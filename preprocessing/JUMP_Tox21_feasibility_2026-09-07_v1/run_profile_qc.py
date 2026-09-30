import csv
import gzip
import hashlib
import io
import json
import platform
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase

from run_metadata_overlap import ASSAYS, parent_key_and_scaffold


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
PROFILE_PATH = INPUTS / "profiles_var_mad_int_featselect_harmony.parquet"
JUMP_PATH = INPUTS / "jump_compound.csv.gz"
TOX21_PATH = INPUTS / "tox21_10k_data_all.sdf.zip"
EXPECTED_BYTES = 2_835_051_686
EXPECTED_MD5 = "c9371af57a36a51e021935c9ca78e506"
META_COLUMNS = ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]
EXCLUDED_VEHICLE_INCHIKEYS = {"IAZDPXIOMUYVGZ-UHFFFAOYSA-N": "dimethyl_sulfoxide_vehicle"}


def file_md5(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_identity_and_labels():
    with gzip.open(JUMP_PATH, "rt", encoding="utf-8") as handle:
        jump = pd.read_csv(handle, usecols=["Metadata_JCP2022", "Metadata_InChIKey"])
    jump = jump.dropna().drop_duplicates()
    jump["connectivity_key"] = jump["Metadata_InChIKey"].str.split("-").str[0]

    rdBase.DisableLog("rdApp.*")
    with zipfile.ZipFile(TOX21_PATH) as archive:
        payload = archive.read(archive.infolist()[0])
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(payload), sanitize=True, removeHs=False)

    records = []
    for mol in supplier:
        if mol is None:
            continue
        parent_key, _, _ = parent_key_and_scaffold(mol)
        if not parent_key:
            continue
        row = {"parent_inchikey": parent_key, "connectivity_key": parent_key.split("-")[0]}
        for assay in ASSAYS:
            value = mol.GetProp(assay).strip() if mol.HasProp(assay) else ""
            row[assay] = value if value in {"0", "1"} else ""
        records.append(row)
    tox = pd.DataFrame(records)

    exact_keys = set(tox["parent_inchikey"]) & set(jump["Metadata_InChIKey"])
    strict_jump = jump[jump["Metadata_InChIKey"].isin(exact_keys)].copy()
    strict_jcp_ids = set(strict_jump["Metadata_JCP2022"])
    jcp_to_connectivity = dict(zip(strict_jump["Metadata_JCP2022"], strict_jump["connectivity_key"]))

    assay_labels = {}
    exact_tox = tox[tox["parent_inchikey"].isin(exact_keys)]
    for assay in ASSAYS:
        values = defaultdict(set)
        for key, value in exact_tox[["connectivity_key", assay]].itertuples(index=False):
            if value in {"0", "1"}:
                values[key].add(int(value))
        assay_labels[assay] = {key: next(iter(v)) for key, v in values.items() if len(v) == 1}
    excluded_vehicle_jcp = set(strict_jump.loc[
        strict_jump["Metadata_InChIKey"].isin(EXCLUDED_VEHICLE_INCHIKEYS), "Metadata_JCP2022"
    ])
    return strict_jcp_ids, excluded_vehicle_jcp, jcp_to_connectivity, assay_labels


def describe_counts(values):
    array = np.asarray(values, dtype=float)
    return {
        "min": int(np.min(array)),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q3": float(np.quantile(array, 0.75)),
        "max": int(np.max(array)),
    }


def main():
    if not PROFILE_PATH.is_file():
        raise FileNotFoundError(PROFILE_PATH)
    size = PROFILE_PATH.stat().st_size
    md5 = file_md5(PROFILE_PATH)
    if size != EXPECTED_BYTES or md5 != EXPECTED_MD5:
        raise RuntimeError(f"Profile fingerprint mismatch: bytes={size}, md5={md5}")

    strict_jcp_ids, excluded_vehicle_jcp, jcp_to_connectivity, assay_labels = load_identity_and_labels()
    parquet = pq.ParquetFile(PROFILE_PATH)
    schema = parquet.schema_arrow
    feature_columns = [name for name in schema.names if name not in META_COLUMNS]
    if len(feature_columns) != 737 or not all(pa.types.is_floating(schema.field(name).type) for name in feature_columns):
        raise RuntimeError("Unexpected profile feature schema")

    total_nulls = Counter()
    total_unique = {name: set() for name in META_COLUMNS}
    matched_frames = []
    for batch in parquet.iter_batches(batch_size=8192):
        ids = batch.column(schema.get_field_index("Metadata_JCP2022")).to_numpy(zero_copy_only=False)
        mask = np.isin(ids, list(strict_jcp_ids))
        for name in META_COLUMNS:
            column = batch.column(schema.get_field_index(name))
            total_nulls[name] += column.null_count
            total_unique[name].update(value for value in column.to_pylist() if value is not None)
        if mask.any():
            selected = batch.filter(pa.array(mask))
            matched_frames.append(selected.to_pandas())

    matched = pd.concat(matched_frames, ignore_index=True)
    covered_jcp = set(matched["Metadata_JCP2022"].dropna().unique())
    missing_jcp = strict_jcp_ids - covered_jcp
    analysis_ready = matched[~matched["Metadata_JCP2022"].isin(excluded_vehicle_jcp)].copy()
    analysis_ready_jcp = set(analysis_ready["Metadata_JCP2022"].unique())
    feature_matrix = matched[feature_columns].to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(feature_matrix)
    per_feature_nonfinite = (~finite).sum(axis=0)

    compound_profiles = matched.groupby("Metadata_JCP2022", sort=True)[feature_columns].mean()
    compound_matrix = compound_profiles.to_numpy(dtype=np.float64, copy=False)
    compound_finite = np.isfinite(compound_matrix)
    variances = np.nanvar(compound_matrix, axis=0)
    replicate_counts = matched.groupby("Metadata_JCP2022").size()
    sources_per_compound = matched.groupby("Metadata_JCP2022")["Metadata_Source"].nunique()
    plates_per_compound = matched.groupby("Metadata_JCP2022")["Metadata_Plate"].nunique()
    ready_replicate_counts = analysis_ready.groupby("Metadata_JCP2022").size()
    ready_sources_per_compound = analysis_ready.groupby("Metadata_JCP2022")["Metadata_Source"].nunique()
    ready_plates_per_compound = analysis_ready.groupby("Metadata_JCP2022")["Metadata_Plate"].nunique()

    endpoint_rows = []
    for assay in ASSAYS:
        labels = assay_labels[assay]
        covered_keys = {jcp_to_connectivity[jcp] for jcp in analysis_ready_jcp if jcp in jcp_to_connectivity}
        eligible = {key: label for key, label in labels.items() if key in covered_keys}
        endpoint_rows.append({
            "assay": assay,
            "profile_covered_labeled_structures": len(eligible),
            "positive_structures": sum(value == 1 for value in eligible.values()),
            "negative_structures": sum(value == 0 for value in eligible.values()),
            "profile_coverage_of_metadata_eligible": None,
        })

    previous = pd.read_csv(ROOT / "endpoint_feasibility.csv")
    previous = previous[previous["match_tier"] == "exact_parent_inchikey"].set_index("assay")
    for row in endpoint_rows:
        denominator = int(previous.loc[row["assay"], "consistent_labeled_structures"])
        row["profile_coverage_of_metadata_eligible"] = round(row["profile_covered_labeled_structures"] / denominator, 6) if denominator else None
    with (ROOT / "profile_endpoint_coverage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=endpoint_rows[0].keys())
        writer.writeheader()
        writer.writerows(endpoint_rows)

    duplicated_profile_keys = int(matched.duplicated(META_COLUMNS, keep=False).sum())
    result = {
        "status": "exploratory_non_confirmatory_profile_qc",
        "file": {
            "bytes": size,
            "md5": md5,
            "expected_bytes_match": size == EXPECTED_BYTES,
            "expected_etag_match": md5 == EXPECTED_MD5,
        },
        "parquet": {
            "rows": parquet.metadata.num_rows,
            "columns": parquet.metadata.num_columns,
            "row_groups": parquet.metadata.num_row_groups,
            "metadata_columns": len(META_COLUMNS),
            "numeric_feature_columns": len(feature_columns),
            "metadata_null_counts": dict(total_nulls),
            "unique_metadata_counts": {name: len(values) for name, values in total_unique.items()},
        },
        "strict_overlap": {
            "metadata_eligible_jcp_ids": len(strict_jcp_ids),
            "profile_covered_jcp_ids": len(covered_jcp),
            "profile_missing_jcp_ids": len(missing_jcp),
            "coverage_fraction": round(len(covered_jcp) / len(strict_jcp_ids), 6),
            "profile_rows": len(matched),
            "unique_sources": int(matched["Metadata_Source"].nunique()),
            "unique_plates": int(matched["Metadata_Plate"].nunique()),
            "unique_wells": int(matched["Metadata_Well"].nunique()),
            "exact_duplicate_profile_key_rows": duplicated_profile_keys,
            "replicates_per_compound": describe_counts(replicate_counts),
            "sources_per_compound": describe_counts(sources_per_compound),
            "plates_per_compound": describe_counts(plates_per_compound),
            "source_profile_counts": {str(k): int(v) for k, v in matched["Metadata_Source"].value_counts().sort_index().items()},
            "vehicle_control_exclusion": {
                "rule": "exclude known vehicle controls before modeling",
                "excluded_jcp_ids": len(excluded_vehicle_jcp),
                "excluded_profile_rows": int(len(matched) - len(analysis_ready)),
                "analysis_ready_jcp_ids": len(analysis_ready_jcp),
                "analysis_ready_profile_rows": len(analysis_ready),
                "replicates_per_compound": describe_counts(ready_replicate_counts),
                "sources_per_compound": describe_counts(ready_sources_per_compound),
                "plates_per_compound": describe_counts(ready_plates_per_compound),
            },
        },
        "features": {
            "matched_profile_values": int(feature_matrix.size),
            "nonfinite_profile_values": int((~finite).sum()),
            "nonfinite_profile_fraction": round(float((~finite).mean()), 10),
            "features_with_any_nonfinite_profile_value": int((per_feature_nonfinite > 0).sum()),
            "compound_aggregated_values": int(compound_matrix.size),
            "nonfinite_compound_aggregated_values": int((~compound_finite).sum()),
            "zero_or_nonfinite_variance_features_after_compound_aggregation": int((~np.isfinite(variances) | (variances <= 0)).sum()),
        },
        "endpoint_coverage": endpoint_rows,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
        "interpretation_boundary": [
            "Profile QC verifies local integrity, overlap coverage, feature finiteness, and replicate/batch structure only.",
            "Rows and wells are technical profiles, not independent compounds; all modeling splits must be compound/scaffold grouped.",
            "No imputation, batch correction, feature selection, model fitting, or endpoint comparison was performed.",
        ],
    }
    (ROOT / "profile_qc_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
