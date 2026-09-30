import csv
import gzip
import io
import json
import platform
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from run_metadata_overlap import ASSAYS, parent_key_and_scaffold


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
PROFILE_PATH = INPUTS / "profiles_var_mad_int_featselect_harmony.parquet"
JUMP_PATH = INPUTS / "jump_compound.csv.gz"
TOX21_PATH = INPUTS / "tox21_10k_data_all.sdf.zip"
META_COLUMNS = ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]
TARGET_ASSAYS = ["SR-ARE", "SR-MMP", "SR-p53", "NR-ER-LBD"]
ASSAY_ROLES = {
    "SR-ARE": "morphology_aligned_candidate",
    "SR-MMP": "morphology_aligned_candidate",
    "SR-p53": "secondary_stress_test",
    "NR-ER-LBD": "lower_alignment_comparator",
}
VEHICLE_KEYS = {"IAZDPXIOMUYVGZ-UHFFFAOYSA-N"}
SEEDS = [17, 43, 101]
N_SPLITS = 5
PCA_COMPONENTS = 64


def load_identity_labels_and_molecules():
    with gzip.open(JUMP_PATH, "rt", encoding="utf-8") as handle:
        jump = pd.read_csv(handle, usecols=["Metadata_JCP2022", "Metadata_InChIKey"])
    jump = jump.dropna().drop_duplicates()
    jump["connectivity_key"] = jump["Metadata_InChIKey"].str.split("-").str[0]

    rdBase.DisableLog("rdApp.*")
    with zipfile.ZipFile(TOX21_PATH) as archive:
        payload = archive.read(archive.infolist()[0])
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(payload), sanitize=True, removeHs=False)

    records = []
    parent_mols = {}
    scaffolds = {}
    for mol in supplier:
        if mol is None:
            continue
        full_key, scaffold, _ = parent_key_and_scaffold(mol)
        if not full_key:
            continue
        connectivity = full_key.split("-")[0]
        row = {"parent_inchikey": full_key, "connectivity_key": connectivity}
        for assay in ASSAYS:
            value = mol.GetProp(assay).strip() if mol.HasProp(assay) else ""
            row[assay] = value if value in {"0", "1"} else ""
        records.append(row)
        if connectivity not in parent_mols:
            parent = rdMolStandardize.FragmentParent(mol)
            Chem.SanitizeMol(parent)
            parent_mols[connectivity] = parent
            scaffolds[connectivity] = scaffold if scaffold else f"ACYCLIC::{connectivity}"
    tox = pd.DataFrame(records)

    exact_keys = (set(tox["parent_inchikey"]) & set(jump["Metadata_InChIKey"])) - VEHICLE_KEYS
    strict_jump = jump[jump["Metadata_InChIKey"].isin(exact_keys)].copy()
    jcp_to_connectivity = dict(zip(strict_jump["Metadata_JCP2022"], strict_jump["connectivity_key"]))

    exact_tox = tox[tox["parent_inchikey"].isin(exact_keys)]
    labels = {}
    for assay in TARGET_ASSAYS:
        values = defaultdict(set)
        for key, value in exact_tox[["connectivity_key", assay]].itertuples(index=False):
            if value in {"0", "1"}:
                values[key].add(int(value))
        labels[assay] = {key: next(iter(v)) for key, v in values.items() if len(v) == 1}
    return jcp_to_connectivity, labels, parent_mols, scaffolds


def load_and_aggregate_morphology(jcp_to_connectivity):
    parquet = pq.ParquetFile(PROFILE_PATH)
    feature_columns = [name for name in parquet.schema_arrow.names if name not in META_COLUMNS]
    target_ids = set(jcp_to_connectivity)
    selected_frames = []
    jcp_index = parquet.schema_arrow.get_field_index("Metadata_JCP2022")
    for batch in parquet.iter_batches(batch_size=8192):
        ids = batch.column(jcp_index).to_numpy(zero_copy_only=False)
        mask = np.isin(ids, list(target_ids))
        if mask.any():
            selected_frames.append(batch.filter(pa.array(mask)).to_pandas())
    profiles = pd.concat(selected_frames, ignore_index=True)
    profiles["connectivity_key"] = profiles["Metadata_JCP2022"].map(jcp_to_connectivity)
    source_level = profiles.groupby(["connectivity_key", "Metadata_Source"], sort=True)[feature_columns].median()
    compound_level = source_level.groupby("connectivity_key", sort=True).median()
    if not np.isfinite(compound_level.to_numpy()).all():
        raise RuntimeError("Non-finite morphology after robust aggregation")
    return profiles, source_level, compound_level, feature_columns


def fingerprints(keys, parent_mols):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    matrix = np.zeros((len(keys), 2048), dtype=np.float32)
    for row, key in enumerate(keys):
        vector = np.zeros(2048, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(parent_mols[key]), vector)
        matrix[row] = vector
    return matrix


def classifier():
    return LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear",
        max_iter=5000, random_state=0,
    )


def score(y_true, probability):
    return {
        "average_precision": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def fit_probability(x_train, y_train, x_test):
    model = classifier()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return model.predict_proba(x_test)[:, 1], converged


def quantile_summary(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)),
        "median": float(np.median(array)),
        "q025": float(np.quantile(array, 0.025)),
        "q975": float(np.quantile(array, 0.975)),
    }


def main():
    jcp_to_connectivity, labels, parent_mols, scaffolds = load_identity_labels_and_molecules()
    profiles, source_level, morphology, feature_columns = load_and_aggregate_morphology(jcp_to_connectivity)
    keys = list(morphology.index)
    x_morph_all = morphology.to_numpy(dtype=np.float64)
    x_struct_all = fingerprints(keys, parent_mols)
    key_to_row = {key: idx for idx, key in enumerate(keys)}

    analysis_units = pd.DataFrame({
        "connectivity_key": keys,
        "murcko_scaffold": [scaffolds[key] for key in keys],
        "profile_rows": profiles.groupby("connectivity_key").size().reindex(keys).to_numpy(),
        "source_groups": source_level.groupby(level=0).size().reindex(keys).to_numpy(),
    })
    for assay in TARGET_ASSAYS:
        analysis_units[assay] = [labels[assay].get(key, np.nan) for key in keys]
    analysis_units.to_csv(ROOT / "analysis_ready_units.csv", index=False, encoding="utf-8-sig")
    morphology.reset_index().to_parquet(ROOT / "analysis_ready_morphology.parquet", index=False)

    fold_rows = []
    convergence_failures = 0
    for assay in TARGET_ASSAYS:
        assay_keys = [key for key in keys if key in labels[assay]]
        indices = np.array([key_to_row[key] for key in assay_keys])
        y = np.array([labels[assay][key] for key in assay_keys], dtype=int)
        groups = np.array([scaffolds[key] for key in assay_keys], dtype=object)
        x_struct = x_struct_all[indices]
        x_morph = x_morph_all[indices]
        for seed in SEEDS:
            splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            for fold, (train, test) in enumerate(splitter.split(x_struct, y, groups), start=1):
                if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
                    raise RuntimeError(f"Single-class fold: {assay}, seed={seed}, fold={fold}")

                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(x_morph[train])
                test_scaled = scaler.transform(x_morph[test])
                components = min(PCA_COMPONENTS, len(train) - 1, train_scaled.shape[1])
                pca = PCA(n_components=components, svd_solver="randomized", random_state=0)
                train_morph = pca.fit_transform(train_scaled)
                test_morph = pca.transform(test_scaled)

                rng = np.random.default_rng(seed * 100 + fold)
                shuffled_train = train_morph[rng.permutation(len(train_morph))]
                shuffled_test = test_morph[rng.permutation(len(test_morph))]
                matrices = {
                    "structure": (x_struct[train], x_struct[test]),
                    "morphology": (train_morph, test_morph),
                    "fusion": (
                        np.hstack([x_struct[train], train_morph]),
                        np.hstack([x_struct[test], test_morph]),
                    ),
                    "morphology_mismatched": (shuffled_train, shuffled_test),
                    "fusion_mismatched": (
                        np.hstack([x_struct[train], shuffled_train]),
                        np.hstack([x_struct[test], shuffled_test]),
                    ),
                }
                for model_name, (x_train, x_test) in matrices.items():
                    probability, converged = fit_probability(x_train, y[train], x_test)
                    convergence_failures += int(not converged)
                    metrics = score(y[test], probability)
                    fold_rows.append({
                        "assay": assay,
                        "assay_role": ASSAY_ROLES[assay],
                        "seed": seed,
                        "fold": fold,
                        "model": model_name,
                        "train_n": len(train),
                        "test_n": len(test),
                        "train_positive": int(y[train].sum()),
                        "test_positive": int(y[test].sum()),
                        "scaffold_overlap": len(set(groups[train]) & set(groups[test])),
                        "converged": converged,
                        **metrics,
                    })

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(ROOT / "benchmark_fold_results.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for (assay, model), frame in folds.groupby(["assay", "model"], sort=True):
        for metric in ["average_precision", "roc_auc", "brier"]:
            summary_rows.append({
                "assay": assay,
                "assay_role": ASSAY_ROLES[assay],
                "model": model,
                "metric": metric,
                "fold_runs": len(frame),
                **quantile_summary(frame[metric]),
            })
    summaries = pd.DataFrame(summary_rows)
    summaries.to_csv(ROOT / "benchmark_summary.csv", index=False, encoding="utf-8-sig")

    comparisons = [
        ("fusion_minus_structure", "fusion", "structure"),
        ("fusion_minus_fusion_mismatched", "fusion", "fusion_mismatched"),
        ("morphology_minus_morphology_mismatched", "morphology", "morphology_mismatched"),
    ]
    delta_rows = []
    decisions = {}
    for assay in TARGET_ASSAYS:
        assay_frame = folds[folds["assay"] == assay]
        pivot = assay_frame.pivot(index=["seed", "fold"], columns="model", values=["average_precision", "roc_auc", "brier"])
        assay_deltas = {}
        for name, left, right in comparisons:
            for metric in ["average_precision", "roc_auc", "brier"]:
                delta = pivot[(metric, left)] - pivot[(metric, right)]
                stats = quantile_summary(delta)
                stats["positive_fraction"] = float((delta > 0).mean())
                delta_rows.append({
                    "assay": assay,
                    "assay_role": ASSAY_ROLES[assay],
                    "comparison": name,
                    "metric": metric,
                    "fold_pairs": len(delta),
                    **stats,
                })
                assay_deltas[(name, metric)] = stats
        fusion_structure = assay_deltas[("fusion_minus_structure", "average_precision")]
        fusion_mismatch = assay_deltas[("fusion_minus_fusion_mismatched", "average_precision")]
        if fusion_structure["median"] > 0.02 and fusion_structure["positive_fraction"] >= 0.8 and fusion_mismatch["median"] > 0.02:
            decision = "retain_morphology"
        elif fusion_structure["median"] <= 0 or fusion_mismatch["median"] <= 0:
            decision = "reject_morphology"
        else:
            decision = "uncertain"
        decisions[assay] = {
            "role": ASSAY_ROLES[assay],
            "engineering_decision": decision,
            "fusion_minus_structure_ap": fusion_structure,
            "fusion_minus_fusion_mismatched_ap": fusion_mismatch,
        }
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(ROOT / "benchmark_deltas.csv", index=False, encoding="utf-8-sig")

    result = {
        "status": "exploratory_non_confirmatory_second_task_benchmark",
        "protocol": "second_task_benchmark_protocol_2026-09-07_v1.md",
        "analysis_ready": {
            "jcp_ids": len(jcp_to_connectivity),
            "connectivity_units": len(keys),
            "profile_rows": len(profiles),
            "source_level_profiles": len(source_level),
            "morphology_features": len(feature_columns),
            "structure_features": x_struct_all.shape[1],
        },
        "folds": {"type": "StratifiedGroupKFold", "groups": "Murcko scaffold", "splits": N_SPLITS, "seeds": SEEDS},
        "decisions": decisions,
        "convergence_failures": convergence_failures,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "boundaries": [
            "Exploratory results; no independent review or formal freeze.",
            "Harmony and upstream feature selection were performed by the public JUMP release, not refit inside each endpoint fold.",
            "Repeated cross-validation folds are dependent; fold quantiles are descriptive and are not confidence intervals.",
            "External Tox21 test/final stress testing remains separate.",
        ],
    }
    (ROOT / "benchmark_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
