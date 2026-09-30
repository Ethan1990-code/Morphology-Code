import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "JUMP_Tox21_feasibility_2026-09-07_v1"
INPUT = SOURCE / "inputs" / "profiles_var_mad_int (1).parquet"
sys.path.insert(0, str(SOURCE))
import run_second_task_benchmark as benchmark  # noqa: E402

EXPECTED_BYTES = 12_111_419_423
EXPECTED_SHA256 = "42028E8C60692DF545E0B1DD087FC9B911F5117C318A8819D768CFF251E4EDDA"
META = ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]
PROXIES = ["Nuclei_Number_Object_Number", "Cytoplasm_Number_Object_Number"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    if INPUT.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("Downloaded file size does not match the server Content-Length")
    observed_hash = sha256(INPUT)
    if observed_hash != EXPECTED_SHA256:
        raise RuntimeError("Downloaded file SHA-256 changed from the independently computed intake hash")

    jcp_to_connectivity, _, _, _ = benchmark.load_identity_labels_and_molecules()
    target_ids = set(jcp_to_connectivity)
    parquet = pq.ParquetFile(INPUT)
    missing = sorted(set(META + PROXIES) - set(parquet.schema_arrow.names))
    if missing:
        raise RuntimeError(f"Required columns missing: {missing}")

    selected = []
    for batch in parquet.iter_batches(batch_size=65_536, columns=META + PROXIES):
        ids = batch.column(3).to_numpy(zero_copy_only=False)
        mask = np.isin(ids, list(target_ids))
        if mask.any():
            selected.append(batch.filter(pa.array(mask)).to_pandas())
    frame = pd.concat(selected, ignore_index=True)
    frame["connectivity_key"] = frame["Metadata_JCP2022"].map(jcp_to_connectivity)

    source_level = frame.groupby(["connectivity_key", "Metadata_Source"], sort=True)[PROXIES].median()
    compound_level = source_level.groupby("connectivity_key", sort=True).median()
    counts = frame.groupby("connectivity_key").size().rename("profile_rows")
    source_counts = source_level.groupby(level=0).size().rename("source_groups")
    output = compound_level.join([counts, source_counts]).reset_index()
    output.to_csv(HERE / "cell_count_proxy_by_compound.csv", index=False, encoding="utf-8-sig")

    correlation = compound_level.corr(method="spearman").iloc[0, 1]
    result = {
        "status": "exploratory_non_confirmatory",
        "input_bytes": INPUT.stat().st_size,
        "input_sha256": observed_hash,
        "parquet_rows": parquet.metadata.num_rows,
        "parquet_columns": parquet.metadata.num_columns,
        "parquet_row_groups": parquet.metadata.num_row_groups,
        "matched_profile_rows": int(len(frame)),
        "matched_compounds": int(len(output)),
        "proxies": PROXIES,
        "missing_values": {name: int(output[name].isna().sum()) for name in PROXIES},
        "nuclei_cytoplasm_spearman": float(correlation),
        "boundary": (
            "The assembled JUMP profile does not contain Image_Count_Cells or "
            "Cells_Number_Object_Number. Nuclei/Cytoplasm Number_Object_Number are "
            "cell-count-correlated proxy features, not direct image-level counts."
        ),
    }
    (HERE / "intake_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
