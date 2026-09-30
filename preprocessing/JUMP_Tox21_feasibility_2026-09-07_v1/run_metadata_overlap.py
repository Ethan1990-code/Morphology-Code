import csv
import gzip
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
JUMP_PATH = INPUTS / "jump_compound.csv.gz"
TOX21_PATH = INPUTS / "tox21_10k_data_all.sdf.zip"
ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53",
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def parent_key_and_scaffold(mol):
    try:
        parent = rdMolStandardize.FragmentParent(mol)
        Chem.SanitizeMol(parent)
        key = Chem.MolToInchiKey(parent)
    except Exception:
        return "", "", False
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent)
        return key, scaffold, False
    except Exception:
        return key, "", True


def summarize_endpoints(frame, match_tier):
    endpoint_rows = []
    for assay in ASSAYS:
        per_structure = defaultdict(set)
        scaffold_by_structure = {}
        for row in frame[["connectivity_key", "murcko_scaffold", assay]].itertuples(index=False):
            key, scaffold, value = row
            if value in {"0", "1"}:
                per_structure[key].add(int(value))
                scaffold_by_structure[key] = scaffold
        consistent = {k: next(iter(v)) for k, v in per_structure.items() if len(v) == 1}
        conflicts = sum(len(v) > 1 for v in per_structure.values())
        positives = {k for k, v in consistent.items() if v == 1}
        negatives = {k for k, v in consistent.items() if v == 0}

        def scaffold_token(k):
            scaffold = scaffold_by_structure.get(k, "")
            return scaffold if scaffold else f"ACYCLIC::{k}"

        scaffolds = {scaffold_token(k) for k in consistent}
        pos_scaffolds = {scaffold_token(k) for k in positives}
        neg_scaffolds = {scaffold_token(k) for k in negatives}
        n = len(consistent)
        p = len(positives)
        q = len(negatives)
        basic = n >= 200 and len(scaffolds) >= 80 and p >= 30 and q >= 30 and len(pos_scaffolds) >= 15 and len(neg_scaffolds) >= 15
        strong = n >= 500 and len(scaffolds) >= 150 and p >= 50 and q >= 100 and len(pos_scaffolds) >= 25 and len(neg_scaffolds) >= 50
        endpoint_rows.append({
            "match_tier": match_tier,
            "assay": assay,
            "consistent_labeled_structures": n,
            "positive_structures": p,
            "negative_structures": q,
            "positive_fraction": round(p / n, 6) if n else None,
            "conflicting_structures_excluded": conflicts,
            "scaffold_groups": len(scaffolds),
            "positive_scaffold_groups": len(pos_scaffolds),
            "negative_scaffold_groups": len(neg_scaffolds),
            "passes_basic_feasibility": basic,
            "passes_strong_feasibility": strong,
        })
    return endpoint_rows


def main():
    rdBase.DisableLog("rdApp.*")
    with gzip.open(JUMP_PATH, "rt", encoding="utf-8") as f:
        jump = pd.read_csv(f, usecols=["Metadata_JCP2022", "Metadata_InChIKey"])
    jump = jump.dropna().drop_duplicates()
    jump["connectivity_key"] = jump["Metadata_InChIKey"].str.split("-").str[0]
    jump_connectivity = set(jump["connectivity_key"])

    with zipfile.ZipFile(TOX21_PATH) as zf:
        payload = zf.read(zf.infolist()[0])
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(payload), sanitize=True, removeHs=False)

    records = []
    parse_invalid = 0
    identity_invalid = 0
    scaffold_errors = 0
    for mol in supplier:
        if mol is None:
            parse_invalid += 1
            continue
        key, scaffold, scaffold_error = parent_key_and_scaffold(mol)
        if not key:
            identity_invalid += 1
            continue
        scaffold_errors += int(scaffold_error)
        row = {
            "tox21_record_id": mol.GetProp("_Name") if mol.HasProp("_Name") else "",
            "parent_inchikey": key,
            "connectivity_key": key.split("-")[0],
            "murcko_scaffold": scaffold,
        }
        for assay in ASSAYS:
            value = mol.GetProp(assay).strip() if mol.HasProp(assay) else ""
            row[assay] = value if value in {"0", "1"} else ""
        records.append(row)

    tox = pd.DataFrame(records)
    jump_full_keys = set(jump["Metadata_InChIKey"])
    overlap = tox[tox["connectivity_key"].isin(jump_connectivity)].copy()
    exact_overlap = tox[tox["parent_inchikey"].isin(jump_full_keys)].copy()
    overlap_keys = set(overlap["connectivity_key"])
    overlap_jump = jump[jump["connectivity_key"].isin(overlap_keys)].copy()

    endpoint_rows = summarize_endpoints(exact_overlap, "exact_parent_inchikey") + summarize_endpoints(overlap, "connectivity_block")
    endpoint_rows.sort(key=lambda x: (x["match_tier"], -x["passes_strong_feasibility"], -x["consistent_labeled_structures"], x["assay"]))
    with (ROOT / "endpoint_feasibility.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=endpoint_rows[0].keys())
        writer.writeheader()
        writer.writerows(endpoint_rows)

    results = {
        "status": "feasibility_probe_non_confirmatory",
        "access_date": "2026-09-07",
        "identity_rule": "standardized_parent_InChIKey_connectivity_block; stereochemistry/protonation collapsed for feasibility only",
        "inputs": {
            "jump_compound_metadata": {"bytes": JUMP_PATH.stat().st_size, "sha256": sha256(JUMP_PATH)},
            "tox21_training_sdf_zip": {"bytes": TOX21_PATH.stat().st_size, "sha256": sha256(TOX21_PATH)},
            "jump_profile_not_downloaded": {
                "bytes": 2835051686,
                "url": "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/profiles_assembled/COMPOUND/v1.0/profiles_var_mad_int_featselect_harmony.parquet",
            },
        },
        "counts": {
            "jump_unique_jcp_ids": int(jump["Metadata_JCP2022"].nunique()),
            "jump_unique_connectivity_keys": int(jump["connectivity_key"].nunique()),
            "tox21_valid_records": int(len(tox)),
            "tox21_parse_invalid_records": int(parse_invalid),
            "tox21_parent_identity_invalid_records": int(identity_invalid),
            "tox21_scaffold_generation_errors": int(scaffold_errors),
            "tox21_unique_connectivity_keys": int(tox["connectivity_key"].nunique()),
            "overlap_exact_parent_inchikeys": int(exact_overlap["parent_inchikey"].nunique()),
            "overlap_unique_connectivity_keys": int(len(overlap_keys)),
            "overlap_jump_jcp_ids": int(overlap_jump["Metadata_JCP2022"].nunique()),
            "overlap_tox21_records": int(len(overlap)),
        },
        "feasibility_thresholds": {
            "basic": "n>=200; scaffolds>=80; positives and negatives>=30; positive and negative scaffolds>=15",
            "strong": "n>=500; scaffolds>=150; positives>=50; negatives>=100; positive scaffolds>=25; negative scaffolds>=50",
            "note": "engineering screening thresholds, not a formal power calculation",
        },
        "endpoints": endpoint_rows,
        "limitations": [
            "Profile-level availability and non-null morphology are not verified without reading the 2.84 GB profile table.",
            "Connectivity-block matching collapses stereochemistry and protonation and must be independently audited before formal analysis.",
            "Biological positive/negative-control roles must be frozen before morphology outcomes are inspected.",
            "This probe does not estimate model performance or support manuscript claims.",
        ],
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results["counts"], indent=2))
    for tier in ["exact_parent_inchikey", "connectivity_block"]:
        print(tier, "strong_endpoints", [x["assay"] for x in endpoint_rows if x["match_tier"] == tier and x["passes_strong_feasibility"]])
        print(tier, "basic_only_endpoints", [x["assay"] for x in endpoint_rows if x["match_tier"] == tier and x["passes_basic_feasibility"] and not x["passes_strong_feasibility"]])


if __name__ == "__main__":
    main()
