from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = PROJECT / "analysis" / "EveBio_positive_control_2026-09-08_v1"
DATA = PROJECT / "data" / "positive_controls_2026-09-08" / "EveBio"
sys.path.insert(0, str(SOURCE))
from run_evebio_scaffold_audit import (  # noqa: E402
    LABELS, coalesce_duplicates, fingerprints_and_scaffolds, smiles_by_inchikey,
)

COUNTS = ["Cells_Number_Object_Number", "Cytoplasm_Number_Object_Number", "Nuclei_Number_Object_Number"]
SEEDS = [17, 43, 101]


def project(train: np.ndarray, test: np.ndarray, pca: bool = False) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    a, b = scaler.fit_transform(train), scaler.transform(test)
    if pca:
        reducer = PCA(n_components=min(64, len(train) - 1, train.shape[1]), svd_solver="randomized", random_state=0)
        a, b = reducer.fit_transform(a), reducer.transform(b)
    return a, b


def main() -> None:
    full = coalesce_duplicates(pd.read_csv(DATA / "CP_all_EveBio.csv"))
    mapping = smiles_by_inchikey(pd.read_csv(DATA / "evebio_assay_data_smiles.csv"))
    structure, scaffolds = fingerprints_and_scaffolds(full.InChIKey.astype(str).tolist(), mapping)
    all_morph = list(full.columns[28:])
    morph_cols = [c for c in all_morph if c not in COUNTS]
    counts = full[COUNTS].to_numpy(float)
    morph = full[morph_cols].to_numpy(float)
    rows = []
    for assay in LABELS:
        valid = full[assay].notna().to_numpy()
        y, s, c, m, g = full.loc[valid, assay].astype(int).to_numpy(), structure[valid], counts[valid], morph[valid], scaffolds[valid]
        for seed in SEEDS:
            splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
            for fold, (tr, te) in enumerate(splitter.split(s, y, g), 1):
                ctr, cte = project(c[tr], c[te])
                mtr, mte = project(m[tr], m[te], pca=True)
                matrices = {
                    "C_logit": (ctr, cte, "logit"), "C+M_logit": (np.c_[ctr, mtr], np.c_[cte, mte], "logit"),
                    "S+C_logit": (np.c_[s[tr], ctr], np.c_[s[te], cte], "logit"),
                    "S+C+M_logit": (np.c_[s[tr], ctr, mtr], np.c_[s[te], cte, mte], "logit"),
                    "C_rf": (c[tr], c[te], "rf"), "C+M_rf": (np.c_[c[tr], mtr], np.c_[c[te], mte], "rf"),
                }
                for name, (xtr, xte, family) in matrices.items():
                    if family == "logit":
                        model = LogisticRegression(C=1, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=0)
                    else:
                        model = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2,
                                                       max_features="sqrt", n_jobs=-1, random_state=seed)
                    p = model.fit(xtr, y[tr]).predict_proba(xte)[:, 1]
                    rows.append({"assay": assay, "seed": seed, "fold": fold, "model": name,
                                 "average_precision": average_precision_score(y[te], p),
                                 "scaffold_overlap": len(set(g[tr]) & set(g[te]))})
    folds = pd.DataFrame(rows)
    folds.to_csv(HERE / "evebio_fold_results.csv", index=False)
    wide = folds.pivot(index=["assay", "seed", "fold"], columns="model", values="average_precision").reset_index()
    wide["SCM_minus_SC_logit"] = wide["S+C+M_logit"] - wide["S+C_logit"]
    wide["CM_minus_C_logit"] = wide["C+M_logit"] - wide["C_logit"]
    wide["CM_minus_C_rf"] = wide["C+M_rf"] - wide["C_rf"]
    summary = wide.groupby("assay", as_index=False).agg(
        SCM_minus_SC_logit_median=("SCM_minus_SC_logit", "median"),
        SCM_minus_SC_logit_positive_fraction=("SCM_minus_SC_logit", lambda x: float((x > 0).mean())),
        CM_minus_C_logit_median=("CM_minus_C_logit", "median"),
        CM_minus_C_rf_median=("CM_minus_C_rf", "median"),
    )
    summary.to_csv(HERE / "evebio_fixed_comparator_summary.csv", index=False)
    result = {"status": "exploratory_non_confirmatory", "rows": len(full), "morphology_features_excluding_counts": len(morph_cols),
              "all_scaffold_overlap_zero": bool((folds.scaffold_overlap == 0).all()),
              "endpoint_median_of_SCM_minus_SC": float(summary.SCM_minus_SC_logit_median.median()),
              "endpoint_positive_fraction": float((summary.SCM_minus_SC_logit_median > 0).mean()),
              "endpoint_median_of_CM_minus_C_rf": float(summary.CM_minus_C_rf_median.median())}
    (HERE / "evebio_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
