from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = PROJECT / "preprocessing" / "JUMP_Tox21_feasibility_2026-09-07_v1"
COUNT_SOURCE = PROJECT / "preprocessing" / "JUMP_cell_count_proxy_intake_2026-09-08_v1"
sys.path.insert(0, str(SOURCE))
from run_second_task_benchmark import fingerprints, load_identity_labels_and_molecules  # noqa: E402

ASSAYS = ["SR-ARE", "SR-MMP", "SR-p53", "NR-ER-LBD"]
SEEDS = [17, 43, 101]


def transform(train: np.ndarray, test: np.ndarray, pca: bool = False) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    a, b = scaler.fit_transform(train), scaler.transform(test)
    if pca:
        reducer = PCA(n_components=min(64, len(train) - 1, train.shape[1]), svd_solver="randomized", random_state=0)
        a, b = reducer.fit_transform(a), reducer.transform(b)
    return a, b


def predict(train: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
    model = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(train, y)
    return model.predict_proba(test)[:, 1]


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def main() -> None:
    units = pd.read_csv(SOURCE / "analysis_ready_units.csv")
    morph = pd.read_parquet(SOURCE / "analysis_ready_morphology.parquet")
    counts = pd.read_csv(COUNT_SOURCE / "cell_count_proxy_by_compound.csv")
    frame = units.merge(morph, on="connectivity_key", validate="one_to_one").merge(
        counts[["connectivity_key", "Nuclei_Number_Object_Number", "Cytoplasm_Number_Object_Number"]],
        on="connectivity_key", validate="one_to_one",
    )
    _, _, parent_mols, _ = load_identity_labels_and_molecules()
    keys = frame["connectivity_key"].tolist()
    structure = fingerprints(keys, parent_mols).astype(float)
    count_cols = ["Nuclei_Number_Object_Number", "Cytoplasm_Number_Object_Number"]
    morph_cols = [c for c in morph.columns if c != "connectivity_key"]
    count = frame[count_cols].to_numpy(float)
    morphology = frame[morph_cols].to_numpy(float)
    groups = frame["murcko_scaffold"].to_numpy(object)

    rows = []
    for assay in ASSAYS:
        valid = frame[assay].notna().to_numpy()
        y = frame.loc[valid, assay].astype(int).to_numpy()
        s, c, m, g = structure[valid], count[valid], morphology[valid], groups[valid]
        for seed in SEEDS:
            split = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
            for fold, (tr, te) in enumerate(split.split(s, y, g), 1):
                ctr, cte = transform(c[tr], c[te])
                mtr, mte = transform(m[tr], m[te], pca=True)
                matrices = {
                    "C": (ctr, cte),
                    "S": (s[tr], s[te]),
                    "C+M": (np.c_[ctr, mtr], np.c_[cte, mte]),
                    "S+M": (np.c_[s[tr], mtr], np.c_[s[te], mte]),
                    "S+C": (np.c_[s[tr], ctr], np.c_[s[te], cte]),
                    "S+C+M": (np.c_[s[tr], ctr, mtr], np.c_[s[te], cte, mte]),
                }
                for name, (xtr, xte) in matrices.items():
                    p = predict(xtr, y[tr], xte)
                    rows.append({"assay": assay, "seed": seed, "fold": fold, "model": name,
                                 "n_train": len(tr), "n_test": len(te),
                                 "scaffold_overlap": len(set(g[tr]) & set(g[te])), **metrics(y[te], p)})
    folds = pd.DataFrame(rows)
    folds.to_csv(HERE / "jump_fold_results.csv", index=False)
    wide = folds.pivot(index=["assay", "seed", "fold"], columns="model", values="average_precision").reset_index()
    for lhs, rhs, label in [("S+C+M", "S+C", "SCM_minus_SC"), ("C+M", "C", "CM_minus_C"), ("S+M", "S", "SM_minus_S")]:
        wide[label] = wide[lhs] - wide[rhs]
    summary = wide.groupby("assay", as_index=False).agg(
        SCM_minus_SC_mean=("SCM_minus_SC", "mean"), SCM_minus_SC_median=("SCM_minus_SC", "median"),
        SCM_minus_SC_positive_fraction=("SCM_minus_SC", lambda x: float((x > 0).mean())),
        CM_minus_C_mean=("CM_minus_C", "mean"), CM_minus_C_median=("CM_minus_C", "median"),
        SM_minus_S_mean=("SM_minus_S", "mean"), SM_minus_S_median=("SM_minus_S", "median"),
    )
    summary.to_csv(HERE / "jump_fixed_comparator_summary.csv", index=False)
    result = {"status": "exploratory_non_confirmatory", "rows": len(frame), "morphology_features": len(morph_cols),
              "fixed_baseline": "S+C", "all_scaffold_overlap_zero": bool((folds.scaffold_overlap == 0).all()),
              "summary": summary.to_dict("records")}
    (HERE / "jump_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
