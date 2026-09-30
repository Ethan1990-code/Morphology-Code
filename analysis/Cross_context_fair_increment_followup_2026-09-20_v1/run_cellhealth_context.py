from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data" / "positive_controls_2026-09-08" / "CellHealth_v2.0"
OLD = PROJECT / "analysis" / "CellHealth_confirmation_2026-09-08_v1"
COUNTS = ["Cells_Number_Object_Number", "Cytoplasm_Number_Object_Number", "Nuclei_Number_Object_Number"]
LINES = ["A549", "ES2", "HCC44"]
N_PERM = 25


def corr(y: np.ndarray, p: np.ndarray) -> float:
    return math.nan if np.std(y) == 0 or np.std(p) == 0 else float(spearmanr(y, p).statistic)


def splits(meta: pd.DataFrame, kind: str):
    if kind == "gene_holdout":
        groups = meta.Metadata_pert_name.str.replace(r"-[^-]+$", "", regex=True)
        yield from ((f"gene_fold_{i+1}", tr, te) for i, (tr, te) in enumerate(GroupKFold(5).split(meta, groups=groups)))
    else:
        for line in LINES:
            test = np.flatnonzero(meta.Metadata_cell_line.to_numpy() == line)
            train = np.flatnonzero(meta.Metadata_cell_line.to_numpy() != line)
            yield f"holdout_{line}", train, test


def permute_within_line(x: np.ndarray, lines: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    for line in np.unique(lines):
        idx = np.flatnonzero(lines == line)
        out[idx] = x[rng.permutation(idx)]
    return out


def main() -> None:
    x = pd.read_csv(DATA / "cell_painting_modz.tsv.gz", sep="\t").set_index("Metadata_profile_id").sort_index()
    y = pd.read_csv(DATA / "cell_health_modz.tsv.gz", sep="\t").set_index("Metadata_profile_id").sort_index()
    assert x.index.equals(y.index)
    meta = x[["Metadata_pert_name", "Metadata_cell_line"]].reset_index(drop=True)
    targets = [c for c in y.columns if not c.startswith("Metadata_")]
    ymat = y[targets].to_numpy(float)
    counts = x[COUNTS].to_numpy(float)
    morph_cols = [c for c in x.columns if not c.startswith("Metadata_") and c not in COUNTS]
    morph = x[morph_cols].to_numpy(float)
    prediction_rows = []
    perm_rows = []
    for kind in ["gene_holdout", "cell_line_holdout"]:
        for fold_no, (fold, tr, te) in enumerate(splits(meta, kind), 1):
            cs = StandardScaler().fit(counts[tr])
            ctr, cte = cs.transform(counts[tr]), cs.transform(counts[te])
            ms = StandardScaler().fit(morph[tr])
            mtr0, mte0 = ms.transform(morph[tr]), ms.transform(morph[te])
            pca = PCA(n_components=64, svd_solver="randomized", random_state=0).fit(mtr0)
            mtr, mte = pca.transform(mtr0), pca.transform(mte0)
            models = {
                "count": Ridge(alpha=10).fit(ctr, ymat[tr]).predict(cte),
                "morphology": Ridge(alpha=10).fit(mtr, ymat[tr]).predict(mte),
                "fusion": Ridge(alpha=10).fit(np.c_[ctr, mtr], ymat[tr]).predict(np.c_[cte, mte]),
            }
            global_mean = np.repeat(ymat[tr].mean(axis=0)[None, :], len(te), axis=0)
            models["global_mean"] = global_mean
            line_pred = global_mean.copy()
            train_lines = meta.iloc[tr].Metadata_cell_line.to_numpy()
            test_lines = meta.iloc[te].Metadata_cell_line.to_numpy()
            for line in np.unique(test_lines):
                available = tr[train_lines == line]
                if len(available):
                    line_pred[test_lines == line] = ymat[available].mean(axis=0)
            models["cell_line_mean"] = line_pred
            for model, pred in models.items():
                for j, target in enumerate(targets):
                    for local, idx in enumerate(te):
                        prediction_rows.append({"split_type": kind, "fold": fold, "row": int(idx),
                                                "cell_line": meta.iloc[idx].Metadata_cell_line,
                                                "target": target, "model": model,
                                                "y": ymat[idx, j], "prediction": pred[local, j]})
            for perm in range(N_PERM):
                rng = np.random.default_rng(100000 * (1 + (kind == "cell_line_holdout")) + 1000 * fold_no + perm)
                pm = permute_within_line(mtr, train_lines, rng)
                pred = Ridge(alpha=10).fit(pm, ymat[tr]).predict(mte)
                for j, target in enumerate(targets):
                    for local, idx in enumerate(te):
                        perm_rows.append({"split_type": kind, "fold": fold, "permutation": perm,
                                          "row": int(idx), "cell_line": meta.iloc[idx].Metadata_cell_line,
                                          "target": target, "y": ymat[idx, j], "prediction": pred[local, j]})

    preds = pd.DataFrame(prediction_rows)
    perms = pd.DataFrame(perm_rows)
    preds.to_csv(HERE / "cellhealth_oof_predictions.csv.gz", index=False, compression="gzip")
    perms.to_csv(HERE / "cellhealth_permutation_predictions.csv.gz", index=False, compression="gzip")
    scored = []
    for keys, z in preds.groupby(["split_type", "cell_line", "target", "model"], sort=False):
        scored.append(dict(zip(["split_type", "cell_line", "target", "model"], keys)) | {"spearman_rho": corr(z.y.to_numpy(), z.prediction.to_numpy())})
    scored = pd.DataFrame(scored)
    pscored = []
    for keys, z in perms.groupby(["split_type", "cell_line", "target", "permutation"], sort=False):
        pscored.append(dict(zip(["split_type", "cell_line", "target", "permutation"], keys)) | {"spearman_rho": corr(z.y.to_numpy(), z.prediction.to_numpy())})
    pscored = pd.DataFrame(pscored)
    scored.to_csv(HERE / "cellhealth_within_line_scores.csv", index=False)
    pscored.to_csv(HERE / "cellhealth_permutation_scores.csv", index=False)
    med = scored.groupby(["split_type", "target", "model"], as_index=False).spearman_rho.median()
    wide = med.pivot(index=["split_type", "target"], columns="model", values="spearman_rho").reset_index()
    null = pscored.groupby(["split_type", "target", "permutation"], as_index=False).spearman_rho.median()
    null_summary = null.groupby(["split_type", "target"], as_index=False).agg(
        permuted_median=("spearman_rho", "median"), permuted_q95=("spearman_rho", lambda v: float(v.quantile(.95))))
    wide = wide.merge(null_summary, on=["split_type", "target"])
    wide["morphology_minus_count"] = wide.morphology - wide["count"]
    wide["fusion_minus_best_simple"] = wide.fusion - wide[["morphology", "count"]].max(axis=1)
    wide["observed_minus_permuted"] = wide.morphology - wide.permuted_median
    line_wide = scored.pivot(index=["split_type", "cell_line", "target"], columns="model", values="spearman_rho").reset_index()
    line_wide["morphology_minus_count"] = line_wide.morphology - line_wide["count"]
    frac = line_wide.groupby(["split_type", "target"], as_index=False).agg(
        morphology_minus_count_positive_fraction=("morphology_minus_count", lambda v: float((v > 0).mean())))
    wide = wide.merge(frac, on=["split_type", "target"])
    mapping = pd.read_csv(DATA / "author_reference" / "feature_mapping_annotated.csv")[["id", "measurement"]]
    wide = wide.merge(mapping, left_on="target", right_on="id", how="left").drop(columns="id")
    wide.to_csv(HERE / "cellhealth_endpoint_context_summary.csv", index=False)

    old = pd.read_csv(OLD / "endpoint_summary.csv")
    median_positive = old.query("consensus == 'median'").groupby("target").morphology_minus_count_median.median().gt(0)
    def count_retained(rho_cut: float, delta_cut: float, perm_cut: float) -> int:
        passed = wide.assign(p=lambda d: (d.morphology >= rho_cut) & (d.morphology_minus_count > delta_cut) &
                             (d.observed_minus_permuted > perm_cut) & (d.morphology_minus_count_positive_fraction >= 2/3))
        both = passed.pivot(index="target", columns="split_type", values="p")
        keep = both.get("gene_holdout", False) & both.get("cell_line_holdout", False)
        keep &= pd.Series({t: bool(median_positive.get(t, False)) for t in both.index})
        return int(keep.sum())
    sensitivity = []
    for name, values in {"rho": [.15, .20, .25], "delta": [.025, .05, .075], "permutation": [.05, .10, .15]}.items():
        for value in values:
            cuts = {"rho": .20, "delta": .05, "permutation": .10}; cuts[name] = value
            sensitivity.append({"varied_gate": name, "value": value, "retained_cross_context": count_retained(cuts["rho"], cuts["delta"], cuts["permutation"])})
    pd.DataFrame(sensitivity).to_csv(HERE / "cellhealth_threshold_sensitivity.csv", index=False)
    global_summary = wide.groupby("split_type", as_index=False).agg(
        median_morphology=("morphology", "median"), median_count=("count", "median"),
        median_cell_line_mean=("cell_line_mean", "median"), median_morphology_minus_count=("morphology_minus_count", "median"),
        median_observed_minus_permuted=("observed_minus_permuted", "median"),
        median_fusion_minus_best_simple=("fusion_minus_best_simple", "median"))
    result = {"status": "exploratory_non_confirmatory", "rows": len(x), "targets": len(targets), "permutations": N_PERM,
              "morphology_features_excluding_counts": len(morph_cols), "global_summary": global_summary.to_dict("records"),
              "primary_gate_retained": count_retained(.20, .05, .10),
              "boundary": "Repeated identity permutations are a stability diagnostic, not an inferential p-value."}
    (HERE / "cellhealth_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
