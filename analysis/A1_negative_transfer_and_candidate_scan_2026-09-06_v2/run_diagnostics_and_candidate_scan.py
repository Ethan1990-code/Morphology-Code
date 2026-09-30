# -*- coding: utf-8 -*-
"""Exploratory, non-confirmatory negative-transfer and candidate scan."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SEED, B = 20260906, 2000
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BASE = PROJECT / "analysis" / "A1_cross_cell_transfer_2026-09-06_v1"


def load_base():
    path = BASE / "run_A1_benchmark.py"
    spec = importlib.util.spec_from_file_location("a1_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def interval(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [float("nan"), float("nan")]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def adjust_bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    good = np.isfinite(p)
    order = np.argsort(p[good])
    ranked = p[good][order]
    adj = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(adj)
    restored[order] = np.minimum(adj, 1)
    out[good] = restored
    return out


def applicability(xs, xl, y, groups):
    fold = np.zeros(len(y), int)
    sim_s, sim_l = np.zeros(len(y)), np.zeros(len(y))
    for k, (tr, te) in enumerate(GroupKFold(5).split(xs, y, groups), 1):
        fold[te] = k
        inter = xs[te] @ xs[tr].T
        union = xs[te].sum(1)[:, None] + xs[tr].sum(1)[None, :] - inter
        sim_s[te] = np.max(inter / np.maximum(union, 1e-12), axis=1)
        a = xl[tr] / np.maximum(np.linalg.norm(xl[tr], axis=1)[:, None], 1e-12)
        b = xl[te] / np.maximum(np.linalg.norm(xl[te], axis=1)[:, None], 1e-12)
        sim_l[te] = np.max(b @ a.T, axis=1)
    return fold, sim_s, sim_l


def negative_transfer(base):
    frame = pd.read_csv(BASE / "analysis_matrix.csv.gz")
    pred = pd.read_csv(BASE / "oof_predictions.csv.gz")
    if not frame.name.equals(pred.name):
        raise RuntimeError("Input rows do not align")
    paths = list(base.PATHWAYS)
    y = np.column_stack([pred[f"observed__{p}"] for p in paths])
    ps = np.column_stack([pred[f"pred_structure__{p}"] for p in paths])
    pm = {m: np.column_stack([pred[f"pred_{m}__{p}"] for p in paths])
          for m in ("l1000", "fusion", "gated_fusion")}
    scale = np.std(y, axis=0, ddof=1)
    scale[scale == 0] = 1
    xs, groups = base.fingerprints(frame.parent_smiles)
    xl = frame[[f"{p}_l1000" for p in paths]].to_numpy(float)
    fold, sim_s, sim_l = applicability(xs, xl, y, groups)
    keep = ["name", "parent_smiles", "dataset", "cell_id", "time_num", "dose_num", "available_signature_n"]
    compounds = frame[keep].copy()
    compounds["fold"], compounds["scaffold"] = fold, groups
    compounds["structure_applicability"] = sim_s
    compounds["l1000_applicability"] = sim_l
    compounds["l1000_vector_norm"] = np.linalg.norm(xl, axis=1)
    compounds["log1p_signature_n"] = np.log1p(compounds.available_signature_n)
    compounds["time_distance_from_24h"] = abs(compounds.time_num - 24)
    compounds["dose_distance_from_10uM_log10"] = abs(np.log10(np.maximum(compounds.dose_num, 1e-4) / 10))
    rng = np.random.default_rng(SEED)
    sq_s = ((y - ps) / scale) ** 2
    summaries, path_rows = {}, []
    for mode, values in pm.items():
        improvement = np.mean(sq_s - ((y - values) / scale) ** 2, axis=1)
        compounds[f"{mode}_incremental_mse"] = improvement
        means, fractions = [], []
        for _ in range(B):
            idx = rng.integers(0, len(y), len(y))
            means.append(np.mean(improvement[idx]))
            fractions.append(np.mean(improvement[idx] > 0))
        summaries[mode] = {
            "mean_incremental_mse_positive_is_better": float(np.mean(improvement)),
            "compound_bootstrap_ci95": interval(means),
            "fraction_compounds_helped": float(np.mean(improvement > 0)),
            "fraction_bootstrap_ci95": interval(fractions),
        }
        for j, pathway in enumerate(paths):
            point = np.sqrt(np.mean((y[:, j] - values[:, j]) ** 2)) / scale[j] - np.sqrt(np.mean((y[:, j] - ps[:, j]) ** 2)) / scale[j]
            boots = []
            for _ in range(B):
                idx = rng.integers(0, len(y), len(y))
                boots.append(np.sqrt(np.mean((y[idx, j] - values[idx, j]) ** 2)) / scale[j] - np.sqrt(np.mean((y[idx, j] - ps[idx, j]) ** 2)) / scale[j])
            lo, hi = interval(boots)
            path_rows.append({"model": mode, "pathway": pathway, "delta_nrmse_model_minus_structure": float(point), "ci95_low": lo, "ci95_high": hi})
    moderators = ["structure_applicability", "l1000_applicability", "l1000_vector_norm", "log1p_signature_n", "time_distance_from_24h", "dose_distance_from_10uM_log10"]
    target, mod_rows = compounds.fusion_incremental_mse.to_numpy(), []
    for name in moderators:
        x = compounds[name].to_numpy()
        if len(np.unique(x[np.isfinite(x)])) < 2:
            mod_rows.append({"moderator": name, "spearman": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "nominal_p": np.nan, "status": "not_estimable_no_variation"})
            continue
        stat = spearmanr(x, target)
        boots = []
        for _ in range(B):
            idx = rng.integers(0, len(x), len(x))
            boots.append(spearmanr(x[idx], target[idx]).statistic)
        lo, hi = interval(boots)
        mod_rows.append({"moderator": name, "spearman": float(stat.statistic), "ci95_low": lo, "ci95_high": hi, "nominal_p": float(stat.pvalue), "status": "estimated_exploratory"})
    mods = pd.DataFrame(mod_rows)
    mods["bh_q"] = adjust_bh(mods.nominal_p)
    contexts = []
    for variable in ("dataset", "cell_id"):
        for level, group in compounds.groupby(variable):
            values = group.fusion_incremental_mse.to_numpy()
            if len(values) < 5:
                continue
            boots = [np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(B)]
            lo, hi = interval(boots)
            contexts.append({"variable": variable, "level": level, "n": len(values), "mean_fusion_incremental_mse": float(np.mean(values)), "ci95_low": lo, "ci95_high": hi, "fraction_helped": float(np.mean(values > 0))})
    compounds.to_csv(HERE / "compound_incremental_errors.csv", index=False)
    pd.DataFrame(path_rows).to_csv(HERE / "pathway_incremental_metrics.csv", index=False)
    mods.to_csv(HERE / "moderator_associations.csv", index=False)
    pd.DataFrame(contexts).to_csv(HERE / "categorical_context_summaries.csv", index=False)
    return summaries, mods, pd.DataFrame(path_rows)


def full_structure(base):
    si, pathways = base.read_si(), base.read_pathways()
    cardiac, coverage = base.cardiac_pathway_slopes(set(si.name), pathways)
    frame = si[["name", "parent_smiles"]].merge(cardiac, on="name", validate="one_to_one")
    targets = list(base.PATHWAYS)
    frame = frame.dropna(subset=targets).sort_values("name").reset_index(drop=True)
    x, groups = base.fingerprints(frame.parent_smiles)
    y = frame[targets].to_numpy(float)
    pn, ps = np.zeros_like(y), np.zeros_like(y)
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(5).split(x, y, groups), 1):
        alpha = base.tune_alpha(x[tr], y[tr], groups[tr])
        pn[te] = np.mean(y[tr], axis=0)
        ps[te] = base.fit_predict(x[tr], y[tr], x[te], alpha)
        folds.append({"fold": k, "train_n": len(tr), "test_n": len(te), "alpha": alpha})
    scale = np.std(y, axis=0, ddof=1)
    scale[scale == 0] = 1
    def metrics(p):
        rmse = np.sqrt(np.mean((y - p) ** 2, axis=0)) / scale
        rho = [spearmanr(y[:, j], p[:, j]).statistic for j in range(y.shape[1])]
        return {"mean_nrmse": float(np.mean(rmse)), "median_spearman": float(np.nanmedian(rho)), "per_pathway_nrmse": rmse.tolist(), "per_pathway_spearman": [float(v) for v in rho]}
    mn, ms = metrics(pn), metrics(ps)
    rng, deltas = np.random.default_rng(SEED + 1), []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        s = np.std(y[idx], axis=0, ddof=1)
        s[s == 0] = 1
        ln = np.mean(np.sqrt(np.mean((y[idx] - pn[idx]) ** 2, axis=0)) / s)
        ls = np.mean(np.sqrt(np.mean((y[idx] - ps[idx]) ** 2, axis=0)) / s)
        deltas.append(ls - ln)
    z = StandardScaler().fit_transform(y)
    pca = PCA().fit(z)
    clusters = []
    for k in range(2, 9):
        labels = KMeans(k, n_init=50, random_state=SEED).fit_predict(z)
        clusters.append({"k": k, "silhouette": float(silhouette_score(z, labels)), "smallest_cluster_n": int(np.bincount(labels).min())})
    ctab = pd.DataFrame(clusters)
    ctab.to_csv(HERE / "response_archetype_feasibility.csv", index=False)
    out = frame[["name", "parent_smiles"]].copy()
    out["scaffold"] = groups
    for j, pathway in enumerate(targets):
        out[f"observed__{pathway}"], out[f"pred_null__{pathway}"], out[f"pred_structure__{pathway}"] = y[:, j], pn[:, j], ps[:, j]
    out.to_csv(HERE / "full_structure_oof_predictions.csv.gz", index=False, compression="gzip")
    frame.to_csv(HERE / "full_cardiac_pathway_slopes.csv.gz", index=False, compression="gzip")
    best = ctab.loc[ctab.silhouette.idxmax()]
    return {
        "cohort": {"compounds": len(frame), "scaffold_groups": len(np.unique(groups))},
        "folds": folds, "null": mn, "structure": ms,
        "structure_minus_null_mean_nrmse": {"point": ms["mean_nrmse"] - mn["mean_nrmse"], "compound_bootstrap_ci95": interval(deltas)},
        "response_archetype_feasibility": {"pca_variance_pc1": float(pca.explained_variance_ratio_[0]), "pca_variance_pc1_pc2": float(sum(pca.explained_variance_ratio_[:2])), "best_k": int(best.k), "best_silhouette": float(best.silhouette), "smallest_cluster_n": int(best.smallest_cluster_n)},
        "cardiac_gene_coverage": coverage,
    }


def main():
    base = load_base()
    negative, moderators, pathways = negative_transfer(base)
    full = full_structure(base)
    result = {
        "status": "exploratory_non_confirmatory_complete", "seed": SEED,
        "bootstrap_replicates": B, "negative_transfer": negative,
        "full_structure_candidate": full,
        "controlled_moderator_family": moderators.to_dict("records"),
        "boundaries": ["Fixed-OOF bootstrap is descriptive, not confirmatory.", "Six continuous moderator tests are exploratory with BH adjustment.", "Clustering is in-sample feasibility only.", "No functional cardiomyocyte phenotype is modeled."],
        "inputs": {"analysis_matrix": str(BASE / "analysis_matrix.csv.gz"), "oof_predictions": str(BASE / "oof_predictions.csv.gz")},
        "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
    }
    (HERE / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"negative_transfer": negative, "full_structure_candidate": full, "moderators": result["controlled_moderator_family"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
