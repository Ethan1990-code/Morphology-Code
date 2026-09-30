from __future__ import annotations

import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RAW = PROJECT / "data" / "raw_public" / "GSE262419"
BASE_DIR = PROJECT / "analysis" / "A1_cross_cell_transfer_2026-09-06_v1"
FULL_SLOPES = PROJECT / "analysis" / "A1_negative_transfer_and_candidate_scan_2026-09-06_v2" / "full_cardiac_pathway_slopes.csv.gz"
MAIN_MATRIX = BASE_DIR / "analysis_matrix.csv.gz"
sys.path.insert(0, str(BASE_DIR))
import run_A1_benchmark as base  # noqa: E402

SEED = 20260920
BOOT = 1000
REPLICATES = ["rep1", "rep2", "rep3"]
JACKKNIFE = ["leave_rep1", "leave_rep2", "leave_rep3"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_column(plate: int, well: str) -> str:
    return f"Plate{plate}-{well[0]}{int(well[1:]):02d}"


def linear_slope(dose: np.ndarray, score: np.ndarray) -> float:
    x = np.log10(1 + np.asarray(dose, float))
    y = np.asarray(score, float)
    return float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))


def icc_consistency(matrix: np.ndarray) -> float:
    x = np.asarray(matrix, float)
    n, k = x.shape
    row_mean = x.mean(axis=1)
    col_mean = x.mean(axis=0)
    grand = x.mean()
    ms_row = k * np.sum((row_mean - grand) ** 2) / (n - 1)
    residual = x - row_mean[:, None] - col_mean[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    return float((ms_row - ms_error) / (ms_row + (k - 1) * ms_error))


def paired_rhos(matrix: np.ndarray) -> list[float]:
    return [float(spearmanr(matrix[:, a], matrix[:, b]).statistic) for a, b in combinations(range(matrix.shape[1]), 2)]


def build_scores(names: set[str], pathways: dict[str, set[str]]) -> pd.DataFrame:
    meta = pd.read_csv(RAW / "GSE262419_hash.csv.gz")
    meta["name"] = meta.Chemical_name.astype(str).str.strip().str.lower()
    wanted_symbols = set().union(*pathways.values())
    records = []
    for plate in range(1, 17):
        one = meta[meta.Plate_ID.eq(plate)].copy()
        treated = one[one.name.isin(names)]
        controls = one[one.name.eq("veh")]
        if treated.empty:
            continue
        treatment_cols = {sample_column(plate, w): (n, float(d)) for w, n, d in treated[["Well_ID", "name", "Chemical_Concentration_uM"]].itertuples(index=False)}
        control_cols = sorted(sample_column(plate, w) for w in controls.Well_ID)
        use = ["Genes", *control_cols, *treatment_cols]
        data = pd.read_csv(RAW / f"GSE262419_Plate{plate:02d}.csv.gz", usecols=use)
        symbols = data.Genes.astype(str).str.rsplit("_", n=1).str[0]
        keep = symbols.isin(wanted_symbols)
        symbols = symbols[keep].to_numpy()
        numeric = np.log2(data.loc[keep, use[1:]].to_numpy(float) + 1)
        positions = {column: i for i, column in enumerate(use[1:])}
        masks = {path: np.isin(symbols, list(genes)) for path, genes in pathways.items()}
        groups = {"full": control_cols}
        for r, label in enumerate(REPLICATES):
            groups[label] = control_cols[r::3]
        for r, label in enumerate(JACKKNIFE):
            groups[label] = [column for i, column in enumerate(control_cols) if i % 3 != r]
        for split, ctrl in groups.items():
            baseline = np.median(numeric[:, [positions[c] for c in ctrl]], axis=1)
            for name, dose in sorted(set(treatment_cols.values())):
                cols = sorted(c for c, key in treatment_cols.items() if key == (name, dose))
                if split in REPLICATES:
                    cols = cols[REPLICATES.index(split)::3]
                elif split in JACKKNIFE:
                    omitted = JACKKNIFE.index(split)
                    cols = [column for i, column in enumerate(cols) if i % 3 != omitted]
                if not cols:
                    continue
                response = np.median(numeric[:, [positions[c] for c in cols]], axis=1) - baseline
                for pathway, mask in masks.items():
                    records.append({"name": name, "dose": dose, "pathway": pathway, "split": split,
                                    "score": float(np.mean(response[mask]))})
    return pd.DataFrame(records)


def slope_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, z in scores.groupby(["name", "pathway", "split"]):
        z = z.groupby("dose", as_index=False).score.median().sort_values("dose")
        if len(z) == 3:
            rows.append(dict(zip(["name", "pathway", "split"], keys)) | {"slope": linear_slope(z.dose, z.score)})
    return pd.DataFrame(rows)


def main() -> None:
    reference = pd.read_csv(FULL_SLOPES)
    names = set(reference.name.astype(str).str.lower())
    pathways = base.read_pathways()
    scores = build_scores(names, pathways)
    slopes = slope_table(scores)
    scores.to_csv(HERE / "cardiac_pathway_dose_scores.csv.gz", index=False, compression="gzip")
    slopes.to_csv(HERE / "cardiac_replicate_slopes.csv.gz", index=False, compression="gzip")
    main_names = set(pd.read_csv(MAIN_MATRIX).name.astype(str).str.lower())

    rng = np.random.default_rng(SEED)
    reliability = []
    for cohort, allowed in [("all_462", None), ("main_119", main_names)]:
        for analysis, labels in [("single_well", REPLICATES), ("leave_one_replicate_out", JACKKNIFE)]:
            for pathway in sorted(pathways):
                z = slopes[(slopes.pathway == pathway) & slopes.split.isin(labels)].pivot(index="name", columns="split", values="slope").dropna()
                if allowed is not None:
                    z = z.loc[z.index.intersection(allowed)]
                matrix = z[labels].to_numpy()
                rhos = paired_rhos(matrix)
                boots = []
                for _ in range(BOOT):
                    idx = rng.integers(0, len(matrix), len(matrix))
                    boots.append(float(np.median(paired_rhos(matrix[idx]))))
                reliability.append({"cohort": cohort, "analysis": analysis, "pathway": pathway, "n_compounds": len(z),
                                    "median_pairwise_spearman": float(np.median(rhos)),
                                    "minimum_pairwise_spearman": float(np.min(rhos)), "bootstrap_ci_low": float(np.quantile(boots, .025)),
                                    "bootstrap_ci_high": float(np.quantile(boots, .975)), "icc_c1": icc_consistency(matrix),
                                    "three_way_sign_concordance": float(np.mean((matrix > 0).all(axis=1) | (matrix < 0).all(axis=1)))})
    reliability = pd.DataFrame(reliability)
    reliability.to_csv(HERE / "cardiac_slope_reliability_summary.csv", index=False)

    full = scores[scores.split.eq("full")]
    leverage_rows = []
    dose_labels = {0.2: "without_low_0.2", 1.0: "without_mid_1", 10.0: "without_high_10"}
    for (name, pathway), z in full.groupby(["name", "pathway"]):
        z = z.groupby("dose", as_index=False).score.median().sort_values("dose")
        if len(z) != 3:
            continue
        row = {"name": name, "pathway": pathway, "full": linear_slope(z.dose, z.score)}
        for omitted, label in dose_labels.items():
            zz = z[~np.isclose(z.dose, omitted)]
            row[label] = linear_slope(zz.dose, zz.score)
        leverage_rows.append(row)
    leverage = pd.DataFrame(leverage_rows)
    leverage.to_csv(HERE / "cardiac_dose_leverage_compound.csv.gz", index=False, compression="gzip")
    dose_summary = []
    for cohort, allowed in [("all_462", None), ("main_119", main_names)]:
        current = leverage if allowed is None else leverage[leverage.name.isin(allowed)]
        for pathway, z in current.groupby("pathway"):
            for label in dose_labels.values():
                dose_summary.append({"cohort": cohort, "pathway": pathway, "omission": label, "n_compounds": len(z),
                                     "spearman_with_full": float(spearmanr(z.full, z[label]).statistic),
                                     "sign_concordance_with_full": float(np.mean(np.sign(z.full) == np.sign(z[label]))),
                                     "median_absolute_change": float(np.median(np.abs(z.full - z[label])))})
    dose_summary = pd.DataFrame(dose_summary)
    dose_summary.to_csv(HERE / "cardiac_dose_leverage_summary.csv", index=False)

    slope_wide = slopes[slopes.split.eq("full")].pivot(index="name", columns="pathway", values="slope").dropna()
    saved = reference.copy()
    saved["name"] = saved.name.astype(str).str.lower()
    saved = saved.set_index("name")[slope_wide.columns].loc[slope_wide.index]
    reconstruction_max_abs = float(np.max(np.abs(saved.to_numpy() - slope_wide.to_numpy())))
    factor_rows, loading_rows = [], []
    for cohort, z in [("all_462", slope_wide), ("main_119", slope_wide.loc[slope_wide.index.intersection(main_names)])]:
        for scaling, matrix in [("centered_raw", z.to_numpy()), ("z_scored", StandardScaler().fit_transform(z))]:
            centered = matrix - matrix.mean(axis=0)
            pca = PCA().fit(centered)
            factor_rows.append({"cohort": cohort, "n_compounds": len(z), "scaling": scaling,
                                "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
                                "pc2_explained_variance": float(pca.explained_variance_ratio_[1])})
            for pathway, loading in zip(z.columns, pca.components_[0]):
                loading_rows.append({"cohort": cohort, "scaling": scaling, "pathway": pathway, "pc1_loading": float(loading)})
    factors = pd.DataFrame(factor_rows)
    factors.to_csv(HERE / "cardiac_common_factor_summary.csv", index=False)
    pd.DataFrame(loading_rows).to_csv(HERE / "cardiac_common_factor_loadings.csv", index=False)

    result = {
        "status": "exploratory_non_confirmatory_complete",
        "technical_replicates_not_biological_replicates": True,
        "doses_um": [0.2, 1.0, 10.0],
        "saved_slope_reconstruction_max_abs": reconstruction_max_abs,
        "main_119_single_well_all_pathways_median_spearman": float(reliability.query("cohort == 'main_119' and analysis == 'single_well'").median_pairwise_spearman.median()),
        "main_119_single_well_all_pathways_median_icc_c1": float(reliability.query("cohort == 'main_119' and analysis == 'single_well'").icc_c1.median()),
        "main_119_leave_one_out_all_pathways_median_spearman": float(reliability.query("cohort == 'main_119' and analysis == 'leave_one_replicate_out'").median_pairwise_spearman.median()),
        "main_119_leave_one_out_all_pathways_median_icc_c1": float(reliability.query("cohort == 'main_119' and analysis == 'leave_one_replicate_out'").icc_c1.median()),
        "main_119_leave_one_out_all_pathways_median_sign_concordance": float(reliability.query("cohort == 'main_119' and analysis == 'leave_one_replicate_out'").three_way_sign_concordance.median()),
        "main_119_worst_omission_median_spearman": float(dose_summary.query("cohort == 'main_119'").groupby("omission").spearman_with_full.median().min()),
        "common_factor": factors.to_dict("records"),
        "input_sha256": {"metadata": sha256(RAW / "GSE262419_hash.csv.gz"), "reference_slopes": sha256(FULL_SLOPES)},
        "boundary": "Technical repeatability and three-dose robustness do not establish cross-donor, cross-laboratory, or nonlinear dose-response reliability."
    }
    (HERE / "cardiac_reliability_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
