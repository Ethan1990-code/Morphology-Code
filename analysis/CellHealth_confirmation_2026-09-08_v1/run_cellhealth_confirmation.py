from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data" / "positive_controls_2026-09-08" / "CellHealth_v2.0"
AUTHOR = DATA / "author_reference"
OUT = PROJECT / "analysis" / "CellHealth_confirmation_2026-09-08_v1"
OUT.mkdir(parents=True, exist_ok=True)

META = ["Metadata_profile_id", "Metadata_pert_name", "Metadata_cell_line"]
COUNTS = [
    "Cells_Number_Object_Number",
    "Cytoplasm_Number_Object_Number",
    "Nuclei_Number_Object_Number",
]
CONSENSUS = ["modz", "median"]
CELL_LINES = ["A549", "ES2", "HCC44"]
RIDGE_ALPHA = 10.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_pair(consensus: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_csv(DATA / f"cell_painting_{consensus}.tsv.gz", sep="\t")
    y = pd.read_csv(DATA / f"cell_health_{consensus}.tsv.gz", sep="\t")
    assert x["Metadata_profile_id"].is_unique
    assert y["Metadata_profile_id"].is_unique
    x = x.set_index("Metadata_profile_id").sort_index()
    y = y.set_index("Metadata_profile_id").sort_index()
    assert x.index.equals(y.index)
    for col in ["Metadata_pert_name", "Metadata_cell_line"]:
        assert x[col].equals(y[col])
    return x, y


def gene_group(values: pd.Series) -> pd.Series:
    return values.str.replace(r"-[^-]+$", "", regex=True)


def split_iterator(meta: pd.DataFrame, split_type: str):
    n = len(meta)
    if split_type == "gene_holdout":
        groups = gene_group(meta["Metadata_pert_name"])
        for fold, (train, test) in enumerate(GroupKFold(5).split(np.zeros(n), groups=groups)):
            yield f"gene_fold_{fold + 1}", train, test
    elif split_type == "cell_line_holdout":
        for cell_line in CELL_LINES:
            test = np.flatnonzero(meta["Metadata_cell_line"].to_numpy() == cell_line)
            train = np.flatnonzero(meta["Metadata_cell_line"].to_numpy() != cell_line)
            yield f"holdout_{cell_line}", train, test
    else:
        raise ValueError(split_type)


def safe_corr(func, y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) == 0 or np.std(pred) == 0:
        return math.nan
    return float(func(y, pred).statistic)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(mean_squared_error(y, pred) ** 0.5)
    sd = float(np.std(y, ddof=1))
    return {
        "spearman_rho": safe_corr(spearmanr, y, pred),
        "pearson_r": safe_corr(pearsonr, y, pred),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 else math.nan,
        "rmse_over_test_sd": rmse / sd if sd > 0 else math.nan,
    }


def permute_within_cell_line(x: np.ndarray, cell_lines: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = x.copy()
    for cell_line in np.unique(cell_lines):
        idx = np.flatnonzero(cell_lines == cell_line)
        out[idx] = x[rng.permutation(idx)]
    return out


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    return Ridge(alpha=RIDGE_ALPHA).fit(x_train, y_train).predict(x_test)


def run_strict() -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    intake: dict = {}
    for consensus in CONSENSUS:
        x, y = load_pair(consensus)
        meta = x[["Metadata_pert_name", "Metadata_cell_line"]].copy()
        x_features = x.drop(columns=["Metadata_pert_name", "Metadata_cell_line"])
        y_targets = y.drop(columns=["Metadata_pert_name", "Metadata_cell_line"])
        assert all(c in x_features for c in COUNTS)
        morph_cols = [c for c in x_features.columns if c not in COUNTS]
        intake[consensus] = {
            "rows": len(x),
            "guides": int(meta["Metadata_pert_name"].nunique()),
            "gene_groups": int(gene_group(meta["Metadata_pert_name"]).nunique()),
            "cell_lines": meta["Metadata_cell_line"].value_counts().to_dict(),
            "targets": y_targets.shape[1],
            "morphology_features_excluding_counts": len(morph_cols),
            "missing_targets": int(y_targets.isna().sum().sum()),
        }

        count_all = x_features[COUNTS].to_numpy(dtype=float)
        morph_all = x_features[morph_cols].to_numpy(dtype=float)
        for split_type in ["gene_holdout", "cell_line_holdout"]:
            for split_number, (fold, train_idx, test_idx) in enumerate(
                split_iterator(meta, split_type), start=1
            ):
                count_pipe = Pipeline([("scale", StandardScaler())])
                morph_pipe = Pipeline(
                    [("scale", StandardScaler()), ("pca", PCA(n_components=64, random_state=0))]
                )
                count_train = count_pipe.fit_transform(count_all[train_idx])
                count_test = count_pipe.transform(count_all[test_idx])
                morph_train = morph_pipe.fit_transform(morph_all[train_idx])
                morph_test = morph_pipe.transform(morph_all[test_idx])
                fusion_train = np.column_stack([morph_train, count_train])
                fusion_test = np.column_stack([morph_test, count_test])
                train_lines = meta.iloc[train_idx]["Metadata_cell_line"].to_numpy()

                for target_number, target in enumerate(y_targets.columns):
                    y_train_all = y_targets.iloc[train_idx][target].to_numpy(dtype=float)
                    y_test_all = y_targets.iloc[test_idx][target].to_numpy(dtype=float)
                    train_ok = np.isfinite(y_train_all)
                    test_ok = np.isfinite(y_test_all)
                    if train_ok.sum() < 20 or test_ok.sum() < 10:
                        continue
                    y_train = y_train_all[train_ok]
                    y_test = y_test_all[test_ok]
                    seed = 10000 * (1 + CONSENSUS.index(consensus)) + 100 * split_number + target_number
                    perm_train = permute_within_cell_line(
                        morph_train[train_ok], train_lines[train_ok], seed
                    )
                    matrices = {
                        "count": (count_train[train_ok], count_test[test_ok]),
                        "morphology": (morph_train[train_ok], morph_test[test_ok]),
                        "fusion": (fusion_train[train_ok], fusion_test[test_ok]),
                        "morphology_permuted": (perm_train, morph_test[test_ok]),
                    }
                    for model, (x_train_model, x_test_model) in matrices.items():
                        pred = fit_predict(x_train_model, y_train, x_test_model)
                        row = {
                            "consensus": consensus,
                            "split_type": split_type,
                            "fold": fold,
                            "target": target,
                            "model": model,
                            "n_train": int(train_ok.sum()),
                            "n_test": int(test_ok.sum()),
                        }
                        row.update(metrics(y_test, pred))
                        rows.append(row)
    return pd.DataFrame(rows), intake


def summarize(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = folds.pivot_table(
        index=["consensus", "split_type", "fold", "target"],
        columns="model",
        values="spearman_rho",
    ).reset_index()
    wide["morphology_minus_count"] = wide["morphology"] - wide["count"]
    wide["fusion_minus_best_simple"] = wide["fusion"] - wide[["morphology", "count"]].max(axis=1)
    wide["observed_minus_permuted"] = wide["morphology"] - wide["morphology_permuted"]
    summary = (
        wide.groupby(["consensus", "split_type", "target"], as_index=False)
        .agg(
            morphology_rho_median=("morphology", "median"),
            count_rho_median=("count", "median"),
            fusion_rho_median=("fusion", "median"),
            permuted_rho_median=("morphology_permuted", "median"),
            morphology_minus_count_median=("morphology_minus_count", "median"),
            morphology_minus_count_positive_fraction=("morphology_minus_count", lambda x: float((x > 0).mean())),
            fusion_minus_best_simple_median=("fusion_minus_best_simple", "median"),
            observed_minus_permuted_median=("observed_minus_permuted", "median"),
        )
    )

    mapping = pd.read_csv(AUTHOR / "feature_mapping_annotated.csv")[["id", "measurement"]]
    summary = summary.merge(mapping, left_on="target", right_on="id", how="left").drop(columns="id")
    assert summary["measurement"].notna().all()
    family = (
        summary.groupby(["consensus", "split_type", "measurement"], as_index=False)
        .agg(
            endpoints=("target", "nunique"),
            morphology_rho_median=("morphology_rho_median", "median"),
            morphology_minus_count_median=("morphology_minus_count_median", "median"),
            fusion_minus_best_simple_median=("fusion_minus_best_simple_median", "median"),
            observed_minus_permuted_median=("observed_minus_permuted_median", "median"),
        )
    )

    primary = summary.query("consensus == 'modz'").copy()
    pivot = primary.pivot(index="target", columns="split_type")
    median_sensitivity = (
        summary.query("consensus == 'median'")
        .groupby("target")["morphology_minus_count_median"]
        .median()
    )
    decisions = []
    for target in sorted(primary.target.unique()):
        def passed(split_type: str) -> bool:
            row = primary[(primary.target == target) & (primary.split_type == split_type)].iloc[0]
            fraction_cutoff = 0.6 if split_type == "gene_holdout" else 2 / 3
            return bool(
                row.morphology_rho_median >= 0.20
                and row.morphology_minus_count_median > 0.05
                and row.observed_minus_permuted_median > 0.10
                and row.morphology_minus_count_positive_fraction >= fraction_cutoff
            )
        gene_pass = passed("gene_holdout")
        cell_pass = passed("cell_line_holdout")
        sensitivity_positive = bool(median_sensitivity.loc[target] > 0)
        target_rows = primary[primary.target == target]
        low_signal = bool(
            target_rows.morphology_rho_median.median() < 0.20
            or target_rows.observed_minus_permuted_median.median() <= 0.10
        )
        if gene_pass and cell_pass and sensitivity_positive:
            decision = "retain_cross_context"
        elif gene_pass ^ cell_pass:
            decision = "retain_context_limited"
        elif not gene_pass and not cell_pass and low_signal:
            decision = "reject"
        else:
            decision = "uncertain"
        decisions.append(
            {
                "target": target,
                "measurement": target_rows.measurement.iloc[0],
                "gene_holdout_pass": gene_pass,
                "cell_line_holdout_pass": cell_pass,
                "median_sensitivity_positive": sensitivity_positive,
                "decision": decision,
            }
        )
    return summary, family, pd.DataFrame(decisions)


def audit_author_outputs() -> dict:
    y = pd.read_csv(DATA / "cell_health_modz.tsv.gz", sep="\t").set_index("Metadata_profile_id")
    pred = pd.read_csv(AUTHOR / "all_model_predictions_modz.tsv", sep="\t").set_index("Metadata_profile_id")
    released = pd.read_csv(AUTHOR / "full_cell_health_regression_modz.tsv.gz", sep="\t")
    targets = [c for c in y.columns if not c.startswith("Metadata_")]
    checks = []
    for target in targets:
        idx = pred.index[(pred["Metadata_data_type"] == "test") & y[target].notna()]
        recomputed = r2_score(y.loc[idx, target], pred.loc[idx, target])
        match = released.query(
            "metric == 'r_two' and data_fit == 'test' and shuffle == 'shuffle_false' and cell_line == 'all' and target == @target"
        )["value"]
        assert len(match) == 1
        checks.append((target, float(recomputed), float(match.iloc[0])))
    frame = pd.DataFrame(checks, columns=["target", "recomputed_r2", "released_r2"])
    return {
        "targets": len(frame),
        "pearson_r": float(frame[["recomputed_r2", "released_r2"]].corr().iloc[0, 1]),
        "max_abs_difference": float((frame.recomputed_r2 - frame.released_r2).abs().max()),
        "released_test_r2_median": float(frame.released_r2.median()),
        "released_test_r2_positive_fraction": float((frame.released_r2 > 0).mean()),
    }


def main() -> None:
    folds, intake = run_strict()
    summary, family, decisions = summarize(folds)
    folds.to_csv(OUT / "fold_results.csv", index=False)
    summary.to_csv(OUT / "endpoint_summary.csv", index=False)
    family.to_csv(OUT / "family_summary.csv", index=False)
    decisions.to_csv(OUT / "endpoint_decisions.csv", index=False)

    primary = summary.query("consensus == 'modz'")
    global_summary = []
    for split_type, group in primary.groupby("split_type"):
        global_summary.append(
            {
                "split_type": split_type,
                "endpoints": int(group.target.nunique()),
                "median_morphology_rho": float(group.morphology_rho_median.median()),
                "median_count_rho": float(group.count_rho_median.median()),
                "median_morphology_minus_count": float(group.morphology_minus_count_median.median()),
                "median_fusion_minus_best_simple": float(group.fusion_minus_best_simple_median.median()),
                "median_observed_minus_permuted": float(group.observed_minus_permuted_median.median()),
            }
        )
    files = [DATA / f"cell_{kind}_{cons}.tsv.gz" for kind in ["painting", "health"] for cons in CONSENSUS]
    result = {
        "status": "exploratory_non_confirmatory",
        "protocol_written_before_results": True,
        "author_output_consistency": audit_author_outputs(),
        "intake": intake,
        "global_primary_summary": global_summary,
        "decision_counts": decisions.decision.value_counts().to_dict(),
        "input_sha256": {p.name: sha256(p) for p in files},
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "boundaries": [
            "fixed effect-size gates are exploratory calibration rules, not formal significance tests",
            "author shuffled control permuted feature positions within rows; strict analysis permutes sample identity within cell line",
            "this analysis is self-audited and not an independent review",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
