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
COUNTS = ["Cells_Number_Object_Number", "Cytoplasm_Number_Object_Number", "Nuclei_Number_Object_Number"]
LINES = ["A549", "ES2", "HCC44"]
CHANNELS = ["AGP", "DNA", "ER", "Mito", "RNA"]


def rho(a: np.ndarray, b: np.ndarray) -> float:
    return math.nan if np.std(a) == 0 or np.std(b) == 0 else float(spearmanr(a, b).statistic)


def split(meta: pd.DataFrame, kind: str):
    if kind == "gene_holdout":
        groups = meta.Metadata_pert_name.str.replace(r"-[^-]+$", "", regex=True)
        yield from ((f"gene_{i+1}", tr, te) for i, (tr, te) in enumerate(GroupKFold(5).split(meta, groups=groups)))
    else:
        for line in LINES:
            te = np.flatnonzero(meta.Metadata_cell_line.to_numpy() == line)
            tr = np.flatnonzero(meta.Metadata_cell_line.to_numpy() != line)
            yield f"holdout_{line}", tr, te


def fit_block(x: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(x[tr])
    a, b = scaler.transform(x[tr]), scaler.transform(x[te])
    pca = PCA(n_components=min(64, len(tr) - 1, x.shape[1]), svd_solver="randomized", random_state=0).fit(a)
    return Ridge(alpha=10).fit(pca.transform(a), y[tr]).predict(pca.transform(b))


def main() -> None:
    summary = pd.read_csv(HERE / "cellhealth_endpoint_context_summary.csv")
    old = pd.read_csv(PROJECT / "analysis" / "CellHealth_confirmation_2026-09-08_v1" / "endpoint_summary.csv")
    median_positive = old[old.consensus.eq("median")].groupby("target").morphology_minus_count_median.median().gt(0)
    wide = summary.pivot(index="target", columns="split_type")
    keep = ((wide[("morphology", "gene_holdout")] >= .20) & (wide[("morphology", "cell_line_holdout")] >= .20) &
            (wide[("morphology_minus_count", "gene_holdout")] > .05) & (wide[("morphology_minus_count", "cell_line_holdout")] > .05) &
            (wide[("observed_minus_permuted", "gene_holdout")] > .10) & (wide[("observed_minus_permuted", "cell_line_holdout")] > .10) &
            (wide[("morphology_minus_count_positive_fraction", "gene_holdout")] >= 2/3) &
            (wide[("morphology_minus_count_positive_fraction", "cell_line_holdout")] >= 2/3))
    targets = list(wide.index[keep & median_positive.reindex(wide.index).fillna(False)])

    x = pd.read_csv(DATA / "cell_painting_modz.tsv.gz", sep="\t").set_index("Metadata_profile_id").sort_index()
    y = pd.read_csv(DATA / "cell_health_modz.tsv.gz", sep="\t").set_index("Metadata_profile_id").sort_index()
    meta = x[["Metadata_pert_name", "Metadata_cell_line"]].reset_index(drop=True)
    columns = [c for c in x.columns if not c.startswith("Metadata_") and c not in COUNTS]
    blocks = {"full": columns}
    for channel in CHANNELS:
        blocks[f"minus_{channel}"] = [c for c in columns if f"_{channel}" not in c]
    ymat = y[targets].to_numpy(float)
    records = []
    for kind in ["gene_holdout", "cell_line_holdout"]:
        for fold, tr, te in split(meta, kind):
            for block, cols in blocks.items():
                pred = fit_block(x[cols].to_numpy(float), ymat, tr, te)
                for local, idx in enumerate(te):
                    for j, target in enumerate(targets):
                        records.append({"split_type": kind, "fold": fold, "cell_line": meta.iloc[idx].Metadata_cell_line,
                                        "row": int(idx), "target": target, "block": block,
                                        "y": ymat[idx, j], "prediction": pred[local, j]})
    raw = pd.DataFrame(records)
    raw.to_csv(HERE / "cellhealth_channel_ablation_predictions.csv.gz", index=False, compression="gzip")
    scores = []
    for keys, z in raw.groupby(["split_type", "cell_line", "target", "block"], sort=False):
        scores.append(dict(zip(["split_type", "cell_line", "target", "block"], keys)) | {"rho": rho(z.y.to_numpy(), z.prediction.to_numpy())})
    scores = pd.DataFrame(scores)
    med = scores.groupby(["split_type", "target", "block"], as_index=False).rho.median()
    pivot = med.pivot(index=["split_type", "target"], columns="block", values="rho").reset_index()
    for channel in CHANNELS:
        pivot[f"loss_when_removing_{channel}"] = pivot["full"] - pivot[f"minus_{channel}"]
    pivot.to_csv(HERE / "cellhealth_channel_ablation_summary.csv", index=False)
    channel_summary = []
    for kind, z in pivot.groupby("split_type"):
        for channel in CHANNELS:
            v = z[f"loss_when_removing_{channel}"]
            channel_summary.append({"split_type": kind, "channel": channel, "median_rho_loss": float(v.median()),
                                    "positive_loss_fraction": float((v > 0).mean())})
    result = {"status": "exploratory_mechanistic_localization_not_causal", "retained_targets": targets,
              "channel_summary": channel_summary,
              "boundary": "Leave-one-channel-out effects are predictive localization, not biological mediation or causal mechanism."}
    (HERE / "cellhealth_ablation_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
