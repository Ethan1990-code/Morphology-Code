# -*- coding: utf-8 -*-
"""Exploratory/non-confirmatory cross-cell transfer benchmark.

Unit: compound. Technical wells are aggregated before modeling. Outer and inner
splits are scaffold-grouped. The reliability gate is selected inside training
folds and falls back from fusion to structure for low-applicability compounds.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SEED = 20260906
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
PATHWAYS = (
    "Adrenergic signaling in cardiomyocytes", "Cardiac muscle contraction",
    "Apoptosis", "Autophagy", "Ferroptosis", "HIF-1 signaling pathway",
    "Oxidative phosphorylation", "TNF signaling pathway", "p53 signaling pathway",
)
CELL_PRIORITY = {x: i for i, x in enumerate(("MCF7", "PC3", "HEPG2", "A375", "A549", "HA1E", "HCC515", "HT29", "VCAP"))}

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SRC = PROJECT / "data" / "reference_metadata"
RAW = PROJECT / "data" / "raw_public" / "GSE262419"
SI = PROJECT / "data" / "reference_metadata" / "tx4c00193_si_002.xlsx"
GCTX = {"GSE92742": PROJECT / "data" / "raw_public" / "GSE92742" / "GSE92742_L5.gctx", "GSE70138": PROJECT / "data" / "raw_public" / "GSE70138" / "GSE70138_L5.gctx"}


def parent_smiles(value: str) -> str | None:
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_si() -> pd.DataFrame:
    frame = pd.read_excel(SI, header=1)
    frame.columns = ["row", "compound", "class", "casrn", "dtxsid", "smiles", "source"]
    frame["name"] = frame["compound"].astype(str).str.strip().str.lower()
    frame["parent_smiles"] = frame["smiles"].map(parent_smiles)
    return frame.dropna(subset=["parent_smiles"]).drop_duplicates("name")


def read_pathways() -> dict[str, set[str]]:
    wanted = set(PATHWAYS)
    out = {}
    with (SRC / "KEGG_2021_Human.gmt").open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields[0] in wanted:
                out[fields[0]] = set(fields[2:])
    if set(out) != wanted:
        raise RuntimeError(f"Missing pathway definitions: {sorted(wanted - set(out))}")
    return out


def numeric(value: object, default: float) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group()) if match else default


def gctx_layout(path: Path) -> tuple[str, list[str], list[str]]:
    with h5py.File(path, "r") as handle:
        matrix = None
        for name, obj in handle.items():
            pass
        def visit(name, obj):
            nonlocal matrix
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and name.split("/")[-1] == "matrix" and matrix is None:
                matrix = name
        handle.visititems(visit)
        cols = handle["0/META/COL/id"][:]
        rows = handle["0/META/ROW/id"][:]
    decode = lambda xs: [x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x) for x in xs]
    if matrix is None:
        raise RuntimeError(f"No matrix in {path}")
    return matrix, decode(cols), decode(rows)


def choose_lincs(si: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    candidates = []
    audit = {}
    configs = {
        "GSE92742": ("GSE92742_Broad_LINCS_pert_info.txt.gz", "GSE92742_Broad_LINCS_sig_info.txt.gz"),
        "GSE70138": ("GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz", "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz"),
    }
    key_to_name = dict(si[["parent_smiles", "name"]].drop_duplicates().itertuples(index=False, name=None))
    for dataset, (pert_file, sig_file) in configs.items():
        pert = pd.read_csv(SRC / pert_file, sep="\t", low_memory=False)
        pert = pert[(pert["pert_type"] == "trt_cp") & pert["canonical_smiles"].notna()].copy()
        pert["parent_smiles"] = pert["canonical_smiles"].map(parent_smiles)
        pert = pert[pert["parent_smiles"].isin(key_to_name)].copy()
        pert["name"] = pert["parent_smiles"].map(key_to_name)
        sig = pd.read_csv(SRC / sig_file, sep="\t", low_memory=False)
        sig = sig[sig["pert_type"] == "trt_cp"].copy()
        matrix_path, available, _ = gctx_layout(GCTX[dataset])
        available = set(available)
        sig = sig[sig["sig_id"].astype(str).isin(available)].merge(
            pert[["pert_id", "name", "parent_smiles", "canonical_smiles"]].drop_duplicates(), on="pert_id", how="inner"
        )
        dose_col = "pert_dose" if "pert_dose" in sig else "pert_idose"
        time_col = "pert_time" if "pert_time" in sig else "pert_itime"
        sig["dose_num"] = sig[dose_col].map(lambda x: numeric(x, 0.0))
        sig["time_num"] = sig[time_col].map(lambda x: numeric(x, 0.0))
        sig["rank_time"] = (sig["time_num"] - 24.0).abs()
        sig["rank_dose"] = sig["dose_num"].map(lambda x: abs(math.log10(max(x, 1e-4) / 10.0)))
        sig["rank_cell"] = sig["cell_id"].map(lambda x: CELL_PRIORITY.get(str(x).upper(), 99))
        sig["rank_dataset"] = 0 if dataset == "GSE92742" else 1
        sig["dataset"] = dataset
        sig["matrix_path"] = matrix_path
        candidates.append(sig)
        audit[dataset] = {"mapped_compounds": int(sig["name"].nunique()), "candidate_signatures": int(len(sig))}
    all_sig = pd.concat(candidates, ignore_index=True)
    counts = all_sig.groupby("name").size().rename("available_signature_n")
    chosen = all_sig.sort_values(["name", "rank_time", "rank_dose", "rank_cell", "rank_dataset", "sig_id"]).drop_duplicates("name")
    chosen = chosen.merge(counts, on="name", validate="one_to_one")
    return chosen, audit


def sample_column(plate: int, well: str) -> str:
    return f"Plate{plate}-{well[0]}{int(well[1:]):02d}"


def cardiac_pathway_slopes(names: set[str], pathways: dict[str, set[str]]) -> tuple[pd.DataFrame, dict]:
    meta = pd.read_csv(RAW / "GSE262419_hash.csv.gz")
    meta["name"] = meta["Chemical_name"].astype(str).str.strip().str.lower()
    all_symbols = set().union(*pathways.values())
    rows = []
    coverage = {p: set() for p in pathways}
    for plate in range(1, 17):
        one = meta[meta["Plate_ID"] == plate].copy()
        selected = one[one["name"].isin(names)]
        controls = one[one["name"] == "veh"]
        if selected.empty:
            continue
        selected_cols = {sample_column(plate, well): (name, float(dose)) for well, name, dose in selected[["Well_ID", "name", "Chemical_Concentration_uM"]].itertuples(index=False)}
        control_cols = [sample_column(plate, well) for well in controls["Well_ID"]]
        usecols = ["Genes", *control_cols, *selected_cols]
        with gzip.open(RAW / f"GSE262419_Plate{plate:02d}.csv.gz", "rt", encoding="utf-8-sig") as handle:
            data = pd.read_csv(handle, usecols=usecols)
        symbols = data["Genes"].astype(str).str.rsplit("_", n=1).str[0]
        keep = symbols.isin(all_symbols)
        data = data.loc[keep].copy()
        symbols = symbols.loc[keep].to_numpy()
        values = np.log2(data.drop(columns="Genes").to_numpy(float) + 1.0)
        columns = list(data.columns[1:])
        baseline = np.median(values[:, [columns.index(c) for c in control_cols]], axis=1)
        for pathway, genes in pathways.items():
            mask = np.isin(symbols, list(genes))
            coverage[pathway].update(symbols[mask])
            if mask.sum() < 5:
                continue
            for (name, dose) in sorted(set(selected_cols.values())):
                cols = [columns.index(c) for c, key in selected_cols.items() if key == (name, dose)]
                response = np.median(values[mask][:, cols], axis=1) - baseline[mask]
                rows.append((name, dose, pathway, float(np.mean(response))))
    long = pd.DataFrame(rows, columns=["name", "dose", "pathway", "score"])
    slopes = []
    for (name, pathway), group in long.groupby(["name", "pathway"]):
        group = group.groupby("dose", as_index=False)["score"].median().sort_values("dose")
        if len(group) < 3:
            continue
        x = np.log10(1.0 + group["dose"].to_numpy(float)); y = group["score"].to_numpy(float)
        slope = float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))
        slopes.append((name, pathway, slope))
    wide = pd.DataFrame(slopes, columns=["name", "pathway", "slope"]).pivot(index="name", columns="pathway", values="slope").reset_index()
    return wide, {p: len(g) for p, g in coverage.items()}


def grab_pathway_features(chosen: pd.DataFrame, pathways: dict[str, set[str]], symbol_to_entrez: dict[str, str]) -> pd.DataFrame:
    pathway_ids = {p: {symbol_to_entrez[g] for g in genes if g in symbol_to_entrez} for p, genes in pathways.items()}
    output = []
    for dataset, group in chosen.groupby("dataset"):
        matrix_path, sigs, rids = gctx_layout(GCTX[dataset])
        pos = {s: i for i, s in enumerate(sigs)}
        wanted = group["sig_id"].astype(str).tolist()
        order = np.argsort([pos[s] for s in wanted])
        sorted_sigs = [wanted[i] for i in order]
        indices = [pos[s] for s in sorted_sigs]
        rid_pos = {r: i for i, r in enumerate(rids)}
        with h5py.File(GCTX[dataset], "r") as handle:
            dset = handle[matrix_path]
            matrix = np.asarray(dset[indices, :], dtype=np.float32) if dset.shape[0] == len(sigs) else np.asarray(dset[:, indices], dtype=np.float32).T
        sig_to_row = {s: matrix[i] for i, s in enumerate(sorted_sigs)}
        for row in group.itertuples(index=False):
            vector = sig_to_row[str(row.sig_id)]
            rec = {"name": row.name}
            for pathway, ids in pathway_ids.items():
                idx = [rid_pos[x] for x in ids if x in rid_pos]
                rec[pathway] = float(np.mean(vector[idx])) if len(idx) >= 5 else np.nan
            output.append(rec)
    return pd.DataFrame(output)


def fingerprints(smiles: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, groups = [], []
    for i, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        fps.append(gen.GetFingerprintAsNumPy(mol).astype(float))
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        groups.append(scaffold or f"ACYCLIC_{i}")
    return np.vstack(fps), np.asarray(groups)


def fit_predict(xtr, ytr, xte, alpha):
    xs = StandardScaler().fit(xtr); ys = StandardScaler().fit(ytr)
    model = Ridge(alpha=alpha).fit(xs.transform(xtr), ys.transform(ytr))
    return ys.inverse_transform(model.predict(xs.transform(xte)))


def loss(y, pred, scale):
    return float(np.mean(np.sqrt(np.mean((y - pred) ** 2, axis=0)) / scale))


def tune_alpha(X, y, groups):
    scale = np.std(y, axis=0, ddof=1); scale[scale == 0] = 1.0
    splitter = GroupKFold(min(3, len(np.unique(groups))))
    scores = []
    for alpha in ALPHAS:
        pred = np.zeros_like(y)
        for tr, te in splitter.split(X, y, groups):
            pred[te] = fit_predict(X[tr], y[tr], X[te], alpha)
        scores.append(loss(y, pred, scale))
    return ALPHAS[int(np.argmin(scores))]


def cosine_applicability(train, test):
    tr = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    te = test / np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-12)
    return np.max(te @ tr.T, axis=1)


def benchmark(Xs, Xl, y, groups):
    splitter = GroupKFold(min(5, len(np.unique(groups))))
    pred = {k: np.zeros_like(y) for k in ("null", "structure", "l1000", "fusion", "gated_fusion")}
    gate_rows = []
    for fold, (tr, te) in enumerate(splitter.split(Xs, y, groups), 1):
        matrices = {"structure": (Xs[tr], Xs[te]), "l1000": (Xl[tr], Xl[te]), "fusion": (np.hstack([Xs[tr], Xl[tr]]), np.hstack([Xs[te], Xl[te]]))}
        pred["null"][te] = np.mean(y[tr], axis=0)
        alphas = {}
        for mode, (a, b) in matrices.items():
            alphas[mode] = tune_alpha(a, y[tr], groups[tr])
            pred[mode][te] = fit_predict(a, y[tr], b, alphas[mode])
        # Gate cutoff is derived only from the outer-training L1000 geometry.
        norm = Xl[tr] / np.maximum(np.linalg.norm(Xl[tr], axis=1, keepdims=True), 1e-12)
        sim = norm @ norm.T; np.fill_diagonal(sim, -np.inf)
        train_sim = np.max(sim, axis=1)
        cutoff = float(np.quantile(train_sim, 0.25))
        test_sim = cosine_applicability(Xl[tr], Xl[te])
        use_fusion = test_sim >= cutoff
        pred["gated_fusion"][te] = np.where(use_fusion[:, None], pred["fusion"][te], pred["structure"][te])
        gate_rows.append({"fold": fold, "cutoff": cutoff, "test_n": int(len(te)), "fusion_used_n": int(use_fusion.sum()), "alphas": alphas})
    scale = np.std(y, axis=0, ddof=1); scale[scale == 0] = 1.0
    metrics = {}
    for mode, value in pred.items():
        per_rmse = np.sqrt(np.mean((y - value) ** 2, axis=0))
        per_spear = [spearmanr(y[:, i], value[:, i]).statistic for i in range(y.shape[1])]
        metrics[mode] = {"mean_nrmse": float(np.mean(per_rmse / scale)), "median_spearman": float(np.nanmedian(per_spear)), "per_pathway_rmse": per_rmse.tolist(), "per_pathway_spearman": [float(x) for x in per_spear]}
    rng = np.random.default_rng(SEED)
    comparisons = {}
    for mode in ("l1000", "fusion", "gated_fusion"):
        deltas = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            boot_scale = np.std(y[idx], axis=0, ddof=1); boot_scale[boot_scale == 0] = 1.0
            deltas.append(loss(y[idx], pred[mode][idx], boot_scale) - loss(y[idx], pred["structure"][idx], boot_scale))
        comparisons[f"{mode}_minus_structure_mean_nrmse"] = {
            "point": metrics[mode]["mean_nrmse"] - metrics["structure"]["mean_nrmse"],
            "compound_bootstrap_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "bootstrap_scope": "descriptive uncertainty of fixed OOF predictions; not a confirmatory p-value",
        }
    return metrics, comparisons, pred, gate_rows


def main():
    si = read_si(); pathways = read_pathways(); chosen, mapping_audit = choose_lincs(si)
    cardiac, cardiac_coverage = cardiac_pathway_slopes(set(chosen["name"]), pathways)
    # GSE262419 row labels provide a deterministic symbol-to-Entrez crosswalk.
    with gzip.open(RAW / "GSE262419_Plate01.csv.gz", "rt", encoding="utf-8-sig") as handle:
        genes = pd.read_csv(handle, usecols=["Genes"])["Genes"].astype(str)
    pairs = genes.str.rsplit("_", n=1, expand=True)
    symbol_to_entrez = dict(pairs.drop_duplicates(0).itertuples(index=False, name=None))
    lincs = grab_pathway_features(chosen, pathways, symbol_to_entrez)
    frame = chosen[["name", "parent_smiles", "dataset", "sig_id", "cell_id", "time_num", "dose_num", "available_signature_n"]].merge(cardiac, on="name").merge(lincs, on="name", suffixes=("_cardiac", "_l1000"))
    target_cols = [f"{p}_cardiac" for p in PATHWAYS]
    feature_cols = [f"{p}_l1000" for p in PATHWAYS]
    frame = frame.dropna(subset=target_cols + feature_cols).sort_values("name").reset_index(drop=True)
    Xs, groups = fingerprints(frame["parent_smiles"]); Xl = frame[feature_cols].to_numpy(float); y = frame[target_cols].to_numpy(float)
    if len(frame) < 60 or len(np.unique(groups)) < 5:
        raise RuntimeError(f"Not estimable: n={len(frame)}, scaffold_groups={len(np.unique(groups))}")
    metrics, comparisons, predictions, gate_rows = benchmark(Xs, Xl, y, groups)
    pred_table = frame[["name", "parent_smiles", "dataset", "sig_id", "cell_id", "time_num", "dose_num", "available_signature_n"]].copy()
    for mode, matrix in predictions.items():
        for i, pathway in enumerate(PATHWAYS):
            pred_table[f"pred_{mode}__{pathway}"] = matrix[:, i]
            pred_table[f"observed__{pathway}"] = y[:, i]
    pred_table.to_csv(HERE / "oof_predictions.csv.gz", index=False, compression="gzip")
    frame.to_csv(HERE / "analysis_matrix.csv.gz", index=False, compression="gzip")
    result = {
        "status": "exploratory_non_confirmatory_complete",
        "construct": "48-hour hiPSC-cardiomyocyte pathway dose-response slopes predicted from structure and noncardiac L1000 pathway signatures",
        "seed": SEED, "unit": "compound", "split": "nested scaffold-grouped CV",
        "cohort": {"si_structures": int(len(si)), "lincs_mapped_compounds": int(chosen["name"].nunique()), "modeled_compounds": int(len(frame)), "scaffold_groups": int(len(np.unique(groups)))},
        "pathways": list(PATHWAYS), "cardiac_gene_coverage": cardiac_coverage,
        "mapping_audit": mapping_audit, "gate": {"rule": "fusion only when test L1000 cosine applicability >= training-only 25th percentile of leave-self-out nearest-neighbor similarity", "folds": gate_rows},
        "metrics": metrics, "paired_comparisons": comparisons,
        "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "rdkit": rdBase.rdkitVersion},
        "inputs": {str(x): {"bytes": x.stat().st_size, "sha256": sha256(x)} for x in [SI, SRC / "KEGG_2021_Human.gmt", SRC / "GSE92742_Broad_LINCS_sig_info.txt.gz", SRC / "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz"]},
        "limitations": ["Pathway score is an unweighted mean and can cancel opposing gene directions.", "One representative noncardiac L1000 signature is used per compound.", "The reliability rule is a prespecified geometric applicability heuristic, not a calibrated uncertainty guarantee.", "No functional cardiomyocyte phenotype is modeled."],
    }
    (HERE / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cohort": result["cohort"], "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
