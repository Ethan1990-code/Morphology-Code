from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data" / "positive_controls_2026-09-08" / "EveBio"
SEEDS = [17, 43, 101]
N_SPLITS = 3
PCA_COMPONENTS = 64
LABELS = list(pd.read_csv(DATA / "CP_all_EveBio.csv", nrows=0).columns[1:25])


def coalesce_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby("InChIKey", sort=False):
        row = group.iloc[0].copy()
        for label in LABELS:
            observed = group[label].dropna().unique()
            if len(observed) > 1:
                raise ValueError(f"Conflicting observed labels for {label}")
            row[label] = observed[0] if len(observed) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def smiles_by_inchikey(source: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    for smiles in source["SMILES"].dropna().astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mapping.setdefault(Chem.MolToInchiKey(mol), Chem.MolToSmiles(mol, canonical=True))
    return mapping


def fingerprints_and_scaffolds(keys: list[str], mapping: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprints, scaffolds = [], []
    for key in keys:
        mol = Chem.MolFromSmiles(mapping[key])
        array = np.zeros(1024, dtype=np.float64)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), array)
        fingerprints.append(array)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.append(scaffold if scaffold else f"ACYCLIC::{key}")
    return np.vstack(fingerprints), np.asarray(scaffolds, dtype=object)


def fit_probability(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def scale(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(train), scaler.transform(test)


def morphology_projection(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_scaled, test_scaled = scale(train, test)
    components = min(PCA_COMPONENTS, len(train) - 1, train.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=0)
    return pca.fit_transform(train_scaled), pca.transform(test_scaled)


def score(y: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def distribution(values: pd.Series) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q025": float(values.quantile(0.025)),
        "q975": float(values.quantile(0.975)),
        "positive_fraction": float((values > 0).mean()),
    }


def main() -> None:
    full = coalesce_duplicates(pd.read_csv(DATA / "CP_all_EveBio.csv"))
    smiles_source = pd.read_csv(DATA / "evebio_assay_data_smiles.csv")
    mapping = smiles_by_inchikey(smiles_source)
    missing_structure = sorted(set(full["InChIKey"]) - set(mapping))
    if missing_structure:
        raise ValueError(f"Missing structures for {len(missing_structure)} entities")

    morphology_columns = list(full.columns[28:])
    count_columns = [
        "Cells_Number_Object_Number",
        "Cytoplasm_Number_Object_Number",
        "Nuclei_Number_Object_Number",
    ]
    keys = full["InChIKey"].astype(str).tolist()
    structures, scaffolds = fingerprints_and_scaffolds(keys, mapping)
    morphology = full[morphology_columns].to_numpy(dtype=np.float64)
    counts = full[count_columns].to_numpy(dtype=np.float64)

    rows = []
    for assay in LABELS:
        valid = full[assay].notna().to_numpy()
        y = full.loc[valid, assay].astype(int).to_numpy()
        x_structure = structures[valid]
        x_morphology = morphology[valid]
        x_count = counts[valid]
        groups = scaffolds[valid]

        for seed in SEEDS:
            splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            permutation = np.random.default_rng(seed).permutation(len(y))
            permuted_morphology = x_morphology[permutation]
            for fold, (train, test) in enumerate(splitter.split(x_structure, y, groups), start=1):
                train_count, test_count = scale(x_count[train], x_count[test])
                train_morph, test_morph = morphology_projection(x_morphology[train], x_morphology[test])
                train_perm, test_perm = morphology_projection(permuted_morphology[train], permuted_morphology[test])
                matrices = {
                    "count": (train_count, test_count),
                    "structure": (x_structure[train], x_structure[test]),
                    "structure_count": (
                        np.hstack([x_structure[train], train_count]),
                        np.hstack([x_structure[test], test_count]),
                    ),
                    "morphology64": (train_morph, test_morph),
                    "fusion64": (
                        np.hstack([x_structure[train], train_morph]),
                        np.hstack([x_structure[test], test_morph]),
                    ),
                    "permuted_fusion64": (
                        np.hstack([x_structure[train], train_perm]),
                        np.hstack([x_structure[test], test_perm]),
                    ),
                }
                for model, (x_train, x_test) in matrices.items():
                    probability = fit_probability(x_train, y[train], x_test)
                    rows.append({
                        "assay": assay,
                        "seed": seed,
                        "fold": fold,
                        "model": model,
                        "train_n": int(len(train)),
                        "test_n": int(len(test)),
                        "train_positive": int(y[train].sum()),
                        "test_positive": int(y[test].sum()),
                        "scaffold_overlap": int(len(set(groups[train]) & set(groups[test]))),
                        **score(y[test], probability),
                    })

    folds = pd.DataFrame(rows)
    folds.to_csv(HERE / "scaffold_fold_results.csv", index=False, encoding="utf-8-sig")
    summaries, decisions = [], {}
    for assay, assay_frame in folds.groupby("assay", sort=True):
        pivot = assay_frame.pivot(index=["seed", "fold"], columns="model", values="average_precision")
        best_simple = pivot[["count", "structure", "structure_count"]].max(axis=1)
        delta = pivot["fusion64"] - best_simple
        permuted_delta = pivot["permuted_fusion64"] - best_simple
        morphology_delta = pivot["morphology64"] - pivot["count"]
        record = {
            "assay": assay,
            "fusion_minus_best_simple_median_ap": float(delta.median()),
            "fusion_minus_best_simple_positive_fraction": float((delta > 0).mean()),
            "morphology_minus_count_median_ap": float(morphology_delta.median()),
            "permuted_fusion_minus_best_simple_median_ap": float(permuted_delta.median()),
            "observed_minus_permuted_median_ap": float((pivot["fusion64"] - pivot["permuted_fusion64"]).median()),
        }
        summaries.append(record)
        decisions[assay] = "pass" if (
            record["fusion_minus_best_simple_median_ap"] > 0.02
            and record["fusion_minus_best_simple_positive_fraction"] >= 0.8
            and record["observed_minus_permuted_median_ap"] > 0
        ) else "not_pass"

    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE / "scaffold_task_summary.csv", index=False, encoding="utf-8-sig")
    ap_pivot = folds.pivot(index=["assay", "seed", "fold"], columns="model", values="average_precision")
    best_simple = ap_pivot[["count", "structure", "structure_count"]].max(axis=1)
    overall_delta = ap_pivot["fusion64"] - best_simple
    observed_vs_permuted = ap_pivot["fusion64"] - ap_pivot["permuted_fusion64"]
    result = {
        "status": "exploratory_non_confirmatory",
        "analysis_units": int(len(full)),
        "structure_match": {"matched": int(len(full)), "missing": 0},
        "morphology_features": len(morphology_columns),
        "count_features": count_columns,
        "fingerprint": "Morgan radius 2, 1024 bits",
        "split": {"type": "StratifiedGroupKFold", "group": "Bemis-Murcko scaffold", "splits": N_SPLITS, "seeds": SEEDS},
        "model": "balanced L2 logistic regression; train-fold scaling and 64-component morphology PCA",
        "all_scaffold_overlaps_zero": bool((folds["scaffold_overlap"] == 0).all()),
        "overall_fusion_minus_best_simple_ap": distribution(overall_delta),
        "overall_observed_fusion_minus_permuted_fusion_ap": distribution(observed_vs_permuted),
        "task_decisions": decisions,
        "passed_tasks": int(sum(value == "pass" for value in decisions.values())),
        "total_tasks": len(decisions),
        "boundary": "one deterministic identity permutation per seed is a diagnostic control, not a formal permutation p-value",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "rdkit": rdBase.rdkitVersion},
    }
    (HERE / "scaffold_audit_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
