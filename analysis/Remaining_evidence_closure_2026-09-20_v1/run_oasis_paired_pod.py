from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = PROJECT / "data" / "positive_controls_2026-09-08" / "OASIS" / "repository_snapshot" / "2024_09_09_Axiom_OASIS-main"
TABLES = ROOT / "2_downstream_analysis" / "compiled_results" / "SI_tables"
SEED = 20260920
BOOT = 5000


def exact_binomial_ci(k: int, n: int) -> tuple[float, float]:
    low = 0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k))
    return low, high


def bootstrap(values: np.ndarray, func, rng: np.random.Generator) -> tuple[float, float]:
    out = [func(values[rng.integers(0, len(values), len(values))]) for _ in range(BOOT)]
    return float(np.quantile(out, .025)), float(np.quantile(out, .975))


def main() -> None:
    hit = pd.read_csv(TABLES / "hit_summary.csv")
    dino = pd.read_csv(TABLES / "cellpainting_dino_pods.csv")
    morph = dino.groupby(["OASIS_ID", "Compound_name"], as_index=False).POD_um.min().rename(columns={"POD_um": "Morphology"})
    frame = hit.loc[hit.Hit_in_all_assays.eq("Yes"), ["OASIS_ID", "Compound_name"]].merge(
        morph, on=["OASIS_ID", "Compound_name"], validate="one_to_one")
    for filename, label in [("mt_pods.csv", "MT"), ("cellcount_pods.csv", "Cell_count"), ("ldh_pods.csv", "LDH")]:
        x = pd.read_csv(TABLES / filename)[["OASIS_ID", "Compound_name", "POD_um"]].rename(columns={"POD_um": label})
        frame = frame.merge(x, on=["OASIS_ID", "Compound_name"], validate="one_to_one")
    assert len(frame) == 121 and np.isfinite(frame[["Morphology", "MT", "Cell_count", "LDH"]]).all().all()
    frame.to_csv(HERE / "oasis_complete_case_pod.csv", index=False)
    rng = np.random.default_rng(SEED)
    rows = []
    for label in ["MT", "Cell_count", "LDH"]:
        ratio = frame[label].to_numpy(float) / frame.Morphology.to_numpy(float)
        log_ratio = np.log10(ratio)
        earlier = ratio > 1
        k = int(earlier.sum())
        bin_low, bin_high = exact_binomial_ci(k, len(ratio))
        geo_ci = bootstrap(log_ratio, lambda v: float(10 ** np.mean(v)), rng)
        med_ci = bootstrap(ratio, lambda v: float(np.median(v)), rng)
        rows.append({"comparator": label, "n": len(ratio), "geometric_mean_ratio_other_over_morphology": float(10 ** log_ratio.mean()),
                     "geometric_ratio_bootstrap_low": geo_ci[0], "geometric_ratio_bootstrap_high": geo_ci[1],
                     "median_paired_ratio": float(np.median(ratio)), "median_ratio_bootstrap_low": med_ci[0],
                     "median_ratio_bootstrap_high": med_ci[1], "morphology_earlier_n": k,
                     "morphology_earlier_fraction": k / len(ratio), "binomial_ci_low": bin_low, "binomial_ci_high": bin_high})
    paired = pd.DataFrame(rows)
    paired.to_csv(HERE / "oasis_paired_pod_summary.csv", index=False)
    result = {
        "status": "exploratory_non_confirmatory_complete",
        "complete_cases": len(frame),
        "paired_results": paired.to_dict("records"),
        "endpoint_context_counts": {"Axiom": 2, "ToxCast_cell_based": 267, "ToxCast_cytotoxicity": 34, "ToxCast_cell_free": 53},
        "selection_boundary": "The compiled positive-POD tables do not encode the full non-detect and censoring process; complete-case estimates are conditional on detection in all four modalities.",
        "independence_boundary": "Endpoint counts are descriptive and are not treated as homogeneous independent biological replicates."
    }
    (HERE / "oasis_paired_pod_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
