from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
pred = pd.read_csv(HERE / "cellhealth_oof_predictions.csv.gz")
rows = []
for keys, z in pred.groupby(["split_type", "target", "model"], sort=False):
    value = spearmanr(z.y, z.prediction).statistic if z.prediction.std() > 0 else float("nan")
    rows.append(dict(zip(["split_type", "target", "model"], keys)) | {"pooled_spearman_rho": value})
out = pd.DataFrame(rows)
out.to_csv(HERE / "cellhealth_pooled_scores.csv", index=False)
print(out.groupby(["split_type", "model"]).pooled_spearman_rho.median().to_string())
