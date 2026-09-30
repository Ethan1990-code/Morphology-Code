"""Only scores hash-locked predictions; cannot refit or choose a method."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from run_models import HERE, INTAKE, SEED, sha, write_json

MODELS = ['S', 'C', 'M', 'B', 'SM', 'early', 'F', 'late_fixed_half',
          'similarity_natural', 'similarity_balanced', 'point', 'simple_ci',
          'context_0', 'context_005', 'context_010']


def paired_macro(frame, a, b, boot=5000):
    d = frame.copy()
    d['loss_difference'] = (d.y-d[a])**2 - (d.y-d[b])**2
    tasks = sorted(d.assay.unique())
    sums = d.pivot_table(index='scaffold', columns='assay', values='loss_difference', aggfunc='sum').reindex(columns=tasks).fillna(0).to_numpy()
    counts = d.pivot_table(index='scaffold', columns='assay', values='loss_difference', aggfunc='count').reindex(columns=tasks).fillna(0).to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(sums), size=(boot, len(sums)))
    vals = (sums[idx].sum(axis=1)/counts[idx].sum(axis=1)).mean(axis=1)
    return dict(mean_difference=float(d.groupby('assay').loss_difference.mean().mean()),
                ci95_low=float(np.quantile(vals, .025)), ci95_high=float(np.quantile(vals, .975)),
                endpoints=len(tasks), independent_scaffolds=len(sums), bootstrap_repetitions=boot)


def main():
    labels = pd.read_csv(HERE/'unit_partition_labels.csv')
    predictions, locks = [], {}
    for family in ['lr', 'rf']:
        lock = json.loads((HERE/(family+'_prediction_lock.json')).read_text())
        f = HERE/(family+'_sealed_predictions.csv')
        assert sha(f) == lock['prediction_sha256']
        assert sha(HERE/(family+'_decision_lock.json')) == lock['decision_lock_sha256']
        df = pd.read_csv(f)
        df['y'] = [labels.set_index('parent_key').loc[k, a] for k, a in zip(df.parent_key, df.assay)]
        assert df.y.notna().all()
        assert np.isfinite(df[MODELS]).all().all() and df[MODELS].ge(0).all().all() and df[MODELS].le(1).all().all()
        predictions.append(df)
        locks[family] = lock
    d = pd.concat(predictions, ignore_index=True)
    d.to_csv(HERE/'scored_predictions.csv', index=False)
    metrics, comparisons, task_comparisons, transport = [], [], [], []
    for keys, g in d.groupby(['family', 'partition', 'site', 'assay']):
        for model in MODELS:
            p, y = g[model].to_numpy(), g.y.to_numpy()
            row = dict(zip(['family', 'partition', 'site', 'assay'], keys), model=model,
                       n=len(g), positive=int(y.sum()), brier=float(np.mean((y-p)**2)),
                       ap=float(average_precision_score(y, p)), auc=float(roc_auc_score(y, p)),
                       mean_predicted=float(p.mean()), prevalence=float(y.mean()),
                       calibration_mean_bias=float(p.mean()-y.mean()))
            order = np.lexsort((g.parent_key.to_numpy(), -p))
            for fraction in [.05, .1, .2]:
                budget = int(np.ceil(fraction*len(g)))
                row['budget_'+str(fraction)] = budget
                row['hits_'+str(fraction)] = int(y[order[:budget]].sum())
            metrics.append(row)
        if keys[1] == 'strict_holdout':
            for a, b in [('F', 'B'), ('context_005', 'simple_ci'), ('similarity_natural', 'B')]:
                task_comparisons.append(dict(zip(['family', 'partition', 'site', 'assay'], keys),
                                              model=a, reference=b, **paired_macro(g, a, b)))
    for keys, g in d.groupby(['family', 'partition', 'site']):
        for a, b in [('context_005', 'simple_ci'), ('context_005', 'B'), ('context_005', 'F'),
                     ('F', 'B'), ('early', 'B'), ('similarity_natural', 'B'), ('similarity_balanced', 'B')]:
            comparisons.append(dict(zip(['family', 'partition', 'site'], keys), model=a, reference=b,
                                    **paired_macro(g, a, b)))
    # Compare gains on the same entity across contexts, not unrelated cohorts.
    for (family, assay), g in d[d.partition.eq('strict_holdout')].groupby(['family', 'assay']):
        g = g.copy()
        g['gain'] = (g.y-g.B)**2-(g.y-g.F)**2
        wide = g.pivot(index=['parent_key', 'scaffold'], columns='site', values='gain')
        for site in ['FMP_HepG2', 'IMTM_HepG2', 'MEDINA_HepG2']:
            pair = wide[[site, 'USC_HepG2']].dropna()
            delta = pair.USC_HepG2-pair[site]
            tmp = pd.DataFrame(dict(scaffold=pair.index.get_level_values('scaffold'), value=delta.to_numpy()))
            sums = tmp.groupby('scaffold').value.agg(['sum', 'count'])
            ix = np.random.default_rng(SEED).integers(0, len(sums), (5000, len(sums)))
            vals = sums['sum'].to_numpy()[ix].sum(axis=1)/sums['count'].to_numpy()[ix].sum(axis=1)
            transport.append(dict(family=family, assay=assay, contrast='USC minus '+site,
                                  n=len(pair), difference_in_gain=float(delta.mean()),
                                  ci95_low=float(np.quantile(vals, .025)), ci95_high=float(np.quantile(vals, .975))))
        pair = wide[['FMP_HepG2', 'FMP_U2OS']].dropna()
        delta = pair.FMP_U2OS-pair.FMP_HepG2
        transport.append(dict(family=family, assay=assay, contrast='FMP U2OS minus FMP HepG2', n=len(pair),
                              difference_in_gain=float(delta.mean()), ci95_low=None, ci95_high=None))
    pd.DataFrame(metrics).to_csv(HERE/'external_task_metrics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(HERE/'external_macro_comparisons.csv', index=False)
    pd.DataFrame(task_comparisons).to_csv(HERE/'external_task_comparisons.csv', index=False)
    pd.DataFrame(transport).to_csv(HERE/'paired_transport_comparisons.csv', index=False)
    mainrows = pd.DataFrame(comparisons).query("family == 'lr' and partition == 'strict_holdout' and site == 'USC_HepG2'")
    write_json('scoring_receipt.json', dict(status='scored_prelocked_predictions_no_refit',
                                          inputs=locks, source_sha256=sha(__file__),
                                          main_results=mainrows.to_dict('records')))
    print(mainrows.to_string(index=False))
    print(pd.DataFrame(metrics).query("partition == 'strict_holdout' and site == 'USC_HepG2' and model in ['B','F','similarity_natural']").to_string(index=False))


if __name__ == '__main__':
    main()
