"""Paired prespecified uncertainty and fixed-hyperparameter RF seed sensitivity."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

import run_models as models
from run_models import HERE, DEVS, load_data, eligible, fit_model, write_json, sha


def interval_by_group(values, groups, seed=20260921):
    g = pd.DataFrame(dict(value=values, group=groups)).groupby('group').value.agg(['sum','count'])
    ix = np.random.default_rng(seed).integers(0,len(g),(5000,len(g)))
    b = g['sum'].to_numpy()[ix].sum(axis=1)/g['count'].to_numpy()[ix].sum(axis=1)
    return dict(point=float(np.mean(values)), low=float(np.quantile(b,.025)), high=float(np.quantile(b,.975)))


def main():
    scored = pd.read_csv(HERE/'scored_predictions.csv')
    rows = []
    for (family, assay), g in scored.query("partition=='strict_holdout'").groupby(['family','assay']):
        gain = (g.y-g.B)**2 - (g.y-g.F)**2
        w = g.assign(gain=gain).pivot(index=['parent_key','scaffold'],columns='site',values='gain')
        pair = w[['FMP_HepG2','FMP_U2OS']].dropna()
        rows.append(dict(family=family,assay=assay,contrast='gain_U2OS_minus_HepG2',
                         **interval_by_group(pair.FMP_U2OS-pair.FMP_HepG2,pair.index.get_level_values('scaffold'))))
    pd.DataFrame(rows).to_csv(HERE/'paired_cell_context_uncertainty.csv',index=False)
    rows = []
    for (family,assay),g in scored.query("partition=='strict_holdout' and site=='USC_HepG2'").groupby(['family','assay']):
        groups = sorted(g.scaffold.unique())
        locs = [np.flatnonzero(g.scaffold.eq(k)) for k in groups]
        rng = np.random.default_rng(20260921)
        differences = []
        for _ in range(2000):
            ix = np.concatenate([locs[i] for i in rng.integers(0,len(groups),len(groups))])
            z = g.iloc[ix]
            if z.y.nunique() == 2:
                differences.append(average_precision_score(z.y,z.F)-average_precision_score(z.y,z.B))
        rows.append(dict(family=family,assay=assay,point=average_precision_score(g.y,g.F)-average_precision_score(g.y,g.B),
                         low=float(np.quantile(differences,.025)),high=float(np.quantile(differences,.975)),valid_bootstraps=len(differences)))
    pd.DataFrame(rows).to_csv(HERE/'external_ap_uncertainty.csv',index=False)
    rows = []
    d = pd.read_csv(HERE/'single_well_versus_aggregate_losses.csv')
    for (family,assay),g in d.groupby(['family','assay']):
        for arm in ['B','F']:
            rows.append(dict(family=family,assay=assay,model=arm,contrast='single_well_minus_aggregate_loss',
                             **interval_by_group(g['brier_'+arm]-g['aggregate_brier_'+arm],g.scaffold)))
    pd.DataFrame(rows).to_csv(HERE/'repeat_aggregation_uncertainty.csv',index=False)
    # Primary seeds/predictions stay untouched. These are training-randomness diagnostics.
    seedrows = []
    for seed in [20260922,20260923]:
        models.SEED = seed
        for assay in eligible():
            frame,X = load_data(assay)
            ti = np.flatnonzero(frame.partition.eq('development') & frame.site.isin(DEVS))
            vi = np.flatnonzero(frame.partition.eq('strict_holdout') & frame.site.eq('USC_HepG2'))
            tr,va = frame.iloc[ti],frame.iloc[vi]
            sel=json.loads((HERE/'rf_prediction_lock.json').read_text())['final_selections'][assay]
            pred={}
            for arm in ['B','M','early']:
                f=fit_model(X[arm][ti],tr.y.to_numpy(),tr,'rf',sel[arm]['parameter'],2048 if arm!='M' else 0,'seed_sensitivity')
                pred[arm]=f.predict_proba(X[arm][vi])[:,1]
            kind=sel['F']['kind']
            if kind=='early':pred['F']=pred['early']
            else:
                weight=float(kind.split('_')[1]);pred['F']=(1-weight)*pred['B']+weight*pred['M']
            for arm in ['B','F']:
                seedrows.append(dict(seed=seed,assay=assay,model=arm,brier=float(np.mean((va.y-pred[arm])**2)),
                                     ap=float(average_precision_score(va.y,pred[arm]))))
    pd.DataFrame(seedrows).to_csv(HERE/'rf_training_seed_sensitivity.csv',index=False)
    write_json('uncertainty_receipt.json',dict(status='completed_no_decision_changes',code_sha256=sha(__file__),
               notes='RF two extra seeds reuse frozen hyperparameters and weights; they are not independent external datasets or a second selection opportunity.'))
    print('UNCERTAINTY_AND_SEED_SENSITIVITY_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
