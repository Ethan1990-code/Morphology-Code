"""Prespecified explanatory diagnostics; never changes locked primary predictions."""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from run_models import (HERE, INTAKE, SITES, DEVS, SEED, TrainTransform, load_data,
                        fit_model, folds, eligible, sha, write_json, weights)
from score_locked import paired_macro


def residuals(tr, va, bt, bv, mt, mv):
    bproc = TrainTransform(2048).fit(bt)
    mproc = TrainTransform().fit(mt)
    model = Ridge(alpha=100, solver='cholesky')
    model.fit(bproc.transform(bt), mproc.transform(mt), sample_weight=weights(tr))
    pred = model.predict(bproc.transform(bv))*mproc.scale_ + mproc.mean_
    out = np.full_like(mv, np.nan)
    out[:, mproc.keep_] = mv[:, mproc.keep_] - pred
    return out


def rowcorr(a, b):
    a, b = a-a.mean(axis=1, keepdims=True), b-b.mean(axis=1, keepdims=True)
    return (a*b).sum(axis=1)/np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-20)


def reliability():
    units = pd.read_csv(HERE/'unit_partition_labels.csv').set_index('parent_key')
    rows = []
    for site in SITES:
        r = np.load(INTAKE/(site+'_replicates.npz'))
        meta = pd.DataFrame(dict(parent_key=r['parent_key'], replicate=r['replicate'], plate=r['plate']))
        dev = units.loc[meta.parent_key, 'partition'].eq('development').to_numpy()
        proc = TrainTransform().fit(r['M'][dev])
        m = proc.transform(r['M'])
        # Two EOS of one parent are averaged within each technical replicate.
        fm = pd.DataFrame(m).assign(parent_key=meta.parent_key, replicate=meta.replicate)
        ag = fm.groupby(['parent_key', 'replicate']).median()
        cm = meta.assign(count=r['C']).groupby(['parent_key', 'replicate']).agg(count=('count', 'median'), plate=('plate', 'first'))
        for part in ['development', 'strict_holdout']:
            partkeys = set(units[units.partition.eq(part)].index)
            for i, j in [('R1','R2'),('R1','R3'),('R1','R4'),('R2','R3'),('R2','R4'),('R3','R4')]:
                a, b = ag.xs(i, level='replicate'), ag.xs(j, level='replicate')
                keys = sorted(set(a.index) & set(b.index) & partkeys)
                av, bv = a.loc[keys].to_numpy(), b.loc[keys].to_numpy()
                ca, cb = cm.xs(i, level='replicate').loc[keys], cm.xs(j, level='replicate').loc[keys]
                nulls = []
                rng = np.random.default_rng(SEED)
                blocks = cb.plate.to_numpy()
                for _ in range(100):
                    order = np.arange(len(keys))
                    for block in set(blocks):
                        ix = np.flatnonzero(blocks == block)
                        order[ix] = rng.permutation(ix)
                    nulls.append(float(np.median(rowcorr(av, bv[order]))))
                rows.append(dict(site=site, partition=part, pair=i+'-'+j, n=len(keys),
                                 morphology_same_entity_median=float(np.median(rowcorr(av, bv))),
                                 within_plate_permutation_median=float(np.median(nulls)),
                                 permutation_p95=float(np.quantile(nulls, .95)),
                                 count_spearman=float(spearmanr(ca['count'], cb['count']).statistic)))
    pd.DataFrame(rows).to_csv(HERE/'repeatability_diagnostics.csv', index=False)


def main():
    residual_out, repeats_out, checks = [], [], []
    scored = pd.read_csv(HERE/'scored_predictions.csv')
    for assay in eligible():
        frame, X = load_data(assay)
        train = np.flatnonzero(frame.partition.eq('development') & frame.site.isin(DEVS))
        valid = np.flatnonzero(frame.partition.eq('strict_holdout'))
        tr, va = frame.iloc[train].reset_index(drop=True), frame.iloc[valid].reset_index(drop=True)
        bt, bv, mt, mv = X['B'][train], X['B'][valid], X['M'][train], X['M'][valid]
        rt = np.full_like(mt, np.nan)
        for ti, vi in folds(tr):
            rt[vi] = residuals(tr.iloc[ti], tr.iloc[vi], bt[ti], bt[vi], mt[ti], mt[vi])
        rv = residuals(tr, va, bt, bv, mt, mv)
        # Fixed exploratory explanatory model, no target-based tuning or replacement of F.
        model = fit_model(np.column_stack([bt, rt]), tr.y.to_numpy(), tr, 'lr', .1, 2048, assay+'/residual_sensitivity')
        pred = model.predict_proba(np.column_stack([bv, rv]))[:, 1]
        out = va[['parent_key','scaffold','site','y']].copy()
        out['assay'], out['residual_fusion'] = assay, pred
        ref = scored.query("family == 'lr' and partition == 'strict_holdout' and assay == @assay")
        out = out.merge(ref[['parent_key','site','B','F']], on=['parent_key','site'], validate='one_to_one')
        residual_out.append(out)
        for family in ['lr', 'rf']:
            lock = json.loads((HERE/(family+'_prediction_lock.json')).read_text())
            selections = lock['final_selections'][assay]
            fitted = {}
            for arm in ['B','M','early']:
                fitted[arm] = fit_model(X[arm][train], tr.y.to_numpy(), tr, family, selections[arm]['parameter'],
                                        2048 if arm != 'M' else 0, assay+'/repeatability_refit/'+arm)
            site = 'USC_HepG2'
            r = np.load(INTAKE/(site+'_replicates.npz'))
            target = frame.iloc[valid].query('site == @site').set_index('parent_key')
            key_list = r['parent_key']
            mask = np.array([k in target.index for k in key_list])
            ks = key_list[mask]
            map_index = frame[frame.site.eq(site)].reset_index().set_index('parent_key')['index']
            sx = X['S'][map_index.loc[ks].to_numpy()]
            bc = np.column_stack([sx, r['C'][mask]])
            mx = r['M'][mask]
            b = fitted['B'].predict_proba(bc)[:, 1]
            m = fitted['M'].predict_proba(mx)[:, 1]
            kind = selections['F']['kind']
            if kind == 'early':
                f = fitted['early'].predict_proba(np.column_stack([bc, mx]))[:, 1]
            else:
                w = float(kind.split('_')[1]); f = (1-w)*b+w*m
            rep = pd.DataFrame(dict(parent_key=ks, replicate=r['replicate'][mask], B=b, F=f))
            rep['y'] = target.loc[ks, 'y'].to_numpy()
            rep['brier_B'], rep['brier_F'] = (rep.y-rep.B)**2, (rep.y-rep.F)**2
            # Equal-weight compound summaries; wells do not inflate uncertainty.
            agg = rep.groupby('parent_key')[['brier_B','brier_F']].mean().reset_index()
            primary = scored.query("family == @family and assay == @assay and site == @site and partition == 'strict_holdout'")
            agg = agg.merge(primary[['parent_key','scaffold','y','B','F']], on='parent_key', validate='one_to_one')
            agg['aggregate_brier_B'], agg['aggregate_brier_F'] = (agg.y-agg.B)**2, (agg.y-agg.F)**2
            agg['assay'], agg['family'] = assay, family
            repeats_out.append(agg)
            ix = map_index.loc[primary.parent_key].to_numpy()
            bcheck = fitted['B'].predict_proba(X['B'][ix])[:, 1]
            mcheck = fitted['M'].predict_proba(X['M'][ix])[:, 1]
            fcheck = fitted['early'].predict_proba(X['early'][ix])[:, 1] if kind == 'early' else (1-w)*bcheck+w*mcheck
            err = float(max(np.max(abs(bcheck-primary.B)), np.max(abs(fcheck-primary.F))))
            assert err < 1e-12
            checks.append(dict(assay=assay, family=family, aggregate_reproduction_max_error=err))
        print('DIAGNOSTICS', assay, flush=True)
    residual = pd.concat(residual_out, ignore_index=True)
    residual.to_csv(HERE/'residual_sensitivity_predictions.csv', index=False)
    rows = []
    for site, g in residual.groupby('site'):
        rows.append(dict(site=site, model='residual_fusion', reference='B', **paired_macro(g, 'residual_fusion','B')))
    pd.DataFrame(rows).to_csv(HERE/'residual_sensitivity_summary.csv', index=False)
    pd.concat(repeats_out, ignore_index=True).to_csv(HERE/'single_well_versus_aggregate_losses.csv', index=False)
    reliability()
    write_json('diagnostics_receipt.json', dict(status='explanatory_not_used_for_decisions',
               fixed_ridge_alpha=100, fixed_residual_LR_C=.1, aggregate_reproduction_checks=checks,
               source_sha256=sha(__file__),
               interpretation='Residualization sensitivity is not a conditional independence or causal test. Single-well comparison uses aggregate-trained models, without retraining on single wells.'))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
