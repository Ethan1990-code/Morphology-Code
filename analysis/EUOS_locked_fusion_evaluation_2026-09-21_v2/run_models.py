"""Nested group/site development; frozen prediction then separate scoring.

Commands: selftest, develop [lr|rf], predict [lr|rf], score.
"""
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
P = HERE.parents[1]
INTAKE = P / 'preprocessing/EUOS_profiles_intake_2026-09-21_v1'
DEVS = ['FMP_HepG2', 'IMTM_HepG2', 'MEDINA_HepG2']
SITES = DEVS + ['USC_HepG2', 'FMP_U2OS']
ARMS = ['S', 'C', 'M', 'B', 'SM', 'early']
SEED = 20260921
FITLOG = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf8')


class TrainTransform(TransformerMixin, BaseEstimator):
    def __init__(self, binary_n=0):
        self.binary_n = binary_n

    def fit(self, X, y=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.keep_ = (np.isfinite(X).mean(axis=0) >= .8) & (np.nanstd(X, axis=0) > 1e-10)
            a = X[:, self.keep_]
            self.median_ = np.nanmedian(a, axis=0)
        a = np.where(np.isfinite(a), a, self.median_)
        self.mean_ = a.mean(axis=0)
        self.scale_ = a.std(axis=0)
        self.scale_[self.scale_ < 1e-10] = 1
        binary = np.arange(X.shape[1])[self.keep_] < self.binary_n
        self.mean_[binary] = 0
        self.scale_[binary] = 1
        assert self.keep_.any()
        return self

    def transform(self, X):
        a = X[:, self.keep_]
        return np.ascontiguousarray((np.where(np.isfinite(a), a, self.median_) - self.mean_) / self.scale_, dtype=float)


def weights(frame):
    return 1 / frame.groupby('parent_key').parent_key.transform('size').to_numpy()


def fit_model(X, y, frame, family, param, binary_n, tag):
    t0 = time.time()
    if family == 'lr':
        model = LogisticRegression(C=param, solver='liblinear', max_iter=2000, random_state=SEED)
    else:
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=param,
                                       max_features='sqrt', random_state=SEED, n_jobs=4)
    pipe = Pipeline([('train_only', TrainTransform(binary_n)), ('model', model)])
    pipe.fit(X, y, model__sample_weight=weights(frame))
    FITLOG.append(dict(tag=tag, family=family, parameter=param, rows=len(y),
                       compounds=frame.parent_key.nunique(), features=int(pipe[0].keep_.sum()),
                       seconds=time.time()-t0))
    return pipe


def folds(frame, seed=SEED):
    units = frame.drop_duplicates('parent_key').reset_index(drop=True)
    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
    out = []
    for tr, va in splitter.split(units, units.y, units.scaffold):
        train_keys, valid_keys = set(units.iloc[tr].parent_key), set(units.iloc[va].parent_key)
        assert not (set(units.iloc[tr].scaffold) & set(units.iloc[va].scaffold))
        ti, vi = np.flatnonzero(frame.parent_key.isin(train_keys)), np.flatnonzero(frame.parent_key.isin(valid_keys))
        assert len(set(frame.iloc[ti].y)) == 2
        out.append((ti, vi))
    return out


def load_data(assay):
    units = pd.read_csv(HERE / 'unit_partition_labels.csv')
    rows, sm, cm, mm = [], [], [], []
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for site in SITES:
        d = np.load(INTAKE / (site + '_profiles.npz'))
        u = units.set_index('parent_key').loc[d['parent_key']].reset_index()
        good = u[assay].notna() & u.all_hepg2_observed & (d['observed_wells'] > 0)
        ix = np.flatnonzero(good)
        sub = u.loc[good, ['parent_key', 'scaffold', 'partition', 'parent_smiles']].copy()
        sub['y'], sub['site'] = u.loc[good, assay].astype(int), site
        rows.append(sub)
        sm.append(np.asarray([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in sub.parent_smiles], dtype='float32'))
        cm.append(d['C'][ix])
        mm.append(d['M'][ix])
    frame = pd.concat(rows, ignore_index=True)
    S, C, M = np.vstack(sm), np.vstack(cm), np.vstack(mm)
    X = dict(S=S, C=C, M=M, B=np.column_stack([S, C]),
             SM=np.column_stack([S, M]), early=np.column_stack([S, C, M]))
    strict = frame.partition.eq('strict_holdout')
    dev = frame.partition.eq('development')
    assert not set(frame.loc[strict, 'parent_key']) & set(frame.loc[dev, 'parent_key'])
    assert not set(frame.loc[strict, 'scaffold']) & set(frame.loc[dev, 'scaffold'])
    return frame, X


def similarity(train, test, Xt, Xv):
    """Top-five active unique compounds, not five site repeats of one compound."""
    active = train.y.to_numpy() == 1
    keys = train.loc[active, 'parent_key'].to_numpy()
    ids = sorted(set(keys))
    st = np.vstack([Xt['S'][active][keys == k][0] for k in ids])
    transform = TrainTransform().fit(Xt['M'])
    mtall = transform.transform(Xt['M'])[active]
    mt = np.vstack([np.median(mtall[keys == k], axis=0) for k in ids])
    mv = transform.transform(Xv['M'])
    sv = Xv['S']
    inter = sv @ st.T
    ts = inter / np.maximum(sv.sum(axis=1)[:, None] + st.sum(axis=1) - inter, 1)
    mt -= mt.mean(axis=1, keepdims=True)
    mv -= mv.mean(axis=1, keepdims=True)
    pc = (mv @ mt.T) / np.maximum(np.linalg.norm(mv, axis=1)[:, None] * np.linalg.norm(mt, axis=1), 1e-20)
    k = min(5, len(ids))
    return np.column_stack([np.sort(ts, axis=1)[:, -k:].mean(axis=1),
                            np.sort(pc, axis=1)[:, -k:].mean(axis=1)])


def predict_bundle(frame, X, train_idx, test_idx, family, tag):
    tr, va = frame.iloc[train_idx].reset_index(drop=True), frame.iloc[test_idx].reset_index(drop=True)
    assert not set(tr.parent_key) & set(va.parent_key)
    Xt = {a: x[train_idx] for a, x in X.items()}
    Xv = {a: x[test_idx] for a, x in X.items()}
    inner = folds(tr)
    y = tr.y.to_numpy()
    grid = [.01, .1, 1.] if family == 'lr' else [1, 3, 8]
    oof, predictions, selected = {}, {}, {}
    for arm in ARMS:
        scores, ps = [], []
        binary = 2048 if arm in ['S', 'B', 'SM', 'early'] else 0
        for param in grid:
            pred = np.full(len(tr), np.nan)
            for j, (ti, vi) in enumerate(inner):
                model = fit_model(Xt[arm][ti], y[ti], tr.iloc[ti], family, param, binary, tag + '/' + arm + '/inner' + str(j))
                pred[vi] = model.predict_proba(Xt[arm][vi])[:, 1]
            assert np.isfinite(pred).all()
            scores.append(float(np.average((y-pred)**2, weights=weights(tr))))
            ps.append(pred)
        best = int(np.argmin(scores))
        oof[arm] = ps[best]
        model = fit_model(Xt[arm], y, tr, family, grid[best], binary, tag + '/' + arm + '/refit')
        predictions[arm] = model.predict_proba(Xv[arm])[:, 1]
        selected[arm] = dict(parameter=grid[best], inner_brier=scores[best])
    candidates = [(f'late_{w}', (1-w)*oof['B']+w*oof['M'], (1-w)*predictions['B']+w*predictions['M'])
                  for w in [0., .25, .5, .75, 1.]]
    candidates.append(('early', oof['early'], predictions['early']))
    losses = [np.average((y - p)**2, weights=weights(tr)) for _, p, _ in candidates]
    win = int(np.argmin(losses))
    predictions['F'] = candidates[win][2]
    predictions['late_fixed_half'] = .5 * (predictions['B'] + predictions['M'])
    selected['F'] = dict(kind=candidates[win][0], inner_brier=float(losses[win]))
    # OOF similarity inputs are constructed using only each fold's training actives.
    simoof = np.empty((len(tr), 2))
    for ti, vi in inner:
        simoof[vi] = similarity(tr.iloc[ti], tr.iloc[vi], {a: x[ti] for a, x in Xt.items()}, {a: x[vi] for a, x in Xt.items()})
    simtest = similarity(tr, va, Xt, Xv)
    metatr = np.column_stack([simoof, oof['S'], oof['M']])
    metava = np.column_stack([simtest, predictions['S'], predictions['M']])
    for balance in [None, 'balanced']:
        t0 = time.time()
        model = LogisticRegression(C=1, class_weight=balance, solver='lbfgs', max_iter=2000, random_state=SEED)
        model.fit(metatr, y, sample_weight=weights(tr))
        key = 'similarity_natural' if balance is None else 'similarity_balanced'
        predictions[key] = model.predict_proba(metava)[:, 1]
        FITLOG.append(dict(tag=tag+'/'+key, family='lr_meta', parameter=1, rows=len(y),
                           compounds=tr.parent_key.nunique(), features=4, seconds=time.time()-t0))
    return predictions, selected


def decision(oof):
    d = oof.copy()
    d['gain'] = (d.y - d.B)**2 - (d.y - d.F)**2
    # Keep every lab measurement of a sampled scaffold together.
    g = d.groupby('scaffold').gain.agg(['sum', 'count'])
    rng = np.random.default_rng(SEED)
    ix = rng.integers(0, len(g), size=(5000, len(g)))
    boot = g['sum'].to_numpy()[ix].sum(axis=1) / g['count'].to_numpy()[ix].sum(axis=1)
    low = float(np.quantile(boot, .05))
    lab = {k: float(v) for k, v in d.groupby('site').gain.mean().items()}
    ci = low > 0
    return dict(mean_gain=float(d.gain.mean()), one_sided_95_low=low,
                ci95=np.quantile(boot, [.025, .975]).tolist(), lab_gain=lab,
                point=bool(d.gain.mean() > 0), simple_ci=bool(ci),
                context_0=bool(ci and min(lab.values()) >= 0),
                context_005=bool(ci and min(lab.values()) >= -.005),
                context_010=bool(ci and min(lab.values()) >= -.01))


def eligible():
    q = pd.read_csv(HERE / 'endpoint_qualification.csv')
    return q.loc[q.eligible_primary, 'assay'].tolist()


def develop(family):
    allrows, selections, decisions, audits = [], {}, {}, []
    for assay in eligible():
        frame, X = load_data(assay)
        devix = np.flatnonzero(frame.partition.eq('development') & frame.site.isin(DEVS))
        fd = frame.iloc[devix].reset_index(drop=True)
        outer = folds(fd)
        predrows = []
        for lab in DEVS:
            for j, (ti, vi) in enumerate(outer):
                train = devix[ti[fd.iloc[ti].site.ne(lab).to_numpy()]]
                valid = devix[vi[fd.iloc[vi].site.eq(lab).to_numpy()]]
                assert not set(frame.iloc[train].scaffold) & set(frame.iloc[valid].scaffold)
                assert lab not in set(frame.iloc[train].site)
                tag = assay+'/'+lab+'/'+str(j)
                t0 = time.time()
                preds, sel = predict_bundle(frame, X, train, valid, family, tag)
                out = frame.iloc[valid][['parent_key', 'scaffold', 'site', 'y']].copy()
                out['assay'], out['fold'], out['family'] = assay, j, family
                for a, v in preds.items():
                    out[a] = v
                predrows.append(out)
                selections[tag] = sel
                audits.append(dict(tag=tag, n_train=len(train), n_valid=len(valid), entity_overlap=0, scaffold_overlap=0, site_overlap=0))
                print('DEVELOP', family, tag, 'seconds', round(time.time()-t0, 1), flush=True)
        od = pd.concat(predrows, ignore_index=True)
        assert not od.duplicated(['parent_key', 'site']).any()
        assert len(od) == len(fd)
        od.to_csv(HERE / (family+'_'+assay+'_development_predictions.csv'), index=False)
        decisions[assay] = decision(od)
        allrows.append(od)
        print('DECISION', family, assay, json.dumps(decisions[assay]), flush=True)
    pd.concat(allrows).to_csv(HERE / (family+'_development_predictions.csv'), index=False)
    pd.DataFrame(FITLOG).to_csv(HERE / (family+'_development_fit_log.csv'), index=False)
    write_json(family+'_development_selections.json', selections)
    write_json(family+'_leakage_checks.json', audits)
    write_json(family+'_decision_lock.json', dict(
        status='locked_before_external_prediction', decisions=decisions, seed=SEED,
        implementation_sha256=sha(__file__), contract_sha256=sha(HERE/'implementation_contract.md'),
        partition_sha256=sha(HERE/'unit_partition_labels.csv'),
        development_prediction_sha256=sha(HERE/(family+'_development_predictions.csv')),
        revision_count=0, final_site='USC_HepG2'))


def predict(family):
    lock = json.loads((HERE/(family+'_decision_lock.json')).read_text())
    assert lock['implementation_sha256'] == sha(__file__)
    assert lock['partition_sha256'] == sha(HERE/'unit_partition_labels.csv')
    outs, sels = [], {}
    for assay in eligible():
        frame, X = load_data(assay)
        train = np.flatnonzero(frame.partition.eq('development') & frame.site.isin(DEVS))
        valid = np.flatnonzero(frame.partition.ne('development'))
        pred, sel = predict_bundle(frame, X, train, valid, family, assay+'/final')
        out = frame.iloc[valid][['parent_key', 'scaffold', 'partition', 'site']].copy()
        out['assay'], out['family'] = assay, family
        # No outcome column is included or scored before this prediction artifact is hashed.
        for a, v in pred.items():
            out[a] = v
        for rule in ['point', 'simple_ci', 'context_0', 'context_005', 'context_010']:
            out[rule] = pred['F'] if lock['decisions'][assay][rule] else pred['B']
        outs.append(out)
        sels[assay] = sel
        print('PREDICTED_NOT_SCORED', family, assay, 'rows', len(out), flush=True)
    file = HERE/(family+'_sealed_predictions.csv')
    pd.concat(outs, ignore_index=True).to_csv(file, index=False)
    write_json(family+'_prediction_lock.json', dict(
        status='predictions_locked_before_scoring', prediction_sha256=sha(file),
        decision_lock_sha256=sha(HERE/(family+'_decision_lock.json')), implementation_sha256=sha(__file__),
        final_selections=sels))
    pd.DataFrame(FITLOG).to_csv(HERE/(family+'_final_fit_log.csv'), index=False)


def selftest():
    rng = np.random.default_rng(20260922)
    x = rng.integers(0, 2, size=(10, 16)).astype(float)
    m = rng.normal(size=(10, 9))
    tr = pd.DataFrame(dict(parent_key=[str(i) for i in range(7)], y=[1]*7))
    va = pd.DataFrame(dict(parent_key=['7', '8', '9'], y=[0]*3))
    fast = similarity(tr, va, dict(S=x[:7], M=m[:7]), dict(S=x[7:], M=m[7:]))
    t = TrainTransform().fit(m[:7])
    mt, mv = t.transform(m[:7]), t.transform(m[7:])
    slow = []
    for i in range(3):
        ts = [sum(np.logical_and(x[i+7], v))/sum(np.logical_or(x[i+7], v)) for v in x[:7]]
        pc = [pearsonr(mv[i], v).statistic for v in mt]
        slow.append([np.mean(sorted(ts)[-5:]), np.mean(sorted(pc)[-5:])])
    assert np.allclose(fast, slow, atol=1e-12)
    tt = TrainTransform().fit(np.array([[1., np.nan, 2.], [2., np.nan, 4.], [3., np.nan, 6.]]))
    before = tt.mean_.copy()
    tt.transform(np.array([[1e9, 2., -1e9]]))
    assert np.array_equal(before, tt.mean_)
    write_json('selftest.json', dict(status='pass', similarity_max_error=float(np.max(np.abs(fast-slow))),
               training_transform_unchanged_by_test=True, nearest_method_scope='formula slice, not reproduction of published scores'))
    print('SELFTEST PASS')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        if sys.argv[1] == 'selftest':
            selftest()
        elif sys.argv[1] == 'develop':
            develop(sys.argv[2])
        elif sys.argv[1] == 'predict':
            predict(sys.argv[2])
