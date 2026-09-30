import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SCENARIOS = ['null', 'redundant', 'increment', 'mispair', 'domain_shift']


def data(rng, n, scenario, noise, visible, target=False):
    s = rng.normal(size=(n, 4))
    z = rng.normal(size=n)
    gamma = 0 if scenario in ['null', 'redundant'] else 1.5
    y = rng.binomial(1, expit(-.6 + s[:, 0] + .5*s[:, 1] + gamma*z))
    ms = []
    for lab in range(1 if target else 3):
        m = rng.normal(size=(n, 10))
        if scenario == 'redundant':
            m[:, :4] = 2*s + noise*rng.normal(size=(n, 4))
        elif scenario not in ['null', 'redundant']:
            sign = -1 if scenario == 'domain_shift' and (target or (visible and lab == 2)) else 1
            m[:, 0] = sign*z + noise*rng.normal(size=n)
            m[:, 1] = s[:, 0]**2 + noise*rng.normal(size=n)
        if scenario == 'mispair':
            blocks = (s[:, 0] > 0).astype(int) + 2*(s[:, 1] > 0)
            for block in range(4):
                eligible = np.flatnonzero(blocks == block)
                ix = rng.choice(eligible, size=len(eligible)//2, replace=False)
                m[ix] = m[rng.permutation(ix)].copy()
        ms.append(m)
    return s, ms, y


def fit_predict(st, mt, y, sv, mv, repeats):
    outs = []
    for xt, xv in [(st, sv), (np.column_stack([st, mt]), np.column_stack([sv, mv]))]:
        scaler = StandardScaler().fit(xt)
        model = LogisticRegression(C=.1, solver='liblinear', max_iter=1000)
        model.fit(scaler.transform(xt), y, sample_weight=np.full(len(y), 1/repeats))
        outs.append(model.predict_proba(scaler.transform(xv))[:, 1])
    return outs


def one(scenario, n, noise, rep, seed):
    rng = np.random.default_rng(seed)
    visible = rep % 2 == 0
    s, ms, y = data(rng, n, scenario, noise, visible)
    order = rng.permutation(n)
    tr, va = order[:n//2], order[n//2:]
    gains = []
    for lab in range(3):
        other = [j for j in range(3) if j != lab]
        b, f = fit_predict(np.tile(s[tr], (2, 1)), np.vstack([ms[j][tr] for j in other]),
                           np.tile(y[tr], 2), s[va], ms[lab][va], 2)
        gains.append((y[va]-b)**2 - (y[va]-f)**2)
    g = np.vstack(gains)
    entity_gain = g.mean(axis=0)
    ix = rng.integers(0, len(va), size=(500, len(va)))
    lower = float(np.quantile(entity_gain[ix].mean(axis=1), .05))
    simple = lower > 0
    rules = dict(always_B=False, always_F=True, point=bool(entity_gain.mean() > 0),
                 simple_ci=bool(simple), context_0=bool(simple and g.mean(axis=1).min() >= 0),
                 context_005=bool(simple and g.mean(axis=1).min() >= -.005),
                 context_010=bool(simple and g.mean(axis=1).min() >= -.01))
    sv, mv, yv = data(rng, 1000, scenario, noise, visible, target=True)
    b, f = fit_predict(np.tile(s, (3, 1)), np.vstack(ms), np.tile(y, 3), sv, mv[0], 3)
    lb, lf = np.mean((yv-b)**2), np.mean((yv-f)**2)
    row = dict(scenario=scenario, n=n, noise=noise, repetition=rep, seed=seed,
               subtype=('visible' if visible else 'hidden') if scenario == 'domain_shift' else 'standard',
               dev_gain=float(entity_gain.mean()), ci_low=lower, baseline_loss=lb, fusion_loss=lf,
               target_gain=lb-lf, information_present=scenario not in ['null', 'redundant'])
    for k, accept in rules.items():
        row[k+'_accept'], row[k+'_loss'] = int(accept), lf if accept else lb
    return row


def main(mode):
    reps, base = (10, 800000) if mode == 'pilot' else (100, 900000)
    rows, t0 = [], time.time()
    for a, scenario in enumerate(SCENARIOS):
        for b, n in enumerate([200, 500]):
            for c, noise in enumerate([.3, 1.]):
                config = a*4 + b*2 + c
                for rep in range(reps):
                    rows.append(one(scenario, n, noise, rep, base+config*1000+rep))
                print(mode, scenario, n, noise, 'completed', len(rows), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(HERE / ('simulation_'+mode+'_replicates.csv'), index=False)
    summaries = []
    for keys, group in d.groupby(['scenario', 'n', 'noise', 'subtype']):
        for rule in ['always_B', 'always_F', 'point', 'simple_ci', 'context_0', 'context_005', 'context_010']:
            acc = group[rule+'_accept']
            target_gain = group.target_gain
            retained = (target_gain.clip(lower=0)*acc).sum() / max(target_gain.clip(lower=0).sum(), 1e-20)
            p = float(acc.mean())
            benefit = group.baseline_loss-group[rule+'_loss']
            summaries.append(dict(zip(['scenario', 'n', 'noise', 'subtype'], keys), rule=rule,
                                  repetitions=len(group), acceptance=p, acceptance_mcse=np.sqrt(p*(1-p)/len(group)),
                                  mean_gain=float(benefit.mean()), gain_mcse=float(benefit.std()/np.sqrt(len(group))),
                                  positive_gain_retention=float(retained), harmful_acceptance=float(((target_gain < 0) & (acc == 1)).mean())))
    pd.DataFrame(summaries).to_csv(HERE / ('simulation_'+mode+'_summary.csv'), index=False)
    (HERE / ('simulation_'+mode+'_run.json')).write_text(json.dumps(dict(
        mode=mode, repetitions=len(rows), runtime_seconds=time.time()-t0,
        contract_sha256=__import__('hashlib').sha256((HERE/'simulation_contract.md').read_bytes()).hexdigest(),
        implementation_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()), indent=2), encoding='utf8')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main(sys.argv[1])
