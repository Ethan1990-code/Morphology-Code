"""Mechanical self-check: not an independent scientific review."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_models import HERE, INTAKE, eligible, sha, write_json


def main():
    checks = []
    units = pd.read_csv(HERE/'unit_partition_labels.csv')
    strict = units[units.partition.eq('strict_holdout')]
    dev = units[units.partition.eq('development')]
    hist = pd.read_csv(INTAKE/'historical_identity_union.csv')
    assert not set(strict.connectivity) & set(hist.connectivity)
    assert not set(strict.scaffold) & set(hist.scaffold)
    assert not set(strict.scaffold) & set(dev.scaffold)
    checks.append('strict entity and scaffold exclusion from historical union and development')
    wells = pd.read_csv(INTAKE/'well_coverage.csv.gz')
    assert not wells.duplicated(['context','Metadata_Plate','Metadata_Batch','Metadata_Well']).any()
    checks.append('no duplicated context/plate/replicate/well keys')
    scored = pd.read_csv(HERE/'scored_predictions.csv')
    metrics = pd.read_csv(HERE/'external_task_metrics.csv')
    for family in ['lr','rf']:
        lock = json.loads((HERE/(family+'_prediction_lock.json')).read_text(encoding='utf8'))
        assert sha(HERE/(family+'_sealed_predictions.csv')) == lock['prediction_sha256']
        assert sha(HERE/(family+'_decision_lock.json')) == lock['decision_lock_sha256']
        dl = json.loads((HERE/(family+'_decision_lock.json')).read_text(encoding='utf8'))
        assert sha(HERE/'run_models.py') == dl['implementation_sha256']
        assert sha(HERE/'unit_partition_labels.csv') == dl['partition_sha256']
        assert set(dl['decisions']) == set(eligible())
        for assay in eligible():
            g = scored.query("family==@family and assay==@assay and partition=='strict_holdout' and site=='USC_HepG2'")
            assert g.parent_key.is_unique
            decision = dl['decisions'][assay]
            for rule in ['point','simple_ci','context_0','context_005','context_010']:
                assert np.allclose(g[rule],g.F if decision[rule] else g.B,atol=1e-14)
            for model in ['B','F']:
                expected = metrics.query("family==@family and assay==@assay and partition=='strict_holdout' and site=='USC_HepG2' and model==@model").brier.iloc[0]
                assert abs(np.mean((g.y-g[model])**2)-expected) < 1e-14
    checks += ['prediction and decision locks match code and input hashes', 'frozen rule choices equal delivered predictions',
               'external Brier scores independently recomputed from row-level prediction files']
    sim = pd.read_csv(HERE/'simulation_formal_replicates.csv',keep_default_na=False)
    assert len(sim)==2000 and sim.seed.is_unique and sim.select_dtypes('number').notna().all().all()
    assert set(sim.scenario)=={'null','redundant','increment','mispair','domain_shift'}
    assert sim.groupby(['scenario','n','noise']).size().eq(100).all()
    assert set(sim.seed).isdisjoint(set(pd.read_csv(HERE/'simulation_pilot_replicates.csv').seed))
    checks += ['20 simulation cells each have 100 unique-seed repetitions', 'pilot and formal seeds disjoint; null label retained as text']
    diag=json.loads((HERE/'diagnostics_receipt.json').read_text(encoding='utf8'))
    assert all(r['aggregate_reproduction_max_error'] < 1e-12 for r in diag['aggregate_reproduction_checks'])
    checks.append('diagnostic refits reproduce locked aggregate predictions')
    paths=sorted(p for p in HERE.iterdir() if p.is_file() and p.name not in ['verification_receipt.json','artifact_manifest.json'])
    manifest=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
    write_json('artifact_manifest.json',dict(status='analysis_artifacts_not_submission_package',artifacts=manifest))
    write_json('verification_receipt.json',dict(status='pass',scope='same-conversation mechanical selfcheck, not independent review',
               checks=checks,independent_review=False,formal_scientific_freeze=False))
    print(json.dumps(dict(status='pass',checks=len(checks),artifacts=len(paths))))


if __name__=='__main__':main()
