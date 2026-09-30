"""Bounded checks; no raw downloads or full model fitting."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
EUOS = 'analysis/EUOS_locked_fusion_evaluation_2026-09-21_v2'
REF = ROOT / 'reference_results' / EUOS


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class ReleaseTests(unittest.TestCase):
    def test_source_integrity(self):
        manifest = json.loads((ROOT / 'provenance/source_manifest.json').read_text('utf-8'))
        names = [r['file'] for r in manifest]
        self.assertEqual(len(names), len(set(names)))
        for record in manifest:
            with self.subTest(file=record['file']):
                path = ROOT / record['file']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record['published_sha256'])

    def test_syntax(self):
        for directory in ['analysis', 'preprocessing', 'figures', 'tools', 'tests']:
            for path in (ROOT / directory).rglob('*.py'):
                with self.subTest(file=path.relative_to(ROOT)):
                    compile(path.read_text('utf-8'), str(path), 'exec')

    def test_training_transform_and_similarity(self):
        obj = module('euos_models_test', ROOT / EUOS / 'run_models.py')
        with tempfile.TemporaryDirectory() as temp:
            obj.HERE = Path(temp)
            obj.selftest()
            result = json.loads((obj.HERE / 'selftest.json').read_text())
            self.assertEqual(result['status'], 'pass')
            self.assertTrue(result['training_transform_unchanged_by_test'])

    def test_corrected_partition(self):
        frame = pd.read_csv(REF / 'unit_partition_labels.csv')
        assays = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                  'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        labeled = frame.loc[frame[assays].notna().any(axis=1)]
        self.assertEqual(labeled.partition.value_counts().to_dict(),
                         {'development': 353, 'strict_holdout': 145, 'new_entity_seen_scaffold': 78})
        dev = frame.loc[frame.partition.eq('development')]
        strict = frame.loc[frame.partition.eq('strict_holdout')]
        self.assertFalse(set(dev.scaffold) & set(strict.scaffold))
        self.assertFalse(set(dev.parent_key) & set(strict.parent_key))
        q = pd.read_csv(REF / 'endpoint_qualification.csv')
        self.assertEqual(set(q.loc[q.eligible_primary, 'assay']), {'SR-ARE', 'SR-MMP'})

    def test_archived_metrics_from_predictions(self):
        scored = pd.read_csv(REF / 'scored_predictions.csv')
        reported = pd.read_csv(REF / 'external_task_metrics.csv')
        checked = 0
        for keys, frame in scored.groupby(['family', 'partition', 'site', 'assay']):
            family, partition, site, assay = keys
            rows = reported.query('family == @family and partition == @partition and site == @site and assay == @assay')
            self.assertGreater(len(rows), 0)
            for row in rows.itertuples():
                pred = frame[row.model].to_numpy()
                y = frame.y.to_numpy()
                with self.subTest(group=keys, model=row.model):
                    self.assertEqual(len(y), row.n)
                    self.assertEqual(int(y.sum()), row.positive)
                    self.assertAlmostEqual(float(np.mean((y-pred)**2)), row.brier, places=12)
                    self.assertAlmostEqual(average_precision_score(y, pred), row.ap, places=12)
                    self.assertAlmostEqual(roc_auc_score(y, pred), row.auc, places=12)
                checked += 1
        self.assertEqual(checked, len(reported))

    def test_known_answer_simulation_slice(self):
        obj = module('euos_sim_test', ROOT / EUOS / 'run_simulations.py')
        saved = pd.read_csv(REF / 'simulation_formal_replicates.csv', keep_default_na=False)
        selected = saved.groupby('scenario', sort=False).head(1)
        self.assertEqual(len(selected), 5)
        with threadpool_limits(limits=1):
            for _, expected in selected.iterrows():
                with self.subTest(scenario=expected.scenario):
                    actual = obj.one(expected.scenario, int(expected.n), float(expected.noise),
                                     int(expected.repetition), int(expected.seed))
                    for key, value in actual.items():
                        if isinstance(value, (float, np.floating)):
                            self.assertAlmostEqual(value, float(expected[key]), places=10)
                        else:
                            self.assertEqual(value, expected[key])

    def test_all_table_sources_exist(self):
        tables = json.loads((ROOT / 'docs/table_source_map.json').read_text())
        expected = {'Table 1', 'Table 2'} | {f'Table S{i}' for i in range(1, 21)}
        self.assertEqual({t['table'] for t in tables}, expected)
        for table in tables:
            for source in table['sources']:
                self.assertTrue((ROOT / source).is_file(), source)
                self.assertGreater(len(pd.read_csv(ROOT / source)), 0)

    def test_restore_refuses_overwrite(self):
        obj = module('restore_test', ROOT / 'tools/restore_reference_results.py')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'reference_results/analysis/test.csv'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'x\n1\n')
            (root / 'provenance').mkdir()
            records = [{'file': 'reference_results/analysis/test.csv',
                        'published_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]
            (root / 'provenance/source_manifest.json').write_text(json.dumps(records))
            self.assertEqual(obj.restore(root), 1)
            self.assertEqual(obj.restore(root), 1)
            target = root / 'analysis/test.csv'
            target.write_bytes(b'x\n2\n')
            with self.assertRaises(FileExistsError):
                obj.restore(root)
            self.assertEqual(target.read_bytes(), b'x\n2\n')


if __name__ == '__main__':
    unittest.main()
