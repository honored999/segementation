"""Pure synthetic tests: no patient IDs, nnU-Net, model or checkpoint imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('archived_run', ROOT / 'run_experiment.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def fixture():
    ids = [f'synthetic_subject_{i:03d}' for i in range(95)]
    folds = []
    for index in range(5):
        val = ids[index * 19:(index + 1) * 19]
        folds.append({'train': [case for case in ids if case not in val], 'val': val})
    return folds, ids[:2] + ids[19:32]


class IntegrityTests(unittest.TestCase):
    def test_shared_selection_original_evaluation_and_train_difference(self):
        folds, excluded = fixture()
        a, selection_a, evaluation_a = run.memberships(folds, excluded, 'A')
        b, selection_b, evaluation_b = run.memberships(folds, excluded, 'B')
        self.assertEqual((len(a), len(b), len(selection_a), len(evaluation_a)), (76, 63, 17, 19))
        self.assertEqual(selection_a, selection_b)
        self.assertEqual(evaluation_a, evaluation_b)
        self.assertEqual(set(a) - set(b), set(excluded) & set(a))
        self.assertEqual(set(evaluation_a) - set(selection_a), set(excluded[:2]))
        self.assertFalse(set(b) & set(evaluation_b))

    def test_reject_overlap_and_cohort_drift(self):
        for defect in ('overlap', 'drift', 'duplicate'):
            with self.subTest(defect=defect):
                folds, excluded = fixture()
                if defect == 'overlap':
                    folds[1]['train'][0] = folds[1]['val'][0]
                elif defect == 'drift':
                    folds[1]['train'][0] = 'synthetic_unknown'
                else:
                    folds[1]['train'].append(folds[1]['train'][0])
                with self.assertRaises(ValueError):
                    run.memberships(folds, excluded, 'B')

    def test_reject_wrong_exclusions(self):
        folds, excluded = fixture()
        for bad in ([], excluded[:-1], excluded[:-1] + [excluded[0]], excluded[:-1] + ['synthetic_unknown'], folds[0]['train'][:15]):
            with self.subTest(exclusions=bad):
                with self.assertRaises(ValueError):
                    run.memberships(folds, bad, 'B')

    def test_output_overlap_rejected_before_dependency_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = {'raw_root': str(base / 'raw'), 'preprocessed_root': str(base / 'prep'),
                   'results_parent': str(base / 'raw' / 'derived'),
                   'pretrained_checkpoint': str(base / 'weights' / 'missing.pth')}
            with self.assertRaisesRegex(ValueError, 'overlaps source'):
                run.verify(cfg, 'A', 0)
            cfg['results_parent'] = str(base / 'outputs')
            (base / 'outputs' / 'seed_0' / 'A').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                run.verify(cfg, 'A', 0)

    def test_metrics_preserve_cohort_and_reject_missing_duplicate_cases(self):
        folds, excluded = fixture()
        _, selection, evaluation = run.memberships(folds, excluded, 'A')
        records = [{'prediction_file': case + '.nii.gz', 'metrics': {'1': {'Dice': .5, 'IoU': 1 / 3, 'TP': 2, 'FP': 1, 'FN': 1}}} for case in reversed(evaluation)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = output / 'summary.json'
            summary.write_text(json.dumps({'metric_per_case': records}), encoding='utf-8')
            result = run.export_metrics(summary, evaluation, selection, output)
            self.assertEqual(result['primary_original19_dice'], .5)
            self.assertEqual(result['selection_cases'], selection)
            for invalid in (records[:-1], records + [records[0]]):
                summary.write_text(json.dumps({'metric_per_case': invalid}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    run.export_metrics(summary, evaluation, selection, output)


if __name__ == '__main__':
    unittest.main()
