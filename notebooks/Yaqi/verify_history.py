"""Check migrated bytes, data lineage and actual RMSE, then build a common catalog."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.cv import get_splits, DATE_COL
from src.data_cleaning import load_train, load_test, TARGET

PERSONAL = ROOT / 'notebooks/Yaqi'
HISTORY = PERSONAL / 'history'


def main():
    migration = json.loads((HISTORY / 'migration_manifest.json').read_text(encoding='utf-8'))
    for entry in migration['files']:
        path = ROOT / entry['destination']
        content = path.read_bytes()
        assert len(content) == entry['bytes'], path
        assert hashlib.sha256(content).hexdigest() == entry['sha256'], path
    train, test = load_train(), load_test()
    folds = {f.definition.name: f for f in get_splits(train, 'rolling')}
    catalog = []
    run_checks = []
    submissions_checked = 0
    for date, expected in [('20261022', 13), ('20261103', 30)]:
        archive = HISTORY / date
        summary = pd.read_csv(archive / 'comparison.csv')
        assert len(summary) == expected and summary.name.is_unique
        signature = json.loads((archive / 'signature.json').read_text(encoding='utf-8'))
        for relative, expected_hash in signature.get('hashes', signature.get('data')).items():
            path = ROOT / relative.replace('\\', '/')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, relative
        for directory in sorted((archive / 'experiments').iterdir()):
            if not directory.is_dir():
                continue
            config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
            prediction = pd.read_csv(directory / 'validation_predictions.csv')
            assert list(prediction) == ['row_index', 'fold', DATE_COL, TARGET, 'Predicted']
            assert not prediction.duplicated(['fold', 'row_index']).any()
            assert set(prediction.fold) == set(folds)
            assert config['seed'] == 42 and config['split'] == 'rolling'
            assert np.isfinite(prediction.Predicted).all()
            scores = {}
            for name, fold in folds.items():
                part = prediction.loc[prediction.fold.eq(name)]
                np.testing.assert_array_equal(part.row_index, fold.valid_idx)
                np.testing.assert_array_equal(part[DATE_COL], train.iloc[fold.valid_idx][DATE_COL])
                np.testing.assert_array_equal(part[TARGET], train.iloc[fold.valid_idx][TARGET])
                scores[name] = float(np.sqrt(np.mean((part[TARGET] - part.Predicted) ** 2)))
                np.testing.assert_allclose(scores[name], config['fold_rmse'][name], atol=1e-9, rtol=0)
            mean = np.mean(list(scores.values()))
            np.testing.assert_allclose(mean, config['mean_rmse'], atol=1e-9, rtol=0)
            run_checks.append({'date': date, 'run_id': directory.name, 'mean_rmse': mean,
                               'validation_rows': len(prediction)})
        for row in summary.to_dict('records'):
            config_path = archive / 'experiments' / row['run_id'] / 'config.json'
            cfg = json.loads(config_path.read_text(encoding='utf-8'))
            np.testing.assert_allclose([row['split_a'], row['split_b'], row['mean_rmse']],
                [cfg['fold_rmse']['holdout'], cfg['fold_rmse']['rolling_2024_04'], cfg['mean_rmse']], atol=1e-9, rtol=0)
            catalog.append({'phase': date, 'name': row['name'], 'run_id': row['run_id'],
                'split_a': row['split_a'], 'split_b': row['split_b'], 'mean_rmse': row['mean_rmse'],
                'seed': 42, 'role': 'historical_experiment',
                'equivalent_configuration': 'catboost_address_24m/base' if row['name'] in ('catboost_address_24m', 'catboost_base') else '',
                'config_path': config_path.relative_to(ROOT).as_posix(),
                'validation_predictions_path': (config_path.parent / 'validation_predictions.csv').relative_to(ROOT).as_posix()})
        log = pd.read_csv(archive / 'experiments/log.csv')
        assert set(log.run_id) == {r['run_id'] for r in run_checks if r['date'] == date}
        for path in (archive / 'submissions').glob('*.csv'):
            frame = pd.read_csv(path)
            assert list(frame) == ['Id', 'Predicted'] and len(frame) == len(test) == 50000
            np.testing.assert_array_equal(frame.Id, test.Id)
            np.testing.assert_array_equal(frame.Id, pd.read_csv(ROOT / 'data/example-submission.csv').Id)
            assert np.isfinite(frame.Predicted).all()
            submissions_checked += 1
    frame = pd.DataFrame(catalog).sort_values(['mean_rmse', 'phase'])
    # Retain both actual runs, but explicitly verify the cross-phase repeat.
    repeat = frame.loc[frame.equivalent_configuration.ne('')]
    left, right = [pd.read_csv(ROOT / p) for p in repeat.validation_predictions_path]
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert len(frame) == 43 and frame.run_id.is_unique
    frame.to_csv(PERSONAL / 'experiment_catalog.csv', index=False)
    official = ROOT / 'results/Yaqi/experiments'
    assert len([p for p in official.iterdir() if p.is_dir()]) == 3
    assert not [p for p in HISTORY.rglob('*') if p.suffix in ('.joblib', '.pkl', '.zip', '.cbm', '.lock')]
    report = {'status': 'passed', 'migrated_files_hash_checked': len(migration['files']),
        'initial_configurations': 13, 'feature_configurations': 30, 'catalog_rows': len(frame),
        'archived_runs_checked_including_final_repeats': len(run_checks),
        'submissions_checked': submissions_checked, 'scores_recomputed_from_predictions': True,
        'row_indices_dates_labels_exact': True, 'data_files_hash_checked_per_phase': 17,
        'cross_phase_catboost_repeat_predictions_identical': True,
        'official_delivery_runs': 3, 'model_weights_included': False, 'runs': run_checks}
    (HISTORY / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
