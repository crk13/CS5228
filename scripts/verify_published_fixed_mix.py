"""Verify the portable fixed-mix experiment archive without retraining."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.cv import get_splits, DATE_COL
from src.data_cleaning import load_train, load_test, TARGET

ARCHIVE = ROOT / 'notebooks/Yaqi/fixed_mix_20261104'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((ARCHIVE / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['kaggle_scores_included'] is False
    for entry in manifest['archive_files']:
        data = (ARCHIVE / entry['path']).read_bytes()
        assert len(data) == entry['bytes'] and digest(data) == entry['sha256'], entry['path']
    for field in ['datasets', 'sources']:
        for relative, entry in manifest[field].items():
            data = (ROOT / relative).read_bytes()
            assert digest(data.replace(b'\r\n', b'\n')) == entry['normalized_lf_sha256'], relative
            if field == 'sources':
                assert digest((ARCHIVE / entry['snapshot']).read_bytes()) == entry['sha256']
    train, test = load_train(), load_test()
    folds = get_splits(train, 'rolling')
    index = pd.read_csv(ARCHIVE / 'experiment_index.csv')
    assert len(index) == 6 and index.run_id.is_unique
    assert set(zip(index.kind, index.variant)) == {
        (kind, variant) for kind in ['catboost', 'xgboost', 'lightgbm']
        for variant in ['raw_rent', 'fixed_mix_last']}
    scores, core_params, labels = [], {}, None
    for entry in manifest['validation_runs']:
        folder = ROOT / entry['run_dir']
        folder.relative_to(ARCHIVE)
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        frame = pd.read_csv(folder / 'validation_predictions.csv')
        assert list(frame) == ['row_index', 'fold', DATE_COL, TARGET, 'Predicted']
        assert len(frame) == 82329 and np.isfinite(frame.Predicted).all()
        assert not frame.duplicated(['fold', 'row_index']).any()
        assert config['seed'] == 42 and config['split'] == 'rolling'
        if labels is None:
            labels = frame.drop(columns='Predicted')
        pd.testing.assert_frame_equal(labels, frame.drop(columns='Predicted'))
        actual_scores = []
        for fold in folds:
            part = frame.loc[frame.fold.eq(fold.definition.name)]
            np.testing.assert_array_equal(part.row_index, fold.valid_idx)
            np.testing.assert_array_equal(part[TARGET], train.iloc[fold.valid_idx][TARGET])
            np.testing.assert_array_equal(part[DATE_COL], train.iloc[fold.valid_idx][DATE_COL])
            value = float(np.sqrt(np.mean((part.Predicted - part[TARGET]) ** 2)))
            np.testing.assert_allclose(value, config['fold_rmse'][fold.definition.name], atol=1e-9, rtol=0)
            actual_scores.append(value)
        actual_scores.append(float(np.mean(actual_scores)))
        np.testing.assert_allclose(actual_scores, [entry['split_a'], entry['split_b'], entry['mean_rmse']], atol=1e-9, rtol=0)
        core_params[(entry['kind'], entry['variant'])] = config['parameters']['estimator']['parameters']
        scores.append({'kind': entry['kind'], 'variant': entry['variant'], 'folds_recomputed': 2})
        if entry['variant'] != 'fixed_mix_last':
            continue
        study = folder.parents[1]
        for fold in folds:
            available = train.iloc[fold.train_idx]
            end = available[DATE_COL].max()
            start = str(pd.Period(end, freq='M') - 23)
            fitting = available.loc[available[DATE_COL].ge(start)]
            audit = json.loads((study / 'fixed_mix_last' / f'fit_through_{end}.json').read_text(encoding='utf-8'))
            assert audit['fit_end'] == audit['index_origin'] == end
            assert audit['fit_start'] == start and audit['actual_fit_rows'] == len(fitting)
            assert end < train.iloc[fold.valid_idx][DATE_COL].min()
            groups = ['2-room', '3-room', '4-room', '5-room', 'executive']
            selected = fitting.loc[fitting.FLAT_TYPE.isin(groups)]
            counts = selected.groupby('FLAT_TYPE').size().reindex(groups)
            weights = counts / counts.sum()
            np.testing.assert_allclose(weights, [audit['index_weights'][g] for g in groups], atol=1e-12, rtol=0)
            medians = selected.groupby([DATE_COL, 'FLAT_TYPE'])[TARGET].median().unstack().reindex(columns=groups)
            saved = pd.read_csv(study / 'fixed_mix_last' / f'index_through_{end}.csv', index_col=DATE_COL)
            np.testing.assert_array_equal(saved.index, medians.index)
            np.testing.assert_allclose(saved[groups], medians, atol=1e-9, rtol=0)
            levels = medians.mul(weights, axis=1).sum(axis=1)
            np.testing.assert_allclose(saved.FIXED_MIX_LEVEL, levels, atol=1e-9, rtol=0)
            np.testing.assert_allclose(saved.RELATIVE_INDEX, levels / levels.iloc[-1], atol=1e-12, rtol=0)
    for kind in ['catboost', 'xgboost', 'lightgbm']:
        raw = core_params[(kind, 'raw_rent')]
        fixed = core_params[(kind, 'fixed_mix_last')]
        assert raw['months'] == fixed['months'] == 24
        assert raw['estimator'] == fixed['estimator']['parameters']['estimator']
    submission_path = ARCHIVE / manifest['catboost_submission']
    frame = pd.read_csv(submission_path)
    receipt = json.loads((submission_path.parent / 'verification.json').read_text(encoding='utf-8'))
    assert digest(submission_path.read_bytes()) == receipt['submission_sha256']
    assert len(frame) == 50000 and list(frame) == ['Id', 'Predicted']
    assert np.isfinite(frame.Predicted).all()
    np.testing.assert_array_equal(frame.Id, test.Id)
    np.testing.assert_array_equal(frame.Id, pd.read_csv(ROOT / 'data/example-submission.csv').Id)
    assert receipt['actual_fit_rows'] == 71040
    assert receipt['fit_start'] == '2023-04' and receipt['fit_end'] == '2025-03'
    assert len([p for p in (ROOT / 'results/Yaqi/experiments').iterdir() if p.is_dir()]) == 3
    assert not [p for p in ARCHIVE.rglob('*') if p.suffix in ['.joblib', '.pkl', '.zip', '.lock']]
    report = {'status': 'passed', 'archive_payloads_hash_verified': len(manifest['archive_files']),
        'data_files_verified': len(manifest['datasets']), 'source_snapshots_verified': len(manifest['sources']),
        'six_prediction_files_aligned': True, 'fold_scores_recomputed': 12,
        'market_indices_rebuilt_from_training_only': True, 'base_model_parameters_identical': True,
        'submission_rows': 50000, 'submission_matches_checked_export': True,
        'formal_delivery_run_count': 3, 'model_weights_included': False,
        'kaggle_scores_included': False, 'runs': scores}
    (ARCHIVE / 'archive_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
