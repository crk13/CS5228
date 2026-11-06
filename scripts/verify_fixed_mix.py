"""Independently check paired predictions and reconstruct each fitted market index."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.cv import get_splits, DATE_COL
from src.data_cleaning import load_train, TARGET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='Study folder, relative to project root')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT / 'outputs/Yaqi')
    study = json.loads((output / 'study_config.json').read_text(encoding='utf-8'))
    for field in ['data_hashes', 'source_hashes']:
        for relative, expected in study[field].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    runs = json.loads((output / 'completed_runs.json').read_text(encoding='utf-8'))
    assert len(runs) == 2 and {r['variant'] for r in runs} == {'raw_rent', 'fixed_mix_last'}
    train = load_train()
    folds = get_splits(train, 'rolling')
    predictions, configs = {}, {}
    scores = {}
    for run in runs:
        folder = ROOT / run['run_dir']
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        frame = pd.read_csv(folder / 'validation_predictions.csv')
        assert config['seed'] == 42 and config['split'] == 'rolling' and config['final'] is False
        assert config['features'] == study['features'] and np.isfinite(frame.Predicted).all()
        assert len(frame) == 82329 and not frame.duplicated(['fold', 'row_index']).any()
        values = []
        for fold in folds:
            part = frame.loc[frame.fold.eq(fold.definition.name)]
            np.testing.assert_array_equal(part.row_index, fold.valid_idx)
            np.testing.assert_array_equal(part[DATE_COL], train.iloc[fold.valid_idx][DATE_COL])
            np.testing.assert_array_equal(part[TARGET], train.iloc[fold.valid_idx][TARGET])
            score = float(np.sqrt(np.mean((part.Predicted - part[TARGET]) ** 2)))
            np.testing.assert_allclose(score, config['fold_rmse'][fold.definition.name], atol=1e-9, rtol=0)
            values.append(score)
        values.append(float(np.mean(values)))
        np.testing.assert_allclose(values, [run['split_a'], run['split_b'], run['mean_rmse']], atol=1e-9, rtol=0)
        scores[run['variant']] = values
        predictions[run['variant']], configs[run['variant']] = frame, config
    raw_core = configs['raw_rent']['parameters']['estimator']['parameters']
    fixed_core = configs['fixed_mix_last']['parameters']['estimator']['parameters']
    assert raw_core['months'] == fixed_core['months'] == 24
    assert raw_core['estimator'] == fixed_core['estimator']['parameters']['estimator']
    assert fixed_core['estimator']['class'] == 'src.models.fixed_mix.FixedMixTargetRegressor'
    pd.testing.assert_frame_equal(predictions['raw_rent'].drop(columns='Predicted'),
                                  predictions['fixed_mix_last'].drop(columns='Predicted'))
    audits = []
    for fold in folds:
        available = train.iloc[fold.train_idx]
        end = available[DATE_COL].max()
        start = str(pd.Period(end, freq='M') - 23)
        fitting = available.loc[available[DATE_COL].ge(start)]
        audit = json.loads((output / 'fixed_mix_last' / f'fit_through_{end}.json').read_text(encoding='utf-8'))
        assert audit['fit_start'] == start and audit['fit_end'] == audit['index_origin'] == end
        assert audit['actual_fit_rows'] == len(fitting)
        assert end < train.iloc[fold.valid_idx][DATE_COL].min()
        groups = study['index_groups']
        selected = fitting.loc[fitting.FLAT_TYPE.isin(groups)]
        counts = selected.groupby('FLAT_TYPE').size().reindex(groups)
        weights = counts / counts.sum()
        np.testing.assert_allclose(weights, [audit['index_weights'][g] for g in groups], atol=1e-12, rtol=0)
        medians = selected.groupby([DATE_COL, 'FLAT_TYPE'])[TARGET].median().unstack().reindex(columns=groups)
        levels = medians.mul(weights, axis=1).sum(axis=1)
        saved = pd.read_csv(output / 'fixed_mix_last' / f'index_through_{end}.csv', index_col=DATE_COL)
        assert len(saved) == 24
        np.testing.assert_array_equal(saved.index, medians.index)
        np.testing.assert_allclose(saved[groups], medians, atol=1e-9, rtol=0)
        np.testing.assert_allclose(saved.FIXED_MIX_LEVEL, levels, atol=1e-9, rtol=0)
        np.testing.assert_allclose(saved.RELATIVE_INDEX, levels / levels.iloc[-1], atol=1e-12, rtol=0)
        assert audit['index_rows'] == len(selected)
        assert audit['excluded_from_index_only_rows'] == len(fitting) - len(selected)
        audits.append({'fold': fold.definition.name, 'fit_start': start, 'fit_end': end,
            'fit_rows': len(fitting), 'index_rows': len(selected), 'last_market_level': levels.iloc[-1]})
    comparison = pd.read_csv(output / 'comparison.csv').set_index('variant')
    for label, values in scores.items():
        np.testing.assert_allclose(comparison.loc[label, ['split_a', 'split_b', 'mean_rmse']].astype(float), values, atol=1e-9, rtol=0)
    report = {'status': 'passed', 'data_files_hash_checked': len(study['data_hashes']),
        'training_source_hash_checked': True, 'both_variants_actually_trained': True,
        'underlying_model_parameters_and_features_identical': True,
        'predictions_aligned_rows_dates_targets': True, 'all_scores_recomputed': True,
        'index_rebuilt_from_actual_recent_training_rows': True, 'fold_audits': audits,
        'rmse_sgd': scores}
    (output / 'independent_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
