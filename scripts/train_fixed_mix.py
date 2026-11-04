"""Matched CatBoost comparison: raw rent vs fixed-flat-type-index adjustment."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'notebooks/Yaqi'))
# The delivery entry establishes seed 42 before loading model factories.
from delivery import build, blocks, hashes, write_json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from threadpoolctl import threadpool_limits
import src.experiment as experiment
from src.cv import get_splits, DATE_COL
from src.data_cleaning import load_train, TARGET
from src.models.fixed_mix import FixedMixTargetRegressor
from src.models.trees import RecentWindowRegressor


class AuditFit(RegressorMixin, BaseEstimator):
    def __init__(self, estimator, output):
        self.estimator = estimator
        self.output = output

    def fit(self, X, y):
        end = X[DATE_COL].max()
        print(f'Start fit through {end}', flush=True)
        started = time.monotonic()
        self.estimator_ = clone(self.estimator).fit(X, y)
        fitted = self.estimator_
        audit = {'available_training_rows': len(X), 'actual_fit_rows': fitted.fit_rows_,
            'fit_start': fitted.fit_start_, 'fit_end': fitted.fit_end_,
            'seconds': time.monotonic() - started}
        adjusted = fitted.estimator_
        if isinstance(adjusted, FixedMixTargetRegressor):
            audit.update({'index_origin': adjusted.origin_, 'index_reference_level': adjusted.reference_level_,
                'index_weights': adjusted.weights_.to_dict(), 'index_rows': adjusted.index_rows_,
                'excluded_from_index_only_rows': adjusted.excluded_from_index_rows_,
                'future_market_level_rule': 'last training month held constant',
                'normalization': 'y / (S_month / S_last_training_month)', 'sample_weight': None})
            table = adjusted.monthly_medians_.copy()
            table['FIXED_MIX_LEVEL'] = adjusted.market_levels_
            table['RELATIVE_INDEX'] = adjusted.relative_index_
            folder = ROOT / self.output
            folder.mkdir(parents=True, exist_ok=True)
            table.to_csv(folder / f'index_through_{end}.csv', index_label=DATE_COL)
        write_json(ROOT / self.output / f'fit_through_{end}.json', audit)
        print(f'Finished fit through {end}: {audit["seconds"]:.1f}s; actual rows={fitted.fit_rows_}', flush=True)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def verify_result(result, train):
    prediction = pd.read_csv(result.run_dir / 'validation_predictions.csv')
    assert list(prediction) == ['row_index', 'fold', DATE_COL, TARGET, 'Predicted']
    assert not prediction.duplicated(['fold', 'row_index']).any()
    assert np.isfinite(prediction.Predicted).all()
    for fold in get_splits(train, 'rolling'):
        part = prediction.loc[prediction.fold.eq(fold.definition.name)]
        np.testing.assert_array_equal(part.row_index, fold.valid_idx)
        np.testing.assert_array_equal(part[TARGET], train.iloc[fold.valid_idx][TARGET])
        np.testing.assert_array_equal(part[DATE_COL], train.iloc[fold.valid_idx][DATE_COL])
        score = np.sqrt(np.mean((part.Predicted - part[TARGET]) ** 2))
        np.testing.assert_allclose(score, result.fold_rmse[fold.definition.name], atol=1e-9, rtol=0)
    return prediction


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'outputs/Yaqi' / f'fixed_mix_catboost_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    before = hashes()
    train = load_train()
    original = json.loads((ROOT / 'notebooks/Yaqi/original_configs/catboost_location_coe.json').read_text(encoding='utf-8'))
    raw = build('catboost')
    assert experiment._json_value(experiment._new_model(raw).get_params(deep=False)) == original['parameters']
    adjusted = RecentWindowRegressor(FixedMixTargetRegressor(raw.estimator), months=raw.months)
    models = {'raw_rent': raw, 'fixed_mix_last': adjusted}
    sources = [ROOT / 'src/models/fixed_mix.py', ROOT / 'src/models/trees.py', Path(__file__)]
    write_json(output / 'study_config.json', {'seed': 42, 'features': blocks('catboost'),
        'kind': 'catboost', 'months': 24, 'threads': 4, 'split': 'rolling', 'final': False,
        'only_change': 'Fixed-mix target normalization; no new input features, weights, or parameter tuning',
        'index_groups': list(adjusted.estimator.groups),
        'index_statistics_fit_on': 'same recent 24 months as CatBoost; computed separately for each fold',
        'future_index': 'fixed at last training month; no validation labels',
        'normalization': 'y / (S_month / S_last_training_month); retain SGD-sized target',
        'data_hashes': before, 'python_executable': sys.executable,
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'threadpoolctl']},
        'source_hashes': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
    rows, predictions = [], {}
    with threadpool_limits(limits=4):
        for label, model in models.items():
            print(f'Start {label}; output={output}', flush=True)
            audit_dir = (output / label).relative_to(ROOT).as_posix()
            result = experiment.run_experiment('yaqi_catboost_' + label, 'Yaqi', blocks('catboost'),
                AuditFit(model, audit_dir), split='rolling', final=False, output_dir=output)
            predictions[label] = verify_result(result, train)
            if label == 'raw_rent':
                for fold, score in result.fold_rmse.items():
                    np.testing.assert_allclose(score, original['fold_rmse'][fold], atol=1e-6, rtol=0)
            rows.append({'variant': label, 'split_a': result.fold_rmse['holdout'],
                'split_b': result.fold_rmse['rolling_2024_04'], 'mean_rmse': result.mean_rmse,
                'run_id': result.run_id, 'run_dir': result.run_dir.relative_to(ROOT).as_posix()})
            write_json(output / 'completed_runs.json', rows)
    comparison = pd.DataFrame(rows)
    for score in ['split_a', 'split_b', 'mean_rmse']:
        comparison['delta_' + score] = comparison[score] - comparison.loc[comparison.variant.eq('raw_rent'), score].iloc[0]
    comparison.to_csv(output / 'comparison.csv', index=False)
    joined = predictions['raw_rent'].merge(predictions['fixed_mix_last'],
        on=['fold', 'row_index', DATE_COL, TARGET], suffixes=('_raw', '_fixed'), validate='one_to_one')
    assert len(joined) == 82329
    monthly = []
    for (fold, month), part in joined.groupby(['fold', DATE_COL], sort=True):
        row = {'fold': fold, DATE_COL: month, 'rows': len(part)}
        for label in ['raw', 'fixed']:
            error = part['Predicted_' + label] - part[TARGET]
            row[label + '_rmse'] = float(np.sqrt(np.mean(error ** 2)))
            row[label + '_bias'] = float(error.mean())
        row['delta_rmse'] = row['fixed_rmse'] - row['raw_rmse']
        monthly.append(row)
    pd.DataFrame(monthly).to_csv(output / 'monthly_comparison.csv', index=False)
    assert hashes() == before
    write_json(output / 'verification.json', {'status': 'passed', 'data_files_unchanged': len(before),
        'both_variants_actually_trained': True, 'raw_reference_scores_reproduced': True,
        'scores_recomputed_from_predictions': True, 'labels_rows_dates_verified': True,
        'paired_validation_rows': len(joined), 'all_predictions_original_sgd': True,
        'index_fit_inside_recent_window': True, 'validation_labels_not_used_in_index_or_forecast': True,
        'models_saved': False, 'formal_results_replaced': False})
    lines = ['# 固定房型指数目标归一化：CatBoost对照实验', '',
        '两种方法均本次实际训练。特征均为原有房屋/楼栋/地址、设施距离与COE，CatBoost参数不变。',
        '种子42、1000轮、深度7、学习率0.04、L2=10、4线程、最近24个月、公共A/B时间验证。', '',
        '固定指数只用当前折实际拟合窗口内五种房型的月度中位数，权重为该窗口内固定房型占比。',
        '1-room不决定指数，但其记录仍参与模型训练。最后训练月指数设为1，历史租金除以相应指数。',
        '未来指数持平为1，完全不使用验证月份租金。未加入平方样本权重，未调参或采用阻尼预测。', '',
        '| 方法 | A RMSE | B RMSE | 均值 | 均值变化 |', '|---|---:|---:|---:|---:|']
    for r in comparison.itertuples():
        lines.append(f'| {r.variant} | {r.split_a:.6f} | {r.split_b:.6f} | {r.mean_rmse:.6f} | {r.delta_mean_rmse:+.6f} |')
    lines += ['', 'RMSE单位SGD，越低越好；变化为新方法减对照，负数表示改善。',
        '这是已有开发验证窗口上的一次实验，A/B有重叠；不能当作Kaggle或独立测试分数。',
        '每折指数、权重和实际拟合范围见对应方法子目录；逐行预测与公共格式配置见experiments/。',
        '仅作比较，未执行最终test训练或替换results/Yaqi的三个正式模型。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(comparison.to_string(index=False), flush=True)
    print(f'Finished. Report: {output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
