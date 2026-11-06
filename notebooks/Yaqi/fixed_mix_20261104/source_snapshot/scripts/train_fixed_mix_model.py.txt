"""Matched raw/index-adjusted runs for a finalized tree-model recipe."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'notebooks/Yaqi'))
from delivery import SELECTED, build, blocks, hashes, write_json
from train_fixed_mix import AuditFit, verify_result
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import src.experiment as experiment
from src.data_cleaning import load_train, TARGET
from src.cv import DATE_COL
from src.models.fixed_mix import FixedMixTargetRegressor
from src.models.trees import RecentWindowRegressor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=list(SELECTED))
    args = parser.parse_args()
    kind = args.kind
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'outputs/Yaqi' / f'fixed_mix_{kind}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    before = hashes()
    train = load_train()
    original = json.loads((ROOT / f'notebooks/Yaqi/original_configs/{kind}_{SELECTED[kind]}.json').read_text(encoding='utf-8'))
    raw = build(kind)
    assert experiment._json_value(experiment._new_model(raw).get_params(deep=False)) == original['parameters']
    adjusted = RecentWindowRegressor(FixedMixTargetRegressor(raw.estimator), months=raw.months)
    source_paths = [Path(__file__), ROOT / 'scripts/train_fixed_mix.py',
        ROOT / 'src/models/fixed_mix.py', ROOT / 'src/models/trees.py',
        ROOT / 'notebooks/Yaqi/delivery.py', ROOT / 'src/features/distance.py', ROOT / 'src/features/macro.py']
    write_json(output / 'study_config.json', {'seed': 42, 'features': blocks(kind),
        'kind': kind, 'scenario': SELECTED[kind], 'months': 24, 'threads': 4, 'split': 'rolling', 'final': False,
        'only_change': 'Fixed-mix target normalization; no new input features, weights, or parameter tuning',
        'index_groups': list(adjusted.estimator.groups),
        'index_statistics_fit_on': 'same recent 24 months as regressor; separately for each fold',
        'future_index': 'fixed at last training month; no validation labels',
        'normalization': 'y / (S_month / S_last_training_month); retain SGD-sized target',
        'data_hashes': before, 'python_executable': sys.executable,
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'xgboost', 'lightgbm', 'threadpoolctl']},
        'source_hashes': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}})
    rows, predictions = [], {}
    with threadpool_limits(limits=4):
        for label, model in {'raw_rent': raw, 'fixed_mix_last': adjusted}.items():
            print(f'Start {kind}/{label}; output={output}', flush=True)
            audit_dir = (output / label).relative_to(ROOT).as_posix()
            result = experiment.run_experiment(f'yaqi_{kind}_{label}', 'Yaqi', blocks(kind),
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
    paired = predictions['raw_rent'].merge(predictions['fixed_mix_last'],
        on=['fold', 'row_index', DATE_COL, TARGET], suffixes=('_raw', '_fixed'), validate='one_to_one')
    assert len(paired) == 82329
    monthly = []
    for (fold, month), part in paired.groupby(['fold', DATE_COL], sort=True):
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
        'paired_validation_rows': len(paired), 'all_predictions_original_sgd': True,
        'index_fit_inside_recent_window': True, 'validation_labels_not_used_in_index_or_forecast': True,
        'models_saved': False, 'formal_results_replaced': False})
    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_fixed_mix.py'),
                    '--output', output.relative_to(ROOT).as_posix()], check=True, cwd=ROOT)
    lines = [f'# {kind}：固定房型指数目标归一化对照', '',
        f'沿用正式配置 {SELECTED[kind]}。种子42、4线程、最近24个月、公共A/B时间验证。',
        '两种方法均本次实际训练；仅目标变换不同，特征与模型参数一致。',
        '每折实际拟合窗口内计算五种房型固定权重与月度中位数；最后训练月指数设为1，未来持平。',
        '1-room不参与指数计算，但其记录仍参与模型训练；不加额外样本权重，不调参。', '',
        '| 方法 | A RMSE | B RMSE | 均值 | 均值变化 |', '|---|---:|---:|---:|---:|']
    for r in comparison.itertuples():
        lines.append(f'| {r.variant} | {r.split_a:.6f} | {r.split_b:.6f} | {r.mean_rmse:.6f} | {r.delta_mean_rmse:+.6f} |')
    lines += ['', 'RMSE单位SGD；变化为新方法减原方法，负数表示改善。',
        'A/B窗口重叠，属于开发验证，尚非Kaggle成绩；未执行最终test训练或替换正式交付。',
        '独立重算分数、重建指数和数据指纹检查见independent_verification.json。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(comparison.to_string(index=False), flush=True)
    print(f'Finished: {output}', flush=True)


if __name__ == '__main__':
    main()
