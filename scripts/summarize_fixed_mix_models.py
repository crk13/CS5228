"""Combine three independently verified matched normalization studies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies', nargs=3, required=True)
    args = parser.parse_args()
    studies = [(ROOT / p).resolve() for p in args.studies]
    for study in studies:
        study.relative_to(ROOT / 'outputs/Yaqi')
    records, folds = [], []
    data_reference = None
    paired_reference = None
    index_reference = {}
    kinds = set()
    for study in studies:
        config = json.loads((study / 'study_config.json').read_text(encoding='utf-8'))
        check = json.loads((study / 'independent_verification.json').read_text(encoding='utf-8'))
        assert check['status'] == 'passed'
        assert check['underlying_model_parameters_and_features_identical']
        kind = config['kind']
        assert kind not in kinds
        kinds.add(kind)
        assert config['seed'] == 42 and config['months'] == 24 and config['threads'] == 4
        if data_reference is None:
            data_reference = config['data_hashes']
        assert config['data_hashes'] == data_reference
        for field in ['data_hashes', 'source_hashes']:
            for relative, expected in config[field].items():
                assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
        comparison = pd.read_csv(study / 'comparison.csv').set_index('variant')
        assert set(comparison.index) == {'raw_rent', 'fixed_mix_last'}
        raw, fixed = comparison.loc['raw_rent'], comparison.loc['fixed_mix_last']
        records.append({'kind': kind, 'features': '+'.join(config['features']),
            'raw_split_a': raw.split_a, 'fixed_split_a': fixed.split_a,
            'delta_split_a': fixed.split_a - raw.split_a,
            'raw_split_b': raw.split_b, 'fixed_split_b': fixed.split_b,
            'delta_split_b': fixed.split_b - raw.split_b,
            'raw_mean_rmse': raw.mean_rmse, 'fixed_mean_rmse': fixed.mean_rmse,
            'delta_mean_rmse': fixed.mean_rmse - raw.mean_rmse,
            'improvement_pct': 100 * (raw.mean_rmse - fixed.mean_rmse) / raw.mean_rmse,
            'study': study.relative_to(ROOT).as_posix()})
        for label in ['raw_rent', 'fixed_mix_last']:
            run = comparison.loc[label]
            frame = pd.read_csv(ROOT / run.run_dir / 'validation_predictions.csv')
            keys = frame.drop(columns='Predicted')
            if paired_reference is None:
                paired_reference = keys
            pd.testing.assert_frame_equal(keys, paired_reference)
            for fold, part in frame.groupby('fold', sort=False):
                score = float(np.sqrt(np.mean((part.Predicted - part.MONTHLY_RENT) ** 2)))
                expected = run.split_a if fold == 'holdout' else run.split_b
                np.testing.assert_allclose(score, expected, atol=1e-9, rtol=0)
                folds.append({'kind': kind, 'variant': label, 'fold': fold,
                              'rmse': score, 'validation_rows': len(part)})
        for origin in ['2023-11', '2024-03']:
            path = study / 'fixed_mix_last' / f'index_through_{origin}.csv'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if origin not in index_reference:
                index_reference[origin] = digest
            assert digest == index_reference[origin], 'All algorithms must use identical market index'
    assert kinds == {'catboost', 'xgboost', 'lightgbm'}
    summary = pd.DataFrame(records).sort_values('fixed_mean_rmse')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'outputs/Yaqi' / ('fixed_mix_three_' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(folds).to_csv(output / 'fold_scores.csv', index=False)
    verification = {'status': 'passed', 'models': summary.kind.tolist(),
        'studies': [p.relative_to(ROOT).as_posix() for p in studies],
        'all_six_variant_predictions_aligned': True, 'all_twelve_fold_scores_recomputed': True,
        'same_training_data_and_market_indices': True, 'index_hashes': index_reference,
        'source_and_data_hashes_verified': True,
        'catboost_source': 'Earlier completed matched experiment from this conversation; not unnecessarily rerun',
        'formal_results_replaced': False, 'model_or_blend_optimality_claim': False}
    (output / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 三个正式树模型：固定房型指数目标归一化比较', '',
        'CatBoost沿用本日先前完成并独立核验的对照实验；XGBoost和LightGBM本轮实际补训原版与归一化版。',
        '三个算法均保持各自正式模型特征与参数；这不是在所有特征组合上重新调参。',
        'CatBoost：位置+COE，1000轮；XGBoost：基础版，700轮；LightGBM：位置+COE，700轮。',
        '全部种子42、4线程、最近24月、公共A/B验证；仅改变目标归一化。',
        '五种房型固定构成指数，最后训练月设为1，预测未来时持平；指数完全在每折实际训练部分计算。',
        '1-room不决定指数但保留训练记录；没有额外样本权重或指数趋势预测。', '',
        '| 模型 | 原版均值 | 指数版均值 | 变化（新减旧） | 降幅 |', '|---|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.kind} | {r.raw_mean_rmse:.6f} | {r.fixed_mean_rmse:.6f} | {r.delta_mean_rmse:+.6f} | {r.improvement_pct:.3f}% |')
    lines += ['', '| 模型 | 原版A | 指数版A | 原版B | 指数版B |', '|---|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.kind} | {r.raw_split_a:.6f} | {r.fixed_split_a:.6f} | {r.raw_split_b:.6f} | {r.fixed_split_b:.6f} |')
    lines += ['', 'RMSE单位SGD，越低越好；均值是A/B的简单平均。两个窗口重叠且已用于模型选择，属于开发验证。',
        '数据与源代码指纹、全部六份逐行预测对齐、十二个折分数重算及三个算法使用相同指数的检查通过。',
        '尚未训练融合模型或确认Kaggle效果；未替换results/Yaqi中的三个正式运行。', '', '## 各模型原始运行与预测', '']
    for r in summary.itertuples():
        lines.append(f'- [{r.kind}报告](../{Path(r.study).name}/REPORT.md)，目录内experiments保存原始配置及验证预测。')
    lines += ['', '## 复现', '', '```powershell',
        'python scripts/train_fixed_mix_model.py --kind catboost',
        'python scripts/train_fixed_mix_model.py --kind xgboost',
        'python scripts/train_fixed_mix_model.py --kind lightgbm', '```',
        '每次输出到新的outputs/Yaqi目录，不覆盖已完成实验。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary.to_string(index=False))
    print(f'Finished. Report: {output / "REPORT.md"}')


if __name__ == '__main__':
    main()
