"""Descriptive feasibility of a fixed-flat-type rental index; no model fitting."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.cv import get_splits, DATE_COL
from src.data_cleaning import load_train, TARGET

SEED = 42
GROUPS = ['2-room', '3-room', '4-room', '5-room', 'executive']


def hashes():
    paths = sorted((ROOT / 'data').rglob('*.csv')) + sorted((ROOT / 'data_cleaned').rglob('*.csv'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def calculate(frame, label, output):
    selected = frame.loc[frame.FLAT_TYPE.isin(GROUPS)]
    counts = selected.groupby([DATE_COL, 'FLAT_TYPE']).size().unstack().reindex(columns=GROUPS)
    assert counts.notna().all().all() and counts.gt(0).all().all()
    medians = selected.groupby([DATE_COL, 'FLAT_TYPE'])[TARGET].median().unstack().reindex(columns=GROUPS)
    weights = counts.sum() / counts.to_numpy().sum()
    shares = counts.div(counts.sum(axis=1), axis=0)
    fixed = medians.mul(weights, axis=1).sum(axis=1)
    changing = medians.mul(shares).sum(axis=1)
    raw = frame.groupby(DATE_COL)[TARGET].median()
    result = pd.DataFrame({'raw_monthly_median': raw, 'fixed_mix_level': fixed,
                           'changing_mix_level': changing,
                           'composition_effect_sgd': changing - fixed,
                           'composition_effect_pct': 100 * (changing / fixed - 1)})
    for name in ['raw_monthly_median', 'fixed_mix_level', 'changing_mix_level']:
        result[name + '_index'] = result[name] / result[name].iloc[:12].mean() * 100
    for group in GROUPS:
        result[group + '_median'] = medians[group]
        result[group + '_count'] = counts[group]
        result[group + '_share'] = shares[group]
    result.to_csv(output / f'{label}_monthly.csv', index_label=DATE_COL)
    detail = pd.DataFrame({'weight': weights, 'monthly_share_min': shares.min(),
        'monthly_share_max': shares.max(), 'monthly_count_min': counts.min(),
        'monthly_count_median': counts.median()})
    detail.to_csv(output / f'{label}_weights.csv', index_label='FLAT_TYPE')
    stats = {'from': str(result.index.min()), 'through': str(result.index.max()),
        'rows': len(frame), 'retained_for_index': len(selected), 'months': len(result),
        'excluded_1_room_rows': int(frame.FLAT_TYPE.eq('1-room').sum()),
        'weights': weights.to_dict(),
        'mean_abs_composition_effect_sgd': float(result.composition_effect_sgd.abs().mean()),
        'max_abs_composition_effect_sgd': float(result.composition_effect_sgd.abs().max()),
        'mean_abs_composition_effect_pct': float(result.composition_effect_pct.abs().mean()),
        'max_abs_composition_effect_pct': float(result.composition_effect_pct.abs().max()),
        'max_abs_composition_effect_month': result.composition_effect_pct.abs().idxmax(),
        'monthly_share_range_pp': ((shares.max() - shares.min()) * 100).to_dict(),
        'raw_vs_fixed_index_correlation': float(result.raw_monthly_median_index.corr(result.fixed_mix_level_index)),
        'raw_monthly_pct_change_std': float(100 * raw.pct_change().std()),
        'fixed_monthly_pct_change_std': float(100 * fixed.pct_change().std()),
        'note': 'In-sample descriptive statistics only; no validation labels used for this view.'}
    return result, shares, weights, stats


def main():
    before = hashes()
    frame = load_train()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / 'outputs/Yaqi' / ('fixed_mix_feasibility_' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    counts = frame.groupby([DATE_COL, 'FLAT_TYPE']).size().unstack(fill_value=0)
    support = pd.DataFrame({'total_rows': counts.sum(), 'monthly_min': counts.min(),
        'monthly_median': counts.median(), 'monthly_max': counts.max(),
        'missing_months': counts.eq(0).sum(), 'total_months': len(counts)})
    support.to_csv(output / 'flat_type_support.csv', index_label='FLAT_TYPE')
    counts.to_csv(output / 'monthly_counts.csv')
    monthly, shares, weights, full_stats = calculate(frame, 'all_train_descriptive', output)
    views = {'all_train_descriptive': full_stats}
    for fold in get_splits(frame, 'rolling'):
        _, _, _, stats = calculate(frame.iloc[fold.train_idx], fold.definition.name + '_training_only', output)
        assert stats['through'] < frame.iloc[fold.valid_idx][DATE_COL].min()
        views[fold.definition.name + '_training_only'] = stats

    # Within-month stratified bootstrap: uncertainty of the component medians.
    # Conditional on fixed weights and independent rows; duplicates/building clustering
    # may make these intervals too narrow. These are not forecast intervals.
    rng = np.random.default_rng(SEED)
    uncertainty = []
    groups = frame.loc[frame.FLAT_TYPE.isin(GROUPS)].groupby([DATE_COL, 'FLAT_TYPE'])[TARGET]
    for month in monthly.index:
        draws = np.zeros(200)
        for group in GROUPS:
            values = groups.get_group((month, group)).to_numpy()
            draws += weights[group] * np.median(rng.choice(values, size=(200, len(values))), axis=1)
        low, high = np.quantile(draws, [.025, .975])
        uncertainty.append({'month': month, 'level': monthly.loc[month, 'fixed_mix_level'],
                            'bootstrap_low': low, 'bootstrap_high': high,
                            'half_width_pct': 100 * (high - low) / (2 * monthly.loc[month, 'fixed_mix_level'])})
    uncertainty = pd.DataFrame(uncertainty)
    uncertainty.to_csv(output / 'bootstrap_diagnostic.csv', index=False)
    full_stats['bootstrap_median_half_width_pct'] = float(uncertainty.half_width_pct.median())
    full_stats['bootstrap_max_half_width_pct'] = float(uncertainty.half_width_pct.max())

    dates = pd.to_datetime(monthly.index)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, layout='constrained')
    for group in GROUPS:
        axes[0].plot(dates, shares[group] * 100, label=group)
    axes[0].set(ylabel='Share (%)', title='Observed flat-type shares (1-room excluded from index only)')
    axes[0].legend(ncol=5, loc='upper center', bbox_to_anchor=(.5, 1.3), frameon=False)
    axes[1].plot(dates, monthly.raw_monthly_median_index, label='Raw monthly median')
    axes[1].plot(dates, monthly.fixed_mix_level_index, label='Fixed mix of type medians')
    axes[1].set(ylabel='Index: 2021 mean = 100', title='Descriptive paths; independently rebased, different aggregations')
    axes[1].legend(frameon=False)
    axes[2].plot(dates, monthly.composition_effect_pct, color='#a4472e')
    axes[2].axhline(0, color='grey', linewidth=.8)
    axes[2].set(ylabel='Difference (%)', xlabel='Month', title='Composition effect: changing-weight / fixed-weight type medians - 1')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output / 'fixed_mix_feasibility.png', dpi=150)
    plt.close(fig)

    assert hashes() == before
    report = {'seed': SEED, 'python_executable': sys.executable,
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'matplotlib']},
        'support': support.reset_index().to_dict('records'), 'views': views,
        'method': 'S_t = sum_g w_g * median(y | month=t, type=g), weights fixed within each training view',
        'not_a_population_median': True,
        'forecast_or_model_trained': False, 'no_forecast_accuracy_claim': True,
        'descriptive_full_train_uses_all_known_train_labels': True,
        'fold_training_views_do_not_use_validation_rows': True,
        'bootstrap_caveat': '200 within-month stratified independent-row resamples, fixed weights; not forecast intervals; clustering not accounted for',
        'data_hashes': before, 'data_unchanged': True}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 固定房型构成指数：可行性检查', '',
        '仅做数据描述，不训练租金模型、不预测未来、不修改正式融合结果。', '',
        '用2-room、3-room、4-room、5-room、executive各月租金中位数，按当前训练视图中固定的房型占比加权。',
        '1-room仅不参与指数计算，原始及模型训练记录不删除。房型中位数的加权平均不是全体租金的中位数。', '',
        '全训练集视图用于描述已知历史。A/B视图分别只用对应训练部分重算权重与指数，没有读取该折验证标签。', '',
        '构成影响通过同一组房型中位数的变动权重和固定权重之差计算；直接比较整体中位数与加权指数不能把差异全归因于构成。', '',
        '样本支持见flat_type_support.csv；月度曲线与权重见各视图CSV；全部实测统计及数据指纹见report.json。', '',
        'Bootstrap仅为月内样本误差诊断，未考虑同楼栋相关性，不是未来预测区间。没有证明该指数可降低模型RMSE。', '',
        '![历史诊断](fixed_mix_feasibility.png)', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': output.as_posix(), 'support': report['support'], 'views': views}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
