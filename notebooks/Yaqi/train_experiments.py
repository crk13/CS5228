"""Run either historical experiment suite through the current public framework."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.config as config
config.SEED = 42
import src.experiment as experiment
experiment.SEED = 42
import src.features.distance
import src.features.macro
import src.models.baselines as baselines
import src.models.trees as trees
baselines.SEED = trees.SEED = 42
from src.features.distance import DISTANCE_COLUMNS
from src.features.macro import COE_COLUMNS, STOCK_COLUMNS
from threadpoolctl import threadpool_limits
from delivery import hashes, write_json

HISTORY = ROOT / 'notebooks/Yaqi/history'
SCENARIOS = {
    'base': ([], []),
    'location': (['distance'], DISTANCE_COLUMNS),
    'location_coe': (['distance', 'macro_coe'], DISTANCE_COLUMNS + COE_COLUMNS),
    'location_stocks': (['distance', 'macro_stocks'], DISTANCE_COLUMNS + STOCK_COLUMNS),
    'location_economy_last': (['distance', 'macro'], DISTANCE_COLUMNS + COE_COLUMNS + STOCK_COLUMNS),
    'location_economy_damped': (['distance', 'macro_damped'], DISTANCE_COLUMNS + COE_COLUMNS + STOCK_COLUMNS),
}


def recipes(suite):
    date = {'initial': '20261022', 'features': '20261103'}[suite]
    records = json.loads((HISTORY / date / 'runs.json').read_text(encoding='utf-8'))
    return [r for r in records if r['status'] == 'complete']


def build(suite, record, threads=4):
    if suite == 'initial':
        specification = {k: v for k, v in record['specification'].items() if k != 'name'}
        return ['basic', 'block'], trees.tree_model(**specification, threads=threads)
    block_names, numeric = SCENARIOS[record['scenario']]
    model = trees.tree_model(record['kind'], months=24, address=True,
                             threads=threads, extra_numeric=tuple(numeric))
    if record['kind'] in ('rf', 'extra'):
        model.set_params(estimator__estimator__prepare__categorical__onehot__max_categories=32)
    return ['basic', 'block'] + block_names, model


def check_recipe(suite, record):
    date = {'initial': '20261022', 'features': '20261103'}[suite]
    original = json.loads((HISTORY / date / 'experiments' / record['run_id'] / 'config.json').read_text(encoding='utf-8'))
    blocks, model = build(suite, record)
    actual = experiment._json_value(experiment._new_model(model).get_params(deep=False))
    if actual != original['parameters'] or blocks != original['features']:
        raise ValueError(f"Current recipe differs from historical parameters: {record['name']}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, choices=['initial', 'features'])
    parser.add_argument('--names', nargs='+', help='Exact experiment names; omit to run the suite')
    parser.add_argument('--list', action='store_true', help='Show recipes without training')
    parser.add_argument('--check-recipes', action='store_true', help='Compare all serialized parameters; no training')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--final', action='store_true', help='Also predict all test rows; no model weights saved')
    args = parser.parse_args()
    records = recipes(args.suite)
    if args.names:
        unknown = set(args.names) - {r['name'] for r in records}
        if unknown:
            parser.error(f'Unknown experiment names: {sorted(unknown)}')
        records = [r for r in records if r['name'] in args.names]
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.list:
        for r in records:
            print(f"{r['name']}: historical A={r['split_a']:.6f}, B={r['split_b']:.6f}")
        return
    for r in records:
        check_recipe(args.suite, r)
    if args.check_recipes:
        print(f'Passed: {len(records)} historical parameter recipes match current models (seed=42, threads=4)')
        return
    # Always create a fresh local output: archives and official fusion runs are immutable here.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'outputs/Yaqi' / f'{stamp}_{args.suite}'
    output.mkdir(parents=True, exist_ok=False)
    before = hashes()
    results = []
    with threadpool_limits(limits=args.threads):
        for r in records:
            blocks, model = build(args.suite, r, args.threads)
            result = experiment.run_experiment('yaqi_' + r['name'], 'Yaqi', blocks, model,
                split='rolling', final=args.final, output_dir=output)
            results.append({'name': r['name'], 'run_id': result.run_id,
                            'fold_rmse': result.fold_rmse, 'mean_rmse': result.mean_rmse,
                            'historical_mean_rmse': r['mean_rmse'],
                            'delta_from_history': result.mean_rmse - r['mean_rmse']})
            write_json(output / 'comparison.json', results)
    if hashes() != before:
        raise RuntimeError('Source data changed during training')
    write_json(output / 'provenance.json', {'suite': args.suite, 'seed': 42,
        'threads': args.threads, 'data_hashes': before, 'datasets_unchanged': True,
        'note': 'These are new runs; historical scores have not been replaced.'})
    print(f'Finished: {output}')


if __name__ == '__main__':
    main()
