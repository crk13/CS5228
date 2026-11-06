"""Refit a validated fixed-mix model once and export a checked Kaggle CSV."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'notebooks/Yaqi'))
from delivery import SELECTED, build, blocks, hashes, write_json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import src.experiment as experiment
from src.data_cleaning import load_train, load_test, TARGET
from src.features import FeaturePipeline
from src.models.fixed_mix import FixedMixTargetRegressor
from src.models.trees import RecentWindowRegressor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=list(SELECTED), required=True)
    parser.add_argument('--study', required=True, help='Verified matched study, relative to project root')
    args = parser.parse_args()
    kind = args.kind
    study = (ROOT / args.study).resolve()
    study.relative_to(ROOT / 'outputs/Yaqi')
    config = json.loads((study / 'study_config.json').read_text(encoding='utf-8'))
    checked = json.loads((study / 'independent_verification.json').read_text(encoding='utf-8'))
    assert config['kind'] == kind and checked['status'] == 'passed'
    assert config['seed'] == 42 and config['months'] == 24 and config['features'] == blocks(kind)
    before = hashes()
    assert before == config['data_hashes'], 'Data changed since validation'
    for relative, expected in config['source_hashes'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    runs = json.loads((study / 'completed_runs.json').read_text(encoding='utf-8'))
    validated = next(r for r in runs if r['variant'] == 'fixed_mix_last')
    validated_config = json.loads((ROOT / validated['run_dir'] / 'config.json').read_text(encoding='utf-8'))
    raw = build(kind)
    template = RecentWindowRegressor(FixedMixTargetRegressor(raw.estimator), months=raw.months)
    recipe = experiment._json_value(experiment._new_model(template).get_params(deep=False))
    assert recipe == validated_config['parameters']['estimator']['parameters']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'outputs/Yaqi' / f'kaggle_fixed_mix_{kind}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    versions = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', kind, 'threadpoolctl']}
    for package, version in versions.items():
        assert version == config['versions'][package], f'Version changed: {package}'
    write_json(output / 'pending_config.json', {'kind': kind, 'seed': 42, 'features': blocks(kind),
        'parameters': recipe, 'versions': versions, 'validation_source': validated,
        'stage': 'final refit only; existing time-validation fits are not repeated'})
    train, test = load_train(), load_test()
    print(f'Final {kind} fit: available train={len(train)}, test={len(test)}, seed=42, recent=24 months', flush=True)
    with threadpool_limits(limits=4):
        experiment._seed()
        features = FeaturePipeline(blocks(kind))
        X_train = features.fit_transform(train)
        X_test = features.transform(test)
        fitted = experiment._new_model(template).fit(X_train, train[TARGET])
        prediction = fitted.predict(X_test)
    assert fitted.fit_start_ == '2023-04' and fitted.fit_end_ == '2025-03'
    assert fitted.fit_rows_ == 71040
    adjusted = fitted.estimator_
    np.testing.assert_array_equal(adjusted.index_for(X_test), np.ones(len(test)))
    submission_path = experiment.write_submission(test, prediction, output / f'{kind}_fixed_mix_submission.csv')
    frame = pd.read_csv(submission_path)
    example = pd.read_csv(ROOT / 'data/example-submission.csv')
    assert len(frame) == 50000 and list(frame) == list(example) == ['Id', 'Predicted']
    np.testing.assert_array_equal(frame.Id, test.Id)
    np.testing.assert_array_equal(frame.Id, example.Id)
    np.testing.assert_allclose(frame.Predicted, prediction, atol=1e-9, rtol=0)
    assert np.isfinite(frame.Predicted).all()
    assert hashes() == before
    index = adjusted.monthly_medians_.copy()
    index['FIXED_MIX_LEVEL'] = adjusted.market_levels_
    index['RELATIVE_INDEX'] = adjusted.relative_index_
    index.to_csv(output / 'final_training_index.csv', index_label='RENT_APPROVAL_DATE')
    receipt = {'status': 'passed', 'kind': kind, 'seed': 42, 'features': blocks(kind),
        'parameters': recipe, 'versions': versions, 'python_executable': sys.executable,
        'validation_source': validated, 'validation_was_not_repeated': True,
        'validation_score_is_not_kaggle_score': True,
        'available_train_rows': len(train), 'actual_fit_rows': fitted.fit_rows_,
        'fit_start': fitted.fit_start_, 'fit_end': fitted.fit_end_,
        'index_origin': adjusted.origin_, 'index_weights': adjusted.weights_.to_dict(),
        'index_reference_level': adjusted.reference_level_,
        'future_index_rule': 'last training month held constant',
        'economy_observations_through': '2025-02' if kind != 'xgboost' else None,
        'test_rows': len(frame), 'columns': list(frame), 'test_ids_match_loader_and_example': True,
        'predictions_finite': True, 'predictions_original_sgd': True,
        'min_prediction': float(frame.Predicted.min()), 'max_prediction': float(frame.Predicted.max()),
        'submission': submission_path.relative_to(ROOT).as_posix(),
        'submission_sha256': hashlib.sha256(submission_path.read_bytes()).hexdigest(),
        'data_hashes': before, 'datasets_unchanged': True, 'model_weights_saved': False,
        'source_hashes': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / 'src/models/fixed_mix.py', ROOT / 'src/models/trees.py',
                ROOT / 'src/features/pipeline.py', ROOT / 'src/features/distance.py', ROOT / 'src/features/macro.py']}}
    write_json(output / 'verification.json', receipt)
    print(json.dumps({key: receipt[key] for key in ['status', 'submission', 'actual_fit_rows',
          'test_rows', 'columns', 'min_prediction', 'max_prediction', 'submission_sha256']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
