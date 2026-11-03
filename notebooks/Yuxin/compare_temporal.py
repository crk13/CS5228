"""Stage 4: matched short-window controls, then bounded temporal-attention search.

Only the frozen property's already fitted OOF ratios are reused. Every market fit
receives the public fold's training labels, never its validation labels. The
selected recipe is subsequently independently retrained by run_experiments.py.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.cv import get_splits
from src.data_cleaning import load_train
from src.models.yuxin_nn.market import MarketForecaster, monthly_market
from notebooks.Yuxin.run_experiments import source_hashes, write_json

OUT = ROOT / "notebooks/Yuxin/stage4"
CARRY = ROOT / "notebooks/Yuxin/tuning/final/experiments/20261103T175001571910Z_a7a4588e"


def candidates():
    common = dict(lookback=6, hidden_size=8, epochs=80, learning_rate=.005,
                  damping=.8, inner_months=6, patience=25, random_state=5228,
                  threads=1, dropout=.1, weight_decay=.01, forecast_strength=.25)
    result = [(f"matched_{kind}", dict(common, kind=kind), "matched_L6")
              for kind in ("carry", "damped", "rnn", "lstm", "transformer")]
    # One-factor changes from the common Transformer, not an unbounded HPO claim.
    for key, value in (("lookback", 3), ("hidden_size", 16), ("learning_rate", .001),
                       ("damping", .95), ("forecast_strength", .5),
                       ("dropout", .2), ("inner_months", 12)):
        result.append((f"transformer_{key}_{value}", dict(common, kind="transformer", **{key: value}),
                       "bounded_attention_search"))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_train()
    reference = pd.read_csv(CARRY / "validation_predictions.csv")
    audits = json.loads((CARRY / "training_audit.json").read_text())
    source_config = json.loads((CARRY / "config.json").read_text())
    assert source_config["yuxin"]["data_hashes"] == source_hashes()
    contexts = []
    for fold in get_splits(train, "rolling"):
        fit, valid = train.iloc[fold.train_idx], train.iloc[fold.valid_idx]
        ref = reference.loc[reference.fold.eq(fold.definition.name)]
        np.testing.assert_array_equal(ref.row_index, fold.valid_idx)
        np.testing.assert_array_equal(ref.MONTHLY_RENT, valid.MONTHLY_RENT)
        audit = next(a for a in audits if a["origin"] == fit.RENT_APPROVAL_DATE.max())
        ratio = ref.Predicted.to_numpy()/ref.RENT_APPROVAL_DATE.map(audit["market_forecast"]).to_numpy()
        contexts.append((fold.definition.name, fit, valid, ratio))
    records, details = [], []
    for name, parameters, scope in candidates():
        scores, evidence = {}, []
        for fold_name, fit, valid, ratio in contexts:
            model = MarketForecaster(**parameters).fit(monthly_market(fit, fit.MONTHLY_RENT))
            prediction = ratio*model.predict_months(valid.RENT_APPROVAL_DATE)
            scores[fold_name] = float(np.sqrt(np.mean((prediction-valid.MONTHLY_RENT.to_numpy())**2)))
            months = pd.period_range(model.origin_, model.origin_+16, freq="M").astype(str)
            forecast = model.predict_months(months)
            np.testing.assert_array_equal(forecast, model.predict_months(months[::-1])[::-1])
            np.testing.assert_array_equal(forecast, [model.predict_months([m])[0] for m in months])
            evidence.append(dict(fold=fold_name, audit=model.audit_,
                                 market_forecast=dict(zip(months, forecast.tolist())),
                                 selection_history=model.selection_history_))
        row = dict(candidate=name, scope=scope, **scores, mean_rmse=float(np.mean(list(scores.values()))))
        records.append(row)
        details.append(dict(**row, parameters=parameters, fits=evidence))
        print(json.dumps(row), flush=True)
    best = min((d for d in details if d["parameters"]["kind"] == "transformer"), key=lambda d: d["mean_rmse"])
    recipe = json.loads((ROOT / "notebooks/Yuxin/tuning/configs/market_carry.json").read_text())
    recipe["parameters"].update(encoder="transformer", temporal_parameters=best["parameters"], n_jobs=4)
    recipe["search"] = dict(candidate=best["candidate"], selection="minimum two-fold development mean among 8 temporal Transformer recipes",
                            global_optimum_proven=False, property_model="frozen stage1-3 four-MLP equal mean",
                            independent_full_retrain_required=True, development_mean_rmse=best["mean_rmse"])
    write_json(OUT / "configs/market_transformer.json", recipe)
    pd.DataFrame(records).sort_values("mean_rmse").to_csv(OUT / "comparison.csv", index=False)
    write_json(OUT / "controlled_comparison.json", dict(
        scope="cached-property controlled experiment, not twelve independently fitted property ensembles",
        property_source=str(CARRY.relative_to(ROOT)), property_source_commit="aba5078",
        property_validation_sha256=hashlib.sha256((CARRY / "validation_predictions.csv").read_bytes()).hexdigest(),
        property_recipe=source_config["yuxin"]["recipe"], data_hashes=source_hashes(),
        temporal_source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (ROOT / "src/models/yuxin_nn/temporal.py", ROOT / "src/models/yuxin_nn/market.py")},
        candidates=details, selected=best["candidate"], all_candidates_completed_both_folds=True,
        selection_bias="Outer folds already used for HPO; no unbiased hidden-test claim"))


if __name__ == "__main__":
    main()
