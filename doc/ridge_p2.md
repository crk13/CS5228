# P2: Ridge on the market-level ratio

Statistical model of the team: a Ridge regression that predicts each flat's rent **relative to the monthly city-wide median rent**, then scales the prediction back to SGD.

## Model

- **Target**: monthly rent divided by the median rent of all training rows in the same `RENT_APPROVAL_DATE` month (the "market level").
- **Prediction**: predicted ratio × the market level of the last training month (carried forward unchanged). The level is 3100 at the end of both validation training periods and 3200 for the final model (2025-03).
- **Features**: one-hot subzone, block ID (`BLOCK` + `STREET`), flat type and flat model; numeric floor area, flat age (rent year − lease commence year) and distance to the nearest opened MRT station (km, from latitude/longitude).
- **Alpha**: chosen inside the training period only (last 6 training months: fit on the first 4, evaluate on the last 2), grid 0.3–100. It is 10 on both folds.
- **Code**: model `src/models/market_ratio.py` (`MarketRatioRidge`), feature block `src/features/mrt_distance.py` (registered as `mrt_distance`).

## Results (validation RMSE, SGD)

| Fold | RMSE |
|---|---:|
| holdout (train 2021-01–2023-11, validate 2023-12–2025-03) | 488.69 |
| rolling_2024_04 (train 2021-01–2024-03, validate 2024-04–2025-03) | 471.94 |
| mean | 480.32 |

Kaggle public leaderboard (model retrained on all training data): **444.84** with level 3200; 448.00 with the mean of the last 3 months (3150).

Step-by-step results on both folds, from the lookup baseline (525 / 506) to the final model, are printed by `notebooks/FXY/p2_linear_ridge.py`. The largest single gain is the ratio target: the same Ridge on all months scores 761 / 711 on raw rent and 489 / 472 on the ratio.

Robustness checks (not part of the delivery): choosing alpha on the ratio target, fitting the inner model on all but the last two training months, and standardising the numeric features all keep alpha at 10 and change the RMSE by at most 0.5.

## Notes

- Rents below 2000 are about 4% of the holdout validation rows but about 28% of its squared error; they are owner-declared and cannot be separated by any column, so they are kept.
- The market level of the test period is unknown and assumed flat.

## Files and reproduction

- Delivery for fusion: `results/FXY/` (validation, training and test predictions, config, log; see its README).
- Reproduce the delivery: `python notebooks/FXY/p2_framework_run.py --deliver`
- Reproduce the step-by-step table and figures: `python notebooks/FXY/p2_linear_ridge.py`
