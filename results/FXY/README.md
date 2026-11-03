# FXY: Ridge on rent / market level (carried forward)

Model: Ridge regression on the ratio of monthly rent to the city-wide median rent of the
same month, with one-hot subzone, block, flat type and flat model plus floor area, flat age
and distance to the nearest open MRT station. A prediction is ratio * the market level of the
last training month (the level is carried forward unchanged). Alpha is chosen inside the
training months only (last 6 months, final 2 as inner validation).
Features: `block`, `mrt_distance`. Split: `rolling`, `final=True`, seed 5228.

Run id: `20261103T110030459326Z_838f1908`

| Fold | RMSE |
|---|---:|
| holdout | 488.690 |
| rolling_2024_04 | 471.943 |
| mean | 480.316 |

Files (under `experiments/<run_id>/` unless noted):

| File | Columns / content |
|---|---|
| `validation_predictions.csv` | `row_index,fold,RENT_APPROVAL_DATE,MONTHLY_RENT,Predicted` (82329 rows; use this for fusion) |
| `training_predictions.csv` | `row_index,fold,RENT_APPROVAL_DATE,MONTHLY_RENT,Predicted,used_for_model_fit` (150000 rows, fold=`final`) |
| `fold_training_predictions.csv` | same columns; each fold's model on its own training rows (217671 rows) |
| `config.json` | run configuration, fold sizes, scores |
| `experiments/log.csv` | one-line log of this run |
| `submissions/<run_id>.csv` | `Id,Predicted` for the 50000 test rows |

The two training files are fitted-value diagnostics, not out-of-sample predictions.
`used_for_model_fit` is True everywhere because the model uses all months of its training part.

Reproduce (from the repository root):

    python notebooks/FXY/p2_framework_run.py --deliver
