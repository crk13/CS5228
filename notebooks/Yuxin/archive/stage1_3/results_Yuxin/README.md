# Yuxin：Transformer、RNN、LSTM 候选预测

本目录按公共 `docs/experiment-interface.md` 的交付格式保存三个运行：
表格 Transformer、市场 RNN＋属性 MLP、市场 LSTM＋属性 MLP。
完整实测结果、局限性和检查证据见`doc/develop_test_yuxin_nn.md`。

| 模型 | run_id | A RMSE | B RMSE | 平均 RMSE |
|---|---|---:|---:|---:|
| 表格 Transformer | 20261103T165254207013Z_e7366de1 | 515.646 | 484.467 | 500.056 |
| 市场 RNN | 20261103T165254212024Z_ae71e28f | 497.557 | 482.783 | 490.170 |
| 市场 LSTM | 20261103T165436355139Z_16342354 | 498.832 | 481.537 | 490.184 |

本轮 carry 对照平均 RMSE 为 485.611，优于这里的两个循环网络。
这些是已实现模型的初始候选，不是已证明优于对照的融合推荐。

每个运行使用公共 `split="rolling", final=True`，种子 5228，原尺度 SGD RMSE。
每折和最终训练均重新建立词表、数值统计和网络；epoch 由训练部分内部时间验证选择。
市场模型不假定同楼栋的出租记录属于同一套房，月度聚合只来自当前训练折。

```text
experiments/log.csv
experiments/<run_id>/config.json
experiments/<run_id>/training_audit.json
experiments/<run_id>/validation_predictions.csv
submissions/<run_id>.csv
verification.json
```

融合读取 `validation_predictions.csv`，按 `(fold,row_index)` 对齐；A 有 47171 行，B 有 35158 行。
两折有重叠，每个模型总计 82329 行，不能当作 82329 个独立验证样本。
最终 `submissions/<run_id>.csv` 有 50000 行 `Id,Predicted`，预测顺序对应 test。

MLP、carry 和公共参考基线在 `notebooks/Yuxin/validation/`，不占用本目录三个运行。
这里保存的是阶段 1—3 的真实候选结果；符合格式不代表胜过参考模型或已证实融合收益。
没有上传 Kaggle，也没有在此阶段训练融合器。完整文件可用个人 `verify_delivery.py` 独立复核。
