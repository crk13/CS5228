# Yuxin：调优后的三个正式候选

三个运行均使用公共 rolling 两折及 `final=True`；实际预测已逐行复核，见 [verification.json](verification.json)。
完整方法、参数、搜索边界和证据见 [develop_tuning.md](../../develop_tuning.md)，初验与最新摘要见 [develop_test.md](../../develop_test.md)。

| 模型 | run_id | A RMSE | B RMSE | 平均 RMSE |
|---|---|---:|---:|---:|
| 4 个表格 Transformer 平均 | 20261103T181754695021Z_493b5f46 | 487.952332 | 474.705755 | **481.329044** |
| 市场 RNN＋4 个 MLP 属性成员 | 20261103T180702097497Z_0c01ed2c | 486.398628 | 469.853216 | **478.125922** |
| 市场 LSTM＋相同属性成员 | 20261103T180702083101Z_b8e1e8c0 | 486.337890 | 469.919411 | **478.128651** |

单位 SGD，越低越好。三者开发均值低于固定 CatBoost 参考 484.521404；但这是同类平均版本，不是单网络成绩，也不是 Kaggle 隐藏分数。
两折已用于调参，区间重叠，不能把 82329 行当作独立无偏测试样本。队友参考使用不同种子、环境及额外特征，不能据此作算法因果排名。

每个 `experiments/<run_id>/` 有 config、training_audit、validation_predictions；对应 `submissions/<run_id>.csv` 有完整 50000 行 `Id,Predicted`。
验证行数 A=47171、B=35158，融合评估按 `(fold,row_index)` 对齐。没有训练集拟合预测冒充时间留出预测，也不声称已证明团队融合收益。

carry 同属性对照均值为 **478.237340**，完整结果在 `notebooks/Yuxin/tuning/final/`，不占本目录 3 个名额。
RNN/LSTM 相对 carry 只改善约 0.11 SGD，主要提升来自属性模型调参与平均。序列是全市月度租金中位数，不是单套房。

原三个初始候选可从 `notebooks/Yuxin/archive/stage1_3/results_Yuxin/` 恢复；重放前的时间赢家另存 `notebooks/Yuxin/tuning/pre_replay/`。
未实现步骤 4 的时间 Transformer，未上传 Kaggle，未推送远端。复现使用 `notebooks/Yuxin/tuning/configs/` 冻结参数，勿向本目录重复追加运行。
