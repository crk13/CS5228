# 阶段 1—3 初验归档

此处完整保留调参之前的三个正式运行及初验比较、逐月误差和复核摘要，没有删除或改写其预测。

- `results_Yuxin/`：原 Transformer/RNN/LSTM 三个候选，run_id 与 CSV 内容保持不变。
- `comparison.csv`、`monthly_errors.csv`、`verification.json`：初验时的分析证据。

历史 JSON 中的原输出目录反映当时位置；现在按此归档目录读取。
原基线、原 MLP 和原 carry 仍在 `notebooks/Yuxin/validation/`。
最新正式候选在 `results/Yuxin/`，最新调参说明见 [develop_tuning.md](../../../../develop_tuning.md)。
