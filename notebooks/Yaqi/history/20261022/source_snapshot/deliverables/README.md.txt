# Yaqi 树模型结果交付

此分支基于 `data-cleaning` 的 `6eb4f01`，主分支 `main` 保持不变。

- [中文实验报告](../outputs/trees/TREE_REPORT.md)
- [训练与复现说明](../docs/TREE_TRAINING.md)
- [Notebook](../notebooks/Yaqi_tree_models.ipynb)
- [13 组结果对比](../outputs/trees/comparison.csv)
- [分数及数据复核记录](../outputs/trees/verification.json)

`yaqi_tree_results.zip` 内含所有已完成实验及最终重训的逐行验证预测、最佳模型文件和 Kaggle 提交 CSV。不包含原始或清洗数据，避免重复上传。各文件 SHA-256 及压缩包校验值见 `manifest.json`。

下载后将压缩包解压到项目根目录，即可恢复 `outputs/trees/experiments/`、`outputs/trees/models/` 和 `outputs/trees/submissions/` 下的生成文件。模型重载需要当前分支的 `src` 模块及记录的 Python 依赖。

最佳配置为 CatBoost、最近 24 个月训练、增加邮编和楼号。开发时间验证的 RMSE 为 A=493.617、B=477.309 SGD，平均 485.463 SGD；不是 Kaggle 排行榜成绩。随机种子为 42；原仓库历史基线使用 5228，因此本次参考模型已重新运行。

提供给融合负责人的预测必须按 `(fold, row_index)` 对齐。两个验证窗口有时间重叠，应逐折处理。
