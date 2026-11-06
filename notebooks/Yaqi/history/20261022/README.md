# 早期树模型实验：13组

从旧分支 `yaqi-tree-models` 原交付包展开，去掉模型权重，保留全部验证预测、最佳模型提交、参数及分析。

- [13组比较](comparison.csv)
- [原实验报告和图表](TREE_REPORT.md)
- [实际拟合月份与行数](actual_fit_windows.csv)
- [置换重要性](permutation_importance.csv)
- [原验证记录](verification.json)
- [全部运行日志](experiments/log.csv)

本阶段最好的配置是 `catboost_address_24m`：A=493.617496，B=477.308813，均值485.463155 SGD。后续阶段 `catboost_base` 复现了这组配置及全部验证预测。本阶段尚未加入新的设施距离和经济数据。

13组配置之外另有一次最佳配置的最终重训，共14个运行目录，附50000行test提交。没有当时未导出的训练集预测文件，未补造历史训练预测。三个正式融合模型的训练诊断和验证预测见 [当前交付](../../../../results/Yaqi/README.md)。

当前可用路径见 [统一实验索引](../../experiment_catalog.csv)；原日志内旧 `outputs/trees/` 路径只是历史记录。复现时使用 [统一训练入口](../../README.md)，不要直接运行 `source_snapshot/` 中的旧脚本。
