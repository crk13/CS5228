# 位置与经济特征实验：30组

五种算法：CatBoost、XGBoost、LightGBM、RandomForest、ExtraTrees。每种算法六个阶段：base、location、location_coe、location_stocks、location_economy_last、location_economy_damped。

- [30组比较](comparison.csv)
- [特征实验报告](FEATURE_REPORT.md)
- [森林模型补充报告](FOREST_REPORT.md)
- [原数据与信息边界核验](verification.json)
- [全部运行日志](experiments/log.csv)

最佳配置为 CatBoost＋位置＋COE：A=492.874313，B=476.168495，均值484.521404 SGD。所有配置固定最近24月训练、地址字段、种子42及相同A/B窗口；森林本阶段限制one-hot类别数，与早期森林参数有差异，应优先在本阶段内部比较特征效果。

共30个验证运行目录，另附4份历史test提交及相应导出记录（最佳、位置版、两种森林）。当时最终模型在独立导出步骤中训练，没有额外生成公共框架运行目录。

报告原样保留，模型权重未纳入；当前源码和复现方式见 [个人入口](../../README.md)。与组员融合时使用 [当前三个正式模型](../../../../results/Yaqi/README.md)，历史提交不自动进入融合。
