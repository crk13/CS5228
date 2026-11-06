# 新旧分支实验归档

| 阶段 | 配置记录 | 内容 |
|---|---:|---|
| [20261022](20261022/README.md) | 13 | 算法比较、全历史/12月/24月窗口、正则化、地址字段、误差分析及置换重要性 |
| [20261103](20261103/README.md) | 30 | 五种算法依次加入位置、COE、股票、持平及阻尼经济输入 |

两阶段统一使用 `comparison.csv`、`experiments/log.csv`、`experiments/<run_id>/config.json`、`validation_predictions.csv`、`submissions/`、`figures/`。这里是个人历史实验归档，融合负责人读取 [results/Yaqi](../../../results/Yaqi/README.md) 中的三个正式运行。

旧阶段直接从 GitHub `yaqi-tree-models` 的提交 `a69bdd546efa377888987b026f1da760e8d4e08f` 下载并展开原始嵌套压缩包，不以已经继续开发的本地旧工作区代替旧提交。后续阶段来自已实际训练并核验的本地30组结果，与原新分支的30组汇总相同。文件逐项来源、原路径、现路径、字节数和SHA-256见 [migration_manifest.json](migration_manifest.json)；本次实际核验见 [verification.json](verification.json)。

原始上传清单中，旧补丁 `yaqi_tree_models.patch` 的校验值与GitHub文件不符；该差异记入迁移清单。原结果压缩包、逐行预测和图表校验通过。补丁没有应用，也没有把旧公共框架覆盖到新框架。

模型权重、压缩包和旧补丁不纳入本次整理。原源码、Notebook、依赖和训练说明保留在各阶段 `source_snapshot/` 下，后缀追加 `.txt`，只供追溯；当前可运行代码使用 `src/models/trees.py`、`src/features/distance.py`、`src/features/macro.py` 和 [统一训练入口](../train_experiments.py)。原 `model_export.json` 等文件只记录当时的导出过程，不表示归档含有权重。

原报告、日志、配置保持原始字节；其中旧路径、旧交付说明和绝对解释器路径属于历史记录。当前运行、路径和交付要求以 [个人入口](../README.md) 和公共 [接口协议](../../../docs/experiment-interface.md) 为准。

本目录单独使用 `.gitattributes` 禁用归档文件的自动换行转换，使Windows和其他系统克隆后的SHA-256仍与原文件一致；未修改仓库的公共Git配置或忽略规则。

这次操作是文件和入口的统一，没有重新训练43组模型或重新调参。实际验证分数由现存逐行预测重新计算。两个时间验证窗口重叠，所有模型仍属于开发验证结果。
