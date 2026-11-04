# 固定房型指数：三个树模型的目标归一化实验

本目录归档2026-11-04实际完成的固定房型指数实验：CatBoost、XGBoost、LightGBM各有原租金与指数归一化两种版本，共六次运行。每次均使用公共框架的A/B时间验证，保留原始配置、逐行验证预测、指数和独立核对记录。

这是个人对比实验，尚未作为新一轮完整融合交付替换 [results/Yaqi](../../../results/Yaqi/README.md)。正式目录仍保留原来的三个运行。指数版目前只有CatBoost生成了最终测试集预测；本轮没有导出三个指数模型的训练集诊断预测，也没有保存或上传模型权重。

## 文件入口

| 内容 | 位置 |
|---|---|
| 六个运行的当前路径索引 | [experiment_index.csv](experiment_index.csv) |
| 三个模型的本地验证对比 | [汇总报告](fixed_mix_three_20261104T071053009381Z/REPORT.md)、[对比CSV](fixed_mix_three_20261104T071053009381Z/comparison.csv) |
| CatBoost实验与逐行预测 | [CatBoost报告](fixed_mix_catboost_20261104T064729648979Z/REPORT.md) |
| XGBoost实验与逐行预测 | [XGBoost报告](fixed_mix_xgboost_20261104T070922647746Z/REPORT.md) |
| LightGBM实验与逐行预测 | [LightGBM报告](fixed_mix_lightgbm_20261104T070922018042Z/REPORT.md) |
| 房型样本支持与指数可行性检查 | [可行性报告](fixed_mix_feasibility_20261104T064239Z/REPORT.md) |
| CatBoost最终测试集预测 | [catboost_fixed_mix_submission.csv](kaggle_fixed_mix_catboost_20261104T071258185386Z/catboost_fixed_mix_submission.csv) |
| 最终训练和提交文件核验 | [verification.json](kaggle_fixed_mix_catboost_20261104T071258185386Z/verification.json) |
| 来源、文件指纹和迁移映射 | [manifest.json](manifest.json) |

每个运行的验证预测列为 `row_index,fold,RENT_APPROVAL_DATE,MONTHLY_RENT,Predicted`。A/B验证窗口重叠，必须按 `(fold,row_index)` 对齐。原日志与配置保持原始字节，其中的 `outputs/Yaqi/...` 是当时本地生成路径；GitHub内现有文件位置以 `experiment_index.csv` 为准。

## 方法与信息边界

在每折实际拟合的最近24个月内，分别计算2-room、3-room、4-room、5-room、executive的月度租金中位数，按该窗口内固定房型占比加权。1-room不参与指数计算，但所有房型的原训练记录仍保留。

令月度加权水平为 `S_t`，最后训练月为 `M`，训练目标为 `MONTHLY_RENT / (S_t / S_M)`。这样保留SGD大小的数值尺度。未来指数按最后训练月持平，预测还原为原始租金尺度；没有读取验证期租金，也没有加入新的输入特征、样本权重或趋势预测。

CatBoost仍使用位置与COE，1000轮；XGBoost仍使用基础特征，700轮；LightGBM仍使用位置与COE，700轮。种子42、4线程，模型参数与各自原版保持一致。经济数据按现有框架的预测时点可得性处理。

最终CatBoost使用完整训练集作为可用输入，按既定最近24个月方案实际拟合2023-04至2025-03的71040条记录，预测test全部50000行，`Id`与样例一致。复用已经完成的验证结果，仅额外进行一次最终训练，没有伪造或重复标记历史训练。

## 复现与核对

使用项目已有Python环境，在项目根目录运行。未修改公共框架或安装新依赖。辅助脚本位于 `scripts/`，模型实现位于 [src/models/fixed_mix.py](../../../src/models/fixed_mix.py)。

```powershell
python -m unittest discover -s tests -v
python scripts/verify_published_fixed_mix.py
python scripts/train_fixed_mix_model.py --kind catboost
python scripts/train_fixed_mix_model.py --kind xgboost
python scripts/train_fixed_mix_model.py --kind lightgbm
```

训练命令始终写入新的 `outputs/Yaqi/` 目录。若要生成最终提交，对新训练输出执行：

```powershell
python scripts/export_fixed_mix_submission.py --kind catboost --study outputs/Yaqi/<刚生成的CatBoost实验目录>
```

`source_snapshot/` 保留实际训练时源文件的原始字节，供核查当时的指纹；当前可运行代码在 `src/` 和 `scripts/`。归档目录单独禁用Git换行转换，源代码和数据另记录LF标准化指纹，使跨系统核验不受换行格式影响。
