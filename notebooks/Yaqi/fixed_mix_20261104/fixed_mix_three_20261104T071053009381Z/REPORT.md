# 三个正式树模型：固定房型指数目标归一化比较

CatBoost沿用本日先前完成并独立核验的对照实验；XGBoost和LightGBM本轮实际补训原版与归一化版。
三个算法均保持各自正式模型特征与参数；这不是在所有特征组合上重新调参。
CatBoost：位置+COE，1000轮；XGBoost：基础版，700轮；LightGBM：位置+COE，700轮。
全部种子42、4线程、最近24月、公共A/B验证；仅改变目标归一化。
五种房型固定构成指数，最后训练月设为1，预测未来时持平；指数完全在每折实际训练部分计算。
1-room不决定指数但保留训练记录；没有额外样本权重或指数趋势预测。

| 模型 | 原版均值 | 指数版均值 | 变化（新减旧） | 降幅 |
|---|---:|---:|---:|---:|
| catboost | 484.521404 | 482.836861 | -1.684543 | 0.348% |
| xgboost | 485.814146 | 483.154808 | -2.659338 | 0.547% |
| lightgbm | 492.244869 | 490.273905 | -1.970964 | 0.400% |

| 模型 | 原版A | 指数版A | 原版B | 指数版B |
|---|---:|---:|---:|---:|
| catboost | 492.874313 | 491.050889 | 476.168495 | 474.622834 |
| xgboost | 495.948263 | 492.286691 | 475.680030 | 474.022925 |
| lightgbm | 501.918151 | 499.283040 | 482.571586 | 481.264769 |

RMSE单位SGD，越低越好；均值是A/B的简单平均。两个窗口重叠且已用于模型选择，属于开发验证。
数据与源代码指纹、全部六份逐行预测对齐、十二个折分数重算及三个算法使用相同指数的检查通过。
尚未训练融合模型或确认Kaggle效果；未替换results/Yaqi中的三个正式运行。

## 各模型原始运行与预测

- [catboost报告](../fixed_mix_catboost_20261104T064729648979Z/REPORT.md)，目录内experiments保存原始配置及验证预测。
- [xgboost报告](../fixed_mix_xgboost_20261104T070922647746Z/REPORT.md)，目录内experiments保存原始配置及验证预测。
- [lightgbm报告](../fixed_mix_lightgbm_20261104T070922018042Z/REPORT.md)，目录内experiments保存原始配置及验证预测。

## 复现

```powershell
python scripts/train_fixed_mix_model.py --kind catboost
python scripts/train_fixed_mix_model.py --kind xgboost
python scripts/train_fixed_mix_model.py --kind lightgbm
```
每次输出到新的outputs/Yaqi目录，不覆盖已完成实验。
