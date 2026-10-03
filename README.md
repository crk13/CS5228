# CS5228：新加坡 HDB 月租金预测

CS5228 小组项目，参加 Kaggle 私有比赛 **CS5228-2610 Project**。根据新加坡 HDB 组屋属性预测 `MONTHLY_RENT`，任务为回归，评估指标为原始租金尺度的 **RMSE**（SGD）。

课程要求对使用或不使用的每类信息，尤其辅助数据，给出论证；报告与分数同等重要。目前已完成基础数据清洗、公共实验框架和两个参考基线，其余模型与特征由组员接入。

## 关键文档与入口

| 文档 / 入口 | 内容 |
|---|---|
| [任务说明原文](CS5228-2610%20Project%20_%20Kaggle.mhtml) | 本地保存的 Kaggle 比赛说明 |
| [AI 开发约定](AGENTS.md) | 数据读取、统一接口、防泄漏与验证要求 |
| [清洗数据说明与数据字典](data_cleaned/README.md) | 清洗步骤、字段类型、已知现象与辅助数据注意事项 |
| [公共特征接口](src/features/base.py) / [注册表](src/features/registry.py) | 特征块扩展与开关 |
| [统一模型接口](src/models/base.py) / [实验入口](src/experiment.py) | 模型接入、目标变换和实验产物 |
| [共享验证方案](src/cv.py) / [框架测试](tests/test_framework.py) | 时间切分及契约检查 |
| [实验日志](experiments/log.csv) / [依赖清单](requirements.txt) | 可比较的实验记录与运行依赖 |

## 项目结构

```text
Project_export/
├── AGENTS.md                       # AI 执行约定
├── README.md                       # 项目说明与团队使用指南
├── requirements.txt
├── CS5228-2610 Project _ Kaggle.mhtml
├── data/                           # 原始数据
│   ├── train.csv
│   ├── test.csv
│   ├── example-submission.csv
│   └── auxiliary/
├── data_cleaned/                    # 基础清洗后的建模数据
│   ├── README.md                    # 数据字典与清洗说明
│   ├── train.csv
│   ├── test.csv                     # 已添加 Id
│   └── auxiliary/
├── src/
│   ├── data_cleaning/              # 清洗、schema、loader 与数据校验
│   ├── features/                   # 特征接口、注册表、basic / block
│   ├── models/                     # 模型包装、目标变换与两个基线
│   ├── config.py                   # 项目路径与固定种子
│   ├── cv.py                       # 共用 holdout / rolling
│   ├── experiment.py               # 训练、评分、日志和提交
│   └── run_baseline.py              # 可运行的基线入口
├── tests/
│   └── test_framework.py
├── experiments/
│   ├── log.csv                     # 共享实验日志
│   └── <run_id>/
│       ├── config.json
│       └── validation_predictions.csv  # 运行生成，Git 忽略
├── submissions/                    # 运行生成的提交 CSV，Git 忽略
└── notebooks/                      # 个人实验约定目录，尚未创建
```

## 快速开始

需要 Python ≥ 3.10，依赖为 NumPy、pandas、scikit-learn ≥ 1.3 和 threadpoolctl。测试使用标准库 unittest。在项目根目录运行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.run_baseline
```

基线入口在 holdout 和 rolling 上运行两个模型，打印各折和平均 RMSE，追加实验日志，并用全量训练数据重训 HistGradientBoosting，生成一个合法提交。入口将计算线程数限制为 4。

- **分组中位数**：使用当前训练折最后 6 个日历月的 `TOWN × FLAT_TYPE` 中位数。未见组合回退到同期户型中位数，再回退到同期全局中位数。
- **HistGradientBoostingRegressor**：使用 `basic + block`，在模型内部编码类别，保留缺失值，关闭随机提前停止验证，不依赖 LightGBM 等额外模型库。

当前默认配置的本地结果见 [共享日志](experiments/log.csv)：

| 基线 | Holdout RMSE | Rolling 平均 RMSE |
|---|---:|---:|
| 最近六个月分组中位数 | 524.659 | 515.448 |
| HistGradientBoosting | 497.937 | 489.241 |

## 数据与建模背景

原始训练集有 150000 行，包含目标；测试集有 50000 行，原始文件没有 Id。清洗后的 test 已添加行号 Id 0..49999。提交格式为 `Id,Predicted`，顺序与测试数据一致。

建模统一使用 `data_cleaned/`：已统一字符串与户型写法、删除常数列 `FURNISHED` / `FEE`、拆分月份、连接楼栋信息并修正辅助表格式。异常值、模型编码和缺失填充由各模型负责。请用 loader 保留邮编前导 0：

```python
from src.data_cleaning import load_train, load_test, load

train = load_train()
test = load_test()
schools = load("auxiliary/schools.csv")
```

辅助数据包括楼栋、MRT、学校、购物中心、COE 和股价。楼栋按小写的 `(BLOCK, STREET)` 与 train / test 100% 匹配。MRT 区分 open / planned；距离或数量特征需注意换乘站重复和开通时间未知。COE 字符串数值已解析；股价为日频，宏观特征需按月聚合。具体文件、字段及口径以 [数据说明](data_cleaned/README.md) 为准。

训练覆盖 2021-01 至 2025-03，测试覆盖 2025-03 至 2026-07，因此共用时间验证方案：主 holdout 用 ≤2023-11 训练，2023-12 至 2025-03 验证；rolling 再加入截止 2024-03 的训练折。租金在 2021–2023 年大涨，2024 年后趋稳（年均约 2121 → 3036 → 3110）；树模型无法直接外推时间，时间趋势需要单独考虑。

数据是面板 / 横截面数据，没有房屋 ID。可探索伪单位键 `BLOCK + STREET + FLAT_TYPE + FLAT_MODEL + FLOOR_AREA_SQM`（约 14.8k 个单位，每单位记录数中位数为 7），但分组统计或固定效应须声明这一假设。同一伪单位、同一月份的租金标准差中位数约 283，RMSE 的合理预期在 300 以上。

## 团队分工

| 组员 | 负责范围 |
|---|---|
| P1 | 数据与公共框架、目标编码、融合与提交 |
| P2 | 统计模型：基线、Ridge、双向固定效应、混合效应；时间趋势处理 |
| P3 | GBDT：LightGBM、XGBoost、CatBoost、RF；调参与 SHAP |
| P4 | 辅助数据、距离特征、KNN、MLP + embedding |

个人实验命名为 `notebooks/<姓名>_<主题>.ipynb`。共享实现放在 `src/`，各模型共用时间切分和实验日志，使结果可比；特征按模块开关，便于消融。新增模型库由相应负责人维护依赖。

## 特征模块

| 模块 | 注册名 | 当前状态与内容 |
|---|---|---|
| F1 基础 | `basic` | 月份索引、房龄、剩余租约、户型有序编码 |
| F2 楼栋 | `block` | 清洗后已连接的楼栋列，直接使用 |
| F3 距离 | `distance` | 预留，待实现 |
| F4 目标编码 | `target_enc` | 预留，待实现；必须防泄漏 |
| F5 宏观 | `macro` | 预留，待实现 |
| F6 空间邻居 | `spatial` | 预留，待实现 |

`features=[]` 保留原始房屋属性；启用 `block` 才加入楼栋原始列。`basic` 的 `MONTH_INDEX` 从 2021-01 的 0 起算；`FLAT_AGE` 为批准年份减建成年份；`REMAINING_LEASE` 为 99 减已使用租约年数；`FLAT_TYPE_ORDINAL` 为 1–6。因此 basic 的房龄也依赖楼栋建成年份，关闭 block 仅移除楼栋直接列。

新特征块继承 `FeatureBlock`，实现 `fit(train_df)` / `transform(df)`，在实验入口导入并注册后即可通过名字启用。例如组员实现 `src/features/my_distance.py` 后：

```python
from src.features import register_feature
from src.features.my_distance import DistanceFeatures

register_feature("distance", DistanceFeatures)
```

注册表接收类或零参数工厂。预留名称未注册时会明确报错。训练 OOF 特征可通过 `fit_transform(train_df)` 钩子接入；完整防泄漏与返回值约定见 [AGENTS.md](AGENTS.md)。

## 模型接入与实验

把自己的 sklearn estimator / Pipeline 放入 `ModelWrapper`，声明列名并在模型内部完成预处理。可选目标变换为 `none` / `log` / `per_sqm`；包装器在预测时还原租金后评分。

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.models import ModelWrapper
from src.experiment import run_experiment

model = ModelWrapper(make_pipeline(SimpleImputer(), StandardScaler(), Ridge()),
                     columns=["FLOOR_AREA_SQM", "MONTH_INDEX"])
result = run_experiment("ridge", "P2", ["basic"], model,
                        split="holdout", final=True)
print(result.mean_rmse, result.submission_path)
```

每折独立创建特征块、clone 模型，统一使用 `src/config.py` 中的种子。`final=True` 在验证后以全量 train 重拟合并写出 `submissions/<run_id>.csv`；只评估时保持默认的 `final=False`。

日志记录时间、运行 ID、名称、作者、特征、模型参数及各折 / 平均 RMSE。每次运行保存配置和验证预测；预测包含原始 train 位置行号 `row_index`、`fold`、月份、真实租金和 `Predicted`。stacking 按 `(fold, row_index)` 对齐，避免混用 rolling 重叠折。日志以本地锁保护并发追加；配置与日志可提交到 Git，验证预测和提交 CSV 默认忽略。
