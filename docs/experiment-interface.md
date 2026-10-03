# 公共实验框架接口协议

本文是全仓库**特征、模型、时间验证、实验运行及产物格式的唯一协议信源**。README、AGENTS 和其他说明文档只链接到本文，不重复定义接口、参数、协议示例或约束。接口变化时同步修改本文、实现和相关契约测试。

字段含义、清洗细节与 loader 的用法由 [清洗数据说明](../data_cleaned/README.md) 维护，本文引用该数据文档，不另建数据字典。项目背景与团队分工见 [项目说明](project.md)，开发和测试命令见 [AGENTS.md](../AGENTS.md)。

## 数据与建模边界

- 建模使用 `data_cleaned/`，通过 `src.data_cleaning` 的 loader 读取；不要直接读取建模 CSV，以免邮编前导 0 丢失。
- `src/data_cleaning/` 只负责与模型无关的基础清洗。特征工程放在 `src/features/`；异常值处理、编码、缩放、缺失填充和目标变换由模型负责，框架不强制统一预处理。
- 目标为 `MONTHLY_RENT`，评分为原始月租金尺度的 RMSE（SGD）。训练标签与特征行按位置对应。
- 数据没有房屋 ID，禁止构造单套房滞后租金或使用单套房 ARIMA / LSTM。只有聚合市场月度指数可按时间序列处理。
- 若用 `BLOCK + STREET + FLAT_TYPE + FLAT_MODEL + FLOOR_AREA_SQM` 作伪单位分组或固定效应，必须在报告中声明这一假设。
- 使用辅助数据前检查数据文档中的时间可用性、缺失值及口径差异；报告须论证每类信息的使用或不使用。

## 时间验证接口

实现：[src/cv.py](../src/cv.py)。

```text
get_splits(df: pd.DataFrame, split: str = "holdout") -> list[Fold]
```

训练数据覆盖 2021-01 至 2025-03，测试数据覆盖 2025-03 至 2026-07。所有实验共用 `SPLITS`，禁止随机 K-fold。

| 方案 / 折名 | 训练区间 | 验证区间 |
|---|---|---|
| holdout / `holdout` | 2021-01 至 2023-11 | 2023-12 至 2025-03 |
| rolling / `holdout` | 2021-01 至 2023-11 | 2023-12 至 2025-03 |
| rolling / `rolling_2024_04` | 2021-01 至 2024-03 | 2024-04 至 2025-03 |

输入必须包含非缺失的 `RENT_APPROVAL_DATE`（`YYYY-MM`）。未知方案、非法日期或空训练 / 验证折会报错。

每个 `Fold` 包含 `definition`、`train_idx`、`valid_idx`。后两者为 NumPy **位置索引**，须使用 `.iloc`；不能把它们当成 DataFrame 的索引标签。每折训练时间严格早于验证时间。rolling 的验证区间重叠，平均 RMSE 为各折 RMSE 的简单平均，不是合并所有预测后计算的 RMSE。

## 特征接口

实现：[基类](../src/features/base.py)、[组合器](../src/features/pipeline.py)、[注册表](../src/features/registry.py)。

### 生命周期与返回值

特征块继承 `FeatureBlock`；每折创建独立实例，最终训练时也重新创建。

| 方法 | 输入与职责 | 返回值 |
|---|---|---|
| `fit(train_df)` | 当前训练折的清洗数据，可读取 `TARGET`；保存该折拟合状态 | `self` |
| `transform(df)` | 清洗数据，框架已移除 `TARGET` 和 `Id`；仅用已拟合状态计算特征 | 新增列组成的 DataFrame |
| `fit_transform(train_df)` | 训练特征钩子；默认先 fit，再移除目标与 Id 后 transform | 与训练行对齐的新增列 DataFrame |

返回值必须严格保持输入索引和行顺序，列名唯一；不能与原始保留列或其他块的输出列重名，不能包含目标列或 Id。训练与后续变换的最终列名及顺序必须一致。块之间独立，输入是原始清洗数据，不读取其他块输出。

`fit` 只能使用当前训练折，不得在块内自行加载全量 train，或读取验证 / 测试标签计算目标统计。需要训练 OOF 编码时覆盖 `fit_transform`：返回 OOF 训练特征，并保留当前整折统计供验证 / 测试 `transform` 使用。默认钩子不会自动生成 OOF 统计。

`FeaturePipeline(names)` 按注册名组合特征：实验使用其 `fit_transform(training)` 和 `transform(validation_or_test)`。基础输入列由实现中的 `BASE_COLUMNS` 定义，保留原始房屋属性，排除目标、Id 和楼栋直接列。

### 注册与可选模块

```text
register_feature(name: str, factory=None)
```

注册值为特征类或零参数工厂，必须返回新的 `FeatureBlock`。可直接注册，也可用装饰器。实验入口须先导入注册模块，再调用实验接口。重复注册名称会报错；未知名称会报错；预留模块未注册时抛出 `NotImplementedError`。一次实验中的特征名称不能重复。

| 模块 | 注册名 | 状态与输出 |
|---|---|---|
| F1 基础 | `basic` | 已实现：`MONTH_INDEX`、`FLAT_AGE`、`REMAINING_LEASE`、`FLAT_TYPE_ORDINAL` |
| F2 楼栋 | `block` | 已实现：直接暴露清洗后已连接的楼栋列 |
| F3 距离 | `distance` | 预留 |
| F4 目标编码 | `target_enc` | 预留 |
| F5 宏观 | `macro` | 预留 |
| F6 空间邻居 | `spatial` | 预留 |

`basic` 的月份索引从 2021-01 的 0 起算；房龄为批准年份减建成年份；剩余租约为 99 减已使用租约年数；户型编码依次为 1-room 至 5-room 的 1–5、executive 的 6。缺失值保持缺失。

`block` 输出 `POSTAL_CODE`、`LATITUDE`、`LONGITUDE`、`MAX_FLOOR`、`YEAR_COMPLETED`、`SUBZONE`、`PLANNING_AREA`、`REGION`。`features=[]` 保留基础房屋列；关闭 block 只移除楼栋直接列，basic 的房龄仍使用楼栋建成年份。

## 模型接口

实现：[src/models/base.py](../src/models/base.py)。

```text
ModelWrapper(estimator, columns: list[str], target_transform: str = "none")
model.fit(X: pd.DataFrame, y)
model.predict(X: pd.DataFrame) -> np.ndarray
```

- `estimator` 是可被 sklearn `clone` 的回归器或 Pipeline。构造参数遵循 sklearn 约定，拟合状态保存在实例的拟合属性中；不能依靠构造参数中的已有拟合状态接入。
- `columns` 为模型使用的非空、无重复列名列表，不能包含目标或 Id。包装器按此列表选列；缺少所需列会报错。
- `X` 是特征组合后的 DataFrame，可包含字符串、缺失值等；编码、缩放、缺失填充与未知类别处理均由模型自身负责。框架不自动为模型选择列或编码类别。
- `y` 为与 X 行位置对应的一维有限目标，长度为 `len(X)`。`fit` 返回 `self`，并在包装器内部 clone estimator 后拟合；传入的 estimator 不被直接训练。
- 包装器内的 estimator 学习并预测**变换后的目标尺度**；包装器 `predict` 负责还原。最终预测必须是形状 `(len(X),)` 的有限数值数组，处于原始租金尺度。

也可直接接入自定义 sklearn 风格模型，但需支持 `clone`、`get_params`、`set_params`，自行选列和预处理，并让 `predict` 返回原始尺度租金。

### 目标变换

| `target_transform` | 训练目标 | 预测还原 | 前提 |
|---|---|---|---|
| `none` | `y` | 不变 | 有限目标 |
| `log` | `log(y)` | `exp(prediction)` | 租金为正 |
| `per_sqm` | `y / FLOOR_AREA_SQM` | `prediction * FLOOR_AREA_SQM` | 训练及预测面积均为有限正数 |

`per_sqm` 从包装器接收到的完整 X 读取面积，即使面积未列入 estimator 的 `columns`，X 也必须包含该列。不要在模型外重复还原预测或以变换后的尺度评分。

## 实验接口

实现：[src/experiment.py](../src/experiment.py)，默认路径和种子：[src/config.py](../src/config.py)。

```text
run_experiment(name, author, features: list[str], model,
               split="holdout", final=False, *, output_dir=None) -> ExperimentResult
```

| 参数 | 含义 |
|---|---|
| `name` / `author` | 实验名称与作者，写入日志和配置 |
| `features` | 要启用的特征注册名列表；所有块在运行前完成注册 |
| `model` | 未拟合的模型配置，满足上面的模型接口 |
| `split` | `holdout` 或 `rolling` |
| `final` | 默认 False；True 时完成验证后重训全量 train 并生成提交 |
| `output_dir` | 产物根目录；默认仓库根目录，可在测试中隔离产物 |

每折流程：取当前训练 / 验证位置索引 → 独立特征实例生成训练特征 → 用同一实例变换验证特征 → clone 模型并拟合 → 原始租金尺度预测与评分。

每折及最终训练前固定 Python / NumPy 随机种子。模型各级参数中名为 `random_state` 的参数统一设为 `src/config.py` 的 `SEED`，会覆盖模型配置中的同名值。特征工厂也应遵守该种子约定。

评分公式为 `sqrt(mean((y_true - prediction) ** 2))`。`final=True` 会重新创建特征块和模型，用全量 train 拟合后变换 test；验证折的拟合状态不直接复用。

返回的 `ExperimentResult` 包含：

| 字段 | 类型 / 内容 |
|---|---|
| `run_id` | 本次运行的唯一标识；调用方不依赖其内部编码格式 |
| `fold_rmse` | 折名到 RMSE 的字典 |
| `mean_rmse` | 各折 RMSE 的简单平均 |
| `run_dir` | 本次配置及验证预测所在 Path |
| `submission_path` | 提交文件 Path；`final=False` 时为 None |

## 实验产物与提交协议

```text
<output_dir>/
├── experiments/
│   ├── log.csv
│   └── <run_id>/
│       ├── config.json
│       └── validation_predictions.csv
└── submissions/
    └── <run_id>.csv                 # 仅 final=True
```

- `log.csv` 追加成功运行记录，列为 `timestamp,run_id,name,author,features,model,parameters,split,seed,final,fold_rmse,mean_rmse`。时间为 UTC；特征列表、模型参数和各折指标以 JSON 字符串保存。共享日志使用本地文件锁保护并发追加。
- `config.json` 保存运行信息、特征参数、模型参数、依赖版本、各折边界 / 行数 / 列名 / 指标，以及最终训练和提交信息（如启用）。
- `validation_predictions.csv` 列为 `row_index,fold,RENT_APPROVAL_DATE,MONTHLY_RENT,Predicted`。`row_index` 是原始 train 的位置行号；stacking 按 `(fold, row_index)` 对齐，不能将 rolling 的重叠折混在一起。
- 提交 CSV 列严格为 `Id,Predicted`，无额外索引列，共 50000 行。Id 来自 loader 的 `test[ID_COL]`，保持测试行顺序 0..49999；每行预测必须有限。
- 预测 CSV 和提交 CSV 默认被 Git 忽略，配置和实验日志可提交。框架不保存训练后的模型对象。

## 接入示例

下面的特征定义、注册和模型配置可放在同一实验入口。自定义块只返回新增列；需要学习统计时另外实现 fit。

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FeatureBlock, register_feature
from src.models import ModelWrapper
from src.experiment import run_experiment


@register_feature("area_sqrt")
class AreaSqrtFeatures(FeatureBlock):
    def transform(self, df):
        return pd.DataFrame({"AREA_SQRT": np.sqrt(df["FLOOR_AREA_SQM"])},
                            index=df.index)


model = ModelWrapper(
    make_pipeline(SimpleImputer(), StandardScaler(), Ridge()),
    columns=["FLOOR_AREA_SQM", "MONTH_INDEX", "AREA_SQRT"],
    target_transform="none",
)
result = run_experiment("ridge", "P2", ["basic", "area_sqrt"], model,
                        split="holdout", final=False)
print(result.fold_rmse, result.mean_rmse, result.run_dir)
```

需要提交时，将同一调用的 `final` 设为 True。验证与维护命令统一见 [AGENTS.md](../AGENTS.md#常用开发与测试命令)。
