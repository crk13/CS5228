# 公共实验框架：开发者指南与接口规范

本文写给**要接入新特征或新模型、用公共流水线训练、验证和生成提交的组员**。前半部分讲怎么用（第 1–6 节），后半部分是接口规范速查（第 7 节）。

本文也是全仓库特征、模型、时间验证、实验运行及产物格式的**唯一协议信源**。README、AGENTS 只引用本文，不复制规则。接口变化时，同步修改本文、实现和 [契约测试](../tests/test_framework.py)。数据字段与清洗细节见 [清洗数据说明](../data_cleaned/README.md)，命令见 [AGENTS.md](../AGENTS.md#常用开发与测试命令)。

## 1. 一张图看懂：你写什么，框架做什么

你只需要写两样东西，再调用一个函数：

| 你写的 | 是什么 | 例子 |
|---|---|---|
| **特征块** `FeatureBlock` | 根据清洗后的数据算出若干新列 | 到 CBD 的距离、楼栋平均租金 |
| **模型** `ModelWrapper` | 一个 sklearn 风格的回归器，加上"要用哪些列" | Ridge、LightGBM、自定义 NN |
| 调用 `run_experiment(...)` | 剩下的全部交给框架 | — |

调用 `run_experiment(name, author, features=["basic", "my_feat"], model=my_model, split="holdout")` 后，框架按下面的流程执行：

```text
加载 data_cleaned/train.csv
按时间切出若干折（holdout 1 折；rolling 2 折）
对每一折：
   ┌──────────────────────── 训练部分（含租金）────────────────────────┐
   │ ① 为每个特征块新建实例，调用 block.fit_transform(训练部分)          │
   │      默认行为 = block.fit(训练部分，含租金)                        │
   │               + block.transform(训练部分，已去掉租金)              │
   │    得到 X_train = 原始房屋列 + 各特征块的新列                      │
   │ ② 复制一份新模型，调用 model.fit(X_train, 训练部分租金)            │
   └──────────────────────────────────────────────────────────────────┘
   ┌──────────────────────── 验证部分（去掉租金）──────────────────────┐
   │ ③ 用同一批已 fit 的特征块调用 block.transform(验证部分) → X_valid │
   │ ④ model.predict(X_valid) → 在原始租金尺度上算 RMSE                │
   └──────────────────────────────────────────────────────────────────┘
写 experiments/log.csv 和 experiments/<run_id>/
若 final=True：用全部 train 从头重复 ①②，对 test 做 ③，写 submissions/<run_id>.csv
```

记住两点就够了：

- **`fit` 只会看到当前折的训练部分，`transform` 永远看不到租金。** 所以只要把"要从租金学的东西"写在 `fit` 里，就不会把验证集的答案泄漏进特征。
- **每一折、以及最终训练，都会新建特征块和模型实例。** 上一折学到的状态不会带到下一折。

## 2. 五分钟跑通

在项目根目录安装依赖并跑一次基线，确认环境正常：

```bash
python -m pip install -r requirements.txt
python -m src.run_baseline
```

然后用现成的特征和一个最简单的模型跑一次自己的实验：

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from src.experiment import run_experiment
from src.models import ModelWrapper

model = ModelWrapper(
    make_pipeline(SimpleImputer(), Ridge()),
    columns=["FLOOR_AREA_SQM", "MONTH_INDEX", "REMAINING_LEASE", "FLAT_TYPE_ORDINAL"],
)
result = run_experiment("ridge_first_try", "P2", ["basic"], model)
print(result.fold_rmse, result.mean_rmse)
```

运行时会打印每折 RMSE。结果自动追加到 [experiments/log.csv](../experiments/log.csv)，配置和验证预测写在 `result.run_dir` 下。

**在 notebook 中使用**：`src` 需要能被导入。如果 notebook 放在 `notebooks/` 下，在第一个单元格执行：

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # 指向项目根目录
```

框架内部用项目根目录的绝对路径读写数据和产物，所以 notebook 放在哪里都不影响结果。

## 3. 接入一个新特征

### 3.1 先看看模型能拿到哪些列

不开任何特征块（`features=[]`）时，模型也能拿到原始房屋列。每开一个块，就在后面追加这个块的输出列：

| 来源 | 列 |
|---|---|
| 始终存在（`BASE_COLUMNS`） | `FLOOR_AREA_SQM`、`LEASE_COMMENCE_DATE`、`RENT_YEAR`、`RENT_MONTH`、`RENT_APPROVAL_DATE`、`TOWN`、`BLOCK`、`STREET`、`FLAT_TYPE`、`FLAT_MODEL` |
| `basic` | `MONTH_INDEX`、`FLAT_AGE`、`REMAINING_LEASE`、`FLAT_TYPE_ORDINAL` |
| `block` | `POSTAL_CODE`、`LATITUDE`、`LONGITUDE`、`MAX_FLOOR`、`YEAR_COMPLETED`、`SUBZONE`、`PLANNING_AREA`、`REGION` |
| 你的特征块 | 你的 `transform` 返回的列 |

注意区分两件事：**特征块的输入**始终是完整的清洗数据表，包含经纬度等楼栋列，不管有没有开启 `block`；但**模型能用的列**只有上表中已开启的部分。比如模型要用 `LATITUDE`，就必须开启 `block`。

想直接看特征长什么样，可以在框架外单独跑特征组合：

```python
from src.data_cleaning import load_train
from src.features import FeaturePipeline

X = FeaturePipeline(["basic", "block"]).fit_transform(load_train())
X.head()
X.dtypes
```

### 3.2 不依赖租金的特征：只写 `transform`

距离、计数、日期换算这类特征，计算时用不到租金，没有什么要"学习"的，所以只需写 `transform`。下面这个例子计算每套房到 CBD 的球面距离：

```python
import numpy as np
import pandas as pd

from src.features import FeatureBlock, register_feature

CBD = (1.2840, 103.8514)  # Raffles Place MRT


@register_feature("cbd_distance")
class CbdDistance(FeatureBlock):
    def transform(self, df):
        lat, lon = np.radians(df["LATITUDE"]), np.radians(df["LONGITUDE"])
        lat0, lon0 = np.radians(CBD)
        a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
        return pd.DataFrame({"DIST_CBD_KM": 6371 * 2 * np.arcsin(np.sqrt(a))}, index=df.index)
```

写 `transform` 时只需遵守三条：

- **只返回新增的列**。不要把输入表原样返回，否则会与原始列重名并报错。
- 返回的表**必须用 `index=df.index`**，并保持输入的行顺序。
- 新列名不能和原始列、其他块的列重名。建议统一用大写，并加上能体现来源的前缀。

辅助数据在 `transform` 或 `__init__` 中用 loader 读取，例如 `load("auxiliary/mrt_stations.csv")`。读取辅助表是允许的，但**不允许在块里自己读取 train 的租金**。

### 3.3 依赖租金的特征：在 `fit` 里学习，在 `transform` 里使用

目标编码、分组平均租金这类特征，需要从租金里学统计量。做法是：在 `fit` 中算好统计量，保存成以下划线结尾的属性；然后在 `transform` 中只读取这些属性。

```python
from src.data_cleaning import TARGET


@register_feature("town_rent")
class TownRent(FeatureBlock):
    def __init__(self, n_splits=5):
        self.n_splits = n_splits  # 构造参数必须原样保存为同名属性，框架会把它记入实验配置

    def fit(self, train_df):  # train_df 是当前折的训练部分，含 MONTHLY_RENT
        rent_sqm = train_df[TARGET] / train_df["FLOOR_AREA_SQM"]
        self.means_ = rent_sqm.groupby(train_df["TOWN"]).mean()
        self.global_ = float(rent_sqm.mean())
        return self

    def transform(self, df):  # df 不含租金；只用 fit 中学到的状态
        value = df["TOWN"].map(self.means_).astype(float).fillna(self.global_)
        return pd.DataFrame({"TOWN_RENT_SQM": value}, index=df.index)
```

这样写对**验证集和测试集**是安全的：它们拿到的统计量只来自训练部分。

### 3.4 训练行自身的泄漏：重写 `fit_transform` 做 OOF

上面的写法还剩一个问题：给**训练行自己**编码时，每一行的统计量里包含了它自己的租金，模型等于提前看到了答案。高基数分组（如楼栋）尤其严重，训练分数会很好看，验证分数却变差。

解决方法是 OOF（out-of-fold）：把训练部分再分成 K 份，每一份的编码只用另外 K−1 份来算。验证集和测试集仍然用整个训练部分的统计量。

在框架里的写法是：重写 `fit_transform`，返回 OOF 编码；最后再用整个训练部分调用一次 `fit`，供之后的 `transform` 使用。在上面的 `TownRent` 类里加上：

```python
# 文件顶部补充导入
from sklearn.model_selection import KFold
from src.config import SEED

# TownRent 类中新增方法
    def fit_transform(self, train_df):
        oof = pd.Series(np.nan, index=train_df.index)
        for fit_pos, enc_pos in KFold(self.n_splits, shuffle=True, random_state=SEED).split(train_df):
            part = TownRent().fit(train_df.iloc[fit_pos])
            oof.iloc[enc_pos] = part.transform(train_df.iloc[enc_pos])["TOWN_RENT_SQM"].to_numpy()
        self.fit(train_df)  # 保存整折统计量，供验证集 / 测试集的 transform 使用
        return pd.DataFrame({"TOWN_RENT_SQM": oof}, index=train_df.index)
```

框架默认的 `fit_transform` 不会自动做 OOF，需要你自己重写。用于切分的随机种子请使用 `src.config.SEED`。内层切分是否也要按时间来，由负责人自己论证决定。

### 3.5 放在哪里、怎么注册

- 正式的特征块放在 `src/features/<名字>.py`，用 `@register_feature("名字")` 注册。
  - 注册发生在模块被导入时，所以**实验脚本需要先导入这个模块**（例如 `import src.features.distance`），再调用 `run_experiment`。
  - `distance`、`target_enc`、`macro`、`spatial` 这几个名字已经为对应负责人预留。未实现前使用会报 `NotImplementedError`。
- 在 notebook 里试验时，可以直接在单元格中定义和注册。但**重复运行同一个单元格会报 `already registered`**：注册表在一个 Python 进程里只能登记一次同名特征。这时重启 kernel 即可，或者把块挪到 `.py` 文件中导入。
- 注册时传入的是类，或者一个无参数、每次都返回新实例的函数。框架会在每一折调用它，新建实例。

## 4. 接入一个新模型

### 4.1 用 `ModelWrapper` 包装

```python
ModelWrapper(estimator, columns=[...], target_transform="none")
```

| 参数 | 你要提供什么 |
|---|---|
| `estimator` | 一个**未训练**的 sklearn 回归器或 Pipeline。类别编码、标准化、缺失值填充都放在这里，**框架不替你做任何预处理** |
| `columns` | 从第 3.1 节的列里，挑出这个模型要用的列 |
| `target_transform` | 模型实际学习的目标形式：`none`（原始租金）、`log`、`per_sqm`（每平米租金）。预测时包装器会自动换算回原始租金，不需要你还原 |

一个完整的例子：线性模型，数值列先填缺失值再标准化，类别列做 one-hot 编码，学习 log 租金。

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models import ModelWrapper

num = ["FLOOR_AREA_SQM", "MONTH_INDEX", "REMAINING_LEASE", "FLAT_TYPE_ORDINAL", "DIST_CBD_KM", "TOWN_RENT_SQM"]
cat = ["TOWN", "FLAT_MODEL"]
ridge = ModelWrapper(
    make_pipeline(
        ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]),
        Ridge(alpha=1.0),
    ),
    columns=num + cat,
    target_transform="log",
)
```

处理数据时要注意两点：

- **缺失值**：训练集有 159 行的 `MAX_FLOOR`、`YEAR_COMPLETED` 是缺失的，`basic` 里由它算出的 `FLAT_AGE` 也会缺失。不能处理 NaN 的模型要自己填充。
- **未见类别**：验证集或测试集里可能出现训练部分没有的类别（尤其是 `BLOCK`、`STREET`），编码器要能处理这种情况，比如设置 `handle_unknown="ignore"`。

### 4.2 GBDT 类模型

参照 [src/models/baselines.py](../src/models/baselines.py) 中的 `hist_gradient_boosting_model()`：数值列直接传入，类别列先做序号编码，再接回归器。换成 LightGBM、XGBoost、CatBoost 时，把最后的回归器替换掉，并把新依赖加入 [requirements.txt](../requirements.txt)。

### 4.3 自定义模型（神经网络、固定效应等）

写一个 sklearn 风格的回归器类，再用 `ModelWrapper` 包起来。下面这个最小例子按 town × 户型取每平米租金的中位数：

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class GroupMedianRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, keys=("TOWN", "FLAT_TYPE")):
        self.keys = keys  # __init__ 里只保存参数，不做任何计算

    def fit(self, X, y):  # X 只含 columns 中声明的列；y 已经过 target_transform
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.medians_ = y.groupby([X[k] for k in self.keys]).median()
        self.global_ = float(y.median())
        return self

    def predict(self, X):  # 返回变换后尺度的预测，由包装器负责还原
        keys = pd.MultiIndex.from_frame(X[list(self.keys)])
        return self.medians_.reindex(keys).fillna(self.global_).to_numpy()


model = ModelWrapper(GroupMedianRegressor(), columns=["TOWN", "FLAT_TYPE"], target_transform="per_sqm")
```

自定义模型能被框架复制（`clone`），需要满足以下要求：

- `__init__` 的参数都要原样保存为同名属性。训练得到的状态保存为以下划线结尾的属性，比如 `medians_`。
- 继承 `BaseEstimator`，这样就自动有 `get_params` 和 `set_params`。
- `fit` 返回 `self`；`predict` 返回形状为 `(len(X),)`、不含 NaN 的数组。
- 神经网络的训练轮数、学习率等超参数也要作为 `__init__` 参数，这样会被记入实验配置。

这个例子用 `per_sqm` 时，`FLOOR_AREA_SQM` 并不在 `columns` 中。这样是可以的：包装器会从完整的特征表里读取面积来做换算。

### 4.4 随机种子

模型参数里所有名为 `random_state` 的参数，都会被框架统一改为 `src.config.SEED`，覆盖你设置的值。这是为了让不同人的结果可以复现、可以比较。如果模型的随机性来自别的参数（例如 PyTorch 的种子），需要在 `fit` 里用 `self.random_state` 自己设置，并把 `random_state` 声明为构造参数。

## 5. 运行实验与生成提交

```python
result = run_experiment(
    "ridge_log_cbd",                          # 实验名，写进日志
    "P2",                                     # 作者
    ["basic", "cbd_distance", "town_rent"],   # 要开启的特征块
    ridge,                                    # 未训练的模型
    split="holdout",                          # holdout 或 rolling
    final=False,                              # True 时额外用全量 train 训练并生成提交
)
```

**选哪种切分**

| `split` | 折 | 建议用途 |
|---|---|---|
| `holdout` | 训练 2021-01 至 2023-11，验证 2023-12 至 2025-03 | 日常迭代，速度快 |
| `rolling` | 上面这一折，再加一折：训练 2021-01 至 2024-03，验证 2024-04 至 2025-03 | 定稿比较、写报告、最终提交前，结论更稳 |

`mean_rmse` 是各折 RMSE 的简单平均。比较不同实验时，要用相同的 `split`。

**生成提交**：把 `final` 设为 `True` 即可。框架先完成验证，再用全部 train 从头训练，预测 test，写出 `submissions/<run_id>.csv`，可以直接上传 Kaggle。最终训练与 `split` 无关，始终使用全部 train。

**试跑不想写进共享日志**：传入 `output_dir=tempfile.mkdtemp()`，所有产物会写到临时目录。

**返回值** `result`：

| 字段 | 内容 |
|---|---|
| `result.fold_rmse` | 每折 RMSE，如 `{"holdout": 497.9}` |
| `result.mean_rmse` | 各折平均 |
| `result.run_dir` | 本次运行的目录，里面有配置和验证预测 |
| `result.submission_path` | 提交文件路径；`final=False` 时为 `None` |
| `result.run_id` | 运行 ID，用来找对应的目录和提交文件。不要依赖它的具体格式 |

## 6. 使用实验结果

每次运行会生成这些产物：

```text
experiments/
├── log.csv                         # 所有人所有实验，一次运行一行   → 提交进 Git
└── <run_id>/
    ├── config.json                 # 完整配置、每折行数和列名、依赖版本 → 提交进 Git
    └── validation_predictions.csv  # 验证集预测，用于融合             → 不进 Git
submissions/<run_id>.csv            # 仅 final=True 时生成              → 不进 Git
```

- **比较实验**：直接打开 `log.csv`，按 `split` 筛选后比较 `mean_rmse`。多人在不同分支上都追加了日志、合并时出现冲突，**保留双方的所有行**即可。
- **融合（stacking）**：读取各模型的 `validation_predictions.csv`，按 `(fold, row_index)` 对齐。`row_index` 是该行在 train 中的位置行号。不要把 rolling 两折的预测混在一起，它们的验证区间是重叠的。

  ```python
  a = pd.read_csv("experiments/<run_a>/validation_predictions.csv")
  b = pd.read_csv("experiments/<run_b>/validation_predictions.csv")
  m = a.merge(b[["fold", "row_index", "Predicted"]], on=["fold", "row_index"], suffixes=("_a", "_b"))
  m = m[m["fold"] == "holdout"]
  ```

  预测文件不进 Git。要融合别人的模型，用仓库中的代码在本地重跑他们的实验即可。

**常见报错**

| 报错 | 原因 | 解决 |
|---|---|---|
| `Feature 'x' is already registered` | notebook 中重复运行了注册单元格，或两个模块用了同一个名字 | 重启 kernel，或换一个名字 |
| `Unknown feature 'x'` | 名字拼错，或没有先导入定义该特征的模块 | 检查名字，在实验前先 `import` 该模块 |
| `NotImplementedError: ... reserved` | 用了尚未实现的预留特征 | 等负责人实现，或先去掉它 |
| `must preserve the input index and row order` | `transform` 返回的表没有用 `index=df.index`，或者重新排了序 | 构造 DataFrame 时传入 `index=df.index` |
| `returned duplicate columns` | 返回了原始列，或与其他块的列重名 | 只返回新列，改个名字 |
| `Feature columns changed between fit and transform` | 训练和验证时返回的列不一致，例如 one-hot 编码出了不同的类别 | 在 `fit` 中固定列集合 |
| `KeyError: ... not in [columns]` | `columns` 里的列不存在，常见原因是用了 `LATITUDE` 却没开启 `block` | 开启对应的特征块，或修改 `columns` |
| `Input X contains NaN` | 模型不支持缺失值 | 在 Pipeline 中加入 `SimpleImputer` |
| `could not convert string to float` | 把类别列直接传给了数值模型 | 在 Pipeline 中加入编码器 |
| `predict must return one finite original-scale rent per row` | 预测中有 NaN 或无穷大，或者形状不对 | 检查未见类别的兜底逻辑；`log` 变换时检查是否溢出 |
| `Experiment log is locked` | 上次运行被强行中断，留下了锁文件 | 确认没有其他实验在跑后，删除 `experiments/log.lock` |

## 7. 接口规范（速查）

### 数据与建模边界

- 建模使用 `data_cleaned/`，通过 `src.data_cleaning` 的 loader 读取。不要直接读取建模 CSV，以免邮编前导 0 丢失。
- `src/data_cleaning/` 只负责与模型无关的基础清洗。特征工程放在 `src/features/`。异常值处理、编码、缩放、缺失值填充和目标变换由模型负责，框架不强制统一预处理。
- 目标为 `MONTHLY_RENT`，评分为原始月租金尺度的 RMSE（SGD）。训练标签与特征行按位置对应。
- 数据没有房屋 ID。禁止构造单套房的滞后租金，禁止对单套房使用 ARIMA / LSTM。只有聚合后的市场月度指数可以按时间序列处理。
- 若使用 `BLOCK + STREET + FLAT_TYPE + FLAT_MODEL + FLOOR_AREA_SQM` 作为伪单位，用于分组或固定效应，必须在报告中声明这一假设。
- 使用辅助数据前，先查阅数据文档中的时间可用性、缺失值及口径差异。报告须论证每类信息为什么使用或不使用。

### 时间验证（[src/cv.py](../src/cv.py)）

```text
get_splits(df: pd.DataFrame, split: str = "holdout") -> list[Fold]
```

| 方案 / 折名 | 训练区间 | 验证区间 |
|---|---|---|
| holdout / `holdout` | 2021-01 至 2023-11 | 2023-12 至 2025-03 |
| rolling / `holdout` | 2021-01 至 2023-11 | 2023-12 至 2025-03 |
| rolling / `rolling_2024_04` | 2021-01 至 2024-03 | 2024-04 至 2025-03 |

- 所有实验共用 `SPLITS`，禁止随机 K-fold。
- 输入必须包含非缺失的 `RENT_APPROVAL_DATE`（格式 `YYYY-MM`）。未知方案、非法日期、训练或验证折为空，都会报错。
- `Fold` 包含 `definition`、`train_idx`、`valid_idx`。后两者是 NumPy **位置索引**，须配合 `.iloc` 使用。
- 每折的训练时间严格早于验证时间。
- rolling 的平均 RMSE 是各折 RMSE 的简单平均，不是把所有预测合并后再算 RMSE。

### 特征块（[base.py](../src/features/base.py)、[pipeline.py](../src/features/pipeline.py)、[registry.py](../src/features/registry.py)）

| 方法 | 何时被调用 | 输入 | 必须返回 |
|---|---|---|---|
| `fit(train_df)` | 默认 `fit_transform` 内部调用 | 当前折训练部分的清洗数据，**含 `TARGET`** | `self` |
| `transform(df)` | 训练部分、验证部分、测试集 | 清洗数据，框架已去掉 `TARGET` 和 `Id` | 只含新增列的 DataFrame |
| `fit_transform(train_df)` | 每折（及最终训练）对训练部分调用一次 | 同 `fit` | 与训练行对齐的新增列 DataFrame |

- 返回的 DataFrame 必须**严格保持输入的 index 和行顺序**，列名唯一。不能与原始保留列或其他块的输出重名，不能包含目标列或 `Id`。训练与之后各次 `transform` 的列名及顺序必须一致。
- 块之间相互独立：输入都是原始清洗数据，不读取其他块的输出。
- `fit` 只能使用当前训练折。不得在块内自行加载全量 train，不得读取验证集或测试集的标签来计算统计量。
- 需要 OOF 编码时重写 `fit_transform`：返回训练行的 OOF 特征，并保留整折统计量供 `transform` 使用。默认实现不做 OOF。
- 构造参数遵循 sklearn 约定（原样保存为同名属性），会通过 `get_params` 记入 `config.json`。
- `FeaturePipeline(names)` 按注册名组合特征，输出 = `BASE_COLUMNS`（原始房屋属性，不含目标、`Id` 和楼栋直接列）+ 各块输出。一次实验中特征名不能重复。`features=[]` 时只保留基础房屋列。关闭 `block` 只是去掉楼栋直接列，`basic` 中的房龄仍然使用楼栋建成年份计算。

```text
register_feature(name: str, factory=None)
```

- `factory` 可以是特征类，也可以是零参数工厂函数，必须每次返回新的 `FeatureBlock`。既可以直接调用注册，也可以当装饰器使用。
- 名称重复注册、名称未知都会报错。预留名称未注册时抛出 `NotImplementedError`。

| 模块 | 注册名 | 状态与输出 |
|---|---|---|
| F1 基础 | `basic` | 已实现：`MONTH_INDEX`、`FLAT_AGE`、`REMAINING_LEASE`、`FLAT_TYPE_ORDINAL` |
| F2 楼栋 | `block` | 已实现：`POSTAL_CODE`、`LATITUDE`、`LONGITUDE`、`MAX_FLOOR`、`YEAR_COMPLETED`、`SUBZONE`、`PLANNING_AREA`、`REGION` |
| F3 距离 | `distance` | 预留 |
| F4 目标编码 | `target_enc` | 预留 |
| F5 宏观 | `macro` | 预留 |
| F6 空间邻居 | `spatial` | 预留 |

`basic` 各列的定义：

- `MONTH_INDEX`：从 2021-01 记为 0 起算的月数。
- `FLAT_AGE`：批准年份 − 楼栋建成年份。
- `REMAINING_LEASE`：99 −（批准年份 − 租约起始年份）。
- `FLAT_TYPE_ORDINAL`：1-room 至 5-room 记为 1–5，executive 记为 6。

缺失值保持缺失，不做填充。

### 模型（[src/models/base.py](../src/models/base.py)）

```text
ModelWrapper(estimator, columns: list[str], target_transform: str = "none")
model.fit(X: pd.DataFrame, y)
model.predict(X: pd.DataFrame) -> np.ndarray
```

- `estimator`：可被 sklearn `clone` 的回归器或 Pipeline。拟合状态保存在实例的拟合属性中，不能通过构造参数传入已训练好的状态。
- `columns`：非空、无重复，不能包含目标或 `Id`。缺少所需列会报错。框架不会自动选列，也不会自动编码类别。
- `X`：特征组合后的 DataFrame，可能包含字符串和缺失值。
- `y`：一维、有限值，长度等于 `len(X)`，按行位置与 `X` 对应。
- `fit` 内部先 clone estimator 再拟合，传入的 estimator 本身不会被训练，返回 `self`。
- 包装器内部的 estimator 在**变换后的目标尺度**上学习和预测，包装器的 `predict` 负责还原。最终预测是形状 `(len(X),)` 的有限数组，处于原始租金尺度。
- 也可以不用包装器，直接接入自定义模型。但这时模型必须支持 `clone`、`get_params`、`set_params`，自己选列和预处理，并让 `predict` 直接返回原始尺度的租金。

| `target_transform` | 训练目标 | 预测还原 | 前提 |
|---|---|---|---|
| `none` | `y` | 不变 | 目标为有限值 |
| `log` | `log(y)` | `exp(prediction)` | 租金为正 |
| `per_sqm` | `y / FLOOR_AREA_SQM` | `prediction * FLOOR_AREA_SQM` | 训练和预测时面积均为有限正数。面积从完整 `X` 读取，不要求在 `columns` 中 |

不要在模型外再还原一次预测，也不要在变换后的尺度上评分。

### 实验运行（[src/experiment.py](../src/experiment.py)，路径与种子见 [src/config.py](../src/config.py)）

```text
run_experiment(name, author, features: list[str], model,
               split="holdout", final=False, *, output_dir=None) -> ExperimentResult
```

| 参数 | 含义 |
|---|---|
| `name` / `author` | 实验名称与作者，写入日志和配置 |
| `features` | 要开启的特征注册名列表，调用前所有块须已注册 |
| `model` | 未拟合的模型，满足上述模型接口 |
| `split` | `holdout` 或 `rolling` |
| `final` | 默认 `False`；为 `True` 时，完成验证后用全量 train 重新训练并生成提交 |
| `output_dir` | 产物根目录，默认是仓库根目录；可指定其他目录来隔离产物 |

- 每折及最终训练前，都会固定 Python 和 NumPy 的随机种子。
- 模型各级参数中名为 `random_state` 的参数，统一设为 `SEED`。特征工厂也应遵守这一种子约定。
- 评分公式为 `sqrt(mean((y_true - prediction) ** 2))`。
- `final=True` 时会重新创建特征块和模型，用全量 train 拟合后变换 test，不复用验证折的拟合状态。
- 返回的 `ExperimentResult` 包含字段：`run_id`、`fold_rmse`（折名 → RMSE）、`mean_rmse`、`run_dir`、`submission_path`。

### 产物与提交格式

- **`log.csv`**：追加成功运行的记录，列为 `timestamp,run_id,name,author,features,model,parameters,split,seed,final,fold_rmse,mean_rmse`。时间为 UTC。特征列表、模型参数和各折指标以 JSON 字符串保存。追加时使用本地文件锁，防止并发写入冲突。
- **`config.json`**：保存运行信息、特征参数、模型参数、依赖版本，每折的边界、行数、列名和指标；启用 `final` 时还包括最终训练和提交信息。
- **`validation_predictions.csv`**：列为 `row_index,fold,RENT_APPROVAL_DATE,MONTHLY_RENT,Predicted`。`row_index` 是该行在原始 train 中的位置行号。
- **提交 CSV**：列严格为 `Id,Predicted`，没有额外的索引列，共 50000 行。`Id` 来自 loader 的 `test[ID_COL]`，保持测试集行顺序 0..49999。每行预测必须是有限值。
- 预测 CSV 和提交 CSV 被 Git 忽略。配置和实验日志可以提交。框架不保存训练好的模型对象。
