# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, Cursor, Copilot, etc.) and human collaborators when working with code in this repository.

## 项目概述

CS5228 小组项目（Kaggle 私有比赛 "CS5228-2610 Project"）：根据新加坡 HDB 组屋属性预测月租金 `MONTHLY_RENT`。
- 任务类型：回归；评估指标：**RMSE**（原始租金尺度）。
- 任务说明原文：`CS5228-2610 Project _ Kaggle.mhtml`。
- 课程要求对使用 / 不使用的每类信息（尤其辅助数据）**给出论证**，报告与分数同等重要。

## 数据

- `data/train.csv`：150,000 行，含目标 `MONTHLY_RENT`。
- `data/test.csv`：50,000 行，无 Id 列；**提交的 `Id` 即测试集行号 0..49999**。
- `data/example-submission.csv`：提交格式 `Id,Predicted`。
- `data/auxiliary/`：
  - `sg-hdb-block.csv`：楼栋经纬度、`MAX_FLOOR`、`YEAR_COMPLETED`、`SUBZONE`、`PLANNING_AREA`、`REGION`。按 (BLOCK, STREET) 小写后与 train/test **100% 匹配**。
  - `sg-mrt-stations.csv`：含 `STATUS`（open / planned），计算距离时注意区分。
  - `sg-schools.csv`、`sg-shopping-malls.csv`：POI 经纬度。
  - `sg-coe-prices.csv`：数值列是带 `$` 和千分位逗号的字符串，需先解析。
  - `sg-stock-prices.csv`：日频股价，需聚合到月。

### 清洗后的数据（建模请用这一份）

- `data_cleaned/` 由 `src/data_cleaning/` 生成，只做了与模型无关的基础清洗：统一格式、删除常数列 `FURNISHED` 和 `FEE`、统一 `FLAT_TYPE` 写法、join 楼栋表、修正辅助表的格式。
- 处理细节、已知的数据现象、每个文件的数据字典以及 `NUM_COLS` / `CAT_COLS`，见 `data_cleaned/README.md`。
- 读取必须使用 `src.data_cleaning` 中的 `load_train()`、`load_test()`、`load("auxiliary/xxx.csv")`。直接 `pd.read_csv` 会丢失 `POSTAL_CODE` 的前导 0。
- 异常值、编码、填充缺失值、特征工程都属于模型相关处理，**不要放进 `src/data_cleaning/`**，由各模型负责人在自己的代码里处理。

```bash
python -m src.data_cleaning.clean    # 重新生成 data_cleaned/（在项目根目录运行）
python -m src.data_cleaning.checks   # 校验输出；修改清洗逻辑后必须跑通
```

## 关键建模约束

- **train/test 按时间切分**：train 覆盖 2021-01 至 2025-03，test 覆盖 2025-03 至 2026-07。这是教授的疏忽，但不会更正。
  - 本地验证必须按时间切分，不要用随机 K-fold。约定主验证为 train ≤ 2023-11，valid 为 2023-12 至 2025-03（16 个月，与测试跨度一致）。
  - 租金 2021 到 2023 年大涨，2024 年后趋稳（年均约 2121 → 3036 → 3110）。树模型无法外推时间。
- **这不是纯时间序列数据**（教授在公告中明确说明）。数据是面板 / 横截面数据，且**没有房屋 ID**。
  - 不要对单套房做 ARIMA、LSTM，也不要构造"同一套房上期租金"之类的滞后特征。
  - 只有聚合后的市场月度指数可以当作时间序列处理。
  - 允许构造"伪单位"键：`BLOCK + STREET + FLAT_TYPE + FLAT_MODEL + FLOOR_AREA_SQM`，共约 14.8k 个单位，中位数每个单位 7 条记录。用作分组统计或固定效应时，必须在报告中声明这是一个假设。
- **噪声下限**：同一伪单位、同一月份的记录之间，租金标准差中位数约 283。RMSE 的合理预期在 300 以上。
- **目标编码 / 分组统计必须防泄漏**：验证时只能用验证切分点之前的数据或 out-of-fold 统计；最终提交时才用全量 train。

## 团队约定（数据清洗已完成，其余为规划）

四人分工：
- P1：数据与公共框架、目标编码、融合与提交。
- P2：统计模型（基线、Ridge、双向固定效应、混合效应）与时间趋势处理。
- P3：GBDT（LightGBM、XGBoost、CatBoost、RF）、调参、SHAP。
- P4：辅助数据、距离特征、KNN、神经网络（MLP + embedding）。

代码组织：
- 公共模块放在 `src/`：`data_cleaning/`、`features/`、`cv.py`、`models/`、`experiment.py`。
- 个人实验放在 `notebooks/<姓名>_<主题>.ipynb`。
- 提交文件放在 `submissions/`。

所有实验通过统一接口、在同一验证切分上运行，结果记入共享实验日志，以便方法之间可比。特征按模块（F1 基础 / F2 楼栋 / F3 距离 / F4 目标编码 / F5 宏观 / F6 空间邻居）组织，可单独开关，用于消融实验。

新增公共模块后，请在此补充对应的运行命令与依赖说明。

## 公共实验框架

依赖：Python ≥ 3.10，NumPy、pandas、scikit-learn ≥ 1.3、threadpoolctl（见 `requirements.txt`）；测试用标准库 unittest。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.run_baseline
```

基线入口运行分组中位数和 HistGradientBoosting 的 holdout / rolling，打印各折及平均 RMSE，追加 `experiments/log.csv`，并用全量 train 训练后者生成一个提交。分组中位数只使用训练折最后 6 个日历月；未见组合回退到同期户型中位数，再回退到同期全局中位数。HistGradientBoosting 在模型内部编码类别，保留缺失值，关闭随机提前停止验证；入口将线程数限制为 4。

- 切分仅在 `src/cv.py` 的 `SPLITS` 定义；`get_splits(df, split)` 返回各折的 `train_idx` / `valid_idx` **位置索引**，用 `.iloc`。rolling 包含 holdout 及截止 2024-03 的第二折；验证区间重叠，平均 RMSE 为各折简单平均。
- `features=[]` 保留原始房屋属性；`block` 才加入楼栋原始列。`basic` 添加 `MONTH_INDEX`（2021-01 为 0）、`FLAT_AGE`（批准年份 − 建成年份）、`REMAINING_LEASE`（99 − 已使用租约年数）、`FLAT_TYPE_ORDINAL`（1–6）。因此 basic 的房龄也使用楼栋建成年份；关闭 block 仅移除楼栋直接列。
- 每折重新创建特征块并 clone 模型；默认种子在 `src/config.py`，模型的各级 `random_state` 统一设为该值。`ModelWrapper.predict()` 自动还原目标变换，指标和提交始终为原始租金尺度。

新增特征块：在 `src/features/` 下继承 `FeatureBlock`，实现 `fit(train_df)` / `transform(df)`，再在实验入口导入并注册：

```python
from src.features import register_feature
from src.features.my_distance import DistanceFeatures  # 组员实现
register_feature("distance", DistanceFeatures)
```

注册表接收类或零参数工厂，允许为预留的 `distance` / `target_enc` / `macro` / `spatial` 注册实现。`fit` 可读取当前训练折的目标；`transform` 收不到目标或 Id，必须返回新增列的 DataFrame，保持索引和行顺序。块之间独立，不读取其他块的输出，也不要自行加载全量 train。若训练目标编码需要 OOF，覆盖 `fit_transform(train_df)` 返回 OOF 特征，同时保存当前整折的统计供后续 `transform` 使用；框架会调用这个钩子。

接入模型：把自己的 sklearn estimator / Pipeline 放进 `ModelWrapper`，明确声明列名；编码、缩放、填充全部在模型内处理。可选 `target_transform="none" / "log" / "per_sqm"`，其中 log 为自然对数，per_sqm 为租金除以面积，要求面积为正。

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.models import ModelWrapper
from src.experiment import run_experiment

model = ModelWrapper(make_pipeline(SimpleImputer(), StandardScaler(), Ridge()),
                     columns=["FLOOR_AREA_SQM", "MONTH_INDEX"])
result = run_experiment("ridge", "P2", ["basic"], model, split="holdout", final=True)
print(result.mean_rmse, result.submission_path)
```

每次运行保存 `experiments/<run_id>/config.json` 与 `validation_predictions.csv`；后者包含原始 train 位置行号 `row_index`、`fold`、月份、真实租金和 `Predicted`，用于 stacking 时按 `(fold, row_index)` 对齐，避免把重叠折混在一起。`final=True` 在验证后以全量 train 重拟合，再按 loader 的测试 Id 顺序写出 50000 行 `submissions/<run_id>.csv`。预测和提交 CSV 默认被 Git 忽略，配置和共享日志可提交；日志用本地锁保护并发追加。
