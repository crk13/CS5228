# AI 开发约定

项目背景、团队分工、目录结构和使用示例见 [README.md](README.md)。动手前阅读本文和 [清洗数据说明](data_cleaned/README.md)。

## 数据与建模边界

- 建模使用 `data_cleaned/`，统一通过以下接口读取，避免邮编前导 0 丢失：

  ```python
  from src.data_cleaning import load_train, load_test, load, NUM_COLS, CAT_COLS, TARGET, ID_COL
  ```

- `src/data_cleaning/` 只负责与模型无关的基础清洗。异常值处理、编码、缩放、缺失填充和目标变换由各模型负责；特征工程放在 `src/features/`。
- 指标为原始月租金尺度的 **RMSE**；模型预测必须先还原目标变换，再评分或提交。
- 数据没有房屋 ID，不能构造单套房滞后租金或使用单套房 ARIMA / LSTM。只有聚合市场月度指数可按时间序列处理。使用 `BLOCK + STREET + FLAT_TYPE + FLAT_MODEL + FLOOR_AREA_SQM` 作为伪单位时，须在报告中声明该假设。
- 辅助数据的使用或不使用须在报告中论证；开工前检查数据说明中的时间可用性、缺失值和口径差异。

## 统一验证与接口

### 时间切分

训练数据覆盖 2021-01 至 2025-03，测试数据覆盖 2025-03 至 2026-07。必须共用 [src/cv.py](src/cv.py) 的 `SPLITS`，禁止随机 K-fold。

| 方案 / 折 | 训练区间 | 验证区间 |
|---|---|---|
| holdout；也是 rolling 第一折 | 2021-01 至 2023-11 | 2023-12 至 2025-03 |
| rolling 第二折 | 2021-01 至 2024-03 | 2024-04 至 2025-03 |

`get_splits(df, split)` 返回各折的 `train_idx` / `valid_idx` **位置索引**，必须用 `.iloc`。rolling 验证区间重叠，平均 RMSE 为各折简单平均。

### 特征块

- 继承 `FeatureBlock`，实现 `fit(train_df)` / `transform(df)`；通过 `register_feature(name, class_or_factory)` 注册类或零参数工厂，每折创建独立实例。
- `fit` 只能使用当前训练折，可读取 `TARGET`。不要在块内自行加载全量 train，或用验证 / 测试目标计算统计。
- `transform` 收不到 `TARGET` 或 `Id`，仅返回新增列的 DataFrame，严格保持输入索引和行顺序。块之间独立，不读取其他块输出；不得返回重复列、目标列或 Id。
- 若训练目标编码需要 OOF，覆盖 `fit_transform(train_df)` 返回 OOF 特征，同时保留当前整折统计供验证 / 测试 `transform` 使用。
- 已实现 `basic` / `block`；`distance` / `target_enc` / `macro` / `spatial` 为预留名称，注册实现后才可启用。具体特征定义见 [README](README.md#特征模块)。

### 模型与实验

- 模型遵循 sklearn 的 `fit(X, y)` / `predict(X)`，支持 `clone`。用 `ModelWrapper(estimator, columns, target_transform="none")` 明确声明列名；自己的 estimator / Pipeline 负责类别编码、缩放、填充等处理。
- 目标变换可选 `none` / `log` / `per_sqm`。`log` 为自然对数，要求租金为正；`per_sqm` 要求面积为有限正数。`ModelWrapper.predict()` 还原到原始租金尺度。
- 所有实验通过 `run_experiment(name, author, features, model, split="holdout", final=False)` 运行。每折重新拟合特征和模型；种子统一在 `src/config.py`，模型各级 `random_state` 由框架设置。
- `final=True` 先完成验证，再以全量 train 重拟合、预测 test。提交为 `Id,Predicted`，共 50000 行，Id 来自 `test[ID_COL]`，按原顺序为 0..49999。
- 共享日志为 `experiments/log.csv`；配置和验证预测保存在 `experiments/<run_id>/`。stacking 按 `(fold, row_index)` 对齐，不混用重叠折。

## 测试与验证

命令均在项目根目录运行；依赖和安装方法见 [README](README.md#快速开始)。

```bash
python -m unittest discover -s tests -v  # 框架契约测试
python -m src.run_baseline              # 完整基线、日志和提交生成
```

修改框架或模型接口后运行契约测试，重点检查时间切分无重叠、验证目标不进入特征变换、目标变换还原和提交格式。基线命令用于验证完整流程，会追加实验日志并生成提交文件；仅改文档无需重新训练。

修改基础清洗逻辑后，重新生成并通过数据校验：

```bash
python -m src.data_cleaning.clean
python -m src.data_cleaning.checks
```

## 代码与文档维护

公共实现放 `src/`，个人实验放 `notebooks/<姓名>_<主题>.ipynb`，提交放 `submissions/`。保持现有代码风格，注释精炼；新增公共模块时同步更新 README 的结构、用法和依赖，接口约束或验证方法变化时同步更新本文。
