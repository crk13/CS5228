# Yuxin：Transformer / RNN / LSTM 开发方案与分支审阅

审阅日期：2026-11-04。本文记录已经读取的仓库事实、模型处理差异及后续架构设计；所有“拟新增”文件和运行命令均为计划，尚未实现。本轮只新增本文，不合并分支、不改模型代码、不训练。

## 1. 当前 Git 状态与读取范围

实际小组仓库位于 `Group Project/CS5228/`，不是外层课程资料目录。当前分支 `yuxin_time_series` 与 `main`、`origin/main` 都指向初始提交 `7def64e`；审阅开始时工作区干净，当前 `README.md` 是 **0 字节**。因此，在自己的分支看不到队友代码，并不代表他们没有实现。

已执行 `git fetch origin --prune` 更新远程引用，再用 `git show`、`git ls-tree` 读取其他分支，没有 checkout。实际分支名是 **`data-cleaning`**，不是 `data_cleaning`；后者只出现在 Python 包路径 `src/data_cleaning/` 中。

| 分支 | 审阅时提交 | 已读取内容及定位 |
|---|---|---|
| `main` / `origin/main` | `7def64e` | 初始原始数据、空 README、旧版 AGENTS、`.gitignore`、Kaggle MHTML |
| `yuxin_time_series` | `7def64e` | 当前个人分支，尚无 `src/`、清洗数据或神经网络实现 |
| `origin/data-cleaning` | `907aa00` | 基础清洗、数据字典、loader、特征与模型接口、时间验证、实验框架、契约测试及基线日志 |
| `origin/merge-data-cleaning` | `32213c4` | 公共框架整合分支；其文件树与上述 `data-cleaning` 相同，合并历史不同 |
| `origin/p2-linear` | `6e7ffcb` | FXY 的线性/Ridge 实验、市场比例模型、MRT 距离、框架交付脚本及 `results/FXY/` |
| `origin/yaqi-tree-models` | `a69bdd5` | 旧树模型报告、13 组实验、patch 和 ZIP；还读取了 ZIP 内模型源码、训练脚本、Notebook 代码及说明 |
| `origin/yaqi-results-20261103` | `3dae63c` | 最新树模型源码、距离/经济特征、30 组阶段比较、交付/核验脚本、配置与 `results/Yaqi/` |

范围说明：已遍历所有上述分支的文件清单，重点阅读源代码、说明、配置、测试和结果汇总；相同 Git blob 不重复解释。数据通过全表统计核对；正式交付的四个模型验证预测已逐行读取并重算 RMSE。没有执行压缩包里的程序、反序列化模型权重或逐张审查结果图片，也没有重新训练队友模型。

**KNN 的实际状态：**所有已获取分支的 Python/Markdown 及旧树模型包中，没有找到 KNN 回归实现或正式成绩。README 只是把 KNN 分配给 P1，旧 AGENTS 则分给 P4。`BallTree` 被用于计算设施距离，不能据此说已实现 KNN 租金预测。如果队友有未推送的 KNN 文件，本次无法读取。

## 2. 数据与公共清洗到底做了什么

### 2.1 原始数据与清洗数据

原始 `data/` 在各开发分支中共用；已核对当前分支 train/test 与清洗分支原始文件的 SHA-256 一致。

| 数据 | 原始形状 | 清洗后形状 | 时间范围 |
|---|---:|---:|---|
| train | 150000 × 11 | 150000 × 19 | 2021-01—2025-03，共 51 个月 |
| test | 50000 × 10 | 50000 × 19 | 2025-03—2026-07，共 17 个月标签 |

test 的首月与 train 最后一个月相同；相对于 2025-03，最远是 **向前预测 16 个月**，不能把 17 个年月标签误写成 17 步未来预测。2025-03 的 test 行仍没有标签，只能使用已提供训练行的历史信息。

清洗后两表各有 18 个特征；train 额外有 `MONTHLY_RENT`，test 额外有 `Id`。`Id` 为测试原行号 0—49999，不是房屋 ID。`MAX_FLOOR` 是楼栋最高楼层，也不是该出租房的实际所在楼层。

### 2.2 `src/data_cleaning/` 的实现

- `clean.py`：统一字符串大小写、去首尾及重复空格；`4 room` → `4-room`；日期拆为 `RENT_YEAR`、`RENT_MONTH`；断言并删除常数 `FURNISHED`、`FEE`。
- 以 `(BLOCK, STREET)` 左连接楼栋表，验证 many-to-one、行数不变、坐标完全匹配及 TOWN 一致；补入邮编、坐标、楼栋最高楼层、建成年份、子区、规划区、大区。
- 保留原始行序、面积、租约起始年份和目标值；只为 test 生成 `Id`。此次已逐行核对 train 标签与街道对应顺序不变。
- 楼栋表：`NIL` 邮编变缺失，`MAX_FLOOR` / `YEAR_COMPLETED` 的 `-1` 变缺失。
- MRT：提取 `LINE`。仍包含 open/planned、LRT、同站多线路记录，使用时要另选口径。
- 学校：`NATRUE_CODE` → `NATURE_CODE`，`REGION` → `ZONE`（MOE 分区，不能当 URA 大区），邮编补前导零，统一 seng kang 拼写。
- COE：解析美元符号/千分位，月份转整数，增加 `YEAR_MONTH`；**尚未按月聚合**。
- 股票：删除 85 条价格字段全空记录，增加 `YEAR_MONTH` / `MARKET` 并排序；**尚未按月聚合**，也未把多币种价格合并。
- `schema.py`：定义类型并提供 `load_train()`、`load_test()`、`load()`，确保 BLOCK 和邮编保留字符串与前导零。
- `checks.py`：检查形状、类型、坐标、连接键唯一性、日期/行序/目标不变及测试编号。

公共清洗**没有**做类别编码、标准化、填补缺失、异常值删除、目标变换或序列构造。这些由模型负责，因此无需为神经网络重新复制一套清洗脚本。

已重新统计的缺失：train 的 `POSTAL_CODE` 5259 行、`MAX_FLOOR` 159 行、`YEAR_COMPLETED` 159 行；test 只有邮编缺失 1739 行。由建成年份计算的 `FLAT_AGE` 也会缺失。原始主表无空值，不等于连接和特征工程后无缺失。

保留重复及低租金记录。不要把同月同楼栋同属性的多条租约擅自去重，也不要因为神经网络输入不方便就删除这些行。

### 2.3 公共特征与框架

`FeaturePipeline` 始终保留基本房屋属性，按开关增加特征：

| 特征块 | 当前公共实现 |
|---|---|
| `basic` | `MONTH_INDEX`、`FLAT_AGE = RENT_YEAR - YEAR_COMPLETED`、`REMAINING_LEASE`、`FLAT_TYPE_ORDINAL` |
| `block` | 已连接楼栋信息的 8 列；只有启用此块，模型才直接拿到经纬度等列 |
| `distance` / `macro` | 公共清洗分支仅预留名字；最新 Yaqi 分支才有具体实现 |
| `target_enc` / `spatial` | 已检查分支中仍未找到正式公共实现 |

每个特征块拿到完整清洗数据，块与块相互独立；模型拿到的是开启后的列集合。所有块输出必须保留输入 index/顺序，只返回新增列。

`run_experiment()` 每折重新实例化特征与模型：`fit_transform(train)` → `model.fit(X, y)` → `transform(valid)` → `predict` → 原尺度 RMSE。`final=True` 会重新用全量 train 拟合，再预测 test。框架没有自动类别编码、缩放或 PyTorch 训练循环。

`ModelWrapper` 支持原始目标、`log`、`per_sqm`；内部模型预测变换后的尺度，包装器负责还原。自定义神经网络也可直接实现 sklearn 风格的 `fit/predict` 并自行返回 SGD，从而拥有序列和目标变换的完整控制权。

## 3. 队友模型怎样实现

### 3.1 FXY：线性与 MarketRatioRidge

主要源码：`origin/p2-linear:src/models/market_ratio.py`、`src/features/mrt_distance.py`、`notebooks/FXY/p2_linear_ridge.py`、`notebooks/FXY/p2_framework_run.py`。

探索脚本依次比较：面积单变量线性回归、最近 6 个月、添加 town/户型/房龄、town 换 subzone、加楼栋、Ridge 正则化、MRT、全历史及市场水平归一。正式交付已经接入公共框架和 loader；早期探索脚本仍直接 `read_csv`，不宜作为新代码的读取范本。

正式模型步骤：

1. 构造 `BLOCK_KEY = BLOCK + " " + STREET`，不会只用 BLOCK。
2. 类别列为 `SUBZONE, BLOCK_KEY, FLAT_TYPE, FLAT_MODEL`；训练折内拟合 `OneHotEncoder(handle_unknown="ignore")`。
3. 数值列为面积、房龄、MRT 距离。这里房龄是 `RENT_YEAR - LEASE_COMMENCE_DATE`，与公共 `basic` 的建成年份口径不同。**实际源码没有 StandardScaler**，不要按“Ridge 通常要标准化”反推其已实现。
4. 在训练折最近 6 个月内，用最后 2 个月作内部验证，在 `(0.3, 1, 3, 10, 30, 100)` 中选择 alpha。
5. 全训练折按月计算全市租金中位数 `m_t`，学习 `q_i = y_i / m_t`。
6. 预测时不读取未来真实租金，使用 `q_hat_i × 最后训练月的 m_t`，即市场水平持平延续。

MRT 距离通过 haversine `BallTree` 计算，只选 `STATUS=open` 并按坐标去重，单位 km；**没有像 Yaqi 那样显式排除 LRT/RTS**，两人的 MRT 列不能视为完全同一个特征。

两处实现细节应保留在比较记录中：alpha 内部选择用的是原始租金，而最终 Ridge 学的是比例；训练目标归一化中的当月中位数包含训练行自身。后者没有读取外层验证标签，但不应把这种当月统计直接复制成无泄漏的输入特征。后续若把它用于神经网络输入，须改为过去月份或交叉拟合。

Ridge 分支曾画过全历史/最近 12 月直线、抛物线、S 曲线外推的探索图；正式交付模型采用的是 **carry**，不能把那些曲线写成最终模型组件。

### 3.2 Yaqi：树模型与特征阶段

最新源码：`origin/yaqi-results-20261103:src/models/trees.py`、`src/features/distance.py`、`src/features/macro.py`、`notebooks/Yaqi/delivery.py`。

通用 `RecentWindowRegressor` 在每个训练折内部筛选最近 N 个日历月，随后 clone 并拟合。初轮比较全历史、12 月、24 月；最新正式交付三个模型都用最近 **24 月**，并增加 `POSTAL_CODE`、`BLOCK` 地址类别。最终可用 150000 行，但实际训练的是 2023-04—2025-03 的 **71040 行**。

| 模型 | 源码中的数据表达和训练设置 |
|---|---|
| HistGradientBoosting 参考模型 | 数值直传，类别 ordinal 并标记为 categorical；原基线 120 轮，不使用外层验证早停 |
| RandomForest / ExtraTrees | 数值中位数填补并加缺失指示；类别缺失转字符串，one-hot，低频阈值 10；160 棵树、深度 24、叶子最少 8、max_features=0.8 |
| LightGBM | 数值 float32，训练词表固定的 pandas category，未知类别变缺失；700 轮、学习率 0.035 |
| XGBoost | 同样固定 category，`enable_categorical=True`、hist；700 轮、深度 6、学习率 0.035 |
| CatBoost | 类别以字符串输入，由库处理类别统计；`fit` 中创建 estimator 以兼容 clone；1000 轮、深度默认 7、学习率 0.04 |

数值主要为面积、时间、房龄/剩余租约、户型顺序、坐标、楼高与建成年份；类别主要为 town、户型、设计、街道、子区、规划区、大区，再按配置加地址。没有对这些树输入统一做 StandardScaler。

最新位置特征：距最近 open MRT（排除 LRT/RTS）、商场、小学的大圆距离。设施是静态快照，缺少历史开通日期；不能称为严格重建了每个历史月的设施。

经济特征：COE 按月及类别取价格均值；股票只使用 SGX、首月固定股票池，以各股首月归一后等权取指数。训练月 t 用 t−1 输入，每折只读取训练截止月 M−1 以前的经济记录，之后按持平或阻尼趋势预测。正式三个模型中，CatBoost/LightGBM 用位置＋COE，XGBoost 用基础特征；三个都没有选择股票。

旧分支 `yaqi-tree-models` 的模型在 ZIP/patch 中，不能只搜索顶层 `src/` 就判定“没有模型”。旧最佳 CatBoost 为 485.463，最新位置＋COE 为 484.521；它们是不同运行。

**报告与源码差异：**下午森林报告写“每字段最多 32 类”，但最新 `trees.py` 的 `OneHotEncoder` 只设置 `min_frequency=10`，没有 `max_categories=32`。森林下午成绩保存在报告/汇总里，当前三个正式交付也不含森林，故这些森林成绩不能据现有入口宣称已完整复现。该问题不影响此次从三个正式模型预测文件重算分数。

### 3.3 已保存结果的独立核对

以下从对应分支 `results/` 内逐行验证预测重新计算，非本轮重训，非 Kaggle 排名分数。A/B 是两折时间窗口，均值是两折 RMSE 的简单平均。

| 正式交付 | A RMSE | B RMSE | 平均 RMSE | 种子 |
|---|---:|---:|---:|---:|
| FXY 市场比例 Ridge | 488.690 | 471.943 | 480.316 | 5228 |
| Yaqi CatBoost＋位置＋COE | 492.874 | 476.168 | 484.521 | 42 |
| Yaqi XGBoost 基础 | 495.948 | 475.680 | 485.814 | 42 |
| Yaqi LightGBM＋位置＋COE | 501.918 | 482.572 | 492.245 | 42 |

Yaqi 入口在进程内把配置及实验模块的种子改为 42，公共 `src/config.py` 仍为 5228。新分支默认遵循公共 5228；如要复现 Yaqi 的有限搜索，应单独明确运行条件，不能静默改公共种子。种子差异也意味着本表不是完全统一随机条件的算法因果比较。

### 3.4 KNN：能判断什么

目前只能说明预期需求：KNN 对尺度、距离定义、类别权重敏感，需要训练折内拟合缩放与编码，并处理高维 one-hot、近期租金水平和查询成本。具体 k、weights、距离度量、时间窗口和成绩在现有分支里均无实现证据。不能把这些预期写成队友已经采用的方法。

## 4. 你的三个模型需要哪些不同处理

基础清洗、样本定义、外层验证、原行号和评分口径全部复用。差异发生在“清洗表如何变为模型输入”这一层。

| 处理 | 当前树模型 | FXY Ridge | KNN（一般需求，未见实现） | 你的神经网络 |
|---|---|---|---|---|
| 数值尺度 | 通常直传/填补 | 源码直传 | 距离尺度必须设计 | 训练折内填补＋标准化，转 float32 |
| 类别 | 原生类别或 one-hot | 稀疏 one-hot | 需编码并设计距离权重 | 独立词表与 embedding；未知/缺失显式处理 |
| 高基数地址 | 可选地址类别 | `BLOCK_KEY` one-hot | 高维稀疏距离可能失效 | 分开编码街道、完整楼栋键；限制 embedding 大小 |
| 缺失 | 部分原生支持 | 当前所选列基本无缺失 | 需填补 | 数值 median＋缺失指示，类别 MISSING |
| 样本组织 | 每行一份房屋特征 | 每行一份房屋特征 | 每行与邻居比较 | 表格 Transformer 仍按行；时间 RNN/LSTM 使用月度聚合窗口 |
| 标签尺度 | 正式树结果为原租金 | 月市场比例 | 未知 | 默认标准化原租金；市场模型单独管理比例及还原 |
| 训练控制 | 固定树轮数 | alpha 内部时间选择 | k 内部选择 | mini-batch、优化器、梯度裁剪、内部时间早停 |

类别至少预留 PAD/UNK/MISSING（按模块需要；表格无 padding 可不用 PAD）。只在当前训练数据拟合词表，不能把任意类别编号当连续数值直接喂网络。`BLOCK` 在不同街道会重复，楼栋 embedding 使用 `(BLOCK, STREET)` 联合键。低频合并的阈值也只能从训练部分学习。

数值预处理对面积、房龄、坐标、距离、时间分别使用训练统计量；连续月份用数值表达，月份可加 sin/cos。不把未来每个月都作为一个未见类别。`MONTHLY_RENT`、`Id`、`row_index` 都不是预测输入。

主损失优先标准化租金上的 MSE，预测还原到 SGD 后选模型。若采用 log 或每平米租金，必须作为单独消融；log 的 MSE 不等价于原租金 RMSE。Scaler 和模型每折重新初始化，预测时使用训练统计量。

## 5. 三个模型应该怎样定义

### 5.1 Transformer 主线：表格特征 token

建议先做小型 **FT-Transformer 风格回归器**：每条租约是一条样本；类别 embedding 与数值 token 投影构成 token 集合，经 Transformer encoder 和 CLS/池化回归头输出一个租金。此处 token 表示不同字段，不是连续月份；不需要给房屋交易行添加时间位置编码。

输入：`x_num: [B, N_num]`、`x_cat: [B, N_cat]`，token 化后 `[B, N_tokens, d_model]`，输出 `[B]`。初始候选可为 d_model=32/64、2 层、4 个头（d_model 必须能整除头数）、dropout=0.1、batch=256/512。这是待验证起点，不是已测最优配置，也不是完整复刻论文的承诺。

先用 basic＋block，稳定后按一致口径加距离；不要先把所有目标统计、股票与大地址 embedding 一次性堆进网络。建立相同预处理的 MLP 对照，检查 Transformer 的收益是否来自 attention。

依据：[FT-Transformer 原论文](https://arxiv.org/abs/2106.11959)、[PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)。

### 5.2 RNN/LSTM 主线：月度市场预测＋房屋属性模型

本仓库的当前/新版约束都明确：没有房屋 ID，不能把同楼栋同户型记录假定为同一套房的时间轨迹。禁止把按日期排序后的 150000 行直接滑窗，或把列顺序当作“时间”。伪单位最多用作有声明的分组/固定效应，不用于伪造单套房 LSTM。

推荐继承 FXY 的市场分解思路：

```text
仅当前训练折的租金
    ├─ 按月聚合 m_t → 历史窗口 → RNN / LSTM → 未来市场水平 m_hat_t
    └─ y_i / m_t + 房屋属性 → 固定的属性回归器 → 比例 q_hat_i
逐行预测 = q_hat_i × 对应月份的 m_hat_t
```

第一版 m_t 用全市月租金中位数，与 FXY 便于对照。属性回归器先固定为同一种小型 embedding MLP；也可在获得对应源码依赖后采用 FXY Ridge 作诊断对照。RNN 和 LSTM 必须使用相同属性回归器、输入、窗口与训练预算，仅替换时间编码器。

为了分离“市场预测”本身的贡献，至少比较 carry-last / 简单阻尼趋势 / RNN / LSTM。若 RNN/LSTM 无法稳定优于 carry，保留真实实验结论，不因模型名称更复杂就交付。

市场比例中的当月 m_t 是训练目标的归一化，不作为房屋输入特征；验证/测试只用预测 m_hat。若扩展为输入型目标统计或联合训练，应采用过去历史/交叉拟合，并单独说明样本自身对月度统计的影响。比例损失与原租金损失权重不同，可用训练月 m_t² 加权比例 MSE 作消融；最终都以逐行 SGD RMSE 判断。

序列张量 `[B, L, D]`，L 首先试 3/6，必要时再试 12；一层单向 RNN 或 LSTM、hidden=16/32、外接 dropout/线性头，避免起步使用大型网络。单层 PyTorch RNN/LSTM 的内部 dropout 不作用于最后输出层，要显式放在网络外部。依据：[RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)、[LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)。

**数据量限制是真实瓶颈：**A 只有 35 个训练月份，B 只有 39 个，全量最终训练只有 51 个。全市单序列 L=6 时，A 只有 29 个一步训练窗口；150000 条租约不等于 150000 个时间步。先用小网络与强基线，不应默认 LSTM 会赢树模型。

第二阶段才考虑共享参数的多市场序列：先按户型，必要时再 town×户型。实际全训练中 town×户型×月份有 5894 个非空组，中位数每组 17 条，其中 2194 组不足 10 条；各折还会更稀疏。阈值、市场集合与回退必须在每折训练部分确定，不按全量未来覆盖挑组。小组可收缩到“户型月水平 → 全市月水平”，同时保留 count/missing mask；它们相关且非独立样本。

### 5.3 如果需要公平比较三个“时间编码器”

表格 Transformer 与上述市场 RNN/LSTM 的建模对象不同，可以比较最终预测质量，但不能据此声称比较出了 attention 与 recurrence 的时间建模优劣。

架构中为 `encoder="transformer"` 保留同一个月度预测接口：使用完全相同的市场序列、L、D、forecast horizon 和属性模型，只把 RNN/LSTM 替换为小型 temporal Transformer，并增加月份位置/时间编码。这样可形成严格的 RNN / LSTM / temporal Transformer 对照；表格 Transformer 另作表格模型实验。正式交付仍最多选择三项最有价值结果，不把每个候选都塞进 results。

## 6. 时间信息与防泄漏的具体设计

### 6.1 外层验证原样复用

| 折 | 训练期间/行数 | 验证期间/行数 |
|---|---|---|
| A / `holdout` | 2021-01—2023-11，102829 | 2023-12—2025-03，47171 |
| B / `rolling_2024_04` | 2021-01—2024-03，114842 | 2024-04—2025-03，35158 |

直接调用公共 `get_splits`，不要重写日期边界。A/B 验证有重叠，两折预测分别保留，禁止当成 82329 条独立租约计算一个“总体”成绩。内层时间验证可以用于选超参数和 epoch，但不能替代这套正式两折。

### 6.2 聚合和窗口必须在切分后构造

1. 先得到当前折 train；只有这一部分的 y 可用于建市场表、fit scaler、选择市场分组或训练属性模型。
2. 对训练标签月份 t，输入只含 t 以前的 L 个完整月份，不能把 m_t 或包含 y_t 的当月聚合放入该窗口；多步训练样本的所有目标月也必须 ≤ 训练截止月。
3. 对缺月先补完整月历；只能用过去信息填补。不可线性插值跨越未来月份或 backward fill，count/mask 区分“无样本”和零租金。
4. 首个 L 月无法训练完整窗口时，时间模块不生成这些窗口；属性分支仍可使用这些行，最终逐行预测覆盖全部验证/test。
5. 全局月度 scaler 按训练市场表拟合；不要把同一月重复到几千条租约后无意改变 scaler 权重。更严格的早期原点训练/内层回测应在各自前缀重新拟合统计量。

### 6.3 固定原点多步预测

A 的原点是 2023-11，整个后续 16 月都不能逐月读入验证真实租金；B 也一样。推荐第一版一步预测器递归 rollout，预测结果填入下一步历史，不 teacher-force 验证标签。`predict` 只依赖拟合历史和查询月份，按最大 horizon 生成缓存；分批预测、打乱查询行后还原顺序，应给出相同值。

最终拟合原点为 2025-03。test 中 2025-03 的查询用当前训练已知月份水平，其余月份递归到 2026-07。不得从无标签 test 的样本分布推算真实月租金或用最终训练状态回填外层验证。

### 6.4 宏观、设施与早停

- COE/股价文件虽包含测试期实测值，主实验沿用队友固定原点假设：截止 M−1，之后只用历史生成的未来预测。先不加宏观，再以 COE 为单独消融；避免把表里“存在”当成原点时“已知”。
- 当月均值必须是月结束才完整可得；训练样本 t 用 t−1，外层整个未来段也不追加真实宏观值。设施沿用快照局限，最好统一采用 Yaqi 的非 MRT 线路排除口径。
- 神经网络 epoch/学习率在外层训练的末端内部月份选择；内层 encoder/scaler/市场历史同样只看内层训练前缀。选定 epoch 后，从头用整个外层训练部分拟合，再仅评估外层 valid。
- 样本窗口构造完成后可以 shuffle 训练 mini-batch，不能 shuffle 原始租约后再切时间窗。每个独立窗口重新初始化隐藏状态，避免跨市场/跨 batch 串状态。
- 固定 Python/NumPy/PyTorch 种子并记录 CPU/CUDA/MPS、版本及确定性设置。框架只设置 Python/NumPy，不会替你设置 torch。

## 7. 拟采用的代码架构

### 7.1 先获取公共基底，后新增个人模块

当前分支缺少框架。开始写代码时，应把已审阅的公共基底 `origin/merge-data-cleaning` 合入个人分支，或使用团队届时真正更新后的 main；不能假定当前 main 已含清洗。**本轮未执行合并。**

无需把 Yaqi 全部结果和旧 ZIP 合入。距离/经济或 FXY 模型若需复用，应有明确来源和模块依赖；公共框架只在公共分支接收独立修复。先用 basic/block，个人专用特征使用 `yuxin_` 前缀注册，避免占用已有/预留名称。

### 7.2 目标目录（尚未创建）

```text
CS5228/
├── develop.md                         # 本文
├── data/                              # 现有原始数据
├── data_cleaned/                       # 复用公共基底
├── docs/experiment-interface.md        # 公共唯一接口协议
├── src/
│   ├── data_cleaning/                  # 复用 loader/清洗
│   ├── cv.py                          # 复用 get_splits
│   ├── experiment.py                  # 复用实验运行与提交
│   ├── features/
│   │   └── yuxin_calendar.py           # 拟新增：周期月份等无标签特征
│   └── models/
│       └── yuxin_nn/                  # 拟新增：个人可复用模型包
│           ├── __init__.py            # 导出 estimator/factory
│           ├── preprocessing.py       # 数值填补/缩放、类别词表及未知处理
│           ├── datasets.py            # 行数据与月度窗口 Dataset
│           ├── tabular.py             # embedding MLP 与表格 Transformer
│           ├── market.py              # 月度聚合、回退、原点、rollout
│           ├── temporal.py            # RNN/LSTM/temporal Transformer
│           ├── training.py            # torch 训练、内部时间早停与重拟合
│           └── estimators.py          # sklearn clone 接口及预测原序还原
├── notebooks/Yuxin/
│   ├── README.md                      # 拟新增：复现命令、配置说明
│   ├── requirements.txt               # 拟新增：NN 直接依赖，复用公共依赖
│   ├── run_experiments.py             # 拟新增：CLI、模型配置、注册导入
│   ├── configs/                       # 拟新增：有限实验配置，JSON
│   │   ├── tabular_transformer.json
│   │   ├── market_rnn.json
│   │   ├── market_lstm.json
│   │   └── market_transformer.json
│   ├── analyze_results.py             # 拟新增：逐月/地区/户型/预测跨度误差
│   ├── verify_delivery.py             # 拟新增：行号、RMSE、提交格式核验
│   └── figures/                       # 拟新增：报告选定图
├── tests/
│   └── test_yuxin_nn.py               # 拟新增：数据契约及时间边界小测试
└── results/Yuxin/                     # 定稿后生成，最多三个正式运行
    ├── README.md
    ├── experiments/log.csv
    ├── experiments/<run_id>/
    │   ├── config.json
    │   └── validation_predictions.csv
    └── submissions/<run_id>.csv
```

不把市场目标序列当普通 `FeatureBlock.transform` 中即时 groupby 的结果：它需要训练标签、截止点、递归状态及与样本的月份映射，统一由模型包内部持有。日历等无标签新增列才放到 features。优先保持模块少而职责清楚，不为三个模型各复制整套训练脚本。

### 7.3 estimator 与内部模块边界

拟提供两个顶层 sklearn 回归器：

- `TabularNNRegressor(kind="mlp" | "transformer", ...)`：`fit(X,y)` 内部拟合词表/scaler、选 epoch、创建 torch 网络；`predict(X)` 输出原尺度 `[N]`。
- `MarketHybridRegressor(encoder="carry" | "rnn" | "lstm" | "transformer", ...)`：同时持有属性模型、训练截止点、市场历史、时间模型、rollout；`predict` 将每条房屋比例乘查询月份市场预测，并恢复原行序。

两个类都继承 `RegressorMixin, BaseEstimator`，构造参数原样保存；不在 `__init__` 创建已拟合网络/读取数据。学习状态放在 `preprocessor_`、`network_`、`market_history_`、`origin_` 等下划线属性。每折 clone 时必须从零开始。

这些自定义 estimator 直接返回 SGD，调用框架时不再叠加 `ModelWrapper` 的 log/per_sqm，还原逻辑只保留一处。`fit(X,y)` 的 X 保留 `RENT_APPROVAL_DATE` 供时间处理，y 按位置对应；市场聚合使用显式位置数组或匹配 index，防止 pandas 隐式对齐引入 NaN。

框架无需接收 `[N,L,D]`：它仍只传 DataFrame，模型内部负责生成二维/三维张量。这样不会破坏队友树模型、Ridge、特征注册与交付流程。

### 7.4 依赖、输出和 `.gitignore`

当前个人分支 `.gitignore` 只忽略缓存、环境与系统文件；公共版本另外忽略默认 `experiments/*/validation_predictions.csv` 和 `submissions/*.csv`，**没有**忽略 `results/<成员名>/`。两版都没有专门忽略 `.pt/.pth`、checkpoint 或输出目录。

后续将 NN 依赖独立放 `notebooks/Yuxin/requirements.txt`（含公共依赖引用、torch、必要绘图库），不把个人完整 `pip freeze` 覆盖公共 requirements。正式配置另外记录实际 Python、torch、numpy、pandas、sklearn 版本和设备。

试跑输出到 `tempfile.mkdtemp()` 等仓库外临时目录；checkpoint 同样放临时目录，避免 `.pt` 权重被误提交，也不需要为个人训练修改公共 `.gitignore`。临时缓存按 fold/origin/config 区分，不跨切分复用市场历史或 scaler。

## 8. 实验顺序与交付

### 8.1 实施顺序

1. 接入公共基底，检查 loader、公共契约测试和基线；建立一个很小的 NN smoke run，先核对 clone、行序、未知类别及原尺度输出。
2. 训练 embedding MLP 作为预处理对照，再做表格 Transformer；初始只开 basic/block。
3. 建立市场聚合、carry 基线与固定属性回归器，确认能生成覆盖全部验证行的预测。再分别接入小 RNN、LSTM。
4. 如要比较时间编码器，接入同窗口 temporal Transformer；先采用短窗口与单步递归，之后才尝试多市场池化/距离/COE。
5. 用统一 rolling 输出逐月、地区、户型、预测跨度的误差；同时对齐队友预测计算误差相关性。神经网络单模稍差仍可能互补，但必须用实际融合验证证明。
6. 锁定每个选中模型的配置与训练轮数方案，最多交付三个正式运行。未优于简单对照的模型也保留实验说明，避免只汇报挑选出的成绩。

主实验默认公共种子 5228；多随机种子稳定性分析可在个人 runner 中明确管理并独立记日志，但公共框架会覆盖名为 `random_state` 的参数，因此不能以为更改构造器参数就改变了框架种子。不要为跑多种子静默 monkey-patch 公共模块。

### 8.2 拟定命令（完成实现之后才能运行）

```bash
python -m pip install -r requirements.txt
python -m pip install -r notebooks/Yuxin/requirements.txt
python -m unittest discover -s tests -v
python notebooks/Yuxin/run_experiments.py --model tabular_transformer --split holdout
python notebooks/Yuxin/run_experiments.py --model market_rnn --split holdout
python notebooks/Yuxin/run_experiments.py --model market_lstm --split holdout
python notebooks/Yuxin/run_experiments.py --model market_transformer --split holdout
```

日常 holdout 仅用于迭代；定稿调用 `run_experiment(..., split="rolling", final=True, output_dir=ROOT_DIR / "results" / "Yuxin")`。公共接口和产物规则以审阅提交的 [实验协议](https://github.com/crk13/CS5228/blob/907aa001f08a23e1c9076db901877a39468dccfc/docs/experiment-interface.md) 为准，本文只是接入设计，不另立协议。

### 8.3 必要验收项

- `clone()` 不携带拟合状态；同配置种子复跑在所记录设备条件下检查可复现性。
- 改动验证租金不影响模型预测；改动截止点之后的宏观实测值不影响主实验预测。
- 月度训练窗口的最大输入月严格早于目标月，多步训练目标不跨训练截止点；缺月仅因果填充。
- 缺失值、未见类别、稀疏市场和首次历史不足均有明确回退。
- 打乱/分批查询再还原，与一次性预测一致，`predict` 不从查询批次更新学习状态。
- A/B 验证文件分别 47171/35158 行，总计 82329；列为公共五列，按 `(fold,row_index)` 唯一，对应原始 train 位置与真实标签。
- test 输出恰好 50000 行 `Id,Predicted`，Id 顺序和样例一致，预测有限，SGD RMSE 可从 CSV 重算。
- 不把模型训练集拟合预测作为融合 OOF。若另存训练诊断，明确标记；融合使用时间留出验证预测。
- 进入融合的运行全部放个人 results、最多三个，并记录配置/版本/原点/输入/目标还原/训练窗口；不提交模型权重。

## 9. 证据与后续复查入口

分支之后可能继续变化，下面链接固定到本次审阅的提交：

- [公共数据清洗说明](https://github.com/crk13/CS5228/blob/907aa001f08a23e1c9076db901877a39468dccfc/data_cleaned/README.md)、[公共接口协议](https://github.com/crk13/CS5228/blob/907aa001f08a23e1c9076db901877a39468dccfc/docs/experiment-interface.md)。
- [FXY 正式模型](https://github.com/crk13/CS5228/blob/6e7ffcb0ccf20bacba42ca9d471084a3ab3aec71/src/models/market_ratio.py)、[FXY 交付说明](https://github.com/crk13/CS5228/blob/6e7ffcb0ccf20bacba42ca9d471084a3ab3aec71/results/FXY/README.md)。
- [Yaqi 树模型源码](https://github.com/crk13/CS5228/blob/3dae63c3ba9e5919fadb9af54474fee20d86f751/src/models/trees.py)、[经济输入边界](https://github.com/crk13/CS5228/blob/3dae63c3ba9e5919fadb9af54474fee20d86f751/src/features/macro.py)、[正式交付说明](https://github.com/crk13/CS5228/blob/3dae63c3ba9e5919fadb9af54474fee20d86f751/results/Yaqi/README.md)。
- [旧树模型 ZIP 入口说明](https://github.com/crk13/CS5228/blob/a69bdd546efa377888987b026f1da760e8d4e08f/YAQI_RESULTS.md)。
- 原始竞赛评分已从本地 `CS5228-2610 Project _ Kaggle.mhtml` 确认是 RMSE；本文没有依赖往届竞赛规则。

本地可用下列只读命令继续检查，无需切换当前分支：

```bash
git branch -a
git show origin/data-cleaning:data_cleaned/README.md
git show origin/data-cleaning:docs/experiment-interface.md
git show origin/p2-linear:src/models/market_ratio.py
git show origin/yaqi-results-20261103:src/models/trees.py
git diff --stat origin/data-cleaning origin/merge-data-cleaning
```
