# Yuxin 开发验收：develop.md §8.1 的步骤 1、2、3

日期：2026-11-04。分支：`yuxin_time_series`。

**三个开发步骤已实现并实际运行，数据、公共接口、时间边界和交付格式检查通过。**
Transformer、RNN、LSTM 各已生成两折完整验证预测及 50000 条测试预测。
本次是初始配置的开发验收，不是调参完成或 Kaggle 排名验收；没有规定最低 RMSE，因此“格式/流程通过”与“优于基线”分别汇报。

## 1. 本轮接入了什么

执行 `git fetch origin --prune` 后，远程 main 已更新到公共框架提交。
在个人分支执行 `git merge --ff-only origin/merge-data-cleaning`，从 `7def64e` 快进到 `32213c435466bf4b081efef9bdcd10a73856b5b5`。
分支名称保持不变，原有 `develop.md` 保留，没有合入队友的个人实验、结果包或树模型源码。

公共基底提供了 `data_cleaned/`、loader、`FeaturePipeline`、时间切分、模型接口、实验运行器和测试。
之后的开发仅新增个人模型、个人运行/核验入口、测试和文档。
公共清洗、`src/cv.py`、`src/experiment.py`、公共 `.gitignore`、requirements、共享日志均未作个人适配修改；原始/清洗数据未重新生成或修改。

实际入口：

- [神经网络模型包](src/models/yuxin_nn/estimators.py)：`TabularNNRegressor`、`MarketHybridRegressor`。
- [个人运行说明](notebooks/Yuxin/README.md)、[实验入口](notebooks/Yuxin/run_experiments.py)、[配置目录](notebooks/Yuxin/configs/)。
- [新增测试](tests/test_yuxin_nn.py)、[逐行交付核验](notebooks/Yuxin/verify_delivery.py)。
- [所有模型比较](notebooks/Yuxin/comparison.csv)、[逐月误差](notebooks/Yuxin/monthly_errors.csv)。
- [完整复核记录](notebooks/Yuxin/verification.json)、[三个正式候选](results/Yuxin/README.md)。

没有提前实现步骤 4 的时间 Transformer、多市场池化，以及后续融合/调参工作。表格 Transformer 已实现，它的 token 表示字段。
本阶段只开公共 `basic` / `block`，因此未创建暂不使用的日历特征块或占用距离/宏观注册名。

## 2. 步骤 1：公共框架、数据和 smoke 验证

公共原有 8 项契约测试全部通过；新增 7 项神经网络测试后，共 **15 项通过**。
`python -m src.data_cleaning.checks` 输出 `All checks passed.`。
小数据 smoke 覆盖五种模型的真实 `fit/predict`、clone 和公共 `run_experiment` 接入，随后在完整数据上运行了全部模型。

公共两折切分未改：

| 折 | 训练范围 | 训练行数 | 验证范围 | 验证行数 |
|---|---|---:|---|---:|
| A / holdout | 2021-01—2023-11 | 102829 | 2023-12—2025-03 | 47171 |
| B / rolling_2024_04 | 2021-01—2024-03 | 114842 | 2024-04—2025-03 | 35158 |

两折平均为两个 RMSE 的简单平均。两个验证区间有重叠，没有把 82329 行当成独立样本合并评分。
全部最终训练使用 150000 行，再预测 test；无最近 24 月截断。

公共基线本次复跑结果与原参考结果吻合：分组中位数平均 RMSE 515.448，HistGradientBoosting 489.241。
为保护共享日志，基线也通过原 `run_experiment` 写入个人验证目录，未调用会修改共享日志的默认输出路径。

## 3. 步骤 2：embedding MLP 和表格 Transformer

预处理全部在每折训练部分拟合：

- 11 个数值字段：面积、租约年份、月份、连续月份、房龄、剩余租约、户型顺序、经纬度、最高楼层、建成年份。
- 数值中位数填补、标准化，每列保留缺失标记；输出 float32。全缺失训练列有 0 填补回退。
- 8 个类别字段：town、户型、设计、街道、子区、规划区、大区、完整楼栋键。
- 完整楼栋键为 `BLOCK|STREET`；词表只在当前训练部分拟合，低频阈值 5。UNK=0、MISSING=1；不把类别编号直接当连续数值。
- 不把租金、Id、row_index、日期字符串作为输入字段；连续月份另以数值表达。

MLP 为 embedding + 128/64 隐藏层、dropout=0.1；最大 30 epoch。
最终表格 Transformer 为 **d_model=16、4 头、1 层、dropout=0.1、batch=1024、最大 18 epoch**。
同一数值字段的值与缺失标记共同构成一个 token，总计 11 数值 token＋8 类别 token＋CLS，即 20 个 token。
这是小型 FT-Transformer 风格实现，不声称完整复现论文。

原租金先按训练部分均值/标准差标准化，以 MSE 学习，预测还原为 SGD；没有使用 log 目标。
在外层训练末端 3 个月内部选 epoch，内层词表、数值和目标统计也仅在内层训练部分拟合。
选定轮数后，重新初始化网络、优化器和预处理，用整个外层训练部分重拟合。

| 模型 | A 选定 epoch | B 选定 epoch | 最终全量训练 epoch |
|---|---:|---:|---:|
| MLP | 1 | 3 | 28 |
| 表格 Transformer | 10 | 16 | 14 |

MLP 的 A/B 早停轮数较少是内部时间验证的真实选择，没有用外层验证标签强行增加轮数。两种模型的容量/训练预算不同，本次比较不是“只替换 attention”的严格消融。

## 4. 步骤 3：市场 carry、RNN、LSTM

实现的是市场分解模型：

```text
当前训练折每月租金中位数 m_t
    ├─ y_i / m_t → 相同的属性 embedding MLP → q_hat_i
    └─ 月度窗口 → carry / RNN / LSTM → m_hat_t
逐条租约预测 = q_hat_i × m_hat_t
```

属性分支在三个市场模型中保持相同：隐藏层 128/64、相同 seed、训练样本、预处理、内部验证和训练预算。
它不使用 MONTH_INDEX / RENT_MONTH，保留其他 9 个数值字段及 8 个类别字段。
训练中的 m_t 只用于训练目标比例归一化，不作为包含自身租金的输入特征；未来使用模型预测的市场水平。

月度分支只有全市一个序列，没有假设楼栋记录属于同一套房，也没有把交易行串成时间步。
训练月数 A/B/最终分别为 35/39/51，lookback=6，窗口数量为 29/33/45。
每个输入窗口截至 t−1，预测 t 的市场水平增量。RNN/LSTM 为单层、hidden=16；每个窗口重新初始化隐藏状态。
月度数值按训练市场表标准化，时间网络最大 160 epoch，内层末端 6 个月按月度市场 RMSE 选 epoch。

| 分支 | A 属性 epoch | B 属性 epoch | 最终属性 epoch | A 时间 epoch | B 时间 epoch | 最终时间 epoch |
|---|---:|---:|---:|---:|---:|---:|
| carry | 11 | 11 | 12 | — | — | — |
| RNN | 11 | 11 | 12 | 3 | 5 | 20 |
| LSTM | 11 | 11 | 12 | 6 | 9 | 10 |

未来以固定原点递归：预测结果加入下一步窗口，不读取验证真实租金；预测增量统一乘 0.8^h 阻尼。
最终原点为 2025-03，test 中同月查询使用已知训练市场水平 3200，之后递归至 2026-07。
最终预测的 2026-07 市场水平：carry=3200，RNN≈3259.931，LSTM≈3263.193。这些是预测，不是真实未来租金。

月历补齐只向前填充，并记录 count/observed。实际本轮全市序列没有缺月；尚未做稀疏多市场序列及其缺失 mask 输入设计，不将这部分写成已完成能力。

已从验证和测试 CSV 中除去各自市场预测，独立恢复属性比例：carry/RNN/LSTM 的比例预测一致，最大绝对差 **4.440892098500626e-16**；属性训练曲线和审计参数也一致。
因此它们之间的成绩变化可以定位到市场分支，而非换了属性模型。

## 5. 完整实测结果

下表均来自保存的逐行预测重新计算，单位 SGD，越低越好；不是 Kaggle 排行榜成绩。
时间包含两折验证及最终全量训练/测试预测，硬件为本机 CPU，非跨机器速度比较。

| 模型 | A RMSE | B RMSE | 两折平均 | 耗时/秒 | 位置 |
|---|---:|---:|---:|---:|---|
| 市场 carry＋属性 MLP | 493.493 | 477.729 | **485.611** | 104.8 | 个人验证目录 |
| HistGradientBoosting | 497.937 | 480.545 | 489.241 | 5.9 | 个人验证目录 |
| 市场 RNN＋属性 MLP | 497.557 | 482.783 | 490.170 | 101.9 | results/Yuxin |
| 市场 LSTM＋属性 MLP | 498.832 | 481.537 | 490.184 | 102.0 | results/Yuxin |
| 表格 Transformer | 515.646 | 484.467 | 500.056 | 708.2 | results/Yuxin |
| 最近六个月分组中位数 | 524.659 | 506.237 | 515.448 | 1.2 | 个人验证目录 |
| 原租金 embedding MLP | 661.392 | 571.097 | 616.244 | 95.6 | 个人验证目录 |

本轮 carry 比树参考基线低约 3.630 SGD。RNN、LSTM 均未优于 carry，也略弱于本次树基线。
两种循环网络平均只差约 0.015 SGD，单种子初始配置不足以支持算法优劣结论。
表格 Transformer 优于当前原租金 MLP，但仍未达到树基线；没有把这些模型包装成“已胜出”。

市场分解模型与原租金 MLP 同时改变了目标表达和时间输入，不能把其差异全部归因于目标归一化。
本阶段未调融合权重，没有实证确认这些候选能改善团队融合。保留较弱结果，是为了完整记录负责模型的真实实现和验证结果。

## 6. 评测条件与复核证据

最终 [完整复核 JSON](notebooks/Yuxin/verification.json) 的 status 为 `passed`，含两个目录的 7 个完整运行及共享属性检查。
三个候选的 [独立交付复核](results/Yuxin/verification.json) 同样为 `passed`。

| 检查 | 实际结果 |
|---|---|
| 公共与新增单元测试 | 15/15 通过；含真实 fit/predict 的小数据 smoke |
| 公共清洗一致性 | All checks passed；邮编类型和训练/测试字段符合 loader 约定 |
| 验证标签扰动 | 对五种模型，把验证租金增大 100000，不改变预测；只改变评分 |
| 时间窗口 | 输入月严格早于目标月，窗口不越过训练截止点；缺月测试未使用未来插值 |
| 内部早停 | 内层训练截止早于内层验证起点，内层验证不超过外层训练截止 |
| clone / 重复运行 | clone 不带拟合状态；同设备同 seed 小数据重复拟合预测一致 |
| 未见类别/NaN | 显式 UNK/MISSING、训练统计填补后可产生有限预测 |
| 推理顺序/分批 | 小数据未来查询、真实拟合 probe 均通过；真实 probe 最大差约 8.94e-5 SGD，低于 0.01 SGD 容差 |
| 验证 CSV | 每运行 82329 行；A=47171、B=35158；原行号、日期、标签逐行匹配，(fold,row_index) 唯一 |
| 指标 | 原尺度 RMSE 重算与 config/log 一致，绝对容差 1e-8 |
| 测试 CSV | 每运行 50000 行，列严格为 Id,Predicted，Id=0..49999 且与 loader 一致 |
| 预测值 | 全部有限、正值；新增最终回归器按租金语义设置最低 1 SGD |
| 共享属性对照 | 两折验证与全量 test 均一致，比例最大差 4.44e-16 |
| 数据完整性 | 17 个原始/清洗 CSV 的 SHA-256 与每次训练前记录相同 |
| 结果目录 | results/Yuxin 恰好 3 个运行，无锁文件，无模型权重 |

本轮没有使用 COE、股票、设施距离等额外输入，因此没有声称完成这些模块的未来实测扰动实验；神经网络实现不读取这些辅助表。
高层训练流程始终由公共框架提供，不能通过目标列或测试行号进入模型特征。

## 7. 环境与运行中处理的问题

- Python 3.12.7；numpy 1.26.4；pandas 2.2.2；scikit-learn 1.5.1；torch 2.15.0。
- macOS arm64，CPU；种子 5228，torch deterministic；神经网络单线程。两个独立训练进程曾并行使用 CPU，计时不代表无竞争基准。
- PyTorch 安装在仓库忽略的 `.venv` 中。环境继承已有公共科学计算包，未卸载或修改全局 Python 包。
- 首轮把 Conda sklearn 与 PyTorch 的 OpenMP 一起切线程时，真实规模 tensor 转换停在 libomp 屏障。采样确认后终止该进程，改为 torch 单线程、runner 只控制 BLAS，重新训练成功。
- 初始 d_model=32 / 2 层 / 31 token 的 Transformer 在 CPU 上成本偏高，未完成首折即停止；改为 16 / 1 层 / 20 token 后，从头完成全流程。此调整基于计算成本，不是看过外层验证后挑分数。
- 两次中断的未完成运行目录已移到 `/tmp/yuxin_interrupted_<run_id>`，未作为成功实验或候选成绩统计。已完成基线未覆盖。

模型源码哈希、实际版本、数据哈希及各折审计均附在每个运行的 config/training_audit 中。
早期基线/MLP/carry 的源码快照可能包含调整前、但未被该模型调用的 Transformer 类；其实际预处理/MLP 路径未改变，跨市场分支的数值一致性已独立核对。
run_id 使用 UTC，因此编号以 20261103T16 开头，对应本地 2026-11-04 凌晨。

## 8. 结果交付索引与复现

正式三个候选位于 `results/Yuxin/`，恰好占用公共规范允许的三个运行：

| 模型 | run_id |
|---|---|
| 表格 Transformer | 20261103T165254207013Z_e7366de1 |
| 市场 RNN | 20261103T165254212024Z_ae71e28f |
| 市场 LSTM | 20261103T165436355139Z_16342354 |

每个 `experiments/<run_id>/` 包含 config、training_audit、validation_predictions；`submissions/<run_id>.csv` 为最终测试预测。
这些是原尺度租金，可按 `(fold,row_index)` 参与后续融合评估；没有用训练集拟合值冒充 OOF。
MLP、carry 和公共基线保存在 `notebooks/Yuxin/validation/`，供核查与后续选择；它们不占正式候选目录名额。

在仓库根目录复核现有产物：

```bash
LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.data_cleaning.checks
.venv/bin/python notebooks/Yuxin/verify_delivery.py \
  --roots notebooks/Yuxin/validation results/Yuxin
.venv/bin/python notebooks/Yuxin/analyze_results.py \
  --roots notebooks/Yuxin/validation results/Yuxin
```

重新训练应写新的开发目录，不覆盖现有交付：

```bash
.venv/bin/python notebooks/Yuxin/run_experiments.py \
  --model mlp market_carry tabular_transformer market_rnn market_lstm \
  --split rolling --final --output /tmp/yuxin_reproduction --verbose
```

本轮按仓库规则做本地 Git 提交，不推送、不上传 Kaggle。
后续应优先检验市场趋势/内部验证长度和模型互补性；没有在本阶段继续搜索外层分数或实现后续步骤。
