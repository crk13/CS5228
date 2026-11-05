# Yuxin 开发验收：develop_yuxin_nn.md §8.1 的步骤 1—6

日期：2026-11-04。分支：`yuxin_time_series`。

> 最新步骤 4—6 的结果与复核见第 10 节；步骤 1—3 调参见第 9 节及 [develop_tuning_yuxin_nn.md](develop_tuning_yuxin_nn.md)。第 1—8 节保留初验记录；原三个候选已归档，不代表当前成绩。历史章节中的“未实施步骤 4/融合”仅描述当时状态。

**三个开发步骤已实现并实际运行，数据、公共接口、时间边界和交付格式检查通过。**
Transformer、RNN、LSTM 各已生成两折完整验证预测及 50000 条测试预测。
本次是初始配置的开发验收，不是调参完成或 Kaggle 排名验收；没有规定最低 RMSE，因此“格式/流程通过”与“优于基线”分别汇报。

## 1. 本轮接入了什么

执行 `git fetch origin --prune` 后，远程 main 已更新到公共框架提交。
在个人分支执行 `git merge --ff-only origin/merge-data-cleaning`，从 `7def64e` 快进到 `32213c435466bf4b081efef9bdcd10a73856b5b5`。
分支名称保持不变，`doc/develop_yuxin_nn.md` 保留，没有合入队友的个人实验、结果包或树模型源码。

公共基底提供了 `data_cleaned/`、loader、`FeaturePipeline`、时间切分、模型接口、实验运行器和测试。
之后的开发仅新增个人模型、个人运行/核验入口、测试和文档。
公共清洗、`src/cv.py`、`src/experiment.py`、公共 `.gitignore`、requirements、共享日志均未作个人适配修改；原始/清洗数据未重新生成或修改。

实际入口：

- [神经网络模型包](../src/models/yuxin_nn/estimators.py)：`TabularNNRegressor`、`MarketHybridRegressor`。
- [个人运行说明](../notebooks/Yuxin/README.md)、[实验入口](../notebooks/Yuxin/run_experiments.py)、[配置目录](../notebooks/Yuxin/configs/)。
- [新增测试](../tests/test_yuxin_nn.py)、[逐行交付核验](../notebooks/Yuxin/verify_delivery.py)。
- [初验模型比较](../notebooks/Yuxin/archive/stage1_3/comparison.csv)、[初验逐月误差](../notebooks/Yuxin/archive/stage1_3/monthly_errors.csv)。
- [初验复核记录](../notebooks/Yuxin/archive/stage1_3/verification.json)、[初验候选归档](../notebooks/Yuxin/archive/stage1_3/results_Yuxin/README.md)。

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
| 市场 RNN＋属性 MLP | 497.557 | 482.783 | 490.170 | 101.9 | 初验归档 |
| 市场 LSTM＋属性 MLP | 498.832 | 481.537 | 490.184 | 102.0 | 初验归档 |
| 表格 Transformer | 515.646 | 484.467 | 500.056 | 708.2 | 初验归档 |
| 最近六个月分组中位数 | 524.659 | 506.237 | 515.448 | 1.2 | 个人验证目录 |
| 原租金 embedding MLP | 661.392 | 571.097 | 616.244 | 95.6 | 个人验证目录 |

本轮 carry 比树参考基线低约 3.630 SGD。RNN、LSTM 均未优于 carry，也略弱于本次树基线。
两种循环网络平均只差约 0.015 SGD，单种子初始配置不足以支持算法优劣结论。
表格 Transformer 优于当前原租金 MLP，但仍未达到树基线；没有把这些模型包装成“已胜出”。

市场分解模型与原租金 MLP 同时改变了目标表达和时间输入，不能把其差异全部归因于目标归一化。
本阶段未调融合权重，没有实证确认这些候选能改善团队融合。保留较弱结果，是为了完整记录负责模型的真实实现和验证结果。

## 6. 评测条件与复核证据

初验 [完整复核 JSON](../notebooks/Yuxin/archive/stage1_3/verification.json) 的 status 为 `passed`，含两个目录的 7 个完整运行及共享属性检查。
初验三个候选的 [独立交付复核](../notebooks/Yuxin/archive/stage1_3/results_Yuxin/verification.json) 同样为 `passed`。

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

初验时三个候选位于 `results/Yuxin/`；现完整迁入 `notebooks/Yuxin/archive/stage1_3/results_Yuxin/`：

| 模型 | run_id |
|---|---|
| 表格 Transformer | 20261103T165254207013Z_e7366de1 |
| 市场 RNN | 20261103T165254212024Z_ae71e28f |
| 市场 LSTM | 20261103T165436355139Z_16342354 |

每个 `experiments/<run_id>/` 包含 config、training_audit、validation_predictions；`submissions/<run_id>.csv` 为最终测试预测。
这些是原尺度租金，可按 `(fold,row_index)` 参与后续融合评估；没有用训练集拟合值冒充 OOF。
MLP、carry 和公共基线保存在 `notebooks/Yuxin/validation/`，供核查与后续选择；它们不占正式候选目录名额。

在仓库根目录复核历史初验产物（最新交付见第 9 节）：

```bash
LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.data_cleaning.checks
.venv/bin/python notebooks/Yuxin/verify_delivery.py \
  --roots notebooks/Yuxin/validation notebooks/Yuxin/archive/stage1_3/results_Yuxin \
  --summary-output notebooks/Yuxin/archive/stage1_3/recheck_verification.json
.venv/bin/python notebooks/Yuxin/analyze_results.py \
  --roots notebooks/Yuxin/validation notebooks/Yuxin/archive/stage1_3/results_Yuxin \
  --output notebooks/Yuxin/archive/stage1_3/recheck_analysis
```

重新训练应写新的开发目录，不覆盖现有交付：

```bash
.venv/bin/python notebooks/Yuxin/run_experiments.py \
  --model mlp market_carry tabular_transformer market_rnn market_lstm \
  --split rolling --final --output /tmp/yuxin_reproduction --verbose
```

本轮按仓库规则做本地 Git 提交，不推送、不上传 Kaggle。
后续应优先检验市场趋势/内部验证长度和模型互补性；没有在本阶段继续搜索外层分数或实现后续步骤。

## 9. 本轮调参完成：步骤 1—3，不进入步骤 4

本轮已实际执行 Optuna TPE、人工暖启动/局部搜索、内部时间早停，并补全所有时间网络剪枝候选的第二折；没有采用不适合本项目从头训练小网络的 QLoRA。
单配置提升不足后，增加了**同一模型家族内最多 4 个配置的等权平均**；没有混入树、Ridge 或 KNN 预测。

以下为独立重训、从完整 CSV 重算的开发 RMSE，单位 SGD；平均为公共 A/B 两折的简单平均：

| 本轮选定方案 | A | B | 平均 | 比队友最佳树低 |
|---|---:|---:|---:|---:|
| 表格 Transformer（4 成员平均） | 487.952332 | 474.705755 | **481.329044** | 3.192360 |
| carry＋4 成员属性 MLP | 486.307715 | 470.166964 | **478.237340** | 6.284065 |
| RNN＋相同属性组合 | 486.398628 | 469.853216 | **478.125922** | 6.395482 |
| LSTM＋相同属性组合 | 486.337890 | 469.919411 | **478.128651** | 6.392753 |
| 队友 CatBoost 固定参考 | 492.874313 | 476.168495 | 484.521404 | — |

参考来自树分支提交 `3dae63c`，本轮未重训队友模型。其种子 42、额外距离/COE 特征及环境与本轮不同，不能把方案分数差异直接归因于算法。
**最佳单 carry=485.201787、单 Transformer=485.629277，尚未胜过 CatBoost；上表胜出的是同类平均版本。**
RNN/LSTM 相对同属性 carry 仅改善约 0.11 SGD，彼此差约 0.003 SGD，不支持时间网络显著优于 carry 或彼此显著优劣的结论。

公共 8 项＋个人 12 项，共 **20/20 测试再次通过**；清洗检查再次 `All checks passed.`。
四个完整运行各有 82329 条验证及 50000 条测试预测，列/行号/日期/标签/Id 全部核对，预测有限、正值、原租金尺度。
重算 RMSE 与 config/log 一致；独立重训与搜索的逐行预测最大差 **4.547473508864641e-13 SGD**。
carry/RNN/LSTM 的属性训练及验证/测试比例预测一致，最大比例差 **4.440892098500626e-16**。
近期窗口实际样本数由原 fold 日期复算；17 个数据 CSV 的哈希未变，公共文件未作个人适配修改，无权重或锁文件交付。

最新入口：

- [详细调参报告](develop_tuning_yuxin_nn.md)：搜索范围、条件最优边界、每个成员的精确配置入口、实际 epoch、耗时及局限。
- [冻结参数](../notebooks/Yuxin/tuning/configs/)、[四方案比较](../notebooks/Yuxin/tuning/comparison.csv)、[最终逐行复核](../notebooks/Yuxin/tuning/final_verification.json)。
- [正式三个候选](../results/Yuxin/README.md)：Transformer/RNN/LSTM，恰好 3 个运行。
- `notebooks/Yuxin/tuning/final/` 保留全部四类完整输出，含 carry 对照；`tuning/pre_replay/` 保留补跑前的两个时间赢家。
- 初验三个正式候选及原始分析文件保留在 `notebooks/Yuxin/archive/stage1_3/`，可恢复，未删除。

长期复核无需临时搜索缓存：

```bash
export LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python notebooks/Yuxin/verify_delivery.py \
  --roots notebooks/Yuxin/tuning/final results/Yuxin \
  --summary-output notebooks/Yuxin/tuning/recheck_verification.json
```

外层两折已经参与调参，存在选择偏差且区间重叠；以上不是无偏隐藏集估计，不证明全局最优，也不保证 Kaggle 排名。
本轮未实施时间 Transformer、跨家族融合实验、Kaggle 上传或远端推送；最终交付按仓库要求本地提交 Git。

## 10. 步骤 4—6：完成与复核

本次按用户新的授权继续执行步骤 4—6，之前“不进入步骤 4”的限制不再适用于本轮。
**三个步骤已实际执行；时间 Transformer 完成两折验证及全量训练/test 预测，互补性通过实际融合计算检验，三个正式运行的配方与训练轮数方案已锁定。**
没有为追求结果继续无界搜索，也没有把未优于 carry 的时间 attention 写成成功提升。

### 10.1 步骤 4：同窗口时间编码器

新增真正的 temporal Transformer：输入为全市历史月中位租金 `[B,L,1]`，不是房屋交易行或表格字段序列。
模型为 1 层、2 头、FFN=2×d_model、GELU、LayerNorm，使用**窗口相对正弦位置编码**与上三角因果 mask；不是按真实年月类别学习，也不声称已学习年度季节性。
回归头零初始化，从 carry 的零增量起步；训练预测一步增量，推理只把自身预测填入历史，再递归至最远 16 步。
缩放、内部早停及市场聚合都只看当前训练前缀；不读取未来租金、未来 COE/股价或测试分布中的目标统计。

先固定步骤 1—3 已选的相同 4 成员 MLP 属性组合，在 L=6、hidden=8、最多 80 epoch、lr=0.005、dropout=0.1、weight_decay=0.01、inner=6、patience=25、damping=0.8、strength=0.25 下替换市场编码器。
仅编码器类型不同；carry/阻尼趋势没有神经训练，参数量也不可能与三种网络完全相等，因此是相同输入/窗口/训练规则对照，不是等参数量实验。

| 同设置受控实验 | A RMSE | B RMSE | 两折均值 |
|---|---:|---:|---:|
| carry | 486.307715 | 470.166964 | **478.237340** |
| 简单阻尼趋势 | 486.535517 | 469.986879 | 478.261198 |
| RNN（L=6） | 487.161576 | 469.990521 | 478.576048 |
| LSTM（L=6） | 487.182244 | 469.964375 | 478.573310 |
| temporal Transformer（L=6） | 487.036876 | 469.954575 | 478.495726 |

这些是复用 carry 运行中**已拟合属性模型的 OOF 比例**、独立重训市场编码器的廉价受控实验，不是重复训练 12 组属性网络。来源 commit/run_id/预测哈希与逐折市场审计保存在 [受控实验记录](../notebooks/Yuxin/stage4/controlled_comparison.json)。
RNN/LSTM 的本表 L=6 配方不是第 9 节各自已调优的 L=12 配方，不能混为同一运行或声称 attention 优于调优后的 recurrence。

随后只对 Transformer 做有限单因素消融：从上述匹配配置出发，分别试 L=3、d=16、lr=0.001、damping=0.95、strength=0.5、dropout=0.2、inner=12。共 **8 组 attention / 12 个总候选**，每组都完成两折，没有剪枝缺折；完整分数见 [comparison.csv](../notebooks/Yuxin/stage4/comparison.csv)。
最好 attention 是 **inner=12，其余保持匹配设置**，均值 478.323802；这是已试配方中的最优，不是全局最优。
它比同属性 carry 高 0.086463 SGD，也没有优于此前调优的 RNN/LSTM，因此不晋升正式候选。

为排除缓存组装掩盖问题，使用冻结 [market_transformer.json](../notebooks/Yuxin/stage4/configs/market_transformer.json)，经未修改的公共框架独立重训全部属性成员及市场模型：

| 项目 | 独立重训结果 |
|---|---|
| run_id | `20261103T190543556826Z_a02818be` |
| A / B / 均值 | 486.665915 / 469.981690 / **478.323802** |
| 拟合可用行数 | 102829 / 114842 / 150000；近期属性成员仍按已声明 24 月窗口筛选 |
| 市场训练月份 / 窗口数 | 35/29、39/33、51/45；L=6 |
| 市场内部选定 epoch（A/B/final） | **17 / 9 / 4**，选定后从头拟合当前全部市场历史 |
| 内层训练截止（A/B/final） | 2022-11 / 2023-03 / 2024-03；之后 12 月只用于内部早停 |
| 最终市场预测 | 2025-03=3200；2026-07=3206.921154 SGD；未来数值是预测，不是实测 |
| 验证 / test | 82329 / 50000 行，原行号与 Id 完整 |
| 耗时 | 274.107 秒，CPU、属性成员 4 进程各单线程；非无竞争性能基准 |
| 与缓存受控预测最大绝对差 | **1.3642420526593924e-12 SGD** |

属性成员训练曲线/审计与 carry/RNN/LSTM 一致；两折验证及最终 test 的属性比例最大差为 **4.440892098500626e-16**。
完整输出在 `notebooks/Yuxin/stage4/full/`，最终证明见 [stage4/verification.json](../notebooks/Yuxin/stage4/verification.json)。本轮没有增加多市场池化/距离/COE：主线已经能检验时间编码器，月度样本很少且细分市场稀疏，未额外扩展该实验范围。

### 10.2 步骤 5：统一误差与残差相关性

[evaluate_complementarity.py](../notebooks/Yuxin/evaluate_complementarity.py) 从正式提交产物读取：本分支三个神经网络、FXY Ridge（`6e7ffcb`）、Yaqi 三个树（`3dae63c`）。
逐个验证 rolling/final 条件，按 `(fold,row_index)` 一对一对齐，再逐行核对日期和真实标签；读取完整 test 并核对 50000 个 Id。没有读取队友的训练拟合预测，也没有重训其模型。
另外将 carry 与未晋升的时间 Transformer 纳入误差/相关性分析，**不将这两个开发对照作为团队融合输入**。

输出覆盖 9 个基础/对照模型及 12 个固定平均组合：

- [group_errors.csv](../notebooks/Yuxin/stage5/group_errors.csv)：**2877 行**，逐月、URA 大区、town、户型、单月 horizon、4 月 horizon_bin，共 6 个维度；保留每组 N、RMSE、MAE、bias（预测−真实，SGD）。
- [fold_scores.csv](../notebooks/Yuxin/stage5/fold_scores.csv)：逐模型逐折，不把两折重叠样本混为一个总体分数。
- [residual_correlations.csv](../notebooks/Yuxin/stage5/residual_correlations.csv)：9 个模型两两、两折分别计算，**72 条 Pearson 残差相关性**。
- [verification.json](../notebooks/Yuxin/stage5/verification.json)：来源完整 commit/run_id、SHA-256、行号与目标核对、范围及局限。

以 RNN 为例：

| 误差关系 | A | B |
|---|---:|---:|
| 与 LSTM 残差相关性 | 0.999956 | 0.999940 |
| 与 CatBoost 残差相关性 | 0.974515 | 0.978442 |
| 与 Ridge 残差相关性 | 0.987526 | 0.987665 |
| central region RMSE / N | 519.132 / 12103 | 504.542 / 9038 |
| north-east region RMSE / N | 433.848 / 11173 | 415.410 / 8308 |
| executive RMSE / N | 590.648 / 2392 | 571.426 / 1778 |

executive 的偏差约 −100.72 / −97.57 SGD，提示有系统低估；1-room 仅 18 / 14 行，不据其分数作稳定优劣结论。
A 的 horizon 1—4 月 RMSE=531.007，13—16 月=450.296；这是不同日历月份与样本构成，不应解释为“预测越远越容易”的因果规律。
RNN/LSTM 高度冗余，相关性低一些也不自动等于融合一定更好，因此继续做实际预测平均。
已审阅的固定分支中没有 KNN 正式预测，不能填造 KNN 实现、成绩或相关性；不保证他人未推送或之后的新分支也没有。

### 10.3 步骤 5：实际融合，而不只看相关性

将三个正式神经网络分别与 Ridge/CatBoost/XGBoost/LightGBM **固定 50:50 平均**，不在这些评估行拟合权重。全部 12 组均保留在 [fixed_blends.csv](../notebooks/Yuxin/stage5/fixed_blends.csv)，其中 10 组均值低于更好的单个组成模型，2 组没有改善。
代表性结果如下，完整表还包含其余组合：

| 固定平均 | A | B | 平均 | 比较好的组成模型低 |
|---|---:|---:|---:|---:|
| RNN＋CatBoost | 485.449732 | 469.654270 | **477.552001** | 0.573921 |
| RNN＋Ridge | 485.919136 | 469.232753 | 477.575945 | 0.549978 |
| RNN＋XGBoost | 486.392499 | 468.993053 | 477.692776 | 0.433146 |
| 表格 Transformer＋CatBoost | 486.544962 | 470.784778 | 478.664870 | 2.664174 |
| RNN＋LightGBM | 488.449107 | 470.544833 | 479.496970 | **−1.371048（退化）** |

因此只能说：**这些开发折上，特定神经网络与队友模型具有实测互补收益，并非所有平均都改善。**
保存了 RNN＋CatBoost 固定平均的实际 [验证 CSV](../notebooks/Yuxin/stage5/diagnostic_blend_validation.csv) 与 [test CSV](../notebooks/Yuxin/stage5/diagnostic_blend_test.csv)，分别 82329 / 50000 行。
它们是诊断产物，不是第四个正式运行；未用新 blend 分数改写任何基础模型的原分数。

另做前向权重迁移：只用 A 折 **2023-12—2024-03 的 12013 行**，解析求一个凸权重 w∈[0,1]，不加截距；然后固定 w，评估 B 折 **2024-04—2025-03 的 35158 行**。
例如 RNN＋CatBoost 的 RNN 权重为 0.516588，B RMSE=469.552473，低于 B 上 RNN 的 469.853216；RNN＋XGBoost 权重 0.670898，B RMSE=468.432592。
全部 12 组见 [forward_transfer.csv](../notebooks/Yuxin/stage5/forward_transfer.csv)，没有用 B 标签求权重，也没有将 A 的未来验证月份反过来校准 B。

**重要边界：**固定平均不拟合权重，前向权重也不读取其评估月份标签；但底层模型此前的超参数已经用外层两折选过，仍存在选择偏差。
这不是完全嵌套、全新未见标签的泛化实验；同时比较多个组合也有挑选效应。两折时间重叠、队友种子/特征/环境不同，不宣称统计显著或算法因果优劣，不能保证 Kaggle 隐藏集提升。

### 10.4 步骤 6：正式候选、配置和轮数锁定

正式 `results/Yuxin/` **仍恰好 3 个运行**；未覆盖原 config/验证/test，未新增第四个运行，既有历史归档保持原样。
本次新建 [selection.json](../results/Yuxin/selection.json)，逐文件锁定 SHA-256、冻结 JSON 配方、实际训练轮数和精确可恢复的原训练源码：

| 正式候选 | run_id | 配方 | 原训练源码 |
|---|---|---|---|
| RNN＋4 MLP | `20261103T180702097497Z_0c01ed2c` | `tuning/configs/market_rnn.json` | `55d2acb` |
| LSTM＋相同属性 | `20261103T180702083101Z_b8e1e8c0` | `tuning/configs/market_lstm.json` | `55d2acb` |
| 表格 Transformer×4 | `20261103T181754695021Z_493b5f46` | `tuning/configs/tabular_transformer.json` | `aba5078` |

这里的源码提交不是随意填写的当前 HEAD：RNN/LSTM 使用的是增加进程并行支持前的 wrapper；锁定脚本将运行保存的每个模型源文件哈希与 Git blob 全量匹配，确认对应原提交。
原配方与对应运行的 recipe SHA 完全一致；新时间模型另行记录新源码哈希，不冒充旧运行的原始源码。

固定轮数**方案**是“冻结最大 epoch、patience、内层月份及其他超参数 → 只在当前训练前缀选择 epoch → 从头重训全折”；不拿外层验证选 epoch，也不要求把所有原点硬设为相同轮数。已完成运行的实际轮数为：

| 组件 | A | B | final |
|---|---|---|---|
| 共享 4 MLP 属性成员 | [11,8,13,6] | [11,5,18,3] | [31,10,10,5] |
| RNN 市场模块 | 14 | 27 | 3 |
| LSTM 市场模块 | 34 | 10 | 6 |
| 4 个表格 Transformer 成员 | [8,18,15,11] | [8,22,11,11] | [19,19,22,24] |

保留这三个是负责家族的比较交付，不声称它们组成全局最优三模型融合；RNN/LSTM 冗余度很高，不建议把它们当独立信息源重复加权。
未优于 carry 的短窗口 RNN/LSTM、阻尼趋势、时间 Transformer、以及未改善的融合组合全部保留实验说明。
carry 和新时间模型开发输出不占正式名额；诊断 blend 也不新增名额。没有上传 Kaggle 或推送远端。

### 10.5 最终复核与复现入口

本轮使用数据验证技能复核指标、连接、重叠窗口及权重的时间边界；它要求保留未改善组合，并将“算术验证通过”与“无偏泛化保证”分开。

| 验收项 | 结果 |
|---|---|
| 公共＋个人测试 | **26/26 通过**：8 公共、14 NN、4 分析测试 |
| 新增模型边界 | clone、未知/缺失、相同种子、验证标签扰动、未来查询顺序/分批、因果 mask 与零增量起步均通过 |
| 清洗检查 | `All checks passed.`；17 个数据 CSV 的哈希未变 |
| 新时间模型完整交付格式 | A=47171、B=35158，test=50000；行号/日期/标签/Id 原样、有限正值 SGD |
| 12 个受控候选 | 两折全部完成，并从已保存市场曲线/属性预测重新核算分数 |
| 独立重训 | 新时间模型与缓存逐行差≤1.37e-12 SGD，市场模型共享属性训练与预测一致 |
| 分组/融合算术 | 每个维度 N 与原折一致；分组加权平方误差恢复全折 MSE；用残差内积恒等式独立重算全部 12 个固定平均 |
| 残差相关性/权重 | 72 个相关性独立中心化内积复算；12 个前向权重/评估分数复算，校准日期严格早于评估日期 |
| 保存的诊断验证/test | 独立按 key 对齐并重算固定平均，原行号与 Id 完整 |
| 配置与来源 | 三正式配方/epoch 方案锁定；运行源码能按完整哈希从固定 Git 提交恢复 |
| 公共文件边界 | 未改框架、公共清洗/特征、共享日志、公共 requirements 或 `.gitignore`；根 README 按用户要求补充 |
| 仓库产物 | 正式仍 3 运行，无权重/锁文件；旧结果未删除或覆盖，全部新实现与产物本地提交 Git |

最终证明：[stage4/verification.json](../notebooks/Yuxin/stage4/verification.json)、[stage5/independent_verification.json](../notebooks/Yuxin/stage5/independent_verification.json)、[正式选择清单](../results/Yuxin/selection.json)。

在根目录复核现有完整结果（不重新训练，不需临时 HPO 缓存）：

```bash
export LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.data_cleaning.checks
.venv/bin/python notebooks/Yuxin/evaluate_complementarity.py
.venv/bin/python notebooks/Yuxin/finalize_stages.py
```

重新训练时间 Transformer 或正式候选时，务必使用新输出目录，命令见 [个人 README](../notebooks/Yuxin/README.md)。
旧 `tune_models.py` / `finalize_tuning.py --prepare` 的源码签名绑定历史研究；新时间模块加入后，不应直接续跑旧 SQLite 或取消哈希保护。冻结配置重训与长期 CSV 审计不依赖旧 `/tmp` 缓存。
