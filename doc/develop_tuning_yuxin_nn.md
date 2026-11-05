# Yuxin：步骤 1—3 调参及验收

日期：2026-11-04；分支：`yuxin_time_series`。本轮仅调优已有表格 Transformer、carry、市场 RNN/LSTM，不实施 `doc/develop_yuxin_nn.md` §8.1 步骤 4 的时间 Transformer。

状态：四类模型搜索、独立重训及完整输出复核均已通过；三个正式候选已更新，旧结果可恢复地归档。未实施步骤 4、未上传 Kaggle、未推送远端。

## 方法与比较边界

不使用 QLoRA。[QLoRA 论文](https://arxiv.org/abs/2305.14314)讨论量化大语言模型的低显存微调，不是本项目小型、从头训练网络的超参数搜索方法。
本轮采用 [Optuna TPE](https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html)，结合人工暖启动/局部候选、首折 median pruning、训练部分内部时间早停。日志逐条区分 queued、TPE、COMPLETE、PRUNED、FAIL；剪枝候选只有首折分数，不能当作完整两折结果。

主要目标是队友已交付 CatBoost 的两折平均 **484.521404**，而非较弱的 HistGradientBoosting 489.240895。
参考来自远程树分支提交 `3dae63c3ba9e5919fadb9af54474fee20d86f751` 的 `results/Yaqi/selected_models.csv` 及对应 config；CatBoost 两折为 492.874313、476.168495。本轮没有重训或修改队友模型。
两者使用相同公共时间折，但队友还有距离/COE 特征，种子为 42、依赖版本不同；本分支仅用 basic/block，固定种子 5228。因此只比较已交付方案的开发分数，不做算法优劣的因果结论。

所有标签、词表、标准化、近期训练窗口和市场聚合只使用相应训练部分。RNN/LSTM 只处理全市月度市场序列，不构造单套房的虚假滞后租金。
表格 Transformer 本轮学习训练月市场中位数归一化后的租金比例，预测时乘以训练截止月市场水平；这是表格特征模型，不是时间编码器。
因此调优后的属性输入去掉 RENT_MONTH/MONTH_INDEX：9 个数值字段、8 个类别字段；Transformer 为 18 个 token（含 CLS），而非初始原租金版本的 20 个。
损失的 market 加权采用训练月中位数平方，使比例 MSE 更接近原租金尺度 MSE；验证时始终使用固定原点预测，不使用真实未来市场水平。

公共 A/B 折用于搜索和选择，因此这里的成绩是**开发集成绩**，不是无偏测试估计。两个验证区间重叠，平均为各折 RMSE 的简单平均，不拼接成独立样本算总分。未上传 Kaggle，隐藏集成绩未知，有限搜索不证明全局最优。

## 搜索范围

- carry 属性 MLP：hidden 64/128/256；embedding 自适应/8/16；dropout 0/0.05/0.1；batch 512/1024/2048；学习率 0.0005—0.004；weight decay 0.000001—0.02；类别频数阈值 1/3/5/10；内部验证末端 3/6 月；训练窗口全部/18/24/36 月；uniform/market 损失。最多 60 epoch、patience 7。
- 表格 Transformer：d_model 16/32、4 heads、1 layer；其余预处理/正则搜索维度同上，最多 24 epoch。人工初始候选和 wider 局部候选均如实记录，不将未使用的维度写成已搜索。
- 时间 RNN/LSTM：lookback 3/6/12；hidden 8/16/32；最多 80/160/240 epoch；学习率 0.001—0.01；阻尼 0.1/0.3/0.5/0.8/0.95；预测增量强度 0.1/0.25/0.5/1；内部末端 6/12 月；dropout 0/0.1/0.2；weight decay 0.000001—0.01。时间网络仍为一层、patience 25。

carry 单配置调优与时间网络调优分阶段进行。冻结获胜属性模型后，RNN/LSTM 才共享其相同逐行比例预测，并独立搜索时间参数；缓存按模型源码、数据哈希和属性配置身份校验，不跨训练截止点复用。

单配置提升不足时，增加一个明确标记的稳定化选项：**同类网络多配置等权平均**。仅在同一模型家族内搜索最多 4 个成员，比较 top-K 与贪心前向选择路径，不混入树/Ridge/KNN 预测。
carry/RNN/LSTM 共用同一 MLP 属性组合；Transformer 仅组合表格 Transformer。单配置最优与平均版本分别汇报，不将集成成绩归于单个网络。

## 证据与复现入口

- 搜索入口：`notebooks/Yuxin/tune_models.py`；同类选择：`select_ensembles.py`。
- 逐候选参数、分折分数、状态、耗时及审计：`notebooks/Yuxin/tuning/*_trials.json`、对应 CSV。
- 最佳单配置：`best_carry.json`、`best_transformer.json`；选中同类属性组合：`selected_carry.json`、`selected_transformer.json`；时间参数：`best_rnn.json`、`best_lstm.json`。
- 最终可重训配置：`notebooks/Yuxin/tuning/configs/`。每个成员的参数都完整保存，不依赖临时 SQLite 或模型权重。
- 独立重训与复核：`notebooks/Yuxin/finalize_tuning.py`、`run_experiments.py`、`verify_delivery.py`。

环境沿用 Python 3.12.7、NumPy 1.26.4、Pandas 2.2.2、sklearn 1.5.1、torch 2.15.0，新增 Optuna 4.9.0，仅装在个人 `.venv`。
CPU、torch 单线程、确定性算法。启动前设置 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`，runner 仅控制 BLAS，避免 Conda/PyTorch OpenMP 冲突。
Transformer 最终重训为 4 个独立进程并行成员，每成员仍单线程；不使用共享全局 RNG 的线程并行。新增串行/并行逐值等价性测试后共 20 项测试通过。
串行模型源码快照已在本地提交 `55d2acb`，后续并行实现保留 `n_jobs=1` 默认值；每个实际运行另存完整模型源码哈希，便于区分两个版本。
初次并行建 SQLite 发生初始化竞态，另一个启动发生 OpenMP abort；均重启处理。中断 trial 明确保留为 FAIL，而非成功结果；原始 carry 配置重训准确复现 485.610900。

选中配置可直接完整重训；若从空数据库重新执行搜索，人工追加候选与恢复顺序会影响 TPE 提议轨迹，因此不声称仅靠 seed 就能复现整个搜索顺序。逐候选日志和冻结配置才是本轮的精确配方依据。

## 已冻结的属性与时间参数

carry 单配置共 25 条记录：17 COMPLETE、7 PRUNED、1 FAIL。最佳完整单配置为 trial 20，A/B=492.579032/477.824541，平均 **485.201787**，仍未超过 CatBoost。
同类选择在 17 个完整候选上比较 top-K 与贪心前向路径（45 次加成员比较），选中 trial **20、7、15、4**，各占 1/4；A/B=486.307715/470.166964，平均 **478.237340**。
这是多配置平均的真实收益，不是单配置成绩。

| 属性成员 trial | hidden | embedding | dropout | batch | 学习率 | weight decay | 最低频数 | 训练月数 | 内部验证月数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 256 | 16 | 0.1 | 1024 | 0.0006728941 | 0.0000528306 | 3 | 全部 | 3 |
| 7 | 256 | 16 | 0 | 512 | 0.0037928122 | 0.0137838362 | 1 | 全部 | 3 |
| 15 | 128 | 8 | 0.05 | 512 | 0.003 | 0.001 | 10 | 全部 | 3 |
| 4 | 128 | 8 | 0 | 1024 | 0.002 | 0.0001 | 3 | 24 | 6 |

四个成员均采用比例目标、uniform 损失、最多 60 epoch、patience 7、seed 5228；表中浮点显示值经过截断，精确值以冻结 JSON 为准。
公共最终训练会接收全部 150000 行，但 trial 4 按已声明的超参数只拟合最后 24 月；不能说所有成员都使用全部训练行。

独立重训中，四个属性成员的实际早停轮数和最终样本量如下；每次都重新初始化、重新拟合训练统计，未把上一折权重带到下一折。

| trial | A epoch | B epoch | 最终 epoch | 最终实际拟合行数 |
|---|---:|---:|---:|---:|
| 20 | 11 | 11 | 31 | 150000 |
| 7 | 8 | 5 | 10 | 150000 |
| 15 | 13 | 18 | 10 | 150000 |
| 4 | 6 | 3 | 5 | 71040（2023-04—2025-03） |

| 时间网络 | trial | lookback | hidden | 最大 epoch | 学习率 | dropout | weight decay | 阻尼 | 增量强度 | 内部验证月数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RNN | 44 | 12 | 8 | 80 | 0.0062077303 | 0.2 | 0.0000931232 | 0.95 | 0.25 | 12 |
| LSTM | 34 | 12 | 8 | 80 | 0.0096709795 | 0.2 | 0.0002451224 | 0.8 | 0.25 | 12 |

首轮 RNN 32 条候选中 15 COMPLETE、17 PRUNED；LSTM 为 16 COMPLETE、16 PRUNED，没有把剪枝首折分数当作两折均值。
由于时间分支廉价，又将全部剪枝参数禁用剪枝后重放：**RNN/LSTM 各 32 组不同参数最终均有完整两折分数**，累计日志分别为 49 条（32 COMPLETE、17 历史 PRUNED）、48 条（32 COMPLETE、16 历史 PRUNED）。
补跑确实更换了赢家：RNN 从 478.142491 降到 **478.125922**；LSTM 从 478.237498 降到 **478.128651**。重放条目记录 `replay_of`，不是伪称多搜索了 33 组新参数。
这是冻结属性组合下的条件最优，不是属性参数和时间参数的联合全局最优。

相同属性组合下，RNN/LSTM 相对 carry 只分别改善约 0.11142/0.10869 SGD，彼此仅相差约 0.00273 SGD。
这些小差异没有独立重复种子/时间留出的显著性证据，不能宣称时间网络明显优于 carry 或 RNN 显著优于 LSTM。主要收益仍来自目标表达、属性模型调参与同类平均；最佳时间参数使用 0.25 的增量收缩。

RNN 的 A/B/全量时间轮数为 **14/27/3**，LSTM 为 **34/10/6**。最终全市市场原点 2025-03 为 3200，预测到 2026-07：carry=3200、RNN≈3242.388、LSTM≈3225.498；这些是预测而非未来真实市场数据。
上一版时间参数的两个完整重训结果保留在 `notebooks/Yuxin/tuning/pre_replay/`，未覆盖或计入最终四类结果。

## 表格 Transformer 的冻结参数

共 8 条候选：6 COMPLETE、2 PRUNED；6 个暖启动/局部候选、2 个 TPE 候选。最佳单网络为 trial 4，A/B=493.396493/477.862062，平均 **485.629277**，仍略弱于 CatBoost。
在完整候选上比较同类组合后选择 trial **4、2、1、5**，各占 1/4；搜索预测 A/B=487.952332/474.705755，平均 **481.329044**。

| trial | d_model | dropout | batch | 学习率 / weight decay | 最低频数 | 训练 / 内部（月） | loss |
|---|---:|---:|---:|---|---:|---|---|
| 4 | 32 | 0 | 1024 | 0.002 / 0.01 | 5 | 全部 / 3 | uniform |
| 2 | 16 | 0.05 | 512 | 0.001 / 0.001 | 5 | 36 / 3 | uniform |
| 1 | 16 | 0.05 | 1024 | 0.0015 / 0.001 | 1 | 全部 / 6 | market |
| 5 | 32 | 0.05 | 1024 | 0.0015 / 0.001 | 1 | 全部 / 3 | market |

全部为 4 heads、1 layer、最多 24 epoch、patience 7、seed 5228、18 个 token（9 数值＋8 类别＋CLS）。最终参数文件额外记录 `n_jobs=4`；该计算并行度不改变每个成员的模型参数或训练样本。

| trial | A epoch | B epoch | 最终 epoch | 最终拟合行数 |
|---|---:|---:|---:|---:|
| 4 | 8 | 8 | 19 | 150000 |
| 2 | 18 | 22 | 19 | 103244（最近 36 月） |
| 1 | 15 | 11 | 22 | 150000 |
| 5 | 11 | 11 | 24 | 150000 |

## 最终独立重训结果与验收

下列均从保存的逐行 CSV 重算，原尺度 SGD RMSE，越低越好。所有“本轮”模型采用上述同类平均属性分支，不是单网络成绩。

| 方案 | A | B | 两折平均 | 比 CatBoost 低 | 完整重训耗时/秒 |
|---|---:|---:|---:|---:|---:|
| 本轮 RNN＋4 个 MLP 属性成员 | 486.398628 | 469.853216 | **478.125922** | 6.395482 | 527.7 |
| 本轮 LSTM＋相同属性成员 | 486.337890 | 469.919411 | **478.128651** | 6.392753 | 525.5 |
| 本轮 carry＋相同属性成员 | 486.307715 | 470.166964 | **478.237340** | 6.284065 | 721.1 |
| 本轮 4 个表格 Transformer 平均 | 487.952332 | 474.705755 | **481.329044** | 3.192360 | 1810.7 |
| 队友 CatBoost 参考 | 492.874313 | 476.168495 | 484.521404 | — | 本轮未重训 |

本轮四个方案开发分数均低于固定参考；改善约 0.66%—1.32%。**最佳单 carry=485.201787、单 Transformer=485.629277，均尚未胜过 CatBoost；胜出的是明确标记的同类平均版本。**
耗时包含两折及最终训练/测试预测，存在 CPU 争用且 Transformer 使用多进程，不能作为等资源算法速度比较。

共保留 130 条 Optuna 记录：87 COMPLETE、42 PRUNED、1 FAIL；其中含 33 次时间候选重放，不是 130 组不同参数。
本轮实际评估 96 组不同配置；carry/Transformer 尚有 9 组只完成首折的剪枝候选，未用于两折最优或同类成员选择。

验收证据为 [最终逐行复核](../notebooks/Yuxin/tuning/final_verification.json)、[四个完整运行复核](../notebooks/Yuxin/tuning/final/verification.json)、[正式候选复核](../results/Yuxin/verification.json)。

- 20/20 测试通过，含公共契约、标签扰动、缺失/未知类别、clone、同种子复现、固定原点及串并行等价性；公共清洗检查再次 `All checks passed.`。
- 每运行验证文件 82329 行（A=47171、B=35158），列、原行号、日期、真实标签逐行匹配，键唯一；RMSE 与 config/log 一致，容差 1e-8。
- 每运行测试文件 50000 行 `Id,Predicted`，Id 与 loader 及 0..49999 顺序完全一致；全部预测有限、正值、原租金尺度。
- 四类独立重训与搜索预测逐行比对，最大差 **4.547473508864641e-13 SGD**。
- carry/RNN/LSTM 各成员的训练曲线、统计审计和验证/测试比例预测一致，比例最大差 **4.440892098500626e-16**；近期窗口的实际行数也由原 fold 日期独立重算。
- 17 个原始/清洗 CSV 的 SHA-256 未变；公共清洗、特征、CV、实验框架、共享日志、公共 requirements 和 `.gitignore` 未修改。没有权重或锁文件交付。

## 结果位置与复现

| 模型 | run_id | 位置 |
|---|---|---|
| Transformer | 20261103T181754695021Z_493b5f46 | `results/Yuxin/` |
| RNN | 20261103T180702097497Z_0c01ed2c | `results/Yuxin/` |
| LSTM | 20261103T180702083101Z_b8e1e8c0 | `results/Yuxin/` |
| carry 对照 | 20261103T175001571910Z_a7a4588e | `notebooks/Yuxin/tuning/final/` |

`tuning/final/` 同时保存全部四个完整运行，正式目录只复制三个网络候选，符合每人最多 3 个运行的团队规范，不代表多训练了三个模型。
原始三个正式候选和初验比较/误差/复核文件保存在 `notebooks/Yuxin/archive/stage1_3/`；没有覆盖初验数据。
冻结参数在 [配置目录](../notebooks/Yuxin/tuning/configs/)，最新四模型比较在 [comparison.csv](../notebooks/Yuxin/tuning/comparison.csv)。完整候选日志和旧时间赢家均保留。

在项目根目录复核现有交付，不依赖临时搜索数据库：

```bash
export LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.data_cleaning.checks
.venv/bin/python notebooks/Yuxin/verify_delivery.py \
  --roots notebooks/Yuxin/tuning/final results/Yuxin \
  --summary-output notebooks/Yuxin/tuning/recheck_verification.json
```

重新训练用新的开发输出目录，不追加到已有正式目录：

```bash
.venv/bin/python notebooks/Yuxin/run_experiments.py \
  --model market_carry tabular_transformer market_rnn market_lstm \
  --config-dir notebooks/Yuxin/tuning/configs --split rolling --final \
  --output /tmp/yuxin_tuned_reproduction_20261104
```

本轮仍没有独立隐藏集分数，也没有完成团队跨家族融合验证。后续是否进入时间 Transformer，等待用户决定。
