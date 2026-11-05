# Yuxin 神经网络：开发阶段 1—6

实现说明和真实运行结果见`doc/develop_test_yuxin_nn.md`；原始设计见 `doc/develop_yuxin_nn.md`。
最新调参报告为 `doc/develop_tuning_yuxin_nn.md`，最新冻结参数为 `tuning/configs/`。
`configs/` 保留初验配置；不带最新 `--config-dir` 的旧命令不会运行本轮优化组合。
公共数据/模型/验证接口以 `docs/experiment-interface.md` 为准。

本目录使用公共 `basic`、`block` 特征，不读取额外宏观或设施数据。
`src/models/yuxin_nn/` 提供 embedding MLP、表格特征 Transformer，以及共享属性 MLP 的市场 carry/RNN/LSTM/temporal Transformer。
RNN/LSTM 的序列单位是全市月度租金中位数，不是单套房或交易行。
时间 Transformer 已在步骤 4 实现；多市场池化、距离/COE 消融没有加入本轮，不能写成已实现能力。

## 环境

项目根目录运行，使用 Python 3.10+，本轮实际采用 Python 3.12、CPU：

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r notebooks/Yuxin/requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.data_cleaning.checks
```

如果希望完全独立环境，可去掉 `--system-site-packages` 再安装 requirements。
本机 Conda 与 torch 同进程混用 OpenMP 时出现过线程屏障停滞，因此默认 torch 单线程 CPU，runner 只限制 BLAS，不统一修改 OpenMP。
不要在公共配置中改种子；本目录默认公共 `SEED=5228`，torch 同样确定性设置。

## 开发运行

```bash
# --smoke 只有两轮训练，用于真实数据管线检查，不是定稿配置
.venv/bin/python notebooks/Yuxin/run_experiments.py --model mlp --smoke

# 默认输出到新临时目录，不修改共享 experiments/log.csv
.venv/bin/python notebooks/Yuxin/run_experiments.py --model mlp --split holdout --verbose
.venv/bin/python notebooks/Yuxin/run_experiments.py --model market_carry --split holdout
.venv/bin/python notebooks/Yuxin/run_experiments.py --model tabular_transformer --split rolling --final
```

参数从 `configs/<model>.json` 读取。模型在当前训练折末端内部月份选择 epoch，再用该折全部训练数据从头拟合。
`training_audit.json` 保存各次拟合截止点、实际轮数、训练曲线、词表大小和未来市场预测。
不保存网络权重，运行可从配置重新训练。

正式结果已经存在时不要再执行下面的交付命令，以免重复运行；新实验另选 `--output`。
初验时首次交付的完整命令为（最新参数见下方调参节）：

```bash
.venv/bin/python notebooks/Yuxin/run_experiments.py \
  --model tabular_transformer market_rnn market_lstm \
  --split rolling --final --deliver --verbose
```

`results/Yuxin/` 最多三个运行。`validation/` 为初验公共基线、MLP 和 carry 对照，不计入融合模型交付，不含权重。
每个模型都输出两折逐行验证预测及完整 50000 条 test 预测；市场预测原点固定，不使用未来租金或未来实测经济变量。

## 复核与结果入口

```bash
.venv/bin/python notebooks/Yuxin/verify_delivery.py \
  --roots notebooks/Yuxin/validation notebooks/Yuxin/archive/stage1_3/results_Yuxin \
          notebooks/Yuxin/tuning/pre_replay notebooks/Yuxin/tuning/final
.venv/bin/python notebooks/Yuxin/analyze_results.py \
  --roots notebooks/Yuxin/validation notebooks/Yuxin/archive/stage1_3/results_Yuxin \
          notebooks/Yuxin/tuning/pre_replay notebooks/Yuxin/tuning/final
```

- `comparison.csv`：从预测重算的 A/B 和两折简单平均 RMSE。
- `monthly_errors.csv`：各模型、各折、每月 RMSE 和偏差。
- `verification.json`：完整行号/日期/标签/分数/编号/信息边界复核。
- `results/Yuxin/verification.json`：三个候选的独立交付复核。
- `training_audit.json` 中的未来市场曲线为固定原点递归预测，不是真实未来租金。

比较神经网络与树模型时请使用同一外层切分；不要把内部早停分数作为最终成绩，也不要把 A/B 重叠验证行作为独立样本合并计算总体 RMSE。

## 步骤 1—3 的调参

`tune_models.py` 使用 Optuna TPE 和首折 median pruning，原始配置保持不变。
先搜索 carry 的属性回归器，再固定该属性模型，分别搜索 RNN/LSTM；Transformer 单独搜索。
该历史调参阶段使用公共两折，不包含时间 Transformer；后者见下方步骤 4—6。QLoRA 是大语言模型的低显存微调方法，不是本任务的超参数搜索器。

本机混合 Conda/PyTorch 环境需要在启动进程前限制线程；以下命令均在项目根目录运行：

```bash
export LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python notebooks/Yuxin/tune_models.py --phase carry --trials 25
.venv/bin/python notebooks/Yuxin/select_ensembles.py --phase carry
.venv/bin/python notebooks/Yuxin/tune_models.py --phase transformer --trials 8
.venv/bin/python notebooks/Yuxin/select_ensembles.py --phase transformer
# 必须在 carry 搜索结束、配置冻结后再运行下面两个搜索
.venv/bin/python notebooks/Yuxin/tune_models.py --phase rnn --trials 32
.venv/bin/python notebooks/Yuxin/tune_models.py --phase lstm --trials 32
.venv/bin/python notebooks/Yuxin/tune_models.py --phase rnn --trials 32 --complete-pruned
.venv/bin/python notebooks/Yuxin/tune_models.py --phase lstm --trials 32 --complete-pruned
.venv/bin/python notebooks/Yuxin/finalize_tuning.py --prepare
.venv/bin/python notebooks/Yuxin/run_experiments.py \
  --model market_carry tabular_transformer market_rnn market_lstm \
  --config-dir notebooks/Yuxin/tuning/configs --split rolling --final \
  --output notebooks/Yuxin/tuning/final
.venv/bin/python notebooks/Yuxin/finalize_tuning.py --check
```

`--trials` 是研究累计预算，含失败/剪枝尝试，不是每次追加的数量。
`--complete-pruned` 只用于廉价时间网络：保留旧剪枝记录，额外重放这些参数并禁用剪枝；
本轮 RNN/LSTM 各 32 组不同参数最终都有完整两折成绩，日志条数因重放分别为 49/48。
SQLite 和逐条搜索预测缓存放在 `/tmp/yuxin_hpo_<模型源码哈希>/`；不要同时首次初始化同一数据库。
`tuning/*_trials.json` 保存状态、参数、分折成绩和训练证据；`best_*.json` 只在完整两折候选中选择。
临时缓存丢失后需重新搜索，或直接用保存的 `tuning/configs/` 重训，不应把临时路径当作预测依赖。
搜索中断的 RUNNING trial 不会自动恢复为完成；先确认对应进程已经退出，再显式标记失败并重试。

本轮还比较了同一模型家族内的 top-K / 贪心前向等权平均，最多 4 个成员；没有用队友模型预测。
最终配置必须标明这是平均版本，不能用其成绩代替最佳单网络成绩。
Transformer 冻结配置使用 `n_jobs=4` 并行独立成员，每个 torch 成员仍为单线程；
其他模型默认串行。20 项测试中包含串行/进程并行的逐值预测、训练曲线和审计等价性。
暖启动/局部候选和恢复顺序以已保存 trial 日志为准，以上新数据库搜索命令不保证重复同一 TPE 轨迹；
复现已选结果应直接从冻结的 `tuning/configs/` 运行完整公共框架。

`finalize_tuning.py --check` 会重算 CSV 分数，并逐条对照搜索缓存，证明完整公共框架重训复现了选中结果。
临时缓存清除后，长期复核使用 `verify_delivery.py --roots notebooks/Yuxin/tuning/final results/Yuxin`，
不依赖 `/tmp`；本轮缓存逐条比对的结论保存在 `tuning/final_verification.json`。
这些外层折已经用于调参，因此是开发集成绩，不是无偏泛化估计；隐藏测试效果必须以 Kaggle 返回为准。
有限 TPE 搜索只能交付已试候选中的最优组合，不证明全局最优，也不保证超过树模型。

## 步骤 4—6：时间对照、互补性与锁定交付

`compare_temporal.py` 先固定属性模型、L=6、hidden=8、训练预算及 rollout，比较 carry、阻尼趋势、RNN、LSTM、时间 Transformer；再完整评估 8 组短窗口 attention 参数。
它复用已提交 carry 运行的属性 OOF 比例，只重训各折的市场编码器，**不是 12 次独立重训属性模型**。完整参数、原点、月度预测和训练证据保存在 `stage4/controlled_comparison.json`。
时间 Transformer 是 1 层、2 头，带窗口相对正弦位置编码和因果 mask；输出一步增量，再在固定原点递归，不读取未来真实市场水平。

最佳候选使用 L=6、d_model=8、lr=0.005、dropout=0.1、weight_decay=0.01、damping=0.8、strength=0.25、内部验证 12 月、最多 80 epoch/patience=25。
独立公共框架完整重训的均值为 **478.323802**，未优于 carry **478.237340**；保留完整验证和 test，不晋升到正式目录。

```bash
export LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# 受控比较与冻结新时间模型配方
.venv/bin/python notebooks/Yuxin/compare_temporal.py
# 重训请写新的目录；stage4/full 中已有完整运行，不要重复追加
.venv/bin/python notebooks/Yuxin/run_experiments.py \
  --model market_transformer --config-dir notebooks/Yuxin/stage4/configs \
  --split rolling --final --output /tmp/yuxin_stage4_reproduction
```

旧 `tune_models.py`/`finalize_tuning.py --prepare` 绑定步骤 1—3 的源码签名。新增时间编码器后，不应在旧研究上直接续跑；复核历史搜索用保存的日志/证据，复现已选方案使用 `tuning/configs/`，不要取消源码或数据哈希检查。

`evaluate_complementarity.py` 仅读取当前 `results/Yuxin` 和队友固定提交下 `results/` 的正式预测来做融合；不读取队友训练拟合值、不 checkout 其他分支、不重训队友模型。
carry 和未晋升的时间 Transformer 只加入误差/相关性分析，不作为团队融合输入。
所有预测按 `(fold,row_index)` 一对一核对日期/标签；逐月、URA 大区、town、户型、单月预测跨度及 4 月跨度分组均输出 RMSE、MAE、bias、样本数。

```bash
# 复核已有产物，不重新训练：先误差/融合分析，再最终审计与锁定
.venv/bin/python notebooks/Yuxin/evaluate_complementarity.py
.venv/bin/python notebooks/Yuxin/finalize_stages.py
```

上述最终审计按已有路径读取 `stage4/full/`，要求该目录恰好一个完整时间 Transformer 运行。新路径重训的输出可先用 `verify_delivery.py --roots /tmp/yuxin_stage4_reproduction --summary-output /tmp/yuxin_stage4_recheck.json` 单独核验，勿覆盖现有证据。

- `stage4/comparison.csv`、`stage4/verification.json`：12 个受控候选和独立重训逐行比对。
- `stage5/group_errors.csv`、`residual_correlations.csv`：9 个基础/对照模型及 12 个固定平均组合的分组误差；相关性逐折计算。
- `stage5/fixed_blends.csv`：三类正式神经网络 × Ridge/CatBoost/XGBoost/LightGBM 的全部 12 组 50:50 比较，含未改善的组合。
- `stage5/forward_transfer.csv`：只用 A 折 2023-12—2024-03 的 12013 行校准一个凸权重，评估 B 折 2024-04—2025-03 的 35158 行；不使用 B 标签选择权重。
- `stage5/diagnostic_blend_validation.csv` / `diagnostic_blend_test.csv`：RNN＋CatBoost 固定平均的实际两折验证/完整 test，**仅诊断，不是第 4 个正式运行**。
- `stage5/verification.json`：队友 commit、run_id、来源哈希与范围；`independent_verification.json` 为独立算式复核。
- `results/Yuxin/selection.json`：锁定三个正式候选，含每个成员 A/B/final 的实际 epoch、冻结配方及产物 SHA-256。

RNN＋CatBoost 固定平均均值 **477.552001**，证明当前开发折上存在互补收益，但所有底层模型已进行外层调参，不能称为无偏嵌套或隐藏集验证。
RNN/LSTM 残差相关接近 1；保留两者是负责家族的比较交付，不建议把三者等同为三个独立信息源。没有找到可读取的 KNN 正式预测，不提供虚构的 KNN 相关性。
正式 RNN/LSTM 原训练源码精确对应 `55d2acb`，表格 Transformer 对应 `aba5078`；锁定脚本逐文件核对 Git blob 与运行保存的源码哈希，避免把新源码冒认为旧运行源码。
