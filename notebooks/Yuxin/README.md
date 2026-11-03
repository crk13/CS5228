# Yuxin 神经网络：开发阶段 1—3

实现说明和真实运行结果见根目录 `develop_test.md`；原始设计见 `develop.md`。
最新调参报告为 `develop_tuning.md`，最新冻结参数为 `tuning/configs/`。
`configs/` 保留初验配置；不带最新 `--config-dir` 的旧命令不会运行本轮优化组合。
公共数据/模型/验证接口以 `docs/experiment-interface.md` 为准。

本目录使用公共 `basic`、`block` 特征，不读取额外宏观或设施数据。
`src/models/yuxin_nn/` 提供 embedding MLP、表格特征 Transformer，以及共享属性 MLP 的市场 carry/RNN/LSTM。
RNN/LSTM 的序列单位是全市月度租金中位数，不是单套房或交易行。
时间 Transformer、多市场池化、距离/COE 消融尚未在本阶段实现。

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
所有比较仍使用公共两折，不实现时间 Transformer。QLoRA 是大语言模型的低显存微调方法，不是本任务的超参数搜索器。

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
