# Yuxin 神经网络：开发阶段 1—3

实现说明和真实运行结果见根目录 `develop_test.md`；原始设计见 `develop.md`。
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
首次交付的完整命令为：

```bash
.venv/bin/python notebooks/Yuxin/run_experiments.py \
  --model tabular_transformer market_rnn market_lstm \
  --split rolling --final --deliver --verbose
```

`results/Yuxin/` 最多三个运行。`validation/` 为公共基线、MLP 和 carry 对照，不计入融合模型交付，不含权重。
每个模型都输出两折逐行验证预测及完整 50000 条 test 预测；市场预测原点固定，不使用未来租金或未来实测经济变量。

## 复核与结果入口

```bash
.venv/bin/python notebooks/Yuxin/verify_delivery.py \
  --roots notebooks/Yuxin/validation results/Yuxin
.venv/bin/python notebooks/Yuxin/analyze_results.py \
  --roots notebooks/Yuxin/validation results/Yuxin
```

- `comparison.csv`：从预测重算的 A/B 和两折简单平均 RMSE。
- `monthly_errors.csv`：各模型、各折、每月 RMSE 和偏差。
- `verification.json`：完整行号/日期/标签/分数/编号/信息边界复核。
- `results/Yuxin/verification.json`：三个候选的独立交付复核。
- `training_audit.json` 中的未来市场曲线为固定原点递归预测，不是真实未来租金。

比较神经网络与树模型时请使用同一外层切分；不要把内部早停分数作为最终成绩，也不要把 A/B 重叠验证行作为独立样本合并计算总体 RMSE。
