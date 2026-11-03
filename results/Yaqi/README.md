# Yaqi：交付融合的树模型预测

按公共 [6.1交付规范](../../docs/experiment-interface.md#61-交付结果给融合) 提供三个运行。每个运行均实际调用最新公共框架的 `split="rolling", final=True`，种子42，最近24个月训练。模型参数保持下午实验设置；三个算法为CatBoost（位置＋COE）、XGBoost（基础版）和LightGBM（位置＋COE）。运行编号、实际分数及数据指纹见 [manifest.json](manifest.json) 和 [selected_models.csv](selected_models.csv)。

## 给融合负责人的入口

| 文件 | 用途 |
|---|---|
| `experiments/<run_id>/validation_predictions.csv` | 两折按时间留出的预测，供训练融合器；列为 `row_index,fold,RENT_APPROVAL_DATE,MONTHLY_RENT,Predicted` |
| `submissions/<run_id>.csv` | 用最终训练模型预测的全部50000条test；列为 `Id,Predicted`，用于最终融合推理 |
| `experiments/<run_id>/training_predictions.csv` | 按要求额外附上的150000条原训练数据预测，仅用于拟合诊断；`fold=final` |
| `experiments/<run_id>/fold_training_predictions.csv` | 两折各自训练部分上的诊断预测，保留fold和原行号 |
| `experiments/<run_id>/config.json` | 公共框架配置、分数、原下午运行编号、依赖版本、实际拟合范围和经济信息截止点 |
| `experiments/log.csv` | 仅这三个正式运行的公共格式日志 |

**训练融合器请使用按时间留出的 `validation_predictions.csv`。** 两份training文件包含模型在自己训练记录上的拟合预测，不能替代OOF输入。`used_for_model_fit` 标明该记录是否属于该模型实际使用的24个月窗口；对于窗口以前的记录，最终模型预测也带有事后信息，不能当作历史回测预测。

A验证2023-12—2025-03，B验证2024-04—2025-03。验证文件共82329行，其中A47171、B35158；同一条训练记录可能在两个fold出现，须分别按 `(fold,row_index)` 对齐，不能当作82329条独立样本。最终模型可用150000条训练数据，按固定24月方案实际拟合2023-04—2025-03的71040条记录。

```python
import json
from pathlib import Path
import pandas as pd

root = Path("results/Yaqi")
models = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["models"]
cat = next(m for m in models if m["kind"] == "catboost")
valid = pd.read_csv(root / "experiments" / cat["run_id"] / "validation_predictions.csv")
valid_a = valid.loc[valid.fold.eq("holdout")]
test = pd.read_csv(root / "submissions" / (cat["run_id"] + ".csv"))
```

所有预测均处于原始月租金尺度（SGD）。数据、行序、标签、分数、训练/验证分离、提交编号及文件完整性检查见 [verification.json](verification.json)。本目录不含模型权重、压缩包或训练数据源文件。

下午全部30组实验汇总及RF/ExtraTrees分析见 [个人实验目录](../../notebooks/Yaqi/README.md)。源码与复现命令也在该目录。
