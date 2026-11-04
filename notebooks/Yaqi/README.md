# Yaqi：树模型实验与融合交付统一入口

新旧分支成果已按当前目录规范整理到此分支。后续以 `yaqi-results-20261103` 为工作入口；旧分支 `yaqi-tree-models` 保留原始版本。公共框架维持最新版本，未把旧补丁或旧公共文件覆盖回来。

| 内容 | 入口 |
|---|---|
| 给组员做融合的三个正式模型 | [results/Yaqi](../../results/Yaqi/README.md) |
| 新旧43条实验记录统一索引 | [experiment_catalog.csv](experiment_catalog.csv) |
| 早期13组算法、窗口、地址实验 | [history/20261022](history/20261022/README.md) |
| 后续30组位置、经济特征实验 | [history/20261103](history/20261103/README.md) |
| 合并过程、来源与校验 | [history](history/README.md) |
| 统一训练入口 | [train_experiments.py](train_experiments.py) |
| 查看结果和运行训练的Notebook | [tree_experiments.ipynb](tree_experiments.ipynb) |

13+30是43条真实配置运行记录，其中旧 `catboost_address_24m` 与新 `catboost_base` 是同一配置的复跑，两折逐行预测完全一致。历史目录还保留一次旧最佳配置的最终重训，因此底层共44个运行目录；后续阶段另附四份最终模型的test预测及导出记录。正式交付的三个模型另存于 `results/Yaqi`。

## 统一运行方式

从项目根目录运行，使用已有 Python 环境。以下两条仅检查历史参数能否由当前代码精确构造，不启动训练：

```powershell
python notebooks/Yaqi/train_experiments.py --suite initial --check-recipes
python notebooks/Yaqi/train_experiments.py --suite features --check-recipes
```

按需要重新训练早期实验或特征实验：

```powershell
python notebooks/Yaqi/train_experiments.py --suite initial --names rf_all extra_all
python notebooks/Yaqi/train_experiments.py --suite features --names rf_base rf_location
python notebooks/Yaqi/train_experiments.py --suite features --names catboost_location_coe --final
```

省略 `--names` 会运行该阶段全部配置；`--list` 只显示配置与历史成绩。默认随机种子42、4线程、两折时间验证；`--final` 额外生成全部test的预测。每次运行写入新的 `outputs/Yaqi/<时间>_<阶段>/`，不会覆盖历史结果或自动增加正式融合模型，也不保存模型权重。新运行的实测分数与历史分数分别记录，不冒充历史训练已经重新执行。

```powershell
python notebooks/Yaqi/verify_history.py
python notebooks/Yaqi/verify_delivery.py
```

合并核验会重算历史44次运行的两折RMSE、检查原行号/月份/标签、17个数据文件指纹及5份历史提交。历史文件原样保留，其中 `runs.json`、旧报告和配置内的 `outputs/...` 路径是当时的记录；现在的有效文件位置以统一索引中的 `config_path` 和 `validation_predictions_path` 为准。

## 后续30组特征实验与正式交付

本次交付基于最新公共框架 `32213c43`（包含 `907aa001` 的交付规范及 `57f8ee8` 的 sklearn 修复）。公共框架、配置、忽略规则和共享日志保持原样；新增的可复用模型与特征分别在 `src/models/trees.py`、`src/features/distance.py`、`src/features/macro.py`。

下午完成五种模型各六阶段，共30组配置。完整成绩见 [afternoon_comparison.csv](afternoon_comparison.csv)，中文分析见 [FEATURE_REPORT.md](FEATURE_REPORT.md) 和 [FOREST_REPORT.md](FOREST_REPORT.md)。这些报告中的 `outputs/`、模型权重路径指向当时本地实验；GitHub 融合交付请使用 [results/Yaqi](../../results/Yaqi/README.md)。

按每人最多三个运行的约定，选择各算法最优配置中验证均值最低的三个算法：CatBoost＋位置＋COE、XGBoost 基础版、LightGBM＋位置＋COE。选择依据是当前开发验证均值，不代表已经优化过融合收益。RF、ExtraTrees 及其他消融保留在完整汇总中。

### 复现三个正式模型

使用项目现有的 Python 环境。直接依赖见 `requirements.in`，本轮实际安装版本见 `requirements.txt`。以下命令在项目根目录执行：

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s notebooks/Yaqi -p test_delivery.py -v
python notebooks/Yaqi/delivery.py --kind catboost
python notebooks/Yaqi/delivery.py --kind xgboost
python notebooks/Yaqi/delivery.py --kind lightgbm
python notebooks/Yaqi/verify_delivery.py
```

已交付的同名运行不会重复追加。若要重新生成最终交付，应先在自己的分支中按公共规范管理旧运行，保持最多三个；不要把日常参数搜索追加到 `results/Yaqi/`。

每次调用公共 `run_experiment(..., split="rolling", final=True, output_dir=ROOT_DIR/"results"/"Yaqi")`。运行入口在个人进程内明确固定种子42，延续下午的实验设定，不修改公共 `src/config.py` 的默认值。所有模型仍使用最近24个月及地址字段，参数与保存的原始配置逐项核对；两折真实分数必须复现到 `1e-6` 才通过交付检查。

### 特征与信息边界

位置特征为最近 open MRT（排除 LRT/RTS）、商场、小学的大圆距离，单位km。设施文件是快照，没有历史启用日期，不能视为严格重建历史设施的回测。

COE按月份、类别平均。训练月份t使用t-1经济输入；每折先截断到训练最大月M-1，随后整个未来区间保持截止点的最后可见值。A只使用到2023-10，B到2024-02，最终到2025-02。文件没有发布时间，一个月可得性延迟是明确假设。股票与阻尼预测代码也保留用于复现下午其他实验，本次三个定稿配置不使用股票。

记录器只在模型完成拟合后额外导出训练诊断，不改变拟合输入、参数或验证过程。其单独测试验证了原行号、真实标签、最近窗口实际拟合标记，以及加入记录器前后预测相同。

验证窗口A/B重叠，应按 `(fold,row_index)` 对齐。它们参与了配置选择，成绩属于开发验证，不是独立测试或Kaggle成绩。
