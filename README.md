# CS5228：新加坡 HDB 月租金预测

CS5228 小组项目，参加 Kaggle 私有比赛 **CS5228-2610 Project**。根据新加坡 HDB 组屋属性预测月租金 `MONTHLY_RENT`，任务为回归，评估指标为原始租金尺度的 **RMSE**（SGD）。

课程要求对使用或不使用的每类信息，尤其辅助数据，给出论证；报告与分数同等重要。目前已完成基础数据清洗、公共实验框架、各成员的模型和最终融合，最终提交见 [融合与最终提交](doc/fusion_p1.md)。各成员工作的详细说明见 [团队分工](#团队分工) 中的文档链接。

## 快速开始

需要 Python ≥ 3.10，第三方依赖见 [requirements.txt](requirements.txt)。在项目根目录运行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.run_baseline
```

基线入口会运行分组中位数和 HistGradientBoosting，打印验证 RMSE、记录实验日志，并生成一个提交文件。新增特征或模型前，请阅读 [公共实验框架接口协议](docs/experiment-interface.md)；完整开发和数据校验命令见 [AGENTS.md](AGENTS.md)。

## 项目结构

```text
CS5228/
├── README.md                       # 项目介绍与使用入口
├── AGENTS.md                       # Agent 常用命令与文档入口
├── docs/
│   └── experiment-interface.md     # 公共框架唯一接口协议
├── doc/                            # 各成员的方法与开发文档
├── requirements.txt
├── CS5228-2610 Project _ Kaggle.mhtml
├── data/                           # 原始数据
│   ├── train.csv
│   ├── test.csv
│   ├── example-submission.csv
│   └── auxiliary/
├── data_cleaned/                    # 基础清洗后的建模数据
│   ├── README.md                    # 清洗说明与数据字典
│   ├── train.csv
│   ├── test.csv
│   └── auxiliary/
├── src/
│   ├── data_cleaning/              # 清洗、schema、loader 与数据校验
│   ├── features/                   # 特征实现与注册表
│   ├── models/                     # 参考基线、KNN；yuxin_nn/ 为个人神经网络包
│   ├── config.py                   # 项目路径与随机种子
│   ├── cv.py                       # 验证方案实现
│   ├── experiment.py               # 实验运行与产物保存
│   └── run_baseline.py              # 基线入口
├── tests/
│   ├── test_framework.py
│   ├── test_yuxin_nn.py
│   └── test_yuxin_analysis.py
├── experiments/
│   ├── log.csv                     # 共享实验日志
│   └── <run_id>/                    # 每次运行的配置与验证预测
├── submissions/                    # 运行生成的提交 CSV
├── results/<成员名>/                # 交付给融合的运行产物，全部进 Git
└── notebooks/<成员名>/              # 个人 notebook、脚本、图表与分析笔记
```

## 数据与建模背景

原始训练集有 150000 行，测试集有 50000 行。辅助数据包括楼栋、MRT、学校、购物中心、COE 和股价；基础清洗已统一字符串与户型写法、连接楼栋信息并修正辅助表格式。建模数据的字段、读取方法、清洗过程和已知口径差异见 [清洗数据说明与数据字典](data_cleaned/README.md)。

租金在 2021–2023 年大涨，2024 年后趋稳（年均约 2121 → 3036 → 3110）；树模型无法直接外推时间，时间趋势需要单独考虑。数据是面板 / 横截面数据，没有房屋 ID。按伪单位统计约有 14.8k 个单位，每单位记录数中位数为 7；同一伪单位、同一月份的租金标准差中位数约 283，RMSE 的合理预期在 300 以上。

时间验证、防泄漏和建模边界的具体要求统一见 [接口协议](docs/experiment-interface.md#数据与建模边界)。

## 团队分工

| 组员 | 负责范围 | 文档 |
|---|---|---|
| P1 | 数据清洗与公共实验框架、KNN、融合与最终提交 | [KNN](doc/knn_p1.md) · [融合与最终提交](doc/fusion_p1.md) |
| P2 | 统计模型：基线、Ridge、双向固定效应、混合效应；时间趋势处理 | [Ridge 文档](doc/ridge_p2.md) · [交付索引](results/FXY/README.md) |
| P3 | GBDT：LightGBM、XGBoost、CatBoost、RF；调参与 SHAP | — |
| P4 | 辅助数据、距离特征、MLP + embedding | [开发方案](doc/develop_yuxin_nn.md) · [验收记录](doc/develop_test_yuxin_nn.md) · [调参报告](doc/develop_tuning_yuxin_nn.md) · [运行说明](notebooks/Yuxin/README.md) |

各组员共用实验框架，以便比较模型表现和进行消融实验。新增模型库由相应负责人维护依赖。

本分支新增的实际负责范围：**Yuxin — 表格 Transformer、市场 RNN/LSTM、时间 Transformer 及同属性 carry 对照**。上表保留公共角色分工，不据此替换其他组员职责。

## 协作与目录约定

`main` 是所有人开发的基底。每个新模型或特征从 `main` 新开分支（建议命名 `<成员名>-<主题>`），完成后通过 PR 合并回 `main`；不要直接修改公共框架来适配个人模型，框架问题单独提出。

| 内容 | 位置 |
|---|---|
| 可复用的特征块 | `src/features/<名字>.py`，按 [接口协议](docs/experiment-interface.md#35-放在哪里怎么注册) 注册 |
| 可复用的模型类 | `src/models/<名字>.py` |
| 个人 notebook、脚本、图表与分析笔记 | `notebooks/<成员名>/` |
| 交付给融合的运行产物 | `results/<成员名>/`，规则见 [交付结果给融合](docs/experiment-interface.md#61-交付结果给融合) |

用 Git 提交和推送，不要通过网页上传压缩包或 patch。根目录只放现有的公共文件，不提交训练好的模型文件和其他大型二进制文件。

## 参考基线

当前提供最近六个月分组中位数和 sklearn HistGradientBoosting 两个基线。默认配置的本地结果见 [共享实验日志](experiments/log.csv)：

| 基线 | Holdout RMSE | Rolling 平均 RMSE |
|---|---:|---:|
| 最近六个月分组中位数 | 524.659 | 515.448 |
| HistGradientBoosting | 497.937 | 489.241 |

## Yuxin 神经网络与步骤 4—6

已完成 `doc/develop_yuxin_nn.md` §8.1 的六个步骤；本次新增短窗口 temporal Transformer、逐月/地区/户型/预测跨度误差、残差相关性及实际融合检验。完整验收和局限见 [develop_test_yuxin_nn.md 第 10 节](doc/develop_test_yuxin_nn.md)。

| 冻结的正式候选 | Rolling 平均 RMSE（SGD） |
|---|---:|
| 市场 RNN＋4 成员属性 MLP 平均 | 478.125922 |
| 市场 LSTM＋相同属性组合 | 478.128651 |
| 4 成员表格 Transformer 平均 | 481.329044 |

这是同类配置平均方案，不是单网络分数。表格 Transformer 的 token 是字段；temporal Transformer 的 token 是历史月份，两者不是同一种实验。
新时间 Transformer 的完整开发均值 **478.323802**，未优于相同属性 carry **478.237340**，因此保留负结果，不替换上述三个正式候选。
RNN 与队友 CatBoost 固定 50:50 平均的开发均值为 **477.552001**，比 RNN 自身低约 0.574 SGD；不意味着所有组合都有效，完整 12 组比较见 [固定融合诊断](notebooks/Yuxin/stage5/fixed_blends.csv)。

外层折已经参与调参，A/B 区间重叠；上述均为开发成绩，不是无偏测试估计或 Kaggle 隐藏分数。队友采用的种子、环境和额外特征存在差异。没有上传 Kaggle，也未推送远端。

复核已保存结果（项目根目录）：

```bash
export LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r notebooks/Yuxin/requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python notebooks/Yuxin/evaluate_complementarity.py
.venv/bin/python notebooks/Yuxin/finalize_stages.py
```

已有 `.venv` 时可跳过创建；这是个人环境，不提交 Git。进一步的训练、环境设置和复现命令见 [Yuxin README](notebooks/Yuxin/README.md)。
[正式交付](results/Yuxin/README.md) 恰好保留三个运行；[选择与配置锁定清单](results/Yuxin/selection.json) 记录参数文件、实际训练轮数、原训练源码提交和产物哈希。
分组误差与融合诊断位于 `notebooks/Yuxin/stage5/`，不是额外正式交付；新时间模型的配置、训练审计与验证摘要位于 `notebooks/Yuxin/stage4/full/`，逐折预测 CSV 按 `.gitignore` 不再提交。

## 重要文档

| 文档 | 内容 |
|---|---|
| [公共实验框架接口协议](docs/experiment-interface.md) | 特征与模型接入、验证方案、实验参数及产物格式；唯一协议信源 |
| [清洗数据说明与数据字典](data_cleaned/README.md) | 数据读取、字段类型、清洗步骤及辅助数据注意事项 |
| [Agent 开发与测试命令](AGENTS.md) | 常用命令、验证方法和开发入口 |
| [Kaggle 任务说明原文](CS5228-2610%20Project%20_%20Kaggle.mhtml) | 本地保存的比赛说明 |
| [共享实验日志](experiments/log.csv) | 各模型的配置和验证结果 |
| [Yuxin 开发方案](doc/develop_yuxin_nn.md) / [验收记录](doc/develop_test_yuxin_nn.md) | 六步实施要求、实际结果与边界 |
| [Yuxin 调参报告](doc/develop_tuning_yuxin_nn.md) | 步骤 1—3 的有限搜索及冻结参数 |
| [Yuxin 运行说明](notebooks/Yuxin/README.md) / [正式候选](results/Yuxin/README.md) | 神经网络复现、步骤 4—6 与交付索引 |
