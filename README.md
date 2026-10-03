# CS5228：新加坡 HDB 月租金预测

CS5228 小组项目，参加 Kaggle 私有比赛 **CS5228-2610 Project**。根据新加坡 HDB 组屋属性预测月租金 `MONTHLY_RENT`，任务为回归，评估指标为原始租金尺度的 **RMSE**（SGD）。

课程要求对使用或不使用的每类信息，尤其辅助数据，给出论证；报告与分数同等重要。目前已完成基础数据清洗、公共实验框架和两个参考基线，其余模型与特征由组员接入。

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
Project_export/
├── README.md                       # 项目介绍与使用入口
├── AGENTS.md                       # Agent 常用命令与文档入口
├── docs/
│   └── experiment-interface.md     # 公共框架唯一接口协议
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
│   ├── models/                     # 模型实现与参考基线
│   ├── config.py                   # 项目路径与随机种子
│   ├── cv.py                       # 验证方案实现
│   ├── experiment.py               # 实验运行与产物保存
│   └── run_baseline.py              # 基线入口
├── tests/
│   └── test_framework.py
├── experiments/
│   ├── log.csv                     # 共享实验日志
│   └── <run_id>/                    # 每次运行的配置与验证预测
├── submissions/                    # 运行生成的提交 CSV
└── notebooks/                      # 个人实验约定目录，尚未创建
```

## 数据与建模背景

原始训练集有 150000 行，测试集有 50000 行。辅助数据包括楼栋、MRT、学校、购物中心、COE 和股价；基础清洗已统一字符串与户型写法、连接楼栋信息并修正辅助表格式。建模数据的字段、读取方法、清洗过程和已知口径差异见 [清洗数据说明与数据字典](data_cleaned/README.md)。

租金在 2021–2023 年大涨，2024 年后趋稳（年均约 2121 → 3036 → 3110）；树模型无法直接外推时间，时间趋势需要单独考虑。数据是面板 / 横截面数据，没有房屋 ID。按伪单位统计约有 14.8k 个单位，每单位记录数中位数为 7；同一伪单位、同一月份的租金标准差中位数约 283，RMSE 的合理预期在 300 以上。

时间验证、防泄漏和建模边界的具体要求统一见 [接口协议](docs/experiment-interface.md#数据与建模边界)。

## 团队分工

| 组员 | 负责范围 |
|---|---|
| P1 | 数据与公共框架、目标编码、融合与提交 |
| P2 | 统计模型：基线、Ridge、双向固定效应、混合效应；时间趋势处理 |
| P3 | GBDT：LightGBM、XGBoost、CatBoost、RF；调参与 SHAP |
| P4 | 辅助数据、距离特征、KNN、MLP + embedding |

公共实现放在 `src/`，个人实验命名为 `notebooks/<姓名>_<主题>.ipynb`。新增模型库由相应负责人维护依赖。各组员共用实验框架，以便比较模型表现和进行消融实验。

## 参考基线

当前提供最近六个月分组中位数和 sklearn HistGradientBoosting 两个基线。默认配置的本地结果见 [共享实验日志](experiments/log.csv)：

| 基线 | Holdout RMSE | Rolling 平均 RMSE |
|---|---:|---:|
| 最近六个月分组中位数 | 524.659 | 515.448 |
| HistGradientBoosting | 497.937 | 489.241 |

## 重要文档

| 文档 | 内容 |
|---|---|
| [公共实验框架接口协议](docs/experiment-interface.md) | 特征与模型接入、验证方案、实验参数及产物格式；唯一协议信源 |
| [清洗数据说明与数据字典](data_cleaned/README.md) | 数据读取、字段类型、清洗步骤及辅助数据注意事项 |
| [Agent 开发与测试命令](AGENTS.md) | 常用命令、验证方法和开发入口 |
| [Kaggle 任务说明原文](CS5228-2610%20Project%20_%20Kaggle.mhtml) | 本地保存的比赛说明 |
| [共享实验日志](experiments/log.csv) | 各模型的配置和验证结果 |
