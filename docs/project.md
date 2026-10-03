# CS5228 项目说明

CS5228 小组项目，参加 Kaggle 私有比赛 **CS5228-2610 Project**。根据新加坡 HDB 组屋属性预测月租金，课程要求对使用或不使用的每类信息，尤其辅助数据，给出论证；报告与分数同等重要。

目前已完成基础数据清洗、公共实验框架和两个参考基线，其余模型与特征由组员接入。接入协议统一见 [公共实验框架接口协议](experiment-interface.md)；开发与测试命令见 [AGENTS.md](../AGENTS.md)，字段和数据读取说明见 [数据字典](../data_cleaned/README.md)。

## 项目结构

```text
Project_export/
├── README.md                       # 文档导航
├── AGENTS.md                       # Agent 常用命令与文档入口
├── docs/
│   ├── project.md                  # 项目说明与团队分工
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

原始训练集有 150000 行，测试集有 50000 行。辅助数据包括楼栋、MRT、学校、购物中心、COE 和股价；基础清洗已统一字符串与户型写法、连接楼栋信息并修正辅助表格式。清洗过程、具体文件和已知口径差异以 [数据说明](../data_cleaned/README.md) 为准。

租金在 2021–2023 年大涨，2024 年后趋稳（年均约 2121 → 3036 → 3110）；树模型无法直接外推时间，时间趋势需要单独考虑。数据是面板 / 横截面数据，没有房屋 ID。按伪单位统计约有 14.8k 个单位，每单位记录数中位数为 7；同一伪单位、同一月份的租金标准差中位数约 283，RMSE 的合理预期在 300 以上。具体建模边界与验证规则见 [接口协议](experiment-interface.md#数据与建模边界)。

## 团队分工

| 组员 | 负责范围 |
|---|---|
| P1 | 数据与公共框架、目标编码、融合与提交 |
| P2 | 统计模型：基线、Ridge、双向固定效应、混合效应；时间趋势处理 |
| P3 | GBDT：LightGBM、XGBoost、CatBoost、RF；调参与 SHAP |
| P4 | 辅助数据、距离特征、KNN、MLP + embedding |

公共实现放在 `src/`，个人实验命名为 `notebooks/<姓名>_<主题>.ipynb`。新增模型库由相应负责人维护依赖。当前需要 Python ≥ 3.10；第三方依赖见 [requirements.txt](../requirements.txt)。

## 参考基线

实现见 [src/models/baselines.py](../src/models/baselines.py)，完整运行命令见 [AGENTS.md](../AGENTS.md#常用开发与测试命令)。

- **分组中位数**：使用训练数据最近 6 个日历月的镇区与户型组合中位数，缺少组合时回退到同期户型中位数，再回退到同期全局中位数。
- **HistGradientBoostingRegressor**：类别编码由模型内部处理，保留缺失值，关闭随机提前停止验证；不依赖额外模型库。

当前默认配置的本地结果见 [共享日志](../experiments/log.csv)：

| 基线 | Holdout RMSE | Rolling 平均 RMSE |
|---|---:|---:|
| 最近六个月分组中位数 | 524.659 | 515.448 |
| HistGradientBoosting | 497.937 | 489.241 |
