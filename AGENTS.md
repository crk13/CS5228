# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, Cursor, Copilot, etc.) and human collaborators when working with code in this repository.

## 项目概述

CS5228 小组项目（Kaggle 私有比赛 "CS5228-2610 Project"）：根据新加坡 HDB 组屋属性预测月租金 `MONTHLY_RENT`。
- 任务类型：回归；评估指标：**RMSE**（原始租金尺度）。
- 任务说明原文：`CS5228-2610 Project _ Kaggle.mhtml`。
- 课程要求对使用 / 不使用的每类信息（尤其辅助数据）**给出论证**，报告与分数同等重要。

## 数据

- `data/train.csv`：150,000 行，含目标 `MONTHLY_RENT`。
- `data/test.csv`：50,000 行，无 Id 列；**提交的 `Id` 即测试集行号 0..49999**。
- `data/example-submission.csv`：提交格式 `Id,Predicted`。
- `data/auxiliary/`：
  - `sg-hdb-block.csv`：楼栋经纬度、`MAX_FLOOR`、`YEAR_COMPLETED`、`SUBZONE`、`PLANNING_AREA`、`REGION`。按 (BLOCK, STREET) 小写后与 train/test **100% 匹配**。
  - `sg-mrt-stations.csv`：含 `STATUS`（open / planned），计算距离时注意区分。
  - `sg-schools.csv`、`sg-shopping-malls.csv`：POI 经纬度。
  - `sg-coe-prices.csv`：数值列是带 `$` 和千分位逗号的字符串，需先解析。
  - `sg-stock-prices.csv`：日频股价，需聚合到月。

### 已确认的数据坑

- `FURNISHED`（全为 yes）和 `FEE`（全为 0）在 train 和 test 中都是常数，应删除。
- `FLAT_TYPE` 混用 "4 room" 和 "4-room"，需统一（test 中只有连字符写法）。
- `STREET` 约 20% 带大写，join 和分组前统一转小写。
- 无缺失值。

## 关键建模约束

- **train/test 按时间切分**：train 覆盖 2021-01 至 2025-03，test 覆盖 2025-03 至 2026-07。这是教授的疏忽，但不会更正。
  - 本地验证必须按时间切分，不要用随机 K-fold。约定主验证为 train ≤ 2023-11，valid 为 2023-12 至 2025-03（16 个月，与测试跨度一致）。
  - 租金 2021 到 2023 年大涨，2024 年后趋稳（年均约 2121 → 3036 → 3110）。树模型无法外推时间。
- **这不是纯时间序列数据**（教授在公告中明确说明）。数据是面板 / 横截面数据，且**没有房屋 ID**。
  - 不要对单套房做 ARIMA、LSTM，也不要构造"同一套房上期租金"之类的滞后特征。
  - 只有聚合后的市场月度指数可以当作时间序列处理。
  - 允许构造"伪单位"键：`BLOCK + STREET + FLAT_TYPE + FLAT_MODEL + FLOOR_AREA_SQM`，共约 14.8k 个单位，中位数每个单位 7 条记录。用作分组统计或固定效应时，必须在报告中声明这是一个假设。
- **噪声下限**：同一伪单位、同一月份的记录之间，租金标准差中位数约 283。RMSE 的合理预期在 300 以上。
- **目标编码 / 分组统计必须防泄漏**：验证时只能用验证切分点之前的数据或 out-of-fold 统计；最终提交时才用全量 train。

## 团队约定（规划中，代码尚未建立）

四人分工：
- P1：数据与公共框架、目标编码、融合与提交。
- P2：统计模型（基线、Ridge、双向固定效应、混合效应）与时间趋势处理。
- P3：GBDT（LightGBM、XGBoost、CatBoost、RF）、调参、SHAP。
- P4：辅助数据、距离特征、KNN、神经网络（MLP + embedding）。

计划的代码组织：
- 公共模块放在 `src/`（清洗、特征、时间切分、`run_experiment()` 接口）。
- 个人实验放在 `notebooks/<姓名>_<主题>.ipynb`。
- 提交文件放在 `submissions/`。

所有实验通过统一接口、在同一验证切分上运行，结果记入共享实验日志，以便方法之间可比。特征按模块（F1 基础 / F2 楼栋 / F3 距离 / F4 目标编码 / F5 宏观 / F6 空间邻居）组织，可单独开关，用于消融实验。

建立代码后，请在此补充实际的运行命令与依赖说明。
