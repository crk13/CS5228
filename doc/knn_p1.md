# P1：KNN 基础模型

本文记录 P1 交付给融合的三个 KNN 运行：模型结构、特征及取舍理由、搜索过程和结果。公共框架的接口和交付规则只以 [接口协议](../docs/experiment-interface.md) 为准，这里不重复。融合方法见 `p1_fusion` 分支的 `doc/fusion_p1.md`。

## 1. 结论

| 运行 | 名称 | Holdout | Rolling 2024-04 | 平均 RMSE |
|---|---|---:|---:|---:|
| `20261103T084841045522Z_9032eefb` | knn_basic | 510.39 | 494.23 | 502.31 |
| `20261103T095144201723Z_6c94ac5f` | knn_v2 ridge_residual + add_age_floor | 504.38 | 488.87 | 496.62 |
| `20261104T110517176853Z_913a37c2` | knn_v3 aux | **500.04** | **484.52** | **492.28** |

- 三个运行都按协议写在 `results/P1/`，用 `split="rolling"`、`final=True`。
- 融合只用 v3（`knn3`）。v1、v2 留作改进过程的记录。
- KNN 单独看不如树模型和神经网络，但它的误差模式不同：它按"相似房子的实际成交价"来预测。在融合的第一层里，它的权重约为 0.13。

## 2. 模型结构

代码在 [src/models/knn.py](../src/models/knn.py)，入口是 `knn_model`（v1）和 `rent_knn_model`（v2、v3）。

### v1：纯 KNN

- 特征：`LATITUDE`、`LONGITUDE`、`FLOOR_AREA_SQM`、`FLAT_TYPE_ORDINAL`、`REMAINING_LEASE`、`MONTH_INDEX`。
- 预处理：缺失值用中位数填充，然后标准化。
- 模型：k=80，按距离倒数加权。
- `MONTH_INDEX` 作为一个距离维度，让预测时间较晚的房子更多地匹配最近的训练记录。

### v2、v3：Ridge 打底 + KNN 修正残差（`RentKNNRegressor`，`base="ridge"`）

1. **Ridge 给出主预测。**
   - 类别变量 `FLAT_TYPE`、`FLAT_MODEL`、`TOWN` 做 one-hot。
   - 数值变量先标准化。
   - alpha=1。
2. **KNN 只学 Ridge 的残差。**
   - 在标准化后的 KNN 特征上找 k=80 个邻居，按距离倒数加权，把它们的平均残差加回 Ridge 预测。
3. **为什么这样设计。**
   - 纯 KNN 在样本稀疏的地方（少见的户型、偏远的楼盘）会去借远处房子的价格，偏差大。
   - Ridge 先把"户型、面积、地区"这些主要规律学好，KNN 再负责 Ridge 学不到的局部细节，比如同一片区里个别楼盘偏贵。
4. **类别变量的处理。**
   - 无序类别只进 Ridge（one-hot），不进 KNN 的距离计算，因为 one-hot 放进欧氏距离里很难定权重。
   - KNN 只用有序的 `FLAT_TYPE_ORDINAL` 和数值特征。

**v2 与 v1 的差别**：v2 换成 Ridge 打底，KNN 特征里加了 `FLAT_AGE` 和 `MAX_FLOOR`。

**v3 与 v2 的差别**：加入了辅助数据特征，见第 3 节。

| 部分 | v3 最终设置 |
|---|---|
| KNN 特征 | `LATITUDE`、`LONGITUDE`、`FLOOR_AREA_SQM`、`FLAT_TYPE_ORDINAL`、`REMAINING_LEASE`、`MONTH_INDEX`、`FLAT_AGE`、`MAX_FLOOR`、`AUX_MRT_KM` |
| Ridge 数值特征 | `FLOOR_AREA_SQM`、`REMAINING_LEASE`、`AUX_MRT_KM`、`AUX_MALL_KM`、`AUX_CBD_KM` |
| Ridge 类别特征 | `FLAT_TYPE`、`FLAT_MODEL`、`TOWN` |
| 超参数 | k=80，按距离倒数加权，Ridge alpha=1，目标不做变换 |

## 3. 辅助数据特征（`p1_aux`）

代码在 [src/features/p1_aux.py](../src/features/p1_aux.py)。数据来自 `data_cleaned/auxiliary/`，说明见 [清洗数据说明](../data_cleaned/README.md)。距离一律用 BallTree 按球面（haversine）距离计算，所有特征都不用到租金本身。

| 组 | 列 | 含义 | 来源 |
|---|---|---|---|
| mrt | `AUX_MRT_KM`、`AUX_MRT_N1KM`、`AUX_MRT_LINES_1KM` | 最近 MRT 站距离、1 km 内站数、1 km 内线路数 | `mrt_stations.csv` |
| rail | `AUX_RAIL_KM` | 最近轨道站距离，含 LRT | `mrt_stations.csv` |
| cbd | `AUX_CBD_KM` | 到 Raffles Place 站的距离 | `mrt_stations.csv` |
| mall | `AUX_MALL_KM`、`AUX_MALL_N1KM` | 最近商场距离、1 km 内商场数 | `shopping_malls.csv` |
| school | `AUX_PRI_KM`、`AUX_PRI_N1KM` | 最近小学距离、1 km 内小学数 | `schools.csv` |
| hdb | `AUX_HDB_N500M` | 500 m 内组屋楼栋数（密度） | `hdb_blocks.csv` |
| macro | `AUX_COE`、`AUX_SGX_INDEX`、`AUX_PROPERTY_INDEX` | 上个月的拥车证价格、新加坡股指、地产股与 REIT 价格指数 | `coe_prices.csv`、`stock_prices.csv` |

需要说明的处理：
- **MRT 只用 2025 年前已运营的线路**（NS、EW、CG、NE、CC、DT、TE）。站点表只是一个快照，没有开通日期，还把 CRL、JRL 等未来线路标成了开通。如果直接用，会让训练期的房子"提前"享受到还没开通的站点。
- **宏观指标滞后一个月**：第 m 个月用第 m−1 个月的值，避免用到当月才公布的数据。

### 各组贡献与取舍

做法是以 v2 的设置为基准（平均 RMSE 496.62），每次只加入一组特征，分别放进 KNN、Ridge 或两者都放，记录平均 RMSE（越低越好）。结果见 [contrib.csv](../notebooks/P1/knn_v3/contrib.csv)：

| 组 | 最好的放法 | 平均 RMSE | 相对基准 | 取舍 |
|---|---|---:|---:|---|
| mrt | KNN+Ridge | 493.13 | −3.49 | 采用，提升最大 |
| 全部空间特征 | Ridge | 493.21 | −3.41 | 只在 Ridge 有效；放进 KNN 会稀释距离 |
| rail | KNN+Ridge | 494.11 | −2.51 | 和 mrt 高度重复，贪心选择时没有入选 |
| mall | KNN+Ridge | 495.44 | −1.18 | 采用 `AUX_MALL_KM`，放进 Ridge |
| school | KNN | 495.69 | −0.93 | 没有入选 |
| cbd | KNN+Ridge | 496.03 | −0.59 | 单独看收益小，组合后放进 Ridge 有效（见第 4 节剔除检验） |
| hdb | KNN | 496.29 | −0.33 | 没有入选 |
| macro | KNN | 500.05 | **+3.43** | **不用** |
| macro | Ridge | 574.32 | **+77.7** | **不用** |

**宏观数据为什么不用。** 它们每个月对所有房子取同一个值，对模型来说其实就是"这个月大盘怎么样"的信号。训练期里它们和租金走势有相关性，但到了验证期就失效了。Ridge 会把它当成线性趋势往外推，误差暴涨到 574。不滞后（`macro lag0`）也一样差。结论是：这些宏观指标对未来几个月的租金大盘没有可靠的预测能力，所以不用。这也是融合阶段要单独处理"整体水平"的原因。

## 4. 搜索过程

脚本是 [knn_v3_search.py](../notebooks/P1/knn_v3_search.py)，各阶段结果在 [notebooks/P1/knn_v3/](../notebooks/P1/knn_v3/)。每组配置都完整跑一遍 rolling 两折，以平均 RMSE 为准。中途中断后重跑，会跳过已经算过的配置。

| 阶段 | 做什么 | 最好结果 |
|---|---|---:|
| contrib | 每组辅助特征单独加入，见上表 | 493.13 |
| greedy_ridge / greedy_knn | 分别给 Ridge 和 KNN 逐个贪心添加特征（按接受规则采纳的步骤） | 493.20 / 493.03 |
| combine | 两边的最优组合放在一起 | **492.28** |
| hyper_a / hyper_b | k、加权方式、alpha、宏观滞后、经纬度换算成公里、按户型分别建 KNN | 492.21 |
| refine / polish | 在最优配置附近微调，再逐个添加剩余特征 | 492.21 / 492.25 |
| dropcheck | 逐个删去最终配置里的特征，检查是否都有用 | 删任何一个都变差（0.07 到 1.86）；删 `MONTH_INDEX` 会升到 649.53 |

**接受规则**：后续阶段的改动，平均 RMSE 至少要降 0.3，并且两折都要变好，才会被采用。refine 和 polish 中最好的改动（alpha=10、k=120、加 `FLAT_AGE` 等）只降了 0.03 到 0.07，属于噪声，所以最终交付的是 combine 阶段的配置。

v2 的单项消融见 [knn_v2_ablation.csv](../notebooks/P1/knn_v2_ablation.csv)：

| 改动 | 相对 v1 | 结论 |
|---|---:|---|
| Ridge 打底，KNN 修正残差 | −3.34 | 采用 |
| 加 `FLAT_AGE`、`MAX_FLOOR` | −1.87 | 采用 |
| 两者组合 | **−5.68** | 即 v2 |
| 按单位面积租金建模 | +6.94 | 不采用 |
| 预测时把月份截断到训练期最后一个月 | +18.81 | 不采用：距离加权下会只剩完全重合的点 |
| 按大盘水平做比例（除以月均值） | +0.90 | 不采用 |
| 经纬度换算成公里 | +0.20 | 不采用 |

## 5. 复现

在 `p1_knn` 分支、项目根目录下运行：

```bash
python notebooks/P1/knn_run.py deliver      # v1；grid 子命令重跑 holdout 网格
python notebooks/P1/knn_v2_run.py deliver   # v2；ablation 子命令重跑消融
python notebooks/P1/knn_v3_search.py contrib   # v3 各阶段依次运行，阶段顺序见脚本开头的说明
python notebooks/P1/knn_v3_search.py deliver   # 把最终配置交付到 results/P1/
```

## 6. 局限

- **整体水平。** 验证期内，2024-07 租金整体跳涨了约 2%。所有模型（包括 KNN）在之后的月份都系统性偏低，没有哪种辅助数据能预测这次跳涨。KNN 预测的整体水平和树模型接近（test 均值约 3139）。这个问题留到融合阶段统一处理。
- **调参用的就是验证折。** rolling 两折既用来选配置，又用来报告分数，所以这些分数是开发成绩，不是无偏的测试估计。
