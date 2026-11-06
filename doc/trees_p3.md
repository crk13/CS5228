# P3：树模型、辅助特征与市场指数归一化

P3 比较树集成模型对 HDB 月租金 `MONTHLY_RENT` 的预测效果，重点检验模型选择、训练历史长度、位置与经济特征，以及市场整体走势的处理。所有下述成绩均来自实际保存的时间验证预测，单位为 SGD，越低越好。

## 1. 模型与训练设置

先完成算法和训练窗口的初步比较，再固定最近 24 个月、地址字段及模型参数，对五类模型各比较六个特征阶段，共 30 组配置。这里的最佳配置指这轮固定设置下的特征比较结果，不代表穷尽历史所有参数组合后的全局最优。

| 模型 | 轮数 / 棵数 | 主要设置 |
|---|---:|---|
| CatBoost | 1000 | 深度 7，学习率 0.04，L2=10 |
| XGBoost | 700 | 深度 6，学习率 0.035，`min_child_weight=30`，L2=20 |
| LightGBM | 700 | 31 叶，学习率 0.035，`min_child_samples=60`，L2=10 |
| Random Forest | 160 | 最大深度 24，叶子最少 8 条，`max_features=0.8` |
| ExtraTrees | 160 | 最大深度 24，叶子最少 8 条，`max_features=0.8` |

使用公共框架的两个时间验证窗口；具体定义以 [接口协议](../docs/experiment-interface.md) 和 [划分实现](../src/cv.py) 为准。每折实际只拟合其训练部分中最近 24 个完整日历月。随机种子固定为 42，使用 4 个线程，不使用外部验证标签提前停止训练。参数在六个特征阶段之间保持不变；森林的 160 棵树是计算预算设置，未证明为最优数量。

类别词表和数值填补只从当前实际拟合窗口学习。CatBoost直接接收类别字符串；LightGBM、XGBoost固定训练类别词表，未见类别转为缺失。两种森林使用数值中位数填补和one-hot编码，低频类别合并，每个类别字段最多保留32个编码类别。

模型实现：[trees.py](../src/models/trees.py)。参数、输入列和原始运行配置见 [完整实验索引](../notebooks/Yaqi/experiment_catalog.csv)。个人入口在运行进程内明确固定种子42，不修改公共配置文件的默认值。

## 2. 房屋、位置与经济特征

基础输入包括面积、租约起始年份、批准月份及月份索引、房龄、剩余租期、房型和型号，以及town、street、subzone、planning area、region、楼栋经纬度、最大楼层和竣工年份。另加入邮编及楼号作为类别字段；street和block分别输入，没有使用与Ridge完全相同的拼接Block ID。树模型房龄为批准年份减楼栋竣工年份。

- **位置特征**：到最近open MRT、商场和小学的大圆直线距离，单位km；MRT排除LRT/RTS，并按坐标去重。
- **COE**：按月份和类别汇总竞价premium。
- **股票**：仅使用SGD计价的SGX股票，日收盘价先按公司取月均值，再以首月确定的固定股票集合构造等权归一指数；不混用币种，也不是官方股票指数。

六个阶段依次为base、location、location+COE、location+stocks、location+两类经济数据（持平预测）、location+两类经济数据（阻尼趋势）。经济观测先按每折截止点截断，再进行汇总和预测；假设延迟一个完整日历月可得。未来输入只使用截止点生成的持平或阻尼预测，不读取验证期实际经济数据。

特征实现：[distance.py](../src/features/distance.py)、[macro.py](../src/features/macro.py)。完整分阶段结果及解释见 [特征报告](../notebooks/Yaqi/FEATURE_REPORT.md) 和 [森林补充报告](../notebooks/Yaqi/FOREST_REPORT.md)。

## 3. 本地验证结果

以下是30组受控特征实验中，各算法平均RMSE最低的原租金配置。A/B平均为两个RMSE的简单平均。两个验证窗口重叠，并且参与了配置选择，属于开发验证结果。

| 模型 | 选定特征阶段 | A | B | 平均 |
|---|---|---:|---:|---:|
| CatBoost | 位置＋COE | 492.874 | 476.168 | 484.521 |
| XGBoost | base | 495.948 | 475.680 | 485.814 |
| LightGBM | 位置＋COE | 501.918 | 482.572 | 492.245 |
| Random Forest | 位置 | 503.187 | 483.193 | 493.190 |
| ExtraTrees | 位置 | 502.874 | 484.617 | 493.746 |

CatBoost均值最低，XGBoost其次。因为模型使用各自的编码和选定特征，以上比较对应完整建模方案，不能将全部差异只归因于算法。

在CatBoost参数不变时，位置特征把平均RMSE从485.463降至484.559；再加入COE只降至484.521，增益很小，而且A折略差、B折较好。股票和两类经济数据组合未优于位置＋COE，阻尼外推也没有改善最佳简单配置。XGBoost在本轮base阶段最好；森林加入位置的收益也较小。更多辅助变量并不必然提高预测精度。

原始表格：[afternoon_comparison.csv](../notebooks/Yaqi/afternoon_comparison.csv)。

## 4. 固定房型指数：小幅补充实验

为减轻租房市场整体走势对目标变量的影响，进一步用固定房型构成的月度指数调整历史租金。指数只在每折实际训练窗口内估计；未来保持最后训练月水平。1-room样本不参与指数计算，但其训练记录保留。

| 模型 | 原租金平均RMSE | 指数版平均RMSE |
|---|---:|---:|
| CatBoost | 484.521 | 482.837 |
| XGBoost | 485.814 | 483.155 |
| LightGBM | 492.245 | 490.274 |

三个模型的A/B两折均改善，但平均降幅仅约0.35%–0.55%，因此作为小幅修正，不视为主要收益来源。实现见 [fixed_mix.py](../src/models/fixed_mix.py)，配置、逐行预测、指数和独立复核见 [实验归档](../notebooks/Yaqi/fixed_mix_20261104/README.md)。

## 5. 产物与复现

| 内容 | 入口 |
|---|---|
| 历史与当前实验总入口 | [notebooks/Yaqi](../notebooks/Yaqi/README.md) |
| 早期算法、窗口和地址字段比较 | [早期13组实验](../notebooks/Yaqi/history/20261022/README.md) |
| 五类模型、六阶段对比 | [30组结果](../notebooks/Yaqi/afternoon_comparison.csv) |
| 供融合使用的三个正式原租金模型 | [results/Yaqi](../results/Yaqi/README.md) |
| 指数归一化对照及CatBoost测试集预测 | [固定房型指数归档](../notebooks/Yaqi/fixed_mix_20261104/README.md) |

正式交付仍是CatBoost（位置＋COE）、XGBoost（base）、LightGBM（位置＋COE）三个原租金运行，附训练诊断、验证和测试预测。指数版属于额外实验归档，目前仅CatBoost生成最终测试预测，未替换上述三个正式融合运行。选择三个算法代表时依据单模型验证均值，未声称已获得最优融合组合。融合读取规则以 [公共接口协议第6.1节](../docs/experiment-interface.md#61-交付结果给融合) 为准。

在项目根目录、已有Python环境中复现；直接依赖与实际安装版本见 [个人依赖说明](../notebooks/Yaqi/requirements.in)、[版本记录](../notebooks/Yaqi/requirements.txt)。以下前两条只核对现有产物，后两条启动新训练：

```powershell
python notebooks/Yaqi/verify_delivery.py
python scripts/verify_published_fixed_mix.py
python notebooks/Yaqi/train_experiments.py --suite features --names catboost_location_coe xgboost_base lightgbm_location_coe
python scripts/train_fixed_mix_model.py --kind catboost
```

指数脚本的 `--kind` 也支持 `xgboost` 和 `lightgbm`。新实验写入新的 `outputs/Yaqi/` 子目录；正式交付和历史归档不会被自动覆盖。输入数据及测试集原行序保留，不提交训练权重。

## 6. 局限与解读

- 树模型不能直接外推训练范围之外的时间趋势；指数版仍假设未来市场水平持平。
- 设施文件缺少历史启用日期，位置特征是文件快照；经济数据的一个月可得性延迟也是明确假设。
- 超参数搜索有限，没有在加入每组特征后联合重调全部参数，不能证明当前配置为最优。
- 两个验证窗口重叠且参与选型，小幅差异不等于统计显著提升，也不能保证未来数据表现。
- 低租金记录和地区间差异仍会造成较大误差；未因标签较低而删除原记录。
- 已有解释性检查包括置换重要性和模型原生重要性；尚未开展SHAP分析，不将相关性或重要性解释为因果作用。
