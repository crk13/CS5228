# data_cleaned —— 清洗后的数据

本目录由 `src/data_cleaning/` 生成，只做了**与模型无关的基础清洗**：格式修正、表连接、一致性检查。
异常值处理、编码、缩放、缺失值填充、特征工程等**模型相关的处理均未做**，由各模型负责人按需处理。

## 生成与读取

在项目根目录运行：

```bash
python -m src.data_cleaning.clean    # 读取 data/，写出 data_cleaned/
python -m src.data_cleaning.checks   # 对输出做一致性检查，全部通过会打印 "All checks passed."
```

**请通过 loader 读取**，不要直接用 `pd.read_csv`。CSV 不保存类型，直接读取会把 `POSTAL_CODE` 的 `"088256"` 读成 `88256`，`BLOCK` 也可能被读成数字。

```python
from src.data_cleaning import load_train, load_test, load, NUM_COLS, CAT_COLS, TARGET, ID_COL

train = load_train()                      # 类别列为 pandas string 类型，数值列为数值类型
test = load_test()
schools = load("auxiliary/schools.csv")   # 辅助表同理
```

提交时：`pd.DataFrame({"Id": test[ID_COL], "Predicted": pred})`。

## 清洗做了哪些事

### train.csv / test.csv

| 处理 | 说明 |
|---|---|
| 字符串统一格式 | `TOWN`、`BLOCK`、`STREET`、`FLAT_TYPE`、`FLAT_MODEL` 去首尾空格、转小写、合并连续空格。原始 `STREET` 约 20% 含大写，如 `"Whampoa Drive"` |
| 统一 `FLAT_TYPE` 写法 | `"4 room"` → `"4-room"`。原始训练集 2021 年的数据基本是空格写法，之后是连字符写法，测试集只有连字符写法。统一后共 6 类 |
| 拆分日期 | 保留 `RENT_APPROVAL_DATE`（`"YYYY-MM"`），新增整数列 `RENT_YEAR`、`RENT_MONTH` |
| 删除常数列 | `FURNISHED`（全为 `"yes"`）和 `FEE`（全为 0）在 train 和 test 中都是常数，没有信息量，已删除。清洗代码中有断言，若不再是常数会报错 |
| 连接楼栋表 | 按 `(BLOCK, STREET)` 左连接 `sg-hdb-block.csv`，新增 `POSTAL_CODE`、`LATITUDE`、`LONGITUDE`、`MAX_FLOOR`、`YEAR_COMPLETED`、`SUBZONE`、`PLANNING_AREA`、`REGION`。train 和 test **100% 匹配**，且两表的 `TOWN` 完全一致（均有断言） |
| 测试集加 `Id` | 新增 `Id` 列，等于行号 0..49999，与提交文件的 `Id` 一一对应 |
| 未改动的列 | `FLOOR_AREA_SQM`、`LEASE_COMMENCE_DATE`、`MONTHLY_RENT` 的数值与原始数据完全一致，行数和行顺序也不变（`checks.py` 中有校验） |

### 辅助数据（`auxiliary/`）

| 输出文件 | 原始文件 | 处理 |
|---|---|---|
| `hdb_blocks.csv` | `sg-hdb-block.csv` | 字符串小写；`POSTAL_CODE` 的 `"NIL"` → 缺失（285 个楼栋）；`MAX_FLOOR` 和 `YEAR_COMPLETED` 的 `-1` → 缺失（11 个楼栋，对应训练集 159 行）。`(BLOCK, STREET)` 唯一 |
| `mrt_stations.csv` | `sg-mrt-stations.csv` | 字符串小写；从 `CODE` 中提取线路前缀，新增 `LINE` 列（如 `ns10` → `ns`） |
| `schools.csv` | `sg-schools.csv` | 修正列名拼写 `NATRUE_CODE` → `NATURE_CODE`；`REGION` 改名为 `ZONE`（见下文）；4 个丢失前导 0 的邮编补齐为 6 位；`PLANNING_AREA` 的 `"seng kang"` → `"sengkang"`，与其他表一致；字符串小写 |
| `shopping_malls.csv` | `sg-shopping-malls.csv` | `NAME` 转小写 |
| `coe_prices.csv` | `sg-coe-prices.csv` | `"$40,609"`、`"1,280"` 等解析为整数；月份英文缩写转为 1–12；新增 `YEAR_MONTH`（`"YYYY-MM"`）；按时间排序 |
| `stock_prices.csv` | `sg-stock-prices.csv` | 删除 85 行价格全空的记录（非交易日）；新增 `YEAR_MONTH`；新增 `MARKET`（由代码后缀判断交易所）；按 `(SYMBOL, DATE)` 排序 |

## 未处理、留给各模型负责人的现象

- **低租金样本**：训练集有 374 行 `MONTHLY_RENT < 1000`，最低 300。可能是录入错误或特殊情况，未删除。
- **完全重复的行**：清洗后训练集有 641 行、测试集有 7,764 行（不计 `Id`）与其他行完全相同。这很可能是同楼栋同属性、同月出租的不同房子，属于合法记录，未去重。
- **缺失值**：只有 `POSTAL_CODE`、`MAX_FLOOR`、`YEAR_COMPLETED` 会缺失（详见数据字典），未填充。
- **`TOWN` 与 `PLANNING_AREA` 是两套划分**：前者是 HDB 镇区，后者是 URA 规划区，不一定相同。例如 `TOWN = "kallang/whampoa"` 的楼栋，`PLANNING_AREA` 可能是 `"novena"`。
- **`LEASE_COMMENCE_DATE` 与 `YEAR_COMPLETED` 不完全一致**：差值多为 0–2 年，范围 −10 到 25。这是两个不同的口径，未做修正。
- **MRT 表**：
  - 每行是一个"站点 × 线路"，换乘站会出现多行（共 51 个同名重复）。统计"附近站点数"时应先按 `NAME` 去重。
  - 表中也包含 LRT 站（`bp`、`pe`、`ptc`、`stc` 等线路）。
  - `STATUS` 是数据采集时的快照（231 个 open，12 个 planned），**没有开通日期**。部分 open 站点可能是在 2021–2025 年间才开通的。
- **学校表的 `ZONE`**：是 MOE 分区（east / north / south / west），**不是**其他表里的 URA `REGION`，不要直接混用。这也是改名的原因。
- **股价的币种**：各股价以上市地货币计价：`sgx` 为 SGD，`hkex` 为 HKD，`us` 为 USD，`euronext` 为 EUR。部分股票的数据并不覆盖完整时间段。
- **COE**：每月通常有 2 轮竞价，少数月份的某些类别只有 1 轮。数据覆盖到 2026-08。

## 数据字典

类型说明：**数值**指可直接作为数值使用；**类别**指以字符串形式存储（读取后是 pandas `string` 类型），包括 ID 类、日期字符串类字段。

### train.csv（150,000 行 × 19 列）/ test.csv（50,000 行 × 19 列）

两表的特征列相同。train 比 test 多 `MONTHLY_RENT`，test 比 train 多 `Id`。

| 列 | 含义 | 类型 | 可能缺失 | 来源 |
|---|---|---|---|---|
| `Id` | 测试集行号，即提交文件中的 Id（仅 test 有） | 数值（int） | 否 | 新增 |
| `RENT_APPROVAL_DATE` | 租约批准年月 `"YYYY-MM"`；train 为 2021-01 至 2025-03，test 为 2025-03 至 2026-07 | 类别 | 否 | 原始 |
| `RENT_YEAR` | 租约批准年份 | 数值（int） | 否 | 由日期拆分 |
| `RENT_MONTH` | 租约批准月份，1–12 | 数值（int） | 否 | 由日期拆分 |
| `TOWN` | HDB 镇区，共 26 个 | 类别 | 否 | 原始 |
| `BLOCK` | 楼栋号，如 `"604a"` | 类别 | 否 | 原始 |
| `STREET` | 街道名 | 类别 | 否 | 原始 |
| `FLAT_TYPE` | 户型：`1-room` / `2-room` / `3-room` / `4-room` / `5-room` / `executive`。有自然顺序 | 类别 | 否 | 原始 |
| `FLAT_MODEL` | 户型设计型号，如 `improved`、`model a`、`dbss`，共 20 种 | 类别 | 否 | 原始 |
| `FLOOR_AREA_SQM` | 建筑面积（平方米） | 数值（float） | 否 | 原始 |
| `LEASE_COMMENCE_DATE` | 99 年租约起始年份 | 数值（int） | 否 | 原始 |
| `POSTAL_CODE` | 6 位邮编，保留前导 0 | 类别 | **是**（train 5,259 行，test 1,739 行） | hdb_blocks |
| `LATITUDE` | 楼栋纬度 | 数值（float） | 否 | hdb_blocks |
| `LONGITUDE` | 楼栋经度 | 数值（float） | 否 | hdb_blocks |
| `MAX_FLOOR` | 楼栋最高楼层 | 数值（int；有缺失时读作 float） | **是**（train 159 行，test 0 行） | hdb_blocks |
| `YEAR_COMPLETED` | 楼栋建成年份 | 数值（int；有缺失时读作 float） | **是**（train 159 行，test 0 行） | hdb_blocks |
| `SUBZONE` | URA 子区 | 类别 | 否 | hdb_blocks |
| `PLANNING_AREA` | URA 规划区 | 类别 | 否 | hdb_blocks |
| `REGION` | URA 大区（如 `central region`），共 5 个 | 类别 | 否 | hdb_blocks |
| `MONTHLY_RENT` | **预测目标**：月租金（SGD，仅 train 有） | 数值（int） | 否 | 原始 |

```python
TARGET = "MONTHLY_RENT"
ID_COL = "Id"  # 仅 test

NUM_COLS = [
    "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "RENT_YEAR", "RENT_MONTH",
    "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED",
]
CAT_COLS = [
    "RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL",
    "POSTAL_CODE", "SUBZONE", "PLANNING_AREA", "REGION",
]
```

使用提示（不是要求）：
- `RENT_APPROVAL_DATE` 放在 `CAT_COLS` 中，是因为它以字符串存储。要表达时间趋势时，通常用 `RENT_YEAR` 和 `RENT_MONTH`，或自行构造连续的月份索引。
- `BLOCK`、`STREET`、`POSTAL_CODE` 是高基数的 ID 类字段，直接 one-hot 编码会非常稀疏。
- `FLAT_TYPE` 有自然顺序，可以按需做有序编码。

### auxiliary/hdb_blocks.csv（9,464 行 × 11 列）

所有 HDB 楼栋的信息，比 train/test 中出现的楼栋更多，可用于空间邻居类特征。`(BLOCK, STREET)` 唯一。

| 列 | 含义 | 类型 | 可能缺失 |
|---|---|---|---|
| `TOWN` | HDB 镇区 | 类别 | 否 |
| `BLOCK` | 楼栋号 | 类别 | 否 |
| `STREET` | 街道名 | 类别 | 否 |
| `POSTAL_CODE` | 6 位邮编 | 类别 | **是**（285 行） |
| `LATITUDE` | 纬度 | 数值 | 否 |
| `LONGITUDE` | 经度 | 数值 | 否 |
| `MAX_FLOOR` | 最高楼层 | 数值 | **是**（11 行） |
| `YEAR_COMPLETED` | 建成年份 | 数值 | **是**（11 行） |
| `SUBZONE` | URA 子区 | 类别 | 否 |
| `PLANNING_AREA` | URA 规划区 | 类别 | 否 |
| `REGION` | URA 大区 | 类别 | 否 |

```python
NUM_COLS = ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED"]
CAT_COLS = ["TOWN", "BLOCK", "STREET", "POSTAL_CODE", "SUBZONE", "PLANNING_AREA", "REGION"]
```

### auxiliary/mrt_stations.csv（243 行 × 9 列）

每行是一个"站点 × 线路"，`CODE` 唯一。

| 列 | 含义 | 类型 | 可能缺失 |
|---|---|---|---|
| `CODE` | 站点编号，如 `ns10` | 类别 | 否 |
| `LINE` | 线路前缀，如 `ns`、`ew`、`dt`、`te`、`cr` | 类别 | 否 |
| `NAME` | 站名；换乘站在多行中重复出现 | 类别 | 否 |
| `STATUS` | `open` / `planned`，为数据采集时的快照 | 类别 | 否 |
| `LATITUDE` | 纬度 | 数值 | 否 |
| `LONGITUDE` | 经度 | 数值 | 否 |
| `SUBZONE` | URA 子区 | 类别 | 否 |
| `PLANNING_AREA` | URA 规划区 | 类别 | 否 |
| `REGION` | URA 大区 | 类别 | 否 |

```python
NUM_COLS = ["LATITUDE", "LONGITUDE"]
CAT_COLS = ["CODE", "LINE", "NAME", "STATUS", "SUBZONE", "PLANNING_AREA", "REGION"]
```

### auxiliary/schools.csv（337 行 × 14 列）

| 列 | 含义 | 类型 | 可能缺失 |
|---|---|---|---|
| `NAME` | 学校名 | 类别 | 否 |
| `URL` | 学校网址 | 类别 | 否 |
| `STREET` | 地址 | 类别 | 否 |
| `POSTAL_CODE` | 6 位邮编 | 类别 | 否 |
| `MRT_STATIONS` | 附近的 MRT 站，自由文本 | 类别 | 否 |
| `BUS_LINES` | 途经的公交线路，逗号分隔的文本 | 类别 | 否 |
| `PLANNING_AREA` | URA 规划区 | 类别 | 否 |
| `ZONE` | MOE 分区：east / north / south / west。原列名为 `REGION` | 类别 | 否 |
| `TYPE_CODE` | 学校类型，如 government school、independent school | 类别 | 否 |
| `NATURE_CODE` | 男校 / 女校 / 混校。原列名拼写为 `NATRUE_CODE` | 类别 | 否 |
| `SESSION_CODE` | single session / full day | 类别 | 否 |
| `MAINLEVEL_CODE` | 学段：primary（179 所）、secondary、junior college、mixed level 等 | 类别 | 否 |
| `LATITUDE` | 纬度 | 数值 | 否 |
| `LONGITUDE` | 经度 | 数值 | 否 |

```python
NUM_COLS = ["LATITUDE", "LONGITUDE"]
CAT_COLS = ["NAME", "URL", "STREET", "POSTAL_CODE", "MRT_STATIONS", "BUS_LINES",
            "PLANNING_AREA", "ZONE", "TYPE_CODE", "NATURE_CODE", "SESSION_CODE", "MAINLEVEL_CODE"]
```

### auxiliary/shopping_malls.csv（113 行 × 3 列）

| 列 | 含义 | 类型 | 可能缺失 |
|---|---|---|---|
| `NAME` | 商场名（小写） | 类别 | 否 |
| `LATITUDE` | 纬度 | 数值 | 否 |
| `LONGITUDE` | 经度 | 数值 | 否 |

```python
NUM_COLS = ["LATITUDE", "LONGITUDE"]
CAT_COLS = ["NAME"]
```

### auxiliary/coe_prices.csv（539 行 × 9 列）

拥车证（COE）竞价结果，2021-01 至 2026-08，每月通常 2 轮。`(YEAR_MONTH, ROUND, CATEGORY)` 唯一。

| 列 | 含义 | 类型 | 可能缺失 |
|---|---|---|---|
| `YEAR_MONTH` | `"YYYY-MM"`，可与 `RENT_APPROVAL_DATE` 对齐 | 类别 | 否 |
| `YEAR` | 年 | 数值 | 否 |
| `MONTH` | 月，1–12 | 数值 | 否 |
| `ROUND` | 当月第几轮竞价（1 或 2） | 数值 | 否 |
| `CATEGORY` | COE 类别 `a` / `b` / `c` / `e` | 类别 | 否 |
| `QUOTA_PREMIUM` | 该轮成交价（SGD） | 数值 | 否 |
| `PREVAILING_QUOTA_PREMIUM` | 现行价（PQP，SGD） | 数值 | 否 |
| `QUOTA` | 配额数量 | 数值 | 否 |
| `BIDS_RECEIVED` | 收到的投标数 | 数值 | 否 |

```python
NUM_COLS = ["YEAR", "MONTH", "ROUND", "QUOTA_PREMIUM", "PREVAILING_QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"]
CAT_COLS = ["YEAR_MONTH", "CATEGORY"]
```

### auxiliary/stock_prices.csv（91,557 行 × 11 列）

日频股价，2021-01-04 至 2026-07-30，共 77 只股票。`(SYMBOL, DATE)` 唯一。

| 列 | 含义 | 类型 | 可能缺失 |
|---|---|---|---|
| `NAME` | 公司名 | 类别 | 否 |
| `SYMBOL` | 股票代码，如 `D05.SI` | 类别 | 否 |
| `MARKET` | 交易所：`sgx` / `us` / `hkex` / `euronext`，决定计价货币 | 类别 | 否 |
| `DATE` | 交易日 `"YYYY-MM-DD"` | 类别 | 否 |
| `YEAR_MONTH` | `"YYYY-MM"`，可与 `RENT_APPROVAL_DATE` 对齐 | 类别 | 否 |
| `OPEN` | 开盘价 | 数值 | 否 |
| `HIGH` | 最高价 | 数值 | 否 |
| `LOW` | 最低价 | 数值 | 否 |
| `CLOSE` | 收盘价 | 数值 | 否 |
| `ADJUSTED_CLOSE` | 复权收盘价 | 数值 | 否 |
| `VOLUME` | 成交量；部分交易日为 0 | 数值 | 否 |

```python
NUM_COLS = ["OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME"]
CAT_COLS = ["NAME", "SYMBOL", "MARKET", "DATE", "YEAR_MONTH"]
```

所有文件的 `NUM_COLS` / `CAT_COLS` 也可以在代码中通过 `src.data_cleaning.schema.SCHEMAS[文件名]` 获取。
