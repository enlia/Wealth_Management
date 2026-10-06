# 数据源踩坑

> 记录**外部数据源 API** 的行为陷阱。
> 与 `PITFALLS.md`（工程/工具链）、`DATA_QUALITY.md`（判据设计）分开——
> 这三类的排查手法完全不同：
>工程看代码，判据看反例，数据源看**接口实际返回**。

---

## S1 Tushare 单次调用有页上限，超了不报错 🔴

**现象**：接口返回行数「看起来很合理」，但**是截断的**。

| 接口 | 无翻页 | 加翻页后 | 倍数 |
|---|---|---|---|
| repurchase | 2,000 | **63,755** | 32× |
| index_basic | 8,000 | **12,535** | 1.6× |
| namechange | 10,000 | **13,889** | 1.4× |

实测存在三种页大小：2,000 / 8,000 / 10,000。
**从返回行数看不出来**——「2,000行」和「1,500 行」的差别只在于是否碰巧被截断。

**最恶劣的案例**：`disclosure_date` 无参翻页 60 页拿到 12 万行，
但数据**停在 2016-04**，近 10 年全缺。行数完全正常，不核查就发现不了。

**修法**：
```python
# research/scripts/tushare_paging.py
first = call(token, api, {**params, "limit": 2000, "offset": 0})
if len(first) < 2000:      # 不足一页→ 没有下一页
    return first
# 否则继续按 offset 翻页
```

**`offset` 是行偏移，不是页号**：
```
{'limit': 2000, 'offset': 2000}   → 第二页（从第 2001 行开始）
```
⚠️ 第一页也必须带统一 `limit`。若首页不带 limit 返回 8,000 行，
第二页就要用 `offset=8000`；误用 `offset=2000` 会拿到重复数据。

**撞页数上限必须抛错**（`fetch_paged_strict`）：
静默返回残缺数据比报错危险得多——流程会以为「下载完成」，
后面用它做因子才发现样本期严重偏短。

**更大的数据量要按时间分段**，不要无限翻页（offset 越大越慢）。
实测按自然年分段 12 次请求就能拿到全量。

---

## S2 Tushare 参数传错静默返回 0 行 🔴

**不报错、不抛异常，就是给你 0 行。**

| 接口 | 错误用法 | 正确用法 |
|---|---|---|
| `dividend` | `{}` / `{'end_date': ...}` → 0 行 | 必须逐只 `{'ts_code': '600519.SH'}` |
| `ggt_top10` | `{'start_date':..., 'end_date':...}` → 0 行 | 必须逐日 `{'trade_date': '20240102'}` |
| `index_classify` | 无 `src` → 359 行（旧版分类） | `src=SW2021` → 511 行 |

第三个最阴险：**不返回 0 行，返回的是旧版数据**，你不会察觉。

**规则：每个任务跑完必须打印实际行数。**
行数 = 0 要当**失败**处理，不是「正常为空」。
`fetch_all_tushare.py` 已内置这个检查。

---

## S3 接口权限会变，禁止沿用历史探测结论

2026-10-05 探测判为不可用的两个接口，2026-06 复测**都能用了**：
- `suspend_d`（停复牌）
- `top_list`（龙虎榜）

**规则：每次补数据前重新实测，不要信上次的探测报告。**
本次重新探测 40 个接口，有数据 31 / 空返回 2 / 权限不足 7。

---

## S4 「文件存在」不等于「下载完成」

**原逻辑**：`if (OUT / f"{tag}.parquet").exists(): skip`

**问题**：中断时 `run_task` 会写 `{tag}_partial.parquet`，
但下次跑只判断 `{tag}.parquet` 存在与否——如果两者同名或
清理不彻底，半成品会被当成完整数据。

**修法**：用**下载账本** `runtime/tushare/_download_manifest.json`
记录每个任务是否真正完成（含行数、耗时、时间戳）。
账本损坏时必须 raise，不能静默当成「全部重跑」。

---

## S5 去重键必须包含区分实体的字段

**原逻辑**：`drop_duplicates(subset=['ts_code', 'period', 'end_date', 'index_code', 'trade_date'])`

**问题**：`index_weight` 同一个 `trade_date` 有多个指数的成分。
`index_code` 在该接口的返回里可能缺失或命名不同，
导致沪深300 和中证500 同一天的权重被合并。

**修法**：按区分度从高到低逐个加键，直到行唯一：
```python
candidates = ["ts_code", "index_code", "con_code", "holder_name",
              "ann_date", "period", "end_date", "month", "trade_date"]
keys = []
for c in candidates:
    keys.append(c)
    if not df.duplicated(subset=keys).any():
        break
df = df.drop_duplicates(subset=keys, keep="last")
```

---

## S6 「碰巧没截断」不等于「保证不截断」

验证时发现：`stk_limit` 单日翻页后总计 5,377 行，
与首页返回的**完全一致**——说明这个接口单页就够，
昨天下载的 `daily_basic` P0 数据是完整的。

但这只能证明**那一个接口那一天的巧合**。
正确的做法仍是所有接口统一走翻页逻辑，
而不是「测几个觉得够了就跳过」。

---

## S7 硬编码优先级列表会掩盖新任务

`--list` 里写死 `for cur in ("P0", "P1", "P2")`，
加了 P3/P4 任务后清单里**根本不显示**，
看列表会以为「这些任务不存在」。

**规则：枚举值从数据源动态取**。
```python
for cur in sorted({s["p"] for s in TASKS.values()}):
```

同类问题：任何「硬编码的枚举清单」都会在数据变化时静默失效。

---

## S8 「同主键不同值」多半不是重复，是多次披露 🔴

**现象**：按 `(ts_code, end_date)` 去重时，几万到十几万行「重复」。

**真相**：这是财务数据与股东数据的**本质特性**，不是错误。

| 字段 | 含义 | 造成的「重复」 |
|---|---|---|
| `update_flag` | 0=原始披露 1=更新公告 | 同一报告期有多个公告版本 |
| `report_type` | 1~4 合并/母公司报表 | 同一主体有4 种报表 |
| `ann_date` | 公告日 | 修正公告与首次披露不同日 |

实测：
```
income         396,253 行 → 去掉多版本后 323,973（-72,280）
balancesheet   382,541 → 267,494（-115,047）
cashflow       376,654 → 292,429（-84,225）
fina_indicator 429,236 → 248,605（-40,334）
```

`top10_holders` 更清楚：
```
002150.SZ  20260630 报告
  20260703 首次披露  hold_float_ratio = NaN
  20260818 修正公告  hold_float_ratio = 1.2305
```

**关键区分：**

| 类型 | 判定 | 处理 |
|---|---|---|
| **整行完全相同** | `drop_duplicates()` 前后列全一样 | 是下载器拼接问题，**必须去重** |
| **同主键但值不同** | 至少一列有差异 | 真实多版本，**要保留** |

**主键必须包含版本维度**：`ann_date`、`update_flag`、`report_type`。

⚠️ 实测 `fina_indicator` 两种情况**同时存在**：
14 万行是整行重复（下载器问题），4 万行是真多版本。
一把梭全去重会丢掉「修正幅度」这个信号 ——
而业绩超预期因子里，修正幅度本身就是因子。

**做法**：整行去重 → 另存 `_clean` 取最新版 → **保留原表**。

---

## S9 存量数据不能用 START 过滤

`START=20151201` 是为了控制行情数据的规模，
但对「存量类」数据（历史更名、股权质押等）会丢历史。

实测 `namechange`：无参 13,889 行中，**7,517 行是 1990~2014** 的历史更名。
按 2015 起分段后只剩 6,373 行，**丢了 72%**。

任务表要支持 `start_year` 显式往前推。

反向的例子：`stock_company` 是**静态数据**，按年分段反而把
同一份公司简介重复 12 次（75,528 = 6,294 × 12）。

**判断标准：数据是「随时间累积」还是「静态快照」？**
- 累积型（更名、质押、解禁）→ 按年分段，`start_year` 往前推
- 快照型（公司简介、指数列表）→ 一次拉完 + 翻页

---

---

## S10 关键词法在金融字段上不可用 🔴 **两次都踩**

判断「某列是不是金额」时，初版用列名关键词猜 —— **误判率极高**。

**第一版：含金额词就换算**
```
assets_turn（资产周转率）    含 assets → 被误换算
assets_yoy（资产同比增速）  含 assets → 被误换算
debt_to_assets（资产负债率）含 assets → 被误换算
bps（每股净资产，元/股）  必须 /1e8 → 实际没换
fina_indicator 一张表误判 5 列
```

**第二版：手写白名单** —— 30 个列名是**凭记忆写的、实际不存在**
（`accounts_payable` 实际叫 `acct_payable`、`currency_borr` 实际叫 `cb_borr`…），
而真实存在却漏写的金额列会**静默漏换算**，后果与不换算完全一样。

**第三版（可用）：脚本从真实 parquet 推导**
```python
# build_money_whitelist.py —— 接口版本变化时重跑，不手写
for c in df.columns:
    if c in NOT_MONEY_EXACT:            # 显式排除
        continue
    if NOT_MONEY_PATTERNS.search(c):    # 排除规则只用于「确定不是金额」
        continue
    if df[c].dtype.kind in "if":# 只取数值列
        cols.append(c)
```

**核心教训：关键词只能用来「排除确定不是金额的」，
不能用来「判断是金额」。** 金融字段的命名无法可靠推断。

**排除规则要覆盖前缀形式** ——实测踩过：
`^eps$` 匹配不到 `basic_eps` / `diluted_eps`，两者都是每股指标，
误/1e8 会变成 2e-6。

⚠️ 生成后仍需人工抽查：白名单挡不住「名字里没有金额词但单位是元」的字段
（`ebit` / `ebitda` 就是这种）。

---

## S11 日期列存成 string —— 最隐蔽的一类静默失效 🔴 **全库20 张表中招**

**实测**：2026-10-06 扫描全部 parquet，发现 **20 张表、2,250 万行**
的 `trade_date` / `ann_date` / `end_date` 等日期列存成 **string**。

**根因**：Tushare 返回的日期是 `'20240102'`（字符串），
`to_parquet` 原样存成 string 列。**没有任何报错。**

**后果** —— 静默失效，不报错：
```python
df[df['trade_date'] == 20240102]        # int 比较 → 0 行
df[df['trade_date'] == '20240102']      # 只有这个写法能匹配
```
下游并库、筛选、join 全部拿到空结果，看起来像「那天没数据」。

实测踩过：用 `stk_limit` 算涨跌停约束时，`trade_date == 20240102`
返回 0 行，**全表11,473,527 行一行的涨跌停价都没用上**。

**修复**：
1. 落盘入口统一归一（`tushare_paths.normalize_types`，挂在 `save()` 上）
   —— **换算只在数据入口做一次**，下游各自转是「东拼西凑」
2. 存量迁移：`research/scripts/migrate_date_types.py`
   （先备份 → 转换 → **校验行数不变** → 不一致自动回滚）

**⚠️ 必须用 pandas 可空的 `Int32`，不能用 numpy `int32`**：
`stk_managers.end_date` 有 73,575 个空值（在任高管本就无离任日期，
是**业务语义**不是数据损坏），numpy int32 存不了 NA，
整表迁移直接失败被回滚。

**回归防护**：`tests/test_tushare_types.py::TestStockedParquet`
直接扫真实文件，新下载的表若再退化成 string 立即失败。

---

---

---

## S14 运行中的长任务不会自动获得代码修复 ⚠️ **时序陷阱**

**实测**：`moneyflow.parquet`（895MB，P1 任务于 12:12 写入）
在`normalize_types` 上线后**仍然是 string 类型**。

**根因**：下载进程在修复前就已启动，Python 在启动时就把
`tushare_paths` 模块载入内存。**改代码不影响已运行的进程。**

**这是个静默失效**：文件正常下载完成、manifest 记了账本、
没有任何错误，但 `df[df.trade_date == 20240102]` 恒返回 0 行。

**规则**：
1. **改了落盘逻辑，必须重跑受影响的任务，或对产出做迁移**。
   代码改了 ≠ 运行中的进程用了新代码。
2. **质量门禁必须扫全库真实文件**，而不是「检查下载器代码是否正确」。
   `tests/test_tushare_types.py::TestStockedParquet` 就是为此存在 ——
   它发现了一个 review 与人工检查都没抓到的问题。
3. 判断文件是否是修复前写入的：看 `mtime` 是否早于修复提交时间。

---

---

## S15 跨年预热链有4 处断点，每处都静默丢数据 🔴 **判据级**

**实测**：`pos250` 因子覆盖率 **0.0%**，逐年排查发现 4 个独立缺陷。
每一个都不报错、不抛异常，只是**安静地丢掉预热数据**。

### 断点 1：预热天数按自然日算

```python
# ❌ 初版
ts = pd.Timestamp(d) - pd.Timedelta(days=260)     # 260 自然日 ≈ 173 交易日
# ✅ 修复
ts - pd.tseries.offsets.BDay(260)                   # 260 交易日
```
`pos250` 需要 **250 个交易日**，自然日只给 173 个 ⇒ 少 77 天。
实测 `_shift_date('20240101', -260)` 返回 `20230416`，正确是 `20230102`。

### 断点 2：下界被 max() 夹回本年元旦

```python
# ❌ lo = max(ys, start[:8])
```
`start` 可能是**预热起点**（如 20180102），而 `ys` 是本年元旦，
`max()` 会把下界夹回 1 月 1 日 ⇒ 预热起点失效。
正确：`lo = start[:8]`（它已是更早的预热起点）。

### 断点 3：首年完全不预热 ⭐ 最隐蔽

```python
# ❌ if y > y0: pad_start = _shift_date(...)
#    else:      pad_start = ys                    # 首年无预热
```
实测 `build_panel(..., '20190101', '20191231')`：
`pos250` 覆盖率 **0.0%**、`mom120_skip20` 7.5%。
长窗口因子在首年必然大面积 NaN ——
表现为「因子没数据」，很容易被误判成「该股不合格」。

### 断点 4：算因子前就裁掉了预热行 ⭐ 真正根因

```python
# ❌ 先裁剪，再算因子
long = long[(long["date"] >= lo) & (long["date"] <= hi)]   # 预热行在这里没了
s = compute_factor(f, long)                                 # 因子看不到预热
```
正确顺序：**先去重 → 再算因子 → 最后裁剪结果**。

### 修好 1~3 之后才暴露的连带问题

`years` 按pad_start 实际落点算后，相邻年份段的预热区间**会重叠** ——
同一 (code, date) 出现两次，`unstack` 直接抛
`ValueError: Index contains duplicate entries`。

**必须在算因子之前**按 `(code, date)` 去重：
```python
long = long.sort_values(["code","date"], kind="stable") \
           .drop_duplicates(subset=["code","date"], keep="last")
```
⚠️ `_prep` 内部排序但**不去重**，必须显式处理。

### 判定规则

1. **长窗口因子的覆盖率必须单独检查**，不能只看整体平均。
   实测整体覆盖率 72.1% 看着正常，但 `pos250` 在 2019 年是 **0.0%**。
2. **「因子没数据」不等于「该股不合格」** ——
   先查是不是窗口不够、预热不足。
3. **修复「样本量不足」类问题时，把每个断点单独修完再验证** ——
   修好一个只暴露下一个，不通读代码根本发现不了断点 1 和 2 的存在。

---

## S13 取数缺依赖列表现为「静默不跑」，不是报错 🔴 **判据级**

**实测**：批量跑 11 个因子的滚动检验，报
`KeyError: 'Column not found: vol'`，整个扫描中断。

但真正的问题不是这次崩溃 —— 是
**`volratio5_60`（依赖 `vol`）与 `amount20`（依赖 `amount`）
从未在滚动检验里跑过**。取数 SQL 硬编码了 OHLC+close_adj，
两列从来没被取出来过，所以这两个因子**一次都没成功跑过**。

**为什么危险**：批量跑时表现为「跑到某因子就崩」，
容易被误判为「这个因子有 bug」，实际是**取数层缺依赖**。
更糟的情况是因子被跳过而非崩溃 —— 扫描表里看不到它的任何痕迹，
**不会报错，只是什么都没发生**。

**规则**：
1. **取数列必须由因子依赖推导**，不能硬编码；
   或者至少显式声明每个因子需要哪些列，并在取数前断言。
2. **批量研究必须配一个「所有因子都能跑」的冒烟测试** ——
   `tests/test_selection.py::TestLongOnlyDataDeps` 用真实库列名
   逐个因子试算，任何一个返回空或抛错都直接失败。
3. **冒烟测试判据要实跑，不能 grep 源码**：
   我自己写错过一次 —— grep `cols = [...]` 只匹配到基础列表，
   漏了后面的 `cols += [...]`，报了**假失败**。
   诊断工具的可信度不高于被诊断代码（见 S12）。

---

## S12 「诊断打印」本身会成为假象来源 🔴 **判据级**

**实测**：因子面板每年打印「1 格」，一度被判定为
「面板损坏、跨年拼接失效」，差点推倒重写取数逻辑。
实际面板有 2,600 日期 × 5,600 只 = **1,460 万格**，完全正常。

**根因**：`sum(len(x) for x in acc.values())` ——
`len(DataFrame)` 返回的是**列数**，单因子面板永远打印「1 格」。
面板是好的，**打印是错的**。

**为什么危险**：诊断工具的可信度**不高于**被诊断代码。
一个错的打印会引导出「数据有问题」的误判，
然后在完好数据上做无意义的修复。

**规则**：
- 所有进度/统计打印，数字必须**明确标注单位与算法**
- 涉及形状的统计一律用 `.shape`，**禁止用 `len(df)`**
- 「数据有问题」的结论，必须有**第二条独立路径交叉验证**后才动手

---

## 相关模块

| 内容 | 位置 |
|---|---|
| 任务表（P1~P4 + 不可用记录） | `research/scripts/tushare_tasks.py` |
| 分页拉取 | `research/scripts/tushare_paging.py` |
| 下载执行器 | `research/scripts/fetch_all_tushare.py` |
| 下载账本 | `runtime/tushare/_download_manifest.json` |
| 质量校验 | `research/scripts/verify_tushare_data.py` |
| 复权因子专项校验 | `research/scripts/verify_adjusted_data.py` |
| **全量质量门禁** | `research/scripts/verify_tushare_full.py` |
| **数据清洗** | `research/scripts/clean_tushare_data.py` |
| **并入本机库** | `research/scripts/merge_tushare_tables.py` |
| **金额列白名单生成** | `research/scripts/build_money_whitelist.py` |
| **日期类型归一（S11）** | `research/scripts/tushare_paths.normalize_types` |
| **存量日期类型迁移** | `research/scripts/migrate_date_types.py` |
| 工具链测试 | `tests/test_tushare_pipeline.py` |
| 类型归一测试 | `tests/test_tushare_types.py` |
| 取数依赖冒烟测试（S13） | `tests/test_selection.py::TestLongOnlyDataDeps` |
| **跨年预热测试（S15）** | `tests/test_selection.py::TestWarmupWindow` |

---

> 外部 API 的行为**只能实测**。文档写「支持分页」不代表默认就翻页了，
> 文档写「需要 5000 积分」不代表 2000 积分完全拿不到（S2 的 `index_classify`）。
> 每条规则都配了实测数字，改动前先重测，别信上一轮的结论。