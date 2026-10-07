# Tushare 全量下载 · 施工方案

> 版本：v1（2026-10-07）｜状态：待主会话评审后分批执行
> 数据截止：本文所有"实测"数字产生于 2026-10-07（探测/冒烟）与 2026-10-06 前（项目既有记录）
> 本文只讲"要做什么、怎么施工、怎么验收"，细节证据在 `.planning/reports/tushare-data-plan.md`
> **本内容不构成投资建议。**

---

## 0. 一句话摘要

Tushare（数据服务商）上本 token 能拿的数据共 57 个接口（实测可用），
目标是全部下载、合理落库（结构化进 `market.db`、大原始包进 `runtime/tushare/` parquet）、
并全部通过 6 道质检门。财务四表（利润表/资产负债表/现金流量表/财务指标）
**已入库、覆盖面完整**（实测 26~33 万行/表、5,591 只、历史到 1990/2001 年），
真正的缺口是：已下载未入库的日频大表、复权因子全覆盖、指数成分多期历史、
以及约 30 个事件/情绪/静态/宏观接口整体缺失。

### 术语速查（第一次出现的人话翻译）

| 术语 | 人话 |
|---|---|
| upsert 键 | 入库去重键：同键数据重复导入时"替换旧值"而不是堆重复行 |
| 质检门 | 一批数据入库前后必须通过的一道自动检查，不过就不算完成 |
| 三要素台账 | 每份数据记录"来源 + 抓取时间 + 覆盖范围"的清单表 |
| 静默截断 | 接口只返回一部分数据但不报错，行数看着正常（本项目 S1 号坑） |
| 静默 0 行 | 参数/权限问题导致返回 0 行但不报错（P5 号坑） |
| 复权因子 | 把除权除息后的价格还原成可比序列的换算系数 |
| 多期历史 | 同一份名单（如指数成分）保留历史上每一期的版本，而不是只有最新一期 |
| by_date / by_stock | 拉取方式：按交易日逐日拉（推荐）/ 按股票逐只拉（只在接口强制时用） |

---

## 1. 现状盘点（先量后写，全部实测）

### 1.1 market.db（SQLite 2.78GB，只读盘点，SQL 与完整结果见报告附录）

| 表 | 行数 | 标的覆盖 | 日期范围 | 说明 |
|---|---|---|---|---|
| bar_daily | 16,476,179 | 9,610 code | 20150914~20261002 | 日线，含指数/基金/B股，已带 close_adj |
| bar_weekly | 3,487,632 | 9,595 | 20151201~20261002 | 周线 |
| stock_info | 8,728 | 8,728 | list_date ~20260930 | 含 PE/PB/市值/换手快照 |
| sector / sector_member | 547 / 69,878 | 5,948 | ~20260930 | 板块+成分归属 |
| **ts_income** | 323,973 | **5,591** | 1990~2026Q2 | 利润表 ✅ 已入库 |
| **ts_balance_sheet** | 267,494 | **5,591** | 2001~2026Q2 | 资产负债表 ✅ |
| **ts_cashflow** | 292,429 | **5,591** | 2001~2026Q2 | 现金流量表 ✅（背景说"0 只"是 10-03 旧快照） |
| **ts_fina_indicator** | 248,605 | **5,591** | 2002~2026Q2 | 财务指标时序 ✅ |
| ts_top10_holders | 257,987 | 4,912 | 20151231~20260630 | 十大股东 ✅ |
| 各 ts_*_meta | 3 行/表 | — | — | 单位口径台账（金额=亿元，入口 /1e8） |

⚠️ 背景描述"利润表 2,470 只 / 现金流量表 0 只 / 财务时序不全"**已过时**：
实测三表+指标均覆盖 5,591 只（2026-10-06 的 feat-tushare-full-download 批次已并库）。
尚缺的是**增量（2026Q3）与质检门复核**，不是重新下载。

### 1.2 runtime/tushare 原料盘点（3.3GB，58 个 parquet）——复用判定

> 版本账本实况（`runtime/versions/history.json` 亲测读出）：既有 `data_version.py` 平台
> 当前版本 `871a003e6d5e`（2026-10-06 14:11）= **53 张 tushare 表 / 48,181,266 行**
> + market.db 16 表 / 21,433,472 行——与下方逐文件盘点合计吻合（48.2M 行）。台账设计见 §7。

| 原料 | 行数 | 覆盖 | 复用判定 |
|---|---|---|---|
| income/balancesheet/cashflow/fina_indicator(.parquet+_clean) | 28.9~39.6 万/表 | 5,591 只 | **已并库**（=ts_* 行数，直接复用对账） |
| daily_basic.parquet | 11,189,704 | 5,905 只 20151201~20260930 | **可直接入库**，未入 |
| stk_limit.parquet | 11,473,527 | 5,975 只 | **可直接入库**，未入 |
| moneyflow.parquet | 10,931,850 | 5,823 只 | **可直接入库**，未入（895MB 大宽表，保留 parquet） |
| adj_factor.parquet | 11,282,949 | 5,591 只 | **可直接入库**，未入 |
| top10_floatholders / stk_managers | 257,985 / 101,942 | 4,852 / 6,536 | 可直接入库，未入 |
| disclosure_date / index_classify / index_member_all / trade_cal / stock_basic(_D) | 4.8 万 / 511 / 5,591 / 2,634 / 5,911 | 全 | 可直接入库，未入 |
| industry_member_shard000~027 | 5,591 | 申万成分 | 与 index_member_all 互备 |
| delisted_daily / delisted_adj_factor | 19,623 / 23,118 | **仅 15 只**（退市股 339 只） | ⚠️ 严重不足，需补抓 324 只 |
| yjbb/*.csv（46 期业绩报表） | ~40MB | 2015Q1~2026Q2 | 独立冗余源，留作对拍 |

### 1.3 Tushare 接口可用性校准（本 token，极小配额实测）

探测口径：每接口 **1 次**请求、窗口压到最小（单股单日/单期）；分钟线等按量计费接口**不探**。
两轮共 **88 次**请求（86 + 纠正 2），结果：**可用 58 / 权限不足 19 / 0 行待复核 5 / 接口名不符 4**。

关键读数（与既有记录的出入以本轮为准，S3："权限会变，禁止沿用历史探测结论"）：

| 现象 | 既有记录 | 本轮实测 |
|---|---|---|
| hs_const 沪深港通成份 | "官方停更，全参数 0 行" | `hs_type=SH` **返回 581 行**（需按官方口径复测全量） |
| top_inst 龙虎榜机构 | 文档标 5,000 积分 | **实测可用**（838 行） |
| cn_pmi PMI | 文档标 5,000 积分 | **实测可用**（261 行） |
| 财务三表/vip | — | 2,000 积分**仅逐只**；全市场按期版 `*_vip` 需 5,000 积分（官方文档 44/33/36/79 明示） |
| 19 个不可用 | cyq_perf 等 7 个 | 筹码/竞价/热榜/游资/连板/THS·DC 概念/stock_st/bak_daily/sw_daily 等 19 个权限不足（多为 5,000~15,000 积分档或独立权限） |

配额限频（官方"积分频次对应表"，<https://tushare.pro/document/1?doc_id=290>，亲测抓取）：
2,000 积分档 = **200 次/分钟、10 万次/天/接口**；分钟/新闻/公告/集合竞价等属**独立付费权限**，与积分无关。

官方效率心法（"如何优雅高效的撸数据"，<https://tushare.pro/document/1?doc_id=230>，亲测抓取）：
全量历史**按 trade_date（交易日）逐日循环**而不是按 ts_code 逐只循环（每年约 220~243 个交易日
对 5,000+ 只，循环次数差 20 倍以上）；先用交易日历（`trade_cal(SSE, is_open='1', 起止)`）；
循环内加重试（3 次 try + sleep(1)）。**例外**：2,000 积分档的财务三表/指标/主营构成
只能逐只 ts_code 拉（见上表），这类接口按 by_stock 施工并把耗时算足。

---

## 2. 缺口矩阵（接口 × market.db 现状 × 目标落位 × 量级 × 优先级）

优先级口径：P0=回测/财务因子立即要用（原 P0 缺口收尾 + 幸存者偏差直接相关）；
P1=显著增强因子库；P2=事件/情绪因子与元数据；P3=静态信息/外围资产；P4=范围外或不可用。

| 接口 | 用途 | market.db 现状 | 目标落位 | 量级（行） | 优先级 |
|---|---|---|---|---|---|
| income | 利润表 | ✅ ts_income 292k 级全 | 增量续持 | +5k/季 | **P0**（收尾质检） |
| balancesheet | 资产负债表 | ✅ ts_balance_sheet 全 | 增量续持 | +5k/季 | **P0**（收尾质检） |
| cashflow | 现金流量表 | ✅ ts_cashflow 292,429 全 | 增量续持（首件冒烟样板） | +5k/季 | **P0**（收尾质检） |
| fina_indicator | 财务指标时序 | ✅ ts_fina_indicator 248,605 全 | 增量续持 | +5k/季 | **P0**（收尾质检） |
| adj_factor | 复权因子 | ❌ 无表（parquet 11.28M 已下） | ts_adj_factor | 11.3M | **P0** |
| daily 系（退市股 324 只） | 退市股行情补历史 | ⚠️ delisted_daily 仅 15 只 | bar_daily + delisted_* | +0.5M | **P0**（幸存者偏差） |
| index_weight | 指数成分权重多期历史 | ❌ 只有当前快照 | ts_index_weight | ~26 万（8 指数×130 月） | **P0** |
| forecast / express | 业绩预告/快报 | ❌ | ts_forecast / ts_express | ~50 万 / 30 万 | **P0**（财务事件轴） |
| daily_basic | PE/PB/市值/换手时序 | ⚠️ 仅 stock_info 快照（parquet 11.19M 已下） | ts_daily_basic | 11.2M | P3（下载缺口=0，可按 by_date 重下兜底；入库 merge 随批顺带） |
| stk_limit | 涨跌停价格 | ❌（parquet 11.47M 已下） | ts_stk_limit | 11.5M | P2（下载缺口=0，入库 merge 随批顺带） |
| moneyflow | 个股资金流 | ❌（parquet 10.93M 已下） | ts_moneyflow | 10.9M | P3（下载缺口=0，可按 by_date 重下兜底；入库 merge 随批顺带） |
| margin_detail / margin | 两融明细/汇总 | ❌ | ts_margin_detail / ts_margin | 530 万 / 5 千 | P1 |
| dividend | 分红送股 | ❌ | ts_dividend | 15~30 万 | P1 |
| share_float | 限售解禁 | ❌ | ts_share_float | ~20 万 | P1 |
| suspend_d | 停复牌 | ❌ | ts_suspend_d | ~5 万 | P1 |
| index_daily | 指数日线（全指数口径） | ⚠️ bar_daily 部分含 | ts_index_daily | 3 万+ | P1 |
| top10_holders | 十大股东 | ✅ 25.8 万 | 增量续持 | +5 万 | P1 |
| top10_floatholders | 十大流通股东 | ❌（parquet 25.8 万已下） | ts_top10_floatholders | 25.8 万 | P1 |
| fina_mainbz | 主营构成 | ❌（探测 0 行待复核参数） | ts_fina_mainbz | ~100 万 | P1 |
| stk_holdernumber | 股东人数 | ❌ | ts_stk_holdernumber | 30~50 万 | P2 |
| pledge_stat / pledge_detail | 股权质押 | ❌（detail 探测 0 行待复核） | ts_pledge_stat / _detail | ~300 万 / ~50 万 | P2 |
| block_trade | 大宗交易 | ❌ | ts_block_trade | ~25 万 | P2 |
| repurchase | 回购 | ❌ | ts_repurchase | 6.4 万 | P2 |
| stk_holdertrade | 股东增减持 | ❌ | ts_holdertrade | ~9 万 | P2 |
| moneyflow_hsgt | 北向资金 | ❌ | ts_moneyflow_hsgt | 5 千 | P2 |
| top_list / top_inst | 龙虎榜 | ❌ | ts_top_list / ts_top_inst | 20 万 / 220 万 | P2 |
| namechange | 曾用名（ST 史） | ❌ | ts_namechange | 1.4 万 | P2 |
| disclosure_date | 财报披露计划 | ❌（parquet 4.8 万已下） | ts_disclosure_date | 4.8 万 | P2 |
| index_basic / index_dailybasic | 指数列表/每日指标 | ❌ | ts_index_basic / _dailybasic | 1.3 万 / 4 万 | P2 |
| index_classify / index_member | 申万分类/分级成分 | ❌（parquet 511 已下） | ts_index_classify / _member | 511 / 5.3 千 | P2 |
| trade_cal | 交易日历 | ❌（parquet 2,634 已下） | ts_trade_cal | 9 千（全历史） | P2 |
| stock_basic / stock_company | 股票/公司信息 | ⚠️ stock_info 部分 | 落 stock_info 增补 | 6 千 | P3 |
| stk_managers / stk_rewards | 高管/薪酬持股 | ❌（managers 10.2 万已下） | ts_stk_managers / _rewards | 10 万 / 250 万 | P3 |
| new_share / hs_const | IPO / 沪深港通成份 | ❌ | ts_new_share / ts_hs_const | 4.3 千 / 2 千 | P3 |
| sz_daily_info / ggt_top10 | 深市统计/港股通十大 | ❌ | ts_sz_daily_info / ts_ggt_top10 | 2.2 万 / 5 万 | P3 |
| cn_cpi/ppi/pmi/gdp/m、shibor 系、index_global、sf_month | 宏观利率 | ❌ | ts_cn_* / ts_shibor_* | 各 <1 万 | P4 |
| cb_basic / cb_issue / cb_daily | 可转债 | ❌ | ts_cb_* | 千 / 千 / ~500 万 | P4（待裁定） |
| fund_daily / fund_nav | ETF/基金行情 | ❌ | ts_fund_* | 视范围 | P4（待裁定） |
| 筹码/竞价/热榜/游资/连板/THS·DC 概念/stock_st/bak_daily/sw_daily 等 19 个 | 特色/情绪 | ❌ | — | — | P4 **实测权限不足**，升级后另案 |
| 分钟/Tick/实时行情、新闻/公告/研报/互动、港美股、期货期权外汇现货、公募基金全系 | 其他资产/语料 | ❌ | 本方案范围外 | — | **范围外**（多数需独立付费权限，见 §10） |

---

## 3. 接口清单表（关键字段 / 限频与配额 / upsert 键）

限频列口径：官方档位（文档）+ 本轮实测；upsert 键沿用 `tushare_state.BUSINESS_KEYS` 白名单
（**不是**"加到唯一为止"——S8 教训：多版本报表会被压没）。

| 接口 | 关键字段（首查） | 单次上限/限频（官方） | upsert 键 |
|---|---|---|---|
| income / balancesheet / cashflow | ts_code,end_date,ann_date,report_type,update_flag + 金额列 | 逐只；2,000 分；全市场版=*_vip(5,000 分) | ts_code,end_date,ann_date,report_type,update_flag |
| fina_indicator | ts_code,end_date,ann_date + roe/grossprofit_margin | 逐只；2,000 分 | ts_code,end_date,ann_date |
| adj_factor | ts_code,trade_date,adj_factor | 逐只/逐日；6,000/次（复权因子类） | ts_code,trade_date |
| daily_basic | ts_code,trade_date,pe,pb,total_mv,circ_mv,turnover_rate | 逐日全市场；6,000/次 | ts_code,trade_date |
| stk_limit | ts_code,trade_date,up_limit,down_limit | 逐日；单日 5,377 行单页可容 | ts_code,trade_date |
| moneyflow | ts_code,trade_date,买/卖单净额 | 逐日 6,000/次 | ts_code,trade_date |
| index_weight | index_code,con_code,trade_date,weight | 6,000/次 | index_code,con_code,trade_date |
| forecast / express | ts_code,ann_date,end_date,type | forecast 单次 3,500 行 | ts_code,ann_date,end_date |
| margin_detail / margin | ts_code,trade_date,rzye/rqye | 6,000 / 4,000 行/次 | ts_code,trade_date / trade_date |
| dividend | ts_code,end_date,div_proc | 2,000/次；**只能逐只** | ts_code,end_date,ann_date(若在列) |
| share_float | ts_code,float_date,ann_date | 6,000/次 | ts_code,float_date,ann_date |
| suspend_d | ts_code,trade_date,suspend_type | 逐日 | ts_code,trade_date |
| top10_holders / top10_floatholders | ts_code,end_date,holder_name,ann_date | 按期全市场 | ts_code,end_date,holder_name,ann_date |
| fina_mainbz | ts_code,end_date,ann_date,type,bz_code | 100 行/次；逐只 | ts_code,end_date,ann_date,type |
| stk_holdernumber | ts_code,end_date,ann_date | 3,000/次 | ts_code,end_date,ann_date |
| pledge_stat / pledge_detail | ts_code,end_date / 质押笔数键 | 1,000/次 | ts_code,end_date / 待实测定 |
| block_trade | ts_code,trade_date,price,vol | 1,000/次 | ts_code,trade_date,vol,price |
| repurchase | ts_code,ann_date,report_period | 2,000/次 | ts_code,ann_date,report_period |
| stk_holdertrade | ts_code,holder_name,in_date | 3,000/次 | ts_code,holder_name,in_date |
| moneyflow_hsgt / ggt_top10 | trade_date / ts_code,trade_date | 逐日 | trade_date / ts_code,trade_date |
| top_list / top_inst | ts_code,trade_date | 10,000/次 | ts_code,trade_date + 细分键实测定 |
| namechange | ts_code,name,start_date | 10,000/次（13,889 全量页 2） | ts_code,name,start_date |
| disclosure_date | ts_code,end_date,pre_date,actual_date | 6,000/次 | ts_code,end_date |
| index_basic / index_dailybasic | ts_code / trade_date | 8,000/次 | ts_code / ts_code,trade_date |
| index_classify / index_member | index_code,src / index_code,con_code | 分级拉 | index_code / index_code,con_code |
| trade_cal | cal_date,is_open | 一次全量 | cal_date |
| stock_basic / stock_company | ts_code / ts_code | 6,000/次（**50 次/分钟**）/4,500/次 | ts_code |
| stk_managers / stk_rewards | ts_code,name,begin_date / ts_code,end_date | 逐只 | ts_code,name,begin_date / 待实测定 |
| new_share / hs_const | ts_code / ts_code,hs_type | 2,000/次 | ts_code / ts_code,hs_type |
| sz_daily_info / cn_* / shibor 系 / index_global / sf_month | trade_date / month / date | 各 ≤6,000/次 | 联合日期键逐表实测定 |
| cb_basic / cb_issue / cb_daily | ts_code / ts_code,trade_date | 2,000/次（cb_daily 名称有效、参数待复核） | ts_code / ts_code,trade_date |

> "待实测定"的键按 BUSINESS_KEYS 原则（先定业务主键，键用尽仍重复→抛错）在该批冒烟时定死。

---

## 4. 统一入口设计（项目铁律：功能必须有清晰入口）

**决定：扩展既有 `research/scripts/fetch_all_tushare.py` 为唯一入口**（不另起炉灶，不东拼西凑），
所有下载/清洗/入库/质检/审计动作从它进出；`--help` 必须齐全。

```
uv run python research/scripts/fetch_all_tushare.py --list              # 任务清单（既有）
uv run python research/scripts/fetch_all_tushare.py --batch P0-C        # 按批次跑
uv run python research/scripts/fetch_all_tushare.py --task index_weight # 单任务（既有）
uv run python research/scripts/fetch_all_tushare.py --smoke cashflow --n 5   # 冒烟 5 只
uv run python research/scripts/fetch_all_tushare.py --stage fetch|clean|merge|audit --batch P0-C
uv run python research/scripts/fetch_all_tushare.py --status             # 账本+台账总览
```

- 任务表（`tushare_tasks.py`）每任务增补：`fields`（首查关键字段）、`kind`（by_date/by_stock/by_period…）、
  `upsert_keys`、`target_table`、`expected_rows`（量级下限，供质检门 G1）、`batch`（批次号）。
- 执行复用既有模块：`tushare_paging`（翻页防截断）、`tushare_state`（限频+账本）、
  `tushare_paths`（落盘+日期类型归一）、`clean_tushare_data`、`merge_tushare_tables`、
  `verify_tushare_full` / `verify_tushare_data`、`audit_data_quality` —— **不重写**。
- 拉取策略固定：日频/事件类按 **trade_date 逐日**（官方心法）；财务类 2,000 分档只能**逐只**，
  任务表必须标 `kind="by_stock"` 并按 §9 预算算足耗时；静态快照类**一次拉完+翻页**（S9）。
- 新增逻辑放新模块（`tushare_stage.py` 分段执行 / 台账字段扩写进既有 `_download_manifest.json`），拆分守 ≤500 行线。

## 5. 分批计划（每批 = 可独立验收的最小闭环）

**每批固定四步**（缺一步不算完）：① 冒烟 5 只（跑通+计时）→ ② 小样本抽查
（3~5 条实际数值+已知真值对照）→ ③ 全量下载入库 → ④ 过 6 道质检门（§8）。

| 批次 | 内容 | 验收数字 |
|---|---|---|
| **P0-A 财务收尾** | income/balancesheet/cashflow/fina_indicator 补 2026Q3 增量 + 质检全跑 | 5,591 只 × 报告期矩阵无洞；增量行数=账本对账 ±0 |
| **P0-B 复权全覆盖** | adj_factor 入库 + 退市股 324 只补 daily/adj + bar_daily 对齐 | verify 覆盖率 ≥95%（现成判据）退市股因子覆盖 ≥300/339 |
| **P0-C 指数成分多期历史** | index_weight：8 指数 × 201512~202609 逐月 | 每指数每月成分 ≥90% 在场；与 csindex 两期快照对拍差异 ≤1pp（权重和 100±2） |
| **P0-D 财务事件** | forecast / express / fina_mainbz / disclosure_date | 每只 ≥0 行不强制；预告覆盖率 ≥60% 报告期 |
| **M-A 本地 merge 收尾**（0 配额） | daily_basic / stk_limit / moneyflow / top10_floatholders / disclosure_date 等已下未入 parquet 并库（moneyflow/daily_basic 保底可按 by_date 重下） | 入库行数 = parquet 行数 - 完全重复行（对账 ±0） |
| **P1-B 情绪资金** | margin_detail / margin / dividend / share_float / suspend_d | 同 G1~G6 |
| **P2-A 事件元数据** | 质押/大宗/回购/增减持/北向/龙虎榜/曾用名/披露/index 系 | 同 G1~G6 |
| **P3-A 静态** | stock_basic/company/managers/rewards/new_share/hs_const/sz_daily_info | 同 G1~G6 |
| **P4 宏观** | cn_* / shibor 系 / index_global / sf_month | 同 G1~G6 |
| **待裁定** | 可转债 / ETF 基金 / 特色数据（升级后）/ 分钟·新闻·港美（独立权限） | 用户裁定后再派单 |

## 6. 错误处理 / 断点续传 / 幂等

- **0 行 = 失败**（P5）：该任务记 `failures.json`、收盘汇总、进程**非零退出**；
  确属"业务上可空"的（如某季无预告）必须在任务表登记 `EMPTY_OK` 白名单，逐条带原因——禁止静默吞。
- **重试只给瞬时错误**（网络/超时/限频），3 次 try + sleep(1)（官方心法）；**参数/权限错误立即上抛**。
- **禁静默 fallback**（沿既有红线）：主键用尽仍重复→抛错；金额白名单缺失→抛错；账本损坏→抛错。
- **断点续传**：下载账本 `_download_manifest.json` 补记 `params_hash + rows + fetched_at + 分段进度`；
  逐只类以 200 只/分片（`fetch_shard` 既有）续跑；逐日类以"任务×年"分段为进度单元。
  中断产物 `{tag}_partial.parquet` 不得当完成（S4）；`params_hash` 变化自动作废旧记录。
- **幂等**：入库=先备份（`runtime/backup/` 既有）→ 事务内按 upsert 键替换 → 行数对账
  （源行数 = 新增 + 替换）→ 不一致自动回滚；重跑不重复、不丢失。

## 7. 存储规范 + 三要素台账

- **结构化表** → `market.db` `ts_` 前缀表：代码用本机格式（`sh600519`）、日期 `Int32/YYYYMMDD`、
  **金额列统一亿元**（入口一次 /1e8，U2；白名单由 `build_money_whitelist.py` 从真实数据推导，禁止手写/关键词猜）。
- **大宽表/原始包** → `runtime/tushare/{api_name}.parquet`（原始单位保留）；
  命名：`{api_name}.parquet` 全量、`_partial` 中断、`_shard###` 分片、`_clean` 去重最新版（S8 原表保留）、
  `delisted_*` 退市补录。
- **三要素台账**（来源+抓取时间+覆盖范围）**复用既有版本账本平台，不另造**：
  1. `runtime/versions/` + `research/scripts/data_version.py`（既有）：每批完成后
     `data_version.py --stamp --note "批次 X"` 打内容哈希版本戳（53 表/4,818 万行口径），
     研究锁版本号、数据变了可发现可回滚；
  2. `_download_manifest.json`（扩展既有账本，不新建表）：每任务补 `source / api_name /
     params_hash / fetched_at / coverage_start / coverage_end / rows_raw / rows_loaded / status`，
     真正把"来源+抓取时间+覆盖范围"记全；
  3. 每张 `ts_` 表配 `{table}_meta` k/v 表（既有模式）记单位换算口径与列数。
- 台账与账本同事务提交，杜绝"入库了但没记账"。

## 8. 质检门设计（每批全过才算完；顺序 G1→G2→G3→G4→G5→G6，先便宜后贵）

| 门 | 复用什么 | 判据 | 拦什么真错（实例） |
|---|---|---|---|
| **G1 防截断/防静默** | `verify_tushare_full` + `fetch_paged(strict)` | 翻页必须见 0 行收尾或抛错；日频 ≥95% 交易日有数、末期 ≤5 交易日前；rows ≥ 任务表量级下限 | S1 分页截断（repurchase 2,000 行 → 实际 63,755）；disclosure_date 行数正常但停在 2016-04；P5 静默 0 行；S4 半成品当完成 |
| **G2 主键/版本** | `dedup_by_business_key` + 唯一性检查 | 业务主键唯一；整行重复=0；同键多版本（report_type/update_flag/ann_date）**保留并计数** | S5 沪深300/中证500 同日权重被合并；S8 8 个合法报表版本被压成 1 行、或反过来把 14 万假重复当"多版本" |
| **G3 类型门** | `tests/test_tushare_types.py`（扫真实文件） | 日期列=Int32/int32、代码列=字符串；新文件 mtime 必须晚于归一化修复上线时间 | S11 日期存 string 导致 `==20240102` 恒 0 行（20 表 2,250 万行）；S14 修复前进程写出的旧文件混入 |
| **G4 量纲反推** | `check_units.py` 扩展 + 已知真值法（U1/U2） | mktcap/shares≈股价 <1%；net_assets/shares≈bps <1%；raw×1e-8=库值（无损）；抽 3~5 条 vs 已知真值 ≤0.5%；每张新表报"查了哪些列"；**回归锚点**（亲测核实）：`bar_daily sh600519 20240830 → close=1443.19 / close_adj=1338.6516544649157`（后复权公式 close×factor(t)/factor(最新)，基准日 2026-09-30） | U1 市值亿元/万元差 1e4；U2 财务元/亿元差 1e8（file `UNITS.md` 两次实锤）；check_units 假绿（只查了市值体系） |
| **G5 独立源对拍** | `src/tools/tdx.py`（通达信 .day）+ akshare 新浪源 | 日/周线抽 30 只×20 日 vs .day：价差 ≤0.01 元（未复权同口径）；财务抽 30 只 vs 新浪：±0.5% | Tushare 单源污染/复权口径错（对拍源独立，通达信 4.4GB 本地不受限流）；新浪限流封窗 10-30 分钟 → 分窗抽样 ≤2,500 只/窗 |
| **G6 数据审计** | `audit_data_quality.py` | 未复权污染：|日收益| 超板限记录占比 <0.3%；后复权序列 annual_drag <0.5pp/年；复权/成分覆盖率 ≥95% | P17 除权跳空未复权（年化偏差 −2.33pp 实锤）；复权因子上市前填充值（920010 判例）；指数成分只有 2 期的幸存者偏差 |

> 每道门的输出必须是**数字判据 + 失败清单**，不许"看起来正常"；任一门红 → 该批未完，禁止开下一批。

## 9. 耗时与配额预算（冒烟实测校准）

- **配额**（官方表）：2,000 积分档 200 次/分钟、**10 万次/天/接口**；项目 `Limiter` 180 次/分钟留 10% 余量。
  by_date 表 2,633 请求/表、by_stock 表 5,591 请求/表、`*_vip` 无权 → **单日配额足够任何单表**。
- **冒烟实测**（cashflow 5 只 = sh600519/sz000001/sz300750/sh688111/920010.BJ，2026-10-07，
  两轮 10 次请求）：单次响应 0.38~0.87s（均值 ≈0.53~0.58s），每次 5 请求得 385 行。
  P7 系数 2.5× → 预算 **1.45s/只**（取收敛口径 0.579s）；限频下限 0.333s 不构成瓶颈。
  **by_stock 全量 5,591 只 ≈ 135 分钟/表**（预算值）。
- **既有大批实测**（账本，2026-10-06）：stk_limit 85.5 分/2,634 日 ≈ 1.95s/请求；
  moneyflow 129.1 分 ≈ 2.94s/请求（含翻页）；adj_factor（逐只）31 分/5,591 只 ≈ 0.33s/只。
  → **by_date 大表预算 90~130 分钟/表；by_stock 表 30~120 分钟/表（响应波动 0.3~1.25s）**。
- **全量总预算**（下载+入库+质检，不含独立权限类）：
  P0 批 ≈ 6~8 小时、P1 批 ≈ 4~6 小时、P2 批 ≈ 4~6 小时、P3+P4 ≈ 2~3 小时，
  M-A 本地 merge 收尾 0 配额 ≈ 1 小时，
  合计 **≈ 20~25 小时机器时间**（建议夜间跑，质检在次日），配额全部在单档日限内。
- 2.5x 系数保留到每批冒烟再校准一次（P7：实际耗时≈响应时间 2.5 倍）。

## 10. 范围外与待裁定

- **实测权限不足（19 个）**：筹码分布/集合竞价/热榜/游资/连板天梯/THS·DC 概念（5,000~15,000 积分或独立权限）；
  stock_st、bak_daily、sw_daily、moneyflow_dc、limit_list_d 等 → 升级积分后按同方案另案。
- **独立付费权限（官方表二）**：分钟/Tick/实时行情、新闻/公告/研报/政策/互动问答、港美股行情财报、
  可转债转股价变动、盘前股本 → 用户明确付费后再立批次（本方案不预设）。
- **其他资产类别**（期货/期权/外汇/现货黄金/公募基金全系/大模型语料/量化因子库/自选组合）：
  与 A 股投研库目标不同主题，列而不动，待用户裁定是否建 `ts_fund_*/ts_fut_*` 独立分库。
- **可转债**：cb_basic/cb_issue 实测可用、cb_daily 名称有效（参数待复核），量级 ~500 万行，
  AGENTS.md 记"可转债用 Ashare"——**两条路线待裁定**，默认 P4。

## 附录 A：批次任务书模板（派单直接套用）

```markdown
# 任务书：Tushare 数据入库批次 {批次号}——{批次名}

## 目标
按 docs/02_方法与结论/tushare全量下载_施工方案.md §5 批次 {批次号} 执行：
{本批接口清单，逐个列 api_name / kind / upsert 键 / 目标落位 / expected_rows}

## 批次边界
- 只做本批 {N} 个接口的「冒烟→抽查→全量→质检门」四步闭环；不碰其他批次任务。
- market.db 写入仅限本批目标 ts_* 表（先备份到 runtime/backup/）；其余表只读。
- 不装包、不动 git、不派子 agent；Python 只用
  C:\Documentation\WorkSpace\Wealth_Management\.venv\Scripts\python.exe。
- token 只从 .env/既有 get_token() 取，**严禁明文写进任何产出**。

## 执行入口（唯一）
C:\Documentation\WorkSpace\Wealth_Management\.venv\Scripts\python.exe \
  research/scripts/fetch_all_tushare.py --batch {批次号}
（--help 必须先跑通；--smoke/--stage 分步执行时按施工方案 §4 用法）

## 四步闭环（缺一步不算完）
1. 冒烟 5 只（sh600519/sz000001/sz300750/sh688111/1 只 .BJ）端到端 + 计时
2. 小样本抽查 3~5 条实际数值 + 已知真值对照（UNITS 法）
3. 全量下载入库（断点续传见施工方案 §6）
4. 质检门 G1~G6 全跑（施工方案 §8），逐门贴数字判据

## 质检门与验收数字
- G1: 翻页见 0 行收尾；覆盖率 ≥95%；rows ≥ {expected_rows 合计} 的下限 {X}
- G2: 业务主键重复 = 0；整行重复 = 0
- G3: 日期列 Int32；代码列字符串；新文件 mtime > {归一化修复时间}
- G4: raw×1e-8 = 库值无损；已知真值差 ≤0.5%；mktcap/shares=股价 <1%
- G5: {本批适用的对拍}，价差 ≤0.01 元 / 数值 ±0.5%
- G6: 超板限记录占比 <0.3%；annual_drag <0.5pp（适用时）
行数对账：源 rows_raw = 入库 rows_loaded + 替换数，±0。

## 红线（越过任一即本批失败）
- market.db 除目标 ts_* 表外只读；0 行必须显式报错记录，禁静默 fallback；
- 同一方法失败 3 次停手记录原因；查不到的写「未知」，禁编造；
- 不做本批范围外的下载；每接口请求量不得超过任务表计划 ±10%；
- 配额 ≤180 次/分钟、单接口 ≤10 万次/天。

## 产出
- 数据：目标 ts_* 表 + runtime/tushare/{api_name}.parquet
- 台账：`_download_manifest.json` 补全本批 7 字段（source/api_name/params_hash/fetched_at/
  coverage_start/coverage_end/status）+ `{table}_meta`；批末 `data_version.py --stamp --note "批次 {批次号}"`
- 报告：.planning/reports/tushare-batch-{批次号}.md
  （四步逐帖实测数字：请求次数/耗时/行数/质检门逐门结果/失败清单/遗留）
```

---

> 口径声明：多空收益为组合口径（A 股散户无法做空）；股票池含幸存者偏差（本方案 P0-B/P0-C 正是为收敛它）；
> 本方案与全部数据产出**不构成投资建议**。