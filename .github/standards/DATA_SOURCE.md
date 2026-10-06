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

## 相关模块

| 内容 | 位置 |
|---|---|
| 任务表（P1~P4 + 不可用记录） | `research/scripts/tushare_tasks.py` |
| 分页拉取 | `research/scripts/tushare_paging.py` |
| 下载执行器 | `research/scripts/fetch_all_tushare.py` |
| 下载账本 | `runtime/tushare/_download_manifest.json` |
| 质量校验 | `research/scripts/verify_tushare_data.py` |
| 复权因子专项校验 | `research/scripts/verify_adjusted_data.py` |

---

> 外部 API 的行为**只能实测**。文档写「支持分页」不代表默认就翻页了，
> 文档写「需要 5000 积分」不代表 2000 积分完全拿不到（S2 的 `index_classify`）。
> 每条规则都配了实测数字，改动前先重测，别信上一轮的结论。