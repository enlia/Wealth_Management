"""Tushare 分页拉取。

为什么要独立模块
----------------
Tushare 的大部分接口**单次调用有上限**（实测 2,000 / 8,000 / 10,000 行三种）。
不翻页拿到的是首页截断，数据是残缺的 ——
而且它**不报错**，行数看着还挺像全量。

实测的三个上限样本：
```
namechange    首页 10,000 行（该接口页大小就是 10,000）
index_basic   首页  8,000 行
repurchase    首页  2,000 行
pledge_detail 首页  1,500 行
```
「首页返回 2,000」和「首页返回 1,500」的差别只在于是否碰巧被截断，
**从行数看不出来**。只能靠翻页验证。

``offset`` 语义
--------------
Tushare 的 ``offset`` 是**行偏移**，不是页码：
```
{'limit': 2000, 'offset': 2000}   → 第二页（从第 2001 行开始）
```
⚠️ 第一页也必须带统一的 ``limit``。若首页不带 limit 返回 8,000 行，
第二页用 ``offset=8000`` 才对；若误用 ``offset=2000`` 会拿到重复数据。

实测验证：``stk_limit`` 单日（2024-01-02）翻页后总计 5,377 行，
与首页返回的 5,377 完全一致 —— 说明该接口**单页已够**，
但这是「碰巧没截断」，不是「保证不截断」，所以仍走翻页逻辑。
"""
from __future__ import annotations

import time
from collections.abc import Callable

import pandas as pd
from fetch_tushare import call

# 统一页大小。选 2000 是因为它是实测到的最小接口上限，
# 用更大的值在部分接口上会超出该接口的页大小而报错。
PAGE_SIZE = 2000

# 单次任务最多翻多少页。60 页 × 2000 = 12 万行，
# 超过这个量级应该改用by_date / by_stock 方式拉，不是继续翻页。
MAX_PAGES = 60


def fetch_paged(
    token: str,
    api: str,
    params: dict,
    limiter: Callable[[], None],
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
    label: str = "",
) -> pd.DataFrame:
    """拉全量：翻页直到返回空或不足一页。

    参数
    ----
    limiter: 限频回调，调用前需已 sleep 到位（用 fetch_all 里的 ``Limiter.wait``）
    page_size: 页大小，默认 2000
    label:进度打印前缀（如 ``20240102``），为空则不打印

    返回
    ----
    拼接去重后的 DataFrame。接口无数据时返回**空 DataFrame**（非 None）。

    ⚠️ 翻页前先看首页行数：多数按日接口单日返回 5,000± 行，
    不足 ``page_size`` 说明没有下一页，不必再请求。
    实测 top_list（单日 64 行）/ ggt_top10（20 行）属这类。
    """
    limiter()
    first = call(token, api, {**params, "limit": page_size, "offset": 0}, retry=2)
    n0 = len(first)
    if n0 == 0 or n0 < page_size:
        return first

    parts = [first]
    offset = n0
    for page in range(1, max_pages):
        limiter()
        df = call(token, api, {**params, "limit": page_size, "offset": offset},
                  retry=2)
        n = len(df)
        if n == 0:
            break
        parts.append(df)
        offset += n
        if n < page_size:
            break
        if label:
            print(f"      {label} 第 {page + 1} 页 +{n:,} 行，累计 {offset:,}",
                  flush=True)

    out = pd.concat(parts, ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(keep="last")
    if len(out) < before and label:
        print(f"      {label} 去重-{before - len(out):,} 行", flush=True)
    return out


def fetch_paged_strict(token: str, api: str, params: dict,
                      limiter: Callable[[], None],
                      page_size: int = PAGE_SIZE) -> pd.DataFrame:
    """翻页到真正末尾；撞页数上限时**抛错**而不是静默返回残缺数据。

    ⚠️ 为什么要 strict 版：
       实测 ``disclosure_date`` 有 40 万+ 行，``max_pages=60`` 只拉到 12 万，
       数据停在 2016-04（近 10 年全缺）。
       若静默接受，会以为「全量下载完成」，后面用它做因子才发现数据是空的。
       **数据不完整必须让流程停下来**，不能靠「看起来有行数」判断成功。

    超大接口的正确做法：改用 by_period / by_date 方式按时间分段拉，
    不要无限翻页（offset 越大越慢，且有上限）。
    """
    parts: list[pd.DataFrame] = []
    offset = 0
    for _page in range(MAX_PAGES):
        limiter()
        df = call(token, api, {**params, "limit": page_size, "offset": offset},
                  retry=2)
        n = len(df)
        if n == 0:
            break
        parts.append(df)
        offset += n
        if n < page_size:
            break
    else:
        raise RuntimeError(
            f"{api} 翻页 {MAX_PAGES} 页仍未到末尾（已拉 {offset:,} 行）。\n"
            f"  该接口数据量超过 {MAX_PAGES * page_size:,} 行上限，\n"
            f"  继续翻页不可靠。解决：\n"
            f"    1. 在 tushare_tasks.py 给 {api} 加时间分段参数（按年/月拆成多次调用）\n"
            f"    2. 或确认该接口是否真的需要全量 ——\n"
            f"       当前拿到的是**前 {offset:,} 行**，不是全量，"
            f"直接用会导致样本期严重偏短"
        )
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(keep="last")


def wait_s(seconds: float) -> Callable[[], None]:
    """返回一个简单的等待回调（测试用或无严格限频需求时）。"""

    def _wait() -> None:
        time.sleep(seconds)

    return _wait
