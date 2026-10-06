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
第二页就要用 ``offset=8000``；误用 ``offset=2000`` 会拿到重复数据。

两个必须守住的规则
------------------
1. **只以「返回 0 行」为终止条件**，不能用「本页不足 page_size」——
   实测存在页大小不一致的接口（第 1 页 2,000、第 2 页 500、第 3 页还有 2,000），
   按行数判末页会漏数据。
2. **撞页数上限必须抛错**。静默返回残缺数据比报错危险得多：
   流程会以为「下载完成」，后面用它做因子才发现样本期严重偏短。
   实测踩过：``disclosure_date`` 无参翻页 60 页拿到 12 万行，
   数据停在 2016-04，近 10 年全缺，而行数完全正常。
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable

import pandas as pd
from fetch_tushare import call

# 统一页大小。选 2000 是因为它是实测到的最小接口上限，
# 用更大的值在部分接口上会超出该接口的页大小而报错。
PAGE_SIZE = 2000

# 单次任务最多翻多少页。60 页 × 2000 = 12 万行，
# 超过这个量级应该改用 by_year / by_date 方式拉，不是继续翻页。
MAX_PAGES = 60


def fetch_paged(
    token: str,
    api: str,
    params: dict,
    limiter: Callable[[], None],
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
    label: str = "",
    strict: bool = True,
) -> pd.DataFrame:
    """拉全量：翻页直到接口返回 0 行。

    参数
    ----
    limiter : 限频回调，调用前需已 sleep 到位
              （用 ``fetch_all_tushare.Limiter.wait``）
    page_size : 页大小，默认 2000
    max_pages : 最多翻多少页
    label : 进度打印前缀（如 ``20240102``），为空则不打印
    strict : 默认 True —— 撞 ``max_pages`` 仍没到末尾时**抛错**。
             确需宽松时显式传 ``strict=False``，会往 stderr 打警告。

    返回
    ----
    拼接去重后的 DataFrame；接口无数据时返回**空 DataFrame**（非 None）。
    """
    limiter()
    first = call(token, api, {**params, "limit": page_size, "offset": 0}, retry=2)
    n0 = len(first)
    if n0 == 0:
        return first

    parts = [first]
    offset = n0
    if n0 < page_size:
        # 不足一页不能直接判定末页 —— 页大小可能不一致，再探一次
        limiter()
        probe = call(token, api,
                     {**params, "limit": page_size, "offset": n0}, retry=2)
        if len(probe) == 0:
            return first
        parts.append(probe)
        offset += len(probe)

    for page in range(1, max_pages):
        limiter()
        df = call(token, api, {**params, "limit": page_size, "offset": offset},
                  retry=2)
        n = len(df)
        if n == 0:
            break
        parts.append(df)
        offset += n
        if label:
            print(f"      {label} 第 {page + 1} 页 +{n:,} 行，累计 {offset:,}",
                  flush=True)
    else:
        # 跑满 max_pages 仍未见 0 行 —— 数据量超上限，不能当成拉完了
        msg = (f"{api}翻页 {max_pages} 页仍未到末尾"
               f"（已拉 {offset:,} 行，超过 {max_pages * page_size:,} 行）。")
        if strict:
            raise RuntimeError(
                msg + "\n"
                f"  当前拿到的是**前 {offset:,} 行**，不是全量，"
                f"直接用会导致样本期严重偏短。\n"
                f"  解决：在 tushare_tasks.py 给 {api} 配 by_year（按年/月分段），"
                f"不要继续翻页。"
            )
        print(f"    ⚠ {msg} 仍按非严格模式返回**残缺数据**。", file=sys.stderr)

    out = pd.concat(parts, ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(keep="last")
    if len(out) < before and label:
        print(f"      {label} 去重 -{before - len(out):,} 行", flush=True)
    return out


def wait_s(seconds: float) -> Callable[[], None]:
    """返回一个简单的等待回调（测试用或无严格限频需求时）。"""

    def _wait() -> None:
        time.sleep(seconds)

    return _wait
