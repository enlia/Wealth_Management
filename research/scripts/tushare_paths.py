"""Tushare 数据源工具：路径、枚举生成、落盘。

从 ``fetch_all_tushare.py`` 拆出（原文件触及 500 行强制拆分区）。
职责单一：只回答「数据在哪」「要拉哪些日期/期次」「怎么存」。
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fetch_tushare import call  # noqa: E402
from tushare_tasks import END, START  # noqa: E402

from factor_lab.config import DB_PATH, is_a_share  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runtime" / "tushare"

def trading_days(token: str) -> list[str]:
    """取交易日历（缓存到本地，避免重复请求）。"""
    cache = OUT / "trade_cal.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = call(token, "trade_cal",
                  {"start_date": START, "end_date": END, "is_open": "1"})
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return sorted(df["cal_date"].astype(str).tolist())


def report_periods() -> list[str]:
    """报告期列表：2015Q4 ~ 2026Q2。"""
    out = []
    for y in range(2015, 2027):
        for q, mmdd in ((1, "0331"), (2, "0630"), (3, "0930"), (4, "1231")):
            if y == 2015 and q != 4:
                continue
            if y == 2026 and (q > 2 or (q == 2 and mmdd > "0930")):
                continue
            out.append(f"{y}{mmdd}")
    return out


def months() -> list[str]:
    out = []
    for y in range(2015, 2027):
        for m in range(1, 13):
            if y == 2015 and m < 12:
                continue
            if y == 2026 and m > 9:
                continue
            out.append(f"{y}{m:02d}")
    return out


def a_share_codes() -> list[str]:
    """本机 A 股代码 → Tushare ts_code。"""
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    codes = [r[0] for r in con.execute("SELECT DISTINCT code FROM bar_daily")
             if is_a_share(r[0])]
    out = []
    for c in codes:
        c = c.lower()
        mkt, num = c[:2], c[2:]
        sfx = {"sh": "SH", "sz": "SZ", "bj": "BJ"}[mkt]
        out.append(f"{num}.{sfx}")
    return sorted(out)


def save(df: pd.DataFrame, tag: str) -> None:
    """落盘到 runtime/tushare/{tag}.parquet。"""
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{tag}.parquet", index=False)
