"""从 SQLite 分块加载前复权行情。

## 为什么单独成模块

`run_long_only.py` 原本514 行，超过 ENGINEERING.md 的 500 行硬上限。
拆分依据是**职责**：

- 本模块：**从数据库取数**（内存/IO 边界）
- `run_long_only.py`：**拿数据算因子、跑组合、出报告**

取数与计算混在一起时，改加载策略必须读懂整套因子逻辑，反之亦然。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from factor_lab.config import DB_PATH  # noqa: E402


def load_long_chunked(codes: list[str], start: str, end: str,
                      years: tuple[int, ...] | None = None,
                      verbose: bool = True) -> pd.DataFrame:
    """分年加载前复权价，避免一次性读入撑爆内存。

    ⚠️ **为什么必须分块**：
    全市场 5,606 只 × 2,611 日 = **1,080 万行 × 9 列**，
    实测 `load_long` 一次性加载直接
    `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 742 MiB`。
    之前「跑 20 分钟」不是慢，是在反复 GC/重试 —— 假象会掩盖真问题。

    做法：按年切片逐年加载，峰值内存降到 1/10。
    因子计算只要 close_adj（有 OHLC 的因子才追加）。
    """
    import sqlite3

    from factor_lab.config import DB_PATH

    if years is None:
        y0, y1 = int(start[:4]), int(end[:4])
        years = tuple(range(y0, y1 + 1))

    chunks: list[pd.DataFrame] = []
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    total = 0
    for y in years:
        ys, ye = f"{y}0101", f"{y}1231"
        if ye < start[:8] or ys > end[:8]:
            continue
        lo = max(ys, start[:8])
        hi = min(ye, end[:8])
        parts = []
        # 分批取代码，避免单条 SQL 的 IN 列表过长
        # ⚠️ **列必须按因子声明的依赖取**，不能只取 OHLC。
        #   实测踩过：volratio5_60 依赖 `vol`、amount20 依赖 `amount`，
        #   而这里硬编码了 OHLC+close_adj ——
        #   结果 `KeyError: 'Column not found: vol'`，
        #   两个因子**从未在滚动检验里跑过**（静默漏测）。
        #   代价可控：多取两列约增加 15% 内存（约 18MB/百万行）。
        cols = ["code", "date", "open", "high", "low", "close", "close_adj"]
        cols += [c for c in ("vol", "amount") if c not in cols]
        for i in range(0, len(codes), 800):
            sub = codes[i:i + 800]
            q = (f"SELECT {', '.join(cols)} "
                 f"FROM bar_daily WHERE date BETWEEN ? AND ? "
                 f"AND close_adj IS NOT NULL AND code IN ({','.join('?' * len(sub))})")
            parts.append(pd.read_sql(q, con, params=[lo, hi, *sub]))
        if parts:
            df = pd.concat(parts, ignore_index=True)
            chunks.append(df)
            total += len(df)
            if verbose:
                print(f"    {y}: {len(df):>10,} 行  累计 {total:>12,}")
    con.close()
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True).sort_values(
        ["code", "date"], kind="stable")
    # ⚠️ **必须转 datetime**：`bar_daily.date` 存的是 int（20260104），
    #   而 `compute_factor` 内部用 `pd.DatetimeIndex(dates)` 构造索引。
    #   int 被当纳秒时间戳 → 索引变成 1970-01-01 → unstack 后每年只有 1 行。
    #   实测踩过：因子面板「每年 1 格」，而原始数据明明有 60 万行。
    #   `load_long` 内部做了同样的转换，这里必须对齐。
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
    out["high_adj"] = out["high"] * out["close_adj"] / out["close"]
    out["low_adj"] = out["low"] * out["close_adj"] / out["close"]
    return out
