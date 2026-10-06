"""单边多头检验的取数层：从 market.db 分块读行情长表。

职责边界（单一职责）：
    · 本模块只做「取数 + 类型转换」，不做任何因子计算/面板拼装
    · 因子面板与价格面板组装在 ``long_only_panel.py``
    · 汇报统计（年化 / 横截面标准化）在 ``long_only_report.py``
    · 入口脚本是 ``run_long_only.py``

⚠️ market.db 一律**只读**打开（``mode=ro``），本模块不存在任何写库路径。
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# 因子最大回看窗口对应的前置**交易日**数（pos250 / mom120 需要 250 日）
WARMUP_TRADING_DAYS = 260


def _shift_date(d: str, n_days: int) -> str:
    """把 YYYYMMDD 往前推 n 个**交易日**（用于跨年预热）。

    ⚠️ **必须按交易日算，不能用自然日**（2026-10-06 实测缺陷）：
       初版用 `pd.Timedelta(days=260)` 减自然日，
       260 自然日 ≈ **173 个交易日**，而 pos250 需要 **250 个交易日** ——
       预热少了约 77 个交易日。

       实测后果：`_shift_date('20240101', -260)` 返回 `20230416`，
       正确应是 `20230102`。
       pos250 / mom120 在每年年初会缺 ~77 个交易日的因子值，
       11 年累积约 **850 天（3.3 年）数据被静默丢弃**。

       这类缺陷不报错、不影响其他因子，只让长窗口因子**样本变少**，
       很难被发现 —— 必须靠「跨年时检查长窗口因子的非空率」来发现。
    """
    ts = pd.Timestamp(d)
    # BDay 只排除周末，不排除法定节假日 —— 偏保守（多取几天数据），
    # 比少取安全。少取会丢样本，多取只是多读一点。
    return (ts - pd.tseries.offsets.BDay(abs(n_days))).strftime("%Y%m%d")


def load_long_chunked(codes: list[str], start: str, end: str,
                      years: tuple[int, ...] | None = None,
                      verbose: bool = True) -> pd.DataFrame:
    """分年加载后复权价，避免一次性读入撑爆内存。

    ⚠️ **为什么必须分块**：
    全市场 5,606 只 × 2,611 日 = **1,080 万行 × 9 列**，
    实测 `load_long` 一次性加载直接
    `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 742 MiB`。
    之前「跑 20 分钟」不是慢，是在反复 GC/重试 —— 假象会掩盖真问题。

    做法：按年切片逐年加载，峰值内存降到 1/10。
    因子计算只要 close_adj（有 OHLC 的因子才追加）。
    """
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
        # ⚠️ **不能用 max(ys, start) 做下界**（实测踩过）：
        #   `start` 可能是**预热起点**（如 20180102），
        #   而 `ys` 是本年 1 月 1 日。max() 会把下界夹回本年元旦，
        #   于是**预热数据一行都读不到**，
        #   长窗口因子（pos250 / mom120）在年初必然全 NaN。
        #
        #   实测：`start=20180102, years=(2018, 2019)`
        #        → 2019 段 lo = max('20190101','20180102') = '20190101'
        #        → 预热完全失效（修复 WARMUP 天数后暴露出来）。
        #
        #   正确做法：下界直接用 start（它已是更早的预热起点）。
        lo = start[:8]
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