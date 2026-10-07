"""库内行情读取与 ATR 计算（本模块唯一的 I/O 入口）。

数据口径（UNITS 与 P23 要求写清）
---------------------------------
- ATR 用 **未复权** 原始 ``high`` / ``low`` / ``close``：
  止损/目标是**挂单报价**，交易所约束的是报价，与复权无关
  （理由同 `data/sqlite_source.load_long` docstring：涨跌停判定与仙股过滤也只对报价成立）。
- ATR 单位 = **元/股**；TR（真实波幅）单位 = 元/股；窗口单位 = 交易日。
- 年化：**不做**（ATR 不是收益率，不做年化）。

ATR 口径
--------
默认 Wilder 平滑（``method='wilder'``，与通达信/多数软件一致）：
``TR_t = max(high-low, |high-close_{t-1}|, |low-close_{t-1}|)``；
``ATR_{n-1} = mean(TR_0..TR_{n-1})``；``ATR_t = (ATR_{t-1}·(n-1) + TR_t) / n``。
``method='sma'`` 为简单均值口径，两者数值不同（实测 sh600519@2024-08-30：
Wilder 28.4497 vs SMA 24.4329，差 16.4%），**卡片必须写明用了哪一个**。

⚠️ 禁止静默 fallback
---------------------
- 数据不足 ``window+1`` 行 → 抛 ``ValueError``（不缩短窗口凑数）。
- ``high``/``low`` 出现 NaN 或 ≤0 → 抛 ``ValueError``（不 dropna 后悄悄继续）。
- 代码在建库不存在 → 抛 ``KeyError``。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...data import sqlite_source
from ...config import DB_PATH

__all__ = ["atr_wilder", "atr_sma", "compute_atr", "load_atr", "load_bars"]

_ATR_METHODS = ("wilder", "sma")


def load_bars(
    code: str,
    start: str,
    end: str,
    db_path: Path | str = DB_PATH,
) -> pd.DataFrame:
    """读一段未复权日线（升序，含 OHLC）。空结果抛错，不返回空表让下游静默跑空。"""
    df = sqlite_source.load_long(codes=[code], start=start, end=end, db_path=db_path)
    if df.empty:
        raise ValueError(
            f"{code} 在 {start}~{end} 无日线数据。\n"
            f"  可能原因：代码不存在、该区间停牌、或日期格式不对（用 YYYY-MM-DD / YYYYMMDD）。\n"
            f"  本模块不静默换代码或换区间。"
        )
    df = df.sort_values("date").reset_index(drop=True)
    for col in ("high", "low", "close"):
        bad = df[col].isna() | (df[col] <= 0)
        if bad.any():
            rows = df.loc[bad, ["date", col]].head(3).to_dict("records")
            raise ValueError(
                f"{code} 的 {col} 有 {int(bad.sum())} 行缺失或 ≤0（例：{rows}）——"
                f"ATR 依赖完整 OHLC，禁止 dropna 后静默继续"
            )
    return df


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def compute_atr(df: pd.DataFrame, window: int = 14, method: str = "wilder") -> float:
    """按给定窗口与口径算最后一个交易日的 ATR（元/股）。"""
    if method not in _ATR_METHODS:
        raise ValueError(f"method 必须是 {_ATR_METHODS} 之一，得到 {method!r}")
    if not isinstance(window, int) or window < 2:
        raise ValueError(f"window 必须是 ≥2 的整数，得到 {window!r}")
    need = window + 1        # 首个 TR 需要前收盘 → 至少 window+1 行
    if len(df) < need:
        raise ValueError(
            f"数据不足：ATR({window}) 需要 ≥{need} 个交易日，实际 {len(df)} 行。\n"
            f"  禁止缩短窗口凑数（那会静默改变风险刻度）。请把 --start 前移。"
        )
    tr = _true_range(df)
    atr_series = tr.rolling(window).mean() if method == "sma" else _wilder(tr, window)
    val = atr_series.iloc[-1]
    if pd.isna(val) or val <= 0:
        raise ValueError(
            f"ATR({window}, {method}) 算得 {val}（应为正的元/股）——"
            f"输入价格序列可能全为同一价或含异常值，需人工核查"
        )
    return float(val)


def _wilder(tr: pd.Series, window: int) -> pd.Series:
    """Wilder 平滑：首值取前 window 个 TR 的均值，其后递推。"""
    out = pd.Series(index=tr.index, dtype="float64")
    out.iloc[window - 1] = tr.iloc[:window].mean()
    for i in range(window, len(tr)):
        out.iloc[i] = (out.iloc[i - 1] * (window - 1) + tr.iloc[i]) / window
    return out


def atr_wilder(df: pd.DataFrame, window: int = 14) -> float:
    """Wilder 口径 ATR（元/股）。默认口径。"""
    return compute_atr(df, window=window, method="wilder")


def atr_sma(df: pd.DataFrame, window: int = 14) -> float:
    """简单均值口径 ATR（元/股）。与 Wilder 口径数值不同，卡片须注明。"""
    return compute_atr(df, window=window, method="sma")


def load_atr(
    code: str,
    as_of: str,
    window: int = 14,
    method: str = "wilder",
    db_path: Path | str = DB_PATH,
) -> tuple[float, pd.DataFrame]:
    """取 ``as_of``（含）及之前 ``window+1`` 个交易日，算 ATR。

    返回 ``(atr, 用到的行情切片)``，切片供人工复核（核对价格与日期是否对得上）。

    ⚠️ 防前视：``as_of`` 之后的数据**不参与**计算（SQL 层已 ``date <= as_of``）。
    """
    df = load_bars(code, start="1990-01-01", end=as_of, db_path=db_path)
    need = window + 1
    if len(df) < need:
        raise ValueError(
            f"{code} 截至 {as_of} 只有 {len(df)} 个交易日，ATR({window}) 需要 ≥{need} 个。"
        )
    tail = df.tail(need).reset_index(drop=True)
    return compute_atr(tail, window=window, method=method), tail
