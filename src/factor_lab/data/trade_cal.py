"""权威交易日历读取：Tushare trade_cal 口径（is_open=1 的日期）。

文件的**写入方**是 ``research/scripts/tushare_paths.trading_days()``（抓取并缓存
为 ``runtime/tushare/trade_cal.parquet``，列 exchange/cal_date/is_open/pretrade_date）；
本模块只做**读取** —— src 层不反向依赖 research 层（ENGINEERING.md 零·扁平），
读写共用同一缓存文件、同一口径。

⚠️ 为什么必须用权威日历而不是输入长表的日期并集（UNITS U4 同源教训）：
   单股/长停牌的大缺口会把「日期并集」压缩，真实 300 个交易日能被数成
   189 个交易日（实测样本被全灭 0 行且不报错）；工作日近似（BDay）也只能
   覆盖 242~243 个真实交易日 / 250 差 7~8 天。计龄按真实交易日，两者都不用。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import OUTPUT_DIR

#: 默认缓存路径（tushare_paths.trading_days 的落盘位置）
DEFAULT_TRADE_CAL = OUTPUT_DIR / "tushare" / "trade_cal.parquet"


def load_trade_cal(path: Path | str | None = None) -> pd.DatetimeIndex:
    """读权威交易日历，返回**升序去重**的 DatetimeIndex（仅开市日）。

    找不到文件时明确报错并给出生成命令，不用工作日/自然日近似兜底
    （近似会数错上市年限，且不报错 —— P16 同类静默错误）。
    """
    p = Path(path) if path is not None else DEFAULT_TRADE_CAL
    if not p.exists():
        raise FileNotFoundError(
            f"交易日历文件不存在: {p}\n"
            f"  生成方式：uv run python research/scripts/fetch_all_tushare.py\n"
            f"  （内部经 tushare_paths.trading_days 缓存为 trade_cal.parquet）\n"
            f"  或显式给 build_universe 传入覆盖完整区间的 trade_cal 参数。"
        )
    df = pd.read_parquet(p)
    if "cal_date" not in df.columns:
        raise ValueError(
            f"交易日历文件缺 cal_date 列: {p}（实际列 {list(df.columns)}）")
    if "is_open" in df.columns:
        df = df[pd.to_numeric(df["is_open"], errors="coerce") == 1]
    days = pd.to_numeric(df["cal_date"], errors="coerce").dropna()
    cal = pd.to_datetime(days.astype("int64").astype("string"),
                         format="%Y%m%d", errors="coerce").dropna()
    if cal.empty:
        raise ValueError(f"交易日历没有可用的开市日: {p}")
    return pd.DatetimeIndex(sorted(cal.unique()))