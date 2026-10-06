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
    ⚠️ **BDay 的日历口径 ≠ 真实交易日历**（market.db 只读统计实测）：
       `BDay(260)` 回推覆盖到的真实交易日只有 **242~243 个**
       （2018 / 2020 / 2022 / 2023 各年段实测均 242~243），
       比 pos250 需要的 250 个交易日**短约 8 个交易日** ——
       即每年年初长窗口因子的最前几行仍可能为 NaN。
       这不是「多取几天」的保守写法，而是略少取：
       BDay 只排除周末、不排除法定节假日。
       预热下限按节假日口径收紧不属于本函数职责（约束本体在
       WARMUP_TRADING_DAYS 与因子窗口长度的比对），这里如实记下边界。
    """
    ts = pd.Timestamp(d)
    return (ts - pd.tseries.offsets.BDay(abs(n_days))).strftime("%Y%m%d")


def dedupe_long(long: pd.DataFrame) -> pd.DataFrame:
    """按 (code, date) 消掉重复行情行；同键不同值**直接报错**。

    两类「重复」必须分开处理（DATA_SOURCE S5 / S8）：
      · **整行完全相同** —— 分块读取的重叠拼接产物，是重复行，允许去掉；
      · **同 (code, date) 但取值不同** —— 数据冲突（多版本 / 复权口径差异），
        ``keep='last'`` 会静默丢掉冲突值、直接污染收益计算，必须抛错。

    返回按 (code, date) 稳定排序的去重结果；
    检测到取值冲突时抛 ValueError，信息含冲突键数量与示例。
    """
    rows_in = len(long)
    out = long.drop_duplicates()          # 整行重复：拼接产物，允许去掉
    conflict = out.duplicated(subset=["code", "date"], keep=False)
    if conflict.any():
        keys = (out.loc[conflict, ["code", "date"]]
                   .drop_duplicates()
                   .sort_values(["code", "date"]))
        examples = ", ".join(
            f"({r.code}, {r.date})" for r in keys.head(5).itertuples())
        raise ValueError(
            f"同一 (code, date) 存在**取值不同**的多行（{len(keys)} 个键），"
            f"不允许静默 keep='last' 去重（DATA_SOURCE S5/S8）。\n"
            f"  冲突键示例：{examples}\n"
            f"  整行完全相同的重复行 {rows_in - len(out):,} 行已允许去掉；"
            f"取值冲突说明数据源存在多版本或复权口径差异，"
            f"必须先定下明确的保留规则（含 ann_date / update_flag 等版本维度）"
            f"再处理。")
    return out.sort_values(["code", "date"], kind="stable")


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

    ⚠️ **年份切片的下界是 max(该年元旦, start)，切片之间不允许重叠**：
       下界统一用 start 时，后面每个切片都从 start 重读到本年末 ——
       同一段数据被读 N 次（2019 段已包含的 2018 预热行，2018 切片也读），
       行数重复靠下游去重兜底，IO 近似随切片数平方放大，
       与「分块控内存」的本意相悖。
       预热数据不会因 max() 丢掉：start 所在年的切片 lo = start，
       各切片并集仍是 [start, hi]；前提是 years 必须覆盖 start 所在年份
       —— 下方有显式校验，缺失直接抛错，不做静默漏读。
    """
    from factor_lab.config import DB_PATH

    if years is None:
        y0, y1 = int(start[:4]), int(end[:4])
        years = tuple(range(y0, y1 + 1))
    if min(years) > int(start[:4]):
        raise ValueError(
            f"years={years} 未覆盖 start={start} 所在年份 —— "
            f"[start, 起始年元旦] 的预热/起始数据会被**整段漏读**。\n"
            f"  years 至少要从 {int(start[:4])} 起"
            f"（build_panel 传 range(pad_year, y+1)，pad_year = pad_start "
            f"实际所在年份）。")

    chunks: list[pd.DataFrame] = []
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    total = 0
    for y in years:
        ys, ye = f"{y}0101", f"{y}1231"
        if ye < start[:8] or ys > end[:8]:
            continue
        # ⚠️ **下界 = max(该年元旦, start)** —— 见函数 docstring：
        #   统一用 start 会让相邻切片重读同一段数据（行数重复 + IO 放大）；
        #   用 max() 后切片互不重叠，并集不变，
        #   但 years 必须覆盖 start 所在年份（上方校验已挡）。
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