"""股票池构建：按流动性、上市年限、价格区间过滤。

⚠️ 幸存者偏差说明（AGENTS.md 质量红线）
   本模块基于【当前】数据库筛选，无法消除退市/暂停上市的历史偏差。
   现有指数成分股快照只有 2 期（2026-09-30 / 08-31），也不足以构建历史成分。
   因此本项目的横截面研究应理解为「当前样本上的检验」，
   结论外推到历史时必须打折扣。这是已知局限，不是可修复的 bug。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ResearchConfig, is_a_share
from .sqlite_source import load_first_dates
from .trade_cal import load_trade_cal


def build_universe(
    long: pd.DataFrame,
    cfg: ResearchConfig,
    info: pd.DataFrame | None = None,
    trade_cal: pd.DatetimeIndex | None = None,
    first_dates: pd.Series | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """构建可交易股票池，返回过滤后的长表。

    过滤项（全部可追溯，verbose=True 时逐项打印淘汰数量）：
      1. 剔除非 A 股（指数/板块/B股/基金/债券）
      2. 价格区间（默认 2~3000 元，剔除仙股与异常高价股）
      3. 上市满 250 个交易日（规避次新股的异常收益模式）
      4. 20 日均成交额 ≥ 5000 万（流动性陷阱：无法在涨停时买入）
      5. 剔除 ST（从名称判断）

    ⚠️ 「上市满 N 个交易日」的口径（钉死，勿改回）：
       · 计龄基准 = ``stock_info.list_date``；该股缺 list_date 时按**表内首日**
        （详见 ``_listing_base``，逐级打印受影响计数）。
       · 交易日序号取自**权威交易日历**（``trade_cal`` 参数；缺省读
         ``runtime/tushare/trade_cal.parquet``，Tushare trade_cal 口径），
         不取输入长表的日期并集 —— 大缺口会把并集压缩（UNITS U4 同源教训）。
       · 上市当天计龄 0；计龄 ≥ cfg.min_list_days 当天即合格
        （=「上市当天算第 1 日」口径下的第 N+1 个交易日）。
       · 计龄基准早于交易日历起点 → 覆盖不到的更早交易日无法计数，
         按已满计并打印受影响计数（据实声明的口径）。
    """
    n0 = len(long)
    d = long[long["code"].map(is_a_share)].copy()
    steps = [("剔除非A股", n0, len(d))]

    m1 = len(d)
    d = d[(d["close"] >= cfg.min_price) & (d["close"] <= cfg.max_price)]
    steps.append(("价格区间", m1, len(d)))

    m2 = len(d)
    if cfg.exclude_new:
        # 🔴 判据必须是「**上市首日至今**的交易日数」，不是「**窗口内**的序号」。
        #
        # 2026-10-06 实测踩坑：原实现是
        #     d.groupby("code")["date"].rank(method="dense") > cfg.min_list_days
        # 这个序号的**上界就是窗口内的交易日总数**。于是只要研究窗口
        # 短于 250 个交易日（2024 年全年只有 242 天），
        # **每一只股票都会被淘汰** —— 股票池直接变成 0 只。
        # 表现为「close_adj 面板为空」，报错信息却指向并库，
        # 把排查方向带偏（第一次就这么被带偏过）。
        # 2026-10-13 又实测到同源哨兵缺陷（P22 同构）：原实现在缺 list_date 时
        # 把 NaN 换成真实位置 0（``.where(_in_win, 0)``），退化分支成死代码，
        # 缺 list_date 的真次新股上市首日即进池、缺 list_date 的老股短窗全灭。
        # 计龄基准解析见 ``_listing_base``，计龄判定见 ``_listing_age_ok``。
        d = d[_listing_age_ok(d, cfg, info,
                              trade_cal=trade_cal, first_dates=first_dates)]
    steps.append(("上市满1年", m2, len(d)))

    m3 = len(d)
    amt = d.groupby("code", sort=False)["amount"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    keep_codes = amt.groupby(d["code"]).max() >= cfg.min_amount_20d
    keep_codes = keep_codes[keep_codes].index
    d = d[d["code"].isin(keep_codes)]
    steps.append(("流动性", m3, len(d)))

    m4 = len(d)
    if cfg.exclude_st and info is not None and "name" in info.columns:
        st = set(info.loc[info["name"].astype(str).str.contains("ST|退", na=False), "code"])
        d = d[~d["code"].isin(st)]
    steps.append(("剔除ST", m4, len(d)))

    if verbose:
        print("股票池构建（逐项淘汰）：")
        for label, before, after in steps:
            pct = (1 - after / before) * 100 if before else 0
            print(f"  {label:12s} {before:>10,} → {after:>10,}   淘汰 {pct:5.1f}%")
        print(f"  最终股票池: {d['code'].nunique():,} 只  {len(d):,} 行")

    return d.reset_index(drop=True)


def _parse_list_dates(info: pd.DataFrame | None) -> pd.Series:
    """解析 ``stock_info.list_date`` → 以 code 为索引的 datetime64 序列。

    缺失/不可解析的条目**直接丢掉**，不占位：
    「有没有上市日」由调用方的独立布尔掩码表达。
    禁止把缺失值换成真实日期/位置 —— 哨兵会撞上真实值（P22）：
    2026-10-13 实测 ``.where(_in_win, 0)`` 把 NaN 换成真实位置 0，
    声称的「缺 list_date → 表内首日」退化分支成死代码（P23）。

    ⚠️ list_date 在库里是 float（缺失为 NaN），直接 astype("Int64")
       会报 invalid literal for int() with base 10: '0.0'
    """
    if info is None or "list_date" not in info.columns:
        return pd.Series(dtype="datetime64[ns]")
    s = info.drop_duplicates("code").set_index("code")["list_date"]
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype="datetime64[ns]")
    s = pd.to_datetime(s.astype("int64").astype("string"),
                       format="%Y%m%d", errors="coerce")
    return s.dropna()


def _require_list_dates(d: pd.DataFrame, info: pd.DataFrame | None) -> pd.Series:
    """返回可解析的上市日期序列；整段不可判定时**显式报错**（原因 + 计数）。

    ⚠️ 禁止静默跳过整段上市年限过滤（铁律 4 / P4）：
       info=None / 无 list_date 列 / list_date 全部不可解析时，旧实现零打印
       跳过整段过滤 —— 实测上市 20 天的 sz301999 照样进池。
       判龄要么有依据要么报错；显式关闭该过滤的出口是 ``cfg.exclude_new=False``。
    """
    n_codes = int(d["code"].nunique())
    if info is None:
        reason = "info 未提供"
    elif "list_date" not in info.columns:
        reason = "stock_info 无 list_date 列"
    else:
        ld = _parse_list_dates(info)
        if not ld.empty:
            return ld
        reason = f"stock_info.list_date 全部不可解析（stock_info {len(info):,} 行）"
    raise ValueError(
        f"上市年限过滤无法执行：{reason}。\n"
        f"  受影响 {n_codes} 只 / {len(d):,} 行 —— 判龄依据缺失，拒绝静默跳过整段过滤"
        f"（静默跳过曾放进上市 20 天的新股）。\n"
        f"  处理：补齐 stock_info.list_date 后重跑；"
        f"或显式设 cfg.exclude_new=False 关闭该过滤（结果不得再引用「已排除次新」口径）。"
    )


def _listing_base(
    d: pd.DataFrame,
    info: pd.DataFrame | None,
    first_dates: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """逐行解析「上市计龄基准日」，返回 (基准日, has_list_date 独立掩码)。

    阶梯（每级退化都打印原因与计数，禁静默 fallback）：
      1. ``stock_info.list_date`` —— 真实上市日（has_list_date=True）
      2. 行情表 ``bar_daily`` 该 code 的首个数据日（``first_dates``；
         参数缺省时由 ``load_first_dates()`` 读库）。
         它是上市日的**下界代理**（首个数据日 ≥ 真实上市日），
         口径上宁可把老股误判为新股（误杀）也不把次新股错放进池。
      3. 本长表内该 code 的首个交易日（连行情表首日都取不到时的最后口径）。

    取舍说明（2026-10-13）：缺 list_date 的人群不整只剔除、也不放弃 250 日
    保护 —— 按表内首日计龄**照常执行同一判据**（对有/无 list_date 语义一致），
    真次新股（如 bj920 段，数据随上市起步）会被正确保护到第 251 个交易日；
    数据晚起点的老股在短窗口可能被保守淘汰（与「次新误放」的方向性错误相比
    是更可接受的一侧），且受影响只数逐次打印可见。
    """
    ld = _require_list_dates(d, info)
    has_ld = d["code"].isin(ld.index)
    base = pd.to_datetime(d["code"].map(ld))

    miss = ~has_ld
    if miss.any():
        fd = first_dates if first_dates is not None else load_first_dates()
        fd = pd.to_datetime(pd.Series(fd))
        from_db = miss & d["code"].isin(fd.index)
        from_tbl = miss & ~from_db
        fill = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
        if from_db.any():
            fill.loc[from_db] = pd.to_datetime(d.loc[from_db, "code"].map(fd)).to_numpy()
        if from_tbl.any():
            tbl_first = d.groupby("code", sort=False)["date"].transform("min")
            fill.loc[from_tbl] = pd.to_datetime(tbl_first[from_tbl]).to_numpy()
        base = base.fillna(fill)
        n_codes = int(d.loc[miss, "code"].nunique())
        n_db = int(d.loc[from_db, "code"].nunique())
        n_tbl = int(d.loc[from_tbl, "code"].nunique())
        print(f"  ⚠️ {n_codes} 只缺 stock_info.list_date，按表内首日处理："
              f"{n_db} 只取行情表首个数据日、{n_tbl} 只取本表首个交易日"
              f"（缺真实上市日，计龄基准取首个数据日=上市日下界，250 日保护照常执行）")
    return base, has_ld


def _resolve_trade_cal(trade_cal: pd.DatetimeIndex | None) -> pd.DatetimeIndex:
    """解析权威交易日历（缺省读 trade_cal.parquet），返回升序去重 DatetimeIndex。"""
    cal = trade_cal if trade_cal is not None else load_trade_cal()
    cal = pd.DatetimeIndex(pd.to_datetime(pd.Index(cal)))
    cal = pd.DatetimeIndex(sorted(pd.DatetimeIndex(cal).unique()))
    if len(cal) == 0:
        raise ValueError("交易日历为空，无法按真实交易日计算上市年限（不做自然日近似）")
    return cal


def _listing_age_ok(
    d: pd.DataFrame,
    cfg: ResearchConfig,
    info: pd.DataFrame | None,
    trade_cal: pd.DatetimeIndex | None = None,
    first_dates: pd.Series | None = None,
) -> pd.Series:
    """逐行判定「上市满 cfg.min_list_days 个交易日」，返回 keep 布尔掩码。

    计龄口径（钉死，勿改回）：
      · 计龄基准见 ``_listing_base``；交易日序号取自 ``_resolve_trade_cal``
        的**权威交易日历**，不取输入长表的日期并集 ——
        单股/长停牌的大缺口会把并集压缩：实测真实计龄 250+ 交易日的样本
        被数成 189 而全灭（不报错），UNITS U4 的 BDay 近似同理不可用；
      · 上市当天计龄 0；计龄 ≥ cfg.min_list_days 当天即合格；
      · 计龄基准早于交易日历起点 → 覆盖不到的更早交易日无法计数，
        按已满计并打印受影响计数（据实声明的口径，不是精确数龄）。
    """
    base, has_ld = _listing_base(d, info, first_dates=first_dates)
    if base.isna().any():
        raise RuntimeError(
            f"{int(base.isna().sum())} 行计龄基准日缺失（_listing_base 阶梯未覆盖），"
            f"拒绝用 NaN 继续判定 —— NaN 一旦被换成占位值就会撞上真实日期（P22）")

    cal = _resolve_trade_cal(trade_cal)
    cal_np = cal.to_numpy()
    dates_np = pd.to_datetime(d["date"]).to_numpy()
    base_np = base.to_numpy()
    if len(dates_np) and (dates_np.min() < cal_np[0] or dates_np.max() > cal_np[-1]):
        raise ValueError(
            f"交易日历未覆盖判断区间："
            f"日历 {pd.Timestamp(cal_np[0]):%Y-%m-%d} ~ {pd.Timestamp(cal_np[-1]):%Y-%m-%d}，"
            f"数据 {pd.Timestamp(dates_np.min()):%Y-%m-%d} ~ "
            f"{pd.Timestamp(dates_np.max()):%Y-%m-%d}。\n"
            f"  上市年限按真实交易日计龄，日历不全会数错龄，拒绝用近似日历兜底。\n"
            f"  处理：确认 runtime/tushare/trade_cal.parquet 覆盖该区间"
            f"（uv run python research/scripts/fetch_all_tushare.py 生成），"
            f"或显式传入覆盖完整区间的 trade_cal 参数。")

    # ⚠️ 计龄基准早于日历起点时更早的交易日不可计数，按已满计并打印 ——
    #    若改按日历位置从 0 数起，窗口起步段的老股会被误判为次新：
    #    计龄序号上界=窗口交易日数（2024 全年 242 天 < 250），短窗全灭。
    #    该人群在判据历史上连栽两次（rank() 序号版、哨兵占位版），勿改回。
    before = base_np < cal_np[0]
    age = (np.searchsorted(cal_np, dates_np)
           - np.searchsorted(cal_np, base_np))
    ok = (age >= cfg.min_list_days) | before
    if before.any():
        n = int(d.loc[before, "code"].nunique())
        print(f"  ℹ️ {n} 只计龄基准早于交易日历起点，按已满 {cfg.min_list_days} 交易日计"
              f"（基准更早的交易日不可计数，口径见 docstring）")
    return pd.Series(np.asarray(ok, dtype=bool), index=d.index)


def summarize_universe(long: pd.DataFrame) -> pd.DataFrame:
    """股票池画像：每个市场的数量、覆盖期、流动性分布。"""
    g = long.groupby("code").agg(
        n_days=("date", "size"),
        first=("date", "min"),
        last=("date", "max"),
        med_amount=("amount", "median"),
        med_close=("close", "median"),
    )
    g["mkt"] = g.index.map(lambda c: c[:2])
    print("\n股票池画像：")
    print(g.groupby("mkt").agg(
        标的数=("n_days", "size"),
        日数中位=("n_days", "median"),
        起始=("first", "min"),
        结束=("last", "max"),
        日成交额中位_亿=("med_amount", lambda s: round(s.mean() / 1e8, 2)),
    ).to_string())
    return g
