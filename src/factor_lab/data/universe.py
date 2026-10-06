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


def build_universe(
    long: pd.DataFrame,
    cfg: ResearchConfig,
    info: pd.DataFrame | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """构建可交易股票池，返回过滤后的长表。

    过滤项（全部可追溯，verbose=True 时逐项打印淘汰数量）：
      1. 剔除非 A 股（指数/板块/B股/基金/债券）
      2. 价格区间（默认 2~3000 元，剔除仙股与异常高价股）
      3. 上市满 250 个交易日（规避次新股的异常收益模式）
      4. 20 日均成交额 ≥ 5000 万（流动性陷阱：无法在涨停时买入）
      5. 剔除 ST（从名称判断）
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
        #
        # 正确口径：以 ``stock_info.list_date``（真实上市日）为基准。
        # 退化路径：info 缺失或该股没有 list_date 时，用「本表内首个交易日」
        # 作为基准 —— 但这会让**窗口之前上市的股票全部被误判为新股**，
        # 所以必须打印受影响数量，不能默默接受（P4 静默fallback）。
        _cal = np.sort(d["date"].unique())
        _pos = {dt: i for i, dt in enumerate(_cal)}
        # _cal 是 numpy.datetime64，_ld 是 Timestamp，直接比较会报
        # "TypeError: '<' not supported between int and Timestamp"，
        # 统一成 Timestamp 再searchsorted。
        _cal_ts = pd.DatetimeIndex(_cal)
        base = d.groupby("code", sort=False)["date"].transform("min")
        n_fallback = 0
        if info is not None and "list_date" in info.columns:
            # ⚠️ list_date 在库里是 float（缺失为 NaN），直接 astype("Int64")
            #    会报 invalid literal for int() with base 10: '0.0'
            _ld = info.drop_duplicates("code").set_index("code")["list_date"]
            _ld = pd.to_numeric(_ld, errors="coerce")
            _ld = _ld.dropna().astype("int64").astype("string")
            _ld = pd.to_datetime(_ld, format="%Y%m%d", errors="coerce")
            if _ld.notna().any():
                _base_ld = d["code"].map(_ld)
                n_fallback = int(_base_ld.isna().sum())
                # 🔴 上市日「早于窗口起点」= 该股在窗口内**每一天**都已上市满 1 年，
                #    直接判合格，不能拿窗口内序号去比 250 ——
                #    窗口内序号上界就是窗口交易日数（2024 全年 242 天 < 250），
                #    于是**每只股票都被淘汰**。本函数在这一点上栽了两次：
                #      ① 用 rank() 窗口内序号 → 短窗口 100% 淘汰
                #      ② 改用上市日基准但仍减 searchsorted → 同样全负数
                #    正确判据只有一句：**上市日早于窗口起点 → 合格**。
                _before_win = _base_ld.map(
                    lambda x: bool(pd.notna(x) and len(_cal_ts)
                                   and x < _cal_ts[0]))
                _in_win = _base_ld.map(
                    lambda x: bool(pd.notna(x) and len(_cal_ts)
                                   and _cal_ts[0] <= x <= _cal_ts[-1]))
                _ld_pos = _base_ld.map(
                    lambda x: (_cal_ts.searchsorted(x) if pd.notna(x) else np.nan))
                age = d["date"].map(_pos) - _ld_pos.where(_in_win, 0)
                # 上市日缺失 → 退化为「表内首日」基准
                bad = age.isna()
                if bad.any():
                    age[bad] = (d.loc[bad, "date"].map(_pos)
                                - base[bad].map(_pos))
                ok = (age >= cfg.min_list_days) | _before_win
                d = d[ok]
        if n_fallback:
            print(f"  ⚠️ {n_fallback:,} 行缺 stock_info.list_date，"
                  "上市年限按表内首日估算（若窗口早于真实上市日，会误判为新股）")
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
