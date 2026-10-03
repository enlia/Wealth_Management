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
        d = d[d.groupby("code")["date"].rank(method="dense") > cfg.min_list_days]
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
