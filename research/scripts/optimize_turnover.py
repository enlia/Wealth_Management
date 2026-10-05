"""降换手优化：在不牺牲信号的前提下把净收益抬上去。

问题
====
本项目实测：rev_20 的 ICIR=0.423（全因子库前列），但年化净收益 −52.8%；
low_vol_120 的 ICIR=0.306，净收益 +2.32%。
差异几乎全部来自**换手率**：
    年化成本 = 往返费率 × 平均单期换手 × 分组数
A 股往返约 20bp、分组 5 → **换手每降 1%，年成本省 0.10pp**。

思路（全部是非深度学习手段）
  1. 缓冲区（buffer）：只在 Top/Bottom 分位的边界附近换仓。
     原做法：每天重排Top20% → 边界附近的名次天天变→ 换手极高。
     改法：持仓到跌出 Top(20%+b) 才卖，涨出 Bottom(20%−b) 才买。
  2. 低频调仓：因子按月/季更新而非每日。财务因子本就季度更新，
     每日重排只是噪声。
  3. 换手平滑：对因子得分做时间维 EWMA，抑制单日跳变。

这些是组合构建层面的工程优化，与「用不用机器学习」无关，
却往往比换因子收益更大。

用法
----
  uv run python scripts/optimize_turnover.py --factors low_vol_120,bp
  uv run python scripts/optimize_turnover.py --factors low_vol_120,bp --all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from factor_lab.analysis.scorecard import SCORING_RULES
from factor_lab.config import DEFAULT_COST, DEFAULT_RESEARCH, OUTPUT_DIR, is_a_share
from factor_lab.data import all_codes, load_long, load_prices, load_stock_info
from factor_lab.data.universe import build_universe
from expand_factor_library import build_pv_factors
from run_financial_study import (build_financial_factors, neutralize,
                                 pick_sample)

TRADING_DAYS = 252


def zscore_cs(s: pd.Series) -> pd.Series:
    """逐日截面标准化。"""
    def one(g):
        sd = g.std()
        return (g - g.mean()) / sd if sd and sd > 0 else g * np.nan
    return s.groupby(level="date", group_keys=False).apply(one)


def to_wide(score: pd.Series) -> pd.DataFrame:
    """(date, asset) Series → date × asset 宽表。"""
    return score.unstack()


def group_rank(w: pd.DataFrame, n_groups: int) -> pd.DataFrame:
    """逐日按分数分位分组（1=最低，n_groups=最高）。NaN 记为 0（不参与）。

    ⚠️⚠️ 两个叠加的坑（曾使分组完全失效但不报错，只是结果全错）：
    1. `w.groupby(level=0).rank(pct=True)` 是**错的**。
       宽表是 index=date、columns=asset，groupby(level=0) 按行分组后
       rank 也在组内（= 整行）排名 → 每行只有一个有效值，全为 1.0。
       正解：`w.rank(axis=1, pct=True)` —— 明确指定「行内排名」。
    2. 分位 → 分组的ceil 转换必须逐行：`.apply(..., axis=1)`。
       DataFrame.apply 默认按列，整列会被同一个标量 ceil。
    """
    r = w.rank(axis=1, pct=True)               # 行内分位，0~1
    g = r.mul(n_groups).apply(lambda row: np.ceil(row), axis=1)
    return g.fillna(0).astype(int)


def buffered_holdings(
    w: pd.DataFrame, n_groups: int, buffer_frac: float,
    n_drop: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """缓冲区持仓法（Qlib 的 TopkDropoutStrategy 同源思路）。

    规则：每期只换掉「跌出 Top 分位边界下方 buffer」的名次，
         即买入新的进入上边界区的，卖出跌出上边界区的。

    buffer_frac = 0 = 每期全换（基准）
    buffer_frac = 0.1 = 上下边界各外扩 10%，实际换仓量大幅下降

    返回 (每日持仓数 Series, 每期换手率 Series)
    """
    g = group_rank(w, n_groups)
    dates = g.index
    top_cut = n_groups                       # 最高分位
    buy_cut = int(np.ceil(n_groups * (1 - buffer_frac)))   # 放宽后的买入边界
    sell_cut = int(np.floor(n_groups * buffer_frac)) + 1     # 放宽后的卖出边界

    held = None
    turns = []
    for d in dates:
        today = g.loc[d]
        valid = today[today > 0]
        if valid.empty:
            turns.append(np.nan)
            continue
        if held is None:
            held = set(valid[valid >= buy_cut].index)
        else:
            still = valid[valid >= sell_cut].index
            # 卖出：原持仓中已跌出 sell_cut
            drop = [c for c in held if c not in set(still)]
            drop = drop[:max(1, int(len(drop) * n_drop))] if drop else []
            held -= set(drop)
            # 买入：从空出的仓位里，按分数最高的补
            cash = len(drop)
            if cash > 0:
                cand = valid[valid >= buy_cut]
                add = [c for c in cand.sort_values(ascending=False).index
                       if c not in held][:cash]
                held |= set(add)
        turns.append(1 - len(held & set(valid.index)) / max(len(held), 1))
    return g, pd.Series(turns, index=dates, dtype=float)


def backtest_holdings(
    w: pd.DataFrame, prices: pd.DataFrame, n_groups: int = 5,
    buffer_frac: float = 0.0, cost: float = DEFAULT_COST.round_trip,
    rebalance_days: int = 1,
) -> dict:
    """按指定缓冲区与调仓频率回测多头端，返回年化毛/净与真实换手。

    缓冲区机制（TopkDropout 思路）
    ------------------------------
    每日算出排名，但**只有跨出缓冲区边界的才换仓**：
      · 买入边界 = 分位上限 ×(1+buffer)  → 排名足够靠前才买
      · 卖出边界 = 分位下限 ×(1−buffer)  → 跌出这个边界才卖
    buffer=0 时退化为每日全量重排（基准）。

    换手率定义：每期实际发生的持仓变动 / 目标持仓数。
    这个数字直接决定成本，必须由持仓变化算出，不能假设。
    """
    g = group_rank(w, n_groups)
    dates = prices.index
    fwd = prices.pct_change()

    # 每组的每日等权收益
    per_group = {}
    for q in range(1, n_groups + 1):
        sel = (g == q).reindex(dates).fillna(False)
        cnt = sel.sum(axis=1).replace(0, np.nan)
        per_group[q] = fwd.where(sel).sum(axis=1) / cnt

    buy_edge = min(n_groups, max(1.0, n_groups * (1 - buffer_frac)))
    sell_edge = max(0.0, n_groups * buffer_frac)

    held: set[str] = set()
    daily_ret, turns = [], []
    prev_target: set[str] = set()

    for i, d in enumerate(dates):
        row = g.loc[d] if d in g.index else None
        valid = row[row > 0] if row is not None and (row > 0).any() else None
        if valid is None:
            daily_ret.append(np.nan); turns.append(0.0); continue

        is_rebal = (i % rebalance_days == 0)
        if is_rebal:
            target = set(valid[valid >= buy_edge].index)
            if buffer_frac > 0 and held:
                # 有持仓时：只卖跌出 sell_edge 的，且按最差优先
                target = (held & set(valid[valid > sell_edge].index))
                add = [c for c in valid.sort_values(ascending=False).index
                       if c not in target]
                need = max(0, len(prev_target) - len(target))
                target |= set(add[:need])
            changed = len(target ^ prev_target)
            turns.append(changed / max(len(target), 1))
            held = target
            prev_target = set(target)
        else:
            # 非调仓日：换手为 0
            turns.append(0.0)

        # 当日收益 = 持仓组等权收益的平均
        vals = [per_group[q][d] for q in range(1, n_groups + 1)
                if d in per_group[q].index and np.isfinite(per_group[q][d])]
        daily_ret.append(np.nanmean(vals) if vals else np.nan)

    pr = pd.Series(daily_ret, index=dates)
    to = pd.Series(turns, index=dates, dtype=float)
    turnover = float(to.mean()) if len(to) else np.nan

    gross = float(pr.mean() * TRADING_DAYS)
    # 成本：调仓日的换手 × 调仓次数 × 往返费率 × 分组数
    # turns 已是「每交易日平均换手」，× 252 得年化换手
    net = gross - cost * turnover * n_groups * TRADING_DAYS
    return {
        "annual_long_gross": gross,
        "annual_long_net": net,
        "turnover": turnover,
        "rebalance_days": rebalance_days,
        "buffer_frac": buffer_frac,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="降换手优化")
    ap.add_argument("--factors", default="low_vol_120,bp")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = DEFAULT_RESEARCH
    full = args.all or args.n == 0
    names = [x.strip() for x in args.factors.split(",") if x.strip()]

    print("=" * 88)
    print("降换手优化：缓冲区 + 低频调仓")
    print("=" * 88)
    print(f"因子: {', '.join(names)}")
    print(f"股票池: {'全市场' if full else f'{args.n} 只'}")
    print(f"成本: 往返 {DEFAULT_COST.round_trip*10000:.0f}bp")

    t0 = time.perf_counter()
    codes = [c for c in all_codes() if is_a_share(c)]
    if not full:
        codes = pick_sample(codes, args.n)
    long = load_long(codes, start=cfg.start_date, end=cfg.end_date)
    info = load_stock_info()
    info = info[info["code"].isin(codes)]
    long = build_universe(long, cfg, info=info, verbose=False)
    alive = long["code"].unique().tolist()
    prices = load_prices(alive, start=cfg.start_date, end=cfg.end_date,
                         field="close")
    print(f"  {len(alive):,} 只 × {prices.shape[0]:,} 日，{time.perf_counter()-t0:.1f}s")

    # 构造合成因子
    parts = []
    panel = pd.read_parquet(
        Path(__file__).resolve().parents[2] / "runtime" / "financial_panel.parquet")
    panel = panel[panel["sym"].isin(alive)]
    shares = info.set_index("code")["shares"] if "shares" in info.columns else None
    fac = build_financial_factors(panel, prices.index, prices, shares=shares)
    daily = fac.pop("_daily")
    ind_map, cap_map = daily["industry"], daily["mktcap"]
    pv = build_pv_factors(long)

    for n in names:
        s = fac[n] if n in fac else pv.get(n)
        if s is None:
            print(f"  ✗ 找不到因子 {n}")
            continue
        s = neutralize(s, ind_map.reindex(s.index).to_numpy(),
                       cap_map.reindex(s.index).to_numpy())
        parts.append(zscore_cs(s).rename(n))
    score = pd.concat(parts, axis=1).mean(axis=1).dropna()
    w = to_wide(score)
    print(f"  合成得分 {len(score):,} 个")

    # 扫描：调仓频率 × 缓冲区
    print("\n" + "-" * 88)
    print(f"{'调仓间隔':>8} {'缓冲区':>8} {'换手':>8} {'年化毛':>10} {'年化净':>10}")
    print("-" * 88)
    rows = []
    for reb in (1, 5, 10, 20, 60):
        for buf in (0.0, 0.1, 0.2, 0.3):
            r = backtest_holdings(w, prices, cfg.quantiles, buf,
                                  DEFAULT_COST.round_trip, reb)
            rows.append(r)
            print(f"{reb:>7}日 {buf:>8.0%} {r['turnover']*100:>7.2f}% "
                  f"{r['annual_long_gross']*100:>9.1f}% {r['annual_long_net']*100:>9.2f}%")

    df = pd.DataFrame(rows).sort_values("annual_long_net", ascending=False)
    print("\n" + "=" * 88)
    print("最优配置（按年化净）")
    print("=" * 88)
    print(df.head(5).to_string(index=False, float_format=lambda x: f"{x:>9.4f}"))

    out_dir = Path(args.out) if args.out else OUTPUT_DIR / "turnover"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"turnover_scan_{'-'.join(names)}.csv",
              index=False, encoding="utf-8-sig")
    print(f"\n结果目录: {out_dir}")
    print("⚠️ 多空口径参考值；此处仅统计多头端收益，且未建模涨跌停不可成交。")
    print("提示: 不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())