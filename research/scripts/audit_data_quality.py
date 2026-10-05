"""数据质量审计：量化研究的前置关卡（务必在任何回测前跑一遍）

本项目实测发现（2026-10-05）
--------------------------------
通达信本机日线数据存在**部分未复权**的除权跳空：
  · A 股 5,593 只中，主板有 3,033 条 |日收益| > 10.5% 的记录（限制是 ±10%）
  · 涉及 1,744 只主板股票
  · 典型案例：平安银行 2016-06-16 单日 −17.91%（当日开盘直接跳空 −17.91%，
    全天几乎无波动 —— 这是 10 送 X 除权，不是真实行情）
  · 宁德时代 2023-04-26 单日 −41.82%（创业板 20% 限，−41% 必然是除权）

影响量级（800 只抽样）：
  · 异常日占全部日收益 0.575%，但贡献 4.7% 的绝对波动
  · 等权买入持有 10 年累计：原始 +29.9% → 修正后 **−3.5%**
  · **年化偏差 3.35pp** —— 足以推翻任何收益结论

⚠️ 关键区分（否则会误伤）
  · **次新股上市首日**：无涨跌幅限制，涨 200% 也合法 → 不是数据错误
  · **老股除权日**：必然是未复权 → 是数据错误
  判别方法：看是否发生在该股票序列的前 2 日内。
  实测：588 只异常股票中只有 28 只（5%）是次新股首日，
  其余 95% 都是真实的未复权污染。

本模块提供
----------
  audit_price_data()      价格数据体检，输出污染规模与影响量级
  clean_returns()         清洗收益率：剔除上市首日 + 标记未复权除权日
  judge_limit()           按板块判定当日涨跌幅是否合法

用法
----
  uv run python scripts/audit_data_quality.py --n 800
  uv run python scripts/audit_data_quality.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from factor_lab.config import DEFAULT_RESEARCH, is_a_share
from factor_lab.data import all_codes, load_prices

# 各板块的日涨跌幅限制（2026 年现行）
LIMIT = {
    "main": 0.10,       # 沪深主板
    "gem": 0.20,       # 创业板 sz30
    "star": 0.20,      # 科创板 sh688
    "bse": 0.30,       # 北交所 bj
}
TOL = 1.005# 容差：涨跌停价按四舍五入，留 0.5% 缓冲


def board_of(code: str) -> str:
    """按代码段判定板块。"""
    c = str(code)
    if c.startswith("sz30"):
        return "gem"
    if c.startswith("sh688"):
        return "star"
    if c.startswith("bj"):
        return "bse"
    return "main"


def judge_limit(code: str, ret: pd.Series) -> pd.Series:
    """判断某股票每日收益是否超出该板块的涨跌幅限制。

    超限即说明该日数据异常（未复权除权 / 数据错误 / 上市首日）。
    """
    lim = LIMIT[board_of(code)] * TOL
    return ret.abs() > lim


def clean_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """清洗收益率，返回 (干净收益率, 异常标记)。

    两类处理：
      1. **上市首日**：无前收、无涨跌幅限制 → 直接置 NaN
      2. **超涨跌停限制的除权日** → 置 NaN（而不是置 0，置 0 会低估波动）
    """
    ret = prices.pct_change(fill_method=None)
    bad = pd.DataFrame(False, index=ret.index, columns=ret.columns)

    for c in ret.columns:
        s = prices[c].dropna()
        if len(s) < 2:
            continue
        # 上市首日：无前收、无涨跌幅限制
        first = s.index[0]
        if first in bad.index:
            bad.at[first, c] = True
        # 超涨跌停限制 → 未复权除权或数据错误
        r = ret[c].dropna()
        over = r.index[r.abs() > LIMIT[board_of(c)] * TOL]
        if len(over):
            # ⚠️ 必须用 .loc[index, col] 逐列赋值。
            #    若 over 为空 Index，`bad.loc[empty_index, c] = True`
            #    在 RangeIndex 上会广播到【全部行】→ 整表变True。
            bad.loc[over, c] = True

    return ret.mask(bad), bad


def audit_price_data(prices: pd.DataFrame, verbose: bool = True) -> dict:
    """价格数据体检。"""
    ret = prices.pct_change(fill_method=None)
    clean, bad = clean_returns(prices)
    n_tot = int(ret.notna().sum().sum())
    n_bad = int(bad.sum().sum())
    abs_all = float(ret.abs().stack().sum())
    abs_bad = float(ret.where(bad).abs().stack().sum())

    # 上市首日 vs 除权污染
    n_first = 0
    for c in prices.columns:
        s = prices[c].dropna()
        if len(s) < 2:
            continue
        r = ret[c].dropna()
        over_idx = r.index[r.abs() > LIMIT[board_of(c)] * TOL]
        if len(over_idx):
            n_first += int(s.index[0] in over_idx)
    n_exright = n_bad - n_first

    # 对等权买入持有的影响
    b_raw = (1 + ret.mean(axis=1)).cumprod().iloc[-1]
    b_clean = (1 + clean.mean(axis=1)).cumprod().iloc[-1]
    years = len(prices) / 252

    out = {
        "n_assets": int(prices.shape[1]),
        "n_days": int(prices.shape[0]),
        "n_ret": n_tot,
        "n_bad": n_bad,
        "pct_bad": n_bad / n_tot,
        "n_first_day": n_first,
        "n_exright": n_exright,
        "abs_share_bad": abs_bad / abs_all if abs_all else np.nan,
        "ew_buy_hold_raw": float(b_raw - 1),
        "ew_buy_hold_clean": float(b_clean - 1),
        "annual_drag": float((b_raw - b_clean) / years),
    }
    if verbose:
        print("=" * 70)
        print("价格数据体检")
        print("=" * 70)
        print(f"  样本           {out['n_assets']:,} 只 × {out['n_days']:,} 日")
        print(f"  日收益样本     {n_tot:,}")
        print(f"  异常           {n_bad:,}（{out['pct_bad']*100:.3f}%）")
        print(f"    ├─ 上市首日  {n_first:,}")
        print(f"    └─ 除权污染  {n_exright:,}")
        print(f"  异常贡献的绝对波动  {out['abs_share_bad']*100:.1f}%")
        print()
        print(f"  等权买入持有 10 年累计")
        print(f"    原始         {(b_raw-1)*100:>+7.1f}%")
        print(f"    清洗后       {(b_clean-1)*100:>+7.1f}%")
        print(f"    → 年化偏差   {out['annual_drag']*100:>+7.3f}pp/年")
        if out["annual_drag"] > 0.005:
            print()
            print("  ⚠️ 偏差超过 0.5pp/年，足以推翻收益结论，必须先清洗")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="数据质量审计")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-09-30")
    args = ap.parse_args()

    codes = [c for c in all_codes() if is_a_share(c)]
    if not args.all:
        import random
        random.seed(42)
        codes = random.sample(codes, min(args.n, len(codes)))
    print(f"审计 {len(codes):,} 只 A 股（{args.start} ~ {args.end}）\n")

    prices = load_prices(codes, args.start, args.end, field="close")
    r = audit_price_data(prices)

    print()
    print("=" * 70)
    print("逐板块异常分布")
    print("=" * 70)
    _, bad = clean_returns(prices)
    boards = pd.Series([board_of(c) for c in prices.columns],
                       index=prices.columns)
    for b in ["main", "gem", "star", "bse"]:
        cols = [c for c in prices.columns if board_of(c) == b]
        if not cols:
            continue
        nb = int(bad[cols].sum().sum())
        nr = int(prices[cols].pct_change(fill_method=None).notna().sum().sum())
        print(f"  {b:<6} {len(cols):>4} 只  异常 {nb:>6} / {nr:>10,} "
              f"= {nb/nr*100:.3f}%")
    print()
    print("限制说明：main ±10%、gem/star ±20%、bse ±30%")
    print("超出限制即为未复权除权或数据错误")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
