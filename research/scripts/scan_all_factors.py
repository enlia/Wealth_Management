"""全因子扫描：找出毛收益能覆盖交易成本的因子。

核心判据（盈亏平衡分析）：
    年成本 = 换手率 × 单次往返成本 × 分组数
    可行 ⟺ 年化毛收益 > 年成本
    盈亏平衡换手率 = 年化毛收益 / (单次往返成本 × 分组数)

同时输出"需要的最低换手率"，直观说明什么样的策略才有可能盈利。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
warnings.filterwarnings("ignore")

from factor_lab.analysis import run_tear_sheet
from factor_lab.config import DEFAULT_COST, DEFAULT_RESEARCH, is_a_share
from factor_lab.data import all_codes, load_long, load_prices
from factor_lab.factors.price_volume import FACTORY, compute_factor

pd.set_option("display.width", 220)

START, END = "2016-01-01", "2026-09-30"


def main() -> None:
    print("=" * 92)
    print("全因子扫描 —— 找毛收益能覆盖成本的因子")
    print(f"区间 {START} ~ {END}   成本模型：往返 {DEFAULT_COST.round_trip*100:.2f}bp"
          f"（买 {DEFAULT_COST.buy_cost*10000:.1f}bp + 卖 {DEFAULT_COST.sell_cost*10000:.1f}bp）")
    print("=" * 92)

    codes = [c for c in all_codes() if is_a_share(c)]
    # 🔴 **必须 adjusted=True**（2026-10-06 review BLOCK-3 修）
    #   `compute_factor` 内部的 `_px()` 要求长表带 `close_adj` / `high_adj`，
    #   缺列时 `raise KeyError` —— 11 个价量因子里有 9 个会直接失败，
    #   只剩 amount20 / volratio5_60 能跑，输出文件**静默缩水到 2 个因子**。
    #   对照：run_factor_study.py:116 本来就传了 adjusted=True。
    long = load_long(codes, start=START, end=END, adjusted=True)
    long = long[long.groupby("code")["date"].rank(method="dense") > 250]
    uni = list(long["code"].unique())
    prices = load_prices(uni, start=START, end=END, field="close_adj")
    print(f"股票池 {len(uni):,} 只   交易日 {len(prices):,} 天   因子值 {len(long):,} 条\n")

    rows = []
    for name in FACTORY:
        try:
            f = compute_factor(name, long).dropna()
            # 离散取值因子（hhhl）需 zero_aware
            za = name.startswith("hhhl")
            r = run_tear_sheet(f, prices, DEFAULT_RESEARCH, DEFAULT_COST, name,
                               zero_aware=za)
            ic1 = r.ic_by_period.loc[1] if 1 in r.ic_by_period.index else None
            cost_annual = r.turnover_mean * DEFAULT_COST.round_trip * DEFAULT_RESEARCH.quantiles
            # 盈亏平衡换手率：毛收益 / (单次成本 × 分组数)
            be_turnover = r.gross_spread / (DEFAULT_COST.round_trip * DEFAULT_RESEARCH.quantiles)
            rows.append({
                "因子": name,
                "IC": float(ic1["mean"]) if ic1 is not None else float("nan"),
                "ICIR": float(ic1["ir"]) if ic1 is not None else float("nan"),
                "t值": float(ic1["tstat"]) if ic1 is not None else float("nan"),
                "换手率": r.turnover_mean,
                "毛收益": r.gross_spread,
                "年成本": cost_annual,
                "净收益": r.net_spread_after_cost,
                "盈亏平衡换手": be_turnover,
            })
            print(f"  {name:16s} IC={rows[-1]['IC']:+.4f} 换手={r.turnover_mean*100:5.1f}%  "
                  f"毛={r.gross_spread*100:+.3f}%  年成本={cost_annual*100:.2f}%  "
                  f"净={r.net_spread_after_cost*100:+.3f}%")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:16s} ✗ {type(e).__name__}: {str(e)[:60]}")

    df = pd.DataFrame(rows)
    if df.empty:
        return
    df = df.sort_values("净收益", ascending=False)
    out = Path(__file__).resolve().parents[2] / "runtime"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "factor_scan.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 92)
    print("排序结果（按净收益，从高到低）")
    print("=" * 92)
    print(df.to_string(index=False, float_format=lambda x: f"{x:>9.4f}"))

    print("\n" + "=" * 92)
    print("判读")
    print("=" * 92)
    viable = df[df["净收益"] > 0]
    if viable.empty:
        best = df.iloc[0]
        print(f"  ✗ 全部 {len(df)} 个因子扣成本后均为负")
        print(f"    最好的是 {best['因子']}：毛 {best['毛收益']*100:+.3f}%，"
              f"成本 {best['年成本']*100:.2f}%，净 {best['净收益']*100:+.3f}%")
        print(f"    它需要换手率降到 {best['盈亏平衡换手']*100:.1f}% 才可能盈利"
              f"（当前 {best['换手率']*100:.1f}%）")
        print(f"    → 即需降到当前的 {best['盈亏平衡换手']/max(best['换手率'],1e-9)*100:.0f}%"
              f"，相当于按月调仓而非每日")
    else:
        for _, x in viable.iterrows():
            print(f"  ✓ {x['因子']} 净 {x['净收益']*100:+.3f}%  换手 {x['换手率']*100:.1f}%")


if __name__ == "__main__":
    main()
