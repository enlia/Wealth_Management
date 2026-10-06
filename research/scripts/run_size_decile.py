"""市值分层混淆检验：因子的收益来自信号本身，还是仅来自「市值暴露」？

为什么必须做这个检验
--------------------
**实测（2026-10-06）**：rev5（5日反转）在滚动检验里
7个窗口只有1 个亏损、最差年份仅 −2.9%，是 11 个价量因子里
唯一接近可用的。**但它选出的 30 只均价11.37 元，
比全市场便宜 36.7%** —— 它选的就是小盘股。

而等权全市场基准本身就是「小盘等权」，2019 年涨 31.9%、2025 年涨 40.2%。
所以存在一个致命的混淆：

> rev5 赚钱，是因为**反转信号**有效？
> 还是仅仅因为它**系统性地买了小盘股**，而小盘股在 A 股长期跑赢？

这两个原因在「全市场选股」的回测里**完全无法区分**。
不区分就下结论，等于把「买小盘」的效果记在「反转因子」头上。

检验方法
--------
按**历史时点流通市值**（Tushare `daily_basic.circ_mv`，万元，1,118 万行）
把股票分三层，**每层内独立选因子最高的 30 只**：

| 结果形态 | 结论 |
|---|---|
| 各层内超额都为正 | ✅ 信号真实存在，与市值无关 |
| 只有小盘层为正 | ❌ 信号无效，收益全来自市值暴露 |
| 只有全市场（跨层）为正 | ❌ 混淆成立，因子不可用 |

**为什么用历史时点市值而不用当前快照**：
本机 `stock_info.mktcap` 是**当前快照**，用它分组会给历史窗口引入
生存偏差 —— 当年濒临退市的公司不会出现在今天的市值表里，
而它们恰恰是「跌得多」的主体。
必须用逐日的 `circ_mv`。

⚠️ **量纲已硬验证**：`circ_mv` 单位是**万元**。
交叉验证：茅台 2024-01-02 `circ_mv / close` = 125,620 万股，
与真实流通股本 12.56 亿股一致（UNITS.md U1）。

用法
----
  uv run python research/scripts/run_size_decile.py --factors rev5,rev20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from factor_lab.analysis.long_only import (  # noqa: E402
    CostModel,
    PortfolioSpec,
    build_long_only,
)
from factor_lab.analysis.tradability import tushare_to_local_code  # noqa: E402
from factor_lab.analysis.walk_forward import WindowSpec, make_windows  # noqa: E402
from factor_lab.config import is_a_share  # noqa: E402
from factor_lab.data import all_codes  # noqa: E402

from run_long_only import _bench_stats, _cross_z, build_panel  # noqa: E402

OUTPUT = ROOT / "runtime" / "walk_forward"
# 分层内至少要有这么多只，否则选不出30 只 + 候选池
MIN_STOCKS_PER_BUCKET = 120


def load_circ_mv(start: str, end: str) -> pd.DataFrame:
    """加载历史时点流通市值面板 (索引=日期, 列=代码)，单位万元。

    ⚠️ **不能用 `stock_info.mktcap`** —— 那是当前快照，
       用它给历史窗口分组会引入**生存偏差**：
       当年濒临退市的公司不在今天的市值表里，
       而它们恰恰是「跌得最多」的那批 ——
       会让反转类因子看起来凭空多出收益。
    """
    import os
    p = ROOT / "runtime" / "tushare" / "daily_basic.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} 不存在，无法做市值分层检验。"
            f"请先下载 daily_basic（流通市值的历史时点数据）。")
    df = pd.read_parquet(p, columns=["trade_date", "ts_code", "circ_mv"])
    lo,hi = int(start), int(end)
    td = pd.to_numeric(df["trade_date"], errors="coerce")
    df = df[(td >= lo) & (td <= hi)].copy()
    df["code"] = tushare_to_local_code(df["ts_code"])
    df["date"] = pd.to_datetime(td[(td >= lo) & (td <= hi)].astype("int64"),
                                format="%Y%m%d")
    panel = df.pivot_table(index="date", columns="code", values="circ_mv",
                           aggfunc="last").sort_index()
    return panel


def _run_sub(z_sub: pd.DataFrame, px_sub: pd.DataFrame,
             cost: CostModel, n_hold: int, n_pick_mult: int,
             factor: str) -> float:
    """在给定的股票子集内跑回测，返回年化收益。"""
    if z_sub.shape[1] < max(n_hold * 4, MIN_STOCKS_PER_BUCKET):
        return float("nan")
    spec = PortfolioSpec(name="layer", n_hold=n_hold,
                         n_pick=n_hold * n_pick_mult, rebalance="M",
                         factor=factor)
    r = build_long_only(z_sub, spec, cost, px_sub)
    return float(r["年化收益"]) if r.get("ok") else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="市值分层混淆检验")
    ap.add_argument("--factors", default="rev5")
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20260930")
    ap.add_argument("--n-hold", type=int, default=30)
    ap.add_argument("--n-pick-mult", type=int, default=3)
    ap.add_argument("--train-years", type=int, default=3)
    args = ap.parse_args()

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    cost = CostModel()

    print("=" * 88)
    print("市值分层混淆检验")
    print("=" * 88)
    print("问题：因子的收益来自信号，还是仅来自「系统性买小盘股」？")
    print("做法：按历史时点流通市值分三层，每层内独立选因子最高的 N 只。\n")

    mcap = load_circ_mv(args.start, args.end)
    print(f"流通市值面板 {mcap.shape}，"
          f"覆盖率 {mcap.notna().to_numpy().mean():.1%}（单位：万元）")

    codes = [c for c in all_codes() if is_a_share(c)]
    panels = build_panel(codes, factors, args.start, args.end)
    price = panels.pop("__price__", None)
    if price is None or price.empty:
        print("✗ 缺价格面板")
        return 1

    spec = WindowSpec(train_years=args.train_years, test_months=12)
    rows: list[dict] = []

    for f in factors:
        z = _cross_z(panels[f])
        dates = sorted(set(z.index) & set(price.index))
        z, price_f = z.loc[dates], price.loc[dates]
        mcap_f = mcap.reindex(index=dates)
        windows = make_windows(dates, spec)
        if not windows:
            print(f"  ⚠ {f}: 无法生成窗口")
            continue

        print(f"\n{'─'*72}\n【{f}】分层年化收益")
        print(f"  {'窗口':>4}{'全市场':>11}{'大盘层':>11}{'中盘层':>11}"
              f"{'小盘层':>11}{'基准':>10}")
        print("  " + "-" * 58)
        agg: dict[str, list[float]] = {k: [] for k in
                                       ("全", "大", "中", "小", "基准")}
        bench = price_f.pct_change(fill_method=None).mean(axis=1)
        for i, (_, te) in enumerate(windows):
            m = mcap_f.loc[te[-1]].reindex(z.columns)
            valid = [c for c in z.columns if pd.notna(m.get(c, np.nan))]
            q1, q2 = m[valid].quantile([1 / 3, 2 / 3])
            buckets = {
                "全": valid,
                "大": [c for c in valid if m[c] >= q2],
                "中": [c for c in valid if q1 <= m[c] < q2],
                "小": [c for c in valid if m[c] < q1],
            }
            vals: dict[str, float] = {}
            for label, cols in buckets.items():
                # ⚠️ **必须只跑测试窗口 `te`**。
                #   初版传的是 `z[cols]`（全期 2016-2026），
                #   而 walk_forward 传的是 `p.loc[te]`（仅测试段）。
                #   口径不一致导致年化被 11 年摊薄：
                #   rev5 全市场在 walk_forward 是 +19.80%，
                #   在这里只报 +2.19% —— **同一因子差 10 倍**，
                #   且基准中位相同（21.14%）说明窗口没错，
                #   只能是策略区间错了。
                #   两个脚本口径必须一致，否则分层结论不可比。
                vals[label] = _run_sub(z.loc[te, cols],
                                        price_f.loc[te, cols], cost,
                                        args.n_hold, args.n_pick_mult, f)
                agg[label].append(vals[label])
            b = _bench_stats(bench.loc[te])
            agg["基准"].append(b)
            rows.append({"因子": f, "窗口": i, **vals, "基准": b})
            print(f"  {i:>4}" + "".join(
                f"{vals[k]:>11.2%}" if pd.notna(vals[k]) else f"{'N/A':>11}"
                for k in ("全", "大", "中", "小")) + f"{b:>10.2%}")

        print("  " + "-" * 58)
        med = {k: float(np.nanmedian(agg[k])) for k in agg}
        print(f"  {'中位':>4}" + "".join(f"{med[k]:>11.2%}"
                                       for k in ("全", "大", "中", "小"))
              + f"{med['基准']:>10.2%}")

        _interpret(f, med)

    if rows:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT / "size_buckets.csv", index=False,
                  encoding="utf-8-sig")

    print("\n以上为统计检验，不构成投资建议。")
    return 0


def _interpret(factor: str, med: dict[str, float]) -> None:
    """按层内表现给结论 —— 这是本脚本存在的全部意义。"""
    print(f"\n  【{factor}】判读：")
    layers = ("大", "中", "小")
    ok = [k for k in layers if pd.notna(med[k])]
    if len(ok) < 2:
        print("    层数不足，无法判读")
        return
    layer_med = np.nanmedian([med[k] for k in ok])
    all_med = med["全"]
    if layer_med > 0.05:
        print(f"    ✅ 层内中位{layer_med:+.2%} > 0 ⇒ **信号真实存在**，"
              f"与市值无关")
    elif layer_med > 0:
        print(f"    ⚠ 层内中位 {layer_med:+.2%} 勉强为正但很弱，"
              f"跨层选（全市场 {all_med:+.2%}）的收益主要来自市值暴露")
    else:
        print(f"    ❌ 层内中位 {layer_med:+.2%} ≤ 0 ⇒ **信号无效**，"
              f"全市场 {all_med:+.2%} 的收益全部来自市值暴露")
    for k in layers:
        if pd.notna(med[k]):
            print(f"       {k}盘层 {med[k]:+.2%}")


if __name__ == "__main__":
    raise SystemExit(main())