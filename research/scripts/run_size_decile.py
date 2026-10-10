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

from run_long_only import _cross_z, build_panel  # noqa: E402

# 🔴 分层逻辑**唯一来源**（2026-10-06 抽离）。
#   初版在本文件与 run_financial_walk_forward.py 各写一份，
#   实测两份在同一份数据上差10 倍（策略区间不一致），
#   且都用窗口末日市值分层 ⇒ 循环论证。
from size_decile_core import (  # noqa: E402
    interpret_layer_median,
    size_decile_run,
)

from factor_lab.analysis.costs import CostModel  # noqa: E402
from factor_lab.analysis.tradability import tushare_to_local_code  # noqa: E402
from factor_lab.analysis.walk_forward import WindowSpec, make_windows  # noqa: E402
from factor_lab.config import is_a_share  # noqa: E402
from factor_lab.data import all_codes  # noqa: E402

OUTPUT = ROOT / "runtime" / "walk_forward"


def load_circ_mv(start: str, end: str) -> pd.DataFrame:
    """加载历史时点流通市值面板 (索引=日期, 列=代码)，单位万元。

    ⚠️ **不能用 `stock_info.mktcap`** —— 那是当前快照，
       用它给历史窗口分组会引入**生存偏差**：
       当年濒临退市的公司不在今天的市值表里，
       而它们恰恰是「跌得最多」的那批 ——
       会让反转类因子看起来凭空多出收益。
    """
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
    print("做法：按历史时点流通市值分三层，每层内独立选因子最高的 N 只。")
    print("⚠ 分层用**窗口首日**市值 —— 用末日市值等于用年内涨幅分层，")
    print("  再拿该层收益当基准是循环论证（实测大盘层基准虚高 18.5pp）。")

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
        windows = make_windows(dates, spec)
        if not windows:
            print(f"  ⚠ {f}: 样本 {dates[0]:%Y-%m-%d} ~ {dates[-1]:%Y-%m-%d}"
                  f"（{len(dates)} 交易日）无法生成窗口。"
                  f" 需 ≥{spec.train_years} 年训练 + "
                  f"{spec.test_months} 月测试 ⇒ 往前推 --start，"
                  f"或用 --train-years 2。")
            continue

        # 🔴 分层逻辑已抽到 `size_decile_core`，与财务因子脚本共用一份。
        #   初版在本文件内各写一份，且用窗口末日市值 ⇒ 循环论证。
        rows += size_decile_run(z, price_f, mcap, windows, cost,
                                args.n_hold, args.n_pick_mult, f)
        report_layers(pd.DataFrame([r for r in rows if r["因子"] == f]), f)

    if rows:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUTPUT / "size_buckets.csv", index=False,
                                 encoding="utf-8-sig")

    print("\n以上为统计检验，不构成投资建议。")
    return 0


def report_layers(df: pd.DataFrame, factor: str) -> None:
    """打印分层年化与层内超额，并按层内中位给判读。"""
    if df.empty:
        print(f"  ⚠ {factor}: 分层无有效数据")
        return
    piv_a = df.pivot_table(index="窗口", columns="层", values="年化")
    piv_e = df.pivot_table(index="窗口", columns="层", values="层内超额")
    print(f"\n{'─' * 72}\n【{factor}】分层年化收益 / 层内超额")
    print(f"  {'窗口':>4}{'全市场':>11}{'大盘层':>11}{'中盘层':>11}"
          f"{'小盘层':>11}{'层基准(全)':>12}")
    print("  " + "-" * 66)
    for w in piv_a.index:
        ba = df[(df["窗口"] == w)]["全市场基准"]
        b = ba.iloc[0] if len(ba) else np.nan
        print(f"  {int(w):>4}" + "".join(
            f"{piv_a.loc[w, k]:>11.2%}" if k in piv_a.columns
            and pd.notna(piv_a.loc[w, k]) else f"{'N/A':>11}"
            for k in ("全", "大", "中", "小"))
            + (f"{b:>12.2%}" if pd.notna(b) else f"{'N/A':>12}"))
    print("  " + "-" * 66)
    med_a = {k: float(np.nanmedian(piv_a[k])) for k in
             ("全", "大", "中", "小") if k in piv_a.columns}
    med_e = {k: float(np.nanmedian(piv_e[k])) for k in
             ("全", "大", "中", "小") if k in piv_e.columns}
    print(f"  {'年化中位':>4}" + "".join(
        f"{med_a[k]:>11.2%}" if k in med_a else f"{'N/A':>11}"
        for k in ("全", "大", "中", "小")))
    print(f"  {'超额中位':>4}" + "".join(
        f"{med_e[k]:>11.2%}" if k in med_e else f"{'N/A':>11}"
        for k in ("全", "大", "中", "小")))

    lvl, msg = interpret_layer_median(med_e)
    print(f"\n  【{factor}】判读：{lvl} {msg}")


if __name__ == "__main__":
    raise SystemExit(main())
