"""滚动样本外检验驱动脚本：在全市场真实数据上判定因子能否用于手动买股。

为什么这个脚本存在
------------------
`run_long_only.py` 的单次 70/30 切分给 rev5 报出「样本外年化 +27.92%」，
看起来很棒。但分年度看跑赢基准只有 4/11 年（36%），
而切点恰好落在 2023 年风格分界上 —— **单次切分的高年化是运气**。

本脚本用 `walk_forward.make_windows` 生成多个**测试段互不重叠**的窗口，
逐窗跑单边多头回测，输出「各窗口超额分布」，这才是可用性的判据。

用法
----
  uv run python research/scripts/run_walk_forward.py --factors rev5
  uv run python research/scripts/run_walk_forward.py --factors rev5,vol60,pos250
  uv run python research/scripts/run_walk_forward.py --buffer-scan rev5
"""
from __future__ import annotations

import argparse
import json
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
from factor_lab.analysis.walk_forward import (  # noqa: E402
    WindowSpec,
    judge,
    make_windows,
)
from factor_lab.config import is_a_share  # noqa: E402
from factor_lab.data import all_codes  # noqa: E402

from run_long_only import (  # noqa: E402
    _bench_stats,
    _cross_z,
    build_panel,
)

OUTPUT = ROOT / "runtime" / "walk_forward"


def run_windows(panels: dict, price: pd.DataFrame, factor: str,
                spec: WindowSpec, cost: CostModel,
                n_hold: int, n_pick_mult: int) -> list[dict]:
    """逐窗口跑回测，收集每个窗口的超额。"""
    # ⚠️ **基准必须用窗口内切片重算，不能全样本算完再 .loc[te]**（review 2026-10-06，BLOCK）：
    #   `price.pct_change()` 在全样本上算，`.loc[te]` 出来的**首行是 te[0] 当天的收益**，
    #   而该收益的分母是 te[0] 的**前一日收盘价 —— 落在训练期里**。
    #   策略侧 `build_long_only(p.loc[te], ..., price.loc[te])` 里
    #   `price_panel.pct_change()` 的首行恒为 NaN，**没有这一天**。
    #   ⇒ 基准比策略多算 1 天，且多的这 1 天来自训练期。
    #   实测（261 日窗口）：基准 n=200 / 策略 n=199，
    #   基准年化 -11.01% vs 策略 -9.61% → 超额 +1.40%；
    #   对齐后基准 -9.57% ⇒ **仅因边界错位，基准年化偏 -1.44%**。
    #   等权基准在 A 股多数年份上涨 ⇒ 基准偏高 ⇒ **所有超额被系统性低估**。
    #   而「超额」是本脚本全部结论的唯一判据。
    #   ⇒ 正确做法在循环内：`price.loc[te].pct_change()`（窗口内重算）。
    p = panels[factor]
    dates = sorted(set(p.index) & set(price.index))
    p = p.loc[dates]
    windows = make_windows(dates, spec)
    if not windows:
        print(f"  ⚠ {factor}: 无法生成窗口（样本 {dates[0]} ~ {dates[-1]}）")
        return []

    n_pick = max(n_hold * n_pick_mult, n_hold)
    out: list[dict] = []
    for i, (tr, te) in enumerate(windows):
        sp = PortfolioSpec(name=f"{factor}-w{i}", n_hold=n_hold,
                           n_pick=n_pick, rebalance="M", factor=factor)
        r = build_long_only(p.loc[te], sp, cost, price.loc[te])
        if not r.get("ok"):
            out.append({"窗口": i, "测试起": str(te[0].date()),
                        "测试止": str(te[-1].date()), "策略": np.nan,
                        "基准": np.nan, "超额": np.nan, "换手": np.nan,
                        "备注": r.get("reason", "失败")})
            continue
        # 与策略同源：窗口内重算，首行 NaN 自然丢弃 ⇒ 两侧区间完全对齐
        bench_ret = price.loc[te].pct_change(fill_method=None).mean(axis=1)
        b = _bench_stats(bench_ret)
        out.append({
            "窗口": i, "测试起": str(te[0].date()), "测试止": str(te[-1].date()),
            "策略": r["年化收益"], "基准": b, "超额": r["年化收益"] - b,
            "换手": r["平均换手"], "回撤": r["最大回撤"], "备注": "",
        })
    return out


def report_windows(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(f"{'窗口':>4}{'测试区间':>26}{'策略':>9}{'基准':>9}"
          f"{'超额':>10}{'换手':>8}{'回撤':>9}")
    print("-" * 80)
    for _, r in df.iterrows():
        if pd.isna(r.get("策略")):
            print(f"{int(r['窗口']):>4}{r['测试起']}~{r['测试止']:>12}"
                  f"{'失败':>9}{'':>9}{'':>10}{'':>8}{'':>9}  {r.get('备注','')}")
            continue
        print(f"{int(r['窗口']):>4}{r['测试起']}~{r['测试止']:>14}"
              f"{r['策略']:>9.2%}{r['基准']:>9.2%}{r['超额']:>+10.2%}"
              f"{r['换手']:>8.1%}{r['回撤']:>9.2%}")


def main() -> int:
    ap = argparse.ArgumentParser(description="滚动样本外检验")
    ap.add_argument("--factors", default="rev5")
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20260930")
    ap.add_argument("--n-hold", type=int, default=30)
    ap.add_argument("--n-pick-mult", type=int, default=3,
                    help="候选池 = n_hold × 该倍数")
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--test-months", type=int, default=12)
    ap.add_argument("--buffer-scan", default=None,
                    help="对指定因子扫描缓冲带倍数，如 rev5")
    args = ap.parse_args()

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    cost = CostModel()
    spec = WindowSpec(train_years=args.train_years,
                      test_months=args.test_months)

    print("=" * 88)
    print("滚动样本外检验（walk-forward）")
    print("=" * 88)
    print(f"窗口切分: {spec.describe()}")
    print(f"持仓: {args.n_hold} 只 / 候选池 {args.n_hold * args.n_pick_mult} 只"
          f"  月频调仓")
    print(f"成本: {cost.describe()}")
    print("\n⚠ 单次切分的高年化是运气 —— 这里的判据是「各窗口超额的分布」。")

    codes = [c for c in all_codes() if is_a_share(c)]
    print(f"\n股票池: {len(codes):,} 只")
    panels = build_panel(codes, factors, args.start, args.end)
    price = panels.pop("__price__", None)
    if price is None or price.empty:
        print("✗ 缺价格面板")
        return 1
    z = {k: _cross_z(v) for k, v in panels.items()}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = []

    for f in factors:
        if f not in z:
            print(f"  ⚠ 跳过 {f}（无数据）")
            continue
        wins = run_windows(z, price, f, spec, cost,
                            args.n_hold, args.n_pick_mult)
        if not wins:
            continue
        df = pd.DataFrame(wins)
        report_windows(df, f"【{f}】各窗口样本外表现")
        res = judge(wins, f)
        print("\n" + res.report())
        ok = (df["超额"] > 0).sum()
        print(f"  超额为正的窗口: {ok}/{len(df)}")
        df.to_csv(OUTPUT / f"windows_{f}.csv", index=False,
                  encoding="utf-8-sig")
        summary.append({
            "因子": f, "窗口数": res.窗口数, "胜率": res.胜率,
            "超额中位数": res.超额中位数, "最差": res.最差超额,
            "最好": res.最好超额, "可用": res.可用, "原因": res.原因,
        })

    # 缓冲带扫描
    if args.buffer_scan:
        f = args.buffer_scan
        print("\n" + "=" * 88)
        print(f"缓冲带扫描：{f}（候选池倍数 vs换手与超额）")
        print("=" * 88)
        print(f"{'倍数':>5}{'候选池':>8}{'胜率':>8}{'超额中位':>11}"
              f"{'平均换手':>11}{'结论':>8}")
        print("-" * 60)
        rows = []
        for mult in (1, 2, 3, 5, 8):
            wins = run_windows(z, price, f, spec, cost, args.n_hold, mult)
            if not wins:
                continue
            dfw = pd.DataFrame(wins)
            r = judge(wins, f"{f}-m{mult}")
            turn = dfw["换手"].mean()
            rows.append({"倍数": mult, "候选池": args.n_hold * mult,
                         "胜率": r.胜率, "超额中位数": r.超额中位数,
                         "平均换手": turn, "可用": r.可用})
            print(f"{mult:>5}{args.n_hold*mult:>8}{r.胜率:>8.0%}"
                  f"{r.超额中位数:>+11.2%}{turn:>11.2%}"
                  f"{'✓' if r.可用 else '✗':>8}")
        if rows:
            pd.DataFrame(rows).to_csv(
                OUTPUT / f"buffer_scan_{f}.csv", index=False,
                encoding="utf-8-sig")

    if summary:
        sdf = pd.DataFrame(summary)
        print("\n" + "=" * 88)
        print("滚动检验汇总")
        print("=" * 88)
        print(sdf.to_string(index=False))
        sdf.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
        try:
            from data_version import build_manifest
            ver = build_manifest()
            (OUTPUT / "run_meta.json").write_text(json.dumps(
                {"数据版本": ver["version"], "打戳": ver["stamped_at"],
                 "窗口": spec.describe(), "因子": factors,
                 "持仓": args.n_hold, "候选池倍数": args.n_pick_mult},
                ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n数据版本: {ver['version']}")
        except Exception as e:                                   # noqa: BLE001
            print(f"⚠ 版本记录失败: {e}")

    print("\n以上为统计检验，不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())