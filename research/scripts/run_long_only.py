"""单边多头策略检验：回答「手动买股票能不能赚钱」。

为什么必须做单边检验
--------------------
因子 IC 是**多空组合**的统计性质，但手动买股票只能做多：
  · 多空可以空掉跌得多的，多头必须持有「剩下的」
  · 多空 IC=0.08 完全可能来自空头端贡献
  · 多头换手 53% 在实盘里是「每周换半仓」

实测（2026-10-06，全市场基线）：
```
因子    IC均值   多空年化毛  多空年化净   平均换手
rev5    0.084+0.68      −0.67        53%
vol60  −0.059    −0.08      −0.28         8%
```
多空毛收益看着不错，扣成本后全负 —— **对手动决策零参考价值**。

本脚本做的事
------------
1. 训练期选因子组合（样本内）
2. **样本外**检验单边多头的年化/波动/回撤/换手/成本
3. 扫参数（持股数、调仓频率、止损）找稳健区间
4. 输出**可直接执行的持仓规则**

⚠️ 三个防过拟合的硬约束：
  · 参数必须**样本外**表现好才采纳，不能只看样本内
  · 相邻参数的表现要**平滑**，孤立尖峰是过拟合
  · 扣成本后仍为正才有效（盈亏平衡换手率是硬约束）

模块划分
--------
  · ``long_only_data.py``   取数层：分块读行情长表、主键去重
  · ``long_only_panel.py``  面板层：逐年算因子、裁剪拼装、对齐检查
  · ``long_only_report.py`` 汇报层：基准年化、横截面标准化、打印

用法
----
  uv run python research/scripts/run_long_only.py --all
  uv run python research/scripts/run_long_only.py --all --factors rev5,vol60,pos250
  uv run python research/scripts/run_long_only.py --all --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from factor_lab.analysis.long_only import (  # noqa: E402
    CostModel,
    PortfolioSpec,
    breakeven_turnover,
    build_long_only,
    holdout_split,
)
from factor_lab.config import is_a_share  # noqa: E402
from factor_lab.data import all_codes  # noqa: E402

# 下面这些名字定义在拆分出的三个模块里，这里平铺转出，
# 供 tests / run_size_decile.py / run_walk_forward.py 沿用既有导入路径。
from long_only_data import (  # noqa: E402,F401
    WARMUP_TRADING_DAYS,
    _shift_date,
    load_long_chunked,
)
from long_only_panel import build_panel  # noqa: E402,F401
from long_only_report import _bench_stats, _cross_z, combine, report  # noqa: E402,F401

OUTPUT = ROOT / "runtime" / "long_only"
BENCH = {"300": "399300.SZ"}


def main() -> int:
    ap = argparse.ArgumentParser(description="单边多头策略检验")
    ap.add_argument("--all", action="store_true", help="全市场")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--factors", default="rev5,vol60,pos250,mom60_skip20")
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20260930")
    ap.add_argument("--hold", type=int, default=30)
    ap.add_argument("--pick", type=int, default=100)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--sweep", action="store_true", help="扫参数找稳健区间")
    args = ap.parse_args()

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    cost = CostModel()
    print("=" * 88)
    print("单边多头策略检验（面向手动买股决策）")
    print("=" * 88)
    print(f"交易成本: {cost.describe()}")
    print(f"  年化20%超额可承受换手率上限 {breakeven_turnover(0.20, cost):.1%}")

    codes = [c for c in all_codes() if is_a_share(c)]
    if not args.all:
        codes = codes[: args.n]
    print(f"\n股票池: {len(codes):,} 只")

    panels = build_panel(codes, factors, args.start, args.end)
    if not panels:
        print("无可用因子")
        return 1

    # 取出价格面板（收益来源），其余是因子
    price = panels.pop("__price__", None)
    if price is None or price.empty:
        print("✗ 缺价格面板，无法计算收益")
        return 1
    # 横截面标准化（每天独立，避免风格漂移）
    z = {k: _cross_z(v) for k, v in panels.items()}

    # ── 基准：等权全市场（必须对照，否则无法判断「跑赢了吗」）──
    # ⚠️ 只看绝对收益没有意义：
    #   实测 rev5 样本内年化 −7.78%，而同期等权基准 +123%
    #   ——策略不是「亏钱」，是**大幅跑输**。
    bench_ret = price.pct_change(fill_method=None).mean(axis=1)
    dates = sorted(set.intersection(*[set(v.index) for v in z.values()]))
    tr, te = holdout_split(dates, args.train_frac)
    print(f"\n样本切分（按时间）: 训练 {len(tr):,} 日 / 测试 {len(te):,} 日")
    print(f"  训练 {tr[0]} ~ {tr[-1]}")
    print(f"  测试 {te[0]} ~ {te[-1]}")

    spec = PortfolioSpec(
        name="等权基线", n_hold=args.hold, n_pick=args.pick,
        rebalance="M", factor=",".join(factors))

    print("\n" + "=" * 88)
    print("单因子：样本内 vs 样本外")
    print("=" * 88)
    rows = []
    for f, p in z.items():
        sp_i = PortfolioSpec(**{**spec.__dict__, "name": f"{f}-IN", "factor": f})
        sp_o = PortfolioSpec(**{**spec.__dict__, "name": f"{f}-OUT", "factor": f})
        ri = build_long_only(p.loc[tr], sp_i, cost, price.loc[tr])
        ro = build_long_only(p.loc[te], sp_o, cost, price.loc[te])
        report(ri, f"{f} 样本内")
        report(ro, f"{f} 样本外")
        if ri.get("ok") and ro.get("ok"):
            b_in = _bench_stats(bench_ret.loc[tr])
            b_te = _bench_stats(bench_ret.loc[te])
            rows.append({
                "因子": f, "内年化": ri["年化收益"], "外年化": ro["年化收益"],
                "外夏普": ro["夏普"], "外回撤": ro["最大回撤"],
                "外换手": ro["平均换手"], "外成本": ro["平均年成本"],
                "衰减": ri["年化收益"] - ro["年化收益"],
                "内基准": b_in, "外基准": b_te,
                "内超额": ri["年化收益"] - b_in,
                "外超额": ro["年化收益"] - b_te,
            })
    if rows:
        df = pd.DataFrame(rows).sort_values("外超额", ascending=False)
        print("\n" + "=" * 92)
        print("因子检验汇总（**超额才是关键**）")
        print("=" * 92)
        print(f"{'因子':<14}{'内年化':>9}{'外年化':>9}{'内基准':>9}{'外基准':>9}"
              f"{'内超额':>10}{'外超额':>10}")
        print("-" * 92)
        for r in rows:
            print(f"{r['因子']:<14}{r['内年化']:>9.2%}{r['外年化']:>9.2%}"
                  f"{r['内基准']:>9.2%}{r['外基准']:>9.2%}"
                  f"{r['内超额']:>+10.2%}{r['外超额']:>+10.2%}")
        print()
        print("⚠ **绝对收益为正但超额为负 = 白做**（跑输基准）。")
        print("  实测 rev5 样本内年化 −7.78%，同期等权基准 +123%")
        print("  → 不是亏钱，是大幅跑输，对手动决策毫无价值。")
        print()
        print("  另：样本外超额为正的因子，才值得继续做参数稳健性检验。")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT / "factor_inout.csv", index=False,
                  encoding="utf-8-sig")

    # 参数扫描
    if args.sweep:
        print("\n" + "=" * 88)
        print("参数扫描（**只看样本外**，找稳健区间）")
        print("=" * 88)
        best = sorted(rows, key=lambda r: -r["外年化"])[:1] if rows else []
        if not best:
            print("无有效因子，跳过")
            return 0
        top = best[0]["因子"]
        p = z[top]
        print(f"\n扫描因子: {top}")
        print(f"{'持股':>5}{'候选':>6}{'年化':>9}{'夏普':>8}{'回撤':>9}"
              f"{'换手':>8}{'成本':>8}")
        print("-" * 60)
        sweep = []
        for nh in (10, 20, 30, 50):
            for np_ in (nh * 2, nh * 3, nh * 5):
                sp = PortfolioSpec(
                    name=f"h{nh}p{np_}", n_hold=nh, n_pick=min(np_, p.shape[1]),
                    rebalance="M", factor=top)
                r = build_long_only(p.loc[te], sp, cost, price.loc[te])
                if r.get("ok"):
                    sweep.append({"持股": nh, "候选": np_, **{
                        k: r[k] for k in ("年化收益", "夏普", "最大回撤",
                                          "平均换手", "平均年成本")}})
                    print(f"{nh:>5}{np_:>6}{r['年化收益']:>9.2%}{r['夏普']:>8.2f}"
                          f"{r['最大回撤']:>9.2%}{r['平均换手']:>8.1%}"
                          f"{r['平均年成本']:>8.2%}")
        if sweep:
            sdf = pd.DataFrame(sweep)
            sdf.to_csv(OUTPUT / "param_sweep.csv", index=False,
                       encoding="utf-8-sig")
            print("\n⚠ 选参数原则：要选**区间内稳定**的，不是最优点。")
            print("  相邻参数表现差异大 = 过拟合，样本外会崩。")

    # 数据版本
    try:
        from data_version import build_manifest
        ver = build_manifest()
        meta = {"数据版本": ver["version"], "打戳": ver["stamped_at"],
                "成本": {"买入": cost.buy, "卖出": cost.sell},
                "因子": factors,
                "训练区间": f"{tr[0]}~{tr[-1]}", "测试区间": f"{te[0]}~{te[-1]}"}
        (OUTPUT / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n数据版本: {ver['version']}（已记录到 run_meta.json）")
    except Exception as e:                                   # noqa: BLE001
        print(f"⚠ 版本记录失败: {e}")

    print("\n提示: 样本外结果才有参考价值；扣成本后为负的方案不可用。")
    print("      以上为统计检验，不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())