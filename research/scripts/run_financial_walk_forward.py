"""财务因子的滚动样本外检验 + 市值分层混淆检验。

为什么财务因子必须过这两关
--------------------------
`run_financial_study` 给的是 **alphalens 多空 tear sheet**：
IC 0.08、多空年化 +6%，看着挺好。但：

1. **多空 ≠ 能买**。A 股散户无法做空，多空收益里有一半来自空头端。
   而项目已实测：11 个价量因子的多空净收益**全负**，
   bp/ep/roe 的多空净收益也分别只有 −0.37% / +4.14% / +4.35%。
2. **单次切分是运气**。rev5 的「样本外 +27.92%」在滚动窗口下 7 个只赢 4 个
   （MEMORY.md 已确立的检验铁律）。
3. **可能是市值暴露而非信号**。若某因子选出的组合天然偏小盘，
   而 A 股小盘长期跑赢等权基准，则「因子有效」是幻觉。
   rev5 就是这样被否掉的。

所以本脚本对财务因子做**与价量因子完全同口径**的两项检验：
  · `run_walk_forward.run_windows` —— 测试段两两不重叠的滚动窗口
  · `run_size_decile.load_circ_mv` —— 历史时点流通市值三层，层内各自选股

⚠️ **两项检验共用同一套基准口径**（等权全市场，窗口内重算 pct_change）。
   基准不一致时，两个脚本的结论不可比 —— 2026-10-06 实测踩过：
   同一因子在 walk_forward 报 +19.80%、在 size_decile 报 +2.19%，
   差 10 倍，根因是 size_decile 初版传了全期面板而非测试段。

用法
----
  uv run python research/scripts/run_financial_walk_forward.py
  uv run python research/scripts/run_financial_walk_forward.py --neutral
  uv run python research/scripts/run_financial_walk_forward.py \\
      --factors bp,ep,roe --skip-decile
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from financial_panels import build_wide_panels  # noqa: E402
from run_long_only import _cross_z  # noqa: E402
from run_size_decile import load_circ_mv  # noqa: E402
from run_walk_forward import run_windows  # noqa: E402

# 🔴 分层逻辑**唯一来源**。初版在本文件内自己写了一份，
#   且用窗口末日市值分层 ⇒ 循环论证（大盘层基准虚高 18.5pp）。
from size_decile_core import (  # noqa: E402
    interpret_layer_median,
    size_decile_run,
)

from factor_lab.analysis.costs import CostModel  # noqa: E402
from factor_lab.analysis.walk_forward import (  # noqa: E402
    WindowSpec,
    judge,
    make_windows,
)

OUTPUT = ROOT / "runtime" / "financial_walk_forward"


def report_decile(rows: list[dict], factor: str) -> dict:
    """按层打印层内超额并判读，返回层内中位（供汇总表用）。"""
    df = pd.DataFrame(rows)
    if df.empty:
        print("  ⚠ 分层无有效数据")
        return {}
    piv = df.pivot_table(index="窗口", columns="层", values="层内超额")
    b_all = df["全市场基准"].dropna()
    print(f"\n  层内超额（策略 − 同层等权基准），"
          f"全市场等权基准中位 {b_all.median():+.2%}" if len(b_all)
          else "\n  层内超额（策略 − 同层等权基准）")
    print("  " + "─" * 56)
    print(f"  {'窗口':>4}{'大盘层':>11}{'中盘层':>11}{'小盘层':>11}")
    for w in piv.index:
        print(f"  {int(w):>4}" + "".join(
            f"{piv.loc[w, k]:>11.2%}" if k in piv.columns
            and pd.notna(piv.loc[w, k]) else f"{'N/A':>11}"
            for k in ("大", "中", "小")))
    med = {k: float(np.nanmedian(piv[k])) for k in ("大", "中", "小")
           if k in piv.columns}
    print("  " + "─" * 56)
    print(f"  {'中位':>4}" + "".join(
        f"{med[k]:>11.2%}" if k in med else f"{'N/A':>11}"
        for k in ("大", "中", "小")))
    lvl, msg = interpret_layer_median(med)
    print(f"    {lvl} {msg}")
    return med


def main() -> int:
    ap = argparse.ArgumentParser(description="财务因子滚动 + 市值分层检验")
    ap.add_argument("--factors", default="bp,ep,roe")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--n-hold", type=int, default=30)
    ap.add_argument("--n-pick-mult", type=int, default=3)
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--neutral", action="store_true",
                    help="行业 + 市值 双中性化")
    ap.add_argument("--skip-decile", action="store_true")
    args = ap.parse_args()

    names = [x.strip() for x in args.factors.split(",") if x.strip()]
    cost = CostModel()
    spec = WindowSpec(train_years=args.train_years, test_months=12)

    print("=" * 88)
    print("财务因子：滚动样本外 + 市值分层混淆检验")
    print("=" * 88)
    # ⚠️ **绩效声明四要素**（P14）：年化口径 / 成本假设 / 基准 / 多空方向。
    #    缺一不可 —— 初版只在前两行打印成本，表头无口径标注（review MINOR）。
    print("─" * 88)
    print("绩效声明（四要素，缺一不可）：")
    print("  1 年化口径：自然日 /365.25（_year_span），**非** len/252"
          "（252 是美股口径，A 股 243~245）")
    print(f"  2 成本假设：{cost.describe()}"
          f"，往返 {cost.round_trip * 10000:.0f}bp，**已扣除**")
    print("  3 基准：同期**同层**股票子集的等权收益（层内基准）"
          " + 全市场等权（参考列）")
    print("     ⚠ 基准在**窗口内重算** pct_change —— 全样本算完再切片"
          "会多算 1 天且那天来自训练期")
    print("  4 多空方向：本表是**单边多头**（A 股散户无法做空），"
          "选因子值最高的 N 只；多空结论见 run_financial_study")
    print("─" * 88)
    print(f"区间 {args.start} ~ {args.end}   窗口 {spec.describe()}")
    print(f"持仓 {args.n_hold} 只 / 候选池 {args.n_hold * args.n_pick_mult} 只"
          f"   月频调仓")
    print(f"中性化 {'行业+市值' if args.neutral else '不做（原始因子）'}")
    print("\n⚠ 判据是「各窗口超额的分布」，不是平均年化。")

    t0 = time.perf_counter()
    panels, price = build_wide_panels(names=names, start=args.start,
                                      end=args.end, neutral=args.neutral)
    print(f"  宽表面构建耗时 {time.perf_counter() - t0:.1f}s")

    mcap = None
    if not args.skip_decile:
        mcap = load_circ_mv(args.start.replace("-", ""),
                            args.end.replace("-", ""))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for f in names:
        raw = panels.get(f)
        if raw is None or raw.empty:
            print(f"\n⚠ 跳过 {f}（无数据）")
            continue
        z = _cross_z(raw)
        dates = sorted(set(z.index) & set(price.index))
        z, price_f = z.loc[dates], price.loc[dates]
        windows = make_windows(dates, spec)
        if not windows:
            print(f"\n⚠ {f}: 样本 {dates[0]:%Y-%m-%d} ~ {dates[-1]:%Y-%m-%d}"
                  f"（{len(dates)} 交易日）无法生成窗口。"
                  f"\n  原因：`make_windows` 要求至少"
                  f" {spec.train_years} 年训练 + {spec.test_months} 月测试，"
                  f"且测试段 ≥60 日、训练段 ≥{spec.min_train_years*200} 日。"
                  f"\n  ⇒ 把 --start 往前推（如 20160101），"
                  f"或用 --train-years 2 缩短训练期。")
            continue

        print(f"\n{'━'*72}\n【{f}】滚动样本外（{len(windows)} 个不重叠测试段）")
        wins = run_windows({f: z}, price_f, f, spec, cost,
                           args.n_hold, args.n_pick_mult)
        if not wins:
            continue
        wdf = pd.DataFrame(wins)
        wdf.to_csv(OUTPUT / f"windows_{f}.csv", index=False,
                   encoding="utf-8-sig")
        res = judge(wins, f)
        print(f"  {'窗口':>4}{'测试区间':>26}{'策略':>9}{'基准':>9}"
              f"{'超额':>10}{'换手':>8}")
        print("  " + "─" * 62)
        for _, r in wdf.iterrows():
            if pd.isna(r.get("策略")):
                print(f"  {int(r['窗口']):>4}{r['测试起']}~{r['测试止']:>14}"
                      f"{'失败':>9}  {r.get('备注','')}")
                continue
            print(f"  {int(r['窗口']):>4}{r['测试起']}~{r['测试止']:>14}"
                  f"{r['策略']:>9.2%}{r['基准']:>9.2%}{r['超额']:>+10.2%}"
                  f"{r['换手']:>8.1%}")
        print("\n  " + res.report().replace("\n", "\n  "))
        print(f"  超额为正的窗口: {(wdf['超额'] > 0).sum()}/{len(wdf)}")

        rec = {"因子": f, "窗口数": res.窗口数, "胜率": res.胜率,
               "超额中位数": res.超额中位数, "最差": res.最差超额,
               "最好": res.最好超额, "可用": res.可用, "原因": res.原因}

        if mcap is not None:
            print(f"\n  市值分层混淆检验（{f}）")
            drows = size_decile_run(z, price_f, mcap, windows, cost,
                                    args.n_hold, args.n_pick_mult, f)
            med = report_decile(drows, f)
            pd.DataFrame(drows).to_csv(
                OUTPUT / f"size_decile_{f}.csv", index=False,
                encoding="utf-8-sig")
            layer_vals = [v for v in med.values() if pd.notna(v)]
            rec["层内超额中位"] = float(np.median(layer_vals)) \
                if layer_vals else np.nan
            for k in ("大", "中", "小"):
                rec[f"{k}盘层超额中位"] = med.get(k, np.nan)

        summary.append(rec)

    if summary:
        sdf = pd.DataFrame(summary)
        print(f"\n{'━'*72}\n汇总")
        print("━" * 88)
        print(sdf.to_string(index=False, float_format=lambda x: f"{x:>9.4f}"))
        tag = "neutral" if args.neutral else "raw"
        sdf.to_csv(OUTPUT / f"summary_{tag}.csv", index=False,
                   encoding="utf-8-sig")
        try:
            from data_version import build_manifest
            ver = build_manifest()
            (OUTPUT / f"run_meta_{tag}.json").write_text(json.dumps(
                {"数据版本": ver["version"], "打戳": ver["stamped_at"],
                 "区间": [args.start, args.end], "窗口": spec.describe(),
                 "因子": names, "持仓": args.n_hold,
                 "候选池倍数": args.n_pick_mult,
                 "中性化": bool(args.neutral),
                 "成本": {"往返": cost.round_trip}},
                ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n数据版本: {ver['version']}（引用结论必须带此戳）")
        except Exception as e:                                   # noqa: BLE001
            print(f"⚠ 版本记录失败: {e}")

    print(f"\n结果目录: {OUTPUT}")
    print("以上为统计检验，不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
