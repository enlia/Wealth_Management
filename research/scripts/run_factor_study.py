"""动量/反转因子有效性检验 —— 主脚本

复用 alphalens-reloaded 做 IC/分组/收益分析，本脚本只负责组装与汇报。

用法
----
  uv run python scripts/run_factor_study.py --n 200            # 小样本冒烟
  uv run python scripts/run_factor_study.py --n 0 --all        # 全市场
  uv run python scripts/run_factor_study.py --factors mom60_skip20,rev5,vol60
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from factor_lab.analysis.alphalens_adapter import run_tear_sheet, subperiod_ic
from factor_lab.config import DEFAULT_COST, DEFAULT_RESEARCH, OUTPUT_DIR, is_a_share
from factor_lab.data import all_codes, load_long, load_prices, load_stock_info
from factor_lab.data.universe import build_universe, summarize_universe
from factor_lab.factors.price_volume import FACTORY, bins_of, compute_factor

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="动量/反转因子有效性检验")
    p.add_argument("--n", type=int, default=200,
                   help="股票池大小上限；0 或 --all 表示全市场")
    p.add_argument("--all", action="store_true", help="全市场")
    p.add_argument("--factors", default="mom60_skip20,rev5,vol60",
                   help=f"逗号分隔的因子名，可用: {','.join(sorted(FACTORY))}")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--out", default=None, help="结果 CSV 输出目录")
    p.add_argument("--skip-audit", action="store_true",
                   help="跳过数据审计前置检查（仅调试用，结论不可引用）")
    return p.parse_args()


def run_precheck(skip: bool) -> None:
    """因子研究的前置关卡：数据审计必须通过（AGENTS.md 硬约束）。

    为什么放在这里而不是靠人工记得跑：
    「年化偏差 < 0.5pp」是硬红线，但审计脚本是手动运行的
    （CI 上没有 market.db，见 ENGINEERING.md 第五节）。
    靠自觉的前置检查等于没有检查—— 本函数把它变成**同一次运行内**的强制步骤。
    """
    if skip:
        print("⚠️ 已跳过数据审计（--skip-audit）：本次结果**不可作为收益结论引用**")
        return
    print("→ 跑数据质量审计（close_adj 口径）…")
    r = subprocess.run(
        [sys.executable,
         str(Path(__file__).parent / "audit_data_quality.py"),
         "--all", "--field", "close_adj"],
        capture_output=True, text=True,
    )
    for ln in r.stdout.split("\n"):
        if "超限率" in ln or "允许做收益结论" in ln or "禁止收益结论" in ln:
            print("   ", ln.strip())
    if r.returncode != 0:
        raise SystemExit(
            "\n✗ 数据审计未通过（规则内超限率 ≥ 0.5%），拒绝做收益研究。\n"
            "  先修数据：uv run python research/scripts/merge_tushare_into_db.py"
        )
    print("   ✓ 数据审计通过")


def main() -> int:
    args = parse_args()
    cfg = DEFAULT_RESEARCH
    if args.start:
        cfg = type(cfg)(**{**cfg.__dict__, "start_date": args.start})
    if args.end:
        cfg = type(cfg)(**{**cfg.__dict__, "end_date": args.end})

    full = args.all or args.n == 0
    run_precheck(args.skip_audit)
    print("=" * 74)
    print("动量 / 反转 / 波动率 因子有效性检验")
    print("=" * 74)
    print(f"区间 {cfg.start_date} ~ {cfg.end_date}   分组 {cfg.quantiles}   "
          f"前瞻期 {cfg.periods}   成本往返 {DEFAULT_COST.round_trip*100:.1f}bp")
    print(f"子区间检验: {list(cfg.sub_periods)}")
    print(f"股票池: {'全市场 A 股' if full else f'前 {args.n} 只 A 股（按代码排序）'}")

    # ── 1. 载入 ──────────────────────────────────────────────
    t0 = time.perf_counter()
    codes = [c for c in all_codes() if is_a_share(c)]
    if not full:
        codes = codes[: args.n]
    print(f"\n[1/5] 读取 {len(codes):,} 只标的长表 …")
    long = load_long(codes, start=cfg.start_date, end=cfg.end_date, adjusted=True)
    print(f"      {len(long):,} 行，耗时 {time.perf_counter()-t0:.1f}s")
    # ⚠️ adjusted=True 是硬要求：未复权价含除权假跳空（年化偏差 2.33pp）。
    #    这里显式断言，不靠「记得传参」——P4「静默fallback」的同类。
    if "close_adj" not in long.columns:
        raise RuntimeError(
            "长表缺少 close_adj 列，拒绝在未复权口径上做收益研究。\n"
            "  解决：确认 bar_daily 已有 close_adj 且调用 load_long(adjusted=True)"
        )
    n_adj = int(long["close_adj"].notna().sum())
    print(f"      后复权覆盖 {n_adj:,} / {len(long):,} = {n_adj/len(long)*100:.3f}%")

    info = load_stock_info()
    info = info[info["code"].isin(codes)]

    # ── 2. 股票池过滤 ───────────────────────────────────────
    print("\n[2/5] 构建股票池")
    long = build_universe(long, cfg, info=info)
    summarize_universe(long)
    alive = long["code"].unique().tolist()

    # ── 3. 价格面板 ─────────────────────────────────────────
    print("\n[3/5] 读取价格宽表（alphalens 输入，后复权）…")
    t0 = time.perf_counter()
    # ⚠️ 必须是 close_adj。alphalens 用 prices.pct_change() 算前瞻收益，
    #    未复权价会把除权跳空当成真实涨跌，直接污染 IC 与分组收益。
    prices = load_prices(alive, start=cfg.start_date, end=cfg.end_date,
                         field="close_adj")
    if prices.empty:
        raise RuntimeError(
            "close_adj 面板为空，拒绝继续（并库未执行？）。\n"
            "  解决：uv run python research/scripts/merge_tushare_into_db.py"
        )
    mem = prices.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"      形状 {prices.shape}，内存 {mem:.0f} MB，耗时 {time.perf_counter()-t0:.1f}s")

    # ── 4. 逐因子检验 ───────────────────────────────────────
    names = [n.strip() for n in args.factors.split(",") if n.strip()]
    print(f"\n[4/5] 检验 {len(names)} 个因子: {', '.join(names)}")
    results = []
    out_dir = Path(args.out) if args.out else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        print("\n" + "-" * 74)
        print(f"因子: {name}")
        print("-" * 74)
        try:
            t0 = time.perf_counter()
            factor = compute_factor(name, long)
            n_valid = int(factor.notna().sum())
            print(f"  有效因子值 {n_valid:,} 个，覆盖 {factor.index.get_level_values('asset').nunique():,} 只")
            if n_valid < 1000:
                print("  ✗ 有效值过少，跳过")
                continue
            r = run_tear_sheet(factor, prices, cfg, DEFAULT_COST, factor_name=name,
                               bins=bins_of(name))
            print(f"  耗时 {time.perf_counter()-t0:.1f}s")
            print("  " + r.summary().replace("\n", "\n  "))
            results.append(r)
            r.ic_by_period.to_csv(out_dir / f"ic_{name}.csv", encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ 失败: {type(e).__name__}: {e}")

    # ── 5. 子区间稳定性 ─────────────────────────────────────
    print("\n[5/5] 子区间稳定性检验（AGENTS.md 强制）")
    sub_tables = []
    for name in names:
        if not any(r.factor_name == name for r in results):
            continue
        print(f"\n  ── {name}")
        factor = compute_factor(name, long)
        sub = subperiod_ic(factor, prices, cfg, DEFAULT_COST, name,
                           bins=bins_of(name))
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print("    " + sub.to_string(index=False).replace("\n", "\n    "))
        sub.insert(0, "因子", name)
        sub_tables.append(sub)

    # ── 汇总 ────────────────────────────────────────────────
    if results:
        print("\n" + "=" * 74)
        print("汇总（首期 1 日前瞻的 IC 与多空年化）")
        print("=" * 74)
        summ = pd.DataFrame([{
            "因子": r.factor_name,
            "IC均值": r.ic_by_period.loc[1, "mean"],
            "ICIR": r.ic_by_period.loc[1, "ir"],
            "IC样本数": int(r.ic_by_period.loc[1, "count"]),
            "多空年化毛": r.gross_spread,
            "多空年化净": r.net_spread_after_cost,
            "平均换手": r.turnover_mean,
        } for r in results if 1 in r.ic_by_period.index])
        summ = summ.sort_values("IC均值", ascending=False)
        print(summ.to_string(index=False, float_format=lambda x: f"{x:>9.4f}"))
        summ.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    if sub_tables:
        allsub = pd.concat(sub_tables, ignore_index=True)
        allsub.to_csv(out_dir / "subperiod_ic.csv", index=False, encoding="utf-8-sig")
        print("\n子区间结果已存 subperiod_ic.csv")

    print(f"\n结果目录: {out_dir}")
    print("提示: 以上为技术形态与价量因子的统计检验，不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
