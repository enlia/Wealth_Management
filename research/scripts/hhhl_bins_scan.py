"""hh_hl_score 等宽分箱复测扫描（实验轮 1·B 线）。

用途
----
在等宽分箱（bins=3，bd64b99 实装）口径下复测 hh_hl_score 还剩多少信号：
  1. 冒烟（--n 5~10）跑通「分箱 → 打分 → 分组检验」全流程；
  2. 全样本 + 三段子区间（2016-2018 / 2019-2022 / 2023-2026）的 IC / 分组收益 / 换手；
  3. 箱数稳健性核对（--bins 2,3,5 对照）——**只用于「参数是否生效」核对
     （检验框架铁律 6：数值两两完全相同 = 死参数），不用于选优**；
     bins=3 是锁定先验（brief 红线：禁过度拟合式参数搜索），扫描结果不改先验。

口径三件套（本脚本全部产出的强制列头）
--------------------------------------
  价格口径 : 后复权 close_adj / high_adj / low_adj（bd64b99 起）
  换手单位 : 单期换手 = %/交易日（alphalens quantile_turnover(period=1) 分位均值）；
             折年换手 = 单期换手 × 252（倍/年）
  年化方式 : 线性 ×252（spread × 252/periods[0]；年化 = 每期均值倍数换算，
             不是几何年化，也不是 /交易日 未年化——D8 口径判决）

P14 绩效四要素
--------------
  年化口径=每期均值 ×252 线性；成本=config.DEFAULT_COST 往返 20bp
  （买 7.5bp + 卖 12.5bp，config.CostModel）；基准=等权全市场；
  多空=最高分位多头 − 最低分位空头（**多空组合口径，A 股散户不可实盘**）。

用法
----
  .venv/Scripts/python.exe research/scripts/hhhl_bins_scan.py --n 8 --skip-audit \\
      --out runtime/experiment1b/smoke
  .venv/Scripts/python.exe research/scripts/hhhl_bins_scan.py --all \\
      --factors hhhl20 --bins 2,3,5 --out runtime/experiment1b/full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 复用既有前置关卡（数据审计过闸），不另造（AGENTS 一·1 复用优先）
from run_factor_study import run_precheck

from factor_lab.analysis.alphalens_adapter import run_tear_sheet, subperiod_ic
from factor_lab.config import DEFAULT_COST, DEFAULT_RESEARCH, is_a_share
from factor_lab.data import all_codes, load_long, load_prices, load_stock_info
from factor_lab.data.universe import build_universe, summarize_universe
from factor_lab.factors.price_volume import bins_of, compute_factor

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 220)

CALIBER = {
    "口径_价格": "后复权 *_adj",
    "口径_换手单位": "单期换手=%/交易日（分位均值）；折年=单期×252 倍/年",
    "口径_年化方式": "线性 ×252（每期均值倍数换算）",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="hh_hl_score 等宽分箱复测扫描")
    p.add_argument("--n", type=int, default=200,
                   help="股票池大小上限；0 或 --all 表示全市场")
    p.add_argument("--all", action="store_true", help="全市场")
    p.add_argument("--seed", type=int, default=20240101)
    p.add_argument("--factors", default="hhhl20", help="逗号分隔因子名")
    p.add_argument("--bins", default="2,3,5",
                   help="等宽箱数扫描列表（死参数核对用，不选优）")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--out", default=None, help="结果 CSV 输出目录")
    p.add_argument("--skip-audit", action="store_true",
                   help="跳过数据审计前置检查（仅冒烟/调试用，结论不可引用）")
    return p.parse_args()


def seg_distribution(seg: pd.DataFrame, label: str) -> dict:
    """三段子区间的判据分布（检验框架铁律 5：判据是分布不是均值）。"""
    nets = [float(v) for v in seg["多空净"] if pd.notna(v)]
    if not nets:
        return {"范围": label, "有效段数": 0, "胜率(段净>0)": "未知",
                "中位段净": "未知", "最差段净": "未知"}
    arr = np.array(nets)
    return {
        "范围": label, "有效段数": len(arr),
        "胜率(段净>0)": f"{int((arr > 0).sum())}/{len(arr)}",
        "中位段净": f"{np.median(arr) * 100:+.2f}%",
        "最差段净": f"{arr.min() * 100:+.2f}%",
    }


def dead_param_check(rows: list[dict], cols: list[str]) -> list[str]:
    """箱数是否为死参数（检验框架铁律 6：数值完全相同 = 参数没生效）。

    对 3 取值离散因子，bins>3 与 bins=3 可能给出**同一分组划分**
    （标签跳号如 [1,3,5]）——此时毛/换手相同是"划分退化"，不是参数没生效；
    但成本乘数取最大标签（5 vs 3），净收益会不同。逐列标注区分两种情形。
    """
    warns = []
    df = pd.DataFrame(rows)
    for name, g in df.groupby("因子"):
        labels = g["实际分位标签"].tolist()
        same_partition = len(set(labels)) < len(labels)
        for c in cols:
            vals = g[c].round(10).tolist()
            if len(vals) > 1 and len(set(vals)) == 1:
                why = ("同划分退化（标签 " + "/".join(map(str, labels))
                       + "，成本乘数随最大标签而变）" if same_partition
                       else "参数未生效")
                warns.append(f"⚠ 数值完全相同: {name} 的 {c} = {vals} → {why}")
    return warns


def main() -> int:
    args = parse_args()
    cfg = DEFAULT_RESEARCH
    if args.start:
        cfg = type(cfg)(**{**cfg.__dict__, "start_date": args.start})
    if args.end:
        cfg = type(cfg)(**{**cfg.__dict__, "end_date": args.end})

    full = args.all or args.n == 0
    run_precheck(args.skip_audit)
    bins_list = [int(b) for b in args.bins.split(",") if b.strip()]

    print("=" * 84)
    print("hh_hl_score 等宽分箱复测扫描（实验轮 1·B 线）")
    print("=" * 84)
    for k, v in CALIBER.items():
        print(f"  {k} = {v}")
    print("  P14 四要素: 年化=线性×252 | 成本=往返 20bp（买 7.5bp+卖 12.5bp，"
          "乘 组数×单期换手） | 基准=等权全市场 | 多空=最高分位−最低分位（不可实盘）")
    print(f"  因子 {args.factors}   箱数扫描 {bins_list}（锁定先验 bins=3，扫描仅供死参数核对）")
    print(f"  区间 {cfg.start_date} ~ {cfg.end_date}   子区间 {list(cfg.sub_periods)}")
    print(f"  股票池: {'全市场 A 股' if full else f'随机 {args.n} 只 A 股（seed={args.seed}）'}")

    t0 = time.perf_counter()
    codes = [c for c in all_codes() if is_a_share(c)]
    if not full:
        rng = np.random.default_rng(args.seed)
        if len(codes) > args.n:
            codes = sorted(rng.choice(codes, size=args.n, replace=False))
    print(f"\n[1/4] 读取 {len(codes):,} 只标的长表（前复权 qfq）…")
    long = load_long(codes, start=cfg.start_date, end=cfg.end_date, adjusted=True)
    if "close_adj" not in long.columns:
        raise RuntimeError("长表缺少 close_adj 列，拒绝在未复权口径上做收益研究")
    print(f"      {len(long):,} 行，耗时 {time.perf_counter()-t0:.1f}s")

    info = load_stock_info()
    info = info[info["code"].isin(codes)]
    print("\n[2/4] 构建股票池")
    long = build_universe(long, cfg, info=info)
    summarize_universe(long)
    alive = long["code"].unique().tolist()

    print("\n[3/4] 读取价格宽表（close_adj）…")
    prices = load_prices(alive, start=cfg.start_date, end=cfg.end_date,
                         field="close_adj")
    if prices.empty:
        raise RuntimeError("close_adj 面板为空，拒绝继续")

    names = [s.strip() for s in args.factors.split(",") if s.strip()]
    out_dir = Path(args.out) if args.out else Path("runtime") / "hhhl_bins_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[4/4] 逐因子 × 逐箱数复测: {names} × {bins_list}")

    rows: list[dict] = []
    dist_rows: list[dict] = []
    for name in names:
        factor = compute_factor(name, long)
        nat_bins = bins_of(name)          # 连续因子为 None（等频分位）
        scan = bins_list if nat_bins else [nat_bins]
        for b in scan:
            run_bins = b if nat_bins else None
            tag = f"bins={b}" if nat_bins else "quantiles=5（连续因子）"
            print(f"\n  ── {name}  {tag}")
            try:
                r = run_tear_sheet(factor, prices, cfg, DEFAULT_COST,
                                   factor_name=name, bins=run_bins)
                labels = sorted(int(x) for x in r.quantile_returns.index)
                cost_coef = (DEFAULT_COST.round_trip * r.quantiles * 252.0)
                print("     " + r.summary().replace("\n", "\n     "))
                rows.append({
                    "因子": name, "箱数": b, "实际分位标签": str(labels),
                    "组数(成本乘数)": r.quantiles,
                    "IC均值": r.ic_by_period.loc[1, "mean"],
                    "ICIR": r.ic_by_period.loc[1, "ir"],
                    "IC样本数": int(r.ic_by_period.loc[1, "count"]),
                    "多空年化毛": r.gross_spread,
                    "多空年化净": r.net_spread_after_cost,
                    "平均单期换手": r.turnover_mean,
                    "折年换手倍数": r.turnover_mean * 252.0,
                    "每单位换手成本系数": cost_coef,
                    **CALIBER,
                })
                sub = subperiod_ic(factor, prices, cfg, DEFAULT_COST, name,
                                   bins=run_bins)
                sub = sub.copy()
                # 段级单期换手由成本恒等式反推：(毛−净) / (0.0020 × 组数 × 252)
                q = r.quantiles
                sub["单期换手_反推"] = (sub["多空年化"] - sub["多空净"]) / (
                    DEFAULT_COST.round_trip * q * 252.0)
                print("     " + sub.to_string(index=False).replace("\n", "\n     "))
                dist_rows.append({"因子": name, "箱数": b,
                                  **seg_distribution(sub, f"{name}/{tag}")})
                sub.insert(0, "因子", name)
                sub.insert(1, "箱数", b)
                for k, v in CALIBER.items():
                    sub[k] = v
                sub.to_csv(out_dir / f"subperiod_{name}_bins{b}.csv",
                           index=False, encoding="utf-8-sig")
            except Exception as e:                     # noqa: BLE001
                # 失败必须显式落表，不静默跳过（工程铁律 4 禁静默 fallback / brief 红线）
                print(f"     ✗ 失败: {type(e).__name__}: {e}")
                rows.append({"因子": name, "箱数": b, "实际分位标签": "失败",
                             "失败": f"{type(e).__name__}: {e}", **CALIBER})
                dist_rows.append({"因子": name, "箱数": b,
                                  "有效段数": 0, "胜率(段净>0)": "失败",
                                  "中位段净": "失败", "最差段净": "失败"})

    summ = pd.DataFrame(rows)
    summ.to_csv(out_dir / "summary_bins.csv", index=False, encoding="utf-8-sig")
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(out_dir / "seg_distribution.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 84)
    print("箱数稳健性 / 死参数核对（检验框架铁律 6：完全相同 = 参数没生效）")
    print("=" * 84)
    print(summ[["因子", "箱数", "实际分位标签", "IC均值", "ICIR",
                "多空年化毛", "多空年化净", "平均单期换手"]].to_string(index=False))
    warns = dead_param_check(
        [r for r in rows if "失败" not in r],
        # 只核对**依赖分箱**的列。IC/ICIR 在原因子值上算（与分箱无关），
        # 各箱数相同是必然、不是死参数（冒烟实测误报一轮，遂收窄口径）。
        ["多空年化毛", "多空年化净", "平均单期换手"])
    for w in warns:
        print(w)
    if not warns:
        print("  ✓ 各箱数数值两两有别，bins 参数生效（非死参数）")

    print("\n三段判据分布（只走分布不判均值）:")
    print(dist.to_string(index=False))

    meta = {
        "口径三件套": CALIBER,
        "因子": args.factors,
        "箱数扫描": bins_list,
        "区间": f"{cfg.start_date} ~ {cfg.end_date}",
        "股票池": "全市场" if full else f"随机 {args.n} 只 (seed={args.seed})",
        "标注": "分箱复核实测·发布待双闸",
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果目录: {out_dir}")
    print("以上为统计检验输出（分箱复核实测·发布待双闸），不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
