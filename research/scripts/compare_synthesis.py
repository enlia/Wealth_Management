"""因子合成方法对比：等权 vs IC加权 vs 最大化IR —— 在已验证因子集上实测。

背景（2026-10-05 本机全市场实测，详见 docs/02_方法与结论/03_A股量化算法全景对比.md）：
  bp         ICIR 0.330，三段扣成本净收益全正(+1.0%/+7.1%/+2.2%)
  cf_quality ICIR 0.185，三段扣成本净收益全正
  → 唯一「三段稳健」的财务因子组合

本脚本要回答：把多个因子合成一个打分，用哪种加权法最好？
卖方（华泰/银河）普遍结论是「等权是难以打败的基准，复杂方法样本外不一定更好」，
本脚本用实测数据检验该说法在 A 股是否成立。

═══════════════════════════════════════════════════════════════════
设计要点（全部来自本项目踩过的坑，勿简化）
═══════════════════════════════════════════════════════════════════
1. 数据形状统一为长表 (date, asset) × 因子列，避免 stack/unstack 方向歧义。
2. 权重只能用【滚动窗口】IC，绝不能用全样本 IC —— 全样本 = 前视偏差。
3. 年化口径：前瞻k 日收益 ×(252/k) 才是年化。
4. 子区间检验：全样本有效≠ 稳定（AGENTS.md 强制）。
5. 成本：往返 20bp × 换手 × 分组数。

用法
----
  uv run python scripts/compare_synthesis.py --n 500# 冒烟
  uv run python scripts/compare_synthesis.py --all       # 全市场
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

from factor_lab.analysis.alphalens_adapter import run_tear_sheet
from factor_lab.config import DEFAULT_COST, DEFAULT_RESEARCH, OUTPUT_DIR, is_a_share
from factor_lab.data import all_codes, load_long, load_prices, load_stock_info
from factor_lab.data.universe import build_universe
from run_financial_study import (build_financial_factors, neutralize,
                                 pick_sample)

# 已验证的因子：三段扣成本净收益全正
DEFAULT_FACTORS = "bp,cf_quality,roe,profit_yoy"


# ════════════════════════════════════════════════════════════════
# 合成方法：入参统一为 (date, asset) MultiIndex × 因子列 的长表
# ════════════════════════════════════════════════════════════════
def zscore_cs(fac: pd.DataFrame) -> pd.DataFrame:
    """逐日截面标准化。必须逐日；用全样本统计量会造成前视偏差。"""
    def one(g):
        sd = g.std()
        if sd.abs().sum() == 0:
            return g * np.nan
        return (g - g.mean()) / sd
    return fac.groupby(level="date", group_keys=False).apply(one)


def synth_equal_w(fac: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    """等权：各因子各占 1/N。无参数、最稳健、最好解释。"""
    z = zscore_cs(fac)
    return z.mean(axis=1)


def synth_ic_w(fac: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    """IC 加权：权重 ∝ 过去 252 日滚动 IC 均值（负 IC 取 0）。

    ⚠️ w 必须是 index=date 的滚动 IC 表；用全样本 IC 是前视偏差。
    实现：把 (date, factor) 权重沿 asset 轴广播成 (date, asset, factor)。
    """
    z = zscore_cs(fac)
    dates = z.index.get_level_values("date")
    # (date, asset) × factor  →  每日一个 (asset × factor) 权重块
    blocks = {}
    for d, wd in w.reindex(dates.unique()).iterrows():
        blocks[d] = wd.reindex(z.columns).fillna(0.0).to_numpy()
    arr = np.vstack([blocks[d] for d in dates])      # (n_rows, n_factor)
    ww = pd.DataFrame(arr, index=z.index, columns=z.columns)
    num = (z * ww).sum(axis=1, min_count=1)
    den = ww.abs().sum(axis=1).reindex(num.index)
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def synth_max_ir(fac: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    """最大化 IR：权重 ∝ |滚动 IC 均值| / 滚动 IC 标准差。偏好【稳定】而非【强】。"""
    z = zscore_cs(fac)
    icm = w.mean().abs()
    ics = w.std().replace(0, np.nan)
    wd = (icm / ics).reindex(z.columns).fillna(0.0)
    dates = z.index.get_level_values("date")
    arr = np.vstack([wd.to_numpy()] * len(dates))
    ww = pd.DataFrame(arr, index=z.index, columns=z.columns)
    num = (z * ww).sum(axis=1, min_count=1)
    den = ww.abs().sum(axis=1).reindex(num.index)
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def rolling_ic(fac: pd.DataFrame, prices: pd.DataFrame,
               window: int = 252, min_periods: int = 120) -> pd.DataFrame:
    """滚动 IC：index=date, columns=factor。

    逐日算当日 IC（因子 vs 次日收益），再对 IC 序列做滚动均值。
    这样每个日期的权重只来自它【之前】的数据，天然无前视偏差。
    """
    fwd1 = prices.shift(-1) / prices - 1.0     # 次日收益
    dates = fac.index.get_level_values("date").unique()
    rows = {}
    for d in dates:
        f = fac.xs(d, level="date")
        r = fwd1.loc[d].reindex(f.index)
        ok = r.notna()
        if ok.sum() < 30:
            continue
        rows[d] = {c: f[c][ok].corr(r[ok], method="spearman") for c in fac.columns}
    ic = pd.DataFrame(rows).T.sort_index()
    return ic.rolling(window, min_periods=min_periods).mean()


# ════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="因子合成方法对比")
    ap.add_argument("--n", type=int, default=0, help="0/--all = 全市场")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--factors", default=DEFAULT_FACTORS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = DEFAULT_RESEARCH
    full = args.all or args.n == 0
    names = [x.strip() for x in args.factors.split(",") if x.strip()]

    print("=" * 74)
    print("因子合成方法对比：等权 vs IC加权 vs 最大化IR")
    print("=" * 74)
    print(f"因子集: {', '.join(names)}")
    print(f"股票池: {'全市场 A 股' if full else f'{args.n} 只（分层抽样）'}")
    print(f"中性化: 行业 + 市值（逐日截面 OLS 残差法）")
    print(f"成本: 往返 {DEFAULT_COST.round_trip*10000:.0f}bp   分组 {cfg.quantiles}")

    # ── 1. 载入 ──────────────────────────────────────────────
    t0 = time.perf_counter()
    codes = [c for c in all_codes() if is_a_share(c)]
    if not full:
        codes = pick_sample(codes, args.n)
    print(f"\n[1/5] 读取 {len(codes):,} 只 …")
    long = load_long(codes, start=cfg.start_date, end=cfg.end_date)
    info = load_stock_info()
    info = info[info["code"].isin(codes)]
    long = build_universe(long, cfg, info=info, verbose=False)
    alive = long["code"].unique().tolist()
    prices = load_prices(alive, start=cfg.start_date, end=cfg.end_date, field="close")
    print(f"      股票池 {len(alive):,} 只 × {prices.shape[0]:,} 日"
          f"，{time.perf_counter()-t0:.1f}s")

    # ── 2. 构造因子长表 ──────────────────────────────────────
    print("\n[2/5] 构造因子长表 …")
    panel = pd.read_parquet(Path(__file__).resolve().parents[2] / "runtime" / "financial_panel.parquet")
    panel = panel[panel["sym"].isin(alive)]
    shares = info.set_index("code")["shares"] if "shares" in info.columns else None
    fac = build_financial_factors(panel, prices.index, prices, shares=shares)
    daily = fac.pop("_daily")
    ind_map, cap_map = daily["industry"], daily["mktcap"]

    parts = []
    for n in names:
        f = fac[n]
        f = neutralize(f, ind_map.reindex(f.index).to_numpy(),
                       cap_map.reindex(f.index).to_numpy())
        f.name = n
        parts.append(f)
    fac_long = pd.concat(parts, axis=1).dropna(how="all")
    print(f"      {fac_long.shape[0]:,} 行 × {fac_long.shape[1]} 因子，"
          f"覆盖 {fac_long.index.get_level_values('date').nunique():,} 天")

    # ── 3. 滚动 IC（只用历史）────────────────────────────────
    print("\n[3/5] 计算滚动 IC（252 日窗口）…")
    ic_hist = rolling_ic(fac_long, prices)
    print(f"      有效 IC 行数 {ic_hist.dropna(how='all').shape[0]}")

    # ── 4. 逐方法检验 ───────────────────────────────────────
    print("\n[4/5] 逐方法检验")
    methods = {
        "equal_w": ("等权", synth_equal_w),
        "ic_w": ("IC加权（滚动252日）", synth_ic_w),
        "max_ir": ("最大化IR", synth_max_ir),
    }
    out_dir = Path(args.out) if args.out else OUTPUT_DIR / "synthesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    results, rows = {}, []

    for key, (label, fn) in methods.items():
        print("\n" + "-" * 74)
        print(f"方法: {label}")
        print("-" * 74)
        try:
            t1 = time.perf_counter()
            score = fn(fac_long, ic_hist)
            score = score.replace([np.inf, -np.inf], np.nan).dropna()
            print(f"  合成因子值 {len(score):,} 个")
            if len(score) < 1000:
                print("  ✗ 有效值过少")
                continue
            r = run_tear_sheet(score, prices, cfg, DEFAULT_COST, factor_name=key)
            print(f"  耗时 {time.perf_counter()-t1:.1f}s")
            print("  " + r.summary().replace("\n", "\n  "))
            results[key] = score
            for p in (1, 5, 20, 60):
                if p in r.ic_by_period.index:
                    rr = r.ic_by_period.loc[p]
                    rows.append({
                        "方法": label, "前瞻期": p,
                        "IC": rr["mean"], "ICIR": rr["ir"], "t值": rr["tstat"],
                        "年化毛": r.gross_spread if p == 1 else np.nan,
                        "年化净": r.net_spread_after_cost if p == 1 else np.nan,
                        "换手": r.turnover_mean,
                    })
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  ✗ 失败: {type(e).__name__}: {e}")
            traceback.print_exc()

    # ── 5. 汇总 + 子区间 ────────────────────────────────────
    if rows:
        summ = pd.DataFrame(rows)
        print("\n" + "=" * 74)
        print("汇总")
        print("=" * 74)
        print(summ.to_string(index=False, float_format=lambda x: f"{x:>9.4f}"))
        summ.to_csv(out_dir / "synthesis_compare.csv", index=False, encoding="utf-8-sig")

        print("\n子区间稳定性（1 日前瞻）：")
        for key, (label, _) in methods.items():
            if key not in results:
                continue
            print(f"\n  ── {label}")
            for lab2, (s, e) in cfg.sub_periods.items():
                rng = pd.date_range(s, e)
                sc = results[key]
                f2 = sc[sc.index.get_level_values("date").isin(rng)]
                p2 = prices.loc[prices.index.intersection(rng)]
                if f2.empty or f2.index.get_level_values("asset").nunique() < 30:
                    continue
                try:
                    r2 = run_tear_sheet(f2, p2, cfg, DEFAULT_COST, factor_name=key)
                    print(f"    {lab2}  IC={r2.ic_by_period.loc[1,'mean']:+.4f} "
                          f"ICIR={r2.ic_by_period.loc[1,'ir']:+.3f} "
                          f"毛={r2.gross_spread*100:+.1f}% "
                          f"净={r2.net_spread_after_cost*100:+.2f}%")
                except Exception as e:  # noqa: BLE001
                    print(f"    {lab2} 失败: {e}")

    print(f"\n结果目录: {out_dir}")
    print("提示: 多空组合口径，A 股散户无法做空。统计检验不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())