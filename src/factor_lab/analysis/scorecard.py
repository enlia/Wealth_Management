"""因子/组合打分卡：统一评估口径，一次跑出所有关键指标。

设计目的
--------
项目已有零散的检验脚本（run_factor_study / run_financial_study /
compare_synthesis），但无法横向对比。本模块提供**唯一评估入口**，
输出统一格式的评分卡，便于回答「A 方案是否优于 B 方案」。

核心指标（全部必看，缺一不可）
  IC          因子值与次日收益的 Spearman 秩相关，衡量排序能力
  ICIR        IC 均值 ÷ IC 标准差，衡量稳定性（业界 >0.3 才算可用）
  年化毛/净   扣 20bp 往返成本前后的多空年化
  三段稳定性  2016-2018 / 2019-2022 / 2023-2026 的 IC 与净收益

评分规则（本项目实证得出，写死在SCORING_RULES 里）
  ✅ 可用因子   三段 IC 同号且 ICIR ≥ 0.15
  ⚠️ 观察区     三段同号但 ICIR < 0.15，或某段样本不足
  ❌ 剔除       IC 三段不一致，或任一段净收益显著为负

⚠️ 为什么强调「三段」：本项目实测发现 ep 的全样本净收益 +6.1%，
   但三段是 +33% / +0.3% / −5.0%；roe 是 +30.7% / −6.4% / −9.9%。
   **均值为正但路径不可持有**，只看全样本会严重误判。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# 评分规则（集中在此，便于调整与审阅）
SCORING_RULES = {
    "min_icir": 0.15,          # ICIR 门槛
    "strong_icir": 0.30,       # 强信号门槛（业界标准）
    "min_t": 2.0,              # t 值门槛
    "neg_net_threshold": -0.02,  # 某段净收益低于 -2% 视为不可用
    "sub_periods": {
        "2016-2018": ("2016-01-01", "2018-12-31"),
        "2019-2022": ("2019-01-01", "2022-12-31"),
        "2023-2026": ("2023-01-01", "2026-09-30"),
    },
}


def grade_row(ic_by_seg: list[float], icir: float, nets: list[float],
              rules: dict | None = None) -> tuple[str, str]:
    """按规则给单行打分，返回 (等级, 理由)。

    等级：✅ 可用 / ⚠️ 观察 / ❌ 剔除
    """
    r = {**SCORING_RULES, **(rules or {})}
    ic_valid = [x for x in ic_by_seg if x is not None and np.isfinite(x)]
    net_valid = [x for x in nets if x is not None and np.isfinite(x)]

    if len(ic_valid) < len(r["sub_periods"]):
        return "⚠️", f"仅 {len(ic_valid)}/{len(r['sub_periods'])} 段有效"

    same_sign = all(x > 0 for x in ic_valid) or all(x < 0 for x in ic_valid)
    if not same_sign:
        return "❌", f"三段 IC 符号不一致：{[f'{x:+.4f}' for x in ic_valid]}"

    bad = [x for x in net_valid if x < r["neg_net_threshold"]]
    if bad:
        return "❌", f"存在净收益 < {r['neg_net_threshold']:.0%} 的子区间：{[f'{x:+.1%}' for x in bad]}"

    if icir < r["min_icir"]:
        return "⚠️", f"ICIR {icir:.3f} < {r['min_icir']}（方向对但强度不足）"

    tier = "强" if icir >= r["strong_icir"] else "可用"
    return "✅", f"{tier}：ICIR {icir:.3f}，三段同号"


def build_scorecard(
    name: str,
    factor: pd.Series,
    prices: pd.DataFrame,
    cost_round_trip: float = 0.0020,
    quantiles: int = 5,
    rules: dict | None = None,
) -> dict:
    """跑一次完整检验并返回打分卡行。

    返回 dict 含 ic/icir/t/annual_gross/annual_net/turnover/
    seg_ic/seg_net/grade/reason。
    """
    from alphalens import performance, utils

    r = {**SCORING_RULES, **(rules or {})}
    periods = (1, 5, 20, 60)

    f = factor.replace([np.inf, -np.inf], np.nan).dropna()
    if len(f) < 1000:
        return {"name": name, "grade": "❌", "reason": f"有效值仅 {len(f):,}，过少"}

    clean = utils.get_clean_factor_and_forward_returns(
        factor=f, prices=prices, quantiles=quantiles,
        periods=list(periods), max_loss=0.5,
    )

    # ── 全样本 IC ──
    ic_daily = performance.factor_information_coefficient(clean)
    s = ic_daily["1D"].dropna()
    if len(s) < 100:
        return {"name": name, "grade": "❌", "reason": "IC 样本不足"}
    sd = s.std()
    icir = s.mean() / sd if sd else np.nan

    # ── 多空收益 + 换手 ──
    qr, _ = performance.mean_return_by_quantile(clean)
    tr = pd.DataFrame({
        q: performance.quantile_turnover(clean["factor_quantile"], q)
        for q in range(1, quantiles + 1)
    })
    turnover = float(tr.mean().mean())
    spread = float(qr.loc[quantiles, "1D"] - qr.loc[1, "1D"])
    ppy = TRADING_DAYS / 1
    gross = spread * ppy
    net = gross - cost_round_trip * turnover * quantiles * ppy

    # ── 分组收益单调性（好因子的 Q1<Q2<...<Q5）──
    qvals = [float(qr.loc[q, "1D"]) for q in range(1, quantiles + 1)]
    monotonic = sum(qvals[i] <= qvals[i + 1] for i in range(len(qvals) - 1))

    # ── 子区间 ──
    seg_ic, seg_net = [], []
    dates = f.index.get_level_values("date")
    for lab, (a, b) in r["sub_periods"].items():
        rng = pd.date_range(a, b)
        f2 = f[dates.isin(rng)]
        # ⚠️ 坑：prices.loc[Index.intersection(...)] 可能退化成普通 Index，
        #    alphalens 的 compute_forward_returns 会因找不到 .tz 报
        #    "'Index' object has no attribute 'tz'"。
        #    正解：显式 reindex 到仍在 prices 里的日期。
        p2 = prices.reindex(prices.index.intersection(rng))
        n_assets = (f2.index.get_level_values("asset").nunique()
                    if len(f2) else 0)
        if f2.empty or n_assets < 30:
            seg_ic.append(np.nan); seg_net.append(np.nan); continue
        try:
            c2 = utils.get_clean_factor_and_forward_returns(
                factor=f2, prices=p2, quantiles=quantiles, periods=[1],
                max_loss=0.5)
            i2 = performance.factor_information_coefficient(c2)["1D"].dropna()
            q2, _ = performance.mean_return_by_quantile(c2)
            t2 = pd.DataFrame({
                q: performance.quantile_turnover(c2["factor_quantile"], q)
                for q in range(1, quantiles + 1)})
            to2 = float(t2.mean().mean())
            sp2 = float(q2.loc[quantiles, "1D"] - q2.loc[1, "1D"])
            g2 = sp2 * TRADING_DAYS
            seg_ic.append(float(i2.mean()) if len(i2) else np.nan)
            seg_net.append(g2 - cost_round_trip * to2 * quantiles * TRADING_DAYS)
        except Exception:  # noqa: BLE001
            seg_ic.append(np.nan); seg_net.append(np.nan)

    grade, reason = grade_row(seg_ic, icir, seg_net, r)

    return {
        "name": name,
        "ic": float(s.mean()),
        "icir": float(icir),
        "t": float(s.mean() / sd * np.sqrt(len(s))) if sd else np.nan,
        "annual_gross": gross,
        "annual_net": net,
        "turnover": turnover,
        "monotonic": f"{monotonic}/{quantiles - 1}",
        "seg_ic": seg_ic,
        "seg_net": seg_net,
        "grade": grade,
        "reason": reason,
    }


def print_scorecard(rows: list[dict], title: str = "因子打分卡") -> None:
    """打印格式化的打分卡。"""
    if not rows:
        print("无数据")
        return
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)
    print(f"{'因子':<22} {'等级':<5} {'IC':>9} {'ICIR':>7} {'t':>7} "
          f"{'年化毛':>9} {'年化净':>9} {'换手':>7} {'单调':>6}  理由")
    print("-" * 108)
    for r in sorted(rows, key=lambda x: (x.get("grade", ""), -(x.get("icir") or 0))):
        if "ic" not in r:
            print(f"{r['name']:<22} {r['grade']:<5} {r['reason']}")
            continue
        print(f"{r['name']:<22} {r['grade']:<5} {r['ic']:>+9.4f} {r['icir']:>7.3f} "
              f"{r['t']:>+7.2f} {r['annual_gross']*100:>8.1f}% {r['annual_net']*100:>8.2f}% "
              f"{r['turnover']*100:>6.1f}% {r['monotonic']:>6}  {r['reason']}")

    print("-" * 108)
    print(f"{'因子':<22} {'三段 IC':<34} {'三段净收益'}")
    for r in rows:
        if "seg_ic" not in r:
            continue
        sic = " / ".join(f"{x:+.4f}" if np.isfinite(x) else "  n/a  " for x in r["seg_ic"])
        sn = " / ".join(f"{x:+.2%}" if np.isfinite(x) else " n/a " for x in r["seg_net"])
        print(f"{r['name']:<22} {sic:<34} {sn}")


def best_available(rows: list[dict], top: int = 5) -> list[str]:
    """筛出评分 ✅ 的因子，按 ICIR 降序。"""
    ok = [r for r in rows if r.get("grade") == "✅" and np.isfinite(r.get("icir", np.nan))]
    ok.sort(key=lambda x: -x["icir"])
    return [r["name"] for r in ok[:top]]

def screen_low_turnover(rows: list[dict], max_turnover: float = 0.06) -> list[dict]:
    """在可用因子中筛出「换手可控」的。

    动机：本项目实测出现大量「IC 强但净收益极差」的因子——
    ma_bias_20 的 ICIR=0.453（很强），年化净收益却 −63%，
    因为换手率高达 34.9%（年成本 = 20bp × 34.9% × 5 组 = 35%）。
    → 光看 IC/ICIR 会选出完全不可用的组合，必须同时约束换手。
    """
    ok = [r for r in rows
          if r.get("grade") == "✅" and np.isfinite(r.get("turnover", np.nan))
          and r["turnover"] <= max_turnover]
    ok.sort(key=lambda x: -(x["icir"] or 0))
    return ok


def rank_by_net(rows: list[dict]) -> list[dict]:
    """按年化净收益排序（最贴近实盘的口径）。"""
    ok = [r for r in rows
          if np.isfinite(r.get("annual_net", np.nan)) and r["annual_net"] > 0]
    ok.sort(key=lambda x: -x["annual_net"])
    return ok
