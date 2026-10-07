"""选股能力评估器：直接回答「准不准」和「赚钱能力」。

为什么需要这个
--------------
此前项目只用 IC / ICIR / 年化收益评估，那是量化视角的语言。
但用户真正关心的是两个非常直接的问题：
  1. **准不准** → 买入我选的股票，涨的概率有多大？比瞎猜高多少？
  2. **赚钱能力** → 一年赚多少？什么时候亏？亏多少？比买入持有强多少？

本模块把这两个问题翻译成可测量的指标，且**不依赖任何机器学习**。

核心指标定义（全部为「多头、实际可执行」口径）
------------------------------------------------
  命中率hit_rate      Top 组未来上涨的比例。基准是同期全市场上涨率，
                     不是 50%。因为 A 股整体上涨率本身就不是 50%。
  超额命中率 edge      Top 组上涨率 − 同期全市场上涨率。真正的选股能力。
  信息比率 IR          年化超额收益 ÷ 超额收益波动。与夏普类似但对基准化。
  最大回撤 MDD         净值从峰到谷的最大跌幅。用户最关心的「最坏情况」。
  Calmar              年化收益 ÷ 最大回撤。风险调整后收益。
  持有期收益分布       按持有期（20/60/120/250 日）算的胜率与盈亏比。

⚠️ 关键口径（AGENTS.md 3.8）
   任何绩效数字必须说明：年化口径、成本假设、基准、多空还是多头。
   本模块统一为：多头、扣往返成本、基准=同期全市场等权。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SCALING_TRADING_DAYS

TRADING_DAYS = SCALING_TRADING_DAYS  # ① 倍数换算层，唯一出处 config


# ════════════════════════════════════════════════════════════════
# 一、准不准：命中率与超额命中率
# ════════════════════════════════════════════════════════════════
def hit_rate_analysis(
    holdings: pd.Series,
    fwd_ret: pd.DataFrame,
    top_frac: float = 0.2,
    periods: tuple = (1, 5, 20, 60),
) -> pd.DataFrame:
    """计算 Top 组的命中率与超额命中率。

    参数
    ----
    holdings   index=date, columns=asset 的 0/1 矩阵，1=持有
    fwd_ret    index=date, columns=asset 的前瞻收益（已用当日价算）
    top_frac   头部比例，取收益最高的这部分
    periods    前瞻期（交易日）

    返回每个期的：Top 组上涨率、基准上涨率、超额、样本数
    """
    rows = []
    for k in periods:
        if k not in fwd_ret.columns:
            continue
        fr = fwd_ret[k]
        # Top 组：持有的股票里，取前瞻收益最高的 top_frac
        sel_hold = holdings.reindex(fr.index).fillna(0) > 0
        n_sel = int(sel_hold.sum(axis=1).mean())
        if n_sel < 10:
            continue
        kth = n_sel
        top = fr.where(sel_hold).apply(
            lambda row: row.nlargest(min(kth, row.notna().sum())).index, axis=1)
        # 逐日算 Top 组的平均收益与上涨比例
        top_ret, top_win = [], []
        for d in fr.index:
            cols = top.loc[d]
            cols = cols[pd.notna(cols)]
            if len(cols) < 5:
                top_ret.append(np.nan); top_win.append(np.nan); continue
            v = fr.loc[d, cols]
            top_ret.append(v.mean())
            top_win.append((v > 0).mean())
        # 基准：同期全市场
        bench_win = (fr.stack().groupby(level=0).apply(lambda s: (s > 0).mean()))
        rows.append({
            "持有期_交易日": k,
            "Top组上涨率": np.nanmean(top_win),
            "全市场上涨率": bench_win.mean(),
            "超额命中率": np.nanmean(top_win) - bench_win.mean(),
            "Top组平均收益": np.nanmean(top_ret),
            "有效天数": int(np.isfinite(top_ret).sum()),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# 二、赚钱能力：净值、超额、回撤
# ════════════════════════════════════════════════════════════════
def equity_metrics(
    port_ret: pd.Series,
    bench_ret: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """从日频收益序列算出赚钱能力指标。"""
    r = port_ret.dropna()
    if len(r) < 30:
        return {"年化收益": np.nan}
    cum = (1 + r).cumprod()
    n = len(r)
    years = n / periods_per_year
    ann = float(cum.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(r.std() * np.sqrt(periods_per_year))

    dd = cum / cum.cummax() - 1
    mdd = float(dd.min())
    calmar = ann / abs(mdd) if mdd else np.nan
    sharpe = ann / vol if vol else np.nan

    out = {
        "年化收益": ann,
        "年化波动": vol,
        "夏普": sharpe,
        "最大回撤": mdd,
        "Calmar": calmar,
        "总收益": float(cum.iloc[-1] - 1),
        "年化天数": n,
    }
    if bench_ret is not None:
        b = bench_ret.reindex(r.index).dropna()
        if len(b) > 30:
            bcum = (1 + b).cumprod()
            bann = float(bcum.iloc[-1] ** (1 / (len(b) / periods_per_year)) - 1)
            ex = r - b
            exann = float((1 + ex).cumprod().iloc[-1]
                          ** (1 / (len(ex) / periods_per_year)) - 1)
            out["基准年化"] = bann
            out["年化超额"] = exann
            out["信息比率"] = (exann / (ex.std() * np.sqrt(periods_per_year))
                             if ex.std() > 0 else np.nan)
    return out


def period_winrate(port_ret: pd.Series, hold_days: int = 60) -> dict:
    """按持有期算胜率与盈亏比——用户最关心的「买了能不能赚」。"""
    r = port_ret.dropna()
    if len(r) < hold_days * 3:
        return {}
    # 用重叠窗口近似持有期收益
    hold = r.rolling(hold_days).sum().dropna()
    win = (hold > 0).mean()
    gains = hold[hold > 0].mean()
    losses = -hold[hold < 0].mean()
    return {
        f"持有{hold_days}日_胜率": float(win),
        f"持有{hold_days}日_盈亏比": float(gains / losses) if losses > 0 else np.nan,
        f"持有{hold_days}日_期望": float(hold.mean()),
    }


# ════════════════════════════════════════════════════════════════
# 三、综合评估报告
# ════════════════════════════════════════════════════════════════
def full_report(
    name: str,
    holdings: pd.Series,
    fwd_ret: pd.DataFrame,
    port_ret: pd.Series,
    bench_ret: pd.Series,
    periods: tuple = (1, 5, 20, 60),
    top_frac: float = 0.2,
) -> dict:
    """一次性输出「准不准 + 赚钱能力」全部指标。"""
    hr = hit_rate_analysis(holdings, fwd_ret, top_frac, periods)
    eq = equity_metrics(port_ret, bench_ret)
    for h in (20, 60, 120):
        eq.update(period_winrate(port_ret, h))
    return {"名称": name, "命中率明细": hr, "收益指标": eq}


def print_report(r: dict) -> None:
    """按用户能直接理解的方式打印。"""
    print("\n" + "=" * 76)
    print(f"【{r['名称']}】")
    print("=" * 76)

    hr = r.get("命中率明细")
    if hr is not None and len(hr):
        print("\n■ 准不准（命中率）")
        print(f"  {'持有期':>10} {'Top组上涨率':>13} {'全市场上涨率':>13} "
              f"{'超额命中率':>12}")
        print("  " + "-" * 54)
        for _, row in hr.iterrows():
            k = int(row["持有期_交易日"])
            label = f"{k}日" + (f"({k/21:.0f}月)" if k >= 21 else "")
            print(f"  {label:>10} {row['Top组上涨率']*100:>12.1f}% "
                  f"{row['全市场上涨率']*100:>12.1f}% "
                  f"{row['超额命中率']*100:>+11.1f}pp")
        best = hr.loc[hr["超额命中率"].idxmax()]
        print(f"  → 最优持有期 {int(best['持有期_交易日'])} 日，"
              f"超额命中率 {best['超额命中率']*100:+.1f}pp")

    eq = r.get("收益指标", {})
    print("\n■ 赚钱能力")
    if "年化收益" in eq and np.isfinite(eq.get("年化收益", np.nan)):
        print(f"  年化收益{eq['年化收益']*100:>8.2f}%   "
              f"年化波动 {eq['年化波动']*100:.2f}%   "
              f"夏普 {eq.get('夏普', float('nan')):.2f}")
        if "年化超额" in eq:
            print(f"  基准年化    {eq['基准年化']*100:>7.2f}%   "
                  f"年化超额 {eq['年化超额']*100:>+7.2f}pp   "
                  f"信息比率 {eq.get('信息比率', float('nan')):.2f}")
        print(f"  最大回撤    {eq['最大回撤']*100:>7.2f}%   "
              f"Calmar {eq.get('Calmar', float('nan')):.2f}")
        print()
        for k in (20, 60, 120):
            wr = eq.get(f"持有{k}日_胜率")
            pf = eq.get(f"持有{k}日_盈亏比")
            if wr is not None:
                print(f"  持有{k:>3}日  胜率 {wr*100:>5.1f}%   "
                      f"盈亏比 {pf:.2f}")


def compare_reports(reports: list[dict]) -> None:
    """多方案横向对比，突出「谁更准、谁更赚」。"""
    if not reports:
        return
    print("\n" + "=" * 96)
    print("方案横向对比")
    print("=" * 96)
    print(f"{'方案':<24}{'超额命中率(20日)':>16}{'年化超额':>11}"
          f"{'最大回撤':>11}{'夏普':>8}")
    print("-" * 96)
    for r in reports:
        hr = r.get("命中率明细")
        eq = r.get("收益指标", {})
        edge = np.nan
        if hr is not None and len(hr):
            m = hr[hr["持有期_交易日"] == 20]
            if len(m):
                edge = float(m.iloc[0]["超额命中率"])
        print(f"{r['名称']:<24}{edge*100:>15.1f}%"
              f"{eq.get('年化超额', float('nan'))*100:>10.2f}pp"
              f"{eq.get('最大回撤', float('nan'))*100:>10.2f}%"
              f"{eq.get('夏普', float('nan')):>8.2f}")
