"""alphalens 适配层。

复用 alphalens-reloaded 的 IC / 分组 / 换手率计算，本模块只做：
  1. 把 factor_lab 的数据转成 alphalens 要求的格式
  2. 调用 alphalens 并把结果整理成便于断言与汇报的结构
不做任何统计公式的重新实现 —— 那是 alphalens 的职责。

════════════════════════════════════════════════════════════════════
alphalens-reloaded 0.4.6 API 实测记录（2026-10-03，勿凭记忆改）
════════════════════════════════════════════════════════════════════
输入约定
  factor : MultiIndex Series，必须是 (date=level0, asset=level1)
          ⚠️ 传反会在 compute_forward_returns 报
             "'Index' object has no attribute 'tz'"（level0 成了字符串 Index）
  prices : DataFrame(index=date, columns=asset)

返回类型
  utils.get_clean_factor_and_forward_returns(...) -> DataFrame
      列 = ['1D','5D','20D', ..., 'factor', 'factor_quantile']（列名为字符串）
  performance.factor_information_coefficient(clean) -> DataFrame
      index=date，列=['1D','5D','20D']，值是【每日的 IC】，需自行聚合均值/ICIR/t
  performance.mean_return_by_quantile(clean, demeaned=) -> (DataFrame, std_err)
      ⚠️ 返回【元组】需解包；index=factor_quantile(1..Q)，列=period 字符串
  performance.quantile_turnover(clean['factor_quantile'], q) -> Series
  performance.compute_mean_returns_spread(mean_returns, upper, lower)
      ⚠️ 要求 mean_returns.index 含 'factor_quantile' MultiIndex
      → 本模块直接用 qr.loc[Q] - qr.loc[1]，更可控（demeaned/zero_aware 变体会错乱）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphalens import performance, utils

from ..config import CostModel, ResearchConfig


def _col(period: int) -> str:
    """alphalens 的 period 列名是字符串：1 -> '1D'。"""
    return f"{period}D"


@dataclass
class TearSheetResult:
    """tear sheet 的结构化摘要。"""

    factor_name: str
    periods: tuple
    quantiles: int
    ic_by_period: pd.DataFrame          # index=period, 列=[mean,std,ir,tstat,count]
    quantile_returns: pd.DataFrame      # index=factor_quantile, 列=period
    gross_spread: float                # 多空年化（毛）
    net_spread_after_cost: float       # 多空年化（扣成本）
    cost_round_trip: float
    turnover_mean: float
    n_assets: int
    n_dates: int
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        L = [f"因子: {self.factor_name}",
             f"样本: {self.n_assets:,} 只 × {self.n_dates:,} 日   分组: {self.quantiles}",
             "-" * 64,
             "信息系数 Spearman IC（★ ICIR≥0.3, · ≥0.1）:"]
        for p in self.ic_by_period.index:
            r = self.ic_by_period.loc[p]
            a = abs(r["ir"])
            star = "★" if a >= 0.3 else ("·" if a >= 0.1 else " ")
            L.append(f"  {star} {int(p):>3}日  IC={r['mean']:+.4f}  ICIR={r['ir']:+.3f}  "
                     f"t={r['tstat']:+6.2f}  n={int(r['count']):,}")
        L.append("-" * 64)
        L.append(f"各分位累计收益（{self.periods[0]} 日）:")
        for q, v in self.quantile_returns[_col(self.periods[0])].items():
            L.append(f"    Q{q}  {v*100:+7.3f}%")
        L.append("-" * 64)
        L.append(f"多空组合（Q{self.quantiles} − Q1）年化毛收益: {self.gross_spread*100:+.2f}%")
        L.append(f"  扣成本 {self.cost_round_trip*100:.2f}% 往返 × 换手 "
                 f"{self.turnover_mean*100:.1f}%  →  净 {self.net_spread_after_cost*100:+.2f}%")
        for n in self.notes:
            L.append(f"  注: {n}")
        return "\n".join(L)


def run_tear_sheet(
    factor: pd.Series,
    prices: pd.DataFrame,
    cfg: ResearchConfig,
    cost: CostModel,
    factor_name: str = "factor",
    groupby: pd.Series | None = None,
    demeaned: bool = False,
    zero_aware: bool = False,
) -> TearSheetResult:
    """跑一次 tear sheet，返回结构化结果。

    demeaned   : False 保留因子绝对值（默认）。True 会按日去均值，只留横截面排序。
    zero_aware : 对有明确正负分界的因子（如 hhhl_score）可设 True；
                 动量/反转类必须 False，否则分位编号错乱。
    """
    periods = list(cfg.periods)

    if groupby is not None:
        assets = factor.index.get_level_values("asset")
        gb = groupby.reindex(assets)
        groupby = pd.Series(gb.to_numpy(), index=factor.index, name="group")

    clean = utils.get_clean_factor_and_forward_returns(
        factor=factor,
        prices=prices,
        quantiles=cfg.quantiles,
        periods=periods,
        groupby=groupby,
        max_loss=cfg.max_loss,
        zero_aware=zero_aware,
    )

    # ── IC：alphalens 只给每日值，均值/ICIR/t 需自行聚合 ──────────
    ic_daily = performance.factor_information_coefficient(clean)
    rows = []
    for p in periods:
        col = _col(p)
        if col not in ic_daily.columns:
            continue
        s = ic_daily[col].dropna()
        if s.empty:
            continue
        sd = s.std()
        mean = s.mean()
        rows.append({
            "period": p, "mean": mean, "std": sd,
            "ir": mean / sd if sd else np.nan,
            "tstat": mean / sd * np.sqrt(len(s)) if sd else np.nan,
            "count": len(s),
        })
    ic_by_period = pd.DataFrame(rows).set_index("period") if rows else pd.DataFrame()

    # ── 分组收益 & 换手率 ───────────────────────────────────────
    qr, _ = performance.mean_return_by_quantile(clean, demeaned=demeaned)
    keep = {_col(p) for p in periods}
    qr = qr[[c for c in qr.columns if c in keep]]
    tr = pd.DataFrame(
        {q: performance.quantile_turnover(clean["factor_quantile"], q)
         for q in range(1, cfg.quantiles + 1)}
    )
    turnover_mean = float(tr.mean().mean()) if not tr.empty else np.nan

    avail = sorted(qr.index.tolist())
    if len(avail) < 2:
        raise ValueError(f"有效分位数不足（{avail}），无法计算多空组合")
    p0 = _col(periods[0])
    spread = float(qr.loc[avail[-1], p0] - qr.loc[avail[0], p0])

    # ⚠️ 年化口径（2026-10-05 修正，勿改回）
    #   mean_return_by_quantile 返回的是【每期平均收益】。原代码写
    #   gross = spread * periods[0]，而 periods[0]=1，于是「年化毛收益」
    #   实际是【每日】差值 —— 数值被低估了约 252 倍，且打印标签写着「年化」，
    #   极易被误读成「扣成本后归零」这类错误结论。
    #   正确做法：先把每期差值折算成年化，再扣年化成本。
    #     年化毛   = 每期差值 × (252 / periods[0])
    #     年化成本 = 往返费率 × 平均单期换手 × 分组数 × (252 / periods[0])
    #   252 = A 股年交易日数近似值。
    periods_per_year = 252.0 / periods[0]
    gross = spread * periods_per_year
    net = gross - cost.round_trip * turnover_mean * cfg.quantiles * periods_per_year

    return TearSheetResult(
        factor_name=factor_name,
        periods=tuple(periods),
        quantiles=cfg.quantiles,
        ic_by_period=ic_by_period,
        quantile_returns=qr,
        gross_spread=gross,
        net_spread_after_cost=net,
        cost_round_trip=cost.round_trip,
        turnover_mean=turnover_mean,
        n_assets=int(clean.index.get_level_values("asset").nunique()),
        n_dates=int(clean.index.get_level_values("date").nunique()),
        notes=[f"demeaned={demeaned}, zero_aware={zero_aware}"],
    )


def subperiod_ic(
    factor: pd.Series,
    prices: pd.DataFrame,
    cfg: ResearchConfig,
    cost: CostModel,
    factor_name: str,
    demeaned: bool = False,
    zero_aware: bool = False,
) -> pd.DataFrame:
    """子区间稳定性检验（AGENTS.md 强制：全样本有效 ≠ 稳定）。"""
    rows = []
    for label, (s, e) in cfg.sub_periods.items():
        rng = pd.date_range(s, e)
        f = factor[factor.index.get_level_values("date").isin(rng)]
        p = prices.loc[prices.index.intersection(rng)]
        n_assets = int(f.index.get_level_values("asset").nunique()) if len(f) else 0
        if f.empty or p.empty or n_assets < 30:
            rows.append({"区间": label, "标的数": n_assets, "日期数": 0,
                         "IC均值": np.nan, "ICIR": np.nan, "t值": np.nan,
                         "多空年化": np.nan, "多空净": np.nan})
            continue
        try:
            r = run_tear_sheet(f, p, cfg, cost, factor_name,
                               demeaned=demeaned, zero_aware=zero_aware)
            ic1 = r.ic_by_period.loc[1] if 1 in r.ic_by_period.index else None
            rows.append({
                "区间": label, "标的数": n_assets, "日期数": r.n_dates,
                "IC均值": float(ic1["mean"]) if ic1 is not None else np.nan,
                "ICIR": float(ic1["ir"]) if ic1 is not None else np.nan,
                "t值": float(ic1["tstat"]) if ic1 is not None else np.nan,
                "多空年化": r.gross_spread,
                "多空净": r.net_spread_after_cost,
            })
        except Exception:  # noqa: BLE001, S110
            rows.append({"区间": label, "标的数": n_assets, "日期数": 0,
                         "IC均值": np.nan, "ICIR": np.nan, "t值": np.nan,
                         "多空年化": np.nan, "多空净": np.nan})
    return pd.DataFrame(rows)
