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

from ..config import CostModel, ResearchConfig, SCALING_TRADING_DAYS


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
        # ⚠️ 标签必须用**实际参与计算的两个分位**，不能用 self.quantiles。
        #   bins 模式下标签可能跳号（实测 [1, 3]），此时 self.quantiles=3 是
        #   「成本乘数」，而 spread 实为 Q3 − Q1；早期版本打印「Q2 − Q1」
        #   与实际计算不符，报告会误导读者（2026-10-06 review 抓出）。
        _hi = self.quantile_returns.index.max()
        _lo = self.quantile_returns.index.min()
        L.append(f"多空组合（Q{_hi} − Q{_lo}，共 {len(self.quantile_returns)} 组）"
                 f"年化毛收益: {self.gross_spread*100:+.2f}%")
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
    bins: int | None = None,
) -> TearSheetResult:
    """跑一次 tear sheet，返回结构化结果。

    demeaned   : False 保留因子绝对值（默认）。True 会按日去均值，只留横截面排序。
    zero_aware : 对有明确正负分界的因子（如 hhhl_score）可设 True；
                 动量/反转类必须 False，否则分位编号错乱。
    bins       : **等宽值域分箱**的箱数。用于取值种类少于分组数的**离散**因子。

    ⚠️ 离散因子必须走 bins，不能靠 zero_aware（2026-10-06 实测）。
       ``hhhl20`` 只有 {0, 1, 2} 三种取值：
         · ``quantiles=5`` → ``pd.qcut(x, 5)`` 在 3 种取值上产生大量 NaN，
           binning 阶段丢弃 90%，触发 MaxLossExceededError（100% exceeded）。
         · ``zero_aware=True``更糟：它按「正/负」各切一半，而 hhhl **没有负值**，
           负半区为空 → 负分位全是 NaN，整段仍失败。
       正确做法是 ``bins=n_distinct``，按**值域**等宽切分。
    """
    periods = list(cfg.periods)
    quantiles = cfg.quantiles

    if groupby is not None:
        assets = factor.index.get_level_values("asset")
        gb = groupby.reindex(assets)
        groupby = pd.Series(gb.to_numpy(), index=factor.index, name="group")

    # ⚠️ alphalens 要求 quantiles 与 bins **二选一**，同时传会抛
    #   ValueError: Either quantiles or bins should be provided（实测）。
    #   早先版本两个都传，注释写了「二选一」但代码没做——注释与实现脱节。
    if bins is not None:
        quantiles_arg: int | None = None
    else:
        quantiles_arg = quantiles

    clean = utils.get_clean_factor_and_forward_returns(
        factor=factor,
        prices=prices,
        quantiles=quantiles_arg,
        periods=periods,
        groupby=groupby,
        max_loss=cfg.max_loss,
        zero_aware=zero_aware,
        bins=bins,
    )
    # bins 模式下实际箱数可能小于请求值（某箱无样本），用实际值做后续口径
    #
    # 🔴 2026-10-06 review 抓出：必须用**真实标签集合**，不能用 nunique()。
    #   alphalens 的 bins 标签**不保证是1..n 连续**。
    #   实测（因子只有 {0,1} 两种取值 + bins=3）：
    #     出现的标签   = [1, 3]← Q2 为空，标签跳号
    #     nunique()    = 2                ← 当成「2 组」
    #     max(标签)     = 3
    #   三处连锁出错：
    #     · tr 用 range(1, nunique+1) = {Q1, Q2} → **Q3 的换手根本没被统计**
    #     · 成本乘数用 2 而非 3 → 净收益偏乐观
    #     · summary 打印「Q2 − Q1」而 spread 实为 Q3 − Q1 → 标签与计算不符
    labels = sorted(int(x) for x in clean["factor_quantile"].unique()) \
        if len(clean) else []
    if bins is not None:
        if len(labels) < 2:
            raise ValueError(
                f"等宽分箱后有效组数 {len(labels)} < 2，无法计算多空组合。"
                f"因子取值可能过于集中，请检查 bins={bins} 是否合理"
            )
        # 成本乘数用**最大标签**（= 组数），换手率覆盖**全部真实标签**
        quantiles = max(labels)

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
        # ⚠️ 覆盖**全部真实标签**（可能是 [1, 3] 而非 [1, 2]），
        #    漏掉最高分位会让 turnover_mean 偏小 → 净收益偏乐观。
        {q: performance.quantile_turnover(clean["factor_quantile"], q)
         for q in (labels or range(1, quantiles + 1))}
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
    #   252 = ① 倍数换算层常数（config.SCALING_TRADING_DAYS），不是
    #   「A 股年均交易日」——时长换算层是 243（YEAR_TRADING_DAYS），两层勿串用。
    periods_per_year = SCALING_TRADING_DAYS / periods[0]
    gross = spread * periods_per_year
    net = gross - cost.round_trip * turnover_mean * quantiles * periods_per_year

    return TearSheetResult(
        factor_name=factor_name,
        periods=tuple(periods),
        quantiles=quantiles,
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
    bins: int | None = None,
) -> pd.DataFrame:
    """子区间稳定性检验（AGENTS.md 强制：全样本有效 ≠ 稳定）。

    ⚠️ 失败必须**显式报告**，不能静默填NaN（ENGINEERING.md 第四节）。
       早先版本用 ``except Exception: pass`` 把真实报错吞成一行NaN，
       输出看起来「只是这个区间没数据」，实际是alphalens 抛异常。
    """
    rows = []
    for label, (s, e) in cfg.sub_periods.items():
        rng = pd.date_range(s, e)
        f = factor[factor.index.get_level_values("date").isin(rng)]
        p = prices.loc[prices.index.intersection(rng)]
        n_assets = int(f.index.get_level_values("asset").nunique()) if len(f) else 0
        base = {"区间": label, "标的数": n_assets,
                # ⚠️ 日期数填**实际交易日数**，不是 0（2026-10-06 review 抓出）。
                #   原先固定 0，导致「有数据但计算失败」与「完全无数据」
                #   两行长得一模一样，读者无法区分。
                "日期数": int(f.index.get_level_values("date").nunique())
                if len(f) else 0,
                "IC均值": np.nan, "ICIR": np.nan, "t值": np.nan,
                "多空年化": np.nan, "多空净": np.nan}
        if f.empty or p.empty:
            rows.append({**base, "说明": "无数据"})
            continue
        if n_assets < cfg.min_assets_subperiod:
            # 样本太少时 alphalens 分位数不可靠，属「主动跳过」不是「计算失败」
            rows.append({**base, "说明": f"标的数 {n_assets} < {cfg.min_assets_subperiod}，主动跳过"})
            continue
        try:
            r = run_tear_sheet(f, p, cfg, cost, factor_name,
                               demeaned=demeaned, zero_aware=zero_aware, bins=bins)
        except Exception as e:  # noqa: BLE001
            # 报出来，但不中断——其他区间仍可能有结论。
            # ⚠️ 只截**中间**，保留头尾：异常消息的根因常在末尾
            #   （如 "cannot reindex on an axis with duplicate labels"），
            #   粗暴截前 120 字会把根因切掉（2026-10-06 review 抓出）。
            msg = f"{type(e).__name__}: {e}"
            if len(msg) > 200:
                msg = msg[:100] + f" …（省略 {len(msg) - 200} 字）… " + msg[-100:]
            rows.append({**base, "说明": f"失败 {msg}"})
            continue
        ic1 = r.ic_by_period.loc[1] if 1 in r.ic_by_period.index else None
        rows.append({
            "区间": label, "标的数": n_assets, "日期数": r.n_dates,
            "IC均值": float(ic1["mean"]) if ic1 is not None else np.nan,
            "ICIR": float(ic1["ir"]) if ic1 is not None else np.nan,
            "t值": float(ic1["tstat"]) if ic1 is not None else np.nan,
            "多空年化": r.gross_spread,
            "多空净": r.net_spread_after_cost,
            "说明": "",
        })
    return pd.DataFrame(rows)
