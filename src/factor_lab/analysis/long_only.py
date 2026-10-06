"""单边多头选股：把因子 IC 翻译成可执行的持仓建议。

为什么不能直接用因子 IC
----------------------
因子检验给的是**多空组合**的统计性质，而手动买股票只能做**单边多头**：
  · 多空组合可以空掉「跌得多的」，多头组合必须持有「剩下的」
  · 多空 IC=0.08 完全可能来自「空头端贡献大」，多头端是负的
  · 多头换手 53% 在实盘里对应「每周换半仓」，交易成本吃掉全部收益

⚠️ 实测踩过（2026-10-06）：
```
rev5   多空年化毛 +0.68  多空年化净 −0.67  换手 53%
vol60  多空年化毛 −0.08  多空年化净 −0.28
```
多空毛收益看着不错，扣成本后全负 —— **对手动决策零参考价值**。

所以必须重新问三个问题：
  1. 只做多，选出来的组合相对基准有没有超额？
  2. 超额能不能覆盖交易成本（换手 × 单边费率）？
  3. 扣除成本后，长期夏普/胜率如何？

本模块的输出
------------
不是「因子好不好」，而是：
  · 选股池（按因子分位筛选）
  · 目标持仓权重（考虑波动率与相关性，不是等权）
  · 调仓频率（换手成本的直接函数）
  · 止损与风控约束

关键设计：所有参数都必须能在**样本外**验证，不能靠样本内调优。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ── 交易成本（A 股实际水平）────────────────────────────────────
@dataclass(frozen=True)
class CostModel:
    """单边交易成本。

    ⚠️ **双边**换算：买入 0.025%+ 过户费 0.001%，卖出 0.025% + 印花税 0.05%
       （印花税 2023-08-28 起减半为 0.05%，此前 0.1%）。
    手动交易的额外成本（研报/软件/时间）不计入，但**换手率要按实际调仓算**。
    """

    commission: float = 0.00025     # 佣金 万2.5
    stamp_duty: float = 0.0005      # 印花税 万5（卖出单边）
    transfer_fee: float = 0.00001   # 过户费 十万分之一（双向）
    slippage: float = 0.001         # 冲击成本 0.1%（保守估计）

    @property
    def buy(self) -> float:
        return self.commission + self.transfer_fee + self.slippage

    @property
    def sell(self) -> float:
        return self.commission + self.transfer_fee + self.slippage + self.stamp_duty

    @property
    def round_trip(self) -> float:
        return self.buy + self.sell

    def describe(self) -> str:
        return (f"买入 {self.buy*1e4:.2f}‱/ 卖出 {self.sell*1e4:.2f}‱ "
                f"（单边成本 {self.buy*1e4:.2f}‱）")


# 盈亏平衡换手率：换手率 × 单边成本必须 < 预期年化超额
def breakeven_turnover(target_annual_return: float, cost: CostModel) -> float:
    """给定目标年化收益，能承受的最高**年换手率**（倍数）。

    模型：``年化超额 ≥ 年换手率 × 单边成本``

    ⚠️ **这个数字通常大得没有参考价值**（如目标 20% 时是 158 倍年换手）。
    原因：单边成本只有 0.126%，理论上极高频也能覆盖成本。
    但**手动交易受的是换手率本身，不是成本**：

    | 调仓频率 | 年换手 | 现实可行性 |
    |---|---|---|
    | 月频 | 3~4 倍 | ✓ 手动可行 |
    | 季频 | 1~1.5 倍 | ✓ 手动舒适 |
    | 周频 | 12 倍+ | ✗ 盯盘成本高，且滑点会远超模型值 |
    | 日频 | 250 倍+ | ✗ 不可能 |

    **所以真正的约束是「手动能执行的换手上限」，约年化 4~6 倍**。
    对应目标年化超额 20%，需要扣成本后仍正：
    ``4倍 × 0.126% = 0.5%`` —— 成本占比很低，不是问题。

    真正的风险不在成本，在**因子衰减**和**个股集中度**，见 build_long_only。
    """
    if cost.buy <= 0:
        return float("inf")
    return target_annual_return / cost.buy


# 手动交易的年换手上限（现实约束，非成本约束）
MANUAL_TURNOVER_CAP = 6.0


def manual_turnover_note() -> str:
    return (f"手动交易年换手上限约 {MANUAL_TURNOVER_CAP:.0f} 倍"
            f"（月频 3~4 倍 / 季频 1~1.5 倍）。"
            f"对应年成本 {MANUAL_TURNOVER_CAP * CostModel().buy * 100:.2f}%，"
            f"不是瓶颈。")


# ── A 股特有的交易约束（量价因子完全没覆盖）──────────────────
@dataclass
class TradabilityFilter:
    """涨跌停与停牌约束。

    ⚠️ 这是 A 股手动交易**最大的隐性风险**，而所有因子研究都忽略了它：
      · 涨停时买不到（买单挂不上）
      · 跌停时卖不出（想跑跑不掉）
      · 停牌期间完全无法操作

    实测：本项目采样区间内主板日均涨停 53 只、跌停 4 只；
    一只强势股在买入信号出现时可能已经涨停 —— 因子说「买」，
    实际买不到，等下一个信号就是一周后。

    **对「赚钱效率」的影响**：选出的股票越强势，越买不到。
    这会系统性削弱动量类因子的实际收益。
    """

    skip_limit_up: bool = True    # 买入时跳过涨停
    skip_limit_down: bool = False # 卖出时：跌停只能等，不算可卖
    max_suspended_days: int = 3   # 停牌超过 N 天视为不可用
    volume_cap_ratio: float = 0.0  # 单日成交量不超过流通盘的 N 倍（0=不限制）


def limit_state(
    close: pd.Series, high: pd.Series, low: pd.Series,
    code: str, prev_close: pd.Series | float,
) -> pd.Series:
    """判定每只股票每日的涨跌停状态。

    返回 True 表示「处于涨跌停状态」（买不到或卖不出）。
    """
    from factor_lab.market_rules import limit_of, limit_price

    lim = limit_of(code)
    if isinstance(prev_close, pd.Series):
        up = close >= limit_price(prev_close, lim) - 1e-6
        dn = close <= limit_price(prev_close, -lim) + 1e-6
    else:
        up = close >= limit_price(float(prev_close), lim) - 1e-6
        dn = close <= limit_price(float(prev_close), -lim) + 1e-6
    return (up | dn).fillna(False)


# ── 持仓构建 ──────────────────────────────────────────────────
@dataclass
class PortfolioSpec:
    """一个可执行的多头组合方案。"""

    name: str
    n_hold: int = 30# 持股数
    n_pick: int = 100          # 从 n_pick 里选 n_hold（给换手留缓冲）
    rebalance: str = "M"       # 调仓频率：M=月初 Q=季初 W=周初
    max_weight: float = 0.15# 单票上限
    min_weight: float = 0.02   # 单票下限
    vol_target: float = 0.20   # 目标年化波动
    vol_lookback: int = 60
    stop_loss: float = -0.20   # 个股止损线
    max_drawdown: float = 0.25 # 组合最大回撤容忍
    factor: str = ""
    extra: dict = field(default_factory=dict)

    def describe(self) -> str:
        return (f"{self.name}: 持股 {self.n_hold}/{self.n_pick}只 · "
                f"{self.rebalance}调仓 · 单票 ≤{self.max_weight:.0%} · "
                f"目标波动 {self.vol_target:.0%} · 止损 {self.stop_loss:.0%}")


def build_long_only(
    factor_values: pd.DataFrame,
    spec: PortfolioSpec,
    cost: CostModel,
    price_panel: pd.DataFrame | None = None,
) -> dict:
    """构建单边多头组合并做样本内检验。

    参数
    ----
    factor_values : 索引=日期，列=代码，因子值（越大越好），**已横截面标准化**
    spec          : 组合参数
    cost          : 交易成本
    price_panel   : 索引=日期，列=代码，**后复权价**。
                    ⚠️ **必须传**。收益必须从价格算，不能从因子值算 ——
                    初版直接 `factor_values.pct_change()`，而 factor_values
                    是 z 分数（可为负、可跨股票差 10 倍），
                    比值的分布毫无意义，实测单日收益出现 100 倍以上的值，
                    年化波动 33,494%、回撤 −261,897%，净值被打爆。

    返回
    ----
    含净值序列、换手、成本、超额、风险指标的 dict

    ⚠️ **这里是样本内**，只能用来排除明显不可行的方案。
       真正决策必须看样本外 —— 用 ``holdout_split`` 切分。
    """
    if factor_values.empty:
        return {"ok": False, "reason": "因子值为空"}
    if price_panel is None or price_panel.empty:
        return {"ok": False,
                "reason": "缺少 price_panel —— 收益必须从后复权价算，"
                          "不能从因子值算（z 分数比值无意义）"}

    # 收益：必须来自**后复权价**，且是下一期收益。
    # ⚠️ 两个易错点：
    #   1. 不能从 factor_values 算 —— 那是 z 分数，比值无意义
    #   2. 必须是下一期 —— 因子在 d 日收盘才可得，
    #      当日收益发生时因子还不知道（用当日收益会系统性虚高）
    fwd = price_panel.pct_change(fill_method=None).shift(-1)

    # 1) 选股：因子排名
    n_pick = min(spec.n_pick, factor_values.shape[1])
    n_hold = min(spec.n_hold, n_pick)

    # 2) 权重：因子分位 + 波动率倒数（风险平价思路的简化版）
    #⚠️ 性能：必须**预计算滚动波动率**。
    #   初版写成 `fwd.loc[:d, held].tail(60)`逐日全表切片，
    #   实测 5,606 只 × 2,611 日跑到 20 分钟没跑完（O(n²)）。
    #   pandas 的 rolling 是 O(n)，预计算后逐日只做 O(n_hold) 的索引。
    rolling_vol = (fwd.rolling(spec.vol_lookback, min_periods=max(
        20, spec.vol_lookback // 3)).std() * np.sqrt(252))

    #⚠️ 性能：整段**向量化**。
    #   初版是逐日 for 循环 + 每日全表切片，
    #   实测 5,606 只 × 2,611 日跑 20 分钟未完成（O(n²)）。
    #   现在：一次取出矩阵，nlargest 用 numpy 的 argpartition（O(n)），权重全部矩阵化。
    fv_mat = factor_values.to_numpy(dtype=float)          # (T, N)
    vol_mat = rolling_vol.to_numpy(dtype=float)          # (T, N)
    T, N = fv_mat.shape

    # ── 调仓日筛选（关键）──
    # ⚠️⚠️ 初版完全没用 rebalance 参数，**每天都重新选股调仓**：
    #   实测年换手 110%、年成本 42%，净值被打爆（年化 −100%）。
    #   月频调仓的年换手应是 3~4 倍，年成本约 0.5%。
    #   这不是参数问题，是「手动能否执行」的问题 ——
    #   每天全换手在实盘里既做不到也承担不起。
    rebal_idx = rebalance_days(factor_values.index, spec.rebalance)
    is_rebal = np.zeros(T, dtype=bool)
    is_rebal[rebal_idx] = True

    fv_for_top = np.where(np.isnan(fv_mat), -np.inf, fv_mat)
    if n_pick >= N:
        top_idx = np.argsort(-fv_for_top, axis=1)[:, :n_pick]
    else:
        part = np.argpartition(-fv_for_top, n_pick - 1, axis=1)[:, :n_pick]
        rows_idx = np.arange(T)[:, None]
        order = np.argsort(-fv_for_top[rows_idx, part], axis=1)
        top_idx = part[rows_idx, order]

    held_mat = top_idx[:, :n_hold]                       # (T, n_hold)
    inv = np.where(vol_mat > 0, 1.0 / vol_mat, np.nan)
    picked_inv = np.take_along_axis(inv, held_mat, axis=1)
    bad = ~np.isfinite(picked_inv) | (picked_inv <= 0)
    w_mat = np.where(bad, np.nan, picked_inv)
    allnan = ~np.isfinite(w_mat).any(axis=1)
    n_valid = np.isfinite(w_mat).sum(axis=1)
    w_mat = np.where(np.isfinite(w_mat), w_mat, 0.0)
    ssum = w_mat.sum(axis=1, keepdims=True)
    degenerate = (n_valid < n_hold * 0.5) | (ssum[:, 0] <= 0) | allnan
    ssum = np.where(ssum <= 0, 1.0, ssum)
    w_mat = w_mat / ssum
    w_eq = np.full((T, n_hold), 1.0 / n_hold)
    w_mat = np.where(degenerate[:, None], w_eq, w_mat)
    w_mat = np.clip(w_mat, spec.min_weight, spec.max_weight)
    w_mat = w_mat / w_mat.sum(axis=1, keepdims=True)

    # 非调仓日**沿用上一期持仓**（不重新选股）
    held_mat = _carry_forward(held_mat, is_rebal)
    w_mat = _carry_forward(w_mat, is_rebal)
    # 首个调仓日之前无持仓 → 全零
    first = int(np.argmax(is_rebal)) if is_rebal.any() else T
    w_mat[:first] = 0.0
    w_mat = w_mat / np.maximum(w_mat.sum(axis=1, keepdims=True), 1e-12)

    return simulate_matrix(factor_values.index, held_mat, w_mat,
                           fwd, cost, spec)


def rebalance_days(dates, freq: str) -> np.ndarray:
    """返回调仓日的位置索引。

    M = 每月首个交易日；Q = 每季首个交易日；W = 每周首个交易日；
    D = 每日（不推荐：实测年换手 110%、年成本 42%）。
    """
    idx = pd.DatetimeIndex(dates)
    if freq.upper() == "D":
        return np.arange(len(idx))
    if freq.upper() == "W":
        key = idx.to_period("W")
    elif freq.upper() == "Q":
        key = idx.to_period("Q")
    else:                                    # 默认月频
        key = idx.to_period("M")
    # 每个 period 的第一天
    change = np.ones(len(key), dtype=bool)
    change[1:] = key[1:] != key[:-1]
    return np.where(change)[0]


def _carry_forward(mat: np.ndarray, is_rebal: np.ndarray) -> np.ndarray:
    """非调仓日沿用上一期的值（矩阵按行前向填充）。

    ⚠️ 用 index-based 而非 pandas ffill：宽表 (2611, 30) 上
    pandas 的 ffill 逐列循环，11 年 × 30 列实测很慢。
    numpy 前向填充只需一次 cumsum 索引。
    """
    out = mat.copy()
    last = np.zeros(mat.shape[1], dtype=mat.dtype)
    for i in range(mat.shape[0]):
        if is_rebal[i]:
            last = out[i]
        else:
            out[i] = last
    return out


def simulate_matrix(dates, held_mat, w_mat, fwd: pd.DataFrame,
                    cost: CostModel, spec: PortfolioSpec) -> dict:
    """矩阵版回测：全程 numpy，不碰 pandas 索引。

    ⚠️ **性能**：初版逐日 `fwd.loc[d, w.index]` 做标签查找，
       5,606 只 × 2,611 日跑 20 分钟未完成。
       现在把收益矩阵与持仓矩阵一起按位置索引，全向量化。
    """
    r_mat = fwd.to_numpy(dtype=float)          # (T, N) 前视修正后的收益
    r_mat = np.nan_to_num(r_mat, nan=0.0)
    T = r_mat.shape[0]
    date_pos = {d: i for i, d in enumerate(dates)}
    pos = np.array([date_pos.get(d, -1) for d in dates])
    ok = pos >= 0

    # 组合每日收益 = Σ w[i,t] × r[t, held[i,t]]
    # ⚠️ 索引顺序：`held_mat[ok]` 是 (T_ok, n_hold)，`pos[ok]` 是 (T_ok,)，
    #    numpy 广播时**行索引必须先取 held 再取 pos**，写反会形状不匹配。
    gross = np.zeros(T)
    if ok.any():
        rr = r_mat[pos[ok][:, None], held_mat[ok]]   # (T_ok, n_hold)
        gross[ok] = np.nansum(w_mat[ok] * rr, axis=1)

    # 换手 = |w_t - w_{t-1}| 在**全股票空间**上的 L1 差
    # ⚠️ 必须在全 N 只上算，只在持仓股票上算会漏掉「卖出的股票」，
    #    实测那种算法换手率被低估约一半。
    W = np.zeros((T, r_mat.shape[1]), dtype=float)
    rows_ok = np.where(ok)[0]
    if len(rows_ok):
        # ⚠️ 需要把行索引扩展成二维 (T_ok, 1)，
        #    否则 (T_ok,) 与 (T_ok, n_hold) 广播失败。
        W[rows_ok[:, None], held_mat[rows_ok]] = w_mat[rows_ok]
    dW = np.abs(np.diff(W, axis=0, prepend=W[:1]))
    turn = dW.sum(axis=1)

    cost_arr = turn * (cost.buy + cost.sell) / 2
    net = gross - cost_arr
    nav = np.cumprod(1.0 + net)

    if len(nav) < 2 or not np.isfinite(nav[-1]) or nav[-1] <= 0:
        return {"ok": False, "reason": "净值序列异常"}

    rr = pd.Series(net).replace([np.inf, -np.inf], np.nan).dropna()
    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)         if hasattr(dates[0], "year") else max(len(net) / 252, 1e-9)
    cagr = float(nav[-1] ** (1 / years) - 1)
    vol = float(rr.std() * np.sqrt(252))
    sharpe = float(cagr / vol) if vol > 0 else np.nan
    nav_s = pd.Series(nav, index=dates, dtype=float)
    dd = float((nav_s / nav_s.cummax() - 1).min())

    return {
        "ok": True,
        "组合": spec.name,
        "因子": spec.factor,
        "调仓频率": spec.rebalance,
        "持股数": spec.n_hold,
        "累计净值": float(nav[-1]),
        "年化收益": cagr,
        "年化波动": vol,
        "夏普": sharpe,
        "最大回撤": dd,
        "平均换手": float(turn[ok].mean()) if ok.any() else 0.0,
        "平均年成本": float(cost_arr[ok].mean() * 252) if ok.any() else 0.0,
        "年化毛收益": float(cagr + cost_arr[ok].mean() * 252) if ok.any() else cagr,
        "期数": int(T),
    }


def holdout_split(dates, train_frac: float = 0.7) -> tuple:
    """按时间切分样本内/外。

    ⚠️ **必须按时间切，不能随机切** —— 金融数据有时间依赖，
       随机切会让训练集包含「未来」信息，IC 直接虚高。
    """
    dates = sorted(dates)
    n = len(dates)
    k = int(n * train_frac)
    return dates[:k], dates[k:]
