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

from factor_lab.analysis.selection import rank_topk, select_with_buffer

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


# ── A 股特有的交易约束 ────────────────────────────────────────
# ⚠️ `TradabilityFilter` / `limit_state` 已拆到 `tradable.py`（2026-10-06）。
#    原因：本文件加注释后达 502 行，超过「≤500 行」硬性门禁。
#    **这两个符号无生产调用方，且涨跌停约束实际从未生效过** ——
#    `limit_state` 的 Series 分支必崩、缺数据口径还与 `limit_masks` 相反。
#    详见 `tradable.py` 的模块 docstring。


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
    # ⚠️⚠️ 波动率**绝不能用 fwd 算**（review 2026-10-06 抓出，前视1 天）：
    #   `fwd[t]` 是「t 收盘买、t+1 收盘卖」的收益，在 t 收盘时**还没发生**；
    #   而 `rolling()` 默认**含当前点**（实测 fwd=[1,2,3] 时
    #   第 2 行得std([1,2,3])）⇒ 逆波动率权重用到了未来一天的收益。
    #   这与本函数开头「因子在 d 日收盘才可得」的口径自相矛盾。
    #   ⇒ 改用 `past = fwd.shift(1)`：t 日的权重只由 t-1 及之前的收益决定。
    #   影响幅度不大（波动率估计差一天），但方向明确违反无前视原则。
    past = fwd.shift(1)
    #⚠️ 性能：必须**预计算滚动波动率**。
    #   初版写成 `fwd.loc[:d, held].tail(60)`逐日全表切片，
    #   实测 5,606 只 × 2,611 日跑到 20 分钟没跑完（O(n²)）。
    #   pandas 的 rolling 是 O(n)，预计算后逐日只做 O(n_hold) 的索引。
    rolling_vol = (past.rolling(spec.vol_lookback, min_periods=max(
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
    # ⚠️ **候选池 + 缓冲带**（2026-10-06 修复）
    #   初版是 `held_mat = top_idx[:, :n_hold]` —— top_idx 已降序排好，
    #   前 n_hold 个元素与 n_pick 毫无关系，**n_pick 是死参数**。
    #   实测 n_pick ∈ {20,60,200} 三组持仓逐位完全相同，
    #   参数扫描表「候选池大小」那一列数值一模一样 —— 那是假象。
    #   现在真正实现缓冲带语义：持仓股只要还在 top n_pick 内就不卖。
    top_idx = rank_topk(fv_for_top, n_pick)
    held_mat = select_with_buffer(top_idx, is_rebal, n_hold)

    inv = np.where(vol_mat > 0, 1.0 / vol_mat, np.nan)
    # ⚠️ **必须先把 -1 哨兵屏蔽掉**，不能用 clip(下界, 0) 代替：
    #   `np.take_along_axis` 用的是**负索引语义** ——
    #   held_mat 里的 -1 会取到**最后一列**（某只真实存在的股票），
    #   而不是「无持仓」。
    #   实测踩过：把 0 当哨兵同样错，列索引 0 是真实股票（如 sh600000）。
    #   所以 sentinel 必须是 -1，且必须**在 take 之前**显式置 NaN。
    held_safe = np.where(held_mat < 0, 0, held_mat)
    picked_inv = np.take_along_axis(inv, held_safe, axis=1)
    bad = (held_mat < 0) | ~np.isfinite(picked_inv) | (picked_inv <= 0)
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
    # ⚠️ **clip 之后必须重新屏蔽哨兵**（review 2026-10-06 抓出，BLOCK）：
    #   上面刚把哨兵槽位置0，但 `np.clip(0, min_weight=0.02, ...)`
    #   会把它抬成 0.02 —— 权重凭空出现。
    #   实测：3槽持仓含1 个哨兵 → 哨兵权重 0 → clip 后 2% → 再归一化后 **6.25%**，
    #   而 `held_safe` 已把 -1 映射到列 0（**一只真实存在的股票**），
    #   下面的散射赋值 `W[rows, held_safe] = w_mat` 就把这 6.25% 打到了它身上。
    #   ⇒注释里「哨兵槽位权重为 0，不会凭空建仓」这句在 clip 之后是**错的**。
    w_mat = np.where(held_mat < 0, 0.0, w_mat)
    w_mat = w_mat / np.maximum(w_mat.sum(axis=1, keepdims=True), 1e-12)

    # 非调仓日**沿用上一期权重**。
    # ⚠️ held_mat **不要**再 carry_forward —— select_with_buffer 内部已经
    #   在非调仓日填了上一期的持仓，再做一次是重复（虽幂等，但会让人
    #   误以为缓冲带没生效）。
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


def _year_span(dates) -> float:
    """区间年数（自然日口径）。

    ⚠️ **整个项目只有一个年化定义**，策略与基准共用。
       初版策略用自然日、基准用 `len/252`，同一份收益算出两个年化，
       「超额」直接偏 0.5pp/年 —— 而超额是所有结论的判据。

    ⚠️ 退化保护：区间不足 1 天时 `days=0` 会除零，故用 `max(..., 1e-9)` 兜底。

    ⚠️⚠️ **短区间年化会严重放大，这是口径的固有性质，不是 bug**：
       几何年化 `nav**(1/years)-1` 在 years 很小时极不稳定。
       实测同一份 +8% 累计收益：
         20 个交易日(0.074 年) → 年化 **+183%**
         60 个交易日(0.227 年) → 年化 **+40%**
         1年(0.931 年)        → 年化 **+8.6%**
       ⇒ **本函数不做下限保护，也不该假装做了**。
         下限保护属于「调用方该不该用这个窗口」的判断，
         混进年化函数里会让调用方以为短窗口是安全的。
         `walk_forward.make_windows` 已用 `len(te) >= 60` 过滤短窗口，
         那里才是该加约束的地方。
       （此前此处 docstring 声称本函数已加天数下限，实测代码里并没有 ——
         注释与实现不符比没有注释更危险，故删除该声明。）
    """
    if len(dates) < 2:
        return 1e-9
    d0, d1 = dates[0], dates[-1]
    if hasattr(d0, "year"):                      # Timestamp 索引
        days = (d1 - d0).days
    else:                                        # 位置索引，兜底按交易日折算
        # 243 = A股实际年均交易日（实测 2026-01~09 为 243日）。
        # ⚠️ 这条分支是**兜底**，与主口径（自然日）不同；
        #   混用会让同一份收益算出两个年数，正是本函数要消灭的问题。
        return max(len(dates) / 243.0, 1e-9)
    return max(days / 365.25, 1e-9)


def simulate_matrix(dates, held_mat, w_mat, fwd: pd.DataFrame,
                    cost: CostModel, spec: PortfolioSpec) -> dict:
    """矩阵版回测：全程 numpy，不碰 pandas 索引。

    ⚠️ **性能**：初版逐日 `fwd.loc[d, w.index]` 做标签查找，
       5,606 只 × 2,611 日跑 20 分钟未完成。
       现在把收益矩阵与持仓矩阵一起按位置索引，全向量化。
    """
    r_mat = fwd.to_numpy(dtype=float)          # (T, N) 前视修正后的收益
    r_mat = np.nan_to_num(r_mat, nan=0.0)
    # ⚠️ **T 必须取自 dates（不是 r_mat）**（review 2026-10-06）：
    #   `gross` / `w_mat` / `held_mat` 全都是**按 dates 对齐**的，
    #   原先写 `T = r_mat.shape[0]` 是假设两者等长。
    #   当 fwd 末位没有下一期收益（行数 = len(dates)-1）时，
    #   T 少 1 → `gross[ok]` 与 `w_mat[ok]` 形状不匹配，报
    #   `IndexError: boolean index did not match ... size 4 but size 5`。
    #   生产路径恰好等长（shift 保留全行、末行填 NaN），所以这个 bug 一直藏着，
    #   直到测试用真实的 (T-1) 行 fwd 才暴露 —— 典型的「靠巧合活着」。
    T = len(dates)
    # fwd 行数可能少于 dates（末位无下一期收益）⇒ 夹住上界
    n_r = min(len(r_mat), T)
    date_pos = {d: i for i, d in enumerate(dates)}
    pos = np.array([date_pos.get(d, -1) for d in dates])
    ok = (pos >= 0) & (pos < n_r)

    # ⚠️⚠️ **哨兵绝不能映射到列 0**（实测发现，比 review 报的 BLOCK-1 更隐蔽）：
    #   `held_safe = np.where(held_mat < 0, 0, held_mat)` 把哨兵指向**第 0 列**，
    #   而第 0 列**本身就是一只真实持仓**（`held_mat` 里可能已经有 0）。
    #   散射赋值 `W[rows, idx] = vals` 对重复下标是**后写覆盖**——
    #   哨兵槽的权重 0 会把真实持仓 c0 的权重**直接抹成 0**。
    #   实测：c0 权重 0.6 被抹成 0，组合日收益从 +0.2% 掉到 −0.4%。
    #   ⇒ 给哨兵一个**独立的垃圾列**：收益矩阵与 W 都多一列（全 0），
    #     真实列仍是0..N-1。该列收益恒 0、权重恒 0，双重零贡献。
    sentinel_col = r_mat.shape[1]             # 真实列是 [0, r_mat.shape[1])
    r_ext = np.column_stack([r_mat, np.zeros(len(r_mat))])
    held_idx = np.where(held_mat < 0, sentinel_col, held_mat)

    # 组合每日收益 = Σ w[i,t] × r[t, held[i,t]]
    gross = np.zeros(T)
    if ok.any():
        # ⚠️ 索引顺序：`held_idx[ok]` 是 (T_ok, n_hold)，`pos[ok]` 是 (T_ok,)，
        #    numpy 广播时**行索引必须先取 held 再取 pos**，写反会形状不匹配。
        rr = r_ext[pos[ok][:, None], held_idx[ok]]   # (T_ok, n_hold)
        gross[ok] = np.nansum(w_mat[ok] * rr, axis=1)

    # 换手 = |w_t - w_{t-1}| 在**全股票空间**上的 L1 差
    # ⚠️ 必须在全 N 只上算，只在持仓股票上算会漏掉「卖出的股票」，
    #    实测那种算法换手率被低估约一半。
    W = np.zeros((T, sentinel_col + 1), dtype=float)   # 末列 = 哨兵垃圾列
    rows_ok = np.where(ok)[0]
    if len(rows_ok):
        # ⚠️ 需要把行索引扩展成二维 (T_ok, 1)，
        #    否则 (T_ok,) 与 (T_ok, n_hold) 广播失败。
        W[rows_ok[:, None], held_idx[rows_ok]] = w_mat[rows_ok]
    dW = np.abs(np.diff(W, axis=0, prepend=W[:1]))
    turn = dW.sum(axis=1)

    # ⚠️⚠️ **换手必须按 ok 夹取**（review 2026-10-06 复核抓出，BLOCK）：
    #   `ok=False` 的行（末位没有下一期收益）其 `W` 全为 0，
    #   而 `np.diff` 会把「有仓 → 0」视为**清仓** ⇒ `turn` = 1.0。
    #   该行既没有收益发生，却被扣一次全额往返成本 ⇒ **凭空少赚**。
    #   实测（T=5、fwd 4 行、默认成本）：末行 turn=1.0、cost=0.00151，
    #   在 4 个交易日的窗口里把年化从 107.46% 压到 80.72%（**低估 26pp**）。
    #   ⚠️ 生产路径恰好等长（`pct_change().shift(-1)` 保留全行）所以不触发，
    #      这正是「靠巧合活着」的典型 —— 一旦有第二个调用方传短 fwd 就中招。
    #   ⇒ 换手与成本都只认有收益发生的行。
    turn = np.where(ok, turn, 0.0)
    cost_arr = turn * (cost.buy + cost.sell) / 2
    net = gross - cost_arr
    # ⚠️ nav 同样只能在 ok 行上复利：非 ok 行既无收益也无成本，
    #   留着只会污染「累计净值」这个对外报出的数字。
    nav_all = np.cumprod(1.0 + net)
    nav = nav_all[ok] if ok.any() else nav_all
    dates = np.asarray(dates)[ok] if ok.any() else np.asarray(dates)

    if len(nav) < 2 or not np.isfinite(nav[-1]) or nav[-1] <= 0:
        return {"ok": False, "reason": "净值序列异常"}

    rr = pd.Series(net).replace([np.inf, -np.inf], np.nan).dropna()

    # ⚠️ **年化口径必须与基准侧一致**，否则「超额 = 策略 − 基准」是错的。
    #   实测踩过（review 2026-10-06发现）：
    #     策略侧用**自然日**：364 天跨度 / 365.25 = 0.9966 年
    #     基准侧用**交易日**：261 个交易日 / 252 = 1.0357 年
    #   同一份收益，两个年化差 0.50pp/年，而「超额」是全部结论的判据。
    #   ⇒ 统一为**自然日**口径（几何年化的标准定义）。
    #
    # ⚠️ 不用 `len(net)/252` 算年数 —— 年化换算**三分层**
    #   （各常数管各语义、勿一刀切，PITFALLS P10(1·补)）：
    #     ① 倍数换算 = 252（×252/periods、波动 ×√252、成本 ×252）；
    #     ② 无日期索引的兜底年跨越 = 交易日数 ÷ 243（年均交易日）；
    #     ③ 有日期的年跨越 = 年数按日期跨度 = 自然日 365.25。
    #   本函数带日期索引 ⇒ 年数取 ③；只剩行数口径才用 ②（len/243）。
    #   ① 的 252 只当乘数：误当除数算年数，会把 10.75 年记成 10.36 年
    #   （PITFALLS P10(1·补) 实测），年化被系统性抬高。
    years = _year_span(dates)
    cagr = float(nav[-1] ** (1 / years) - 1) if nav[-1] > 0 else -1.0
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
