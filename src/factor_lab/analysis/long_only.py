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

import numpy as np
import pandas as pd

from factor_lab.analysis.costs import CostModel, breakeven_turnover
from factor_lab.analysis.engine import (
    _carry_forward,
    _year_span,
    simulate_matrix,
)
from factor_lab.analysis.selection import (
    align_masks,
    rank_topk,
    select_with_buffer,
)
from factor_lab.analysis.spec import (
    PortfolioSpec,
    _normalize_bounded,
)
from factor_lab.config import SCALING_TRADING_DAYS

# ⚠️ **re-export**：`PortfolioSpec` / `_normalize_bounded` 已搬到 `spec.py`，
#   但外部调用方（含既有测试）从 `long_only` 导入它们。
#   ⇒ 这里显式再导出，避免一次拆分就打断所有调用方。
__all__ = [
    "PortfolioSpec",
    "_normalize_bounded",
    "build_long_only",
    "simulate_matrix",
    "rebalance_days",
    "holdout_split",
]


def build_long_only(
    factor_values: pd.DataFrame,
    spec: PortfolioSpec,
    cost: CostModel,
    price_panel: pd.DataFrame | None = None,
    buy_ok: pd.DataFrame | None = None,
    sell_ok: pd.DataFrame | None = None,
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
    buy_ok/sell_ok: 索引=日期，列=代码的 bool 面板，
                    True = 当日可买/ 可卖。None = 不施加涨跌停约束。
                    ⚠️ **必须与 factor_values 的索引、列完全一致** ——
                       `align_masks` 按**标签**校验，索引或列不一致会
                       **直接抛错**（ValueError），不会静默错配。
                       （早期版本按下标对齐，列序不同会静默错配；
                        标签校验即为此而加。）

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
        20, spec.vol_lookback // 3)).std() * np.sqrt(SCALING_TRADING_DAYS))

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

    # ⚠️⚠️ **涨跌停约束接在这里**（2026-10-06 接线，此前从未生效）：
    #   `tradability.py` 建好了、`limit_masks` 有覆盖率自检，
    #   但**全项目零生产调用方** —— 所有已披露结论都是在
    #   「假设涨跌停板随时能成交」下得出的。
    #   本项目的因子（反转）选的正是「刚跌过」的股票，
    #   而跌停股恰恰是「跌得最狠、最卖不掉」的那一批 ⇒ 影响不是随机噪声。
    #
    #   对齐与形状校验在 `selection.align_masks`（选股侧职责）。
    buy_m, sell_m = align_masks(buy_ok, sell_ok, factor_values)
    held_mat = select_with_buffer(top_idx, is_rebal, n_hold,
                                  buy_ok=buy_m, sell_ok=sell_m)

    inv = np.where(vol_mat > 0, 1.0 / vol_mat, np.nan)
    # ⚠️ **必须先把 -1 哨兵挡住，不能让它进 `np.take_along_axis`**，也不能用
    #   clip(下界, 0) 代替：
    #   `np.take_along_axis` 用的是**负索引语义** ——
    #   held_mat 里的 -1 会取到**最后一列**（某只真实存在的股票），
    #   而不是「无持仓」。
    #   实测踩过：把 0 当哨兵同样错，列索引 0 是真实股票（如 sh600000）。
    #   ⇒ 实际步骤是两步：take 时用 `held_safe` 把哨兵槽**临时占位**成列 0，
    #     take 之后立刻由 `bad` 掩码（含 `held_mat < 0`）把哨兵槽的取值
    #     连同非法逆波动率一起置 NaN —— 哨兵槽不携带任何列 0 的信息。
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
    # ⚠️ `degenerate`（有效持仓不足一半）⇒ 退化为等权。
    #   必须**在带约束归一化之前**处理：等权1/n_hold 天然满足上下限，
    #   而逆波动率权重经过归一化后**必然触顶**（少数高波动槽被压到hi，
    #   剩余权重分摊给其余槽位），这正是需要约束投影的场景。
    w_eq = np.full((T, n_hold), 1.0 / n_hold)
    w_mat = np.where(degenerate[:, None], w_eq, w_mat)
    # ⚠️ **哨兵槽位（held_mat < 0）权重恒为 0**，不能参与归一化 ——
    #   否则 `n_hold=30` 里若有 1 个哨兵，就会拿 29 个槽位去凑 100%，
    #   凭空把其余权重抬高 3.4%。它们代表「这一格没有持仓」，
    #   `held_safe` 会把 -1 映射到列 0（**一只真实存在的股票**），
    #   权重算错就等于给一只没打算买的股票建仓。
    w_mat = np.where(held_mat < 0, 0.0, w_mat)
    # 🔴🔴 **BLOCK（review 2026-10-06 抓出）：clip 后不能再等比归一化**
    #   初版是 `np.clip(...)` 接 `w_mat / w_mat.sum(...)`，
    #   但**除以行和必然把权重和拉回 1**，而 `sum < 1` 时是**等比放大**
    #   ⇒ clip 的效果被完全抹掉。
    #
    #   实测（max_weight=0.15，3 个有效槽 + 1 哨兵）：
    #       原始逆波动率 [0.50, 0.30, 0.20] → clip → [0.15, 0.15, 0.15, 0.02]
    #       → 等比归一化 → **[0.333, 0.333, 0.333]**  = 上限的 **2.22 倍**
    #
    #   ⇒ `max_weight` 是**死参数**，而 `PortfolioSpec.describe()`
    #      正在向用户打印「单票 ≤15%」—— **文档声明与实际行为相反**。
    #      这正是 ENGINEERING.md 第四节点名的「不报错但结论完全颠倒」。
    #
    #   ⚠️ 1397eac / f08667d 都把它误标为「`degenerate` 兜底绕过 max_weight」
    #      的局部问题。实测**正常路径同样绕过**，
    #      只改 `degenerate` 分支会漏掉主路径。
    #
    #   正解：**带上下限约束的归一化**（水位法投影）。
    #   不是「先 clip 再归一化」，而是「归一化时就把上下限算进去」。
    w_mat = _normalize_bounded(w_mat, spec.min_weight, spec.max_weight,
                               active_mask=held_mat >= 0)

    # 非调仓日**沿用上一期权重**。
    # ⚠️ held_mat **不要**再 carry_forward —— select_with_buffer 内部已经
    #   在非调仓日填了上一期的持仓，再做一次是重复（虽幂等，但会让人
    #   误以为缓冲带没生效）。
    w_mat = _carry_forward(w_mat, is_rebal)
    # 首个调仓日之前无持仓 → 全零
    first = int(np.argmax(is_rebal)) if is_rebal.any() else T
    w_mat[:first] = 0.0
    w_mat = w_mat / np.maximum(w_mat.sum(axis=1, keepdims=True), 1e-12)

    res = simulate_matrix(factor_values.index, held_mat, w_mat,
                          fwd, cost, spec)
    # ⚠️ **必须把 held_mat 带出去**（2026-10-06 review BLOCK）：
    #   「涨跌停约束到底生效了没有」只能通过**检查实际持仓**回答 ——
    #   持仓里若还有封涨停的票，说明约束没生效。
    #   而 `held_mat` 原先是 `simulate_matrix` 的局部变量，调用方拿不到
    #   ⇒ `run_limit_constraint` 里的校验函数 `verify_constraint_active`
    #   **接不上**（签名要held_mat，返回值里没有），只能空转。
    #   ⇒ 无条件附带（非 ok 时也给 None），调用方自行判None。
    res["held_mat"] = held_mat if res.get("ok") else None
    return res


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


def holdout_split(dates, train_frac: float = 0.7) -> tuple:
    """按时间切分样本内/外。

    ⚠️ **必须按时间切，不能随机切** —— 金融数据有时间依赖，
       随机切会让训练集包含「未来」信息，IC 直接虚高。
    """
    dates = sorted(dates)
    n = len(dates)
    k = int(n * train_frac)
    return dates[:k], dates[k:]
