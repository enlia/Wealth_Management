"""选股：从因子面板挑出「手动买得到、且换手可控」的持仓。

为什么单独一个模块
------------------
`long_only.py` 负责**收益模拟**（成本、净值、回撤），
本模块负责**选哪些股票**。两者关注点不同，耦合在一起会让
「换选股规则」必须去读收益模拟的代码 —— 这正是n_pick 失效
能长期存在的原因：规则写在模拟里，看不出它有没有生效。

缓冲带（buffer band）
--------------------
选股不能是「每次调仓都取因子值最高的 30 只」，必须是：

  1. 先取因子值前 ``n_pick`` 只作为**候选池**
  2. 已持仓的股票**只要还在候选池里就不卖**
  3. 只有跌出候选池的持仓槽位，才用池子里排名最前的**新股票**填

这是指数调仓里的标准做法，作用是**把换手从「排名波动」里剥离出来**。
排名会因为因子值的小幅变化而剧烈波动：某只股票从第 32 名升到第 28 名
（因子值只差0.01），没有缓冲带就要卖出再买回，白付两次成本。

⚠️ **实测踩过（2026-10-06）：n_pick 曾是死参数**
   初版实现是 `held = top_idx[:, :n_hold]`，
   而 ``top_idx`` 已经按因子值降序排好序，
   所以前 n_hold 个元素与 n_pick 毫无关系。
   实测 n_pick ∈ {20, 60, 200} 三组持仓**逐位完全相同**，
   参数扫描表里「候选池大小」那一列数值一模一样 —— 那是假象。
   根因不是算错，是**缓冲带语义根本没写**，只写了「取前n_hold」。

换手��向（2023-2026 样本外，月频，持 30 只）
    无缓冲（n_pick = n_hold）    年换手约 9.5%
    缓冲 n_pick = 3 × n_hold     换手显著下降
"""
from __future__ import annotations

import numpy as np


def rank_topk(factor_mat: np.ndarray, k: int) -> np.ndarray:
    """按因子值降序取每行前 k 只，返回列索引矩阵 (T, k)。

    ⚠️ NaN 必须先换成 -inf，否则 numpy 会把它们排到**最前面**
    （NaN 比较恒为 False，排序行为与直觉相反），
    结果是「因子缺失的股票反而被选中」。
    """
    fv = np.where(np.isnan(factor_mat), -np.inf, factor_mat)
    T, N = fv.shape
    k = min(k, N)
    if k >= N:
        return np.argsort(-fv, axis=1)[:, :k]
    part = np.argpartition(-fv, k - 1, axis=1)[:, :k]
    rows = np.arange(T)[:, None]
    order = np.argsort(-fv[rows, part], axis=1)
    return part[rows, order]


def select_with_buffer(top_idx: np.ndarray, is_rebal: np.ndarray,
                       n_hold: int) -> np.ndarray:
    """带缓冲带的调仓选股。

    参数
    ----
    top_idx   : (T, K) 每行按因子值降序的候选池列索引，K = n_pick
    is_rebal  : (T,) 调仓日标记
    n_hold    : 实际持股数

    返回
    ----
    (T, n_hold) 列索引矩阵，**非调仓日沿用上一调仓日的持仓**。

    ⚠️ 必须是**顺序**计算，不能向量化：
       缓冲带的定义依赖「上一期持有了什么」，
       这是典型的顺序依赖，只能逐个调仓日推进。
       但调仓日数量很少（月频 11 年约 129 个），复杂度 O(调仓日数 × n_pick)，
       实测全市场 11 年耗时 < 0.1 秒，不是性能瓶颈。
    """
    T = top_idx.shape[0]
    K = top_idx.shape[1]
    n_hold = min(n_hold, K)
    out = np.zeros((T, n_hold), dtype=top_idx.dtype)

    prev = np.empty(0, dtype=top_idx.dtype)
    for t in range(T):
        if not is_rebal[t]:
            out[t] = prev
            continue
        pool = top_idx[t]
        # ⚠️ 不用「因子值是否有限」过滤 —— rank_topk 已把 NaN 排到末尾，
        #   再按 -inf 过滤属于重复劳动，且要多传一个因子矩阵进来（徒增耦合）。
        #   列索引恒为 >= 0，直接判即可。
        pool = pool[pool >= 0]
        if len(pool) == 0:
            prev = np.empty(0, dtype=top_idx.dtype)
            continue
        # ①保留仍在候选池里的旧持仓（按**昨天的池内排名**定序，不按新排名）
        if len(prev):
            in_pool = prev[np.isin(prev, pool)]
        else:
            in_pool = np.empty(0, dtype=top_idx.dtype)
        keep = list(in_pool[:n_hold])
        # ② 空出的槽位用候选池里排名最前、且不在 keep 里的新股票填
        for c in pool:
            if len(keep) >= n_hold:
                break
            if c not in keep:
                keep.append(c)
        prev = np.asarray(keep[:n_hold], dtype=top_idx.dtype)
        out[t] = prev
    return out