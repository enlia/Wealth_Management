"""回测引擎：把持仓矩阵与收益矩阵算成净值序列。

## 为什么从 `long_only.py` 拆出来

`long_only.py` 原本 681 行，超过ENGINEERING.md 的 500 行硬上限
（CI 检查 3 是硬门禁，不通过就不能合并）。

拆分依据是**职责**而非行数：
- 本模块：**给定持仓与权重 ⇒ 算净值/年化/换手/成本**（纯计算，无选股逻辑）
- `long_only.py`：**因子 ⇒ 选股 ⇒ 定权重**（策略构建）

初版 `costs.py` 拆分时已经犯过一次「撞线才拆」的错，
这次直接按职责切，避免第三次。

## 本模块不做的事

- **不选股**（那是 `selection.py` 的事）
- **不定权重**（那是 `long_only.py` 的事）
- **不读文件、不查数据库**（数据由调用方准备好）

⇒ 可以脱离整个回测流程单独测试：给定 `held_mat` / `w_mat` / `fwd`
   就能验证净值、成本、换手、年化口径。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from factor_lab.analysis.costs import CostModel

if TYPE_CHECKING:                      # 避免运行期循环导入
    from factor_lab.analysis.long_only import PortfolioSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


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
    # ⚠️ 不用 `len(net)/252`：A 股一年实际约 243~245 个交易日，
    #   252 是美股口径，用它会让「年数」偏大约 3.5%，年化被系统性压低。
    #
    # ⚠️ 下方 vol/成本的年化仍用固定系数 252（`std×√252`、`cost×252`）：
    #   这是日频波动/成本放大到年频的惯例系数，与「年数」的自然日口径
    #   互不相干（年数只管 CAGR），两者并存是有意的，不是漏改。
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

