"""组合参数与权重约束归一化。

## 为什么从 `long_only.py` 拆出来

`long_only.py` 加了带约束归一化后达到 519 行，超过 ENGINEERING.md
的500 行硬上限（CI 检查 3 是硬门禁）。

拆分依据是**职责**：
- 本模块：**「目标持仓长什么样」** —— 参数定义 + 权重约束求解
- `long_only.py`：**「怎么从因子得到持仓」** —— 选股/ 定权重 / 调引擎
- `engine.py`：**「持仓怎么变成净值」** —— 纯计算

⚠️ 本项目已因同一文件撞线两次（`TradabilityFilter` → `tradable.py`、
`CostModel` → `costs.py`），第三次才按职责切 ——
说明**该拆的是职责，不是行数**。

## 关键约束：`n_hold` 与 `max_weight` 不是独立的

    n_hold × max_weight ≥ 1     否则无法满仓
    n_hold × min_weight ≤ 1     否则无法买满

实测默认值下`n_hold=2/3/5` 全部违反第一条（2 × 15% = 30%），
而旧的「clip + 等比归一化」实现**不报错**、只是静默把权重
放大到上限之上（实测 2.22 倍）⇒ `max_weight` 成了死参数。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _normalize_bounded(w: np.ndarray, lo: float, hi: float,
                       active_mask: np.ndarray | None = None,
                       iters: int = 60) -> np.ndarray:
    """带上下限的行归一化：每行缩放到和为 1，且全部落在 `[lo, hi]`。

    ## 为什么不能用「先 clip 再除以行和」

    除以行和必然把和拉回 1；而 `sum < 1` 时是**等比放大**，
    于是 clip 的效果被完全抹掉。实测 `max_weight=0.15` 时
    最终出现 **0.333** 的单票权重 = 上限的 2.22 倍，
    而 `PortfolioSpec.describe()` 正在打印「单票 ≤15%」。

    ## 算法：对「截断阈值 λ」做二分，而不是迭代分摊

    带上下限的等比归一化没有初等闭式解，但可以**逐元素二分**：

        求唯一 λ ≥ 0 使得   Σ clip(w_i·λ, lo, hi) = 1

    λ 的左边函数 Σ clip(·) 对 λ **严格单调递增**（lo ≤ hi 时）
    ⇒ 二分必然收敛，且**结果精确可行**（残差只来自二分的浮点比较）。

    ⚠️ **不要用「迭代分摊」**（我先写了这版，实测有问题）：
       把越界槽位钉到边界、再把缺口按余量分摊回去，看似合理，
       实测**收敛速度是线性的且会振荡** ——
       越界槽位数在 6↔2 之间来回跳，每轮缺口只减半。
       `n_hold=10, hi=10%`、权重差 20 倍时，
       **60 次迭代后仍超出上限 1.3e-8**，
       终检（tol=1e-6）虽然放行，但离「守约束」差 5 个数量级。
       ⇒ 二分法 40 次即可达到机器精度，且没有振荡。

    参数
    ----
    w : (T, N) 非负权重，**允许行和为 0**（全零行原样返回）
    lo, hi : 下限与上限。必须 `0 <= lo <= hi`
    active_mask : (T, N) bool，标记**真正要归一化的槽位**。
        ⚠️ 哨兵槽位（`held_mat < 0`，代表「这一格没有持仓」）
           必须置 False —— 它们既不参与权重分配，也不能计入
           可行性检查的分母。否则 `n_hold=30` 里若有 1 个哨兵，
           就会拿 29 个槽位去凑 100%，凭空抬高其余权重。
    iters : 二分次数。40 次远超 double 的有效位（~17 位十进制）

    ## 不可行时的处理（分两种，不是统一抛错）

    逐行判 `hi * k < 1`（k = 该行有效槽位数）：

    | 情形 | 处理 | 理由 |
    |---|---|---|
    | **部分行**不可行 | 该行退化为**等权** + 往 stderr 报行号 | 等权是该行唯一可行解；不报会静默降级 |
    | **多行**都不可行（≥2 且 ≥10%） | `raise ValueError` | 系统性配置错误，必须显式失败 |

    ⚠️ 早期版本对不可行一律 `raise`，实测会崩在离原因很远的地方：
    `n_hold=10, max_weight=10%` 需要 10 个有效槽位，
    而 `degenerate` 阈值是 `n_valid < n_hold*0.5 = 5`
    ⇒ 实际持仓 6~9 只时不进 `degenerate`，但归一化必然失败 ⇒ 整个回测崩。
    """
    if not (0.0 <= lo <= hi):
        raise ValueError(f"权重上下限非法: lo={lo}, hi={hi}")
    out = np.array(w, dtype=float, copy=True)
    if out.ndim == 1:
        # ⚠️ 允许单行输入。实测踩过：传一维时 `T, N = out.shape` 直接
        #   `ValueError: not enough values to unpack`，而调用方
        #   （测试、单只股票调试）很自然会传一维。
        out = out.reshape(1, -1)
    if out.size == 0:
        return out
    T, N = out.shape
    if active_mask is None:
        active_mask = np.ones((T, N), dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool)
    # ⚠️⚠️ **非 active 槽位必须清零**（实测踩过）：
    #   本函数只在 active 槽位上做除法与迭代，**非 active 位置会保留
    #   输入的原始值**。实测传 `raw` + 前 6 槽 inactive：
    #   输出行和 = **8.89**（其中 7.89 来自那6 个本该为 0 的槽位）。
    #   ⇒ 函数不能依赖调用方预先清零 —— 那是隐式契约，
    #     换个调用方就静默出错（多建仓/权重和爆表）。
    out = np.where(active_mask, out, 0.0)
    m = active_mask.sum(axis=1)

    # ⚠️⚠️ **可行性按「每行有效槽位数」判，且不可行时不能抛错**
    #   某一行持仓数不足时，`hi * k < 1` ⇒ 该行**不存在**满足上限的权重。
    #   实测这不是理论风险：`n_hold=10, max_weight=10%` 需要 10 个有效槽位，
    #   而 `degenerate` 阈值是 `n_valid < n_hold*0.5 = 5`
    #   ⇒ 实际持仓 6~9 只时**不**进degenerate，但带约束归一化必然失败。
    #   早期版本直接 raise ⇒ 整个回测崩溃，且崩在离原因很远的地方。
    #
    #   正确处理：不可行的行**退化为等权并显式报出**。
    #   ⚠️ 这不是静默 fallback —— 等权是**唯一可行**的选择（该行无论如何
    #      都满足不了上限），且会打印行号与槽位数，不会被忽略。
    #   反过来，若用户把 `max_weight` 设成 0.4% 这种全局不可能值，
    #   则**所有**行都不可行 ⇒ 直接抛错（那是配置错误，必须显式失败）。
    impossible = (m > 0) & (hi * m < 1.0 - 1e-9)
    n_impossible = int(impossible.sum())
    # ⚠️ 只有**多行**不可行才抛错。单行不可行是正常现象
    #   （某天可买的票不够），为此中断整个回测是错的 ——
    #   实测 n_hold=10、hi=10% 时，持仓 6~9 只的行**每天**都不可行。
    #   ⚠️ 判据不能用纯百分比：T=1 时「1 行不可行」就是 100%，
    #   会把单行退化误判成配置错误。必须有绝对下限。
    n_live_rows = int((m > 0).sum())
    thresh = max(2, int(0.10 * n_live_rows))
    if n_live_rows and n_impossible >= thresh and n_impossible >= 2:
        raise ValueError(
            f"权重上限不可行：{n_impossible}/{n_live_rows} 行的有效槽位数不足"
            f"（至少需要 {int(np.ceil(1 / hi - 1e-9))} 个）。\n"
            f"  max_weight={hi:.2%} 与当前持仓数系统性不兼容，"
            f"请调高 max_weight 或增加 n_hold。")
    if n_live_rows and lo > 0 and int(
            ((m > 0) & (lo * m > 1.0 + 1e-9)).sum()) >= thresh:
        raise ValueError(
            f"权重下限不可行：≥{thresh} 行 × 下限 {lo:.2%}超过 100%。\n"
            f"  请调低 min_weight 或增加 n_hold。")
    if n_impossible:
        first = int(np.argmax(impossible))
        print(f"  ⚠ {n_impossible}/{n_live_rows} 行持仓数不足以满足 "
              f"max_weight={hi:.2%}（至少需 {int(np.ceil(1 / hi - 1e-9))} 只），"
              f"这些行退化为等权。首次出现于第 {first} 行"
              f"（{int(m[first])} 只有效）。",
              file=sys.stderr)

    for t in range(T):
        act = active_mask[t]
        k = int(act.sum())
        if k == 0:
            continue
        if hi * k < 1.0 - 1e-9:
            # 该行无法满足上限 ⇒ 等权（唯一可行解）
            out[t] = np.where(act, 1.0 / k, 0.0)
            continue
        row = out[t]
        base = row[act].copy()
        if not np.isfinite(base).all() or base.min() < 0:
            raise ValueError(
                f"第 {t} 行权重含非有限值或负值，不可归一化："
                f"{np.round(base, 4).tolist()[:10]}")
        total = float(base.sum())
        if total <= 0:
            # ⚠️ **全零有效槽必须保持全零**，不能退化成等权。
            #   「有效但权重和为 0」= 没有任何真实持仓
            #   ⇒ 填等权等于**凭空建仓**（CODE_TRUST P22 同一类错误）。
            #   ⚠️ 与上面「不可行行退化为等权」是**两种不同情况**：
            #      那里 k 个槽位确实有持仓，只是满足不了上限；
            #      这里根本没有持仓可分配。
            out[t] = 0.0
            continue
        # ── 二分求 λ：Σ clip(base·λ, lo, hi) = 1 ──
        # 左端取使所有槽位都触hi 的 λ（此时和最大 = k·hi ≥ 1，可行时）
        # 右端取使所有槽位都触 lo 的 λ（和最小 = k·lo ≤ 1）
        lo_lam = 0.0
        hi_lam = 1.0
        while float(np.clip(base * hi_lam, lo, hi).sum()) < 1.0:
            hi_lam *= 2.0
            if hi_lam > 1e300:
                raise RuntimeError(
                    f"第 {t} 行二分上界溢出：权重 {base[:5].tolist()}，"
                    f"上下限 [{lo}, {hi}] 下无法凑到 1")
        for _ in range(iters):
            mid = 0.5 * (lo_lam + hi_lam)
            s = float(np.clip(base * mid, lo, hi).sum())
            if s < 1.0:
                lo_lam = mid
            else:
                hi_lam = mid
        lam = 0.5 * (lo_lam + hi_lam)
        row_out = np.zeros(N)
        # ⚠️ 不能用 `np.where(act, clip(base*lam), 0.0)`：
        #   `base` 是 `row[act]` 取出来的**压缩数组**（长度 k），
        #   与长度 N 的 act 广播不上（实测 shapes (30,) vs (24,)）。
        #   必须先算压缩结果，再写回原位置。
        row_out[act] = np.clip(base * lam, lo, hi)
        out[t] = row_out

    # 终检：约束必须真的成立，否则说明没收敛。
    # ⚠️ 只对「本来可行」的行检查 —— 不可行行已退化为等权，
    #    等权可能超过 hi（如 6 只股票、hi=0.15 ⇒ 每只 16.7%），
    #    那是**已报出的**降级，不是未收敛。
    chk = active_mask & ~impossible[:, None]
    chk &= (out > hi + 1e-6) | (out < lo - 1e-6)
    if chk.any():
        r = int(np.argmax(chk.any(axis=1)))
        raise RuntimeError(
            f"带约束归一化未收敛：第 {r} 行权重 "
            f"{np.round(out[r], 4).tolist()}，"
            f"超出 [{lo:.4f}, {hi:.4f}]。\n"
            f"  迭代 {iters} 次仍不收敛，通常是 lo/hi 贴着可行域边界。")
    return out


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

    def __post_init__(self) -> None:
        """🔴 **校正数学上不可行的上下限**，否则整个回测无法运行。

        ## 为什么必须在这里处理

        `n_hold` 只与 `max_weight` 存在硬约束：

            n_hold × max_weight ≥ 1     （否则无法满仓）
            n_hold × min_weight ≤ 1     （否则无法买满）

        实测默认值下`n_hold=2/3/5` **全部违反**第一条
        （2 × 15% = 30% < 100%）⇒ 任何正确实现都会失败。
        而旧的「clip + 等比归一化」实现**不报错**，
        只是静默把权重放大到上限之上 ⇒ 这正是 BLOCK 的本体。

        ⇒ 两种选择：
          (a) 让 `_normalize_bounded` 抛错 ⇒ 冒烟测试与小样本研究全部不可用
          (b) 在这里把上下限**收敛到可行域内**并显式告警

        选(b)，因为 `n_hold < 7` 只是**小样本测试配置**，
        而 `max_weight=15%` 是给 30+ 只的真实组合设计的。
        自动收敛不改变真实配置的结论（30 只时 15% 本就可行），
        只让小样本能跑起来。

        ⚠️ 必须在 `describe()` **之前**生效，
        否则打印的仍是用户传入的、与实际行为不一致的数。
        """
        floor_hi = 1.0 / self.n_hold
        if self.max_weight < floor_hi:
            print(f"  ⚠ n_hold={self.n_hold} 时单票上限 {self.max_weight:.2%} "
                  f"数学上不可行（{self.n_hold} 只最多只能到 "
                  f"{floor_hi:.2%}），已上调至 {floor_hi:.2%}。",
                  file=sys.stderr)
            self.max_weight = floor_hi
        ceil_lo = 1.0 / self.n_hold
        if self.min_weight > ceil_lo:
            print(f"  ⚠ n_hold={self.n_hold} 时单票下限 {self.min_weight:.2%} "
                  f"数学上不可行（{self.n_hold} 只至少要 {ceil_lo:.2%}/只），"
                  f"已下调至 {ceil_lo:.2%}。", file=sys.stderr)
            self.min_weight = ceil_lo

    def describe(self) -> str:
        return (f"{self.name}: 持股 {self.n_hold}/{self.n_pick}只 · "
                f"{self.rebalance}调仓 · 单票 ≤{self.max_weight:.0%} · "
                f"目标波动 {self.vol_target:.0%} · 止损 {self.stop_loss:.0%}")
