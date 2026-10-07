"""止损位 / 目标位 / 买点区间：把「逻辑上的赔率 R」落成「具体挂单价格」。

两种互斥口径（二选一，禁止同时给、也禁止都不给）
------------------------------------------------
1. **ATR 倍数**：``止损距离 = stop_atr × ATR(n)``，``目标距离 = target_atr × ATR(n)``
   默认 ``2×ATR`` / ``4×ATR`` ⇒ ``R = target_atr / stop_atr = 2``。
2. **用户给定百分比**：``止损距离 = price × stop_pct``，``目标距离 = price × target_pct``。

量纲（UNITS 口径）
------------------
- ``price`` / ``atr`` / 全部输出价格：**元/股**，A 股未复权报价口径（挂单价）。
- ``stop_mult`` / ``target_mult`` / ``entry_fraction``：**无量纲倍数**。
- ``r_ratio``：**无量纲**（目标距离 ÷ 止损距离 = 每 1 元风险对应的收益，A 股多头无做空）。

⚠️ 禁止静默 fallback（ENGINEERING 第四节）
-------------------------------------------
- 缺 ATR / ATR ≤ 0 / ATR 为 NaN → **抛 ValueError**，不偷偷改用百分比口径。
- 止损距离或目标距离算出来 ≤ 0 → **抛 ValueError**，不返回负价格的「建议」。
- ``target_distance ≤ stop_distance``（R ≤ 1）→ **抛 ValueError**，那不是可执行的建议。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["AtrLevels", "Levels", "PercentLevels", "build_levels"]

_UNIT_NOTE = (
    "价格单位=元/股（A 股未复权报价口径，挂单价）；"
    "mult/pct/r_ratio 无量纲；本模块不做年化（输出单笔 EV 与仓位，非年化收益）"
)


def _require_positive(value: float | None, name: str, hint: str) -> float:
    """校验一个必须存在且 > 0 的浮点数，缺失一律抛错（不静默取默认值）。"""
    if value is None:
        raise ValueError(f"{name} 缺失（None）—— {hint}")
    try:
        v = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} 无法解析为数字：{value!r}（类型 {type(value).__name__}）") from e
    if math.isnan(v):
        raise ValueError(f"{name} 是 NaN —— {hint}")
    if math.isinf(v):
        raise ValueError(f"{name} 是无穷值 —— 源数据异常，{hint}")
    if v <= 0:
        raise ValueError(f"{name} 必须 > 0，得到 {v} —— {hint}")
    return v


@dataclass(frozen=True)
class Levels:
    """一次买卖的价格三件套（元/股，未复权报价口径）。

    正常建议满足 ``stop < entry_low ≤ entry_high < target``；
    不满足时 ``__post_init__`` 直接抛错（不给「看起来像建议」的坏数据）。
    """

    method: str            # 'atr' 或 'percent'
    price: float           # 现价/参考价（元）
    entry_low: float
    entry_high: float
    stop: float            # 止损价（元）：跌破即离场
    target: float          # 目标价（元）
    stop_distance: float   # 每股风险（元）
    target_distance: float # 每股收益（元）
    r_ratio: float         # 赔率 = target_distance / stop_distance
    unit_note: str         # 量纲说明，进卡片

    def __post_init__(self) -> None:
        if self.stop_distance <= 0:
            raise ValueError(f"stop_distance 必须 > 0，得到 {self.stop_distance}")
        if self.target_distance <= self.stop_distance:
            raise ValueError(
                f"目标距离({self.target_distance}) 必须 > 止损距离({self.stop_distance})，"
                f"否则 R ≤ 1：不是可执行的建议"
            )
        if self.entry_low > self.entry_high:
            raise ValueError(
                f"买点区间上下界反了：entry_low={self.entry_low} > entry_high={self.entry_high}"
            )
        if not self.stop < self.entry_low:
            raise ValueError(
                f"止损价必须低于买点下界：stop={self.stop} 未 < entry_low={self.entry_low}"
            )
        if not self.entry_high < self.target:
            raise ValueError(
                f"目标价必须高于买点上界：target={self.target} 未 > entry_high={self.entry_high}"
            )
        if abs(self.r_ratio - self.target_distance / self.stop_distance) > 1e-9:
            raise ValueError(
                f"r_ratio 与距离不一致：r_ratio={self.r_ratio} "
                f"而 target/stop={self.target_distance / self.stop_distance}"
            )


@dataclass(frozen=True)
class AtrLevels(Levels):
    """ATR 口径：止损/目标由 ATR(n) 的倍数决定。"""

    atr: float = 0.0            # 元/股
    atr_window: int = 14
    stop_atr: float = 2.0
    target_atr: float = 4.0


@dataclass(frozen=True)
class PercentLevels(Levels):
    """百分比口径：止损/目标由现价的百分比决定。"""

    stop_pct: float = 0.0
    target_pct: float = 0.0


def _entry_zone(price: float, fraction: float) -> tuple[float, float]:
    """买点区间 = 现价 ± fraction。fraction 必须在 [0,1) 内。"""
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"entry_fraction 必须在 [0,1) 内，得到 {fraction}")
    return price * (1.0 - fraction), price * (1.0 + fraction)


def build_levels(
    price: float,
    *,
    atr: float | None = None,
    stop_atr: float = 2.0,
    target_atr: float | None = None,
    atr_window: int = 14,
    stop_pct: float | None = None,
    target_pct: float | None = None,
    r: float | None = None,
    entry_fraction: float = 0.01,
) -> AtrLevels | PercentLevels:
    """构造止损/目标/买点区间。ATR 口径与百分比口径**二选一**。

    参数
    ----
    price : 现价（元/股，未复权报价）。
    atr : ATR(n) 值（元/股）。给 ``atr`` 即走 ATR 口径。
    stop_atr : 止损 ATR 倍数（默认 2.0）。
    target_atr : 目标 ATR 倍数（默认 4.0 ⇒ R=2）。
    r : **赔率覆盖**。给了 ``r`` 就用 ``目标距离 = 止损距离 × r``
        （此时 ``target_atr`` 必须未指定，否则两处同时指定赔率 = 口径冲突，抛错）。
    stop_pct, target_pct : 百分比小数（0.08 = 8%）。
    entry_fraction : 买点区间半宽（相对现价），默认 0.01 = ±1%。

    赔率的唯一来源（防止「价位的 R」与「EV 的 R」不是同一个数）
    ------------------------------------------------------------
    优先级与互斥：``r`` ⟂ ``target_atr`` ⟂ ``target_pct`` 三者只能有一个生效：
    - 给 ``r`` → 目标距离由 r 推出；
    - 没给 ``r`` → 目标距离由 ``target_atr``（ATR 口径，默认 4.0）
      或 ``target_pct``（百分比口径）推出。
    """
    px = _require_positive(price, "price", "现价必须为正的元/股报价")
    use_atr = atr is not None
    use_pct = stop_pct is not None or target_pct is not None
    if use_atr and use_pct:
        raise ValueError(
            "ATR 口径与百分比口径**互斥**：同时给了 atr 与 stop_pct/target_pct。"
            "请只保留一种（--atr 或 --stop-pct/--target-pct）"
        )
    if not use_atr and not use_pct:
        raise ValueError(
            "atr 缺失且未给百分比口径 —— 必须二选一给出止损/目标口径："
            "要么给 atr，要么同时给 stop_pct 与 target_pct。"
            "禁止在缺 ATR 时静默改用别的口径或默认倍数"
        )
    r_override = None if r is None else _require_positive(r, "r", "赔率必须 > 1")
    if r_override is not None and r_override <= 1.0:
        raise ValueError(f"r 必须 > 1（赔率），得到 {r}")

    entry_low, entry_high = _entry_zone(px, entry_fraction)

    if use_atr:
        a = _require_positive(
            atr, "atr",
            "缺失 ATR 时禁止改用百分比口径静默兜底；请先算 ATR 或显式指定百分比",
        )
        s_mult = _require_positive(stop_atr, "stop_atr", "止损 ATR 倍数必须为正")
        if r_override is not None:
            if target_atr is not None:
                raise ValueError(
                    "赔率来源冲突：同时给了 r 与 target_atr。"
                    "请只留一个（--r 或 --target-atr），否则卡片上的赔率有两个出处"
                )
            t_mult = s_mult * r_override
        else:
            t_mult = _require_positive(
                target_atr if target_atr is not None else 4.0,
                "target_atr", "目标 ATR 倍数必须为正",
            )
        sd, td = s_mult * a, t_mult * a
        _guard_executable(px, sd, td, entry_low)
        return AtrLevels(
            method="atr", price=px, entry_low=entry_low, entry_high=entry_high,
            stop=px - sd, target=px + td,
            stop_distance=sd, target_distance=td, r_ratio=td / sd, unit_note=_UNIT_NOTE,
            atr=a, atr_window=int(atr_window), stop_atr=s_mult, target_atr=t_mult,
        )

    if stop_pct is None or target_pct is None:
        raise ValueError(
            f"百分比口径必须**同时**给 stop_pct 与 target_pct，"
            f"得到 stop_pct={stop_pct!r} target_pct={target_pct!r}"
        )
    sp = _require_positive(stop_pct, "stop_pct", "止损百分比必须为正（0.08 = 8%）")
    tp = _require_positive(target_pct, "target_pct", "目标百分比必须为正（0.16 = 16%）")
    if sp >= 1.0 or tp >= 1.0:
        raise ValueError(
            f"百分比口径请用小数（0.08 = 8%）：得到 stop_pct={sp} target_pct={tp}，"
            f"≥1 说明误把百分数当小数传了"
        )
    if r_override is None and tp <= sp:
        raise ValueError(
            f"目标百分比({tp}) 必须 > 止损百分比({sp})，否则 R ≤ 1；"
            f"若想用赔率控制目标位，请改用 r 参数"
        )
    sd = px * sp
    if r_override is not None:
        tp = r_override * sp
    td = px * tp
    _guard_executable(px, sd, td, entry_low)
    return PercentLevels(
        method="percent", price=px, entry_low=entry_low, entry_high=entry_high,
        stop=px - sd, target=px + td,
        stop_distance=sd, target_distance=td, r_ratio=td / sd, unit_note=_UNIT_NOTE,
        stop_pct=sp, target_pct=tp,
    )


def _guard_executable(px: float, sd: float, td: float, entry_low: float) -> None:
    """可执行性硬校验：止损价必须为正，且必须落在买点下界之下。"""
    if sd <= 0:
        raise ValueError(
            f"止损距离必须 > 0，算得 {sd}（元/股）——检查 ATR/百分比输入，"
            f"禁止返回无意义的负止损"
        )
    if td <= sd:
        raise ValueError(
            f"目标距离({td}) 必须 > 止损距离({sd})，否则 R ≤ 1，不是可执行建议"
        )
    if px - sd <= 0:
        raise ValueError(
            f"止损价算得 {px - sd} ≤ 0：止损距离({sd}) ≥ 现价({px})，输入不合理"
        )
    if px - sd >= entry_low:
        raise ValueError(
            f"止损价({px - sd}) 未低于买点下界({entry_low})："
            f"止损距离({sd}) 相对买点区间半宽太小，请放大 ATR 倍数或收窄 entry_fraction"
        )
