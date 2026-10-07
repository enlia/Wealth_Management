"""期望值闸门 + 分数凯利仓位（López de Prado 论文公式自实现）。

为什么自实现而不复用 mlfinlab
------------------------------
2026-10-13 实测 `find_spec('mlfinlab')` = None，本机**不可用**
（商用许可受限、未安装；见 AGENTS.md 一·1 勘误与 `pyproject.toml:38-40`）。
且 mlfinlab 的 ``bet_sizing`` 是「按预测概率缩放注码」的
``bet_size_probability`` / ``bet_size_budget`` 族，**不含期望值闸门**。
故按论文公式自实现（公式不受版权保护，不复制 GPL 源码）。

公式与量纲（全部为「每 1 元投入」的比率，UNITS 同纲才能相加）
--------------------------------------------------------------
``EV_gross = p·R − (1−p)``
``EV_net   = p·R − (1−p) − c``      ← 闸门用这个，``EV_net ≤ 0`` 即否决
``c        = 该笔买卖往返成本 ÷ 该笔投入金额``（费率口径，无量纲）

其中 ``R`` 是**赔率**（盈亏比）＝ 目标距离 ÷ 止损距离。A 股无法做空，
每次只赌「赢则 +R 元、输则 −1 元」，故 R 同时就是凯利公式里的 ``b``：

``f* = (p·R − (1−p)) / R = p − (1−p)/R``   （全凯利）

⚠️ **凯利前提不满足**：Kelly 假设 IID，A 股收益非 IID（LTCM 教训，AGENTS.md 八节）。
故默认乘收缩系数 ``shrinkage=0.25``（分数凯利），上限由 `sizing` 的三道硬约束兜底。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["EvGate", "evaluate_ev", "fractional_kelly", "full_kelly"]


def _as_float(value, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} 无法解析为数字：{value!r}") from e


def _prob(value, name: str) -> float:
    v = _as_float(value, name)
    if math.isnan(v):
        raise ValueError(f"{name} 是 NaN —— 概率必须有实测/假设来源，不允许缺失")
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} 必须在 [0,1] 内，得到 {v}")
    return v


@dataclass(frozen=True)
class EvGate:
    """一次买卖的期望值体检结果。

    ``approved`` 为 False 时 ``reason`` 必须非空（否决理由要能直接打印给人看）；
    ``approved`` 为 True 时 ``reason`` 必须为空，附带说明走 ``note``
    （避免把「通过」和「否决」两类文字混在一个字段里 —— P23「声称≠事实」同族）。
    """

    p: float                # 胜率（0~1）
    r: float                # 赔率 R = 目标距离/止损距离（>1）
    cost_ratio: float       # c = 往返成本 ÷ 投入金额（每单位投入，无量纲）
    ev_gross: float         # p·R − (1−p)
    ev_net: float           # p·R − (1−p) − c
    approved: bool
    reason: str             # 否决理由（approved=False 时必填）
    note: str = ""          # 通过时的附加说明（approved=True 时可为非空）

    def __post_init__(self) -> None:
        if self.approved and self.reason:
            raise ValueError(f"已通过的闸门不该带否决理由：{self.reason!r}")
        if not self.approved and not self.reason:
            raise ValueError(
                "被否决的闸门必须给出理由（ENGINEERING 第四节：禁止静默否决/静默通过）"
            )
        if not self.approved and self.note:
            raise ValueError(f"被否决的闸门不该带通过说明：{self.note!r}")


def evaluate_ev(p: float, r: float, cost_ratio: float) -> EvGate:
    """期望值闸门：``EV_net = p·R − (1−p) − c``，``EV_net ≤ 0`` 直接否决。

    参数
    ----
    p : 胜率（0~1）。**必须来自调用方**（本模块不产出 p）。
    r : 赔率 = 目标距离 ÷ 止损距离，必须 > 1（否则 R ≤ 1 无意义）。
    cost_ratio : c = 往返成本 ÷ 投入金额。默认本项目 ``CostModel.round_trip`` = 0.0015
        （双边 15bp，见 `factor_lab.config.DEFAULT_COST`）。

    边界情形**显式报错或返回未通过并说明**，不做静默兜底：
    - ``p`` 不在 [0,1] → 抛 ValueError
    - ``r ≤ 1`` → 抛 ValueError（R ≤ 1 的交易不该进入闸门）
    - ``cost_ratio < 0`` → 抛 ValueError
    - ``p = 0`` → 未通过，理由「p=0：无胜率」
    - ``p = 1`` 且 ``r > 1`` → 通过，但理由里写明「p=1 属退化假设」
    - ``cost_ratio ≥ 1`` → 未通过（成本已 ≥ 全部投入）
    """
    pp = _prob(p, "p")
    rr = _as_float(r, "r")
    if math.isnan(rr) or math.isinf(rr):
        raise ValueError(f"r 必须是有限数，得到 {r}")
    if rr <= 1.0:
        raise ValueError(
            f"r 必须 > 1（赔率=目标距离/止损距离），得到 {r}；"
            f"R ≤ 1 的交易不应进入闸门（应先由 build_levels 拦下）"
        )
    cc = _as_float(cost_ratio, "cost_ratio")
    if math.isnan(cc) or math.isinf(cc):
        raise ValueError(f"cost_ratio 必须是有限数，得到 {cost_ratio}")
    if cc < 0:
        raise ValueError(
            f"cost_ratio 不能为负，得到 {cc}；负成本=凭空返佣，需先核实费率口径"
        )

    ev_gross = pp * rr - (1.0 - pp)
    ev_net = ev_gross - cc

    if pp == 0.0:
        return EvGate(pp, rr, cc, ev_gross, ev_net, False,
                      "p=0：无胜率，期望净 EV 必为负（成本与败率双重拖累）")
    if cc >= 1.0:
        return EvGate(pp, rr, cc, ev_gross, ev_net, False,
                      f"c={cc:.6f} ≥ 1：往返成本已吃掉全部投入，费率口径需复核")
    if ev_net <= 0.0:
        need = (1.0 - pp + cc) / rr
        return EvGate(
            pp, rr, cc, ev_gross, ev_net, False,
            f"EV_net={ev_net:.6f} ≤ 0：毛期望 {ev_gross:.6f} 不足以覆盖成本 c={cc:.6f}；"
            f"同等赔率下 p 需 ≥ {need:.4f} 才转正（当前 {pp:.4f}，差 {need - pp:+.4f}）",
        )
    note = ""
    if pp >= 1.0:
        note = "p=1 属退化假设（真实市场不存在），仅当 p 有实测来源时才可信"
    return EvGate(pp, rr, cc, ev_gross, ev_net, True, "", note)


def full_kelly(p: float, r: float) -> float:
    """全凯利仓位比例：``f* = (p·R − (1−p)) / R = p − (1−p)/R``。

    返回**占投入资金的比例**（无量纲）。可为负 —— 负值含义是「该方向期望为负，
    不该下注（反向才对）」；调用方（`sizing`）必须显式处理，不得 clip 成 0 后继续。
    """
    pp = _prob(p, "p")
    rr = _as_float(r, "r")
    if math.isnan(rr) or math.isinf(rr) or rr <= 0:
        raise ValueError(f"r 必须为有限正数，得到 {r}")
    return (pp * rr - (1.0 - pp)) / rr


def fractional_kelly(p: float, r: float, shrinkage: float) -> float:
    """分数凯利：``shrinkage × full_kelly(p, R)``。

    ⚠️ ``shrinkage=1.0`` 时**退化为全凯利**（对拍手算例子见
    ``tests/test_risk_budget.py::test_shrinkage_one_equals_full_kelly``）。

    ``shrinkage`` 必须在 ``(0, 1]``：=0 等于不给仓位（无意义），>1 是加杠杆（禁止）。
    """
    s = _as_float(shrinkage, "shrinkage")
    if math.isnan(s):
        raise ValueError("shrinkage 是 NaN")
    if not 0.0 < s <= 1.0:
        raise ValueError(
            f"shrinkage 必须在 (0,1] 内，得到 {s}；"
            f"=0 表示不下注（无意义），>1 表示加杠杆（本项目禁止）"
        )
    return s * full_kelly(p, r)
