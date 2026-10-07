"""仓位裁剪：分数凯利 → 三道硬上限（单票 ≤2% / 单行业 ≤20% / 总仓位 ≤100%）。

裁剪顺序（**先算 EV，再按 EV 从高到低排序，逐笔取 min(剩余额度)**）
------------------------------------------------------------------
1. 候选按 ``ev_net`` **从高到低**排序（超限时优先保 EV 高的）。
2. 每个候选依次被三道上限夹住：
   ``w = min(分数凯利, 单票剩余额度, 该行业剩余额度, 组合总剩余额度)``
3. ``w`` 低于 ``min_position`` → **丢弃并写明理由**为「低于最小建仓比例」
   （不返回 0.0001 这种挂不出去的单）。

为什么必须「显式报错或返回 0 并说明」（★目标 与 P22 要求）
------------------------------------------------------------
- 分数凯利为负 → **不 clip 成 0 悄悄放过**，而是 ``w=0`` 且 ``binding='ev_negative'``，
  理由写明「全凯利为负，方向不该下注」。**禁止**把哨兵值或 0 当成真实权重静默返回。
- 所有上限必须真实生效：``w`` 恰好等于上限（裁剪后）时 ``binding`` 记录是哪一道。
- 上限本身非法（≤0 或 >1）→ 抛 ``ValueError``，不默默用默认值。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .ev import fractional_kelly

__all__ = [
    "DEFAULT_MAX_INDUSTRY",
    "DEFAULT_MAX_POSITION",
    "DEFAULT_MIN_POSITION",
    "DEFAULT_SHRINKAGE",
    "Allocation",
    "Candidate",
    "RiskParams",
    "apply_portfolio_caps",
]

DEFAULT_SHRINKAGE = 0.25      # 分数凯利收缩系数（Kelly 假设 IID，股票收益非 IID）
DEFAULT_MAX_POSITION = 0.02   # 单票 ≤ 账户 2%
DEFAULT_MAX_INDUSTRY = 0.20   # 单申万一级行业合计 ≤ 账户 20%
DEFAULT_MIN_POSITION = 0.001  # 单票低于 0.1% 账户即丢弃（挂不出去）


def _unit_rate(value, name: str, *, upper_open: bool = True) -> float:
    """校验一个必须在 (0,1] （或 (0,1)）内的比例。"""
    try:
        v = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} 无法解析为数字：{value!r}") from e
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} 必须是有限数，得到 {value}")
    if v <= 0.0:
        raise ValueError(f"{name} 必须 > 0，得到 {v}（=0 会让该上限失效，禁止）")
    if upper_open and v > 1.0:
        raise ValueError(f"{name} 必须 ≤ 1（=总仓位），得到 {v}")
    if not upper_open and v >= 1.0:
        raise ValueError(f"{name} 必须 < 1，得到 {v}")
    return v


@dataclass(frozen=True)
class RiskParams:
    """风险预算参数。全部为**占账户总本金的比例**（无量纲）。"""

    shrinkage: float = DEFAULT_SHRINKAGE
    max_position: float = DEFAULT_MAX_POSITION
    max_industry: float = DEFAULT_MAX_INDUSTRY
    max_total: float = 1.0
    min_position: float = DEFAULT_MIN_POSITION

    def __post_init__(self) -> None:
        _unit_rate(self.shrinkage, "shrinkage")
        if not 0.0 < self.shrinkage <= 1.0:
            raise ValueError(f"shrinkage 必须在 (0,1] 内，得到 {self.shrinkage}")
        _unit_rate(self.max_position, "max_position")
        _unit_rate(self.max_industry, "max_industry")
        _unit_rate(self.max_total, "max_total")
        _unit_rate(self.min_position, "min_position")
        if self.min_position > self.max_position:
            raise ValueError(
                f"min_position({self.min_position}) 不能大于 max_position"
                f"({self.max_position})：否则永远无法建仓"
            )
        if self.max_position > self.max_total:
            raise ValueError(
                f"max_position({self.max_position}) 不能大于 max_total({self.max_total})"
            )


@dataclass(frozen=True)
class Candidate:
    """一个候选标的（p/R 已由调用方给出，本模块只做守门与裁剪）。"""

    code: str
    p: float
    r: float
    ev_net: float
    ev_gross: float
    ev_approved: bool
    industry_l1: str
    reason_prefix: str = ""   # 上游（例如闸门）带过来的理由，透传进裁剪说明


@dataclass
class Allocation:
    """裁剪结果：一只标的最终建议仓位 = 占**账户总本金**的比例。"""

    code: str
    industry_l1: str
    kelly_raw: float          # 分数凯利原始值（可为负）
    weight: float             # 最终仓位（≥0，占账户总本金）
    binding: str              # 生效的约束名
    reason: str
    ev_net: float = 0.0
    ev_gross: float = 0.0
    dropped: bool = False

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError(f"{self.code}: Allocation 必须带说明理由（禁止静默裁剪）")
        if self.dropped and self.weight != 0.0:
            raise ValueError(
                f"{self.code}: dropped=True 时权重必须恰为 0，得到 {self.weight}"
            )
        if not self.dropped and self.weight <= 0.0:
            raise ValueError(
                f"{self.code}: 未丢弃却给出非正权重 {self.weight}——"
                f"请用 dropped=True + reason 显式表达「不下注」"
            )


def _industry_cap_for(industry: str, params: RiskParams, caps: dict[str, float]) -> float:
    """取该行业的上限。``caps`` 里没有该行业时用全局 ``max_industry``。"""
    return float(caps.get(industry, params.max_industry))


def apply_portfolio_caps(
    candidates: list[Candidate],
    params: RiskParams,
    *,
    existing_industry: dict[str, float] | None = None,
    industry_caps_override: dict[str, float] | None = None,
) -> list[Allocation]:
    """按 EV 从高到低逐笔裁剪，返回全部候选的仓位结果（含被丢弃项）。

    参数
    ----
    candidates : 候选列表。``ev_approved=False`` 的候选**不会被静默忽略**，
        而是以 ``binding='ev_gate_veto'``、``weight=0`` 明确出现在结果里。
    params : 三道硬上限参数。
    existing_industry : **已有持仓**的行业仓位 ``{行业名: 占账户比例}``，
        会从该行业的剩余额度里扣掉（不扣会突破 20% 行业上限）。
    industry_caps_override : 逐行业覆盖上限（如某行业单独设 10%）。

    ⚠️ **空输入**：``candidates=[]`` 直接抛 ``ValueError``，
    禁止用空表糊弄成「没有超限」（P18：空索引不得广播成「全部合格」）。
    """
    if not isinstance(candidates, list):
        raise ValueError(f"candidates 必须是 list，得到 {type(candidates).__name__}")
    if not candidates:
        raise ValueError(
            "候选列表为空：无标的可裁剪。"
            "上游未产出任何候选时请显式报错，不要返回空表让人误以为「全部通过」（P18）"
        )
    if not isinstance(params, RiskParams):
        raise ValueError(f"params 必须是 RiskParams，得到 {type(params).__name__}")

    held = dict(existing_industry or {})
    for k, v in held.items():
        if float(v) < 0:
            raise ValueError(f"已有持仓行业 {k!r} 的仓位为负：{v}（口径错误，需核实）")

    caps = dict(industry_caps_override or {})

    # 排序：EV 从高到低；同 EV 时按 code 稳定排序，保证结果可复现
    ordered = sorted(
        enumerate(candidates),
        key=lambda t: (-t[1].ev_net, t[1].code),
    )

    ind_used: dict[str, float] = {k: float(v) for k, v in held.items()}
    total_used = sum(ind_used.values())
    if total_used > params.max_total + 1e-12:
        raise ValueError(
            f"已有持仓合计 {total_used:.6f} 已超过总仓位上限 {params.max_total}："
            f"请先减仓，本模块不做「自动砍已有持仓」（那会静默改动真实组合）"
        )

    results: list[Allocation] = []
    for _, cand in ordered:
        if not cand.ev_approved:
            results.append(Allocation(
                code=cand.code, industry_l1=cand.industry_l1, kelly_raw=float("nan"),
                weight=0.0, binding="ev_gate_veto", dropped=True,
                reason=(cand.reason_prefix or "期望值闸门否决（EV ≤ 0）"),
                ev_net=cand.ev_net, ev_gross=cand.ev_gross,
            ))
            continue

        raw = fractional_kelly(cand.p, cand.r, params.shrinkage)
        if raw <= 0.0:
            results.append(Allocation(
                code=cand.code, industry_l1=cand.industry_l1, kelly_raw=raw,
                weight=0.0, binding="kelly_nonpositive", dropped=True,
                reason=(f"分数凯利 {raw:.6f} ≤ 0："
                        f"p={cand.p:.4f}·R={cand.r:.3f} 不足以覆盖败率，方向不该下注"),
                ev_net=cand.ev_net, ev_gross=cand.ev_gross,
            ))
            continue

        ind = cand.industry_l1
        ind_cap = _industry_cap_for(ind, params, caps)
        ind_room = ind_cap - ind_used.get(ind, 0.0)
        total_room = params.max_total - total_used
        # binding 用 math.isclose 判定（浮点等值比较会错标约束名；
        # 约束名错标 = 卡片给用户的裁剪理由错，属 CODE_TRUST P23 同类「声称≠事实」）
        contributors = {
            "kelly": raw,
            "max_position": params.max_position,
            "max_industry": ind_room,
            "max_total": total_room,
        }
        w = min(contributors.values())
        binding = next(
            k for k, v in contributors.items()
            if math.isclose(v, w, rel_tol=1e-12, abs_tol=1e-15)
        )

        if w <= 0.0:
            results.append(Allocation(
                code=cand.code, industry_l1=ind, kelly_raw=raw, weight=0.0,
                binding="no_room", dropped=True,
                reason=(f"剩余额度为 {w:.6f}（行业剩余 {ind_room:.6f} / 总剩余 "
                        f"{total_room:.6f}）：该笔无法建仓"),
                ev_net=cand.ev_net, ev_gross=cand.ev_gross,
            ))
            continue

        if w < params.min_position:
            results.append(Allocation(
                code=cand.code, industry_l1=ind, kelly_raw=raw, weight=0.0,
                binding="below_min_position", dropped=True,
                reason=(f"裁剪后仓位 {w:.6f} < 最小建仓比例 {params.min_position}："
                        f"该笔挂不出去，已丢弃（原分数凯利 {raw:.6f}）"),
                ev_net=cand.ev_net, ev_gross=cand.ev_gross,
            ))
            continue

        note = {
            "kelly": f"分数凯利 {raw:.6f} 未被上限触及",
            "max_position": f"触及单票上限 {params.max_position:.4f}（原分数凯利 {raw:.6f}）",
            "max_industry": (f"触及行业「{ind}」上限 {ind_cap:.4f}"
                             f"（已用 {ind_used.get(ind, 0.0):.4f}，剩余 {ind_room:.6f}）"),
            "max_total": f"触及总仓位上限 {params.max_total:.4f}（剩余 {total_room:.6f}）",
        }[binding]
        results.append(Allocation(
            code=cand.code, industry_l1=ind, kelly_raw=raw, weight=w,
            binding=binding, reason=note,
            ev_net=cand.ev_net, ev_gross=cand.ev_gross,
        ))
        ind_used[ind] = ind_used.get(ind, 0.0) + w
        total_used += w

    # 后置不变量：裁剪结果绝不能越过任何一道上限（自检，防实现漂移）
    _assert_caps_hold(results, params, held, caps)
    return results


def _assert_caps_hold(
    results: list[Allocation],
    params: RiskParams,
    held: dict[str, float],
    caps: dict[str, float],
) -> None:
    """独立自检：任何一条越限都抛错（不靠调用方信任）。"""
    total = 0.0
    per_ind: dict[str, float] = dict(held)
    for a in results:
        if a.dropped:
            continue
        if a.weight > params.max_position + 1e-12:
            raise RuntimeError(
                f"内部缺陷：{a.code} 权重 {a.weight} 超过单票上限 {params.max_position}"
            )
        total += a.weight
        per_ind[a.industry_l1] = per_ind.get(a.industry_l1, 0.0) + a.weight
    if total > params.max_total + 1e-12:
        raise RuntimeError(f"内部缺陷：合计仓位 {total} 超过总上限 {params.max_total}")
    for ind, w in per_ind.items():
        cap = _industry_cap_for(ind, params, caps)
        if w > cap + 1e-12:
            raise RuntimeError(f"内部缺陷：行业「{ind}」合计 {w} 超过上限 {cap}")


@dataclass
class PortfolioPlan:
    """组合级裁剪结果（供卡片/报告消费）。"""

    allocations: list[Allocation] = field(default_factory=list)

    def total_weight(self) -> float:
        return sum(a.weight for a in self.allocations if not a.dropped)

    def industry_totals(self, held: dict[str, float] | None = None) -> dict[str, float]:
        out = {k: float(v) for k, v in (held or {}).items()}
        for a in self.allocations:
            if not a.dropped:
                out[a.industry_l1] = out.get(a.industry_l1, 0.0) + a.weight
        return out
