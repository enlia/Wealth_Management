"""建议卡：把闸门/仓位/价位/行业四块拼成一张**可执行**的卡片。

卡片字段（缺任一即视为不合格 —— AGENTS.md ★目标第 3、4 条）
------------------------------------------------------------
标的代码 / 名称 / 买点区间 / 目标位 / 止损位 / 建议仓位% /
该票所属板块及其合计仓位% / 期望净 EV / 依据的样本量与口径三件套 /
「不构成投资建议」。

量纲自查（UNITS）
-----------------
- 价格类字段（买点/目标/止损）单位 **元/股**，未复权报价口径。
- 仓位类字段（仓位%/行业合计%）单位 **占账户总本金百分比**（``23.5`` 表示 23.5%）。
- ``EV`` 单位 **每 1 元投入的比率**（0.042 表示每投入 1 元期望净赚 0.042 元）；
  ``ev_net_account`` 才是**占账户**口径 = ``ev_net × 仓位``（★目标第 4 条要求
  改用「占账户」表述时必须乘仓位）。两者**禁止混用**，卡片分行打印并各带单位。
- ``R`` 无量纲（赔率）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .levels import Levels

__all__ = ["DISCLAIMER", "AdviceCard", "build_card", "render_card"]

DISCLAIMER = "不构成投资建议"

# 口径三件套（AGENTS.md 九·7 要求任何收益/换手/成本引用带此三件）
PRICE_CALIBER = "未复权报价（挂单价口径；止损/目标为交易所约束的报价，与复权无关）"
TURNOVER_UNIT = "不涉及换手（本卡为单笔交易建议，不含换手率统计）"
ANNUALIZATION = "不做年化（本卡输出单笔期望 EV 与仓位，非年化收益）"

_REQUIRED_TEXT = (
    "code", "name", "industry_l1", "price_caliber", "turnover_unit",
    "annualization", "sample_note", "p_source", "r_source",
    "disclaimer", "position_reason",
)

_PCT_LIMIT = 100.0


@dataclass(frozen=True)
class AdviceCard:
    """一张建议卡。所有字段都是**必填**，缺一个建议卡就不该被产出。"""

    # ── 标的与价位 ────────────────────────────────────────────
    code: str
    name: str
    entry_low: float          # 元/股
    entry_high: float         # 元/股
    target: float             # 元/股
    stop: float               # 元/股
    price: float              # 参考现价，元/股

    # ── 仓位与集中度（占账户总本金 %）────────────────────────
    position_pct: float
    industry_l1: str
    industry_pct: float       # 该行业**合计**仓位（含已有持仓）
    max_position_pct: float
    max_industry_pct: float

    # ── 期望值与依据 ─────────────────────────────────────────
    p: float
    r: float
    cost_ratio: float
    ev_gross: float
    ev_net: float
    sample_n: int
    sample_note: str
    p_source: str
    r_source: str

    # ── 口径三件套 + 披露 ────────────────────────────────────
    price_caliber: str
    turnover_unit: str
    annualization: str
    disclaimer: str
    position_reason: str

    def __post_init__(self) -> None:
        missing = [
            f for f in _REQUIRED_TEXT
            if not isinstance(getattr(self, f), str) or not getattr(self, f).strip()
        ]
        if missing:
            raise ValueError(
                f"{self.code}: 建议卡缺必填文字字段 {missing} ——"
                f"建议卡字段完整性是硬要求（★目标三要素），缺任一即不合格"
            )
        if not self.stop < self.entry_low:
            raise ValueError(
                f"{self.code}: 止损({self.stop}) 必须 < 买点下界({self.entry_low})"
            )
        if not self.entry_high < self.target:
            raise ValueError(
                f"{self.code}: 目标({self.target}) 必须 > 买点上界({self.entry_high})"
            )
        if self.position_pct < 0:
            raise ValueError(f"{self.code}: 仓位不能为负：{self.position_pct}")
        if self.position_pct > self.max_position_pct + 1e-9:
            raise ValueError(
                f"{self.code}: 单票仓位 {self.position_pct:.4f}% 超过上限 "
                f"{self.max_position_pct:.4f}%（三要素之一失守）"
            )
        if self.industry_pct > self.max_industry_pct + 1e-9:
            raise ValueError(
                f"{self.code}: 行业「{self.industry_l1}」合计 {self.industry_pct:.4f}% 超过上限 "
                f"{self.max_industry_pct:.4f}%（三要素之一失守）"
            )
        if self.industry_pct + 1e-9 < self.position_pct:
            raise ValueError(
                f"{self.code}: 行业合计 {self.industry_pct:.4f}% 小于本票仓位 "
                f"{self.position_pct:.4f}% —— 口径错误（行业合计必须包含本票）"
            )
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"{self.code}: p 必须在 [0,1]，得到 {self.p}")
        if not self.r > 1.0:
            raise ValueError(f"{self.code}: R 必须 > 1，得到 {self.r}")
        if self.sample_n <= 0:
            raise ValueError(
                f"{self.code}: 样本量必须为正整数，得到 {self.sample_n} ——"
                f"「无样本」的建议不允许出卡"
            )
        if abs(self.ev_net - (self.ev_gross - self.cost_ratio)) > 1e-9:
            raise ValueError(
                f"{self.code}: EV 不自洽：ev_net={self.ev_net} 而 ev_gross-c="
                f"{self.ev_gross - self.cost_ratio}"
            )
        if self.ev_net <= 0:
            raise ValueError(
                f"{self.code}: ev_net={self.ev_net} ≤ 0 却出了建议卡 ——"
                f"期望值闸门的作用就是拦下这种交易（核心价值，禁止绕过）"
            )
        if self.disclaimer != DISCLAIMER:
            raise ValueError(
                f"{self.code}: 风险披露必须恰为「{DISCLAIMER}」，得到 {self.disclaimer!r}"
            )

    # ── 派生量（占账户口径，★目标第 4 条）────────────────────
    @property
    def position_frac(self) -> float:
        return self.position_pct / _PCT_LIMIT

    @property
    def ev_net_account(self) -> float:
        """期望净 EV 的**占账户**口径 = ``ev_net × 仓位``（每 1 元账户本金）。"""
        return self.ev_net * self.position_frac

    @property
    def risk_amount_pct(self) -> float:
        """单笔最大亏损（占账户 %）= 仓位 × 止损距离 ÷ 现价。"""
        return self.position_pct * (self.price - self.stop) / self.price

    def lines(self) -> list[str]:
        """渲染成文本行（供 CLI / 报告直接打印）。"""
        star = "" if self.p_source.lower().startswith(("measured", "实测")) else "  ⚠️"
        return [
            "┌─ 建议卡 " + "─" * 62,
            f"│ 标的代码      {self.code}   名称  {self.name}",
            f"│ 买点区间      {self.entry_low:,.2f} ~ {self.entry_high:,.2f} 元"
            f"   （参考现价 {self.price:,.2f} 元）",
            f"│ 目标位        {self.target:,.2f} 元"
            f"   （+{self.target / self.price - 1:.2%}，{(self.target - self.price):,.2f} 元/股）",
            f"│ 止损位        {self.stop:,.2f} 元"
            f"   （{self.stop / self.price - 1:.2%}，跌破即离场）",
            f"│ 建议仓位      {self.position_pct:.4f}% 账户"
            f"   （上限 {self.max_position_pct:.2f}%；理由：{self.position_reason}）",
            f"│ 板块          {self.industry_l1}"
            f"   该板块合计 {self.industry_pct:.4f}%（上限 {self.max_industry_pct:.2f}%）",
            f"│ 期望净 EV     {self.ev_net:+.6f} /元投入"
            f"   = p·R − (1−p) − c = {self.p:.4f}×{self.r:.3f}"
            f" − {1 - self.p:.4f} − {self.cost_ratio:.6f}",
            f"│               占账户口径 {self.ev_net_account:+.6f} /元账户"
            f"（= EV × {self.position_pct:.4f}% 仓位）",
            f"│ 单笔最大亏损  {self.risk_amount_pct:.4f}% 账户（仓位 × 止损距离/现价）",
            f"│ 样本量        n={self.sample_n}   {self.sample_note}",
            f"│ p 来源        {self.p_source}{star}",
            f"│ R 来源        {self.r_source}{star}",
            f"│ 口径三件套    ①价格口径：{self.price_caliber}",
            f"│               ②换手单位：{self.turnover_unit}",
            f"│               ③年化方式：{self.annualization}",
            f"└─ ⚠️ {self.disclaimer}｜历史统计不预示未来，决策与后果自担 " + "─" * 12,
        ]


def build_card(
    *,
    code: str,
    name: str,
    levels: Levels,
    position_pct: float,
    industry_l1: str,
    industry_pct: float,
    max_position_pct: float,
    max_industry_pct: float,
    p: float,
    r: float,
    cost_ratio: float,
    sample_n: int,
    sample_note: str,
    p_source: str,
    r_source: str,
    position_reason: str,
    ev_gross: float | None = None,
) -> AdviceCard:
    """组装建议卡（所有量纲转换集中在此，禁止调用方各自换算）。

    ``levels.r_ratio`` 与传入的 ``r`` 必须一致（容差 1e-6）——
    价位的赔率与 EV 用的赔率若不是同一个数，卡片就是自相矛盾的。
    """
    if not math.isclose(levels.r_ratio, r, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(
            f"{code}: 价位隐含赔率 R={levels.r_ratio:.6f} 与 EV 用的 R={r:.6f} 不一致 ——"
            f"禁止用不同赔率拼一张卡（价位与 EV 必须同源）"
        )
    g = p * r - (1.0 - p) if ev_gross is None else float(ev_gross)
    net = g - cost_ratio
    if net <= 0:
        raise ValueError(
            f"{code}: ev_net={net:.6f} ≤ 0 却要出建议卡 ——"
            f"期望值闸门的作用就是拦下这种交易（本模块核心价值，禁止绕过）；"
            f"请先在 evaluate_ev() 得到 approved=True 再出卡"
        )
    return AdviceCard(
        code=code, name=name,
        entry_low=levels.entry_low, entry_high=levels.entry_high,
        target=levels.target, stop=levels.stop, price=levels.price,
        position_pct=position_pct, industry_l1=industry_l1, industry_pct=industry_pct,
        max_position_pct=max_position_pct, max_industry_pct=max_industry_pct,
        p=p, r=r, cost_ratio=cost_ratio, ev_gross=g, ev_net=net,
        sample_n=sample_n, sample_note=sample_note, p_source=p_source, r_source=r_source,
        price_caliber=PRICE_CALIBER, turnover_unit=TURNOVER_UNIT,
        annualization=ANNUALIZATION, disclaimer=DISCLAIMER,
        position_reason=position_reason,
    )


def render_card(card: AdviceCard) -> str:
    """渲染成一段可直接粘贴的文本。"""
    return "\n".join(card.lines())
