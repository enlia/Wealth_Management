"""风险预算模块：把「任意信号排序」变成「带风控的可执行买卖建议」。

设计边界（**必读，防止误用**）
------------------------------
本模块**不产出 p（胜率）与 R（赔率）**。本项目实测 11 个价量因子扣 20bp 后
多空净收益为正 **0/11**（`docs/03_项目报告/12_价量因子有效性检验报告.md`），
即当前**没有可用选股信号**。所以本模块的价值是：
**当你已经有 p 与 R（来自文献、样本外回测、或其他信号）时，它当守门员与风险控制**——
- 期望值闸门：``EV = p·R − (1−p) − c ≤ 0`` 时**直接否决**并打印理由（否决权，本模块核心价值）
- 仓位：分数凯利 + 三道硬上限（单票 / 单行业 / 总仓位）
- 止损/目标位：ATR 倍数或用户给定百分比，输出具体价格
- 板块集中度：复用库内申万行业映射

调用方若**没有** p/R，应显式传 ``--p-source``/``--r-source example``，
卡片会打上「示例假设值，非实测」标记 —— 禁止伪造实测口径。

子模块
------
- ``levels``    止损/目标/买点区间（ATR 或百分比）
- ``ev``        期望值闸门 + 分数凯利
- ``sizing``    三道硬上限裁剪
- ``industry``  申万行业映射与集中度
- ``market``    库内行情读取（ATR 等）
- ``card``      建议卡组装与渲染

三件套管径（UNITS/AGENTS 九·7）
------------------------------
价格口径 = **未复权报价**（止损/目标是挂单价，交易所约束的是报价）；
换手单位 = **不涉及**（本模块不做换手率统计）；
年化方式 = **不做年化**（本模块输出的是单笔交易的 EV 与仓位，不是年化收益）。
"""
from __future__ import annotations

from .card import DISCLAIMER, AdviceCard, build_card, render_card
from .ev import (
    EvGate,
    evaluate_ev,
    fractional_kelly,
    full_kelly,
)
from .industry import IndustryMap, IndustryUsage, industry_caps
from .levels import AtrLevels, PercentLevels, build_levels
from .market import atr_sma, atr_wilder, load_atr
from .sizing import (
    DEFAULT_MAX_INDUSTRY,
    DEFAULT_MAX_POSITION,
    DEFAULT_MIN_POSITION,
    DEFAULT_SHRINKAGE,
    Allocation,
    Candidate,
    RiskParams,
    apply_portfolio_caps,
)

__all__ = [
    "DISCLAIMER",
    "DEFAULT_MAX_INDUSTRY",
    "DEFAULT_MAX_POSITION",
    "DEFAULT_MIN_POSITION",
    "DEFAULT_SHRINKAGE",
    "AdviceCard",
    "Allocation",
    "AtrLevels",
    "Candidate",
    "EvGate",
    "IndustryMap",
    "IndustryUsage",
    "PercentLevels",
    "RiskParams",
    "apply_portfolio_caps",
    "atr_sma",
    "atr_wilder",
    "build_card",
    "build_levels",
    "evaluate_ev",
    "fractional_kelly",
    "full_kelly",
    "industry_caps",
    "load_atr",
    "render_card",
]
