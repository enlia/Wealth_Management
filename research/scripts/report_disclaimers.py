"""报告输出的口径声明文本：P14 绩效声明四要素 + AGENTS.md 第九节四条。

为什么单独成模块
----------------
`run_limit_constraint.py` 与 `run_long_only.py` 的报告输出都要附口径声明。
文本只应有一份 —— 两处各写一份必然漂移，而漂移的声明比没有声明更危险
（CODE_TRUST P23：说的和做的不一致，会让读的人按错的口径下结论）。

声明里每一句都对应代码里的一个具体位置：
  · 年化收益        → `engine._year_span`（自然日 365.25）几何年化
  · 年化波动/年成本 → `engine.TRADING_DAYS`（243，A股实测）折算
  · 「夏普」        → 年化收益 ÷ 年化波动（**非标准夏普**）
  · 成本假设        → `factor_lab.analysis.costs.CostModel`

模块级不碰数据库、不读文件 —— 只拼文本，任何环境都能导入
（CI 检查 2 的 import 链验证同样过）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.analysis.engine import TRADING_DAYS  # noqa: E402

# 项目 AGENTS.md 第九节「口径声明」的四条 —— 写报告时必须附带（原文照录）。
AGENTS_CALIBER = (
    "1. 「多空收益」是**多空组合**口径（做多高分位 + 做空低分位）。"
    "A 股散户无法做空，实际只能拿多头部分，且要再扣自己那一半成本。",
    "2. 股票池基于**当前**数据库构建，无法回溯已退市/暂停上市股票 → 幸存者偏差。",
    "3. 市值中性化用的 shares 来自**当前快照**，不是历史股本 → 不无偏。",
    "4. 检验多个因子时**未做多重比较校正**，IC 最大值会系统性偏高。",
)


def perf_declaration(portfolio: str, cost_desc: str, benchmarks: str) -> str:
    """绩效声明（P14 四要素：年化口径 / 成本假设 / 基准 / 多空口径）。

    参数
    ----
    portfolio  : 多空口径一句话（如「单边多头组合（非多空）」）
    cost_desc  : 成本假设一句话（如 `CostModel.describe()` 的输出）
    benchmarks : 基准一句话（Q18：不得只报单一基准）
    """
    return "\n".join([
        "绩效声明（P14 四要素，缺一不可）：",
        f"  ① 年化口径：年化收益 = 自然日几何年化（365.25 天/年，"
        f"engine._year_span，策略与基准同一定义）；"
        f"年化波动/年成本按日频 × {TRADING_DAYS} 交易日/年折算"
        "（A股实测，非美股 252）；"
        "「夏普」= 年化收益 ÷ 年化波动，**非标准夏普**"
        "（标准定义是超额均值/波动）。",
        f"  ② 成本假设：{cost_desc}；换手成本按日从收益中扣减进净值。",
        f"  ③ 基准：{benchmarks}；超额 = 策略年化 − 基准年化。",
        f"  ④ 多空口径：{portfolio}。",
    ])


def agents_caliber() -> str:
    """项目 AGENTS.md 第九节的四条口径声明，报告输出必须附带。"""
    return "\n".join(("口径声明（AGENTS.md 九）：",) + AGENTS_CALIBER)
