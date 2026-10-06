"""报告输出的口径声明必须完整：P14 四要素 + AGENTS.md 九的四条。

## 为什么要钉

绩效数字缺口径声明就没法读：同一个「年化 +2%」，
多空口径与多头口径、有基准超额与无基准绝对值，含义完全不同
（P14 实测：同因子多空 −14.32%、多头 +11.74%，方向相反；
 幸存者偏差/多重比较不声明则结论不可引用）。

声明是**对外契约**：缺任一要素、缺任一条即失败，
防止后续改动悄悄删减声明文本。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from report_disclaimers import (  # noqa: E402
    AGENTS_CALIBER,
    agents_caliber,
    perf_declaration,
)

from factor_lab.analysis.costs import CostModel  # noqa: E402


class TestPerfDeclaration:
    def test_绩效声明必含P14四要素(self):
        text = perf_declaration(
            portfolio="单边多头组合（非多空）",
            cost_desc=CostModel().describe(),
            benchmarks="等权全市场 + 沪深300")
        for key in ("① 年化口径", "② 成本假设", "③ 基准", "④ 多空口径"):
            assert f"{key}：" in text, f"绩效声明缺 P14 要素：{key}"

    def test_年化成本与夏普定义必须写明(self):
        """三个口径都要落在声明里：365.25（自然日年数）、
        「年化按 252 交易日惯例（PITFALLS P10 公式口径）」
        + 改 243 需单独立项的边界说明、以及「非标准夏普」的定性说明。"""
        text = perf_declaration(portfolio="p", cost_desc="c", benchmarks="b")
        assert "365.25" in text
        assert "年化按 252 交易日惯例（PITFALLS P10 公式口径）" in text
        assert "需单独立项" in text
        assert "非标准夏普" in text


class TestAgentsCaliber:
    def test_口径声明必含AGENTS第九节四条(self):
        text = agents_caliber()
        assert len(AGENTS_CALIBER) == 4
        for item in AGENTS_CALIBER:          # 条目自带序号（1.~4.）
            assert item in text, f"口径声明缺条目：{item[:20]}"
        for i in range(1, 5):
            assert f"{i}. " in text, f"口径声明缺第 {i} 条的序号"
        for key in ("做空", "幸存者偏差", "快照", "多重比较校正"):
            assert key in text, f"口径声明缺关键声明：{key}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
