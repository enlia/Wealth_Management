"""日频 → 年频折算统一按 **243 交易日**（A股实测），不再是 252。

## 为什么要钉

`engine.simulate_matrix` 的年化波动、平均年成本曾按 252（美股口径）折算，
而同文件注释、`_year_span` 兜底分支都认定 A 股一年实际约 243 个交易日 ——
同一份数据两套折算，年频量系统性差约 3.5%，且不报错（P10 尾巴）。

## 判据

按 UNITS 的「找已知真值反推」：手工给定持仓/权重/收益，
把年化波动与年成本的**已知真值**写死在断言里，
并加「还原旧版（×252）必须失败」的反向断言 —— 防止悄悄退回 252。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.analysis.costs import CostModel  # noqa: E402
from factor_lab.analysis.engine import simulate_matrix  # noqa: E402
from factor_lab.analysis.spec import PortfolioSpec  # noqa: E402

DATES = pd.date_range("2023-01-02", periods=3, freq="B")


class TestDailyToYearScale:
    def test_年化波动按243交易日折算_不是252(self):
        """日收益 [0.01, 0, 0] 的已知真值：std(ddof=1)×√243。"""
        fwd = pd.DataFrame({"sh600000": [0.01, 0.0, 0.0]}, index=DATES)
        res = simulate_matrix(
            DATES, np.array([[0], [0], [0]]), np.array([[1.0], [1.0], [1.0]]),
            fwd, CostModel(), PortfolioSpec(name="t", n_hold=1))
        assert res["ok"], res.get("reason")
        daily_std = float(np.std([0.01, 0.0, 0.0], ddof=1))
        assert res["年化波动"] == pytest.approx(
            daily_std * math.sqrt(243), rel=1e-9)
        assert res["年化波动"] != pytest.approx(
            daily_std * math.sqrt(252), rel=1e-3), (
            "又退回 252 折算了 —— A 股一年约 243 个交易日，见模块注释")

    def test_平均年成本按243交易日折算_不是252(self):
        """一次调仓换手 L1=2 的已知真值：turn×(buy+sell)/2 按 243 年化。

        持仓第 1 天从列 0 换到列 1：全股票空间 L1 差 = 2（卖 1 买 1）。
        """
        fwd = pd.DataFrame({"sh600000": [0.0, 0.0, 0.0],
                            "sz000001": [0.0, 0.0, 0.0]}, index=DATES)
        res = simulate_matrix(
            DATES, np.array([[0], [1], [1]]), np.array([[1.0], [1.0], [1.0]]),
            fwd, CostModel(), PortfolioSpec(name="t", n_hold=1))
        assert res["ok"], res.get("reason")
        cost = CostModel()
        per_day = [0.0, 2.0 * (cost.buy + cost.sell) / 2, 0.0]
        assert res["平均年成本"] == pytest.approx(
            sum(per_day) / 3 * 243, rel=1e-9)
        assert res["平均年成本"] != pytest.approx(
            sum(per_day) / 3 * 252, rel=1e-3), (
            "又退回 252 折算了 —— A 股一年约 243 个交易日，见模块注释")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
