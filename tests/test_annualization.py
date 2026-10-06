"""日频 → 年频折算按 **252 交易日惯例**（PITFALLS P10 公式口径）。

## 口径来源

规范真源 `.github/standards/PITFALLS.md` P10 钉死公式：
`gross = spread * (252 / periods[0])` —— 全规范与既有报告数字均为 252 口径。
`engine.TRADING_DAYS` 是全仓该常数的唯一出处。
⚠️ 若改 243（A股实际年均交易日）属**全项目口径变更**，
   需单独立项重刷历史数字的可比性；本文件即该决议的守门人。

## 判据

按 UNITS「找已知真值反推」：给定持仓/权重/收益，年化波动与平均年成本的
算式逐步写在用例注释里（可复核、可重算）；
另有反向断言钉住「退回 243 必须失败」、「常数被改必须失败」。
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
from factor_lab.analysis.engine import TRADING_DAYS, simulate_matrix  # noqa: E402
from factor_lab.analysis.spec import PortfolioSpec  # noqa: E402

DATES = pd.date_range("2023-01-02", periods=3, freq="B")


class TestDailyToYearScale:
    def test_年化波动按252交易日折算(self):
        """日收益 [0.01, 0, 0] 的已知真值（算式可复核）：

        均值        = 0.01/3                             ≈ 0.0033333
        Σ(x−mean)²  = 0.0066667² + 2×0.0033333²          = 6.66667e-5
        std(ddof=1) = √(6.66667e-5 / 2)                  ≈ 0.0057735
        年化波动     = std × √252 ≈ 0.0057735 × 15.87451  ≈ 0.091652
        """
        fwd = pd.DataFrame({"sh600000": [0.01, 0.0, 0.0]}, index=DATES)
        res = simulate_matrix(
            DATES, np.array([[0], [0], [0]]), np.array([[1.0], [1.0], [1.0]]),
            fwd, CostModel(), PortfolioSpec(name="t", n_hold=1))
        assert res["ok"], res.get("reason")
        daily_std = float(np.std([0.01, 0.0, 0.0], ddof=1))
        assert res["年化波动"] == pytest.approx(
            daily_std * math.sqrt(252), rel=1e-9)
        assert res["年化波动"] != pytest.approx(
            daily_std * math.sqrt(243), rel=1e-3), (
            "折算被改成 243 —— 全项目口径是 252（PITFALLS P10 公式口径），"
            "改 243 需单独立项重刷历史数字可比性，不能悄悄改")

    def test_平均年成本按252交易日折算(self):
        """一次调仓换手 L1=2 的已知真值（算式可复核）：

        CostModel 默认：commission=0.00025、transfer_fee=0.00001、
          slippage=0.001 ⇒ buy = 0.00126；stamp_duty=0.0005 ⇒ sell = 0.00176
        单边均值 (buy+sell)/2         = 0.00151
        持仓列 0→1（卖 1 买 1）L1 = 2 ⇒ 日成本 = 2×0.00151 = 0.00302
        3 行均值 = 0.00302/3           ≈ 0.00100667
        平均年成本 = 0.00100667 × 252  ≈ 0.255680
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
            sum(per_day) / 3 * 252, rel=1e-9)
        assert res["平均年成本"] != pytest.approx(
            sum(per_day) / 3 * 243, rel=1e-3), (
            "折算被改成 243 —— 全项目口径是 252（PITFALLS P10 公式口径），"
            "改 243 需单独立项重刷历史数字可比性，不能悄悄改")

    def test_折算常数钉死252并与P10公式口径一致(self):
        """常数 = 252，与 PITFALLS P10 公式 `gross = spread * (252 / periods[0])`
        的 252 同一口径；全仓该常数的唯一出处是 engine.TRADING_DAYS。"""
        assert TRADING_DAYS == 252, (
            f"TRADING_DAYS={TRADING_DAYS} —— 全项目年化口径是 252"
            "（PITFALLS P10 公式口径），"
            "改口径需单独立项并重刷历史数字可比性")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
