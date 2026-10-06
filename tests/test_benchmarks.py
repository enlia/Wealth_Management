"""多基准对比（compare_benchmarks.py）的判据测试。

核心判据：**价格序列与日收益序列的年化方式不能混用**。
实测踩过：统一用 `(1+r).prod()` 对指数价格序列算出 `inf%`。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from compare_benchmarks import (  # noqa: E402
    annual_return,
    annual_return_from_r,
)


class TestAnnualReturn:
    def test_价格序列用首尾比(self):
        """⚠️ **价格序列不能复利**（实测踩坑）：
        指数 close 已从 1000 涨到 5000（内部已复利 5 倍），
        再 `(1+r).prod()` 会溢出成 inf。
        正确：末日 / 首日 − 1。

        ⚠️ 250 个**交易日**跨约 347 **自然日** = 0.95 年，
        不是 1.0 年 —— 断言值必须按真实跨度算，不能想当然。"""
        idx = pd.bdate_range("2023-01-02", periods=250)
        prices = pd.Series(np.linspace(1000, 5000, len(idx)), index=idx)
        v = annual_return(prices)
        assert np.isfinite(v), f"价格序列年化溢出成 {v} —— 用错了公式"
        years = (idx[-1] - idx[0]).days / 365.25
        expect = 5 ** (1 / years) - 1
        assert abs(v - expect) < 1e-9, f"年化 {v:.4%} 应为 {expect:.4%}"
        assert 4.0 < v < 5.0, f"年化 {v:.2%} 量级不合理"

    def test_日收益序列用复利(self):
        """日收益序列必须 `(1+r).prod()` 再开方。"""
        idx = pd.bdate_range("2023-01-02", periods=250)
        r = pd.Series(0.0004, index=idx)      # 日均 +4bp
        v = annual_return_from_r(r)
        assert 0.09 < v < 0.12, f"年化 {v:.2%}（应约 10.7%）"

    def test_两者对同一序列结果相近但不相同(self):
        """把价格序列转成日收益后，两者应接近 ——
        若相差很大说明其中一个用错了公式。"""
        idx = pd.bdate_range("2023-01-02", periods=250)
        prices = pd.Series(np.linspace(1000, 1100, len(idx)), index=idx)
        r = prices.pct_change().dropna()
        a, b = annual_return(prices), annual_return_from_r(r)
        assert abs(a - b) < 0.01, f"价格 {a:.4%} vs 收益 {b:.4%} 差太多"

    def test_下跌序列返回负(self):
        idx = pd.bdate_range("2023-01-02", periods=250)
        prices = pd.Series(np.linspace(5000, 1000, len(idx)), index=idx)
        v = annual_return(prices)
        # 5000→1000 是 −80%，跨 0.95 年 ⇒ 年化约 −81.6%
        assert -0.95 < v < -0.70, f"年化 {v:.2%} 应在 −70% ~ −95%"

    def test_空序列返回nan而非崩溃(self):
        idx = pd.bdate_range("2023-01-02", periods=0)
        assert np.isnan(annual_return(pd.Series(dtype=float, index=idx)))
        assert np.isnan(annual_return_from_r(pd.Series(dtype=float, index=idx)))

    def test_首价为零返回nan而非无穷(self):
        """首价 0 会导致除零 → inf。
        必须返回 nan（不可计算），不能返回 inf
        —— inf 混进统计会污染整列。"""
        idx = pd.bdate_range("2023-01-02", periods=10)
        prices = pd.Series([0.0] + [100.0] * 9, index=idx)
        v = annual_return(prices)
        assert np.isnan(v), f"首价 0 应返回 nan，实得 {v}"


class TestRealBenchmarkData:
    """用真实数据验证基准差异确实很大（Q18 的核心事实）。"""

    @pytest.fixture(scope="class")
    def bench_csv(self):
        p = ROOT / "runtime" / "benchmarks.csv"
        if not p.exists():
            pytest.skip("尚未跑 compare_benchmarks.py")
        return pd.read_csv(p, encoding="utf-8-sig", index_col=0)

    def test_基准间存在显著差异(self, bench_csv):
        """⚠️ **「跑输基准」必须说清是哪个基准**。
        实测各年最优基准与最差基准中位差 28.72%、最大 49.50%。"""
        spread = bench_csv.max(axis=1) - bench_csv.min(axis=1)
        assert spread.median() > 0.15, (
            f"基准间中位差仅 {spread.median():.2%} —— "
            f"若确实这么小，Q18 的结论需要重新审视")
        assert spread.max() > 0.30, f"最大差仅 {spread.max():.2%}"

    def test_2021年沪深300与等权全市场符号相反(self):
        """2021 年沪深300 −6.28% / 等权全市场 +24.80% ——
        差 31 个百分点，**同一策略的结论完全取决于选哪个基准**。"""
        p = ROOT / "runtime" / "benchmarks.csv"
        if not p.exists():
            pytest.skip("尚未跑 compare_benchmarks.py")
        df = pd.read_csv(p, encoding="utf-8-sig", index_col=0)
        if 2021 not in df.index:
            pytest.skip("数据不含 2021 年")
        hs300 = df.loc[2021, "沪深300"]
        ew = df.loc[2021, "等权全市场"]
        assert hs300 < 0 < ew, (
            f"2021 年应沪深300<0<等权全市场，实得 {hs300:.2%} / {ew:.2%}")
        assert ew - hs300 > 0.20, f"两者应差 20pp 以上，实差 {ew-hs300:.2%}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))