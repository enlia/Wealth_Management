"""Deflated Sharpe（Bailey & López de Prado 2014）公式的守护用例。

锁三件事：① N=1 退化=普通显著性检验（SR0=0）；② DSR 随尝试次数单调不增
（多重比较惩罚方向）；③ 异常输入显式抛错（短序列/零方差/非法 N）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.analysis.deflated import (  # noqa: E402
    EULER_GAMMA,
    deflated_sharpe_ratio,
)

R = pd.Series([0.01, 0.02, -0.005, 0.015, 0.0, 0.008, -0.002, 0.012])


class TestDeflatedFormula:
    def test_N1退化为普通显著性检验(self) -> None:
        d = deflated_sharpe_ratio(R, n_trials=1, periods_per_year=1.0)
        sd, t = float(R.std(ddof=1)), len(R)
        sr = float(R.mean()) / sd
        skew = float(stats.skew(R, bias=False))
        kurt = float(stats.kurtosis(R, bias=False, fisher=False))
        quad = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
        assert d["sr0"] == 0.0
        assert d["dsr"] == pytest.approx(
            stats.norm.cdf(sr * math.sqrt(t - 1) / math.sqrt(quad)), rel=1e-12)

    def test_SRa公式锚(self) -> None:
        d = deflated_sharpe_ratio(R, n_trials=1, periods_per_year=252.0)
        assert d["sr_annual"] == pytest.approx(
            float(R.mean()) / float(R.std(ddof=1)) * math.sqrt(252.0), rel=1e-12)

    def test_DSR随尝试次数单调不增(self) -> None:
        dsrs = [deflated_sharpe_ratio(R, n_trials=n, periods_per_year=1.0)["dsr"]
                for n in (1, 2, 10, 100)]
        assert dsrs == sorted(dsrs, reverse=True), dsrs
        assert dsrs[0] > dsrs[-1]

    def test_择优期望SR0公式锚(self) -> None:
        d = deflated_sharpe_ratio(R, n_trials=10, periods_per_year=1.0)
        sr = d["sr_annual"]
        skew, kurt, t = d["skew"], d["kurtosis"], d["n_obs"]
        quad = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
        var = quad / (t - 1)
        z1 = stats.norm.ppf(1 - 1 / 10)
        z2 = stats.norm.ppf(1 - 1 / (10 * math.e))
        assert d["sr0"] == pytest.approx(
            math.sqrt(var) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2), rel=1e-12)

    @pytest.mark.parametrize("n", [3, 10])
    def test_序列过短显式抛错(self, n: int) -> None:
        with pytest.raises(ValueError, match="不足以估计"):
            deflated_sharpe_ratio(pd.Series([0.01] * n)[:2], n_trials=1)

    def test_零方差显式抛错(self) -> None:
        with pytest.raises(ValueError, match="标准差为 0"):
            deflated_sharpe_ratio(pd.Series([0.01, 0.01, 0.01]), n_trials=1)

    @pytest.mark.parametrize("n", [0, -2])
    def test_非法尝试次数显式抛错(self, n: int) -> None:
        with pytest.raises(ValueError, match="n_trials"):
            deflated_sharpe_ratio(R, n_trials=n)

    # 边界注：`quad <= 0`（极高 SR×偏度组合）为公式固有防护分支，自然收益序列
    # 难以触发，未单测——该分支显式抛错不静默，属「抛错而非算错」路径。
