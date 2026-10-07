"""低换手月频骨架的算式守护（换手/成本/前视/死参数）。

为什么必须有这个文件
--------------------
骨架的可信度全在几条**能手算的算式**上：单边换手 = ½·Σ|Δw|、
期成本 = 单边换手 × config.DEFAULT_COST.round_trip、收益取**次期**实现
（防前视）、缓冲档参数必须**数值真变**（防死参数）。这几条一旦被改坏，
冒烟数字照样「看着合理」。夹具全人造、不读 market.db。

运行
----
  uv run pytest tests/test_low_turnover_monthly.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from run_low_turnover_monthly import (  # noqa: E402
    SMOKE_LABEL,
    assert_distinct_pools,
    lag_by_months,
    map_buffer,
    render_report,
    simulate,
    target_weights,
)

from factor_lab.config import DEFAULT_COST  # noqa: E402


class TestTurnoverManualAnchor:
    """换手算式手算锚：建仓日 = ½×建仓幅度；换仓日 = k÷n_valid。"""

    @staticmethod
    def _fixture():
        # 4 股、3 槽 × 2 个月首日；首月建仓 3 槽，次月换 1 槽（0 退、3 进）
        dates = pd.DatetimeIndex(["2024-01-02", "2024-02-01"])
        price = pd.DataFrame({"a": [10.0, 11.0], "b": [10.0, 10.0],
                              "c": [10.0, 10.0], "d": [10.0, 10.0]}, index=dates)
        held = np.array([[0, 1, 2], [1, 2, 3]])   # 次月：0 退 3 进（真换 1 槽）
        is_rebal = np.array([True, True])
        return price, held, is_rebal

    def test_建仓换手按半幅度计(self) -> None:
        price, held, is_rebal = self._fixture()
        sim = simulate(price, held, target_weights(held), is_rebal)
        ev = sim["events"].iloc[0]
        assert ev["event"] == "build"
        assert ev["turn_one_side"] == pytest.approx(0.5)          # ½×(3×1/3)
        assert ev["manual_one_side"] == pytest.approx(0.5)
        assert ev["check_rel"] == pytest.approx(0.0, abs=1e-12)

    def test_换仓一槽为一除以n_valid(self) -> None:
        price, held, is_rebal = self._fixture()
        sim = simulate(price, held, target_weights(held), is_rebal)
        ev = sim["events"].iloc[1]
        assert ev["event"] == "swap"
        assert ev["turn_one_side"] == pytest.approx(1 / 3)        # k=1 ÷ n_valid=3
        assert ev["check_rel"] == pytest.approx(0.0, abs=1e-12)

    def test_建仓被算入换手不许归零(self) -> None:
        """还原必失败：前补 W[:1] 的旧写法会把建仓换手压成 0。"""
        price, held, is_rebal = self._fixture()
        sim = simulate(price, held, target_weights(held), is_rebal)
        assert sim["turn_one_side"][0] > 0.49


class TestCostFormula:
    def test_成本取单边换手乘config往返费率(self) -> None:
        assert DEFAULT_COST.round_trip == pytest.approx(0.002)   # 20bp/往返，引用不自造
        price, held, is_rebal = TestTurnoverManualAnchor._fixture()
        sim = simulate(price, held, target_weights(held), is_rebal)
        for _, ev in sim["events"].iterrows():
            assert ev["cost"] == pytest.approx(
                ev["turn_one_side"] * DEFAULT_COST.round_trip)


class TestNoLookahead:
    def test_收益取次期实现不吃选定日(self) -> None:
        """选中日的涨幅不计、次日涨幅才计（fwd = pct_change.shift(-1)）。"""
        dates = pd.DatetimeIndex(["2024-01-02", "2024-02-01"])
        # a 号票在选定日 d0→d1 间涨 20%；b 号票平。持仓 a/b 各半。
        price = pd.DataFrame({"a": [10.0, 12.0], "b": [10.0, 10.0]}, index=dates)
        held = np.array([[0, 1], [0, 1]])
        sim = simulate(price, held, target_weights(held),
                       np.array([True, True]))
        # d0 的组合收益 = 0.5×20%（a）+ 0.5×0（b）——只吃「选定之后」那段
        assert sim["gross"][0] == pytest.approx(0.5 * 0.2)
        assert sim["gross"][1] == 0.0          # 末日无下一期收益 → 0（防借巧合）


class TestSignalLag:
    def test_滞后整月取月末快照(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=8)   # 跨 1~2 月
        z = pd.DataFrame({"a": np.arange(8.0), "b": -np.arange(8.0)}, index=dates)
        out = lag_by_months(z, 1)
        jan_end = z[z.index.month == 1].iloc[-1]           # 1 月快照（月末值）
        feb_rows = out.index.month == 2
        assert (out.loc[feb_rows, "a"].to_numpy() == jan_end["a"]).all()
        assert out.loc[feb_rows, "b"].eq(jan_end["b"]).all()

    def test_零滞后原样返回对象(self) -> None:
        z = pd.DataFrame({"a": [1.0, 2.0]},
                         index=pd.DatetimeIndex(["2024-01-02", "2024-02-01"]))
        assert lag_by_months(z, 0) is z


class TestBufferDeadParameterGuard:
    def test_映射无缓冲即自选(self) -> None:
        assert map_buffer(7, 0.0) == 7
        assert map_buffer(7, 0.1) == 8
        assert map_buffer(7, 0.2) == 9

    def test_映射重复必须抛错防死参数(self) -> None:
        with pytest.raises(ValueError, match="死参数"):
            assert_distinct_pools({0.0: 5, 0.1: 6, 0.2: 6}, n_pool_eff=9)

    def test_封顶重复必须抛错防隐性死参数(self) -> None:
        with pytest.raises(ValueError, match="封顶"):
            assert_distinct_pools({0.0: 7, 0.1: 8, 0.2: 9}, n_pool_eff=8)

    def test_正常档距放行并回有效池值(self) -> None:
        eff = assert_distinct_pools({0.0: 7, 0.1: 8, 0.2: 9}, n_pool_eff=9)
        assert eff == {0.0: 7, 0.1: 8, 0.2: 9}


class TestReportTemplate:
    def test_模板含强制列头与免责尾句(self) -> None:
        dates = pd.DatetimeIndex(["2024-01-02", "2024-02-01", "2024-03-01"])
        price = pd.DataFrame({"a": [10.0, 11.0, 12.1], "b": [10.0, 9.5, 9.9]},
                             index=dates)
        held = np.array([[0, 1], [0, 1], [0, 1]])
        sim = simulate(price, held, target_weights(held),
                       np.array([True, True, True]))
        rep = render_report(sim, price, ["rev5"], {"n_hold": 2}, attempts=1)
        for head in ("口径三件套", "价格口径", "换手单位", "年化方式",
                     "①年化口径", "②成本假设", "③基准", "④多空/多头",
                     "多重比较声明", SMOKE_LABEL, "不构成投资建议"):
            assert head in rep, head

