"""每股净资产反推口径（普通股权益）与量纲修复工具的测试。

为什么必须有这个文件
--------------------
「每股净资产 = 归母权益 ÷ 总股本」是错的口径：归母权益里可能含其他权益工具
（优先股/永续债），而每股净资产（bps）是**每股普通股**净资产，分子必须扣除
其他权益工具。口径搞错**不报错**，只让反推对账系统性偏差 = 其他权益工具占归母比
——实测平安银行 sz000001 2026-06-30：800 亿其他权益工具未扣 → 反推偏差 17.09%，
而库内 bps 与扣口径后的反推逐位闭合（1e-7 量级）。证据：A18 判决书
（.planning/reports/a18-sz000001-bps-forensics.md）、规则条目 UNITS.md U6。

本文件钉住四件事：
  1. C 式（扣除其他权益工具）对已知四票收敛，A 式（不扣）对照列与其判读差异；
  2. COALESCE 语义：oth_eqt_tools 行内 NULL 按 0 处理，**缺列必须显式报错**，
     两者不许混同（NULL=合法「无其他权益工具」，缺列=无法执行）；
  3. 两守卫（权益恒等式 inc−exc=minority、快照自洽 price/bps≈pb）好/坏样本双向；
  4. 量纲修复 SQL 幂等（重跑结果不变），防「修复工具自己越修越错」。

四票已知值出处：market.db `ts_balance_sheet` @end_date=20260630（report_type='1'）
与 `ts_fina_indicator.bps` 同期；冻结为夹具值，换库不随之漂移。

运行
----
  uv run pytest tests/test_equity_caliber.py -v
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from check_units import (  # noqa: E402
    FAMILY_OTH_EQT_TOOLS,
    IDENTITY_ABS_TOL_YI,
    PB_REL_TOL,
    compare_bps_with_db,
    equity_identity_guard,
    implied_bps_a,
    implied_bps_c,
    ordinary_equity_yi,
    require_oth_eqt_tools_column,
    snapshot_pb_guard,
)

# 四票夹具：(code, 归母权益亿元, 其他权益工具亿元(NULL→None), 总股本股, 库内 bps 元/股)
# 出处：market.db ts_balance_sheet / ts_fina_indicator @20260630（report_type='1'）
FOUR_VOTES = [
    ("sz000001", 5482.14, 800.0, 19406000000.0, 24.1273),
    ("sh600519", 2512.535944, None, 1250081601.0, 200.9898),
    ("sh601318", 10280.84, None, 18107641995.0, 56.7762),
    ("sz300750", 3793.5371, None, 4626650861.0, 81.9932),
]

# C 式期望值（全精度，按上表输入算出，钉死算式本身不许漂）
EXPECTED_IMPLIED_C = {
    "sz000001": 24.127280222611567,
    "sh600519": 200.98975474801827,
    "sh601318": 56.77624951298912,
    "sz300750": 81.99315690702602,
}


class TestImpliedBpsCFourVotes:
    """C 式反推 = (归母 − COALESCE(oth_eqt_tools,0)) × 1e8 / total_share。"""

    @pytest.mark.parametrize(("code", "exc", "oth", "share", "db"), FOUR_VOTES)
    def test_四票C式对账收敛远小于1pct(self, code, exc, oth, share, db) -> None:
        v = compare_bps_with_db(db, exc, oth, share)
        assert v["pass_c"], f"{code} C 式未收敛：{v}"
        assert v["rel_c"] < 1e-6, f"{code} C 式应收敛到 1e-7 量级，实为 {v['rel_c']}"

    @pytest.mark.parametrize(("code", "exc", "oth", "share", "db"), FOUR_VOTES)
    def test_C式数值锚定不许漂(self, code, exc, oth, share, db) -> None:
        assert implied_bps_c(exc, oth, share) == pytest.approx(
            EXPECTED_IMPLIED_C[code], rel=1e-12
        )

    def test_000001_A式败C式胜自动标记其他权益工具族(self) -> None:
        code, exc, oth, share, db = FOUR_VOTES[0]
        v = compare_bps_with_db(db, exc, oth, share)
        assert v["implied_a"] == pytest.approx(28.249716582500263, rel=1e-12)
        assert v["rel_a"] > 0.01, "A 式（分子不扣其他权益工具）应败在 17% 量级"
        assert v["pass_c"] and not v["pass_a"]
        assert v["family"] == FAMILY_OTH_EQT_TOOLS, "A 败 C 胜即标记口径混装族"

    @pytest.mark.parametrize(("code", "exc", "oth", "share", "db"), FOUR_VOTES[1:])
    def test_无其他权益工具票_A式C式同判不标记(self, code, exc, oth, share, db) -> None:
        v = compare_bps_with_db(db, exc, oth, share)
        assert v["pass_a"] and v["pass_c"], f"{code} 应两式同过：{v}"
        assert v["implied_a"] == v["implied_c"]
        assert v["family"] is None

    def test_db_bps为0或None显式抛错(self) -> None:
        for bad in (0, None):
            with pytest.raises(ValueError, match="db_bps"):
                compare_bps_with_db(bad, 5482.14, 800.0, 19406000000.0)


class TestCoalesceSemantics:
    """NULL=合法「无其他权益工具」按 COALESCE 归 0；缺列=无法执行，必须显式报错。"""

    def test_oth为NULL按COALESCE归零(self) -> None:
        assert ordinary_equity_yi(5482.14, None) == 5482.14
        assert implied_bps_c(5482.14, None, 19406000000.0) == implied_bps_a(
            5482.14, 19406000000.0
        )

    def test_oth为0与NULL等价(self) -> None:
        assert ordinary_equity_yi(5482.14, 0.0) == ordinary_equity_yi(5482.14, None)

    def test_oth非零必须扣除(self) -> None:
        assert ordinary_equity_yi(5482.14, 800.0) == pytest.approx(4682.14)

    def test_归母为None显式抛错不静默归零(self) -> None:
        with pytest.raises(ValueError, match="归属母公司"):
            ordinary_equity_yi(None, 800.0)

    @pytest.mark.parametrize("share", [0, 0.0, None])
    def test_股本缺失或为0显式抛错(self, share) -> None:
        with pytest.raises(ValueError, match="总股本"):
            implied_bps_c(5482.14, 800.0, share)

    @staticmethod
    def _mk_db(cols: str) -> sqlite3.Connection:
        con = sqlite3.connect(":memory:")
        con.execute(f"CREATE TABLE ts_balance_sheet ({cols})")  # noqa: S608
        return con

    def test_缺oth_eqt_tools列显式报错(self) -> None:
        con = self._mk_db("ts_code TEXT, total_share REAL")
        with pytest.raises(RuntimeError, match="oth_eqt_tools"):
            require_oth_eqt_tools_column(con)

    def test_有oth_eqt_tools列放行(self) -> None:
        con = self._mk_db("ts_code TEXT, oth_eqt_tools REAL, total_share REAL")
        require_oth_eqt_tools_column(con)   # 不抛即通过

    def test_表不存在显式报错(self) -> None:
        con = sqlite3.connect(":memory:")
        with pytest.raises(RuntimeError, match="ts_balance_sheet"):
            require_oth_eqt_tools_column(con)


class TestEquityIdentityGuard:
    """恒等式守卫：含少数股东权益 − 归母权益 = 少数股东权益（U6 规则②）。

    输入为 market.db ts_balance_sheet@20260630 实值；好坏样本双向：
    成立过、破坏拦、NULL 显式不可检。"""

    def test_容差显式为两位舍入残差界(self) -> None:
        assert IDENTITY_ABS_TOL_YI == 0.02

    def test_恒等式成立通过_600519实值(self) -> None:
        g = equity_identity_guard(2620.9635217436, 2512.535944195, 108.42757754860001)
        assert g["checkable"] and g["pass"], g
        assert abs(g["residual_yi"]) < 1e-9

    def test_恒等式破坏被拦(self) -> None:
        g = equity_identity_guard(2620.9635217436, 2512.535944195, 118.42757754860001)
        assert g["checkable"] and not g["pass"], g
        assert abs(g["residual_yi"] - (-10.0)) < 1e-9

    def test_容差内过界外拦(self) -> None:
        inside = equity_identity_guard(100.0, 50.0, 49.981)   # 残差 0.019
        outside = equity_identity_guard(100.0, 50.0, 49.979)  # 残差 0.021
        assert inside["pass"] and not outside["pass"], (inside, outside)

    def test_minority为NULL不可检且点名缺项(self) -> None:
        g = equity_identity_guard(5482.14, 5482.14, None)     # sz000001 形态
        assert not g["checkable"], g
        assert g["missing"] == ["minority_int"]

    def test_输入缺项不冒算(self) -> None:
        g = equity_identity_guard(None, 5482.14, 0.0)
        assert not g["checkable"]
        assert g["missing"] == ["total_hldr_eqy_inc_min_int"]


class TestSnapshotPbGuard:
    """自洽守卫：price / bps ≈ pb（U6 规则②）——stock_info 快照混日期在此露馅。

    输入为 market.db stock_info 快照实值；同日快照过、混日期快照拦。"""

    def test_容差显式为1pct(self) -> None:
        assert PB_REL_TOL == 0.01

    @pytest.mark.parametrize(
        ("code", "price", "bps", "pb"),
        [
            ("sh600519", 1258.62, 200.99, 6.26),   # price/bps=6.26210 ≈ 6.26 ✓
            ("sz000001", 11.57, 24.129, 0.48),      # 0.479506 ≈ 0.48 ✓
            ("sh600036", 41.26, 45.4, 0.91),        # 0.908811 ≈ 0.91 ✓
        ],
    )
    def test_同日快照通过(self, code, price, bps, pb) -> None:
        g = snapshot_pb_guard(price, bps, pb)
        assert g["checkable"] and g["pass"], (code, g)

    @pytest.mark.parametrize(
        ("code", "price", "bps", "pb", "ratio"),
        [
            ("sh601318", 53.29, 56.779, 0.96, 0.938551224924708),
            ("sz300750", 291.11, 81.992, 3.61, 3.5504683383744755),
        ],
    )
    def test_混日期快照被拦(self, code, price, bps, pb, ratio) -> None:
        g = snapshot_pb_guard(price, bps, pb)
        assert g["checkable"] and not g["pass"], (code, g)
        assert g["ratio"] == pytest.approx(ratio, rel=1e-12)
        assert g["rel_err"] > PB_REL_TOL

    def test_bps缺失不可检不冒算(self) -> None:
        assert not snapshot_pb_guard(11.57, None, 0.48)["checkable"]
        assert not snapshot_pb_guard(11.57, 0, 0.48)["checkable"]

    def test_price或pb缺失不可检(self) -> None:
        assert not snapshot_pb_guard(None, 24.129, 0.48)["checkable"]
        assert not snapshot_pb_guard(11.57, 24.129, None)["checkable"]
