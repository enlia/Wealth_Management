"""U6 两守卫与缺件探针回归的测试（从 test_equity_caliber.py 纯搬移拆出）。

拆分动机：test_equity_caliber.py 555 行越过「代码文件 ≤ 500 行」硬线（CI 行数门
口径 = splitlines 含空行），按 400~500 强制拆分铁律按测试簇拆两件：本文件收
「守卫族」（权益切分恒等式 / 快照自洽 / 缺 total_share 显式缺件态与逐票捕获），
公式族、派生口径、幂等与真库探针留在 test_equity_caliber.py。**用例本体逐字
搬移、语义零变**；跨文件共用名经 check_units 原名导入。

运行
----
  uv run pytest tests/test_equity_guards.py -v
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
    equity_identity_guard,
    probe_equity_row,
    snapshot_pb_guard,
)


def _run_check_with_probes(probes, con) -> tuple[list[str], list[str]]:
    """临时换探针清单跑 check_equity_caliber（跑完还原），返回 (msgs, warns)。"""
    import check_units as cu

    saved = cu.PROBES
    cu.PROBES = probes
    try:
        return cu.check_equity_caliber(con)
    finally:
        cu.PROBES = saved


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

    def test_price为0停牌形态不可检不误报(self) -> None:
        """price=0（停牌/退市形态）必须落不可检分支：不许算出 ratio=0 冒判混日期嫌疑。"""
        g = snapshot_pb_guard(0, 24.129, 0.48)
        assert not g["checkable"], g
        assert g["missing"] == ["price"], g

    def test_price为负脏数据不可检不误报(self) -> None:
        g = snapshot_pb_guard(-3.5, 24.129, 0.48)
        assert not g["checkable"], g
        assert g["missing"] == ["price"], g

    def test_低pb舍入界内不误拦_pb048(self) -> None:
        """pb 两位小数的舍入界是绝对 ±0.005：pb=0.48 时相对界 1.04%，旧 1% 相对容差会误拦
        同日快照低 pb 票；复合容差 max(0.0051, 0.01×pb) 必须放行。"""
        g = snapshot_pb_guard(0.485, 1.0, 0.48)   # |ratio−pb| = 0.0050
        assert g["checkable"] and g["pass"], g

    def test_复合容差边界双向_pb048(self) -> None:
        inside = snapshot_pb_guard(0.485, 1.0, 0.48)     # 0.0050 ≤ max(0.0051, 0.01×0.48)=0.0051
        outside = snapshot_pb_guard(0.4853, 1.0, 0.48)   # 0.0053 > 0.0051
        assert inside["pass"] and not outside["pass"], (inside, outside)

    def test_复合容差常显式(self) -> None:
        import check_units as cu

        assert cu.PB_ROUND_ABS_TOL == 0.0051, "两位小数舍入界（绝对）"
        assert cu.PB_REL_TOL == 0.01, "相对带（既有名，与 test_容差显式为1pct 同源）"


class TestProbeExplicitMissingShare:
    """缺 total_share → 显式缺件态、单票不中断全轮（活反例 sh600919@20260630 total_share=NULL）。

    修复前该形态在 probe 内抛 ValueError 整轮崩，与 docstring「缺行/缺基准/归母 NULL
    → status 显式」不一致。"""

    @staticmethod
    def _mk_db() -> sqlite3.Connection:
        con = sqlite3.connect(":memory:")
        con.execute(
            "CREATE TABLE ts_balance_sheet (ts_code TEXT, end_date INT,"
            " total_hldr_eqy_exc_min_int REAL, total_hldr_eqy_inc_min_int REAL,"
            " minority_int REAL, oth_eqt_tools REAL, total_share REAL, report_type TEXT)"
        )
        con.execute("CREATE TABLE ts_fina_indicator (ts_code TEXT, end_date INT, bps REAL)")
        con.execute("CREATE TABLE stock_info (code TEXT, price REAL, bps REAL, pb REAL)")
        con.executemany(
            "INSERT INTO ts_balance_sheet VALUES (?,?,?,?,?,?,?,?)",
            [
                # sh600919@20260630 活反例实值（market.db 只读取证）：total_share=NULL
                ("sh600919", 20260630, 3513.36365, 3636.85184,
                 123.48819, 799.7783, None, "1"),
                # 健康对照行（sz000001 夹具值，oth=800 亿 → 必产出族标记记录）
                ("sz000001", 20260630, 5482.14, 5482.14,
                 None, 800.0, 19406000000.0, "1"),
            ],
        )
        con.executemany(
            "INSERT INTO ts_fina_indicator VALUES (?,?,?)",
            [("sh600919", 20260630, 14.7869), ("sz000001", 20260630, 24.1273)],
        )
        con.executemany(
            "INSERT INTO stock_info VALUES (?,?,?,?)",
            [("sh600919", 12.38, 14.79, 0.84), ("sz000001", 11.57, 24.129, 0.48)],
        )
        con.commit()
        return con

    def test_600919缺total_share返回显式缺件态不冒算(self) -> None:
        con = self._mk_db()
        p = probe_equity_row(con, "sh600919")           # 不许抛错
        assert p["status"] != "ok", p
        assert "total_share" in p["status"], p
        assert "verdict" not in p, "缺分母不许产出对账判读"

    def test_缺total_share单票不中断全轮(self) -> None:
        con = self._mk_db()
        msgs, warns = _run_check_with_probes(
            [("sh600919", "600919.SH", "活反例缺股本"),
             ("sz000001", "000001.SZ", "健康对照")],
            con,
        )
        assert any("活反例缺股本" in m and "total_share" in m for m in msgs), msgs
        assert any("健康对照" in w and FAMILY_OTH_EQT_TOOLS in w for w in warns), \
            "缺件票之后的健康票必须继续处理并留痕（其他权益工具族记录）"

    def test_逐票异常被捕获归msgs不中断全轮(self, monkeypatch) -> None:
        def _boom(_con, code):
            if code == "sh600919":
                raise RuntimeError("单票异常测试桩")
            return {
                "code": code, "status": "ok",
                "verdict": {"pass_c": True, "rel_c": 1e-7, "rel_a": 1e-7,
                            "family": None, "implied_c": 1.0},
                "oth_eqt_tools_yi": 0.0, "db_bps": 1.0,
                "identity": {"checkable": False, "missing": ["x"]},
                "pb": {"checkable": False, "missing": ["y"]},
            }

        monkeypatch.setattr("check_units.probe_equity_row", _boom)
        msgs, _warns = _run_check_with_probes(
            [("sh600919", "600919.SH", "异常票"), ("sh600519", "600519.SH", "健康票")],
            self._mk_db(),
        )
        assert any("异常票" in m and "捕获" in m for m in msgs), msgs
