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
sys.path.insert(0, str(ROOT / "src" / "tools"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

import pandas as pd  # noqa: E402
from build_sqlite import SCHEMA, STOCK_INFO_CALIBER_META, derive_equity_caliber  # noqa: E402
from check_units import (  # noqa: E402
    FAMILY_OTH_EQT_TOOLS,
    FIX_MKTCAP_UNIT_SQL,
    PROBES,
    apply_fix_sql,
    assert_fix_sql_idempotent,
    check_equity_caliber,
    compare_bps_with_db,
    implied_bps_a,
    implied_bps_c,
    ordinary_equity_yi,
    probe_equity_row,
    require_oth_eqt_tools_column,
    stock_info_fix_copy,
)

from factor_lab.config import DB_PATH  # noqa: E402

DB_SKIP = pytest.mark.skipif(
    not Path(DB_PATH).exists(),
    reason=f"数据产物缺失：{DB_PATH} 不存在（无产物口径），真库权益探针跳过",
)


def _ro_con() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


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


class TestStockInfoEquityCaliber:
    """stock_info 派生口径统一：普通股权益口径列 + 口径记入 meta（U6 规则③/U3 处方）。

    原生 net_assets（通达信 JZC）是【含其他权益工具】口径、bps（TZMGJZ）是
    【普通股】口径，两列互推会错 17% 量级（A18 实证）。派生列
    net_assets_ord = bps × shares 提供扣口径分子（≈ 归母 − oth_eqt_tools），
    原生列保留、口径写进 meta，后缀 _ord = 普通股权益。"""

    def test_net_assets_ord派生为bps乘shares_000001实值(self) -> None:
        df = derive_equity_caliber(pd.DataFrame({
            "code": ["sz000001"], "bps": [24.129],
            "shares": [194.059187], "net_assets": [5482.14016],
        }))
        assert df["net_assets_ord"][0] == pytest.approx(4682.4541231230005, rel=1e-12)

    def test_派生不覆写原生net_assets(self) -> None:
        df = derive_equity_caliber(pd.DataFrame({
            "code": ["sz000001"], "bps": [24.129],
            "shares": [194.059187], "net_assets": [5482.14016],
        }))
        assert df["net_assets"][0] == 5482.14016, "原生含口径列必须原样保留"

    def test_NULL传播不冒算(self) -> None:
        df = derive_equity_caliber(pd.DataFrame({"bps": [None], "shares": [1.0]}))
        assert df["net_assets_ord"].isna()[0]

    def test_缺派生列显式抛错(self) -> None:
        with pytest.raises(ValueError, match="net_assets_ord"):
            derive_equity_caliber(pd.DataFrame({"bps": [24.129]}))

    def test_meta口径记录带后缀与扣减式(self) -> None:
        keys = {k for k, _ in STOCK_INFO_CALIBER_META}
        assert keys == {
            "stock_info.net_assets", "stock_info.net_assets_ord",
            "stock_info.bps", "stock_info.shares",
        }
        by = dict(STOCK_INFO_CALIBER_META)
        assert "oth_eqt_tools" in by["stock_info.net_assets_ord"]
        assert "普通股" in by["stock_info.bps"]
        assert "含其他权益工具" in by["stock_info.net_assets"]
        assert "股" in by["stock_info.shares"], "shares 必须标注股口径（U3 同名不同纲）"

    def test_schema带口径派生列(self) -> None:
        assert "net_assets_ord REAL" in SCHEMA


class TestProbeListSeven:
    """探针清单 6→7：补 sh600011 双高边界锚（oth 53.25% + minority 52.75%，敞口谱推到极端档）。"""

    def test_探针恰7只且含双高边界锚(self) -> None:
        codes = [p[0] for p in PROBES]
        assert len(codes) == 7
        assert codes == [
            "sh600519", "sz000001", "sh601318", "sz300750",
            "sh600036", "sh601288", "sh600011",
        ]


class TestFixSqlIdempotent:
    """市值量纲修复 SQL 必须幂等：重跑结果不变（修复工具自己不许带二次事故）。

    旧提示 `UPDATE stock_info SET mktcap *= 1e4` 是整列乘法，重跑再放大 1e4。"""

    @staticmethod
    def _fixture() -> sqlite3.Connection:
        con = sqlite3.connect(":memory:")
        con.execute(
            "CREATE TABLE stock_info (code TEXT PRIMARY KEY, mktcap REAL,"
            " float_mktcap REAL, price REAL, shares REAL)"
        )
        con.executemany(
            "INSERT INTO stock_info VALUES (?,?,?,?,?)",
            [
                # 亿元形态行（体量反推价 = 价格，该被修一次）
                ("bad", 4682.14, 3000.0, 24.1273, 194.06),
                # 已是万元形态行（反推价 = 价格，不许动）
                ("ok", 25125359.4, 25125359.4, 200.9898, 12.500816),
                # 两种形态都对不上：属数值问题不是量纲问题，不许盲乘
                ("odd", 999.0, 999.0, 50.0, 2.0),
            ],
        )
        con.commit()
        return con

    def test_只修亿元形态行且恰放大一次(self) -> None:
        con = self._fixture()
        apply_fix_sql(con)
        rows = {r[0]: r[1:] for r in con.execute("SELECT * FROM stock_info")}
        assert rows["bad"][0] == pytest.approx(4682.14 * 1e4, rel=1e-12)
        assert rows["bad"][1] == pytest.approx(3000.0 * 1e4, rel=1e-12)
        assert rows["ok"] == (25125359.4, 25125359.4, 200.9898, 12.500816), "万元行不许动"
        assert rows["odd"] == (999.0, 999.0, 50.0, 2.0), "形态不明行不许盲乘"

    def test_重跑零变动_幂等断言(self) -> None:
        con = self._fixture()
        r1 = assert_fix_sql_idempotent(con)
        assert r1 == {"first_run_rows": 1, "second_run_rows": 0, "idempotent": True}
        r2 = assert_fix_sql_idempotent(con)
        assert r2["first_run_rows"] == 0 and r2["second_run_rows"] == 0

    def test_修复SQL带形态条件_非整列乘法(self) -> None:
        assert "UPDATE stock_info" in FIX_MKTCAP_UNIT_SQL
        assert "WHERE" in FIX_MKTCAP_UNIT_SQL, "必须带量纲形态条件，否则重跑即二次放大"
        assert "*= 1e4" not in FIX_MKTCAP_UNIT_SQL

    def test_旧式整列乘法重跑必变_缺陷写法反例(self) -> None:
        legacy = "UPDATE stock_info SET mktcap = mktcap * 1e4, float_mktcap = float_mktcap * 1e4"
        con = self._fixture()
        con.execute(legacy)
        once = sorted(con.execute("SELECT * FROM stock_info"))
        con.execute(legacy)
        twice = sorted(con.execute("SELECT * FROM stock_info"))
        assert once != twice, "整列乘法若重跑不变，本反例失效（口径已改动？）"

    def test_stock_info副本不触真库(self) -> None:
        src = self._fixture()
        copy = stock_info_fix_copy(src)
        apply_fix_sql(copy)
        rows = {r[0] for r in src.execute("SELECT code FROM stock_info WHERE mktcap > 1e7")}
        assert rows == {"ok"}, "修复试验只许在副本上做"


@DB_SKIP
class TestEquityProbeOnMarketDb:
    """真库（market.db 只读）六探针与两守卫的双向实证；无产物口径下整类跳过。"""

    def test_全探针C式全收敛(self) -> None:
        con = _ro_con()
        try:
            for code, _, _name in PROBES:
                p = probe_equity_row(con, code)
                assert p["status"] == "ok", (code, p)
                assert p["verdict"]["pass_c"], (code, p["verdict"])
                assert p["verdict"]["rel_c"] < 1e-5, (code, p["verdict"])
        finally:
            con.close()

    def test_600011双高边界锚relC应小于十万分之一(self) -> None:
        con = _ro_con()
        try:
            p = probe_equity_row(con, "sh600011")
        finally:
            con.close()
        assert p["status"] == "ok", p
        assert p["verdict"]["rel_c"] < 1e-5, p["verdict"]

    def test_族标记落在四只其他权益工具票(self) -> None:
        con = _ro_con()
        try:
            fam = {code: probe_equity_row(con, code)["verdict"]["family"]
                   for code, _, _ in PROBES}
        finally:
            con.close()
        # oth_eqt_tools>0 的四票（000001=800 亿/600036=1999.89 亿/601288=4700 亿/
        # 600011=734.75 亿）必须 A 败 C 胜并标记；其余三票两式同判，不标记。
        for code in ("sz000001", "sh600036", "sh601288", "sh600011"):
            assert fam[code] == FAMILY_OTH_EQT_TOOLS, (code, fam)
        for code in ("sh600519", "sh601318", "sz300750"):
            assert fam[code] is None, (code, fam)

    def test_恒等式守卫真库双向(self) -> None:
        con = _ro_con()
        try:
            g = {code: probe_equity_row(con, code)["identity"] for code, _, _ in PROBES}
        finally:
            con.close()
        assert not g["sz000001"]["checkable"], "000001 minority_int=NULL 应显式不可检"
        for code in ("sh600519", "sh601318", "sz300750", "sh600036", "sh601288", "sh600011"):
            assert g[code]["checkable"] and g[code]["pass"], (code, g[code])

    def test_自洽守卫真库能拦到混日期快照(self) -> None:
        con = _ro_con()
        try:
            g = {code: probe_equity_row(con, code)["pb"] for code, _, _ in PROBES}
        finally:
            con.close()
        for code in ("sh600519", "sz000001", "sh600036", "sh601288"):
            assert g[code]["pass"], (code, g[code])
        n_flagged = sum(1 for code in ("sh601318", "sz300750") if not g[code]["pass"])
        assert n_flagged >= 1, "已知混日期快照票应被守卫拦下（A18 §五附 2 票违例）"

    def test_检查3返回两级清单_幂等断言常驻(self) -> None:
        con = _ro_con()
        try:
            msgs, warns = check_equity_caliber(con)
            fix_probe = assert_fix_sql_idempotent(stock_info_fix_copy(con))
        finally:
            con.close()
        assert isinstance(msgs, list) and isinstance(warns, list)
        assert fix_probe["idempotent"] and fix_probe["second_run_rows"] == 0
