"""Tushare 工具链的关键逻辑测试。

为什么必须有
------------
这批代码（分页、去重、单位换算、校验判据）**零测试**，
而评审实测发现CI 的 5 项检查对它们几乎没有保护：
  · F841 抓不到「变量被重新赋值」的死代码
  · 行数检查只对 >500 阻断
  · pytest 跑的是别的模块的测试

后果是三个判据级 bug 都进了分支：
  · 预筛用更宽阈值 → 85 条真实超限被丢弃
  · 去重缺版本列 → 8 个合法报表版本被压成 1 行
  · 金额零换算 → 与 stock_info 差 1e8 倍

每个断言都对应一个**真实踩过的坑**。
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from build_money_whitelist import NOT_MONEY_EXACT, NOT_MONEY_PATTERNS  # noqa: E402
from fetch_all_tushare import dedup_by_business_key  # noqa: E402
from merge_tushare_tables import ts_to_local  # noqa: E402
from verify_tushare_full import check_one  # noqa: E402


class TestBusinessKeyDedup:
    """S8：同主键不同值是**多版本**，不能压成 1 行。"""

    def test_8个合法报表版本全保留(self) -> None:
        """财务三表同 (ts_code,end_date,ann_date) 有 4 report_type × 2 update_flag
        = 8 个合法版本，缺维度列时会被压成 1 行（实测踩过）。"""
        combos = list(itertools.product([1, 2, 3, 4], [0, 1]))
        df = pd.DataFrame({
            "ts_code": ["A"] * 8, "end_date": ["20250630"] * 8,
            "ann_date": ["20250815"] * 8,
            "report_type": [c[0] for c in combos],
            "update_flag": [c[1] for c in combos],
            "v": range(8),
        })
        out = dedup_by_business_key(df, "income")
        assert len(out) == 8, f"8 个合法版本被压成 {len(out)} 行"

    def test_主键相同值不同都保留(self) -> None:
        df = pd.DataFrame({
            "ts_code": ["A", "A"], "end_date": ["20250630"] * 2,
            "ann_date": ["20250815"] * 2,
            "report_type": [1, 2], "update_flag": [0, 1], "v": [1, 2],
        })
        assert len(dedup_by_business_key(df, "income")) == 2

    def test_同日多指数不合并(self) -> None:
        """S5：index_weight 同一 trade_date 有沪深300 与中证500 两套成分，
        缺 index_code 会被合并掉。"""
        df = pd.DataFrame({
            "index_code": ["HS300", "HS300", "ZZ500", "ZZ500"],
            "con_code": ["1", "2", "1", "2"],
            "trade_date": ["20240131"] * 4, "w": [1, 2, 3, 4],
        })
        assert len(dedup_by_business_key(df, "index_weight")) == 4

    def test_缺版本列时抛错而非静默去重(self) -> None:
        """主键用尽仍不唯一 → 必须 raise。
        静默 drop_duplicates 会悄悄丢数据。"""
        df = pd.DataFrame({
            "ts_code": ["A", "A"], "end_date": ["20250630"] * 2,
            "ann_date": ["20250815"] * 2, "v": [1, 2],
        })
        with pytest.raises(RuntimeError, match="未识别的版本维度"):
            dedup_by_business_key(df, "income")

    def test_未配置主键的表不做业务去重(self) -> None:
        """没有配置主键的表，本层**不动**它 ——
        整行去重由上游的 df.drop_duplicates() 负责。
        ⚠️ 这里的 2 行值不同，业务主键也未配置，
        若本层擅自 drop_duplicates 会静默丢数据。"""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
        out = dedup_by_business_key(df, "not_configured")
        assert len(out) == 3, "未配置主键时不应做业务去重"


class TestMoneyWhitelist:
    """P19：金额单位差 1e8 倍且不报错。"""

    def test_每股指标不换算(self) -> None:
        """bps=200.99 元/股，若/1e8 会变成 2e-6。"""
        for c in ("eps", "dt_eps", "bps", "basic_eps", "diluted_eps",
                  "revenue_ps", "total_revenue_ps", "cfps"):
            assert NOT_MONEY_PATTERNS.search(c) or c.lower() in NOT_MONEY_EXACT, \
                f"{c} 是每股指标，不应换算成亿元"

    def test_比率字段不换算(self) -> None:
        for c in ("roe", "debt_to_assets", "assets_turn", "assets_yoy",
                  "or_yoy", "grossprofit_margin", "inv_turn_days"):
            assert NOT_MONEY_PATTERNS.search(c) or c.lower() in NOT_MONEY_EXACT, \
                f"{c} 是比率/增速，不应换算"

    def test_金额字段要换算(self) -> None:
        for c in ("total_revenue", "n_income", "total_assets", "total_liab",
                  "money_cap", "n_cashflow_act"):
            assert not (NOT_MONEY_PATTERNS.search(c)
                        or c.lower() in NOT_MONEY_EXACT), \
                f"{c} 是金额列，必须换算"

    def test_股数不换算(self) -> None:
        assert NOT_MONEY_PATTERNS.search("total_share")

    def test_白名单文件存在(self) -> None:
        """缺失必须 raise —— 静默用空列表会让人以为「已经换算过了」。

        ⚠️ `runtime/tushare/` 是下载/推导出的**本地产物**（不进 git）：
        整个产物目录都不存在时跳过（CI 干净环境、未跑过下载的环境）；
        目录在、白名单却缺失 = 真缺陷，仍按原判据失败。
        """
        from merge_tushare_tables import WHITELIST_FILE
        if not WHITELIST_FILE.parent.exists():
            pytest.skip(f"缺 {WHITELIST_FILE.parent}（本地产物不进 git），"
                        "无产物环境跳过")
        assert WHITELIST_FILE.exists(), (
            f"白名单不存在：{WHITELIST_FILE}\n"
            f"  解决：uv run python research/scripts/build_money_whitelist.py")

    def test_实际白名单不含比率字段(self) -> None:
        import json

        from merge_tushare_tables import WHITELIST_FILE
        if not WHITELIST_FILE.exists():
            pytest.skip(f"缺 {WHITELIST_FILE}（本地产物不进 git），"
                        "无产物环境跳过")
        wl = json.loads(WHITELIST_FILE.read_text(encoding="utf-8"))
        for cols in wl.values():
            for c in cols:
                assert c.lower() not in NOT_MONEY_EXACT, \
                    f"{c} 不该在金额白名单里"

    def test_fina_indicator整表不换算(self) -> None:
        """每股/比率类表，换算意义不大且易错。"""
        from merge_tushare_tables import NO_MONEY_CONVERT
        assert "fina_indicator_clean" in NO_MONEY_CONVERT


class TestTsToLocal:
    @pytest.mark.parametrize(("ts", "expect"), [
        ("600519.SH", "sh600519"),
        ("000001.SZ", "sz000001"),
        ("430047.BJ", "bj430047"),
        ("600519.HK", None),      # 港股不支持
        ("600519", None),          # 无后缀
        ("600519.XY", None),       # 非法后缀
        ("", None),
    ])
    def test_代码转换(self, ts: str, expect: str | None) -> None:
        assert ts_to_local(ts) == expect


class TestVerifyBlocking:
    """「零异常比异常很多更该警惕」—— 缺失必须算阻塞。"""

    def test_缺失表的blocking为True(self) -> None:
        r = check_one("no_such_table_xyz", verbose=False)
        assert r["status"] == "缺失"
        assert r["blocking"] is True, (
            "缺失表若不blocking，main() 只按 sus 判退出码 → "
            "会输出「全绿」却漏掉整张表")

    def test_真实表通过(self) -> None:
        p = ROOT / "runtime" / "tushare" / "adj_factor.parquet"
        if not p.exists():
            pytest.skip("adj_factor 未下载")
        r = check_one("adj_factor", verbose=False)
        assert r["status"] == "OK", r.get("issues")


class TestPagingTermination:
    """分页只以「返回 0 行」为终止条件。"""

    def test_不足一页时仍探测下一页(self) -> None:
        """页大小可能不一致（第1页2000/第2页500/第3页还有），
        按「不足一页就停」会漏数据。"""
        import tushare_paging

        # ⚠️ 各页的值必须**互不重复** —— fetch_paged 末尾会 drop_duplicates，
        #    用 range() 造数据会因值重叠被误判为重复行。
        pages = {
            0: pd.DataFrame({"v": range(0, 500)}),        # 首页不足一页
            500: pd.DataFrame({"v": range(500, 800)}),   # 还有下一页
            800: pd.DataFrame(),# 到末尾
        }
        calls: list[int] = []

        def fake_call(token, api, params, retry=3):
            off = params.get("offset", 0)
            calls.append(off)
            return pages.get(off, pd.DataFrame())

        orig = tushare_paging.call
        tushare_paging.call = fake_call
        try:
            out = tushare_paging.fetch_paged(
                "tk", "api", {}, lambda: None, page_size=2000, max_pages=10)
        finally:
            tushare_paging.call = orig

        assert len(out) == 800, f"应拉到 800 行，实际 {len(out)}"
        assert calls == [0, 500, 800]

    def test_撞页数上限抛错(self) -> None:
        import tushare_paging

        def fake_call(token, api, params, retry=3):
            return pd.DataFrame({"v": range(params.get("limit", 10))})

        orig = tushare_paging.call
        tushare_paging.call = fake_call
        try:
            with pytest.raises(RuntimeError, match="仍未到末尾"):
                tushare_paging.fetch_paged(
                    "tk", "api", {}, lambda: None,
                    page_size=10, max_pages=3)
        finally:
            tushare_paging.call = orig
