"""全量评估驱动的算式与契约守护（成本映射/档映射/池掩码/分布裁决/模板列头）。

为什么必须有
------------
全量单的每个数字都站在四条可手算的线与两份模板契约上：引擎成本映射必须
逐位=config 20bp/往返、缓冲档必须逐档真异池、build_universe 池掩码必须真淘汰
（次新/ST）、judge 必须按分布裁决、报告模板强制列头缺一不可。夹具全人造。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from low_turnover_full_report import LABEL, render_full  # noqa: E402
from run_low_turnover_full import (  # noqa: E402
    N_PICKS,
    cost_twenty_bp,
    judge_dict,
    universe_mask,
)

from factor_lab.config import DEFAULT_COST  # noqa: E402


class TestCostMapping:
    def test_引擎成本逐位等于config二十bp往返(self) -> None:
        c = cost_twenty_bp()
        assert c.buy == 0.00075 and c.sell == 0.00125
        assert c.round_trip == DEFAULT_COST.round_trip == 0.002, (
            "裁定③：20bp/往返单收，映射失准即结论级错误")

    def test_档映射逐档真异池(self) -> None:
        assert N_PICKS == {0.0: 30, 0.1: 33, 0.2: 36}   # 30 + max(1, ceil(30×pct))


class TestUniverseMask:
    """build_universe 池掩码手锚：次新与 ST 必须真被淘汰（还原必失败对象）。"""

    @staticmethod
    def _fixtures():
        cal = pd.bdate_range("2023-02-01", periods=280)
        codes = ["sh600001", "sh600002", "sh600003"]
        rows = [{"code": c, "date": d, "close": 10.0, "amount": 1e8}
                for c in codes for d in cal]
        slim = pd.DataFrame(rows)
        info = pd.DataFrame({
            "code": codes, "name": ["旧股AAA", "次新BBB", "*ST CCC"],
            "list_date": ["20100101", "20231201", "20100101"],
        })
        z = pd.DataFrame(1.0, index=cal, columns=codes)
        return slim, info, z, cal

    def test_池掩码保留旧股淘汰次新与ST(self) -> None:
        slim, info, z, cal = self._fixtures()
        mask = universe_mask(slim, z, info, trade_cal=cal)
        assert bool(mask["sh600001"].all()), "满龄旧股应全保留"
        assert not bool(mask["sh600002"].any()), "上市 250 交易日内次新必须淘汰"
        assert not bool(mask["sh600003"].any()), "ST 名称票必须淘汰"
        assert list(mask.index) == list(cal) and list(mask.columns) == info["code"].tolist()

    def test_掩码外选股面置Nan不进排名(self) -> None:
        slim, info, z, cal = self._fixtures()
        masked = z.where(universe_mask(slim, z, info, trade_cal=cal))
        assert masked["sh600002"].isna().all()
        assert masked["sh600001"].notna().all()


class TestJudgeWiring:
    def test_全胜窗口判可用(self) -> None:
        j = judge_dict([{"超额": 0.01}] * 7, "t")
        assert j["可用"] and j["原因"] == "全部判据通过", j
        assert j["窗口数"] == 7 and j["胜率"] == 1.0

    def test_胜率与最差双破必须点名(self) -> None:
        rows = [{"超额": 0.01}] * 4 + [{"超额": -0.06}] * 3
        j = judge_dict(rows, "t")
        assert not j["可用"]
        assert "胜率" in j["原因"] and "最差窗口" in j["原因"], j["原因"]

    def test_窗口不足必须点名(self) -> None:
        j = judge_dict([{"超额": 0.01}] * 3, "t")
        assert not j["可用"] and "窗口数" in j["原因"]


class TestReportContract:
    def test_模板强制列头缺一不可(self) -> None:
        res = {"windows": [], "judgments": [], "guards": [], "segments": [],
               "layers": [],
               "deflated": {"n_trials": 6, "rows": [
                   {"name": "rev5-lag1-b0%", "sr_annual": 1.0, "sr0": 0.5,
                    "dsr": 0.9, "n_obs": 7}]}}
        md = render_full(res)
        for head in ("口径三件套", "价格口径", "换手单位", "年化方式",
                     "①年化口径", "②成本假设", "③基准", "④多空/多头",
                     "裁定采纳留痕", "护栏", "判据分布表", "三段子区间",
                     "层内混淆检验", "多重比较", "Deflated Sharpe", LABEL,
                     "不构成投资建议"):
            assert head in md, head
