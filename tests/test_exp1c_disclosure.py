"""实验 1·C 线新判据代码的守护用例（披露日历级联 + 窗口卫生 + DSR 等效件）。

判据代码 = exp1c_bp_walkforward.py 的 apply_disclosure_calendar / window_hygiene /
deflated_sharpe。三条都属于「判据错了会安静输出合理数字」的高危面
（检验框架铁律·静默失效三类），故各配反向断言，还原形态必须 FAILED。
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

from exp1c_bp_walkforward import (  # noqa: E402
    apply_disclosure_calendar,
    deflated_sharpe,
    window_hygiene,
)

_PNL = pd.DataFrame({
    "sym": ["sh600000", "sh600001", "sh600002", "sh600003"],
    "report_period": pd.to_datetime(["2024-03-31"] * 4),
})
_DISC = pd.DataFrame({
    "sym_": ["sh600000", "sh600001", "sh600002"],
    "report_period": pd.to_datetime(["2024-03-31"] * 3),
    "opdate": pd.to_datetime(["2024-04-10", "2024-04-20", None]),
    "regdate": pd.to_datetime(["2024-04-15", None, "2024-04-25"]),
})


class Test披露日历级联:
    def test_ymd必须认数值型日期且坏值为NaT(self):
        """ts_ 表日期列是数值型（20240331 整数）——按字符串 to_datetime 会全体
        coerce NaT 而被 dropna 光（join 0 行、级联 100% 落兜底的假象）。"""
        from exp1c_bp_walkforward import _ymd
        got = _ymd(pd.Series([20240331, 20241231.0, None, "garbage"]))
        assert got.iloc[0] == pd.Timestamp("2024-03-31")
        assert got.iloc[1] == pd.Timestamp("2024-12-31")
        assert pd.isna(got.iloc[2])
        assert pd.isna(got.iloc[3]), "坏值必须显式 NaT，不得静默给默认日期"

    def test_ts_to_local双态映射(self):
        """库内 ts_ 表实测为前缀形（bj920000），tushare 原始形为后缀形
        （920000.BJ）——两态都须落到本机口径，漏一态会让 join 全空。"""
        from exp1c_bp_walkforward import _ts_to_local
        assert _ts_to_local("sh600519") == "sh600519"      # 前缀形直通
        assert _ts_to_local("600519.SH") == "sh600519"     # 后缀形走现成映射
        assert _ts_to_local("920680.BJ") == "bj920680"
        assert _ts_to_local("garbage") is None, "垃圾值必须显式 None（→被 dropna）"

    def test_cascade逐级延顺_op优先于reg再兜底80日(self):
        """级联已知真值（算式可复核，+1 日）：
        sh600000 有 opdate 04-10 ⇒ 04-11（虽 regdate 04-15 更晚不采）
        sh600001 只有 opdate 04-20 ⇒ 04-21
        sh600002 只有 regdate 04-25 ⇒ 04-26
        sh600003 全无   ⇒ 03-31 + 80 自然日 = 06-19
        """
        out = apply_disclosure_calendar(_PNL, _DISC, "cascade")
        got = out.set_index("sym")["available_date"]
        assert got["sh600000"] == pd.Timestamp("2024-04-11")
        assert got["sh600001"] == pd.Timestamp("2024-04-21")
        assert got["sh600002"] == pd.Timestamp("2024-04-26")
        assert got["sh600003"] == pd.Timestamp("2024-06-19")

    def test_cascade级序颠倒必须失败(self):
        """反向断言：把 regdate 当第一优先（级序颠倒）必须落红 ——
        登记日晚于披露日，颠倒会让因子提前进入回测（前视）。"""
        out = apply_disclosure_calendar(_PNL, _DISC, "cascade")
        got = out.set_index("sym")["available_date"]
        assert got["sh600000"] != pd.Timestamp("2024-04-16"), (
            "级序颠倒成 reg 优先 —— opdate+1 才是第一级")

    def test_registry只认登记日路径且无日期记录丢弃(self):
        out = apply_disclosure_calendar(_PNL, _DISC, "registry")
        assert set(out["sym"]) == {"sh600000", "sh600001", "sh600002"}, (
            "registry 口径必须丢弃无任何披露日的记录（sh600003）")
        got = out.set_index("sym")["available_date"]
        assert got["sh600002"] == pd.Timestamp("2024-04-26")
        assert (out["available_date"] != pd.Timestamp("2024-06-19")).all(), (
            "registry 口径混进了 80 日启发式 —— 必须只走登记日路径")

    def test_heur80全量end_date加80自然日(self):
        out = apply_disclosure_calendar(_PNL, _DISC, "heur80")
        assert (out["available_date"] == pd.Timestamp("2024-06-19")).all()


class Test窗口卫生双证:
    def test重叠测试段必须抛错(self):
        d = pd.date_range("2020-01-01", periods=40, freq="B")
        # 第二窗测试段与第一窗测试段重叠 5 日（d[25:30]）
        wins = [(list(d[:20]), list(d[20:30])), (list(d[:20]), list(d[25:35]))]
        with pytest.raises(AssertionError):
            window_hygiene(wins, "污染形态")

    def test正常双窗必须零重叠零越界且报拼接空窗(self):
        d = pd.date_range("2020-01-01", periods=44, freq="B")
        wins = [(list(d[:12]), list(d[12:24])), (list(d[:24]), list(d[24:36]))]
        out = window_hygiene(wins, "正常形态")
        assert out["测试段重叠日"] == 0
        assert out["训练/测试越界"] == 0
        assert out["拼接段空窗日"] == [0], "相邻测试段应首尾相接（空窗 0）"

    def test训练测试越界必须抛错(self):
        d = pd.date_range("2020-01-01", periods=30, freq="B")
        wins = [(list(d[:20]), list(d[15:30]))]      # 训练含测试头 5 日 = 前视
        with pytest.raises(AssertionError):
            window_hygiene(wins, "越界形态")


class TestDeflatedSharpe等效件:
    def _ar1_series(self, mu: float, n: int = 400) -> pd.Series:
        rng = np.random.default_rng(7)
        return pd.Series(mu + rng.normal(0, 0.01, n))

    def test_正超额序列DSR高于零超额序列(self):
        hi = deflated_sharpe(self._ar1_series(0.002), n_trials=2)
        lo = deflated_sharpe(self._ar1_series(0.0), n_trials=2)
        assert hi["DSR"] > lo["DSR"], "均值更高的序列 DSR 必须更高"

    def test_多重比较惩罚必须压低DSR(self):
        one = deflated_sharpe(self._ar1_series(0.001), n_trials=2)
        many = deflated_sharpe(self._ar1_series(0.001), n_trials=13)
        assert many["DSR"] < one["DSR"], (
            "试验数 2→13 SR0 必须抬升、DSR 必须被压低（多重比较惩罚）")
        assert many["SR0"] > one["SR0"]

    def test_N1必须显式报错而非假绿(self):
        """反向断言：N=1 处 SR0 无定义（Φ⁻¹(0)=−∞ 会让 DSR 恒 1 = 假绿），
        必须显式 raise，不得返回数值。"""
        s = self._ar1_series(0.001)
        with pytest.raises(ValueError):
            deflated_sharpe(s, n_trials=1)

    def test_样本不足必须显式报出(self):
        r = deflated_sharpe(pd.Series([0.1, 0.2]), n_trials=3)
        assert np.isnan(r["DSR"]), "T<3 不得悄悄给出 DSR 数值"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
