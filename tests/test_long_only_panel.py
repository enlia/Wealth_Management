"""long_only_panel 面板组装的守护测试。

断言都按「还原 bug 版必须失败」设计（CODE_TRUST P23：声称有保护就要测它）：
  · 对齐检查只比第一个因子      → 第二个因子索引不一致必须报错
  · 一个因子都没有就打印「对齐通过」→ 必须报错
  · 价格面板去重只落局部变量    → 去重必须落在返回值上
全部打桩取数（monkeypatch），不读 market.db：跑得快、结果可复现。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

import long_only_panel  # noqa: E402
from long_only_panel import assemble_frames, build_panel  # noqa: E402

DATES = pd.bdate_range("2019-01-02", periods=6)
CODE = "sh600000"


def _long_frame() -> pd.DataFrame:
    return pd.DataFrame({"code": CODE, "date": DATES, "close_adj": 1.0})


def _patch(monkeypatch, factor_dates: dict) -> None:
    """打桩取数与因子计算：每个因子返回各自的日期集合（None = 算不出）。"""
    monkeypatch.setattr(
        long_only_panel, "load_long_chunked",
        lambda codes, start, end, years=None, verbose=True: _long_frame())

    def fake_compute(name, long_df):
        dts = factor_dates[name]
        if dts is None:
            return None
        idx = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(dts), [CODE] * len(dts)], names=["date", "asset"])
        return pd.Series(1.0, index=idx)

    monkeypatch.setattr(long_only_panel, "compute_factor", fake_compute)


class TestAssembleFrames:
    def test_重复日期只留最后一份且去重落在返回值(self):
        """跨年重叠片段拼接时：同一日期取后一份，返回值本身已去重。"""
        f1 = pd.DataFrame({"close_adj": [1.0, 2.0]},
                          index=pd.to_datetime(["2019-01-02", "2019-01-03"]))
        f2 = pd.DataFrame({"close_adj": [9.0, 4.0]},
                          index=pd.to_datetime(["2019-01-03", "2019-01-04"]))
        out = assemble_frames([f1, f2], "price")
        assert out.index.is_unique, "返回值必须已去重"
        assert len(out) == 3
        assert float(out.loc[pd.Timestamp("2019-01-03"), "close_adj"]) == 9.0

    def test_无重复时原样返回(self):
        f1 = pd.DataFrame({"close_adj": [1.0]},
                          index=pd.to_datetime(["2019-01-02"]))
        out = assemble_frames([f1], "price")
        assert list(out.index) == list(f1.index)


class TestBuildPanelAlignment:
    def test_第二个因子索引不一致必须报错(self, monkeypatch):
        """还原 bug 版只拿第一个因子与 price 比对 → 本用例会静默通过。"""
        _patch(monkeypatch, {"a": DATES, "b": DATES[:-1]})
        with pytest.raises(ValueError, match="因子面板 b"):
            build_panel([CODE], ["a", "b"], "20190101", "20191231")

    def test_无因子面板必须报错(self, monkeypatch):
        """还原 bug 版拿 price 与自身比对后照样打印「对齐通过」。"""
        _patch(monkeypatch, {"a": None})
        with pytest.raises(ValueError, match="没有任何因子面板"):
            build_panel([CODE], ["a"], "20190101", "20191231")

    def test_索引一致时返回去重后的价格面板(self, monkeypatch):
        _patch(monkeypatch, {"a": DATES})
        out = build_panel([CODE], ["a"], "20190101", "20191231")
        px = out["__price__"]
        assert px.index.is_unique, "返回的价格面板必须已去重"
        assert px.index.equals(out["a"].index), "两面板日期索引必须一致"
        assert len(px) == len(DATES)