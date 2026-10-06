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

    def test_无因子面板必须报错(self, monkeypatch, capsys):
        """还原 bug 版拿 price 与自身比对后照样打印「对齐通过」。
        跳过还必须打印「跳过 X + 原因 Y」（ENGINEERING §四 / DATA_SOURCE S2：
        0 行不是「正常为空」）。"""
        _patch(monkeypatch, {"a": None})
        with pytest.raises(ValueError, match="没有任何因子面板"):
            build_panel([CODE], ["a"], "20190101", "20191231")
        out = capsys.readouterr().out
        assert "跳过因子 a" in out and "原因" in out, (
            f"静默跳过不允许，必须打印因子名与原因：{out!r}")

    def test_零行片段必须打印跳过与原因(self, monkeypatch, capsys):
        monkeypatch.setattr(
            long_only_panel, "load_long_chunked",
            lambda codes, start, end, years=None, verbose=True: pd.DataFrame())
        with pytest.raises(ValueError):
            build_panel([CODE], ["a"], "20190101", "20191231")
        out = capsys.readouterr().out
        assert "跳过整年" in out and "0 行" in out, (
            f"取数 0 行必须打印年份与原因：{out!r}")

    def test_索引一致时返回去重后的价格面板(self, monkeypatch):
        _patch(monkeypatch, {"a": DATES})
        out = build_panel([CODE], ["a"], "20190101", "20191231")
        px = out["__price__"]
        assert px.index.is_unique, "返回的价格面板必须已去重"
        assert px.index.equals(out["a"].index), "两面板日期索引必须一致"
        assert len(px) == len(DATES)


class TestPriceWriteBack:
    def test_多片段重叠日期拼接后去重必须落在返回值(self, monkeypatch):
        """R3b 的真实触发形态：**多个年份片段各含同一批日期** ⇒ 拼接出重复日期。

        单片段打桩造不出该形态（价格片段逐年的裁剪区间在真实日历上天然互斥），
        故打桩裁剪边界：`long_only_panel.pd` 换成替身，ys 输入（…0101 结尾）
        给 `Timestamp.min`、其余给 `Timestamp.max` —— 每个年份片段都保留
        全部日期，3 个片段（2018/2019/2020）各 4 日 = 12 行进拼接。
        （用例 start/end 固定 …0101 / …1231 结尾，与替身判定规则绑定。）

        还原「价格去重只落局部变量、返回值还是旧对象」的写法：
        本用例 **FAILED**（返回的价格面板残留重复日期）；
        现写法（去重写回返回值）：**PASSED**。双向均已实跑留证。
        """
        dates = pd.to_datetime(["2019-01-02", "2019-01-03",
                                "2020-01-02", "2020-01-03"])
        call_no = {"n": 0}

        def fake_loader(codes, start, end, years=None, verbose=True):
            call_no["n"] += 1
            return pd.DataFrame({"code": CODE, "date": dates,
                                 "close_adj": [float(call_no["n"])] * 4})

        def fake_compute(name, long_df):
            idx = pd.MultiIndex.from_arrays([dates, [CODE] * 4],
                                            names=["date", "asset"])
            return pd.Series(1.0, index=idx)

        class _BroadPd:
            """build_panel 对 pd 只用 Timestamp（裁剪边界）与 concat。"""

            concat = staticmethod(pd.concat)

            @staticmethod
            def Timestamp(s):
                return (pd.Timestamp.min if str(s).endswith("0101")
                        else pd.Timestamp.max)

        monkeypatch.setattr(long_only_panel, "pd", _BroadPd)
        monkeypatch.setattr(long_only_panel, "load_long_chunked", fake_loader)
        monkeypatch.setattr(long_only_panel, "compute_factor", fake_compute)

        out = build_panel([CODE], ["a"], "20180101", "20201231")
        px = out["__price__"]
        assert call_no["n"] == 3, "应产 3 个年份片段（2018/2019/2020）"
        assert px.index.is_unique, (
            "去重必须落在返回值：多片段含重叠日期时，返回的价格面板仍带重复日期")
        assert sorted(px.index) == sorted(dates), f"应剩 4 个日期，实际 {len(px)}"
        assert float(px.loc[pd.Timestamp("2019-01-02"), CODE]) == 3.0, (
            "重复日期按 keep='last' 保留最后片段的值")
