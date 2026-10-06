"""long_only_data 取数层的守护测试（去重规则与年份切片边界）。

断言都按「还原缺陷版必须失败」设计：
  · keep='last' 一刀切去重   → 同键不同值必须报错，整行重复才允许去掉
打桩/合成数据，不读 market.db：跑得快、结果可复现。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

import long_only_data  # noqa: E402
from long_only_data import dedupe_long  # noqa: E402


class TestDedupeLong:
    def test_整行重复允许去掉(self):
        """整行完全相同 = 分块拼接产物，是重复行，该去重。"""
        df = pd.DataFrame({"code": ["sh600000"] * 3,
                           "date": pd.to_datetime(["2019-01-02"] * 3),
                           "close_adj": [1.0, 1.0, 1.0]})
        out = dedupe_long(df)
        assert len(out) == 1

    def test_同键不同值必须报错(self):
        """同 (code, date) 取值不同是数据冲突（S5/S8）——
        还原 keep='last' 会静默丢掉冲突值、价格冲突直接进收益计算。"""
        df = pd.DataFrame({"code": ["sh600000"] * 2,
                           "date": pd.to_datetime(["2019-01-02"] * 2),
                           "close_adj": [1.0, 2.0]})
        with pytest.raises(ValueError, match="取值不同"):
            dedupe_long(df)

    def test_重复与冲突并存时报错且给出计数与示例(self):
        df = pd.DataFrame({"code": ["sh600000"] * 3,
                           "date": pd.to_datetime(
                               ["2019-01-02", "2019-01-02", "2019-01-02"]),
                           "close_adj": [1.0, 1.0, 3.0]})
        with pytest.raises(ValueError) as ei:
            dedupe_long(df)
        msg = str(ei.value)
        assert "1 个键" in msg, f"错误信息应给出冲突键计数：{msg}"
        assert "冲突键示例" in msg, f"错误信息应给出冲突键示例：{msg}"

    def test_无重复时按code_date排序返回(self):
        df = pd.DataFrame({"code": ["sh600001", "sh600000"],
                           "date": pd.to_datetime(["2019-01-03", "2019-01-02"]),
                           "close_adj": [1.0, 2.0]})
        out = dedupe_long(df)
        assert list(out["code"]) == ["sh600000", "sh600001"]
        assert len(out) == 2


class TestLoadLongChunkedBounds:
    def test_years不覆盖start所在年份必须报错(self):
        """years 缺 start 所在年份时，[start, 起始年元旦] 会被整段漏读；
        还原「静默缺年」的写法必须失败。"""
        with pytest.raises(ValueError, match="未覆盖"):
            long_only_data.load_long_chunked(
                ["sh600000"], "20180102", "20191231", years=(2019,))

    def test_相邻切片下界逐年前移且互不重叠(self, monkeypatch):
        """下界统一用 start 会让 2019 切片重读 2018 切片已读过的预热行
        （行数重复 + IO 随切片数放大）—— 下界必须逐年抬到该年元旦。"""
        captured: list[list[str]] = []

        class _FakeCon:
            def close(self) -> None:
                pass

        class _FakeSqlite:
            @staticmethod
            def connect(*a, **k):
                return _FakeCon()

        def fake_read_sql(q, con, params):
            captured.append([params[0], params[1]])
            return pd.DataFrame(columns=["code", "date", "open", "high",
                                         "low", "close", "close_adj",
                                         "vol", "amount"])

        monkeypatch.setattr(long_only_data, "sqlite3", _FakeSqlite)
        monkeypatch.setattr(long_only_data.pd, "read_sql", fake_read_sql)
        out = long_only_data.load_long_chunked(
            ["sh600000"], "20180102", "20191231", years=(2018, 2019),
            verbose=False)
        assert captured == [["20180102", "20181231"], ["20190101", "20191231"]], (
            f"切片下界应逐年前移且互不重叠，实际 {captured}")
        assert out.empty, "打桩只回空表，本用例只看切片边界"