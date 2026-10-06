"""落盘类型归一（normalize_types）的判据测试。

核心判据：**日期列必须是数值类型，且 int 比较能匹配到行**。
这是实测踩过的最大一处静默失效 —— 19 张表日期列存成 string，
下游 `df[df.trade_date == 20240102]` 恒返回 0 行且不报错。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from tushare_paths import DATE_COLS, normalize_types  # noqa: E402


class TestNormalizeTypes:
    def test_字符串日期转int32(self):
        df = pd.DataFrame({"trade_date": ["20240102", "20240103"]})
        out = normalize_types(df)
        assert out["trade_date"].dtype == "int32"
        assert len(out[out["trade_date"] == 20240102]) == 1, \
            "int 比较必须能匹配到行 —— 这是踩过的核心坑"

    def test_已为int不重复处理(self):
        df = pd.DataFrame({"ann_date": [20240102, 20240103]})
        assert normalize_types(df)["ann_date"].dtype == "int32"

    def test_空值不报错且保持缺失(self):
        """⚠️ 实测踩过：stk_managers.end_date 有 73,575 个空值
        （在任高管本就无离任日期，是业务语义不是数据损坏）。
        用 numpy int32 会抛 `cannot convert NA to integer` 并整表迁移失败。"""
        df = pd.DataFrame({"end_date": ["20290105", None, "20270909"]})
        out = normalize_types(df)
        assert out["end_date"].isna().sum() == 1
        assert len(out[out["end_date"] == 20290105]) == 1

    @pytest.mark.parametrize("empty", ["", "None", "nan", "NaT", "  "])
    def test_各种空值写法都识别(self, empty):
        """Tushare 的空值写法不统一，漏一个就会抛异常中断整表下载。"""
        df = pd.DataFrame({"cal_date": ["20240102", empty]})
        out = normalize_types(df)
        assert out["cal_date"].isna().sum() == 1

    def test_带空格会清理(self):
        df = pd.DataFrame({"trade_date": [" 20240102 "]})
        assert normalize_types(df)["trade_date"].iloc[0] == 20240102

    def test_浮点日期有小数则报错(self):
        """浮点日期说明数据已经损坏（含时分秒），必须报错而不是静默取整。
        静默 round 会把 2024.75 也变成 2024 —— 一天丢两次。"""
        df = pd.DataFrame({"trade_date": [20240102.75]})
        with pytest.raises(ValueError, match="非整数小数"):
            normalize_types(df)

    def test_浮点整数日期正常转(self):
        df = pd.DataFrame({"trade_date": [20240102.0]})
        assert normalize_types(df)["trade_date"].iloc[0] == 20240102

    def test_不改动非日期列(self):
        df = pd.DataFrame({"ts_code": ["600519.SH"], "close": [1.5],
                           "trade_date": ["20240102"]})
        out = normalize_types(df)
        assert out["ts_code"].dtype == object
        assert out["close"].dtype == float

    def test_多个日期列同时处理(self):
        df = pd.DataFrame({"ann_date": ["20240102"], "end_date": ["20231231"],
                           "trade_date": ["20240102"]})
        out = normalize_types(df)
        for c in ("ann_date", "end_date", "trade_date"):
            assert out[c].dtype == "int32", f"{c} 未转换"

    def test_缺列不报错(self):
        df = pd.DataFrame({"close": [1.0]})
        assert len(normalize_types(df)) == 1

    def test_所有DATE_COLS都被覆盖(self):
        """防止新增日期列名时忘了加进 DATE_COLS —— 那是静默失效的温床。"""
        real = pd.DataFrame({c: ["20240102"] for c in DATE_COLS})
        out = normalize_types(real)
        for c in DATE_COLS:
            assert out[c].dtype == "int32", f"{c} 未被归一"


class TestStockedParquet:
    """真实存量文件必须全部是数值日期（回归防护）。"""

    def test_运行时parquet无string日期列(self):
        out_dir = ROOT / "runtime" / "tushare"
        if not out_dir.exists():
            pytest.skip("尚未下载数据")
        bad: list[str] = []
        for f in sorted(out_dir.glob("*.parquet")):
            sch = pq.read_schema(f)
            for c in DATE_COLS:
                if c in sch.names and str(sch.field(c).type) == "string":
                    bad.append(f"{f.name}:{c}")
        assert not bad, (
            f"以下文件的日期列仍是 string，int 比较会静默返回 0 行: {bad}")

    def test_stk_limit的int比较能命中(self):
        """实测踩过：stk_limit.trade_date 是 string，
        `df[df.trade_date == 20240102]` 恒返回 0 行 —— 不报错，
        只是「那天没有任何涨跌停数据」，看起来像正常结果。"""
        p = ROOT / "runtime" / "tushare" / "stk_limit.parquet"
        if not p.exists():
            pytest.skip("尚未下载 stk_limit")
        df = pd.read_parquet(p, columns=["trade_date"])
        assert df["trade_date"].dtype.kind in "iu", \
            f"trade_date 应为整数类型，实为 {df['trade_date'].dtype}"
        assert (df["trade_date"] == 20240102).sum() > 0, \
            "int 比较一天都匹配不到 —— 日期列类型仍是错的"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))