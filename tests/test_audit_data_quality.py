"""``audit_data_quality`` 判定链路的行为测试（入口行为，不测内部实现）。

为什么单独一个文件
------------------
``tests/test_market_rules.py`` 测的是 market_rules 纯函数加一份最小复刻组合，
刻意不 import research/ 脚本。但审计脚本自身的组装 —— 板块判定取自哪里、
容差怎么取、缺口日怎么处理、残留归因怎么分桶 —— 一直没人从入口测过。
实测踩过：脚本内置的第二套 board_of/TOL 与 market_rules 漂移，
sz302132 的 20% 涨停被按主板 ±10% 误判、sh689009 被按前缀切片判成主板。
本文件直接 import 脚本本体，把入口行为钉死。

运行
----
  uv run pytest tests/test_audit_data_quality.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "scripts"))

import audit_data_quality as audit  # noqa: E402


def _panel(code: str, closes: list[float], start: str = "2024-01-02") -> pd.DataFrame:
    """单列价格面板：连续工作日，行数 = len(closes)。"""
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame({code: closes}, index=idx)


def _flat_then(code: str, base: float, last: float, n: int = 8) -> pd.DataFrame:
    """前 n-1 天横盘、最后一天跳到 last 的面板（跳变日 = 第 n 个交易日）。"""
    return _panel(code, [base] * (n - 1) + [last])


class TestCleanReturnsJudgement:
    """审计入口的超限判定行为：板块限幅、价格感知容差、缺口不跨日。"""

    def test_创业板302段按20percent限幅(self) -> None:
        """sz302132（302 段）+20.0% 封板属规则内，不得判超限。"""
        px = _flat_then("sz302132", 10.00, 12.00)
        clean, bad = audit.clean_returns(px, quoted_prev=px.shift(1), timeline=px)
        assert bool(bad.iloc[7]["sz302132"]) is False

    def test_科创板689段按20percent限幅(self) -> None:
        """sh689009（689 CDR 段）+15% 在 20% 限内，不得判超限。"""
        px = _flat_then("sh689009", 10.00, 11.50)
        clean, bad = audit.clean_returns(px, quoted_prev=px.shift(1), timeline=px)
        assert bool(bad.iloc[7]["sh689009"]) is False

    def test_主板15percent必须判超限(self) -> None:
        px = _flat_then("sz000001", 10.00, 11.50)
        clean, bad = audit.clean_returns(px, quoted_prev=px.shift(1), timeline=px)
        assert bool(bad.iloc[7]["sz000001"]) is True
        assert np.isnan(clean.iloc[7]["sz000001"])  # 超限日置 NaN，不是置 0

    def test_低价股真实封板不得误判(self) -> None:
        """0.26 → 0.29 是精确涨停（+11.54%），动态容差必须放行。

        固定容差 1.005 会把它判成超限（Q2：低价股误判率 87%）。
        """
        px = _flat_then("sz000001", 0.26, 0.29)
        clean, bad = audit.clean_returns(px, quoted_prev=px.shift(1), timeline=px)
        assert bool(bad.iloc[7]["sz000001"]) is False

    def test_注册制新股无限幅窗口内不判超限(self) -> None:
        """全面注册制（2023-02-17）后主板新股上市前 5 日无涨跌幅限制。"""
        px = _panel("sh603124", [10.0, 12.3, 15.0, 18.0, 21.0, 25.0, 30.0, 34.0],
                    start="2025-03-20")
        clean, bad = audit.clean_returns(px, quoted_prev=px.shift(1), timeline=px)
        # 第 2~5 日（窗口内）大幅波动都不得判超限
        assert [bool(bad.iloc[i]["sh603124"]) for i in (1, 2, 3, 4)] == [False] * 4
        # 第 8 日（窗口外）+13% 小于 ±10%? 34/30-1 = 13.3% → 超限
        assert bool(bad.iloc[7]["sh603124"]) is True

    def test_上市首日单列为异常标记(self) -> None:
        px = _panel("sz000001", [10.00, 10.20, 10.40])
        clean, bad = audit.clean_returns(px, quoted_prev=px.shift(1), timeline=px)
        assert bool(bad.iloc[0]["sz000001"]) is True      # 首日无前收，单列
        assert bool(bad.iloc[1]["sz000001"]) is False
        assert np.isnan(clean.iloc[0]["sz000001"])

    def test_缺口相邻日不算单日涨跌(self) -> None:
        """停牌（NaN）两侧的收益算不出，不得跨缺口判超限。"""
        px = _panel("sz000001", [10.00, np.nan, 13.00, 13.10])
        clean, bad = audit.clean_returns(px, quoted_prev=px.shift(1), timeline=px)
        assert bool(bad.iloc[2]["sz000001"]) is False   # 10 → 13 是缺口两侧，不是单日 +30%
