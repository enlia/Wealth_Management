"""涨跌停约束（tradability.py）的判据测试。

这个模块此前**零测试覆盖**，而它的存在意义就是消除静默偏差 ——
一个自己会静默失效的模块最危险（review2026-10-06 报为 BLOCK）。

每个断言对应 review 实测出的具体失败路径。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.analysis.tradability import (  # noqa: E402
    apply_to_buy,
    coverage_report,
    limit_masks,
    tushare_to_local_code,
)


class TestCodeConversion:
    def test_ts_code转本机口径(self):
        """⚠️ **交易所是后缀不是前缀**（review 实测 BLOCK）。
        初版写 slice(0,2)+slice(2,8)，产出 '000001.S'，
        与本机代码交集为 0 —— 涨跌停约束 100% 失效却不报错。
        """
        s = pd.Series(["600519.SH", "000001.SZ", "300750.SZ", "000003.S"])
        got = tushare_to_local_code(s).tolist()
        assert got == ["sh600519", "sz000001", "sz300750", "sz000003"]

    def test_退市股后缀S也能处理(self):
        assert tushare_to_local_code(pd.Series(["000003.S"])).iloc[0] == "sz000003"

    def test_北交所不崩(self):
        assert tushare_to_local_code(pd.Series(["920000.BJ"])).iloc[0] == "bj920000"

    def test_未知后缀必须报错(self):
        """静默放过未知后缀会让该批股票列名对不上、约束失效 ——
        这正是本模块要消除的那类静默偏差，不能自己犯。"""
        with pytest.raises(ValueError, match="未知交易所后缀"):
            tushare_to_local_code(pd.Series(["600519.XX"]))

    def test_缺后缀必须报错(self):
        with pytest.raises(ValueError, match="缺少交易所后缀"):
            tushare_to_local_code(pd.Series(["600519"]))


class TestLimitMasks:
    @staticmethod
    def _panel():
        """构造 2 日 × 3 只的价格/涨跌停价。

        第 0 日：sh600519 收10.00==涨停价（封板）
                  sz000001 收 20.00 < 涨停 22.00（未封板）
                  sz300750 收 30.00 < 涨停 33.00（未封板）
        第 1 日：sh600519 收 11.00 > 跌停价 9.90（未封跌停）
                  sz000001 收 22.00 == 涨停价（封板）
                  sz300750 收 30.00== 跌停价 30.00（封跌停）
        """
        idx = pd.to_datetime(["2023-01-03", "2023-01-04"])
        cols = ["sh600519", "sz000001", "sz300750"]
        close = pd.DataFrame([[10.0, 20.0, 30.0],
                              [11.0, 22.0, 30.0]], index=idx, columns=cols)
        up = pd.DataFrame([[10.0, 22.0, 33.0],
                           [11.0, 22.0, 33.0]], index=idx, columns=cols)
        dn = pd.DataFrame([[9.0, 18.0, 27.0],
                           [9.9, 19.8, 30.0]], index=idx, columns=cols)
        return close, up, dn

    def test_识别封涨停(self):
        close, up, dn = self._panel()
        lu, _ = limit_masks(close, up, dn)
        assert bool(lu.iloc[0, 0]) is True, "收盘==涨停价应判封板"
        assert bool(lu.iloc[1, 1]) is True
        assert bool(lu.iloc[0, 1]) is False, "20<22 未封板"

    def test_同时判跌停(self):
        """⚠️ **必须同时判涨停和跌停**（DATA_QUALITY Q12）。
        初版只判涨停（`close >= up_limit`），
        跌停日被当成正常交易日 ——
        实测 343 条「真污染」逐条核查发现绝大多数是**真实跌停**。
        """
        close, up, dn = self._panel()
        lu, ld = limit_masks(close, up, dn)
        assert bool(ld.iloc[1, 2]) is True, "收盘==跌停价应判封跌停"
        assert bool(ld.iloc[1, 0]) is False, "11>9.9 未封跌停"
        # 同一天可能既封涨停也可能封跌停（极端情形），两者都要独立判
        assert bool(lu.iloc[1, 1]) and bool(lu.iloc[1, 2]) is False

    def test_未覆盖的格子视为不可交易(self):
        """NaN → True（不可交易）而非 False。
        理由：填False 会让「没数据的票」变成可随意买卖。"""
        close, up, dn = self._panel()
        up2 = up.copy()
        up2.iloc[0, 0] = np.nan
        lu, _ = limit_masks(close, up2, dn)
        assert bool(lu.iloc[0, 0]) is True

    def test_面板未对齐必须报错(self):
        """⚠️ review 实测：索引/列名完全错位时覆盖率 0%，
        而 fillna(True) 会把**全部**股票判成不可交易（一只都买不了）
        且不报错。故必须显式抛错。"""
        close, _, _ = self._panel()
        bad = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        with pytest.raises(ValueError, match="覆盖率异常"):
            limit_masks(close, bad, bad)

    def test_传后复权价会被检出(self):
        """review 实测：传close_adj 时封涨停率从 0.79% 稀释到 0.25%，
        不报任何错。这里用「close 数值整体偏移」模拟复权因子。"""
        close, up, dn = self._panel()
        adj = close * 0.91
        lu_real, _ = limit_masks(close, up, dn)
        lu_adj, _ = limit_masks(adj, up, dn)
        # 覆盖率都正常，所以不抛错 —— 但封板数应当明显变少
        assert lu_real.to_numpy().sum() > lu_adj.to_numpy().sum()


class TestApplyToBuy:
    def test_全部可买时不变(self):
        held = np.array([[0, 1, 2]])
        tradable = np.ones((1, 5), dtype=bool)
        out = apply_to_buy(held, tradable, 3)
        assert out.tolist() == [[0, 1, 2]]

    def test_封涨停的用池内顺位补(self):
        """⚠️ **补位必须限定在候选池内**（review 实测 MAJOR）。
        初版没传 pool，退化成从 range(N) 全市场取 ——
        被顶掉的持仓换成「全市场任意可买股票」，因子信号被彻底丢弃。
        """
        held = np.array([[0, 1, 2]])
        tradable = np.ones((1, 6), dtype=bool)
        tradable[0, 0] = False               # 0 封涨停
        pool = np.array([[0, 1, 2, 3, 4]])   # 候选池
        out = apply_to_buy(held, tradable, 3, pool=pool)
        assert out.tolist() == [[1, 2, 3]], "应从候选池补 3，不是 5"

    def test_不传pool时从全市场补(self):
        """向后兼容：不传 pool 时用全市场，但**必须仍不抛错**。"""
        held = np.array([[0, 1, 2]])
        tradable = np.ones((1, 6), dtype=bool)
        tradable[0, 0] = False
        out = apply_to_buy(held, tradable, 3)
        assert len(out[0]) == 3 and 0 not in out[0]

    def test_可买不足必须报错而非补齐(self):
        """⚠️ **不能用列索引占位补齐**（review 实测 BLOCK）。
        列索引 0 是**真实存在的股票**，填 0 等于「凭空持有它」。
        初版直接赋值会 `ValueError: could not broadcast (29,) into (30,)`，
        现在显式抛错并说明处理方式。
        """
        held = np.array([[0, 1, 2]])
        tradable = np.zeros((1, 5), dtype=bool)
        tradable[0, 0] = True                # 只有 1 只可买
        pool = np.array([[0, 1, 2]])
        with pytest.raises(ValueError, match="仅 1/3 只可买"):
            apply_to_buy(held, tradable, 3, pool=pool)

    def test_多日逐日独立处理(self):
        held = np.array([[0, 1, 2], [0, 1, 2]])
        tradable = np.ones((2, 6), dtype=bool)
        tradable[0, 0] = False
        tradable[1, 1] = False
        pool = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
        out = apply_to_buy(held, tradable, 3, pool=pool)
        assert out[0].tolist() == [1, 2, 3]
        assert out[1].tolist() == [0, 2, 3]


class TestCoverageReport:
    def test_能区分真封板与缺数据(self):
        """⚠️ **实测踩过（review MAJOR）**：
        北交所 `stk_limit` 覆盖不全，整体封板率虚高到 **37%**，
        而真实 A 股封涨停率只有 0.5%~0.8%。
        拆开看：真封板 0.50% + 缺数据被当不可交易 32%。
        若不拆开，会得出「A 股三分之一的日子在涨停」的错误结论。
        故 `coverage_report` 必须能分别报出两者。
        """
        idx = pd.to_datetime(["2023-01-03", "2023-01-04"])
        cols = [f"sh60000{i}" for i in range(1, 5)] + \
               [f"bj92000{i}" for i in range(1, 5)]
        close = pd.DataFrame(10.0, index=idx, columns=cols)
        up = pd.DataFrame(11.0, index=idx, columns=cols[:4])   # 北交所未覆盖
        txt = coverage_report(close, up, "test")
        assert "真封涨停率" in txt
        assert "北交所" in txt and "沪主板" in txt

        # ⚠️ **核心判据：零覆盖列绝不能被算成「100% 封板」**。
        #   若混进去，北交所那几列会以「全部不可交易」的形式
        #   把整体封板率虚高 —— 实测该缺陷曾让 2023 年整体封板率
        #   报成 37%（真值0.50%）。
        #   正确做法：显式报出「零覆盖列」并从板块统计中排除。
        assert "零涨跌停价列" in txt, \
            "必须显式报出零覆盖列，否则缺数据会被当成封板"
        assert "4/8" in txt, f"零覆盖列数应报出 4/8，实际: {txt}"

        # 沪主板全覆盖部分：真封涨停率应为 0（close=10 远低于 up=11）
        line = [ln for ln in txt.splitlines() if "沪主板" in ln][0]
        assert "0.000%" in line, f"沪主板未封板应报 0%，实际: {line}"

        # ⚠️ **不能出现「北交所 …真封涨停率 100%」** —— 那是缺数据被当封板
        bj = [ln for ln in txt.splitlines() if "北交所" in ln][0]
        assert "100.00%" not in bj, (
            f"北交所未覆盖却报 100% —— 缺数据被当成封板了: {bj}")

    def test_真封板率为零时明确提示(self):
        idx = pd.to_datetime(["2023-01-03"])
        cols = ["sh600001"]
        close = pd.DataFrame([[10.0]], index=idx, columns=cols)
        up = pd.DataFrame([[20.0]], index=idx, columns=cols)  # 远高于收盘
        txt = coverage_report(close, up)
        assert "0.000%" in txt


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))