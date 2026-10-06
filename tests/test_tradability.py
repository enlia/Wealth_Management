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

    def test_传前复权价会被检出(self):
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

    def test_指数列不得混入板块统计(self):
        """⚠️⚠️ **本组测试的还原验证实测过**（2026-10-06）：
        把 `ok_cols` 改回「不排除零覆盖列」，本测试**会失败**，
        而 `test_能区分真封板与缺数据` **照样通过**——
        因为 `true_limit` 内部已有 `& both`，封板率算不出差别。
        ⇒ 后者是**橡皮章测试**：看着在验证覆盖率口径，实际验证不了。

        真实缺陷（2026-10-06 实测）：`bar_daily` 里混着 2,862 列指数
        （sh000001、sh000300…），指数**永远没有涨跌停价**。
        而本函数按 `startswith('sh')` 分组，于是指数全被算进「沪主板」：
          沪主板 4,181 只 → 涨停价覆盖率被拉到 **52.24%**（真值 97.8%），
        看起来像「数据烂了一半」，实际是**分母混了不该有的东西**。
        ⚠️ 与 P16（分组排名方向）同类：代码能跑、有输出，
           但统计的对象不是你要研究的对象。

        判据：沪深主板的「只数」必须等于**有涨跌停价的**沪深股票数，
        指数不计入。
        """
        idx = pd.to_datetime(["2023-01-03", "2023-01-04"])
        # 2 只真实沪市A股 + 5 列指数（sh000001 / sh000300 等）
        cols = ["sh600001", "sh600002",
                "sh000001", "sh000300", "sh000905", "sh000016", "sh000852"]
        close = pd.DataFrame(10.0, index=idx, columns=cols)
        # 只有两只真实股票有涨跌停价，指数列全缺
        up = pd.DataFrame(11.0, index=idx, columns=cols[:2])

        txt = coverage_report(close, up, "test")

        # 零覆盖列必须被显式报出
        assert "零涨跌停价列" in txt, f"未报出零覆盖列:\n{txt}"
        assert "5/7" in txt, f"零覆盖列数应为 5/7:\n{txt}"

        # ⚠️ **核心判据**：沪主板只数必须是 2（真实股票），不是 7（含指数）
        line = [ln for ln in txt.splitlines() if "沪主板" in ln][0]
        assert "2 只" in line, (
            f"指数列被算进沪主板了（应为 2 只）:\n{line}")
        # 覆盖率必须是 100%（真实股票全覆盖），不是被指数拉低的 28.6%
        assert "100.00%" in line, (
            f"沪主板覆盖率被指数列拉低（应为 100.00%）:\n{line}")

    def test_板块全零覆盖必须报警而非静默(self):
        """⚠️ 上一条的**副作用约束**：排除零覆盖列后，
        若某板块**全部**零覆盖，不能静默消失 —— 必须显式报警。

        背景：北交所 `stk_limit` 覆盖不全是真实存在的问题。
        若改代码时顺手把零覆盖列全filter 掉，
        `coverage_report` 就不再报警 ⇒ 恰好丢掉了它存在的意义。
        """
        idx = pd.to_datetime(["2023-01-03"])
        cols = ["sh600001", "bj920001", "bj920002"]
        close = pd.DataFrame(10.0, index=idx, columns=cols)
        up = pd.DataFrame(11.0, index=idx, columns=["sh600001"])  # 北交所全缺

        txt = coverage_report(close, up, "test")
        bj_lines = [ln for ln in txt.splitlines() if "北交所" in ln]
        assert bj_lines, f"北交所行消失了 —— 缺数据不再报警:\n{txt}"
        assert "0.00%" in bj_lines[0], (
            f"北交所全零覆盖必须报出 0.00%:\n{bj_lines[0]}")
        assert "未覆盖" in bj_lines[0] or "只" in bj_lines[0]


class TestNoPxDoesNotBlockUnlisted:
    """🔴 **BLOCK 回归**：`no_px` 曾让 `unlisted_state` 修复完全失效。

    ## 缺陷本体

    初版：
        no_px = up.isna() | dn.isna()
        limit_up = (close >= up - tol) | no_px      # ← 无条件封板

    Tushare `stk_limit` **在未上市/已退市的日子里本来就没有行**
    ⇒ 未上市 ⇒ `up` 为 NaN ⇒ `no_px=True` ⇒ 未上市股票被判「封涨停」。

    ⇒ `unlisted_state="tradable"` 只在「有涨跌停价但无 close」时生效，
      而「未上市」这个**主体场景**在更早的 `no_px` 分支就被拦下了。

    ## 实测影响（2016-2026 全市场 A 股 5,921 只 × 2,611 日）

    | 口径 | 买不进 | 卖不掉 |
    |---|---|---|
    | 初版 | **27.817%** | **27.179%** |
    | 修正版 | 1.164% | 0.526% |
    | 真实封板强度 | 1.048% | 0.410% |

    `no_px` 的 4,138,520 格里 **99.6%** 是「close 也缺失」，
    只有 0.116% 是真正的上市期间数据空洞。
    """

    @staticmethod
    def _panels():
        """3 日 × 2 只。sz000001 第 2 日起才上市（第 0、1 日未上市）。"""
        idx = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
        cols = ["sh600001", "sz000001"]
        close = pd.DataFrame(
            [[10.0, np.nan],       # 第 0 日：sz 未上市
             [10.2, np.nan],       # 第 1 日：sz 未上市
             [10.1, 20.0]],         # 第 2 日：两只都在交易
            index=idx, columns=cols)
        # 关键：stk_limit 对未上市日**没有行**⇒ up/dn 也是 NaN
        up = pd.DataFrame(
            [[11.0, np.nan], [11.2, np.nan], [11.1, 22.0]],
            index=idx, columns=cols)
        dn = pd.DataFrame(
            [[9.0, np.nan], [9.2, np.nan], [9.1, 18.0]],
            index=idx, columns=cols)
        return close, up, dn

    def test_未上市不得被判买不进(self):
        """🔴 核心判据：未上市格子的 `limit_up` 必须为 False。"""
        close, up, dn = self._panels()
        limit_up, _ = limit_masks(close, up, dn, unlisted_state="tradable")

        assert not bool(limit_up.loc[pd.Timestamp("2023-01-03"), "sz000001"]), \
            "未上市被判成买不进 —— no_px 又把 unlisted_state 绕过去了"
        assert not bool(limit_up.loc[pd.Timestamp("2023-01-04"), "sz000001"]), \
            "未上市被判成买不进"

    def test_未上市不得被判卖不掉(self):
        """🔴 退市股永久锁仓的根源在**卖出端**，必须单独守。"""
        close, up, dn = self._panels()
        _, limit_dn = limit_masks(close, up, dn, unlisted_state="tradable")

        assert not bool(limit_dn.loc[pd.Timestamp("2023-01-03"), "sz000001"]), \
            "未上市被判成卖不掉 ⇒ 退市股会僵尸锁仓"

    def test_买不进比例必须接近真实封板率(self):
        """整体口径自检：不能把「未上市」算进约束强度。

        这是**唯一能抓住该bug 的整体判据** ——
        单点断言容易和别的口径混淆，比率不会。
        """
        close, up, dn = self._panels()
        limit_up, _ = limit_masks(close, up, dn, unlisted_state="tradable")
        # 3 日 × 2 只 = 6 格，其中真封板0 格 ⇒ 买不进必须是 0
        assert float(limit_up.to_numpy().mean()) == 0.0, (
            f"买不进比例应为 0，实测 "
            f"{float(limit_up.to_numpy().mean()):.2%} —— 未上市被误封")

    def test_有close但缺价仍然保守封板(self):
        """⚠️ **另一侧不能一起放掉**：真数据空洞仍须判不可交易。

        上市期间有 close 却没涨跌停价 ⇒ 无法判断是否封板 ⇒ 保守封 True。
        这是安全的一侧（不会高估可交易性），不能为了修未上市而放掉。

        ⚠️ 面板整体覆盖率必须 ≥50%（`limit_masks` 的口径自检会抛错），
        所以用 4×4 面板、只让一格缺价来构造。
        """
        idx = pd.to_datetime(["2023-01-03", "2023-01-04",
                              "2023-01-05", "2023-01-06"])
        cols = [f"sh60000{i}" for i in range(4)]
        close = pd.DataFrame(10.0, index=idx, columns=cols)
        up = pd.DataFrame(11.0, index=idx, columns=cols)
        dn = pd.DataFrame(9.0, index=idx, columns=cols)
        # 只让 sh600000 第 0 日缺涨跌停价（真实数据空洞）
        up.loc[idx[0], "sh600000"] = np.nan
        dn.loc[idx[0], "sh600000"] = np.nan

        limit_up, limit_dn = limit_masks(close, up, dn)

        assert bool(limit_up.loc[idx[0], "sh600000"]), \
            "有 close 但缺价应保守判买不进"
        assert bool(limit_dn.loc[idx[0], "sh600000"]), \
            "有 close 但缺价应保守判卖不掉"
        # 其余格子不应受影响
        assert not bool(limit_up.loc[idx[1], "sh600000"]), \
            "只有缺价那一格该被封"

    def test_真封涨停仍然被识别(self):
        """修复不能把真封板也一起放掉。"""
        idx = pd.to_datetime(["2023-01-03"])
        cols = ["sh600001"]
        close = pd.DataFrame([[11.0]], index=idx, columns=cols)  # 收在涨停
        up = pd.DataFrame([[11.0]], index=idx, columns=cols)
        dn = pd.DataFrame([[9.0]], index=idx, columns=cols)

        limit_up, limit_dn = limit_masks(close, up, dn)

        assert bool(limit_up.iloc[0, 0]), "收在涨停价必须判买不进"
        assert not bool(limit_dn.iloc[0, 0]), "涨停不应同时判跌停"

    def test_blocked模式仍然封未上市(self):
        """`unlisted_state="blocked"` 的显式契约不能被这次修复破坏。"""
        close, up, dn = self._panels()
        limit_up, _ = limit_masks(close, up, dn, unlisted_state="blocked")

        assert bool(limit_up.loc[pd.Timestamp("2023-01-03"), "sz000001"]), \
            "blocked 模式必须封未上市（调用方显式要求）"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))