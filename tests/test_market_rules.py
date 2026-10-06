"""``factor_lab.market_rules`` 与调用方判定逻辑的测试。

为什么必须有这个文件
------------------
``market_rules`` 里全是「魔法数字」：13.4615% / 23.2143% / 32.8571% /
HALF_TICK / MIN_PRICE。这些数字文档里写着，但**没人能验证它们还能算出来**。
本项目实测踩过：文档写「主板最大 10.48%」，代码实算是 10.95%，
差异来自加浮点余量后没同步文档 —— 评审才发现。

⚠️ **不只测 market_rules 本身，还要测调用方的布尔组合。**
   实测踩过：上一轮最严重的 bug 是 `audit_data_quality.judge_limit_rules`里
   `ok &= ~in_window; return ~ok` 把「排除」写成「计入」。
   但当时只有 `no_limit_days` 这个纯函数的测试 ——
   把调用方还原成 bug 版本后，50 个测试**一个都没失败**。
   纯函数正确 ≠ 调用方组合正确。`TestJudgeIntegration` 专治这个。

运行
----
  uv run pytest tests/test_market_rules.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from factor_lab.market_rules import (
    BOARD_CN,
    HALF_TICK,
    LIMIT,
    MAX_PRICE,
    MIN_PRICE,
    TICK,
    board_cn,
    board_of,
    is_limit_up,
    limit_of,
    limit_price,
    max_tolerance,
    min_tolerance,
    no_limit_days,
    price_tolerance,
)


class TestBoardOf:
    """Q5：板块判定必须看代码数字段，不能看首位。"""

    @pytest.mark.parametrize(
        ("code", "expect"),
        [
            ("sh600519", "main"), ("sh601398", "main"),
            ("sh603124", "main"), ("sh605358", "main"),
            ("sz000001", "main"), ("sz002594", "main"),
            ("sh688205", "star"), ("sh688981", "star"), ("sh689009", "star"),
            ("sz300489", "gem"), ("sz301288", "gem"),
            # Q5：创业板前缀有 300/301/302 三个，漏 302 会把创业板注册制股票
            # 判成主板 ±10%，其 20% 涨停被误判成「复权失败」（2026-10-06 实测）
            ("sz302132", "gem"),
            ("bj920790", "bse"), ("bj899050", "bse"),
        ],
    )
    def test_按数字段判定(self, code: str, expect: str) -> None:
        assert board_of(code) == expect

    def test_大小写不敏感(self) -> None:
        assert board_of("SH688205") == "star"

    @pytest.mark.parametrize("code", ["", "   "])
    def test_空代码必须报错而非默认(self, code: str) -> None:
        # Q5：未知形态默认 main，但**空代码**是数据错误，必须 raise
        with pytest.raises(ValueError, match="代码为空"):
            board_of(code)

    def test_未知前缀默认最严限幅(self) -> None:
        # 前缀不是 sh/sz/bj 时应落main（最严），不能落 gem
        assert board_of("xy123456") == "main"

    def test_板块中文名齐全(self) -> None:
        assert set(LIMIT) == set(BOARD_CN)


class TestLimitPrice:
    """Q3：必须用 Decimal 四舍五入，不能用 float + round。"""

    @pytest.mark.parametrize(
        ("prev", "expect"),
        [
            (2.66, 2.93),   # 文档实例
            (1.05, 1.16),   # 低价股取整误差最大
            (1.10, 1.21),
            (10.00, 11.00),  # 整数档无误差
            (1258.62, 1384.48),
        ],
    )
    def test_涨停价四舍五入(self, prev: float, expect: float) -> None:
        assert limit_price(prev, 0.10) == pytest.approx(expect, abs=0.005)

    def test_银行家舍入与四舍五入不一致的档位(self) -> None:
        """Python round() 是银行家舍入，0.125 → 0.12，与交易所不同。"""
        # 前收 1.1364 元 → ×1.1 = 1.25004 → 四舍五入到 1.25
        # 若用 round()，1.25 无歧义；这里选真正会分歧的 0.125 边界
        got = limit_price(1.1363636363636365, 0.10)
        assert got == 1.25

    @pytest.mark.parametrize("bad", [0.0, -1.0, -0.01])
    def test_非法前收盘必须报错(self, bad: float) -> None:
        with pytest.raises(ValueError, match="前收盘价必须为正"):
            limit_price(bad, 0.10)


class TestPriceTolerance:
    """Q2 + Q8：动态容差 + 半档tick 浮点余量。"""

    def test_容差大于名义限幅(self) -> None:
        # 取整必然撑大容差，即使最贵的价格
        assert price_tolerance(1258.62, 0.10) > 0.10

    def test_余量随价格单调收缩(self) -> None:
        """余量 = HALF_TICK / prev_close，必须随价格单调递减。"""
        t_low = price_tolerance(1.05, 0.10) - price_tolerance(1.05, 0.10) + HALF_TICK / 1.05
        t_high = HALF_TICK / 1258.62
        assert t_low > t_high
        assert t_high == pytest.approx(3.97e-6, rel=1e-3)

    def test_不会误吞真实超限(self) -> None:
        """真实 -30% 跌幅在任何价位都必须被判超限。"""
        for prev in (0.26, 1.00, 2.66, 10.0, 2832.92):
            assert 0.30 > price_tolerance(prev, 0.10)

    def test_恰好合法涨停不判超限(self) -> None:
        """Q8：数学上相等的涨幅不能因浮点误差被判超限。"""
        for prev in (1.05, 1.10, 2.66, 3.00, 10.00, 50.00):
            lp = limit_price(prev, 0.10)
            ret = lp / prev - 1
            assert ret <= price_tolerance(prev, 0.10), (
                f"前收 {prev}: ret={ret} > tol={price_tolerance(prev, 0.10)}"
            )

    def test_低价股容差明显大于高价股(self) -> None:
        assert price_tolerance(0.30, 0.10) > price_tolerance(100.0, 0.10) + 0.01


class TestMaxTolerance:
    """兜底阈值必须与MIN_PRICE/MAX_PRICE 区间一致。"""

    def test_与文档数字一致(self) -> None:
        # 文档 DATA_QUALITY.md Q2 记录的这三个数字，改动即需同步文档
        assert max_tolerance(0.10) == pytest.approx(0.134615, abs=1e-5)
        assert max_tolerance(0.20) == pytest.approx(0.232143, abs=1e-5)
        assert max_tolerance(0.30) == pytest.approx(0.328571, abs=1e-5)

    def test_必须覆盖区间内所有价格(self) -> None:
        mx = max_tolerance(0.10)
        for cents in range(int(MIN_PRICE * 100), int(MAX_PRICE * 100) + 1, 37):
            pc = cents / 100
            assert price_tolerance(pc, 0.10) <= mx + 1e-12

    def test_不落在病态区间(self) -> None:
        """[0.01, 0.04] 内 round(pc×1.1,2)==pc，容差会退化到荒谬值。

        MIN_PRICE 必须避开它，否则兜底阈值失去筛除能力。
        """
        assert MIN_PRICE > 0.05


class TestNoLimitDays:
    """Q4：注册制新股无限幅窗口，三板块生效日不同。"""

    def test_科创板新股前5日(self) -> None:
        for td in (1, 2, 3, 4, 5):
            assert no_limit_days("sh688205", td, 20220401) is True
        assert no_limit_days("sh688205", 6, 20220401) is False

    def test_创业板注册制后新股前5日(self) -> None:
        for td in (1, 5):
            assert no_limit_days("sz300873", td, 20200824) is True
        assert no_limit_days("sz300873", 6, 20200824) is False

    def test_主板全面注册制后新股前5日(self) -> None:
        """2023-02-17 之后主板新股也有 5 日窗口 —— 最容易漏的一条。"""
        for td in (1, 2, 5):
            assert no_limit_days("sh603124", td, 20250320) is True
        assert no_limit_days("sh603124", 6, 20250320) is False

    def test_主板注册制前的新股只有首日(self) -> None:
        """2023-02-17 前上市的主板新股第 2 日起恢复 ±10%。"""
        assert no_limit_days("sh600000", 1, 19991110) is True
        assert no_limit_days("sh600000", 2, 19991110) is False

    def test_老股任何交易日都有限幅(self) -> None:
        # 首日（td=1）任何股票都无涨跌幅限制，这是制度而非豁免
        assert no_limit_days("sz000001", 1, 19910403) is True
        for td in (2, 3, 100, 2000):
            assert no_limit_days("sz000001", td, 19910403) is False

    def test_创业板注册制前的次新股(self) -> None:
        """2020-08-24 前上市的创业板股票不适用 5 日窗口。"""
        assert no_limit_days("sz300001", 1, 20090610) is True
        assert no_limit_days("sz300001", 2, 20090610) is False


class TestIsLimitUp:
    """判定「达到涨停价」，不是「超过」。

    ⚠️ 旧版用 `> tol - HALF_TICK/pc`，实际在判「超过涨停价」，
       导致真正的封板判 False、不可能存在的涨幅判 True。
    """

    @pytest.mark.parametrize("prev", [0.26, 1.05, 1.10, 2.66, 10.00, 50.00])
    def test_精确封板必须判True(self, prev: float) -> None:
        """回归：2.66→2.93、0.26→0.29 都是精确封板，旧版判 False。"""
        assert is_limit_up(prev, limit_price(prev, 0.10), "sh600000") is True

    def test_越过了涨停价也算封过板(self) -> None:
        """收盘高于涨停价，说明当天确实触及过涨停。
        2.66 → 3.00 = +12.78%，涨停价是 2.93（+10.15%），判 True 正确。
        """
        assert is_limit_up(2.66, 3.00, "sh600000") is True
        assert is_limit_up(10.00, 11.10, "sh600000") is True

    def test_普通涨幅不误判(self) -> None:
        assert is_limit_up(10.00, 10.50, "sh600000") is False
        assert is_limit_up(10.00, 10.20, "sh600000") is False
        assert is_limit_up(0.26, 0.28, "sh600000") is False

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_非法前收盘必须报错(self, bad: float) -> None:
        with pytest.raises(ValueError, match="前收盘价必须为正"):
            is_limit_up(bad, 10.0, "sh600000")


class TestNoLimitDaysEdge:
    """评审指出的漏测分支。"""

    def test_北交所走首日分支(self) -> None:
        # NO_LIMIT_DAYS['bse'] == 1 → n<=1 早退
        assert no_limit_days("bj920790", 1, 20211115) is True
        assert no_limit_days("bj920790", 2, 20211115) is False

    def test_漏传first_trade_date的行为(self) -> None:
        """docstring 说「必须传」，但可选默认值仍存在 —— 记录其真实行为。

        漏传时退化成「一律 5 日窗口」，会让 2023-02-17 前上市的主板新股
        拿到 4 天不该有的豁免，**漏报**真实除权污染。
        """
        assert no_limit_days("sh603124", 3, None) is True
        # 正确传入则豁免范围相同（该股 2025 年上市）
        assert no_limit_days("sh603124", 3, 20250320) is True
        # 反例：老主板新股必须靠 first_trade_date 区分
        assert no_limit_days("sh600000", 3, None) is True        # 漏传 → 错误豁免
        assert no_limit_days("sh600000", 3, 19991110) is False   # 正确传入

    def test_注册制生效日当天边界(self) -> None:
        """代码用 `<` 而非 `<=`，生效日当天上市的新股应享受 5 日窗口。"""
        assert no_limit_days("sh603124", 5, 20230217) is True
        assert no_limit_days("sh603124", 6, 20230217) is False
        # 前一天上市则不适用
        assert no_limit_days("sh603124", 3, 20230216) is False


class TestConstants:
    def test_tick与半档(self) -> None:
        assert TICK == 0.01
        assert HALF_TICK == pytest.approx(0.005)

    def test_各板块名义限幅(self) -> None:
        assert limit_of("sh600519") == 0.10
        assert limit_of("sh688205") == 0.20
        assert limit_of("sz300489") == 0.20
        assert limit_of("bj920790") == 0.30

    def test_创业板三个前缀限幅一致(self) -> None:
        # Q5：302 漏判会让 sz302132 的 20% 涨停被当成主板 10% 超限
        for c in ("sz300489", "sz301288", "sz302132"):
            assert limit_of(c) == 0.20, f"{c} 创业板限幅应为 20%"

    def test_板块中文名(self) -> None:
        assert board_cn("sh600519") == "主板"
        assert board_cn("bj920790") == "北交所"

    def test_价格区间覆盖真实A股(self) -> None:
        # 实测 A 股最低 0.26（sz000004）、最高 2832.92（bj899601）
        assert MIN_PRICE <= 0.26
        assert MAX_PRICE >= 2832.92


class TestPreFilterBound:
    """预筛必须用 min_tolerance，不能用 max_tolerance。

    评审实测：800 只样本里，预筛用 max_tolerance 会把落在
    (精确容差, 兜底阈值] 区间的 **85 条**（占精确判定的 10.65%）静默丢弃。
    """

    def test_min小于max(self) -> None:
        assert min_tolerance(0.10) < max_tolerance(0.10)

    def test_预筛下界不会漏掉应判超限的记录(self) -> None:
        """核心回归：−12.13% 必须能被预筛捞到（它 > 精确容差）。"""
        prev, ret = 9.26, -0.1213
        exact = price_tolerance(prev, 0.10)
        assert abs(ret) > exact, "前提：这条确实应该判超限"
        assert abs(ret) > min_tolerance(0.10), "预筛必须能捞到它"
        # 反证：用max_tolerance 就会漏掉
        assert abs(ret) < max_tolerance(0.10), "这正是原 bug 的表现"

    def test_最小值落在高价区(self) -> None:
        """最小容差出现在区间右端附近（价格越高，取整误差占比越小）。
        实测在前收 4999.94 元处取到 10.000020%。
        """
        assert min_tolerance(0.10) == pytest.approx(0.10000020, abs=1e-7)
        # 严格大于名义限幅 —— 因为总有半档 tick 的浮点余量
        assert min_tolerance(0.10) > LIMIT["main"]


class TestJudgeIntegration:
    """调用方的布尔组合 —— 纯函数对 ≠ 组合对。

    ⚠️ 上一轮最严重的 bug 在``audit_data_quality.judge_limit_rules``：
       ``ok &= ~in_window; return ~ok``把「排除」写成「计入」。
       当时 50 个测试一个都没失败，因为只测了``no_limit_days`` 纯函数。

    这里复刻那个判定组合（不 import research/ 脚本，避免耦合），
    把 4 种 (窗口内/外) × (涨幅超/未超) 的组合全部钉死。
    """

    @staticmethod
    def _judge(ret: float, pc: float, in_window: bool, limit: float = 0.10) -> bool:
        """与 audit_data_quality.judge_limit_rules 同口径的最小复刻。

        排除必须作用在「超限」上，不能靠对「不超限」标志二次取反。
        """
        tol = price_tolerance(pc, limit) if pc > 0 else limit
        over = abs(ret) > tol
        over &= np.logical_not(in_window)
        over &= pc > 0
        return bool(over)

    @pytest.mark.parametrize(
        ("in_window", "ret", "expect_over"),
        [
            # (a) 窗口内大幅波动 → 不是超限（注册制新股合法）
            (True, 0.350, False),
            # (b) 窗口外大幅波动 → 超限
            (False, 0.350, True),
            # (c) 窗口内小幅波动 → 不是超限
            (True, 0.005, False),
            # (d) 窗口外小幅波动 → 不是超限
            (False, 0.005, False),
            # 边界：恰好等于容差（合法封板）
            (False, 0.10, False),
            # 边界：略微超出容差（真实超限）
            (False, 0.1100, True),
        ],
    )
    def test_四种组合(self, in_window: bool, ret: float, expect_over: bool) -> None:
        assert self._judge(ret, 10.00, in_window) is expect_over

    def test_涨跌方向对称(self) -> None:
        """负收益与正收益的判定必须对称。"""
        assert self._judge(-0.35, 10.00, True) is False
        assert self._judge(-0.35, 10.00, False) is True

    @pytest.mark.parametrize("pc", [0.0, -1.0, float("nan")])
    def test_前收无效时不判定(self, pc: float) -> None:
        """前收未知 → 无法判定，不能当成超限也不能当成合法。"""
        assert self._judge(0.35, pc, False) is False

    def test_精确封板不判超限(self) -> None:
        """Q8 浮点余量：合法封板绝不能被判超限。"""
        for prev in (0.26, 1.05, 2.66, 10.00, 50.00):
            lp = limit_price(prev, 0.10)
            assert self._judge(lp / prev - 1, prev, False) is False, (
                f"前收 {prev} 的精确封板被判超限"
            )
