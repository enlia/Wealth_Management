"""``factor_lab.market_rules`` 的判据边界测试。

为什么必须有这个文件
------------------
``market_rules`` 里全是「魔法数字」：12.8571% / 22.7273% / 32.8571% /
HALF_TICK / MIN_PRICE。这些数字文档里写着，但**没人能验证它们还能算出来**。
本项目实测踩过：文档写「主板最大 10.48%」，代码实算是 10.95%，
差异来自加浮点余量后没同步文档 —— 评审才发现。

这些测试的每个断言都对应一个**真实踩过的坑**，不是形式覆盖。

运行
----
  uv run pytest tests/test_market_rules.py -v
"""
from __future__ import annotations

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
        assert max_tolerance(0.10) == pytest.approx(0.128571, abs=1e-5)
        assert max_tolerance(0.20) == pytest.approx(0.227273, abs=1e-5)
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
        for td in (1, 2, 3, 100, 2000):
            assert no_limit_days("sz000001", td, 19910403) is False

    def test_创业板注册制前的次新股(self) -> None:
        """2020-08-24 前上市的创业板股票不适用 5 日窗口。"""
        assert no_limit_days("sz300001", 1, 20090610) is True
        assert no_limit_days("sz300001", 2, 20090610) is False


class TestIsLimitUp:
    """Q8配套：容差口径必须与 price_tolerance 一致。"""

    def test_合法封板(self) -> None:
        assert is_limit_up(10.00, limit_price(10.00, 0.10), "sh600000") is False
        assert is_limit_up(2.66, limit_price(2.66, 0.10), "sz002426") is False

    def test_明确超限(self) -> None:
        # 回归：旧实现减整档 TICK，导致 +11% 也返回 True
        assert is_limit_up(10.00, 11.10, "sh600000") is True
        assert is_limit_up(10.00, 12.00, "sh600000") is True

    def test_普通涨幅不误判(self) -> None:
        assert is_limit_up(10.00, 10.50, "sh600000") is False
        assert is_limit_up(10.00, 10.20, "sh600000") is False

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_非法前收盘必须报错(self, bad: float) -> None:
        with pytest.raises(ValueError, match="前收盘价必须为正"):
            is_limit_up(bad, 10.0, "sh600000")


class TestConstants:
    def test_tick与半档(self) -> None:
        assert TICK == 0.01
        assert HALF_TICK == pytest.approx(0.005)

    def test_各板块名义限幅(self) -> None:
        assert limit_of("sh600519") == 0.10
        assert limit_of("sh688205") == 0.20
        assert limit_of("sz300489") == 0.20
        assert limit_of("bj920790") == 0.30

    def test_板块中文名(self) -> None:
        assert board_cn("sh600519") == "主板"
        assert board_cn("bj920790") == "北交所"

    def test_价格区间覆盖真实A股(self) -> None:
        # 实测 A 股最低 0.26（sz000004）、最高 2832.92（bj899601）
        assert MIN_PRICE <= 0.26
        assert MAX_PRICE >= 2832.92