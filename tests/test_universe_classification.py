"""标的分类规则的回归测试。

为什么必须有这个文件
-------------------
分类规则是**股票池的第一道闸门**：判错一个代码段，整段标的要么被静默剔除，
要么混进不该进的研究池。本项目实测踩过两次：

  · ``sz302132``（中航成飞，创业板注册制）在 ``board_of`` 里漏判→ ±10%
  · ``bj899050``（北证50，**指数**）在 ``is_index`` 里漏判 → 被当 A 股普通股

两次都是「代码能跑、结果有数字、但样本池是错的」—— 不会抛异常，
只会安静地把结论做偏。所以这些边界必须用测试钉死。
"""
from __future__ import annotations

import pytest

from factor_lab.config import is_a_share, is_b_share, is_index, is_sector


class TestIsIndex:
    @pytest.mark.parametrize(
        "code",
        [
            "sh000001",   # 上证指数
            "sh000300",   # 沪深300
            "sh950001",   # 上证 950 段
            "sz399001",   # 深证成指
            "sz399006",   # 创业板指
            # ⚠️ 2026-10-06 实测补：北交所指数段 899xxx 早先漏判，
            #    bj899050（北证50）被当 A 股，close_adj 全 NULL，
            #    把 60 只样本的价格面板缺失率拉到 67%
            "bj899050",
            "bj899601",
        ],
    )
    def test_指数段(self, code: str) -> None:
        assert is_index(code) is True

    @pytest.mark.parametrize(
        "code",
        [
            "sh600519",   # 贵州茅台
            "sz000001",   # 平安银行（与 sh000001 同数字段，靠市场前缀区分）
            "sz300750",   # 宁德时代
            "sz302132",   # 中航成飞（创业板注册制）
            "bj920000",   # 北交所新代码段
            "bj430047",   # 北交所旧代码段
            "bj830799",   # 北交所 8xxxxx 股票段（不是 899 指数段）
        ],
    )
    def test_非指数段(self, code: str) -> None:
        assert is_index(code) is False


class TestIsAShare:
    @pytest.mark.parametrize(
        "code",
        [
            "sh600519", "sh601398", "sh603124", "sh605358", "sh688981",
            "sz000001", "sz001979", "sz002594", "sz300489", "sz301288",
            "sz302132",                    # 创业板注册制（2026-10-06 补）
            "bj430047", "bj830799", "bj920000",
        ],
    )
    def test_普通股算A股(self, code: str) -> None:
        assert is_a_share(code) is True

    @pytest.mark.parametrize(
        "code",
        [
            "sh000001",   # 指数
            "sh900901",   # B股（sh 900xxx）
            "sz399001",   # 指数
            "sz200011",   # B股（sz 200xxx）
            "sh510300",   # ETF
            "sh113050",   # 可转债
            "bj899050",   # ⚠️ 北证50指数—— 早先被误判为 A 股
            "bj899601",   # ⚠️ 北证专精特新指数 —— 同上
        ],
    )
    def test_非A股(self, code: str) -> None:
        assert is_a_share(code) is False


class TestIsBShare:
    @pytest.mark.parametrize("code", ["sh900901", "sh900933", "sz200011"])
    def test_B股(self, code: str) -> None:
        assert is_b_share(code) is True

    @pytest.mark.parametrize("code", ["sh600519", "sz000001", "sz300750"])
    def test_非B股(self, code: str) -> None:
        assert is_b_share(code) is False


class TestIsSector:
    def test_板块指数(self) -> None:
        assert is_sector("sh880507") is True

    def test_个股不是板块(self) -> None:
        assert is_sector("sh600519") is False


class TestPrefixIsRespected:
    """坑1（2026-10-03 实测）：带前缀的代码必须直接采用前缀，不能重新推断。

    ``sh000001``（上证指数 3842点）与 ``sz000001``（平安银行 11.57 元）
    数字段完全相同。若丢掉前缀按首位猜市场，sh000001 会被读成平安银行。
    """

    def test_同数字段不同市场(self) -> None:
        assert is_index("sh000001") is True
        assert is_index("sz000001") is False
        assert is_a_share("sh000001") is False
        assert is_a_share("sz000001") is True

    def test_带前缀与不带前缀结果一致(self) -> None:
        assert is_a_share("sh600519") == is_a_share("600519")
        assert is_index("sz399001") == is_index("399001")
