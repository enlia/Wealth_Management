"""股票池「上市满 N 天」判据的守护用例（2 BLOCK + 1 口径缺陷的回归）。

出处：`.planning/reports/review-retro-2e5c653-2357b79.md` 二·2.5（缺陷真源）。

  · BLOCK-1 哨兵死代码（P22 同构 + P23）：
    ``_ld_pos.where(_in_win, 0)`` 把「缺上市日」的 NaN 换成**真实位置 0**，
    ``age.isna()`` 恒 False，「缺 list_date → 按表内首日计龄」分支不可达。
    后果实测双向：缺 list_date 的**真次新股上市首日即进池**（250 日保护不存在），
    缺 list_date 的**老股在短窗口里被全灭**（旧 bug 原样复发，样本 3 只灭 1 只 /
    242 行全删）。暴露面：bj920 新号段整段缺 list_date。
  · BLOCK-2 静默跳过（铁律 4 违规）：
    info=None / 无 list_date 列 / list_date 全 NaN 时整段上市年限过滤**零打印跳过**，
    上市 20 天的 sz301999 照样进池。
  · 口径缺陷：计龄序号取自输入长表的日期并集，单股/长停牌的大缺口会把「日历」
    压缩 —— 真实 299 个交易日的样本能被数成 0 行（S7b）。

每个用例都写明「还原 bug 版的失败形态」：还原任一修复后必须 FAILED。
"""
from __future__ import annotations

import pandas as pd
import pytest

from factor_lab.config import DB_PATH, ResearchConfig
from factor_lab.data.universe import build_universe


def _long(code: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    """单只股票的最小长表：价格 10 元、日成交额 1 亿（不触发价格/流动性过滤）。"""
    return pd.DataFrame({
        "code": [code] * len(days),
        "date": pd.DatetimeIndex(days),
        "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
        "amount": 1e8, "vol": 1e6,
    })


class TestListDateFallback:
    """缺 list_date 的退化分支必须做实（独立掩码 + 表内首日 + 打印计数）。

    还原 bug 版（``.where(_in_win, 0)`` 哨兵 + 死分支）后，
    本类各用例必须 FAILED —— 失败形态见各用例 docstring。
    """

    @pytest.mark.parametrize("has_list_date", [True, False])
    def test_真次新股250日保护必须生效(self, has_list_date: bool) -> None:
        """暴露面样本一：bj920 段（库内实测整段缺 list_date）的真次新股。

        上市/数据起点 = 日历第 261 日（0 基），窗口内上市：
        上市当天计龄 0 → 第 251 个交易日起合格，此前必须淘汰。
        有/无 list_date 两个变体的结果必须**完全一致**（保护语义一致）。

        还原 bug 版失败形态（缺 list_date 变体）：
        哨兵把基准换成位置 0 → 计龄=全局窗口序号 → 340 行**全部首日进池**。
        """
        cal = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=600))
        code = "bj920888"
        days = cal[260:]                     # 上市第 1 天 = 日历第 261 个交易日
        long = _long(code, days)
        ld = [float(pd.Timestamp(days[0]).strftime("%Y%m%d"))] if has_list_date else [None]
        info = pd.DataFrame({"code": [code], "name": ["北证次新"], "list_date": ld})
        first = pd.Series({code: days[0]})

        out = build_universe(long, ResearchConfig(), info=info,
                             first_dates=first, verbose=False)

        ok = days[250:]                      # 第 251 个交易日起合格
        assert len(out) == len(ok) == 90, \
            f"250 日保护失效：应保留 {len(ok)} 行，实际 {len(out)} 行"
        assert pd.Timestamp(out["date"].min()) == days[250]

    def test_缺list_date的老股短窗不得全灭(self) -> None:
        """暴露面样本二：老股 + 短窗口（242 日 < 250）+ 缺 list_date（S4c 形态）。

        计龄基准取行情表（bar_daily）内该股首个数据日 = 2015-09-14（远早于窗口），
        短窗口内每一天都已满 250 交易日 → 242 行必须全保留。

        还原 bug 版失败形态：计龄=窗口内序号（0..241 < 250）→ **242 行全删**
        （旧 bug 在该人群原样复发）。
        """
        cal = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=242))
        code = "sh600036"
        long = _long(code, cal)
        info = pd.DataFrame({
            "code": [code, "sh600519"],
            "name": ["招商银行", "贵州茅台"],
            "list_date": [None, 20010827.0],     # 同表有可解析上市日 → 不触发整段判据缺失
        })
        first = pd.Series({code: pd.Timestamp("2015-09-14")})

        out = build_universe(long, ResearchConfig(), info=info,
                             first_dates=first, verbose=False)
        assert len(out) == 242, f"缺 list_date 的老股被全灭，剩 {len(out)}/242 行"

    def test_缺list_date处理必须打印来源与计数(self) -> None:
        """铁律 4（P4）：退化分支必须打印原因 + 计数，禁静默。

        还原 bug 版失败形态：死分支不可达，打印文案要么缺失、
        要么声称「按表内首日估算」而实际按窗口内序号计龄（P23 文案失实）。
        """
        import io
        import contextlib

        # 数据晚起点的次新股：行情表首日与本表首日两个来源各一只
        cal = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=300))
        code_db, code_tbl = "bj920881", "bj920882"
        days = cal[50:100]
        long = pd.concat([_long(code_db, days), _long(code_tbl, days)],
                         ignore_index=True)
        info = pd.DataFrame({
            "code": [code_db, code_tbl, "sh600519"],
            "name": ["北证A", "北证B", "贵州茅台"],
            "list_date": [None, None, 20010827.0],
        })
        # 只有 code_db 能取到行情表首日；code_tbl 连行情表首日都没有 → 本表首日
        first = pd.Series({code_db: days[0]})

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_universe(long, ResearchConfig(), info=info,
                           first_dates=first, verbose=False)
        out_txt = buf.getvalue()
        assert "缺 stock_info.list_date" in out_txt, "退化处理零打印（铁律 4 违规）"
        assert "按表内首日处理" in out_txt, "未声明退化口径（P23 文案要求如实）"
        assert "2 只" in out_txt, f"未打印受影响只数：{out_txt!r}"
        assert "1 只取行情表首个数据日" in out_txt, f"来源拆分缺失：{out_txt!r}"
        assert "1 只取本表首个交易日" in out_txt, f"来源拆分缺失：{out_txt!r}"

    def test_退化计龄同样执行250日保护(self) -> None:
        """退化基准（本表首日）计龄同样生效：数据晚起点的次新股不得首日进池。

        还原 bug 版失败形态：计龄基准被换成位置 0 → 首日即合格（保护不存在）。
        """
        cal = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=400))
        code = "bj920883"                     # 行情表首日也没有 → 退到本表首日
        days = cal[100:150]                   # 表内仅 50 行
        long = _long(code, days)
        info = pd.DataFrame({
            "code": [code, "sh600519"],
            "name": ["北证C", "贵州茅台"],
            "list_date": [None, 20010827.0],
        })

        out = build_universe(long, ResearchConfig(), info=info,
                             first_dates=pd.Series(dtype="datetime64[ns]"),
                             verbose=False)
        assert len(out) == 0, \
            f"退化基准下 50 行全部未满 250 日，应 0 行，实际 {len(out)} 行"


class TestLoadFirstDates:
    """缺 list_date 退化路径的数据源：行情表内首个数据日（上市日下界代理）。"""

    @pytest.mark.skipif(not DB_PATH.exists(),
                        reason="缺产物 market.db（data/ 不入库），跳过依赖真实库的读取用例")
    def test_读真实行情表首日(self) -> None:
        from factor_lab.data import load_first_dates

        fd = load_first_dates(["sh600519", "bj920000"])
        assert list(fd.index) == ["bj920000", "sh600519"] or set(fd.index) == \
            {"sh600519", "bj920000"}, f"索引应为 code：{fd}"
        # 亲测读数（2026-10-13）：sh600519 首个数据日 < 2016-06-30（库覆盖期初段）；
        # bj920000 首个数据日 = 2020-12-23（该代码由旧号段转来，行情早于 920 号段启用）
        assert pd.Timestamp(fd["sh600519"]) < pd.Timestamp("2016-06-30")
        assert pd.Timestamp(fd["bj920000"]) > pd.Timestamp("2017-01-01")
        assert fd["bj920000"] > fd["sh600519"], "首个数据日应随代码逐一给出"

    def test_空清单返回空序列不报错(self) -> None:
        from factor_lab.data import load_first_dates

        out = load_first_dates([])
        assert out.empty


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))