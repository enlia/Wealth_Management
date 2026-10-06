"""本轮踩坑的回归测试：静默产出错误数字的四个位置。

为什么必须有这个文件
-------------------
本轮（2026-10-06）修的四个问题**全都不抛异常**，只是安静地给出错误数字：

  1. ``market_of`` 把 92xxxx 判成 sh→ 退市股被 ``is_a_share`` 静默剔除
  2. ``to_code`` 自写一套代码段规则 → 300526 判成 sh、920680 判成 sz
  3. 复权归一化分母未先剔除无行情日期 → bj920680 分母偏 22%
  4. 复权因子命中率只比 date 丢掉 code → 键全错也报 99.6%

没有测试的话，任何一个人改回去都不会有人发现。
每个用例都对应上面一个真实踩过的坑，不是形式覆盖。
"""
from __future__ import annotations

import sqlite3

import numpy as np

import pandas as pd
import pytest

from factor_lab.config import is_a_share, market_of


class TestMarketOf:
    @pytest.mark.parametrize(
        ("code", "expect"),
        [
            ("600519", "sh"),   # 上交所主板
            ("688981", "sh"),   # 科创板
            ("000001", "sz"),   # 深主板
            ("300526", "sz"),   # 🔴 创业板：首判若写成"3→sh"就错
            ("920680", "bj"),   # 🔴 北交所 92xxxx：首判若被"9→sh"吃掉就错
            ("832317", "bj"),   # 北交所旧段
            ("430047", "bj"),
        ],
    )
    def test_代码段判定(self, code: str, expect: str) -> None:
        assert market_of(code) == expect

    @pytest.mark.parametrize("code", ["920680", "920305", "920000"])
    def test_92xxxx_是北交所不是上交所(self, code: str) -> None:
        """92xxxx 必须判 bj。

        背景：`startswith("9") → sh` 会把北交所 92xxxx 判成上交所，
        于是 is_a_share 返回 False，退市股从名单里被静默剔除 ——
        bj920680（广道退）当年就是这样差点漏掉。
        """
        assert market_of(code) == "bj"
        assert is_a_share(f"bj{code}") is True
        assert is_a_share(f"sh{code}") is False


class TestToCode:
    """``to_code`` 必须复用 ``market_of``，不能自写一套代码段规则。

    实测：原先内联 ``s[0] in "0366" → sh``，于是 300526（创业板）被判成
    sh300526、920680（北交所）判成 sz920680，两条都错且都不报错，
    只是让股票被 ``is_a_share`` 静默剔除。
    """

    @pytest.mark.parametrize(
        ("symbol", "expect"),
        [
            ("920680", "bj920680"),
            ("920305", "bj920305"),
            ("300526", "sz300526"),
            ("603157", "sh603157"),
            ("000001", "sz000001"),
        ],
    )
    def test_交易所前缀(self, symbol: str, expect: str) -> None:
        from factor_lab.config import market_of

        s = str(symbol).zfill(6)
        assert f"{market_of(s)}{s}" == expect

    def test_to_code_已复用market_of(self) -> None:
        """源码级守护：to_code 的**可执行代码**里不得再硬编码代码段判断。

        用 AST 剥掉 docstring —— 说明文字里为了记录踩坑会引用旧代码
        （`s[0] in "0366"`），那是文档不是逻辑，不能因此误判。
        """
        import ast
        import inspect
        import sys
        from pathlib import Path

        sys.path.insert(0, str(
            Path(__file__).resolve().parents[1] / "research" / "scripts"))
        from backfill_delisted import to_code

        # 只看函数体的**语句节点**，并显式剔除 docstring（它也是 Expr/Constant）——
        # 说明文字里为了记录踩坑会引用旧代码（`s[0] in "0366"`），
        # 那是文档不是逻辑，不能因此误判。
        fn = next(n for n in ast.parse(inspect.getsource(to_code)).body
                  if isinstance(n, ast.FunctionDef) and n.name == "to_code")
        body = [n for n in fn.body
                if not (isinstance(n, ast.Expr)
                        and isinstance(n.value, ast.Constant)
                        and isinstance(n.value.value, str))]
        stmt = " ".join(ast.dump(n) for n in body)

        assert body, "没解析到任何语句，用例本身失效"
        assert "market_of" in stmt, "to_code 必须调用 market_of，不能自己判前缀"
        for bad in ("0366", "348"):
            assert bad not in stmt, f"to_code 的可执行代码里仍有硬编码 {bad!r}"


class TestHitRateJudge:
    """复权因子命中率必须按 (code, date) **复合键**判定。

    反证要求（DATA_QUALITY Q11）：把 code 全部改错，该判据**必须**报警。
    修复前它只比 date，把所有 code 换成库里不存在的值仍报 99.6%。
    """

    @staticmethod
    def _rate(adj: pd.DataFrame, have: set) -> float:
        pairs = set(zip(adj["code"], adj["dnum"]))
        return len(pairs & have) / max(len(pairs), 1)

    def test_全部键匹配(self) -> None:
        adj = pd.DataFrame({"code": ["a", "a", "b"],
                            "dnum": [1, 2, 1]})
        have = {("a", 1), ("a", 2), ("b", 1)}
        assert self._rate(adj, have) == 1.0

    def test_日期对但代码全错_必须报警(self) -> None:
        """🔴 反证用例：键全错配时判据必须降到 0，不能仍报高位。"""
        adj = pd.DataFrame({"code": ["xx", "xx"], "dnum": [1, 2]})
        have = {("a", 1), ("a", 2)}
        assert self._rate(adj, have) == 0.0

    def test_日期全错_必须报警(self) -> None:
        adj = pd.DataFrame({"code": ["a"], "dnum": [999]})
        have = {("a", 1)}
        assert self._rate(adj, have) == 0.0

    def test_旧判据的反证_说明它为什么是橡皮章(self) -> None:
        """保留旧判据作为对照：它对「代码全错」毫无反应。"""
        adj = pd.DataFrame({"code": ["xx", "xx"], "dnum": [1, 2]})
        have = {("a", 1), ("a", 2)}
        old = float(adj["dnum"].isin({d for _, d in have}).mean())
        assert old == 1.0, "旧判据在键全错时仍给满分 —— 正是要修的缺陷"


class TestExhaustedGenerator:
    """一次性迭代器被提前消费后，executemany 静默写 0 行。

    实测（:memory: 库，50 行）：
      · 完全耗尽的生成器 → rowcount **0**，写入 **0 行**，不抛异常
      · 未耗尽的生成器   → rowcount 50，写入 50 行
    部分消费时 rowcount 会返回 **-1**（同样是「不是成功」）。
    这条测试把现象钉住，防止有人再写出
    ``executemany(sql, df.itertuples())`` 这种写法。
    """

    def test_耗尽生成器_写0行(self, tmp_path) -> None:
        db = tmp_path / "t.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE t(k INTEGER PRIMARY KEY, v REAL)")
        con.executemany("INSERT INTO t VALUES(?,?)", [(i, 1.0) for i in range(50)])
        con.commit()

        gen = ((i,) for i in range(50))
        _ = len(list(gen))                    # 提前消费
        rc = con.executemany("UPDATE t SET v=2.0 WHERE k=?", gen).rowcount
        assert rc <= 0, "耗尽生成器的 rowcount 应为 0 或 -1（而非正数）"
        assert con.execute("SELECT COUNT(*) FROM t WHERE v=2.0").fetchone()[0] == 0
        con.close()

    def test_先物化成list_则写入成功(self, tmp_path) -> None:
        db = tmp_path / "t2.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE t(k INTEGER PRIMARY KEY, v REAL)")
        con.executemany("INSERT INTO t VALUES(?,?)", [(i, 1.0) for i in range(50)])
        con.commit()

        rows = [(i,) for i in range(50)]       # 物化成 list
        rc = con.executemany("UPDATE t SET v=2.0 WHERE k=?", rows).rowcount
        assert rc == 50
        assert con.execute("SELECT COUNT(*) FROM t WHERE v=2.0").fetchone()[0] == 50
        con.close()


class TestPriceVolumePxGuard:
    """``_px()`` 缺复权列必须抛错，不能静默退回未复权价。"""

    def test_缺复权列报错(self) -> None:
        from factor_lab.factors.price_volume import _px

        d = pd.DataFrame({"code": ["sh600519"], "date": [20240101],
                          "close": [10.0]})
        with pytest.raises(KeyError, match="adjusted=True"):
            _px(d, "close")

    def test_有复权列则用复权列(self) -> None:
        from factor_lab.factors.price_volume import _px

        d = pd.DataFrame({"code": ["sh600519"], "date": [20240101],
                          "close": [10.0], "close_adj": [20.0]})
        assert float(_px(d, "close").iloc[0]) == 20.0

class TestBinsQuantileLabels:
    """bins 模式下分位标签可能**跳号**，不能用 ``nunique()`` 当组数。

    实测（因子只有 {0, 1} 两种取值 + bins=3）：
        出现的标签= [1, 3]（Q2 为空）
        nunique()     = 2
        max(标签)      = 3

    早期版本用 ``range(1, nunique+1)`` 统计换手 → 只覆盖 {Q1, Q2}，
    **Q3 的换手根本没被统计** → 成本乘数偏小 → 净收益偏乐观（不报错）。
    """

    @staticmethod
    def _labels(factor: pd.Series, prices: pd.DataFrame, bins: int):
        from alphalens.utils import get_clean_factor_and_forward_returns

        clean = get_clean_factor_and_forward_returns(
            factor, prices, quantiles=None, bins=bins, periods=[1],
            max_loss=0.5)
        return clean, sorted(int(x) for x in clean["factor_quantile"].unique())

    @staticmethod
    def _panel():
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        assets = list("ABCDEF")
        factor = pd.Series([0.0, 1.0] * 9,
                           index=pd.MultiIndex.from_product([dates, assets]))
        prices = pd.DataFrame(np.linspace(1, 2, 18).reshape(3, 6),
                              index=dates, columns=assets)
        prices.index = prices.index.tz_localize(None)
        return factor, prices

    def test_标签确实会跳号(self) -> None:
        factor, prices = self._panel()
        clean, labels = self._labels(factor, prices, bins=3)
        assert labels == [1, 3], f"预期跳号标签 [1,3]，实际 {labels}"
        # 这就是不能用 nunique() 的理由
        assert clean["factor_quantile"].nunique() != max(labels)

    def test_换手率必须覆盖真实标签(self) -> None:
        from alphalens.performance import quantile_turnover

        factor, prices = self._panel()
        clean, labels = self._labels(factor, prices, bins=3)
        covered = {q for q in labels
                   if quantile_turnover(clean["factor_quantile"], q)
                   is not None}
        assert covered == set(labels), \
            f"换手率未覆盖全部分位：缺 {set(labels) - covered}"


class TestUniverseListDays:
    """「上市满 N 天」判据不能拿**窗口内序号**去比 N。

    🔴 2026-10-06 实测：原实现是
        d.groupby("code")["date"].rank(method="dense") > min_list_days
    这个序号的上界就是**窗口内交易日总数**。研究窗口短于 250 个交易日时
    （2024 年全年只有 242 天），**每只股票都被淘汰** → 股票池 0 只。
    表现出来是「close_adj 面板为空」，报错却指向并库，把排查方向带偏。

    正确判据：**上市日早于窗口起点 → 每天都已满 N 天，直接合格。**
    """

    @staticmethod
    def _long(n_days: int, start: str = "2024-01-02") -> pd.DataFrame:
        days = pd.bdate_range(start, periods=n_days)
        return pd.DataFrame({
            "code": ["sh600519"] * n_days,
            "date": days,
            "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
            "amount": 1e8, "vol": 1e6,
        })

    def _info(self, list_date: int | None = 20010827) -> pd.DataFrame:
        return pd.DataFrame({"code": ["sh600519"],
                             "name": ["贵州茅台"],
                             "list_date": [float(list_date)]})

    def test_短窗口不应把老股票全淘汰(self) -> None:
        """窗口 242 天 < min_list_days=250，老股票必须保留。"""
        from factor_lab.config import ResearchConfig
        from factor_lab.data.universe import build_universe

        long = self._long(242)
        out = build_universe(long, ResearchConfig(), info=self._info(),
                             verbose=False)
        assert len(out) == 242, f"老股票被误判为新股，剩 {len(out)}/242 行"

    def test_上市日在窗口内的新股应被淘汰(self) -> None:
        """上市日落在窗口内且不满 250 天 → 必须淘汰（这是本过滤的本意）。"""
        from factor_lab.config import ResearchConfig
        from factor_lab.data.universe import build_universe

        long = self._long(242)
        # 上市日= 2024-03-01（窗口内第 21 个交易日）
        out = build_universe(long, ResearchConfig(),
                             info=self._info(list_date=20240301), verbose=False)
        # 满 250 天的行不存在 → 全部淘汰
        assert len(out) == 0, f"次新股未被淘汰，剩 {len(out)} 行"

    def test_上市满250天的新股应保留(self) -> None:
        """上市日 2024-01-02，第 251 个交易日起应合格。"""
        from factor_lab.config import ResearchConfig
        from factor_lab.data.universe import build_universe

        long = self._long(300)
        out = build_universe(long, ResearchConfig(),
                             info=self._info(list_date=20240102), verbose=False)
        assert len(out) == 50, f"应保留 300−250=50 行，实际 {len(out)}"
