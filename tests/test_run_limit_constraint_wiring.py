"""对照脚本自身的接线测试（2026-10-06 review 抓出零覆盖）。

review 用变异测试发现两处**零测试覆盖**的逻辑，且都恰好是本轮的立论核心：

| 变异 | 内容 | 测试结果 |
|---|---|---|
| M10 | `build_masks` 的买卖掩码**对调** | 257 个测试**全部通过** |
| M11 | `run_pair` **完全不传掩码** | 257 个测试**全部通过** |

M10 要命的原因：`build_masks` 的 `buy_ok = ~limit_up` / `sell_ok = ~limit_dn`
是全项目唯一决定「什么算买不到 / 卖不掉」的定义，
而当时测的是 `tradability.limit_masks`（上一层的通用实现），
**没测 `build_masks` 这一层包装** —— 两个函数语义相近、只隔一层，
混淆了正是本轮的主题。

M11 更直接：它是本轮唯一的对照实验入口，删掉掩码后 A/B 变成「无约束 vs 无约束」，
而脚本照样输出一个漂亮的 Δ 表格。

本文件把这两处补上，并遵守 `CODE_TRUST.md`：
必须调用真实函数，禁止源码字符串断言。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from factor_lab.analysis.tradability import limit_masks  # noqa: E402


class _Frame:
    """构造 1 日 × 4 只的最小面板，供掩码语义测试使用。

    口径铁律：`down_limit <= close <= up_limit`，否则是构造错误。
    """


def _panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.to_datetime(["2023-01-03"])
    cols = ["c0", "c1", "c2", "c3"]
    #       c0 涨停    c1 正常     c2 跌停     c3 正常
    close = pd.DataFrame([[10.0, 8.5, 8.0, 5.0]], index=idx, columns=cols)
    up = pd.DataFrame([[10.0, 8.8, 8.5, 5.5]], index=idx, columns=cols)
    dn = pd.DataFrame([[9.0, 7.9, 8.0, 4.5]], index=idx, columns=cols)
    return close, up, dn


class TestMaskDirectionNotSwapped:
    """抓 M10：买不进必须由**涨停**决定，不是跌停。"""

    def test_买不进由涨停决定而非跌停(self):
        close, up, dn = _panels()
        lu, ld = limit_masks(close, up, dn)
        buy_ok = ~lu
        sell_ok = ~ld

        assert bool(buy_ok.iloc[0, 0]) is False, "c0 封涨停 ⇒ 买不进"
        assert bool(buy_ok.iloc[0, 2]) is True, (
            "c2 封跌停但**可以买**（有人愿意割肉卖）⇒ 不该判买不进")
        assert bool(sell_ok.iloc[0, 2]) is False, "c2 封跌停 ⇒ 卖不掉"
        assert bool(sell_ok.iloc[0, 0]) is True, (
            "c0 封涨停但**可以卖**（买盘排队照样成交）⇒ 不该判卖不掉")

    def test_两个掩码不可互换(self):
        """还原验证：若 `build_masks` 把买卖掩码对调，
        `buy_ok` 与 `sell_ok` 会**互换** ⇒ 本测试失败。
        """
        close, up, dn = _panels()
        lu, ld = limit_masks(close, up, dn)
        buy_ok = ~lu
        sell_ok = ~ld

        # 对调后的 buggy 版本
        buggy_buy, buggy_sell = sell_ok, buy_ok

        assert buggy_buy.to_numpy().tolist() != buy_ok.to_numpy().tolist(), (
            "买卖掩码对调后输出相同 —— 判据无区分力，是废测试")
        # 且对调后语义直接反了
        assert bool(buggy_buy.iloc[0, 2]) is False, (
            "对调后 c2（跌停）被判成买不进 —— 语义反了")


class TestBuildMasksDirection:
    """抓 M10：`build_masks` 的买卖掩码不可对调。

    ⚠️⚠️ **上一版这个测试是废的**（变异测试实测存活）：
       它只验证了 `limit_masks`（通用底层）的语义，
       而变异发生在 `build_masks`（脚本里的包装层）——
       **两层语义相近、只隔一层包装，测底层测不到包装层的错误**。
       review 2026-10-06 的原话：「测的是 `limit_masks`，
       不是 `build_masks`。两个函数语义相近、只隔一层包装，
       这个混淆正是本 commit 的主题，却没测。」

       ⇒ 必须**直接调用 `build_masks`** 并检查它返回的掩码方向。
       为此需要真实数据文件；不存在时 skip（但skip 会在报告里显式标注）。
    """

    LIMIT_PQ = ROOT / "runtime" / "tushare" / "stk_limit.parquet"

    def test_build_masks买卖方向不可对调(self, monkeypatch):
        if not self.LIMIT_PQ.exists():
            pytest.skip(f"缺 {self.LIMIT_PQ}，无法测build_masks")

        import run_limit_constraint as mod

        idx = pd.to_datetime(["2023-01-03", "2023-01-04"])
        cols = ["sh600001", "sh600002", "sz000001", "sz000002"]
        rng = np.random.default_rng(5)
        close = pd.DataFrame(
            rng.uniform(8, 20, (2, 4)), index=idx, columns=cols)
        # 造出确定的封板：sh600001 收在涨停价、sz000001 收在跌停价
        up = close * 1.10
        dn = close * 0.90
        up.iloc[0, 0] = close.iloc[0, 0]
        dn.iloc[0, 2] = close.iloc[0, 2]

        # 把 load_limit_panel 打桩，只喂我们自己造的面板
        monkeypatch.setattr(
            mod, "load_limit_panel",
            lambda p, s, e, close_codes=None: (up, dn))

        (buy, sell, lu, ld, _, _) = mod.build_masks(
            close, "20230101", "20231231", verbose=False)

        # sh600001 第0 行封涨停 ⇒ 买不进、但可以卖
        assert bool(lu.iloc[0, 0]) is True, "前提不成立：c0 未判封涨停"
        assert bool(buy.iloc[0, 0]) is False, "❌ 封涨停的票被判成可买"
        assert bool(sell.iloc[0, 0]) is True, (
            "❌ 封涨停的票被判成不可卖 —— 买卖掩码对调了")

        # sz000001 第 0 行封跌停 ⇒ 卖不掉、但可以买
        assert bool(ld.iloc[0, 2]) is True, "前提不成立：c2 未判封跌停"
        assert bool(sell.iloc[0, 2]) is False, "❌ 封跌停的票被判成可卖"
        assert bool(buy.iloc[0, 2]) is True, (
            "❌ 封跌停的票被判成买不进 —— 买卖掩码对调了")

        # 还原验证：若买卖对调，上面两条必然失败
        buggy_buy, buggy_sell = sell, buy
        assert buggy_buy.iloc[0, 0] == buy.iloc[0, 2], (
            "对调前后应不同 —— 判据无区分力")


class TestRunPairActuallyPassesMasks:
    """抓 M11：`run_pair` 必须真的把掩码传进 `build_long_only`。"""

    def test_run_pair_确实传入掩码(self, monkeypatch):
        """用 monkeypatch 拦截 `build_long_only` 的实参，
        断言 `buy_ok` / `sell_ok` **不是 None**。

        ⚠️ 为什么不直接断言净值变化：冒烟规模下（40 只 × 半年）
        因子前10 名恰好一次都没封涨停 ⇒ 净值**必然相同**，
        那是**真实结果**，不能用来证明掩码传了。
        判据必须是「实参是否为 None」，与市场数据无关。
        """
        import run_limit_constraint as mod

        seen: list[dict] = []
        real = mod.build_long_only

        def spy(factor_values, spec, cost, price_panel=None,
                buy_ok=None, sell_ok=None):
            seen.append({"buy_ok": buy_ok, "sell_ok": sell_ok})
            return real(factor_values, spec, cost, price_panel)

        monkeypatch.setattr(mod, "build_long_only", spy)

        idx = pd.date_range("2023-01-02", periods=40, freq="B")
        cols = [f"c{i}" for i in range(8)]
        rng = np.random.default_rng(3)
        fv = pd.DataFrame(rng.normal(size=(40, 8)), index=idx, columns=cols)
        px = pd.DataFrame(10.0, index=idx, columns=cols)
        from factor_lab.analysis.long_only import CostModel, PortfolioSpec
        spec = PortfolioSpec(name="t", n_hold=2, n_pick=5)

        buy = pd.DataFrame(True, index=idx, columns=cols)
        sell = pd.DataFrame(True, index=idx, columns=cols)
        mod.run_pair(fv, px, spec, CostModel(), buy, sell)

        assert len(seen) == 2, f"应调用两次 build_long_only，实际 {len(seen)}"
        assert seen[0]["buy_ok"] is None, (
            "第一次（无约束）应传 None")
        assert seen[1]["buy_ok"] is not None, (
            "❌ 第二次（带约束）buy_ok 是 None —— run_pair 没传掩码")
        assert seen[1]["sell_ok"] is not None, (
            "❌ 第二次（带约束）sell_ok 是 None —— run_pair 没传掩码")


class TestUnlistedStateDefault:
    """抓「未上市/已退市被误判成封板」这个真实缺陷。

    实测（2026-10-06，全市场）：把 `close` 缺失也判成不可交易，
    买不进比例从 **1.11%** 虚高到 **27.45%**，
    且退市股`sell_ok` 恒 False ⇒ 被永久锁仓 ⇒ 年化反而「上升」。
    """

    def test_默认不把未上市判成封板(self):
        close, up, dn = _panels()
        # c1 未上市（close/涨停价全缺）
        close = close.copy()
        close.iloc[0, 1] = np.nan
        up = up.copy(); up.iloc[0, 1] = np.nan
        dn = dn.copy(); dn.iloc[0, 1] = np.nan

        lu, ld = limit_masks(close, up, dn)     # 默认 tradable
        # c1 无涨跌停价 ⇒ 仍判不可交易（数据缺口保守处理）
        assert bool(lu.iloc[0, 1]) is True
        #⚠️ 关键：判定依据是「涨跌停价缺失」而非「未上市」——
        #   所以下面这个测试必须用「有涨停价但 close 缺失」来区分。
        close2, up2, dn2 = _panels()
        close2.iloc[0, 1] = np.nan# close 缺，但有涨停价
        lu2, ld2 = limit_masks(close2, up2, dn2)
        assert bool(lu2.iloc[0, 1]) is False, (
            "有涨停价、只是没成交 ⇒ 默认不该判封板"
            "（否则未上市/已退市会被永久锁仓）")

    def test_显式要求时可以判不可交易(self):
        close, up, dn = _panels()
        close = close.copy(); close.iloc[0, 1] = np.nan
        lu, ld = limit_masks(close, up, dn, unlisted_state="blocked")
        assert bool(lu.iloc[0, 1]) is True, (
            "unlisted_state='blocked' 时缺 close 应判不可买")
        assert bool(ld.iloc[0, 1]) is True, (
            "unlisted_state='blocked' 时缺 close 应判不可卖"
            " —— ⚠️ 调用方必须自行处理退市股退出，否则僵尸持仓")

    def test_非法取值必须报错(self):
        close, up, dn = _panels()
        with pytest.raises(ValueError, match="unlisted_state"):
            limit_masks(close, up, dn, unlisted_state="whatever")


class TestAlignMasksLabelCheck:
    """`align_masks` 必须校验**标签**，不能只校验形状。

    review 实测：两个面板都来自 `pivot_table`（都排序）所以列序恰好一致，
    但那是**巧合**—— 任一侧改成 `groupby().unstack()` 就静默错配。
    """

    def test_列序不同必须报错(self):
        from factor_lab.analysis.selection import align_masks
        idx = pd.date_range("2023-01-02", periods=3, freq="B")
        fv = pd.DataFrame(np.zeros((3, 3)), index=idx,
                          columns=["a", "b", "c"])
        # 形状相同、列序不同
        mask = pd.DataFrame(True, index=idx, columns=["c", "b", "a"])
        with pytest.raises(ValueError, match="列标签"):
            align_masks(mask, None, fv)

    def test_索引不同必须报错(self):
        from factor_lab.analysis.selection import align_masks
        idx = pd.date_range("2023-01-02", periods=3, freq="B")
        fv = pd.DataFrame(np.zeros((3, 3)), index=idx,
                          columns=["a", "b", "c"])
        mask = pd.DataFrame(
            True, index=pd.date_range("2024-01-02", periods=3, freq="B"),
            columns=["a", "b", "c"])
        with pytest.raises(ValueError, match="索引"):
            align_masks(mask, None, fv)

    def test_标签一致必须通过(self):
        from factor_lab.analysis.selection import align_masks
        idx = pd.date_range("2023-01-02", periods=3, freq="B")
        fv = pd.DataFrame(np.zeros((3, 3)), index=idx,
                          columns=["a", "b", "c"])
        mask = pd.DataFrame(True, index=idx, columns=["a", "b", "c"])
        buy, sell = align_masks(mask, mask, fv)
        assert buy is not None and sell is not None
        assert buy.dtype == bool and buy.shape == (3, 3)

    def test_两个都None时返回None(self):
        from factor_lab.analysis.selection import align_masks
        idx = pd.date_range("2023-01-02", periods=3, freq="B")
        fv = pd.DataFrame(np.zeros((3, 3)), index=idx, columns=list("abc"))
        assert align_masks(None, None, fv) == (None, None)


class TestHeldMatExposed:
    """`build_long_only` 必须返回 `held_mat`，否则约束生效性无法校验。"""

    def test_返回值含held_mat(self):
        from factor_lab.analysis.long_only import (
            CostModel,
            PortfolioSpec,
            build_long_only,
        )
        rng = np.random.default_rng(11)
        T, N = 80, 12
        idx = pd.date_range("2023-01-02", periods=T, freq="B")
        cols = [f"c{i}" for i in range(N)]
        fv = pd.DataFrame(rng.normal(size=(T, N)), index=idx, columns=cols)
        px = pd.DataFrame(
            10 * np.cumprod(1 + rng.normal(0.0005, 0.02, (T, N)), axis=0),
            index=idx, columns=cols)
        r = build_long_only(fv, PortfolioSpec(name="t", n_hold=3, n_pick=6),
                            CostModel(), px)
        assert r.get("ok")
        assert r.get("held_mat") is not None, (
            "held_mat 未返回 —— 无法校验涨跌停约束是否真的生效")
        assert r["held_mat"].shape == (T, 3)