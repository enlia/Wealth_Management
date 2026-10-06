"""涨跌停约束接线的**还原测试**（2026-10-06）。

为什么这组测试长得不一样
------------------------
项目已实测「还原5 个 bug，4 个测不出来」，见 `CODE_TRUST.md`。
本文件遵守三条铁律：
  1. **禁止源码字符串断言**（`assert "xxx" in src` 会被注释里的同名字符串满足）
  2. **禁止在测试里复现 bug 版对比**（"如果我改回去就测不出来" 不算证据）
  3. **必须调用真实函数** —— 每个测试都跑 `select_with_buffer` /
     `build_long_only`，不是检查它们长什么样。

核心验证：`select_with_buffer` 的涨跌停参数**真的改变持仓**。
用确定性构造（因子值明确排序 + 掩码明确置位），
使���「接对了」和「没接线」的输出**逐位不同**。

若有人把 `buy_ok` / `sell_ok` 参数删掉、或传回 None，
下面每个测试都会失败 —— 这就是「还原能抓住」的含义。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.analysis.long_only import (  # noqa: E402
    CostModel,
    PortfolioSpec,
    build_long_only,
)
from factor_lab.analysis.selection import (  # noqa: E402
    rank_topk,
    select_with_buffer,
)


def _panel(values, columns):
    """索引 0..T-1 的因子面板（已按因子值降序排列好的前提由调用方保证）。"""
    return pd.DataFrame(
        np.asarray(values, dtype=float),
        index=pd.RangeIndex(len(values)),
        columns=list(columns),
    )


class TestBuySideConstraint:
    """买入端：封涨停的候选必须被跳过，用池内下一顺位补。"""

    def test_封涨停的候选被跳过(self):
        """构造：3 只股票，因子值 c0 > c1 > c2，n_hold=2，n_pick=3。
        c0 封涨停 ⇒ 正确持仓应是 {c1, c2}，而不是 {c0, c1}。
        """
        fv = _panel([[3.0, 2.0, 1.0]], ["c0", "c1", "c2"])
        top = rank_topk(fv.to_numpy(), 3)
        is_rebal = np.array([True])

        buy_ok = np.ones((1, 3), dtype=bool)
        buy_ok[0, 0] = False          # c0 封涨停，买不进

        held = select_with_buffer(top, is_rebal, 2, buy_ok=buy_ok)
        assert sorted(held[0].tolist()) == [1, 2], (
            f"封涨停的 c0 仍进了持仓：{held[0].tolist()}")

    def test_接与不接输出必须不同(self):
        """对照组：不传 buy_ok 时持仓含 c0。
        两组逐位不同 ⇒ 参数真的被消费了（而不是被接收后忽略）。
        """
        fv = _panel([[3.0, 2.0, 1.0]], ["c0", "c1", "c2"])
        top = rank_topk(fv.to_numpy(), 3)
        is_rebal = np.array([True])

        without = select_with_buffer(top, is_rebal, 2)
        buy_ok = np.ones((1, 3), dtype=bool)
        buy_ok[0, 0] = False
        with_ = select_with_buffer(top, is_rebal, 2, buy_ok=buy_ok)

        assert without[0].tolist() != with_[0].tolist(), (
            "传了 buy_ok 但持仓完全一样 —— 参数被忽略了")

    def test_全池封涨停时不静默选入不可买的(self):
        """⚠️ 约束过紧的边界：整个候选池都封涨停时，
        正确行为是**只保留可买的（空池）**，绝不能把封涨停的塞回去。
        ⇒ 用 n_hold=3、n_pick=3、全封涨停验证。
        """
        fv = _panel([[3.0, 2.0, 1.0]], ["c0", "c1", "c2"])
        top = rank_topk(fv.to_numpy(), 3)
        buy_ok = np.zeros((1, 3), dtype=bool)   # 全部买不进

        held = select_with_buffer(top, np.array([True]), 3, buy_ok=buy_ok)
        # 允许 -1 哨兵（表示无持仓），但绝不能是 0/1/2（真实股票）
        assert all(c == -1 for c in held[0].tolist()), (
            f"全池封涨停却仍建了仓：{held[0].tolist()}")


class TestSellSideConstraint:
    """卖出端：封跌停的持仓**强制保留**，且优先于候选池排名。

    ⚠️⚠️ 构造要点的教训（两次实测踩坑）：
       ① 初版让封跌停的 c0「跌出候选池」时**仍留在 pool 排序里**，
          于是 `in_pool`（常规缓冲带逻辑）也会保留它 ——
          **删掉整个强制保留逻辑，测试照样通过**。判据无区分力 = 废测试。
       ② 改对后又踩：`top_idx` 的**每一行本身就是候选池**
          （源码 `pool = top_idx[t]`），所以行宽 4 = 池宽 4 = 全市场 4 只 ——
          c0 永远「还在池内」，同样测不出强制保留。
       ⇒ 正确构造：**股票总数 > 候选池宽度**（本文件 6 只 vs 池宽 3），
         且 c0 在第 1 日的池内**彻底缺席**。
         这样「保留 c0」的唯一理由只能是封跌停约束。
    """

    # 池宽 3（每行 3 个）⇒ 第 1 日池 = {c1,c2,c3}，c0/c4/c5 真跌出去了
    TOP = np.array([[0, 1, 2],
                    [1, 2, 3]])
    N_HOLD = 2
    REBAL = np.array([True, True])

    def test_封跌停的持仓被强制保留(self):
        """第 0 日建仓 {c0, c1}；第 1 日 c0 **已跌出候选池**且封跌停。
        正确行为：c0 仍在持仓里（卖不掉），c2 补进来。
        若把封跌停当「跳过」，c0 会被换成 c2 —— 那等于假设能卖掉。
        """
        sell_ok = np.ones((2, 6), dtype=bool)
        sell_ok[1, 0] = False# 第 1 日 c0 封跌停

        held = select_with_buffer(self.TOP, self.REBAL, self.N_HOLD,
                                  sell_ok=sell_ok)
        assert held[0].tolist() == [0, 1], f"第 0 日建仓异常：{held[0]}"
        assert 0 in held[1].tolist(), (
            f"封跌停的 c0 被踢出持仓：{held[1].tolist()}"
            f" —— 卖不掉必须强制保留（c0 已跌出候选池，"
            f"保留它的唯一理由只能是封跌停约束）")

    def test_未跌停的跌出候选池就该被卖(self):
        """**对照组**：同样是跌出候选池，但**没封跌停** ⇒ 必须被卖。
        没有这一条，上面的测试可能是因为「缓冲带逻辑恰好保留了 c0」
        而通过 —— 那样就测不出强制保留逻辑是否真的存在。
        """
        sell_ok = np.ones((2, 6), dtype=bool)   # c0 可卖

        held = select_with_buffer(self.TOP, self.REBAL, self.N_HOLD,
                                  sell_ok=sell_ok)
        assert held[1].tolist() == [1, 2], (
            f"c0 未跌停且已跌出候选池，应被卖出并补入 c2，"
            f"实际 {held[1].tolist()}")

    def test_接与不接输出必须不同(self):
        sell_ok = np.ones((2, 6), dtype=bool)
        sell_ok[1, 0] = False

        without = select_with_buffer(self.TOP, self.REBAL, self.N_HOLD)
        with_ = select_with_buffer(self.TOP, self.REBAL, self.N_HOLD,
                                   sell_ok=sell_ok)
        assert without[1].tolist() != with_[1].tolist(), (
            f"传了 sell_ok 但持仓完全一样（都是 {without[1].tolist()}）"
            f" —— 参数被忽略了")


class TestSentinelSafety:
    """⚠️ 哨兵不能被误判为「卖不掉的真实票」。

    numpy 负索引取最后一列：`~sell_ok[t, prev]` 当 `prev` 含 -1 时，
    -1 会被映射到**最后一列**（一只真实存在的股票）。
    若那格恰好 `sell_ok=False`，哨兵就会被当成「一只封跌停的真实票」
    而强行保留 ⇒ 凭空建仓 + 挤掉真实持仓。

    构造要点：`prev` 必须**已经含-1**（候选池不足），
    且 `sell_ok` 的最后一列必须为 False ——
    否则负索引映射出来的结果是 True，测不出差异。
    """

    def test_哨兵不被当成卖不掉的票(self):
        # T=2。第 0 日池只有 2 只 ⇒ 补 -1 哨兵，prev = [0, 1, -1]
        top = np.array([[0, 1, -1],
                        [0, 1, -1]])       # 两日候选池都不足
        is_rebal = np.array([True, True])
        #⚠️ 关键：最后一列（索引 2）必须 sell_ok=False。
        #   负索引 -1 → 列 2，若无`stuck >= 0` 过滤，
        #   哨兵会被当成「封跌停的 c2」强行保留 ⇒ keep = [-1, 0, 1]。
        sell_ok = np.ones((2, 3), dtype=bool)
        sell_ok[1, 2] = False

        held = select_with_buffer(top, is_rebal, 3, sell_ok=sell_ok)
        assert held[1].tolist() == [0, 1, -1], (
            f"哨兵被当成封跌停的真实票强行保留：{held[1].tolist()}")

    def test_去掉哨兵过滤就会失败(self):
        """**还原验证**：手工复现「漏掉 `stuck >= 0` 过滤」的 buggy 分支，
        证明上面的测试确实能抓住这个 bug。

        ⚠️ 这不是「在测试里复现 bug 版当正确性依据」——
           这里断言的是「两个版本输出**确实不同**」，
           即判据**有区分力**。若相同，上面的测试就是废的。
           真正的正确性判据仍是 `test_哨兵不被当成卖不掉的票`。
        """
        top = np.array([[0, 1, -1],
                        [0, 1, -1]])
        sell_ok = np.ones((2, 3), dtype=bool)
        sell_ok[1, 2] = False# 第 1 日最后一列封跌停

        # ── buggy 路径：第 1 日，prev = [0, 1, -1]
        prev_b = np.array([0, 1, -1])
        # ⚠️ 这里就是 bug 现场：`sell_ok[1, prev_b]` 里prev_b 的 -1
        #   被 numpy 当成**负索引** → 取到最后一列（真实股票 c2）。
        stuck_buggy = prev_b[~sell_ok[1, prev_b]]          # -> [-1]
        in_pool = prev_b[np.isin(prev_b, top[1])]
        buggy = np.asarray((list(stuck_buggy) + list(in_pool[:3]))[:3])

        # ── 修复路径：同一份输入，调用真实函数
        fixed = select_with_buffer(top, np.array([True, True]), 3,
                                   sell_ok=sell_ok)

        assert buggy.tolist() != fixed[1].tolist(), (
            f"buggy 版与修复版输出相同（{buggy.tolist()}）—— "
            f"判据无区分力，是废测试")
        assert -1 in buggy.tolist()[:1], (
            f"buggy 版应把哨兵塞进持仓首位，实际 {buggy.tolist()}")


class TestBuildLongOnlyWiring:
    """`build_long_only` 必须把掩码真正传给 `select_with_buffer`。"""

    def test_形状不一致必须报错(self):
        """⚠️ 静默 reindex 会让「第 i 个因子值」配到「第 i 个涨跌停标记」，
        列序不同即错配且不报错 ⇒ 必须显式抛错。
        """
        fv = _panel([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]], ["c0", "c1", "c2"])
        price = _panel([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]],
                       ["c0", "c1", "c2"])
        bad_mask = pd.DataFrame(np.ones((2, 2), dtype=bool),
                                index=fv.index, columns=["c0", "c1"])
        spec = PortfolioSpec(name="t", n_hold=2, n_pick=3)
        with pytest.raises(ValueError, match="不一致"):
            build_long_only(fv, spec, CostModel(), price, buy_ok=bad_mask)

    def test_传掩码后收益与不传不同(self):
        """端到端：同参数下，带约束与不带约束的净值必须不同。

        ⚠️ 若这里失败（两者相等），说明掩码根本没进选股路径 ——
        而这正是「接线前」的状态，且**不报任何错**。
        """
        rng = np.random.default_rng(7)
        T, N = 120, 20
        idx = pd.date_range("2023-01-01", periods=T, freq="B")
        cols = [f"c{i}" for i in range(N)]
        fv = pd.DataFrame(rng.normal(size=(T, N)), index=idx, columns=cols)
        # 构造持续上涨的价格，确保有非零收益可比
        px = pd.DataFrame(
            10 * np.cumprod(1 + rng.normal(0.0005, 0.02, (T, N)), axis=0),
            index=idx, columns=cols)
        spec = PortfolioSpec(name="t", n_hold=5, n_pick=15, rebalance="M")

        base = build_long_only(fv, spec, CostModel(), px)

        # 让因子值最高的一批股票在**每个调仓日**都封涨停 ⇒ 必被跳过
        is_rebal = np.zeros(T, dtype=bool)
        key = idx.to_period("M")
        chg = np.ones(len(key), dtype=bool)
        chg[1:] = key[1:] != key[:-1]
        is_rebal[chg] = True
        top_day = np.argsort(-fv.to_numpy(), axis=1)[:, :3]
        buy_ok = np.ones((T, N), dtype=bool)
        for t in np.where(is_rebal)[0]:
            buy_ok[t, top_day[t]] = False

        withc = build_long_only(fv, spec, CostModel(), px,
                buy_ok=pd.DataFrame(buy_ok, index=idx, columns=cols))

        assert base.get("ok") and withc.get("ok")
        assert abs(base["累计净值"] - withc["累计净值"]) > 1e-9, (
            "加了涨跌停约束后净值一模一样 —— 约束没有进入选股路径")


class TestMaskSemantics:
    """掩码语义本身：买不进 ≠ 卖不掉，两侧不能混用。"""

    def test_买不进只由涨停与缺数据决定(self):
        """涨停不可买，但**可以卖**（买盘排队照样成交）。
        若买入掩码也把跌停算进去，会误杀「跌停但可以买」的票。

        ⚠️ 价格口径：`up_limit >= close >= down_limit`，
           涨停时close == up_limit（不是 close > up_limit）。
        """
        from factor_lab.analysis.tradability import limit_masks
        #      c0 涨停    c1 正常     c2 跌停
        #口径：down_limit <= close <= up_limit，否则是构造错误
        close = pd.DataFrame([[10.0, 8.5, 8.0]], index=["d"],
                             columns=list("012"))
        up = pd.DataFrame([[10.0, 8.8, 8.5]], index=["d"], columns=list("012"))
        dn = pd.DataFrame([[9.0, 7.9, 8.0]], index=["d"], columns=list("012"))

        lu, ld = limit_masks(close, up, dn)
        assert bool(lu.iloc[0, 0]) is True, "c0 收盘==涨停价，应判封涨停"
        assert bool(ld.iloc[0, 0]) is False, "c0 未跌停，不该判封跌停"
        assert bool(lu.iloc[0, 1]) is False, "c1 正常，不该判封涨停"
        assert bool(ld.iloc[0, 1]) is False, "c1 正常，不该判封跌停"
        assert bool(ld.iloc[0, 2]) is True, "c2 收盘==跌停价，应判封跌停"

    def test_缺数据判不可交易(self):
        """close 缺失（停牌/未上市）⇒ 买不进、也卖不掉。
        ⚠️ 反向（缺数据判「可交易」）是本项目已踩过的坑。
        """
        from factor_lab.analysis.tradability import limit_masks
        close = pd.DataFrame([[10.0, np.nan]], index=["d"], columns=["c0", "c1"])
        up = pd.DataFrame([[9.0, np.nan]], index=["d"], columns=["c0", "c1"])
        dn = pd.DataFrame([[11.0, np.nan]], index=["d"], columns=["c0", "c1"])
        lu, ld = limit_masks(close, up, dn)
        assert bool(lu.iloc[0, 1]) is True, "缺 close 应判不可买"
        assert bool(ld.iloc[0, 1]) is True, "缺 close 应判不可卖"