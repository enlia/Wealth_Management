"""市值分层的时点与口径还原测试。

为什么这类测试值得单独写
------------------------
2026-10-06 实测的 BLOCK 是：`run_size_decile.py` 用**窗口末日**市值分层，
而市值 ≈ 股价 × 股本 ⇒ 用年末市值分层等于用「年内涨幅」分层，
再拿该层的收益当基准 —— **循环论证**。
实测 2025 窗口大盘层：按年初市值 +31.25% vs 按年末市值 +49.80%。

⚠️ **注释与 docstring 都会说谎，只有测试不会**（CODE_TRUST P23）。
   本项目已多次出现「注释写 A、代码做 B 且不报错」的情况，
   所以口径约束必须有**还原测试**兜底，不能只写在注释里。

⚠️ 这些测试**不依赖本机数据库**（CI 上可跑）。
   数据口径类测试若依赖本机文件，在 CI 上会被 skip 掉 ——
   而 CI 恰恰是最需要它们的地方。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from size_decile_core import (  # noqa: E402
    MIN_STOCKS_PER_BUCKET,
    interpret_layer_median,
    split_buckets,
)


class TestSplitBuckets:
    """分位数切层本身。"""

    def test_三等分数量均衡(self):
        n = MIN_STOCKS_PER_BUCKET * 3
        m = pd.Series(np.arange(float(n)), index=[f"c{i}" for i in range(n)])
        b = split_buckets(m)
        assert len(b["大"]) == len(b["小"])
        assert len(b["大"]) < len(b["全"]), "层不能比全集还大"
        assert set(b["大"]) | set(b["中"]) | set(b["小"]) == set(m.index)
        assert not (set(b["大"]) & set(b["小"]))

    def test_缺失市值不进入任何层(self):
        """🔴 缺失若被填 0，会全部落进「小盘层」= 收益最高的一层。"""
        # ⚠️ 池子要留够 3×MIN_STOCKS_PER_BUCKET 的**有效**股票：
        #   门槛卡的是有效样本，不是全池（见 split_buckets 的注释）。
        #   初版取 n=360、缺失 20% ⇒ 有效仅 288 < 360 ⇒ 门槛正确地返回 {}，
        #   测试却直接取 b["大"] 而 KeyError ⇒ **测试自己没跟上口径**。
        n = MIN_STOCKS_PER_BUCKET * 6
        m = pd.Series(np.arange(float(n)), index=[f"c{i}" for i in range(n)])
        n_missing = n // 2
        m.iloc[:n_missing] = np.nan
        b = split_buckets(m)
        全部 = set(b["大"]) | set(b["中"]) | set(b["小"])
        assert 全部 == {f"c{i}" for i in range(n_missing, n)}
        assert not any(i in 全部 for i in [f"c{i}" for i in range(n_missing)])

    def test_样本不足返回空(self):
        m = pd.Series(np.arange(10.0), index=[f"c{i}" for i in range(10)])
        assert split_buckets(m) == {}


class TestLookAheadInBucketing:
    """🔴🔴 BLOCK 还原：分层必须用窗口【首日】市值。"""

    def test_用首日市值时大盘层不等于涨得最多的(self):
        """构造：年初小盘股年内大涨，年末市值超过大盘股。

        若实现用末日市值，这只小盘股会被划进「大」层 ——
        而它之所以涨，恰恰是它**年初**的小市值没预测到的。
        """
        n = MIN_STOCKS_PER_BUCKET * 3
        codes = [f"c{i:04d}" for i in range(n)]
        # 年初：c0000 是**最小**的一只；其余 100~200
        年初 = pd.Series([1.0] + [100.0 + i for i in range(n - 1)],
                         index=codes)
        # 年末：c0000 涨 100 倍，变成**最大**的一只
        年末 = 年初.copy()
        年末.iloc[0] = 1e6

        by_early = split_buckets(年初)
        by_late = split_buckets(年末)

        assert codes[0] not in by_early["大"], "按年初市值必须是中小盘层"
        assert codes[0] in by_late["大"], "按年末市值会变成大盘层（循环论证）"

    def test_两口径的大盘层收益可以差很多(self):
        """量化这个偏差：按结果分组会系统性高估基准。"""
        n = MIN_STOCKS_PER_BUCKET * 3
        codes = [f"c{i:04d}" for i in range(n)]
        市值 = pd.Series([1.0] + [100.0 + i for i in range(n - 1)],
                         index=codes)
        # 只有 c0000 涨（它涨 100 倍），其余全不动
        收益 = pd.Series(0.0, index=codes)
        收益.iloc[0] = 100.0

        年末 = 市值.copy()
        年末.iloc[0] = 1e6

        按年末 = float(收益[split_buckets(年末)["大"]].mean())
        按年初 = float(收益[split_buckets(市值)["大"]].mean())
        assert 按年末 > 按年初 * 3, (
            f"按结果分组把涨得最多的股票放进大盘层，基准被抬高："
            f"年末口径 {按年末:.4f} vs 年初口径 {按年初:.4f}")

    def test_size_decile_run_使用窗口首日(self):
        """🔴 核心 BLOCK 还原：`size_decile_run` 必须用**窗口首日**市值分层。

        做法：spy 住 `run_in_subset`，记录每次收到的股票集合。
        由于「大」层与「小」层**大小相同**，不能用集合大小区分
        ⇒ 改为直接断言「codes[0] 不在最大市值的三分之一里」：
        它在窗口首日（`dates[200]`）市值恒为 10，是全池最小。
        若实现用末日（`dates[-1]`）分层，它的市值变成 9000，会进大层。
        """
        import size_decile_core
        from size_decile_core import size_decile_run

        from factor_lab.analysis.costs import CostModel

        dates = pd.bdate_range("2025-01-01", periods=250)
        # 门槛卡的是「每层 ≥MIN_STOCKS_PER_BUCKET」⇒ 全池需 3×，见 split_buckets
        n = MIN_STOCKS_PER_BUCKET * 3 + 20
        codes = [f"sh6000{i:02d}" for i in range(n)]
        px = pd.DataFrame(100.0, index=dates, columns=codes)
        # 🔴 市值必须**各不相同且单调递增**。
        #    初版给 139 只票同一个市值 1000，导致层内均值只差 21，
        #    「按层内均值识别大盘层」失效 ⇒ 变异后测试依然全绿
        #    （实测：te[-1] 变异下 12 passed）。
        #    ⇒ **退化数据 = 橡皮章测试**，与「恒等断言」同一类陷阱。
        base = pd.Series(np.linspace(100.0, 2000.0, n), index=codes)
        mcap = pd.DataFrame(np.tile(base.to_numpy(), (len(dates), 1)),
                            index=dates, columns=codes)
        # codes[0] 首日最小（100）、末日变成最大（1e6）
        mcap.iloc[-1, 0] = 1e6

        te = list(dates[200:])
        windows = [([d for d in dates[:200]], te)]
        seen: list[set] = []

        def spy(z_sub, px_sub, *a, **kw):
            seen.append(set(px_sub.columns))
            return 0.0, ""

        orig = size_decile_core.run_in_subset
        size_decile_core.run_in_subset = spy
        try:
            size_decile_run(pd.DataFrame(1.0, index=dates, columns=codes),
                            px, mcap, windows, CostModel(), 30, 3, "f")
        finally:
            size_decile_core.run_in_subset = orig

        assert len(seen) == 4, f"应有 4 层（全+大中小），spy 记录 {len(seen)}"
        n = len(codes)
        三层 = [s for s in seen if len(s) < n]
        assert len(三层) == 3

        # 🔴 核心断言：**用层内平均市值识别出「大盘层」，再检查成员**。
        #
        # ⚠️ **不能靠 `seen[i]` 的下标猜哪层是大盘**（dict 遍历顺序不保证），
        #   也不能靠集合大小（大层与小层大小相同）。
        #   只能用「层内平均市值」这个**可观测且顺序无关**的量。
        #
        # ⚠️ **不能用 quantile 复刻「期望集合」来对比** ——
        #   实现用 `rank` 切层（修「层比全集还大」的 bug），
        #   与 quantile 分位在重复值多时不等价；
        #   那样写的断言比较两套不同算法 ⇒ 恒不相等 ⇒ 变成橡皮章
        #   （2026-10-06 实测：这样写的测试在 bug 存在时依然 12 passed）。
        首日市值 = mcap.loc[te[0]]
        末日市值 = mcap.loc[dates[-1]]
        # ⚠️ pandas 不接受 set 作索引器，必须转 list
        大盘层 = max(三层, key=lambda s: 首日市值[list(s)].mean())
        小盘层 = min(三层, key=lambda s: 首日市值[list(s)].mean())
        assert 大盘层 is not 小盘层, (
            "按首日市值识别的最大层与最小层相同，构造有问题")

        # codes[0] 首日市值最小、末日市值最大
        # ⇒ 首日口径下它必属小盘层，绝不在大盘层
        assert codes[0] not in 大盘层, (
            f"{codes[0]} 在窗口首日市值最小（{首日市值[codes[0]]:.0f}），"
            f"却出现在大盘层（层内均值 {首日市值[list(大盘层)].mean():.0f}）⇒ "
            f"分层用的是窗口【末日】市值（前视，循环论证）")
        assert codes[0] in 小盘层, "构造前提不成立：首日口径下它应在小盘层"

        # 反向校验：用末日市值识别的大盘层，它**必须**在
        assert codes[0] in max(三层, key=lambda s: 末日市值[list(s)].mean()), (
            "构造前提不成立：末日口径下它应是大盘层")


class TestInterpret:
    def test_层内为正判通过(self):
        lvl, msg = interpret_layer_median({"大": 0.08, "中": 0.06, "小": 0.02})
        assert lvl == "✅"
        assert "真实存在" in msg

    def test_层内为负判无效(self):
        lvl, msg = interpret_layer_median({"大": -0.02, "中": -0.05, "小": -0.01})
        assert lvl == "❌"
        assert "信号无效" in msg

    def test_层数不足不硬判(self):
        lvl, _ = interpret_layer_median({"大": 0.08})
        assert lvl == "⚠"

    def test_阈值不是零(self):
        """+0.3% 的层内超额不应判为「信号真实存在」。"""
        lvl, _ = interpret_layer_median({"大": 0.003, "中": 0.002, "小": 0.001})
        assert lvl == "⚠"


def test_分层逻辑只存在于_size_decile_core():
    """🔴 两份实现 = 结论不可比（实测 rev5 因此差 10 倍）。

    ⚠️ **必须是行为断言，不能是文本匹配**（review 2026-10-06 实测）：
       初版检查 `"quantile([1 / 3, 2 / 3])" not in源码` +
       正则 `mcap_f.loc[te[-1]]`（转义后）。
       实测往 `run_size_decile.py` 注入一份**完整独立**的分层实现
       （用 `te[-1]`、变量名全改、rank 切层）⇒ **12 passed**。
       ⇒ 文本匹配守的是**字符串**不是**行为**，换个变量名就绕过，
       正是「橡皮章门禁」（ENGINEERING 第五节）。

    正解：断言两个脚本的模块命名空间里**没有本地定义**分层函数。
    """
    import run_financial_walk_forward
    import run_size_decile

    for mod in (run_size_decile, run_financial_walk_forward):
        # ⚠️ 判据是「**本地定义**」，不是「有没有这个名字」——
        #   合法 `from size_decile_core import size_decile_run`
        #   也会让名字出现在 vars(mod) 里。
        #   ⚠️ 也**不能**用 `__module__ != mod.__name__`：
        #   脚本被 `python xxx.py` 直接运行时 `__name__ == "__main__"`，
        #   而导入进来的函数 `__module__ == "size_decile_core"` ——
        #   两者不等，判据会把合法导入误判为本地定义（实测已踩）。
        #   正解：比 `inspect.getsourcefile`（真实文件路径）。
        import inspect
        本文件 = Path(inspect.getsourcefile(mod)).resolve()
        本地 = [n for n in ("split_buckets", "size_decile_run",
                            "run_in_subset", "interpret_layer_median")
                if n in vars(mod)
                and Path(inspect.getsourcefile(vars(mod)[n])).resolve()
                == 本文件]
        assert 本地 == [], (
            f"{mod.__name__} 自行定义了分层逻辑 {本地}，"
            f"必须改从 size_decile_core 导入（两份实现结论不可比）")


class TestTiedMarketCaps:
    """🔴 并列市值下三层仍须互斥且均衡。

    review 实测：旧的 `>= quantile` 实现在 d=2 时中盘层为空、
    d=1 时三层完全重合（1000 只同时属大/中/小）⇒ 混淆检验彻底失效。
    而**初版测试数据全部互异**（`np.arange`）⇒ 对该 bug 零敏感度
    （变异测试：改回 quantile 后仍 12 passed）。
    """

    @pytest.mark.parametrize("d", [1, 2, 10, 50])
    def test_并列越多三层仍互斥均衡(self, d):
        n = MIN_STOCKS_PER_BUCKET * 3
        # ⚠️ 不能用 `np.repeat(arange(d), n // d)` ——
        #   d 不整除 n 时长度对不上（实测 d=50,n=360 → 350 vs 360）。
        #   正解：按名次取整分箱，值一定落在 0..d-1 且长度精确为 n。
        vals = (np.arange(n) * d // n).astype(float)
        assert len(vals) == n and vals.max() == d - 1
        m = pd.Series(vals, index=[f"c{i}" for i in range(n)])
        b = split_buckets(m)
        assert len(b["大"]) == len(b["中"]) == len(b["小"]), (
            f"d={d} 时层大小失衡："
            f"{len(b['大'])}/{len(b['中'])}/{len(b['小'])}")
        assert not (set(b["大"]) & set(b["小"])), (
            f"d={d} 时大盘层与小盘层重叠 "
            f"{len(set(b['大']) & set(b['小']))} 只")
        assert not (set(b["大"]) & set(b["中"]))
        assert not (set(b["中"]) & set(b["小"]))
        全部 = set(b["大"]) | set(b["中"]) | set(b["小"])
        assert 全部 == set(m.index), f"d={d} 时三层并集不等于全集"

    def test_全同值时三层仍不相交(self):
        """d=1 是最坏情况：旧实现下三层完全重合。"""
        n = MIN_STOCKS_PER_BUCKET * 3
        m = pd.Series(1000.0, index=[f"c{i}" for i in range(n)])
        b = split_buckets(m)
        assert len(b["大"]) == len(b["中"]) == len(b["小"]) == n // 3
        assert not (set(b["大"]) & set(b["小"]))


class TestBenchWindowAlignment:
    """🔴 BLOCK-2 还原：全市场基准必须**窗口内重算** `pct_change`。

    `run_walk_forward.py:58-69` 用 60 行注释记录此坑：
      · 全样本 `pct_change()` 的首行是 `te[0]` 当天的收益，
        其分母是 `te[0]` 的**前一日收盘价 —— 落在训练期里**
      · 策略侧 `build_long_only(price.loc[te])` 的 `pct_change()` 首行恒 NaN
        ⇒ **基准比策略多算1 天，且多的这 1 天来自训练期**
      · 实测 rev5 全A 股 7 窗口，基准年化中位偏高 **2.15pp**

    ⚠️ 变异测试确认过这条判据有效（改回全样本切片 ⇒ 本测试 FAIL）。
    """

    def test_全市场基准不含训练期那一天(self):
        import size_decile_core
        from size_decile_core import size_decile_run

        from factor_lab.analysis.costs import CostModel

        # 门槛卡的是「每层 ≥MIN_STOCKS_PER_BUCKET」⇒ 全池需 3×，见 split_buckets
        n = MIN_STOCKS_PER_BUCKET * 3 + 20
        codes = [f"sh6000{i:02d}" for i in range(n)]
        dates = pd.bdate_range("2025-01-01", periods=250)
        px = pd.DataFrame(100.0, index=dates, columns=codes)
        # 🔴 让 `te[0]` 前一天（即训练期最后一天）发生暴涨 ——
        #   若基准在**全样本**上算pct_change，这一天会被算进窗口基准。
        px.iloc[199] = 200.0
        mcap = pd.DataFrame(1000.0, index=dates, columns=codes)

        te = list(dates[200:])
        窗口 = [([d for d in dates[:200]], te)]
        orig = size_decile_core.run_in_subset

        def spy(z_sub, px_sub, *a, **kw):
            return 0.0, ""

        size_decile_core.run_in_subset = spy
        try:
            rows = size_decile_run(pd.DataFrame(1.0, index=dates,
                                                columns=codes),
                                   px, mcap, 窗口, CostModel(),
                                   30, 3, "f")
        finally:
            size_decile_core.run_in_subset = orig
        基准列 = [r["全市场基准"] for r in rows]
        assert 基准列 and all(pd.notna(b) for b in 基准列)

        # 窗口内重算的基准：首行 NaN（无前一日），共 49 个有效日
        窗口内 = px.loc[te].pct_change(fill_method=None).mean(axis=1)
        assert 窗口内.notna().sum() == len(te) - 1, "前提不成立"

        from run_long_only import _bench_stats
        期望 = _bench_stats(窗口内)
        for b in 基准列:
            assert abs(b - 期望) < 1e-12, (
                f"全市场基准 {b:.6%} ≠ 窗口内重算 {期望:.6%} ⇒ "
                f"基准把训练期那一天算进了窗口")


class TestWindowSlicingGuard:
    """🔴 BLOCK 守护：`size_decile_run` 传给 `run_in_subset` 的必须是**窗口切片**。

    背景（2026-10-06 review 实测）：
    历史事故 —— 价量侧传 `z.loc[te]`（仅测试段）、财务侧传 `z[cols]`（全期 2016-2026），
    同一因子年化差 **10 倍**（rev5 +19.80% vs +2.19%），而基准中位相同（21.14%）
    证明窗口没错，只能是策略区间错了。

    为什么已有测试拦不住：
    `test_size_decile_run_使用窗口首日` 只 spy **列集合**
    （`set(px_sub.columns)`），与日期区间无关 ⇒
    把 `z.loc[te, cols]` 变异成 `z[cols].loc[dates]`（即重新引入该 bug）
    后，**17 个测试全部通过**。这是结构性盲区，必须补日期维断言。
    """

    def test_传入的是窗口切片而非全期面板(self):
        import size_decile_core
        from size_decile_core import size_decile_run

        from factor_lab.analysis.costs import CostModel

        dates = pd.bdate_range("2025-01-01", periods=250)
        # 门槛卡的是「每层 ≥MIN_STOCKS_PER_BUCKET」⇒ 全池需 3×，见 split_buckets
        n = MIN_STOCKS_PER_BUCKET * 3 + 20
        codes = [f"sh6000{i:02d}" for i in range(n)]
        rs = np.random.default_rng(7)
        z = pd.DataFrame(rs.normal(size=(len(dates), n)),
                         index=dates, columns=codes)
        px = pd.DataFrame(
            100 * np.exp(np.cumsum(rs.normal(0, 0.01, (len(dates), n)), axis=0)),
            index=dates, columns=codes)
        mcap = pd.DataFrame(
            np.tile(np.linspace(100.0, 2000.0, n), (len(dates), 1)),
            index=dates, columns=codes)

        te = list(dates[200:])          # 50 个交易日的测试窗口
        windows = [([d for d in dates[:200]], te)]
        seen_idx: list[pd.Index] = []
        orig = size_decile_core.run_in_subset

        def spy(z_sub, px_sub, cost, n_hold, n_pick_mult, factor):
            seen_idx.append(z_sub.index)
            return orig(z_sub, px_sub, cost, n_hold, n_pick_mult, factor)

        size_decile_core.run_in_subset = spy
        try:
            size_decile_run(z, px, mcap, windows, CostModel(), 30, 3, "f")
        finally:
            size_decile_core.run_in_subset = orig

        assert seen_idx, "spy 没被调用，测试本身失效"
        for idx in seen_idx:
            assert len(idx) == len(te), (
                f"传入 {len(idx)} 行，期望窗口切片 {len(te)} 行 "
                f"⇒ 传的是全期面板，历史事故就是差 10 倍的来源")
            assert idx[0] == te[0] and idx[-1] == te[-1], (
                f"日期区间 [{idx[0].date()} ~ {idx[-1].date()}] "
                f"≠ 窗口 [{te[0].date()} ~ {te[-1].date()}]")
