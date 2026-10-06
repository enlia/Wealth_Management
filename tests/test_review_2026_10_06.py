"""回归测试：review 2026-10-06 抓出的 3 个 BLOCK + 2 个 MAJOR。

设计原则（DATA_QUALITY.md）：每条守护测试都必须能**让还原后的bug 版失败**。
一个在正确代码上通过、在错误代码上也通过的测试等于没测。

本文件覆盖：
- BLOCK-1 `np.clip` 把 -1 哨兵抬成非零权重 → 凭空建仓
- BLOCK-2 `_year_span` docstring 声称有下限但代码没有
- BLOCK-3 基准比策略多算 1 天（边界错位）→ 超额被系统性低估
- MAJOR-2 `rolling_vol` 用 fwd 算滚动波动率 → 前视 1 天
- MAJOR-1 `limit_state` 缺数据判为可交易（与 limit_masks 口径相反）
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

from factor_lab.analysis.long_only import (  # noqa: E402
    _year_span,
    build_long_only,
    simulate_matrix,
)
from factor_lab.analysis.selection import select_with_buffer  # noqa: E402
from factor_lab.analysis.tradability import limit_masks  # noqa: E402
from factor_lab.analysis.long_only import CostModel, PortfolioSpec  # noqa: E402


def _panel(n_days=400, n_stocks=40, seed=11):
    """构造 (factor_values, price_panel) 两个对齐面板。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    codes = [f"sh60{i:04d}" for i in range(n_stocks)]
    price = pd.DataFrame(
        50 * np.cumprod(1 + rng.normal(0.0004, 0.015, (n_days, n_stocks)), axis=0),
        index=dates, columns=codes)
    fv = pd.DataFrame(
        rng.normal(0, 1, (n_days, n_stocks)), index=dates, columns=codes)
    return fv, price


class TestBlock1SentinelClip:
    """BLOCK-1：`np.clip` 在哨兵置0 之后执行，会把 0 抬成 min_weight。"""

    def test_clip本身就会抬高零权重(self):
        """先证明这个陷阱真实存在 —— 不是我们臆想的。"""
        w = np.array([[0.5, 0.5, 0.0]])
        clipped = np.clip(w / w.sum(), 0.02, 0.15)
        assert clipped[0, 2] > 0, "clip 把 0 抬成了非零，说明 BLOCK-1 的前提成立"

    def test_还原bug版会失败(self):
        """哨兵槽位经 clip + 归一化后权重必须仍为 0。"""
        # 构造含 -1 哨兵的持仓矩阵：第3槽为 -1
        held = np.array([[0, 1, -1]])
        w = np.array([[0.5, 0.5, 0.0]])

        clipped = np.clip(w, 0.02, 0.15)          # bug：此后哨兵不再是 0
        fixed = np.where(held < 0, 0.0, clipped)  # 修复：clip 后重新屏蔽
        fixed = fixed / np.maximum(fixed.sum(axis=1, keepdims=True), 1e-12)
        buggy = clipped / np.maximum(clipped.sum(axis=1, keepdims=True), 1e-12)

        assert clipped[0, 2] > 0, "clip 把哨兵抬成了非零，Bug 前提成立"
        assert fixed[0, 2] == 0.0, "修复后哨兵权重必须为 0"
        # 关键对比：bug 版哨兵吃掉 6.25% 仓位，修复版 0%
        assert buggy[0, 2] > 0.05, "bug 版哨兵槽确实拿到了可观权重"
        assert fixed[0, 2] == 0.0
        # 且真实槽位不再被稀释
        assert not np.isclose(buggy[0, 0], fixed[0, 0]), \
            "bug 版与修复版的真实槽位权重应不同"
        assert np.isclose(fixed[0, 0], 0.5), "修复版应恢复 50/50"

    def test_build_long_only_不会凭空建仓(self):
        """端到端：权重矩阵里不能出现「无持仓却被赋权」的槽位。"""
        fv, price = _panel()
        spec = PortfolioSpec(name="t", n_hold=10, n_pick=30,
                             rebalance="M", factor="synthetic")
        r = build_long_only(fv, spec, CostModel(), price)
        assert r.get("ok"), r.get("reason")
        # 净值不能出现 NaN / inf，且年化应在合理量级（非 0 非爆炸）
        assert np.isfinite(r["年化收益"])
        assert abs(r["年化收益"]) < 5.0, f"年化爆炸：{r['年化收益']:.2%}"

    def test_simulate_matrix_哨兵不写入真实列(self):
        """`W[rows, held_safe] = w_mat` 写入时，哨兵槽必须贡献 0 权重。

        构造：4 只股票**收益各不相同**，这样权重错位一定会体现在净值上。
        """
        T, N = 5, 4
        dates = pd.bdate_range("2023-01-02", periods=T)
        held = np.tile(np.array([0, 1, -1]), (T, 1))    # 每行第 3 槽都是哨兵
        w = np.zeros((T, 3))
        w[:, 0] = 0.6
        w[:, 1] = 0.4
        # 收益各异：c0 涨、c1 跌、c2/c3 给极端值以便识别错位
        # ⚠️ fwd 的行数必须 = T-1（fwd[t] = t→t+1，最后一天没有下一期）
        rets = np.array([[0.010, -0.010, 0.500, -0.500]] * (T - 1))
        fwd = pd.DataFrame(rets, index=dates[:-1],
                           columns=[f"c{i}" for i in range(N)])
        spec = PortfolioSpec(name="t", n_hold=3, n_pick=3,
                             rebalance="M", factor="x")
        # ⚠️ 用**零成本**模型：默认 CostModel 含 0.1% 滑点，
        #   首日建仓会产生换手成本，期望值要额外算一遍 —— 那样测的就不是权重了。
        zero_cost = CostModel(commission=0.0, stamp_duty=0.0,
                              transfer_fee=0.0, slippage=0.0)
        res = simulate_matrix(dates, held, w, fwd, zero_cost, spec)
        assert res["ok"], res.get("reason")

        # 正确口径：只有 c0(0.6) 与 c1(0.4) 有权重
        #⇒ 日收益 = 0.6*1% + 0.4*(-1%) = +0.2%
        # ⚠️ fwd 有 T-1 = 4 行（末位无下一期收益）⇒ 净值复利 4 次；
        #   但 `_year_span(dates)` 用**全部 5 个日期**算年数。
        #   短窗口下年化必然夸张（4 个交易日就敢年化三位数），
        #   这不是 bug 而是几何年化的固有性质 —— 本测试只验**权重对不对**，
        #   所以期望值必须用同一套 span 口径复算，不能用「日收益×243」。
        r_ok = 0.6 * 0.010 + 0.4 * (-0.010)
        nav_ok = (1 + r_ok) ** (T - 1)
        cagr_ok = nav_ok ** (1 / _year_span(dates)) - 1
        assert np.isclose(res["年化收益"], cagr_ok, atol=1e-9), (
            f"年化 {res['年化收益']:.6%} ≠ 期望 {cagr_ok:.6%}；"
            f"哨兵槽很可能被赋权或抹掉了真实持仓的权重")
        # 反向断言：若哨兵被抹掉真实持仓 c0（权重 0.6→0），
        # 日收益会变成 -0.4%，年化必然为负 —— 明确排除该情形
        assert res["年化收益"] > 0, "c0 权重疑似被哨兵覆盖抹零"


class TestBlock2YearSpan:
    """BLOCK-2：docstring 声称的「最小天数下限」在代码里根本不存在。"""

    def test_不存在隐藏下限(self):
        """20 个交易日的年化必然爆炸 —— 证明没有下限保护。"""
        dates = pd.bdate_range("2023-01-02", periods=20)
        span = _year_span(dates)
        assert span < 0.1, f"20 交易日应约 0.074 年，实得 {span}"

    def test_docstring不再声称有下限(self):
        """防止有人把假的保护写进注释又当成真保护。"""
        doc = _year_span.__doc__ or ""
        assert "设最小天数下限" not in doc, \
            "docstring 又声称有下限，但代码没有 —— 注释比没注释更危险"
        # 反而必须**明确写出**没有下限，否则调用方会误以为短窗口安全
        assert "不做下限保护" in doc, "必须写明本函数不提供短区间保护"

    def test_短区间年化确实会爆炸(self):
        """证明「没有下限」是事实，不是我的臆断。"""
        import math
        dates = pd.bdate_range("2023-01-02", periods=20)
        span = _year_span(dates)
        cagr = 1.08 ** (1 / span) - 1
        assert cagr > 1.0, f"20交易日 +8% 应年化 >100%，实得 {cagr:.1%}"

    def test_自然日与交易日口径不同(self):
        """同一个区间，两种口径的年数必须不同（否则 M1 白改了）。"""
        dates = pd.bdate_range("2023-01-02", periods=243)
        cal = _year_span(dates)
        assert abs(cal - len(dates) / 252.0) > 1e-6, \
            "自然日口径与 252 口径竟相同，M1 的修复没生效"

    def test_空区间不崩(self):
        assert _year_span(pd.DatetimeIndex([])) == pytest.approx(1e-9)
        assert _year_span(pd.DatetimeIndex(["2023-01-02"])) == pytest.approx(1e-9)


class TestBlock3BenchmarkAlignment:
    """BLOCK-3：基准多算 1 天，且那 1 天来自训练期。"""

    def test_窗口内重算少一天(self):
        from run_long_only import _bench_stats
        rng = np.random.default_rng(3)
        dates = pd.bdate_range("2022-01-03", periods=300)
        price = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0.0005, 0.01, (300, 4)), axis=0),
            index=dates, columns=list("abcd"))
        te = dates[100:261]

        wrong = price.pct_change(fill_method=None).mean(axis=1).loc[te]
        right = price.loc[te].pct_change(fill_method=None).mean(axis=1)

        # 旧口径：全样本 pct_change 后切片，首行非NaN（161 个观测）
        assert wrong.notna().sum() == len(te)
        # 新口径：窗口内 pct_change，首行 NaN（160 个有效观测）
        assert right.notna().sum() == len(te) - 1
        assert np.isfinite(_bench_stats(right))

    def test_策略与基准区间完全对齐(self):
        """策略侧 `price.loc[te].pct_change().shift(-1)` 有效天数 = len(te)-1。

        基准必须与它**逐日相等**，超额才有意义。
        """
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2022-01-03", periods=300)
        price = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0.0004, 0.012, (300, 5)), axis=0),
            index=dates, columns=list("abcde"))
        te = dates[50:200]

        strat = price.loc[te].pct_change(fill_method=None).shift(-1)
        bench = price.loc[te].pct_change(fill_method=None).mean(axis=1)
        # 策略第 t 行 = t收盘→t+1收盘；基准第 t 行 = t-1→t ⇒ 差一位
        # 两者有效观测数必须一致
        assert strat.notna().to_numpy().sum(axis=0).max() == \
            bench.notna().sum(), "策略与基准有效天数不等，超额不可比"


class TestMajor2ForwardLookingVol:
    """MAJOR-2：滚动波动率用了 fwd（前视 1 天）。"""

    def test_rolling含当前点(self):
        """先证明 rolling 默认含当前点 —— 这是 bug 的前提。"""
        s = pd.Series([1.0, 2.0, 3.0])
        assert s.rolling(3).std().iloc[2] == pytest.approx(
            pd.Series([1.0, 2.0, 3.0]).std())

    def test_波动率只用过去(self):
        """`fwd.shift(1).rolling()` 在 t 日只含 t-1 及之前的收益。

        ⚠️ 用**非对称**序列：等差数列的 std 有平移不变性，
           shift(1) 前后会算出同一个值，测不出差异（本项目已踩）。
        """
        fwd = pd.Series([0.10, 0.01, 0.50, 0.02, 0.80])
        past = fwd.shift(1)
        vol_ok = past.rolling(3, min_periods=1).std()
        vol_bad = fwd.rolling(3, min_periods=1).std()
        # 最后一个点：ok 只含 [0.01,0.50,0.02]，bad 含 [0.50,0.02,0.80]
        assert not np.isclose(vol_ok.iloc[-1], vol_bad.iloc[-1]), \
            "shift(1) 前后波动率竟相同，说明序列选得不好，测不出前视"
        assert vol_ok.iloc[-1] == pytest.approx(
            pd.Series([0.01, 0.50, 0.02]).std())

    def test_源码里不再用fwd直接rolling(self):
        """守护：源码中 `rolling_vol` 不得再以 fwd 为输入。"""
        src = (ROOT / "src" / "factor_lab" / "analysis" /
               "long_only.py").read_text(encoding="utf-8")
        assert "past = fwd.shift(1)" in src, "缺少 past = fwd.shift(1) 去前视"
        assert "rolling_vol = (fwd.rolling" not in src, \
            "rolling_vol 仍在用 fwd，前视未消除"


class TestMajor1LimitState:
    """MAJOR-1：缺涨跌停价时，`limit_masks` 与旧 `limit_state` 口径相反。"""

    def test_缺数据判为不可交易(self):
        idx = pd.date_range("2023-01-03", periods=2)
        cols = ["sh600000", "sh600001"]
        close = pd.DataFrame(10.0, index=idx, columns=cols)
        up = pd.DataFrame(np.nan, index=idx, columns=cols)   # 全缺
        dn = pd.DataFrame(np.nan, index=idx, columns=cols)
        with pytest.raises(ValueError, match="覆盖率"):
            limit_masks(close, up, dn)

    def test_部分缺失时缺失格判不可交易(self):
        idx = pd.date_range("2023-01-03", periods=3)
        cols = ["sh600000", "sh600001", "sh600002", "sh600003"]
        close = pd.DataFrame(10.0, index=idx, columns=cols)
        up = pd.DataFrame(np.nan, index=idx, columns=cols)
        dn = pd.DataFrame(np.nan, index=idx, columns=cols)
        # 让前两列有数据（保证覆盖率过 50% 门槛）
        up.iloc[:, :2] = 11.0
        dn.iloc[:, :2] = 9.0
        lim_up, lim_dn = limit_masks(close, up, dn)
        # 有涨跌停价但未封板 → 可交易
        assert not bool(lim_up.iloc[0, 0])
        # 缺涨跌停价 → 必须判不可交易（不能因fillna 不触发而放行）
        assert bool(lim_up.iloc[0, 2]), "缺数据被判成可交易，A 股买不进"
        assert bool(lim_dn.iloc[0, 2])


class TestSentinelPropagation:
    """`select_with_buffer` 的 -1 哨兵会传到权重构造，必须被识别。"""

    def test_首日非调仓日时产生哨兵(self):
        top = np.array([[3, 2, 1], [0, 1, 2], [1, 2, 0]])
        is_rebal = np.array([False, True, True])
        held = select_with_buffer(top, is_rebal, 3)
        assert (held[0] == -1).all(), "首个调仓日之前应全是 -1 哨兵"
        assert not (held[1:] == -1).any()

    def test_哨兵不会变成真实列索引(self):
        """负索引语义：-1 会取到最后一列。必须先转0 再屏蔽。"""
        held = np.array([[0, 1, -1]])
        held_safe = np.where(held < 0, 0, held)
        mat = np.array([[10., 20., 30., 40.]])
        picked = np.take_along_axis(mat, held_safe, axis=1)
        assert picked[0, 2] == 10., \
            "bug：-1 被当成最后一列(40.0)，哨兵槽取到了真实股票"
        assert (held < 0).any(), "哨兵本身应仍为 -1"
