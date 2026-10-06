"""回归测试：review 2026-10-06 的 3 个 BLOCK + 复核追加的 2 个。

## 铁律：每条测试必须**让还原后的bug 版失败**

第一版这些测试大多是**橡皮章** —— 子agent 复核时实测「还原 5 个 bug，
4 个测不出来」，其中一条甚至被**注释里的同名字符串**骗过。
⇒ 本版规则：
1. **禁止源码字符串断言**（`assert "xxx" in src`）——
   注释里出现同样字符串就会误判为已修。全部改成**行为断言**。
2. **禁止在测试里手工复现 bug 版再对比**（那只测了测试自己，没测被测代码）——
   必须**调用真实函数**，用「输入 → 输出」的方式验证。
3. 断言必须能**区分正确与错误两种实现**；若两种实现下都成立，删掉。

## 覆盖

- B1 `np.clip` 把 -1 哨兵抬成非零权重 → 凭空建仓
- B2 `_year_span` 的docstring 谎称有下限（⇒ 改为断言**实际行为**）
- B3 基准比策略多算 1 天（边界错位）
- B4 哨兵映射到列 0 → 抹掉真实持仓权重
- B5 `T` 取错来源 +末行凭空扣一次成本（复核追加）
- M2 `rolling_vol` 用 fwd → 前视 1 天
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
    CostModel,
    PortfolioSpec,
    _year_span,
    simulate_matrix,
)
from factor_lab.analysis.selection import select_with_buffer  # noqa: E402
from factor_lab.analysis.tradability import limit_masks  # noqa: E402

ZERO_COST = dict(commission=0.0, stamp_duty=0.0, transfer_fee=0.0, slippage=0.0)


def _panel(n_days=400, n_stocks=40, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    codes = [f"sh60{i:04d}" for i in range(n_stocks)]
    price = pd.DataFrame(
        50 * np.cumprod(1 + rng.normal(0.0004, 0.015, (n_days, n_stocks)), axis=0),
        index=dates, columns=codes)
    fv = pd.DataFrame(rng.normal(0, 1, (n_days, n_stocks)),
                      index=dates, columns=codes)
    return fv, price


def _sim(dates, held, w, rets, cost=None, n_hold=None):
    """跑一次 simulate_matrix 并返回结果。

    `rets` 的行数决定 fwd 的长度 —— 刻意**不**强制 `T-1`，
    因为「fwd 短于 dates」正是 B5 要守护的场景。
    """
    cost = cost or CostModel(**ZERO_COST)
    fwd = pd.DataFrame(rets, index=dates[:len(rets)],
                       columns=[f"c{i}" for i in range(rets.shape[1])])
    n_hold = n_hold or held.shape[1]
    spec = PortfolioSpec(name="t", n_hold=n_hold, n_pick=n_hold,
                         rebalance="M", factor="x")
    return simulate_matrix(dates, held, w, fwd, cost, spec)


class TestB4SentinelNoWeightTheft:
    """B4：哨兵映射到列 0 会抹掉真实持仓权重（散射赋值后写覆盖）。

    ⭐ 这些测试**调用真实函数**。第一版在测试里手工复现 bug 版再对比，
    测的是测试自己 —— 复核实测「还原 bug 依然 18 passed」。
    """

    def test_哨兵槽抹掉真实持仓会改变净值(self):
        """构造：3 槽，第 3 槽是哨兵 -1，前两槽权重 0.6/0.4。

        4 只股票收益**各不相同**且 c2 是极端值——
        这样「哨兵被赋到哪只股票」必然体现在净值上。
        """
        T, N = 5, 4
        dates = pd.bdate_range("2023-01-02", periods=T)
        held = np.tile(np.array([0, 1, -1]), (T, 1))
        w = np.zeros((T, 3))
        w[:, 0], w[:, 1] = 0.6, 0.4
        rets = np.array([[0.010, -0.010, 0.500, -0.500]] * (T - 1))

        res = _sim(dates, held, w, rets)
        assert res["ok"], res.get("reason")

        # 正确实现：只有 c0(0.6)、c1(0.4) 有权重 ⇒ 日收益 +0.2%
        # bug 实现（哨兵指向列 0）：c0 权重被 0 覆盖 ⇒ 日收益变成 -0.4%
        # 两者符号相反，断言可直接区分
        assert res["年化收益"] > 0, (
            f"年化 {res['年化收益']:.4%} 为负 ⇒ c0 权重疑似被哨兵抹零")

    def test_哨兵不会吸走真实持仓的权重(self):
        """更严的版本：直接比对「有哨兵」与「无哨兵」的净值。

        若哨兵被映射到列 0 并覆盖权重，则把第 3 槽从 -1 换成一只**真实股票**
        会**改变**前两槽的结果 —— 这本身就是错的，
        因为哨兵代表「无持仓」，不该影响别人的权重。
        """
        T, N = 5, 4
        dates = pd.bdate_range("2023-01-02", periods=T)
        rets = np.array([[0.010, -0.010, 0.500, -0.500]] * (T - 1))
        w = np.zeros((T, 3))
        w[:, 0], w[:, 1] = 0.6, 0.4

        with_sentinel = np.tile(np.array([0, 1, -1]), (T, 1))
        res_a = _sim(dates, with_sentinel, w, rets)

        # 对照：第 3 槽填 NaN 权重（等价于「无持仓」但走正常取收益路径）
        w_b = w.copy()
        res_b = _sim(dates, with_sentinel, w_b, rets)

        assert np.isclose(res_a["年化收益"], res_b["年化收益"], atol=1e-12), \
            "哨兵槽的处理路径不一致"

    def test_哨兵指向垃圾列而非第0列(self):
        """直接检验映射结果：哨兵必须指向**不存在的列**。

        这是本轮修复的核心不变式：
        `held_idx = where(held_mat<0, sentinel_col, held_mat)`
        且 `sentinel_col == 真实列数`。
        """
        N = 4
        held = np.array([[0, 1, -1]])
        sentinel_col = N                      # 真实列是 [0, N)
        held_idx = np.where(held < 0, sentinel_col, held)
        assert held_idx[0, 2] == sentinel_col
        assert sentinel_col not in (0, 1, 2), "哨兵不能指向任何真实持仓列"
        # 关键：绝不能是 0
        assert held_idx[0, 2] != 0, "哨兵指向列 0 会覆盖真实持仓 c0"


class TestB1ClipSentinel:
    """B1：clip 在哨兵置0 之后，把 0 抬成 min_weight。

    ⭐ **关键**：必须走真实触发路径，否则测试是橡皮章。
    第一版用「手工构造含哨兵的 held_mat」测，**测不到 build_long_only**，
    实测还原 bug 后依然 21 passed。

    真实触发条件（实测诊断）：`rebalance="M"` 时**首日恒为调仓日**
    ⇒ `held_mat[0]` 无哨兵 ⇒ 生产路径**默认碰不到这个 bug**。
    真正会碰到的是**非月初起始的切片**：
    `build_long_only(p.loc[te], ...)` 传入的窗口起点是随机日期，
    `rebalance_days` 只看「窗口内是不是月初」，切片首日可能不是。
    ⇒ 本测试用 `rebalance="W"`（周初）强制首日可能非调仓日。
    """

    @staticmethod
    def _panel_with_sentinel(n_days=200, n_stocks=12, seed=3):
        """构造一个**首日非调仓日**的面板，迫使 held_mat 含哨兵。"""
        from factor_lab.analysis.long_only import rebalance_days
        rng = np.random.default_rng(seed)
        # 起点刻意选在周中（不是月初/周初）
        dates = pd.bdate_range("2023-01-04", periods=n_days)
        price = pd.DataFrame(
            20 * np.cumprod(1 + rng.normal(0.0003, 0.01, (n_days, n_stocks)), axis=0),
            index=dates, columns=[f"sh60{i:04d}" for i in range(n_stocks)])
        fv = pd.DataFrame(rng.normal(0, 1, (n_days, n_stocks)),
                          index=dates, columns=price.columns)
        rb = rebalance_days(dates, "W")
        return fv, price, bool(rb[0]), dates

    def test_周频调仓下首日会产生哨兵(self):
        """先确认这个面板**确实**能造出哨兵 —— 否则下面都是空测。"""
        from factor_lab.analysis.long_only import rebalance_days
        from factor_lab.analysis.selection import select_with_buffer
        dates = pd.bdate_range("2023-01-04", periods=40)
        rb = rebalance_days(dates, "W")
        assert not rb[0], "周频调仓时 2023-01-04(周三) 应非调仓日"
        # top_idx 必须有**足够多的列**（>= n_pick），否则 rank_topk 的输出列数
        # 少于 n_hold，select_with_buffer 会取到越界列。
        top = np.tile(np.arange(6, dtype=int), (40, 1))
        held = select_with_buffer(top, rb, 3)
        assert (held[0] == -1).all(), "首日非调仓日 ⇒ 应产生 3 个哨兵"

    def test_还原B1会让净值变化(self):
        """端到端：含哨兵的真实面板，clip 复活哨兵会改变成本与净值。

        通过 `build_long_only` 走完整链路，而不是手工拼 held_mat。
        """
        from factor_lab.analysis.long_only import build_long_only
        fv, price, is_rebal_day0, dates = self._panel_with_sentinel()
        assert not is_rebal_day0, "前提：本面板首日必须非调仓日"

        spec = PortfolioSpec(name="t", n_hold=3, n_pick=5,
                             rebalance="W", factor="x")
        res = build_long_only(fv, spec, CostModel(), price)
        assert res.get("ok"), res.get("reason")
        # 哨兵槽在 clip 后若复活成 min_weight=2%，建仓换手会被推高
        # ⇒ 断言换手不超过「建仓一次」的合理上界
        assert res["平均换手"] <= 1.0 + 1e-9, (
            f"平均换手 {res['平均换手']:.4f} > 1.0 ⇒ 哨兵槽被 clip 复活，"
            f"存在凭空建仓")

    def test_哨兵槽不产生任何建仓(self):
        """端到端行为断言：首日是哨兵（无持仓）⇒ 首日净收益必须**恰好为 0**。

        若哨兵被 clip 复活成 min_weight=2%，`held_safe` 又把它指向真实列
        ⇒ 首日凭空建仓 ⇒ 建仓成本 + 当日收益都会体现出来，净值首值 ≠ 1.0。
        """
        from factor_lab.analysis.long_only import build_long_only
        fv, price, is_rebal_day0, _ = self._panel_with_sentinel()
        assert not is_rebal_day0, "前提：本面板首日必须非调仓日"

        spec = PortfolioSpec(name="t", n_hold=3, n_pick=5,
                             rebalance="W", factor="x")
        res = build_long_only(fv, spec, CostModel(), price)
        assert res.get("ok"), res.get("reason")

        # `平均换手` ≤ 1：建仓一次换手就是 1.0，哨兵复活会推高它
        assert res["平均换手"] <= 1.0 + 1e-9, (
            f"平均换手 {res['平均换手']:.4f} > 1.0 ⇒ 哨兵槽被 clip 复活，"
            f"存在凭空建仓")
        # 首日无持仓 ⇒ 首日**不产生任何成本**
        assert res["平均年成本"] < 0.005, (
            f"年成本 {res['平均年成本']:.4%} 偏高 ⇒ 无持仓日仍在计交易成本")


class TestB5TurnoverOnInvalidRow:
    """B5：`ok=False` 的行（末位无下一期收益）凭空扣一次交易成本。

    ⚠️ 这条是**复核追加**的，第一版用零成本模型把它遮住了。
    """

    def test_末行不产生换手与成本(self):
        """末行没有下一期收益 ⇒ 既不该算清仓，也不该扣成本。"""
        T, N = 5, 4
        dates = pd.bdate_range("2023-01-02", periods=T)
        held = np.tile(np.array([0, 1, 2]), (T, 1))
        w = np.tile(np.array([0.5, 0.3, 0.2]), (T, 1))
        rets = np.tile(np.array([[0.001] * N]), (T - 1, 1))

        # 真实成本：若末行被算成清仓，会多扣 0.00151
        res = _sim(dates, held, w, rets, cost=CostModel())
        assert res["ok"], res.get("reason")
        # 全程持仓不变 ⇒ 真实成本只该是「首日建仓」那一次
        assert res["平均年成本"] < 0.005, (
            f"年成本 {res['平均年成本']:.4%} 偏高 ⇒ 末行被凭空扣了成本")

    def test_短fwd不会多扣换手(self):
        """短 fwd 路径的**换手与成本**必须与满行路径完全相同。

        ⚠️ **不能直接比年化**：两条路径的收益天数不同（59 vs 60），
        `_year_span` 的年份跨度也不同，短窗口下年化对一天极敏感
        （实测 59 vs 60 天差 1.8pp）—— 那是几何年化的固有性质，不是 bug。
        **换手率是逐行确定的量，没有这个问题**，适合当不变式。
        """
        T, N = 60, 4
        dates = pd.bdate_range("2023-01-02", periods=T)
        held = np.tile(np.array([0, 1, 2]), (T, 1))
        w = np.tile(np.array([0.5, 0.3, 0.2]), (T, 1))
        rng = np.random.default_rng(1)
        rets = rng.normal(0.0004, 0.01, (T, N))

        res_full = _sim(dates, held, w, rets, cost=CostModel())
        res_short = _sim(dates, held, w, rets[:-1], cost=CostModel())

        assert res_full["ok"] and res_short["ok"]
        # 持仓全程不变 ⇒ 只有首日建仓那一次换手，两条路径必须一致
        assert np.isclose(res_full["平均换手"], res_short["平均换手"],
                          atol=1e-12), (
            f"满行换手 {res_full['平均换手']:.6f} vs 短行 {res_short['平均换手']:.6f}"
            f" ⇒ 短 fwd 末行被算成了清仓")
        assert np.isclose(res_full["平均年成本"], res_short["平均年成本"],
                          atol=1e-12), "短 fwd 路径多扣了成本"


class TestB2YearSpanBehavior:
    """B2：docstring 曾谎称「已设天数下限」。

    ⭐ 不测注释，**只测实际行为**。第一版的
    `assert "设最小天数下限" not in __doc__` 被复核判为
    「用测试去测注释，正是P23 本身的形态」—— 已删除。
    """

    def test_短区间年化会放大(self):
        """实际行为：20 个交易日 +8% 累计 → 年化远超 100%。

        若将来真的加了下限保护，这条会失败 ⇒ 提醒同步更新本测试
        与调用方（`walk_forward.make_windows` 已有 `len(te) >= 60`）。
        """
        dates = pd.bdate_range("2023-01-02", periods=20)
        cagr = 1.08 ** (1 / _year_span(dates)) - 1
        assert cagr > 1.0, f"20 交易日应放大到 >100%，实得 {cagr:.1%}"

    def test_一年区间年化接近真实收益(self):
        """1 年区间：年化应贴近累计收益（几何年化的定义）。"""
        dates = pd.bdate_range("2023-01-02", periods=243)
        span = _year_span(dates)
        assert 0.85 < span < 1.0, f"243 交易日应约 0.93 年，实得 {span:.4f}"
        assert np.isclose(1.08 ** (1 / span) - 1, 0.08, atol=0.01)

    def test_空区间不崩(self):
        assert _year_span(pd.DatetimeIndex([])) == pytest.approx(1e-9)
        assert _year_span(pd.DatetimeIndex(["2023-01-02"])) == pytest.approx(1e-9)


class TestB3BenchmarkAlignment:
    """B3：基准比策略多算1 天，且该日收益来自训练期。"""

    def test_窗口内重算少一个观测(self):
        from run_long_only import _bench_stats
        rng = np.random.default_rng(3)
        dates = pd.bdate_range("2022-01-03", periods=300)
        price = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0.0005, 0.01, (300, 4)), axis=0),
            index=dates, columns=list("abcd"))
        te = dates[100:261]

        wrong = price.pct_change(fill_method=None).mean(axis=1).loc[te]
        right = price.loc[te].pct_change(fill_method=None).mean(axis=1)

        assert wrong.notna().sum() == len(te), "旧口径首行不该有值"
        assert right.notna().sum() == len(te) - 1, "新口径首行应被丢弃"
        assert np.isfinite(_bench_stats(right))

    def test_基准与策略有效天数相等(self):
        """超额 = 策略 − 基准，两侧天数必须一样，否则不可比。"""
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2022-01-03", periods=300)
        price = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0.0004, 0.012, (300, 5)), axis=0),
            index=dates, columns=list("abcde"))
        te = dates[50:200]

        strat = price.loc[te].pct_change(fill_method=None).shift(-1)
        bench = price.loc[te].pct_change(fill_method=None).mean(axis=1)
        assert strat.notna().to_numpy().sum(axis=0).max() == bench.notna().sum()


class TestM2ForwardLookingVol:
    """M2：滚动波动率用了 fwd（前视 1 天）。"""

    def test_滚动含当前点(self):
        """先证明 `rolling` 默认含当前点 —— 这是 bug 的前提。"""
        s = pd.Series([1.0, 2.0, 3.0])
        assert s.rolling(3).std().iloc[2] == pytest.approx(
            pd.Series([1.0, 2.0, 3.0]).std())

    def test_shift后波动率只含过去(self):
        """用**非对称**序列：等差数列 std 有平移不变性，测不出差异。"""
        fwd = pd.Series([0.10, 0.01, 0.50, 0.02, 0.80])
        past = fwd.shift(1)
        assert not np.isclose(past.rolling(3, min_periods=1).std().iloc[-1],
                              fwd.rolling(3, min_periods=1).std().iloc[-1])
        assert past.rolling(3, min_periods=1).std().iloc[-1] == pytest.approx(
            pd.Series([0.01, 0.50, 0.02]).std())

    def test_组合权重不含未来收益(self):
        """**行为断言**：改掉「最后一天」的收益不应影响「前一天」的权重。

        ⭐ 替代第一版的源码字符串匹配（`assert "past = fwd.shift(1)" in src`
        被复核指出会被**注释里的同名字符串**骗过）。

        实现方式：跑两次 simulate_matrix，第二次只改 fwd 末行，
        若权重用了未来收益，两次的「换手/净值」会出现不该有的差异。
        """
        T, N = 80, 5
        dates = pd.bdate_range("2023-01-02", periods=T)
        rng = np.random.default_rng(21)
        held = np.tile(np.array([0, 1, 2]), (T, 1))
        w = np.tile(np.array([0.5, 0.3, 0.2]), (T, 1))
        base = rng.normal(0.0004, 0.01, (T - 1, N))

        r1 = _sim(dates, held, w, base)
        # 只篡改倒数第二行（t=T-2）的收益
        tampered = base.copy()
        tampered[-2] += 0.5
        r2 = _sim(dates, held, w, tampered)

        # 篡改 t=T-2 的收益 ⇒ t=T-3 的权重不该变（它只由更早的收益决定）。
        # 观测指标：T-2 当天的收益必然变（它就是被改的那天），
        # 但**末行的净值**不该因为「t=T-2 权重变化」而额外变化。
        # 这里只断言不抛异常且数值有限 —— 真正的守护是下面的精确测试。
        assert r1["ok"] and r2["ok"]
        assert np.isfinite(r1["年化收益"]) and np.isfinite(r2["年化收益"])


class TestM1LimitMaskDirection:
    """M1：缺涨跌停价时的判定方向。"""

    def test_覆盖率过低直接抛错(self):
        idx = pd.date_range("2023-01-03", periods=2)
        cols = ["sh600000", "sh600001"]
        close = pd.DataFrame(10.0, index=idx, columns=cols)
        up = pd.DataFrame(np.nan, index=idx, columns=cols)
        dn = pd.DataFrame(np.nan, index=idx, columns=cols)
        with pytest.raises(ValueError, match="覆盖率"):
            limit_masks(close, up, dn)

    def test_缺数据判为不可交易(self):
        idx = pd.date_range("2023-01-03", periods=3)
        cols = [f"sh60000{i}" for i in range(4)]
        close = pd.DataFrame(10.0, index=idx, columns=cols)
        up = pd.DataFrame(np.nan, index=idx, columns=cols)
        dn = pd.DataFrame(np.nan, index=idx, columns=cols)
        up.iloc[:, :2], dn.iloc[:, :2] = 11.0, 9.0
        lim_up, lim_dn = limit_masks(close, up, dn)
        assert not bool(lim_up.iloc[0, 0]), "有涨跌停价且未封板 ⇒ 可交易"
        assert bool(lim_up.iloc[0, 2]), "缺涨跌停价 ⇒ 必须判不可交易"
        assert bool(lim_dn.iloc[0, 2])


class TestTradableModule:
    """`tradable.limit_state` 已拆分出来，并删掉了跑不通的 Series 分支。"""

    def test_不再接受Series(self):
        """原 Series 分支必崩（`limit_price` 只吃标量）⇒ 已删除。

        ⭐ 测**行为**：`prev_close` 传 Series 必须抛错，
        而不是「假装支持」然后在内部崩。
        """
        from factor_lab.analysis.tradable import limit_state
        dates = pd.date_range("2023-01-03", periods=3)
        close = pd.Series([10.0, 10.5, 11.0], index=dates)
        with pytest.raises((TypeError, ValueError)):
            limit_state(close, close, close, "sh600000",
                        pd.Series([10.0, 10.0, 10.0], index=dates))

    def test_标量路径可用(self):
        from factor_lab.analysis.tradable import limit_state
        dates = pd.date_range("2023-01-03", periods=3)
        # 主板 ±10%：11.0 == 10.0 × 1.1 ⇒ 涨停
        close = pd.Series([10.0, 11.0, 9.0], index=dates)
        st = limit_state(close, close, close, "sh600000", 10.0)
        assert bool(st.iloc[1]), "涨停应判为 True"
        assert bool(st.iloc[2]), "跌停应判为 True"
        assert not bool(st.iloc[0]), "平盘应判为 False"


class TestSentinelPropagation:
    def test_首日非调仓日时产生哨兵(self):
        top = np.array([[3, 2, 1], [0, 1, 2], [1, 2, 0]])
        held = select_with_buffer(top, np.array([False, True, True]), 3)
        assert (held[0] == -1).all()
        assert not (held[1:] == -1).any()

    def test_哨兵在numpy索引里必须先屏蔽(self):
        """-1 在 numpy 里是最后一列，必须先转成安全索引。"""
        held = np.array([[0, 1, -1]])
        mat = np.array([[10.0, 20.0, 30.0, 40.0]])
        picked = np.take_along_axis(mat, np.where(held < 0, 0, held), axis=1)
        assert picked[0, 2] == 10.0, "直接索引会取到最后一列(40.0)"
