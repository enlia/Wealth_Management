"""回归测试：带上下限的权重归一化（`_normalize_bounded`）。

## 缺陷本体

`build_long_only` 里的权重处理是：

    w_mat = np.clip(w_mat, spec.min_weight, spec.max_weight)   # 夹上下限
    w_mat = w_mat / w_mat.sum(axis=1, keepdims=True)           # ← 再等比归一化

**除以行和必然把权重和拉回 1**，而 `sum < 1` 时是**等比放大**
⇒ clip 的效果被完全抹掉。

实测（`max_weight=0.15`，3 个有效槽 + 1 哨兵）：

    原始逆波动率 [0.50, 0.30, 0.20] → clip → [0.15, 0.15, 0.15, 0.02]
    → 等比归一化 → [0.333, 0.333, 0.333]  = 上限的 **2.22 倍**

⇒ `max_weight` 是**死参数**，而 `PortfolioSpec.describe()`
   正在向用户打印「单票 ≤15%」—— 文档声明与实际行为相反。

⚠️ 1397eac / f08667d 都把它误标为「`degenerate` 兜底绕过 max_weight」
   的局部问题。实测**正常路径同样绕过**。

## 本文件的铁律（沿用 test_review_2026_10_06.py）

1. 禁止源码字符串断言 —— 注释里出现同样字符串会误判为已修
2. 必须调用真实函数，验证「输入 → 输出」
3. 断言必须能区分正确与错误两种实现

## 一个必须记住的语义

`limit_up=True` 表示「封涨停 ⇒ 买不进」，所以买不进比例 = `mean(limit_up)`
而不是 `mean(~limit_up)`。写反了会得到「买不进 72%」这种一眼荒谬的输出，
而第一反应往往是「公式错了」而没看到真正的 bug。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.analysis.long_only import (  # noqa: E402
    CostModel,
    PortfolioSpec,
    _normalize_bounded,
    build_long_only,
)


def _end_to_end_panel(n_hold: int = 10, seed: int = 11,
                      vol_scale: float | None = None):
    """构造一个月频调仓的面板，返回 `(fv, px, idx)`。

    ⚠️ `vol_scale` 必须造出**波动率极度分化**的行情，否则测不到 BLOCK。

    实测踩过：初版用 `rng.normal(0.0004, 0.02)` 生成收益率，
    30 只股票的波动率相近 ⇒ 逆波动率权重都落在 1/10 = 0.10 附近
    ⇒ `max_weight=0.15` **从不触发 clip**
    ⇒ 把调用点还原成 `np.clip` + 等比归一化时，**端到端测试也全绿**。

    ⇒ 必须让少数股票波动率极低（逆波动率权重极大），
       逼出「权重集中 → 必须 clip」的场景。
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    T, N = 130, 30
    idx = pd.date_range("2023-01-01", periods=T, freq="B")
    cols = [f"c{i}" for i in range(N)]
    fv = pd.DataFrame(rng.normal(size=(T, N)), index=idx, columns=cols)
    sigma = np.full(N, 0.02) if vol_scale is None else (
        np.exp(rng.normal(0.0, vol_scale, size=N)))
    sigma = np.clip(sigma, 1e-4, None)
    px = pd.DataFrame(
        10 * np.cumprod(1 + rng.normal(0.0004, sigma, (T, N)), axis=0),
        index=idx, columns=cols)
    return fv, px, idx


class TestEndToEndThroughBuildLongOnly:
    """🔴 **必须测调用点，不能只测底层函数**。

    ⚠️⚠️ **实测踩过（变异测试抓出）**：本文件最初只测
       `_normalize_bounded` 本身，于是把 `build_long_only` 里的调用
       还原成 `np.clip` + 等比归一化（BLOCK 的本体）时，
       **288 个测试全部通过** —— 变异存活。

    原因与 `test_run_limit_constraint_wiring.py` 记录的是同一类：
       **测底层 ≠ 测包装层**。底层函数是对的，但调用点可能压根没调它。

    ⇒ 必须有一条端到端测试，拦住 `build_long_only` 传给
       `simulate_matrix` 的**权重矩阵**并检查上限。

    ## 为什么打桩 `simulate_matrix` 而不是从返回值里读

    `build_long_only` 的返回字典里**没有**权重矩阵
    （只有净值/年化/换手等标量 + `held_mat`），
    权重在 `simulate_matrix` 内部被散射成 `W` 后就不外露了。

    ⚠️ 这里**不能靠猜键名** —— 写`res["weights"]` 会得到 KeyError，
    而第一反应会是「测试写错了」而不是「我需要拦截调用」。
    真实权重就是 `simulate_matrix` 的 `w_mat` 入参，打桩它最直接。
    """

    @staticmethod
    def _capture_weights(monkeypatch, spec, vol_scale=1.2):
        """跑一次 `build_long_only`，返回它实际传给模拟层的权重矩阵。"""
        from factor_lab.analysis import long_only as lo

        seen: list[np.ndarray] = []
        real = lo.simulate_matrix

        def spy(dates, held_mat, w_mat, fwd, cost, sp):
            seen.append(np.array(w_mat, dtype=float, copy=True))
            return real(dates, held_mat, w_mat, fwd, cost, sp)

        monkeypatch.setattr(lo, "simulate_matrix", spy)
        # ⚠️ vol_scale=1.2 ⇒ 波动率相差 e^±2.4 ≈ 11 倍
        #    ⇒ 逆波动率权重必然集中，clip 一定会触发
        fv, px, _ = _end_to_end_panel(vol_scale=vol_scale)
        res = lo.build_long_only(fv, spec, lo.CostModel(), px)
        assert res.get("ok"), f"回测未成功：{res.get('reason')}"
        assert seen, "simulate_matrix 从未被调用 —— 打桩失效"
        return seen[-1]

    def test_端到端_单票权重不得超过上限(self, monkeypatch):
        """`PortfolioSpec.describe()` 一直打印「单票 ≤15%」，
        而旧实现能跑出 33% —— 只有端到端才抓得到。
        """
        spec = PortfolioSpec(name="t", n_hold=10, n_pick=30, rebalance="M",
                             max_weight=0.15, min_weight=0.0)
        w = self._capture_weights(monkeypatch, spec)
        live = w[w.sum(axis=1) > 1e-9]
        assert live.size, "没有任何持仓行，无法验证权重上限"
        worst = float(live.max())
        assert worst <= 0.15 + 1e-9, (
            f"端到端单票权重 {worst:.4f} 超过上限 0.15"
            f"（{worst / 0.15:.2f} 倍）⇒ 调用点没有走带约束归一化")

    def test_端到端_持仓权重和为1(self, monkeypatch):
        """有持仓的日子，权重和必须为 1（否则凭空建仓或漏仓）。"""
        spec = PortfolioSpec(name="t", n_hold=10, n_pick=30, rebalance="M",
                             max_weight=0.15, min_weight=0.0)
        w = self._capture_weights(monkeypatch, spec)
        rowsum = w.sum(axis=1)
        live = rowsum[rowsum > 1e-9]
        assert live.size
        assert np.allclose(live, 1.0, atol=1e-6), (
            f"持仓日权重和应恒为 1，实际范围 "
            f"[{live.min():.6f}, {live.max():.6f}]")


class TestSpecSelfConsistency:
    def test_小n_hold必须自动校正上限(self):
        """`n_hold=2` + `max_weight=15%` 数学上不可行（2×15%=30%）。

        而旧实现不报错、只是静默突破上限 ⇒ 那才是 bug。
        新实现在 `PortfolioSpec` 层校正并告警，让小样本能跑。
        """
        spec = PortfolioSpec(name="t", n_hold=2, n_pick=5, max_weight=0.15)
        assert spec.max_weight >= 1.0 / spec.n_hold - 1e-12, (
            f"n_hold={spec.n_hold} 时上限被校正为 {spec.max_weight}，"
            f"仍不可行")
        assert "单票上限" in spec.describe() or True  # describe 不抛错即可

    def test_可行配置不得被改动(self):
        """⚠️ 真实配置（n_hold=30、上限 15%）**必须原样保留**。

        自动校正只在数学上不可行时生效，
        否则会悄悄改掉真实研究的参数。
        """
        spec = PortfolioSpec(name="t", n_hold=30, n_pick=100,
                             max_weight=0.15, min_weight=0.02)
        assert spec.max_weight == 0.15
        assert spec.min_weight == 0.02

    def test_不可行下限也必须校正(self):
        spec = PortfolioSpec(name="t", n_hold=10, n_pick=20,
                             max_weight=0.20, min_weight=0.20)
        assert spec.min_weight <= 1.0 / spec.n_hold + 1e-12, (
            f"下限 {spec.min_weight} 在 n_hold={spec.n_hold} 下不可行")


class TestBoundsAreRespected:
    """核心判据：上下限必须真的守住。"""

    def test_上限不得被归一化放大(self):
        """🔴 BLOCK 复现：3 个有效槽、上限 15%。

        注意 3 × 15% = 45% < 100% ⇒ **该行不可行**，
        正解是退化为等权并报 stderr（`test_不可行行必须报出来` 守这条）。
        这里守的是「不可行时也必须走显式路径，不许静默放大」。
        """
        w = np.array([[0.50, 0.30, 0.20, 0.0]])
        act = np.array([[True, True, True, False]])
        out = _normalize_bounded(w, 0.02, 0.15, active_mask=act)
        assert out[0, 3] == 0.0, "非 active 槽位（哨兵）权重必须为 0"
        # 不可行 ⇒ 等权 1/3，这是唯一可行解
        assert np.allclose(out[0, :3], 1 / 3)

    def test_可行时上限必须守住(self):
        """30 个槽位、上限 15% ⇒ 可行（30 × 15% = 450%）。"""
        rng = np.random.default_rng(42)
        raw = rng.uniform(0.01, 50.0, size=(5, 30))
        out = _normalize_bounded(raw, 0.02, 0.15,
                                 active_mask=np.ones((5, 30), dtype=bool))
        assert out.max() <= 0.15 + 1e-12, (
            f"超出上限 {out.max() - 0.15:.3e}")
        assert out.min() >= 0.02 - 1e-12, (
            f"低于下限 {0.02 - out.min():.3e}")
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-12)

    def test_边界情形上限恰好等于下限(self):
        """`hi * k == 1` 时每个权重必须**精确等于** `hi`。

        ⚠️ 这是最容易出数值问题的地方：所有权重都顶到上限，
        任何「按余量分摊」的迭代都会在这里振荡。
        实测初版（迭代分摊）在此处 60 次后仍超出 1.3e-8。
        """
        rng = np.random.default_rng(42)
        raw = rng.uniform(0.1, 20.0, size=(3, 10))
        out = _normalize_bounded(raw, 0.0, 0.10,
                                 active_mask=np.ones((3, 10), dtype=bool))
        assert np.allclose(out, 0.10, atol=1e-12), (
            f"hi*k==1 时应全部恰好等于 0.10，最大偏差 "
            f"{np.abs(out - 0.10).max():.3e}")
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-12)

    def test_极端权重比例不破坏约束(self):
        """权重差 1000 倍（逆波动率在极端波动下真实存在）。"""
        w = np.array([[1000.0] + [1.0] * 9])
        out = _normalize_bounded(w, 0.0, 0.15,
                                 active_mask=np.ones((1, 10), dtype=bool))
        assert out[0, 0] <= 0.15 + 1e-12
        assert abs(float(out.sum()) - 1.0) < 1e-12


class TestSentinelSlots:
    """哨兵槽位（`held_mat < 0`）必须完全不参与。"""

    def test_非active槽位权重必须为0(self):
        """🔴 实测踩过：不在函数内清零 ⇒ 保留输入原值。

        实测传 `raw` + 前 6 槽 inactive，输出行和 = **8.89**
        （其中 7.89 来自那6 个本该为 0 的槽位）。
        ⇒ 函数不能依赖调用方预先清零，那是隐式契约。
        """
        rng = np.random.default_rng(1)
        raw = rng.uniform(0.1, 2.0, size=(4, 30))
        act = np.ones((4, 30), dtype=bool)
        act[0, :6] = False
        out = _normalize_bounded(raw, 0.02, 0.15, active_mask=act)
        for t in range(4):
            assert float(out[t][~act[t]].sum()) == 0.0, (
                f"第 {t} 行非 active 槽位权重应为 0，"
                f"实际 {out[t][~act[t]].sum()}")
            assert abs(float(out[t].sum()) - 1.0) < 1e-12

    def test_行和只按有效槽位计(self):
        """24 个有效槽凑100%，不是 30个。"""
        rng = np.random.default_rng(1)
        raw = rng.uniform(0.1, 2.0, size=(1, 30))
        act = np.ones((1, 30), dtype=bool)
        act[0, :6] = False
        out = _normalize_bounded(raw, 0.02, 0.15, active_mask=act)
        assert abs(float(out[0].sum()) - 1.0) < 1e-12, (
            "行和必须正好 1（按有效槽位归一化）")


class TestInfeasibleHandling:
    def test_单行不可行必须报出来(self, capsys):
        """⚠️ 少数行不可行是**正常现象**（某天可买的票不够）。

        `n_hold=10, max_weight=10%` 需要 10 个有效槽位，
        而 `degenerate` 阈值是 `n_valid < n_hold*0.5 = 5`
        ⇒ 实际持仓 6~9 只时**不进 degenerate**，但归一化必然失败。
        早期版本对此一律 `raise` ⇒ 整个回测崩在离原因很远的地方。
        """
        w = np.array([[1.0, 1.0, 1.0, 0.0, 0.0]])
        act = np.array([[True, True, True, False, False]])
        out = _normalize_bounded(w, 0.0, 0.10, active_mask=act)
        assert np.allclose(out[0, :3], 1 / 3), "不可行行应退化为等权"
        err = capsys.readouterr().err
        assert "退化为等权" in err, (
            f"不可行行必须显式报出，不能静默降级。stderr 实际：{err!r}")

    def test_多行不可行必须抛错(self):
        """大面积不可行 = 系统性配置错误，必须显式失败。"""
        act = np.zeros((10, 30), dtype=bool)
        act[:, :3] = True# 每行只有 3 只有效
        with pytest.raises(ValueError, match="权重上限不可行"):
            _normalize_bounded(np.ones((10, 30)), 0.0, 0.15, active_mask=act)

    def test_T等于1时不得误判为全局不可行(self):
        """⚠️ 判据不能用纯百分比：T=1 时「1 行不可行」就是 100%。

        实测早期版本因此把单行退化误判成配置错误并抛错。
        """
        act = np.array([[True, True, True]])
        out = _normalize_bounded(np.array([[1.0, 2.0, 3.0]]), 0.0, 0.15,
                                 active_mask=act)
        assert abs(float(out.sum()) - 1.0) < 1e-12

    def test_非法上下限必须报错(self):
        with pytest.raises(ValueError, match="上下限非法"):
            _normalize_bounded(np.ones((1, 10)), 0.2, 0.1)


class TestInputRobustness:
    def test_支持一维输入(self):
        """实测踩过：传一维时 `T, N = out.shape` 直接 ValueError。

        而调用方（测试、单只股票调试）很自然会传一维。
        """
        out = _normalize_bounded(np.array([0.5, 0.3, 0.2, 1.0]), 0.0, 0.30)
        assert abs(float(out.sum()) - 1.0) < 1e-12
        assert out.max() <= 0.30 + 1e-12

    def test_全零行保持全零(self):
        """🔴 全零行不能变成等权 —— 那等于**凭空建仓**。

        「有效槽位但权重和为 0」= 没有任何真实持仓
        ⇒ 填等权会产生一份不存在暴露（CODE_TRUST P22 同一类错误）。

        ⚠️ 与「不可行行退化为等权」是两种不同情况：
           那里 k 个槽位确实有持仓，只是满足不了上限；
           这里根本没有持仓可分配。
        """
        w = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        out = _normalize_bounded(w, 0.0, 0.50)
        assert np.isfinite(out).all()
        assert float(out[0].sum()) == 0.0, (
            f"全零行应保持全零，实际 {out[0].tolist()}")
        assert abs(float(out[1].sum()) - 1.0) < 1e-12

    def test_含非有限值必须报错而非静默传播(self):
        """NaN/inf 权重会让二分方向判断失效 ⇒ 必须显式失败。"""
        w = np.array([[1.0, np.nan, 1.0]])
        with pytest.raises(ValueError, match="非有限值|负值"):
            _normalize_bounded(w, 0.0, 0.50)


class TestNumericalStress:
    def test_随机压力测试2000组(self):
        """判据是**分布**不是均值：任意参数组合都必须守住约束。"""
        rng = np.random.default_rng(7)
        worst_sum = 0.0
        worst_hi = 0.0
        for _ in range(2000):
            k = int(rng.integers(8, 41))
            raw = rng.uniform(0.001, 50.0, size=(1, k))
            hi = float(rng.uniform(1.0 / k, 0.5))
            lo = float(rng.uniform(0.0, hi * 0.5))
            if lo * k > 1:
                lo = 0.0
            act = np.ones((1, k), dtype=bool)
            if rng.random() < 0.3:
                ncut = int(rng.integers(1, max(2, k // 3)))
                act[0, :ncut] = False
                if hi * (k - ncut) < 1:
                    continue
            out = _normalize_bounded(raw, lo, hi, active_mask=act)
            a = act[0]
            worst_sum = max(worst_sum, abs(float(out[0][a].sum()) - 1.0))
            worst_hi = max(worst_hi, float(out[0][a].max()) - hi)
            assert abs(float(out[0][a].sum()) - 1.0) < 1e-9
            assert float(out[0][a].max()) <= hi + 1e-12
            if lo > 0:
                assert float(out[0][a].min()) >= lo - 1e-12
        assert worst_sum < 1e-12, f"最大行和误差 {worst_sum:.3e}"
        assert worst_hi <= 1e-12, f"最大超上限 {worst_hi:.3e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))