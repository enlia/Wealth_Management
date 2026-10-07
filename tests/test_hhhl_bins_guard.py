"""hh_hl_score 等宽分箱双向守护测试（实验轮 1·B 线）。

背景（AGENTS.md 七·下一步 1 处方作废记录；brief=.planning/experiment-round1-B-brief.md）：
    hh_hl_score 取值恒 {0,1,2}（无负半区），"离散因子需 zero_aware=True" 的旧处方
    数学上不成立；正解 = 等宽分箱 bins=3
    （price_volume.DISCRETE_FACTORS["hhhl20"] = 3 + alphalens_adapter 的 bins 路径）。

双向守护（CODE_TRUST P23 实例 4 教训：还原测试的打桩必须复现真实触发形态）：
    ① 修复版（bins=3）→ 必须 PASS，且「净 = 毛 − 往返费率 × 单期换手 × 组数 × 252」
       成本恒等式精确闭合（组数 = 真实最大分位标签）；
    ② 还原 bug 版（退回等频分位 quantiles=5 / 旧 zero_aware 形态）→ 必须 FAILED
       （显式异常，不许静默出数）。
fixture 复现真实触发形态：同一横截面上 {0,1,2} 三种取值**同时大量存在**（各 10/30 只）
    —— 这正是 pd.qcut(x, 5) 在 3 取值上丢样本、触发 MaxLossExceededError 的真实条件；
    单一取值或单形态打桩测不出问题（假绿，见 CODE_TRUST 实例 4）。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from factor_lab.analysis.alphalens_adapter import run_tear_sheet
from factor_lab.config import DEFAULT_COST, DEFAULT_RESEARCH
from factor_lab.factors.price_volume import (
    DISCRETE_FACTORS,
    bins_of,
    compute_factor,
)

WIN = 20               # hh_hl_score 的 FACTORY 注册窗口
N_PER = 10             # 每种形态 10 只 → 横截面 30 只、三取值同场（真实触发形态）
# 300 交易日：前 2*WIN-1 日为评分预热段（因子值 NaN）。预热占比必须小，
# 否则 max_loss 消耗在前瞻收益/预热上，测不到分箱环节（CODE_TRUST 实例 4
# 的"打桩造不出真实触发形态"坑 —— 曾以 80 日序列假绿/假红一轮）。
DATES = pd.bdate_range("2020-01-01", periods=300)
CODES = [f"syn{k:03d}" for k in range(3 * N_PER)]
# 检验配置只开 1D/5D 前瞻（spread 取 periods[0]='1D'）：
# 默认 (1,5,10,20,60) 会让 60D 尾部把 max_loss 吃满，与分箱无关。
CFG = type(DEFAULT_RESEARCH)(**{**DEFAULT_RESEARCH.__dict__, "periods": (1, 5)})


def _pattern_long() -> pd.DataFrame:
    """构造三形态横截面长表（走 compute_factor 的真实输入口径）。

    形态设计（窗口内极值 vs 上一窗口极值，hh_hl_score 的真实判据）：
      rise → HH✓ HL✓ = 2 分；mix → HH✓ HL✗ = 1 分；fall → HH✗ HL✗ = 0 分。
    """
    t = np.arange(len(DATES), dtype=float)
    rows = []
    for k, code in enumerate(CODES):
        if k < N_PER:                       # rise
            hi = 100 + t
            lo = hi - 1
        elif k < 2 * N_PER:                 # mix：高点抬升、低点下移
            hi = 100 + t
            lo = 100 - t
        else:                               # fall
            hi = 100 - t
            lo = hi - 1
        close = (hi + lo) / 2
        rows.append(pd.DataFrame({
            "code": code, "date": DATES,
            "high_adj": hi, "low_adj": lo,
            "close": close, "close_adj": close,
        }))
    return pd.concat(rows, ignore_index=True)


def _prices() -> pd.DataFrame:
    """价格宽表（close_adj）：带噪声的几何随机游走，仅用于前瞻收益，不喂 hh_hl_score。"""
    rng = np.random.default_rng(20261013)
    rets = rng.normal(0.0, 0.01, size=(len(DATES), len(CODES)))
    px = 50.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(px, index=DATES, columns=CODES)


LONG = _pattern_long()
PRICES = _prices()
FACTOR = compute_factor("hhhl20", LONG)


class Test因子形态与登记:
    """守护 P23 残留（docstring 曾声称取值 {-1,0,1,2}）与 DISCRETE_FACTORS 登记。"""

    def test_取值集合恒为0_1_2三值同场(self):
        vals = set(FACTOR.dropna().unique())
        assert vals == {0.0, 1.0, 2.0}, (
            f"hh_hl_score 取值集合应恒为 {{0,1,2}}，实测 {sorted(vals)}；"
            "出现 -1 即与实现不符（docstring 声称形态回归）")

    def test_无负半区_旧zero_aware处方前提不成立(self):
        """旧处方「离散因子用 zero_aware=True」要求取值有正负分界；
        实测恒非负 ⇒ 负半区必空 ⇒ 处方数学上不成立（AGENTS 七·下一步 1）。"""
        assert (FACTOR.dropna() >= 0).all()
        assert int((FACTOR.dropna() < 0).sum()) == 0

    def test_离散因子已登记分箱数(self):
        assert DISCRETE_FACTORS["hhhl20"] == 3
        assert bins_of("hhhl20") == 3
        assert bins_of("rev5") is None          # 连续因子走等频分位
        with pytest.raises(KeyError):
            bins_of("不存在的因子")


class Test修复版必过:
    """修复版（等宽分箱 bins=3）必须可用，且成本恒等式精确闭合。"""

    def test_bins3_可跑通成本乘数取真实最大分位(self):
        r = run_tear_sheet(FACTOR, PRICES, CFG, DEFAULT_COST,
                           factor_name="hhhl20", bins=3)
        labels = sorted(int(x) for x in r.quantile_returns.index)
        assert len(labels) >= 2, f"等宽分箱后有效分位不足: {labels}"
        assert r.quantiles == max(labels), "成本乘数必须 = 真实最大分位标签"

    def test_bins3_成本恒等式精确闭合(self):
        """净 = 毛 − 0.0020 × 平均单期换手 × 组数 × 252（线性年化，口径三件套见报告）。"""
        r = run_tear_sheet(FACTOR, PRICES, CFG, DEFAULT_COST,
                           factor_name="hhhl20", bins=3)
        expected_cost = (DEFAULT_COST.round_trip * r.turnover_mean
                         * r.quantiles * 252.0)
        assert (r.gross_spread - r.net_spread_after_cost) == pytest.approx(
            expected_cost, rel=1e-12)

    def test_bins3_IC与换手为有效数值(self):
        r = run_tear_sheet(FACTOR, PRICES, CFG, DEFAULT_COST,
                           factor_name="hhhl20", bins=3)
        assert int(r.ic_by_period.loc[1, "count"]) > 0
        assert np.isfinite(r.ic_by_period.loc[1, "mean"])
        assert np.isfinite(r.turnover_mean)
        assert np.isfinite(r.gross_spread) and np.isfinite(r.net_spread_after_cost)


class Test还原bug版必失败:
    """还原 bug 版（不分箱 / 旧 zero_aware 处方）必须 FAILED —— 新守护要"咬人"。

    复现的是 bd64b99 之前的真实失败形态：横截面 3 取值进 5 等频分位，
    分箱丢样本超 max_loss ⇒ MaxLossExceededError（显式报错，不静默出数）。
    """

    def test_还原bug_退回等频分位必失败(self, capsys):
        with pytest.raises(Exception) as ei:
            run_tear_sheet(FACTOR, PRICES, CFG, DEFAULT_COST,
                           factor_name="hhhl20", bins=None)
        assert "MaxLossExceededError" in type(ei.value).__name__ or \
               "max_loss" in str(ei.value), (
            f"还原版应触发分箱丢样本超限的显式失败，实测异常: "
            f"{type(ei.value).__name__}: {ei.value}")
        self._assert_binning_loss(capsys)

    def test_还原bug_旧zero_aware处方必失败(self, capsys):
        with pytest.raises(Exception) as ei:
            run_tear_sheet(FACTOR, PRICES, CFG, DEFAULT_COST,
                           factor_name="hhhl20", bins=None, zero_aware=True)
        assert "MaxLossExceededError" in type(ei.value).__name__ or \
               "max_loss" in str(ei.value) or \
               "分位" in str(ei.value) or "quantile" in str(ei.value).lower(), (
            f"旧 zero_aware 处方在无负半区因子上必须显式失败，实测异常: "
            f"{type(ei.value).__name__}: {ei.value}")
        self._assert_binning_loss(capsys)

    @staticmethod
    def _assert_binning_loss(capsys) -> None:
        """丢失必须发生在**分箱阶段**，不是前瞻收益/预热（对照组=修复版同 fixture
        零丢失通过 ⇒ 两条路径唯一差异就是分箱）。"""
        out = capsys.readouterr().out
        m = re.search(r"in forward returns computation and ([\d.]+)% "
                      r"in binning phase", out)
        assert m, f"未捕获到 alphalens 丢失分解输出，实测 stdout: {out!r}"
        assert float(m.group(1)) > 0, (
            f"丢失应发生在分箱阶段（binning），实测分解: {out!r}")
