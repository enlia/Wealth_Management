"""Deflated Sharpe Ratio（多重比较校正后的夏普）——Bailey & López de Prado (2014)。

为什么需要它
------------
扫描 N 个策略/参数组合后，**最好的那个夏普必然偏高**（选择效应，多重比较）。
检验框架共同红线要求「多重比较声明（引用 mlfinlab Deflated Sharpe）」：
把观测夏普换算成「在 N 次尝试的择优下仍显著」的概率尺度。

取舍注记（据实声明）
--------------------
· 裁定原文指向 `mlfinlab.backtest_statistics` 现成件；本机**实测缺位**
  （find_spec('mlfinlab') = None，P1：新版需商业许可、不可用），仓内亦无实现。
  按 AGENTS 一.1 对 P1 的既有先例（公式不受版权保护、不复制源码），本文件
  按 Bailey & López de Prado (2014, *Journal of Portfolio Management* 40(5),
  eq.18–20「The Deflated Sharpe Ratio」) 公式自实现，逐式列于下方。
· 采用 **i.i.d. 形式**（未做 Newey-West 自相关修正）：策略为月频调仓低换手，
  日收益残余自相关未评估——若序列自相关显著，DSR 会偏乐观，边界已注明。
· N=1 退化分支：择优期望 max 单抽 = 0，取 SR0=0，DSR 退化为普通显著性概率
  Φ(z)（公式软最大近似对 N=1 发散，不能直接用）。

算式（年化口径，√ 项倍数换算层取 SCALING_TRADING_DAYS）
------------------------------------------------------
  SR_a = (mean(r)/std(r)) × √ppy                     # ppy=每年期数（默认 252）
  V    = (1 − g3·SR_a + (g4−1)/4·SR_a²) / (T−1)      # SR_a 估计量方差
  SR0  = √V × ((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))) # N 次尝试的择优期望
         （γ=0.5772156649… 欧拉常数；N=1 时 SR0=0）
  DSR  = Φ( (SR_a − SR0)·√(T−1) / √(1 − g3·SR_a + (g4−1)/4·SR_a²) )
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from factor_lab.config import SCALING_TRADING_DAYS

EULER_GAMMA = 0.5772156649015329


def _quad(sr_a: float, skew: float, kurt: float) -> float:
    """非正态二次型 1 − g3·SR + (g4−1)/4·SR²（式内分母公共项）。"""
    return 1.0 - skew * sr_a + (kurt - 1.0) / 4.0 * sr_a ** 2


def deflated_sharpe_ratio(returns, n_trials: int,
                          periods_per_year: float = SCALING_TRADING_DAYS) -> dict:
    """给定逐期收益序列与尝试次数，返回 (SR_a, SR0, DSR) 明细。

    returns: 逐期净收益（1 列可迭代）；n_trials: 同口径下试过的组合总数。
    """
    r = pd.Series(np.asarray(returns, dtype=float)).dropna()
    t = len(r)
    if t < 3:
        raise ValueError(f"收益序列仅 {t} 期（<3）—— 不足以估计夏普及其方差")
    sd = float(r.std(ddof=1))
    if sd <= 0:
        raise ValueError("收益序列标准差为 0 —— 夏普未定义，禁止静默给 0")
    if n_trials < 1:
        raise ValueError(f"n_trials={n_trials} < 1 —— 尝试次数必须为正")
    sr_a = float(r.mean()) / sd * math.sqrt(float(periods_per_year))
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, bias=False, fisher=False))   # 正态=3
    quad = _quad(sr_a, skew, kurt)
    if quad <= 0:
        raise ValueError(f"非正态二次型非正（{quad:.6g}）—— 高阶矩下公式退化，需改用模拟法")
    var = quad / (t - 1)
    if n_trials == 1:
        sr0 = 0.0                      # 无择优 → 退化为普通显著性检验
    else:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(var) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    z = (sr_a - sr0) * math.sqrt(t - 1) / math.sqrt(quad)
    return {
        "sr_annual": sr_a, "sr0": sr0, "dsr": float(stats.norm.cdf(z)),
        "z": z, "n_obs": t, "n_trials": int(n_trials), "skew": skew, "kurtosis": kurt,
        "periods_per_year": float(periods_per_year),
    }
