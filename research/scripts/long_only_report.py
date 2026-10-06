"""单边多头检验的汇报统计：基准年化、横截面标准化、结果打印。

职责边界（单一职责）：
    · 只做「统计换算 + 打印格式」，不取数、不拼面板、不跑组合回测
    · 取数在 ``long_only_data.py``，面板组装在 ``long_only_panel.py``
    · 组合构建在 ``factor_lab.analysis.long_only``，入口是 ``run_long_only.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.analysis.long_only import _year_span  # noqa: E402


def combine(panels: dict[str, pd.DataFrame],
            weights: dict[str, float]) -> pd.DataFrame:
    """加权合成多因子（已标准化的 z 分数）。"""
    acc, wsum = None, 0.0
    for name, w in weights.items():
        p = panels.get(name)
        if p is None:
            continue
        acc = p * w if acc is None else acc + p * w
        wsum += abs(w)
    return acc / wsum if wsum else None


def report(r: dict, label: str) -> None:
    if not r.get("ok"):
        print(f"  {label}: 失败（{r.get('reason')}）")
        return
    print(f"  {label:<22} 年化 {r['年化收益']:>7.2%}  波动 {r['年化波动']:>6.2%}  "
          f"夏普 {r['夏普']:>5.2f}  回撤 {r['最大回撤']:>7.2%}  "
          f"换手 {r['平均换手']:>5.1%}  成本 {r['平均年成本']:>6.2%}")


def _bench_stats(rets: pd.Series) -> float:
    """等权基准的年化收益。

    ⚠️ **必须用自然日口径**（`_year_span`），与策略侧完全一致。
       初版这里用 `len(r)/252`，而策略侧用 `(日期差)/365.25`，
       同一份收益两个年化，基准被低估约 0.5pp/年。
       而「超额 = 策略 − 基准」是所有结论的判据 ——
       口径不一致会让**所有超额都偏悲观**。

    ⚠️ 252 是美股口径。A 股一年实际约 243~245 个交易日，
       用 252 会让年数偏大约 3.5%。
    """
    r = rets.dropna()
    if len(r) < 2:
        return float("nan")
    nav = float((1 + r).prod())
    if nav <= 0:
        return -1.0
    years = _year_span(r.index)
    return nav ** (1 / years) - 1


def _cross_z(df: pd.DataFrame) -> pd.DataFrame:
    """横截面 z-score 标准化（每天独立）。

    ⚠️ **必须横截面，不能全样本**：
    不同年份市场风格不同（2017 小盘占优 vs 2020 大盘），
    全样本标准化会让因子含义随时间漂移，样本外必然失效。

    ⚠️ **性能**：初版逐列 `groupby(level=0).transform(z)`，
    5,600 列 × 2,611 行跑不出来（groupby 在宽表上极慢）。
    改成按行 numpy 广播：均值/标准差都是 (T,1)，与 (T,N) 一次广播完成。
    """
    m = df.to_numpy(dtype=float)
    valid = np.isfinite(m)
    cnt = valid.sum(axis=1, keepdims=True)
    s = np.where(valid, m, 0.0).sum(axis=1, keepdims=True)
    mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    var = np.where(valid, (m - mean) ** 2, 0.0).sum(axis=1, keepdims=True)
    std = np.where(cnt > 1, np.sqrt(var / np.maximum(cnt - 1, 1)), np.nan)
    z = np.where(std > 0, (m - mean) / std, 0.0)
    z[~valid] = np.nan
    return pd.DataFrame(z, index=df.index, columns=df.columns)