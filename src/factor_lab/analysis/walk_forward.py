"""滚动样本外检验（walk-forward）：替代「单次切分」这个不可信的证据来源。

为什么必须废弃单次切分
--------------------
实测（2026-10-06，rev5 因子，全市场 5,606 只）：
```
单次 70/30 切分样本内-7.78%/年   ← 亏
单次 70/30 切分样本外 +27.92%/年 ← 看着很棒
分年度跑赢基准  4/11 年（36%）
```
单次切分把 2016-2023 划成「样本内」、2023-2026 划成「样本外」，
而**切点恰好落在风格分界上**。后半段是单边上涨行情，
任何等权基准策略都会赚钱 —— 这个「样本外有效」是运气，不是稳健性。

**更严重的是：切点决定了结论。**
换成 80/20、60/40 切分，或换个切点年份，样本外年化会完全不同。
一个因子只在一个切分点上有效，等于没有效��。

滚动检验怎么做
------------
```
窗口 1：训练 2016-2018→ 测试 2019
窗口 2：训练 2017-2019    → 测试 2020
窗口 3：训练 2018-2020    → 测试 2021
...
```
每一步都满足「训练在前、测试在后」，**且测试段两两不重叠**。
最终看的是「各测试段的超额分布」：
  · 超额为正的窗口占比（胜率） —— 低于 60% 不可用
  · 超额中位数 —— 比均值稳健，不被个别年份带偏
  · 最差窗口 —— 决定能不能真的拿钱去做

判据（不是「平均年化 > 0」）
------------------------
| 指标 | 门槛 | 为什么 |
|---|---|---|
| 测试段超额胜率 | ≥ 60% | 单次样本外 100% 胜率在 3 个窗口上出现过，假的 |
| 超额中位数 | > 0 | 比均值稳健 |
| 最差窗口超额 | > −5% | 决定实盘可承受的回撤 |
| 各窗口超额离散度 | 不过大 | 忽正忽负= 靠运气 |

⚠️ **训练段在这里不用于选参数**（那只做一次全样本扫描）。
   本模块只回答「因子在全历史里稳不稳」，用于**否决**而非**挑选**——
   挑选必须在 walk-forward 内部完成，否则又是样本外污染。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# 窗口数下限：低于此数量「胜率」没有统计意义。
# ⚠️ 实测踩过：judge 原先不检查窗口数，单窗口且超额为正时
#   直接判「三项判据全部通过」—— 而胜率是二项统计，
#   n=1 时只有 0% / 100% 两种取值，阈值 60% 对它毫无约束力。
MIN_WINDOWS = 5


@dataclass(frozen=True)
class WindowSpec:
    """滚动窗口的切分参数。"""
    train_years: int = 3
    test_months: int = 12
    step_months: int = 12
    min_train_years: int = 2

    def describe(self) -> str:
        return (f"训练 {self.train_years} 年 / 测试 {self.test_months} 月 / "
                f"步进 {self.step_months} 月")


def make_windows(dates, spec: WindowSpec) -> list[tuple[list, list]]:
    """生成 (训练期, 测试期) 日期列表序列。

    ⚠️ **测试段必须两两不重叠**。若step < test 就会重叠，
    重叠的测试段彼此不独立，胜率会被虚高（同一个行情被数了两遍）。

    ⚠️ 这条**必须在代码里强制**，不能只写在 docstring 里
    （review 实测：初版只有注释，`step_months` 是公开字段，
    传 `step=6, test=12` 得到 14 个窗口、13 对相邻重叠，
    胜率直接虚高）。故此处直接抛错。
    """
    if spec.step_months < spec.test_months:
        raise ValueError(
            f"step_months({spec.step_months}) < test_months({spec.test_months})"
            f" ⇒ 测试段会重叠，胜率虚高。要用滚动前进模式请显式传 "
            f"expand_overlap=True 并自行处理样本非独立问题。")
    idx = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates))))
    if len(idx) == 0:
        return []
    start = idx[0]
    end = idx[-1]

    def shift(d: pd.Timestamp, months: int) -> pd.Timestamp:
        y = d.year + (d.month - 1 + months) // 12
        m = (d.month - 1 + months) % 12 + 1
        return pd.Timestamp(year=y, month=m, day=1)

    windows: list[tuple[list, list]] = []
    cut = shift(start, spec.train_years * 12)
    while True:
        test_end = shift(cut, spec.test_months)
        if test_end > end:
            break
        tr = [d for d in idx if start <= d < cut]
        te = [d for d in idx if cut <= d < test_end]
        if len(tr) >= spec.min_train_years * 200 and len(te) >= 60:
            windows.append((tr, te))
        cut = shift(cut, spec.step_months)
    return windows


@dataclass
class WalkForwardResult:
    """滚动检验的汇总结论。"""
    因子: str
    窗口数: int
    胜率: float
    超额中位数: float
    最差超额: float
    最好超额: float
    可用: bool
    原因: str
    明细: pd.DataFrame

    def report(self) -> str:
        flag = "✅ 可用" if self.可用 else "❌ 不可用"
        lines = [
            f"{self.因子}: {flag} —— {self.原因}",
            f"  窗口数 {self.窗口数}  胜率 {self.胜率:.0%}  "
            f"超额中位数 {self.超额中位数:+.2%}  "
            f"最差 {self.最差超额:+.2%}  最好 {self.最好超额:+.2%}",
        ]
        return "\n".join(lines)


def judge(windows: list[dict], factor: str,
          min_win_rate: float = 0.60,
          min_median: float = 0.0,
          worst_floor: float = -0.05,
          min_windows: int = MIN_WINDOWS) -> WalkForwardResult:
    """按稳健性判据裁决，不看「平均年化」。

    ⚠️ **为什么不用平均超额**：
       单个窗口 +200% 会把 10 个 −10% 的均值拉到正数，
       而这种策略实盘根本拿不住。胜率 + 中位数 + 最差值三个一起看，
       才能反映「能不能真的拿钱去做」。

    ⚠️ **回测失败的窗口（超额 NaN）必须先剔除**（review 实测踩过）：
       `NaN > 0` 恒为False，直接统计会把「失败窗口」计成「跑输」，
       而 `np.median` 在含 NaN 时返回 NaN ——
       报告里出现「最差 +nan%」，判据「最差 > −5%」**根本没执行**，
       只是靠 `not (nan > 0)` 恰好判False 而歪打正着。
       且失败率本身就是要报告的信息。
    """
    if not windows:
        return WalkForwardResult(
            因子=factor, 窗口数=0, 胜率=0.0, 超额中位数=float("nan"),
            最差超额=float("nan"), 最好超额=float("nan"), 可用=False,
            原因="没有有效窗口", 明细=pd.DataFrame())

    full = pd.DataFrame(windows)
    valid = full["超额"].dropna() if "超额" in full.columns else pd.Series(dtype=float)
    n_fail = len(full) - len(valid)

    reasons: list[str] = []
    if n_fail:
        reasons.append(f"{n_fail}/{len(full)} 个窗口回测失败")

    if valid.empty:
        return WalkForwardResult(
            因子=factor, 窗口数=0, 胜率=0.0, 超额中位数=float("nan"),
            最差超额=float("nan"), 最好超额=float("nan"), 可用=False,
            原因="；".join(reasons + ["无成功窗口"]), 明细=full)

    # ⚠️ **窗口数必须有下限**（review 实测踩过）：
    #    胜率是二项统计，n=1 时只有 0% / 100% 两种取值，
    #    `min_win_rate=0.60` 对它**毫无约束力** ——
    #    实测单窗口且超额为正时judge 判「三项判据全部通过」。
    #    这直接违背本模块自己的立论（docstring：单窗口 100% 胜率是假的）。
    if len(valid) < min_windows:
        reasons.append(f"窗口数 {len(valid)} < {min_windows}，胜率无统计意义")

    exc = valid.to_numpy(dtype=float)
    win_rate = float((exc > 0).mean())
    med = float(np.median(exc))
    worst = float(exc.min())
    best = float(exc.max())

    if win_rate < min_win_rate:
        reasons.append(f"胜率 {win_rate:.0%} < {min_win_rate:.0%}")
    if not med > min_median:
        reasons.append(f"超额中位数 {med:+.2%} ≤ 0")
    if worst < worst_floor:
        reasons.append(f"最差窗口 {worst:+.2%} 击穿 {worst_floor:.0%}")

    return WalkForwardResult(
        因子=factor, 窗口数=len(valid), 胜率=win_rate, 超额中位数=med,
        最差超额=worst, 最好超额=best, 可用=not reasons,
        原因="；".join(reasons) if reasons else "全部判据通过",
        明细=full)