"""市值分层混淆检验的**共享实现** —— 两个脚本必须用同一份。

为什么必须共享
--------------
`run_size_decile.py`（价量因子）与 `run_financial_walk_forward.py`（财务因子）
都要做「按市值分层、层内各自选股」的检验。
2026-10-06 实测踩坑：初版把逻辑各写一份，结果
  · 价量侧传 `z.loc[te]`（仅测试段）
  · 财务侧传 `z[cols]`（全期 2016-2026）
⇒ 同一因子年化差 10 倍（rev5 +19.80% vs +2.19%），
  而基准中位相同（21.14%）证明窗口没错，只能是策略区间错了。
**两份实现 = 结论不可比。**

🔴🔴 **BLOCK（2026-10-06 抓出）：分层必须用窗口【首日】市值，不能用末日**
------------------------------------------------------------
初版用 `mcap_f.loc[te[-1]]`（窗口最后一个交易日）分层。致命之处：

> 市值 = 股价 × 股本，而股本在一年内基本不变
> ⇒ **用年末市值分层 = 用「这一年的涨幅」分层**

于是「大盘层」被定义成「年内涨得最多的那一批」，
再拿它的收益当基准 —— **循环论证**。

实测（2025 窗口，全A 股）：
```
按 2025-01-02 市值分层的大盘层   累计 +31.25%
按 2025-12-31 市值分层的大盘层   累计 +49.80%
     差 18.5pp —— 全部来自「按结果分组」
```
后果：基准被系统性高估 ⇒ **所有层内超额被系统性低估**，
而层内超额正是「信号是否独立于市值」的判据。

正确做法：用 `te[0]`（窗口首日）市值。
它在 `te[0]` 收盘时**已经可知**，且不含任何窗口内收益信息。

⚠️ 窗口首日不是精确的调仓日（策略在窗口内按月调仓），
   但它至少是**事前可知**的时点，而末日时点不是。
   若要更严格，应在每个调仓日重新分层 —— 那是另一种实现，
   本函数不提供，因为会与 `build_long_only` 的月度调仓口径打架。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from run_long_only import _bench_stats  # noqa: E402

from factor_lab.analysis.costs import CostModel  # noqa: E402
from factor_lab.analysis.long_only import (  # noqa: E402
    PortfolioSpec,
    build_long_only,
)

# 分层内至少要有这么多只，否则选不出 n_hold + 候选池
MIN_STOCKS_PER_BUCKET = 120

LAYERS = ("全", "大", "中", "小")


def split_buckets(m: pd.Series) -> dict[str, list[str]]:
    """按流通市值三分位切层。`m` 是单日横截面（索引=代码）。

    ⚠️ 缺失市值的代码直接剔除（`valid`），不做任何填充——
       填0 或前向填充都会把它们塞进「小盘层」，
       而小盘层是收益最高的一层，等于凭空造收益。

    ⚠️ **层用排名切，不用 `>= quantile` 切**（2026-10-06 修复）。
       `>= quantile(q2)` 会把**所有等于 q2 的行**整批纳入大盘层。
       实测缺陷（n=1000，不同市值个数 d）：

       | d（不同市值个数） | 旧: 大/中/小 | 新: 大/中/小 |
       |---|---|---|
       | 1000（全互异） | 334/333/333 | 334/333/333 |
       | 50| 340/320/340 | 334/333/333 |
       | 10| **400/200/400** | 334/333/333 |
       | 2 | **500/0/500** | 334/333/333 |
       | 1（全同值） | **1000/0/1000** | 334/333/333 |

       即：
         · 层大小失衡（d=10 时大/小层各占全集 40%）
         · **中盘层可能为空**（d=2 时中=0）
         · **层间重叠**（d=1 时大∩小 = 全部 1000 只，三层完全重合，
           混淆检验彻底失效）

       ⚠️ 顺带纠正一个错误诊断：初版 docstring 写
       「大盘层 334 只比全层 333 只还多」——
       **334 并没有超过全集 1000**，那句话本身是错的（CODE_TRUST P23：
       错误的原因描述会误导下一次修复）。真实缺陷是上表的三条。

    ⚠️ 真实数据上并列极少（实测 2023-01-03：5,062 只仅 1 组并列、2 只股票），
       所以这个 bug 在生产数据上影响 ≤2 只 —— **但它让测试失真**：
       旧实现同样满足「层大小相等」以外的大多数直觉断言。
    """
    valid = [c for c in m.index if pd.notna(m[c])]
    #🔴 MAJOR（2026-10-06 review）：门槛必须是**每层**的门槛 ×3。
    #   初版拿 `MIN_STOCKS_PER_BUCKET`(120) 卡**全池**，
    #   而 `run_in_subset` 要求**每层** ≥120 ⇒ 全池 120~359 时三层全部 NaN，
    #   但「全」层照样出数字 ⇒ **混淆检验静默空转**，
    #   汇总表里该行与「正常跑完」长得一样，只是多了几个 NaN。
    #   实测：359 只 → [120,120,119]（小层刚好不够）；360 只 → [120,120,120]。
    if len(valid) < MIN_STOCKS_PER_BUCKET * 3:
        return {}
    s = m[valid]
    n = len(valid)
    rank = s.rank(method="first", ascending=True)
    #前 1/3 小盘，中间 1/3 中盘，后 1/3 大盘
    小 = [c for c in valid if rank[c] <= n / 3]
    大 = [c for c in valid if rank[c] > 2 * n / 3]
    中 = [c for c in valid if c not in set(小) | set(大)]
    return {"全": valid, "大": 大, "中": 中, "小": 小}


def run_in_subset(z_sub: pd.DataFrame, px_sub: pd.DataFrame,
                  cost: CostModel, n_hold: int, n_pick_mult: int,
                  factor: str) -> tuple[float, str]:
    """在给定的股票子集内跑回测，返回 (年化收益, 跳过原因)。

    ⚠️ `z_sub` / `px_sub` 必须是**测试窗口切片**，不是全期面板。
       全期会被 11 年摊薄（实测 rev5 因此差 10倍）。

    ⚠️ **跳过必须带原因**（ENGINEERING 第四节：禁止静默 fallback）。
       初版直接 `return float("nan")` 不打印任何东西，
       上层只能笼统说「分层无有效数据」，
       区分不出「层内样本不足」与「回测失败」。
    """
    min_need = max(n_hold * 4, MIN_STOCKS_PER_BUCKET)
    if z_sub.shape[1] < min_need:
        return float("nan"), f"样本 {z_sub.shape[1]} < {min_need}"
    spec = PortfolioSpec(name="layer", n_hold=n_hold,
                         n_pick=n_hold * n_pick_mult, rebalance="M",
                         factor=factor)
    r = build_long_only(z_sub, spec, cost, px_sub)
    if not r.get("ok"):
        return float("nan"), str(r.get("reason", "回测失败"))
    return float(r["年化收益"]), ""


def size_decile_run(z: pd.DataFrame, price: pd.DataFrame,
                    mcap: pd.DataFrame, windows: list,
                    cost: CostModel, n_hold: int, n_pick_mult: int,
                    factor: str) -> list[dict]:
    """逐窗口 × 逐层跑回测，返回明细行。

    基准口径：**层内自己的等权收益**（不是全市场等权）。
    ⚠️ 用全市场基准比，会把「牛市里大盘涨得多」误读成「因子有效」。
       实测 rev5（修复分层时点后）：大盘层年化中位 +0.43%，
       而层内超额中位 −8.15% ⇒ 判不可用。

    分层时点：**窗口首日** `te[0]`，见模块 docstring 的 BLOCK 说明。

    🔴 BLOCK（review 2026-10-06 抓出）：全市场基准必须**窗口内重算**
       `pct_change`，不能全样本算完再 `.loc[te]`。
       全样本 `pct_change()` 的首行是 `te[0]` 当天的收益，
       而其分母是 `te[0]` 的**前一日收盘价 —— 落在训练期里**。
       策略侧 `build_long_only(p.loc[te], ..., price.loc[te])` 的
       `pct_change()` 首行恒为 NaN，**没有这一天**
       ⇒ 基准比策略多算 1 天，且多的这 1 天来自训练期。
       `run_walk_forward.py:58-69` 已用 60 行注释记录此坑，
       本文件初版把同一个 bug 原样引入 ——
       实测 rev5 全A 股 7 窗口，基准年化中位偏高 **2.15pp**
       （窗口内重算 18.99% vs 全样本切片 21.14%）。
       ⚠️ 只影响 `全市场基准` 这一列；`层内基准` 用的是切片后的 `px_b`，
         本来就对 ⇒ **层内超额不受影响**。
    """
    # ⚠️ 日期轴只取 `z ∩ price`，**不把 mcap 放进交集**。
    # 调用方就是用 `z ∩ price` 生成 windows 的；若这里再 ∩ mcap，
    # mcap 缺日会把交易日从策略轴上抹掉，随后 `z.loc[te]` / `price_f.loc[te]`
    # 抛裸 KeyError（报错是长串 Timestamp，看不出根因是「市值缺日」）。
    # 正确做法：策略轴由 z/price 决定，mcap 单向 reindex 过来 + 前向填充。
    dates = sorted(set(z.index) & set(price.index))
    z, price_f = z.loc[dates], price.loc[dates]
    mcap_f = mcap.sort_index().ffill().reindex(dates)
    if len(mcap.index) and dates[0] < mcap.index[0]:
        print(f"⚠ 窗口起点 {dates[0].date()} 早于市值面板首日 "
              f"{mcap.index[0].date()}，该日用首个可用市值")

    rows: list[dict] = []
    for i, (_, te) in enumerate(windows):
        # 🔴 **窗口首日**，不是末日 —— 见模块 docstring
        if te[0] not in mcap_f.index or mcap_f.loc[te[0]].isna().all():
            rows.append({"因子": factor, "窗口": i, "层": "全",
                         "年化": np.nan, "层内基准": np.nan,
                         "层内超额": np.nan, "全市场基准": np.nan,
                         "备注": f"窗口首日 {te[0].date()} 无市值数据，跳过"})
            continue
        m = mcap_f.loc[te[0]].reindex(z.columns)
        buckets = split_buckets(m)
        # ⚠️ 基准与层内基准同源：**先切片再 pct_change**
        px_te = price_f.loc[te]
        b_all = _bench_stats(
            px_te.pct_change(fill_method=None).mean(axis=1))
        if not buckets:
            rows.append({"因子": factor, "窗口": i, "层": "全",
                         "年化": np.nan, "层内基准": np.nan,
                         "层内超额": np.nan, "全市场基准": b_all,
                         "备注": f"有效市值样本不足 "
                                 f"{MIN_STOCKS_PER_BUCKET * 3} 只"
                                 f"（每层需 ≥{MIN_STOCKS_PER_BUCKET}）"})
            continue
        for label, cols in buckets.items():
            px_b = px_te[cols]
            a, why = run_in_subset(z.loc[te, cols], px_b, cost,
                                   n_hold, n_pick_mult, factor)
            b = _bench_stats(px_b.pct_change(fill_method=None).mean(axis=1))
            rows.append({"因子": factor, "窗口": i, "层": label,
                         "年化": a, "层内基准": b, "层内超额": a - b,
                         "全市场基准": b_all, "备注": why})
    return rows


def interpret_layer_median(med: dict[str, float], threshold: float = 0.05
                           ) -> tuple[str, str]:
    """按层内表现给结论 —— 返回 (等级, 说明)。

    ⚠️ 阈值 5% 而非 0：层内年化超额 +0.3% 在扣掉成本与波动后无意义。
    """
    ok = [med[k] for k in ("大", "中", "小")
          if k in med and pd.notna(med[k])]
    if len(ok) < 2:
        return "⚠", "层数不足，无法判读"
    layer_med = float(np.median(ok))
    detail = "".join(f"\n       {k}盘层 {med[k]:+.2%}"
                     for k in ("大", "中", "小")
                     if k in med and pd.notna(med[k]))
    if layer_med > threshold:
        return "✅", (f"层内中位 {layer_med:+.2%} > +{threshold:.0%} ⇒ "
                      f"**信号真实存在**，与市值无关{detail}")
    if layer_med > 0:
        return "⚠", (f"层内中位 {layer_med:+.2%} 勉强为正但很弱 ⇒ "
                     f"跨层选的收益主要来自市值暴露{detail}")
    return "❌", (f"层内中位 {layer_med:+.2%} ≤ 0 ⇒ **信号无效**，"
                  f"收益全部来自市值暴露{detail}")
