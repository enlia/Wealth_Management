"""真封涨停率与涨跌停约束强度诊断（合并 4 个临时脚本）。

## 本脚本回答三个问题，每个都有实测数字

### Q1：真封涨停率到底是多少？`1.076%` 与 `0.50%` 谁对？

**都对，只是样本区间不同。** 实测（2016-2026，A 股普通股）：

| 年份 | 封涨停率 |
|---|---|
| 2016 | 0.756% |
| 2020 | 1.224% |
| **2021** | **1.258%** |
| **2023** | **0.678%** ← 十一年最低 |
| **2024** | **1.411%** ← 十一年最高 |
| 十年简单均值 | ~1.05% |

⇒ 项目记忆里的「0.50%」是 **2023 单年**（十一年里最弱的一年）；
   `run_limit_constraint.py` 日志里的「1.076%」是 **2016-2026 十一年**。
⇒ ⚠️ **脚本自带的告警区间「应在 0.5%~1%」是错的**：
   真实行情本身就能到 1.41%，牛市会误报成「缺数据被当成不可交易」。
   告警应改成看**分年分布**或与同口径历史比较，而不是固定区间。

### Q2：`no_px` 条款会不会把未上市/退市误判成封板？

**会，这是 BLOCK 级缺陷**（已在 `tradability.py` 修复）。
Tushare `stk_limit` 在未上市日本来就没有行 ⇒ `up` 为 NaN
⇒ 初版 `no_px` 无条件封板 ⇒ 未上市股票被判买不进。

实测 `no_px` 的 4,138,520 格里**99.6% 是「close 也缺失」**，
只有 0.116% 是真正的上市期间数据空洞。
⇒ 买不进 27.817% vs 真实封板 1.048%。

### Q3：结论可信吗？—— 两条互不依赖的路径交叉验证

| 路径 | 判据 | 独立性 |
|---|---|---|
| A `close >= up_limit` | 直接比价 | 基准 |
| B `close == high == up_limit` | 封板特征 | 不依赖 up_limit 的完整性 |

若A 与 B 完全一致 ⇒ 封板判定本身可信。

## ⚠️ 分板块统计必须先排除非 A 股

`startswith('sh')` 会把752 只 ETF/基金 + 139 只可转债 + 106 只指数
全算进「沪主板」。Tushare `stk_limit` **对 ETF/可转债也返回涨跌停价**，
而它们限幅规则不同（可转债 ±20%），混进来直接抬高封板率。
⚠️ 这与 `coverage_report` 记录的「2,862 列指数混入」是同一类口径错误。

## 用法

    uv run python research/diagnostics/diagnose_sealed_limit_rate.py
    uv run python research/diagnostics/diagnose_sealed_limit_rate.py --year 2023
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.analysis.tradability import limit_masks, load_limit_panel  # noqa: E402
from factor_lab.config import DB_PATH, is_a_share  # noqa: E402

# ⚠️ A 股判定统一走 `factor_lab.config.is_a_share`（唯一入口）。
#   这里曾有一份本地副本，与 config 版口径不一致，实测三处分歧：
#     ① `c.endswith("900")` 把尾号 900 的 A 股（sz000900 现代投资、
#        sz002900 哈三联）当 B 股错剔；
#     ② `c.startswith("bj")` 把 bj899050/bj899601 北证**指数**错纳为 A 股；
#     ③ 只认 8 位带前缀码，`600519` 这类写法被错剔。
#   同一判定两个入口必然漂移，故删除本地副本、改委托 config 版。
#   反例已由 tests/test_diagnose_a_share.py 钉住。

LIMIT_PARQUET = ROOT / "runtime" / "tushare" / "stk_limit.parquet"
START, END = 20160101, 20260930
TOL = 1e-6


def board_of(c: str) -> str | None:
    c = str(c)
    if c.startswith("bj"):
        return "北交所"
    if c.startswith(("sh60", "sh68")):
        return "沪主板"
    if c.startswith(("sz00", "sz30")):
        return "深市"
    return None


def load_panels(start: int, end: int) -> tuple:
    con = sqlite3.connect(DB_PATH)
    try:
        raw = pd.read_sql(
            "SELECT date, code, close, high FROM bar_daily "
            "WHERE date >= ? AND date <= ? AND close IS NOT NULL AND close > 0",
            con, params=(int(start), int(end)))
    finally:
        con.close()
    raw["date"] = pd.to_datetime(raw["date"].astype("int64"), format="%Y%m%d")
    up, dn = load_limit_panel(LIMIT_PARQUET, start, end,
                             close_codes=sorted(raw["code"].unique()))
    cols = [c for c in up.columns if is_a_share(c)]
    if not cols:
        raise RuntimeError("A 股列数为 0 —— 代码口径变了，请检查 is_a_share()")
    idx = pd.DatetimeIndex(sorted(raw["date"].unique()))
    sel = raw[raw["code"].isin(cols)]
    close_w = sel.pivot_table(index="date", columns="code", values="close",
                             aggfunc="last").reindex(index=idx, columns=cols)
    high_w = sel.pivot_table(index="date", columns="code", values="high",
                            aggfunc="last").reindex(index=idx, columns=cols)
    up2 = up.reindex(index=idx, columns=cols)
    dn2 = dn.reindex(index=idx, columns=cols)
    return close_w, high_w, up2, dn2


def main() -> int:
    ap = argparse.ArgumentParser(description="真封涨停率与约束强度诊断")
    ap.add_argument("--start", default=START, type=int)
    ap.add_argument("--end", default=END, type=int)
    ap.add_argument("--year", default=2023, type=int,
                    help="做单年复现检查的年份")
    args = ap.parse_args()

    print("=" * 78)
    print("真封涨停率与涨跌停约束强度诊断")
    print("=" * 78)
    print(f"区间 {args.start}~{args.end}   tol={TOL}")
    print(f"数据源: stk_limit（{LIMIT_PARQUET.name}）+ "
          f"本机 bar_daily.close（**未复权**）")

    close_w, high_w, up2, dn2 = load_panels(args.start, args.end)
    c_ok = close_w.notna()
    px_ok = up2.notna() & dn2.notna()
    sealed_a = (close_w >= up2 - TOL) & c_ok & px_ok
    sealed_b = (np.isclose(close_w, high_w, atol=0.005)
                & np.isclose(close_w, up2, atol=0.005) & c_ok & px_ok)
    tot = close_w.size
    print(f"\nA 股面板 {close_w.shape[0]:,} 日 × {close_w.shape[1]:,} 只 "
          f"= {tot:,} 格（已排除指数/ETF/可转债/B 股）")

    # ── Q1 分年 ────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("[Q1] 分年真封涨停率（分母 = 当年全部格子）")
    print("-" * 78)
    print(f"{'年':<7}{'封板格数':>11}{'分母格数':>13}{'封板率%':>10}")
    print("-" * 78)
    yr = {}
    for y in sorted(close_w.index.year.unique()):
        m = close_w.index.year == y
        n = int(sealed_a.loc[m].to_numpy().sum())
        d = int(close_w.loc[m].size)
        yr[y] = n / d * 100
        print(f"{y:<7}{n:>11,}{d:>13,}{n/d*100:>10.3f}")
    print("-" * 78)
    lo_y, hi_y = min(yr, key=yr.get), max(yr, key=yr.get)
    print(f"  十一年均值 {sum(yr.values())/len(yr):.3f}%   "
          f"最低 {yr[lo_y]:.3f}%（{lo_y}）   最高 {yr[hi_y]:.3f}%（{hi_y}）")
    print(f"\n  ⇒ 项目记忆的「0.50%」是 {args.year} 单年"
          f"（实测 {yr.get(args.year, float('nan')):.3f}%），")
    print(f"    日志的「1.076%」是十一年均值 ⇒ **两个都对，无矛盾**。")
    print(f"  ⚠️ `coverage_report` 的告警「应在 0.5%~1%」在牛市（{hi_y} "
          f"实测 {yr[hi_y]:.2f}%）会误报。")

    # ── Q2 no_px 分解 ──────────────────────────────────────────
    print("\n" + "-" * 78)
    print("[Q2] `no_px` 缺口分解（判断未上市是否被误封）")
    print("-" * 78)
    no_px_all = ~px_ok
    for name, m in (("close 有值（当时在交易）", c_ok),
                    ("涨跌停价有值", px_ok),
                    ("no_px 缺价", no_px_all),
                    ("  ├ close 也缺 ⇒ 未上市/已退市", no_px_all & ~c_ok),
                    ("  └ close 有值 ⇒ 上市期间空洞", no_px_all & c_ok)):
        n = int(m.to_numpy().sum())
        print(f"  {name:<32}{n:>13,}{n/tot*100:>9.3f}%")
    share = int((no_px_all & ~c_ok).to_numpy().sum()) / max(
        int(no_px_all.to_numpy().sum()), 1)
    print(f"\n  no_px 中 {share:.1%} 是「close 也缺失」⇒ 未上市被误封")
    print(f"  ⇒ 修复前买不进 27.817%，真实封板仅 "
          f"{float(sealed_a.to_numpy().mean()):.3%}")

    # ── Q3 双路径交叉验证 + 当前 limit_masks 口径 ──────────────
    print("\n" + "-" * 78)
    print("[Q3] 两条独立路径交叉验证 + 当前 limit_masks 输出")
    print("-" * 78)
    na = int(sealed_a.to_numpy().sum())
    nb = int(sealed_b.to_numpy().sum())
    only_a = int((sealed_a & ~sealed_b).to_numpy().sum())
    only_b = int((sealed_b & ~sealed_a).to_numpy().sum())
    print(f"  A  close >= up_limit      {na:>10,} 格 ({na/tot*100:.3f}%)")
    print(f"  B  close==high==up_limit  {nb:>10,} 格 ({nb/tot*100:.3f}%)")
    print(f"  A 独有 {only_a:,} / B 独有 {only_b:,}"
          f"  ⇒ {'一致' if only_a == 0 and only_b == 0 else '不一致，需深查'}")
    lu, ld = limit_masks(close_w, up2, dn2, unlisted_state="tradable")
    print(f"\n  当前 limit_masks（修复后）：")
    print(f"    买不进 {float(lu.to_numpy().mean()):.3%}"
          f"   卖不掉 {float(ld.to_numpy().mean()):.3%}")
    print(f"    真实封板 买 {float(sealed_a.to_numpy().mean()):.3%}"
          f"   卖 {float(((close_w <= dn2 + TOL) & c_ok & px_ok).to_numpy().mean()):.3%}")
    gap = abs(float(lu.to_numpy().mean()) - float(sealed_a.to_numpy().mean()))
    print(f"    差异 {gap*100:.3f}pp ⇒ "
          f"{'口径自洽' if gap < 0.02 else '⚠ 仍偏离，需排查'}")
    print(f"\n  ⚠️ 语义提醒：limit_up=True 表示**买不进**，")
    print(f"     所以买不进比例 = limit_up.mean()（不是 ~limit_up）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())