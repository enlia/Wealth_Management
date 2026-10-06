"""复权数据质量验证：确认除权跳空被消除，且不误伤真实涨停。

判据演进（三次推翻，结论已实测固化）
------------------------------------
**v1 用「等权累计收益之差」当偏差** —— 错。那是分红再投资价值：
  茅台 2015-12~2026-09 未复权 +488.7%、前复权 +642.3%，差额 +153.6%
  正是 10.7 年累计分红，两者都不「错」，未复权只是不含分红。

**v2 用固定容差 TOL=1.005 判定超限** —— 也错。涨停价按 0.01 元
  四舍五入，真实涨幅会超名义限幅，且低价股最严重：
    前收 2.66 → 涨停 2.93 = +10.15%
    前收 1.10 → 涨停 1.21 = +10.00%
  固定 10.5% 容差把 sz002426 的 31 条真实封板涨停误判成 27 条污染。

**v3（当前）用价格相关的动态容差** `market_rules.price_tolerance()`，
  逐日按该股前收盘价算出的真实涨停阈值判定。
  预筛必须用 `min_tolerance()`（容差下界）——
  用 `max_tolerance()` 会把落在 (精确容差, 兜底阈值] 区间的真实超限丢弃。

同时用 `close == high`（涨停）/ `close == low`（跌停）做交叉验证：
真实封板时收盘必然等于最高/最低价，而复权失败造成的跳空不会。两者独立，能互为证据。

⚠️ 预筛必须用 `min_tolerance()`（容差下界），不能用 `max_tolerance()`。
   用兜底阈值筛「严标准要的东西」会把真实超限静默丢弃 ——
   实测 800 只样本丢 85 条（占精确判定的 10.65%）。

为什么必须实测而非推断
----------------------
复权公式简单，但有多个可能出错的地方：
  · 因子日期错位（本项目已踩：Tushare adj_factor 是倒序的，
    不排序会让归一化系数错 1.26倍，累计收益从 33% 虚增到 125%）
  · 归一化方向反了（前复权写成前复权）
  · 拼接时把 NaN 当 0（会让整段价格变 0）
  · 诊断脚本 ffill 未按 code 分组（会让因子跨股票对齐）

任何一个出错都会让偏差**变大或方向相反**，所以只能实测。

判据
----
  复权后「非真实涨停」的跳空数 = 0

用法
----
  uv run python research/scripts/verify_adjusted_data.py
  uv run python research/scripts/verify_adjusted_data.py --samples 2000
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from factor_lab.config import DB_PATH, is_a_share  # noqa: E402
from factor_lab.data import all_codes  # noqa: E402
from factor_lab.data.adjust import AdjustedPriceBuilder  # noqa: E402
from factor_lab.market_rules import (  # noqa: E402
    board_cn,
    board_of,
    limit_of,
    min_tolerance,
    no_limit_days,
    price_tolerance,
)

TICK = 0.01


def pick_codes(n_total: int, seed: int) -> list[str]:
    """分层抽样：按板块比例抽，避免 bj 排最前时全抽北交所（见 PITFALLS）。

    ⚠️ 必须用 `is_a_share()` 过滤，不能用 SQL 的 `code LIKE 'sz3%'`。
       sz3% 会把指数（sz399003、sz399356 等）一起抽进来 ——
       它们的 adj_factor 不在 Tushare 因子表里。
       实测 149 只样本里有 1 只指数，它本次恰好因「无复权因子」
       被 last_missing 剔除而侥幸无害 —— 那是巧合，不是设计。
    """
    universe = [c for c in all_codes() if is_a_share(c)]
    by_board: dict[str, list[str]] = {}
    for c in universe:
        by_board.setdefault(board_of(c), []).append(c)

    rng = np.random.default_rng(seed)
    picked: list[str] = []
    for _b, lst in sorted(by_board.items()):
        n = max(1, round(n_total * len(lst) / len(universe)))
        picked += [lst[i] for i in rng.choice(len(lst), size=min(n, len(lst)), replace=False)]
    return sorted(picked)


def scan(price: pd.DataFrame, high_raw: pd.DataFrame, low_raw: pd.DataFrame,
         close_raw: pd.DataFrame, label: str) -> dict:
    """逐股找超限跳空，并按「是否真实封板 / 新股窗口」分类。

    price:     待检价格面板（索引=日期 int32，列=代码）
    high_raw:  原始未复权最高价面板，用于 close==high（涨停）交叉验证
    low_raw:   原始未复权最低价面板，用于 close==low（跌停）交叉验证
    close_raw: 原始未复权收盘价面板（与上面同形状，便于逐列比较）

    ⚠️ 性能：不能用 panel.at[(date, (code, "high"))]，那是 O(n) 的
       MultiIndex 查找，2611 日 × 800 股会跑到十几分钟。
       改成预先按列取numpy 数组，循环体内只做 O(1) 索引。
    """
    ret = price.pct_change(fill_method=None)
    n_tot = int(ret.notna().sum().sum())
    rows = []
    for c in price.columns:
        s = price[c].to_numpy(dtype=float)
        r = ret[c].to_numpy(dtype=float)
        hi = high_raw[c].to_numpy(dtype=float)
        lo_raw = low_raw[c].to_numpy(dtype=float)
        cl = close_raw[c].to_numpy(dtype=float)
        lim = limit_of(c)
        # ⚠️ 预筛必须用 **min_tolerance**（容差下界），不能用 max_tolerance。
        #    max_tolerance 比逐日精确容差更宽，用它筛选等于
        #    「用宽标准挑严标准要的东西」——
        #    落在 (精确容差, 兜底阈值] 区间的真实超限记录会被静默丢弃。
        #    实测踩过：sh600000 2016-06-23 跌 -12.13%，
        #    精确容差 10.10% 应判超限，兜底 13.46% 把它筛掉了；
        #    800 只样本因此丢了 85 条（占精确判定的 10.65%）。
        #    预筛的下界语义是「任何价位都不可能超限」，故取最小容差。
        lo = min_tolerance(lim)
        over = np.where(np.abs(r) > lo)[0]
        if len(over) == 0:
            continue
        # ⚠️ 交易日序号必须按**实际有行情的天数**累计，不能用面板行号 i。
        #    pivot 出来的是矩形面板（停牌处为 NaN），行号 ≠ 交易日序号。
        #    实测踩过：sz300873 上市仅3 个交易日，面板行号却是 1132，
        #    导致新股无涨跌幅限制窗口完全失效。
        seq = np.cumsum(np.isfinite(s))
        finite = np.isfinite(s)
        # 首个行情日：首个非 NaN 的位置对应 price.index[该位置]
        first_date = (int(price.index[int(np.argmax(finite))])
                      if finite.any() else None)
        for i in over:
            if i == 0 or not np.isfinite(s[i - 1]) or s[i - 1] <= 0:
                continue
            prev = float(s[i - 1])
            close = float(s[i])
            tol = price_tolerance(prev, lim)
            # 真实封板：原始收盘 == 原始最高（涨停）或 == 原始最低（跌停）。
            # ⚠️ 只判涨停会漏掉跌停：跌停日 close==low 而非 close==high。
            #    实测 150 只样本里 343 条「真污染」绝大多数是真实跌停
            #    （sz002076 1.37→1.23 正是 round(1.37×0.9, 2)=1.23 的跌停价）。
            #    涨跌停取整方向相反：涨停向上、跌停向下，
            #    所以跌停幅度**总小于**price_tolerance（按涨停算的上界），判据安全。
            closed_high = bool(np.isfinite(hi[i]) and np.isfinite(cl[i])
                               and abs(hi[i] - cl[i]) < TICK)
            closed_low = bool(np.isfinite(lo_raw[i]) and np.isfinite(cl[i])
                              and abs(lo_raw[i] - cl[i]) < TICK)
            sealed = closed_high or closed_low
            td = int(seq[i])
            rows.append({
                "code": c, "board": board_of(c),
                "date": int(price.index[i]),
                "ret": float(r[i]), "tol": tol,
                "sealed": sealed, "prev": prev, "close": close,
                "trade_day": td,
                "no_limit": no_limit_days(c, td, first_date),
            })
    df = pd.DataFrame(rows)
    if df is None or df.empty:
        return {"label": label, "样本数": n_tot, "超限数": 0,
                "真实封板": 0, "新股窗口": 0, "真污染": 0, "污染率%": 0.0,
                "_df": df}
    true_bad = df[~df["sealed"] & ~df["no_limit"]]
    return {
        "label": label,
        "样本数": n_tot,
        "超限数": len(df),
        "真实封板": int(df["sealed"].sum()),
        "新股窗口": int(df["no_limit"].sum()),
        "真污染": len(true_bad),
        "污染率%": len(true_bad) / max(n_tot, 1) * 100,
        "_df": df,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=800, help="抽样股票数")
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20260930")
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    builder = AdjustedPriceBuilder()

    codes = pick_codes(args.samples, args.seed)
    dist = {board_cn(c): 0 for c in codes}
    for c in codes:
        dist[board_cn(c)] += 1
    print(f"抽样 {len(codes):,} 只  分层 {dist}")

    q = f"""
    SELECT date, code, close, high, low FROM bar_daily
     WHERE date BETWEEN ? AND ? AND code IN ({','.join('?' * len(codes))})
    """
    raw = pd.read_sql(q, con, params=[args.start, args.end, *codes])
    price_raw = raw.pivot(index="date", columns="code", values="close").sort_index()
    high_raw = raw.pivot(index="date", columns="code", values="high").sort_index()
    low_raw = raw.pivot(index="date", columns="code", values="low").sort_index()
    print(f"行情面板 {price_raw.shape[0]:,} 日 × {price_raw.shape[1]:,} 只")

    price_adj = builder.adjust_frame(price_raw, "close")
    missing = getattr(builder, "last_missing", [])
    if missing:
        print(f"剔除 {len(missing)} 只无复权因子：{missing[:6]}"
              f"{'...' if len(missing) > 6 else ''}")
        keep = [c for c in price_adj.columns if c not in missing]
        price_raw, price_adj = price_raw[keep], price_adj[keep]
        high_raw, low_raw = high_raw[keep], low_raw[keep]

    r_raw = scan(price_raw, high_raw, low_raw, price_raw, "未复权")
    r_adj = scan(price_adj, high_raw, low_raw, price_raw, "前复权")

    print()
    print("=" * 78)
    print("复权前后对比（动态容差 + 封板交叉验证）")
    print("=" * 78)
    print(f"{'指标':<16}{'未复权':>16}{'前复权':>16}{'变化':>16}")
    print("-" * 78)
    for k in ("超限数", "真实封板", "新股窗口", "真污染", "污染率%"):
        a, b = r_raw[k], r_adj[k]
        print(f"{k:<16}{a:>16.3f}{b:>16.3f}{b - a:>+16.3f}")
    print()
    print("  超限数 = 涨跌幅超过该股当日真实涨停阈值的记录数")
    print("  真实封板 = close==high（涨停）或 close==low（跌停），是真行情")
    print("  新股窗口 = 上市前 5 个交易日（注册制无涨跌幅限制）")
    print("  真污染 = 超限且两者都不是 = 复权失败")

    ew_raw = price_raw.pct_change(fill_method=None).mean(axis=1)
    ew_adj = price_adj.pct_change(fill_method=None).mean(axis=1)
    c_raw = float((1 + ew_raw.fillna(0)).prod() - 1)
    c_adj = float((1 + ew_adj.fillna(0)).prod() - 1)
    print()
    print(f"{'等权累计%':<16}{c_raw * 100:>16.2f}{c_adj * 100:>16.2f}"
          f"{(c_adj - c_raw) * 100:>+16.2f}")
    print("  （差额 = 分红再投资价值，不是偏差）")

    df = r_adj.get("_df")
    ok = r_adj["真污染"] == 0
    print()
    print("=" * 78)
    if ok:
        print(f"✓ 复权后真污染 0 条（{r_adj['超限数']} 条超限已全部归因为"
              f"真实封板 {r_adj['真实封板']} / 新股窗口 {r_adj['新股窗口']}）")
        print(f"  未复权时真污染 {r_raw['真污染']} 条 → 100% 消除")
        print("  复权价可用于回测与因子计算")
    else:
        print(f"✗ 复权后仍有 {r_adj['真污染']} 条真污染，需排查")
        bad = df[~df["sealed"] & ~df["no_limit"]]
        print(bad.groupby("board").size().to_string())
        print(bad.sort_values("ret", key=abs, ascending=False).head(20).to_string(index=False))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
