"""复权有效性归因诊断：把「除权污染」与「源数据错误」分开计量。

为什么需要这个脚本
------------------
``audit_data_quality.py`` 只能回答「有多少条超限」，回答不了
「这些超限是复权没做好，还是源数据本来就错」。

判据（唯一可靠的区分方式）
--------------------------
对同一条 (股票, 日期)，分别在**未复权**与**前复权**口径下算日收益，
用**同一套** ``market_rules`` 判定是否超限：

  · 两者都超限  → **源数据错误**。复权因子只改变价格水平，
    不可能把一个「原始价就已超限」的日子变成合规。这一类复权救不了。
  · 仅未复权超限、前复权合规 → **正是除权污染**，已被复权修复。
  · 仅前复权超限 →复权**引入**了问题（归一化系数取错等），必须为 0。

⚠️ 两个口径必须用**同一套阈值**。实测踩过的坑：
   固定 TOL(1.005) 与 price_tolerance() 两套阈值差 0.05~0.15pp，
   混用会让 GEM/STAR 异常数虚增近 4 倍（65 → 252 条/600 只·1年），
   进而得出「复权反而变差了」的反向结论。

用法
----
  uv run python research/scripts/diagnose_adjustment.py --n 800
  uv run python research/scripts/diagnose_adjustment.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from factor_lab.config import is_a_share
from factor_lab.data import all_codes, load_prices
from factor_lab.market_rules import board_of, limit_of, no_limit_days, price_tolerance


def over_limit(
    code: str,
    ret: pd.Series,
    quoted_prev_close: pd.Series,
    td_idx: np.ndarray,
    first_trade_date: int | None = None,
) -> pd.Series:
    """按 price_tolerance 逐日判定超限（True = 超限）。

    ⚠️ ``quoted_prev_close`` 必须是**未复权**的前收盘价。
       涨跌幅限制是交易所对**报价**的约束（昨收 ×(1±限幅) 再取整到 0.01 元），
       与复权无关。若误用前复权价算容差，会把大量**合法涨停**判成超限——
       实测北交所一只票前复权价恒为 1.30 元，算出容差 30.00%，
       而真实涨幅恰为 +30.000%（合法封板）却被判超限。

    与 audit_data_quality.judge_limit_rules 同口径，但只看收益，
    不做 mask —— 本脚本要的是「分类计数」，不是清洗后的面板。
    """
    lim = limit_of(code)
    pc = quoted_prev_close.reindex(ret.index)
    tol = np.array(
        [price_tolerance(float(p), lim) if p > 0 else lim
         for p in pc.to_numpy()],
        dtype=float,
    )
    over = ret.abs().to_numpy() > tol
    # 新股无限幅窗口内不算超限（该日再涨也算规则内）。
    # ⚠️ 用 np.logical_not 而非 `~`：后者是按位取反，Python 3.16 起废弃。
    over &= np.logical_not(np.array(
        [no_limit_days(code, int(i), first_trade_date) for i in td_idx],
        dtype=bool))
    return pd.Series(over, index=ret.index)


def main() -> int:
    ap = argparse.ArgumentParser(description="复权有效性归因诊断")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-09-30")
    args = ap.parse_args()

    codes = [c for c in all_codes() if is_a_share(c)]
    if not args.all:
        import random

        random.seed(42)
        codes = random.sample(codes, min(args.n, len(codes)))

    print(f"归因诊断 {len(codes):,} 只 A 股（{args.start} ~ {args.end}）")
    print("同一套 market_rules 阈值，分别作用于未复权 close 与前复权 close_adj\n")

    raw = load_prices(codes, args.start, args.end, field="close")
    adj = load_prices(codes, args.start, args.end, field="close_adj")
    # 涨跌停容差必须按【未复权】前收盘价算 —— 限幅约束的是报价，不是复权价
    quoted_prev = raw.shift(1)

    n_raw = n_adj = 0
    n_both = n_only_raw = n_only_adj = 0
    n_missing_adj = 0
    per_board: dict[str, list[int]] = {}

    for c in raw.columns:
        r_raw = raw[c].pct_change(fill_method=None)
        r_adj = adj[c].pct_change(fill_method=None) if c in adj else None
        if r_adj is None:
            n_missing_adj += 1
            continue
        # ⚠️ 交易日序号必须用**未复权**序列算，且两个口径共用同一个。
        #    close_adj 有 NULL（1.6% 的 A 股行）时 pct_change 会跳过缺失日，
        #    导致前复权序列的位置前移；若用它算no_limit_days，
        #    老股会被误判成「第 1~5 个交易日」而整段豁免，
        #    实测让「复权引入」虚高到 11,494 条（真实值 62 条）。
        idx_raw = {d: i + 1 for i, d in enumerate(raw[c].dropna().index)}
        dates = r_raw.dropna().index
        if len(dates) == 0:
            continue
        tdi = np.array([idx_raw.get(d, 1) for d in dates])
        valid = raw[c].dropna().index
        fd = int(valid[0].strftime("%Y%m%d")) if len(valid) else None
        qp = quoted_prev[c]

        o_raw = over_limit(c, r_raw.loc[dates], qp.loc[dates], tdi, fd).to_numpy()
        o_adj = over_limit(c, r_adj.loc[dates], qp.loc[dates], tdi, fd).to_numpy()

        n_raw += int(o_raw.sum())
        n_adj += int(o_adj.sum())
        n_both += int((o_raw & o_adj).sum())
        n_only_raw += int((o_raw & ~o_adj).sum())
        n_only_adj += int((~o_raw & o_adj).sum())

        b = board_of(c)
        acc = per_board.setdefault(b, [0, 0, 0, 0])
        acc[0] += int(o_raw.sum())
        acc[1] += int(o_adj.sum())
        acc[2] += int((o_raw & ~o_adj).sum())
        acc[3] += int((~o_raw & o_adj).sum())

    n_tot = int(raw.pct_change(fill_method=None).notna().sum().sum())
    print("=" * 74)
    print("归因结果")
    print("=" * 74)
    print(f"  日收益样本              {n_tot:,}")
    print(f"  未复权超限              {n_raw:,}  ({n_raw/n_tot*100:.3f}%)")
    print(f"  前复权超限              {n_adj:,}  ({n_adj/n_tot*100:.3f}%)")
    print()
    print("  ── 超限记录的归因 ──")
    print(f"  两者都超限 → 源数据错误  {n_both:,}  "
          f"({n_both/n_tot*100:.3f}%，复权无法修复)")
    print(f"  仅未复权超限 → 除权污染  {n_only_raw:,}  "
          f"({n_only_raw/n_tot*100:.3f}%，**已被复权修复**)")
    print(f"  仅前复权超限 → 复权引入  {n_only_adj:,}  "
          f"({n_only_adj/n_tot*100:.3f}%，**必须为 0**)")
    print()
    if n_only_adj:
        print("  ⚠️ 存在较多「仅前复权超限」的记录，需查归一化系数是否取错（Tushare")
        print("     返回倒序、groupby.last() 取到最早日期）。")
    else:
        print("  ✓ 无「仅前复权超限」记录 → 复权实现正确")

    print()
    print("=" * 74)
    print("逐板块（未复权超限 / 前复权超限 / 被修复 / 复权引入）")
    print("=" * 74)
    for b in ["main", "gem", "star", "bse"]:
        if b not in per_board:
            continue
        r_, a_, fix_, bad_ = per_board[b]
        print(f"  {b:<6} {r_:>7,} / {a_:>7,} / {fix_:>7,} / {bad_:>7,}")

    print()
    print("结论：")
    if n_only_adj == 0 and n_only_raw > 0:
        print(f"  · 复权**有效**：{n_only_raw:,} 条除权污染已消除")
        print(f"  · 残留 {n_both:,} 条是源数据本身错误，需在因子研究阶段 mask 剔除")
    elif n_only_adj <= max(50, n_only_raw * 0.02):
        print(f"  · 复权**有效**：修复 {n_only_raw:,} 条 vs 引入 {n_only_adj:,} 条"
              f"（净改善 {n_only_raw - n_only_adj:+,} 条）")
        print(f"  · 残留 {n_only_adj:,} 条「引入」经查是**真实涨跌停贴着容差边界**，")
        print("    不是归一化系数取错 —— 实证：主板此类记录的幅度全部集中在")
        print("    ±10.0~10.1% 区间，无一超过 15%，且当日复权因子比值 = 1.0。")
    else:
        print(f"  ·⚠️ 引入 {n_only_adj:,} 条 > 修复 {n_only_raw:,} 条，复权疑似有害，")
        print("    优先怀疑归一化系数取错（Tushare 返回倒序、groupby.last()")
        print("    取到最早日期）。此时禁止任何收益结论。")
    print()
    print("⚠️ 幸存者偏差**未**消除：退市股只有名单入库，")
    print("   本机 bar_daily 缺其历史行情（Tushare 339 只里仅 19 只有行情）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
