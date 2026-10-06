"""数据质量审计：量化研究的前置关卡（务必在任何回测前跑一遍）

本项目实测发现（2026-10-05）
--------------------------------
通达信本机日线数据存在**部分未复权**的除权跳空：
  · A 股 5,593 只中，主板有 3,033 条 |日收益| > 10.5% 的记录（限制是 ±10%）
  · 涉及 1,744 只主板股票
  · 典型案例：平安银行 2016-06-16 单日 −17.91%（当日开盘直接跳空 −17.91%，
    全天几乎无波动 —— 这是 10 送 X 除权，不是真实行情）
  · 宁德时代 2023-04-26 单日 −41.82%（创业板 20% 限，−41% 必然是除权）

影响量级（800 只抽样，2026-10-06 实测复核）：
  · 异常日占全部日收益 0.240%，贡献 1.4% 的绝对波动
  · 等权买入持有 10 年累计：原始 +29.5% → 清洗后 **+53.6%**
  · **年化偏差 −2.332pp/年** —— 足以推翻任何收益结论

⚠️ 踩坑记录：本文档曾写「年化偏差 3.35pp、修正后 −3.5%」，
  那是**早期 588 只样本 + 误伤次新股**时的错误数字，
  2026-10-06 重跑已修正为 −2.332pp / +53.6%。
  → **教训：docstring 里的实测数字必须与代码实算结果对齐，
  否则文档会变成误导来源。改统计口径后必须同步更新。**

⚠️ 关键区分（否则会误伤）
  · **次新股上市首日**：无涨跌幅限制，涨 200% 也合法 → 不是数据错误
  · **老股除权日**：必然是未复权 → 是数据错误
  判别方法：看是否发生在该股票序列的前 2 日内。
  实测：当前口径下上市首日误判数为 0（588 只异常股票中曾有 28 只被误判）。
  实测：588 只异常股票中只有 28 只（5%）是次新股首日，
  其余 95% 都是真实的未复权污染。

本模块提供
----------
  audit_price_data()      价格数据体检，输出污染规模与影响量级
  clean_returns()         清洗收益率：剔除上市首日 + 标记未复权除权日
  judge_limit_rules()     按 market_rules 判定超限（价格感知容差 + 新股窗口）
  classify_residuals()    残留超限按形态三分类计数（成因不硬判）

判定「是否修好了」的口径（重要）
--------------------------------
`annual_drag` 是用**同一批价格**分别算「原始」与「剔除超限日」得到的差，
它衡量的是**污染的潜在影响量级**，而不是「复权修好了没有」。

  · 对**未复权**价格：annual_drag ≈ −2.365pp/年（污染确实存在）
  · 对**前复权**价格：超限记录应降到 ~0，annual_drag 也应趋近 0

所以审计必须能分别读两个字段。本脚本用 ``--field`` 控制：
  --field close      未复权（默认，体检用）
  --field close_adj  前复权（修好后回归验证用）

⚠️ 若`--field close_adj` 的异常数与`--field close` 几乎一样，
说明并库没生效，不要通过调大容差来「修复」。

用法
----
  uv run python research/scripts/audit_data_quality.py --n 800
  uv run python research/scripts/audit_data_quality.py --all
  uv run python research/scripts/audit_data_quality.py --all --field close_adj
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from factor_lab.config import YEAR_TRADING_DAYS, is_a_share
from factor_lab.data import all_codes, load_prices
from factor_lab.market_rules import (
    board_of,
    limit_of,
    no_limit_days,
    price_tolerance,
)

# ⚠️ 板块判定、限幅表与容差**只有 market_rules 一份实现**（统一入口）。
#    本脚本曾内置第二套 board_of / LIMIT / 固定容差 TOL=1.005：
#    与 market_rules 双实现并存必然漂移 —— 本地 board_of 按前缀切片把
#    sh689009（科创板 CDR）判成主板 ±10%，固定容差把低价股真实封板
#    误判成超限（Q2 实测误判率 87%），均已证伪。此处统一引用 market_rules。


def judge_limit_rules(
    code: str, ret: pd.Series, quoted_prev_close: pd.Series,
    trade_day_idx: pd.Series, first_trade_date: int | None = None,
) -> pd.Series:
    """按 ``market_rules`` 判定超限，比固定 TOL 口径更准。

    三处改进：
      1. **新股无涨跌幅窗口**：注册制新股上市前 5 日不设限，
         这些日子动辄 ±40%，按 ±10% 判定会全部误报。
      2. **容差按未复权前收盘价动态算**（``price_tolerance``）：
         交易所涨停价 = 前收 × (1+限幅) 再向上取整到 0.01 元，
         真实涨停幅度会**超过**名义限幅，低价股最严重。
      3. **跨缺失日不算单日涨跌**：close_adj 有 NULL 时 pct_change 会跨日，
         把两天的涨幅算成一天的超限。

    ⚠️⚠️ ``quoted_prev_close`` 必须是**未复权**前收盘价，且**不能**用
       ``min(lim*TOL, …)`` 这类近似代替price_tolerance。

       涨跌幅限制是交易所对**报价**的约束，与复权无关。
       实测：若用前复权价算容差，北交所某票前复权价恒为 1.30 元，
       容差被算成 30.00%，而其真实涨幅恰为 +30.000%（合法封板）却被判超限；
       创业板/科创板异常数虚增近 4 倍（65 → 252 条/600 只·1 年），
       会得出「复权反而使污染变多 4 倍」的反向结论。

    返回与 ``ret`` 同索引的布尔 Series（True = 超限）。
    """
    lim = limit_of(code)
    pc = quoted_prev_close.reindex(ret.index)
    # 前收盘无效时退化为名义限幅（下方 ok &= pc > 0 会把该日排除）
    tol = np.array(
        [price_tolerance(float(p), lim) if p > 0 else lim
         for p in pc.to_numpy()],
        dtype=float,
    )
    # ⚠️ 三个条件是「与」关系：超限 = 涨幅超阈值 且 不在无限幅窗口 且 前收有效。
    #
    #    实测踩过的坑：写成 `ok = within_tol & ~in_window` 再 `return ~ok`，
    #    语义反了 —— 窗口内的日子 ok=False，取反后被判成超限，
    #    200 只样本里 13 条「规则内超限」有 11 条是窗口内日子被误计。
    #    排除必须作用在「超限」上，不能靠二次取反。
    over = np.abs(ret.to_numpy()) > tol
    # 新股无限幅窗口内不判超限（该日再涨也算规则内）。
    # ⚠️ 必须传 first_trade_date：2023-02-17 前上市的主板新股
    #    不适用「前 5 日无限幅」，漏传会把窗口多算 4 天。
    # ⚠️ 用 np.logical_not 而不是 `~`：后者是按位取反，Python 3.16 起废弃。
    in_window = np.array([no_limit_days(code, int(i), first_trade_date)
                          for i in trade_day_idx], dtype=bool)
    over &= np.logical_not(in_window)
    # 前收盘无效（非正或NaN）时无法判定，不计入异常
    over &= (pc > 0).to_numpy()
    return pd.Series(over, index=ret.index)


def _trade_day_index(
    timeline: pd.DataFrame, prices: pd.DataFrame, code: str, index: pd.Index,
) -> tuple[pd.Series, int | None]:
    """按【未复权】时间线算交易日序号（1 起）与首个行情日（YYYYMMDD）。

    ⚠️ 序号与首个行情日必须取自未复权时间线（timeline），不能用被审字段
       自身：它有 NULL 时位置整体错位，会把新股无限幅窗口判到错误的日期上，
       实测让异常数从 112 条虚高到 11,183 条（100 倍）。
    """
    src = timeline if code in timeline else prices
    valid = src[code].dropna().index
    pos = {d: i + 1 for i, d in enumerate(valid)}
    tdi = pd.Series([pos.get(d, 1) for d in index], index=index)
    # 首个行情日（YYYYMMDD），用于判断是否适用注册制新股 5 日窗口
    fd = int(valid[0].strftime("%Y%m%d")) if len(valid) else None
    return tdi, fd


def clean_returns(
    prices: pd.DataFrame,
    quoted_prev: pd.DataFrame | None = None,
    timeline: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """清洗收益率，返回 (干净收益率, 异常标记)。

    两类处理：
      1. **上市首日**：无前收、无涨跌幅限制 → 直接置 NaN
      2. **超涨跌停限制的除权日** → 置 NaN（而不是置 0，置 0 会低估波动）

    参数
    ----
    prices : 被检验的价格面板（``close`` 或 ``close_adj``）
    quoted_prev : **未复权**前收盘价面板，必须传入（缺省退化为
        ``prices.shift(1)``，仅当 prices 本身就是报价时才正确）。
        涨跌停容差基准必须是**未复权**前收，见 ``judge_limit_rules`` 的警告。
    timeline : **未复权**价格面板，仅用于确定「交易日序号」与首个行情日。
        ⚠️ 检验 ``close_adj`` 时**必须**传入：
        close_adj 有 NULL（1.6% 的 A 股行）时，用它自己的非空位置算序号会
        把位置整体前移/后移，导致新股无涨跌幅窗口判定错位 ——
        实测会让异常数从112 条虚高到 11,183 条（100 倍）。
    """
    ret = prices.pct_change(fill_method=None)
    bad = pd.DataFrame(False, index=ret.index, columns=ret.columns)
    prev_close = prices.shift(1) if quoted_prev is None else quoted_prev
    tl = prices if timeline is None else timeline

    for c in ret.columns:
        s = prices[c].dropna()
        if len(s) < 2:
            continue
        # 上市首日：无前收、无涨跌幅限制
        first = s.index[0]
        if first in bad.index:
            bad.at[first, c] = True
        r = ret[c].dropna()
        if len(r) == 0:
            continue
        pc = prev_close[c].reindex(r.index)
        tdi, fd = _trade_day_index(tl, prices, c, r.index)
        over = r.index[judge_limit_rules(c, r, pc, tdi, fd).to_numpy()]
        if len(over):
            # ⚠️ 必须用 .loc[index, col] 逐列赋值。
            #    若 over 为空 Index，`bad.loc[empty_index, c] = True`
            #    在 RangeIndex 上会广播到【全部行】→ 整表变True。
            bad.loc[over, c] = True

    return ret.mask(bad), bad


def classify_residuals(
    prices: pd.DataFrame,
    quoted: pd.DataFrame,
    bad: pd.DataFrame,
    quoted_prev: pd.DataFrame | None = None,
    timeline: pd.DataFrame | None = None,
) -> dict[str, int]:
    """把「规则内超限」残留逐条按**形态**分三类计数（Q10：三态分开）。

    只做形态分类，**不做成因定性** —— 成因要在公司行为记录上逐条核对，
    库内暂无逐日公司行为明细（ts_income 的分红是期值不是逐日事件）。

    ``两口径同超限``
        当日**未复权**口径在同一判定链（judge_limit_rules）下也超限。
        已知反例边界：「除权跳空残留 + 复权因子缺失」正是**复权失败**的一种
        形态，与「源数据报价错误」在行情数据内不可区分 —— 故此形态只能记
        「成因存疑」，**不得**据此断言「属源数据错误、非复权失败」。
    ``仅复权口径超限``
        当日未复权口径在限内、仅前复权口径超限 ⇒ 复权因子当日有跳变
        （分红/送转计入），与「除权日总收益按规则略超报价限幅」一致
        （sh600095 2021-05-26 形态：复权收益 +10.02%、报价涨幅 +9.40%，
        差额即当日分红收益率）。
        已知反例边界：复权因子过矫、重复计入分红也呈此形态。
    ``无法判定``
        当日未复权收益算不出（相邻报价缺失），单列不并入前两类（Q10）。

    上市首日不属归因对象（由 n_first_day 单列）。
    """
    ret_q = quoted.pct_change(fill_method=None)
    prev_close = quoted.shift(1) if quoted_prev is None else quoted_prev
    tl = quoted if timeline is None else timeline
    counts = {"两口径同超限": 0, "仅复权口径超限": 0, "无法判定": 0}
    for c in prices.columns:
        flagged = bad.index[bad[c].to_numpy()]
        if not len(flagged):
            continue
        s = prices[c].dropna()
        first = s.index[0] if len(s) else None
        r = ret_q[c].dropna()
        pc = prev_close[c].reindex(r.index)
        tdi, fd = _trade_day_index(tl, prices, c, r.index)
        over_q = judge_limit_rules(c, r, pc, tdi, fd)
        for d in flagged:
            if first is not None and d == first:
                continue                    # 上市首日单列（n_first_day）
            if d not in r.index:
                counts["无法判定"] += 1      # 当日未复权收益缺失
            elif bool(over_q[d]):
                counts["两口径同超限"] += 1
            else:
                counts["仅复权口径超限"] += 1
    return counts


def audit_price_data(
    prices: pd.DataFrame,
    verbose: bool = True,
    quoted_prev: pd.DataFrame | None = None,
    timeline: pd.DataFrame | None = None,
) -> dict:
    """价格数据体检。

    判定走 ``market_rules`` 的价格感知容差 + 新股窗口（统一入口，无第二套口径）。
    quoted_prev / timeline 为**未复权**面板；检验 close_adj 时**必须**传入，
    否则会用前复权价算容差、把合法涨跌停误判成超限。
    """
    ret = prices.pct_change(fill_method=None)
    clean, bad = clean_returns(prices, quoted_prev=quoted_prev,
                               timeline=timeline)
    n_tot = int(ret.notna().sum().sum())
    n_bad = int(bad.sum().sum())
    abs_all = float(ret.abs().stack().sum())
    abs_bad = float(ret.where(bad).abs().stack().sum())

    # 上市首日 vs 其余异常
    n_first = 0
    for c in prices.columns:
        s = prices[c].dropna()
        if len(s) < 2:
            continue
        if s.index[0] in bad.index:
            n_first += int(bad.at[s.index[0], c])
    n_other = n_bad - n_first

    # 对等权买入持有的影响
    b_raw = (1 + ret.mean(axis=1)).cumprod().iloc[-1]
    b_clean = (1 + clean.mean(axis=1)).cumprod().iloc[-1]
    # 交易日数折年数属 ② 时长换算兜底（无日期索引输入：交易日数 ÷ 年均交易日），
    # 口径值 YEAR_TRADING_DAYS = 243。年化三层口径（PITFALLS P10「年化三层口径裁决」）：
    #   ① 倍数换算 ×252/periods、σ×√252 → SCALING_TRADING_DAYS
    #   ② 时长换算兜底（本行属此层）    → YEAR_TRADING_DAYS = 243
    #   ③ 有日期的年跨越                → 自然日 365.25
    # 此前注记的「层间分歧」在本行处理：历史输出按 252 折算、值冻结不倒改；
    # 本次起新输出按 ② 层 = 243，差值打印行随行带折年系数注。
    # （① 层系数串进 ② 层曾使年数记小约 3.57% =1−243/252、
    #   annual_drag 记大约 3.7% =252/243−1。）
    years = len(prices) / YEAR_TRADING_DAYS

    out = {
        "n_assets": int(prices.shape[1]),
        "n_days": int(prices.shape[0]),
        "n_ret": n_tot,
        "n_bad": n_bad,
        "n_first_day": n_first,
        "n_other": n_other,
        # 关卡只看「规则内超限」，上市首日本就无前收、不该算异常
        "pct_bad": n_other / n_tot,
        "abs_share_bad": abs_bad / abs_all if abs_all else np.nan,
        "ew_buy_hold_raw": float(b_raw - 1),
        "ew_buy_hold_clean": float(b_clean - 1),
        "annual_drag": float((b_raw - b_clean) / years),
    }
    if verbose:
        print("=" * 70)
        print("价格数据体检")
        print("=" * 70)
        print(f"  样本           {out['n_assets']:,} 只 × {out['n_days']:,} 日")
        print(f"  日收益样本     {n_tot:,}")
        print(f"  异常           {n_bad:,}（{out['pct_bad']*100:.3f}%）")
        print(f"    ├─ 上市首日  {n_first:,}")
        print(f"    └─ 规则内超限{n_other:,}")
        print(f"  异常贡献的绝对波动  {out['abs_share_bad']*100:.1f}%")
        print()
        print("  等权买入持有 10 年累计")
        print(f"    原始         {(b_raw-1)*100:>+7.1f}%")
        print(f"    清洗后       {(b_clean-1)*100:>+7.1f}%")
        print(f"    → 两口径之差 {(out['annual_drag'])*100:>+7.3f}pp/年"
              f"（②层时长兜底折年：年数 = 行数/{YEAR_TRADING_DAYS}）")
        if verbose:
            print()
            print("  ⚠️ 这个差值**不是复权质量指标**。它衡量「把超限日置 NaN 会改变多少」：")
            print("     · 对未复权价：超限多为除权假跳空 → 差值大 = 污染重")
            print("     · 对前复权价：残留超限的成因分类见关卡判定处的运行时三分类，")
            print("       无论归入哪一类，置 NaN 都会抹掉当日真实收益 → 差值大 ≠ 污染重")
            print("     → 判断复权是否成功，只看 `异常率`，不要看这个差值")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="数据质量审计")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--field", default="close",
                    choices=["close", "close_adj"],
                    help="close=未复权（体检）；close_adj=前复权（修好后回归）")
    args = ap.parse_args()

    codes = [c for c in all_codes() if is_a_share(c)]
    if not args.all:
        import random
        random.seed(42)
        codes = random.sample(codes, min(args.n, len(codes)))
    print(f"审计 {len(codes):,} 只 A 股（{args.start} ~ {args.end}）"
          f"字段={args.field}\n")

    prices = load_prices(codes, args.start, args.end, field=args.field)
    if prices.empty:
        raise RuntimeError(
            f"字段 {args.field} 读出空面板。\n"
            "  若选的是 close_adj，说明并库未执行，请先运行：\n"
            "    uv run python research/scripts/merge_tushare_into_db.py"
        )
    # 涨跌停容差必须按【未复权】前收盘价算 —— 限幅约束的是报价，与复权无关
    quoted_panel = load_prices(codes, args.start, args.end, field="close")
    quoted_prev = quoted_panel.shift(1)
    # audit_price_data 内部直接打印全部结果（verbose=True），
    # 返回的 dict 仅作调用方备用，此处无需接收。
    out = audit_price_data(prices, quoted_prev=quoted_prev,
                             timeline=quoted_panel)

    print()
    print("=" * 70)
    print("逐板块异常分布")
    print("=" * 70)
    _, bad = clean_returns(prices, quoted_prev=quoted_prev,
                             timeline=quoted_panel)
    for b in ["main", "gem", "star", "bse"]:
        cols = [c for c in prices.columns if board_of(c) == b]
        if not cols:
            continue
        nb = int(bad[cols].sum().sum())
        nr = int(prices[cols].pct_change(fill_method=None).notna().sum().sum())
        print(f"  {b:<6} {len(cols):>4} 只  异常 {nb:>6} / {nr:>10,} "
              f"= {nb/nr*100:.3f}%")
    print()
    print("限制说明：main ±10%、gem/star ±20%、bse ±30%（已按前收盘价计算取整容差，")
    print("并剔除注册制新股上市前 5 日的无涨跌幅窗口）")

    # ── 关卡判定（AGENTS.md 硬约束：偏差必须 < 0.5pp/年）──────────
    print()
    print("=" * 70)
    print("关卡判定")
    print("=" * 70)
    if args.field == "close_adj":
        # 前复权：判据是「规则内超限率」，不是与清洗后的差值。
        # ⚠️ 残留超限的成因**必须逐条归因，不能假设**；下述计数由
        #    classify_residuals() 在**本次运行时**算出。一次性快照写死进
        #    代码/文档正文后，下次重跑就会说谎（「99.7% 源数据错误」与
        #    「33 条（29.5%）+ 79 条」两版写死数字已先后作废）。
        #    归因口径、已知反例边界与实测快照见
        #    docs/03_项目报告/12_价量因子有效性检验报告.md
        #    （一、数据关卡 · 残留超限归因口径）。
        att = classify_residuals(prices, quoted=quoted_panel, bad=bad,
                                 quoted_prev=quoted_prev)
        n_att = sum(att.values())
        ok = out["pct_bad"] < 0.005
        if ok:
            print(f"  ✓ 前复权规则内超限率 {out['pct_bad']*100:.3f}% < 0.5%"
                  f" → 允许做收益结论")
        else:
            print(f"  ✗ 前复权超限率 {out['pct_bad']*100:.3f}% ≥ 0.5% → **禁止收益结论**")
        print(f"    残留超限 {n_att} 条（不含上市首日）按形态分类，成因不硬判：")
        print(f"      · 两口径同超限   {att['两口径同超限']:>6} 条 —— 成因存疑："
              f"可能是源数据报价错误，")
        print("        也可能是除权跳空残留 + 复权因子缺失（后者恰是复权失败的特征之一，")
        print("        不逐条对照公司行为记录不能定性；已知反例边界："
              "「未复权也超限」≠「非复权失败」）")
        print(f"      · 仅复权口径超限 {att['仅复权口径超限']:>6} 条 —— 当日复权因子跳变"
              f"（分红/送转计入），")
        print("        与「除权日总收益按规则略超报价限幅」一致"
              "（sh600095 2021-05-26 形态：复权 +10.02%、报价 +9.40%）；")
        print("        已知反例边界：复权因子过矫 / 重复计入分红也呈此形态")
        print(f"      · 无法判定       {att['无法判定']:>6} 条 —— 当日未复权收益缺失，"
              f"按 Q10 单列，不并入前两类")
        print("    归因口径与实测快照：docs/03_项目报告/12_价量因子有效性检验报告.md"
              "（一、数据关卡 · 残留超限归因口径）")
        print("    因子研究阶段仍用 mask 剔除（见 run_factor_study.py）")
        return 0 if ok else 2
    if out["pct_bad"] >= 0.005:
        print(f"  ⚠️ 未复权超限率 {out['pct_bad']*100:.3f}% ≥ 0.5%，"
              f"须用 close_adj 做收益研究")
    else:
        print("  ✓ 未复权价格已无显著除权污染")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
