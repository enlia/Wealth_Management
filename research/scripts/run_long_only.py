"""单边多头策略检验：回答「手动买股票能不能赚钱」。

为什么必须做单边检验
--------------------
因子 IC 是**多空组合**的统计性质，但手动买股票只能做多：
  · 多空可以空掉跌得多的，多头必须持有「剩下的」
  · 多空 IC=0.08 完全可能来自空头端贡献
  · 多头换手 53% 在实盘里是「每周换半仓」

实测（2026-10-06，全市场基线）：
```
因子    IC均值   多空年化毛  多空年化净   平均换手
rev5    0.084+0.68      −0.67        53%
vol60  −0.059    −0.08      −0.28         8%
```
多空毛收益看着不错，扣成本后全负 —— **对手动决策零参考价值**。

本脚本做的事
------------
1. 训练期选因子组合（样本内）
2. **样本外**检验单边多头的年化/波动/回撤/换手/成本
3. 扫参数（持股数、调仓频率、止损）找稳健区间
4. 输出**可直接执行的持仓规则**

⚠️ 三个防过拟合的硬约束：
  · 参数必须**样本外**表现好才采纳，不能只看样本内
  · 相邻参数的表现要**平滑**，孤立尖峰是过拟合
  · 扣成本后仍为正才有效（盈亏平衡换手率是硬约束）

用法
----
  uv run python research/scripts/run_long_only.py --all
  uv run python research/scripts/run_long_only.py --all --factors rev5,vol60,pos250
  uv run python research/scripts/run_long_only.py --all --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from factor_lab.analysis.long_only import (  # noqa: E402
    CostModel,
    PortfolioSpec,
    _year_span,
    breakeven_turnover,
    build_long_only,
    holdout_split,
)
from factor_lab.config import is_a_share  # noqa: E402
from factor_lab.data import all_codes  # noqa: E402
from factor_lab.factors.price_volume import compute_factor  # noqa: E402

OUTPUT = ROOT / "runtime" / "long_only"
BENCH = {"300": "399300.SZ"}


def load_long_chunked(codes: list[str], start: str, end: str,
                      years: tuple[int, ...] | None = None,
                      verbose: bool = True) -> pd.DataFrame:
    """分年加载后复权价，避免一次性读入撑爆内存。

    ⚠️ **为什么必须分块**：
    全市场 5,606 只 × 2,611 日 = **1,080 万行 × 9 列**，
    实测 `load_long` 一次性加载直接
    `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 742 MiB`。
    之前「跑 20 分钟」不是慢，是在反复 GC/重试 —— 假象会掩盖真问题。

    做法：按年切片逐年加载，峰值内存降到 1/10。
    因子计算只要 close_adj（有 OHLC 的因子才追加）。
    """
    import sqlite3

    from factor_lab.config import DB_PATH

    if years is None:
        y0, y1 = int(start[:4]), int(end[:4])
        years = tuple(range(y0, y1 + 1))

    chunks: list[pd.DataFrame] = []
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    total = 0
    for y in years:
        ys, ye = f"{y}0101", f"{y}1231"
        if ye < start[:8] or ys > end[:8]:
            continue
        # ⚠️ **不能用 max(ys, start) 做下界**（实测踩过）：
        #   `start` 可能是**预热起点**（如 20180102），
        #   而 `ys` 是本年 1 月 1 日。max() 会把下界夹回本年元旦，
        #   于是**预热数据一行都读不到**，
        #   长窗口因子（pos250 / mom120）在年初必然全 NaN。
        #
        #   实测：`start=20180102, years=(2018, 2019)`
        #        → 2019 段 lo = max('20190101','20180102') = '20190101'
        #        → 预热完全失效（修复 WARMUP 天数后暴露出来）。
        #
        #   正确做法：下界直接用 start（它已是更早的预热起点）。
        lo = start[:8]
        hi = min(ye, end[:8])
        parts = []
        # 分批取代码，避免单条 SQL 的 IN 列表过长
        # ⚠️ **列必须按因子声明的依赖取**，不能只取 OHLC。
        #   实测踩过：volratio5_60 依赖 `vol`、amount20 依赖 `amount`，
        #   而这里硬编码了 OHLC+close_adj ——
        #   结果 `KeyError: 'Column not found: vol'`，
        #   两个因子**从未在滚动检验里跑过**（静默漏测）。
        #   代价可控：多取两列约增加 15% 内存（约 18MB/百万行）。
        cols = ["code", "date", "open", "high", "low", "close", "close_adj"]
        cols += [c for c in ("vol", "amount") if c not in cols]
        for i in range(0, len(codes), 800):
            sub = codes[i:i + 800]
            q = (f"SELECT {', '.join(cols)} "
                 f"FROM bar_daily WHERE date BETWEEN ? AND ? "
                 f"AND close_adj IS NOT NULL AND code IN ({','.join('?' * len(sub))})")
            parts.append(pd.read_sql(q, con, params=[lo, hi, *sub]))
        if parts:
            df = pd.concat(parts, ignore_index=True)
            chunks.append(df)
            total += len(df)
            if verbose:
                print(f"    {y}: {len(df):>10,} 行  累计 {total:>12,}")
    con.close()
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True).sort_values(
        ["code", "date"], kind="stable")
    # ⚠️ **必须转 datetime**：`bar_daily.date` 存的是 int（20260104），
    #   而 `compute_factor` 内部用 `pd.DatetimeIndex(dates)` 构造索引。
    #   int 被当纳秒时间戳 → 索引变成 1970-01-01 → unstack 后每年只有 1 行。
    #   实测踩过：因子面板「每年 1 格」，而原始数据明明有 60 万行。
    #   `load_long` 内部做了同样的转换，这里必须对齐。
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
    out["high_adj"] = out["high"] * out["close_adj"] / out["close"]
    out["low_adj"] = out["low"] * out["close_adj"] / out["close"]
    return out


def build_panel(codes, factors, start, end) -> dict[str, pd.DataFrame]:
    """算各因子的宽表面（索引=日期，列=代码）。

    ⚠️ **必须边加载边算，不能全部加载完再算**：
    全市场 11 年 = 1,080 万行 × 9 列 ≈ 1.2 GB，
    实测一次性加载直接 `_ArrayMemoryError: Unable to allocate 742 MiB`。
    「跑 20 分钟」不是慢，是在反复 GC/重试 —— **假象会掩盖真问题**。

    做法：按年加载 → 立刻算因子 → 丢掉原始数据 → 只保留因子面板。
    因子面板是 (日期 × 代码) 的浮点矩阵，11 年约 5,600×2,611×8 = 117 MB。

    ⚠️ **跨年窗口**：因子最大回看窗口（mom250 / pos250 / vol60…）会跨年。
       每年多加载 `WARMUP_TRADING_DAYS` 个**前置交易日**用于预热，
       但只保留本年计算结果 —— 否则 mov250 在年初会全NaN，
       相当于每年初白扔 250 个交易日的信号。
    """
    y0, y1 = int(start[:4]), int(end[:4])
    acc: dict[str, list[pd.DataFrame]] = {f: [] for f in factors}
    price_parts: list[pd.DataFrame] = []
    n_rows = 0

    for y in range(y0, y1 + 1):
        ys = f"{y}0101" if y > y0 else start[:8]
        ye = f"{y}1231" if y < y1 else end[:8]
        # 多读 WARMUP 天前置数据（用上一年末），只为本年预热
        if y > y0:
            pad_start = _shift_date(ys, -WARMUP_TRADING_DAYS)
        else:
            pad_start = ys
        long = load_long_chunked(codes, pad_start, ye, years=(y - 1, y),
                                 verbose=False)
        if long.empty:
            continue
        # 只保留 [ys, ye] 区间内的计算结果
        # （date 此时已是 datetime —— load_long_chunked 已转换）
        if "date" in long.columns:
            lo = pd.Timestamp(ys)
            hi = pd.Timestamp(ye)
            long = long[(long["date"] >= lo) & (long["date"] <= hi)]
        n_rows += len(long)
        # 价格面板（后复权）也要留一份 —— 收益必须从价格算
        price_parts.append(long.pivot_table(
            index="date", columns="code", values="close_adj",
            aggfunc="last").sort_index())
        for f in factors:
            s = compute_factor(f, long)
            if s is None or s.empty:
                continue
            # compute_factor 返回 MultiIndex(date, asset) —— 层名是 asset
            acc[f].append(s.unstack("asset").sort_index())
        del long
        # ⚠️ **不能用 len(DataFrame)** 算「格数」——
        #   len(df) 返回的是**列数**，单因子面板永远打印「1 格」。
        #   实测踩过：面板实际有 2,600 日期 × 5,600 只 = 1,460 万格，
        #   打印却是「1 格」，一度被误判为「面板损坏、跨年拼接失效」。
        #
        # ⚠️ 也不能写 `x.shape` —— acc[f] 是**逐年 append 的列表**，
        #   元素是 DataFrame，列表本身没有 .shape（实测 AttributeError）。
        #   必须逐元素累加 shape 的乘积。
        n_cells = sum(f.shape[0] * f.shape[1]
                      for parts in acc.values() for f in parts)
        print(f"    {y}: 原始 {n_rows:>11,} 行，因子面板 {n_cells:>12,} 格")

    out = {}
    for f, parts in acc.items():
        if not parts:
            print(f"  ⚠ 因子 {f} 无数据")
            continue
        #⚠️ 必须 axis=0（按日期索引纵向堆叠）。
        #   axis=1 是横向拼列，会把 11 年的因子并成 5,600×11 列，
        #   实测表现为「每年只有 1 格」。
        wide = (pd.concat(parts, axis=0) if len(parts) > 1 else parts[0])
        # 跨年预热期可能与下一年重叠，同一 (日期, 代码) 只保留一份
        if wide.index.duplicated().any():
            wide = wide[~wide.index.duplicated(keep="last")]
        wide = wide.sort_index().loc[
            (wide.index >= pd.Timestamp(f"{y0}0101"))
            & (wide.index <= pd.Timestamp(f"{y1}1231"))]
        out[f] = wide
        cov = float(wide.notna().mean().mean())
        print(f"  ✓ {f:<16} 覆盖率 {cov*100:5.1f}%  "
              f"非空 {wide.notna().sum().sum():>12,}  "
              f"内存 {wide.memory_usage(deep=True).sum()/1e6:.0f}MB")
    out["__price__"] = (pd.concat(price_parts, axis=0)
                        if price_parts else pd.DataFrame())
    p = out["__price__"]
    if p.index.duplicated().any():
        p = p[~p.index.duplicated(keep="last")]
    print(f"  ✓ {'price':<16} 覆盖率 "
          f"{float(p.notna().mean().mean())*100:5.1f}%  "
          f"内存 {p.memory_usage(deep=True).sum()/1e6:.0f}MB")
    return out


# 因子最大回看窗口对应的前置**交易日**数（pos250 / mom120 需要 250 日）
WARMUP_TRADING_DAYS = 260


def _shift_date(d: str, n_days: int) -> str:
    """把 YYYYMMDD 往前推 n 个**交易日**（用于跨年预热）。

    ⚠️ **必须按交易日算，不能用自然日**（2026-10-06 实测缺陷）：
       初版用 `pd.Timedelta(days=260)` 减自然日，
       260 自然日 ≈ **173 个交易日**，而 pos250 需要 **250 个交易日** ——
       预热少了约 77 个交易日。

       实测后果：`_shift_date('20240101', -260)` 返回 `20230416`，
       正确应是 `20230102`。
       pos250 / mom120 在每年年初会缺 ~77 个交易日的因子值，
       11 年累积约 **850 天（3.3 年）数据被静默丢弃**。

       这类缺陷不报错、不影响其他因子，只让长窗口因子**样本变少**，
       很难被发现 —— 必须靠「跨年时检查长窗口因子的非空率」来发现。
    """
    ts = pd.Timestamp(d)
    # BDay 只排除周末，不排除法定节假日 —— 偏保守（多取几天数据），
    # 比少取安全。少取会丢样本，多取只是多读一点。
    return (ts - pd.tseries.offsets.BDay(abs(n_days))).strftime("%Y%m%d")


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


def main() -> int:
    ap = argparse.ArgumentParser(description="单边多头策略检验")
    ap.add_argument("--all", action="store_true", help="全市场")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--factors", default="rev5,vol60,pos250,mom60_skip20")
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20260930")
    ap.add_argument("--hold", type=int, default=30)
    ap.add_argument("--pick", type=int, default=100)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--sweep", action="store_true", help="扫参数找稳健区间")
    args = ap.parse_args()

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    cost = CostModel()
    print("=" * 88)
    print("单边多头策略检验（面向手动买股决策）")
    print("=" * 88)
    print(f"交易成本: {cost.describe()}")
    print(f"  年化20%超额可承受换手率上限 {breakeven_turnover(0.20, cost):.1%}")

    codes = [c for c in all_codes() if is_a_share(c)]
    if not args.all:
        codes = codes[: args.n]
    print(f"\n股票池: {len(codes):,} 只")

    panels = build_panel(codes, factors, args.start, args.end)
    if not panels:
        print("无可用因子")
        return 1

    # 取出价格面板（收益来源），其余是因子
    price = panels.pop("__price__", None)
    if price is None or price.empty:
        print("✗ 缺价格面板，无法计算收益")
        return 1
    # 横截面标准化（每天独立，避免风格漂移）
    z = {k: _cross_z(v) for k, v in panels.items()}

    # ── 基准：等权全市场（必须对照，否则无法判断「跑赢了吗」）──
    # ⚠️ 只看绝对收益没有意义：
    #   实测 rev5 样本内年化 −7.78%，而同期等权基准 +123%
    #   ——策略不是「亏钱」，是**大幅跑输**。
    bench_ret = price.pct_change(fill_method=None).mean(axis=1)
    dates = sorted(set.intersection(*[set(v.index) for v in z.values()]))
    tr, te = holdout_split(dates, args.train_frac)
    print(f"\n样本切分（按时间）: 训练 {len(tr):,} 日 / 测试 {len(te):,} 日")
    print(f"  训练 {tr[0]} ~ {tr[-1]}")
    print(f"  测试 {te[0]} ~ {te[-1]}")

    spec = PortfolioSpec(
        name="等权基线", n_hold=args.hold, n_pick=args.pick,
        rebalance="M", factor=",".join(factors))

    print("\n" + "=" * 88)
    print("单因子：样本内 vs 样本外")
    print("=" * 88)
    rows = []
    for f, p in z.items():
        sp_i = PortfolioSpec(**{**spec.__dict__, "name": f"{f}-IN", "factor": f})
        sp_o = PortfolioSpec(**{**spec.__dict__, "name": f"{f}-OUT", "factor": f})
        ri = build_long_only(p.loc[tr], sp_i, cost, price.loc[tr])
        ro = build_long_only(p.loc[te], sp_o, cost, price.loc[te])
        report(ri, f"{f} 样本内")
        report(ro, f"{f} 样本外")
        if ri.get("ok") and ro.get("ok"):
            b_in = _bench_stats(bench_ret.loc[tr])
            b_te = _bench_stats(bench_ret.loc[te])
            rows.append({
                "因子": f, "内年化": ri["年化收益"], "外年化": ro["年化收益"],
                "外夏普": ro["夏普"], "外回撤": ro["最大回撤"],
                "外换手": ro["平均换手"], "外成本": ro["平均年成本"],
                "衰减": ri["年化收益"] - ro["年化收益"],
                "内基准": b_in, "外基准": b_te,
                "内超额": ri["年化收益"] - b_in,
                "外超额": ro["年化收益"] - b_te,
            })
    if rows:
        df = pd.DataFrame(rows).sort_values("外超额", ascending=False)
        print("\n" + "=" * 92)
        print("因子检验汇总（**超额才是关键**）")
        print("=" * 92)
        print(f"{'因子':<14}{'内年化':>9}{'外年化':>9}{'内基准':>9}{'外基准':>9}"
              f"{'内超额':>10}{'外超额':>10}")
        print("-" * 92)
        for r in rows:
            print(f"{r['因子']:<14}{r['内年化']:>9.2%}{r['外年化']:>9.2%}"
                  f"{r['内基准']:>9.2%}{r['外基准']:>9.2%}"
                  f"{r['内超额']:>+10.2%}{r['外超额']:>+10.2%}")
        print()
        print("⚠ **绝对收益为正但超额为负 = 白做**（跑输基准）。")
        print("  实测 rev5 样本内年化 −7.78%，同期等权基准 +123%")
        print("  → 不是亏钱，是大幅跑输，对手动决策毫无价值。")
        print()
        print("  另：样本外超额为正的因子，才值得继续做参数稳健性检验。")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT / "factor_inout.csv", index=False,
                  encoding="utf-8-sig")

    # 参数扫描
    if args.sweep:
        print("\n" + "=" * 88)
        print("参数扫描（**只看样本外**，找稳健区间）")
        print("=" * 88)
        best = sorted(rows, key=lambda r: -r["外年化"])[:1] if rows else []
        if not best:
            print("无有效因子，跳过")
            return 0
        top = best[0]["因子"]
        p = z[top]
        print(f"\n扫描因子: {top}")
        print(f"{'持股':>5}{'候选':>6}{'年化':>9}{'夏普':>8}{'回撤':>9}"
              f"{'换手':>8}{'成本':>8}")
        print("-" * 60)
        sweep = []
        for nh in (10, 20, 30, 50):
            for np_ in (nh * 2, nh * 3, nh * 5):
                sp = PortfolioSpec(
                    name=f"h{nh}p{np_}", n_hold=nh, n_pick=min(np_, p.shape[1]),
                    rebalance="M", factor=top)
                r = build_long_only(p.loc[te], sp, cost, price.loc[te])
                if r.get("ok"):
                    sweep.append({"持股": nh, "候选": np_, **{
                        k: r[k] for k in ("年化收益", "夏普", "最大回撤",
                                          "平均换手", "平均年成本")}})
                    print(f"{nh:>5}{np_:>6}{r['年化收益']:>9.2%}{r['夏普']:>8.2f}"
                          f"{r['最大回撤']:>9.2%}{r['平均换手']:>8.1%}"
                          f"{r['平均年成本']:>8.2%}")
        if sweep:
            sdf = pd.DataFrame(sweep)
            sdf.to_csv(OUTPUT / "param_sweep.csv", index=False,
                       encoding="utf-8-sig")
            print("\n⚠ 选参数原则：要选**区间内稳定**的，不是最优点。")
            print("  相邻参数表现差异大 = 过拟合，样本外会崩。")

    # 数据版本
    try:
        from data_version import build_manifest
        ver = build_manifest()
        meta = {"数据版本": ver["version"], "打戳": ver["stamped_at"],
                "成本": {"买入": cost.buy, "卖出": cost.sell},
                "因子": factors,
                "训练区间": f"{tr[0]}~{tr[-1]}", "测试区间": f"{te[0]}~{te[-1]}"}
        (OUTPUT / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n数据版本: {ver['version']}（已记录到 run_meta.json）")
    except Exception as e:                                   # noqa: BLE001
        print(f"⚠ 版本记录失败: {e}")

    print("\n提示: 样本外结果才有参考价值；扣成本后为负的方案不可用。")
    print("      以上为统计检验，不构成投资建议。")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
