"""按持有期正确评估因子 + 周期匹配的组合构建。

本项目最重要的一次修正
======================
此前用「1 日前瞻 IC」筛选因子，得出：
  ✅ 可用：bp（IC +0.024）、low_vol_120（IC +0.037）、cf_quality
  ❌ 剔除：skew_60、range_20、vol_ratio_20_60、idio_vol_20（IC 为负）

但换成 20~120 日前瞻后，结论**完全颠倒**（全市场实测）：

  因子            IC_1日IC_20日    IC_60日   IC_120日
  skew_60         -0.006    +0.139   +0.305   +0.196
  range_20        -0.035    +0.167   +0.269   +0.274
  vol_ratio_20_60 -0.010    +0.231   +0.146   +0.063
  low_vol_120     +0.030    +0.019   -0.037   -0.142
  bp-0.002    -0.060   -0.111   -0.149
  cf_quality       +0.002    +0.005   +0.008   +0.015

**根因：A 股 T+1 制度使日内与隔夜收益方向相反**（全A 日内 +28.22% /
隔夜 −13.85%），所以任何因子的 1 日 IC 都被日内噪声污染。
→ **评估因子必须用与预期持有期匹配的前瞻期。**

第二个修正：换手与持有期必须匹配
--------------------------------
skew_60 / range_20 / vol_ratio 在日频下换手 10%~20%，扣费后为负。
但它们的信号是 20~120 日级别的 → **调仓频率也应该是 20~120 天**，
换手会自然大幅下降。

本脚本：
  1. 用多前瞻期（20/60/120 日）重新评估每个因子
  2. 只保留「在目标持有期上 IC 稳定为正」的因子
  3. 用【与持有期匹配的调仓频率】回测
  4. 输出命中率、净收益、最大回撤、持有期胜率
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from factor_lab.analysis.selection_quality import (TRADING_DAYS,
                                                  equity_metrics,
                                                  hit_rate_analysis,
                                                  period_winrate)
from factor_lab.config import DEFAULT_COST, DEFAULT_RESEARCH, OUTPUT_DIR, is_a_share
from factor_lab.data import all_codes, load_long, load_factor_prices, load_stock_info
from factor_lab.data.universe import build_universe
from expand_factor_library import build_pv_factors
from optimize_turnover import group_rank, zscore_cs
from run_financial_study import (build_financial_factors, neutralize,
                                 pick_sample)

# 用 60 日口径评估（覆盖季度调仓）
EVAL_PERIOD = 60


def ic_by_period(fac_w: pd.DataFrame, prices: pd.DataFrame,
                 periods=(1, 20, 60, 120), sample_every: int = 5,
                 min_obs: int = 100) -> pd.DataFrame:
    """各因子在不同前瞻期上的 IC。返回 index=factor, columns=IC_<period>。

    ⚠️ 长前瞻期（60/120 日）的收益窗口高度重叠，
       若逐日抽样会把同一段行情重复计入 → IC 虚高且 t 值失真。
       正解：每 sample_every 日抽一次。
    ⚠️ min_obs 之前设 100，在小样本（如 400 只）下几乎所有日期都跳过，
       导致结果全空。改为 50，并把阈值提到参数里。
    """
    rows = {}
    # 预先算好各前瞻期的未来收益，避免在因子循环里反复计算
    fwds = {k: prices.pct_change(k) for k in periods}
    for c in fac_w.columns:
        row = {}
        # 长表 (date, asset) → 宽表 date × asset
        a_all = fac_w[c].unstack()
        for k in periods:
            fwd = fwds[k]
            common = a_all.columns.intersection(fwd.columns)
            if len(common) < 50:
                row[f"IC_{k}"] = np.nan
                continue
            w2, f2 = a_all[common], fwd[common]
            step = 1 if k <= 20 else sample_every
            ics = []
            for d in w2.index[::step]:
                a, b = w2.loc[d], f2.loc[d]
                m = a.notna() & b.notna()
                if m.sum() < min_obs:
                    continue
                ics.append(a[m].corr(b[m], method="spearman"))
            row[f"IC_{k}"] = float(np.mean(ics)) if ics else np.nan
        rows[c] = row
    return pd.DataFrame(rows).T


def backtest_with_rebalance(
    w: pd.DataFrame, prices: pd.DataFrame, n_groups: int = 5,
    rebalance_days: int = 60, n_drop: int = 20,
    cost: float = DEFAULT_COST.round_trip,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, float]:
    """固定持仓数 + 每次只换最差 n_drop 只（Qlib TopkDropout 思路）。

    ⚠️ 本项目在这个函数上错了两次，都记录在此：
    第 1 版：用「绝对分位边界」保留持仓 → 调仓越慢换手反而越高（反常）。
    第 2 版：用「缓冲区扩大的目标持仓数」→ 单次换手高达 90%+，
           因为持仓集合本身在漂移，且换手口径混用了「单次」与「年化」。

    正确做法（与 Qlib TopkDropoutStrategy 一致）：
      · 持仓数固定 N = 全池/n_groups
      · 每次调仓：按因子分数排序，卖掉【已持仓中最差的 n_drop 只】，
        买入【未持仓中最好的 n_drop 只】
      · 换手 = n_drop / N（与排名重排幅度无关，只由 n_drop 决定）
      → 换手完全可控：年换手 = n_drop / N × (252 / rebalance_days)

    返回 (日频组合收益, 日频基准收益, 持仓矩阵, 年换手率)
    """
    g = group_rank(w, n_groups)
    dates = prices.index
    fwd = prices.pct_change()
    bench = fwd.mean(axis=1)

    n_hold = max(5, int(round(len(prices.columns) / n_groups)))
    n_drop = max(1, min(n_drop, n_hold // 2))

    held: set = set()
    port, hold_mat = [], []
    per_rebal_turn = []

    for i, d in enumerate(dates):
        if i % rebalance_days == 0:
            row = g.loc[d] if d in g.index else None
            valid = row[row > 0] if row is not None and (row > 0).any() else None
            if valid is not None:
                score = w.loc[d].reindex(valid.index)
                order = list(score.sort_values(ascending=False).index)
                if not held:
                    held = set(order[:n_hold])
                else:
                    # 卖：已持仓里分数最差的 n_drop 只
                    cur = [(c, score[c]) for c in held if c in score.index]
                    cur.sort(key=lambda x: x[1])
                    sell = [c for c, _ in cur[:n_drop]]
                    # 买：未持仓里分数最好的 n_drop 只
                    pool = [c for c in order if c not in held][:n_drop]
                    per_rebal_turn.append(len(set(sell) | set(pool)) / n_hold)
                    held = (held - set(sell)) | set(pool)
        if held:
            r = fwd.loc[d, list(held)]
            port.append(float(np.nanmean(r.to_numpy(dtype=float))))
        else:
            port.append(np.nan)
        hold_mat.append(dict.fromkeys(held, 1.0))

    pr = pd.Series(port, index=dates, dtype=float)
    hm = pd.DataFrame(0, index=dates, columns=prices.columns, dtype=float)
    if len(hold_mat) == len(dates):
        col_idx = {c: j for j, c in enumerate(prices.columns)}
        for i, h in enumerate(hold_mat):
            if h:
                for c in h:
                    j = col_idx.get(c)
                    if j is not None:
                        hm.iat[i, j] = 1.0
    # 年换手率 = 单次换手 × 年调仓次数
    per = float(np.mean(per_rebal_turn)) if per_rebal_turn else np.nan
    annual_turn = per * (TRADING_DAYS / rebalance_days)
    return pr, bench, hm, annual_turn


def main() -> int:
    ap = argparse.ArgumentParser(description="按持有期正确评估 + 周期匹配回测")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = DEFAULT_RESEARCH
    full = args.all or args.n == 0
    names = ["low_vol_120", "low_vol_60", "bp", "cf_quality", "ep", "roe",
             "skew_60", "range_20", "vol_ratio_20_60", "idio_vol_20",
             "max_ret_20", "price_pos250", "drawdown_60", "amihud_20",
             "illiq_accel", "turnover_20", "volume_shock", "low_vol_60"]

    print("=" * 88)
    print("按持有期正确评估因子 + 周期匹配组合构建")
    print("=" * 88)
    print(f"股票池: {'全市场 A 股' if full else f'{args.n} 只'}")
    print(f"成本: 往返 {DEFAULT_COST.round_trip*10000:.0f}bp")

    t0 = time.perf_counter()
    codes = [c for c in all_codes() if is_a_share(c)]
    if not full:
        codes = pick_sample(codes, args.n)
    long = load_long(codes, start=cfg.start_date, end=cfg.end_date)
    info = load_stock_info()
    info = info[info["code"].isin(codes)]
    long = build_universe(long, cfg, info=info, verbose=False)
    alive = long["code"].unique().tolist()
    # 🔴 两种价格口径必须分开取（2026-10-06 修）：
    #    PB/EP 用未复权（bps/eps 是财报披露的原始数字），
    #    收益/IC/回测用前复权（未复权价除权日有假跳空，
    #    实测全市场等权口径年化偏差 5~17pp/年）。
    px_raw, prices = load_factor_prices(alive, start=cfg.start_date,
                                    end=cfg.end_date)
    panel = pd.read_parquet(
        Path(__file__).resolve().parents[2] / "runtime" / "financial_panel.parquet")
    panel = panel[panel["sym"].isin(alive)]
    shares = info.set_index("code")["shares"] if "shares" in info.columns else None
    fac = build_financial_factors(panel, prices.index, px_raw, shares=shares)
    daily = fac.pop("_daily")
    ind_map, cap_map = daily["industry"], daily["mktcap"]
    pv = build_pv_factors(long)
    print(f"  {len(alive):,} 只 × {prices.shape[0]:,} 日，{time.perf_counter()-t0:.1f}s")

    Z = {}
    for n in names:
        s = fac.get(n) if n in fac else pv.get(n)
        if s is None:
            continue
        s = neutralize(s, ind_map.reindex(s.index).to_numpy(),
                       cap_map.reindex(s.index).to_numpy())
        Z[n] = zscore_cs(s)
    Z = pd.DataFrame(Z)
    print(f"  {Z.shape[1]} 个中性化因子\n")

    out_dir = Path(args.out) if args.out else OUTPUT_DIR / "holding"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 多前瞻期 IC ──
    print("[1/3] 各因子多前瞻期 IC …")
    ic_tbl = ic_by_period(Z, prices, min_obs=60)
    print(ic_tbl.sort_values(f"IC_{EVAL_PERIOD}", ascending=False)
          .to_string(float_format=lambda x: f"{x:>+8.4f}"))
    ic_tbl.to_csv(out_dir / "ic_multi_period.csv", encoding="utf-8-sig")

    # 选因子：在目标持有期上 IC 为正，且符号在 20/60/120 日基本一致
    print(f"\n[2/3] 按 {EVAL_PERIOD} 日口径筛选因子 …")
    good, bad = [], []
    for c in ic_tbl.index:
        signs = [np.sign(ic_tbl.loc[c, f"IC_{k}"]) for k in (20, 60, 120)]
        consistent = len(set(signs)) == 1 and signs[0] > 0
        if ic_tbl.loc[c, f"IC_{EVAL_PERIOD}"] > 0.02 and consistent:
            good.append(c)
        else:
            bad.append(c)
    print(f"  ✅ 在 20/60/120 日均为正且 IC60>0.02: {len(good)} 个 → {good}")
    print(f"  ❌ 不满足: {len(bad)} 个")
    if not good:
        print("  无可用因子，退出")
        return 1

    # ── 3. 周期匹配回测 ──
    print("\n[3/3] 周期匹配回测（调仓频率 = 持有期）…")
    score = Z[good].mean(axis=1).dropna()
    w = score.unstack()
    results = []
    for reb, n_drop in ((20, 5), (60, 5), (60, 10), (120, 5), (120, 10)):
        pr, br, hm, to_annual = backtest_with_rebalance(
            w, prices, cfg.quantiles, reb, n_drop, DEFAULT_COST.round_trip)
        # 年成本 = 往返费率 × 年换手 × 分组数（每笔成交涉及买卖双侧）
        cost_ann = DEFAULT_COST.round_trip * to_annual * cfg.quantiles
        pr_net = pr - cost_ann / TRADING_DAYS
        label = f"每{reb}日调仓/每次换{n_drop}只"
        print(f"\n  ── {label}（年换手 {to_annual*100:.1f}%，"
              f"年成本 {cost_ann*100:.2f}%）")
        fr = pd.concat({k: prices.pct_change(k) for k in (20, 60, 120)}, axis=1)
        hr = hit_rate_analysis(hm, fr, top_frac=1.0)
        eq2 = equity_metrics(pr_net, br)
        for k in (60, 120):
            eq2.update(period_winrate(pr_net, k))
        results.append({"配置": label, "调仓": reb, "n_drop": n_drop,
                        "年换手": to_annual, "年成本": cost_ann, **eq2})

        print(f"     年化(净) {eq2.get('年化收益', np.nan)*100:+.2f}%  "
              f"基准 {eq2.get('基准年化', np.nan)*100:+.2f}%  "
              f"超额 {eq2.get('年化超额', np.nan)*100:+.2f}pp")
        print(f"     最大回撤 {eq2.get('最大回撤', np.nan)*100:.2f}%  "
              f"夏普 {eq2.get('夏普', np.nan):.2f}  "
              f"信息比率 {eq2.get('信息比率', np.nan):.2f}")
        if hr is not None and len(hr):
            for _, r in hr.iterrows():
                print(f"     命中：持有{int(r['持有期_交易日']):>3}日 "
                      f"Top组 {r['Top组上涨率']*100:>5.1f}%  "
                      f"全市场 {r['全市场上涨率']*100:>5.1f}%  "
                      f"超额 {r['超额命中率']*100:>+5.1f}pp")
        wr = eq2.get("持有60日_胜率")
        if wr:
            print(f"     持有60日胜率 {wr*100:.1f}%  "
                  f"盈亏比 {eq2['持有60日_盈亏比']:.2f}")

    pd.DataFrame(results).to_csv(out_dir / "holding_period_test.csv",
                                 index=False, encoding="utf-8-sig")
    print(f"\n结果目录: {out_dir}")
    print("提示: 多头口径，扣 20bp 往返。基准 = 同期全市场等权。")
    print("统计检验不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())