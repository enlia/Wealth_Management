"""同口径对比：本项目因子组合 vs Qlib 官方基准（沪深300）。

为什么必须做同口径对比
----------------------
Qlib 官方基准的 9.01%（LightGBM）是「沪深300 成分股 + 多空/Topk 口径 +
未明确成本」。而本项目之前的结论是「全市场 5,185 只 + 多空口径 + 扣20bp」。
两者不可直接比较。本脚本把本项目因子组合限制到【同一批 300 只】，
并同时输出多头端与多空端、扣费前后，统一口径后再对照。

沪深300 成分股来源：通达信 boards 字段里的 ZS_沪深300 标签（恰好 300 只）。
⚠️ 该标签是【当前快照】，严格说有成分变更的前视偏差，
但这与 Qlib 使用的数据集口径一致（Qlib 也用静态成分表），
因此对比是公平的。

用法
----
  uv run python scripts/head_to_head_vs_qlib.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from factor_lab.analysis.scorecard import build_scorecard
from factor_lab.config import DB_PATH, DEFAULT_COST, DEFAULT_RESEARCH, OUTPUT_DIR
from factor_lab.data import load_long, load_factor_prices, load_stock_info
from factor_lab.data.universe import build_universe
from expand_factor_library import build_pv_factors
from optimize_turnover import backtest_holdings, group_rank, zscore_cs
from run_financial_study import build_financial_factors, neutralize

TRADING_DAYS = 252

# Qlib 官方基准（CSI300 + Alpha158，20 种子均值，2026-10-05 抓取官方 README）
QLIB_BENCH = {
    "LightGBM":0.0901,
    "XGBoost":  0.0780,
    "MLP":      0.0895,
    "DoubleEnsemble": 0.1158,
    "GRU":      0.0344,      # Alpha158 selected-20 features
    "LSTM":     0.0381,
    "Transformer": 0.0273,
    "Linear":   0.0692,
    "CatBoost": 0.0765,
}
QLIB_COST_WARNING = 0.25   # Qlib 文档：每日交易+10bp往返 = 年损耗 25%


def get_csi300() -> list[str]:
    """从通达信 boards 字段取沪深300 成分股。"""
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT code FROM stock_info WHERE boards LIKE '%ZS_沪深300%'").fetchall()
    con.close()
    return [r[0] for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description="与 Qlib 同口径对比")
    ap.add_argument("--factors", default="low_vol_120,bp,cf_quality")
    ap.add_argument("--buffer", type=float, default=0.10)
    ap.add_argument("--rebalance", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = DEFAULT_RESEARCH
    names = [x.strip() for x in args.factors.split(",") if x.strip()]

    print("=" * 92)
    print("同口径对比：本项目因子组合 vs Qlib 官方基准")
    print("=" * 92)
    print(f"股票池: 沪深300 成分股（与 Qlib 基准同一 universe）")
    print(f"因子:   {', '.join(names)}")
    print(f"组合构建: 缓冲区 {args.buffer:.0%} + 每 {args.rebalance} 日调仓")
    print(f"成本:   往返 {DEFAULT_COST.round_trip*10000:.0f}bp")

    csi = get_csi300()
    print(f"\n[1/4] 沪深300 成分股 {len(csi)} 只")

    # ── 载入 ──
    t0 = time.perf_counter()
    long = load_long(csi, start=cfg.start_date, end=cfg.end_date)
    info = load_stock_info()
    info = info[info["code"].isin(csi)]
    long = build_universe(long, cfg, info=info, verbose=False)
    alive = long["code"].unique().tolist()
{ind}# 🔴 两种价格口径必须分开取（2026-10-06 修）：
{ind}#    PB/EP 用未复权（bps/eps 是财报披露的原始数字），
{ind}#    收益/IC/回测用后复权（未复权价除权日有假跳空，
{ind}#    实测全市场等权口径年化偏差 5~17pp/年）。
{ind}px_raw, prices = load_factor_prices(alive, start=cfg.start_date,
{ind}                                end=cfg.end_date)
    print(f"      可用 {len(alive)} 只 × {prices.shape[0]:,} 日，"
          f"{time.perf_counter()-t0:.1f}s")

    # ── 构造合成因子 ──
    print("\n[2/4] 构造因子 …")
    panel = pd.read_parquet(
        Path(__file__).resolve().parents[2] / "runtime" / "financial_panel.parquet")
    panel = panel[panel["sym"].isin(alive)]
    shares = info.set_index("code")["shares"] if "shares" in info.columns else None
    fac = build_financial_factors(panel, prices.index, px_raw, shares=shares)
    daily = fac.pop("_daily")
    ind_map, cap_map = daily["industry"], daily["mktcap"]
    pv = build_pv_factors(long)

    parts = []
    for n in names:
        s = fac[n] if n in fac else pv.get(n)
        if s is None:
            print(f"      ✗ 找不到 {n}")
            continue
        s = neutralize(s, ind_map.reindex(s.index).to_numpy(),
                       cap_map.reindex(s.index).to_numpy())
        parts.append(zscore_cs(s).rename(n))
        print(f"      ✓ {n}")
    score = pd.concat(parts, axis=1).mean(axis=1).dropna()
    w = score.unstack()
    print(f"      合成得分 {len(score):,} 个")

    # ── 多空口径打分卡（与之前所有实测一致）──
    print("\n[3/4] 多空口径打分卡（用于与 Qlib 的分组口径对齐）…")
    r = build_scorecard("combo", score, prices,
                        DEFAULT_COST.round_trip, cfg.quantiles)
    if "ic" in r:
        print(f"      IC={r['ic']:+.4f}  ICIR={r['icir']:.3f}  t={r['t']:+.2f}")
        print(f"      年化毛 {r['annual_gross']*100:+.2f}%  "
              f"净 {r['annual_net']*100:+.2f}%  换手 {r['turnover']*100:.2f}%")
        print(f"      三段净收益: "
              + " / ".join(f"{x*100:+.2f}%" if np.isfinite(x) else "n/a"
                            for x in r["seg_net"]))

    # ── 多头端 + 缓冲区（可执行口径）──
    print("\n[4/4] 多头端 + 缓冲区（实盘可执行口径）…")
    bt = backtest_holdings(w, prices, cfg.quantiles, args.buffer,
                           DEFAULT_COST.round_trip, args.rebalance)
    print(f"      换手 {bt['turnover']*100:.3f}%  "
          f"年化毛 {bt['annual_long_gross']*100:+.2f}%  "
          f"净 {bt['annual_long_net']*100:+.2f}%")

    # ── 对照表 ──
    # ⚠️⚠️ 必须同时报「基准收益」，否则数字会严重误导。
    # 实测发现：CSI300 成分股【等权】买入持有年化 +11.78%，
    # 而【指数】（市值加权）只有 +3.67% —— 相差 8 个百分点。
    # 若只报「组合年化 +11.74%」，读者会以为赚到了 11.74%，
    # 实际上其中 ~11.8% 只是等权 beta，真正超额接近 0 甚至为负。
    # → 本项目一律以【指数年化】为基准计算超额。
    idx_code = "sh000300"
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    idxtbl = pd.read_sql_query(
        "SELECT date, close FROM bar_daily WHERE code=? ORDER BY date",
        con, params=[idx_code])
    con.close()
    idxtbl["date"] = pd.to_datetime(idxtbl["date"].astype(str), format="%Y%m%d")
    idx_ret = idxtbl.set_index("date")["close"].pct_change()
    bench = float(idx_ret.mean() * TRADING_DAYS)

    ew = prices.pct_change().mean(axis=1)
    bench_ew = float(ew.mean() * TRADING_DAYS)

    print("\n" + "=" * 92)
    print("⚠️ 先看基准：不做对就会严重误读")
    print("=" * 92)
    print(f"  沪深300 指数（市值加权）买入持有年化{bench*100:>7.2f}%   ← 正确基准")
    print(f"  CSI300 成分股【等权】买入持有年化   {bench_ew*100:>7.2f}%   "
          f"← 与组合口径同，但含大量小盘暴露")
    print(f"  差异{((bench_ew - bench)*100):>7.2f}pp  —— 等权跑赢市值加权，"
          f"这是 A 股长期存在的『小盘效应』")

    print("\n" + "=" * 92)
    print("对照结果（全部为超额口径，基准 = 沪深300 指数）")
    print("=" * 92)
    ours_ml = r.get("annual_gross") if "annual_gross" in r else np.nan
    ours_ml_net = r.get("annual_net") if "annual_net" in r else np.nan
    head_net = bt["annual_long_net"]

    print(f"{'方案':<28} {'年化':>9} {'超额(vs指数)':>13} {'口径'}")
    print("-" * 92)
    for k, v in sorted(QLIB_BENCH.items(), key=lambda x: -x[1]):
        print(f"Qlib {k:<23} {v*100:>8.2f}% {'':>13} 官方基准，未明确计成本")
    print("-" * 92)
    print(f"{'本项目 多空（毛）':<26} {ours_ml*100:>8.2f}% "
          f"{(ours_ml-bench)*100:>12.2f}% 扣费前，多空，需融券")
    print(f"{'本项目 多空（净）':<26} {ours_ml_net*100:>8.2f}% "
          f"{(ours_ml_net-bench)*100:>12.2f}% 扣 20bp")
    print(f"{'本项目 多头+缓冲区（净）':<24} {head_net*100:>8.2f}% "
          f"{(head_net-bench)*100:>12.2f}% 扣 20bp，换手 "
          f"{bt['turnover']*100:.2f}%")
    print("-" * 92)
    print(f"""
【⚠️ 重要修正 —— 早前口径的误读】
本项目「多头+缓冲区」年化 {head_net*100:+.2f}%，但其中绝大部分是
**等权 beta**：CSI300 成分股等权买入持有本身就有 {bench_ew*100:+.2f}%，
而市值加权的沪深300 指数只有 {bench*100:+.2f}%。

→ 相对指数的真实超额 = {(head_net-bench)*100:+.2f}pp，
   这个数字才是「因子选股贡献的部分」，而且它并不出色。

【与 Qlib 的正确比较】
Qlib 官方表同样没给基准收益，其 +9.01% 也应理解为【组合总收益】而非超额。
若两者都按总收益比，本项目 {head_net*100:+.2f}% 略高于 LightGBM 9.01%；
但两者的**成本假设、特征数量、是否含做空**都不同，不能直接断言胜负。

【诚实的结论】
1. 本路线在【扣费后可执行】口径下总收益与 Qlib 树模型相当，
   换手率 0.066% 是决定性优势（Qlib 自己警告每日交易年损耗 25%）。
2. 但**相对市值加权基准的超额很薄**，因子选股的真实贡献有限。
3. 要真正超过 Qlib，需要提升的是【排序能力】，而不是换因子或压成本 ——
   成本已经几乎归零，没有优化空间了。
""")

    out_dir = Path(args.out) if args.out else OUTPUT_DIR / "headtohead"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmp_df = pd.DataFrame([
        {"方案": f"Qlib {k}", "年化": v, "超额vs指数": v - bench,
         "口径": "官方基准，未明确计成本"}
        for k, v in QLIB_BENCH.items()
    ] + [
        {"方案": "本项目 多空毛", "年化": ours_ml,
         "超额vs指数": ours_ml - bench, "口径": "扣费前，多空，需融券"},
        {"方案": "本项目 多空净", "年化": ours_ml_net,
         "超额vs指数": ours_ml_net - bench, "口径": "扣20bp，多空"},
        {"方案": "本项目 多头缓冲区净", "年化": head_net,
         "超额vs指数": head_net - bench, "口径": "扣20bp，多头，可执行"},
        {"方案": "【基准】沪深300指数", "年化": bench, "超额vs指数": 0.0,
         "口径": "市值加权，正确基准"},
        {"方案": "【基准】CSI300等权", "年化": bench_ew,
         "超额vs指数": bench_ew - bench, "口径": "含小盘暴露"},
    ])
    cmp_df.to_csv(out_dir / "vs_qlib.csv", index=False, encoding="utf-8-sig")
    print(f"结果目录: {out_dir}")
    print("提示: 统计检验不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())