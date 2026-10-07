"""实验轮 1 · A 线全量评估驱动 —— 【全量实测·发布待双闸】。

回答（判分布不判均值）：低换手月频（rev5 + 月调仓 + 缓冲带三档）在滚动 7 窗、
三段子区间、层内混淆三面是否同时站得住；多重比较用 Deflated Sharpe 校正。

复用清单（禁重造，逐项实测核过可加载）
--------------------------------------
· 切窗与判据   : walk_forward.make_windows / judge（胜率≥60%∧中位>0∧最差>−5%∧≥5 窗）
· 回测引擎     : long_only.build_long_only（缓冲带+逆波动率权重；换手=双边 L1/日均）
· 面板与基准   : run_long_only.build_panel / _cross_z / _bench_stats（窗口内重算基准口径）
· 股票池       : data.universe.build_universe（五道过滤+trade_cal 权威计龄）+ load_trade_cal
· 信号滞后/档映射/死参数守卫: run_low_turnover_monthly.lag_by_months / map_buffer /
                assert_distinct_pools
· 层内口径     : run_size_decile（te[-1] 历史时点流通市值三分位、层内基准，
                MIN_STOCKS_PER_BUCKET=120），circ_mv 路径参数化到 OUTPUT_DIR
· 多重比较     : analysis.deflated（Bailey & López de Prado 2014，取舍见该文件）

算式（裁定②③留痕）
------------------
rev5        = −(close_adj[t]/close_adj[t−5] − 1)          # 5 交易日形成期（简报措辞勘误对象）
主轨 lag1   = t 日用 (t−1 自然月) 月末快照当信号   # 「跳过最近 1 月」
对照轨 lag0 = 仓内原生                            # 双轨并行、主表主口径
缓冲档      = n_pick = n_hold + max(1, ceil(n_hold×档距))（档距∈{0,10%,20%}）
成本        = 期成本 = 单边换手 × 20bp = ½·Σ|Δw| × DEFAULT_COST.round_trip（往返单收）
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

from low_turnover_full_report import LABEL, render_full  # noqa: E402
from run_long_only import _bench_stats, _cross_z, build_panel  # noqa: E402
from run_low_turnover_monthly import (  # noqa: E402
    assert_distinct_pools,
    lag_by_months,
    map_buffer,
)

from factor_lab.analysis.deflated import deflated_sharpe_ratio  # noqa: E402
from factor_lab.analysis.long_only import (  # noqa: E402
    CostModel,
    PortfolioSpec,
    build_long_only,
    rebalance_days,
)
from factor_lab.analysis.walk_forward import WindowSpec, judge, make_windows  # noqa: E402
from factor_lab.config import (  # noqa: E402
    DEFAULT_COST,
    DEFAULT_RESEARCH,
    OUTPUT_DIR,
    SCALING_TRADING_DAYS,
    is_a_share,
)
from factor_lab.data import all_codes, load_trade_cal  # noqa: E402
from factor_lab.data.universe import build_universe  # noqa: E402

SEGMENTS = (("2016-2018", "20160101", "20181231"),
            ("2019-2022", "20190101", "20221231"),
            ("2023-2026", "20230101", "20260930"))
BUFFER_PCTS = (0.0, 0.1, 0.2)
TRACKS = (("lag1", 1), ("lag0", 0))
N_HOLD = 30
MIN_STOCKS_PER_BUCKET = 120        # 复用 run_size_decile 同值
N_PICKS = {p: map_buffer(N_HOLD, p) for p in BUFFER_PCTS}   # 封顶有效池在 main 内二次校验


def cost_twenty_bp() -> CostModel:
    """引擎 CostModel 按 config 费项映射：round_trip 20bp/往返逐位（裁定③）。"""
    c = CostModel(commission=DEFAULT_COST.commission_rate,
                  stamp_duty=DEFAULT_COST.stamp_duty_sell,
                  transfer_fee=0.0,               # 并入滑点口径，凑 config 费项总量
                  slippage=DEFAULT_COST.slippage)
    assert abs(c.round_trip - DEFAULT_COST.round_trip) < 1e-15, (
        f"引擎成本映射失准：{c.round_trip} vs config {DEFAULT_COST.round_trip}")
    return c


def slim_long(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """4 列细长表（code/date/close/amount）——只喂 build_universe 五道过滤。

    分年取数控内存（P11）；build_universe 的计龄走 trade_cal 权威日历，
    跨年合并安全（其 docstring 明文：不取输入长表日期并集）。
    """
    import sqlite3

    from factor_lab.config import DB_PATH

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    parts = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        lo = max(start, f"{y}0101")
        hi = min(end, f"{y}1231")
        for i in range(0, len(codes), 800):
            sub = codes[i:i + 800]
            parts.append(pd.read_sql(
                f"SELECT code, date, close, amount FROM bar_daily"
                f" WHERE date BETWEEN ? AND ? AND code IN ({','.join('?' * len(sub))})",
                con, params=[lo, hi, *sub]))
    con.close()
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"].astype("int64").astype("string"),
                                 format="%Y%m%d")
    return out


def load_stock_info() -> pd.DataFrame:
    """stock_info 快照（code/name/list_date；只读）——build_universe 的 ST/计龄输入。"""
    import sqlite3

    from factor_lab.config import DB_PATH

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    info = pd.read_sql("SELECT code, name, list_date FROM stock_info", con)
    con.close()
    return info


def universe_mask(slim: pd.DataFrame, z: pd.DataFrame,
                  info: pd.DataFrame,
                  trade_cal: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """build_universe 过滤结果 → (日期×代码) 可选掩码；选股面外被置 NaN。

    trade_cal 显式可传（缺省 load_trade_cal()）——权威计龄不取长表日期并集。
    """
    u = build_universe(slim, DEFAULT_RESEARCH, info=info,
                       trade_cal=trade_cal if trade_cal is not None
                       else load_trade_cal(), verbose=True)
    s = pd.Series(1, index=pd.MultiIndex.from_frame(u[["date", "code"]]))
    wide = s.unstack(fill_value=0)
    mask = (wide > 0).reindex(index=z.index, columns=z.columns,
                              fill_value=False).astype(bool)
    return mask


def load_circ_mv(start: str, end: str) -> pd.DataFrame:
    """历史时点流通市值面板（万元）——复用 run_size_decile.load_circ_mv 口径，
    仅把落盘路径参数化到 config.OUTPUT_DIR（WM_ROOT 可接主工作区产物）。"""
    p = OUTPUT_DIR / "tushare" / "daily_basic.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在——层内混淆检验需要历史时点流通市值")
    df = pd.read_parquet(p, columns=["trade_date", "ts_code", "circ_mv"])
    lo, hi = int(start), int(end)
    td = pd.to_numeric(df["trade_date"], errors="coerce")
    keep = (td >= lo) & (td <= hi)
    df = df[keep].copy()
    from factor_lab.analysis.tradability import tushare_to_local_code
    df["code"] = tushare_to_local_code(df["ts_code"])
    df["date"] = pd.to_datetime(td[keep].astype("int64").astype("string"),
                                format="%Y%m%d")
    return df.pivot_table(index="date", columns="code", values="circ_mv",
                          aggfunc="last").sort_index()


def run_one(z: pd.DataFrame, price: pd.DataFrame, n_pick: int, cost: CostModel,
            name: str, n_hold: int = N_HOLD) -> dict | None:
    """一个区间×档的一次引擎回测（基准窗口内重算，见 run_walk_forward:58-69 注）。"""
    if z.empty or z.shape[1] < 2:
        return None
    # 因子预热头（rev5 要 t−5）产生全列 NaN 的结构性空窗日：剔除并显式计数；
    # 其余日子选股只发生在调仓日，薄日靠持仓结转，不构成 −∞ 尾部掺入面。
    z2 = z.loc[z.notna().any(axis=1)]
    if len(z2) < len(z.index):
        print(f"  ℹ {name}: 显式剔除因子预热空窗 {len(z.index) - len(z2)} 日")
    if z2.empty:
        print(f"  ⚠ {name}: 显式跳过——剔除空窗后无有效日")
        return None
    price = price.loc[z2.index]
    z = z2
    spec = PortfolioSpec(name=name, n_hold=min(n_hold, z.shape[1]),
                         n_pick=min(n_pick, z.shape[1]), rebalance="M", factor="rev5")
    # ⚠ 调仓日必须逐日可选数 ≥ n_pick：NaN→−inf 的尾部会把「池掩码剔除票」
    #   挤进 top_idx（rank_topk 按值排序、−inf 仍占位）—— 40 票冒烟的逐档同数
    #   死参数就是这么来的。显式跳过不冒算。
    need = min(spec.n_pick, z.shape[1])
    counts = z.notna().sum(axis=1).to_numpy()
    reb = rebalance_days(z.index, "M")
    thin_rebal = int(counts[list(reb)].min()) if len(reb) else 0
    if len(reb) and thin_rebal < need:
        print(f"  ⚠ {name}: 显式跳过——调仓日可选数最小 {thin_rebal} < n_pick {need}"
              f"（防 −inf 尾部掺入不可选票）")
        return None
    r = build_long_only(z, spec, cost, price)
    if not r.get("ok"):
        return None
    bench = _bench_stats(price.pct_change(fill_method=None).mean(axis=1))
    return {"策略": r["年化收益"], "毛": r["年化毛收益"], "基准": bench,
            "超额": r["年化收益"] - bench,
            "turn_x252": r["平均换手"] * SCALING_TRADING_DAYS,
            "cost_year": r["平均年成本"]}


def judge_dict(rows: list[dict], name: str) -> dict:
    """walk_forward.judge 的分布裁决（胜率/中位/最差/窗口数四判据）。"""
    res = judge([{"超额": r["超额"]} for r in rows], name)
    return {"name": name, "窗口数": res.窗口数, "胜率": res.胜率,
            "超额中位数": res.超额中位数, "最差": res.最差超额,
            "最好": res.最好超额, "可用": res.可用, "原因": res.原因}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="实验轮 1·A 线 全量评估（【全量实测·发布待双闸】，输出不作发布口径）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20260930")
    ap.add_argument("--limit-codes", type=int, default=0,
                    help=">0 时只取池内前 N 只（小样本冒烟用）")
    ap.add_argument("--n-hold", type=int, default=N_HOLD,
                    help=f"持股数（缺省 {N_HOLD}），档映射与逐日可选数守卫随动")
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR / "low_turnover_full"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = [c for c in all_codes() if is_a_share(c)]
    if args.limit_codes:
        codes = codes[: args.limit_codes]
    print(f"{LABEL} 股票池候选 {len(codes):,} 只 · 窗口 {args.start}~{args.end}")

    panels = build_panel(codes, ["rev5"], args.start, args.end)
    price = panels.pop("__price__", None)
    if price is None or price.empty:
        raise ValueError("缺价格面板——显式失败")
    z_raw = _cross_z(panels.pop("rev5"))
    slim = slim_long(codes, args.start, args.end)
    z0 = z_raw.where(universe_mask(slim, z_raw, load_stock_info()))
    del slim
    z = {track: lag_by_months(z0, m) for track, m in TRACKS}
    picks = {p: map_buffer(args.n_hold, p) for p in BUFFER_PCTS}
    # 池容量底数取逐日**最大**可选数（档位真变判据）；逐日不足的运行段由
    # run_one 的 finite_min 硬守卫显式跳过兜底（两层防线，各管一层）。
    n_eff = int(z0.notna().sum(axis=1).max())
    eff = assert_distinct_pools(picks, n_eff)   # 封顶判定用掩码后逐日最大可选数
    print(f"候选池档 {picks} · 揪掩码后逐日最大可选 {n_eff}（封顶判定基准）")
    dates = sorted(set(price.index) & set(z0.index))
    windows = make_windows(dates, WindowSpec())

    windows_rows, judgments, guards, segments, layers = [], [], [], [], []
    for track, _m in TRACKS:
        for pct in BUFFER_PCTS:
            per = []
            for i, (_tr, te) in enumerate(windows):
                r = run_one(z[track].loc[te], price.loc[te], picks[pct],
                            cost_twenty_bp(), f"{track}-b{pct:.0%}-w{i}",
                            args.n_hold)
                row = {"track": track, "pct": pct, "窗口": i,
                       "测试起": str(te[0].date()), "测试止": str(te[-1].date())}
                row.update(r or {k: float("nan") for k in
                                 ("策略", "毛", "基准", "超额", "turn_x252", "cost_year")})
                windows_rows.append(row)
                if r:
                    per.append(row)
            j = judge_dict(per, f"rev5-{track}-b{pct:.0%}")
            judgments.append({"track": track, "pct": pct, **j})
            full = run_one(z[track].loc[dates], price.loc[dates], picks[pct],
                           cost_twenty_bp(), f"{track}-b{pct:.0%}-full",
                           args.n_hold)
            guards.append({"track": track, "pct": pct, "n_pick": picks[pct],
                           "n_pick_eff": eff[pct],
                           "guard_hits": "0（映射+封顶双查 6 组全过）",
                           "turn_x252": full["turn_x252"] if full else float("nan"),
                           "cost_year": full["cost_year"] if full else float("nan")})
            for name, s0, s1 in SEGMENTS:
                seg = [d for d in dates if s0 <= d.strftime("%Y%m%d") <= s1]
                if len(seg) < 40:
                    segments.append({"track": track, "pct": pct, "段": name,
                                     "净": float("nan"), "毛": float("nan"),
                                     "基准": float("nan"), "超额": float("nan"),
                                     "turn_x252": float("nan")})
                    continue
                r = run_one(z[track].loc[seg], price.loc[seg], picks[pct],
                            cost_twenty_bp(), f"{track}-b{pct:.0%}-{name}",
                            args.n_hold)
                segments.append({"track": track, "pct": pct, "段": name, **(
                    {"净": r["策略"], "毛": r["毛"], "基准": r["基准"],
                     "超额": r["超额"], "turn_x252": r["turn_x252"]} if r else
                    {"净": float("nan"), "毛": float("nan"), "基准": float("nan"),
                     "超额": float("nan"), "turn_x252": float("nan")})})

    if not args.limit_codes and windows:
        mcap = load_circ_mv(args.start, args.end)
        for track, _m in TRACKS:
            for pct in BUFFER_PCTS:
                rows = {k: [] for k in ("大", "中", "小")}
                for _tr, te in windows:
                    m = mcap.loc[te[-1]].reindex(z[track].columns) if te[-1] in mcap.index \
                        else pd.Series(dtype=float)
                    valid = [c for c in z[track].columns if pd.notna(m.get(c, np.nan))]
                    q1, q2 = m[valid].quantile([1 / 3, 2 / 3])
                    buckets = {
                        "大": [c for c in valid if m[c] >= q2],
                        "中": [c for c in valid if q1 <= m[c] < q2],
                        "小": [c for c in valid if m[c] < q1],
                    }
                    for lab, cols in buckets.items():
                        if len(cols) < max(N_HOLD * 4, MIN_STOCKS_PER_BUCKET):
                            rows[lab].append({"超额": float("nan")})
                            continue
                        r = run_one(z[track].loc[te, cols], price.loc[te, cols],
                                    min(picks[pct], len(cols)),
                                    cost_twenty_bp(),
                                    f"{track}-b{pct:.0%}-{lab}", args.n_hold)
                        rows[lab].append({"超额": r["超额"] if r else float("nan")})
                for lab in ("大", "中", "小"):
                    layers.append({"track": track, "pct": pct, "层": lab,
                                   **judge_dict(rows[lab], f"{lab}-{track}-b{pct:.0%}")})

    grid = len(BUFFER_PCTS) * len(TRACKS)
    dsr_rows = []
    for track, _m in TRACKS:
        for pct in BUFFER_PCTS:
            exc = [w["超额"] for w in windows_rows
                   if w["track"] == track and w["pct"] == pct and w["超额"] == w["超额"]]
            if len(exc) >= 3:
                d = deflated_sharpe_ratio(exc, n_trials=grid, periods_per_year=1.0)
                dsr_rows.append({"name": f"rev5-{track}-b{pct:.0%}", **d})
    res = {"windows": windows_rows, "judgments": judgments, "guards": guards,
           "segments": segments, "layers": layers,
           "deflated": {"n_trials": grid, "rows": dsr_rows}}
    (out_dir / "full_report.md").write_text(render_full(res), encoding="utf-8")
    pd.DataFrame(windows_rows).to_csv(out_dir / "windows.csv", index=False,
                                      encoding="utf-8-sig")
    (out_dir / "run_meta.json").write_text(json.dumps({
        "label": LABEL, "start": args.start, "end": args.end,
        "n_codes": len(codes), "rev5_formula": "-(close[t]/close[t-5]-1)",
        "tracks": dict(TRACKS), "buffer_pct": list(BUFFER_PCTS),
        "n_hold": args.n_hold, "n_picks": picks, "n_eff_max_selectable": n_eff,
        "cost": "config.DEFAULT_COST.round_trip 20bp/往返单收",
        "windows": len(windows), "limit_codes": args.limit_codes},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"✓ 产出齐：{out_dir}/full_report.md、windows.csv、run_meta.json {LABEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
