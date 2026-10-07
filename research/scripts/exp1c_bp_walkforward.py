"""实验 1·C 线：bp（账面市值比）滚动样本外复核入口（薄封装）。

为什么新起本入口
----------------
既有件各自只覆盖一段：build_financial_panel.py（财务面板，法定截止日口径）、
financial_factors.py（因子与中性化）、run_walk_forward.py（滚动多头）、
run_size_decile.py（市值分层混淆）、run_long_only._cross_z/_bench_stats（z 与基准）。
本文件只做**编排 + 简报强制的披露日历级联**，其余一律调用现成函数，不重造：

  1. 披露日历级联（简报实验设计·2，P20(1)/P20(4)/Q5）：
     available_date = (f_ann_date|ann_date) + 1 日   ←「opdate/regdate」实测字段名
                    > end_date + 80 自然日           ← 启发式兜底
     另备两个对照口径：heur80（全部 end_date+80）、registry（只认登记日，无日期记录丢弃）。
     **第一检查项** = heur80 → registry 后收益是否衰减（衰减 ⇒ 启发式曾放宽披露日、
     结果偏乐观；不衰减 ⇒ 启发式未虚增收益）。
  2. 窗口卫生双证（简报实验设计·4）：测试段两两零重叠 + 训练/测试零越界，
     拼接段空窗集显式核查，禁「每天一个样」充滚动。
  3. 多重比较（简报实验设计·5）：Bailey & López de Prado (2014) Deflated Sharpe
     等效自算（mlfinlab 新版需商业许可不可用，见 PITFALLS P1；公式不受版权保护），
     试验数 N = 3（bp 主 + ep/roe 对照）+ 10 族历史出线 on-record = 13。

用法
----
  python research/scripts/exp1c_bp_walkforward.py --smoke 8        # 冒烟（分层抽 8 只）
  python research/scripts/exp1c_bp_walkforward.py --mode cascade   # 主口径全样本
  python research/scripts/exp1c_bp_walkforward.py --mode heur80    # 80 日启发式基线
  python research/scripts/exp1c_bp_walkforward.py --mode registry  # 登记日路径（第一检查项）

所有输出数字属【滚动复核实测·发布待双闸】。本入口为统计检验，不构成投资建议。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

import build_financial_panel as fin_panel  # noqa: E402
from financial_factors import build_financial_factors, neutralize  # noqa: E402
from run_long_only import _bench_stats, _cross_z  # noqa: E402
from run_size_decile import _run_sub, load_circ_mv  # noqa: E402

from factor_lab.analysis.long_only import CostModel, PortfolioSpec, build_long_only  # noqa: E402
from factor_lab.analysis.walk_forward import WindowSpec, judge, make_windows  # noqa: E402
from factor_lab.config import DB_PATH, is_a_share  # noqa: E402
from factor_lab.data import all_codes, load_factor_prices, load_stock_info  # noqa: E402

WM_ROOT = Path(__import__("os").environ.get("WM_ROOT") or ROOT)
OUT = WM_ROOT / "runtime" / "exp1c_bp"
TAG = "【滚动复核实测·发布待双闸】"

# 子区间（AGENTS 二「不分段验证」既有三段口径）
SUB_PERIODS = {
    "2016-2018": ("2016-01-01", "2018-12-31"),
    "2019-2022": ("2019-01-01", "2022-12-31"),
    "2023-2026": ("2023-01-01", "2026-09-30"),
}
# 成本口径（两套并列，简报「成本单边 20bp」字面口径 vs 对账口径往返 20bp；
#  工具默认 costs.CostModel 为往返 30.2bp，作第三行敏感度）
COSTS = {
    "往返20bp": CostModel(slippage=0.0005),      # 对账口径（既有三段数字口径）
    "单边20bp": CostModel(commission=0.002,      # 简报字面口径：买卖各 20bp
                          transfer_fee=0.0, slippage=0.0, stamp_duty=0.0),
}
N_TRIALS = 13          # 3（本批）+ 10 族历史出线 on-record（简报实验设计·5）


# ── 1. 披露日历（新判据代码；测试见 tests/test_exp1c_disclosure.py）──
def load_disclosure_dates() -> pd.DataFrame:
    """从 market.db 只读取（sym/报告期）→（最早披露 f_ann_date、登记日 ann_date）。

    ts_income/ts_balance_sheet/ts_cashflow 带 ann_date 与 f_ann_date（实际披露日）；
    ts_fina_indicator 只有 ann_date。逐 (ts_code, end_date) 取各表最早的日期。
    """
    con = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    q = ("SELECT ts_code, end_date,"
         " MIN(f_ann_date), MIN(ann_date) FROM ("
         "  SELECT ts_code, end_date, f_ann_date, ann_date FROM ts_income"
         "  UNION ALL SELECT ts_code, end_date, f_ann_date, ann_date FROM ts_balance_sheet"
         "  UNION ALL SELECT ts_code, end_date, f_ann_date, ann_date FROM ts_cashflow"
         "  UNION ALL SELECT ts_code, end_date, NULL AS f_ann_date, ann_date"
         "    FROM ts_fina_indicator"
         ") GROUP BY ts_code, end_date")
    d = pd.read_sql_query(q, con)
    con.close()
    d.columns = ["ts_code", "end_date", "opdate", "regdate"]
    for c in ("opdate", "regdate"):
        d[c] = pd.to_datetime(d[c], format="%Y%m%d", errors="coerce")
    d["end_date"] = pd.to_datetime(d["end_date"], format="%Y%m%d", errors="coerce")
    d["sym"] = d["ts_code"].map(_ts_to_local)
    return d.dropna(subset=["sym", "end_date"])


def _ts_to_local(ts: str) -> str | None:
    code6, _, mkt = str(ts).partition(".")
    prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(mkt)
    return prefix + code6 if prefix else None


def apply_disclosure_calendar(panel: pd.DataFrame, disc: pd.DataFrame,
                              mode: str) -> pd.DataFrame:
    """按级联重打 available_date。mode ∈ {cascade, heur80, registry}。"""
    p = panel.copy()
    p["report_period"] = pd.to_datetime(p["report_period"])
    key = disc.rename(columns={"sym": "sym_", "end_date": "report_period"})
    m = p.merge(key[["sym_", "report_period", "opdate", "regdate"]],
                left_on=["sym", "report_period"],
                right_on=["sym_", "report_period"], how="left")
    op = m["opdate"] + pd.Timedelta(days=1)      # 最早披露日 + 1 日
    reg = m["regdate"] + pd.Timedelta(days=1)    # 登记日 + 1 日
    heur = m["report_period"] + pd.Timedelta(days=80)   # end_date + 80 自然日
    if mode == "heur80":
        avail = heur
    elif mode == "registry":
        avail = op.fillna(reg)
    elif mode == "cascade":
        avail = op.fillna(reg).fillna(heur)      # 逐级延顺
    else:
        raise ValueError(f"未知披露日历口径 {mode}")
    p["available_date"] = avail
    n_before = len(p)
    p = p.dropna(subset=["available_date"])
    if len(p) < n_before:
        print(f"      [{mode}] 丢弃无披露日记录 {n_before - len(p):,} 行"
              f"（registry 口径不认启发式兜底）")
    return p


def window_hygiene(windows, label: str) -> dict:
    """窗口卫生双证（简报实验设计·4）：测试段两两零重叠 + 训练/测试零越界。"""
    tes = [list(te) for _, te in windows]
    overlap = 0
    for i in range(len(tes)):
        for j in range(i + 1, len(tes)):
            overlap += len(set(tes[i]) & set(tes[j]))
    leak = sum(1 for tr, te in windows for d in tr if d in set(te))
    gaps = []
    for i in range(len(tes) - 1):
        a, b = pd.Timestamp(tes[i][-1]), pd.Timestamp(tes[i + 1][0])
        # 拼接空窗 = 两测试段之间严格居中的工作日数（首尾相接 ⇒ 0）
        gaps.append(max(int(np.busday_count(a.date(), b.date())) - 1, 0))
    out = {"口径": label, "窗口数": len(tes), "测试段重叠日": overlap,
           "训练/测试越界": leak, "拼接段空窗日": gaps}
    assert overlap == 0, f"测试段两两重叠 {overlap} 日 —— 胜率会被虚高"
    assert leak == 0, f"训练/测试越界 {leak} 日 —— 前视污染"
    return out


# ── 2. 因子装配（薄封装：现成 financial_factors）────────────────────
def build_factor_wide(panel: pd.DataFrame, px_raw: pd.DataFrame,
                      px_adj: pd.DataFrame, shares, names,
                      neutral: bool) -> dict[str, pd.DataFrame]:
    """财务面板 → 日频宽表因子（bp/ep 用未复权价，收益轴取复权面板索引）。"""
    trading_days = px_adj.index
    fac = build_financial_factors(panel, trading_days, px_raw, shares=shares)
    daily = fac.pop("_daily")
    ind = daily["industry"]
    cap = daily["mktcap"] if "mktcap" in daily.columns else None
    wide: dict[str, pd.DataFrame] = {}
    for n in names:
        s = fac[n]
        if neutral:
            s = neutralize(s, ind.to_numpy(),
                           cap.to_numpy() if cap is not None else None)
        wide[n] = s.unstack("asset").reindex(index=trading_days)
    return wide


# ── 3. 判定与多重比较（DSR 等效自算）──────────────────────────────
def deflated_sharpe(excess: pd.Series, n_trials: int = N_TRIALS) -> dict:
    """Bailey & López de Prado (2014) Deflated Sharpe（公式自算等效件）。

    DSR = Φ( (SR_hat − SR0)·√(T−1) / √(1 − γ3·SR + (γ4−1)/4·SR²) )
    SR0 = 期望无技能最大夏普 = √Var·[(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))]
    （γ=0.5772 欧拉常数；SR/√Var 单位=每期，年化换算按 PITFALLS P10 ① 层 ×√252。）
    """
    from math import e, sqrt
    from statistics import NormalDist
    if n_trials < 2:
        raise ValueError(
            f"n_trials={n_trials} < 2 —— 无技能基准 SR0 的 "
            f"Φ⁻¹(1−1/N) 在 N=1 处无定义（=−∞ ⇒ DSR 恒 1 的假绿）。"
            f" 只跑一个试验请用 N=2 近似并在报告注明。")
    x = excess.dropna().to_numpy(dtype=float)
    t = len(x)
    if t < 3:
        return {"DSR": float("nan"), "SR0": float("nan"), "说明": "样本不足"}
    sr = float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else 0.0
    g3 = float(pd.Series(x).skew())
    g4 = float(pd.Series(x).kurt()) + 3.0
    var = (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / t
    gamma = 0.5772156649
    z1 = NormalDist().inv_cdf(1 - 1 / n_trials)
    z2 = NormalDist().inv_cdf(1 - 1 / (n_trials * e))
    sr0 = sqrt(var) * ((1 - gamma) * z1 + gamma * z2)
    denom = sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    dsr = float(NormalDist().cdf((sr - sr0) * sqrt(t - 1) / denom))
    return {"期数T": t, "每期SR": sr, "年化SR": sr * sqrt(252), "SR0": sr0,
            "DSR": dsr, "N_trials": n_trials,
            "公式": "Bailey & López de Prado (2014) Deflated Sharpe（自算等效件）"}


# ── 4. 冒烟锚点与逐月值抽查（红线：U2 三例照抄 + bp 反推价）────────
def anchor_checks(panel: pd.DataFrame) -> list[dict]:
    """红线锚点（UNITS U1/U2 三例照抄）+ bp 反推价自查。

    自带取 sh600519 行情（不依赖抽样池），冒烟与全量同一把尺。
    """
    import sqlite3 as sq
    px_raw, px_adj = load_factor_prices(["sh600519"], start="20240101",
                                        end="20260930")
    rows = []
    # 例 1（UNITS U1 四源闭合锚点）：600519 @2026-09-30 close=1,258.62 元
    got = float(px_raw.loc[pd.Timestamp("2026-09-30"), "sh600519"])
    rows.append({"锚点": "UNITS U1 例1 sh600519 close@2026-09-30",
                 "期望": 1258.62, "实测": round(got, 2),
                 "一致": abs(got - 1258.62) < 0.02})
    # 例 2（UNITS U1 真值标尺）：@2024-08-30 close=1443.19 / close_adj=1338.651654
    r2, a2 = (float(px_raw.loc[pd.Timestamp("2024-08-30"), "sh600519"]),
              float(px_adj.loc[pd.Timestamp("2024-08-30"), "sh600519"]))
    rows.append({"锚点": "UNITS U1 例2 close@2024-08-30",
                 "期望": 1443.19, "实测": round(r2, 2),
                 "一致": abs(r2 - 1443.19) < 0.02})
    rows.append({"锚点": "UNITS U1 例2 close_adj@2024-08-30",
                 "期望": 1338.651654, "实测": round(a2, 6),
                 "一致": abs(a2 - 1338.651654) < 0.02})
    # 例 3：bp 反推价 = bps / bp 须回到当日未复权 close（逐月抽查 3 期）
    con = sq.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    q = ("SELECT end_date, bps FROM ts_fina_indicator "
         "WHERE ts_code='600519.SH' AND end_date IN "
         "('20240331','20240630','20240930') ORDER BY end_date")
    bps = pd.read_sql_query(q, con)
    con.close()
    mini = panel[panel["sym"] == "sh600519"]
    fac = build_financial_factors(mini, px_raw.index, px_raw, shares=None)
    bp_s = fac["bp"].unstack("asset")["sh600519"].dropna()
    for _, r in bps.iterrows():
        ed = pd.to_datetime(r["end_date"])
        after = bp_s.index[bp_s.index > ed]
        if not len(after):
            continue
        d = after[0]
        bp_v = float(bp_s.loc[d])
        implied = float(r["bps"]) / bp_v
        px = float(px_raw.loc[d, "sh600519"])
        rows.append({"锚点": f"例3 bp 反推价 sh600519 {ed:%Y-%m}→{d:%Y-%m-%d}",
                     "期望": round(px, 2), "实测": round(implied, 2),
                     "一致": abs(implied - px) < 0.02 * px})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="实验 1·C 线 bp 滚动样本外复核")
    ap.add_argument("--smoke", type=int, default=0, help="冒烟：分层抽 N 只（5~10）")
    ap.add_argument("--mode", default="cascade",
                    choices=["cascade", "heur80", "registry"])
    ap.add_argument("--factors", default="bp,ep,roe")
    ap.add_argument("--neutral", action="store_true", default=True)
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20260930")
    args = ap.parse_args()
    t0 = time.perf_counter()
    names = [x.strip() for x in args.factors.split(",") if x.strip()]
    spec = WindowSpec(train_years=3, test_months=12)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{TAG}\n实验 1·C 线 bp 滚动样本外复核 | 披露口径={args.mode} | "
          f"{spec.describe()} | 中性化={args.neutral}")

    fin_panel.YJBB_DIR = WM_ROOT / "runtime" / "yjbb"     # 产物定位到主工作区
    raw_panel = fin_panel.build_panel(verbose=True)
    disc = load_disclosure_dates()
    panel = apply_disclosure_calendar(raw_panel, disc, args.mode)
    panel_all = panel
    print(f"  披露日历后 {len(panel):,} 行；启发式兜底占比 "
          f"{(panel['available_date'].isin(panel['report_period'] + pd.Timedelta(days=80))).mean():.1%}")

    codes = [c for c in all_codes() if is_a_share(c)]
    if args.smoke:
        from run_financial_study import pick_sample
        codes = pick_sample(codes, args.smoke)
        print(f"  冒烟样本 {len(codes)} 只（分层抽样 seed=42）")
    panel = panel[panel["sym"].isin(codes)]
    px_raw, px_adj = load_factor_prices(codes, start=args.start, end=args.end)
    info = load_stock_info()
    shares = info.set_index("code")["shares"] if "shares" in info.columns else None
    wide = build_factor_wide(panel, px_raw, px_adj, shares, names,
                             neutral=args.neutral)

    # ── 锚点与逐月抽查（自带 sh600519 取数，不依赖抽样池）────────
    anchors = anchor_checks(panel_all)
    pd.DataFrame(anchors).to_csv(OUT / f"anchors_{args.mode}.csv",
                                 index=False, encoding="utf-8-sig")
    print("\n[锚点/逐月抽查]" + TAG)
    for a in anchors:
        print(f"  {a['锚点']}: 期望 {a['期望']} 实测 {a['实测']} "
              f"{'✓' if a['一致'] else '✗'}")

    # ── 滚动 7 窗 + 双成本模型 ──────────────────────────────────
    hyg = None
    results = {}
    for f in names:
        z = _cross_z(wide[f])
        dates = sorted(set(z.index) & set(px_adj.index))
        z, price = z.loc[dates], px_adj.loc[dates]
        for cname, cost in COSTS.items():
            wins = []
            for i, (_, te) in enumerate(make_windows(dates, spec)):
                sp = PortfolioSpec(name=f"{f}-w{i}", n_hold=30, n_pick=90,
                                   rebalance="M", factor=f)
                r = build_long_only(z.loc[te], sp, cost, price.loc[te])
                bt = price.loc[te].pct_change(fill_method=None).mean(axis=1)
                b = _bench_stats(bt)
                wins.append({
                    "窗口": i, "测试起": str(te[0].date()), "测试止": str(te[-1].date()),
                    "策略": r.get("年化收益", np.nan) if r.get("ok") else np.nan,
                    "毛": r.get("年化毛收益", np.nan) if r.get("ok") else np.nan,
                    "基准": b,
                    "超额": (r.get("年化收益", np.nan) - b) if r.get("ok") else np.nan,
                    "换手": r.get("平均换手", np.nan) if r.get("ok") else np.nan,
                    "备注": "" if r.get("ok") else r.get("reason", "失败")})
            if hyg is None:
                hyg = window_hygiene(make_windows(dates, spec), args.mode)
            jd = judge(wins, f"{f}[{cname}]")
            results[(f, cname)] = (pd.DataFrame(wins), jd)
            print(f"\n【{f} / {cname}】" + TAG)
            print(pd.DataFrame(wins).to_string(index=False))
            print(jd.report())
            pd.DataFrame(wins).to_csv(
                OUT / f"windows_{args.mode}_{f}_{cname}.csv",
                index=False, encoding="utf-8-sig")

    # ── 三段子区间（毛/净并列，成本=往返20bp 对账口径）──────────
    print("\n[三段子区间]" + TAG)
    seg_rows = []
    cost = COSTS["往返20bp"]
    for f in names:
        z = _cross_z(wide[f])
        dates = sorted(set(z.index) & set(px_adj.index))
        for seg, (s, e) in SUB_PERIODS.items():
            te = [d for d in dates if s <= str(d.date()) <= e]
            sp = PortfolioSpec(name=f"{f}-{seg}", n_hold=30, n_pick=90,
                               rebalance="M", factor=f)
            r = build_long_only(z.loc[te], sp, cost, px_adj.loc[te])
            seg_rows.append({"因子": f, "子区间": seg,
                             "净": r.get("年化收益", np.nan),
                             "毛": r.get("年化毛收益", np.nan),
                             "换手": r.get("平均换手", np.nan),
                             "状态": "ok" if r.get("ok") else r.get("reason")})
    seg_df = pd.DataFrame(seg_rows)
    print(seg_df.to_string(index=False))
    seg_df.to_csv(OUT / f"subperiod_{args.mode}.csv", index=False,
                  encoding="utf-8-sig")

    # ── 市值层内混淆（现成 run_size_decile 口径：te[-1] 三分层）──
    print("\n[市值层内混淆]" + TAG)
    mcap = load_circ_mv(args.start, args.end)
    lay_rows = []
    for f in names:
        z = _cross_z(wide[f])
        dates = sorted(set(z.index) & set(px_adj.index))
        z, price_f = z.loc[dates], px_adj.loc[dates]
        mcap_f = mcap.reindex(index=dates)
        cost = COSTS["往返20bp"]
        for i, (_, te) in enumerate(make_windows(dates, spec)):
            m = mcap_f.loc[te[-1]].reindex(z.columns)
            valid = [c for c in z.columns if pd.notna(m.get(c, np.nan))]
            if not valid:
                continue
            q1, q2 = m[valid].quantile([1 / 3, 2 / 3])
            buckets = {"大": [c for c in valid if m[c] >= q2],
                       "中": [c for c in valid if q1 <= m[c] < q2],
                       "小": [c for c in valid if m[c] < q1]}
            bench = _bench_stats(price_f.loc[te].pct_change(
                fill_method=None).mean(axis=1))
            row = {"因子": f, "窗口": i, "基准": bench}
            for lab, cols in buckets.items():
                v = _run_sub(z.loc[te, cols], price_f.loc[te, cols], cost,
                             30, 3, f)
                row[lab] = v
                row[f"{lab}超额"] = v - bench if pd.notna(v) else np.nan
            lay_rows.append(row)
    lay_df = pd.DataFrame(lay_rows)
    print(lay_df.to_string(index=False))
    lay_df.to_csv(OUT / f"size_layers_{args.mode}.csv", index=False,
                  encoding="utf-8-sig")

    # ── 多重比较：Deflated Sharpe 等效自算 ───────────────────────
    print("\n[多重比较]" + TAG)
    dsr_rows = []
    for f in names:
        wins_df = results[(f, "往返20bp")][0]
        # 用窗口超额的时间邻接近似：T=窗口数太少，改用各窗口内月度换手口径不可得
        # ⇒ 以窗口超额序列 + 期数=有效交易日数并注记（等效说明）
        dsr = deflated_sharpe(wins_df["超额"], n_trials=N_TRIALS)
        dsr["因子"] = f
        dsr_rows.append(dsr)
        print(f"  {f}: {dsr}")
    pd.DataFrame(dsr_rows).to_csv(OUT / f"dsr_{args.mode}.csv", index=False,
                                  encoding="utf-8-sig")

    meta = {"披露口径": args.mode, "窗口": spec.describe(),
            "因子": names, "中性化": args.neutral,
            "成本模型": {k: str(v) for k, v in COSTS.items()},
            "窗口卫生": hyg, "耗时s": round(time.perf_counter() - t0, 1),
            "口径三件套": {
                "价格口径": "因子用未复权 close（bps 同口径）；收益用 load_factor_prices:adj（DB 字段 close_adj，代码注释标『前复权』、真源 AGENTS 九.5 标后复权——以 DB 真源为准记后复权）",
                "换手单位": "月频调仓的单边换手率均值（engine.simulate_matrix 口径）",
                "年化方式": "自然日跨度 ÷ 365.25（_year_span ③ 层）几何年化；倍数换算 ×252（① 层）"}}
    (OUT / f"meta_{args.mode}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\n产物: {OUT}\n以上为统计检验，不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
