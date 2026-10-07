# -*- coding: utf-8 -*-
"""事件驱动漂移 —— 管道演练（事件源 = forecast 已到手 200 只子样本）。

⚠️ 首要目的是**证明管道可跑通 + 量化样本能否回答问题**，不是下有效性结论。
   结论段显式声明：样本仅覆盖全市场 3.58%，不可外推。

口径（项目方法论铁律）
--------------------
1. 事件 T = ``first_ann_date``（缺失回退 ``ann_date``）＝盘后披露；
   买入 = **T 之后第一个交易日开盘**（严格用交易日历取 next()，非 T+1 自然日）。
2. 价格口径 = **前复权（qfq）** ``close_adj``（U3 锚：sh600519 adj/close
   归一 0.793056→1.000000@末日）。T+1 开盘用「同日 close_adj/close」换算，
   同日因子恒定 ⇒ 恒等式，不引入跨期推测。
3. 基准 = **同市值层（买入日前一交易日 circ_mv 三分位）× 同申万一级行业**，
   同买点同持有期、层内等权，层内成员 <5 只则置 NaN（不静默兜底）。
4. 成本 = 单边 15bp / 20bp 两档；双边 30bp / 40bp。
5. 判据 = 7 窗口分布（胜率≥60% / 中位>0 / 最差>−5% / ≥5 窗）。
6. 多重比较 = n_trials 显式声明 + DSR（Bailey & López de Prado 2014，i.i.d. 自算件）。
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.config import is_a_share  # noqa: E402

RT = ROOT / "runtime" / "tushare"
HOLDS = (1, 3, 5, 10)
POS_TYPES = {"预增", "略增", "扭亏", "续盈"}
MIN_LAYER_MEMBERS = 5


# ══════════════════════════════════════════════════════════════
# 数据装载
# ══════════════════════════════════════════════════════════════
def ts_to_local(ts: str) -> str:
    """'000001.SZ' → 'sz000001'。"""
    return f"{ts[-2:].lower()}{ts[:6]}"


def local_to_ts(code: str) -> str:
    """'sz000001' → '000001.SZ'。"""
    return f"{code[2:]}.{code[:2].upper()}"


def load_events() -> pd.DataFrame:
    shards = sorted(RT.glob("forecast_shard*.parquet"))
    if not shards:
        raise FileNotFoundError("无 forecast 分片 —— 事件源缺失")
    d = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    d["ann_date"] = d["ann_date"].astype(int)
    d["end_date"] = d["end_date"].astype(int)
    d["T"] = d["first_ann_date"].fillna(d["ann_date"]).astype(int)
    d["side"] = np.where(d["type"].isin(POS_TYPES), "正面", "负面")
    d = (d.sort_values(["ts_code", "T", "ann_date"])
           .drop_duplicates(subset=["ts_code", "T", "end_date"], keep="first"))
    return d.reset_index(drop=True)


def load_price_panel(codes: list[str]) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect("file:data/market.db?mode=ro", uri=True)
    q = ("select code, date, open, close, close_adj from bar_daily "
         "where code in (%s) and date >= 20150101 order by code, date"
         % ",".join("?" * len(codes)))
    df = pd.read_sql(q, con, params=codes)
    con.close()
    df["ds"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    return {f: df.pivot_table(index="ds", columns="code", values=f,
                              aggfunc="last").sort_index()
            for f in ("open", "close", "close_adj")}


def load_layers() -> tuple[pd.DataFrame, pd.Series]:
    """(历史时点流畅市值面板[万元], 申万一级映射 code→l1_name)。"""
    db = pd.read_parquet(RT / "daily_basic.parquet",
                         columns=["ts_code", "trade_date", "circ_mv"])
    db["code"] = db["ts_code"].map(ts_to_local)
    db["ds"] = pd.to_datetime(db["trade_date"].astype(str), format="%Y%m%d")
    mv = db.pivot_table(index="ds", columns="code", values="circ_mv",
                        aggfunc="last").sort_index()
    ind = pd.read_parquet(RT / "index_member_all.parquet",
                          columns=["ts_code", "l1_name"])
    ind["code"] = ind["ts_code"].map(ts_to_local)
    return mv, ind.set_index("code")["l1_name"]


# ══════════════════════════════════════════════════════════════
# 收益与层内基准
# ══════════════════════════════════════════════════════════════
@dataclass
class Cal:
    """事件日历 + 全市场价格/分层。

    ⚠️ 存取**一律用 .loc / .iloc，禁止用 .at/.iat 配 Timestamp 标签**
       （实测踩过，2026-10-07）：``df.at[Timestamp(...), 'sz000021']``
       在「DatetimeIndex + 字符串列」的混合类型下会**退化为位置索引**，
       返回位置的整数值而不是该日该股的价格 —— 不报错、不复现，
       后果是复权因子恒等于错误的 1.0，基准收益被算成 -20%~+132% 的乱数。
       同 P12/P16 家族（构造器/索引方向陷阱）。
    """
    days: pd.DatetimeIndex
    cadj: pd.DataFrame
    open_: pd.DataFrame
    factor: pd.DataFrame          # close_adj / close（同日恒等）
    layer: pd.DataFrame           # 小/中/大
    l1: pd.Series

    def next_day(self, T: pd.Timestamp) -> pd.Timestamp | None:
        i = self.days.searchsorted(T, side="right")
        return self.days[i] if i < len(self.days) else None

    def day_index(self, d: pd.Timestamp) -> int:
        return int(self.days.searchsorted(d))

    def get(self, df: pd.DataFrame, d: pd.Timestamp,
            code: str) -> float:
        """安全取标量：df.loc[d, code]。"""
        if code not in df.columns or d not in df.index:
            return float("nan")
        return float(df.loc[d, code])


def build_event_returns(ev: pd.DataFrame, cal: Cal,
                        max_hold: int) -> pd.DataFrame:
    """逐事件算持有 1..max_hold 日的前复权收益 + 层内超额。

    基准 = **全市场**同市值层 × 同申万一级行业 的等权组合（不是从子样本里抽），
    否则层内成员不足会大面积缺格（实测：只用 189 只时缺格 78%）。
    """
    rows: list[dict] = []
    n_skip = {"无交易日": 0, "无价格": 0, "持有期越界": 0}

    for e in ev.itertuples(index=False):
        code = ts_to_local(e.ts_code)
        if code not in cal.cadj.columns:
            n_skip["无价格"] += 1
            continue
        T = pd.to_datetime(str(e.T), format="%Y%m%d")
        d1 = cal.next_day(T)
        if d1 is None:
            n_skip["无交易日"] += 1
            continue
        i1 = cal.day_index(d1)
        if i1 + max_hold >= len(cal.days):
            n_skip["持有期越界"] += 1
            continue
        p_in = cal.get(cal.open_, d1, code) * cal.get(cal.factor, d1, code)
        if not np.isfinite(p_in) or p_in <= 0:
            n_skip["无价格"] += 1
            continue

        rec = {"ts_code": e.ts_code, "code": code, "type": e.type,
               "side": e.side, "T": T, "buy_date": d1, "entry": p_in,
               "end_date": e.end_date, "i1": i1}
        for h in HOLDS:
            d_exit = cal.days[i1 + h]
            p_out = cal.get(cal.cadj, d_exit, code)
            rec[f"r{h}"] = p_out / p_in - 1 if np.isfinite(p_out) else np.nan
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        print("  跳过统计:", n_skip)
        return out
    print(f"  事件保留 {len(out):,} / {len(ev):,}  跳过明细 {n_skip}")

    # ── 每日预计算层内基准收益（全市场、层×行业 等权，只算一次）──
    bench_by_day: dict[tuple, dict[int, float]] = {}
    for d0 in sorted(out["buy_date"].unique()):
        i1 = cal.day_index(d0)
        d_ref = cal.days[i1 - 1]
        if d_ref not in cal.layer.index:
            continue
        L = cal.layer.loc[d_ref]
        f0 = cal.factor.loc[d0]
        o0 = cal.open_.loc[d0]
        p0 = (o0 * f0).reindex(cal.cadj.columns)      # 复权口径的 T+1 开盘
        cls: dict[tuple, list[str]] = {}
        for c in L.index:
            lay = L.loc[c]
            ind = cal.l1.get(c)
            if lay is None or (isinstance(lay, float) and np.isnan(lay)) or ind is None:
                continue
            if not np.isfinite(p0.get(c, np.nan)) or p0.get(c, 0) <= 0:
                continue
            cls.setdefault((lay, ind), []).append(c)
        for key, members in cls.items():
            if len(members) < MIN_LAYER_MEMBERS:
                continue
            d = {}
            for h in HOLDS:
                i2 = i1 + h
                if i2 >= len(cal.days):
                    continue
                p1 = cal.cadj.iloc[i2][members]
                ok = np.isfinite(p0[members].to_numpy()) & p1.notna().to_numpy()
                if ok.sum() < MIN_LAYER_MEMBERS:
                    continue
                d[h] = float((p1.to_numpy()[ok] / p0[members].to_numpy()[ok] - 1).mean())
            bench_by_day[(d0, key)] = d

    n_miss = 0
    bench: dict[int, list[float]] = {h: [] for h in HOLDS}
    for r in out.itertuples(index=False):
        i1 = cal.day_index(r.buy_date)
        d_ref = cal.days[i1 - 1]
        L = cal.layer.loc[d_ref] if d_ref in cal.layer.index else None
        key = None
        if L is not None:
            lay = L.get(r.code)
            ind = cal.l1.get(r.code)
            if lay is not None and ind is not None and not (
                    isinstance(lay, float) and np.isnan(lay)):
                key = (lay, ind)
        d = bench_by_day.get((r.buy_date, key), {}) if key else {}
        if not d:
            n_miss += 1
        for h in HOLDS:
            bench[h].append(d.get(h, np.nan))

    for h in HOLDS:
        out[f"b{h}"] = bench[h]
        out[f"x{h}"] = out[f"r{h}"] - out[f"b{h}"]
    print(f"  层内基准缺格 {n_miss:,} / {len(out):,}"
          f"（层内成员<{MIN_LAYER_MEMBERS} 或行业/市值/停牌缺失）")
    return out


# ══════════════════════════════════════════════════════════════
# 统计
# ══════════════════════════════════════════════════════════════
def deflated_sharpe(sr: float, n_obs: int, n_trials: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """DSR（Bailey & López de Prado 2014，i.i.d. 形式，自算件）。"""
    nd = NormalDist()
    if n_obs < 2 or n_trials < 1 or not np.isfinite(sr):
        return float("nan")
    emc = 0.5772156649015329
    e = math.e
    z = nd.inv_cdf(1 - 1.0 / n_trials)
    var = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if var <= 0:
        return float("nan")
    sr0 = math.sqrt(var) * ((1 - emc) * z + emc * nd.inv_cdf(1 - 1.0 / (n_trials * e)))
    return float(nd.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(var)))


def hit_rate_stats(x: pd.Series) -> dict:
    v = x.dropna()
    if v.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan, "win": np.nan,
                "worst": np.nan, "sd": np.nan}
    return {"n": len(v), "mean": float(v.mean()), "median": float(v.median()),
            "win": float((v > 0).mean()), "worst": float(v.min()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan}


def main() -> int:
    ap = argparse.ArgumentParser(description="事件驱动漂移管道演练")
    ap.add_argument("--max-hold", type=int, default=10)
    ap.add_argument("--out", default="runtime/event_drift/event_returns.csv")
    args = ap.parse_args()

    print("=" * 84)
    print("事件驱动漂移（业绩预告）· 管道演练")
    print("事件源 = runtime/tushare/forecast_shard000-003.parquet（已到手 200 只）")
    print("=" * 84)

    ev = load_events()
    print(f"\n[1] 事件装载 {len(ev):,} 条 / {ev['ts_code'].nunique()} 只"
          f"  正面 {int((ev['side']=='正面').sum()):,} / "
          f"负面 {int((ev['side']=='负面').sum()):,}")
    print(f"    T 范围 {ev['T'].min()} ~ {ev['T'].max()}")

    codes = sorted({ts_to_local(c) for c in ev["ts_code"].unique()})
    # ⚠️ 价格面板必须是**全市场**：基准是同市值层×同申万一级的层内等权，
    #    只装事件涉及的股票会让层内成员不足（实测缺格 78%）。
    con = sqlite3.connect("file:data/market.db?mode=ro", uri=True)
    all_a = [r[0] for r in con.execute("select distinct code from bar_daily")
             if is_a_share(r[0])]
    con.close()
    px = load_price_panel(sorted(all_a))
    mv, l1 = load_layers()
    days = px["close_adj"].index
    print(f"\n[2] 全市场价格面板 {px['close_adj'].shape[0]} 交易日 × "
          f"{px['close_adj'].shape[1]} 只（事件涉及 {len(codes)} 只）")
    # 层：按买入日可得的历史流通市值做当日横截面三分位
    mv = mv.reindex(days).ffill()
    rank = mv.rank(axis=1, pct=True)
    layer = pd.cut(rank.stack(), [0, 1/3, 2/3, 1.0],
                   labels=["小", "中", "大"]).unstack()
    layer = layer.reindex(columns=px["close_adj"].columns)
    factor = (px["close_adj"] / px["close"].replace(0, np.nan)
              ).reindex(columns=px["close_adj"].columns)
    cal = Cal(days=days, cadj=px["close_adj"], open_=px["open"],
              factor=factor, layer=layer, l1=l1)
    print(f"    市值层覆盖 {layer.notna().any(axis=1).sum()} 日；"
          f"申万一级映射 {l1.notna().sum()} 只")

    er = build_event_returns(ev, cal, args.max_hold)
    if er.empty:
        print("✗ 无有效事件")
        return 1

    # ── 中间抽查 + 锚点对拍 ──
    print("\n[3] 中间结果抽查（前 3 条）")
    show = ["ts_code", "type", "T", "buy_date", "entry", "r1", "r5",
            "b5", "x5"]
    print(er[show].head(3).to_string(index=False))

    chk = er.iloc[1]
    con = sqlite3.connect("file:data/market.db?mode=ro", uri=True)
    raw = pd.read_sql("select date, open, close, close_adj from bar_daily "
                      "where code=? and date=?", con,
                      params=[chk["code"],
                              int(chk["buy_date"].strftime("%Y%m%d"))])
    # 该事件的层内基准成员
    i1 = cal.day_index(chk["buy_date"])
    d_ref = cal.days[i1 - 1]
    L = cal.layer.loc[d_ref]
    members = [c for c in L.index[(L == L.get(chk["code"])).to_numpy()]
               if cal.l1.get(c) == cal.l1.get(chk["code"])
               and c in cal.cadj.columns]
    con.close()
    print(f"\n    对拍 {chk['code']} {chk['buy_date'].date()}：")
    print("      库内原始行:", raw.to_dict("records"))
    if len(raw):
        f = raw["close_adj"].iloc[0] / raw["close"].iloc[0]
        manual = raw["open"].iloc[0] * f
        print(f"      手算 open×同日因子 = {manual:.6f}   管道 entry = "
              f"{chk['entry']:.6f}   一致={abs(manual-chk['entry'])<1e-6}")
    print(f"      层内基准成员数 = {len(members)}"
          f"（行业={cal.l1.get(chk['code'])}，市值层={L.get(chk['code'])}）")
    print(f"      成员样例: {members[:8]}")
    print(f"      基准 b5 = {chk['b5']:.6f}   个股 r5 = {chk['r5']:.6f}"
          f"   超额 x5 = {chk['x5']:.6f}")

    # ── 全样本 ──
    print("\n[4] 全样本（200 只子样本）")
    print(f"{'持有':>4}{'n':>7}{'原始均值':>11}{'超额均值':>11}"
          f"{'超额中位':>11}{'超额胜率':>11}{'σ':>9}")
    print("-" * 66)
    for h in HOLDS:
        s = hit_rate_stats(er[f"x{h}"])
        print(f"{h:>4}{s['n']:>7}{er[f'r{h}'].mean():>11.3%}"
              f"{s['mean']:>11.3%}{s['median']:>11.3%}{s['win']:>11.1%}"
              f"{s['sd']:>9.3%}")

    # ── 分方向 ──
    print("\n[5] 分方向（正面=预增/略增/扭亏/续盈 vs 负面）")
    print(f"{'持有':>4}{'正面n':>7}{'正面超额':>11}{'负面n':>7}"
          f"{'负面超额':>11}{'多空差':>10}")
    print("-" * 64)
    for h in HOLDS:
        p = er.loc[er["side"] == "正面", f"x{h}"].dropna()
        n = er.loc[er["side"] == "负面", f"x{h}"].dropna()
        print(f"{h:>4}{len(p):>7}{p.mean():>11.3%}{len(n):>7}{n.mean():>11.3%}"
              f"{p.mean()-n.mean():>10.3%}")

    # ── 扣成本 ──
    print("\n[6] 扣成本（单位=每笔投入；双边 30bp 起算 + 40bp 敏感性档）")
    print(f"{'持有':>4}{'毛超额':>10}{'净@30bp':>11}{'净@40bp':>11}"
          f"{'年化(线性×252/h)':>18}")
    print("-" * 68)
    for h in HOLDS:
        g = er[f"x{h}"].dropna().mean()
        print(f"{h:>4}{g:>10.3%}{g-0.0030:>11.3%}{g-0.0040:>11.3%}"
              f"{(g-0.0030)*252/h:>18.2%}")

    # ── 7 窗口（按自然年序列切，测试段两两不重叠）──
    print("\n[7] 7 互不重叠窗口（按自然年分段：训练 3 年 / 测试 1 年 / 步进 1 年）")
    er["yr"] = er["buy_date"].dt.year
    years = sorted(er["yr"].unique())
    wins = [(y0, y0 + 1) for y0 in range(2019, 2026)]
    wrows = []
    for h in (5,):
        print(f"\n    持有 {h} 日：")
        print(f"    {'窗':>3}{'测试年':>8}{'事件':>7}{'股票':>6}"
              f"{'毛超额':>10}{'净@30bp':>11}{'胜率':>8}")
        print("    " + "-" * 56)
        for k, (y0, y1) in enumerate(wins):
            sub = er[(er["yr"] >= y0) & (er["yr"] < y1)]
            v = sub[f"x{h}"].dropna()
            wrows.append({
                "窗口": k, "测试年": y0, "事件数": len(sub),
                "股票数": sub["code"].nunique(),
                "毛超额": float(v.mean()) if len(v) else np.nan,
                "净超额": float(v.mean()) - 0.0030 if len(v) else np.nan,
                "胜率": float((v > 0).mean()) if len(v) else np.nan})
            r = wrows[-1]
            print(f"    {k:>3}{y0:>8}{r['事件数']:>7}{r['股票数']:>6}"
                  f"{r['毛超额']:>10.3%}{r['净超额']:>11.3%}{r['胜率']:>8.1%}")
        wdf = pd.DataFrame(wrows)
        exc = wdf["净超额"].dropna()
        print(f"\n    → 窗口数 {len(exc)}  胜率 {(exc>0).mean():.0%}  "
              f"中位 {exc.median():+.3%}  最差 {exc.min():+.3%}  "
              f"最好 {exc.max():+.3%}")
        print(f"    判据（胜率≥60% ∧ 中位>0 ∧ 最差>−5% ∧ ≥5 窗）: ", end="")
        ok = ((exc > 0).mean() >= 0.60 and exc.median() > 0
              and exc.min() > -0.05 and len(exc) >= 5)
        print("全部通过" if ok else "未通过")

    # ── DSR ──
    print("\n[8] 多重比较：Deflated Sharpe（Bailey & López de Prado 2014, i.i.d.）")
    n_trials = len(HOLDS) * 2
    print(f"    n_trials = {n_trials}（4 持有期 × 2 方向）")
    for h in HOLDS:
        v = er[f"x{h}"].dropna()
        if len(v) < 3 or v.std(ddof=1) == 0:
            continue
        sr = v.mean() / v.std(ddof=1)
        dsr = deflated_sharpe(sr, len(v), n_trials,
                              skew=float(v.skew()),
                              kurt=float(v.kurtosis() + 3))
        print(f"    持有{h:>3}日  单期SR={sr:+.4f}  n={len(v):>5}  DSR={dsr:.4f}")

    # ── 三段 ──
    print("\n[9] 三段子区间（持有 5 日，净@双边30bp）")
    for name, (a, b) in {"2016-2018": (2016, 2018), "2019-2022": (2019, 2022),
                         "2023-2026": (2023, 2026)}.items():
        sub = er[(er["yr"] >= a) & (er["yr"] <= b)]
        v = sub["x5"].dropna()
        if not len(v):
            print(f"    {name}: 无样本")
            continue
        print(f"    {name}: n={len(v):<5} 毛 {v.mean():+.3%}  "
              f"净 {v.mean()-0.0030:+.3%}  胜率 {(v>0).mean():.1%}")

    # ── 功效 ──
    print("\n[10] 统计功效：这份样本能不能回答问题")
    v = er["x5"].dropna()
    sd, n = float(v.std(ddof=1)), len(v)
    for target in (0.01, 0.02):
        need = math.ceil(((1.96 + 0.84) * sd / target) ** 2)
        print(f"    检出 {target:.0%} 漂移（5% 显著 / 80% 功效）需 "
              f"{need:,} 个独立事件")
    print(f"    现有 n={n:,} 独立事件，σ={sd:.3%} → 可检出最小漂移 "
          f"≈ {(1.96+0.84)*sd/math.sqrt(n):.2%}")
    print(f"    子样本覆盖 200/5,591 = 3.58% 标的 ⇒ 全市场事件数约为其 "
          f"{5591/200:.0f} 倍（毛估 {n*5591//200:,} 个）")

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    er.to_csv(outp, index=False, encoding="utf-8-sig")
    pd.DataFrame(wrows).to_csv(outp.with_name("windows.csv"), index=False,
                               encoding="utf-8-sig")
    print(f"\n产物: {outp}")
    print("\n⚠️ 本样本仅覆盖全市场 3.58%（仅深市 000/001/002 前缀），"
          "**不可外推为有效性结论**。历史统计不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
