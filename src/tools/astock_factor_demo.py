"""
A 股多因子选股 —— 最小可运行实现（Python + AkShare / Tushare）

⚠️ 先读这段，再决定要不要跑
═══════════════════════════════════════════════════════════════════
本项目在 2016-2026 全市场 4,925 只 A 股上实测的结论（详见
docs/02_方法与结论/03_A股量化算法全景对比.md）：

  · 价值因子 bp 是唯一三段扣成本后净收益全正的因子
    （+1.0% / +7.1% / +2.2% 年化，ICIR 0.330）
  · ep 净收益 +6.1%，但路径是「+33% → +0.3% → −5%」，2019 年后失效
  · roe 净收益集中在 2016-2018，之后两段为负
  · 价量类（动量/反转/波动率）在 A 股扣成本后无一为正
  · A 股 T+1 导致日内动量与隔夜反向，月频动量在总收益层面相互抵消

所以下面这份代码**不追求高收益**，而追求「把上面这些结论复现出来」。
如果你只想找「能赚钱的策略」，请先看第七节的三道必问。

用法
----
  pip install akshare pandas numpy alphalens-reloaded pyarrow
  python astock_factor_demo.py --source akshare --mode value
  python astock_factor_demo.py --source akshare --mode all      # 全因子
  python astock_factor_demo.py --source tushare --token 你的token

数据源说明
----------
  AkShare  : 免费无需注册，但接口不稳定（东财系常需代理），历史财务
             只有报告期没有披露日 → 本文件用【固定 45 天延迟】近似，
             详见 _disclosure_lag() 的警告
  Tushare  : 有真实披露日字段（ann_date/f_ann_date），更严谨，
             但需要积分（1 元 = 10 积分，权限中心需登录）

本文件的财务因子必须自行处理「披露延迟」，否则就是前视偏差。
═══════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

TRADING_DAYS_PER_YEAR = 252

# ── A 股交易成本（务必按自己的实际费率调整）──────────────────────
# 佣金万2.5（双边） + 印花税千0.5（仅卖出） + 滑点万5（双边估算）
COST_ROUND_TRIP = 0.0020  # 20bp

# ── 因子定义 ────────────────────────────────────────────────────
FACTORS = {
    "bp":  "价值：账面市值比 = 每股净资产 / 股价。买最便宜的",
    "ep":  "价值：盈利收益率 = 每股收益 / 股价",
    "roe": "质量：净资产收益率",
    "gross_margin": "定价权：销售毛利率",
    "profit_yoy": "成长：净利润同比增速",
    "cf_quality": "盈利质量：每股经营现金流 / 每股收益",
}


# ════════════════════════════════════════════════════════════════
# 1. 数据获取
# ════════════════════════════════════════════════════════════════
def load_prices_akshare(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """用 AkShare 取日线，返回 close 宽表（index=date, columns=code）。

    ⚠️ AkShare 的股票代码是 6 位裸码，不带 sh/sz 前缀。
       本文件统一用 6 位裸码做列名，与 AkShare 保持一致。
    """
    import akshare as ak
    import time

    close = {}
    for i, code in enumerate(codes):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=start.replace("-", ""),
                                    end_date=end.replace("-", ""),
                                    adjust="hfq")  # 后复权，规避除权跳空
        except Exception as e:  # noqa: BLE001
            print(f"    {code} 失败: {e}")
            continue
        if df is None or df.empty:
            continue
        s = pd.to_numeric(df["收盘"], errors="coerce")
        s.index = pd.to_datetime(df["日期"])
        close[code] = s
        if (i + 1) % 50 == 0:
            print(f"    已取 {i+1}/{len(codes)}，暂停 1s防限流")
            time.sleep(1)
    if not close:
        raise RuntimeError("未取到任何行情，请检查网络或代理")
    return pd.DataFrame(close).sort_index()


def load_prices_tushare(codes: list[str], start: str, end: str,
                        token: str) -> pd.DataFrame:
    """用 Tushare 取日线。

    Tushare 代码需带交易所后缀：600519 → 600519.SH
    """
    import tushare as ts
    pro = ts.pro_api(token)

    df = pro.daily(
        ts_code=",".join(_to_tushare_code(c) for c in codes),
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
    )
    if df is None or df.empty:
        raise RuntimeError("Tushare 返回空。请确认积分是否够 daily 接口权限")
    df["date"] = pd.to_datetime(df["trade_date"])
    return df.pivot(index="date", columns="ts_code", values="close").sort_index()


def _to_tushare_code(code6: str) -> str:
    c = str(code6).zfill(6)
    if c.startswith(("6", "9")):
        return f"{c}.SH"
    if c.startswith(("0", "3")):
        return f"{c}.SZ"
    if c.startswith(("4", "8", "92")):
        return f"{c}.BJ"
    return f"{c}.SZ"


def load_financials_akshare(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """用 AkShare 取业绩报表（东财 stock_yjbb_em 接口）。

    ⚠️⚠️ 关键：返回的数据只有 report_period（报告期），没有实际披露日。
       必须自行施加披露延迟，见 _disclosure_lag()。
       这是 A 股量化研究最常见的前视偏差来源。
    """
    import akshare as ak
    import time

    periods = _quarter_ends(start, end)
    parts = []
    for rp in periods:
        df = ak.stock_yjbb_em(date=rp.strftime("%Y%m%d"))
        if df is None or df.empty:
            continue
        df["report_period"] = rp
        parts.append(df)
        print(f"    {rp:%Y-%m-%d} → {len(df):,} 行")
        time.sleep(0.6)  # 东财限流较严
    if not parts:
        raise RuntimeError("未取到财务数据。AkShare 的东财接口常需代理")
    d = pd.concat(parts, ignore_index=True)
    d["code"] = d["code"].astype(str).str.extract(r"(\d{6})", expand=False)
    return d[d["code"].isin(codes)]


def _quarter_ends(start: str, end: str) -> list[pd.Timestamp]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    y = s.year
    while y <= e.year:
        for m, d in (3, 6, 9, 12):
            ts = pd.Timestamp(y, m, d)
            if s <= ts <= e:
                out.append(ts)
        y += 1
    return out


# ════════════════════════════════════════════════════════════════
# 2. 披露延迟 —— 前视偏差的唯一防线
# ════════════════════════════════════════════════════════════════
def disclosure_lag(report_period: pd.Timestamp) -> pd.Timestamp:
    """报告期 → 该报告【最早可能已公开】的日期。

    沪深主板法定披露截止日：
      一季报 03-31 → 当年 04-30
      半年报 06-30 → 当年 08-31
      三季报 09-30 → 当年 10-31
      年报   12-31 → 次年 04-30

    ⚠️ 千万不要用「报告期 + 固定天数」。本项目初版用 +45 天，
       对年报提前了 2.5 个月、对半年报提前 17 天 ——
       等于回测时用了尚未公布的信息，会系统性高估基本面因子。

    若用 Tushare，直接用它的 ann_date / f_ann_date 字段更严谨。
    """
    m = report_period.month
    if m == 3:
        return pd.Timestamp(report_period.year, 4, 30)
    if m == 6:
        return pd.Timestamp(report_period.year, 8, 31)
    if m == 9:
        return pd.Timestamp(report_period.year, 10, 31)
    return pd.Timestamp(report_period.year + 1, 4, 30)


# ════════════════════════════════════════════════════════════════
# 3. 因子构建
# ════════════════════════════════════════════════════════════════
def winsorize_mad(s: pd.Series, n: float = 5.0) -> pd.Series:
    """单日横截面 MAD 去极值。财务同比增速可达 1e9%（业绩暴雷/重组）。"""
    v = s.dropna()
    if len(v) < 20:
        return s
    med = v.median()
    mad = (v - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    return s.clip(med - n * 1.4826 * mad, med + n * 1.4826 * mad)


def build_factor_panel(fin: pd.DataFrame, prices: pd.DataFrame,
                       name: str) -> pd.Series:
    """把季度财务展开成日频因子，返回 index=(date, asset) 的 Series。"""
    f = fin.copy()
    f["report_period"] = pd.to_datetime(f["report_period"])
    f["available_date"] = f["report_period"].map(disclosure_lag)

    num = {"roe": "净资产收益率",
           "profit_yoy": "净利润-同比增长",
           "gross_margin": "销售毛利率",
           "eps": "每股收益",
           "bps": "每股净资产",
           "cfps": "每股经营现金流"}
    col = num.get(name, name)
    if col not in f.columns:
        raise KeyError(f"财务表里没有 {col}。实际字段: {list(f.columns)[:20]}")
    f["v"] = pd.to_numeric(f[col], errors="coerce")

    # 派生
    if name == "cf_quality":
        eps = pd.to_numeric(f["每股收益"], errors="coerce")
        f["v"] = f["v"] / eps.replace(0, np.nan)

    # 估值倍数用当日价
    if name == "bp":
        wide = f.pivot_table(index="available_date", columns="code",
                             values="bps", aggfunc="last").sort_index()
    elif name == "ep":
        wide = f.pivot_table(index="available_date", columns="code",
                             values="eps", aggfunc="last").sort_index()
    else:
        wide = f.pivot_table(index="available_date", columns="code",
                             values="v", aggfunc="last").sort_index()

    union = wide.index.union(prices.index).sort_values()
    aligned = wide.reindex(union).ffill().reindex(prices.index)
    px = prices.reindex(columns=aligned.columns)

    if name == "bp":
        with np.errstate(divide="ignore", invalid="ignore"):
            aligned = aligned / px          # B/P = BPS / Price
    elif name == "ep":
        with np.errstate(divide="ignore", invalid="ignore"):
            aligned = aligned / px          # E/P = EPS / Price

    out = aligned.stack(future_stack=True)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.groupby(level="date").transform(winsorize_mad)
    out.name = name
    return out


def neutralize(s: pd.Series, industry: pd.Series | None = None) -> pd.Series:
    """行业中性化：逐日对因子在行业组内去均值。

    ⚠️ 本项目踩过的坑：A 股行业标签多、每行业股票少，
       若再叠加市值分位做「行业 × 市值」细分组，每格往往只有 2 只股票，
       组内去均值会把它们全压成 0.0；加最小组数阈值后中性化又退化成
       恒等变换，且【输出与原始因子一字不差】，极易误判。
       → 若要做市值中性，请用逐日截面 OLS 残差法，不要用细分组去均值。
       这里只做行业去均值（行业粒度足够，不会踩这个坑）。
    """
    if industry is None:
        return s
    df = pd.DataFrame({"f": s, "ind": industry.reindex(s.index)})
    df["_date"] = df.index.get_level_values("date")
    df["_n"] = df.groupby(["_date", "ind"])["f"].transform("size")
    df["_mu"] = df.groupby(["_date", "ind"])["f"].transform("mean")
    # 组内不足 20 只就不去均值（该组贡献横截面排序信息而非被压平）
    out = np.where(df["_n"] >= 20, df["f"] - df["_mu"], df["f"])
    return pd.Series(out, index=s.index, name=s.name)


# ════════════════════════════════════════════════════════════════
# 4. 检验
# ════════════════════════════════════════════════════════════════
def ic_and_quantiles(factor: pd.Series, prices: pd.DataFrame,
                     quantiles: int = 5, periods: tuple = (1, 5, 20, 60),
                     cost: float = COST_ROUND_TRIP) -> dict:
    """IC + 分组多空收益 + 扣成本净收益。

    ⚠️⚠️ 年化口径（本项目踩过最贵的坑）：
       前瞻 k 日的收益要乘 (252 / k) 才是年化。
       直接把每期差值当年化，数值会被低估约 252 倍，
       且很容易据此得出「扣成本后归零」的错误结论。
    """
    from alphalens import performance, utils

    clean = utils.get_clean_factor_and_forward_returns(
        factor=factor, prices=prices, quantiles=quantiles,
        periods=list(periods), max_loss=0.5,
    )

    ic_daily = performance.factor_information_coefficient(clean)
    ic = {}
    for k in periods:
        col = f"{k}D"
        if col in ic_daily.columns:
            s = ic_daily[col].dropna()
            if len(s):
                sd = s.std()
                ic[k] = {
                    "IC": s.mean(),
                    "ICIR": s.mean() / sd if sd else np.nan,
                    "t": s.mean() / sd * np.sqrt(len(s)) if sd else np.nan,
                    "n": len(s),
                }

    qr, _ = performance.mean_return_by_quantile(clean)
    tr = pd.DataFrame({q: performance.quantile_turnover(clean["factor_quantile"], q)
                       for q in range(1, quantiles + 1)})
    turnover = float(tr.mean().mean())

    out = {"ic": ic, "turnover": turnover, "periods": {}}
    for k in periods:
        col = f"{k}D"
        if col not in qr.columns:
            continue
        spread = float(qr.loc[quantiles, col] - qr.loc[1, col])
        ppy = TRADING_DAYS_PER_YEAR / k          # ← 正确的年化系数
        gross = spread * ppy
        net = gross - cost * turnover * quantiles * ppy
        out["periods"][k] = {
            "Q1": float(qr.loc[1, col]), "Q5": float(qr.loc[quantiles, col]),
            "多空年化毛": gross, "多空年化净": net,
        }
    return out


def subperiod_stability(factor: pd.Series, prices: pd.DataFrame,
                        sub_periods: dict, cost: float) -> pd.DataFrame:
    """子区间检验：全样本有效 ≠ 稳定（AGENTS.md 强制要求）。"""
    rows = []
    for label, (s, e) in sub_periods.items():
        rng = pd.date_range(s, e)
        f = factor[factor.index.get_level_values("date").isin(rng)]
        p = prices.loc[prices.index.intersection(rng)]
        if f.empty or p.empty:
            rows.append({"区间": label, "IC": np.nan, "ICIR": np.nan,
                         "多空年化净": np.nan})
            continue
        try:
            r = ic_and_quantiles(f, p, cost=cost)
            k = r["periods"].get(1, {})
            i1 = r["ic"].get(1, {})
            rows.append({"区间": label,
                         "标的数": f.index.get_level_values("asset").nunique(),
                         "IC": i1.get("IC"), "ICIR": i1.get("ICIR"),
                         "多空年化毛": k.get("多空年化毛"),
                         "多空年化净": k.get("多空年化净")})
        except Exception as ex:  # noqa: BLE001
            rows.append({"区间": label, "IC": np.nan, "ICIR": np.nan,
                         "多空年化净": np.nan, "错误": str(ex)[:40]})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# 5. 主流程
# ════════════════════════════════════════════════════════════════
SUB_PERIODS = {
    "2016-2018": ("2016-01-01", "2018-12-31"),
    "2019-2022": ("2019-01-01", "2022-12-31"),
    "2023-2026": ("2023-01-01", "2026-09-30"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="A 股多因子选股最小实现")
    ap.add_argument("--source", choices=["akshare", "tushare"], default="akshare")
    ap.add_argument("--token", default=None, help="Tushare token")
    ap.add_argument("--mode", default="bp",
                    help=f"因子名，可选 {list(FACTORS)} 或 all")
    ap.add_argument("--n", type=int, default=300, help="股票数（先小样本验证）")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-09-30")
    args = ap.parse_args()

    print("=" * 70)
    print("A 股多因子选股 —— 最小可运行实现")
    print("=" * 70)
    print(f"数据源 {args.source}   因子 {args.mode}   股票数 {args.n}")
    print(f"区间 {args.start} ~ {args.end}")
    print(f"往返成本 {COST_ROUND_TRIP*10000:.0f}bp")
    print("披露延迟：按法定披露截止日（年报次年 4/30）")

    # 1) 取股票列表
    if args.source == "akshare":
        import akshare as ak
        codes = [c.zfill(6) for c in ak.stock_info_a_code_name()["code"].tolist()]
    else:
        import tushare as ts
        pro = ts.pro_api(args.token)
        codes = pro.stock_basic(exchange="", list_status="L",
                                fields="ts_code")["ts_code"].tolist()[: args.n]
    codes = [c for c in codes if c][: args.n]
    print(f"\n[1/5] 股票池 {len(codes)} 只")

    # 2) 取行情
    print("\n[2/5] 取日线（首次运行会较慢，AkShare 需逐只请求）…")
    if args.source == "akshare":
        prices = load_prices_akshare(codes, args.start, args.end)
    else:
        prices = load_prices_tushare(codes, args.start, args.end, args.token)
    print(f"      价格宽表 {prices.shape}")

    # 3) 取财务
    print("\n[3/5] 取季度财务…")
    fin = load_financials_akshare(codes, args.start, args.end)
    print(f"      财务 {len(fin):,} 行，{fin['report_period'].nunique()} 期")

    # 4) 检验
    names = list(FACTORS) if args.mode == "all" else args.mode.split(",")
    print(f"\n[4/5] 检验因子: {', '.join(names)}")
    for name in names:
        print("\n" + "-" * 70)
        print(f"因子: {name}  —— {FACTORS.get(name, '')}")
        print("-" * 70)
        try:
            factor = build_factor_panel(fin, prices, name)
            factor = factor[factor.index.get_level_values("date").isin(prices.index)]
            print(f"  有效值 {len(factor):,}")
            if len(factor) < 1000:
                print("  ✗ 有效值过少，跳过")
                continue
            r = ic_and_quantiles(factor, prices)
            print(f"  平均换手 {r['turnover']*100:.2f}%")
            print(f"  {'期限':>6} {'IC':>9} {'ICIR':>8} {'t':>8} {'年化毛':>10} {'年化净':>10}")
            for k, v in r["ic"].items():
                q = r["periods"].get(k, {})
                print(f"  {k:>5}日 {v['IC']:>+9.4f} {v['ICIR']:>+8.3f} "
                      f"{v['t']:>+8.2f} {q.get('多空年化毛', float('nan')):>9.1%} "
                      f"{q.get('多空年化净', float('nan')):>9.1%}")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  ✗ 失败: {e}")
            traceback.print_exc()
            continue

        print(f"\n  子区间稳定性（{name}）:")
        try:
            sub = subperiod_stability(factor, prices, SUB_PERIODS, COST_ROUND_TRIP)
            print("    " + sub.to_string(index=False).replace("\n", "\n    "))
        except Exception as e:  # noqa: BLE001
            print(f"    失败: {e}")

    # 5) 必读
    print("\n" + "=" * 70)
    print("第七节 · 看到任何回测数字前，先问三个问题")
    print("=" * 70)
    print("""
  1. 换手率是多少？
     没有换手率的收益数字无法评估成本。A 股往返约 20bp，
     换手 10%/日 → 年成本 20%，几乎吃掉任何毛收益。

  2. N 次试验才找到这个最优参数？
     试 7 组参数/2年 → 样本外 Sharpe 归零（Bailey et al. 2014）。
     不披露试验次数的回测无法评估过拟合。

  3. 扣掉成本后还剩多少？子区间还成立吗？
     本项目实测：ep 全样本净 +6.1%，但三段是 +33% / +0.3% / −5%。
     均值为正，路径不可持有。

  ⚠️ 本项目所有「多空收益」都是做多高分位 + 做空低分位的口径。
     A 股散户无法做空，实际只能拿多头部分，还要再扣自己那半成本。

  ⚠️ 实盘通常比修正后的回测差 30%~50%。回测夏普 1.2，实盘按 0.6~0.8 规划。
""")
    print("本文件为研究演示，不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())