"""财务因子（价值/质量/成长/盈利质量）有效性检验。

复用 alphalens-reloaded 的 IC / 分组 / 换手率计算，本模块只做财务特有的处理：

  1. **报告期对齐**：财务数据按 available_date（法定披露截止日）ffill 到交易日，
     保证任一天看到的都是「当时已公布」的最新财报。
  2. **去极值（winsorize）**：财务同比增速（profit_yoy/revenue_yoy）
     极端值可达 1e9%（业绩暴雷或重组），直接进 Spearman 会污染整个横截面。
     采用 MAD 法（median ± 5×1.4826×MAD）而非分位裁剪——因为分布本就重尾，
     分位裁剪会把有效信息也删掉。
  3. **行业中性化**：A 股行业轮动极强，不中性化的话「财务因子」可能只是
     「行业因子」的代理变量。做法是逐日对每个行业做横截面去均值（demean by industry）。
  4. **PB / PE 的价格对齐**：PB = 当日收盘价 / bps，必须用当日价，不能用财报期价格。

用法
----
  uv run python scripts/run_financial_study.py --n 300          # 冒烟
  uv run python scripts/run_financial_study.py --all
  uv run python scripts/run_financial_study.py --all --neutral   # 行业中性
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factor_lab.analysis.alphalens_adapter import run_tear_sheet, subperiod_ic
from factor_lab.config import DEFAULT_COST, DEFAULT_RESEARCH, OUTPUT_DIR, is_a_share
from factor_lab.data import all_codes, load_long, load_prices, load_stock_info
from factor_lab.data.universe import build_universe, summarize_universe

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 220)

PANEL = Path(__file__).parent.parent / "out" / "financial_panel.parquet"


# ── 工具 ────────────────────────────────────────────────────────
def winsorize_mad(s: pd.Series, n: float = 5.0) -> pd.Series:
    """MAD 去极值。s 为单日横截面序列。"""
    v = s.dropna()
    if len(v) < 20:
        return s
    med = v.median()
    mad = (v - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    lo, hi = med - n * 1.4826 * mad, med + n * 1.4826 * mad
    return s.clip(lower=lo, upper=hi)


def demean_by_group(df: pd.DataFrame, keys: list[str], by: str) -> pd.DataFrame:
    """按 by 列的取值在 keys 指定的横截面内去均值。

    df 需为长表，含 keys 列与 by 列。
    """
    idx_cols = keys
    g = df.groupby([*idx_cols, by])[list(df.columns.difference(idx_cols + [by]))]
    return g.transform(lambda x: x - x.mean())


def build_financial_factors(
    panel: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    prices_close: pd.DataFrame,
    winsor: bool = True,
    shares: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """把季度财务面板展开为日频因子 Series（index=(date, asset)）。

    流程
    ----
    1. **一次性全局 asof 对齐**：把面板 pivot 成 index=available_date 的宽表，
       在【全部交易日 + 全部财报可用日】的并集轴上 reindex + ffill，
       再取交易日行。这样得到的是「任一交易日看到的都是当时已公布的最新财报」。
       ⚠️ 为什么不用逐标的 merge_asof：5,593 只 × 2,611 日逐个循环，
          中等样本就要几十秒，全市场不可接受。全局 ffill 是 O(N log N)，
          且结果完全等价（同一天所有标的共享同一个时间轴）。
    2. 计算 PB / EP / BP（用当日收盘价 —— 这是唯一正确的价格口径）。
    3. MAD 去极值（同比增速极端值可达 1e9%）。
    4. 组装 (date, asset) MultiIndex Series。
    """
    raw_cols = ["eps", "bps", "roe", "profit_yoy",
                "revenue_yoy", "gross_margin", "cf_quality"]

    # ⚠️ 踩坑：pivot_table 会把缺失的 available_date（NaT）当成合法索引值，
    #    之后 MultiIndex.from_product 会对含 NaT 的 DatetimeIndex 调 isna 而
    #    抛 NotImplementedError。pandas 不会静默丢弃，必须显式剔。
    n_before = len(panel)
    panel = panel[panel["available_date"].notna() & panel["sym"].notna()]
    if len(panel) < n_before:
        print(f"      剔除 available_date 缺失 {n_before - len(panel):,} 行")

    p = panel.sort_values("available_date")
    wide = p.pivot_table(index="available_date", columns="sym",
                         values=raw_cols, aggfunc="last")
    # ⚠️ 踩坑：pivot_table 传 values=[多列] 时 columns 变成 MultiIndex(字段, sym)，
    #    且展平顺序是【字段主序】（先所有 eps 列，再所有 bps 列……）。
    #    直接 .to_numpy().reshape(-1, 7) 会把不同字段的数混在一起。
    #    正解是逐字段取子表再 stack。
    if not isinstance(wide.columns, pd.MultiIndex):
        raise TypeError(f"预期 MultiIndex 列，实际 {type(wide.columns)}")
    wide = wide.sort_index()

    # 并集轴：财报可用日 ∪ 交易日 → ffill → 取交易日
    union = wide.index.union(trading_days).sort_values()

    aligned: dict[str, pd.DataFrame] = {}
    for c in raw_cols:
        sub = wide[c].reindex(union).ffill().reindex(trading_days)
        aligned[c] = sub

    dates = trading_days
    codes = wide.columns.get_level_values("sym").unique()
    stacked = {c: sub.reindex(columns=codes).stack(future_stack=True)
               for c, sub in aligned.items()}
    daily = pd.DataFrame(stacked)
    daily.index.names = ["date", "asset"]

    # 行业标签：取该交易日最新财报的行业（仅用于中性化，缺失填 'NA'）
    ind_wide = p.pivot_table(index="available_date", columns="sym",
                             values="industry", aggfunc="last").sort_index()
    ind_wide = ind_wide.reindex(union).ffill().reindex(dates)
    ind_wide = ind_wide.reindex(columns=codes)
    ind_series = ind_wide.stack(future_stack=True).reindex(daily.index)
    daily["industry"] = ind_series.to_numpy()

    # 当日收盘价（价格对齐）
    close = prices_close.reindex(dates).reindex(columns=codes)
    close_s = close.stack(future_stack=True).reindex(daily.index)
    daily["close"] = close_s.to_numpy()

    # ── 规模（市值）代理变量：收盘价 × 总股本 ────────────────────
    # ⚠️ 已知局限：shares 来自通达信【当前】快照，不是历史股本。
    #    严格做法需要逐期股本（可用 akshare 历史股本，但覆盖不全）。
    #    股本变动相对缓慢且多数为增发/回购，用当前值近似是可接受的误差，
    #    但它【不是】无偏的，因此市值中性结论只能当稳健性参考。
    # ⚠️ 顺序坑：必须在按 close 有效性过滤【之前】算，否则长度不一致会广播失败。
    shares_map = shares.reindex(codes) if shares is not None else None
    if shares_map is not None:
        sh = shares_map.to_numpy(dtype=float)
        # stack 顺序 = (date 外层, code 内层)，与 daily 索引顺序一致
        daily["mktcap"] = daily["close"].to_numpy() * np.tile(sh, len(dates))

    daily = daily[np.isfinite(daily["close"]) & (daily["close"] > 0)].copy()
    daily["industry"] = daily["industry"].fillna("NA").astype(str)

    with np.errstate(divide="ignore", invalid="ignore"):
        daily["pb"] = daily["close"] / daily["bps"].replace(0, np.nan)
        daily["ep"] = daily["eps"] / daily["close"]      # 盈利收益率 E/P
        daily["bp"] = 1.0 / daily["pb"]                  # 账面市值比 B/P

    fcols = ["bp", "ep", "roe", "profit_yoy",
             "revenue_yoy", "gross_margin", "cf_quality"]
    if winsor:
        for c in fcols:
            daily[c] = daily.groupby(level="date")[c].transform(winsorize_mad)

    out: dict[str, pd.Series] = {}
    for c in fcols:
        s = daily[c].replace([np.inf, -np.inf], np.nan).dropna()
        out[c] = s
    out["_daily"] = daily
    return out


MIN_GROUP_SIZE = 5
"""中性化时的最小组内样本数（见 neutralize 的说明）。

取值权衡：格子 = 行业 × 市值分位。全市场约 4,900 只、约 100 个行业标签、
市值分 5 档 → 平均每格约 10 只。因此阈值不能设大（20 会让中性化几乎失效），
也不能设 1~2（会把整个横截面压成常数）。5 是可用折中。
"""


def neutralize(s: pd.Series, industry: np.ndarray,
               mktcap: np.ndarray | None = None,
               min_group: int = MIN_GROUP_SIZE) -> pd.Series:
    """行业中性化（可选叠加市值中性化）。

    为什么必须做：财务因子与市值、行业高度共线。
      · bp（账面市值比）天然高 ↔ 市值小；2017 年后 A 股小市值股大幅跑输，
        不控市值的话「价值因子」实测到的其实是「小市值因子」。
      · roe 天然高 ↔ 传统行业（银行、地产）；行业轮动会让结果被行业主导。
    做法：逐日对因子在 (行业 × 市值分位组) 内去均值。
          组内样本 < min_group 的格子【不去均值】、保留原值——否则细分组里
          只有 1~2 只股票时，去均值会把它们全压成 0.0，整个横截面变成常数，
          alphalens 分组全部 NaN，报 MaxLossExceededError 100%。

    ⚠️ 踩坑：索引是 MultiIndex(date, asset)，`groupby(cols, level=0)` 会把
       【列名当索引层】去映射（"ind" 在第 0 层找不到），
       报 'numpy.ndarray' object is not callable。
       正确做法是把 date 从索引还原成普通列，再按列 groupby。
    """
    ind = pd.Series(index=s.index, data=industry)
    tmp = pd.DataFrame({"f": s, "ind": ind})
    tmp["_date"] = tmp.index.get_level_values("date")
    if mktcap is not None:
        tmp["cap"] = pd.Series(index=s.index, data=mktcap)
        valid = tmp["cap"].where(np.isfinite(tmp["cap"]) & (tmp["cap"] > 0))
        # ⚠️ 关键：市值分位必须在【各行业内部】计算。
        #    早期版本用全截面排名，结果在「行业 × 市值分位」组内去均值时
        #    把因子几乎抹平（每日横截面 nunique 中位数 = 1，分组完全失效）。
        #    原因：组内市值同质 → 组内因子均值≈该行业该市值层均值 → 全被减掉。
        tmp["capq"] = (valid.groupby([tmp["_date"], tmp["ind"]])
                       .rank(pct=True).fillna(0.5))
        keys = ["_date", "ind", "capq"]
    else:
        keys = ["_date", "ind"]
    grp = tmp.groupby(keys)["f"]
    cnt = grp.transform("size")
    mu = grp.transform("mean")
    # ⚠️ np.where 返回 ndarray，必须包回 Series 才能保留 (date, asset) 索引
    out = np.where(cnt >= min_group, tmp["f"] - mu, tmp["f"])
    return pd.Series(out, index=s.index, name=s.name)


def pick_sample(codes: list[str], n: int, seed: int = 42) -> list[str]:
    """从 A 股代码里抽 n 只做冒烟。

    ⚠️ 踩坑记录（2026-10-05）：早期版本直接取 `codes[:n]`，
       而 codes 是按代码排序的，bj4xxxxx（北交所旧代码段）排在最前，
       于是 n=200 抽出来的全是北交所股票 → 与财务面板交集为 0，
       且北交所 2021 年才有数据，整个检验直接空跑。
       → 改为【分层抽样】：先按市场/板块分层，各层等比例抽。

    分层依据：sh / sz / bj 三个市场。
    """
    if n <= 0 or n >= len(codes):
        return codes
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[str]] = {}
    for c in codes:
        buckets.setdefault(c[:2], []).append(c)
    # 按层大小比例分配名额，每层至少 1 只
    total = len(codes)
    out: list[str] = []
    for mkt, lst in sorted(buckets.items()):
        k = max(1, round(n * len(lst) / total))
        k = min(k, len(lst))
        idx = rng.choice(len(lst), size=k, replace=False)
        out.extend(lst[i] for i in sorted(idx))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="财务因子有效性检验")
    ap.add_argument("--n", type=int, default=300, help="股票池上限，0/--all = 全市场（分层抽样）")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--neutral", action="store_true",
                    help="行业 + 市值 双中性化")
    ap.add_argument("--factors", default="bp,ep,roe,profit_yoy,revenue_yoy,gross_margin,cf_quality")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = type(DEFAULT_RESEARCH)(**{**DEFAULT_RESEARCH.__dict__,
                                    "start_date": args.start, "end_date": args.end})
    full = args.all or args.n == 0
    names = [x.strip() for x in args.factors.split(",") if x.strip()]

    print("=" * 74)
    print("财务因子有效性检验（价值 / 质量 / 成长 / 盈利质量）")
    print("=" * 74)
    print(f"区间 {cfg.start_date} ~ {cfg.end_date}   分组 {cfg.quantiles}   "
          f"前瞻期 {cfg.periods}   往返成本 {DEFAULT_COST.round_trip*10000:.0f}bp")
    print(f"股票池 {'全市场 A 股' if full else f'前 {args.n} 只'}")
    print(f"中性化: {'行业 + 市值' if args.neutral else '不做（原始因子）'}")
    print(f"披露口径: 法定披露截止日（年报次年 4/30 / 半年报 8/31 / 三季报 10/31）")

    # ── 1. 价量数据 ────────────────────────────────────────────
    t0 = time.perf_counter()
    codes = [c for c in all_codes() if is_a_share(c)]
    if not full:
        codes = pick_sample(codes, args.n)
        print(f"（分层抽样 {len(codes)} 只，按 sh/sz/bj 比例随机，seed=42）")
    print(f"\n[1/6] 读取 {len(codes):,} 只标的长表 …")
    long = load_long(codes, start=cfg.start_date, end=cfg.end_date)
    print(f"      {len(long):,} 行，{time.perf_counter()-t0:.1f}s")

    info = load_stock_info()
    info = info[info["code"].isin(codes)]
    print("\n[2/6] 构建股票池")
    long = build_universe(long, cfg, info=info)
    summarize_universe(long)
    alive = long["code"].unique().tolist()

    print("\n[3/6] 读取价格宽表 …")
    prices = load_prices(alive, start=cfg.start_date, end=cfg.end_date, field="close")
    print(f"      {prices.shape}，{time.perf_counter()-t0:.1f}s")

    # ── 2. 财务面板 ───────────────────────────────────────────
    print("\n[4/6] 构建财务因子日频面板 …")
    panel = pd.read_parquet(PANEL)
    panel = panel[panel["sym"].isin(alive)]
    print(f"      面板 {len(panel):,} 行，{panel['sym'].nunique():,} 只")

    trading_days = prices.index
    # 总股本（用于市值中性化）。shares 单位待核，见 build_financial_factors 注释。
    shares = None
    if "shares" in info.columns:
        shares = info.set_index("code")["shares"]
    fac = build_financial_factors(panel, trading_days, prices, shares=shares)
    daily = fac.pop("_daily")
    print(f"      展开为日频 {len(daily):,} 行，去极值: MAD±5")
    ind_map = daily["industry"]
    cap_map = daily["mktcap"] if "mktcap" in daily.columns else None
    if cap_map is not None:
        ok = cap_map.notna() & (cap_map > 0)
        print(f"      市值代理变量有效率 {ok.mean()*100:.1f}%（close × shares）")

    def _neut(f: pd.Series) -> pd.Series:
        ind = ind_map.reindex(f.index)
        cap = cap_map.reindex(f.index) if cap_map is not None else None
        return neutralize(f, ind.to_numpy(),
                          cap.to_numpy() if cap is not None else None)

    # ── 3. 逐因子检验 ─────────────────────────────────────────
    print(f"\n[5/6] 检验 {len(names)} 个因子: {', '.join(names)}")
    out_dir = Path(args.out) if args.out else OUTPUT_DIR / "financial"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for name in names:
        print("\n" + "-" * 74)
        print(f"因子: {name}")
        print("-" * 74)
        try:
            t0 = time.perf_counter()
            f = fac[name]
            if args.neutral:
                f = _neut(f)
            n = int(f.notna().sum())
            print(f"  有效值 {n:,}，覆盖 {f.index.get_level_values('asset').nunique():,} 只")
            if n < 1000:
                print("  ✗ 有效值过少，跳过")
                continue
            r = run_tear_sheet(f, prices, cfg, DEFAULT_COST, factor_name=name)
            print(f"  耗时 {time.perf_counter()-t0:.1f}s")
            print("  " + r.summary().replace("\n", "\n  "))
            results.append(r)
            r.ic_by_period.to_csv(out_dir / f"ic_{name}.csv", encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  ✗ 失败: {type(e).__name__}: {e}")
            traceback.print_exc()

    # ── 4. 子区间 + 汇总 ─────────────────────────────────────
    print("\n[6/6] 子区间稳定性检验")
    sub_tables = []
    for name in names:
        if not any(r.factor_name == name for r in results):
            continue
        print(f"\n  ── {name}")
        f = fac[name]
        if args.neutral:
            f = _neut(f)
        sub = subperiod_ic(f, prices, cfg, DEFAULT_COST, name)
        print("    " + sub.to_string(index=False).replace("\n", "\n    "))
        sub.insert(0, "因子", name)
        sub_tables.append(sub)

    if results:
        print("\n" + "=" * 74)
        print("汇总（1 日前瞻）")
        print("=" * 74)
        summ = pd.DataFrame([{
            "因子": r.factor_name,
            "IC均值": r.ic_by_period.loc[1, "mean"],
            "ICIR": r.ic_by_period.loc[1, "ir"],
            "t值": r.ic_by_period.loc[1, "tstat"],
            "IC样本": int(r.ic_by_period.loc[1, "count"]),
            "多空年化毛": r.gross_spread,
            "多空年化净": r.net_spread_after_cost,
            "平均换手": r.turnover_mean,
        } for r in results if 1 in r.ic_by_period.index])
        summ = summ.sort_values("IC均值", ascending=False)
        print(summ.to_string(index=False, float_format=lambda x: f"{x:>9.4f}"))
        tag = "neutral" if args.neutral else "raw"
        summ.to_csv(out_dir / f"summary_{tag}.csv", index=False, encoding="utf-8-sig")

    if sub_tables:
        tag = "neutral" if args.neutral else "raw"
        pd.concat(sub_tables, ignore_index=True).to_csv(
            out_dir / f"subperiod_{tag}.csv", index=False, encoding="utf-8-sig")

    print(f"\n结果目录: {out_dir}")
    print("提示: 以上为统计检验，不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())