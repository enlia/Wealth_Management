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
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from factor_lab.analysis.alphalens_adapter import run_tear_sheet, subperiod_ic
from factor_lab.config import DEFAULT_COST, DEFAULT_RESEARCH, OUTPUT_DIR, is_a_share
from factor_lab.data import (  # noqa: E402
    all_codes,
    load_factor_prices,
    load_long,
    load_stock_info,
)
from factor_lab.data.universe import build_universe, summarize_universe

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 220)

PANEL = Path(__file__).resolve().parents[2] / "runtime" / "financial_panel.parquet"


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
        # ⚠️ 此处 mktcap 的单位是【亿元】：close(元/股) × shares(亿股) = 亿元
        #    与 stock_info.mktcap（【万元】，见 factor_lab.config.MARKET_CAP_UNIT）
        #    口径不同，不要拿它与 stock_info.mktcap 直接比。
        #    用于 log 市值中性化时常数倍会被回归截距吸收，故结果一致
        #    （实测残差差异 6.66e-16）。
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


MIN_OBS_PER_DAY = 30
"""逐日截面回归的最小样本数。低于此值当日不做中性化（返回原值）。"""

# 并行度：本机 8 物理核 / 16 逻辑核。取逻辑核一半：超线程对 numpy 小矩阵
# 几无收益，反而抢内存（本机 13.9GB，单进程已用到 78%）。
N_JOBS = max(1, (os.cpu_count() or 4) // 2)


def _neutralize_one_day(pos, y, ind_codes, logcap):
    """单日截面回归。返回 (位置, 因子残差)。供 joblib 并行调用。"""
    n = len(y)
    if n < MIN_OBS_PER_DAY:
        return pos, y
    parts = [np.ones((n, 1))]
    k = 1
    if logcap is not None:
        v = np.asarray(logcap, dtype=float)
        if np.isfinite(v).sum() >= MIN_OBS_PER_DAY and np.nanstd(v) > 0:
            parts.append(np.nan_to_num(v, nan=float(np.nanmean(v)))[:, None])
            k += 1
    uniq, inv = np.unique(ind_codes, return_inverse=True)
    if uniq.size > 1:
        # drop_first：去掉第一个行业，避免与截距完全共线
        D = np.zeros((n, uniq.size - 1), dtype=float)
        keep = inv > 0
        D[keep, inv[keep] - 1] = 1.0
        parts.append(D)
        k += D.shape[1]
    if k >= n - 2:
        return pos, y
    X = np.column_stack(parts)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() <= k + 2 or ok.sum() < MIN_OBS_PER_DAY:
        return pos, y
    try:
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    except np.linalg.LinAlgError:
        return pos, y
    return pos, y - X @ beta


def neutralize(s: pd.Series, industry: np.ndarray,
               mktcap: np.ndarray | None = None,
               n_jobs: int = N_JOBS) -> pd.Series:
    """行业 + 市值中性化（逐日截面回归残差法，Barra 式）。

    为什么必须做：财务因子与市值、行业高度共线。
      · bp（账面市值比）天然高 ↔ 市值小；2017 年后 A 股小市值股大幅跑输，
        不控市值的话「价值因子」实测到的其实是「小市值因子」。
      · roe 天然高 ↔ 传统行业（银行、地产）；行业轮动会让结果被行业主导。

    ⚠️ 为什么不用「分组去均值」——踩过的坑，务必记住：
       最初实现是逐日在 (行业 × 市值分位) 组内去均值，实测
       `changed frac == 0.0`，即【完全没生效】，而结果看上去与原始因子
       一字不差，极易误判为「中性化不影响结论」。失效链条：
         1. 市值分位在【行业内】排名 → 每格只剩约 2 只股票
         2. 组内只有 1~2 只时，去均值会把它们全压成 0.0
            → 加 min_group 阈值保护，否则 alphalens 分组全 NaN，
              报 MaxLossExceededError 100%
         3. 但 min_group=5 > 每格实际样本数(约 2)
            → 阈值把所有格子都保护掉 → 中性化退化成恒等变换
       即「分组去均值 + 样本阈值」在 A 股这种「行业标签多、每行业股票少」的
       横截面上结构性不可用。两版踩坑记录都保留在案。

    正解：逐日截面 OLS 取残差
        f_t = a + b·log(市值)_t + Σ_c γ_c·行业哑变量_c + ε_t
      对行业是完全控制，对市值是线性控制，不依赖格子样本量。

    ⚠️ 性能：2,551 个交易日串行做截面回归是最大瓶颈（实测单核占绝大多数时间）。
       现按日拆成任务用 joblib 并行，本机 8 核加速约 6~8 倍。
       用线程池而非进程池：每日任务是小矩阵 lstsq，numpy 会释放 GIL；
       进程池还要复制/序列化 1,000 万行数据，反而更慢更占内存。
    """
    df = pd.DataFrame({
        "f": s.to_numpy(dtype=float),
        "_date": np.asarray(s.index.get_level_values("date")),
    })
    df["_ind"] = pd.factorize(np.asarray(industry))[0]
    df["_pos"] = np.arange(len(df))

    logcap_full = None
    if mktcap is not None:
        cap = pd.Series(np.asarray(mktcap, dtype=float))
        cap = cap.where(np.isfinite(cap) & (cap > 0))
        logcap_full = np.log(cap.to_numpy())

    tasks = [
        (g["_pos"].to_numpy(),
         g["f"].to_numpy(dtype=float),
         g["_ind"].to_numpy(),
         logcap_full[g["_pos"].to_numpy()] if logcap_full is not None else None)
        for _, g in df.groupby("_date", sort=False)
    ]

    if n_jobs > 1 and len(tasks) > 8:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_neutralize_one_day)(*t) for t in tasks)
    else:
        results = [_neutralize_one_day(*t) for t in tasks]

    out = df["f"].to_numpy(dtype=float).copy()
    for pos, yhat in results:
        out[pos] = yhat
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
    # 🔴🔴 **必须同时取两种价格口径**（2026-10-06 实测 BLOCK）
    #   初版只取 `field="close"`（未复权），一个面板同时充当两个角色：
    #     · PB = close / bps        → 确实需要**未复权**价（bps 是财报披露的原始数字）
    #     · alphalens 前瞻收益 / IC → **必须**用后复权价
    #   ⇒ 收益端被除权缺口污染。实测全市场等权口径年化偏差：
    #       2016 −16.80pp / 2019 −10.80pp / 2021~2026 每年 −5.1 ~ −8.0pp
    #   **这个偏差大于本项目声称的任何因子收益**（最高的 ep 也只有 +6.1%）
    #   ⇒ 修复前跑出的 bp/ep/roe IC 与多空收益全部不可信。
    px_raw, px_adj = load_factor_prices(alive, start=cfg.start_date,
                                       end=cfg.end_date)
    prices_raw = px_raw
    prices = px_adj          # 收益口径：后复权
    print(f"      未复权 {px_raw.shape} / 后复权 {px_adj.shape}"
          f"（索引已校验一致），{time.perf_counter()-t0:.1f}s")

    # ── 2. 财务面板 ───────────────────────────────────────────
    print("\n[4/6] 构建财务因子日频面板 …")
    panel = pd.read_parquet(PANEL)
    panel = panel[panel["sym"].isin(alive)]
    print(f"      面板 {len(panel):,} 行，{panel['sym'].nunique():,} 只")

    # ⚠️ **交易日轴取后复权面板**（它才是收益口径的时间轴），
    #   但**PB/EP 的价格必须传未复权面板** —— bps/eps 是财报披露的原始数字，
    #   与未复权价同口径。若这里传 `prices`（后复权），
    #   PB 会被复权因子抬高，等于按「今天的股本」算历史 PB。
    trading_days = prices.index
    # 总股本（用于市值中性化）
    # shares 单位 = 亿股（已核验：mktcap / shares 精确等于股价，见
    # research/scripts/check_units.py）；市值口径为【亿元】，与
    # stock_info.mktcap 的【万元】不同口径，仅用于 log 中性化。
    shares = None
    if "shares" in info.columns:
        shares = info.set_index("code")["shares"]
    fac = build_financial_factors(panel, trading_days, prices_raw, shares=shares)
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
    # ⚠️ 性能：中性化与 tear sheet 都是纯函数且无副作用，算一次即可复用。
    #    早期版本在主循环和子区间循环各算一遍 → 重复 2 倍耗时。
    prepared: dict[str, pd.Series] = {}

    def _get(name: str) -> pd.Series:
        if name not in prepared:
            f = fac[name]
            if args.neutral:
                f = _neut(f)
            prepared[name] = f
        return prepared[name]

    for name in names:
        print("\n" + "-" * 74)
        print(f"因子: {name}")
        print("-" * 74)
        try:
            t0 = time.perf_counter()
            f = _get(name)
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
        sub = subperiod_ic(_get(name), prices, cfg, DEFAULT_COST, name)
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