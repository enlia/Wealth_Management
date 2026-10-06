"""财务因子构建层：季度财务面板 → 日频因子序列，含去极值与中性化。

从 run_financial_study.py 迁出（入口文件职责拆分，函数体逐字未改）：
  · 工具：winsorize_mad（MAD 去极值）、demean_by_group（组内去均值）
  · build_financial_factors：财报面板 asof 对齐 + PB/EP/BP + 去极值 + (date, asset) 组装
  · 中性化：MIN_OBS_PER_DAY / N_JOBS / _neutralize_one_day / neutralize
    （逐日截面 OLS 残差法，joblib 线程池并行）

对外名字与迁移前一致；run_financial_study.py 重导出这些名字，
compare_synthesis.py 等 5 个脚本的 `from run_financial_study import ...` 不受影响。
入口与检验编排（CLI、抽样、逐因子检验、子区间汇总）见 run_financial_study.py。

用法
----
  from financial_factors import build_financial_factors, neutralize
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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
        #
        # 🔴 **`close` 传的是未复权价**（与 PB 的分母 bps 同口径，见 run_financial_study.py 的 main()）。
        #    初版注释写「用于 log 市值中性化时常数倍会被回归截距吸收，
        #    故结果一致（实测残差差异 6.66e-16）」—— **这个论证是错的**，
        #    已被 2026-10-06 review 实测推翻：
        #
        #    log(cap_adj) − log(cap_raw) = log(adj_factor)，而复权因子
        #    **逐股逐日不同**，不是全样本一个常数 ⇒ 截距吸收不成立。
        #    实测逐日截面 std(log adj − log raw)：中位 0.0168、最大 0.0276
        #    （极差最大 0.05）—— 显著非零。
        #
        #    那句 6.66e-16 来自「亿元 vs 万元」的单位换算差（**真**常数倍），
        #    被错误推广到「复权口径」这个完全不同的场景 —— CODE_TRUST P23。
        #
        #    现状取舍：**接受**用未复权市值。
        #    理由只有「PB 分母是账面价值（bps），与未复权价同口径」这一条；
        #    若市值改用前复权，控制变量与因子分母就用两套「价格」口径。
        #    代价如实记录：未复权市值使市值中性化残差带有 ~0.02 量级扰动
        #    （即上文逐日截面 std(log adj − log raw)：中位 0.0168、最大 0.0276），
        #    该偏差【不会】被回归截距吸收（复权因子逐股逐日变化，非常数倍），
        #    也【不只】影响截面排序（OLS 中性化使用 log 市值的数值本身）。
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
