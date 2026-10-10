"""财务因子的宽表面加载器 —— 把季度面板展开成 (日期 × 代码) 矩阵。

为什么不直接复用 `run_financial_study.build_financial_factors`
-------------------------------------------------------------
那个函数返回的是 `index=(date, asset)` 的 **Series**，供 alphalens 算 IC 用。
而滚动窗口检验与市值分层检验需要的是 **宽表面**（索引=日期，列=代码），
因为 `build_long_only` 的选股是 `rank_topk(fv_mat, n_pick)` —— 吃的是
`(T, N)` 的 numpy 矩阵。

两者不能互相替代：
  · Series 版每天的股票集合随财报披露变化（有 NaN 行），
    转宽表时会引入大量「当日不存在的列」
  · 宽表版本必须保证 **行=交易日轴、列=标的集合** 固定，
    缺失处以 NaN 表示（`rank_topk` 把 NaN 排到末尾）

⚠️ **价格口径是本模块存在的全部理由**（2026-10-06 实测 BLOCK）
   `close_adj` 实测为**前复权 qfq**，不是后复权。
   收益端必须用它，否则除权缺口污染收益，偏差 5~17pp/年。
   而 PB 的分母 `bps` 是财报披露的**原始账面价值**，
   必须配**未复权价** —— 一个面板不能同充当两个角色。

⚠️ **qfq 的可复现性陷阱**：前复权因子每遇分红送转就整体重算历史，
   所以本模块的输出只在**同一份数据快照**内自洽。
   任何跨版本引用必须带 `data_version.build_manifest()` 的版本戳。
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from run_financial_study import PANEL, build_financial_factors  # noqa: E402

from factor_lab.data import (  # noqa: E402
    all_codes,
    load_factor_prices,
    load_stock_info,
)
from factor_lab.data.universe import build_universe  # noqa: E402

# 宽表面里允许保留的因子（与 build_financial_factors 的输出同名）
FACTOR_NAMES = ("bp", "ep", "roe", "profit_yoy",
                "revenue_yoy", "gross_margin", "cf_quality")


def _build_wide_chunked(panel: pd.DataFrame, trading_days: pd.DatetimeIndex,
                        px_raw: pd.DataFrame, names: list[str],
                        shares: pd.Series | None, neutral: bool,
                        verbose: bool) -> dict[str, pd.DataFrame]:
    """按年分块构建宽表面，避免一次性构建 11 年的 daily 长表。

    🔴 **为什么必须分块**（2026-10-06 实测 OOM）
       `build_financial_factors` 内部构造 `daily`，形状 = (日期 × 标的) 长表。
       全市场 5,267 只 × 2,611 日 = **1,375 万行 × 9 列 float64 = 990 MB**，
       而 pandas 的 `.copy()` 会做 block 合并，峰值需**同时持有多份** ——
       实测 `_ArrayMemoryError: Unable to allocate 687 MiB`
       （本机 13.9 GB，free 10.1 GB）。

    ⚠️ **分块不破坏财务对齐**：
       `build_financial_factors` 的对齐轴是
       `union = 全部 available_date ∪ trading_days`，
       其中 `available_date` 来自**整份 panel**（与传入的 trading_days 无关）
       ⇒ 传「只含本年交易日」的子集，ffill 链依然从历史报告日接上来，
       不会把1 月初的因子变成 NaN。
       ⚠️ 这与 `run_long_only.build_panel` 的跨年预热是**不同**的问题
          （那里是 rolling/shift 需要前置行），此处是 ffill 对齐，不需要预热。
    """
    panels: dict[str, list[pd.DataFrame]] = {n: [] for n in names}
    years = sorted({d.year for d in trading_days})
    for y in years:
        days = trading_days[trading_days.year == y]
        if len(days) == 0:
            continue
        fac = build_financial_factors(panel, days, px_raw, shares=shares)
        daily = fac.pop("_daily")
        for n in names:
            wide = (fac[n].unstack(level="asset")
                    .reindex(index=days, columns=px_raw.columns))
            panels[n].append(wide)
        if neutral:
            _neutralize_chunk(panels, daily, names, days, px_raw.columns)
        del daily, fac
        if verbose:
            got = panels[names[0]][-1]
            print(f"    {y} 年: {len(days)} 交易日 × "
                  f"{got.shape[1]} 标的，"
                  f"{names[0]} 有效 {int(got.notna().to_numpy().sum()):,}")

    out = {n: pd.concat(parts) if parts else pd.DataFrame()
           for n, parts in panels.items()}
    return out


def _neutralize_chunk(panels: dict[str, list[pd.DataFrame]],
                      daily: pd.DataFrame, names: list[str],
                      days: pd.DatetimeIndex,
                      codes: pd.Index) -> None:
    """对某一年的 chunk 做行业 + 市值中性化，结果追加到 parts 列表。

    🔴 BLOCK（review 2026-10-06 实测抓出，此初版 100% 崩溃）：
       `run_financial_study.neutralize` 内部是
       `s.index.get_level_values("date")` ⇒ 要求 `s` 带 `(date, asset)`
       MultiIndex。而本函数传的是 `row[ok]`（单日、纯 Index(代码)），
       实测 `KeyError: 'Requested level (date) does not match index name (None)'`。

    🔴 第二个 bug（被上面那个挡住，从未暴露）：
       `pd.DataFrame(acc, index=days)` 产出的是 (日期 × 标的)，
       初版又 `.T` 转成 (标的 × 日期) 再 reindex 回 (日期 × 标的)
       ⇒ 行列标签全不匹配，**输出全 NaN**。
       若只修第一个 bug，`--neutral` 会「跑通」但结果全 NaN ——
       正是 CODE_TRUST描述的「不报错、结果全错」形态。

       实测（review 打桩 neutralize 为恒等函数）：
         输出与输入逐元素相等： False；输出非NaN 数 0 / 240。
    """
    from run_financial_study import neutralize

    ind = daily["industry"]
    cap = daily["mktcap"] if "mktcap" in daily.columns else None
    for n in names:
        wide = panels[n][-1]
        ind_w = ind.unstack(level="asset").reindex(index=days, columns=codes)
        cap_w = (cap.unstack(level="asset").reindex(index=days, columns=codes)
                 if cap is not None else None)
        logcap = np.log(cap_w) if cap_w is not None else None
        acc = []
        for dt in days:
            row = wide.loc[dt]
            if dt not in ind_w.index:
                acc.append(row)
                continue
            ok = row.notna() & ind_w.loc[dt].notna()
            if ok.sum() < 30:
                acc.append(row)
                continue
            # ⚠️ **必须补上 (date, asset) MultiIndex** —— neutralize 依赖它分组
            sub = row[ok]
            mi = pd.MultiIndex.from_arrays(
                [np.repeat(dt, len(sub)), sub.index.to_numpy()],
                names=["date", "asset"])
            resid = neutralize(
                pd.Series(sub.to_numpy(dtype=float), index=mi),
                ind_w.loc[dt][ok].to_numpy(),
                (logcap.loc[dt][ok].to_numpy()
                 if logcap is not None else None))
            r = row.copy()
            r.loc[ok.index] = resid.to_numpy()
            acc.append(r)
        # ⚠️ **不要再 .T** —— `DataFrame(acc, index=days)` 已是 (日期 × 标的)
        panels[n][-1] = pd.DataFrame(acc, index=days).reindex(
            index=days, columns=codes)


def build_wide_panels(names: list[str] | None = None,
                      start: str = "2016-01-01",
                      end: str = "2026-09-30",
                      neutral: bool = False,
                      verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    """构建财务因子宽表面 + 前复权价格面板。

    返回
    ----
    (panels, price)
      panels: {因子名: DataFrame(索引=交易日, 列=代码)}，**未做横截面标准化**
              （标准化交给调用方的 `_cross_z`，与价量因子路径保持一致）
      price : DataFrame，**前复权收盘价**，行=交易日，列=代码

    ⚠️ `price` 只返回**前复权**一份：未复权价仅在构建 PB 时用一次，
       不需要外泄给回测引擎 —— 让调用方有机会误用。
    """
    names = list(names or FACTOR_NAMES)
    unknown = [n for n in names if n not in FACTOR_NAMES]
    if unknown:
        raise ValueError(f"未知财务因子 {unknown}，可选 {list(FACTOR_NAMES)}")

    # ── 1. 股票池 ────────────────────────────────────────────
    from factor_lab.config import DEFAULT_RESEARCH, is_a_share
    from factor_lab.data import load_long

    # ⚠️ **必须用 `dataclasses.replace`，不能用 `__dict__` 拷贝**
    #   （ENGINEERING 第四节：静默fallback）。
    #   `{**obj.__dict__, ...}` 会静默丢掉 `field(init=False)`
    #   或 `__slots__` 的字段 —— 当前能跑通不代表字段变化后还跑得通。
    cfg = replace(DEFAULT_RESEARCH, start_date=start, end_date=end)
    codes = [c for c in all_codes() if is_a_share(c)]
    long = load_long(codes, start=cfg.start_date, end=cfg.end_date)
    info = load_stock_info()
    long = build_universe(long, cfg, info=info, verbose=verbose)
    alive = sorted(long["code"].unique().tolist())
    if not alive:
        raise ValueError("股票池为空 —— 检查 build_universe 的过滤条件")

    # ── 2. 双口径价格 ────────────────────────────────────────
    px_raw, px_adj = load_factor_prices(alive, start=cfg.start_date,
                                        end=cfg.end_date)
    trading_days = px_adj.index

    # ── 3. 财务因子（按年分块，避免 OOM）────────────────────
    panel = pd.read_parquet(PANEL)
    panel = panel[panel["sym"].isin(alive)]
    shares = (info.set_index("code")["shares"]
              if "shares" in info.columns else None)
    built = _build_wide_chunked(panel, trading_days, px_raw, names,
                                shares, neutral, verbose)

    # ⚠️ **必须与价格面板的列完全对齐**。
    #   `build_long_only` 内部用 `np.take_along_axis(fv_mat, held_mat)`，
    #   因子列序与价格列序不一致 ⇒ 静默错配（因子值取自 A 股、收益取自 B 股）。
    panels: dict[str, pd.DataFrame] = {}
    for n in names:
        wide = built[n].reindex(index=trading_days, columns=px_adj.columns)
        n_ok = int(wide.notna().to_numpy().sum())
        panels[n] = wide
        if verbose:
            print(f"    {n:>12}: 宽表 {wide.shape}，有效值 {n_ok:,}，"
                  f"覆盖 {n_ok / max(wide.size, 1):.1%}")

    return panels, px_adj
