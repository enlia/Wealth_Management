"""因子面板与价格面板的组装：逐年算因子、按统一区间裁剪、拼装与对齐。

职责边界（单一职责）：
    · 取数在 ``long_only_data.py``；本模块只把长表变成「日期 × 代码」宽表面板
    · 面板的消费方是 ``run_long_only.py``（单边多头检验入口）

契约：``build_panel`` 返回 dict —— 每个因子一个 DataFrame（行=日期，列=代码），
外加 ``out["__price__"]`` 价格面板；两个面板的日期索引在返回前已对齐。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from factor_lab.factors.price_volume import compute_factor  # noqa: E402
from long_only_data import (  # noqa: E402
    WARMUP_TRADING_DAYS,
    _shift_date,
    load_long_chunked,
)


def build_panel(codes, factors, start, end) -> dict[str, pd.DataFrame]:
    """算各因子的宽表面（索引=日期，列=代码）。

    ⚠️ **必须边加载边算，不能全部加载完再算**：
    全市场 11 年 = 1,080 万行 × 9 列 ≈ 1.2 GB，
    实测一次性加载直接 `_ArrayMemoryError: Unable to allocate 742 MiB`。
    「跑 20 分钟」不是慢，是在反复 GC/重试 —— **假象会掩盖真问题**。

    做法：按年加载 → 立刻算因子 → 丢掉原始数据 → 只保留因子面板。
    因子面板是 (日期 × 代码) 的浮点矩阵，11 年约 5,600×2,611×8 = 117 MB。

    ⚠️ **跨年窗口**：因子最大回看窗口（mom250 / pos250 / vol60…）会跨年。
       每年多加载 `WARMUP_TRADING_DAYS` 个**前置交易日**用于预热，
       但只保留本年计算结果 —— 否则 mov250 在年初会全NaN，
       相当于每年初白扔 250 个交易日的信号。
    """
    y0, y1 = int(start[:4]), int(end[:4])
    acc: dict[str, list[pd.DataFrame]] = {f: [] for f in factors}
    price_parts: list[pd.DataFrame] = []
    n_rows = 0

    for y in range(y0, y1 + 1):
        ys = f"{y}0101" if y > y0 else start[:8]
        ye = f"{y}1231" if y < y1 else end[:8]
        # ⚠️ **首年也必须预热**（实测踩过）：
        #   初版写 `if y > y0: pad_start = ...` else `pad_start = ys`，
        #   于是**首年（y0）完全没有预热数据**。
        #   实测 `build_panel(..., '20190101', '20191231')`：
        #     pos250 覆盖率 **0.0%**（需要 250 日窗口，2019 年内凑不满）
        #     mom120_skip20 覆盖率 7.5%
        #   长窗口因子在首年必然大面积 NaN —— 而这**不报错**，
        #   表现为「因子没数据」，很容易被误判成「该股不合格」。
        #
        #   正确做法：每一年都往前预热 WARMUP_TRADING_DAYS 个交易日。
        pad_start = _shift_date(ys, -WARMUP_TRADING_DAYS)
        # ⚠️ **years 必须按 pad_start 实际落到的年份算**，不能写死 (y-1, y)。
        #   260 个交易日可能跨到 y−2（如 2024-01-01 → 2023-01-02，刚好 y−1；
        #   但 2021-01-01 → 2019-12 附近就会跨到 y−2）。
        #   写死 (y-1, y) 会让跨到 y−2 的那部分预热数据读不到，
        #   而 `load_long_chunked` 内部按整年边界切片，漏掉的年份整段跳过。
        pad_year = int(pad_start[:4])
        long = load_long_chunked(codes, pad_start, ye,
                                 years=tuple(range(pad_year, y + 1)),
                                 verbose=False)
        if long.empty:
            continue
        # ⚠️⚠️ **预热数据必须先按 (code, date) 去重**（实测踩过）：
        #   `years` 现在按 pad_start 实际落点算（如 2017~2019），
        #   相邻年份段的预热区间**会重叠** ——
        #   2019 年的数据同时出现在「2018 段」的预热和「2019 段」主体里。
        #   不去重则 `unstack('asset')` 直接抛
        #   `ValueError: Index contains duplicate entries`。
        #
        #   ⚠️ 去重必须在**算因子之前**，否则因子会在重复行上算错。
        #   `_prep` 内部按 ["code","date"] 排序但**不去重**，
        #   所以这里必须显式处理。
        before = len(long)
        long = (long.sort_values(["code", "date"], kind="stable")
                    .drop_duplicates(subset=["code", "date"], keep="last"))
        if len(long) != before:
            print(f"    {y}: 去重 {before:,} → {len(long):,} 行"
                  f"（预热区间与相邻年份重叠）")
        # ⚠️ **不能在这里裁剪** —— 这是预热失效的真正根因（实测踩过）：
        #   预热数据（上一年的行）在这里被 `>= lo` 全部删掉，
        #   **然后才调用 compute_factor** —— 因子根本看不到预热数据。
        #   后果：pos250 / mom120 在每年年初必然全 NaN。
        #
        #   正确顺序：**先去重 → 再算因子 → 最后裁剪结果**。
        # ⚠️ 因子面板与价格面板**必须用同一区间**，
        #    否则两者行数不一致，下游 take_along_axis 报 shape mismatch。
        lo = pd.Timestamp(ys)
        hi = pd.Timestamp(ye)
        n_rows += len(long)
        for f in factors:
            s = compute_factor(f, long)
            if s is None or s.empty:
                continue
            # compute_factor 返回 MultiIndex(date, asset) —— 层名是 asset
            wide_f = s.unstack("asset").sort_index()
            # **算完再裁剪**：只保留 [ys, ye]，去掉与相邻年份重复的预热行。
            # 裁剪必须在因子计算之后，否则预热数据白读（见上方注释）。
            wide_f = wide_f[(wide_f.index >= lo) & (wide_f.index <= hi)]
            if wide_f.empty:
                continue
            acc[f].append(wide_f)
        # 价格面板**必须用同一区间裁剪**（实测踩过）：
        #   因子面板裁了、price 没裁 →两者行数不一致
        #   （实测 price 488 行 vs factor 244 行），
        #   下游 `take_along_axis` 直接抛
        #   `IndexError: shape mismatch ... (488,1) (244,30)`。
        lo = pd.Timestamp(ys)
        hi = pd.Timestamp(ye)
        px_wide = long.pivot_table(
            index="date", columns="code", values="close_adj",
            aggfunc="last").sort_index()
        px_wide = px_wide[(px_wide.index >= lo) & (px_wide.index <= hi)]
        if not px_wide.empty:
            price_parts.append(px_wide)
        # ⚠️ **不能用 len(DataFrame)** 算「格数」——
        #   len(df) 返回的是**列数**，单因子面板永远打印「1 格」。
        #   实测踩过：面板实际有 2,600 日期 × 5,600 只 = 1,460 万格，
        #   打印却是「1 格」，一度被误判为「面板损坏、跨年拼接失效」。
        #
        # ⚠️ 也不能写 `x.shape` —— acc[f] 是**逐年 append 的列表**，
        #   元素是 DataFrame，列表本身没有 .shape（实测 AttributeError）。
        #   必须逐元素累加 shape 的乘积。
        n_cells = sum(f.shape[0] * f.shape[1]
                      for parts in acc.values() for f in parts)
        print(f"    {y}: 原始 {n_rows:>11,} 行，因子面板 {n_cells:>12,} 格")

    out = {}
    for f, parts in acc.items():
        if not parts:
            print(f"  ⚠ 因子 {f} 无数据")
            continue
        #⚠️ 必须 axis=0（按日期索引纵向堆叠）。
        #   axis=1 是横向拼列，会把 11 年的因子并成 5,600×11 列，
        #   实测表现为「每年只有 1 格」。
        wide = (pd.concat(parts, axis=0) if len(parts) > 1 else parts[0])
        # 跨年预热期可能与下一年重叠，同一 (日期, 代码) 只保留一份
        if wide.index.duplicated().any():
            wide = wide[~wide.index.duplicated(keep="last")]
        wide = wide.sort_index().loc[
            (wide.index >= pd.Timestamp(f"{y0}0101"))
            & (wide.index <= pd.Timestamp(f"{y1}1231"))]
        out[f] = wide
        cov = float(wide.notna().mean().mean())
        print(f"  ✓ {f:<16} 覆盖率 {cov*100:5.1f}%  "
              f"非空 {wide.notna().sum().sum():>12,}  "
              f"内存 {wide.memory_usage(deep=True).sum()/1e6:.0f}MB")
    out["__price__"] = (pd.concat(price_parts, axis=0)
                        if price_parts else pd.DataFrame())
    p = out["__price__"]
    if p.index.duplicated().any():
        p = p[~p.index.duplicated(keep="last")]
    print(f"  ✓ {'price':<16} 覆盖率 "
          f"{float(p.notna().mean().mean())*100:5.1f}%  "
          f"内存 {p.memory_usage(deep=True).sum()/1e6:.0f}MB")

    # ⚠️⚠️ **因子面板与价格面板的日期索引必须完全一致**（实测踩过）：
    #   因子裁剪了、price 没裁 → price 488 行 vs factor 244 行，
    #   下游 `take_along_axis` 抛
    #   `IndexError: shape mismatch ... (488,1) (244,30)`。
    #   这个错在下游报出来，**看起来像 numpy 的问题**，
    #   实际是取数层两个面板口径不一致。
    #   故在源头断言，错误信息直接指向根因。
    if out:
        ref = next(iter(out.values())).index
        if not p.index.equals(ref):
            missing = len(ref.difference(p.index))
            extra = len(p.index.difference(ref))
            raise ValueError(
                f"价格面板与因子面板日期索引不一致："
                f"price {len(p.index)} 行 vs factor {len(ref)} 行"
                f"（price 缺 {missing} 天，多 {extra} 天）。\n"
                f"  根因通常是两者用了不同的裁剪区间 —— "
                f"必须用**同一个** [ys, ye] 裁剪。")
        print(f"  ✓ 面板对齐检查通过（{len(ref):,} 个交易日 × "
              f"{len(next(iter(out.values())).columns):,} 只）")
    return out