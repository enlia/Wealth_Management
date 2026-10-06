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
    dedupe_long,
    load_long_chunked,
)


def assemble_frames(parts: list[pd.DataFrame], label: str) -> pd.DataFrame:
    """把多年份片段沿日期轴拼成一张宽表；同一日期重复时只留最后一份。

    ⚠️ **去重结果就是返回值本身** —— 调用方与下游只应拿到去重后的对象。
       不允许「去重落在局部变量、返回值还是旧对象」：那样保护形同虚设，
       测试与运行时读到的恰是没被保护的那个面板。
    """
    #⚠️ 必须 axis=0（按日期索引纵向堆叠）。
    #   axis=1 是横向拼列，会把 11 年的因子并成 5,600×11 列，
    #   实测表现为「每年只有 1 格」。
    wide = pd.concat(parts, axis=0) if len(parts) > 1 else parts[0]
    # 跨年预热期可能与下一年重叠，同一 (日期, 代码) 只保留一份
    dup = wide.index.duplicated(keep="first")
    if dup.any():
        n_dup = int(dup.sum())
        wide = wide[~wide.index.duplicated(keep="last")]
        print(f"    {label}: 拼接片段含 {n_dup} 个重复日期（跨年重叠），"
              f"已按 keep='last' 去重为 {len(wide):,} 行")
    return wide


def check_alignment(out: dict[str, pd.DataFrame]) -> None:
    """因子面板与价格面板的日期索引必须完全一致 —— **逐个因子**与价格比对。

    ⚠️⚠️ 这是实测踩过的错：因子裁剪了、price 没裁 →
       price 488 行 vs factor 244 行，下游 `take_along_axis` 抛
       `IndexError: shape mismatch ... (488,1) (244,30)`。
       这个错在下游报出来，**看起来像 numpy 的问题**，
       实际是取数层两个面板口径不一致 —— 故在源头断言，错误指向根因。

    ⚠️ 保护必须覆盖**所有**因子面板：
       只拿第一个因子与 price 比对时，其余因子的索引不一致不会被发现，
       而下游 shape mismatch 照样发生；价格面板也不能拿自己跟自己比 ——
       一个因子都没有时必须按失败抛错，不许打印「对齐通过」。
    """
    factor_names = [k for k in out if k != "__price__"]
    price = out.get("__price__")
    if price is None or price.empty:
        raise ValueError(
            "价格面板为空 —— 没有价格就没有收益可算，"
            "禁止以空面板冒充对齐通过。")
    if not factor_names:
        raise ValueError(
            "没有任何因子面板（逐年因子结果全为空）—— "
            "价格面板单独存在没有意义，禁止打印「对齐通过」。")
    ref = price.index
    if not ref.is_unique:
        raise ValueError(
            f"价格面板日期索引仍有重复（{int(ref.duplicated().sum())} 处）——"
            f"去重必须落在返回对象上。")
    for name in factor_names:
        idx = out[name].index
        if not idx.equals(ref):
            missing = len(ref.difference(idx))
            extra = len(idx.difference(ref))
            raise ValueError(
                f"因子面板 {name} 与价格面板日期索引不一致："
                f"price {len(ref)} 行 vs factor {len(idx)} 行"
                f"（{name} 缺 {missing} 天，多 {extra} 天）。\n"
                f"  根因通常是两者用了不同的裁剪区间 —— "
                f"必须用**同一个** [ys, ye] 裁剪。")


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
        #   相邻年份切片的预热区间会重叠，同一 (code, date) 出现多行，
        #   不去重则 `unstack('asset')` 直接抛
        #   `ValueError: Index contains duplicate entries`。
        #
        #   ⚠️ 去重必须在**算因子之前**，否则因子会在重复行上算错。
        #   `_prep` 内部按 ["code","date"] 排序但**不去重**，
        #   所以这里必须显式处理（long_only_data.dedupe_long）。
        #
        #   ⚠️ 去重规则按 DATA_SOURCE S5/S8 分两类：
        #   整行完全相同的重复 = 拼接产物，允许去掉；
        #   同 (code, date) 取值不同 = 数据冲突，
        #   **不做 keep='last' 静默丢弃**，由 dedupe_long 直接报错。
        before = len(long)
        long = dedupe_long(long)
        if len(long) != before:
            print(f"    {y}: 去重 {before:,} → {len(long):,} 行"
                  f"（预热区间与相邻年份重叠产生的整行重复）")
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
        wide = assemble_frames(parts, f)
        wide = wide.sort_index().loc[
            (wide.index >= pd.Timestamp(f"{y0}0101"))
            & (wide.index <= pd.Timestamp(f"{y1}1231"))]
        out[f] = wide
        cov = float(wide.notna().mean().mean())
        print(f"  ✓ {f:<16} 覆盖率 {cov*100:5.1f}%  "
              f"非空 {wide.notna().sum().sum():>12,}  "
              f"内存 {wide.memory_usage(deep=True).sum()/1e6:.0f}MB")
    # ⚠️ **价格面板的去重必须落在 out["__price__"] 上**（实测踩过的反例：
    #   去重结果只赋给局部变量 p，返回 dict 里仍是未去重对象，
    #   测试与运行时读到的恰是没被保护的那个）。
    out["__price__"] = (assemble_frames(price_parts, "price")
                        if price_parts else pd.DataFrame())
    p = out["__price__"]
    print(f"  ✓ {'price':<16} 覆盖率 "
          f"{float(p.notna().mean().mean())*100:5.1f}%  "
          f"内存 {p.memory_usage(deep=True).sum()/1e6:.0f}MB")

    # 对齐检查：逐个因子与价格面板全量比对，失败直接抛错（见 check_alignment）
    check_alignment(out)
    ref = out["__price__"].index
    print(f"  ✓ 面板对齐检查通过（{len(ref):,} 个交易日 × "
          f"{len(out['__price__'].columns):,} 只，"
          f"{sum(1 for k in out if k != '__price__')} 个因子逐一比对）")
    return out