"""基准选择：等权全市场 vs 真实指数 —— 决定「跑输多少」的关键。

为什么必须做这个
----------------
实测（2026-10-06）：同一年度，不同基准的收益**差 31 个百分点**。

| 年份 | 沪深300 | 中证500 | 等权全市场 |
|---|---|---|---|
| 2019 | +37.95% | +27.49% | +31.60% |
| 2020 | +25.51% | +18.65% | +20.45% |
| 2021 | **−6.21%** | +13.52% | **+24.96%** |
| 2022 | −21.27% | −20.26% | −9.62% |
| 2023 | −11.75% | −8.84% | +9.52% |
| 2024 | +16.20% | +5.85% | +12.15% |
| 2025 | +21.19% | +34.62% | +39.28% |

2021 年沪深300 跌 6.21%，等权全市场涨 24.96% ——
**同一个策略，说「大幅跑输」还是「小幅跑赢」全看选哪个基准。**

### 三个基准各自的含义

| 基准 | 含义 | 适合回答什么 |
|---|---|---|
| 等权全市场 | 全 A 股等权，含大量小盘 | 「我手动买股能不能赢过无脑等权买入全市场」 |
| 沪深300 | 大盘蓝筹 | 「我能不能赢过专业基金」 |
| 中证500 | 中盘 | 「我能不能赢过中盘成长风格」 |

**对手动投资者，等权全市场是最严苛也最公平的基准**：
它代表「把钱平均分给所有股票」这个最朴素的做法。
若策略连它都跑不赢，至少说明选股没带来任何价值。

但**不能只看一个基准** —— 一个基准下的「跑输」
可能只是风格差异（策略偏小盘 vs 基准偏大盘）。
必须报告**多个基准**下的超额，让使用者自己判断。

用法
----
  uv run python research/scripts/compare_benchmarks.py
  uv run python research/scripts/compare_benchmarks.py --years
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.analysis.long_only import _year_span  # noqa: E402
from factor_lab.config import DB_PATH  # noqa: E402

INDEX_DAILY = ROOT / "runtime" / "tushare" / "index_daily.parquet"

# 常用指数（Tushare ts_code）
BENCHMARKS = {
    "沪深300": "000300.SH",
    "中证500": "000905.SH",
    "上证指数": "000001.SH",
    "创业板指": "399006.SZ",
}


def load_index_panel(codes: list[str] | None = None) -> pd.DataFrame:
    """加载指数日线面板 (索引=日期, 列=指数代码)。"""
    if not INDEX_DAILY.exists():
        raise FileNotFoundError(
            f"{INDEX_DAILY} 不存在，无法用真实指数做基准。\n"
            f"请先跑 fetch_all_tushare.py --prio P1。")
    cols = ["ts_code", "trade_date", "close"]
    df = pd.read_parquet(INDEX_DAILY, columns=cols)
    if codes:
        df = df[df.ts_code.isin(codes)]
    df["date"] = pd.to_datetime(df.trade_date.astype("int64"),
                                format="%Y%m%d")
    return (df.pivot_table(index="date", columns="ts_code", values="close",
                           aggfunc="last").sort_index())


def equal_weight_bench(con, lo: str, hi: str) -> pd.Series:
    """等权全市场基准（从主库算）。"""
    q = f"""SELECT date,
        (close_adj - LAG(close_adj) OVER (PARTITION BY code ORDER BY date))
        / LAG(close_adj) OVER (PARTITION BY code ORDER BY date) AS r
        FROM bar_daily
        WHERE date BETWEEN {lo} AND {hi} AND close_adj IS NOT NULL"""
    d = pd.read_sql(q, con)
    d = d[d.r.notna()]
    return d.groupby("date").r.mean()


def annual_return(series: pd.Series) -> float:
    """价格序列的年化收益。

    ⚠️ **必须区分「价格序列」与「日收益序列」**（2026-10-06 实测踩坑）：
    - **价格序列**（指数 close）：`末日 / 首日 − 1`，复利在序列内部已完成
    - **日收益序列**（等权全市场 r）：`(1+r).prod()` 再开方

    初版统一用 `(1+r).prod()`，对价格序列算出 `inf%` ——
    因为价格已经涨了几千倍，再复利一次就溢出了。
    这与今天早些时候「用次日收益当因变量」是同一类错误：
    **价格与收益的量纲搞混，结果必然离谱到应该被立刻质疑。**
    """
    s = series.dropna()
    if len(s) < 2:
        return float("nan")
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    if first <= 0:
        return float("nan")
    years = _year_span(s.index)
    return (last / first) ** (1 / years) - 1


def annual_return_from_r(r: pd.Series) -> float:
    """日收益序列的年化。"""
    s = r.dropna()
    if len(s) < 2:
        return float("nan")
    nav = float((1 + s).prod())
    if nav <= 0:
        return -1.0
    return nav ** (1 / _year_span(s.index)) - 1


def benchmark_annual(start: str, end: str) -> dict[str, float]:
    """区间内基准年化收益：**等权全市场 + 沪深300**。

    ⚠️ 本项目判据（DATA_QUALITY Q18）：报超额不得只报单一基准 ——
       实测基准间单年可差 31pp，只报一个的「跑赢/跑输」结论换个基准就反转。
       故一次给两个，让读者自己判断。

    ⚠️ 两个基准的**年化口径必须与策略侧一致**（`_year_span` 自然日），
       否则「超额 = 策略 − 基准」直接偏 0.5pp/年（见 `annual_return` 注释）。

    ⚠️ 沪深300 依赖 `runtime/tushare/index_daily.parquet`（本地产物，不进 git）：
       缺文件时**显式打印跳过原因**并返回 NaN —— 不静默当 0，
       也不悄悄换成别的基准（禁止静默 fallback）。
    """
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        out = {"等权全市场":
               annual_return_from_r(equal_weight_bench(con, start, end))}
    finally:
        con.close()
    if not INDEX_DAILY.exists():
        print(f"  ⚠ 跳过沪深300 基准：缺 {INDEX_DAILY}（本地产物不进 git）")
        out["沪深300"] = float("nan")
        return out
    idx = load_index_panel(["000300.SH"])
    s = idx.loc[(idx.index >= pd.Timestamp(start)) &
                (idx.index <= pd.Timestamp(end)), "000300.SH"]
    out["沪深300"] = annual_return(s) if len(s.dropna()) > 2 else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="多基准对比")
    ap.add_argument("--start", default="20190101")
    ap.add_argument("--end", default="20251231")
    args = ap.parse_args()

    print("=" * 84)
    print("基准对比：等权全市场 vs 真实指数")
    print("=" * 84)
    print("用途：「跑输基准」这句话换个基准结论就变，必须同时报告多个。\n")

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    idx = load_index_panel(list(BENCHMARKS.values()))

    lo_y, hi_y = int(args.start[:4]), int(args.end[:4])
    header = f"{'年份':<7}" + "".join(f"{k:>12}" for k in BENCHMARKS) + f"{'等权全市场':>12}"
    print(header)
    print("-" * len(header))
    rows = []
    for y in range(lo_y, hi_y + 1):
        lo, hi = f"{y}0101", f"{y}1231"
        row = {"年份": y}
        line = f"{y:<7}"
        for name, code in BENCHMARKS.items():
            s = idx.loc[(idx.index >= pd.Timestamp(lo)) &
                        (idx.index <= pd.Timestamp(hi)), code]
            v = annual_return(s) if len(s.dropna()) > 2 else float("nan")
            row[name] = v
            line += f"{v:>12.2%}" if pd.notna(v) else f"{'N/A':>12}"
        b = equal_weight_bench(con, lo, hi)
        # 等权全市场是「日收益序列」，指数是「价格序列」——
        # 两者年化方式不同（见上面两个函数的 docstring）。
        v = annual_return_from_r(b)
        row["等权全市场"] = v
        line += f"{v:>12.2%}"
        rows.append(row)
        print(line)

    df = pd.DataFrame(rows).set_index("年份")
    print("-" * len(header))
    print(f"{'中位':<7}" + "".join(f"{df[c].median():>12.2%}" for c in df.columns))

    spread = df.max(axis=1) - df.min(axis=1)
    print(f"\n各年「最优基准 − 最差基准」中位差 {spread.median():.2%}，"
          f"最大 {spread.max():.2%}")
    print("\n⇒ 单一基准下的「跑输 X%」没有意义，必须说清是哪个基准。")

    out = ROOT / "runtime" / "benchmarks.csv"
    df.to_csv(out, encoding="utf-8-sig")
    print(f"\n已保存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())