"""财务因子构建：从 46 期业绩报表转成可回测的日频面板。

⚠️ 前视偏差控制（本项目第一红线）
   业绩报表只有报告期（report_period），没有实际披露日。
   本项目【按法定披露截止日】作为最早可用日，而不是「报告期 + 固定天数」：

       报告期类型   法定披露截止日     最早可用日
       Q1  (03-31)  当年 04-30        当年 04-30
       H1  (06-30)  当年 08-31        当年 08-31
       Q3  (09-30)  当年 10-31        当年 10-31
       年报(12-31)  次年 04-30        次年 04-30

   ⚠️ 历史踩坑：初版用「报告期 + 45 天」统一延迟，对 Q3 保守、对
      Q1 略保守，但【对年报提前 2.5 个月、对半年报提前 17 天】——
      大量公司 3~4 月才出年报，用 2/14 当可用日就是用了尚未公布的信息。
      统一天数是错的，必须按报告期类型取法定上限。

   若严格偏保守，还可再加缓冲（见 EXTRA_BUFFER_DAYS），
   但会牺牲样本；本项目默认不加，因为法定截止日已是最晚合法时点。
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parents[2]
YJBB_DIR = LAB / "runtime" / "yjbb"
OUT_DIR = LAB / "runtime" / "financial_panel.parquet"

# 额外缓冲天数（0 = 只用法定截止日）。调大更保守但样本更少。
EXTRA_BUFFER_DAYS = 0

_PREFIX_RULES = [
    ("sh", ("6", "9", "5", "1", "88")),
    ("sz", ("0", "3", "2")),
    ("bj", ("4", "8", "92")),
]


def to_full_code(code6: str) -> str | None:
    """6 位裸码 → 带市场前缀。无法判断时返回 None。"""
    c = str(code6).strip().zfill(6)
    for mkt, prefixes in _PREFIX_RULES:
        if c.startswith(prefixes):
            return mkt + c
    return None


def earliest_available_date(report_period: pd.Timestamp) -> pd.Timestamp:
    """报告期 → 法定最早可用披露日（= 法定披露截止日）。

    沪深主板规则（《证券法》+ 交易所规则）：
      一季报  03-31 → 当年 04-30
      半年报  06-30 → 当年 08-31
      三季报  09-30 → 当年 10-31
      年报    12-31 → 次年 04-30
    """
    m = report_period.month
    if m == 3:
        return pd.Timestamp(report_period.year, 4, 30)
    if m == 6:
        return pd.Timestamp(report_period.year, 8, 31)
    if m == 9:
        return pd.Timestamp(report_period.year, 10, 31)
    if m == 12:
        return pd.Timestamp(report_period.year + 1, 4, 30)
    # 非标准报告期（业绩快报等）保守处理：按季末 + 3 个月
    return report_period + pd.DateOffset(months=3)


def load_yjbb() -> pd.DataFrame:
    fs = sorted(glob.glob(str(YJBB_DIR / "yjbb_*.csv")))
    if not fs:
        raise FileNotFoundError(f"未找到业绩报表: {YJBB_DIR}")
    parts = []
    for f in fs:
        d = pd.read_csv(f, low_memory=False, dtype={"code": str})
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["code6"] = d["code"].astype(str).str.extract(r"(\d{6})", expand=False)
    d["sym"] = d["code6"].map(to_full_code)
    d = d.dropna(subset=["sym"])
    d["report_period"] = pd.to_datetime(d["report_period"], format="%Y%m%d")
    # 最早可用日 = 该报告期类型的法定披露截止日
    d["available_date"] = d["report_period"].map(earliest_available_date)
    if EXTRA_BUFFER_DAYS:
        d["available_date"] += pd.Timedelta(days=EXTRA_BUFFER_DAYS)
    return d


def build_panel(verbose: bool = True) -> pd.DataFrame:
    d = load_yjbb()
    if verbose:
        print(f"原始: {len(d):,} 行, {d['sym'].nunique():,} 只, "
              f"{d['report_period'].nunique()} 期 "
              f"({d['report_period'].min():%Y-%m-%d} ~ {d['report_period'].max():%Y-%m-%d})")

    num = ["eps", "revenue", "revenue_yoy", "net_profit", "profit_yoy",
           "bps", "roe", "cfps", "gross_margin"]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # 同一 sym 同期可能有重复（不同来源口径），去重取最新
    d = d.sort_values("available_date").drop_duplicates(
        subset=["sym", "report_period"], keep="last")

    # 派生因子
    # 1) 现金流质量：每股经营现金流 / 每股收益
    d["cf_quality"] = d["cfps"] / d["eps"].replace(0, np.nan)
    # 2) 盈利能力：ROE（已是百分数）
    # 3) 成长：净利润同比（已是百分数）
    # 4) 估值：PB 需在回测时用当日价 ÷ bps 计算，此处只准备 bps

    d = d.sort_values(["sym", "available_date"]).reset_index(drop=True)
    if verbose:
        print(f"去重后: {len(d):,} 行, {d['sym'].nunique():,} 只")
        print(f"可用日范围: {d['available_date'].min():%Y-%m-%d} ~ "
              f"{d['available_date'].max():%Y-%m-%d}")
    return d


def to_daily_panel(d: pd.DataFrame) -> pd.DataFrame:
    """把季度财务数据展开成「每只股票每个可用日」的长表，并 ffill 到日频。"""
    cols = ["sym", "available_date", "eps", "bps", "roe", "profit_yoy",
            "revenue_yoy", "gross_margin", "cfps", "cf_quality", "industry"]
    out = d[cols].sort_values(["sym", "available_date"])
    return out


def main() -> None:
    d = build_panel()
    dp = to_daily_panel(d)

    # 保存：财务事件表（季度粒度，可用日对齐）
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    dp.to_parquet(OUT_DIR, index=False, compression="zstd")
    size = OUT_DIR.stat().st_size / 1024 / 1024

    print("\n" + "=" * 66)
    print("财务因子面板已生成")
    print("=" * 66)
    print(f"记录数      {len(dp):,}")
    print(f"股票数      {dp['sym'].nunique():,}")
    print(f"字段        {list(dp.columns[2:])}")
    print(f"文件        {OUT_DIR}  ({size:.1f} MB, zstd 压缩)")
    print("\n披露延迟    【按报告期类型的法定披露截止日】，非固定天数")
    print("            03-31→04-30   06-30→08-31   09-30→10-31   12-31→次年04-30")
    print(f"            额外缓冲      {EXTRA_BUFFER_DAYS} 天")
    print("\n可用因子：")
    print("  roe           净资产收益率 → 质量因子")
    print("  profit_yoy    净利润同比   → 成长因子")
    print("  revenue_yoy   营收同比     → 成长因子")
    print("  gross_margin  销售毛利率   → 定价权")
    print("  cf_quality    每股经营现金流/每股收益 → 盈利质量")
    print("  bps           每股净资产   → 算 PB 用（PB = 当日收盘价 / bps）")


if __name__ == "__main__":
    main()
