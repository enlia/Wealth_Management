"""抓取全市场历史业绩报表（东财源，走代理可达）。

解决的核心问题：此前财务因子无法做历史检验，因为只有单期快照。
本脚本按季度抓取 2015Q1~2026Q2 的全市场业绩报表，
得到净利润同比、ROE、毛利率、每股经营现金流等**时间序列**。

数据源：akshare.stock_yjbb_em（东方财富，业绩报表）
用法：需先设置代理
  export HTTPS_PROXY=http://127.0.0.1:7897
  uv run python scripts/fetch_yjbb.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

PROXY = os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTPS_PROXY", PROXY)

import akshare as ak  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent.parent / "out" / "yjbb"
OUT.mkdir(parents=True, exist_ok=True)

# 报告期列表：Q1=0331, 中报=0630, Q3=0930, 年报=1231
PERIODS = [f"{y}{q}" for y in range(2015, 2027)
           for q in ("0331", "0630", "0930", "1231")]

# 列名映射（akshare 的原始列名 → 简洁名）
COLS = {
    "股票代码": "code",
    "股票简称": "name",
    "每股收益": "eps",
    "营业总收入-营业总收入": "revenue",
    "营业总收入-同比增长": "revenue_yoy",
    "营业总收入-季度环比增长": "revenue_qoq",
    "净利润-净利润": "net_profit",
    "净利润-同比增长": "profit_yoy",
    "净利润-季度环比增长": "profit_qoq",
    "每股净资产": "bps",
    "净资产收益率": "roe",
    "每股经营现金流量": "cfps",
    "销售毛利率": "gross_margin",
    "所处行业": "industry",
}


def main() -> int:
    print("=" * 74)
    print("抓取全市场历史业绩报表（东财源）")
    print(f"代理: {PROXY}")
    print(f"目标报告期: {PERIODS[0]} ~ {PERIODS[-1]}  共 {len(PERIODS)} 期")
    print("=" * 74)

    ok, fail, empty = 0, 0, []
    for i, p in enumerate(PERIODS, 1):
        f = OUT / f"yjbb_{p}.csv"
        if f.exists() and f.stat().st_size > 1000:
            ok += 1
            continue
        t0 = time.perf_counter()
        try:
            d = ak.stock_yjbb_em(date=p)
            if d is None or d.empty:
                empty.append(p)
                continue
            keep = {k: v for k, v in COLS.items() if k in d.columns}
            d = d[list(keep)].rename(columns=keep)
            # 股票代码补零到 6 位
            d["code"] = d["code"].astype(str).str.extract(r"(\d{6})", expand=False)
            d = d.dropna(subset=["code"])
            d["report_period"] = p
            d.to_csv(f, index=False, encoding="utf-8-sig")
            ok += 1
            print(f"  [{i:>2}/{len(PERIODS)}] {p}  {len(d):>6,} 行  "
                  f"{time.perf_counter()-t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  [{i:>2}/{len(PERIODS)}] {p}  ✗ {type(e).__name__}: {str(e)[:50]}")
            time.sleep(2)
    print("=" * 74)
    print(f"成功 {ok} 期，失败 {fail} 期，空 {len(empty)} 期")
    if empty:
        print(f"空报告期: {empty}（通常是因为该期尚未披露或接口无数据）")
    print(f"输出目录: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
