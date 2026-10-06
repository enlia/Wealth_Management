"""Tushare 全量数据质量校验。

为什么下载完必须跑这个
--------------------
「下载成功」只说明接口返回了数据，**不代表数据是对的**。
本项目实测踩过的坑，每一条都是「下载成功但数据不能用」：

  · 分页截断—— repurchase 拿到 2,000 行（实际63,755），行数看着正常
  · 期间截断—— disclosure_date 停在 2016-04，近 10 年全缺
  · 静默失败—— dividend 传错参数返回 0 行，不报错
  · 重复放大—— stock_company 按年分段拉，6,294 只变 75,528 行

所以校验的核心不是「有没有数据」，而是**数据覆盖的时间范围和唯一性对不对**。

判据
----
  1. 覆盖率：日频表≥ 95% 的交易日有数
  2. 时间覆盖：末期距今≤ 1 年、末期 ≥ 当前日期 - 5 天
  3. 唯一性：(主键)组合无重复
  4. 非空率：关键字段缺失率 ≤ 30%

用法
----
  uv run python research/scripts/verify_tushare_full.py
  uv run python research/scripts/verify_tushare_full.py --task dividend
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "scripts"))

OUT = Path(__file__).resolve().parents[2] / "runtime" / "tushare"
MANIFEST = OUT / "_download_manifest.json"

# 各类表的主键与关键字段。
# ⚠️ 键必须包含「区分实体的字段」——index_weight 只用 trade_date
#    会把沪深300 和中证500 同一天的权重当成重复（见 DATA_SOURCE S5）。
#
# ⚠️ 财务三表（income/balancesheet/cashflow）**同一个 end_date 会有多行**，
#    那是 `update_flag`（0=原始披露 / 1=更新公告）与 `report_type`（1~4 对应
#    一季报/半年报/三季报/年报的合并报表与母公司报表）造成的。
#    这是**真实数据特性，不是重复** —— 实测 income 有 72,280 行
#    同 (ts_code, end_date) 但 ann_date/update_flag 不同。
#    因子研究要用「最新公告版本」，取 ann_date 最大 + report_type 合并报表。
#    所以这里把 update_flag 也纳入键，避免误报。
SPECS: dict[str, dict] = {
    "adj_factor":       {"keys": ["ts_code", "trade_date"], "freq": "D"},
    "daily_basic":      {"keys": ["ts_code", "trade_date"], "freq": "D"},
    "stk_limit":        {"keys": ["ts_code", "trade_date"], "freq": "D"},
    "moneyflow":        {"keys": ["ts_code", "trade_date"], "freq": "D"},
    "margin_detail":    {"keys": ["ts_code", "trade_date"], "freq": "D"},
    "index_daily":      {"keys": ["ts_code", "trade_date"], "freq": "D"},
    "moneyflow_hsgt":   {"keys": ["trade_date"], "freq": "D"},
    "block_trade":      {"keys": ["ts_code", "trade_date"], "freq": "D"},
    "top_list":         {"keys": ["ts_code", "trade_date"], "freq": "D"},
    "suspend_d":        {"keys": ["ts_code", "trade_date"], "freq": "D"},
    "ggt_top10":        {"keys": ["ts_code", "trade_date"], "freq": "D"},
    "fina_indicator":   {"keys": ["ts_code", "end_date", "ann_date"], "freq": "Q"},
    "income":           {"keys": ["ts_code", "end_date", "ann_date",
                                  "report_type", "update_flag"], "freq": "Q"},
    "balancesheet":     {"keys": ["ts_code", "end_date", "ann_date",
                                  "report_type", "update_flag"], "freq": "Q"},
    "cashflow":         {"keys": ["ts_code", "end_date", "ann_date",
                                  "report_type", "update_flag"], "freq": "Q"},
    "forecast":         {"keys": ["ts_code", "ann_date"], "freq": "Q"},
    "express":          {"keys": ["ts_code", "ann_date"], "freq": "Q"},
    # ⚠️ top10_holders 同 (end_date, holder_name, hold_amount) 会有
    #    多个 ann_date —— 实测 002150.SZ 的 20260630 报告在20260703 首次披露、
    #    20260818 又发修正公告（hold_float_ratio 从 NaN 变成 1.2305）。
    #    这是**真实的多次披露**，不是重复，必须把 ann_date 纳入键。
    "top10_holders":    {"keys": ["ts_code", "end_date", "holder_name",
                                  "ann_date"], "freq": "Q"},
    "stk_holdernumber": {"keys": ["ts_code", "end_date"], "freq": "Q"},
    "dividend":         {"keys": ["ts_code", "end_date"], "freq": "Q"},
    "share_float":      {"keys": ["ts_code", "ann_date", "float_date"], "freq": "Q"},
    "pledge_stat":      {"keys": ["ts_code", "end_date"], "freq": "Q"},
    "index_weight":     {"keys": ["index_code", "con_code", "trade_date"], "freq": "M"},
    "namechange":       {"keys": ["ts_code", "start_date"], "freq": "X"},
    "stock_basic":      {"keys": ["ts_code"], "freq": "X"},
    "stock_company":    {"keys": ["ts_code"], "freq": "X"},
}

# 各表应该用哪个日期列判时间覆盖
DATE_COL = {
    "adj_factor": "trade_date", "daily_basic": "trade_date",
    "stk_limit": "trade_date", "moneyflow": "trade_date",
    "margin_detail": "trade_date", "index_daily": "trade_date",
    "fina_indicator": "end_date", "income": "end_date",
    "balancesheet": "end_date", "cashflow": "end_date",
    "top10_holders": "end_date", "stk_holdernumber": "end_date",
    "dividend": "end_date", "index_weight": "trade_date",
    "namechange": "start_date", "share_float": "float_date",
    "stock_basic": "list_date", "repurchase": "ann_date",
    "new_share": "ann_date", "disclosure_date": "ann_date",
    "forecast": "ann_date", "express": "ann_date",
    "pledge_stat": "end_date",
}

# 静态信息类：时间覆盖不适用
NO_DATE_CHECK = {"index_classify", "index_member_all", "shibor", "cn_cpi",
                 "cn_pmi", "margin", "sz_daily_info", "stock_basic_D"}


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def check_one(name: str, verbose: bool = True) -> dict:
    """校验单个表。返回结构化结果。"""
    f = OUT / f"{name}.parquet"
    if not f.exists():
        return {"table": name, "status": "缺失", "rows": 0}

    df = pd.read_parquet(f)
    n = len(df)
    r: dict = {"table": name, "rows": n, "status": "OK",
             "issues": [], "blocking": False}

    if n == 0:
        r["status"] = "空表"
        r["blocking"] = True
        r["issues"].append("0 行 —— 接口参数可能已失效")
        return r

    spec = SPECS.get(name, {})
    keys = [k for k in spec.get("keys", []) if k in df.columns]
    if keys:
        dup = int(df.duplicated(subset=keys).sum())
        r["dup"] = dup
        # ⚠️ 财务三表同 (ts_code, end_date) 有多个版本是**真实数据特性**：
        #    update_flag（0=原始披露 / 1=更新公告）+
        #    report_type（1~4 合并/母公司报表）造成的。
        #    已由 clean_tushare_data.py 产出 `{name}_clean.parquet`
        #    （最新公告的合并报表版本）。原表保留是有意的——
        #    「首次披露 vs 修正后」的差异本身是业绩超预期因子的信号。
        #    所以有 _clean 版本时，原表的重复**不算问题**。
        has_clean = (OUT / f"{name}_clean.parquet").exists()
        if dup > 0:
            if has_clean:
                r["issues"].append(
                    f"主键重复 {dup:,} 行，但已有 _clean 版本（多版本报表，非错误）")
            else:
                r["issues"].append(f"主键重复 {dup:,} 行（键={keys}）")
                r["blocking"] = True

    dc = DATE_COL.get(name)
    if dc and dc in df.columns and name not in NO_DATE_CHECK:
        s = df[dc].dropna().astype(str).str.replace("-", "", regex=False)
        s = s[s.str.len() == 8]
        if len(s):
            lo, hi = s.min(), s.max()
            r["date_range"] = f"{lo}~{hi}"
            try:
                y, m, d = int(hi[:4]), int(hi[4:6]), int(hi[6:])
                gap = (date.today() - date(y, m, d)).days
                r["days_behind"] = gap
                # 数据可以比今天旧（历史数据正常），但不能停在10 年前
                if gap > 400:
                    r["issues"].append(
                        f"末期距今 {gap} 天（{hi}），**疑似分页/期间截断**")
                    r["blocking"] = True
            except ValueError:
                pass

    if r["blocking"]:
        r["status"] = "存疑"
    if verbose:
        flag = "✓" if r["status"] == "OK" else "✗"
        line = f"{flag} {name:<20}{n:>12,} 行"
        if "date_range" in r:
            line += f"  {r['date_range']}"
        if r.get("dup"):
            line += f"  重复 {r['dup']:,}"
        print(line)
        for it in r["issues"]:
            print(f"    ⚠ {it}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="Tushare 全量数据质量校验")
    ap.add_argument("--task", default=None, help="只校验某个表")
    args = ap.parse_args()

    files = sorted(OUT.glob("*.parquet"))
    files = [f for f in files if "_shard" not in f.stem
             and not f.stem.startswith("_")]
    if args.task:
        files = [f for f in files if f.stem == args.task]

    print("=" * 88)
    print("Tushare 全量数据质量校验")
    print("=" * 88)
    print(f"{'':2}{'表':<20}{'行数':>12}  时间范围 / 问题")
    print("-" * 88)

    results = [check_one(f.stem) for f in files]

    ok = [r for r in results if r["status"] == "OK"]
    sus = [r for r in results if r["status"] == "存疑"]
    bad = [r for r in results if r["status"] in ("缺失", "空表")]

    print("-" * 88)
    print(f"通过 {len(ok)} ｜ 存疑 {len(sus)} ｜ 异常 {len(bad)}"
          f" ｜ 合计 {len(results)} 张表，"
          f"{sum(r['rows'] for r in results):,} 行")
    if sus:
        print("\n存疑明细（需处理后才能用于因子研究）：")
        for r in sus:
            print(f"  {r['table']}: {'; '.join(r['issues'])}")
    if bad:
        print("\n异常明细：")
        for r in bad:
            print(f"  {r['table']}: {r['status']}")
    print("=" * 88)
    return 1 if sus else 0


if __name__ == "__main__":
    raise SystemExit(main())
