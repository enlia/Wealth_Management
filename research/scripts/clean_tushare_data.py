"""清洗已下载的 Tushare 数据：去完全重复行 + 财务表取最新公告版本。

为什么需要
----------
2026-10-06 校验发现 `fina_indicator` 有 140,297 行**完全重复**（占 33%），
2,411 只股票各出现 100 次而正常只该有约 42 个报告期。
根因是逐只拉取的任务在中断重跑时把重复分片拼在了一起。

⚠️ **完全重复 ≠ 数据有多个版本**：
  · 完全重复（整行一模一样）→ 下载器的拼接问题，必须去重
  · 同主键不同值（update_flag 0/1、report_type 1~4）→ 真实数据特性，
    是「原始披露 vs 更新公告」「合并报表 vs 母公司报表」，**要保留**

所以清洗分两步，不能一把梭：
  1. 整行去重（丢掉下载器造成的假重复）
  2. 财务表按 (ts_code, end_date) 取最新 ann_date + 合并报表，
     另存为 `{table}_clean.parquet`，**不覆盖原表**

原表保留的理由：因子研究可能需要「首次披露 vs 修正后」的差异信息
（业绩超预期因子里，修正幅度本身是信号）。

用法
----
  uv run python research/scripts/clean_tushare_data.py--dry-run
  uv run python research/scripts/clean_tushare_data.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "runtime" / "tushare"

# 财务三表 + 财务指标：同 (ts_code, end_date) 有多版本
FIN_TABLES = {
    "income":         {"prefer_report_type": "1", "label": "利润表"},
    "balancesheet":   {"prefer_report_type": "1", "label": "资产负债表"},
    "cashflow":       {"prefer_report_type": "1", "label": "现金流量表"},
    "fina_indicator": {"prefer_report_type": None, "label": "财务指标"},
}


def drop_exact_dup(name: str, dry: bool) -> dict:
    """第 1 步：整行去重。"""
    f = OUT / f"{name}.parquet"
    if not f.exists():
        return {"table": name, "skipped": "文件不存在"}
    df = pd.read_parquet(f)
    before = len(df)
    out = df.drop_duplicates()
    removed = before - len(out)
    r = {"table": name, "before": before, "after": len(out),
         "exact_dup_removed": removed}
    if removed and not dry:
        out.to_parquet(f, index=False)
    return r


def build_fin_clean(name: str, dry: bool) -> dict:
    """第 2 步：财务表取「最新公告的合并报表」版本。"""
    spec = FIN_TABLES.get(name)
    if spec is None:
        return {"table": name, "skipped": "非财务表"}
    f = OUT / f"{name}.parquet"
    if not f.exists():
        return {"table": name, "skipped": "文件不存在"}

    df = pd.read_parquet(f).drop_duplicates()
    if "ts_code" not in df.columns or "end_date" not in df.columns:
        return {"table": name, "skipped": "缺 ts_code/end_date"}

    before = len(df)
    keys = ["ts_code", "end_date"]
    # 过滤 A 股（Tushare 里可能有 B 股/港股混入）
    df = df[df["ts_code"].astype(str).str.endswith((".SH", ".SZ", ".BJ"))]
    # 优先合并报表（report_type=1）；无该列时全部保留
    rt = spec.get("prefer_report_type")
    if rt and "report_type" in df.columns:
        merged = df[df["report_type"].astype(str) == rt]
        if len(merged):
            df = merged
    # 同 (ts_code, end_date) 取最新 ann_date
    if "ann_date" in df.columns:
        df = df.sort_values(["ts_code", "end_date", "ann_date"])
    out = df.drop_duplicates(subset=keys, keep="last")

    r = {"table": name, "label": spec["label"], "before": before,
         "after": len(out), "multi_version_removed": before - len(out)}
    if not dry:
        out.to_parquet(OUT / f"{name}_clean.parquet", index=False)
        r["out"] = f"{name}_clean.parquet"
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="清洗 Tushare 已下载数据")
    ap.add_argument("--dry-run", action="store_true", help="只报告不写文件")
    args = ap.parse_args()
    mode = "[DRY-RUN] " if args.dry_run else ""

    print("=" * 76)
    print(f"清洗 Tushare 数据 {mode}")
    print("=" * 76)

    print("\n第 1 步：整行去重（下载器造成的假重复）")
    print("-" * 76)
    targets = sorted({p.stem for p in OUT.glob("*.parquet")
                      if "_shard" not in p.stem
                      and not p.stem.endswith("_clean")})
    total_removed = 0
    for name in targets:
        r = drop_exact_dup(name, args.dry_run)
        if r.get("skipped"):
            continue
        n = r["exact_dup_removed"]
        total_removed += n
        flag = "⚠" if n else "✓"
        print(f"{flag} {name:<22}{r['before']:>12,} → {r['after']:>12,}"
              f"  -{n:,}")
    print(f"\n合计去掉完全重复行 {total_removed:,}")

    print("\n第 2 步：财务表取最新公告版本（另存 _clean，不覆盖原表）")
    print("-" * 76)
    for name in FIN_TABLES:
        r = build_fin_clean(name, args.dry_run)
        if r.get("skipped"):
            print(f"○ {name:<22}{r['skipped']}")
            continue
        print(f"✓ {name:<22}{r['label']:<8}{r['before']:>10,} → "
              f"{r['after']:>10,}  去掉多版本 {r['multi_version_removed']:,}")
        if "out" in r:
            print(f"{'':24}→ {r['out']}")

    print()
    print("=" * 76)
    print("原表已保留：因子研究可能需要「首次披露 vs 修正后」的差异")
    print("（业绩超预期因子里，修正幅度本身是信号）")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
