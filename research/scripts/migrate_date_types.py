"""一次性迁移：把存量 parquet 的日期列从 string 修正为 int32。

为什么必须迁移存量数据
----------------------
`normalize_types` 只在**新下载**时生效（挂在 `save()` 上）。
但实测 19 张存量表全都已经中招：日期列存成 string，
导致 `df[df.trade_date == 20240102]`（int 比较）恒返回 0 行，
且**不报任何错**。

这类静默失效最难发现 —— 下游看到的是「该股票没有数据」，
而不是「类型不匹配」。若在它之上建了结论，结论就是错的。

⚠️ 迁移前先备份，迁移后逐表校验行数不变（类型转换不该改变行数）。
   行数变了说明数据被破坏了，必须回滚。

用法
----
  uv run python research/scripts/migrate_date_types.py --dry-run
  uv run python research/scripts/migrate_date_types.py --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from tushare_paths import DATE_COLS, OUT, normalize_types  # noqa: E402

BACKUP = ROOT / "runtime" / "tushare_backup_pre_datetype"


def date_cols_of(path: Path) -> list[str]:
    sch = pq.read_schema(path)
    return [c for c in DATE_COLS if c in sch.names]


def string_date_cols(path: Path) -> list[str]:
    """返回「类型错误」的日期列名。"""
    sch = pq.read_schema(path)
    bad = []
    for c in DATE_COLS:
        if c not in sch.names:
            continue
        t = str(sch.field(c).type)
        if t in ("string", "large_string", "double", "float",
                 "timestamp", "date32", "date64"):
            bad.append(c)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description="迁移 parquet 日期列类型")
    ap.add_argument("--apply", action="store_true",
                    help="真正写入（默认只报告不改动）")
    args = ap.parse_args()

    files = sorted(OUT.glob("*.parquet"))
    todo = []
    for f in files:
        bad = string_date_cols(f)
        if bad:
            todo.append((f, bad))

    if not todo:
        print("✓ 没有类型异常的表")
        return 0

    print(f"发现 {len(todo)} 张表日期列类型异常\n")
    print(f"{'表':<34}{'异常列':<30}{'行数':>12}")
    print("-" * 78)
    for f, bad in todo:
        n = pq.ParquetFile(f).metadata.num_rows
        print(f"{f.name:<34}{','.join(bad):<30}{n:>12,}")

    if not args.apply:
        print("\n（dry-run，加 --apply 真正执行）")
        return 0

    BACKUP.mkdir(parents=True, exist_ok=True)
    print(f"\n备份到 {BACKUP}")
    failed = []
    for f, bad in todo:
        before = pq.ParquetFile(f).metadata.num_rows
        shutil.copy2(f, BACKUP / f.name)
        try:
            df = pd.read_parquet(f)
            out = normalize_types(df)
            out.to_parquet(f, index=False)
            after = pq.ParquetFile(f).metadata.num_rows
            if before != after:
                raise RuntimeError(
                    f"行数变了：{before:,} → {after:,}，已从备份恢复")
            fixed = string_date_cols(f)
            if fixed:
                raise RuntimeError(f"迁移后仍是string: {fixed}")
            print(f"  ✓ {f.name:<32} {before:>12,} 行")
        except Exception as e:                                   # noqa: BLE001
            failed.append((f.name, str(e)))
            shutil.copy2(BACKUP / f.name, f)      # 回滚
            print(f"  ✗ {f.name:<32} {e}（已回滚）")

    print()
    if failed:
        print(f"❌ {len(failed)} 张表迁移失败并已回滚：")
        for name, err in failed:
            print(f"   {name}: {err}")
        return 1
    print(f"✅ 全部 {len(todo)} 张表迁移完成，行数校验通过")

    try:
        from data_version import build_manifest
        ver = build_manifest()
        print(f"\n数据版本（新）: {ver['version']}  "
              f"{ver['stamped_at']}  "
              f"{ver['counts']['ts_tables']} 张 tushare 表 / "
              f"{ver['counts']['db_tables']} 张库表")
    except Exception as e:                                       # noqa: BLE001
        print(f"\n⚠ 版本打戳失败: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())