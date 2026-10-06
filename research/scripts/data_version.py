"""数据版本管理：给每次研究锁定一份可复现的数据快照。

为什么必须有
------------
因子研究的核心风险是**「结论建立在什么数据上说不清」**：
  · 数据被重新下载 / 清洗过，旧结论不再对应同一份数据
  · 半年后想复现「当时那个 IC=0.08」，不知道用的是哪版
  · 数据出问题要回滚，没有版本就无从下手

所以：**每次研究必须绑定一个数据版本号**，版本号变了结论必须重跑。

版本号怎么算
------------
不是 git commit（数据文件在 .gitignore 里），而是**内容哈希**：
```
version = sha256(表名 + 行数 + 各表文件内容哈希)[:12]
```
同一份数据 → 同一版本号；数据变一个字节 → 版本号变。

⚠️ **哈希必须基于文件内容，不能只用行数**：
实测 `disclosure_date` 去重前 12 万行、去重后 4.7 万行，
行数变了能发现；但同报告期的修正公告版本变化时行数不变，
只有内容哈希能发现。

快照包含什么
------------
1. ``manifest.json`` —— 版本号、生成时间、各表的行数/大小/哈希
2. ``db_meta.json``   —— 本机库的表清单与关键字段统计
3. 因子研究用的数据区间、股票池大小

⚠️ **不做数据拷贝**。1.2 GB 的 parquet 拷一份很慢且浪费。
快照存的是「指纹」，真要复现时用同样的代码+ 同样的版本号重新生成即可。

用法
----
  uv run python research/scripts/data_version.py --stamp      # 生成当前版本
  uv run python research/scripts/data_version.py --show       # 查看当前版本
  uv run python research/scripts/data_version.py --history    # 历史版本
  uv run python research/scripts/data_version.py--check      # 校验数据是否被改动
  uv run python research/scripts/data_version.py --stamp --note "加完P1 涨跌停"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from factor_lab.config import DB_PATH  # noqa: E402

TS_DIR = ROOT / "runtime" / "tushare"
VERSION_DIR = ROOT / "runtime" / "versions"
HISTORY_FILE = VERSION_DIR / "history.json"

# 参与版本号的表（这些变了会影响研究结论）。
# ⚠️ 不含 trade_cal（交易日历是固定的）与 _money_whitelist（配置不是数据）。
SKIP_PREFIX = ("_",)
SKIP_NAMES = {"trade_cal.parquet", "_money_whitelist.json"}


def file_sha256(p: Path, chunk: int = 1 << 20) -> str:
    """文件内容哈希（分块读，避免大文件占内存）。"""
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parquet_fingerprint(p: Path) -> dict:
    """parquet 指纹：行数 + 列名 + 文件哈希。

    只用文件哈希不够 —— parquet 的元数据可能变而数据没变；
    只用行数更不够 —— 同报告期的修正版本行数不变。
    """
    try:
        import pyarrow.parquet as pq
        n = pq.ParquetMetadata(str(p)).num_rows
        cols = pq.ParquetSchema(str(p)).names
    except Exception:                                    # noqa: BLE001
        df = pd.read_parquet(p)
        n, cols = len(df), list(df.columns)
    return {
        "rows": int(n),
        "ncols": len(cols),
        "bytes": p.stat().st_size,
        "sha256": file_sha256(p)[:16],
    }


def db_fingerprint() -> dict:
    """本机库指纹：各表行数 + 关键列的非空率。"""
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    out: dict = {"path": str(DB_PATH), "tables": {}}
    names = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    for t in names:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.Error:
            continue
        out["tables"][t] = int(n)
    con.close()
    return out


def build_manifest() -> dict:
    files: dict[str, dict] = {}
    for p in sorted(TS_DIR.glob("*.parquet")):
        if p.name in SKIP_NAMES or p.name.startswith(SKIP_PREFIX):
            continue
        files[p.stem] = parquet_fingerprint(p)

    db = db_fingerprint()

    # 版本号 = 全部指纹的哈希
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode())
        h.update(files[name]["sha256"].encode())
    h.update(json.dumps(db["tables"], sort_keys=True).encode())
    version = h.hexdigest()[:12]

    return {
        "version": version,
        "stamped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ts_files": files,
        "db": db,
        "totals": {
            "ts_tables": len(files),
            "ts_rows": sum(f["rows"] for f in files.values()),
            "db_tables": len(db["tables"]),
            "db_rows": sum(db["tables"].values()),
        },
    }


def stamp(note: str = "") -> dict:
    VERSION_DIR.mkdir(parents=True, exist_ok=True)
    m = build_manifest()
    m["note"] = note
    m["ts_files"].pop("_money_whitelist", None)

    # 存历史
    hist = json.loads(HISTORY_FILE.read_text(encoding="utf-8")
                     ) if HISTORY_FILE.exists() else []
    # 同一版本不重复记
    if not any(h["version"] == m["version"] for h in hist):
        hist.append({
            "version": m["version"],
            "at": m["stamped_at"],
            "note": note,
            "ts_rows": m["totals"]["ts_rows"],
            "ts_tables": m["totals"]["ts_tables"],
            "db_rows": m["totals"]["db_rows"],
        })
        hist.sort(key=lambda x: x["at"])
        HISTORY_FILE.write_text(
            json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")

    (VERSION_DIR / f"manifest_{m['version']}.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    # latest 指针
    (VERSION_DIR / "latest.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return m


def show() -> None:
    p = VERSION_DIR / "latest.json"
    if not p.exists():
        print("尚未打版本戳。先跑：--stamp")
        return
    m = json.loads(p.read_text(encoding="utf-8"))
    t = m["totals"]
    print(f"当前数据版本: {m['version']}   {m['stamped_at']}")
    if m.get("note"):
        print(f"  备注: {m['note']}")
    print(f"  Tushare {t['ts_tables']} 张表 / {t['ts_rows']:,} 行")
    print(f"  本机库{t['db_tables']} 张表 / {t['db_rows']:,} 行")


def history() -> None:
    if not HISTORY_FILE.exists():
        print("尚无历史版本")
        return
    hist = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    print(f"共 {len(hist)} 个数据版本\n")
    print(f"{'版本':<14}{'打戳时间':<20}{'Tushare行':>12}{'库行':>14}备注")
    print("-" * 92)
    for h in hist:
        print(f"{h['version']:<14}{h['at']:<20}{h['ts_rows']:>12,}"
              f"{h['db_rows']:>14,}  {h.get('note', '')}")


def check() -> int:
    """校验当前数据是否与最新版本一致。"""
    if not (VERSION_DIR / "latest.json").exists():
        print("尚未打版本戳，无法校验")
        return 1
    old = json.loads((VERSION_DIR / "latest.json").read_text(encoding="utf-8"))
    cur = build_manifest()

    diff: list[str] = []
    for name, fp in cur["ts_files"].items():
        o = old["ts_files"].get(name)
        if o is None:
            diff.append(f"+ {name}:新增（{fp['rows']:,} 行）")
        elif o["sha256"] != fp["sha256"]:
            diff.append(f"~ {name}: 内容已变（{o['rows']:,} → {fp['rows']:,} 行）")
    for name in old["ts_files"]:
        if name not in cur["ts_files"]:
            diff.append(f"- {name}: 已删除")

    print(f"打戳版本: {old['version']}  ({old['stamped_at']})")
    print(f"当前数据: {cur['version']}")
    if old["version"] == cur["version"]:
        print("\n✓ 数据未变动")
        return 0
    print(f"\n⚠ 数据已变动，{len(diff)} 处差异：")
    for d in diff[:30]:
        print(f"  {d}")
    if len(diff) > 30:
        print(f"  ... 另有 {len(diff) - 30} 处")
    print("\n→ 旧版本的研究结论不再对应当前数据，需重跑或改用：")
    print("    uv run python research/scripts/data_version.py --stamp")
    print(f"  旧 manifest 仍保留在 {VERSION_DIR}/manifest_<版本>.json")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description="数据版本管理")
    ap.add_argument("--stamp", action="store_true", help="生成当前数据版本")
    ap.add_argument("--show", action="store_true", help="查看当前版本")
    ap.add_argument("--history", action="store_true", help="历史版本")
    ap.add_argument("--check", action="store_true", help="校验数据是否被改动")
    ap.add_argument("--note", default="", help="版本备注")
    args = ap.parse_args()

    if args.stamp:
        m = stamp(args.note)
        print(f"✓ 已打版本戳: {m['version']}   {m['stamped_at']}")
        t = m["totals"]
        print(f"  Tushare {t['ts_tables']} 张表 / {t['ts_rows']:,} 行")
        print(f"  本机库   {t['db_tables']} 张表 / {t['db_rows']:,} 行")
        if args.note:
            print(f"  备注: {args.note}")
        return 0
    if args.check:
        return check()
    if args.history:
        history()
        return 0
    # 默认 show
    if args.show:
        show()
    else:
        show()
        print("\n用法: --stamp / --show / --history / --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
