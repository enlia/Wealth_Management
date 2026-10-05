"""把 Tushare 补全数据并入本机数据库（复权 + 行业 + 退市股）。

⚠️ 并库前必须先跑 verify_tushare_data.py，覆盖率 <95% 时禁止执行。

本脚本做的事
------------
1. **复权**：用 adj_factor 把 bar_daily 的 close 修正为后复权价
   close_adj = close × adj_factor / latest_adj_factor
   → 只改 close，保留原始 open/high/low/close 作为备份
2. **行业**：用申万 index_member_all 补全 stock_info.industry（当前缺 37.2%）
3. **退市股**：扩展股票池清单（本机仅 16 只含退/ST，Tushare 有 339 只）

⚠️ 关键约束（实测踩过的坑）
   Tushare 的 adj_factor 包含**上市前的填充值**（adj_factor=1.0 恒定）。
   直接使用会让早期日期的复权价全错。
   → 必须按【本机实际有行情的日期】做交集，只保留真实交易日。

用法
----
  uv run python scripts/merge_tushare_into_db.py --dry-run    # 预演
  uv run python scripts/merge_tushare_into_db.py--adj-only
  uv run python scripts/merge_tushare_into_db.py                # 全量
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from factor_lab.config import DB_PATH, is_a_share  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "runtime" / "tushare"
BACKUP = Path(__file__).resolve().parents[2] / "runtime" / "backup"


def to_local(ts: str) -> str:
    num, ex = ts.split(".")
    pre = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(ex.upper())
    return f"{pre}{num}" if pre else ""


def merge_adj(con, dry_run: bool) -> int:
    """用 adj_factor 修正收盘价为后复权。"""
    print("\n" + "=" * 76)
    print("并库 1：复权修正")
    print("=" * 76)

    files = sorted(OUT.glob("adj_shard*.parquet"))
    if not files:
        p = OUT / "adj_factor.parquet"
        if not p.exists():
            print("  ✗ 无复权因子文件")
            return 0
        files = [p]
    adj = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    adj["code"] = adj["ts_code"].map(to_local)
    adj["dnum"] = adj["trade_date"].astype(int)
    adj = adj[adj["code"].str.len() > 0]
    print(f"  原始 {len(adj):,} 行，覆盖 {adj['code'].nunique():,} 只")

    # 只保留本机确实有行情的日期（剔除上市前填充）
    codes = adj["code"].unique().tolist()
    have = {}
    for i in range(0, len(codes), 400):
        ch = codes[i:i + 400]
        q = ",".join("?" * len(ch))
        for c, d in con.execute(
                f"SELECT code, date FROM bar_daily WHERE code IN ({q})", ch):
            have.setdefault(c, set()).add(int(d))
    mask = [d in have.get(c, ()) for c, d in zip(adj["code"], adj["dnum"])]
    dropped = (~pd.Series(mask, index=adj.index)).sum()
    adj = adj[mask]
    print(f"  剔除上市前填充 {int(dropped):,} 行后剩 {len(adj):,} 行")

    # 建索引
    adj_i = adj.set_index(["code", "dnum"])["adj_factor"]
    n_upd = 0
    t0 = time.perf_counter()
    for i, c in enumerate(adj["code"].unique()):
        sub = adj[adj["code"] == c][["dnum", "adj_factor"]]
        if len(sub) < 2:
            continue
        latest = sub["adj_factor"].iloc[-1]
        # 后复权：把末端价格拉平
        factor = sub["adj_factor"] / latest
        rows = con.execute(
            "SELECT date, close FROM bar_daily WHERE code=?", (c,)).fetchall()
        upd = [(c, d, cl, cl * f) for d, cl in
               ((r[0], r[1]) for r in rows)
               for f in [factor.get(d, np.nan)] if np.isfinite(f)]
        if not upd:
            continue
        if dry_run:
            n_upd += len(upd)
        else:
            con.executemany(
                "UPDATE bar_daily SET close_adj=? WHERE code=? AND date=?",
                [(u[3], u[0], u[1]) for u in upd])
            n_upd += len(upd)
        if (i + 1) % 500 == 0:
            el = (time.perf_counter() - t0) / 60
            print(f"    {i+1:,} 只  {n_upd:,} 行  {el:.1f} 分钟", flush=True)
    if not dry_run:
        con.commit()
    print(f"  {'预演' if dry_run else '已更新'} {n_upd:,} 条收盘价为后复权")
    return n_upd


def merge_industry(con, dry_run: bool) -> int:
    """用申万行业补全 stock_info.industry。"""
    print("\n" + "=" * 76)
    print("并库 2：行业补全")
    print("=" * 76)
    p = OUT / "index_member_all.parquet"
    if not p.exists():
        print("  ✗ 无行业成分文件")
        return 0
    d = pd.read_parquet(p)
    d["code"] = d["ts_code"].map(to_local)
    d = d[d["code"].str.len() > 0]
    # 取每只股票最细且最新的分类（L3 > L2 > L1）
    d["_lv"] = d["l3_name"].fillna("").replace("", np.nan)
    d["_lv"] = d["_lv"].fillna(d["l2_name"]).fillna(d["l1_name"])
    best = d.sort_values("_lv").drop_duplicates("code", keep="last")
    n = 0
    for _, r in best.iterrows():
        if dry_run:
            n += 1
        else:
            con.execute("UPDATE stock_info SET industry=? WHERE code=?",
                        (r["_lv"], r["code"]))
            n += 1
    if not dry_run:
        con.commit()
    print(f"  {'预演' if dry_run else '已更新'} {n:,} 只的行业分类")
    return n


def merge_delisted(con, dry_run: bool) -> int:
    """把退市股加入股票池清单（不拉行情，只标记）。"""
    print("\n" + "=" * 76)
    print("并库 3：退市股")
    print("=" * 76)
    p = OUT / "stock_basic_D.parquet"
    if not p.exists():
        print("  ✗ 无退市股文件")
        return 0
    d = pd.read_parquet(p)
    have = {r[0] for r in con.execute("SELECT code FROM stock_info")}
    new = []
    for _, r in d.iterrows():
        lc = to_local(r["ts_code"])
        if not lc or lc in have:
            continue
        new.append((lc, r.get("name", ""), r.get("list_date", ""),
                    r.get("delist_date", ""), "D"))
    if new and not dry_run:
        con.executemany(
            "INSERT OR IGNORE INTO stock_info"
            "(code, name, list_date, industry) VALUES (?,?,?,?)",
            [(a, b, c, "退市") for a, b, c, _, _ in new])
        con.commit()
    print(f"  {'预演' if dry_run else '已插入'} {len(new):,} 只退市股")
    return len(new)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tushare 数据并库")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--adj-only", action="store_true")
    ap.add_argument("--no-industry", action="store_true")
    args = ap.parse_args()

    # 前置检查
    chk = subprocess_run_verify()
    if chk != 0:
        print("\n✗ 质量校验未通过，拒绝并库")
        print("  请先修复问题，或运行 verify_tushare_data.py 查看详情")
        return 1

    print("=" * 76)
    print(f"Tushare 数据并库（{'预演' if args.dry_run else '实际写入'}）")
    print("=" * 76)
    print(f"数据库: {DB_PATH}")

    if not args.dry_run:
        BACKUP.mkdir(parents=True, exist_ok=True)
        bak = BACKUP / f"market_{time.strftime('%Y%m%d_%H%M%S')}.db"
        print(f"备份到: {bak.name}")
        shutil.copy2(DB_PATH, bak)

    con = sqlite3.connect(DB_PATH)
    t0 = time.perf_counter()
    merge_adj(con, args.dry_run)
    if not args.adj_only:
        if not args.no_industry:
            merge_industry(con, args.dry_run)
        merge_delisted(con, args.dry_run)
    con.close()
    print()
    print(f"耗时 {(time.perf_counter()-t0)/60:.1f} 分钟")
    print("下一步：uv run python scripts/audit_data_quality.py --all")
    return 0


def subprocess_run_verify() -> int:
    """并库前强制跑质量校验。"""
    print("→ 先跑质量校验…")
    import subprocess
    r = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "verify_tushare_data.py"),
         "--adj-only"],
        capture_output=True, text=True)
    tail = [ln for ln in r.stdout.split("\n") if "通过" in ln or "结论" in ln]
    for ln in tail[-3:]:
        print("   ", ln.strip())
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
