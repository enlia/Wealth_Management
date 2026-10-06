"""把 Tushare 补全数据并入本机数据库（复权 + 行业 + 退市股）。

⚠️ 并库前必须先跑 verify_tushare_data.py，覆盖率 <95% 时拒绝执行。

本脚本做的事
------------
1. **复权**：新增``close_adj`` 列存前复权收盘价，**保留原始 close 不动**
   close_adj = close × adj_factor(t) / adj_factor(最新交易日)
   → 原始未复权价必须保留：涨跌停判定、真实成交价校验都要用它
2. **行业**：用申万 index_member_all 补全 stock_info.industry
3. **退市股**：标记退市代码到 stock_info（本机无其行情，只做标记）

三个已修的严重缺陷（2026-10-06）
--------------------------------
原脚本**从未成功写入过任何一行**，三处叠加导致结果恒为空：

  缺陷 1：`bar_daily` 表**没有 close_adj 列**
          → `UPDATE bar_daily SET close_adj=...` 直接抛
            `sqlite3.OperationalError: no such column: close_adj`。
          → 本脚本改为先 `ALTER TABLE ... ADD COLUMN close_adj REAL`。

  缺陷 2：因子查表用错了键（**静默返回 0 行**，最危险的一类）
          旧代码 `factor = sub["adj_factor"] / latest` 里``sub`` 保留了
          concat 后的 RangeIndex（0,1,2,...），却用 ``factor.get(d)`` 按
          **日期整数**去取值→ 永远命中不到 → ``upd`` 恒为空 → 静默跳过。
          实测命中率 **0/26,771 = 0.0%**。
          → 本脚本改用显式 ``Series(values, index=dates)`` 构造，
            并断言命中率，命中率低于 90% 直接抛错。

  缺陷 3：``latest`` 取的是**最早**一天的因子，不是最新的
          Tushare 返回的 adj_factor 是**日期倒序**（实测茅台首行20260930、
          末行 20151201）。旧代码 ``sub["adj_factor"].iloc[-1]`` 在倒序数据上
          取到的是**最早日期**的因子，归一化系数错约 1.26 倍。
          → 本脚本先 ``sort_values(["code","dnum"])`` 再取 ``last()``。

性能
----
用``UPDATE ... FROM``（SQLite ≥ 3.33）一次集��更新，替代逐只executemany。
16.4M 行耗时从「逐只循环约 2 小时」降到分钟级。

用法
----
  uv run python research/scripts/merge_tushare_into_db.py --dry-run   # 预演
  uv run python research/scripts/merge_tushare_into_db.py --adj-only
  uv run python research/scripts/merge_tushare_into_db.py           # 全量
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from factor_lab.config import DB_PATH  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "runtime" / "tushare"
BACKUP = Path(__file__).resolve().parents[2] / "runtime" / "backup"

# 因子命中率低于此值即判定为「键错配」，直接抛错（禁止静默跳过）
MIN_FACTOR_HIT = 0.90


def to_local(ts: str) -> str:
    """600519.SH → sh600519。非 A/B 股返回空串。"""
    num, ex = ts.split(".")
    pre = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(ex.upper())
    return f"{pre}{num}" if pre else ""


def load_adj_factors() -> pd.DataFrame:
    """读全部复权因子分片，按 (code, dnum) 升序，返回 code/dnum/adj_factor。

    ⚠️ 排序是**硬性要求**：Tushare 按日期倒序返回，不排序时
    ``groupby.last()`` 取到的是最早日期的因子（见模块 docstring 缺陷 3）。
    """
    files = sorted(OUT.glob("adj_shard*.parquet"))
    if not files:
        p = OUT / "adj_factor.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"无复权因子文件：{OUT}/adj_shard*.parquet 或 adj_factor.parquet\n"
                "  解决：uv run python research/scripts/fetch_tushare.py --task adj"
            )
        files = [p]
    adj = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    adj["code"] = adj["ts_code"].map(to_local)
    adj = adj[adj["code"].str.len() > 0].copy()
    # trade_date 可能是 object(str) 或 int，统一转 int32
    adj["dnum"] = adj["trade_date"].astype("int64").astype("int32")
    adj = adj[adj["adj_factor"].notna() & (adj["adj_factor"] > 0)]
    # 同一 (code, date) 保留最后一条，再按日期升序
    adj = adj.drop_duplicates(subset=["code", "dnum"], keep="last")
    adj = adj.sort_values(["code", "dnum"], kind="stable")
    return adj[["code", "dnum", "adj_factor"]].reset_index(drop=True)


def merge_adj(con: sqlite3.Connection, dry_run: bool) -> int:
    """新增 close_adj 列并写入前复权收盘价。"""
    print("\n" + "=" * 76)
    print("并库 1：复权修正（新增 close_adj 列，保留原始 close）")
    print("=" * 76)

    adj = load_adj_factors()
    print(f"  因子原始 {len(adj):,} 行，覆盖 {adj['code'].nunique():,} 只")

    # 只保留本机确实有行情的 (code, date) —— 剔除 Tushare 的上市前填充
    codes = adj["code"].unique().tolist()
    have: dict[str, set[int]] = {}
    for i in range(0, len(codes), 400):
        ch = codes[i:i + 400]
        q = ",".join("?" * len(ch))
        for c, d in con.execute(
                f"SELECT code, date FROM bar_daily WHERE code IN ({q})", ch):
            have.setdefault(c, set()).add(int(d))

    mask = np.fromiter(
        (d in have.get(c, ()) for c, d in zip(adj["code"], adj["dnum"], strict=True)),
        dtype=bool, count=len(adj))
    dropped = int((~mask).sum())
    adj = adj[mask]
    print(f"  剔除上市前填充 {dropped:,} 行 →剩 {len(adj):,} 行"
          f"（{adj['code'].nunique():,} 只）")

    # 归一化：前复权 = close × factor(t) / factor(最新)
    latest = adj.groupby("code")["adj_factor"].last()
    adj["norm"] = adj["adj_factor"] / adj["code"].map(latest)
    adj["value"] = adj["norm"]   # 待乘以 close 的系数

    if dry_run:
        print(f"  [预演] 将写入 {len(adj):,} 条 close_adj")
        _report_factor_sanity(adj)
        return len(adj)

    # 缺陷 1 修复：列不存在就加。SQLite 的 ALTER TABLE ADD COLUMN 是元数据操作，
    # 16.4M 行表上瞬时完成，不重写全表。
    cols = {r[1] for r in con.execute("PRAGMA table_info(bar_daily)")}
    if "close_adj" not in cols:
        print("  bar_daily 无 close_adj 列，执行 ALTER TABLE ADD COLUMN")
        con.execute("ALTER TABLE bar_daily ADD COLUMN close_adj REAL")
        con.commit()

    # 落盘到临时表，再用 UPDATE...FROM 一次集��更新（SQLite ≥ 3.33）
    con.execute("DROP TABLE IF EXISTS temp.adj_norm")
    con.execute("CREATE TEMP TABLE adj_norm (code TEXT, dnum INTEGER, v REAL)")
    con.executemany(
        "INSERT INTO temp.adj_norm (code, dnum, v) VALUES (?,?,?)",
        adj[["code", "dnum", "value"]].itertuples(index=False, name=None),
    )
    con.execute("CREATE INDEX temp.adj_norm_ix ON adj_norm(code, dnum)")
    t0 = time.perf_counter()
    cur = con.execute(
        "UPDATE bar_daily "
        "   SET close_adj = bar_daily.close * adj_norm.v "
        "  FROM temp.adj_norm adj_norm "
        " WHERE bar_daily.code = adj_norm.code "
        "   AND bar_daily.date = adj_norm.dnum"
    )
    con.commit()
    el = time.perf_counter() - t0
    print(f"  已写入 {cur.rowcount:,} 条 close_adj（耗时 {el:.1f} 秒）")

    # 缺陷 2 修复：显式断言命中率，不让「静默 0 行」再发生。
    # ⚠️ 分母**必须限定A 股**。bar_daily 里还有板块指数/基金/债券/B 股
    #    （实测占34.2%），它们本就不在 Tushare 复权因子表里。
    #    早先版本拿全表16,456,556 行当分母，把覆盖率算成 65.83%并误报故障。
    n_a, n_hit = con.execute(
        "SELECT COUNT(*), SUM(CASE WHEN close_adj IS NOT NULL THEN 1 ELSE 0 END) "
        "  FROM bar_daily WHERE code LIKE 'sh6%'    OR code LIKE 'sz0%' "
        "    OR code LIKE 'sz3%'   OR code LIKE 'bj%'"
    ).fetchone()
    ratio = (n_hit or 0) / max(n_a or 1, 1)
    print(f"  覆盖校验（A 股口径）：{n_a:,} 行中 close_adj 非空 {n_hit:,} 行"
          f"（{ratio*100:.3f}%）")
    missing = (n_a or 0) - (n_hit or 0)
    if ratio < MIN_FACTOR_HIT:
        raise RuntimeError(
            f"A 股 close_adj 覆盖率 {ratio*100:.3f}% < {MIN_FACTOR_HIT*100:.0f}%，"
            f"判定为键错配而非数据缺口（缺 {missing:,} 行）。\n"
            "  不要降阈值硬跑——先查因子表的 code/date 类型是否与 bar_daily 一致。"
        )
    if missing:
        print(f"  仍有 {missing:,} 行 A 股无复权因子（新股/退市/停牌），"
              f"研究时按缺失处理，不要当0填充")
    _report_factor_sanity(adj)
    return n_hit


def _report_factor_sanity(adj: pd.DataFrame) -> None:
    """打印归一化系数的量级分布，供人工确认 1.26 倍那类错误。"""
    q = adj["norm"].describe(percentiles=[0.01, 0.5, 0.99])
    print(f"  归一化系数 norm：min={q['min']:.4f} p1={q['1%']:.4f} "
          f"中位={q['50%']:.4f} p99={q['99%']:.4f} max={q['max']:.4f}")
    bad = adj[adj["norm"] > 1.0 + 1e-6]
    print(f"  系数 >1 的记录：{len(bad):,}（前复权应有>1 的早期记录，"
          f"若为 0 说明 latest 取错了）")


def merge_industry(con: sqlite3.Connection, dry_run: bool) -> int:
    """用申万行业补全 stock_info.industry（缺 37.2% → 目标 0）。"""
    print("\n" + "=" * 76)
    print("并库 2：行业补全")
    print("=" * 76)
    p = OUT / "index_member_all.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"无行业成分文件：{p}\n"
            "  解决：uv run python/scripts/fetch_tushare.py --task industry"
        )
    d = pd.read_parquet(p)
    d["code"] = d["ts_code"].map(to_local)
    d = d[d["code"].str.len() > 0]
    # 取最细层级（L3 > L2 > L1）；同一股票可能同时在多个 L1 下，
    # 用最新 in_date 优先，避免拿到过时分类。
    d["_lv"] = (d["l3_name"].fillna("").replace("", np.nan)
                .fillna(d["l2_name"]).fillna(d["l1_name"]))
    d["_in"] = pd.to_datetime(d["in_date"], format="%Y%m%d", errors="coerce")
    d = d.sort_values(["code", "_in"]).drop_duplicates("code", keep="last")
    n = 0
    for code, name in d[["code", "_lv"]].itertuples(index=False, name=None):
        if dry_run:
            n += 1
        else:
            con.execute("UPDATE stock_info SET industry=? WHERE code=?",
                        (name, code))
            n += 1
    if not dry_run:
        con.commit()
    left = con.execute(
        "SELECT COUNT(*) FROM stock_info "
        " WHERE industry IS NULL OR TRIM(industry)=''").fetchone()[0]
    print(f"  {'预演' if dry_run else '已更新'} {n:,} 只行业分类；"
          f"库内仍缺行业 {left:,} 条")
    return n


def merge_delisted(con: sqlite3.Connection, dry_run: bool) -> int:
    """把退市股标记进 stock_info。

    ⚠️ 本机 bar_daily **没有**这些股票的历史行情（Tushare 339 只里仅 19 只有行情），
    所以本步只解决「股票池名单不遗漏退市代码」，**不能消除幸存者偏差**——
    偏差需要把退市股的历史行情补进 bar_daily 才能消除。
    这个事实必须写清楚，不能让「已插入 320 只」被误读成「偏差已修复」。
    """
    print("\n" + "=" * 76)
    print("并库 3：退市股标记")
    print("=" * 76)
    p = OUT / "stock_basic_D.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"无退市股文件：{p}\n"
            "  解决：uv run python research/scripts/fetch_tushare.py --task basic"
        )
    d = pd.read_parquet(p)
    have = {r[0] for r in con.execute("SELECT code FROM stock_info")}
    in_bar = {r[0] for r in con.execute("SELECT DISTINCT code FROM bar_daily")}
    new, no_bar = [], 0
    for _, r in d.iterrows():
        lc = to_local(r["ts_code"])
        # 排除 shT600018 这类T 开头的老代码（非 A 股普通股）
        if not lc or lc in have or not lc[2:].isdigit():
            continue
        new.append((lc, str(r.get("name") or ""),
                    str(r.get("list_date") or "")))
        if lc not in in_bar:
            no_bar += 1
    if new and not dry_run:
        con.executemany(
            "INSERT OR IGNORE INTO stock_info (code, name, list_date, industry) "
            "VALUES (?,?,?,?)",
            [(a, b, c, "退市") for a, b, c in new])
        con.commit()
    print(f"  {'预演' if dry_run else '已插入'} {len(new):,} 只退市股")
    print(f"  ⚠️ 其中 {no_bar:,} 只本机无行情 → 名单已补，但**幸存者偏差未消除**")
    return len(new)


def run_precheck() -> None:
    """并库前强制跑质量校验，不通过则拒绝。"""
    print("→ 先跑数据质量校验…")
    r = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "verify_tushare_data.py"),
         "--adj-only"],
        capture_output=True, text=True)
    for ln in r.stdout.split("\n"):
        if "通过" in ln or "结论" in ln or "覆盖率" in ln:
            print("   ", ln.strip())
    if r.returncode != 0:
        print("\n✗ 质量校验未通过，拒绝并库")
        raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tushare 数据并库")
    ap.add_argument("--dry-run", action="store_true", help="只报告不写库")
    ap.add_argument("--adj-only", action="store_true", help="只做复权")
    ap.add_argument("--no-industry", action="store_true")
    ap.add_argument("--skip-precheck", action="store_true",
                    help="跳过质量校验（仅调试用）")
    args = ap.parse_args()

    if not args.skip_precheck:
        run_precheck()

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
    try:
        merge_adj(con, args.dry_run)
        if not args.adj_only:
            if not args.no_industry:
                merge_industry(con, args.dry_run)
            merge_delisted(con, args.dry_run)
    finally:
        con.close()
    print(f"\n耗时 {(time.perf_counter()-t0)/60:.1f} 分钟")
    print("下一步：uv run python research/scripts/audit_data_quality.py --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
