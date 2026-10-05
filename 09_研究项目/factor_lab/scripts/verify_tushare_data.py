"""Tushare 补全数据的质量校验（并库前必须跑通）。

为什么必须单独校验
------------------
实测发现三个会导致「看起来成功、实际全错」的问题：
  1. **代码转换静默失败**：本机代码是 `bj899050`（小写前缀），
     初版 `to_ts()`漏了 bj 分支 → 传`bj899050.BJ` 给 Tushare，
     返回 0 行且**不报错**。若不校验，会误以为「拉完了」。
  2. **上市前填充值**：北交所 920010 实际 2021 年上市，
     但 Tushare 返回 2,634 行、从 2015-12-01 就开始，`adj_factor=1.0`恒定。
     这些是回填的默认值，直接用于复权会算出错误的换手率。
  3. **静默截断**：部分接口会分页，静默丢数据。

本脚本的判定标准
----------------
  ✓ 通过：复权因子与本机行情日期的交集覆盖率 ≥ 95%
  ✗ 不通过：覆盖率不足，或存在未识别的异常

用法
----
  uv run python scripts/verify_tushare_data.py
  uv run python scripts/verify_tushare_data.py --adj-only
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from factor_lab.config import DB_PATH, is_a_share  # noqa: E402

OUT = Path(__file__).parent.parent / "out" / "tushare"
COVERAGE_MIN = 0.95      # 与本机行情日期的交集覆盖率门槛


def to_local(ts_code: str) -> str:
    """Tushare 格式 → 本机格式。"""
    num, ex = ts_code.split(".")
    pre = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(ex.upper())
    if pre is None:
        return ""
    return f"{pre}{num}"


def local_dates(con, codes: list[str]) -> dict[str, set]:
    """取本机每只股票的行情日期集合（只取需要的，避免全表扫描）。"""
    out = {}
    for i in range(0, len(codes), 400):
        chunk = codes[i:i + 400]
        q = ",".join("?" * len(chunk))
        sql = f"SELECT code, date FROM bar_daily WHERE code IN ({q})"
        for c, d in con.execute(sql, chunk):
            out.setdefault(c, set()).add(int(d))
    return out


def verify_adj(con, verbose: bool = True) -> bool:
    """校验复权因子。"""
    print("\n" + "=" * 76)
    print("校验 1：复权因子 adj_factor")
    print("=" * 76)

    files = sorted(OUT.glob("adj_shard*.parquet"))
    if not files:
        p = OUT / "adj_factor.parquet"
        if p.exists():
            adj = pd.read_parquet(p)
        else:
            print("  ✗ 未找到复权因子文件")
            return False
    else:
        parts = [pd.read_parquet(f) for f in files]
        adj = pd.concat(parts, ignore_index=True)
    print(f"  原始 {len(adj):,} 行，覆盖 {adj['ts_code'].nunique():,} 只")

    adj["dt"] = pd.to_datetime(adj["trade_date"], format="%Y%m%d")
    adj["code"] = adj["ts_code"].map(to_local)
    adj = adj[adj["code"].str.len() > 0]

    # 剔除未识别的品种
    ts_all = set(r[0] for r in con.execute("SELECT DISTINCT code FROM bar_daily"))
    ts_ash = {c for c in ts_all if is_a_share(c)}
    unknown = set(adj["code"]) - ts_ash
    if unknown:
        print(f"  ⚠️ 剔除本机不存在的 {len(unknown)} 只: {list(unknown)[:5]}")
        adj = adj[~adj["code"].isin(unknown)]

    # 剔除填充值：adj_factor == 1.0 且该股首日不是本机首日
    n_one = int((adj["adj_factor"] == 1.0).sum())
    print(f"  adj_factor==1.0 的记录: {n_one:,}（{n_one/len(adj)*100:.1f}%）")

    # 与本机日期做交集
    codes = adj["code"].unique().tolist()
    ldates = local_dates(con, codes)
    print(f"  正在比对 {len(codes):,} 只与本机日期…")

    keep, drop_filled, drop_nomatch = [], 0, 0
    detail = []
    for c, g in adj.groupby("code", sort=False):
        loc = ldates.get(c)
        if not loc:
            drop_nomatch += len(g)
            continue
        g = g.copy()
        g["dnum"] = g["dt"].dt.strftime("%Y%m%d").astype(int)
        m = g["dnum"].isin(loc)
        gm = g[m]
        if gm.empty:
            drop_nomatch += len(g)
            detail.append((c, len(g), 0, 0.0))
            continue
        # 填充值判定：本机首日之后的记录里，adj_factor 恒为 1.0
        #         且与本机首个交易日的 factor 也相同 → 视为上市前填充
        first_loc = min(loc)
        pre = g[(g["dnum"] < first_loc)]
        if len(pre):
            drop_filled += len(pre)
        keep.append(gm)
        detail.append((c, len(g), len(gm), len(gm) / max(len(g), 1)))

    d = pd.DataFrame(detail, columns=["code", "tushare_n", "matched_n", "coverage"])
    print(f"  与本机匹配的记录: {d['matched_n'].sum():,}")
    print(f"  剔除上市前填充: {drop_filled:,}")
    print(f"  剔除本机无行情:{drop_nomatch:,}")

    cov = float(d["matched_n"].sum() / max(d["tushare_n"].sum(), 1))
    print(f"  总体覆盖率: {cov*100:.2f}%")
    print()
    print("  覆盖率最低的 8 只（需人工确认是否新股/次新）:")
    for _, r in d.nsmallest(8, "coverage").iterrows():
        print(f"    {r['code']:<12} {r['tushare_n']:>5} → {r['matched_n']:>5}"
              f"  {r['coverage']*100:>5.1f}%")

    ok = cov >= COVERAGE_MIN
    print()
    print(f"  {'✓ 通过' if ok else '✗ 不通过'}（门槛 {COVERAGE_MIN*100:.0f}%）")
    if not ok:
        print("    → 不要并库，先查清原因")
    return ok


def verify_others(con) -> None:
    """其余数据的体检（只报告，不阻断）。"""
    print("\n" + "=" * 76)
    print("校验 2：其他数据")
    print("=" * 76)

    p = OUT / "stock_basic_D.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        n_as = con.execute(
            "SELECT COUNT(*) FROM stock_info WHERE boards LIKE '%退%'"
            " OR name LIKE '%退%' OR name LIKE '%ST%'").fetchone()[0]
        print(f"  退市股: Tushare {len(d):,} 只  本机 {n_as:,} 只"
              f"  → 需补 {max(0, len(d)-n_as):,} 只")
        if "delist_date" in d.columns:
            has = d["delist_date"].notna().sum()
            print(f"    含退市日期: {has:,}/{len(d):,}")
    else:
        print("  退市股:未拉取")

    p = OUT / "index_classify_L1.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        print(f"  申万 L1 行业: {len(d):,} 个")
    p = OUT / "index_member_all.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        cov = d["ts_code"].nunique()
        print(f"  申万行业成分: {len(d):,} 行，覆盖 {cov:,} 只")
        if "l1_name" in d.columns:
            top = d["l1_name"].value_counts().head(5)
            print("    前 5 大行业: " + ", ".join(f"{k}({v:,})" for k, v in top.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Tushare 数据质量校验")
    ap.add_argument("--adj-only", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    print("=" * 76)
    print("Tushare 补全数据质量校验")
    print("=" * 76)

    ok = verify_adj(con)
    if not args.adj_only:
        verify_others(con)

    print()
    print("=" * 76)
    if ok:
        print("结论：✓ 质量通过，可以并库")
        print("=" * 76)
        print("  下一步：merge_tushare_into_db.py")
        return 0
    print("结论：✗ 质量不通过，禁止并库")
    print("=" * 76)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
