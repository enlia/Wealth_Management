"""把 Tushare 补全的日频/事件表并入本机库（独立 ts_ 前缀表）。

为什么单独一个模块
------------------
``merge_tushare_into_db.py`` 专攻复权（改``bar_daily`` 加 close_adj 列）
与静态信息（补 stock_info 的行业/退市标记）。
本模块处理的是**新增的独立表**，两者写入方式完全不同：
  · 复权：UPDATE 已有表
  · 这里：CREATE TABLE + 批量 INSERT

混在一起会让并库脚本破500 行，也会让「复权出错」和「新表导入出错」互相干扰。

表命名
------
统一 ``ts_`` 前缀，与本机原有表隔离：
  ts_stk_limit / ts_moneyflow / ts_margin_detail / ts_dividend ...

⚠️ **不写进 bar_daily**。因子研究需要同时用「未复权价（算涨跌停）」
和「后复权价（算收益）」，两张表并列才不会互相污染。

数据类型约定
------------
Tushare 的 ts_code 是 ``600519.SH``，本机是 ``sh600519``。
统一存**本机格式**（``ts_to_local``），因子查表时不用每次转换。
日期统一存**整数** YYYYMMDD，与 ``bar_daily.date`` 同类型，
可以直接 JOIN 而不用类型转换。

用法
----
  uv run python research/scripts/merge_tushare_tables.py --dry-run
  uv run python research/scripts/merge_tushare_tables.py
  uv run python research/scripts/merge_tushare_tables.py --only dividend
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from factor_lab.config import DB_PATH  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "runtime" / "tushare"
BACKUP = Path(__file__).resolve().parents[2] / "runtime" / "backup"

# 要并入的表：tushare 文件名 → (本机表名, 日期列, 额外索引列)
# ⚠️ 财务类多版本表用 _clean 版本（见 clean_tushare_data.py）
TABLES: dict[str, tuple[str, str, list[str]]] = {
    # 日频
    "stk_limit":       ("ts_stk_limit", "trade_date", ["ts_code", "trade_date"]),
    "moneyflow":       ("ts_moneyflow", "trade_date", ["ts_code", "trade_date"]),
    "margin_detail":   ("ts_margin_detail", "trade_date", ["ts_code", "trade_date"]),
    "moneyflow_hsgt":  ("ts_moneyflow_hsgt", "trade_date", ["trade_date"]),
    "block_trade":     ("ts_block_trade", "trade_date", ["ts_code", "trade_date"]),
    "top_list":        ("ts_top_list", "trade_date", ["ts_code", "trade_date"]),
    "suspend_d":       ("ts_suspend_d", "trade_date", ["ts_code", "trade_date"]),
    "ggt_top10":       ("ts_ggt_top10", "trade_date", ["ts_code", "trade_date"]),
    "index_daily":     ("ts_index_daily", "trade_date", ["ts_code", "trade_date"]),
    # 事件
    "dividend":        ("ts_dividend", "end_date", ["ts_code", "end_date"]),
    "share_float":     ("ts_share_float", "float_date", ["ts_code", "float_date"]),
    "pledge_stat":     ("ts_pledge_stat", "end_date", ["ts_code", "end_date"]),
    "stk_holdernumber": ("ts_holder_number", "end_date", ["ts_code", "end_date"]),
    "top10_holders":   ("ts_top10_holders", "end_date", ["ts_code", "end_date"]),
    "repurchase":      ("ts_repurchase", "ann_date", ["ts_code", "ann_date"]),
    "new_share":       ("ts_new_share", "ann_date", ["ts_code", "ann_date"]),
    # 财务（用 _clean：最新公告的合并报表）
    "fina_indicator_clean": ("ts_fina_indicator", "end_date",
                             ["ts_code", "end_date"]),
    "income_clean":    ("ts_income", "end_date", ["ts_code", "end_date"]),
    "balancesheet_clean": ("ts_balance_sheet", "end_date", ["ts_code", "end_date"]),
    "cashflow_clean":  ("ts_cashflow", "end_date", ["ts_code", "end_date"]),
    "forecast":        ("ts_forecast", "ann_date", ["ts_code", "ann_date"]),
    "express":         ("ts_express", "ann_date", ["ts_code", "ann_date"]),
    "fina_mainbz":     ("ts_main_business", "end_date", ["ts_code", "end_date"]),
    # 指数
    "index_weight":    ("ts_index_weight", "trade_date",
                        ["index_code", "trade_date"]),
}

# ts_code → 本机代码
_SUF = {"SH": "sh", "SZ": "sz", "BJ": "bj"}


def ts_to_local(ts_code) -> str | None:
    """``600519.SH`` → ``sh600519``。格式不对返回 None。"""
    s = str(ts_code)
    if "." not in s:
        return None
    num, suf = s.split(".", 1)
    pre = _SUF.get(suf.upper())
    if pre is None or not num.isdigit():
        return None
    return pre + num


def normalize(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """统一类型：ts_code→ 本机格式、日期 → int32 YYYYMMDD。"""
    out = df.copy()
    if "ts_code" in out.columns:
        out["ts_code"] = out["ts_code"].map(ts_to_local)
        # 转换失败的（港股/指数/债券等）直接剔除，不静默留 NaN
        out = out[out["ts_code"].notna()]
    if date_col in out.columns:
        d = out[date_col]
        if d.dtype == object or str(d.dtype).startswith("datetime"):
            d = pd.to_datetime(d, format="mixed", errors="coerce")
            d = d.dt.strftime("%Y%m%d")
        out[date_col] = (pd.to_numeric(d, errors="coerce")
                         .dropna().astype("int64"))
        out = out[out[date_col].between(19900101, 20301231)]
    return out


def merge_one(con: sqlite3.Connection, src: str, dst: str,
              date_col: str, indexes: list[str], dry: bool) -> dict:
    """并入单张表。"""
    f = OUT / f"{src}.parquet"
    if not f.exists():
        return {"src": src, "dst": dst, "skipped": "文件不存在"}

    df = normalize(pd.read_parquet(f), date_col)
    n_raw = len(df)
    if n_raw == 0:
        return {"src": src, "dst": dst, "skipped": "清洗后 0 行"}

    # 库中已有行数（用于报告是新增还是覆盖）
    existed = 0
    if not dry:
        con.execute(f"DROP TABLE IF EXISTS {dst}")
    else:
        row = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (dst,)).fetchone()
        if row:
            existed = con.execute(f"SELECT COUNT(*) FROM {dst}").fetchone()[0]

    if not dry:
        # pandas → sqlite：列名里的特殊字符保持原样，SQLite 允许
        df.to_sql(dst, con, index=False, if_exists="replace", chunksize=50_000)
        for col in indexes:
            if col in df.columns:
                con.execute(
                    f"CREATE INDEX IF NOT EXISTS ix_{dst}_{col} "
                    f"ON {dst}({col})")
        con.commit()

    return {"src": src, "dst": dst, "rows": n_raw, "raw": n_raw, "existed": existed}


def main() -> int:
    ap = argparse.ArgumentParser(description="Tushare 补充表并入本机库")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="只并入某个源表")
    args = ap.parse_args()

    targets = TABLES if args.only is None else {
        k: v for k, v in TABLES.items() if k == args.only or v[0] == args.only}
    if not targets:
        print(f"✗ 未知表：{args.only}")
        print(f"  可用：{', '.join(TABLES)}")
        return 1

    print("=" * 84)
    print(f"Tushare 补充表并入本机库{'（预演）' if args.dry_run else ''}")
    print("=" * 84)
    print(f"数据库: {DB_PATH}")
    print(f"待并入{len(targets)} 张表\n")

    if not args.dry_run:
        BACKUP.mkdir(parents=True, exist_ok=True)
        bak = BACKUP / f"market_{time.strftime('%Y%m%d_%H%M%S')}.db"
        import shutil
        print(f"备份到: {bak.name}")
        shutil.copy2(DB_PATH, bak)
        con = sqlite3.connect(DB_PATH)
    else:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    results = []
    t0 = time.perf_counter()
    try:
        for src, (dst, dc, idx) in targets.items():
            r = merge_one(con, src, dst, dc, idx, args.dry_run)
            results.append(r)
            if r.get("skipped"):
                print(f"  ○ {src:<24} {r['skipped']}")
            else:
                print(f"  ✓ {src:<24} → {dst:<22} {r['rows']:>10,} 行"
                      f"  {time.perf_counter() - t0:>5.1f}s")
    finally:
        con.close()

    ok = [r for r in results if r.get("rows")]
    total = sum(r["rows"] for r in ok)
    print("\n" + "=" * 84)
    print(f"并入 {len(ok)} / {len(results)} 张表  共 {total:,} 行  "
          f"用时 {(time.perf_counter() - t0)/60:.1f} 分钟")
    if not args.dry_run and ok:
        print("\n下一步：")
        print("  1. uv run python research/scripts/verify_tushare_full.py  质量校验")
        print("  2. 因子研究阶段用 ts_ 前缀表与 bar_daily 联合取数")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
