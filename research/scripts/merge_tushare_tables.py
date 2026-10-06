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
和「前复权价（算收益）」，两张表并列才不会互相污染。

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
from factor_lab.config import DB_PATH, YUAN_TO_YI  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "runtime" / "tushare"
BACKUP = Path(__file__).resolve().parents[2] / "runtime" / "backup"

# ── 金额单位换算（P19 同类问题的第二个实例）────────────────────
# ⚠️ 本机 stock_info 用【亿元】，Tushare 财务表用【元】，差 1e8 倍。
#    实测硬验证（茅台 2026-06-30）：
#      stock_info.net_assets 2,512.536亿 ÷ shares 12.500815亿股
#        = 200.98978 元/股 = ts_fina_indicator.bps 200.9898✓
#      直接相除：9.23e10 / 907.03 = 101,736,220（荒谬但不报错）
#
#    规矩：**换算只在数据入口做一次**（本函数），禁止下游各自换算。
#    入库后所有 ts_ 财务表的金额列统一为【亿元】。

# ⚠️⚠️ **按表配置，绝不用列名关键词猜测** ——
#   初版用「含revenue/income/assets 等关键词」判定，实测误判率极高：
#     assets_turn（资产周转率）    含 assets → 被误换算
#     assets_yoy（资产同比增速）  含 assets → 被误换算
#     debt_to_assets（资产负债率）含 assets → 被误换算
#     bps（每股净资产，元/股）  必须 /1e8 → 实际没换
#   fina_indicator 一张表就误判 5 列。**关键词法在金融字段上不可用。**
# 不换算的表（键为 TABLES 里的源文件名）
NO_MONEY_CONVERT = {"fina_indicator_clean", "top10_holders",
                    "top10_floatholders", "stk_managers", "index_weight"}

# 白名单由 build_money_whitelist.py 从**真实数据**推导并存为 JSON。
# ⚠️ 绝不手写：初版手写时有 30 个列名是凭记忆写的、实际不存在
#   （accounts_payable 实际叫 acct_payable…），
#   而真实存在却漏写的金额列会**静默漏换算**。
WHITELIST_FILE = OUT / "_money_whitelist.json"


def _load_whitelist() -> dict[str, list[str]]:
    """读白名单。文件缺失时 raise —— 禁止静默用空列表导致「以为换算过了」。"""
    import json
    if not WHITELIST_FILE.exists():
        raise FileNotFoundError(
            f"金额列白名单不存在：{WHITELIST_FILE}\n"
            f"  解决：uv run python research/scripts/build_money_whitelist.py"
        )
    return json.loads(WHITELIST_FILE.read_text(encoding="utf-8"))


def _money_cols_for(src: str, df: pd.DataFrame) -> list[str]:
    """返回该表需要换算的列名列表。

    ⚠️ **必须显式配置**，不在白名单里的列一律不换算 ——
    金融字段的命名无法靠关键词可靠推断（见MONEY_COLS 上方注释）。
    """
    if src in NO_MONEY_CONVERT:
        return []
    allow = set(_load_whitelist().get(src, []))
    return [c for c in df.columns if c in allow]

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


def normalize(df: pd.DataFrame, date_col: str, src: str) -> tuple[pd.DataFrame,
                                                                   list[str]]:
    """统一三件事：ts_code → 本机格式、日期 → int32、**金额 → 亿元**。

    ⚠️ 金额换算是本函数存在的核心原因之一（P19 同类问题）。
    没有它，因子研究时``ts_income.total_revenue``（元）
    与 ``stock_info.revenue``（亿元）直接相除会得到 1e8 量级的荒谬值，
    而且**不报错**。
    """
    out = df.copy()
    n_in = len(out)
    if "ts_code" in out.columns:
        out["ts_code"] = out["ts_code"].map(ts_to_local)
        # 转换失败的（港股/指数/债券等）直接剔除
        out = out[out["ts_code"].notna()]
        n_dropped = n_in - len(out)
        if n_dropped:
            # 禁止静默：明确报告剔除了多少
            print(f"    {src}: 剔除 {n_dropped:,} 行非 A 股代码"
                  f"（港股/指数/债券等）")
    if date_col in out.columns:
        d = out[date_col]
        if d.dtype == object or str(d.dtype).startswith("datetime"):
            d = pd.to_datetime(d, format="mixed", errors="coerce")
            d = d.dt.strftime("%Y%m%d")
        # ⚠️ 用 Int64 而非 int64：astype("int64") 遇到 NaN 会静默变 float64，
        #    导致 JOIN bar_daily（date 是 integer）时类型不匹配、索引失效。
        out[date_col] = pd.to_numeric(d, errors="coerce").astype("Int64")
        out = out[out[date_col].between(19900101, 20301231)]
    # 金额：元 → 亿元（显式白名单，不猜）
    money_cols = _money_cols_for(src, out)
    if money_cols:
        out[money_cols] = out[money_cols] * YUAN_TO_YI
    return out, money_cols


def merge_one(con: sqlite3.Connection, src: str, dst: str,
              date_col: str, indexes: list[str], dry: bool) -> dict:
    """并入单张表。金额列在入库前统一换算成【亿元】。"""
    f = OUT / f"{src}.parquet"
    if not f.exists():
        return {"src": src, "dst": dst, "skipped": "文件不存在"}

    df, money_cols = normalize(pd.read_parquet(f), date_col, src)
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
        # 记录单位口径 —— 防止下游忘记「已经是亿元了」
        con.execute(f"CREATE TABLE IF NOT EXISTS {dst}_meta "
                    f"(k TEXT PRIMARY KEY, v TEXT)")
        con.execute(f"INSERT OR REPLACE INTO {dst}_meta VALUES (?,?)",
                    ("金额单位", "亿元（Tushare 原始单位为元，入口已/1e8）"))
        con.execute(f"INSERT OR REPLACE INTO {dst}_meta VALUES (?,?)",
                    ("每股指标单位", "元/股（未换算）"))
        con.execute(f"INSERT OR REPLACE INTO {dst}_meta VALUES (?,?)",
                    ("换算列数", str(len(money_cols))))
        con.commit()

    return {"src": src, "dst": dst, "rows": n_raw, "existed": existed,
            "money_cols": len(money_cols)}


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
                      f"  金额列 {r.get('money_cols', 0)} 个已转亿元"
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
