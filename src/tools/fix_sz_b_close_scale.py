"""深市 B 股 bar 价格 ×10 归一：迁移交付件（UNITS U7 / U1 修正三问三模式）。

⚠️ **只读纪律**：对 market.db 一律 ``mode=ro``，本脚本绝不落库。
   交付物 = 迁移 SQL 文件（交数据线批次执行）+ 修正对照表 CSV + G4 守卫口径。

用法::

    python src/tools/fix_sz_b_close_scale.py audit   [--db PATH] [--out DIR]
    python src/tools/fix_sz_b_close_scale.py impact  [--db PATH] [--stk-limit PATH]
    python src/tools/fix_sz_b_close_scale.py sql     [--db PATH] [--out DIR]

U1「修正三问」对位（UNITS.md U7③ 指名走此三问）::

    ①改哪张表 / 导出物连动：bar_daily(open/high/low/close) 与 bar_weekly 同列
      ×10 归一；close_adj 对 B 股全历史 0 覆盖（实测 2,473~2,634 行/码全 NULL）
      **不入迁移面**；stock_info.price（通达信导出物）为真币价、不动。
    ②守卫对拍：G4 量纲恒等式 amount/(vol×close) 双口径（全市场 / 深 B）修复
      前后对照；阈值照抄 UNITS U7②（比值 >1.5 或 <0.67 即疑量纲污染）。
    ③历史产物影响面：封板判定（tradability.py 判据式）修复前后差异表 + 历史
      产出数字断点声明（修复落库前后 B 股价格类数字不可直接对比）。

幂等三道防线（测试 test_sz_b_close_scale.py 钉死）::

    PRECHECK → 污染谓词圈行 UPDATE（比值 ∈ [6,15] = 10× 净痕区间，复跑 0 命中）
             → 零成交/缺值行走原值三键逐条 UPDATE（code,date,close 原值匹配）
    POSTCHECK → 污染残留 = 0
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

_G4_HIGH = 1.5          # UNITS U7② 原文阈值
_G4_LOW = 0.67          # 同上
_CONTAM_LOW = 6.0       # 污染区间（10× 净痕居中 9.95；U5 的 0.099 净痕另诉）
_CONTAM_HIGH = 15.0
SZB_GLOBS = ("sz200*", "sz201*", "sz202*")   # 深 B 三段（与 config.is_b_share 同义）
SCB_SCALE = 10.0        # 迁移倍率（还原必败守护：改 1.0 测试立即转红）


def g4_ratio(amount, vol, close):
    """G4 量纲恒等式比值 r = amount/(vol×close)；任一缺/非正 → None（禁填空）。"""
    try:
        a, v, c = float(amount), float(vol), float(close)
    except (TypeError, ValueError):
        return None
    if not (a > 0 and v > 0 and c > 0):
        return None
    return a / (v * c)


def g4_verdict(ratio):
    """UNITS U7② 原文判据：比值 >1.5 或 <0.67 疑量纲污染；缺值=UNKNOWN 不猜。"""
    if ratio is None:
        return "unknown"
    if ratio > _G4_HIGH or ratio < _G4_LOW:
        return "suspect"
    return "ok"


def is_sz_b_share(code: str) -> bool:
    """深市 B 股（200/201/202 段，可带 sz 前缀）。

    与 ``factor_lab.config.is_b_share`` 同义；跨面一致性由
    tests/test_units_g4_identity.py 对拍钉死。
    """
    c = str(code).lower()
    num = c[2:] if len(c) == 8 and c[:2] in ("sh", "sz", "bj") else c
    return num.startswith(("200", "201", "202")) and (
        len(c) != 8 or c[:2] == "sz")


def is_sz_b_contaminated(code, amount, vol, close) -> bool:
    """污染谓词（= 幂等圈行条件）：深 B 且 G4 比值落在 10× 净痕区间。"""
    if not is_sz_b_share(code):
        return False
    r = g4_ratio(amount, vol, close)
    return r is not None and _CONTAM_LOW <= r <= _CONTAM_HIGH


def build_migration_sql(daily_pairs, weekly_pairs) -> str:
    """生成迁移 SQL 文本。daily_pairs/weekly_pairs = [(code, date, close 原值)]。

    零成交/缺值行无 G4 比值可判 → 逐条原值三键匹配（幂等：修过即不匹配）。
    """
    glob = " OR ".join(f"code GLOB '{g}'" for g in SZB_GLOBS)
    q = "amount / (vol * close) BETWEEN 6.0 AND 15.0"
    head = [
        "-- 深市 B 股 bar 价格 ×10 归一（UNITS U7）｜由 fix_sz_b_close_scale.py 生成",
        "-- 执行主体：数据线批次（本 SQL 未在本工具内执行过）；建议整包单事务执行",
        "BEGIN;",
        "-- PRECHECK 污染行数（预期 >0；首跑后复跑应为 0 → 幂等自证）",
        f"SELECT COUNT(*) FROM bar_daily WHERE ({glob}) AND close IS NOT NULL"
        f" AND vol > 0 AND amount > 0 AND {q};",
        "-- PRECHECK bar_weekly 同口径",
        f"SELECT COUNT(*) FROM bar_weekly WHERE ({glob}) AND close IS NOT NULL"
        f" AND vol > 0 AND amount > 0 AND {q};",
        "-- 主修复：污染谓词圈行（复跑 0 命中，天然幂等）",
        "UPDATE bar_daily SET open=open*10.0, high=high*10.0, low=low*10.0,"
        " close=close*10.0",
        f" WHERE ({glob}) AND close IS NOT NULL AND vol > 0 AND amount > 0 AND {q};",
        "UPDATE bar_weekly SET open=open*10.0, high=high*10.0, low=low*10.0,"
        " close=close*10.0",
        f" WHERE ({glob}) AND close IS NOT NULL AND vol > 0 AND amount > 0 AND {q};",
        "-- 零成交/缺值行：原值三键逐条（code,date,close 原值匹配 → 复跑 0 命中）",
    ]
    lines = list(head)
    for tab, pairs in (("bar_daily", daily_pairs), ("bar_weekly", weekly_pairs)):
        for code, date, close in pairs:
            lines.append(
                f"UPDATE {tab} SET open=open*10.0, high=high*10.0, low=low*10.0,"
                f" close=close*10.0 WHERE code='{code}' AND date={int(date)}"
                f" AND close={float(close)!r};")
    tail = [
        "-- POSTCHECK：污染残留（预期 0）与分组均价比值（预期 ≈1.00）",
        f"SELECT COUNT(*) FROM bar_daily WHERE ({glob}) AND close IS NOT NULL"
        f" AND vol > 0 AND amount > 0 AND {q};",
        f"SELECT code, AVG(amount/(vol*close)) FROM bar_daily WHERE ({glob})"
        " AND vol > 0 AND amount > 0 GROUP BY code LIMIT 8;",
        "COMMIT;",
    ]
    return "\n".join(lines + tail)


def _connect_ro(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def g4_summary(con) -> pd.DataFrame:
    """分组 G4 对照（全市场其余 / 基金债 / 沪 B / 深 B），单代表日 + 全历史深 B。"""
    rows = con.execute(
        "SELECT code, close, vol, amount FROM bar_daily WHERE vol>0 AND close>0"
        " AND amount>0").fetchall()
    df = pd.DataFrame(rows, columns=["code", "close", "vol", "amount"])
    df["r"] = df.amount / (df.vol * df.close)

    def grp(c: str) -> str:
        n = c[2:]
        if n.startswith(("200", "201", "202")) and c.startswith("sz"):
            return "深B"
        if c.startswith("sh900"):
            return "沪B"
        if n.startswith(("15", "16", "18")) or c.startswith(("sh5", "sh1", "sz1", "sh11", "sh12")):
            return "基金/债"
        return "A股等"

    df["g"] = df.code.map(grp)
    out = df.groupby("g")["r"].agg(["count", "median", "mean"]).round(4)
    return out.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description="深市 B 股价格 ×10 归一迁移交付件（只读）")
    ap.add_argument("mode", choices=["audit", "impact", "sql"])
    ap.add_argument("--db", default=None, help="market.db 路径（只读打开）")
    ap.add_argument("--stk-limit", default=None, help="stk_limit.parquet 路径（impact 用）")
    ap.add_argument("--out", default=None, help="SQL/对照表输出目录（默认 <db>/../runtime/fix_b14）")
    args = ap.parse_args()
    db = Path(args.db or _discover_db())
    out = Path(args.out) if args.out else db.parent.parent / "runtime" / "fix_b14"
    out.mkdir(parents=True, exist_ok=True)
    con = _connect_ro(db)
    glob = " OR ".join(f"code GLOB '{g}'" for g in SZB_GLOBS)

    if args.mode in ("audit", "sql"):
        s = g4_summary(con)
        print("== G4 双口径对照（全历史；UNITS U7② 阈值 0.67~1.5）==")
        print(s.to_string(index=False))
        z = con.execute(
            f"SELECT COUNT(*) FROM bar_daily WHERE ({glob}) AND close IS NOT NULL").fetchone()[0]
        bad = con.execute(
            f"SELECT COUNT(*) FROM bar_daily WHERE ({glob}) AND close IS NOT NULL"
            " AND vol > 0 AND amount > 0"
            " AND amount/(vol*close) BETWEEN 6.0 AND 15.0").fetchone()[0]
        loose = con.execute(
            f"SELECT COUNT(*) FROM bar_daily WHERE ({glob}) AND close IS NOT NULL"
            " AND (vol <= 0 OR amount <= 0 OR amount/(vol*close) NOT BETWEEN 6.0"
            " AND 15.0)").fetchone()[0]
        w = con.execute(
            f"SELECT COUNT(*) FROM bar_weekly WHERE ({glob}) AND close IS NOT NULL"
            " AND vol > 0 AND amount > 0"
            " AND amount/(vol*close) BETWEEN 6.0 AND 15.0").fetchone()[0]
        nclose = con.execute(
            f"SELECT COUNT(*) FROM bar_daily WHERE ({glob}) AND close IS NULL").fetchone()[0]
        print(f"\n深 B 污染行（谓词命中）: bar_daily {bad:,} / bar_weekly {w:,}")
        print(f"零成交/缺值行走三键路径: {loose:,}（占深 B 行 {z:,}）"
              f"；close IS NULL 空转行 {nclose:,}（不迁移，open/high/low 同行为空转）")
        # ⚠️ 三值逻辑：vol=0 时 amount/(vol*close) 为 NULL，
        #    `NOT (NULL BETWEEN …)` 不命中 → 必须用 OR 平铺 + 前项 TRUE 短路。
        loose_where = (" AND close IS NOT NULL AND (vol <= 0 OR amount <= 0"
                       " OR amount/(vol*close) NOT BETWEEN 6.0 AND 15.0)")
        pairs_d = con.execute(
            f"SELECT code, date, close FROM bar_daily WHERE ({glob}){loose_where}").fetchall()
        pairs_w = con.execute(
            f"SELECT code, date, close FROM bar_weekly WHERE ({glob}){loose_where}").fetchall()
        sql = build_migration_sql(pairs_d, pairs_w)
        sqlp = out / "fix_sz_b_close_scale.sql"
        sqlp.write_text(sql, encoding="utf-8")
        print(f"迁移 SQL → {sqlp}（{len(pairs_d)+len(pairs_w)} 条三键行 + 谓词圈行）")
        samp = con.execute(
            f"SELECT code, date, close, vol, amount FROM bar_daily WHERE ({glob})"
            " AND vol > 0 AND amount > 0 ORDER BY code, date LIMIT 20").fetchall()
        with (out / "correction_map.csv").open("w", encoding="utf-8", newline="") as f:
            f.write("code,date,close_old,close_new,g4_before,g4_after,verdict_before,"
                    "verdict_after\n")
            for code, date, cl, v, a in samp:
                rb = g4_ratio(a, v, cl)
                ra = g4_ratio(a, v, cl * SCB_SCALE)
                f.write(f"{code},{date},{cl},{cl * SCB_SCALE},{rb:.4f},{ra:.4f},"
                        f"{g4_verdict(rb)},{g4_verdict(ra)}\n")
        print(f"修正对照表（抽样 20 行）→ {out / 'correction_map.csv'}")

    if args.mode == "impact":
        _impact(con, Path(args.stk_limit) if args.stk_limit
                else db.parent.parent / "runtime" / "tushare" / "stk_limit.parquet")
    con.close()
    return 0


def _impact(con, stk_limit: Path) -> None:
    """封板影响面：sz200×3 + sh600×1，2023 年跌停判定修复前后差异（UNITS U7③）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from factor_lab.analysis.tradability import tushare_to_local_code

    sl = pd.read_parquet(stk_limit, columns=["trade_date", "ts_code", "down_limit"])
    sl = sl[(sl.trade_date >= 20230101) & (sl.trade_date <= 20231231)]
    keys = zip(tushare_to_local_code(sl["ts_code"]), sl["trade_date"], strict=True)
    dn = dict(zip(keys, sl["down_limit"], strict=True))
    print("\n== 封板影响面（2023 跌停判定修复前后；判据=tradability.py:191/210 原文式）==")
    print(f"  {'code':<10}{'格数':>5}{'修复前真封':>9}{'修复后真封':>9}{'翻转':>6}   样例(date/close前/close后/dn/前→后)")
    for code in ("sz200011", "sz200012", "sz200017", "sh600519"):
        rows = con.execute(
            "SELECT date, close FROM bar_daily WHERE code=? AND date BETWEEN"
            " 20230101 AND 20231231 ORDER BY date", (code,)).fetchall()
        n = before = after = 0
        flip = []
        scale = SCB_SCALE if is_sz_b_share(code) else 1.0
        for date, cl in rows:
            d = dn.get((code, date))
            if d is None or cl is None:
                continue
            n += 1
            b = cl <= d + 1e-6
            a = cl * scale <= d + 1e-6
            before += b
            after += a
            if b != a:
                flip.append((date, cl, cl * scale, d, b, a))
        ex = "; ".join(f"{d}/{c:.3f}/{c*scale:.2f}/{dl:.2f}/{'封' if b else '未'}→"
                       f"{'封' if a else '未'}" for d, c, c2, dl, b, a in
                       [(f[0], f[1], f[2], f[3], f[4], f[5]) for f in flip[:3]])
        print(f"  {code:<10}{n:>5}{before:>9}{after:>9}{len(flip):>6}   {ex}")
    print("  （sh600519 为对照组：深 B 修复不应波及 → 翻转应为 0）")


def _discover_db() -> str:
    try:
        from factor_lab.config import DB_PATH

        return str(DB_PATH)
    except Exception as e:                     # 配置发现失败必须显式报错（禁静默兜底）
        raise SystemExit(
            f"无法发现 market.db（{e}）——请显式传 --db 路径；本工具对库只读") from e


if __name__ == "__main__":
    raise SystemExit(main())
