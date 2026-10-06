"""补齐退市股票的历史行情（消除幸存者偏差）。

为什么需要这个脚本
------------------
`merge_tushare_into_db.py` 只能**标记**退市股，无法补它们的历史行情 ——
通达信本地库（vipdoc）里没有这些票的任何数据。
实测（2026-10-06）：`stock_info` 318 只带 "(退)" 标记的股票中，
**有行情的是 0 只**。

但「本机完全没有退市股」是错的，真实情况更微妙：
  · 2026-02-12 集体退市的那批（退市熊猫/立方退/退市太和…）**有完整历史行情**，
    末行正好是 20260212 —— 通达信保留了已退市股票的存量数据
  · 真正缺的是**更早退市**的那批：sh603157 退拉夏、sh603133 退碳元…
    实测这 15 只在 2016 年后上市、如今已退市，本机一条行情都没有

这 15 只是**真实的幸存者偏差敞口**：它们在 2016~2022 期间是正常可交易的股票，
却因为最终退市而从任何历史横截面里消失。因子研究若只用存续股，
等于系统性地只看「活下来的赢家」。

已实测 Tushare `daily` 对退市股**有数据**，且末行正好是退市日：
    sh603157 退拉夏  1,109 行  20170925 ~ 20220517
    sh603133 退碳元  1,736 行  20170320 ~ 20240626
    sz300526 中潜退  1,505 行  20160802 ~ 20230711
    sh688086 退紫晶    781 行  20200226 ~ 20230630
对照组 sz000003 PT金田A（2002 年退市）返回 0 行 —— 说明接口只保留较近的退市股，
这也解释了为什么更早的退市股补不齐：它们**不在 Tushare 的保留期内**。

因此本脚本的定位是**尽力补齐 + 明确标注补不到的部分**，
不是声称消除了幸存者偏差。剩余敞口由 audit 脚本量化并写入报告。

用法
----
  uv run python research/scripts/backfill_delisted.py --dry-run   # 只探测不落盘
  uv run python research/scripts/backfill_delisted.py             # 拉取并落盘 parquet
  uv run python research/scripts/backfill_delisted.py --merge     # 顺带并入 bar_daily
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fetch_tushare import call, get_token, to_ts  # noqa: E402
from factor_lab.config import DB_PATH, is_a_share  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runtime" / "tushare"
PARQ = OUT / "delisted_daily.parquet"
PARQ_ADJ = OUT / "delisted_adj_factor.parquet"

START, END = "20150101", "20260930"
RATE_LIMIT_S = 0.34          # 180 次/分钟 ≈ 0.33 s/次，留余量


def to_code(symbol: str) -> str | None:
    """6 位代码 → 本机代码（交易所由代码段唯一决定）。

    ⚠️不要用 ``ts_code.str.slice(0,2)`` 取前缀 —— Tushare 的 ts_code
    形如 ``000003.S``，交易所写在**后缀**，slice(0,2) 得到 "00"，
    全表对不上，「重合 0 只」的结论完全是假的（2026-10-06 实测踩过）。
    """
    s = str(symbol).zfill(6)
    if s[0] in "0366":
        return f"sh{s}"
    if s[0] in "348":
        return f"bj{s}"
    return f"sz{s}"


def find_missing(start_after: str = "20160101") -> pd.DataFrame:
    """找出「研究窗口内上市、已退市、本机无任何行情」的股票。

    这才是**真实偏差敞口**。上市日早于窗口的退市股（2016 年前退市）
    补不补都不影响结论 —— 它们在窗口起点就已经不在市场上了。
    """
    con = sqlite3.connect(DB_PATH)
    have = {r[0] for r in con.execute("SELECT DISTINCT code FROM bar_daily")}
    info = {r[0]: (r[1], r[2]) for r in
            con.execute("SELECT code, name, list_date FROM stock_info")}
    con.close()

    rows = []
    # 来源 1：Tushare 退市名单
    d = pd.read_parquet(OUT / "stock_basic_D.parquet")
    for _, r in d.iterrows():
        c = to_code(r["symbol"])
        if c and is_a_share(c):
            rows.append({"code": c, "name": r["name"], "list_date": r["list_date"]})
    # 来源 2：stock_info 名字带 "(退)"
    for c, (n, ld) in info.items():
        if is_a_share(c) and "(退)" in (n or ""):
            rows.append({"code": c, "name": n, "list_date": ld})

    df = pd.DataFrame(rows).drop_duplicates(subset=["code"])
    df = df[df["code"].isin(have) == False]          # noqa: E712
    df["ld"] = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
    miss = df[df["ld"] >= pd.Timestamp(start_after)].copy()
    return miss.sort_values("list_date").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="补齐退市股历史行情")
    ap.add_argument("--dry-run", action="store_true", help="只探测，不落盘")
    ap.add_argument("--merge", action="store_true", help="并入 bar_daily")
    ap.add_argument("--merge-only", action="store_true",
                    help="跳过下载，直接用已落盘的 parquet 重跑并库")
    args = ap.parse_args()

    miss = find_missing()
    print("=" * 70)
    print("退市股行情补齐")
    print("=" * 70)

    # --merge-only 必须**先于**「无需补齐」的早退判断：
    # 首次并库可能只补了 close 没补上 close_adj（实测踩过），
    # 此时 find_missing() 已返回 0，但复权列仍需重跑。
    if args.merge_only:
        if not PARQ.exists() or not PARQ_ADJ.exists():
            raise SystemExit(f"✗ 找不到 {PARQ.name} / {PARQ_ADJ.name}，"
                             "请先跑一次不带 --merge-only 的下载")
        print("→ --merge-only：复用已下载的 parquet，不重新请求 Tushare")
        return merge_into_db(pd.read_parquet(PARQ), pd.read_parquet(PARQ_ADJ))

    print(f"研究窗口内上市、已退市、本机无行情: {len(miss)} 只")
    for _, r in miss.iterrows():
        print(f"  {r['code']} {r['name']:12s} 上市 {r['list_date']}")
    if miss.empty:
        print("\n✓ 无需补齐")
        return 0

    if args.dry_run:
        token = get_token()
        if not token:
            raise SystemExit("✗ 取不到 Tushare token")
        print("\n=== 探测 Tushare daily 可得性 ===")
        got = miss[miss["code"].isin(
            [c for c in miss["code"] if len(call(
                token, "daily",
                {"ts_code": to_ts(c), "start_date": START, "end_date": END})) > 0])]
        print(f"有数据 {len(got)} / {len(miss)}")
        print(f"无数据 {len(miss) - len(got)} 只 → 这些将**保留为已知偏差敞口**")
        return 0

    # ── 正式拉取 ────────────────────────────────────────────────
    if args.merge_only:
        if not PARQ.exists() or not PARQ_ADJ.exists():
            raise SystemExit(f"✗ 找不到 {PARQ.name} / {PARQ_ADJ.name}，"
                             "请先跑一次不带 --merge-only 的下载")
        print("→ --merge-only：复用已下载的 parquet，不重新请求 Tushare")
        return merge_into_db(pd.read_parquet(PARQ), pd.read_parquet(PARQ_ADJ))

    token = get_token()
    if not token:
        raise SystemExit("✗ 取不到 Tushare token")
    frames, adj_frames, empty = [], [], []
    for i, code in enumerate(miss["code"], 1):
        d = call(token, "daily",
                 {"ts_code": to_ts(code), "start_date": START, "end_date": END})
        if len(d) == 0:
            empty.append(code)
            print(f"  [{i}/{len(miss)}] {code} 无数据（Tushare 未保留该股）")
        else:
            d = d.copy()
            d["code"] = code
            frames.append(d)
            # 🔴 必须同时取 adj_factor：只补 close 不补 close_adj，
            # 这 15 只在因子研究里会因复权列全NULL 被整段跳过 ——
            # 等于白补，还制造「已消除偏差」的错觉。
            a = call(token, "adj_factor",
                     {"ts_code": to_ts(code), "start_date": START, "end_date": END})
            if len(a) == 0:
                raise SystemExit(
                    f"✗ {code} 有日线但无复权因子，拒绝写入。\n"
                    "  写入未复权价会污染 bar_daily，且 close_adj 为空导致\n"
                    "  因子研究跳过该股 —— 看似补齐、实则制造偏差。")
            a = a.copy()
            a["code"] = code
            adj_frames.append(a)
            print(f"  [{i}/{len(miss)}] {code} 日线 {len(d):5d} 行 "
                  f"{d['trade_date'].min()} ~ {d['trade_date'].max()}"
                  f" / 复权 {len(a):5d} 行")
        time.sleep(RATE_LIMIT_S)

    if not frames:
        raise SystemExit("✗ 一只都没取到，拒绝写出空文件")
    allq = pd.concat(frames, ignore_index=True)
    alladj = pd.concat(adj_frames, ignore_index=True)
    if PARQ.exists():
        allq = pd.concat([pd.read_parquet(PARQ), allq], ignore_index=True)
    if PARQ_ADJ.exists():
        alladj = pd.concat([pd.read_parquet(PARQ_ADJ), alladj],
                           ignore_index=True)
    allq = allq.drop_duplicates(subset=["code", "trade_date"])
    alladj = alladj.drop_duplicates(subset=["code", "trade_date"])
    PARQ.parent.mkdir(parents=True, exist_ok=True)
    allq.to_parquet(PARQ, index=False)
    alladj.to_parquet(PARQ_ADJ, index=False)
    print(f"\n已写出 {PARQ.name}（{len(allq):,} 行，{allq['code'].nunique()} 只）")
    print(f"已写出 {PARQ_ADJ.name}（{len(alladj):,} 行）")
    if empty:
        print(f"⚠️ {len(empty)} 只 Tushare 无数据，偏差敞口无法归零：{empty}")

    if args.merge:
        return merge_into_db(allq, alladj)
    return 0


def merge_into_db(df: pd.DataFrame, adj: pd.DataFrame) -> int:
    """把退市股日线 + 后复权收盘价并入 bar_daily。

    复权算法与 ``merge_tushare_into_db.py`` 完全一致（同一口径，不能两套）：
        close_adj = close × adj_factor(t) / adj_factor(最新交易日)
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    have = {r[0] for r in cur.execute("SELECT DISTINCT code FROM bar_daily")}
    new = sorted(set(df["code"]) - have)
    if new:
        cols = ["code", "date", "open", "high", "low", "close", "amount", "vol"]
        src = df[df["code"].isin(new)].copy()
        # 🔴 bar_daily.date 存的是 **INTEGER**（实测 typeof(date)='integer'），
        #    传字符串会静默匹配 0 行。入库前显式转 int 并断言。
        src["date"] = src["trade_date"].astype(int)
        cur.execute("SELECT typeof(date) FROM bar_daily LIMIT 1")
        db_dt = cur.fetchone()[0]
        if db_dt != "integer":
            con.close()
            raise SystemExit(f"✗ bar_daily.date 类型是 {db_dt}，与预期 integer 不符，"
                             "拒绝写入（口径不明时不能猜）")
        rows = src[cols].itertuples(index=False, name=None)
        cur.executemany(
            f"INSERT OR IGNORE INTO bar_daily ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})", rows)
        con.commit()
        n = cur.execute("SELECT COUNT(*) FROM bar_daily WHERE code IN "
                        f"({','.join('?' * len(new))})", new).fetchone()[0]
        print(f"已并入 {len(new)} 只退市股，共 {n:,} 行")
    else:
        print("✓ 日线已在库，仅重跑复权回填")
        src = df.copy()
        src["date"] = src["trade_date"].astype(int)

    # ── close_adj：按并库脚本同口径回填 ──────────────────────────
    # ⚠️ 过滤基准是「parquet 里的全部代码」而不是 `new`：
    #    --merge-only 场景下 new 为空（日线已在库），用 new 过滤会得到
    #    空 adj，命中率算成 NaN，然后报「19,623 行匹配不到」——
    #    错误信息指向了错的行，排查方向被带偏（实测踩过）。
    a = adj[adj["code"].isin(set(df["code"]))].copy()
    a["dnum"] = a["trade_date"].astype(int)
    # ⚠️ Tushare adj_factor 是**日期倒序**，必须先排序再取 last()。
    #    早先并库脚本取 iloc[-1] 拿到最早日期，归一化系数错 1.26 倍。
    latest = a.sort_values(["code", "dnum"]).groupby("code")["adj_factor"].last()
    a["norm"] = a["code"].map(latest)
    hit = a["adj_factor"] / a["norm"]
    have_dates = set(zip(src["code"], src["date"].astype(int)))
    rate = float((a["dnum"].isin({d for _, d in have_dates})).mean())
    print(f"复权因子命中率 {rate:.1%}（低于 90% 说明键错配，应报错）")
    if not rate >= 0.90:
        con.close()
        raise SystemExit(f"✗ 复权因子命中率仅 {rate:.1%}，键错配，已中止")

    # daily 与 adj_factor 行数可能不同（停牌日 adj_factor 有、daily 无），
    # 用 merge 对齐，绝不按行号positional 拼接（P3）。
    d2 = src[["code", "date", "close"]].copy()
    d2["dnum"] = d2["date"].astype(int)
    m = d2.merge(a[["code", "dnum", "adj_factor", "norm"]],
                 on=["code", "dnum"], how="left")
    if m["adj_factor"].isna().any():
        con.close()
        raise SystemExit(
            f"✗ {int(m['adj_factor'].isna().sum())} 行匹配不到复权因子，已中止")
    m["close_adj"] = m["close"].astype(float) * m["adj_factor"] / m["norm"]
    # 🔴 **逐条 execute，不用 executemany**（实测 2026-10-06，SQLite 3.53.1）：
    #    同样的 19,623 条参数、同一张表、同一个连接：
    #      · executemany → 写入 0 行，**不报错**，同连接内也查不到
    #      · 逐条 execute → 命中 19,623/19,623
    #    逐条慢（十几秒）但结果可验证。对19k 行的回填完全可接受，
    #    拿「快」换「静默写不进去」不值。
    #    另：参数用普通 tuple 并显式转 int/float，不接受 numpy 标量。
    upd = [(str(r.code), int(r.dnum), float(r.close_adj))
           for r in m.itertuples(index=False)]
    miss = 0
    for c, dt, ca in upd:
        if cur.execute(
            "UPDATE bar_daily SET close_adj=? WHERE code=? AND date=?",
            (ca, c, dt)).rowcount == 0:
            miss += 1
    con.commit()
    if miss:
        con.close()
        raise SystemExit(
            f"✗ {miss:,}/{len(upd):,} 行 UPDATE 未命中，已中止。\n"
            "  日线与复权因子的键对不上，不能带着全 NULL 的复权列继续。")
    # 🔴 写后**独立验证**命中数：UPDATE 不报错 ≠ 写进去了
    codes = sorted(set(df["code"]))
    got = cur.execute("SELECT COUNT(*) FROM bar_daily WHERE code IN "
                      f"({','.join('?' * len(codes))}) AND close_adj IS NOT NULL",
                      codes).fetchone()[0]
    print(f"已回填 close_adj：{got:,} 行")
    total = cur.execute("SELECT COUNT(*) FROM bar_daily WHERE code IN "
                        f"({','.join('?' * len(codes))})", codes).fetchone()[0]
    if got < 0.90 * total:
        con.close()
        raise SystemExit(
            f"✗ close_adj 只命中 {got}/{total} 行（<90%），已中止。\n"
            "  多半是 date 类型或键格式不匹配 —— 不允许带着全 NULL 的复权列继续。")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())