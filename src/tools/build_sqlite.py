"""构建 SQLite 行情库
设计要点：
  1. 用 PRIMARY KEY(code,date) 代替独立索引 —— 更紧凑，查询同样走 B-tree
  2. 日线全量 + 周线/月线聚合层（聚合层体积小，查询快）
  3. 附带股票信息、板块定义、成分股、财务快照
"""
import os
import sys
import time
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# 允许直接运行本文件：把 src/ 和 src/tools/ 都挂上
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent / "src"))
sys.path.insert(0, str(_HERE.parent))

from factor_lab.config import DB_PATH, MARKET_CAP_UNIT, to_wan  # noqa: E402
from tdx import read_day, all_codes, market_of  # noqa: E402

# ⚠️ 踩坑记录（2026-10-05）：原代码硬编码
#    DB = r"C:\Documentation\Wealth_Management\data\market.db"
#    → 换机器/换路径即崩。改为复用 factor_lab.config 的路径自动发现。
DB = str(DB_PATH)

SCHEMA = """
PRAGMA journal_mode = WAL;      -- 写入更快
PRAGMA synchronous = NORMAL;   -- 断电保护略降，换 3~5 倍写入速度
PRAGMA temp_store = MEMORY;
PRAGMA cache_size = -400000;   -- 400 MB 页缓存

CREATE TABLE IF NOT EXISTS bar_daily (
    code    TEXT NOT NULL,
    date    INTEGER NOT NULL,     -- YYYYMMDD
    open    REAL, high REAL, low REAL, close REAL,
    amount  REAL,                 -- 成交额(元)
    vol     REAL,                 -- 成交量(股)
    PRIMARY KEY (code, date)
);

CREATE TABLE IF NOT EXISTS bar_weekly (
    code    TEXT NOT NULL,
    date    INTEGER NOT NULL,     -- 标签=周内最后交易日，与通达信周线对齐
    open    REAL, high REAL, low REAL, close REAL,
    amount  REAL, vol REAL,
    PRIMARY KEY (code, date)
);

CREATE TABLE IF NOT EXISTS stock_info (
    code TEXT PRIMARY KEY, name TEXT, price REAL, chg REAL, pe REAL, pb REAL,
    mktcap REAL, float_mktcap REAL, turnover REAL, vol_ratio REAL,
    revenue REAL, net_profit REAL, net_assets REAL, roe REAL, bps REAL,
    net_assets_ord REAL,        -- 归属母公司普通股股东权益（亿元，U6 扣其他权益工具口径派生列）
    shares REAL, report_date TEXT, list_date TEXT, industry TEXT, region TEXT,
    boards TEXT, n_boards INT
);

CREATE TABLE IF NOT EXISTS sector (
    code TEXT PRIMARY KEY, name TEXT, n_members INT, updated TEXT
);

CREATE TABLE IF NOT EXISTS sector_member (
    sector_code TEXT, code TEXT,
    PRIMARY KEY (sector_code, code)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT
);
"""

# ── stock_info 权益口径记录（U6 规则③：不带口径描述的 net_assets/bps 不入合成面）──
# 后缀约定：_ord = 归属母公司普通股股东权益（已扣其他权益工具）。
STOCK_INFO_CALIBER_META = (
    ("stock_info.net_assets",
     "归属母公司股东权益合计_含其他权益工具_亿元（通达信 JZC 实测口径；U6："
     "= ts_balance_sheet.total_hldr_eqy_exc_min_int，未扣 oth_eqt_tools）"),
    ("stock_info.net_assets_ord",
     "归属母公司普通股股东权益_亿元（U6 派生列，_ord=普通股权益；= bps × shares "
     "≈ 归母 − COALESCE(oth_eqt_tools,0)，含 bps 三位小数舍入误差界 ~1e-4 相对；"
     "精确扣减用 UNITS U6 ① 的 C 式）"),
    ("stock_info.bps",
     "每股普通股净资产_元（通达信 TZMGJZ 实测口径；U6："
     "= (归母 − COALESCE(oth_eqt_tools,0)) × 1e8 / total_share）"),
    ("stock_info.shares",
     "总股本_亿股（U3 同名不同纲：ts_balance_sheet.total_share 为【股】）"),
)


def derive_equity_caliber(u2: pd.DataFrame) -> pd.DataFrame:
    """派生 U6 普通股权益口径列 net_assets_ord = bps × shares（亿元）。

    stock_info 原生两列的权益切分口径不同（A18 判决实证）：net_assets（通达信
    JZC）是【含其他权益工具】口径、bps（TZMGJZ）是【普通股】口径，
    net_assets/shares 当 bps 反推会系统性偏差 = 其他权益工具占归母比
    （sz000001 实测 17.09%）。派生列用已扣口径的 bps×shares 给普通股权益，
    原生列不覆写、口径记入 meta（STOCK_INFO_CALIBER_META，U6 规则③）。
    NULL 传播不冒算：任一输入为空 → 派生值为空。
    """
    missing = [c for c in ("bps", "shares") if c not in u2.columns]
    if missing:
        raise ValueError(
            f"net_assets_ord 派生缺列（{missing}）—— universe.csv 无每股净资产/"
            f"总股本列（base_dbf.csv 缺失时如此），补财务数据后重跑，不静默置 NULL"
        )
    out = u2.copy()
    bps = pd.to_numeric(out["bps"], errors="raise")
    shares = pd.to_numeric(out["shares"], errors="raise")
    out["net_assets_ord"] = bps * shares
    return out


def build():
    if os.path.exists(DB):
        os.remove(DB)
    for ext in ("-wal", "-shm"):
        if os.path.exists(DB + ext):
            os.remove(DB + ext)
    os.makedirs(os.path.dirname(DB), exist_ok=True)

    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)

    # ⚠️ 必须用 with_prefix=True：sh000001(上证指数) 与 sz000001(平安银行) 代码相同，
    #    去掉前缀会导致 25 个重名代码混淆、3124 个标的（9xx/1xx/2xx/5xx）被漏掉
    codes = all_codes(with_prefix=True)
    print(f"待导入标的: {len(codes)}")

    d_rows, w_rows = [], []
    t0 = time.perf_counter()
    for i, sym in enumerate(codes):
        mkt, c = sym[:2], sym[2:]
        d = read_day(c, mkt=mkt)
        ymd = d["date"].dt.strftime("%Y%m%d").astype(int).tolist()
        o = d["open"].tolist(); h = d["high"].tolist()
        lo = d["low"].tolist(); cl = d["close"].tolist()
        am = d["amount"].tolist(); vo = d["vol"].tolist()
        d_rows.extend((sym, ymd[j], o[j], h[j], lo[j], cl[j], am[j], vo[j])
                      for j in range(len(ymd)))

        g = d.set_index("date")
        rule = g.resample("W-FRI")
        last = rule["close"].apply(lambda s: s.index[-1] if len(s) else pd.NaT)
        w = rule.agg({"open": "first", "high": "max", "low": "min",
                      "close": "last", "amount": "sum", "vol": "sum"}).dropna()
        wl = last[w.index]
        wd = wl.dt.strftime("%Y%m%d").astype(int).tolist()
        w_rows.extend((sym, wd[j], w["open"].iloc[j], w["high"].iloc[j],
                       w["low"].iloc[j], w["close"].iloc[j],
                       w["amount"].iloc[j], w["vol"].iloc[j])
                      for j in range(len(wd)))
        if (i + 1) % 1500 == 0:
            print(f"  {i+1}/{len(codes)}  日线 {len(d_rows):,} 条  "
                  f"周线 {len(w_rows):,} 条  {time.perf_counter()-t0:.0f}s")

    print(f"解析完成: 日线 {len(d_rows):,} / 周线 {len(w_rows):,} 条 "
          f"({time.perf_counter()-t0:.0f}s)，开始写入...")
    t1 = time.perf_counter()
    con.executemany("INSERT OR REPLACE INTO bar_daily VALUES(?,?,?,?,?,?,?,?)", d_rows)
    con.commit()
    print(f"  日线写入 {time.perf_counter()-t1:.0f}s")
    t1 = time.perf_counter()
    con.executemany("INSERT OR REPLACE INTO bar_weekly VALUES(?,?,?,?,?,?,?,?)", w_rows)
    con.commit()
    print(f"  周线写入 {time.perf_counter()-t1:.0f}s")

    # 股票信息
    caliber_meta = list(STOCK_INFO_CALIBER_META)
    if os.path.exists("universe.csv"):
        u = pd.read_csv("universe.csv", dtype={"代码": str})
        cols = {
            "代码": "code", "名称": "name", "现价": "price", "涨跌幅%": "chg",
            "PE_TTM": "pe", "PB": "pb", "总市值亿": "mktcap", "流通市值亿": "float_mktcap",
            "换手率%": "turnover", "量比": "vol_ratio", "营收_待核": "revenue",
            "净利润": "net_profit", "净资产": "net_assets", "ROE%": "roe",
            "每股净资产": "bps", "总股本亿股": "shares", "财报期": "report_date",
            "上市日期": "list_date", "行业代码": "industry", "地域代码": "region",
            "所属板块": "boards", "板块数": "n_boards",
        }
        u = u.rename(columns=cols)
        keep = [v for v in cols.values() if v in u.columns]
        u2 = u[keep].where(u[keep].notna(), None)

        # ⚠️ 单位统一（2026-10-06）：universe.csv 的市值是【亿元】，
        #    而 Tushare 的 total_mv / circ_mv 是【万元】。
        #    本项目统一用【万元】，否则两者混用差 1 万倍，会彻底污染市值因子。
        #    实测反推：mktcap(亿) / shares(亿股) = 股价 ✓
        for col in ("mktcap", "float_mktcap"):
            if col in u2.columns:
                u2[col] = u2[col].map(to_wan)

        # ⚠️ 权益口径统一（U6/A18 判决）：net_assets=含其他权益工具口径、
        #    bps=普通股口径，禁止互推；派生 net_assets_ord（普通股权益）随行，
        #    口径写进 meta（U3 处方款）。
        if {"bps", "shares"}.issubset(u2.columns):
            u2 = derive_equity_caliber(u2)
            keep = keep + ["net_assets_ord"]
        else:
            caliber_meta.append(("stock_info.net_assets_ord",
                                 "未派生：universe.csv 缺 bps/shares 列"
                                 "（base_dbf.csv 缺失）——显式记档，不静默置 NULL"))

        con.executemany(
            f"INSERT OR REPLACE INTO stock_info({','.join(keep)}) "
            f"VALUES({','.join('?'*len(keep))})", u2.itertuples(index=False, name=None))
        con.commit()
        print(f"  股票信息 {len(u2)} 条（市值已统一为{MARKET_CAP_UNIT}）")

    # 板块
    if os.path.exists("sectors_index.csv"):
        s = pd.read_csv("sectors_index.csv", dtype={"代码880": str})
        s = s.rename(columns={"代码880": "code", "板块名": "name",
                              "成分数": "n_members", "更新日期": "updated"})
        s = s.where(s.notna(), None)
        con.executemany("INSERT OR REPLACE INTO sector VALUES(?,?,?,?)",
                        s[["code", "name", "n_members", "updated"]]
                        .itertuples(index=False, name=None))
        n = 0
        with open("sector_members.txt", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 3 or not p[2]:
                    continue
                sc, mem = p[0], [x for x in p[2].split(",") if x]
                con.executemany("INSERT OR IGNORE INTO sector_member VALUES(?,?)",
                                [(sc, m) for m in mem])
                n += len(mem)
        con.commit()
        print(f"  板块 {len(s)} 个 / 成分关系 {n} 条")

    con.executemany("INSERT OR REPLACE INTO meta VALUES(?,?)", [
        ("built_at", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("source", "本机通达信 vipdoc (不复权)"),
        ("daily_rows", str(len(d_rows))),
        ("weekly_rows", str(len(w_rows))),
        ("n_codes", str(len(codes))),
        *caliber_meta,          # stock_info 权益/股本口径记录（U6 规则③/U3）
    ])
    con.commit()
    con.execute("ANALYZE")
    con.commit()
    con.close()

    sz = os.path.getsize(DB) / 1024 / 1024
    print(f"\n完成: {DB}  {sz:.0f} MB  总耗时 {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    build()
