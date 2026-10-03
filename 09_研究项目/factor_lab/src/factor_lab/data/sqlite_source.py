"""数据访问层：只负责从 SQLite 读出 DataFrame，不做任何计算。

设计约束（AGENTS.md 架构要求）：
- 本层不依赖 factors / analysis，依赖单向向下
- 所有返回的宽表索引为 DatetimeIndex(date)，列名为带市场前缀的代码
  （因为 sh000001 上证指数 与 sz000001 平安银行 代码相同，必须靠前缀区分）
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ..config import DB_PATH

# SQLite 连接参数：只读 + mmap（WAL 模式下可与写入共存）
_CONN_KW = {
    "check_same_thread": False,
    "timeout": 30.0,
}


def _connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(
            f"数据库不存在: {p}\n"
            f"请先运行 04_工具脚本/build_sqlite.py 构建"
        )
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, **_CONN_KW)
    con.execute("PRAGMA query_only = ON")
    return con


def available_code_count(db_path: Path | str = DB_PATH) -> int:
    with _connect(db_path) as con:
        return con.execute("SELECT COUNT(DISTINCT code) FROM bar_daily").fetchone()[0]


def all_codes(db_path: Path | str = DB_PATH) -> list[str]:
    with _connect(db_path) as con:
        return [r[0] for r in con.execute("SELECT DISTINCT code FROM bar_daily ORDER BY code")]


def load_prices(
    codes: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    field: str = "close",
    db_path: Path | str = DB_PATH,
) -> pd.DataFrame:
    """读出宽表面板：行=date，列=code。

    这是 alphalens 的 `prices` 输入格式。

    参数
    ----
    codes : 标的列表，带市场前缀如 'sh600519'。None 表示全部。
    start, end : 'YYYY-MM-DD' 或 'YYYYMMDD'。None 表示不限。
    field : close / open / high / low / amount / vol

    内存提示
    --------
    全市场 9,595 只 × 2,634 交易日 × 8 字节 ≈ 1.6 GB（float64）。
    本机 13.9 GB，单次全市场读取可行但会挤占内存；
    若只做横截面研究，建议先用 300~1000 只的池子。
    """
    allowed = {"close", "open", "high", "low", "amount", "vol"}
    if field not in allowed:
        raise ValueError(f"field 必须是 {allowed} 之一，得到 {field!r}")

    s = _norm_date(start)
    e = _norm_date(end)

    sql = f"SELECT code, date, {field} AS v FROM bar_daily WHERE 1=1"
    params: list = []
    if codes is not None:
        if not codes:
            return pd.DataFrame()
        ph = ",".join("?" * len(codes))
        sql += f" AND code IN ({ph})"
        params.extend(codes)
    if s:
        sql += " AND date >= ?"
        params.append(s)
    if e:
        sql += " AND date <= ?"
        params.append(e)
    sql += " ORDER BY code, date"

    with _connect(db_path) as con:
        rows = con.execute(sql, params).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["code", "date", "v"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    wide = df.pivot(index="date", columns="code", values="v")
    # 列名规范化成 str，便于后续 is_a_share 等判断
    wide.columns.name = "code"
    return wide.sort_index()


def load_long(
    codes: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    db_path: Path | str = DB_PATH,
) -> pd.DataFrame:
    """读出长表：code / date / open / high / low / close / amount / vol

    因子计算用长表更自然（groupby code），转宽表交给调用方。
    """
    s, e = _norm_date(start), _norm_date(end)
    sql = (
        "SELECT code, date, open, high, low, close, amount, vol "
        "FROM bar_daily WHERE 1=1"
    )
    params: list = []
    if codes is not None:
        if not codes:
            return pd.DataFrame()
        ph = ",".join("?" * len(codes))
        sql += f" AND code IN ({ph})"
        params.extend(codes)
    if s:
        sql += " AND date >= ?"
        params.append(s)
    if e:
        sql += " AND date <= ?"
        params.append(e)
    sql += " ORDER BY code, date"

    with _connect(db_path) as con:
        rows = con.execute(sql, params).fetchall()
    cols = ["code", "date", "open", "high", "low", "close", "amount", "vol"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.reset_index(drop=True)


def load_stock_info(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """股票静态信息：PE/PB/市值/换手率/ROE/行业/所属板块。"""
    with _connect(db_path) as con:
        return pd.read_sql_query("SELECT * FROM stock_info", con)


def load_sectors(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as con:
        return pd.read_sql_query("SELECT * FROM sector", con)


def load_sector_members(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as con:
        return pd.read_sql_query("SELECT * FROM sector_member", con)


def load_weekly(
    codes: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    field: str = "close",
    db_path: Path | str = DB_PATH,
) -> pd.DataFrame:
    """周线宽表（本地已按通达信口径聚合，标签=周内最后实际交易日）。"""
    return _load_weekly_impl(codes, start, end, field, db_path)


def _load_weekly_impl(codes, start, end, field, db_path) -> pd.DataFrame:
    allowed = {"close", "open", "high", "low", "amount", "vol"}
    if field not in allowed:
        raise ValueError(f"field 必须是 {allowed} 之一")
    s, e = _norm_date(start), _norm_date(end)
    sql = f"SELECT code, date, {field} AS v FROM bar_weekly WHERE 1=1"
    params: list = []
    if codes is not None:
        if not codes:
            return pd.DataFrame()
        sql += " AND code IN (%s)" % ",".join("?" * len(codes))
        params.extend(codes)
    if s:
        sql += " AND date >= ?"; params.append(s)
    if e:
        sql += " AND date <= ?"; params.append(e)
    with _connect(db_path) as con:
        rows = con.execute(sql + " ORDER BY code, date", params).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["code", "date", "v"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    w = df.pivot(index="date", columns="code", values="v").sort_index()
    w.columns.name = "code"
    return w


def _norm_date(d: str | None) -> int | None:
    if d is None:
        return None
    return int(str(d).replace("-", ""))
