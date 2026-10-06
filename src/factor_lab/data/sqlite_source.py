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
            f"请先运行 src/tools/build_sqlite.py 构建"
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
    codes: list[str] | None,
    start: str | None,
    end: str | None,
    field: str,
    db_path: Path | str = DB_PATH,
) -> pd.DataFrame:
    """读出宽表面板：行=date，列=code。

    这是 alphalens 的 `prices` 输入格式。

    🔴 **`field` 是必填参数，没有默认值**（2026-10-06 改）。
       此前默认 `close`（未复权），被 6 个研究脚本继承 ——
       它们把未复权价喂给 alphalens 算前瞻收益，
       除权日的向下假跳空被当成真实跌幅。

       **实测全市场 5,606 只等权口径的年化偏差（未复权 − 前复权）：**

           2016 −16.80pp / 2019 −10.80pp / 2025 −6.97pp
           2021~2026 每年 −5.1 ~ −8.0pp

       这个量级**大于本项目声称的任何因子收益**（最高的 ep 也只有 +6.1%）。

       为什么不给默认值：**「忘了写口径」应当报错，而不是猜一个。**
       保留默认值等于把设计缺陷藏进 API —— 门禁代替不了设计
       （ENGINEERING.md 第五节）。忘了写会得到 `TypeError`，
       而 TypeError 是吵的，静默的 −16.80pp 不是。

    参数
    ----
    codes : 标的列表，带市场前缀如 'sh600519'。
    start, end : 'YYYY-MM-DD' 或 'YYYYMMDD'。None 表示不限。
    field : close / open / high / low / amount / vol / close_adj
        ⚠️ 收益研究必须用 ``close_adj``（前复权）。``close`` 是未复权价。
        ⚠️ **只有 PB/PE/EP 才允许用未复权 ``close``**（要与 bps/eps 同口径），
           那种场景建议用 :func:`factor_lab.data.load_factor_prices`
           显式取两种口径。

    内存提示
    --------
    全市场 9,595 只 × 2,634 交易日 × 8 字节 ≈ 1.6 GB（float64）。
    本机 13.9 GB，单次全市场读取可行但会挤占内存；
    若只做横截面研究，建议先用 300~1000 只的池子。
    """
    allowed = {"close", "open", "high", "low", "amount", "vol", "close_adj"}
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
    adjusted: bool = False,
) -> pd.DataFrame:
    """读出长表：code / date / open / high / low / close / amount / vol

    因子计算用长表更自然（groupby code），转宽表交给调用方。

    参数
    ----
    adjusted : False（默认）只返回**未复权** OHLC。
        True 时**额外**返回三列前复权价，且**不改动**原始 OHLC：
        ``close_adj`` / ``high_adj`` / ``low_adj``。

    ⚠️ **收益研究必须 adjusted=True**（AGENTS.md 第二节质量红线）。
       ``close`` 是未复权价，除权日出现假跳空：
       实测主板 0.291% 的日收益超过 ±10% 限制，年化偏差 2.33pp。

       为什么**保留**未复权 OHLC 而不是就地替换：
       涨跌停判定必须用**报价**（交易所约束的是报价，与复权无关），
       仙股过滤（2~3000 元）也只对报价成立。直接替换会让这两个判断失效。

       ``high_adj`` / ``low_adj`` 由``high * close_adj/close`` 推导：
       复权是乘性调整，对同一天的高/低价位同样适用。
       ⚠️ ``close_adj`` 为 NULL 的行（实测占 A 股 1.616%）比值为 NaN，
       这三列在该行为空——**按缺失处理，不要填 0**。
    """
    s, e = _norm_date(start), _norm_date(end)
    adj_cols = ", close_adj" if adjusted else ""
    sql = (
        "SELECT code, date, open, high, low, close, amount, vol"
        f"{adj_cols} "
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
    if adjusted:
        cols.append("close_adj")
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    if adjusted:
        # 复权比例：同一天三个价格位共用一个乘数。
        # 用 where 屏蔽 close<=0（脏数据）避免除零产生 inf。
        ratio = (df["close_adj"] / df["close"]).where(df["close"] > 0)
        df["high_adj"] = df["high"] * ratio
        df["low_adj"] = df["low"] * ratio
    return df.reset_index(drop=True)


def load_stock_info(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """股票静态信息：PE/PB/市值/换手率/ROE/行业/所属板块。"""
    with _connect(db_path) as con:
        return pd.read_sql_query("SELECT * FROM stock_info", con)


def load_first_dates(
    codes: list[str] | None = None,
    db_path: Path | str = DB_PATH,
) -> pd.Series:
    """每个 code 在行情表 ``bar_daily`` 的**首个数据日期**（code 索引）。

    用途：``stock_info.list_date`` 缺失（实测 944/8,728 只，bj920 新号段整段缺）
    时，该日期是上市日的**下界代理**（首个数据日 ≥ 真实上市日）——
    拿它计龄是保守口径：数据晚起点的老股会被判新（误杀），
    但真次新股不会被错放（宁可误杀不可错放次新，250 日保护必须生效）。
    """
    sql = "SELECT code, MIN(date) AS d FROM bar_daily"
    params: list = []
    if codes is not None:
        if not codes:
            return pd.Series(dtype="datetime64[ns]")
        sql += " WHERE code IN (%s)" % ",".join("?" * len(codes))
        params.extend(codes)
    sql += " GROUP BY code"
    with _connect(db_path) as con:
        rows = con.execute(sql, params).fetchall()
    if not rows:
        return pd.Series(dtype="datetime64[ns]")
    df = pd.DataFrame(rows, columns=["code", "d"])
    first = pd.to_datetime(pd.to_numeric(df["d"]).astype("int64").astype("string"),
                           format="%Y%m%d")
    return pd.Series(first.to_numpy(), index=df["code"], name="first_date")


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
