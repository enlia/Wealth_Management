"""Tushare 数据源工具：路径、枚举生成、落盘。

从 ``fetch_all_tushare.py`` 拆出（原文件触及 500 行强制拆分区）。
职责单一：只回答「数据在哪」「要拉哪些日期/期次」「怎么存」。
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fetch_tushare import call  # noqa: E402
from tushare_tasks import END, START  # noqa: E402

from factor_lab.config import DB_PATH, is_a_share  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runtime" / "tushare"

def trading_days(token: str) -> list[str]:
    """取交易日历（缓存到本地，避免重复请求）。"""
    cache = OUT / "trade_cal.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = call(token, "trade_cal",
                  {"start_date": START, "end_date": END, "is_open": "1"})
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return sorted(df["cal_date"].astype(str).tolist())


def report_periods() -> list[str]:
    """报告期列表：2015Q4 ~ 2026Q2。"""
    out = []
    for y in range(2015, 2027):
        for q, mmdd in ((1, "0331"), (2, "0630"), (3, "0930"), (4, "1231")):
            if y == 2015 and q != 4:
                continue
            if y == 2026 and (q > 2 or (q == 2 and mmdd > "0930")):
                continue
            out.append(f"{y}{mmdd}")
    return out


def months() -> list[str]:
    out = []
    for y in range(2015, 2027):
        for m in range(1, 13):
            if y == 2015 and m < 12:
                continue
            if y == 2026 and m > 9:
                continue
            out.append(f"{y}{m:02d}")
    return out


def a_share_codes() -> list[str]:
    """本机 A 股代码 → Tushare ts_code。"""
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    codes = [r[0] for r in con.execute("SELECT DISTINCT code FROM bar_daily")
             if is_a_share(r[0])]
    out = []
    for c in codes:
        c = c.lower()
        mkt, num = c[:2], c[2:]
        sfx = {"sh": "SH", "sz": "SZ", "bj": "BJ"}[mkt]
        out.append(f"{num}.{sfx}")
    return sorted(out)


# Tushare 日期字段的标准命名（Tushare 全用 YYYYMMDD 数字串）
DATE_COLS: tuple[str, ...] = (
    "trade_date", "ann_date", "end_date", "start_date", "list_date",
    "delist_date", "float_date", "cal_date", "f_ann_date", "f_end_date",
    "update_date", "release_date", "lock_date",
    # ⚠️ 下面这几个是实测扫描补上的（2026-10-06 review 发现漏迁）：
    #   in_date  指数/行业成分「纳入日期」，29 张表都有 —— 做 universe 必用
    #   actual_date / pre_date  财报预约披露日期
    #   begin_date / pretrade_date 上交所债券字段
    #   out_date  指数成分「剔除日期」
    "in_date", "out_date", "begin_date", "actual_date", "pre_date",
    "pretrade_date",
)

# 兜底：列名含 date 且类型是 string/float 时也当日期处理。
# ⚠️ **为什么必须留兜底**：靠人肉枚举 DATE_COLS 必然漏 ——
#   实测第一轮枚举了 13 个，扫全库发现还漏 5 个
#   （in_date 涉及 29 张表），其中 in_date 是指数成分纳入日期，
#   构造股票池时必然要用。漏一个就是一处静默失效。
# ⚠️ **排除 update_flag**：它含 "date" 子串但是 0/1 标志位，不是日期。
DATE_EXCLUDE = frozenset({"update_flag", "update_date_flag"})
DATE_HINT = re.compile(r"date$|^date_|_date_", re.I)


def is_date_col(name: str, dtype_kind: str) -> bool:
    """该列是否应按日期处理。"""
    if name in DATE_EXCLUDE:
        return False
    if name in DATE_COLS:
        return True
    return dtype_kind in "OSUf" and bool(DATE_HINT.search(name))


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """落盘前统一类型：**日期列一律 int32，代码列一律字符串**。

    ⚠️ **实测踩过（2026-10-06）：日期列存成 string会导致静默错误**
    Tushare 返回的日期是 `'20240102'`（字符串），`to_parquet` 会原样存成
    string 列。实测 **19张表全部中招**（stk_limit / adj_factor / 三大财务表…）。
    后果不是报错，而是**静默失效**：
        df[df['trade_date'] == 20240102]        # int 比较 → 0 行
        df[df['trade_date'] == '20240102']      # 唯一能匹配的写法
    下游并库、筛选、join 全部拿到空结果，却**没有任何错误提示**。

    同理代码列：`600519.SH` 这类值一旦被推断成别的类型，
    `.str.slice()` 会炸或返回 NaN。

    **为什么放在 save() 里统一做**：转换只在数据入口发生一次。
    下游各自转是「东拼西凑」，且总会有人忘记 —— 规范禁止那种写法。
    """
    out = df.copy()
    for c in out.columns:
        kind = out[c].dtype.kind
        if not is_date_col(c, kind):
            continue
        s = out[c]
        if s.dtype.kind in "OSU":
            # 空值可能是 ''、'None'、None、NaN —— 统一成 NaN 再转
            cleaned = s.astype("string").str.strip().replace(
                {"": pd.NA, "None": pd.NA, "nan": pd.NA, "NaT": pd.NA})
            num = pd.to_numeric(cleaned, errors="coerce")
            # ⚠️ 必须用 pandas 可空的 **Int32**，不能用 numpy int32。
            #   实测踩过：stk_managers.end_date 有 73,575 个空值
            #   （在任高管本就无离任日期，是业务语义而非数据损坏），
            #   numpy int32 无法存 NA，整表迁移失败被回滚。
            #   Int32 落盘后仍是 int32 物理类型，但支持缺失值。
            if num.isna().any():
                out[c] = num.astype("Int32")
            else:
                out[c] = num.astype("int32")
        elif s.dtype.kind == "f":
            # float 存日期（小数化过的）→ 必须是整数，无余量才转
            if not np.allclose(s.dropna() % 1, 0):
                raise ValueError(
                    f"日期列 {c!r} 含非整数小数，无法安全转int32："
                    f"样例 {s.dropna().head(3).tolist()}")
            out[c] = s.round().astype("Int32")
        elif s.dtype.kind in "iu":
            # ⚠️ **pandas 可空整数必须保持可空**。
            #   实测踩过：`Int32`（含 73,575 个 NA）被强转 numpy `int32`
            #   会抛 `ValueError: cannot convert NA to integer`。
            #   这让 `normalize_types` **不幂等** ——
            #   对已迁移的表再跑一次就崩，迁移脚本无法重跑。
            #   故只有「确认无 NA」时才用 numpy int32（物理类型更紧凑）。
            if s.isna().any():
                out[c] = s.astype("Int32")
            else:
                out[c] = s.astype("int32")
    return out


def save(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """落盘到 runtime/tushare/{tag}.parquet，落盘前统一类型。

    返回归一后的 DataFrame —— 内存里的对象与磁盘保持一致，
    否则会出现「内存是 int、磁盘是 string」的幽灵差异。
    """
    OUT.mkdir(parents=True, exist_ok=True)
    out = normalize_types(df)
    out.to_parquet(OUT / f"{tag}.parquet", index=False)
    return out
