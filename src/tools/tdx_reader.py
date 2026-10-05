# -*- coding: utf-8 -*-
"""
通达信本地 .day 文件解析器
==========================
.day 格式：每 32 字节一条记录
  偏移 0  uint32  date    YYYYMMDD
  偏移 4  uint32  open    价格*100
  偏移 8  uint32  high    价格*100
  偏移 12 uint32  low     价格*100
  偏移 16 uint32  close   价格*100
  偏移 20 float32 amount  成交额
  偏移 24 uint32  volume  成交量(股)
  偏移 28 uint32  保留
注意：本地 .day 为【不复权】数据（除权除息未调整）
"""

import os
import struct
import pandas as pd

VIPDOC = r"C:\SoftWare\TONGDAXIN\Body\vipdoc"


def code_to_path(code):
    """股票代码 -> .day 文件路径"""
    code = str(code).zfill(6)
    if code[0] == "6":
        return os.path.join(VIPDOC, "sh", "lday", f"sh{code}.day")
    if code[0] in ("0", "3"):
        return os.path.join(VIPDOC, "sz", "lday", f"sz{code}.day")
    return os.path.join(VIPDOC, "bj", "lday", f"bj{code}.day")


def read_day(code):
    """解析单只股票日线，返回 DataFrame"""
    path = code_to_path(code)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        buf = f.read()
    n = len(buf) // 32
    if n < 60:
        return None

    fmt = "<IIIIIfII"
    rows = []
    for i in range(n):
        d, o, h, l, c, amt, vol, _ = struct.unpack(fmt, buf[i * 32:(i + 1) * 32])
        rows.append((d, o / 100.0, h / 100.0, l / 100.0, c / 100.0, amt, vol))

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close",
                                     "amount", "volume"])
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    return df


def to_weekly(df):
    """日线重采样为周线"""
    x = df.set_index("date")
    w = x.resample("W").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return w


def list_codes(market=None, min_records=300):
    """列出本地可用的股票代码（按数据量筛选，排除新股）"""
    codes = []
    dirs = []
    if market in (None, "sh"):
        dirs.append(os.path.join(VIPDOC, "sh", "lday"))
    if market in (None, "sz"):
        dirs.append(os.path.join(VIPDOC, "sz", "lday"))
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".day"):
                continue
            code = fn[2:8]
            # 只要个股，排除指数/基金
            if not (code[0] == "6" or code[0] in ("0", "3")):
                continue
            p = os.path.join(d, fn)
            if os.path.getsize(p) // 32 >= min_records:
                codes.append(code)
    return sorted(set(codes))


if __name__ == "__main__":
    df = read_day("603259")
    print("药明康德 603259 本地日线：", len(df), "条")
    print(df.tail(3)[["date", "high", "low", "close"]].to_string(index=False))
    codes = list_codes()
    print("\n本地可用个股数（>=300 条日线）:", len(codes))
