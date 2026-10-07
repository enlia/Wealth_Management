#!/usr/bin/env python
"""
tdx.py —— 本机通达信数据读取工具（纯离线，不联网）

存储格式（已实测验证）：
  vipdoc/<市场>/<类型>/<前缀><代码>.<ext>
  .day 行情 = 32 字节定长记录，struct 格式 "<IIIIIfII"
      date   uint32  YYYYMMDD
      open   uint32  价格分（×100）
      high   uint32  同上
      low    uint32  同上
      close  uint32  同上
      amount float32 成交额（元）
      vol    uint32  成交量（股）
      rsv    uint32  保留
  注意：本地 .day 是【不复权】原始价。

用法示例：
  python tdx.py info                        查看数据覆盖情况
  python tdx.py show 600519 --last 20       打印日线尾部
  python tdx.py show 600519 --period weekly 打印周线
  python tdx.py ind 600519                  打印最新技术指标
  python tdx.py export 600519               导出 CSV 到 out/
  python tdx.py export-all                  全市场批量导出（多进程）
  python tdx.py qfq 600519                  取前复权数据（走新浪，需联网）
  python tdx.py list                        列出全部本地标的
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

# ── 路径配置（按你本机实际位置）──────────────────────────────
TDX = r"C:\SoftWare\TONGDAXIN\Body\vipdoc"
OUT = "out"

DTYPE = np.dtype([
    ("date", "<u4"), ("open", "<u4"), ("high", "<u4"),
    ("low", "<u4"), ("close", "<u4"), ("amount", "<f4"),
    ("vol", "<u4"), ("rsv", "<u4"),
])
assert DTYPE.itemsize == 32, "记录长度必须是 32 字节"


def market_of(code, mkt=None):
    """判断市场目录。
    重要：88xxxx 是【通达信板块指数】，存放在 sh 目录，不能按北交所处理。
    另外 sh000001(上证指数) 与 sz000001(平安银行) 代码相同，
    需要用 mkt 参数显式区分：read_day('000001', mkt='sh') 取上证指数。
    ⚠️ 补 2026-10-03：原先 9xx(B股)/1xx/2xx/5xx(基金债券) 会返回 None，
       导致 3124 个标的被漏掉。已补全。
    """
    if mkt:
        return mkt
    c = code
    # 沪市目录 sh：6xxxxx A股 / 900xxx B股 / 5xxxxx 基金 / 1xxxxx 债券 / 000xxx 指数 / 88xxxx 板块
    if c.startswith(("6", "9", "5", "1", "88")):
        return "sh"
    # 深市目录 sz：0xxxxx 主板 / 3xxxxx 创业板 / 200xxx B股 / 1xxxxx 基金债券 / 399xxx 指数
    if c.startswith(("0", "3", "2")):
        return "sz"
    # 北交所目录 bj：4xxxxx / 8xxxxx
    if c.startswith(("4", "8")):
        return "bj"
    return None


def all_codes(kind="lday", with_prefix=False):
    """列出本地所有标的代码。
    ⚠️ 2026-10-03 修正：默认**不带**前缀（保持向后兼容），
       但 with_prefix=True 时返回 'sh600519' 形式——
       因为 sh000001(上证指数) 与 sz000001(平安银行) 代码相同，
       落库时必须用带前缀的形式才不会混淆。
    """
    ext = {"lday": ".day", "minline": ".lc1", "fzline": ".lc5"}.get(kind, ".day")
    out = []
    for m in ("sh", "sz", "bj"):
        for f in glob.glob(os.path.join(TDX, m, kind, f"*{ext}")):
            base = os.path.basename(f)[:-len(ext)]
            out.append(m + base[2:] if with_prefix else base[2:])
    return sorted(set(out))


def day_path(code, kind="lday", mkt=None):
    m = market_of(code, mkt)
    if not m:
        return None
    ext = {"lday": ".day", "minline": ".lc1", "fzline": ".lc5"}.get(kind, ".day")
    p = os.path.join(TDX, m, kind, f"{m}{code}{ext}")
    return p if os.path.exists(p) else None


def price_scale(code, mkt=None):
    """价格缩放因子（真实价 = 原始整数 × 本系数）
    ⚠️ 交叉验证发现（源自 mootdx/tdxpy 的 SECURITY_COEFFICIENT）：
    并非所有品种都是 ×0.01。通达信按品种分级：
        A股 / 指数 / 板块指数 : 0.01
        沪 B股 / 基金 / 债券  : 0.001
        深 B股（200/201/202）: 0.01   ← 与沪 B **不同纲**（U7，见下）
    实测对照：
        宁波银行B 900948  原始 3736    → ×0.001 = 3.736    ✅（×0.01=37.36 错）
        深物业B  200011   库内 0.59   → ×0.01  = 5.90     ✅（×0.001=0.59 错 10×）
        上证指数  000001   原始 384219  → ×0.01  = 3842.19  ✅
        生物制药  880402   原始 239123  → ×0.01  = 2391.23  ✅
    注意：沪市 000xxx 是【指数】不是债券，别把 sh000001 误判成债券。

    ⚠️ **深 B 与沪 B 必须分开判**（UNITS U7，2026-10-13 三方对拍定案）：
       深 B（sz200/201/202 段）行情整数实为 ×0.01 缩放，旧版与基金/债券同归
       0.001 → 库内 close 小 10 倍、跌停判定对深 B 单侧假阳性（2023 假封 9,189 格）。
       定案证据：G4 恒等式 amount/(vol×close) 深 B 中位 9.95、沪 B 0.99；
       Tushare 权威限价对拍 ×10 后 114/114 命中（对照 A 股 ×1 必中、×10 全不中）；
       通达信导出物 stock_info.price/库内末日 close = 10.00。
       （vipdoc 原始整数层原件不在机，该层直接证据 UNKNOWN；三方对拍定案。）
    """
    c = code
    m = mkt or market_of(c)
    if c.startswith("88"):
        return 0.01                                   # 板块指数
    if c.startswith("900"):
        return 0.001                                  # 沪 B（900xxx，×0.001 实测）
    if c.startswith(("200", "201", "202")):
        return 0.01                                   # 深 B（200/201/202 段，U7 ×0.01）
    if m == "sh":
        if c.startswith("000") or c.startswith("950"):
            return 0.01                               # 沪市指数
        if c[0] in "12345":
            return 0.001                              # 沪市债券/基金
    elif m == "sz":
        if c.startswith("399"):
            return 0.01                               # 深证指数
        if c[0] in "03":
            return 0.01                               # 深市 A股（000/001/002/003/301/302）
        return 0.001                                  # 1x 基金/债券（2x B股已上移 0.01，U7）
    return 0.01                                       # A股（00/30 开头）与北交所


def read_day(code, kind="lday", mkt=None):
    """读取单只标的的日线，返回 DataFrame（未复权）
    mkt: 强制市场目录 'sh'/'sz'/'bj'，用于区分 sh000001(指数) 与 sz000001(股票)"""
    m = market_of(code, mkt)
    p = day_path(code, kind, mkt)
    if not p:
        return None
    raw = np.fromfile(p, dtype=DTYPE)
    if raw.size == 0:
        return None
    k = price_scale(code, m)      # 0.01 表示 原始整数 × 0.01 = 真实价
    d = pd.DataFrame({
        "date": pd.to_datetime(raw["date"].astype(str), format="%Y%m%d"),
        "open": raw["open"] * k,
        "high": raw["high"] * k,
        "low": raw["low"] * k,
        "close": raw["close"] * k,
        "amount": raw["amount"].astype("float64"),
        "vol": raw["vol"].astype("float64"),
    })
    return d.dropna(subset=["close"]).reset_index(drop=True)


def resample(d, period):
    """日线 → 周/月线。
    关键：标签用【周内最后一个实际交易日】，与通达信周线对齐。
    （若用 W-FRI 直接标记，9/30(周三) 会被错标成 10/02，与通达信不一致）"""
    if period == "daily":
        return d
    rule = {"weekly": "W-FRI", "monthly": "ME"}.get(period)
    if rule is None:
        raise ValueError("period 只能是 daily / weekly / monthly")
    g = d.set_index("date")
    res = g.resample(rule)
    # last_date 记录每组真实的最后交易日
    last_dates = res["close"].apply(lambda s: s.index[-1] if len(s) else pd.NaT)
    out = res.agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "amount": "sum", "vol": "sum"}).dropna()
    out["date"] = last_dates[out.index]
    return out.reset_index(drop=True)


def indicators(d):
    """计算常用技术指标"""
    c = d["close"]
    out = {}
    for n in (5, 10, 20, 60, 120, 250):
        if len(c) >= n:
            out[f"MA{n}"] = c.rolling(n).mean().iloc[-1]
    if len(c) >= 34:
        dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        dea = dif.ewm(span=9, adjust=False).mean()
        out["MACD_DIF"] = dif.iloc[-1]
        out["MACD_DEA"] = dea.iloc[-1]
        out["MACD_BAR"] = (dif.iloc[-1] - dea.iloc[-1]) * 2
    if len(c) >= 20:
        sd = c.rolling(20).std(ddof=0)
        mid = c.rolling(20).mean()
        out["BOLL_UP"] = mid.iloc[-1] + 2 * sd.iloc[-1]
        out["BOLL_LOW"] = mid.iloc[-1] - 2 * sd.iloc[-1]
    if len(c) >= 15:
        delta = c.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        dn = (-delta.clip(upper=0)).rolling(14).mean()
        out["RSI14"] = 100 * up.iloc[-1] / (up.iloc[-1] + dn.iloc[-1]) if (up.iloc[-1] + dn.iloc[-1]) else np.nan
    if len(c) >= 20:
        pv = d["vol"].rolling(20).mean().iloc[-1]
        out["量比20"] = d["vol"].iloc[-1] / pv if pv else np.nan
    # 波动率（年化，%）
    if len(c) >= 61:
        r = np.log(c / c.shift(1)).dropna()
        out["年化波动率%"] = r.iloc[-60:].std() * np.sqrt(244) * 100
    return out


# ── 子命令 ────────────────────────────────────────────────
def cmd_info(args):
    print("=" * 58)
    print("本机通达信数据概况")
    print("=" * 58)
    print(f"数据目录: {TDX}")
    codes = all_codes()
    print(f"标的总数: {len(codes)}  (sh/sz/bj 合计日线文件数)")
    for m, label in (("sh", "上交所"), ("sz", "深交所"), ("bj", "北交所")):
        n = len(glob.glob(os.path.join(TDX, m, "lday", "*.day")))
        print(f"  {label} {m}: {n} 个")
    print()
    # 抽样看时间覆盖
    print("抽样标的（日期范围 / 记录数 / 末日收盘）:")
    for c in (["600519", "000001", "603259", "002142", "300750", "688202"]):
        d = read_day(c)
        if d is not None and len(d):
            print(f"  {c}: {len(d):5d} 条  "
                  f"{d['date'].iloc[0]:%Y-%m-%d} ~ {d['date'].iloc[-1]:%Y-%m-%d}  "
                  f"末收 {d['close'].iloc[-1]}")
    print()
    for kind, desc, ext in (("lday", "日线", ".day"),
                            ("minline", "1分钟线", ".lc1"),
                            ("fzline", "5分钟线", ".lc5")):
        n = len(glob.glob(os.path.join(TDX, "sh", kind, f"*{ext}")))
        print(f"  {desc}({kind}/*.{ext[1:]}): 沪市 {n} 个文件")
    print()
    print("提示: 本地 .day 为【不复权】数据；需要前复权用 `qfq` 子命令。")
    print("=" * 58)


def cmd_show(args):
    d = read_day(args.code)
    if d is None:
        print(f"未找到 {args.code} 的本地数据"); return
    d = resample(d, args.period)
    n = args.last
    if args.start:
        d = d[d["date"] >= pd.Timestamp(args.start)]
    print(f"=== {args.code} {args.period} 近 {n} 根 "
          f"(共 {len(d)} 根, 数据止于 {d['date'].iloc[-1]:%Y-%m-%d}) ===")
    show = d.tail(n).copy()
    show["date"] = show["date"].dt.strftime("%Y-%m-%d")
    show["涨跌%"] = (show["close"] / show["close"].shift(n) - 1) * 100 if args.period == "daily" \
        else (show["close"] / show["open"] - 1) * 100
    show["量(万手)"] = (show["vol"] / 1e6).round(2)
    show["额(亿)"] = (show["amount"] / 1e8).round(2)
    print(show[["date", "open", "high", "low", "close", "涨跌%",
                "量(万手)", "额(亿)"]].to_string(index=False, float_format=lambda x: f"{x:8.2f}"))


def cmd_ind(args):
    d = read_day(args.code)
    if d is None:
        print(f"未找到 {args.code} 的本地数据"); return
    last = d.iloc[-1]
    print(f"=== {args.code} 技术指标 @ {last['date']:%Y-%m-%d} ===")
    print(f"收盘 {last['close']:.2f}   开 {last['open']:.2f}  高 {last['high']:.2f}  低 {last['low']:.2f}")
    print(f"成交额 {last['amount']/1e8:.2f} 亿   成交量 {last['vol']/1e6:.2f} 万股")
    print("-" * 46)
    for k, v in indicators(d).items():
        print(f"  {k:<14} {v:10.2f}")
    print("-" * 46)
    h = d["high"].max(); l = d["low"].min()
    print(f"区间最高 {h:.2f}  距高点 {(last['close']/h-1)*100:+.2f}%")
    print(f"区间最低 {l:.2f}  距低点 {(last['close']/l-1)*100:+.2f}%")


def _export_one(code):
    try:
        d = read_day(code)
        if d is None or len(d) < 2:
            return 0
        d.to_csv(os.path.join(OUT, f"{code}.csv"),
                 index=False, encoding="utf-8-sig")
        return 1
    except Exception:
        return 0


def cmd_export_all(args):
    os.makedirs(OUT, exist_ok=True)
    codes = all_codes()
    if args.limit:
        codes = codes[:args.limit]
    print(f"导出 {len(codes)} 只到 ./{OUT}/ ...")
    from multiprocessing import Pool
    with Pool(args.workers) as p:
        ok = sum(p.imap_unordered(_export_one, codes, chunksize=64))
    print(f"完成：成功 {ok} / {len(codes)}")


def cmd_export(args):
    os.makedirs(OUT, exist_ok=True)
    d = read_day(args.code)
    if d is None:
        print(f"未找到 {args.code}"); return
    p = os.path.join(OUT, f"{args.code}_{args.period}.csv")
    resample(d, args.period).to_csv(p, index=False, encoding="utf-8-sig")
    print(f"已导出 {len(d)} 行 -> {p}")


def cmd_qfq(args):
    """前复权数据：本地无除权表，走新浪接口（需联网，单只查询）"""
    try:
        import akshare as ak
    except ImportError:
        print("需要先安装 akshare"); return
    prefix = "sh" if args.code[0] == "6" else ("bj" if args.code[0] in "48" else "sz")
    try:
        d = ak.stock_zh_a_daily(symbol=prefix + args.code, adjust="qfq")
    except Exception as e:
        print(f"新浪接口失败: {type(e).__name__} {str(e)[:80]}")
        print("该数据源在本机可能被代理拦截，改用东方财富也不行时请用本地不复权数据。")
        return
    d["date"] = pd.to_datetime(d["date"])
    if args.start:
        d = d[d["date"] >= pd.Timestamp(args.start)]
    d = d.reset_index(drop=True)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{args.code}_qfq.csv")
    d.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"前复权 {len(d)} 行 -> {p}")
    print(d.tail(args.last)[["date", "open", "high", "low", "close"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.2f}"))


def cmd_list(args):
    codes = all_codes()
    print(f"共 {len(codes)} 个标的，前 60 个:")
    print("  ".join(codes[:60]) + (" ..." if len(codes) > 60 else ""))


def main():
    ap = argparse.ArgumentParser(
        description="本机通达信数据读取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="数据覆盖概况").set_defaults(fn=cmd_info)
    sub.add_parser("list", help="列出全部标的").set_defaults(fn=cmd_list)

    p = sub.add_parser("show", help="打印K线")
    p.add_argument("code"); p.add_argument("--period", default="daily",
            choices=["daily", "weekly", "monthly"])
    p.add_argument("--last", type=int, default=20); p.add_argument("--start")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("ind", help="技术指标")
    p.add_argument("code"); p.set_defaults(fn=cmd_ind)

    p = sub.add_parser("export", help="导出单只CSV")
    p.add_argument("code"); p.add_argument("--period", default="daily",
            choices=["daily", "weekly", "monthly"])
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("export-all", help="全市场批量导出")
    p.add_argument("--limit", type=int, default=0); p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_export_all)

    p = sub.add_parser("qfq", help="前复权数据(联网)")
    p.add_argument("code"); p.add_argument("--start"); p.add_argument("--last", type=int, default=10)
    p.set_defaults(fn=cmd_qfq)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
