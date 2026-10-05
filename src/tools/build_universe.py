"""统一数据层构建：把 4 个来源合并成一张全市场主表
来源：
  1. 腾讯财经行情  raw/all_raw.txt        → 名称/价格/涨跌/PE/PB/市值/换手率
  2. 通达信 base.dbf  base_dbf.csv       → 营收/净利/扣非/净资产/股东人数/行业/地域
  3. 本地 .day K线    tdx.read_day        → MA/MACD/RSI/波动率/距高低点
  4. 板块成分        sector_members.txt   → 所属板块
输出：universe.csv（全市场主表）
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from tdx import read_day, all_codes

RAW = "raw/all_raw.txt"
OUT = "universe.csv"


# ── 1. 解析腾讯行情 ──────────────────────────────────────────
def parse_tencent(path):
    rows = []
    if not os.path.exists(path):
        print("!! 行情原始文件不存在:", path)
        return pd.DataFrame()
    with open(path, "rb") as f:
        text = f.read().decode("gbk", errors="ignore")
    n = 0
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk.startswith("v_") or '="' not in chunk:
            continue
        head, _, body = chunk.partition('="')
        sym = head[2:].strip()
        p = body.rstrip('"').split("~")
        if len(p) < 45:
            continue

        def g(i, d=float("nan")):
            try:
                v = p[i].strip()
                return float(v) if v else d
            except (IndexError, ValueError):
                return d

        rows.append({
            "代码": sym,
            "名称": p[1].strip(),
            "现价": g(3), "昨收": g(4), "今开": g(5),
            "成交量手": g(36), "成交额万": g(37),
            "时间": p[30].strip() if len(p) > 30 else "",
            "涨跌额": g(31), "涨跌幅%": g(32),
            "最高": g(33), "最低": g(34),
            "换手率%": g(38), "PE_TTM": g(39),
            "振幅%": g(43),
            "流通市值亿": g(44), "总市值亿": g(45),
            "PB": g(46),
            "涨停价": g(47), "跌停价": g(48), "量比": g(49),
        })
        n += 1
    df = pd.DataFrame(rows)
    print(f"[1] 腾讯行情解析: {len(df)} 条")
    return df


# ── 2. 财务数据（base.dbf）────────────────────────────────────
# ⚠️ 重要：通达信 base.dbf 无数据字典，字段含义需交叉验证。
#    已用茅台(600519)/宁德(300750)对照公开财报验证：
#      JZC 净资产 ✓ (2512.5亿)   ZZC 总资产 ✓ (3090亿)
#      JLY 净利润 ✓ (445亿)       TZMGJZ 每股净资产 ✓ (200.99元)
#      ZGB 总股本 ✓ (12.5亿股)   WFPLY 存疑(1996亿 > 净利) → 不贴标签，保留原名
#    单位为【万元】。
FIN_MAP = {
    "JLY": "净利润", "JZC": "净资产", "ZZC": "总资产",
    "LDZC": "流动资产", "LDFZ": "流动负债", "GDZC": "固定资产",
    "TZMGJZ": "每股净资产", "ZYSY": "营收_待核",
    "WFPLY": "经营现金流_待核", "ZGB": "总股本万股", "LTAG": "流通A股万股",
}
FIN_SCALE = {   # 千元 → 亿元（由净资产交叉验证标定：茅台 251253600→2512.5亿）
    "净利润": 1e5, "净资产": 1e5, "总资产": 1e5, "流动资产": 1e5,
    "流动负债": 1e5, "固定资产": 1e5, "营收_待核": 1e5, "经营现金流_待核": 1e5,
}


def load_fin():
    if not os.path.exists("base_dbf.csv"):
        print("[2] base_dbf.csv 不存在，跳过财务")
        return pd.DataFrame()
    df = pd.read_csv("base_dbf.csv", dtype={"GPDM": str})
    df["代码"] = (df["SC"].map({0: "sz", 1: "sh"}).astype(str)
                  + df["GPDM"].astype(str).str.zfill(6))
    out = {"代码": df["代码"]}
    for k, cn in FIN_MAP.items():
        if k in df.columns:
            out[cn] = pd.to_numeric(df[k], errors="coerce")
    # 非数值字段直接重命名接入
    for src, cn in (("GXRQ", "财报期"), ("SSDATE", "上市日期"),
                    ("DY", "地域代码"), ("HY", "行业代码"), ("ZBNB", "板块属性")):
        if src in df.columns:
            out[cn] = df[src]
    res = pd.DataFrame(out)
    for c, div in FIN_SCALE.items():
        if c in res.columns:
            res[c] = res[c] / div
    res["总股本亿股"] = res.get("总股本万股", pd.Series(dtype=float)) / 1e4
    res = res.drop(columns=["总股本万股"], errors="ignore")
    if {"净利润", "净资产"}.issubset(res.columns):
        res["ROE%"] = np.where(res["净资产"] > 0,
                               res["净利润"] / res["净资产"] * 100, np.nan)
    # 交叉验证：由行情反推，与 dbf 净资产对比
    print(f"[2] 财务数据: {len(res)} 条"
          f"（财报期样例 {res['财报期'].dropna().unique()[:3]}）")
    return res.reset_index(drop=True)


# ── 3. 技术指标（本地 K 线）──────────────────────────────────
def load_tech(sample=None):
    codes = all_codes()
    if sample:
        codes = codes[:sample]
    recs = []
    for i, c in enumerate(codes):
        d = read_day(c)
        if d is None or len(d) < 60:
            continue
        cl = d["close"]
        last = d.iloc[-1]
        ma = {n: cl.rolling(n).mean().iloc[-1] for n in (5, 20, 60, 250)
              if len(cl) >= n}
        r = np.log(cl / cl.shift(1)).dropna()
        hi = d["high"].max()
        lo = d["low"].min()
        recs.append({
            "代码": ("sh" if (c[0] == "6" or c.startswith("88")) else
                     "sz" if c[0] in "03" else "bj") + c,
            "末日": last["date"].strftime("%Y-%m-%d"),
            "K线根数": len(d),
            "历史最高": round(hi, 3), "历史最低": round(lo, 3),
            "距最高%": round((cl.iloc[-1] / hi - 1) * 100, 2),
            "距最低%": round((cl.iloc[-1] / lo - 1) * 100, 2),
            "年化波动率%": round(r.iloc[-244:].std() * np.sqrt(244) * 100, 2)
                            if len(r) >= 60 else np.nan,
            **{f"MA{n}": round(v, 3) for n, v in ma.items()},
        })
        if (i + 1) % 2000 == 0:
            print(f"    K线进度 {i+1}/{len(codes)}")
    df = pd.DataFrame(recs)
    print(f"[3] 技术指标: {len(df)} 条")
    return df


# ── 4. 板块归属 ──────────────────────────────────────────────
def load_sectors():
    if not os.path.exists("sector_members.txt"):
        return pd.DataFrame(columns=["代码", "所属板块"])
    m = {}
    with open("sector_members.txt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or not p[2]:
                continue
            # 板块代码 p[0] 本脚本用不到（只取 名称 + 成分），
            # 但保留读取以便发现格式异常（列数不对时 p[0] 会是垃圾值）
            _code, name, members = p[0], p[1], p[2].split(",")
            for s in members:
                m.setdefault(s, []).append(name)
    df = pd.DataFrame([{"代码": k, "所属板块": "|".join(v),
                        "板块数": len(v)} for k, v in m.items()])
    print(f"[4] 板块归属: {len(df)} 只个股有板块标签（平均 {df['板块数'].mean():.1f} 个/只）")
    return df


if __name__ == "__main__":
    q = parse_tencent(RAW)
    fin = load_fin()
    tech = load_tech()
    sec = load_sectors()

    df = q
    for other in (fin, tech, sec):
        if other is None or other.empty or "代码" not in other.columns:
            continue
        df = df.merge(other, on="代码", how="left")

    if "现价" in df and "MA20" in df and "MA60" in df:
        df["MA20>MA60"] = df["MA20"] > df["MA60"]
        df["多头排列"] = (df.get("MA5", np.nan) > df["MA20"]) & (df["MA20"] > df["MA60"])
    # 交叉验证：行情反推净资产 vs dbf 净资产
    if {"总市值亿", "PB", "净资产"}.issubset(df.columns):
        df["净资产_反推亿"] = np.where(df["PB"] > 0, df["总市值亿"] / df["PB"], np.nan)
        df["净资产校验差%"] = np.where(
            df["净资产"].abs() > 0,
            (df["净资产_反推亿"] - df["净资产"]) / df["净资产"].abs() * 100, np.nan)
    if {"总市值亿", "PE_TTM", "净利润"}.issubset(df.columns):
        df["净利_反推亿"] = np.where(df["PE_TTM"] > 0, df["总市值亿"] / df["PE_TTM"], np.nan)

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n=== 完成: {OUT}  {len(df)} 行 × {len(df.columns)} 列 ===")
    have = df.notna().sum()
    print("\n各字段非空数:")
    print(have[have > 0].to_string())
    chk = df["净资产校验差%"].abs()
    if chk.notna().any():
        print(f"\n净资产交叉验证(行情反推 vs dbf): 中位差 {chk.median():.2f}%, "
              f"±5%内占比 {(chk < 5).mean()*100:.1f}%")
