"""数据资产总览：统计已获取的数据规模与质量"""
import os
import pandas as pd

pd.set_option("display.width", 200)

print("=" * 70)
print("本机 + 网络 股市数据资产总览")
print("=" * 70)

# 1. 本地通达信
V = r"C:\SoftWare\TONGDAXIN\Body\vipdoc"
print("\n【一】本机通达信 vipdoc (4.4 GB) —— 离线、免费、完整")
for kind, desc, ext in (("lday", "日线", ".day"), ("minline", "1分钟线", ".lc1"),
                        ("fzline", "5分钟线", ".lc5")):
    tot = 0
    for m in ("sh", "sz", "bj"):
        d = os.path.join(V, m, kind)
        if os.path.isdir(d):
            tot += len([f for f in os.listdir(d) if f.endswith(ext)])
    print(f"   {desc:8s} {tot:>6,} 个文件")

# 2. 通达信内部数据库
print("\n【二】通达信内部数据库 (T0002/hq_cache) —— 本次新发现")
items = [
    ("base.dbf", "证券信息库", "7,992 条 × 40 字段：净资产/净利/总资产/营收/现金流/每股净资产/行业/地域/上市日期/财报期"),
    ("infoharbor_block.dat", "板块定义", "547 个板块：名称 + 880代码 + 79,455 条成分股归属"),
    ("spblock.dat", "融资融券标的", "两融标的清单"),
    ("gbbq", "除权除息表", "5.6 MB 加密二进制，无法解析（前复权需另走接口）"),
]
for f, d, desc in items:
    p = os.path.join(V, "..", "T0002", "hq_cache", f)
    p = os.path.normpath(p)
    sz = os.path.getsize(p) if os.path.exists(p) else 0
    print(f"   {d:12s} {f:22s} {sz/1024:>9.0f} KB   {desc}")

# 3. 网络抓取
print("\n【三】网络补充（腾讯财经 qt.gtimg.cn，4分42秒 / 8474 标的）")
if os.path.exists("raw/all_raw.txt"):
    print(f"   原始行情 {os.path.getsize('raw/all_raw.txt')/1024/1024:.1f} MB")

# 4. 主表
print("\n【四】整合主表 universe.csv")
if os.path.exists("universe.csv"):
    df = pd.read_csv("universe.csv")
    print(f"   {len(df):,} 行 × {len(df.columns)} 列")
    groups = {
        "行情快照": ["现价", "涨跌幅%", "最高", "最低", "换手率%", "量比", "振幅%"],
        "估值": ["PE_TTM", "PB", "总市值亿", "流通市值亿"],
        "财务": ["净利润", "净资产", "总资产", "营收_待核", "经营现金流_待核", "每股净资产", "ROE%"],
        "技术面": ["MA5", "MA20", "MA60", "MA250", "年化波动率%", "距最高%", "距最低%"],
        "标签": ["所属板块", "板块数", "行业代码", "地域代码", "财报期", "上市日期"],
    }
    for g, cols in groups.items():
        ok = [c for c in cols if c in df.columns and df[c].notna().any()]
        n = sum(int(df[c].notna().sum()) for c in ok)
        print(f"   {g:6s}: {len(ok)} 个字段, 有效值 {n:,}")

    # 质量校验
    chk = df["净资产校验差%"].abs()
    print(f"\n   质量校验（净资产：dbf vs 行情反推）: 中位差 {chk.median():.2f}%, "
          f"±5%内 {(chk<5).mean()*100:.1f}%")
    print(f"   数据日期: 行情 {df['时间'].dropna().max()}  |  K线 {df['末日'].dropna().max()}")

    # 快速预览
    print("\n【五】主表预览（按总市值前 8）")
    cols = [c for c in ["代码", "名称", "现价", "涨跌幅%", "PE_TTM", "PB",
                        "总市值亿", "ROE%", "距最高%", "所属板块"] if c in df.columns]
    v = df.nlargest(8, "总市值亿")[cols].copy()
    v["所属板块"] = v["所属板块"].fillna("").apply(
        lambda s: "|".join(s.split("|")[:2]))
    print(v.to_string(index=False))
print("=" * 70)
