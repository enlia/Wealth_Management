"""CSV vs SQLite 真实性能实测
用同一批真实 K 线数据（取本地通达信 300 只标的），
分别走 CSV 和 SQLite 两条路，对比体积/写入/查询/筛选速度。
"""
import os
import sys
import time
import sqlite3

sys.path.insert(0, ".")
from tdx import read_day, all_codes

N = 300
CSV = "bench_data.csv"
DB = "bench_data.db"
if os.path.exists(DB):
    os.remove(DB)

codes = all_codes()[:N]

# ── 读取源数据 ────────────────────────────────────────────
t0 = time.perf_counter()
rows = []
for c in codes:
    d = read_day(c)
    if d is None:
        continue
    mkt = "sh" if (c[0] == "6" or c.startswith("88")) else ("sz" if c[0] in "03" else "bj")
    sym = mkt + c
    for r in d.itertuples(index=False):
        rows.append((sym, r.date.year, r.date.month, r.date.day, r.date.strftime("%Y%m%d"),
                     round(r.open, 2), round(r.high, 2), round(r.low, 2), round(r.close, 2),
                     float(r.amount), float(r.vol)))
t_read = time.perf_counter() - t0
print(f"源数据: {len(codes)} 只标的, {len(rows):,} 条日线, 解析耗时 {t_read:.2f}s")

# ── 方案A: CSV ────────────────────────────────────────────
t0 = time.perf_counter()
with open(CSV, "w", encoding="utf-8") as f:
    f.write("code,y,m,d,ymd,open,high,low,close,amount,vol\n")
    f.writelines(f"{a},{b},{c},{d_},{e},{f_},{g},{h},{i},{j},{k}\n"
                 for a, b, c, d_, e, f_, g, h, i, j, k in rows)
t_csv_write = time.perf_counter() - t0
csv_mb = os.path.getsize(CSV) / 1024 / 1024

import pandas as pd
t0 = time.perf_counter()
df_csv = pd.read_csv(CSV, dtype={"code": str})
t_csv_read = time.perf_counter() - t0
mem_csv = df_csv.memory_usage(deep=True).sum() / 1024 / 1024

t0 = time.perf_counter()
r1 = df_csv[(df_csv.ymd >= 20250101) & (df_csv.ymd <= 20250930) & (df_csv.close > 20)]
t_csv_q1 = time.perf_counter() - t0

t0 = time.perf_counter()
g = df_csv.groupby("code")["close"].agg(["mean", "std", "max"])
t_csv_q2 = time.perf_counter() - t0
print(f"  CSV  写入 {t_csv_write:.2f}s | 读入 {t_csv_read:.2f}s | "
      f"筛选 {t_csv_q1*1000:.0f}ms | 分组聚合 {t_csv_q2*1000:.0f}ms")
print(f"       体积 {csv_mb:.1f} MB | 内存 {mem_csv:.1f} MB | 命中 {len(r1)} 行")

# ── 方案B: SQLite ─────────────────────────────────────────
con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS bar(
    code TEXT NOT NULL, y INT, m INT, d INT, ymd INT,
    open REAL, high REAL, low REAL, close REAL, amount REAL, vol REAL)""")
t0 = time.perf_counter()
cur.executemany("INSERT INTO bar VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
con.commit()
t_db_write = time.perf_counter() - t0
raw_mb = os.path.getsize(DB) / 1024 / 1024

t0 = time.perf_counter()
cur.execute("CREATE INDEX idx_code ON bar(code)")
cur.execute("CREATE INDEX idx_ymd ON bar(ymd)")
con.commit()
t_idx = time.perf_counter() - t0
db_mb = os.path.getsize(DB) / 1024 / 1024

t0 = time.perf_counter()
n1 = cur.execute("SELECT COUNT(*) FROM bar WHERE ymd>=20250101 AND ymd<=20250930 AND close>20").fetchone()[0]
t_db_q1 = time.perf_counter() - t0

t0 = time.perf_counter()
g2 = cur.execute("SELECT code, AVG(close), MAX(close) FROM bar GROUP BY code").fetchall()
t_db_q2 = time.perf_counter() - t0

t0 = time.perf_counter()
one = cur.execute("SELECT * FROM bar WHERE code=? ORDER BY ymd", ("sh600519",)).fetchall()
t_db_q3 = time.perf_counter() - t0
print(f"  SQL  写入 {t_db_write:.2f}s | 建索引 {t_idx:.2f}s | "
      f"筛选 {t_db_q1*1000:.0f}ms | 分组聚合 {t_db_q2*1000:.0f}ms | 单股取全史 {t_db_q3*1000:.0f}ms")
print(f"       体积(无索引) {raw_mb:.1f} MB → (有索引) {db_mb:.1f} MB | 命中 {n1} 行")

# ── 全量外推 ──────────────────────────────────────────────
print("\n" + "=" * 66)
print("全市场外推（16,456,556 条日线）")
print("=" * 66)
scale = 16_456_556 / len(rows)
print(f"  CSV   预计体积 {csv_mb*scale:>7.0f} MB | 写入 {t_csv_write*scale/60:>6.1f} 分钟")
print(f"  SQL   预计体积 {db_mb*scale:>7.0f} MB | 写入+索引 {(t_db_write+t_idx)*scale/60:>6.1f} 分钟")
print(f"  内存  加载全量 DataFrame 需 {mem_csv*scale/1024:.1f} GB  ← 本机仅 13.9GB")
con.close()
os.remove(CSV)
os.remove(DB)
