"""解析通达信 base.dbf（证券信息库）——含代码、名称、拼音、行业等
DBF 结构：32字节头 + N×32字节字段描述 + 数据区（记录长固定，以 0x0D 结束）
"""
import struct
import pandas as pd

PATH = r"C:\SoftWare\TONGDAXIN\Body\T0002\hq_cache\base.dbf"

with open(PATH, "rb") as f:
    raw = f.read()

ver = raw[0]
y, m, d = raw[1], raw[2], raw[3]
n_rec = struct.unpack("<I", raw[4:8])[0]
hdr_len = struct.unpack("<H", raw[8:10])[0]
rec_len = struct.unpack("<H", raw[10:12])[0]
print(f"DBF 版本 0x{ver:02X}  更新日期 {y:04d}-{m:02d}-{d:02d}")
print(f"记录数 {n_rec}  头长度 {hdr_len}  记录长度 {rec_len}")

# 字段描述
fields = []
pos = 32
while raw[pos] != 0x0D:
    fd = raw[pos:pos + 32]
    name = fd[:11].split(b"\x00")[0].decode("gbk", errors="ignore").strip()
    ftype = chr(fd[11])
    flen = fd[16]
    fdec = fd[17]
    fields.append((name, ftype, flen, fdec))
    pos += 32
print(f"字段数 {len(fields)}")
print("字段列表:", [f[0] for f in fields])

# 读记录
recs = []
off = hdr_len
for i in range(n_rec):
    chunk = raw[off:off + rec_len]
    if not chunk or chunk[0:1] == b"\x1a":
        break
    p = 1
    row = {}
    for (name, ftype, flen, fdec) in fields:
        val = chunk[p:p + flen]
        s = val.decode("gbk", errors="ignore").strip()
        if ftype in "NF" and s:
            try:
                s = float(s) if fdec else int(s)
            except ValueError:
                pass
        row[name] = s
        p += flen
    recs.append(row)
    off += rec_len

df = pd.DataFrame(recs)
print(f"\n成功解析 {len(df)} 条记录")
print("\n前 5 条:")
print(df.head(5).to_string())
df.to_csv("base_dbf.csv", index=False, encoding="utf-8-sig")
print("\n已保存 base_dbf.csv")
