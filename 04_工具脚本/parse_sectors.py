"""解析通达信 infoharbor_block.dat → 板块名称 + 880代码 + 成分股
格式：
  #GN_板块名,成分股数,880代码,创建日,更新日,,
  0#000408,1#600519,...          ← 成分股，格式 市场#代码
"""
import re
import pandas as pd

PATH = r"C:\SoftWare\TONGDAXIN\Body\T0002\hq_cache\infoharbor_block.dat"

with open(PATH, "rb") as f:
    text = f.read().decode("gbk", errors="ignore")

blocks = []
cur = None
for line in text.splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith("#"):
        parts = line.lstrip("#").split(",")
        if len(parts) < 4:
            continue
        cur = {
            "name": parts[0],
            "n_declared": int(parts[1]) if parts[1].isdigit() else 0,
            "code": parts[2].strip(),
            "created": parts[3].strip(),
            "updated": parts[4].strip() if len(parts) > 4 else "",
            "stocks": [],
        }
        blocks.append(cur)
    elif cur is not None:
        for tok in line.split(","):
            tok = tok.strip()
            m = re.match(r"^(\d)#(\d{6})$", tok)
            if m:
                mkt, cd = m.group(1), m.group(2)
                pre = {"0": "sz", "1": "sh", "2": "bj"}.get(mkt, "sh")
                cur["stocks"].append(pre + cd)

rows = []
for b in blocks:
    rows.append({
        "板块名": b["name"],
        "代码880": b["code"],
        "成分数": len(b["stocks"]),
        "声明数": b["n_declared"],
        "更新日期": b["updated"],
        "成分股": ",".join(b["stocks"]),
    })

df = pd.DataFrame(rows)
print(f"解析板块数: {len(df)}")
print(f"有效成分股总计: {sum(len(b['stocks']) for b in blocks):,}")
print(f"成分数为0的板块: {(df['成分数'] == 0).sum()}")
print("\n--- 按成分股数前 20 大板块 ---")
top = df.nlargest(20, "成分数")[["板块名", "代码880", "成分数", "更新日期"]]
print(top.to_string(index=False))

df[["板块名", "代码880", "成分数", "更新日期"]].to_csv(
    "sectors_index.csv", index=False, encoding="utf-8-sig")
with open("sector_members.txt", "w", encoding="utf-8") as f:
    for b in blocks:
        f.write(f"{b['code']}\t{b['name']}\t{','.join(b['stocks'])}\n")
print("\n已保存 sectors_index.csv 与 sector_members.txt")
