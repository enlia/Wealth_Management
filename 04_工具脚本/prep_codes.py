"""生成本地通达信全部代码清单（腾讯格式），供批量下载行情
关键：前缀必须按 .day 文件【实际所在目录】决定，不能按代码首位。
     因为 sh000001=上证指数，sz000001=平安银行，代码相同但市场不同。
"""
import os
import glob

TDX = r"C:\SoftWare\TONGDAXIN\Body\vipdoc"

all_items = []      # (prefix, code)
sector_items = []   # (prefix, code)  880/881 板块指数

for m in ("sh", "sz", "bj"):
    d = os.path.join(TDX, m, "lday")
    for f in sorted(glob.glob(os.path.join(d, "*.day"))):
        base = os.path.basename(f)[:-4]     # 如 sh600519
        code = base[2:]
        if code.startswith("88"):
            sector_items.append((m, code))
        else:
            all_items.append((m, code))

print(f"非板块标的: {len(all_items)}   板块指数: {len(sector_items)}   合计 {len(all_items)+len(sector_items)}")

with open("codes_all.txt", "w") as f:
    f.write("\n".join(m + c for m, c in all_items) + "\n")
with open("codes_sector.txt", "w") as f:
    f.write("\n".join(m + c for m, c in sector_items) + "\n")
print("已生成 codes_all.txt / codes_sector.txt")

# 验证代码冲突情况
from collections import Counter
dup = [c for c, n in Counter(c for _, c in all_items).items() if n > 1]
print(f"存在 {len(dup)} 个代码在多个市场出现（需靠前缀区分），例: {dup[:6]}")
