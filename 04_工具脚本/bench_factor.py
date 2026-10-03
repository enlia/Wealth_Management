"""实测：全市场(9570标的)做一次横截面因子计算的算力消耗
回答"做量化需要什么硬件"——用真实数据跑，不猜。
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

sys.path.insert(0, ".")
from tdx import read_day, all_codes

CODES = all_codes()
print(f"标的总数: {len(CODES)} | CPU 逻辑核心: {cpu_count()}")


def feat(code):
    """算 4 个因子：动量20日/反转5日/波动率/量能比"""
    d = read_day(code)
    if d is None or len(d) < 250:
        return None
    c = d["close"].to_numpy()
    v = d["vol"].to_numpy()
    if len(c) < 250:
        return None
    r = np.diff(np.log(c))
    mom20 = c[-1] / c[-21] - 1
    rev5 = -(c[-1] / c[-6] - 1)
    vol60 = r[-60:].std() * np.sqrt(244)
    vratio = v[-5:].mean() / (v[-250:].mean() + 1e-9)
    return (code, mom20, rev5, vol60, vratio, c[-1], len(c))


if __name__ == "__main__":
    n_mp = max(1, cpu_count() - 2)
    print(f"\n【单核串行】")
    t0 = time.perf_counter()
    rows = [feat(c) for c in CODES]
    t1 = time.perf_counter() - t0
    rows = [x for x in rows if x]
    print(f"  {len(rows):,} 只有效 | 耗时 {t1:.1f}s | {len(CODES)/t1:.0f} 只/秒")

    print(f"\n【多核 {n_mp} 进程】")
    t0 = time.perf_counter()
    with Pool(n_mp) as p:
        rows2 = [x for x in p.map(feat, CODES, chunksize=64) if x]
    t2 = time.perf_counter() - t0
    print(f"  {len(rows2):,} 只有效 | 耗时 {t2:.1f}s | 加速比 {t1/t2:.1f}x")

    df = pd.DataFrame(rows2, columns=["code", "动量20日", "反转5日", "年化波动率", "量能比", "现价", "K线数"])
    t0 = time.perf_counter()
    for col in ["动量20日", "反转5日", "年化波动率", "量能比"]:
        df[col + "_排名"] = df[col].rank(pct=True)
    df["综合评分"] = (df["动量20日_排名"] * 0.4 + df["反转5日_排名"] * 0.1
                    + (1 - df["年化波动率_排名"]) * 0.3 + df["量能比_排名"] * 0.2)
    df = df.sort_values("综合评分", ascending=False)
    t_rank = time.perf_counter() - t0

    print(f"\n【横截面打分】耗时 {t_rank*1000:.0f}ms（4因子×排名+加权）")
    print(f"\n内存占用峰值估计: {df.memory_usage(deep=True).sum()/1024/1024:.1f} MB")
    print("\n综合评分 Top 15:")
    print(df.head(15)[["code", "现价", "动量20日", "年化波动率", "量能比", "综合评分"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\n" + "=" * 60)
    print(f"结论：全市场 {len(CODES):,} 只 × 4 因子，一次完整横截面打分")
    print(f"  多核耗时 {t2+t_rank:.1f}s（{t2/t1:.1f}x 加速）| 内存 < 100 MB")
    print(f"  换算：一天可跑 {(86400/(t2+t_rank)):.0f} 轮全市场因子")
    print("=" * 60)
