# -*- coding: utf-8 -*-
"""
提取【带完整日期链】的黄金分割案例
==================================
条件：日线 + 周线【同时】HH/HL 上涨趋势 → 回调 H→L → 回升触及 0.618 位
输出每个案例的：股票代码、H日期、L日期、0.618位、触及日期、结局及日期

用于生成可视化证据报告。
"""

import sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from tdx_reader import read_day, to_weekly, list_codes

MIN_DROP = 0.10
HORIZON_D = 120
K_D = 12
K_W = 4
GAP_LIMIT = 0.15


def find_swings(high, low, k):
    n = len(high)
    if n < 2 * k + 10:
        return []
    is_h = np.zeros(n, bool); is_l = np.zeros(n, bool)
    for i in range(k, n - k):
        if high[i] >= high[i - k:i + k + 1].max():
            is_h[i] = True
        if low[i] <= low[i - k:i + k + 1].min():
            is_l[i] = True
    sw = []
    for i in range(n):
        if not (is_h[i] or is_l[i]):
            continue
        kind = "H" if is_h[i] else "L"
        p = high[i] if kind == "H" else low[i]
        if sw and sw[-1][2] == kind:
            if (p > sw[-1][1]) if kind == "H" else (p < sw[-1][1]):
                sw[-1] = (i, p, kind)
        else:
            sw.append((i, p, kind))
    return sw


def uptrend_at(swings, idx):
    prior = [s for s in swings if s[0] <= idx]
    if len(prior) < 4:
        return False
    p1, p2, p3, p4 = prior[-4], prior[-3], prior[-2], prior[-1]
    if not (p1[2] == "H" and p2[2] == "L" and p3[2] == "H" and p4[2] == "L"):
        return False
    return (p3[1] > p1[1]) and (p4[1] > p2[1])


def analyze(code):
    d = read_day(code)
    if d is None or len(d) < 300:
        return []
    w = to_weekly(d)
    dh = d["high"].values.astype(float)
    dl = d["low"].values.astype(float)
    dc = d["close"].values.astype(float)
    dd = pd.to_datetime(d["date"]).values
    wh = w["high"].values.astype(float)
    wl = w["low"].values.astype(float)
    wd = pd.to_datetime(w["date"]).values

    sw_d = find_swings(dh, dl, K_D)
    sw_w = find_swings(wh, wl, K_W)
    if len(sw_d) < 5 or len(sw_w) < 5:
        return []

    rets = np.abs(np.diff(dc) / dc[:-1])
    has_gap = bool((rets > GAP_LIMIT).any())

    out = []
    for i in range(len(sw_d)):
        if i < 3 or sw_d[i][2] != "L":
            continue
        if sw_d[i-1][2] != "H" or sw_d[i-2][2] != "L" or sw_d[i-3][2] != "H":
            continue
        i_l2, L2, _ = sw_d[i]
        i_h2, H2, _ = sw_d[i-1]
        L1 = sw_d[i-2][1]; H1 = sw_d[i-3][1]
        i_l1 = sw_d[i-2][0]; i_h1 = sw_d[i-3][0]
        if H2 <= L2 or (H2 - L2) / H2 < MIN_DROP:
            continue

        d_up = (H2 > H1) and (L2 > L1)
        pos = int(np.searchsorted(wd, dd[i_l2], side="right")) - 1
        if pos < 0:
            continue
        w_up = uptrend_at(sw_w, min(pos, len(wh) - 1))
        if not (d_up and w_up):
            continue

        start = i_l2 + K_D
        if start >= len(dh) - 5:
            continue
        end = min(len(dh) - 1, start + HORIZON_D)
        rng = H2 - L2
        # 逐档记录"首次触及"（均在突破前），保证与统计脚本口径一致
        first = {}
        broke_idx = None
        for j in range(start, end + 1):
            for lv in (0.382, 0.500, 0.618, 0.786):
                if lv not in first and dh[j] >= L2 + rng * lv:
                    first[lv] = j
            if dh[j] >= H2:
                broke_idx = j
                break
        if 0.618 not in first:
            continue   # 未触及 0.618，不属于本次研究对象
        hit_idx = first[0.618]
        target = L2 + rng * 0.618

        # 收益检验：在首次触及各档位当日收盘买入，持有 40 个交易日
        rr = {}
        for lv in (0.382, 0.500, 0.618, 0.786):
            k = first.get(lv, -1)
            rr[lv] = ((dc[k + 40] / dc[k] - 1) * 100) if (k >= 0 and k + 40 < len(dc)) else np.nan

        out.append({
            "code": code,
            "ret382": rr[0.382], "ret500": rr[0.500],
            "ret618": rr[0.618], "ret786": rr[0.786],
            "hit382": 0.382 in first, "hit500": 0.500 in first,
            "hit618": True, "hit786": 0.786 in first,
            "h1_date": pd.Timestamp(dd[i_h1]).date(), "h1": H1,
            "l1_date": pd.Timestamp(dd[i_l1]).date(), "l1": L1,
            "h2_date": pd.Timestamp(dd[i_h2]).date(), "h2": H2,
            "l2_date": pd.Timestamp(dd[i_l2]).date(), "l2": L2,
            "drop_pct": round((H2 - L2) / H2 * 100, 2),
            "fib618": round(target, 3),
            "hit_date": pd.Timestamp(dd[hit_idx]).date(),
            "hit_idx": int(hit_idx),
            "l2_idx": int(i_l2),
            "broke": broke_idx is not None,
            "broke_date": pd.Timestamp(dd[broke_idx]).date() if broke_idx else None,
            "broke_idx": int(broke_idx) if broke_idx else -1,
            "max_retr_pct": round((dh[start:end+1].max() - L2) / rng * 100, 1),
            "has_gap": has_gap,
            "n_bars": len(d),
        })
    return out


def main():
    codes = list_codes()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(codes)
    codes = codes[:limit]
    print(f"扫描 {len(codes)} 只...")
    all_c = []
    for j, c in enumerate(codes, 1):
        try:
            all_c.extend(analyze(c))
        except Exception:
            pass
        if j % 1000 == 0:
            print(f"  {j} 只，案例 {len(all_c)}")
    d = pd.DataFrame(all_c)
    print(f"\n【口径统一后】触及 0.618 的案例共 {len(d)} 个")
    print(f"  其中突破前高: {int(d['broke'].sum())}  未突破(失效): {int((~d['broke']).sum())}")
    print()
    print("各档位条件成功率（与主统计脚本对齐）:")
    base = d["broke"].mean()
    print(f"  {'档位':<8}{'样本':>7}{'突破':>7}{'成功率':>9}{'较基准':>10}{'增量':>9}")
    print("  " + "-" * 52)
    print(f"  {'(基准)':<8}{len(d):>7}{int(d['broke'].sum()):>7}{base*100:>8.1f}%{'-':>10}{'-':>9}")
    prev = None
    for col, name in (("hit382", "0.382"), ("hit500", "0.500"),
                      ("hit618", "0.618"), ("hit786", "0.786")):
        sub = d[d[col]]
        if len(sub) < 5:
            continue
        r = sub["broke"].mean()
        inc = f"{(r-prev)*100:+.1f}" if prev is not None else "-"
        print(f"  {name:<8}{len(sub):>7}{int(sub['broke'].sum()):>7}"
              f"{r*100:>8.1f}%{(r-base)*100:>+9.1f}pp{inc:>9}")
        prev = r
    print()
    print("收益检验：首次触及该档位当日收盘买入，持有 40 个交易日")
    print(f"  {'档位':<8}{'样本':>7}{'收益中位数':>12}{'收益均值':>11}{'胜率':>9}")
    print("  " + "-" * 47)
    for col, name in (("ret382", "0.382"), ("ret500", "0.500"),
                      ("ret618", "0.618"), ("ret786", "0.786")):
        r = d[col].dropna()
        if len(r) < 5:
            continue
        print(f"  {name:<8}{len(r):>7}{r.median():>11.2f}%{r.mean():>10.2f}%"
              f"{(r>0).mean()*100:>8.1f}%")

    # 失效案例的反弹进度
    f = d[~d["broke"]]
    if len(f):
        print(f"\n失效案例 {len(f)} 个，其反弹进度(max_retr_pct)分布:")
        print(f"  中位数 {f['max_retr_pct'].median():.1f}%  "
              f"最小 {f['max_retr_pct'].min():.1f}%  最大 {f['max_retr_pct'].max():.1f}%")
        print(f"  其中进度 >61.8% 的占比: {(f['max_retr_pct']>61.8).mean()*100:.1f}%")

    d.to_csv("cases_dated.csv", index=False, encoding="utf-8-sig")
    print("\n已保存 cases_dated.csv")

    # 挑选代表性案例
    d["fail"] = ~d["broke"]
    # 失效组：优先选回调幅度适中(15%-35%)、无除权干扰、走势清晰的
    f = d[(d["fail"]) & (~d["has_gap"]) & (d["drop_pct"].between(15, 40))]
    s = d[(~d["fail"]) & (~d["has_gap"]) & (d["drop_pct"].between(15, 40))]
    print(f"\n候选 —— 失效组 {len(f)} 只，成功组 {len(s)} 只")

    f.sort_values("drop_pct", ascending=False).head(30).to_csv(
        "cand_fail.csv", index=False, encoding="utf-8-sig")
    s.sort_values("drop_pct", ascending=False).head(30).to_csv(
        "cand_success.csv", index=False, encoding="utf-8-sig")
    print("候选已保存: cand_fail.csv / cand_success.csv")


if __name__ == "__main__":
    main()
