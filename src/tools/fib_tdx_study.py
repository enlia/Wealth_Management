# -*- coding: utf-8 -*-
"""
【严格条件】黄金分割 0.618 大样本实证研究
==========================================
数据来源：本机通达信 .day 文件（C:\\SoftWare\\TONGDAXIN\\Body\\vipdoc）
数据范围：2021-08 ~ 2026-02，5168 只个股
注意：本地 .day 为不复权数据，已做除权跳空检测与稳健性检验

研究问题（用户指定）：
  在【日线 + 周线同时处于上涨趋势(HH/HL)】的前提下，
  出现回调波段 H→L，价格随后从低点 L 回升并到达 0.618 位，
  统计其中"继续涨上去(突破前高H)"与"掉头失效"的比例。

关键设计：
  · 趋势判定用道氏 HH/HL：H2>H1 且 L2>L1（高点点位与低点点位同步抬升）
  · 日线、周线【同时】判定为上涨趋势才纳入样本
  · 对照组：0.382 / 0.500 / 0.618 / 0.786 四档 + 基准
  · 稳健性：剔除存在异常跳空(疑似除权)的个股后重算
"""

import sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from tdx_reader import read_day, to_weekly, list_codes

MIN_DROP = 0.10        # 回调幅度下限
HORIZON_D = 120        # 日线观察窗口（交易日）
LEVELS = [("0.382", 0.382), ("0.500", 0.500), ("0.618", 0.618), ("0.786", 0.786)]
GAP_LIMIT = 0.15       # 单日跳空 >15% 视为疑似除权


def find_swings(high, low, k):
    """识别摆动高低点，返回 [(idx, price, 'H'/'L')]"""
    n = len(high)
    if n < 2 * k + 10:
        return []
    is_h = np.zeros(n, bool)
    is_l = np.zeros(n, bool)
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
    """
    判定在 idx 处是否处于上涨趋势(HH/HL)。
    取 <= idx 的最后 4 个摆动点，要求形态为 H1,L1,H2,L2
    且 H2>H1 且 L2>L1。
    """
    prior = [s for s in swings if s[0] <= idx]
    if len(prior) < 4:
        return False
    p1, p2, p3, p4 = prior[-4], prior[-3], prior[-2], prior[-1]
    if not (p1[2] == "H" and p2[2] == "L" and p3[2] == "H" and p4[2] == "L"):
        return False
    H1, L1, H2, L2 = p1[1], p2[1], p3[1], p4[1]
    return (H2 > H1) and (L2 > L1)


def analyze(code, k_daily=12, k_weekly=4):
    """分析单只股票，返回案例列表"""
    d = read_day(code)
    if d is None or len(d) < 300:
        return []
    w = to_weekly(d)

    dh = d["high"].values.astype(float)
    dl = d["low"].values.astype(float)
    dc = d["close"].values.astype(float)
    wh = w["high"].values.astype(float)
    wl = w["low"].values.astype(float)

    sw_d = find_swings(dh, dl, k_daily)
    sw_w = find_swings(wh, wl, k_weekly)
    if len(sw_d) < 5 or len(sw_w) < 5:
        return []

    # 除权跳空检测
    rets = np.abs(np.diff(dc) / dc[:-1])
    has_gap = bool((rets > GAP_LIMIT).any())

    # 日线 -> 周线索引映射（按日期）
    w_dates = pd.to_datetime(w["date"]).values
    d_dates = pd.to_datetime(d["date"]).values

    cases = []
    for i in range(len(sw_d)):
        # 需要 prior 至少 4 个点：当前 i 是 L，前一个是 H，再前是 L，再前是 H
        if sw_d[i][2] != "L" or i < 3:
            continue
        if sw_d[i - 1][2] != "H" or sw_d[i - 2][2] != "L" or sw_d[i - 3][2] != "H":
            continue

        i_l2, L2, _ = sw_d[i]
        i_h2, H2, _ = sw_d[i - 1]
        L1 = sw_d[i - 2][1]
        H1 = sw_d[i - 3][1]

        if H2 <= L2:
            continue
        drop = (H2 - L2) / H2
        if drop < MIN_DROP:
            continue

        # === 双周期上涨趋势判定 ===
        d_up = (H2 > H1) and (L2 > L1)
        # 周线：找到 L2 对应日期在周线中的位置
        dt = d_dates[i_l2]
        pos = int(np.searchsorted(w_dates, dt, side="right")) - 1
        if pos < 0:
            continue
        pos = min(pos, len(wh) - 1)
        w_up = uptrend_at(sw_w, pos)

        if not (d_up and w_up):
            continue   # ← 只保留双周期共振上涨趋势的样本

        # === 观察从 L2 的回升 ===
        # 【修正前视偏差】摆动低点 L2 需 k_daily 根K线后才能被确认，
        # 真实交易里只能从"确认点"之后开始观察和行动。
        start = i_l2 + k_daily
        if start >= len(dh) - 5:
            continue
        end = min(len(dh) - 1, start + HORIZON_D)
        rng = H2 - L2
        first_hit = {}
        broke = False
        for j in range(start, end + 1):
            hj = dh[j]
            if hj >= H2:
                broke = True
                break
            for name, lv in LEVELS:
                if name not in first_hit and hj >= L2 + rng * lv:
                    first_hit[name] = j
        if end <= start:
            continue

        row = {"code": code, "H": H2, "L": L2, "drop": drop,
               "broke_H": broke, "has_gap": has_gap,
               "max_retr": (dh[start:end + 1].max() - L2) / rng}
        for name, _ in LEVELS:
            row["hit_" + name] = name in first_hit
        # 触及档位后买入、持有 40 交易日的收益
        for name, _ in LEVELS:
            j = first_hit.get(name, -1)
            if j >= 0 and j + 40 < len(dc):
                row["ret_" + name] = (dc[j + 40] / dc[j] - 1) * 100
            else:
                row["ret_" + name] = np.nan
        cases.append(row)
    return cases


def report(d, title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    base = d["broke_H"].mean()
    print(f"  双周期上涨趋势下的回调波段样本: {len(d)}")
    print(f"  无条件突破前高(基准): {base*100:.1f}%\n")
    print(f"{'档位':<9}{'触及样本':>9}{'突破数':>8}{'成功率':>9}{'较基准':>10}{'失效(掉头)':>11}")
    print("-" * 72)
    prev = None
    res = {}
    for name, _ in LEVELS:
        sub = d[d["hit_" + name]]
        if len(sub) < 10:
            continue
        r = sub["broke_H"].mean()
        res[name] = r
        inc = f"{(r-prev)*100:+.1f}" if prev is not None else "-"
        print(f"{name:<9}{len(sub):>9}{int(sub['broke_H'].sum()):>8}"
              f"{r*100:>8.1f}%{(r-base)*100:>+9.1f}pp{(1-r)*100:>10.1f}%{inc:>8}")
        prev = r
    print("\n  相邻档位增量（看 0.618 是否跳变）:")
    ks = list(res.keys())
    for a, b in zip(ks[:-1], ks[1:]):
        mark = "   <<<< 关键" if b == "0.618" else ""
        print(f"    {a} -> {b}: {(res[b]-res[a])*100:+.1f}pp{mark}")
    return base, res


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    codes = list_codes()
    if limit:
        codes = codes[:limit]
    print(f"本地通达信样本: {len(codes)} 只 | 日线 2021-08~2026-02 | 不复权")
    print(f"筛选条件: 日线+周线【同时】HH/HL 上涨趋势，回调幅度 >= {MIN_DROP*100:.0f}%\n")

    all_cases = []
    for j, c in enumerate(codes, 1):
        try:
            all_cases.extend(analyze(c))
        except Exception:
            pass
        if j % 400 == 0:
            print(f"  {j}/{len(codes)} 只，符合条件样本 {len(all_cases)}")

    print(f"\n共提取【双周期上涨趋势 + 回调】样本 {len(all_cases)} 个")
    if len(all_cases) < 60:
        print("样本不足")
        return
    d = pd.DataFrame(all_cases)

    report(d, "【主检验】双周期上涨趋势下，回升到各档位后的结局")

    # --- 用户最关心：0.618 档的失效详情 ---
    s = d[d["hit_0.618"]]
    if len(s) >= 20:
        print("\n" + "=" * 72)
        print("【重点】到达 0.618 位后的结局分解")
        print("=" * 72)
        ok = int(s["broke_H"].sum())
        fail = len(s) - ok
        print(f"  到达 0.618 位样本: {len(s)}")
        print(f"    继续涨上去(突破前高): {ok}  ({ok/len(s)*100:.1f}%)")
        print(f"    掉头失效          : {fail}  ({fail/len(s)*100:.1f}%)")
        f = s[~s["broke_H"]]
        if len(f):
            print(f"\n  失效样本的反弹进度中位数: {f['max_retr'].median()*100:.1f}%")
            print("    （>61.8% 说明多数是冲过黄金分割后才掉头，")
            print("      而非在 61.8% 处获得支撑后企稳）")

    # --- 收益检验 ---
    print("\n" + "=" * 72)
    print("【收益检验】触及档位当日收盘买入，持有 40 个交易日")
    print("=" * 72)
    print(f"{'档位':<9}{'样本':>8}{'收益中位数':>12}{'收益均值':>11}{'胜率':>9}")
    print("-" * 72)
    for name, _ in LEVELS:
        col = "ret_" + name
        if col not in d.columns:
            continue
        r = d[col].dropna()
        if len(r) < 10:
            continue
        print(f"{name:<9}{len(r):>8}{r.median():>11.2f}%{r.mean():>10.2f}%"
              f"{(r>0).mean()*100:>8.1f}%")

    # --- 稳健性：剔除疑似除权个股 ---
    clean = d[~d["has_gap"]]
    if len(clean) >= 60:
        print("\n" + "=" * 72)
        print("【稳健性检验】剔除存在异常跳空(疑似除权)的个股后")
        print("=" * 72)
        report(clean, f"  剔除后样本 {len(clean)}（主样本 {len(d)}）")

    d.to_csv("fib_tdx_cases.csv", index=False, encoding="utf-8-sig")
    print("\n明细: fib_tdx_cases.csv")


if __name__ == "__main__":
    main()
