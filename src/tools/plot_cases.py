# -*- coding: utf-8 -*-
"""
为黄金分割案例生成日线 + 周线可视化证据图
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from tdx_reader import read_day, to_weekly

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#131519"; FG = "#e8e9ed"; GRID = "#2a2d35"
UP = "#e5484d"; DOWN = "#30a46c"; GOLD = "#e8b03d"; GRAY = "#8a909c"

OUT = "charts"
os.makedirs(OUT, exist_ok=True)


def candle(ax, o, h, l, c, x0=0):
    """简化为 OHLC 柱：竖线=高低，横杠=开收"""
    n = len(c)
    for i in range(n):
        x = x0 + i
        col = UP if c[i] >= o[i] else DOWN
        ax.plot([x, x], [l[i], h[i]], color=col, linewidth=0.7, zorder=2)
        ax.plot([x - 0.32, x + 0.32], [o[i], o[i]], color=col, linewidth=1.4, zorder=3)
        ax.plot([x - 0.32, x + 0.32], [c[i], c[i]], color=col, linewidth=1.4, zorder=3)


def style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=10, pad=6, loc="left")
    ax.tick_params(colors=GRAY, labelsize=7.5)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.55)


def plot_case(code, h2d, l2d, hitd, broked, status, tag, name_map):
    d = read_day(code)
    if d is None:
        return None
    w = to_weekly(d)

    d["date"] = pd.to_datetime(d["date"])
    w["date"] = pd.to_datetime(w["date"])
    h2d = pd.Timestamp(h2d); l2d = pd.Timestamp(l2d); hitd = pd.Timestamp(hitd)
    broked = pd.Timestamp(broked) if (broked and not pd.isna(broked)) else None

    # 定位索引
    i_h2 = int((d["date"] - h2d).abs().idxmin())
    i_l2 = int((d["date"] - l2d).abs().idxmin())
    i_hit = int((d["date"] - hitd).abs().idxmin())
    if broked is not None:
        i_end = int((d["date"] - broked).abs().idxmin())
    else:
        i_end = min(len(d) - 1, i_hit + 120)
    i_beg = max(0, i_h2 - 70)

    H = d["high"].values[i_h2]
    L = d["low"].values[i_l2]
    fib = L + (H - L) * 0.618

    seg = d.iloc[i_beg:i_end + 1].reset_index(drop=True)
    seg_w = w[(w["date"] >= seg["date"].iloc[0]) & (w["date"] <= seg["date"].iloc[-1])].reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6), facecolor=BG,
                             gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.28})

    for ax, data, per in ((axes[0], seg, "日线"), (axes[1], seg_w, "周线")):
        if len(data) < 5:
            continue
        data = data.reset_index(drop=True)
        candle(ax, data["open"].values, data["high"].values,
               data["low"].values, data["close"].values)
        n = len(data)
        style(ax, f"{code} {name_map.get(code,'')} · {per}")

        # 水平线
        ax.axhline(H, color=UP, linewidth=1.1, linestyle="--", alpha=0.85, zorder=1)
        ax.axhline(L, color=DOWN, linewidth=1.1, linestyle="--", alpha=0.85, zorder=1)
        ax.axhline(fib, color=GOLD, linewidth=1.3, linestyle="-.", alpha=0.95, zorder=1)

        ax.text(n * 0.004, H, f"  H={H:.2f}", color=UP, fontsize=7.5, va="bottom")
        ax.text(n * 0.004, L, f"  L={L:.2f}", color=DOWN, fontsize=7.5, va="top")
        ax.text(n * 0.004, fib, f"  0.618={fib:.2f}", color=GOLD, fontsize=7.5, va="bottom")

        # 用【日期】定位标注点（日线/周线各自索引不同）
        dd = pd.to_datetime(data["date"])

        def near(t):
            if t is None:
                return None
            k = int((dd - t).abs().idxmin())
            return k if 0 <= k < n else None

        for tgt, lab, col, is_high in ((h2d, "H", UP, True),
                                       (l2d, "L", DOWN, False),
                                       (hitd, "触及0.618", GOLD, True)):
            k = near(tgt)
            if k is None:
                continue
            y = data["high"].values[k] if is_high else data["low"].values[k]
            ax.scatter([k], [y], s=44, facecolor="none", edgecolor=col,
                       linewidth=1.7, zorder=6)
            ax.annotate(lab, (k, y), textcoords="offset points",
                        xytext=(0, 9 if is_high else -16),
                        color=col, fontsize=7.5, ha="center", zorder=7)

        # 结局标注
        if status == "fail":
            k = near(pd.Timestamp(d["date"].iloc[i_end]))
            if k is None:
                k = n - 1
            ax.scatter([k], [data["low"].values[k]], marker="v", s=72,
                       color=DOWN, zorder=7)
            ax.annotate("掉头失效", (k, data["low"].values[k]),
                        textcoords="offset points", xytext=(0, -19),
                        color=DOWN, fontsize=8.5, fontweight="bold",
                        ha="center", zorder=7)
        else:
            k = near(broked)
            if k is not None:
                ax.scatter([k], [data["high"].values[k]], marker="^", s=72,
                           color=UP, zorder=7)
                ax.annotate("突破前高", (k, data["high"].values[k]),
                            textcoords="offset points", xytext=(0, 11),
                            color=UP, fontsize=8.5, fontweight="bold",
                            ha="center", zorder=7)

    sub = f"{tag} ｜ 回调 {(H-L)/H*100:.1f}% ｜ 0.618 位 {fib:.2f} ｜ " + \
          (f"{'失效：未突破前高' if status=='fail' else '突破前高'}"
           + f"，参考日 {broked.date() if broked is not None else '—'}")
    fig.suptitle(f"{code} {name_map.get(code,'')}　{sub}", color=FG,
                 fontsize=11.5, y=0.975)
    fig.text(0.01, 0.012, "数据来源：本机通达信 .day（不复权）｜ 涨=红 跌=绿 ｜ 仅供方法论研究，不构成投资建议",
             color="#5a5f6b", fontsize=7.5)

    path = os.path.join(OUT, f"{tag}_{code}.png")
    fig.savefig(path, dpi=135, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


# 案例清单（来自 extract_cases.py 的候选筛选）
FAIL = [
    ("600026", "2022-09-22", "2022-10-13", "2022-11-03", None,  "中远海能"),
    ("603839", "2025-08-01", "2025-08-28", "2025-10-10", None,  "安正时尚"),
    ("000736", "2022-04-18", "2022-04-25", "2022-05-16", None,  "中交地产"),
    ("600802", "2025-11-17", "2025-11-24", "2025-12-10", None,  "福建水泥"),
    ("000790", "2022-12-12", "2023-02-27", "2023-06-21", None,  "华神科技"),
    ("002593", "2025-11-18", "2025-11-21", "2025-12-09", None,  "日上集团"),
]
SUCC = [
    ("002134", "2024-11-11", "2024-11-27", "2024-12-13", "2025-03-05", "天津普林"),
    ("603011", "2025-04-02", "2025-04-29", "2025-05-20", "2025-05-26", "合锻智能"),
    ("002362", "2022-12-13", "2022-12-30", "2023-01-31", "2023-02-02", "汉王科技"),
]

if __name__ == "__main__":
    name_map = {c: n for c, *_r, n in FAIL + SUCC}
    made = []
    for code, h, l, hit, brk, name in FAIL:
        p = plot_case(code, h, l, hit, brk, "fail", "失效", name_map)
        print("失效:", code, name, "->", p)
        made.append(p)
    for code, h, l, hit, brk, name in SUCC:
        p = plot_case(code, h, l, hit, brk, "succ", "对照", name_map)
        print("成功:", code, name, "->", p)
        made.append(p)
    print(f"\n共生成 {len([m for m in made if m])} 张图 -> {OUT}/")
