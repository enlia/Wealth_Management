"""量纲校验：市值单位必须是【万元】（2026-10-06 统一）

为什么需要这个检查
------------------
量纲错误**不报错**，只会让因子静默失真。本项目踩过两次：

1. 本机`universe.csv` 导出的市值是**亿元**
2. Tushare 的 `total_mv` / `circ_mv` 是**万元**

两者混用差1 万倍。若不校验，市值中性化的对数市值因子会完全失效，
而回测照常输出数字 —— 属于典型的静默失败。

判定原理
--------
市值 / 股本 = 股价。已知股价与股本，就能反推市值的真实量纲：

    mktcap单位是万元 → mktcap(万元) / shares(万股) = 元/股 ✓
    mktcap单位是亿元 → mktcap(亿元) / shares(亿股) = 元/股 ✓

两种量纲都能反推出正确股价，所以单看反推无法区分 ——
必须与**已知的真值**（Tushare 数据）对比才能发现量纲错误。

用法
----
  uv run python research/scripts/check_units.py
  uv run python research/scripts/check_units.py --skip-tushare# 无 Tushare 时只查本机自洽性
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from factor_lab.config import DB_PATH, MARKET_CAP_UNIT, YI_TO_WAN  # noqa: E402

# 抽查标的：沪深主板 + 创业板 + 保险 + 宁德，覆盖不同量级
PROBES = [
    ("sh600519", "600519.SH", "贵州茅台（大盘蓝筹）"),
    ("sz000001", "000001.SZ", "平安银行（深主板）"),
    ("sh601318", "601318.SH", "中国平安（保险）"),
    ("sz300750", "300750.SZ", "宁德时代（创业板）"),
]

TOL = 0.01          # 1% 容差（Tushare 与通达信都只保留有限小数）


def check_self_consistency(con: sqlite3.Connection) -> list[str]:
    """本机自洽性：mktcap / shares 能否还原出股价。"""
    msgs = []
    print("=" * 72)
    print(f"检查 1：本机量纲自洽性（期望单位 = {MARKET_CAP_UNIT}）")
    print("=" * 72)
    print(f"{'标的':<26}{'股价':>10}{'市值(万)':>18}{'股本(亿)':>12}{'反推价':>10}判定")
    print("-" * 72)

    for code, ts, name in PROBES:
        r = con.execute(
            "SELECT price, mktcap, shares FROM stock_info WHERE code=?", (code,)
        ).fetchone()
        if not r or not r[0] or not r[1] or not r[2]:
            msgs.append(f"{name} {code}：数据缺失，跳过")
            continue
        price, mktcap, shares = float(r[0]), float(r[1]), float(r[2])
        # 万元 / (亿股) → 元/股
        implied = mktcap / YI_TO_WAN / shares   # 万元 → 亿元 / 亿股 = 元/股
        err = abs(implied - price) / price
        ok = err < 0.01
        mark = "✓" if ok else "✗ 量纲错误"
        print(f"{name:<26}{price:>10,.2f}{mktcap:>18,.0f}{shares:>12,.2f}"
              f"{implied:>10,.2f}  {mark}")
        if not ok:
            msgs.append(
                f"{name} {code}：反推价 {implied:,.2f} 与股价 {price:,.2f} 差 "
                f"{err*100:.1f}% —— 量纲或数值有问题"
            )
    return msgs


def check_against_tushare(con: sqlite3.Connection) -> list[str]:
    """与 Tushare 对比：这是发现量纲错误的唯一可靠方法。"""
    msgs = []
    tpath = Path(__file__).resolve().parents[2] / "runtime" / "tushare" / "daily_basic.parquet"
    if not tpath.exists():
        print("\n跳过 Tushare 对比：daily_basic.parquet 不存在")
        return [f"未找到 {tpath.name}，跳过 Tushare 对比"]

    import pandas as pd

    print()
    print("=" * 72)
    print(f"检查 2：与 Tushare 对比（两边都应是{MARKET_CAP_UNIT}）")
    print("=" * 72)
    print(f"{'标的':<26}{'Tushare(万)':>18}{'本机(万)':>18}{'相对差':>10}判定")
    print("-" * 72)

    df = pd.read_parquet(tpath)
    # ⚠️ 取最新交易日，不能硬编码 —— 数据会滚动，写死日期会每天误报失败
    latest = df["trade_date"].astype(str).max()
    df = df[df["trade_date"].astype(str) == latest]
    print(f"（对比日期：{latest}，自动取Tushare 最新交易日）")

    for code, ts, name in PROBES:
        t = df[df["ts_code"] == ts]
        l = con.execute("SELECT mktcap FROM stock_info WHERE code=?", (code,)).fetchone()
        if not len(t) or not l or not l[0]:
            msgs.append(f"{name} {code}：Tushare 或本机数据缺失，跳过")
            continue
        tv = float(t.iloc[0]["total_mv"])
        lv = float(l[0])
        err = abs(tv - lv) / tv
        ok = err < TOL
        mark = "✓" if ok else "✗ 量纲不一致"
        print(f"{name:<26}{tv:>18,.0f}{lv:>18,.0f}{err*100:>9.4f}%  {mark}")
        if not ok:
            ratio = tv / lv if lv else float("nan")
            hint = ""
            if 0.5 < ratio < 2:
                hint = "（数值差异，可能是数据时点不同）"
            elif 5e3 < ratio < 2e4:
                hint = " → 本机是【亿元】，差 1e4 倍！"
            elif 0.5e-4 < ratio < 2e-4:
                hint = " → 本机是【万元】但值偏大 1e4 倍，疑似重复换算！"
            msgs.append(
                f"{name} {code}：Tushare {tv:,.0f} vs 本机 {lv:,.0f}，"
                f"差 {err*100:.2f}%，比值 {ratio:.4g}{hint}"
            )
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tushare", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    msgs: list[str] = check_self_consistency(con)
    if not args.skip_tushare:
        msgs += check_against_tushare(con)
    con.close()

    print()
    print("=" * 72)
    if msgs:
        print(f"✗ 发现 {len(msgs)} 个问题：")
        for m in msgs:
            print(f"  · {m}")
        print()
        print("处理方式：不要猜单位。执行下面任一方案后重跑本脚本——")
        print("  A. 若本机是亿元：UPDATE stock_info SET mktcap *= 1e4, float_mktcap *= 1e4")
        print("  B. 若 Tushare 侧需换算：在 merge脚本中统一，不要在下游各自换")
        print("=" * 72)
        return 1

    print(f"✓ 量纲校验通过，市值单位统一为 {MARKET_CAP_UNIT}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
