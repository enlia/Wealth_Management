"""Tushare 2000 积分全量数据补全：按「性价比」排序，分批落盘 + 断点续跑。

实测结论（2026-10-05，probe_all_interfaces.py 探测 27 个接口）
-----------------------------------------------------------
  可用 23 / 27。不可用 4 个：suspend_d（停复牌）、top_list（龙虎榜）、
  bak_daily、cyq_perf（筹码分布）。

  意外收获：文档标注高门槛的 6 个接口实际都能用
  forecast（业绩预告）、express（业绩快报）、fina_mainbz（主营构成）、
  report_rc（研报评级）、stk_holdernumber（股东人数）、pledge_stat（股权质押）。

关键优化：按「交易日」批量拉，而非按「股票」逐只拉
-----------------------------------------------------------
  实测单日全市场 daily_basic 返回 5,561 行 / 0.4 秒。
  按交易日拉 2,634 天 = 2,634 次请求；按股票拉 5,591 只 = 5,591 次。
  耗时差2 倍以上，且行数一样。

  ⚠️ 但财务类接口（fina_indicator / income / balancesheet / cashflow）
     只能按股票或报告期拉 —— 实测按报告期一次返回 5,591 行，
     80 个报告期 = 80 次请求，这反而比逐只快 70倍。

优先级排序（按「对因子检验的边际价值 / 耗时」）
-------------------------------------------------
  P0 daily_basic     日频估值/市值/换手   18 min  1,448万行  ← 最高价值
  P0 fina_indicator  财务指标全历史        1 min5万行
  P0 income          利润表              2 min
  P0 balancesheet    资产负债表            2 min
  P0 cashflow        现金流量表            2 min
  P1 stk_limit       涨跌停价格           11 min  1,475万行
  P1 moneyflow       资金流向             24 min  1,448万行
  P1 index_daily     指数日线7 min
  P1 index_weight    指数成分权重           1 min
  P2 trade_cal       交易日历             <1 min
  P2 其余（forecast / express / fina_mainbz / report_rc /
     stk_holdernumber / pledge_stat / namechange / hs_const）  合计 < 5 min

用法
----
  uv run python research/scripts/fetch_all_tushare.py --list       # 看清单
  uv run python research/scripts/fetch_all_tushare.py --task p0    # 只拉 P0
  uv run python research/scripts/fetch_all_tushare.py              # 全拉
  uv run python research/scripts/fetch_all_tushare.py --dry-run    # 只估时间
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "scripts"))
from factor_lab.config import DB_PATH, env_get, is_a_share  # noqa: E402
from fetch_tushare import call  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runtime" / "tushare"
START, END = "20151201", "20260930"
RATE = 180          # 次/分钟，留10% 余量


# ── 任务定义 ────────────────────────────────────────────────────
# kind: "by_date"  按交易日逐日  |  "by_period" 按报告期 |  "once" 一次拉完
TASKS: dict[str, dict] = {
    # ── P0 ──
    "daily_basic": {
        "kind": "by_date", "p": "P0", "desc": "每日指标(PE/PB/市值/换手/换手率)",
        "est_min": 18, "note": "本机只有单期快照，这个是日频历史，量级差异最大",
    },
    "fina_indicator": {
        "kind": "by_stock", "p": "P0", "desc": "财务指标(ROE/毛利率/负债率/增长率)",
        "est_min": 50, "note": "盈利能力/成长/质量因子的核心输入",
    },
    "income": {
        "kind": "by_stock", "p": "P0", "desc": "利润表(营收/净利/毛利)",
        "est_min": 48, "note": "构建利润质量因子：净利与经营现金流的背离",
    },
    "balancesheet": {
        "kind": "by_stock", "p": "P0", "desc": "资产负债表(资产/负债/净资产)",
        "est_min": 50, "note": "杠杆因子、偿债能力因子",
    },
    "cashflow": {
        "kind": "by_stock", "p": "P0", "desc": "现金流量表(经营/投资/筹资现金流)",
        "est_min": 49, "note": "现金流质量因子 —— 比净利润更难操纵",
    },
    # ── P1 ──
    "stk_limit": {
        "kind": "by_date", "p": "P1", "desc": "涨跌停价格",
        "est_min": 11, "note": "涨跌停因子必需；可精确判断「是否封板」",
    },
    "moneyflow": {
        "kind": "by_date", "p": "P1", "desc": "资金流向(大单/超大单净额)",
        "est_min": 24, "note": "资金流因子；注意这是交易行为数据，非基本面",
    },
    "index_daily": {
        "kind": "by_date", "p": "P1", "desc": "指数日线",
        "est_min": 7, "note": "基准净值 —— 算超额收益必需",
    },
    "index_weight": {
        "kind": "by_period_month", "p": "P1", "desc": "指数成分权重",
        "est_min": 1, "note": "沪深300/中证500 成分与权重",
    },
    # ── P2 ──
    "trade_cal": {
        "kind": "once", "p": "P2", "params": {"start_date": START, "end_date": END},
        "desc": "交易日历",
        "est_min": 0.1, "note": "所有时间对齐的基础",
    },
    "forecast": {
        "kind": "by_stock", "p": "P2", "desc": "业绩预告",
        "est_min": 40, "note": "预告净利润增速 —— 事件因子",
    },
    "express": {
        "kind": "by_period", "p": "P2", "desc": "业绩快报",
        "est_min": 2, "note": "快报与正式财报的差异本身就是信号",
    },
    "fina_mainbz": {
        "kind": "by_stock", "p": "P2", "desc": "主营构成",
        "est_min": 50, "note": "业务多元化程度；单主业公司更易形成能力预期",
    },
    "report_rc": {
        "kind": "by_period_month", "p": "P2",
        "params": {"index_code": None},
        "desc": "研报评级(全市场按月)",
        "est_min": 6,
        "note": "卖方共识因子；注意可能反向 —— 过度拥挤的预期已被price in",
    },
    "stk_holdernumber": {
        "kind": "by_period", "p": "P2", "desc": "股东人数",
        "est_min": 2, "note": "股东人数变化 = 筹码集中度，散户化程度",
    },
    "pledge_stat": {
        "kind": "by_period", "p": "P2", "desc": "股权质押",
        "est_min": 2, "note": "质押率 = 股东风险偏好，高质押股易暴跌",
    },
}

# 指数列表（index_daily / index_weight 需要）
INDEXES = [
    "000300.SH",   # 沪深300
    "000905.SH",   # 中证500
    "000852.SH",   # 中证1000
    "399006.SZ",   # 创业板指
    "000001.SH",   # 上证指数
    "399001.SZ",   # 深证成指
]


def trading_days(token: str) -> list[str]:
    """取交易日历（缓存到本地，避免重复请求）。"""
    cache = OUT / "trade_cal.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = call(token, "trade_cal",
                  {"start_date": START, "end_date": END, "is_open": "1"})
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return sorted(df["cal_date"].astype(str).tolist())


def report_periods() -> list[str]:
    """报告期列表：2015Q4 ~ 2026Q2。"""
    out = []
    for y in range(2015, 2027):
        for q, mmdd in ((1, "0331"), (2, "0630"), (3, "0930"), (4, "1231")):
            if y == 2015 and q != 4:
                continue
            if y == 2026 and (q > 2 or (q == 2 and mmdd > "0930")):
                continue
            out.append(f"{y}{mmdd}")
    return out


def months() -> list[str]:
    out = []
    for y in range(2015, 2027):
        for m in range(1, 13):
            if y == 2015 and m < 12:
                continue
            if y == 2026 and m > 9:
                continue
            out.append(f"{y}{m:02d}")
    return out


def a_share_codes() -> list[str]:
    """本机 A 股代码 → Tushare ts_code。"""
    import sqlite3
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    codes = [r[0] for r in con.execute("SELECT DISTINCT code FROM bar_daily")
             if is_a_share(r[0])]
    out = []
    for c in codes:
        c = c.lower()
        mkt, num = c[:2], c[2:]
        sfx = {"sh": "SH", "sz": "SZ", "bj": "BJ"}[mkt]
        out.append(f"{num}.{sfx}")
    return sorted(out)


def save(df: pd.DataFrame, tag: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{tag}.parquet", index=False)


class Limiter:
    """限频：RATE 次/分钟，遇接口报错自动退避。"""

    def __init__(self, rate: int = RATE) -> None:
        self.interval = 60.0 / rate
        self.last = 0.0

    def wait(self) -> None:
        dt = time.perf_counter() - self.last
        if dt < self.interval:
            time.sleep(self.interval - dt)
        self.last = time.perf_counter()


def run_task(name: str, spec: dict, token: str, lim: Limiter,
             dry: bool) -> dict:
    """执行单个任务的全量下载。"""
    tag = name
    final = OUT / f"{tag}.parquet"
    if final.exists():
        d = len(pd.read_parquet(final, columns=[list(pd.read_parquet(final).columns)[0]]))
        print(f"  ✓ {name:<16} 已完成，跳过（{d:,} 行）")
        return {"task": name, "skipped": True, "rows": d}

    kind = spec["kind"]
    print(f"\n  → {name}  {spec['desc']}")
    print(f"    {spec['note']}")
    if dry:
        print(f"    [dry-run] 预估{spec['est_min']} 分钟")
        return {"task": name, "dry": True}

    t0 = time.perf_counter()
    parts: list[pd.DataFrame] = []
    fail = 0

    try:
        if kind == "once":
            lim.wait()
            parts.append(call(token, name, spec.get("params", {})))

        elif kind == "by_date":
            days = trading_days(token)
            print(f"    {len(days):,} 个交易日，每交易日 1 次请求")
            for i, d in enumerate(days, 1):
                lim.wait()
                try:
                    df = call(token, name, {"trade_date": d}, retry=2)
                    if len(df):
                        parts.append(df)
                except Exception:                                # noqa: BLE001
                    fail += 1
                if i % 200 == 0 or i == len(days):
                    el = time.perf_counter() - t0
                    got = sum(len(p) for p in parts)
                    eta = el / i * (len(days) - i)
                    print(f"      {i:>5,}/{len(days):,} 日  {got:>12,} 行  "
                          f"已用 {el/60:.1f}m  剩余 {eta/60:.1f}m")

        elif kind == "by_period":
            periods = report_periods()
            print(f"    {len(periods)} 个报告期，每期 1 次请求")
            for i, p in enumerate(periods, 1):
                lim.wait()
                try:
                    df = call(token, name, {"period": p}, retry=2)
                    if len(df):
                        parts.append(df)
                except Exception:                                # noqa: BLE001
                    fail += 1
                if i % 20 == 0 or i == len(periods):
                    el = time.perf_counter() - t0
                    got = sum(len(p) for p in parts)
                    print(f"      {i:>3}/{len(periods)} 期  {got:>10,} 行  "
                          f"{el/60:.1f}m")

        elif kind == "by_period_month":
            ms = months()
            if name == "report_rc":
                # 研报评级：按月份区间拉全市场（不按指数）
                print(f"    {len(ms)} 个月，每月 1 次请求")
                for i, m in enumerate(ms, 1):
                    lim.wait()
                    y, mm = int(m[:4]), m[4:]
                    last = {"01": "31", "02": "28", "03": "31", "04": "30",
                            "05": "31", "06": "30", "07": "31", "08": "31",
                            "09": "30", "10": "31", "11": "30", "12": "31"}[mm]
                    try:
                        df = call(token, name,
                                  {"start_date": f"{y}{mm}01", "end_date": f"{y}{mm}{last}"},
                                  retry=2)
                        if len(df):
                            parts.append(df)
                    except Exception:                            # noqa: BLE001
                        fail += 1
                    if i % 12 == 0 or i == len(ms):
                        print(f"      {i:>3}/{len(ms)} 月  "
                              f"{sum(len(p) for p in parts):>8,} 行")
                df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                if len(df):
                    save(df, tag)
                    el = (time.perf_counter() - t0) / 60
                    print(f"    ✓ {len(df):,} 行  用时 {el:.1f} 分钟")
                return {"task": name, "rows": len(df)}

            idxs = INDEXES
            print(f"    {len(ms)} 个月 × {len(idxs)} 个指数")
            for i, m in enumerate(ms, 1):
                for ix in idxs:
                    lim.wait()
                    try:
                        df = call(token, name,
                                  {"index_code": ix, "start_date": f"{m}01",
                                   "end_date": f"{m}31"}, retry=2)
                        if len(df):
                            parts.append(df)
                    except Exception:                            # noqa: BLE001
                        fail += 1
                if i % 30 == 0 or i == len(ms):
                    print(f"      {i:>3}/{len(ms)} 月  {sum(len(p) for p in parts):>8,} 行")

        elif kind == "by_stock":
            codes = a_share_codes()
            print(f"    {len(codes):,} 只股票（低频接口，耗时最长）")
            for i, ts in enumerate(codes, 1):
                lim.wait()
                try:
                    df = call(token, name, {"ts_code": ts}, retry=2)
                    if len(df):
                        parts.append(df)
                except Exception:                                # noqa: BLE001
                    fail += 1
                if i % 300 == 0 or i == len(codes):
                    el = time.perf_counter() - t0
                    got = sum(len(p) for p in parts)
                    eta = el / i * (len(codes) - i)
                    print(f"      {i:>5,}/{len(codes):,} 只  {got:>10,} 行  "
                          f"{el/60:.1f}m  剩余 {eta/60:.1f}m")
    except KeyboardInterrupt:
        if parts:
            save(pd.concat(parts, ignore_index=True), f"{tag}_partial")
            print(f"\n    已中断，部分结果存为 {tag}_partial.parquet")
        return {"task": name, "interrupted": True, "rows": sum(len(p) for p in parts)}

    if not parts:
        print(f"    ✗ 无数据返回（失败 {fail} 次）")
        return {"task": name, "failed": True}

    df = pd.concat(parts, ignore_index=True)
    # 财务类接口同报告期可能多次覆盖，去重后保存
    before = len(df)
    if kind in ("by_period", "by_period_month") and len(df.columns) > 2:
        keys = [c for c in ("ts_code", "period", "end_date", "index_code", "trade_date")
                if c in df.columns]
        if keys:
            df = df.drop_duplicates(subset=keys, keep="last")
    save(df, tag)
    el = (time.perf_counter() - t0) / 60
    print(f"    ✓ {len(df):,} 行（去重前 {before:,}）  用时 {el:.1f} 分钟"
          f"{f'  失败 {fail} 次' if fail else ''}")
    return {"task": name, "rows": len(df), "minutes": round(el, 1), "fail": fail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="只跑某个任务名")
    ap.add_argument("--prio", default=None, help="只跑某优先级 P0/P1/P2")
    ap.add_argument("--list", action="store_true", help="列出任务清单")
    ap.add_argument("--dry-run", action="store_true", help="只估算不执行")
    args = ap.parse_args()

    if args.list:
        print("=" * 92)
        print("Tushare 补全任务清单（2000 积分，实测可用）")
        print("=" * 92)
        print(f"{'任务':<18}{'优先级':<7}{'方式':<17}{'预估':>7}  说明")
        print("-" * 92)
        total = 0
        for cur in ("P0", "P1", "P2"):
            for n, s in TASKS.items():
                if s["p"] != cur:
                    continue
                done = (OUT / f"{n}.parquet").exists()
                mark = "✓" if done else " "
                m = s["est_min"]
                total += m
                unit = "s" if m < 1 else "m"
                print(f"{mark} {n:<16}{s['p']:<7}{s['kind']:<17}{m:>6.1f}{unit}  {s['desc']}")
        print("-" * 92)
        print(f"合计预估 {total:.0f} 分钟（约 {total/60:.1f} 小时）")
        return 0

    token = env_get("TUSHARE_TOKEN")
    if not token:
        print("✗ 未配置 TUSHARE_TOKEN（在项目根 .env 中设置）")
        return 1

    todo = TASKS
    if args.task:
        if args.task not in TASKS:
            print(f"✗ 未知任务：{args.task}")
            print(f"  可用：{', '.join(TASKS)}")
            return 1
        todo = {args.task: TASKS[args.task]}
    elif args.prio:
        todo = {n: s for n, s in TASKS.items() if s["p"] == args.prio}

    print("=" * 92)
    print(f"Tushare 全量补全  {'[DRY-RUN] ' if args.dry_run else ''}"
          f"共 {len(todo)} 个任务")
    print("=" * 92)

    lim = Limiter()
    results = []
    t_all = time.perf_counter()
    for i, (name, spec) in enumerate(todo.items(), 1):
        print(f"\n[{i}/{len(todo)}] {spec['p']}")
        results.append(run_task(name, spec, token, lim, args.dry_run))

    el = (time.perf_counter() - t_all) / 60
    print("\n" + "=" * 92)
    ok = [r for r in results if r.get("rows")]
    print(f"完成{len(ok)} / {len(results)} 个任务  总用时 {el:.1f} 分钟")
    if ok:
        print(f"新增数据 {sum(r['rows'] for r in ok):,} 行")
    print("=" * 92)
    print("\n下一步：")
    print("  1. research/scripts/verify_tushare_data.py质量校验")
    print("  2. research/scripts/merge_tushare_into_db.py  并入本机库")
    print("  3. research/scripts/audit_data_quality.py     确认年化偏差 < 0.5pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
