"""Tushare 2000 积分全量数据下载：按「性价比」排序，分批落盘 + 断点续跑。

任务表已抽到 ``tushare_tasks.py``（本文件原本 443 行，触及 500 行强制拆分区）。
接口可用性实测、踩坑记录、优先级依据都在那个模块，本文件只管执行。

用法
----
  uv run python research/scripts/fetch_all_tushare.py --list# 看清单
  uv run python research/scripts/fetch_all_tushare.py --prio P1     # 只拉 P1
  uv run python research/scripts/fetch_all_tushare.py --task moneyflow
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
from fetch_tushare import call  # noqa: E402
from tushare_paging import fetch_paged  # noqa: E402
from tushare_tasks import END, INDEXES, START, TASKS  # noqa: E402

from factor_lab.config import DB_PATH, env_get, is_a_share  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runtime" / "tushare"
RATE = 180# 次/分钟，留 10% 余量




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


MANIFEST = OUT / "_download_manifest.json"

# Tushare 单页上限。实测：
#   namechange 首页 10,000（上限 10,000）    index_basic 首页 8,000（实际 8,000 满页）
#   repurchase/pledge_detail 首页 2,000/1,500
# ⚠️ **必须翻页**，否则拿到的是首页截断，数据是残缺的。
#   实测 new_share offset=4000 只剩 340 行 —— 说明 4,340 就是全量。
PAGE_SIZE = 2000


def load_manifest() -> dict:
    """读下载账本：记录每个任务是否**真正完成**。

    ⚠️ 不能用「文件存在」判断完成 ——
    中断时 `run_task` 会写 `{tag}_partial.parquet`，
    若下次跑只判断 `{tag}.parquet` 存在，会把半成品当完成。
    """
    if not MANIFEST.exists():
        return {}
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # 账本损坏必须显式报错，不能静默当成「全部重跑」
        raise RuntimeError(
            f"下载账本损坏：{MANIFEST}\n"
            f"  解决：删除该文件后重跑（会重新下载全部任务）"
        ) from None


def mark_done(name: str, rows: int, minutes: float) -> None:
    m = load_manifest()
    m[name] = {"rows": rows, "minutes": round(minutes, 1),
               "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def run_task(name: str, spec: dict, token: str, lim: Limiter,
             dry: bool) -> dict:
    """执行单个任务的全量下载。"""
    tag = name
    final = OUT / f"{tag}.parquet"
    manifest = load_manifest()

    if name in manifest and final.exists():
        d = manifest[name]["rows"]
        print(f"  ✓ {name:<16} 已完成，跳过（{d:,} 行）")
        return {"task": name, "skipped": True, "rows": d}

    if final.exists() and name not in manifest:
        print(f"  ⚠ {name:<16} 存在 {final.name} 但账本无记录"
              f"（可能是早期版本下载的，按已完成处理）")
        try:
            d = len(pd.read_parquet(final))
        except Exception:                                      # noqa: BLE001
            d = 0
        mark_done(name, d, 0.0)
        return {"task": name, "skipped": True, "rows": d}

    kind = spec["kind"]
    print(f"\n  → {name}  {spec['desc']}")
    print(f"    {spec['note']}")
    if dry:
        print(f"    [dry-run] 预估 {spec['est_min']} 分钟")
        return {"task": name, "dry": True}

    t0 = time.perf_counter()
    parts: list[pd.DataFrame] = []
    fail = 0

    try:
        if kind == "once":
            parts.append(fetch_paged(token, name, spec.get("params", {}), lim.wait))

        elif kind == "by_date":
            days = trading_days(token)
            print(f"    {len(days):,} 个交易日，每交易日 1 次请求")
            for i, d in enumerate(days, 1):
                lim.wait()
                try:
                    df = fetch_paged(token, name, {"trade_date": d}, lim.wait, label=d)
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
                    mark_done(name, len(df), el)
                    print(f"    ✓ {len(df):,} 行  用时 {el:.1f} 分钟")
                else:
                    print(f"    ✗ {len(ms)} 个月全部返回 0 行 —— "
                          f"参数可能已失效，需重新核实接口文档")
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

        elif kind == "by_year":
            # 按自然年分段拉。用于数据量超过翻页上限的接口：
            # 实测 disclosure_date 无参数翻页 60 页只拿到 12 万行，
            # 数据停在 2016-04，近 10 年全缺 —— 分段才是可靠做法。
            y0, y1 = int(START[:4]), int(END[:4])
            years = list(range(y0, y1 + 1))
            print(f"    {len(years)} 个自然年（{y0}~{y1}），每年 1 次请求+翻页")
            for i, y in enumerate(years, 1):
                lim.wait()
                try:
                    df = fetch_paged(token, name,
                                     {"start_date": f"{y}0101",
                                      "end_date": f"{y}1231"},
                                     lim.wait, label=str(y))
                    if len(df):
                        parts.append(df)
                except Exception as e:                              # noqa: BLE001
                    fail += 1
                    print(f"      {y} 失败：{str(e)[:80]}")
                if i % 3 == 0 or i == len(years):
                    print(f"      {i:>3}/{len(years)} 年  "
                          f"{sum(len(p) for p in parts):>9,} 行  "
                          f"{(time.perf_counter() - t0)/60:.1f}m")

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
    # 财务类接口同报告期可能多次覆盖，去重后保存。
    # ⚠️ 去重键必须包含「区分不同实体的字段」：
    #   index_weight 按 trade_date 去重会丢掉同一天其它指数的成分
    #   （实测沪深300 与中证500 权重同日返回，不带 index_code 会被合并）。
    before = len(df)
    if kind in ("by_period", "by_period_month") and len(df.columns) > 2:
        # 从最具体到最宽泛，逐个补齐可用键
        keys: list[str] = []
        for cand in ("ts_code", "index_code", "con_code", "holder_name",
                     "period", "end_date", "ann_date", "month", "trade_date"):
            if cand in df.columns:
                keys.append(cand)
        # 候选键按「区分度从高到低」排列，逐个加入直到行唯一。
        # ⚠️ 不能固定用trade_date —— index_weight 同一天有多个指数的成分，
        #   少了 index_code 会把沪深300 和中证500 合并掉（实测踩过）。
        candidates = ["ts_code", "index_code", "con_code", "holder_name",
                      "ann_date", "period", "end_date", "month", "trade_date"]
        candidates = [c for c in candidates if c in df.columns]
        keys: list[str] = []
        for c in candidates:
            keys.append(c)
            if not df.duplicated(subset=keys).any():
                break
        if keys:
            df = df.drop_duplicates(subset=keys, keep="last")
    save(df, tag)
    el = (time.perf_counter() - t0) / 60
    mark_done(name, len(df), el)
    print(f"    ✓ {len(df):,} 行（去重前 {before:,}）  用时 {el:.1f} 分钟"
          f"{f'  失败 {fail} 次' if fail else ''}")
    return {"task": name, "rows": len(df), "minutes": round(el, 1), "fail": fail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="只跑某个任务名")
    ap.add_argument("--prio", default=None, help="只跑某优先级 P1/P2/P3/P4")
    ap.add_argument("--list", action="store_true", help="列出任务清单")
    ap.add_argument("--dry-run", action="store_true", help="只估算不执行")
    args = ap.parse_args()

    if args.list:
        from tushare_tasks import EMPTY, UNAVAILABLE

        print("=" * 92)
        print("Tushare 补全任务清单（2000 积分，2026-10-06 实测）")
        print("=" * 92)
        print(f"{'':2}{'任务':<18}{'优先级':<7}{'方式':<17}{'预估':>7}  说明")
        print("-" * 92)
        # ⚠️ 优先级必须从任务表动态取，不能硬编码 ("P0","P1","P2")——
        #    实测踩过：加了 P3/P4 后这里不显示，看起来像「任务不存在」。
        manifest = load_manifest()
        total = 0.0
        for cur in sorted({s["p"] for s in TASKS.values()}):
            for n, s in TASKS.items():
                if s["p"] != cur:
                    continue
                done = n in manifest and (OUT / f"{n}.parquet").exists()
                mark = "✓" if done else " "
                m = s["est_min"]
                if not done:
                    total += m
                unit = "s" if m < 1 else "m"
                print(f"{mark} {n:<16}{s['p']:<7}{s['kind']:<17}{m:>6.1f}{unit}  {s['desc']}")
        print("-" * 92)
        print(f"待下载预估 {total:.0f} 分钟（约 {total/60:.1f} 小时）")

        print()
        print("实测不可用（2000 积分）：")
        for n, why in UNAVAILABLE.items():
            print(f"  ✗ {n:<20} {why}")
        print("接口存在但无数据：")
        for n, why in EMPTY.items():
            print(f"  ○ {n:<20} {why}")
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
