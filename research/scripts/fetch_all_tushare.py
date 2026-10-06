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
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "scripts"))
from fetch_tushare import call  # noqa: E402
from tushare_paging import fetch_paged  # noqa: E402
from tushare_paths import (  # noqa: E402
    OUT,
    a_share_codes,
    months,
    report_periods,
    save,
    trading_days,
)
from tushare_state import Limiter, load_manifest, mark_done  # noqa: E402
from tushare_tasks import END, INDEXES, START, TASKS  # noqa: E402

from factor_lab.config import env_get  # noqa: E402

# 各表的业务主键（白名单，**不是「加到唯一为止」**）。
# ⚠️ 为什么不用「逐个加候选键直到行唯一」：
#   那个策略天然会**主动丢弃区分字段**来追求唯一。
#   实测踩过：财务三表 4 个 report_type × 2 个 update_flag = 8 个合法版本，
#   候选键里没有 report_type/update_flag，全部被压成 1 行。
#   正确做法是先确定该表的**业务主键**，键用尽仍不唯一时抛错。
BUSINESS_KEYS: dict[str, list[str]] = {
    # 财务三表：主体 + 报告期 + 公告日 + 报表类型 + 是否更新公告
    "income":            ["ts_code", "end_date", "ann_date",
                          "report_type", "update_flag"],
    "balancesheet":      ["ts_code", "end_date", "ann_date",
                          "report_type", "update_flag"],
    "cashflow":          ["ts_code", "end_date", "ann_date",
                          "report_type", "update_flag"],
    "fina_indicator":    ["ts_code", "end_date", "ann_date"],
    "forecast":          ["ts_code", "ann_date", "end_date"],
    "express":           ["ts_code", "ann_date", "end_date"],
    "fina_mainbz":       ["ts_code", "end_date", "ann_date", "type"],
    "top10_holders":     ["ts_code", "end_date", "holder_name", "ann_date"],
    "top10_floatholders": ["ts_code", "end_date", "holder_name", "ann_date"],
    "stk_holdernumber":  ["ts_code", "end_date", "ann_date"],
    "pledge_stat":       ["ts_code", "end_date"],
    # ⚠️ dividend 主键必须含 **div_proc**（2026-10-06 实测踩过）：
    #   分红是**多阶段流程**，同一报告期会有多条记录：
    #   预案 → 股东大会通过 → 实施，各阶段ann_date 不同。
    #   实测 9 种阶段：股东大会通过 118,458 / 预案 77,983 /
    #   实施 57,381 / 预披露 1,176 / 股东提议 210 ...
    #   原主键 [ts_code, end_date] 把它们全判成重复 ——
    #   实测 255,573 行里186,890 行涉及重复（73%），全是误报。
    #
    #   ⚠️ 但**光加 div_proc 还不够**：实测 000001.SZ 20080630 有两条
    #   都是「实施」、div_proc 相同，只有 ann_date 不同
    #   （20080926 与 20081016）⇒ 那是**修正公告**，
    #   修正幅度本身是信息（业绩超预期因子里要用），不能去重。
    "dividend":          ["ts_code", "end_date", "div_proc", "ann_date"],
    # ⚠️ share_float 主键必须含 **holder_name**（2026-10-06 实测）：
    #   同一 (ts_code, ann_date, float_date) 会有**多个股东**的解禁记录 ——
    #   实测 000878.SZ 20260311/20310317 有两条：
    #     中国铝业集团有限公司 → 公开增发一般股份
    #     中国铜业有限公司     → 定增股份
    #   它们是**两条合法记录**（不同股东的限售股），
    #   原键 [ts_code, float_date, ann_date] 把 4,251 行里的 3,257 行判成重复。
    #   实测加 holder_name 后重复归 0。
    #
    #   ⚠️ 另注：`float_date` 是**计划解禁日期**，实测含未来日期
    #   （最远 20330711）⇒ **不可用于判时间覆盖**，
    #   门禁的 DATE_COL必须用 ann_date。
    "share_float":       ["ts_code", "float_date", "ann_date",
                          "holder_name"],
    # 指数权重：必须带 index_code，否则同日的沪深300/中证500 会被合并
    "index_weight":      ["index_code", "con_code", "trade_date"],
    "report_rc":         ["ts_code", "ann_date", "end_date", "org_name"],
}

# 业务主键用尽后仍不唯一时的兜底：只加这些「补充区分列」
EXTRA_DISAMBIGUATORS = ["f_ann_date", "comp_type", "end_type", "holder_type"]


def dedup_by_business_key(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """按业务主键去重；主键用尽仍不唯一时抛错，不静默丢数据。"""
    keys = [k for k in BUSINESS_KEYS.get(name, []) if k in df.columns]
    if not keys:
        # 没有配置主键的表：只去整行重复，不做业务去重
        return df

    if df.duplicated(subset=keys).any():
        # 补充区分列（能救几个是几个）
        for c in EXTRA_DISAMBIGUATORS:
            if c in df.columns and c not in keys:
                keys.append(c)
                if not df.duplicated(subset=keys).any():
                    break

    remain = int(df.duplicated(subset=keys).sum())
    if remain:
        raise RuntimeError(
            f"{name} 按业务主键 {keys} 去重后仍有 {remain:,} 行重复。\n"
            f"  说明该表还有未识别的版本维度。**禁止静默 drop_duplicates** ——\n"
            f"  实测踩过：财务三表缺 report_type/update_flag 时，\n"
            f"  8 个合法报表版本被压成 1 行（见 DATA_SOURCE.md S8）。\n"
            f"  解决：把缺失的区分列加到 BUSINESS_KEYS['{name}']"
        )
    return df.drop_duplicates(subset=keys, keep="last")


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
            #
            # ⚠️ start_year 缺省用 START 的年份，但**存量数据**要显式往前推：
            #   实测 namechange 无参 13,889 行中 7,517 行是 1990~2014 的历史更名，
            #   若从 START(20151201) 的2015 开始分段，这 15 年全丢。
            y0 = int(spec.get("start_year", START[:4]))
            y1 = int(END[:4])
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
    # ⚠️ 先做**完全行**去重，再做主键去重。
    #    逐只拉的任务（by_stock）在中断重跑时会拿到重复分片，
    #    实测 fina_indicator 有 140,297 行完全重复（占 33%，
    #    2,411 只股票各出现 100 次而正常只该有约 42 个报告期）。
    #    整行相同说明是同一次请求被重复拼接，不是数据本身有多个版本。
    n_raw = len(df)
    df = df.drop_duplicates()
    n_exact_dup = n_raw - len(df)
    if n_exact_dup:
        print(f"    去完全重复行 -{n_exact_dup:,}（{n_exact_dup/n_raw*100:.1f}%）")

    # 财务类接口同报告期可能多次覆盖，去重后保存。
    # ⚠️⚠️ 主键必须包含**版本维度**，否则会把合法的多版本报表压成 1 行。
    #   财务三表的版本维度是 update_flag（0=原始披露 / 1=更新公告）
    #   与 report_type（1~4 合并/母公司报表），
    #   实测 4 report_type × 2 update_flag = 8 个合法版本，
    #   缺键时会被压成 1 行（见 DATA_SOURCE.md S8）。
    #   index_weight 的区分维度是 index_code ——
    #   同一 trade_date 有沪深300 与中证500 两套成分，缺 index_code 会合并。
    before = len(df)
    if kind in ("by_period", "by_period_month") and len(df.columns) > 2:
        df = dedup_by_business_key(df, name)
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
