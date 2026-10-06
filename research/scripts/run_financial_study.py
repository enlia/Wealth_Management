"""财务因子（价值/质量/成长/盈利质量）有效性检验。

复用 alphalens-reloaded 的 IC / 分组 / 换手率计算，本项目只做财务特有的处理。
数据装配与中性化（下述 1~4 的实现）在同目录 financial_factors.py；
本文件是入口与检验编排（CLI、分层抽样、逐因子检验、子区间汇总），
并重导出迁出前的原名字（build_financial_factors / neutralize / pick_sample 等）。

  1. **报告期对齐**：财务数据按 available_date（法定披露截止日）ffill 到交易日，
     保证任一天看到的都是「当时已公布」的最新财报。
  2. **去极值（winsorize）**：财务同比增速（profit_yoy/revenue_yoy）
     极端值可达 1e9%（业绩暴雷或重组），直接进 Spearman 会污染整个横截面。
     采用 MAD 法（median ± 5×1.4826×MAD）而非分位裁剪——因为分布本就重尾，
     分位裁剪会把有效信息也删掉。
  3. **行业中性化**：A 股行业轮动极强，不中性化的话「财务因子」可能只是
     「行业因子」的代理变量。做法是逐日对每个行业做横截面去均值（demean by industry）。
  4. **PB / PE 的价格对齐**：PB = 当日收盘价 / bps，必须用当日价，不能用财报期价格。

用法
----
  uv run python scripts/run_financial_study.py --n 300          # 冒烟
  uv run python scripts/run_financial_study.py --all
  uv run python scripts/run_financial_study.py --all --neutral   # 行业中性
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from factor_lab.analysis.alphalens_adapter import run_tear_sheet, subperiod_ic
from factor_lab.config import DEFAULT_COST, DEFAULT_RESEARCH, OUTPUT_DIR, is_a_share
from factor_lab.data import (  # noqa: E402
    all_codes,
    load_factor_prices,
    load_long,
    load_stock_info,
)
from factor_lab.data.universe import build_universe, summarize_universe

# 兼容重导出：build_financial_factors / neutralize 等已迁至 financial_factors.py，
# compare_synthesis.py 等 5 个脚本仍从本模块取这些名字 —— 对外名字保持不变。
from financial_factors import (  # noqa: E402,F401
    MIN_OBS_PER_DAY,
    N_JOBS,
    build_financial_factors,
    demean_by_group,
    neutralize,
    winsorize_mad,
)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 220)

PANEL = Path(__file__).resolve().parents[2] / "runtime" / "financial_panel.parquet"


def pick_sample(codes: list[str], n: int, seed: int = 42) -> list[str]:
    """从 A 股代码里抽 n 只做冒烟。

    ⚠️ 踩坑记录（2026-10-05）：早期版本直接取 `codes[:n]`，
       而 codes 是按代码排序的，bj4xxxxx（北交所旧代码段）排在最前，
       于是 n=200 抽出来的全是北交所股票 → 与财务面板交集为 0，
       且北交所 2021 年才有数据，整个检验直接空跑。
       → 改为【分层抽样】：先按市场/板块分层，各层等比例抽。

    分层依据：sh / sz / bj 三个市场。
    """
    if n <= 0 or n >= len(codes):
        return codes
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[str]] = {}
    for c in codes:
        buckets.setdefault(c[:2], []).append(c)
    # 按层大小比例分配名额，每层至少 1 只
    total = len(codes)
    out: list[str] = []
    for mkt, lst in sorted(buckets.items()):
        k = max(1, round(n * len(lst) / total))
        k = min(k, len(lst))
        idx = rng.choice(len(lst), size=k, replace=False)
        out.extend(lst[i] for i in sorted(idx))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="财务因子有效性检验")
    ap.add_argument("--n", type=int, default=300, help="股票池上限，0/--all = 全市场（分层抽样）")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--neutral", action="store_true",
                    help="行业 + 市值 双中性化")
    ap.add_argument("--factors", default="bp,ep,roe,profit_yoy,revenue_yoy,gross_margin,cf_quality")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = type(DEFAULT_RESEARCH)(**{**DEFAULT_RESEARCH.__dict__,
                                    "start_date": args.start, "end_date": args.end})
    full = args.all or args.n == 0
    names = [x.strip() for x in args.factors.split(",") if x.strip()]

    print("=" * 74)
    print("财务因子有效性检验（价值 / 质量 / 成长 / 盈利质量）")
    print("=" * 74)
    print(f"区间 {cfg.start_date} ~ {cfg.end_date}   分组 {cfg.quantiles}   "
          f"前瞻期 {cfg.periods}   往返成本 {DEFAULT_COST.round_trip*10000:.0f}bp")
    print(f"股票池 {'全市场 A 股' if full else f'前 {args.n} 只'}")
    print(f"中性化: {'行业 + 市值' if args.neutral else '不做（原始因子）'}")
    print(f"披露口径: 法定披露截止日（年报次年 4/30 / 半年报 8/31 / 三季报 10/31）")

    # ── 1. 价量数据 ────────────────────────────────────────────
    t0 = time.perf_counter()
    codes = [c for c in all_codes() if is_a_share(c)]
    if not full:
        codes = pick_sample(codes, args.n)
        print(f"（分层抽样 {len(codes)} 只，按 sh/sz/bj 比例随机，seed=42）")
    print(f"\n[1/6] 读取 {len(codes):,} 只标的长表 …")
    long = load_long(codes, start=cfg.start_date, end=cfg.end_date)
    print(f"      {len(long):,} 行，{time.perf_counter()-t0:.1f}s")

    info = load_stock_info()
    info = info[info["code"].isin(codes)]
    print("\n[2/6] 构建股票池")
    long = build_universe(long, cfg, info=info)
    summarize_universe(long)
    alive = long["code"].unique().tolist()

    print("\n[3/6] 读取价格宽表 …")
    # 🔴🔴 **必须同时取两种价格口径**（2026-10-06 实测 BLOCK）
    #   初版只取 `field="close"`（未复权），一个面板同时充当两个角色：
    #     · PB = close / bps        → 确实需要**未复权**价（bps 是财报披露的原始数字）
    #     · alphalens 前瞻收益 / IC → **必须**用前复权价
    #   ⇒ 收益端被除权缺口污染。实测全市场等权口径年化偏差：
    #       2016 −16.80pp / 2019 −10.80pp / 2021~2026 每年 −5.1 ~ −8.0pp
    #   **这个偏差大于本项目声称的任何因子收益**（最高的 ep 也只有 +6.1%）
    #   ⇒ 修复前跑出的 bp/ep/roe IC 与多空收益全部不可信。
    px_raw, px_adj = load_factor_prices(alive, start=cfg.start_date,
                                       end=cfg.end_date)
    prices_raw = px_raw
    prices = px_adj          # 收益口径：前复权
    print(f"      未复权 {px_raw.shape} / 前复权 {px_adj.shape}"
          f"（索引已校验一致），{time.perf_counter()-t0:.1f}s")

    # ── 2. 财务面板 ───────────────────────────────────────────
    print("\n[4/6] 构建财务因子日频面板 …")
    panel = pd.read_parquet(PANEL)
    panel = panel[panel["sym"].isin(alive)]
    print(f"      面板 {len(panel):,} 行，{panel['sym'].nunique():,} 只")

    # ⚠️ **交易日轴取前复权面板**（它才是收益口径的时间轴），
    #   但**PB/EP 的价格必须传未复权面板** —— bps/eps 是财报披露的原始数字，
    #   与未复权价同口径。若这里传 `prices`（前复权），
    #   PB 会被复权因子抬高，等于按「今天的股本」算历史 PB。
    trading_days = prices.index
    # 总股本（用于市值中性化）
    # shares 单位 = 亿股（已核验：mktcap / shares 精确等于股价，见
    # research/scripts/check_units.py）；市值口径为【亿元】，与
    # stock_info.mktcap 的【万元】不同口径，仅用于 log 中性化。
    shares = None
    if "shares" in info.columns:
        shares = info.set_index("code")["shares"]
    fac = build_financial_factors(panel, trading_days, prices_raw, shares=shares)
    daily = fac.pop("_daily")
    print(f"      展开为日频 {len(daily):,} 行，去极值: MAD±5")
    ind_map = daily["industry"]
    cap_map = daily["mktcap"] if "mktcap" in daily.columns else None
    if cap_map is not None:
        ok = cap_map.notna() & (cap_map > 0)
        print(f"      市值代理变量有效率 {ok.mean()*100:.1f}%（close × shares）")

    def _neut(f: pd.Series) -> pd.Series:
        ind = ind_map.reindex(f.index)
        cap = cap_map.reindex(f.index) if cap_map is not None else None
        return neutralize(f, ind.to_numpy(),
                          cap.to_numpy() if cap is not None else None)

    # ── 3. 逐因子检验 ─────────────────────────────────────────
    print(f"\n[5/6] 检验 {len(names)} 个因子: {', '.join(names)}")
    out_dir = Path(args.out) if args.out else OUTPUT_DIR / "financial"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    # ⚠️ 性能：中性化与 tear sheet 都是纯函数且无副作用，算一次即可复用。
    #    早期版本在主循环和子区间循环各算一遍 → 重复 2 倍耗时。
    prepared: dict[str, pd.Series] = {}

    def _get(name: str) -> pd.Series:
        if name not in prepared:
            f = fac[name]
            if args.neutral:
                f = _neut(f)
            prepared[name] = f
        return prepared[name]

    for name in names:
        print("\n" + "-" * 74)
        print(f"因子: {name}")
        print("-" * 74)
        try:
            t0 = time.perf_counter()
            f = _get(name)
            n = int(f.notna().sum())
            print(f"  有效值 {n:,}，覆盖 {f.index.get_level_values('asset').nunique():,} 只")
            if n < 1000:
                print("  ✗ 有效值过少，跳过")
                continue
            r = run_tear_sheet(f, prices, cfg, DEFAULT_COST, factor_name=name)
            print(f"  耗时 {time.perf_counter()-t0:.1f}s")
            print("  " + r.summary().replace("\n", "\n  "))
            results.append(r)
            r.ic_by_period.to_csv(out_dir / f"ic_{name}.csv", encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  ✗ 失败: {type(e).__name__}: {e}")
            traceback.print_exc()

    # ── 4. 子区间 + 汇总 ─────────────────────────────────────
    print("\n[6/6] 子区间稳定性检验")
    sub_tables = []
    for name in names:
        if not any(r.factor_name == name for r in results):
            continue
        print(f"\n  ── {name}")
        sub = subperiod_ic(_get(name), prices, cfg, DEFAULT_COST, name)
        print("    " + sub.to_string(index=False).replace("\n", "\n    "))
        sub.insert(0, "因子", name)
        sub_tables.append(sub)

    if results:
        print("\n" + "=" * 74)
        print("汇总（1 日前瞻）")
        print("=" * 74)
        summ = pd.DataFrame([{
            "因子": r.factor_name,
            "IC均值": r.ic_by_period.loc[1, "mean"],
            "ICIR": r.ic_by_period.loc[1, "ir"],
            "t值": r.ic_by_period.loc[1, "tstat"],
            "IC样本": int(r.ic_by_period.loc[1, "count"]),
            "多空年化毛": r.gross_spread,
            "多空年化净": r.net_spread_after_cost,
            "平均换手": r.turnover_mean,
        } for r in results if 1 in r.ic_by_period.index])
        summ = summ.sort_values("IC均值", ascending=False)
        print(summ.to_string(index=False, float_format=lambda x: f"{x:>9.4f}"))
        tag = "neutral" if args.neutral else "raw"
        summ.to_csv(out_dir / f"summary_{tag}.csv", index=False, encoding="utf-8-sig")

    if sub_tables:
        tag = "neutral" if args.neutral else "raw"
        pd.concat(sub_tables, ignore_index=True).to_csv(
            out_dir / f"subperiod_{tag}.csv", index=False, encoding="utf-8-sig")

    print(f"\n结果目录: {out_dir}")
    print("提示: 以上为统计检验，不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
