"""扩充因子库：把可用的非深度学习因子补到 15+ 个，并逐一打分。

背景
====
Qlib 官方基准里 LightGBM 用 158 个特征拿到 +9.01% 年化；
而本项目此前只有 7 个财务因子，其中仅 2 个稳健。差距首先在因子数量。

本脚本补充两类因子（均非深度学习）：

A. 价量自建因子（完全无前视偏差，只用当日及历史行情）
   low_vol_60    低波动      —— 文献与本项目前测均显示稳健
   amihud_20     Amihud 非流动性 = |日收益| / 成交额
   turnover_20   换手率
   amt_log20     成交额对数（大额资金关注度）
   range_20      振幅（高低价差 / 收盘）
   ret_vol_ratio 波动率比（短期波动 / 长期波动，识别异常放量）
   skew_60       收益偏度（彩票型股票在 A 股被系统性高估）

B. 通达信 boards 标签（⚠️ 有快照偏差，见下）
   high_div      FG_高股息股
   broken_net    FG_破净资产
   low_pe        FG_低市盈率
   quality_tag   FG_绩优股
   north_hold    FG_北上重仓（北向资金重仓）
   fund_holdFG_基金重仓

   ⚠️⚠️ 前视偏差警告（必读）
   boards 字段是【2026-09-30 当前快照】，没有历史版本。
   直接用它回测2016 年 = 用了 2026 年才知道的信息，会高估效果。
   本脚本的处理：
     1. 只选【逻辑上跨期稳定】的标签（高股息/破净/绩优等），
        排除时变标签（最近异动/昨日涨停/近期新高等）
     2. 明确标注为「快照偏差」证据等级，不与无偏因子同等看待
     3. 结论中必须带上此警告

用法
----
  uv run python scripts/expand_factor_library.py --n 600# 冒烟
  uv run python scripts/expand_factor_library.py --all       # 全市场
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from factor_lab.analysis.scorecard import (build_scorecard, print_scorecard)
from factor_lab.config import DEFAULT_COST, DEFAULT_RESEARCH, OUTPUT_DIR, is_a_share
from factor_lab.data import all_codes, load_long, load_prices, load_stock_info
from factor_lab.data.universe import build_universe
from run_financial_study import (build_financial_factors, neutralize,
                                 pick_sample)

# ════════════════════════════════════════════════════════════════
# A. 价量自建因子（无前视偏差）
# ════════════════════════════════════════════════════════════════
def build_pv_factors(long: pd.DataFrame) -> dict[str, pd.Series]:
    """从长表行情构造价量因子。全部只用当日及历史数据，无前视偏差。

    ⚠️ 实现要点（踩坑记录）
       对已按 (code, date) 建索引的 Series 再 `groupby("code").rolling(...)`，
       pandas 会产生**三层索引** (code, code, date)，因为分组键与原索引层重复。
       后果：后续与其他两层 Series 相减时抛
       "Index._join_level on non-unique index is not implemented"。
       正解：所有 rolling 结果统一用 `_to_da()` 转成 (date, code) 的 DataFrame，
       再按同一列对齐做算术，全程不用 Series 自动对齐。
    """
    g = long.sort_values(["code", "date"])
    codes = g["code"].to_numpy()
    dates = pd.DatetimeIndex(g["date"].to_numpy())
    idx = pd.MultiIndex.from_arrays([dates, codes], names=["date", "asset"])

    col = {k: pd.Series(g[k].to_numpy(dtype=float), index=idx)
           for k in ("open", "high", "low", "close", "amount", "vol")}
    ret = col["close"].groupby(level="asset", group_keys=False).pct_change()

    def R(s: pd.Series, w: int, mp: int | None = None, how="mean"):
        """按 asset 分组的rolling，输出【同索引】的 Series。

        ⚠️ groupby(level="asset").rolling() 会把分组键再插一层，
        产出 (asset, asset, date) 三层索引 → 后续算术会报
        "The name asset occurs multiple times, use a level number"。
        正解：算完立刻 drop 掉重复层，回到 (date, asset)。
        """
        r = s.groupby(level="asset", group_keys=False).rolling(
            w, min_periods=mp or max(w // 3, 2))
        out = getattr(r, how)() if how in ("mean", "std", "max", "min", "skew") \
            else r.mean()
        if isinstance(out.index, pd.MultiIndex) and out.index.nlevels > 2:
            out = out.droplevel(0)
        return out.reindex(idx)

    def ops(a, b):
        """同一索引上安全做四则运算（b 为 0 时置 NaN）。"""
        return a / b.replace(0, np.nan) if not isinstance(b, int) else a + b

    close, amt = col["close"], col["amount"] / 1e8
    vol = col["vol"]
    out: dict[str, pd.Series] = {}

    # ── 低波 / 反转类 ──
    vol60 = R(ret, 60, 30, "std")
    out["low_vol_60"] = -vol60
    out["low_vol_120"] = -R(ret, 120, 40, "std")
    out["rev_20"] = -(close / close.groupby(level="asset").shift(20) - 1)
    out["max_ret_20"] = -R(ret, 20, 10, "max")
    # 特质波动：剔除市场收益后的残差波动
    resid = ret - ret.groupby(level="date").transform("mean")
    out["idio_vol_20"] = -R(resid, 20, 10, "std")

    # ── 流动性 / 关注度 ──
    out["amihud_20"] = R(ret.abs() / amt, 20, 10)
    out["turnover_20"] = np.log1p(R(vol, 20, 10))
    out["amt_log20"] = R(np.log1p(amt), 20, 10)
    out["volume_shock"] = -(R(vol, 20, 10) / R(vol, 60, 20))

    # ── 形态 / 趋势质量 ──
    out["range_20"] = R((col["high"] - col["low"]) / close, 20, 10)
    out["vol_ratio_20_60"] = R(ret, 20, 10, "std") / vol60
    out["skew_60"] = R(ret, 60, 30, "skew")
    out["ma_bias_20"] = -(close / R(close, 20, 10) - 1)

    rmax = R(close, 250, 100, "max")
    rmin = R(close, 250, 100, "min")
    out["price_pos250"] = -((close - rmin) / (rmax - rmin))
    out["drawdown_60"] = close / R(close, 60, 20, "max")
    out["illiq_accel"] = R(amt.groupby(level="asset", group_keys=False).diff(), 20, 10)

    # ── 统一清洗 ──
    res = {}
    for k, v in out.items():
        s = v.replace([np.inf, -np.inf], np.nan).dropna()
        s.index = s.index.set_names(["date", "asset"])
        # ⚠️ 坑：`s[s.index.duplicated(keep='last')]` 在【没有重复】时
        #    返回空 Series（mask 全 False），不是「原样返回」。
        #    曾因此把所有因子变成 len=0 且不报错。
        #    正解：先判断有无重复，无重复就不动。
        if s.index.duplicated().any():
            s = s[~s.index.duplicated(keep="last")]
        res[k] = s
    return res
    """从长表行情构造价量因子。全部只用当日及历史数据，无前视偏差。

    分组说明：
      低波/ 反转类（文献与前测显示 A 股稳健）
        low_vol_60负60 日波动率
        low_vol_120负 120 日波动率
        id_vol_20   20 日特质波动（残差波动）
        max_ret_20  过去 20 日最大单日收益（彩票效应，A 股被系统性高估）
        rev_5跳过最近 5 日的短期反转
      流动性/ 关注度类
        amihud_20   Amihud 非流动性
        turnover_20 换手率对数
        amt_log20   成交额对数
        volume_shock 量比：20 日均量 / 60 日均量
      形态 / 趋势质量类
        range_20    20 日振幅
        vol_ratio_20_60 波动率比
        skew_60     60 日收益偏度
        ma_bias_20  收盘价对 20 日均线的偏离
        price_pos250 250 日价格分位（长周期位置）
        drawdown_60 当前距 60 日高点的回撤
        illiq_accel 成交额的二阶动量（量能加速）
    """
    g = long.sort_values(["code", "date"]).set_index(["code", "date"])
    out: dict[str, pd.Series] = {}

    ret = g["close"].groupby("code", group_keys=False).pct_change()
    close = g["close"]

    def roll(s, w, mp=None, fn=None):
        r = s.groupby("code", group_keys=False).rolling(w, min_periods=mp or max(w // 3, 2))
        return (r.apply(fn, raw=True) if fn else r.mean())

    # ── 低波/ 反转类 ──
    vol60 = ret.groupby("code", group_keys=False).rolling(60, min_periods=30).std()
    out["low_vol_60"] = -vol60
    out["low_vol_120"] = -ret.groupby(
        "code", group_keys=False).rolling(120, min_periods=40).std()

    # 短期反转：跳过最近 5 日，取 -5~20 日收益的负值
    out["rev_5skip5"] = -(close / close.groupby(
        "code", group_keys=False).shift(20) - 1)

    # 过去 20 日最大单日收益（彩票型特征，A 股中被系统性高估 → 取负）
    out["max_ret_20"] = -ret.groupby("code", group_keys=False).rolling(
        20, min_periods=10).max()

    # 特质波动：残差波动率（对市场收益回归后的残差 std）
    mkt_ret = ret.groupby(level="date").transform("mean")
    resid = ret - mkt_ret
    out["idio_vol_20"] = -resid.groupby("code", group_keys=False).rolling(
        20, min_periods=10).std()

    # ── 流动性 / 关注度 ──
    amt_yi = g["amount"] / 1e8
    out["amihud_20"] = (ret.abs() / amt_yi.replace(0, np.nan)).groupby(
        "code", group_keys=False).rolling(20, min_periods=10).mean()
    out["turnover_20"] = np.log1p(roll(g["vol"], 20, 10))
    out["amt_log20"] = roll(np.log1p(amt_yi), 20, 10)

    # 量比：20 日均量 / 60 日均量（放量往往是情绪高点 → 取负）
    v20 = roll(g["vol"], 20, 10)
    v60 = g["vol"].groupby("code", group_keys=False).rolling(60, min_periods=20).mean()
    out["volume_shock"] = -(v20 / v60.replace(0, np.nan))

    # ── 形态 / 趋势质量 ──
    out["range_20"] = roll((g["high"] - g["low"]) / g["close"].replace(0, np.nan), 20, 10)
    out["vol_ratio_20_60"] = ret.groupby(
        "code", group_keys=False).rolling(20, min_periods=10).std() / vol60.replace(0, np.nan)
    out["skew_60"] = ret.groupby("code", group_keys=False).rolling(
        60, min_periods=30).skew()

    # 均线偏离：收盘对 20 日均线的偏离（乖离率）
    # ⚠️ 坑：close 是 (code, date) 两层，rolling 结果也是两层但顺序/名称
    #    可能不一致 → 直接相减会触发 join 报
    #    "The name code occurs multiple times, use a level number"。
    #    正解：统一用 level 编号 0 取第一层后再算。
    c1 = close.droplevel(0)                    # 变成 (date, code)
    ma20 = roll(close, 20, 10).droplevel(0)
    out["ma_bias_20"] = -(c1 / ma20.replace(0, np.nan) - 1)

    # 250 日价格分位（长周期位置，高位跑输 → 取负）
    roll_max = close.groupby("code", group_keys=False).rolling(
        250, min_periods=100).max().droplevel(0)
    roll_min = close.groupby("code", group_keys=False).rolling(
        250, min_periods=100).min().droplevel(0)
    out["price_pos250"] = -(c1 - roll_min) / (
        (roll_max - roll_min).replace(0, np.nan))

    # 距 60 日高点回撤（新高附近动能弱 → 取负）
    hi60 = close.groupby("code", group_keys=False).rolling(
        60, min_periods=20).max().droplevel(0)
    out["drawdown_60"] = c1 / hi60.replace(0, np.nan)

    # 成交额二阶动量（量能加速度）
    d_amt = amt_yi.groupby("code", group_keys=False).diff()
    out["illiq_accel"] = d_amt.groupby("code", group_keys=False).rolling(
        20, min_periods=10).mean()

    # 统一成 (date, asset) 索引
    res = {}
    for k, v in out.items():
        s = v.droplevel(0)
        s = s.replace([np.inf, -np.inf], np.nan)
        s.index = s.index.set_names(["date", "asset"])
        res[k] = s.dropna()
    return res


# ════════════════════════════════════════════════════════════════
# B. 通达信 boards 标签因子（⚠️ 快照偏差）
# ════════════════════════════════════════════════════════════════
# 只选逻辑上跨期稳定的标签；排除时变标签
# ⚠️ 实测发现：这些标签几乎全部恰好 200 只 → 通达信每类只存 Top200 名单，
#    覆盖率仅约 3.9%（200/5185），横向信息量非常有限，且是当前快照（带前视偏差）。
#    因此本项目把它们降级为【辅助因子】，主力是价量自建的连续因子。
BOARD_TAGS = {
    "high_div": "FG_高股息股",
    "broken_net": "FG_破净资产",
    "low_pe_tag": "FG_低市盈率",
    "low_pb_tag": "FG_低市净率",
    "quality_tag": "FG_绩优股",
    "north_hold": "FG_北上重仓",
    "fund_hold": "FG_基金重仓",
    "no_dividend": "FG_久不分红",
    "loss_tag": "FG_亏损股",
    "micro_cap_tag": "FG_微盘股",
    "large_cap_tag": "FG_大盘股",
    "margin_low": "FG_小盘非融",
}
# ⚠️ 明确排除的时变标签（回测会严重前视偏差）
EXCLUDED_TAGS = [
    "FG_最近异动", "FG_最近情绪", "FG_昨日涨停", "FG_昨日振荡", "FG_昨日上榜",
    "FG_最近多板", "FG_近期新高", "FG_近期新低", "FG_近期弱势", "FG_次日新低",
    "FG_活跃股", "FG_不活跃股", "FG_通达信热", "FG_今日涨停",
]


def build_board_factors(stock_info: pd.DataFrame,
                        prices: pd.DataFrame,
                        alive: list[str]) -> dict[str, pd.Series]:
    """把 boards 标签展开成因子。

    标签是【当前快照】，因此这里生成的是「这只股票现在属于该类」的静态标记，
    沿用到全历史。这是【已知偏差】，会在结果解读时明确标注。
    """
    si = stock_info[stock_info["code"].isin(alive)].copy()
    si["boards"] = si["boards"].fillna("").astype(str)
    tag_sets = {c: set(t.split("|")) for c, t in si["boards"].items()}
    trading_days = prices.index          # DatetimeIndex
    codes = [c for c in prices.columns if c in tag_sets]
    if not codes:
        return {}

    out = {}
    for name, tag in BOARD_TAGS.items():
        mask = np.array([tag in tag_sets[c] for c in codes], dtype=float)
        n_hit = int(mask.sum())
        # 门槛：标签成分股至少 20 只，且占池子 2% 以上，否则横截面信息太少
        if n_hit < 20 or n_hit < 0.02 * len(codes):
            print(f"    跳过 {name:<14}标签 {tag:<16}"
                  f"命中 {n_hit:>4} 只（{n_hit/max(len(codes),1)*100:.1f}%）")
            continue
        # 二值因子（是=1，否=0）。截面标准化在合成阶段做
        df = pd.DataFrame(
            np.tile(mask, (len(trading_days), 1)),
            index=trading_days, columns=codes,
        )
        s = df.stack(future_stack=True)
        s = s[s > 0]
        s.index = s.index.set_names(["date", "asset"])
        out[name] = s
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="扩充因子库 + 打分卡")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-board", action="store_true",
                    help="跳过 boards 标签（完全无偏模式）")
    args = ap.parse_args()

    cfg = type(DEFAULT_RESEARCH)(**{**DEFAULT_RESEARCH.__dict__,
                                    "start_date": args.start,
                                    "end_date": args.end})
    full = args.all or args.n == 0

    print("=" * 100)
    print("扩充因子库 + 统一打分卡")
    print("=" * 100)
    print(f"区间 {cfg.start_date} ~ {cfg.end_date}   成本 {DEFAULT_COST.round_trip*10000:.0f}bp")
    print(f"股票池 {'全市场 A 股' if full else f'{args.n} 只（分层抽样）'}")
    print("中性化: 行业+市值（逐日截面 OLS 残差）")
    if not args.skip_board:
        print("⚠️ boards 标签因子含【当前快照偏差】，见文件头警告")

    t0 = time.perf_counter()
    codes = [c for c in all_codes() if is_a_share(c)]
    if not full:
        codes = pick_sample(codes, args.n)
    print(f"\n[1/6] 读取 {len(codes):,} 只行情 …")
    long = load_long(codes, start=cfg.start_date, end=cfg.end_date)
    info = load_stock_info()
    info = info[info["code"].isin(codes)]
    long = build_universe(long, cfg, info=info, verbose=False)
    alive = long["code"].unique().tolist()
    prices = load_prices(alive, start=cfg.start_date, end=cfg.end_date, field="close")
    print(f"      {len(alive):,} 只 × {prices.shape[0]:,} 日，{time.perf_counter()-t0:.1f}s")

    # ── 财务因子 ──
    print("\n[2/6] 财务因子 …")
    panel = pd.read_parquet(
        Path(__file__).parent.parent / "out" / "financial_panel.parquet")
    panel = panel[panel["sym"].isin(alive)]
    shares = info.set_index("code")["shares"] if "shares" in info.columns else None
    fac = build_financial_factors(panel, prices.index, prices, shares=shares)
    daily = fac.pop("_daily")
    ind_map, cap_map = daily["industry"], daily["mktcap"]

    # ── 价量因子 ──
    print("\n[3/6] 价量因子（无前视偏差）…")
    pv = build_pv_factors(long)
    print(f"      {len(pv)} 个: {', '.join(pv)}")

    # ── boards 标签因子 ──
    bf = {}
    if not args.skip_board:
        print("\n[4/6] 通达信 boards 标签因子（⚠️ 快照偏差）…")
        bf = build_board_factors(info, prices, alive)
        print(f"      {len(bf)} 个: {', '.join(bf)}")
    else:
        print("\n[4/6] 跳过 boards 标签")

    # ── 合并 & 打分 ──
    print("\n[5/6] 逐因子打分（中性化 + 三段检验）…")
    rows = []
    t0 = time.perf_counter()

    def _score(name: str, s: pd.Series, note: str = "") -> None:
        try:
            if len(s) < 1000:
                rows.append({"name": name, "grade": "❌",
                             "reason": f"有效值仅 {len(s):,}"})
                return
            sn = neutralize(s, ind_map.reindex(s.index).to_numpy(),
                            cap_map.reindex(s.index).to_numpy())
            r = build_scorecard(name, sn, prices, DEFAULT_COST.round_trip,
                                cfg.quantiles)
            if note:
                r["reason"] = f"[{note}] {r['reason']}"
            rows.append(r)
            g = r.get("grade", "?")
            print(f"    {name:<20} {g}IC={r.get('ic', float('nan')):+.4f} "
                  f"ICIR={r.get('icir', float('nan')):.3f} "
                  f"净={r.get('annual_net', float('nan'))*100:+.2f}%")
        except Exception as e:  # noqa: BLE001
            rows.append({"name": name, "grade": "❌", "reason": f"失败 {e}"})

    for n, s in pv.items():
        _score(n, s, note="价量")
    for n, s in fac.items():
        if n == "_daily":
            continue
        _score(n, s, note="财务")
    for n, s in bf.items():
        _score(n, s, note="快照偏差")

    print(f"\n      打分耗时 {time.perf_counter()-t0:.1f}s")

    # ── 输出 ──
    out_dir = Path(args.out) if args.out else OUTPUT_DIR / "scorecard"
    out_dir.mkdir(parents=True, exist_ok=True)
    print_scorecard(rows, f"因子打分卡（{len(rows)} 个因子）")

    flat = []
    for r in rows:
        if "ic" not in r:
            flat.append({"因子": r["name"], "等级": r["grade"], "理由": r["reason"]})
            continue
        d = {"因子": r["name"], "等级": r["grade"], "IC": r["ic"], "ICIR": r["icir"],
             "t": r["t"], "年化毛": r["annual_gross"], "年化净": r["annual_net"],
             "换手": r["turnover"], "单调": r["monotonic"]}
        for i, (lab, _) in enumerate(SCORE_PERIODS.items()):
            d[f"IC_{lab}"] = r["seg_ic"][i] if i < len(r["seg_ic"]) else np.nan
            d[f"净_{lab}"] = r["seg_net"][i] if i < len(r["seg_net"]) else np.nan
        d["理由"] = r["reason"]
        flat.append(d)
    pd.DataFrame(flat).to_csv(out_dir / "scorecard.csv", index=False,
                              encoding="utf-8-sig")

    ok = [r for r in rows if r.get("grade") == "✅"]
    ok.sort(key=lambda x: -x["icir"])
    print(f"\n[6/6] 可用因子（✅ 三段同号且 ICIR≥0.15）共 {len(ok)} 个，"
          f"按 ICIR 降序：")
    for r in ok:
        print(f"    {r['name']:<22} ICIR={r['icir']:.3f}  净={r['annual_net']*100:+.2f}%")

    print(f"\n结果目录: {out_dir}")
    print("⚠️ boards 标签因子含当前快照偏差，解读时务必扣除。")
    print("提示: 多空组合口径，A 股散户无法做空。统计检验不构成投资建议。")
    return 0


SCORE_PERIODS = {
    "2016-2018": ("2016-01-01", "2018-12-31"),
    "2019-2022": ("2019-01-01", "2022-12-31"),
    "2023-2026": ("2023-01-01", "2026-09-30"),
}

if __name__ == "__main__":
    raise SystemExit(main())