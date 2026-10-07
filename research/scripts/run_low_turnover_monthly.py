"""实验轮 1 · A 线（低换手月频）全流程骨架 —— 冒烟预备单交付物。

回答什么
--------
「月频调仓 + 缓冲区（跌出候选池才卖）能否把 rev5 类反转组合的年换手压小、
让多头口径净收益转正」的**可复跑骨架**：数据读入 → 因子打分 → 月频调仓（缓冲带）
→ 换手统计 → 成本扣减 → 绩效输出（P14 四要素模板）。

⚠ 本脚本只做**骨架 + 小样本冒烟**：输出数字一律标【冒烟样本·不作为结论】，
   不写入任何正式报告面。全量（滚动 7 窗、层内混淆、三段子区间、缓冲三档对比）
   留待数据线批次 1 质检门全过后按 .planning/experiment-round1-A-brief.md 执行。

流程与复用（禁重造，仓规一.1）
----------------------------
1. 数据读入   ：run_long_only.load_long_chunked（分年加载前复权（qfq）价；market.db
                只读 mode=ro；自带 int→datetime 等实测修正）；逐码覆盖清单显式打印
2. 因子打分   ：factor_lab.factors.price_volume.compute_factor +
                run_long_only._cross_z（横截面 z）+ run_long_only.combine；
                --signal-lag-months 实现简报红线「跳过最近 1 月」（信号滞后整月，
                按月切片不用 BDay——U4：BDay(n) 非真实交易日数）
3. 月频调仓   ：long_only.rebalance_days（M=每月首个交易日）+
                selection.rank_topk / select_with_buffer（缓冲带：已持仓还在
                候选池就不卖）；缓冲档映射与死参数守卫见 map_buffer
4. 换手统计   ：全股票空间权重 L1 差（引擎 simulate_matrix 同口径）；
                单边换手 = ½·Σ|Δw|；事件表带手算锚（n_enter/n_valid）
5. 成本扣减   ：**引用 config.DEFAULT_COST（CostModel 20bp 往返，不自造）**：
                期成本 = 单边换手 × round_trip（20bp/完整往返 = 买入 7.5bp +
                卖出 12.5bp）⚠ 分析层 long_only.CostModel（往返 30.2bp）与
                config 口径不同，本脚本按任务书固定 config 口径
6. 绩效输出   ：收益**从前复权（qfq）价算**（禁从因子值算）；年化几何 `_year_span`
                （自然日/365.25，与基准同一函数）；报告模板=P14 四要素列头 +
                口径三件套列头 + 多重比较声明位 + 「不构成投资建议」尾句

口径三件套（D8 判决强制列头）
----------------------------
  价格口径：前复权（qfq） close_adj ｜ 换手单位：倍/月·倍/年（单边，算式 ½·Σ|Δw|）
  年化方式：几何 _year_span（年数=自然日/365.25）；252 仅倍数换算层（铁律 7 分层）

用法（冒烟示例）
----
  python research/scripts/run_low_turnover_monthly.py \\
      --codes sh600519,sz000001,sh600036,sh600053,sh603391,sz301565,sz300799 \\
      --start 20240102 --end 20251231 --n-hold 5 --buffer-pct 0,0.1,0.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from run_long_only import _bench_stats, _cross_z, combine, load_long_chunked  # noqa: E402

from factor_lab.analysis.long_only import _year_span, rebalance_days  # noqa: E402
from factor_lab.analysis.selection import rank_topk, select_with_buffer  # noqa: E402
from factor_lab.config import DEFAULT_COST, SCALING_TRADING_DAYS  # noqa: E402
from factor_lab.factors.price_volume import compute_factor  # noqa: E402

SMOKE_LABEL = "【冒烟样本·不作为结论】"
OUTPUT = ROOT / "runtime" / "low_turnover_monthly"


# ── 1) 数据读入 ────────────────────────────────────────────────
def load_inputs(codes: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读入长表（算因子）与前复权（qfq）价格宽表（算收益）；逐码覆盖清单显式打印。

    ⚠ 覆盖清单必须打印：退市/停牌/次新可能被 SQL 的 `close_adj IS NOT NULL`
      静默剔除，不列清单就是静默失败（空结果 ≠ 真没有）。
    """
    long = load_long_chunked(list(codes), start, end, verbose=False)
    if long.empty:
        raise ValueError(f"窗口 {start}~{end} 内 {codes} 无可用行情（close_adj 全空？）"
                         f"—— 显式失败，不静默出空结果")
    price = long.pivot_table(index="date", columns="code", values="close_adj",
                             aggfunc="last").sort_index()
    for c in codes:
        sub = long[long["code"] == c]
        if sub.empty:
            print(f"  ⚠ 覆盖缺口：{c} 窗口内 0 行（退市后段/停牌/次新未上市/close_adj 缺失）")
        else:
            print(f"  · {c}: {len(sub):,} 行  {sub['date'].min().date()} ~ "
                  f"{sub['date'].max().date()}")
    return long, price


# ── 2) 因子打分（+ 简报红线「跳过最近 1 月」的信号滞后）─────────────
def score_panel(long: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """横截面 z 分数合成（多因子等权；单因子=自身）。"""
    panels: dict[str, pd.DataFrame] = {}
    for f in factors:
        s = compute_factor(f, long)
        if s is None or s.empty:
            raise ValueError(f"因子 {f} 无产出——显式失败（空结果 ≠ 真没有，先排查取数）")
        panels[f] = _cross_z(s.unstack("asset").sort_index())
    z = panels[factors[0]] if len(factors) == 1 else combine(panels, {f: 1.0 for f in factors})
    return z


def lag_by_months(z: pd.DataFrame, months: int) -> pd.DataFrame:
    """信号滞后整月（简报红线「跳过最近 1 月」：t 日用 t−months 个月的月末快照）。

    ⚠ 按**日历月**对齐，不用 BDay 近似（UNITS U4：BDay(n) 非真实交易日数）。
      months<=0 原样返回（仓内 rev5 原生口径）。
    """
    if months <= 0:
        return z
    snap = z.resample("ME").last()
    lut = {p: i for i, p in enumerate(snap.index.to_period("M"))}
    want = z.index.to_period("M") - months
    arr = snap.to_numpy()
    out = np.full((len(z), z.shape[1]), np.nan)
    for i, p in enumerate(want):
        k = lut.get(p, -1)
        if k >= 0:
            out[i] = arr[k]
    return pd.DataFrame(out, index=z.index, columns=z.columns)


# ── 3) 月频调仓（缓冲带）────────────────────────────────────────
def map_buffer(n_hold: int, buffer_pct: float) -> int:
    """缓冲档 → 候选池大小 n_pick（映射显式，简报未定义其落地算法，记录待裁）。

    采用：n_pick = n_hold + max(1, ceil(n_hold × buffer_pct))（buffer_pct=0 → 无缓冲）。
    ⚠ 死参数守卫（铁律 6）：不同档必须落不同 n_pick，否则「参数没生效」。
    """
    if buffer_pct <= 0:
        return n_hold
    return n_hold + max(1, int(np.ceil(n_hold * buffer_pct)))


def assert_distinct_pools(picks: dict, n_pool_eff: int) -> dict:
    """死参数守卫（铁律 6）：映射值与封顶后有效池都必须逐档不同。

    ⚠ 冒烟实证：n_hold=5 时 10%/20% 两档经取整同落 6 只、9 票池时 20% 档被
      票数封顶回落到 9 —— 只查映射值不查有效池会漏掉「参数没生效」。
    """
    if len(set(picks.values())) != len(picks):
        raise ValueError(f"缓冲档映射出重复 n_pick（{picks}）—— 参数没生效即死参数，"
                         f"加大 --n-hold 或拉开档距再跑（铁律：参数扫描核数值真变）")
    eff = {b: min(k, n_pool_eff) for b, k in picks.items()}
    if len(set(eff.values())) != len(eff):
        raise ValueError(f"缓冲档被股票数封顶后重复有效池（{eff}，池上限 {n_pool_eff}）"
                         f"—— 档位落到同一候选池 = 参数没生效，加股票数或减 --n-hold 再跑")
    return eff


def monthly_holdings(z: pd.DataFrame, n_hold: int, n_pick: int
                     ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """月度调仓 + 缓冲带选股：返回 (持仓列索引 (T,n_hold)-1 哨兵, 调仓日掩码, 日期)。

    语义（selection.select_with_buffer）：已持仓只要还在前 n_pick 名就不卖。
    """
    n_pick = min(n_pick, z.shape[1])
    is_rebal = np.zeros(len(z.index), dtype=bool)
    is_rebal[rebalance_days(z.index, "M")] = True
    top_idx = rank_topk(z.fillna(-np.inf).to_numpy(dtype=float), n_pick)
    held = select_with_buffer(top_idx, is_rebal, min(n_hold, n_pick))
    return held, is_rebal, z.index


def target_weights(held: np.ndarray) -> np.ndarray:
    """等权目标权重（骨架口径）：哨兵 -1 槽 → 0，其余槽均分后按行归一。

    ⚠ 哨兵只能是 -1 且必须显式压 0（引擎教训：0 是真实股票列，
      clip 抬权重会凭空建仓）。
    """
    n_hold = held.shape[1]
    w = np.where(held < 0, 0.0, 1.0 / n_hold)
    s = w.sum(axis=1, keepdims=True)
    return w / np.where(s <= 0, 1.0, s)


# ── 4/5) 换手 + 成本 + 收益模拟 ─────────────────────────────────
def simulate(price: pd.DataFrame, held: np.ndarray, w_slots: np.ndarray,
             is_rebal: np.ndarray) -> dict:
    """等权目标权重、非调仓日沿用的月频模拟；收益从前复权（qfq）价算（次期实现，防前视）。

    换手算式：单边换手 = ½·Σ|Δw|（全股票空间权重 L1 差，向哨兵列双零贡献）；
      事件表手算锚：n_valid 恒定的调仓日，单边换手 = 新进槽数 ÷ n_valid。
    成本算式（引用 config.DEFAULT_COST，不自造）：
      期成本 = 单边换手 × DEFAULT_COST.round_trip（20bp/完整往返）。
    """
    dates = price.index
    fwd = price.pct_change(fill_method=None).shift(-1)   # 收益从价格算 + 次期实现
    fwd_mat = np.nan_to_num(fwd.to_numpy(dtype=float), nan=0.0)
    T, N = fwd_mat.shape

    sentinel = N                                  # 哨兵独占垃圾列，绝不映射真实列 0
    r_ext = np.column_stack([fwd_mat, np.zeros(T)])
    W = np.zeros((T, N + 1))
    slot_pos = np.where(held < 0, sentinel, held)
    W[np.arange(T)[:, None], slot_pos] = w_slots
    first = int(np.argmax(is_rebal)) if is_rebal.any() else T
    W[:first] = 0.0

    picked = r_ext[np.arange(T)[:, None], slot_pos]      # (T, n_slot) 次期收益
    gross = np.nansum(w_slots * picked, axis=1)
    gross[:first] = 0.0

    # ⚠ 前补**全零行**而不是 W[:1]：首调仓日的建仓是一次真实买入，
    #   单边换手应 = 新进槽数÷n_valid（手算锚 1.0）；补 W[:1] 会把建仓
    #   换手静默压成 0（冒烟手算对账抓出）。与引擎 simulate_matrix 的
    #   「首日不计建仓」口径不同，本骨架**计入**并在算式里写明。
    dW = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, W.shape[1]))))
    turn_one_side = dW.sum(axis=1) / 2.0
    cost = turn_one_side * DEFAULT_COST.round_trip        # 20bp/往返（config 引用）
    net = gross - cost

    ev, prev_set = [], set()
    for t in np.where(is_rebal)[0]:
        cur = set(slot_pos[t][slot_pos[t] < N])
        n_valid = len(cur)
        n_enter = len(cur - prev_set) if prev_set else len(cur)
        w_rep = float(w_slots[t].max()) if n_valid else 0.0
        # 手算锚按定义式 ½·Σ|Δw| 分两类：
        #   建仓日（无前仓）：单向成交 ⇒ ½ × 建仓幅度 = 0.5×n_enter÷n_valid
        #   换仓日（k 进 k 出）：双边对称 ⇒ k÷n_valid
        # n_valid 在事件间发生变动（补位/哨兵增减）时另有归一化漂移项，
        # check_rel 会把它显式顶出来，不许静默抹平。
        if not n_valid:
            manual = 0.0
        elif prev_set:
            manual = n_enter / n_valid
        else:
            manual = 0.5 * n_enter / n_valid
        ev.append({"date": dates[t], "event": "build" if not prev_set else "swap",
                   "n_valid": n_valid, "n_enter": n_enter,
                   "turn_one_side": float(turn_one_side[t]),
                   "manual_one_side": manual,
                   "check_rel": (abs(turn_one_side[t] - manual) / manual) if manual else 0.0,
                   "w_each": w_rep, "cost": float(cost[t]),
                   "gross_next_day": float(gross[t])})
        prev_set = cur
    return {"dates": dates, "gross": gross, "cost": cost, "net": net,
            "turn_one_side": turn_one_side, "first": first,
            "events": pd.DataFrame(ev)}


# ── 6) 绩效输出（P14 四要素 + 口径三件套 + 多重比较位 + 免责）──────
def render_report(sim: dict, price: pd.DataFrame, factors: list[str], spec: dict,
                  attempts: int) -> str:
    dates = sim["dates"]
    net = pd.Series(sim["net"], index=dates, dtype=float)
    chain = net.iloc[sim["first"]:].replace([np.inf, -np.inf], np.nan).dropna()
    nav = float((1 + chain).prod()) if len(chain) else float("nan")
    years = _year_span(chain.index) if len(chain) >= 2 else float("nan")
    cagr = nav ** (1 / years) - 1 if nav and nav > 0 else float("nan")
    vol = float(chain.std() * np.sqrt(SCALING_TRADING_DAYS)) if len(chain) >= 2 else float("nan")
    bench = _bench_stats(price.pct_change(fill_method=None).mean(axis=1)
                         .loc[chain.index])
    turn_year = float(sim["turn_one_side"][sim["first"]:].sum()) / years
    cost_year = float(sim["cost"][sim["first"]:].sum()) / years
    y = SMOKE_LABEL
    return "\n".join([
        f"# 实验轮 1·A 线 低换手月频 —— 骨架冒烟 {y}",
        "",
        "## 口径三件套（强制列头）",
        "| 价格口径 | 换手单位 | 年化方式 |", "|---|---|---|",
        "| 前复权（qfq） close_adj | 倍/月·倍/年（单边）；算式 ½·Σ&#124;Δw&#124; | "
        "几何 _year_span（自然日/365.25）；252 仅倍数换算层 |",
        "",
        "## P14 四要素（每格均带冒烟标）",
        "| ①年化口径 | ②成本假设 | ③基准 | ④多空/多头 |", "|---|---|---|---|",
        f"| 几何 nav^(1/年数)−1，年数=_year_span {y} "
        f"| config.DEFAULT_COST：20bp/往返（买 7.5bp+卖 12.5bp），"
        f"期成本=单边换手×20bp（每档附算式）{y} "
        f"| 同池等权买持有（_bench_stats，同一 _year_span）{y} "
        f"| 单边多头（A 股散户现实口径）{y} |",
        "",
        "## 冒烟绩效",
        "| 指标 | 数值 |", "|---|---|",
        f"| 累计净值 | {nav:.6f} {y} |",
        f"| 年化净收益 | {cagr:+.4%} {y} |",
        f"| 年化波动（σ×√252 倍数层） | {vol:.4%} {y} |",
        f"| 基准年化（同池等权买持有） | {bench:+.4%} {y} |",
        f"| 超额（策略−基准） | {cagr - bench:+.4%} {y} |",
        f"| 年换手（倍/年·单边） | {turn_year:.4f} {y} |",
        f"| 年成本（年换手×20bp） | {cost_year:+.4%} {y} |",
        "",
        "换手/成本算式（手算锚）：单边换手 = ½·Σ|Δw|（n_valid 恒定的调仓日 = 新进槽数÷n_valid，"
        "见 smoke_events.csv 手算列）；期成本 = 单边换手 × 20bp。",
        f"运行规格：{spec}",
        "",
        "## 多重比较声明（留位，如实列）",
        f"本轮尝试组合数 = {attempts}（因子 {factors} × 缓冲档数）；**未做多重比较校正**；"
        f"正式结论单须用本仓自实现 Deflated Sharpe 校正（src/factor_lab/analysis/deflated.py，Bailey & López de Prado 2014 公式；mlfinlab 本机不可用）——task_plan.md 共同红线，此处留位。",
        "",
        "## 免责",
        SMOKE_LABEL + " 全部数字为小样本骨架冒烟产物，不构成投资建议。",
    ])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="实验轮 1·A 线 低换手月频骨架（冒烟预备；输出不作为结论）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", required=True,
                    help="逗号分隔的显式股票清单（冒烟 5~10 只，覆盖异质形态）")
    ap.add_argument("--start", default="20240102")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--factors", default="rev5",
                    help="逗号分隔；默认 rev5（price_volume.reversal(5)，仓内既有定义）")
    ap.add_argument("--n-hold", type=int, default=5, help="持股数（骨架等权）")
    ap.add_argument("--buffer-pct", default="0,0.1,0.2",
                    help="缓冲档（候选池扩张比），逗号多值；映射见 map_buffer")
    ap.add_argument("--signal-lag-months", type=int, default=1,
                    help="信号滞后整月数（简报红线「跳过最近 1 月」=1；0=仓内原生口径）")
    ap.add_argument("--out-dir", default=str(OUTPUT))
    args = ap.parse_args()

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    codes = [c.strip().lower() for c in args.codes.split(",") if c.strip()]
    buffers = [float(x) for x in str(args.buffer_pct).split(",") if str(x).strip()]
    if not 5 <= len(codes) <= 10:
        print(f"⚠ 股票数 {len(codes)} 不在冒烟判据 5~10 区间（AGENTS 一.3 第 1 步）")
    picks = {b: map_buffer(args.n_hold, b) for b in buffers}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{SMOKE_LABEL} 实验轮 1·A 线 低换手月频骨架（前复权（qfq）口径）")
    long, price = load_inputs(codes, args.start, args.end)
    n_pool_eff = len(set(price.columns)) if len(price.columns) else 0
    eff = assert_distinct_pools(picks, n_pool_eff)
    z = lag_by_months(score_panel(long, factors), args.signal_lag_months)
    attempts = len(factors) * len(buffers)
    reports, ev_all = [], []
    for b in buffers:
        held, is_rebal, _ = monthly_holdings(z, args.n_hold, picks[b])
        w_slots = target_weights(held)
        sim = simulate(price, held, w_slots, is_rebal)
        spec = {"codes": codes, "factors": factors, "n_hold": args.n_hold,
                "buffer_pct": b, "n_pick": picks[b], "n_pick_eff": eff[b],
                "signal_lag_months": args.signal_lag_months,
                "调仓": "每月首个交易日（rebalance_days=M）",
                "成本": f"config.DEFAULT_COST round_trip="
                        f"{DEFAULT_COST.round_trip * 1e4:.1f}bp/往返"}
        reports.append(render_report(sim, price, factors, spec, attempts))
        ev = sim["events"].copy()
        ev["buffer_pct"] = b
        ev_all.append(ev)
        print(f"> 缓冲档 {b:.0%} → n_pick={picks[b]}（数值真变守卫过）；"
              f"事件明细见 smoke_events.csv {SMOKE_LABEL}")

    (out_dir / "smoke_report.md").write_text("\n\n---\n\n".join(reports),
                                             encoding="utf-8")
    pd.concat(ev_all, ignore_index=True).to_csv(out_dir / "smoke_events.csv",
                                                index=False, encoding="utf-8-sig")
    print(f"\n✓ 产出：{out_dir / 'smoke_report.md'}、{out_dir / 'smoke_events.csv'}")
    print(SMOKE_LABEL + " 全部输出为骨架冒烟数字，不写入正式报告面；不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

