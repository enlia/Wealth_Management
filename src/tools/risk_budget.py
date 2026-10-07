"""风险预算建议卡 CLI —— 「有了 p/R 之后的守门员与风控」单一入口。

用法示例
--------
::

    # 1) 看帮助
    .venv\\Scripts\\python.exe src/tools/risk_budget.py --help

    # 2) 单票建议卡（ATR 口径，R 由 2×ATR 止损 / 4×ATR 目标 推出 = 2.0）
    .venv\\Scripts\\python.exe src/tools/risk_budget.py \\
        --codes sh600519 --as-of 2024-08-30 --p sh600519=0.45 \\
        --p-source "示例假设值，非实测" --r-source "ATR 倍数口径 R=4/2=2"

    # 3) 多票 + 行业集中度（已有持仓从额度里扣掉）
    .venv\\Scripts\\python.exe src/tools/risk_budget.py \\
        --codes sh600519,sz000858 --as-of 2026-09-30 \\
        --p sh600519=0.45,sz000858=0.42 --existing sh600519=0.01

本模块**不产出 p 与 R**
-----------------------
本项目 11 个价量因子扣 20bp 后多空净收益为正 **0/11**，当前无可用选股信号。
`--p` 是必填项，`--p-source` 亦必填（来源写不清就不出卡）——
单元测试里可以用「明示为示例的假设值」，卡片上会标 ⚠️。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接 `python src/tools/risk_budget.py` 运行（与 src/tools 下其他工具同款）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_lab.analysis.risk_budget import (  # noqa: E402
    DEFAULT_MAX_INDUSTRY,
    DEFAULT_MAX_POSITION,
    DEFAULT_MIN_POSITION,
    DEFAULT_SHRINKAGE,
    Candidate,
    IndustryMap,
    RiskParams,
    apply_portfolio_caps,
    build_card,
    build_levels,
    evaluate_ev,
    load_atr,
    render_card,
)
from factor_lab.analysis.risk_budget.industry import industry_caps, IndustryUsage  # noqa: E402
from factor_lab.config import DEFAULT_COST, WORKSPACE  # noqa: E402


def _parse_code_values(raw: str | None, flag: str) -> dict[str, float]:
    """解析 ``sh600519=0.45,sz000858=0.42`` → dict。缺等号/非数字一律抛错。"""
    if raw is None:
        return {}
    out: dict[str, float] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            raise ValueError(f"{flag} 里有空项：{raw!r}")
        if "=" not in item:
            raise ValueError(
                f"{flag} 的项必须写成 code=value，得到 {item!r}（整串：{raw!r}）"
            )
        code, val = item.split("=", 1)
        code = code.strip()
        if not code:
            raise ValueError(f"{flag} 的代码为空：{item!r}")
        try:
            out[code] = float(val)
        except ValueError as e:
            raise ValueError(f"{flag} 的 {code} 值无法解析为数字：{val!r}") from e
    return out


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="risk_budget.py",
        description=(
            "风险预算建议卡：期望值闸门(EV=p·R−(1−p)−c) + 分数凯利仓位 + "
            "单票/单行业/总仓位三道硬上限 + ATR 或百分比止损目标位。"
            "⚠️ 本模块不产出 p 与 R，必须由调用方给出并写明来源。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "量纲：价格为元/股（未复权报价）；仓位为占账户总本金百分比；\n"
            "      EV 为「每 1 元投入」比率，占账户口径 = EV × 仓位。\n"
            "全部输出不构成投资建议。"
        ),
    )
    ap.add_argument("--codes", required=True,
                    help="标的代码，逗号分隔，带市场前缀（如 sh600519,sz000858）")
    ap.add_argument("--as-of", required=True,
                    help="决策基准日 YYYY-MM-DD；ATR 与现价只用该日及之前数据（防前视）")

    g = ap.add_argument_group("p / R 来源（本模块不产出，必须写清来源）")
    g.add_argument("--p", required=True,
                   help="胜率 p，格式 code=值，逗号分隔；例 sh600519=0.45")
    g.add_argument("--p-source", required=True,
                   help="p 的来源描述（如『示例假设值，非实测』或『XXX 回测 n=... 』）")
    g.add_argument("--r", type=float, default=None,
                   help="赔率 R（>1）。给 R 则目标距离=止损距离×R，与 --target-atr 互斥")
    g.add_argument("--r-source", default=None,
                   help="R 的来源描述；不给则按 ATR 倍数自动生成")
    g.add_argument("--sample-n", type=int, default=1,
                   help="p/R 所依据的样本量（如回测交易笔数）。<380 时卡片会提示不足")

    g = ap.add_argument_group("止损/目标口径（ATR 与百分比二选一）")
    g.add_argument("--atr-window", type=int, default=14, help="ATR 窗口（交易日），默认 14")
    g.add_argument("--atr-method", choices=("wilder", "sma"), default="wilder",
                   help="ATR 口径，默认 wilder（与通达信/多数软件一致）")
    g.add_argument("--stop-atr", type=float, default=2.0, help="止损 ATR 倍数，默认 2.0")
    g.add_argument("--target-atr", type=float, default=None,
                   help="目标 ATR 倍数，默认 4.0（不给 --r 时生效；与 --r 互斥）")
    g.add_argument("--stop-pct", type=float, default=None,
                   help="止损百分比小数（0.08=8%%）；给则改走百分比口径")
    g.add_argument("--target-pct", type=float, default=None,
                   help="目标百分比小数（0.16=16%%）；百分比口径必须与 --stop-pct 同时给")
    g.add_argument("--entry-fraction", type=float, default=0.01,
                   help="买点区间半宽（相对现价），默认 0.01 = ±1%%")

    g = ap.add_argument_group("仓位与集中度上限（占账户总本金）")
    g.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE,
                   help=f"分数凯利收缩系数 (0,1]，默认 {DEFAULT_SHRINKAGE}"
                        f"（Kelly 假设 IID，股票收益非 IID）")
    g.add_argument("--max-position", type=float, default=DEFAULT_MAX_POSITION,
                   help=f"单票上限，默认 {DEFAULT_MAX_POSITION}（2%%）")
    g.add_argument("--max-industry", type=float, default=DEFAULT_MAX_INDUSTRY,
                   help=f"单申万一级行业合计上限，默认 {DEFAULT_MAX_INDUSTRY}（20%%）")
    g.add_argument("--max-total", type=float, default=1.0,
                   help="总仓位上限，默认 1.0（100%%）")
    g.add_argument("--min-position", type=float, default=DEFAULT_MIN_POSITION,
                   help=f"低于此仓位即丢弃，默认 {DEFAULT_MIN_POSITION}")
    g.add_argument("--existing", default=None,
                   help="已有持仓 code=比例，逗号分隔；其行业占用会从行业额度扣掉")

    g = ap.add_argument_group("成本（默认复用 factor_lab.config.DEFAULT_COST）")
    g.add_argument("--cost-bp", type=float, default=None,
                   help="手动指定单次往返成本（bp，1bp=0.01%%）；不给则用 CostModel.round_trip")
    g.add_argument("--commission", type=float, default=None, help="佣金率，如 0.00025")
    g.add_argument("--stamp-duty", type=float, default=None, help="卖出印花税率，如 0.0005")
    g.add_argument("--slippage", type=float, default=None, help="单边滑点，如 0.0005")

    ap.add_argument("--json", action="store_true", help="额外输出 JSON（便于程序消费）")
    ap.add_argument("--industry-file", default=None,
                    help="申万映射 parquet 路径；不给则自动定位 runtime/tushare/index_member_all.parquet")
    return ap


def _round_trip(args) -> tuple[float, str]:
    """返回 (c, 口径说明)。c = 往返成本 ÷ 投入金额（无量纲）。"""
    if args.cost_bp is not None:
        if args.cost_bp <= 0:
            raise ValueError(f"--cost-bp 必须 > 0，得到 {args.cost_bp}")
        c = args.cost_bp / 1e4
        return c, f"手动指定往返 {args.cost_bp:g}bp = {c:.6f}（每 1 元投入）"
    if any(v is not None for v in (args.commission, args.stamp_duty, args.slippage)):
        from factor_lab.config import CostModel
        cm = CostModel(
            commission_rate=args.commission if args.commission is not None
            else DEFAULT_COST.commission_rate,
            stamp_duty_sell=args.stamp_duty if args.stamp_duty is not None
            else DEFAULT_COST.stamp_duty_sell,
            slippage=args.slippage if args.slippage is not None
            else DEFAULT_COST.slippage,
        )
    else:
        cm = DEFAULT_COST
    c = cm.round_trip
    return c, (f"factor_lab.config.CostModel：佣金 {cm.commission_rate:.6g} + "
               f"印花税 {cm.stamp_duty_sell:.6g} + 双边滑点 {2 * cm.slippage:.6g} "
               f"= {c:.6f}（每 1 元投入）")


def _as_of_close(code: str, as_of: str) -> tuple[float, str]:
    """取 as_of 当日（或之前最近交易日）的未复权收盘价与日期标签。"""
    from factor_lab.analysis.risk_budget.market import load_bars
    df = load_bars(code, start="1990-01-01", end=as_of)
    last = df.iloc[-1]
    return float(last["close"]), str(int(last["date"]))


def _stock_name(code: str) -> str:
    from factor_lab.data import sqlite_source
    info = sqlite_source.load_stock_info()
    hit = info.loc[info["code"] == code, "name"]
    if hit.empty or not str(hit.iloc[0]).strip():
        return code          # 名称缺失只影响展示，不改成猜测名（也不阻断建议）
    return str(hit.iloc[0])


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    if not codes:
        raise SystemExit("--codes 不能为空")
    p_map = _parse_code_values(args.p, "--p")
    existing_map = _parse_code_values(args.existing, "--existing")

    missing = [c for c in codes if c not in p_map]
    if missing:
        raise SystemExit(
            f"以下代码缺 --p：{missing}。本模块不产出 p，禁止用默认值兜底"
        )

    params = RiskParams(
        shrinkage=args.shrinkage,
        max_position=args.max_position,
        max_industry=args.max_industry,
        max_total=args.max_total,
        min_position=args.min_position,
    )

    imap = (IndustryMap.from_parquet(args.industry_file) if args.industry_file
            else IndustryMap.from_workspace())
    cost, cost_note = _round_trip(args)

    # ── 已有持仓：换算成行业占用，从行业额度里扣 ──
    held_by_industry: dict[str, float] = {}
    for code, w in existing_map.items():
        held_by_industry[imap.l1(code)] = held_by_industry.get(imap.l1(code), 0.0) + w
    for code, w in existing_map.items():
        industry_caps(
            [IndustryUsage(imap.l1(code), w, args.max_industry, "existing")],
            args.max_industry,
        )

    r_arg = args.r
    if r_arg is not None and args.target_atr is not None:
        raise SystemExit(
            "--r 与 --target-atr 都是赔率来源，只能留一个（否则卡片上的 R 有两个出处）"
        )
    if r_arg is not None and r_arg <= 1.0:
        raise SystemExit(f"--r 必须 > 1（赔率），得到 {r_arg}")

    print("=" * 78)
    print("风险预算建议卡 —— 期望值闸门 + 分数凯利 + 三道硬上限")
    print("=" * 78)
    print(f"决策基准日 as-of : {args.as_of}")
    print(f"成本 c           : {cost_note}")
    print(f"仓位上限         : 单票 ≤{params.max_position:.2%}  "
          f"单行业 ≤{params.max_industry:.2%}  总仓位 ≤{params.max_total:.2%}  "
          f"收缩系数 {params.shrinkage}")
    print(f"p 来源           : {args.p_source}")
    print(f"样本量           : n={args.sample_n}")
    if args.sample_n < 380:
        print(f"  ⚠️ 样本量 {args.sample_n} < 380：按统计学结论，"
              f"不足以证明胜率优势（95% 置信、p=0.55 时需 ≥380 笔）")
    if "示例" in args.p_source or "假设" in args.p_source or "example" in args.p_source.lower():
        print("  ⚠️ p 是**示例假设值，非实测** —— 本卡仅演示风控链路，不可当交易依据")
    print(f"申万映射         : {len(imap)} 只（来源见模块 docstring）")
    print("=" * 78)

    levels_by_code = {}
    candidates: list[Candidate] = []
    for code in codes:
        price, day = _as_of_close(code, args.as_of)
        use_pct = args.stop_pct is not None or args.target_pct is not None
        atr = None if use_pct else load_atr(
            code, args.as_of, window=args.atr_window, method=args.atr_method,
        )[0]
        lv = build_levels(
            price, atr=atr, stop_atr=args.stop_atr, target_atr=args.target_atr,
            atr_window=args.atr_window, stop_pct=args.stop_pct,
            target_pct=args.target_pct, r=r_arg,
            entry_fraction=args.entry_fraction,
        )
        levels_by_code[code] = lv
        gate = evaluate_ev(p_map[code], lv.r_ratio, cost)
        print(f"\n[{code}] 现价 {price:,.2f} 元（{day}）  赔率 R={lv.r_ratio:.4f}")
        if atr is not None:
            print(f"  ATR({args.atr_window},{args.atr_method}) = {atr:,.4f} 元/股"
                  f"  → 止损 {lv.stop:,.2f} / 目标 {lv.target:,.2f}")
        else:
            print(f"  百分比口径 stop={args.stop_pct} target={args.target_pct}"
                  f"  → 止损 {lv.stop:,.2f} / 目标 {lv.target:,.2f}")
        tag = "✅ 通过" if gate.approved else "⛔ 否决"
        print(f"  EV 闸门 {tag}：EV_net={gate.ev_net:+.6f} /元投入"
              f"（毛 {gate.ev_gross:+.6f} − c {gate.cost_ratio:.6f}）")
        if gate.reason:
            print(f"    理由：{gate.reason}")
        cand = Candidate(
            code=code, p=gate.p, r=gate.r, ev_net=gate.ev_net,
            ev_gross=gate.ev_gross, ev_approved=gate.approved,
            industry_l1=imap.l1(code),
            reason_prefix=gate.reason,
        )
        candidates.append(cand)

    allocs = apply_portfolio_caps(
        candidates, params, existing_industry=held_by_industry,
    )
    by_code = {a.code: a for a in allocs}

    total_new = sum(a.weight for a in allocs if not a.dropped)
    pct_of = 100.0
    cards = []
    print("\n" + "=" * 78)
    print("仓位裁剪结果（按 EV 从高到低）")
    print("=" * 78)
    for a in allocs:
        state = "丢弃" if a.dropped else "建仓"
        print(f"  {a.code}  {state}  仓位 {a.weight * pct_of:.4f}%  约束={a.binding}")
        print(f"      {a.reason}")

    for code in codes:
        a = by_code[code]
        if a.dropped:
            continue
        lv = levels_by_code[code]
        gate = evaluate_ev(p_map[code], lv.r_ratio, cost)
        ind_total = held_by_industry.get(a.industry_l1, 0.0) + sum(
            x.weight for x in allocs
            if (not x.dropped) and x.industry_l1 == a.industry_l1
        )
        r_source = args.r_source or (
            f"ATR 倍数口径：目标 {lv.target_atr:g}×ATR ÷ 止损 {lv.stop_atr:g}×ATR"
            if lv.method == "atr" else
            f"百分比口径：目标 {lv.target_pct:.2%} ÷ 止损 {lv.stop_pct:.2%}"
        )
        cards.append(build_card(
            code=code, name=_stock_name(code), levels=lv,
            position_pct=a.weight * pct_of,
            industry_l1=a.industry_l1,
            industry_pct=ind_total * pct_of,
            max_position_pct=params.max_position * pct_of,
            max_industry_pct=params.max_industry * pct_of,
            p=gate.p, r=gate.r, cost_ratio=gate.cost_ratio,
            sample_n=args.sample_n,
            sample_note=(f"{args.sample_n} 笔（<380，不足以证明胜率优势）"
                         if args.sample_n < 380 else f"{args.sample_n} 笔"),
            p_source=args.p_source, r_source=r_source,
            position_reason=a.reason, ev_gross=gate.ev_gross,
        ))

    print("\n" + "=" * 78)
    print(f"建议卡 {len(cards)} 张（被闸门否决或低于最小建仓比例的标的不出卡）")
    print("=" * 78)
    for c in cards:
        print(render_card(c))
        print()

    print("-" * 78)
    print(f"新建仓合计 {total_new:.4f}% 账户"
          f"（已有持仓 {sum(existing_map.values()):.4f}% → "
          f"合计 {(total_new + sum(existing_map.values())):.4f}%）")
    for ind, w in sorted(
        (k, held_by_industry.get(k, 0.0) + sum(
            x.weight for x in allocs if (not x.dropped) and x.industry_l1 == k))
        for k in {a.industry_l1 for a in allocs if not a.dropped} | set(held_by_industry)
    ):
        print(f"  行业「{ind}」合计 {w:.4f}%  上限 {params.max_industry:.2%}")
    print("-" * 78)
    print("⚠️ 不构成投资建议｜历史统计不预示未来，决策与后果自担")

    if args.json:
        payload = {
            "as_of": args.as_of,
            "cost_ratio": cost,
            "cost_note": cost_note,
            "params": {
                "shrinkage": params.shrinkage,
                "max_position": params.max_position,
                "max_industry": params.max_industry,
                "max_total": params.max_total,
                "min_position": params.min_position,
            },
            "allocations": [
                {"code": a.code, "industry_l1": a.industry_l1,
                 "kelly_raw": a.kelly_raw, "weight": a.weight,
                 "binding": a.binding, "reason": a.reason, "dropped": a.dropped,
                 "ev_net": a.ev_net}
                for a in allocs
            ],
            "cards": [
                {k: v for k, v in vars(c).items()} for c in cards
            ],
            "disclaimer": "不构成投资建议",
        }
        out = WORKSPACE / "runtime" / "risk_budget_cards.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
