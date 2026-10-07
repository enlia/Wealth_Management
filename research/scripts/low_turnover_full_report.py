"""实验轮 1 · A 线全量报告渲染（模板强制列头集中在此）。

模板强制（裁定与 task_plan.md:31-33）：口径三件套列头 + P14 四要素 +
多重比较声明 + 「不构成投资建议」尾句；数字一律带【全量实测·发布待双闸】标；
判据按**分布**列（胜率/中位/最差/窗口数），不按均值。
"""
from __future__ import annotations

LABEL = "【全量实测·发布待双闸】"

CALIBER_HEADER = (
    "| 价格口径 | 换手单位 | 年化方式 |\n|---|---|---|\n"
    "| 前复权（qfq）close_adj（口径断点见 AGENTS 九·5/6） | 倍/年（双边 L1=引擎口径，单边=½）"
    "与 %/日 均值×252；算式列逐行给出 | 几何 _year_span（自然日/365.25，策略与基准同函数）；"
    "252 仅倍数换算层（SCALING_TRADING_DAYS） |"
)

P14_HEADER = (
    "| ①年化口径 | ②成本假设 | ③基准 | ④多空/多头 |\n|---|---|---|---|\n"
    "| 几何 nav^(1/年数)−1，年数=_year_span " + LABEL + " "
    "| config.DEFAULT_COST 往返 20bp 单收（买 7.5bp+卖 12.5bp），引擎 CostModel 按同费项映射、"
    "round_trip=0.002 逐位断言 " + LABEL + " "
    "| 同池等权买持有（_bench_stats，同一 _year_span）；分层检验用层内等权基准 " + LABEL + " "
    "| 单边多头（A 股散户现实口径，多空数字不适用） " + LABEL + " |"
)

RULED_NOTES = (
    "裁定采纳留痕：①缓冲档映射 n_pick = n_hold + max(1, ceil(n_hold×档距))，"
    "护栏 assert_distinct_pools（映射值+封顶有效池双查）命中数见护栏表；"
    "②信号双轨：主=lag1（信号取前一自然月月末快照，即「跳过最近 1 月」），"
    "对=lag0（仓内原生）；rev5 算式 = −(close[t]/close[t−5] − 1)（5 交易日形成期，"
    "简报「1 个月形成期」字样按裁定以本算式勘误）；"
    "③成本=config.DEFAULT_COST.round_trip=20bp/往返单收（D8 判决口径），"
    "简报「单边 20bp」旧文同轮勘误。"
)


def _fmt_pct(x: float) -> str:
    return f"{x:+.2%}" if x == x else "N/A"


def render_full(res: dict) -> str:
    """res 键：judgments/ windows/ segments/ layers/ guards/ deflated/ meta。"""
    y = LABEL
    out: list[str] = [
        "# 实验轮 1·A 线 低换手月频 —— 全量评估 " + y, "",
        "## 判决依据与裁定留痕", RULED_NOTES, "",
        "## 口径三件套（强制列头）", CALIBER_HEADER, "",
        "## P14 四要素（强制列头）", P14_HEADER, "",
    ]

    out += ["## 换手三档护栏（裁定①：逐档真异池证明）",
            "| 轨 | 档 | n_pick/有效池 | 守卫命中(拦截) | 年换手(倍/年·双边) | 年换手(单边) | 年成本 |",
            "|---|---|---|---|---|---|---|"]
    for g in res["guards"]:
        out.append(f"| {g['track']} | {g['pct']:.0%} | {g['n_pick']}/{g['n_pick_eff']} "
                   f"| {g['guard_hits']} | {g['turn_x252']:.4f} | {g['turn_x252']/2:.4f} "
                   f"| {g['cost_year']:+.4%} |")

    out += ["", "## 滚动 7 窗明细（主表=主口径 lag1；对照轨随附）",
            "| 轨 | 档 | 窗 | 测试区间 | 策略年化净 | 毛 | 基准 | 超额 |", "|---|---|---|---|---|---|---|---|"]
    for w in res["windows"]:
        out.append(f"| {w['track']} | {w['pct']:.0%} | {w['窗口']} | {w['测试起']}~{w['测试止']} "
                   f"| {_fmt_pct(w['策略'])} | {_fmt_pct(w['毛'])} | {_fmt_pct(w['基准'])} "
                   f"| {_fmt_pct(w['超额'])} |")

    out += ["", "## 判据分布表（判分布不判均值：胜率≥60% ∧ 中位>0 ∧ 最差>−5% ∧ 窗口≥5）",
            "| 轨 | 档 | 窗口数 | 胜率 | 超额中位 | 最差 | 最好 | 可用 | 原因 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for j in res["judgments"]:
        out.append(f"| {j['track']} | {j['pct']:.0%} | {j['窗口数']} | {j['胜率']:.0%} "
                   f"| {_fmt_pct(j['超额中位数'])} | {_fmt_pct(j['最差'])} | {_fmt_pct(j['最好'])} "
                   f"| {'✓' if j['可用'] else '✗'} | {j['原因']} |")

    out += ["", "## 三段子区间（2016-2018 / 2019-2022 / 2023-2026，毛/净并列）",
            "| 轨 | 档 | 段 | 净年化 | 毛年化 | 段内基准 | 超额 | 年换手(单边) |",
            "|---|---|---|---|---|---|---|---|"]
    for s in res["segments"]:
        out.append(f"| {s['track']} | {s['pct']:.0%} | {s['段']} | {_fmt_pct(s['净'])} "
                   f"| {_fmt_pct(s['毛'])} | {_fmt_pct(s['基准'])} | {_fmt_pct(s['超额'])} "
                   f"| {s['turn_x252']/2:.4f} |")

    out += ["", "## 层内混淆检验（历史时点流通市值三分位，层内基准）",
            "| 轨 | 档 | 层 | 窗口数 | 胜率 | 层内超额中位 | 最差 | 可用 |",
            "|---|---|---|---|---|---|---|---|"]
    for a in res["layers"]:
        out.append(f"| {a['track']} | {a['pct']:.0%} | {a['层']} | {a['窗口数']} | {a['胜率']:.0%} "
                   f"| {_fmt_pct(a['超额中位数'])} | {_fmt_pct(a['最差'])} "
                   f"| {'✓' if a['可用'] else '✗'} |")

    out += ["", "## 多重比较校正：Deflated Sharpe（Bailey & López de Prado 2014）",
            f"尝试组合数 n_trials = {res['deflated']['n_trials']}（缓冲档 3 × 信号轨 2；"
            "口径=窗口年化超额序列、T=7、ppy=1、i.i.d. 形式未做自相关修正）",
            "| 候选（最优先） | SR_a | SR0(择优期望) | DSR | 观测数 |", "|---|---|---|---|---|"]
    for d in res["deflated"]["rows"]:
        out.append(f"| {d['name']} | {d['sr_annual']:.4f} | {d['sr0']:.4f} | {d['dsr']:.4f} | {d['n_obs']} |")

    out += ["", "## 与冒烟数字勾稽", res.get("勾稽", "（见 .planning/reports/experiment1a-smoke.md（会话台账层，未入 git） §三：冒烟=9 票池持 7、等权手工模拟；"
            "全量=全池 build_universe、引擎逆波动率权重、7 窗/三段/层内——口径不同不可直接比大小，"
            "换手量级与档距方向性一致才可勾稽）"), ""]
    out += ["## 多重比较声明", "本轮参数网格 6 组合 + 层内 18 + 三段 18 列（列数如实）；"
            "Deflated Sharpe 对 6 组合择优口径校正（上表）；其余列作稳健性记录不作挑选依据。", ""]
    out += ["## 免责", y + " 数字待「review + 发布」双闸后方可引用；本报告不构成投资建议。"]
    return "\n".join(out)
