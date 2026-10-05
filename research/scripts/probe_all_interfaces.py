"""Tushare 2000 积分接口实测探测：对每个候选接口发真实请求，报告能否用 + 实际返回量。

为什么必须实测（而不是查文档）
--------------------------------
1. 文档标注的「积分要求」与账号实际档位可能不一致
2. **「有权限」≠「能拿到全量数据」** —— 限量字段、静默截断、只返回一年
3. 不同接口的限频不同，一次探测不出来的结论不可靠
4. 账号刚充值时积分可能延迟到账

因此本脚本对每个接口实际发一次请求，输出四要素：
  权限 / 实际返回行数 / 单次耗时 / 全量耗时估算

用法
----
  uv run python research/scripts/probe_all_interfaces.py
  uv run python research/scripts/probe_all_interfaces.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "scripts"))
from factor_lab.config import env_get  # noqa: E402
from fetch_tushare import call  # noqa: E402  复用统一的重试与限频逻辑

# ── 候选接口清单 ────────────────────────────────────────────────
# 字段：(api_name, 说明, 分类, 探测参数, 全量规模估算依据)
# 分�� P0=因子检验必需 / P1=显著增强 / P2=锦上添花 / P3=超出2000 积分
PROBES: list[tuple[str, str, str, dict, str]] = [
    # ── P0：因子检验的核心输入 ──
    ("adj_factor", "复权因子", "P0",
     {"ts_code": "600519.SH"}, "5,591 只 × 全历史"),
    ("daily_basic", "每日指标(PE/PB/市值/换手)", "P0",
     {"ts_code": "600519.SH", "start_date": "20260901", "end_date": "20260930"},
     "全市场日频，量最大"),
    ("fina_indicator", "财务指标(ROE/毛利率/负债率)", "P0",
     {"ts_code": "600519.SH"}, "5,591 只 × 8 季"),
    ("income", "利润表", "P0",
     {"ts_code": "600519.SH"}, "5,591 只 × 全期"),
    ("balancesheet", "资产负债表", "P0",
     {"ts_code": "600519.SH"}, "5,591 只 × 全期"),
    ("cashflow", "现金流量表", "P0",
     {"ts_code": "600519.SH"}, "5,591 只 × 全期"),

    # ── P1：显著增强（换手率/波动/流动性因子的必需输入）──
    ("daily", "日线(含复权价)", "P1",
     {"ts_code": "600519.SH", "start_date": "20260901", "end_date": "20260930"},
     "可替代本机未复权数据"),
    ("index_weight", "指数成分权重", "P1",
     {"index_code": "000300.SH", "start_date": "20260101", "end_date": "20260930"},
     "沪深300 每月"),
    ("index_daily", "指数日线", "P1",
     {"ts_code": "000300.SH", "start_date": "20260101", "end_date": "20260930"},
     "宽基指数行情"),
    ("stk_limit", "涨跌停价格", "P1",
     {"ts_code": "600519.SH", "start_date": "20260901", "end_date": "20260930"},
     "涨跌停因子必需"),
    ("suspend_d", "停复牌信息", "P1",
     {"ts_code": "600519.SH", "start_date": "20260101", "end_date": "20260930"},
     "停牌日处理"),

    # ── P2：锦上添花 ──
    ("stock_basic", "股票列表", "P2",
     {"list_status": "L"}, "全市场 L/D/P 三态"),
    ("trade_cal", "交易日历", "P2",
     {"start_date": "20260101", "end_date": "20261231"},
     "全年"),
    ("namechange", "更名记录", "P2",
     {"ts_code": "600519.SH"}, "变更历史"),
    ("index_classify", "指数分类(申万)", "P2",
     {"level": "L1", "src": "SW2021"}, "31 个"),
    ("index_member_all", "指数成分(全)", "P2",
     {"ts_code": "600519.SH"}, "5,591 只"),
    ("hs_const", "沪深港通成分", "P2",
     {"hs_type": "SH"}, "A股标的"),
    ("top_list", "龙虎榜", "P2",
     {"trade_date": "20260925"}, "每日"),
    ("moneyflow", "资金流向", "P2",
     {"ts_code": "600519.SH", "start_date": "20260901", "end_date": "20260930"},
     "全市场日频"),

    # ── P3：预期超出 2000 积分，测了以���知 ──
    ("bak_daily", "日线备份", "P3", {"ts_code": "600519.SH"}, "—"),
    ("cyq_perf", "筹码分布", "P3", {"ts_code": "600519.SH"}, "—"),
    ("forecast", "业绩预告", "P3", {"ts_code": "600519.SH"}, "—"),
    ("express", "业绩快报", "P3", {"ts_code": "600519.SH"}, "—"),
    ("fina_mainbz", "主营构成", "P3", {"ts_code": "600519.SH"}, "—"),
    ("report_rc", "研报评级", "P3", {"ts_code": "600519.SH"}, "—"),
    ("stk_holdernumber", "股东人数", "P3", {"ts_code": "600519.SH"}, "—"),
    ("pledge_stat", "股权质押", "P3", {"ts_code": "600519.SH"}, "—"),
]


def probe(token: str, api: str, params: dict) -> dict:
    """发一次真实请求，返回探测结果。"""
    t0 = time.perf_counter()
    try:
        df = call(token, api, params, retry=1)
        el = time.perf_counter() - t0
        return {
            "ok": len(df) > 0,
            "rows": len(df),
            "cols": list(df.columns)[:12],
            "elapsed": round(el, 2),
        }
    except Exception as e:                                    # noqa: BLE001
        return {
            "ok": False,
            "rows": 0,
            "cols": [],
            "elapsed": round(time.perf_counter() - t0, 2),
            "error": f"{type(e).__name__}: {e}"[:200],
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="把结果写入 JSON")
    ap.add_argument("--only", default=None, help="只测指定分类，如 P0")
    args = ap.parse_args()

    token = env_get("TUSHARE_TOKEN")
    if not token:
        print("✗ 未配置 TUSHARE_TOKEN（在项目根 .env 中设置）")
        return 1

    probes = [p for p in PROBES if not args.only or p[2] == args.only]

    print("=" * 92)
    print("Tushare 接口实测探测（真实请求，非文档推断）")
    print("=" * 92)
    print(f"{'接口':<22}{'档':<4}{'状态':<7}{'行数':>9}{'耗时':>7}  说明")
    print("-" * 92)

    results = {}
    cur_class = None
    for api, desc, cls, params, scale in probes:
        if cls != cur_class:
            label = {"P0": "P0 因子检验必需", "P1": "P1 显著增强",
                     "P2": "P2 锦上添花", "P3": "P3 预期超权限"}[cls]
            print(f"\n【{label}】")
            cur_class = cls

        r = probe(token, api, params)
        results[api] = {"desc": desc, "class": cls, "scale": scale, **r}
        mark = "✓ 可用" if r["ok"] else "✗ 拒绝"
        print(f"{api:<22}{cls:<4}{mark:<7}{r['rows']:>9,}{r['elapsed']:>6.1f}s  {desc}")

    # ── 汇总 ──
    avail = [k for k, v in results.items() if v["ok"]]
    denied = [k for k, v in results.items() if not v["ok"]]
    print()
    print("=" * 92)
    print(f"可用 {len(avail)} / {len(results)}    拒绝 {len(denied)}")
    if denied:
        print(f"不可用：{', '.join(denied)}")
    print("=" * 92)

    # 按分类给出可获取的数据清单
    print("\n【按用途归类：2000 积分实际能拿到的数据】")
    for cls, label in (("P0", "因子检验核心"), ("P1", "显著增强"),
                       ("P2", "锦上添花"), ("P3", "超出权限")):
        got = [v for v in results.values() if v["class"] == cls and v["ok"]]
        if not got:
            continue
        print(f"\n  {label}（{len(got)} 个）")
        for v in got:
            print(f"    {v['desc']:<28} 全量规模：{v['scale']}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
