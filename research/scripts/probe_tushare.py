"""Tushare 接口可用性实测：用真实 token 逐个试，不靠猜。

为什么必须实测
--------------
Tushare 的积分门槛在文档里写得很清楚，但：
  1. 不同接口的「积分要求」标注分散，容易看错
  2. 账号刚注册时积分可能未到账、或有延迟
  3. 同一档位下不同接口权限可能不同（如 daily_basic vs limit_list_d）
  4. **「有权限」≠「能拿到全量数据」**（限量字段、静默截断）

所以本脚本对每个候选接口实际发一次请求，报告：
  · 权限是否放开
  · 实际返回行数（判断是否被限量静默截断）
  · 耗时（用于估算全市场拉取的可行时间）

用法
----
  export TUSHARE_TOKEN=你的token
  uv run python scripts/probe_tushare.py
  uv run python scripts/probe_tushare.py --token xxx --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from factor_lab.config import env_get  # noqa: E402

ENDPOINT = "http://api.tushare.pro"

# 候选接口 → (说明, 该项目是否必需, 期望最小积分档)
CANDIDATES = [
    # ── P0：本项目当前缺口 ──
    ("adj_factor",      "复权因子",           True,  2000, {"ts_code": "600519.SH"}),
    ("daily",           "日线（含复权）",     True,  120,  {"ts_code": "600519.SH",
                                                  "start_date": "20240101",
                                                  "end_date": "20240110"}),
    ("stock_basic",     "股票列表",           True,  120,  {"list_status": "D"}),
    ("pro_bar",         "复权行情（一体化）", True,  2000, {"ts_code": "600519.SH",
                                                   "start_date": "20240101",
                                                   "end_date": "20240110",
                                                   "adj": "hfq"}),
    ("daily_basic",     "每日指标PE/PB/市值", True,  120,  {"ts_code": "600519.SH",
                                                   "trade_date": "20240110"}),
    ("index_weight",    "指数成分权重",       True,  2000, {"index_code": "000300.SH",
                                                   "start_date": "20240101",
                                                   "end_date": "20240110"}),
    # ── P1：提升数据质量 ──
    ("fina_indicator",  "财务指标",           True,  2000, {"ts_code": "600519.SH"}),
    ("income",          "利润表",             True,  120,  {"ts_code": "600519.SH",
                                                   "period": "20231231"}),
    ("balancesheet",    "资产负债表",         True,  120,  {"ts_code": "600519.SH",
                                                   "period": "20231231"}),
    ("cashflow",        "现金流量表",         True,  120,  {"ts_code": "600519.SH",
                                                   "period": "20231231"}),
    ("index_classify",  "申万行业分类",       True,  2000, {"level": "L1",
                                                   "src": "SW2021"}),
    ("index_member",    "申万行业成分",       True,  2000, {"l1_code": "801780.SI"}),
    ("limit_list_d",    "涨跌停价格",         True,  2000, {"trade_date": "20240110"}),
    ("suspend_d",       "停复牌信息",         False, 2000, {"trade_date": "20240110"}),
    ("namechange",      "股票曾用名",         False, 2000, {"ts_code": "600519.SH"}),
    # ── P2：可选的因子数据 ──
    ("top_list",        "龙虎榜",             False, 2000, {"trade_date": "20240110"}),
    ("margin_detail",   "融资融券明细",       False, 2000, {"trade_date": "20240110"}),
    ("moneyflow_hsgt",  "沪深港通资金",       False, 2000, {"trade_date": "20240110"}),
    ("ths_index",       "同花顺概念",         False, 3000, {}),
    ("cyq_perf",        "筹码分布",           False, 5000, {"ts_code": "600519.SH"}),
    ("forecast",        "盈利预测",           False, 8000, {"ts_code": "600519.SH"}),
]


def call(token: str, api: str, params: dict, timeout: int = 30) -> dict:
    body = json.dumps({"api_name": api, "token": token,
                       "params": params, "fields": ""}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    out["_elapsed"] = time.perf_counter() - t0
    return out


def probe(token: str, api: str, params: dict) -> dict:
    try:
        r = call(token, api, params)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "code": -1, "msg": f"{type(e).__name__}: {str(e)[:60]}",
                "n": 0, "elapsed": 0.0}
    items = (r.get("data") or {}).get("items") or []
    return {"ok": r.get("code") == 0, "code": r.get("code"),
            "msg": (r.get("msg") or "")[:70], "n": len(items),
            "elapsed": r.get("_elapsed", 0.0),
            "fields": len((r.get("data") or {}).get("fields") or [])}


def main() -> int:
    ap = argparse.ArgumentParser(description="Tushare 接口实测")
    ap.add_argument("--token", default=None)
    ap.add_argument("--all", action="store_true", help="测全部候选接口")
    args = ap.parse_args()

    token = args.token or env_get("TUSHARE_TOKEN")
    if not token:
        print("=" * 78)
        print("缺少 token")
        print("=" * 78)
        print("  获取方式：登录 tushare.pro →右上角头像→ 用户中心 → 接口TOKEN")
        print("  配置方式：把 token 填入项目根的 .env（参考 .env.example）")
        print("            .env 已被 .gitignore 排除，不会误提交")
        print("  临时覆盖：TUSHARE_TOKEN=xxx uv run python ...")
        print("            或 --token 参数")
        return 1

    print("=" * 78)
    print("Tushare 接口实测（结果以实际返回为准，不靠猜）")
    print("=" * 78)

    # 先测最基础的 stock_basic，确认 token 有效
    r = probe(token, "stock_basic", {"list_status": "L"})
    if not r["ok"]:
        print(f"\n✗ token 无效或积分不足：[{r['code']}] {r['msg']}")
        print("  若提示「积分不足」，说明当前档位还没到这个接口的门槛。")
        print("  免费新用户通常是 120 积分。")
        return 1
    print(f"\n✓ token 有效。stock_basic 返回 {r['n']:,} 条，{r['fields']} 个字段"
          f"，耗时 {r['elapsed']:.2f}s")

    todo = CANDIDATES if args.all else [c for c in CANDIDATES if c[2]]
    print()
    print(f"{'接口':<18}{'必需':<6}{'需积分':>7}{'状态':<8}"
          f"{'行数':>9}{'字段':>6}{'耗时':>8}  说明")
    print("-" * 78)
    need, ok_cnt, fail = [], 0, 0
    for api, desc, need_flag, need_pts, params in todo:
        r = probe(token, api, params)
        flag = "★" if need_flag else " "
        if r["ok"]:
            status = "✓"
            ok_cnt += 1
            if need_flag:
                need.append((api, need_pts, r["n"]))
        else:
            status = "✗"
            fail += 1
            if need_flag:
                need.append((api, need_pts, None))
        note = r["msg"] if not r["ok"] else ""
        print(f"{flag}{api:<17}{'必需' if need_flag else '可选':<6}"
              f"{need_pts:>7}{status:<8}{r['n']:>9,}{r['fields']:>6}"
              f"{r['elapsed']:>7.2f}s  {note}")

    print("-" * 78)
    print(f"可调 {ok_cnt} / 不可调 {fail} / 共 {len(todo)}")

    if need:
        print()
        print("本项目必需但被拒的接口：")
        for api, pts, n in need:
            print(f"  · {api:<18} 需 {pts:>5} 积分"
                  + (f"（当前可用）" if n is not None else " ← **需升级**"))
        if any(n is None for _, _, n in need):
            max_pts = max(p for _, p, n in need if n is None)
            print()
            print(f"→ 需至少 {max_pts:,} 积分档位。")

    print()
    print("提示：返回行数异常少（如某个按日接口只返回 1~2 条）可能触发了"
          "每日限量或被静默截断，需调高积分档位。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
