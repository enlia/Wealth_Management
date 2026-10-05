"""Tushare 数据补全：按缺陷优先级拉取，本地合并入库。

背景（2026-10-05 审计）
----------------------
本机通达信数据存在三处可量化缺陷：
  1. 未复权除权污染：主板 0.291%，年化偏差 2.3pp
  2. 行业字段缺 37.2%（6,082 只无行业）
  3. 幸存者偏差：退市股基本被剔除

时间预算（实测网络延迟 89ms/次，2000 积分档限频200 次/分）
  下载 22,374 次调用 → 频次受限约 112 分钟
  清洗 + 重建 + 重跑检验 → 约 98 分钟
  合计 ≈ 3.5 小时

用法
----
  export TUSHARE_TOKEN=xxx
  uv run python scripts/fetch_tushare_supplement.py --all
  uv run python scripts/fetch_tushare_supplement.py --adj-factor-only
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from factor_lab.config import DB_PATH, env_get, is_a_share

ENDPOINT = "http://api.tushare.pro"
OUT_DIR = Path(__file__).resolve().parents[2] / "runtime" / "tushare"

# 2000 积分档：200 次/分钟
RATE_PER_MIN = 200
# 留 10% 余量，避免触发限流惩罚
SAFE_RATE = int(RATE_PER_MIN * 0.9)


def call(token: str, api: str, params: dict, retry: int = 3) -> pd.DataFrame:
    """调一次接口，返回 DataFrame。带重试与限流感知。"""
    body = json.dumps({"api_name": api, "token": token,
                       "params": params, "fields": ""}).encode()
    for attempt in range(retry):
        req = urllib.request.Request(
            ENDPOINT, data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read().decode())
            if out.get("code") == 0:
                d = out.get("data") or {}
                items, fields = d.get("items") or [], d.get("fields") or []
                return pd.DataFrame(items, columns=fields)
            msg = str(out.get("msg", ""))
            if "每分钟" in msg or "频率" in msg:
                time.sleep(20)
                continue
            return pd.DataFrame()
        except Exception:  # noqa: BLE001
            if attempt < retry - 1:
                time.sleep(2 ** attempt * 3)
    return pd.DataFrame()


def rate_limit_sleep(n_calls: int) -> None:
    """按限频要求休眠。"""
    need = n_calls / (SAFE_RATE / 60)
    if need > 0:
        time.sleep(need)


def to_ts_code(code6: str) -> str:
    """6 位裸码 → Tushare 格式。"""
    c = str(code6).zfill(6)
    if c.startswith(("6", "9")):
        return f"{c}.SH"
    if c.startswith(("0", "3")):
        return f"{c}.SZ"
    return f"{c}.BJ"


class Fetcher:
    def __init__(self, token: str, start: str, end: str):
        self.token = token
        self.start = start.replace("-", "")
        self.end = end.replace("-", "")
        self.calls = 0
        self.t0 = time.perf_counter()
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _tick(self):
        self.calls += 1
        if self.calls % SAFE_RATE == 0:
            elapsed = time.perf_counter() - self.t0
            target = self.calls / (SAFE_RATE / 60)
            if target > elapsed:
                time.sleep(target - elapsed)

    def fetch_by_stock(self, api: str, codes: list[str],
                       extra: dict | None = None) -> pd.DataFrame:
        """逐只股票拉取（一次拿该股全历史，最省调用）。"""
        parts, t0 = [], time.perf_counter()
        for i, c in enumerate(codes):
            params = {"ts_code": c, "start_date": self.start,
                      "end_date": self.end}
            if extra:
                params.update(extra)
            df = call(self.token, api, params)
            if len(df):
                parts.append(df)
            self._tick()
            if (i + 1) % 500 == 0:
                el = time.perf_counter() - t0
                print(f"    {i+1}/{len(codes)}  {el/60:.1f} 分钟  "
                      f"累计 {self.calls} 次", flush=True)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def report(self):
        el = time.perf_counter() - self.t0
        print(f"    完成：{self.calls:,} 次调用，{el/60:.1f} 分钟")


def main() -> int:
    ap = argparse.ArgumentParser(description="Tushare 数据补全")
    ap.add_argument("--token", default=None)
    ap.add_argument("--start", default="2015-12-01")
    ap.add_argument("--end", default="2026-10-02")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--adj-factor-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只拉前N 只（试跑）")
    args = ap.parse_args()

    token = args.token or env_get("TUSHARE_TOKEN")
    if not token:
        print("缺少 token。请在项目根的 .env 中设置 TUSHARE_TOKEN=<你的token>")
        print("（参考 .env.example；.env 已被 .gitignore 排除，不会误提交）")
        return 1

    import urllib.request  # noqa: F401  （call 内部用）
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    codes = [r[0] for r in con.execute("SELECT DISTINCT code FROM bar_daily")]
    codes = [to_ts_code(c) for c in codes if is_a_share(c)]
    if args.limit:
        codes = codes[: args.limit]
    print("=" * 74)
    print("Tushare 数据补全")
    print("=" * 74)
    print(f"标的 {len(codes):,} 只   区间 {args.start} ~ {args.end}")
    print(f"限频 {SAFE_RATE} 次/分钟（2000 积分档上限 200，留 10% 余量）")
    print(f"预计 {len(codes)*3:,} 次调用 ≈ {len(codes)*3/(SAFE_RATE/60)/60:.0f} 分钟")
    print(f"输出目录: {OUT_DIR}")

    f = Fetcher(token, args.start, args.end)

    # ① 复权因子（最关键）
    print("\n[1/3] 复权因子 adj_factor（修正 2.3pp 年化偏差）…")
    adj = f.fetch_by_stock("adj_factor", codes)
    if len(adj):
        adj.to_parquet(OUT_DIR / "adj_factor.parquet", index=False)
        print(f"    ✓ {len(adj):,} 行 → adj_factor.parquet")
    else:
        print("    ✗ 空返回（检查权限或区间）")
    f.report()
    if args.adj_factor_only:
        return 0

    # ② 退市股 + 申万行业（各 1 次调用）
    print("\n[2/3] 退市股 / 申万行业分类…")
    for api, kw in (("stock_basic", {"list_status": "D"}),
                    ("index_classify", {"level": "L1", "src": "SW2021"}),
                    ("index_classify", {"level": "L2", "src": "SW2021"})):
        df = call(token, api, kw)
        f.calls += 1
        if len(df):
            name = f"{api}_{kw.get('level', kw.get('list_status', 'x'))}"
            df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
            print(f"    ✓ {api:<16} {len(df):>6,} 行 → {name}.parquet")
        else:
            print(f"    ✗ {api} 空返回")

    # ③ 行业成分 + 每日指标
    print("\n[3/3] 申万行业成分 / 每日指标 PE/PB…")
    for api in ("index_member_all", "daily_basic"):
        df = f.fetch_by_stock(api, codes)
        if len(df):
            df.to_parquet(OUT_DIR / f"{api}.parquet", index=False)
            print(f"    ✓ {api:<18} {len(df):>10,} 行")
        else:
            print(f"    ✗ {api} 空返回")
        f.report()

    print()
    print("=" * 74)
    print("下一步：把新数据并入本机数据库")
    print("=" * 74)
    print("  1. 用 adj_factor 修正 bar_daily 的 close（后复权）")
    print("  2. 用 index_member_all 补全 industry 字段")
    print("  3. 用 stock_basic(D) 扩展股票池，加入退市股")
    print("  4. 跑 audit_data_quality.py 确认偏差 < 0.5pp/年")
    print("  5. 重跑 expand_factor_library.py 与holding_period_test.py")
    print(f"\n输出: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    import urllib.request
    raise SystemExit(main())
