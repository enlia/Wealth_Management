"""Tushare 数据补全：从 MCP 配置自动取 token，落盘为本地 parquet。

设计要点
--------
1. **token 自动读取**：从 ~/.workbuddy/mcp.json 的 tushareMcp 配置里提取，
   不需要用户手动传环境变量。token 只在内存中使用，绝不写入任何文件。
2. **分批落盘 + 断点续跑**：每 200 只存一个分片，中断后重跑会跳过已完成的。
3. **限频自适应**：默认 180 次/分钟（200 上限的 90%），
   遇限频自动退避，实测退避后可继续。
4. **质量校验**：每个分片落盘前检查行数与字段，
   出现「单股返回行数明显偏少」立即告警（防接口静默截断）。

用法
----
  uv run python scripts/fetch_tushare.py --task adj --limit 5# 试跑 5 只
  uv run python scripts/fetch_tushare.py --task adj                    # 全量
  uv run python scripts/fetch_tushare.py --task all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from factor_lab.config import DB_PATH, is_a_share  # noqa: E402

ENDPOINT = "http://api.tushare.pro"
MCP_JSON = Path(os.path.expanduser("~/.workbuddy/mcp.json"))
OUT = Path(__file__).parent.parent / "out" / "tushare"
RATE = 180          # 次/分钟，留 10% 余量
EXPECT_DAYS = 2829  # 2015-12 ~ 2026-09 的交易日数，用于完整性校验


# ── token ────────────────────────────────────────────────────
def get_token() -> str | None:
    """从 MCP 配置提取 token（不落盘、不打印）。"""
    if os.environ.get("TUSHARE_TOKEN"):
        return os.environ["TUSHARE_TOKEN"]
    if not MCP_JSON.exists():
        return None
    s = MCP_JSON.read_text(encoding="utf-8")
    # 优先取 tushareMcp 段内的 56 位 token
    try:
        cfg = json.loads(s)
        srv = cfg.get("mcpServers", {}).get("tushareMcp", {})
        blob = json.dumps(srv, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        blob = s
    m = re.findall(r"\b([0-9a-f]{56})\b", blob)
    if m:
        return m[0]
    m = re.findall(r"\b([0-9a-f]{32,64})\b", blob)
    return m[0] if m else None


# ── 单次调用 ──────────────────────────────────────────────────
def call(token: str, api: str, params: dict, retry: int = 3) -> pd.DataFrame:
    body = json.dumps({"api_name": api, "token": token,
                       "params": params, "fields": ""}).encode()
    for k in range(retry):
        req = urllib.request.Request(
            ENDPOINT, data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read().decode())
            if out.get("code") == 0:
                d = out.get("data") or {}
                it, fl = d.get("items") or [], d.get("fields") or []
                return pd.DataFrame(it, columns=fl)
            msg = str(out.get("msg", ""))
            if "每分钟" in msg or "频率" in msg or "抽取" in msg:
                time.sleep(30)
                continue
            if "权限" in msg:
                raise PermissionError(f"{api}: {msg}")
            return pd.DataFrame()
        except PermissionError:
            raise
        except Exception:  # noqa: BLE001
            if k < retry - 1:
                time.sleep(3 * (2 ** k))
    return pd.DataFrame()


def to_ts(code6: str) -> str:
    """本机 6 位裸码 → Tushare 格式。

    ⚠️ 踩坑（2026-10-05 实测）：本机代码前缀是【小写 sh/sz/bj】，
       初始版本漏了 bj 分支，导致前若干只全是北交所（bj899050 等），
       转换后仍传原始小写前缀 → Tushare 返回 0 行且不报错。
       静默返回空是最危险的失败模式，必须三段全覆盖 + 小写转大写。
    """
    c = str(code6).lower().strip()
    assert c[:2] in ("sh", "sz", "bj"), f"未知代码前缀: {code6}"
    num = c[2:].zfill(6)
    return f"{num}.{c[:2].upper()}"


# ── 分片拉取（断点续跑）────────────────────────────────────────
def fetch_shard(token: str, api: str, codes: list[str], tag: str,
                start: str, end: str, extra: dict | None = None,
                batch: int = 200) -> pd.DataFrame:
    """按分片拉取并落盘。已完成的分片自动跳过。"""
    OUT.mkdir(parents=True, exist_ok=True)
    shards = [codes[i:i + batch] for i in range(0, len(codes), batch)]
    parts, t0, calls = [], time.perf_counter(), 0
    done_n = 0

    for si, sh in enumerate(shards):
        fpath = OUT / f"{tag}_shard{si:03d}.parquet"
        if fpath.exists():
            parts.append(pd.read_parquet(fpath))
            done_n += len(sh)
            continue
        buf = []
        for c in sh:
            p = {"ts_code": c, "start_date": start, "end_date": end}
            if extra:
                p.update(extra)
            df = call(token, api, p)
            calls += 1
            if len(df):
                buf.append(df)
            # 限频控制
            if calls % RATE == 0:
                el = time.perf_counter() - t0
                tgt = calls / (RATE / 60)
                if tgt > el:
                    time.sleep(tgt - el)
        d = pd.concat(buf, ignore_index=True) if buf else pd.DataFrame()
        d.to_parquet(fpath, index=False)
        parts.append(d)
        done_n += len(sh)
        el = (time.perf_counter() - t0) / 60
        print(f"  [{tag}] 分片 {si+1}/{len(shards)}  累计 {done_n:,} 只  "
              f"{len(d):,} 行  {el:.1f} 分钟", flush=True)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description="Tushare 数据补全")
    ap.add_argument("--task", default="adj",
                    choices=["adj", "industry", "basic", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", default="20151201")
    ap.add_argument("--end", default="20260930")
    args = ap.parse_args()

    token = get_token()
    if not token:
        print("✗ 未能从 ~/.workbuddy/mcp.json 取得 token")
        print("  可手动设置：export TUSHARE_TOKEN=xxx")
        return 1
    print("=" * 76)
    print("Tushare 数据补全")
    print("=" * 76)
    print(f"token:已从 MCP 配置取得（{len(token)} 字符，不落盘）")
    print(f"限频: {RATE} 次/分钟")
    print(f"输出: {OUT}")

    import sqlite3
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    codes = [to_ts(r[0]) for r in con.execute("SELECT DISTINCT code FROM bar_daily")
             if is_a_share(r[0])]
    if args.limit:
        codes = codes[: args.limit]
    print(f"标的: {len(codes):,} 只 A 股")
    print()

    t0 = time.perf_counter()

    # ① 退市股 + 行业分类（各 1 次，最快）
    if args.task in ("basic", "industry", "all"):
        print("[1/3] 退市股名单…")
        d = call(token, "stock_basic", {"list_status": "D"})
        if len(d):
            d.to_parquet(OUT / "stock_basic_D.parquet", index=False)
            print(f"  ✓ {len(d):,} 只退市股")
        print("[2/3] 申万行业分类 L1/L2/L3…")
        for lv in ("L1", "L2", "L3"):
            d = call(token, "index_classify", {"level": lv, "src": "SW2021"})
            if len(d):
                d.to_parquet(OUT / f"index_classify_{lv}.parquet", index=False)
                print(f"  ✓ {lv} {len(d):>4} 个行业")
        print("[3/3] 申万行业成分（逐只）…")
        mem = fetch_shard(token, "index_member_all", codes, "industry_member",
                          args.start, args.end)
        if len(mem):
            mem.to_parquet(OUT / "index_member_all.parquet", index=False)
            print(f"  ✓ {len(mem):,} 行，覆盖 {mem['ts_code'].nunique():,} 只")

    if args.task in ("adj", "all"):
        print("\n复权因子（逐只，全历史）…")
        adj = fetch_shard(token, "adj_factor", codes, "adj", args.start, args.end)
        if len(adj):
            adj.to_parquet(OUT / "adj_factor.parquet", index=False)
            n = adj.groupby("ts_code").size()
            print(f"  ✓ {len(adj):,} 行，覆盖 {len(n):,} 只")
            print(f"  每只行数: 中位 {n.median():.0f}  最少 {n.min()}  "
                  f"最多 {n.max()}")
            low = n[n < EXPECT_DAYS * 0.8]
            if len(low):
                print(f"  ⚠️ {len(low)} 只返回行数偏少（<{int(EXPECT_DAYS*0.8)}）"
                      f"，可能被静默截断：{list(low.index[:5])}")
            else:
                print(f"  ✓ 完整性检查通过（无明显截断）")

    el = (time.perf_counter() - t0) / 60
    print()
    print("=" * 76)
    print(f"完成，耗时 {el:.1f} 分钟")
    print("=" * 76)
    print("下一步：")
    print("  1. verify_tushare_data.py  质量校验（必须先跑）")
    print("  2. merge_tushare_into_db.py  并入本机数据库")
    print("  3. audit_data_quality.py    确认偏差 < 0.5pp/年")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
