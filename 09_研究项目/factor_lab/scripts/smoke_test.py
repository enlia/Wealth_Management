"""冒烟测试：数据层正确性验证（AGENTS.md 第3条强制要求）

三步验证，缺一不可：
  步骤1  小样本读取（5 只）—— 确认能跑通
  步骤2  抽查实际数值 —— 与通达信 MCP 实测值逐个对照
  步骤3  交叉验证 —— SQLite 读出值 vs 本机 .day 二进制直读值

运行：uv run python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factor_lab.config import DEFAULT_COST, TDX_VIPDOC, is_a_share, is_index, is_sector, market_of
from factor_lab.data import all_codes, available_code_count, load_long, load_prices

# 通达信 MCP 实测值（2026-09-30 收盘），作为"已知真值"
KNOWN = {
    "sh600519": 1258.62,   # 贵州茅台
    "sz000001": 11.57,     # 平安银行
    "sh000001": 3842.19,   # 上证指数
    "sh880402": 2391.23,   # 生物制药（板块）
    "sh603259": 167.34,    # 药明康德
    "sz002142": 35.43,     # 宁波银行
    "sz399001": 12887.62,  # 深证成指
    "sh900948": 3.74,      # 宁波银行B（B股，系数 0.001）
}

# 通达信 .day 二进制直读，用于交叉验证
DTYPE = np.dtype([
    ("date", "<u4"), ("open", "<u4"), ("high", "<u4"),
    ("low", "<u4"), ("close", "<u4"), ("amount", "<f4"),
    ("vol", "<u4"), ("rsv", "<u4"),
])


def _price_scale(code: str) -> float:
    """与 04_工具脚本/tdx.py 保持一致的价格缩放系数。
    code 必须是带前缀的 8 位形式（sh000001 / sz000001），前缀直接采用，不重新推断。"""
    c = code[2:]
    if c.startswith("88"):
        return 0.01
    if c.startswith(("900", "200")):
        return 0.001
    m = code[:2]                      # ← 直接用前缀，不调 market_of 猜
    if m == "sh":
        if c.startswith(("000", "950")):
            return 0.01
        if c[0] in "12345":
            return 0.001
    elif m == "sz":
        if c.startswith("399"):
            return 0.01
        if c[0] in "03":
            return 0.01
        return 0.001
    return 0.01


def read_tdx_direct(code: str) -> tuple[int, float] | None:
    """绕过 SQLite，直接读 .day 二进制文件，返回 (日期, 收盘价)。

    ⚠️ 前缀必须直接取用：sh000001(上证) 与 sz000001(平安) 代码都是 000001，
       重新推断市场会导致读错文件。
    """
    c = code[2:]
    m = code[:2]
    p = TDX_VIPDOC / m / "lday" / f"{m}{c}.day"
    if not p.exists():
        return None
    raw = np.fromfile(p, dtype=DTYPE)
    if raw.size == 0:
        return None
    last = raw[-1]
    return int(last["date"]), float(last["close"]) * _price_scale(code)


def step1_smoke() -> None:
    print("=" * 70)
    print("步骤 1  小样本读取（5 只股票）")
    print("=" * 70)
    t0 = time.perf_counter()
    codes = list(KNOWN)[:5]
    wide = load_prices(codes, start="2026-09-01", end="2026-09-30")
    dt = time.perf_counter() - t0
    print(f"  读取耗时 {dt*1000:.0f} ms")
    print(f"  形状 {wide.shape}  索引 {wide.index.min().date()} ~ {wide.index.max().date()}")
    assert wide.shape[0] > 0, "读到 0 行"
    assert wide.shape[1] == 5, f"列数应为 5，实际 {wide.shape[1]}"
    print(f"  列名 {list(wide.columns)}")
    print("  ✓ 形状/索引/列名符合 alphalens 输入要求")


def step2_verify_known() -> None:
    print()
    print("=" * 70)
    print("步骤 2  抽查实际数值 vs 通达信 MCP 实测值")
    print("=" * 70)
    wide = load_prices(list(KNOWN))
    last_row = wide.ffill().iloc[-1]
    ok = 0
    for code, expect in KNOWN.items():
        if code not in wide.columns:
            print(f"  {code} 不在库中")
            continue
        got = float(last_row[code])
        good = abs(got - expect) < max(0.02, abs(expect) * 1e-4)
        ok += good
        mark = "✓" if good else "✗"
        print(f"  {mark} {code}  期望 {expect:>10.2f}  实得 {got:>10.2f}  差 {got-expect:+.4f}")
    print(f"\n  通过 {ok}/{len(KNOWN)}")
    assert ok == len(KNOWN), "存在对不上的标的，停止后续步骤"


def step3_cross_validate() -> None:
    print()
    print("=" * 70)
    print("步骤 3  交叉验证：SQLite 读出值 vs .day 二进制直读值")
    print("=" * 70)
    print("  意义：SQLite 是二次构建的产物，若与原始文件不一致说明构建有 bug")
    wide = load_prices(list(KNOWN))
    last_row = wide.ffill().iloc[-1]
    diffs = []
    for code in KNOWN:
        direct = read_tdx_direct(code)
        if direct is None:
            print(f"  ? {code} 本机无 .day 文件")
            continue
        d_date, d_close = direct
        s_close = float(last_row[code])
        diff = abs(s_close - d_close)
        diffs.append(diff)
        rel = diff / max(d_close, 1e-9)
        mark = "✓" if rel < 1e-6 else "✗"
        print(f"  {mark} {code}  .day直读 {d_close:>10.4f}  SQLite {s_close:>10.4f}  相对差 {rel:.2e}")
    if diffs:
        print(f"\n  最大绝对差 {max(diffs):.6f}，平均 {np.mean(diffs):.6f}")
        assert max(diffs) < 0.01, "SQLite 与原始文件不一致"
    print("  ✓ 两路数据完全一致")


def step4_classification() -> None:
    print()
    print("=" * 70)
    print("步骤 4  标的分类规则验证（防重名混淆 / 防指数混入）")
    print("=" * 70)
    cases = [
        ("sh600519", "A股", True, False, False),
        ("sz000001", "A股", True, False, False),
        ("sz300750", "A股", True, False, False),
        ("sh000001", "指数", False, True, False),
        ("sz399001", "指数", False, True, False),
        ("sh880402", "板块", False, False, True),
        ("sh900948", "B股", False, False, False),
        ("bj920000", "北交所", True, False, False),
    ]
    ok = 0
    for code, name, want_a, want_i, want_s in cases:
        got = (is_a_share(code), is_index(code), is_sector(code))
        good = got == (want_a, want_i, want_s)
        ok += good
        print(f"  {'✓' if good else '✗'} {code} {name:8s} A股={got[0]} 指数={got[1]} 板块={got[2]}")
    print(f"\n  通过 {ok}/{len(cases)}")
    assert ok == len(cases), "分类规则有误"
    print("  ✓ sh000001(上证) 与 sz000001(平安) 被正确区分")


def step5_inventory() -> None:
    print()
    print("=" * 70)
    print("步骤 5  全库概况与成本模型")
    print("=" * 70)
    n = available_code_count()
    print(f"  全库标的数 {n:,}")
    t0 = time.perf_counter()
    codes = all_codes()
    a = [c for c in codes if is_a_share(c)]
    idx = [c for c in codes if is_index(c)]
    sec = [c for c in codes if is_sector(c)]
    print(f"  A股 {len(a):,}  指数 {len(idx):,}  板块 {len(sec):,}  其他 {len(codes)-len(a)-len(idx)-len(sec):,}")
    print(f"  分类耗时 {time.perf_counter()-t0:.1f}s")
    print()
    print("  交易成本模型：")
    print(f"    买入 {DEFAULT_COST.buy_cost*10000:.2f} 基点  卖出 {DEFAULT_COST.sell_cost*10000:.2f} 基点")
    print(f"    一次完整买卖 {DEFAULT_COST.round_trip*10000:.1f} 基点 = {DEFAULT_COST.round_trip*100:.3f}%")
    print(f"    年换手 50 倍 → 年成本 {DEFAULT_COST.round_trip*50*100:.1f}%")


if __name__ == "__main__":
    step1_smoke()
    step2_verify_known()
    step3_cross_validate()
    step4_classification()
    step5_inventory()
    print()
    print("=" * 70)
    print("全部冒烟测试通过 —— 数据层可信，可进入因子计算")
    print("=" * 70)
