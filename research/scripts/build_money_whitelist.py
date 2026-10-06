"""从**真实返回数据**推导金额列白名单，输出 `_money_whitelist.json`。

为什么必须用脚本生成，不能手写
------------------------------
初版手写了白名单，结果 30 个列名是**凭记忆写的、实际不存在**
（`accounts_payable` 实际叫 `acct_payable`，`currency_borr` 实际叫 `cb_borr`…）。
而真实存在但没写进白名单的金额列会**静默漏换算** —— 后果与不换算一样。

改成从真实 parquet 推导 + 显式排除规则，接口版本变化时重跑即可。

排除规则（不可换算）
--------------------
| 类别 | 例子 | 为什么 |
|---|---|---|
| 每股指标 | eps, bps, cfps, *_ps | 单位【元/股】，不是元 |
| 比率 | roe, grossprofit_margin, debt_to_assets | 无量纲 |
| 增速 | or_yoy, assets_yoy | % |
| 天数 | inv_turn_days | 天 |
| 周转率 | assets_turn | 次 |
| 股数 | total_share | 股 |

⚠️ 关键词法**只能用来排除**，不能用来判断「是金额」——
   `assets_turn`（周转率）含 assets，`ebit`（元）不含任何金额词。

用法
----
  uv run python research/scripts/build_money_whitelist.py
  uv run python research/scripts/build_money_whitelist.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "runtime" / "tushare"
WHITELIST_FILE = OUT / "_money_whitelist.json"

# 明确不是金额的列
# ⚠️ 排除规则要覆盖**前缀形式**（basic_eps / diluted_eps），
#    只写 ^eps$ 会漏掉—— 实测踩过：income 表的 basic_eps、
#    diluted_eps 都被误当成金额列。
NOT_MONEY_PATTERNS = re.compile(
    # 每股类：eps / bps / cfps 及其前缀组合（basic_eps、diluted_eps…）
    r"(eps$|^eps|_ps$|_bps$|^bps|^cfps|^ncfps|^ocfps"
    r"|_to_|_ratio|_margin|_rate|_pct|_yoy|_ytd|days$|_turn"
    r"|^total_share$|_share$|^share_|_per$)",
    re.I,
)
# 单独的比率字段（不含上述后缀，需显式列出）
NOT_MONEY_EXACT = {
    "roe", "roe_waa", "roe_dt", "roa", "roic", "roic_yearly",
    "bps", "cfps", "ncfps", "ocfps", "eps", "dt_eps", "basic_eps",
    "diluted_eps", "total_share", "capital_rese_ps", "surplus_rese_ps",
    "undist_profit_ps", "total_revenue_ps", "revenue_ps",
}

# 标识列（永远不是金额）
ID_COLS = {"ts_code", "ann_date", "f_ann_date", "end_date", "report_type",
           "comp_type", "end_type", "update_flag", "first_ann_date"}

# 整个表都不换算
SKIP_TABLES = {"fina_indicator_clean", "top10_holders", "top10_floatholders",
               "stk_managers", "index_weight", "stock_basic"}


def derive(path: Path) -> list[str]:
    """从 parquet 推导金额列。"""
    df = pd.read_parquet(path)
    cols = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        if c.lower() in NOT_MONEY_EXACT:
            continue
        if NOT_MONEY_PATTERNS.search(c):
            continue
        if df[c].dtype.kind not in "if":        # 只取数值列
            continue
        cols.append(c)
    return sorted(cols)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成金额列白名单")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(p for p in OUT.glob("*_clean.parquet"))
    if not files:
        print(f"✗ 没有找到 *_clean.parquet：{OUT}")
        return 1

    result: dict[str, list[str]] = {}
    print("=" * 76)
    print("推导金额列白名单（从真实数据）")
    print("=" * 76)

    for f in files:
        name = f.stem
        if name in SKIP_TABLES:
            print(f"  ○ {name:<24} 跳过（每股/比率类表，不换算）")
            continue
        cols = derive(f)
        result[name] = cols
        total = len(pd.read_parquet(f).columns)
        print(f"  ✓ {name:<24} 金额列 {len(cols):>3} / 共 {total:>3}")

    # 抽样展示，方便人工核对
    print("\n抽样核对（每表前 8 个金额列）:")
    for k, v in result.items():
        print(f"  {k}: {v[:8]}")

    if not args.dry_run:
        WHITELIST_FILE.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {WHITELIST_FILE.name}"
              f"（{WHITELIST_FILE.stat().st_size:,} 字节）")
    print("=" * 76)
    print("⚠️ 生成后仍需人工抽查：白名单只能排除确定不是金额的列，")
    print("   漏进来的比率字段会被错误地/1e8。")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
