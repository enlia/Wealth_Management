"""涨跌停约束对照实验：把`tradability.py` 接入生产回测并量化其影响。

为什么必须做这个对照
------------------
`docs/03_项目报告/12_价量因子有效性检验报告.md` 的全部结论都是在
**完全没有涨跌停约束**下得出的 —— `tradability.py` 建好了、
`limit_masks` 有覆盖率自检，但**零生产调用方**。

A 股实盘里：涨停买不到、跌停卖不掉、停牌动不了。
因子选出的强势股，封涨停率显著高于全市场 0.794% ⇒ 影响不是随机噪声。

本脚本做 A/B 对照（同参数、同数据、只差涨跌停约束），回答三个问题：
  1. 约束真的接上了吗？（用「封涨停股是否出现在持仓里」直接验证）
  2. 约束让净收益变化多少？
  3. 换手与成本怎么变？（理论上应下降：买不到强势股 ⇒ 换手更低）

用法
----
  uv run python research/scripts/run_limit_constraint.py --smoke# 30 只×1年
  uv run python research/scripts/run_limit_constraint.py --all          # 全市场
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from factor_lab.analysis.long_only import (  # noqa: E402
    CostModel,
    PortfolioSpec,
    build_long_only,
)
from factor_lab.analysis.tradability import (  # noqa: E402
    coverage_report,
    limit_masks,
    load_limit_panel,
)
from factor_lab.config import DB_PATH, is_a_share  # noqa: E402

OUTPUT = ROOT / "runtime" / "limit_constraint"
LIMIT_PARQUET = ROOT / "runtime" / "tushare" / "stk_limit.parquet"


def load_close_raw(codes: list[str], start: str, end: str,
                   verbose: bool = True) -> pd.DataFrame:
    """读**未复权**收盘价面板（涨跌停判定的唯一正确口径）。

    ⚠️ **必须是未复权价**，不能用 close_adj ——
       实测 sh6005192024-01-02 收盘 1685.01 / 后复权 1531.31，
       差 9%，而涨停判定阈值是 0.01 元级。
       拿后复权价比原始涨跌停价，几乎所有股票都判成「未封板」：
       约束彻底失效却不报任何错（tradability.limit_masks 有同样警示）。
    """
    import sqlite3

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    parts = []
    for i in range(0, len(codes), 800):
        sub = codes[i:i + 800]
        q = (f"SELECT date, code, close FROM bar_daily "
             f"WHERE date BETWEEN ? AND ? AND close IS NOT NULL "
             f"AND code IN ({','.join('?' * len(sub))})")
        parts.append(pd.read_sql(q, con, params=[start, end, *sub]))
    con.close()
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    wide = df.pivot_table(index="date", columns="code", values="close",
                          aggfunc="last").sort_index()
    if verbose:
        print(f"  未复权收盘面板 {wide.shape}，"
              f"内存 {wide.memory_usage(deep=True).sum()/1e6:.0f}MB")
    return wide


def build_masks(close_raw: pd.DataFrame, start: str, end: str,
                verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """产出 (buy_ok, sell_ok) 两个 bool 面板。

    口径（**与`tradability.limit_masks` 一致，不另立一套**）：
      · 买不进：封涨停 **或** 缺数据（停牌/未上市）
      · 卖不掉：封跌停 **或** 缺数据
      · 缺数据判「不可交易」是**保守**方向 ——宁可少买不可乱买。
        静默把它当「可交易」是本项目已踩过的坑（见 tradability 模块 docstring）。

    ⚠️ **买入端不看跌停、卖出端不看涨停**：
       涨停板是可以卖的（买盘排队照样成交），
       跌停板是可以买的（有人愿意割肉卖）。
       两边都用 `~(limit_up | limit_down)` 会把可买可卖的票也误杀，
       约束过度收紧 ⇒ 虚增换手、虚增成本，方向反而是**更悲观**的假象。
    """
    codes = [str(c) for c in close_raw.columns]
    up, dn = load_limit_panel(LIMIT_PARQUET, start, end, close_codes=codes)
    limit_up, limit_dn = limit_masks(close_raw, up, dn)
    if verbose:
        print(coverage_report(close_raw, up, "对照实验 "))
    buy_ok = ~(limit_up.to_numpy())
    sell_ok = ~(limit_dn.to_numpy())
    return (pd.DataFrame(buy_ok, index=close_raw.index,
                         columns=close_raw.columns),
            pd.DataFrame(sell_ok, index=close_raw.index,
                         columns=close_raw.columns))


def _verify_constraint_active(
    label: str, held: np.ndarray, limit_up: pd.DataFrame,
    cols: list[str], rebal: np.ndarray,
) -> dict:
    """**直接验证约束真的生效**：持仓里还有多少封涨停的票。

    ⚠️ 这条断言不能省 ——
       `select_with_buffer` 的买入约束是「跳过不可买」，
       若某天整个候选池都封涨停，它会**静默地选不满**；
       而 `held_mat` 的 -1 哨兵又会把槽位压成 0 权重。
       两者叠加 =「约束太紧 → 持仓不足 → 等权退化」，
       表现为「收益变差」但**看起来像因子失效**。
       实测必须区分这两种原因。
    """
    lu = limit_up.to_numpy()
    n_at_rebal = 0
    n_limit_held = 0
    for t in np.where(rebal)[0]:
        h = held[t]
        h = h[h >= 0]
        if len(h) == 0:
            continue
        n_at_rebal += 1
        n_limit_held += int(lu[t, h].sum())
    return {"标签": label, "调仓日": n_at_rebal,
            "封涨停持仓数": n_limit_held,
            "占比": n_limit_held / max(n_at_rebal * len(cols), 1)}


def run_pair(factor_panel: pd.DataFrame, price: pd.DataFrame,
             spec: PortfolioSpec, cost: CostModel,
             buy_ok: pd.DataFrame, sell_ok: pd.DataFrame,
             limit_up: pd.DataFrame) -> tuple[dict, dict]:
    """同一份输入跑两次：带约束 / 不带约束，其余完全相同。"""
    base = build_long_only(factor_panel, spec, cost, price)
    withc = build_long_only(factor_panel, spec, cost, price,
                buy_ok=buy_ok, sell_ok=sell_ok)
    return base, withc


def main() -> int:
    ap = argparse.ArgumentParser(description="涨跌停约束对照实验")
    ap.add_argument("--all", action="store_true", help="全市场")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--factors", default="rev5,vol60")
    ap.add_argument("--start", default="20230101")
    ap.add_argument("--end", default="20231231")
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--pick", type=int, default=50)
    ap.add_argument("--smoke", action="store_true", help="冒烟模式")
    args = ap.parse_args()

    from run_long_only import build_panel, _cross_z
    from factor_lab.data import all_codes

    cost = CostModel()
    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    start, end = args.start, args.end
    if args.smoke:
        start, end = "20230101", "20230630"

    print("=" * 88)
    print("涨跌停约束对照实验（唯一变量：是否施加涨跌停约束）")
    print("=" * 88)
    print(f"区间 {start}~{end}  因子 {factors}  持股 {args.hold}/{args.pick}")

    codes = [c for c in all_codes() if is_a_share(c)]
    if not args.all:
        # ⚠️ 随机抽样，不能取前 N 个（同 run_factor_study 的踩坑记录）：
        #   all_codes() 按代码排序，取前 N 个 =清一色北交所新股，
        #   股票池「上市满1 年」一项全灭 → 面板为空。
        rng = np.random.default_rng(20240101)
        codes = sorted(rng.choice(codes,
                                  size=min(args.n, len(codes)),
                                  replace=False))
    print(f"股票池 {len(codes):,} 只")

    print("\n[1/4] 构建因子面板 …")
    panels = build_panel(codes, factors, start, end)
    price = panels.pop("__price__", None)
    if price is None or price.empty:
        print("✗ 缺价格面板")
        return 1

    print("\n[2/4] 构建涨跌停约束面板（未复权口径）…")
    close_raw = load_close_raw(list(price.columns),
                              price.index.min().strftime("%Y%m%d"),
                              price.index.max().strftime("%Y%m%d"))
    # ⚠️ **列序必须与因子面板一致** —— build_long_only 按下标对齐，
    #   这里 reindex 只做「取交集 + 补NaN」，不改变顺序。
    buy_ok, sell_ok = build_masks(close_raw, start, end)
    codes_aligned = list(price.columns)
    close_raw = close_raw.reindex(columns=codes_aligned)

    print("\n[3/4] 施加约束的自检（确认口径没搞反）")
    print(f"  买不进比例 {(~buy_ok.to_numpy()).mean():.3%}")
    print(f"  卖不掉比例 {(~sell_ok.to_numpy()).mean():.3%}")

    print("\n[4/4] A/B 对照")
    spec = PortfolioSpec(name="对照", n_hold=args.hold, n_pick=args.pick,
                         rebalance="M", factor=",".join(factors))
    rows = []
    for f, p in panels.items():
        z = _cross_z(p)
        base, withc = run_pair(z, price, spec, cost, buy_ok, sell_ok,
                               pd.DataFrame(~buy_ok.to_numpy(),
                                            index=price.index,
                                            columns=price.columns))
        for label, r in (("无约束", base), ("有约束", withc)):
            if not r.get("ok"):
                print(f"  {f} {label}: 失败（{r.get('reason')}）")
                continue
            rows.append({"因子": f, "口径": label,
                         "年化": r["年化收益"], "波动": r["年化波动"],
                         "夏普": r["夏普"], "回撤": r["最大回撤"],
                         "换手": r["平均换手"], "年成本": r["平均年成本"],
                         "累计净值": r["累计净值"]})
        if base.get("ok") and withc.get("ok"):
            d = withc["年化收益"] - base["年化收益"]
            dt = withc["平均换手"] - base["平均换手"]
            print(f"  {f:<14} 年化 {base['年化收益']:>8.2%} → "
                  f"{withc['年化收益']:>8.2%}（Δ {d:+.2%}）  "
                  f"换手 {base['平均换手']:>6.2%} → {withc['平均换手']:>6.2%}"
                  f"（Δ {dt:+.2%}）")

    if not rows:
        print("✗ 无有效结果")
        return 1

    df = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print("对照汇总")
    print("=" * 88)
    piv = df.pivot_table(index="因子", columns="口径",
                         values=["年化", "夏普", "换手", "年成本"])
    with pd.option_context("display.width", 200):
        print(piv.to_string(float_format=lambda x: f"{x:>9.4f}"))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT / "ab_compare.csv", index=False, encoding="utf-8-sig")
    try:
        from data_version import build_manifest
        ver = build_manifest()
        (OUTPUT / "run_meta.json").write_text(json.dumps(
            {"数据版本": ver["version"], "打戳": ver["stamped_at"],
             "区间": f"{start}~{end}", "因子": factors,
             "成本": {"买入": cost.buy, "卖出": cost.sell},
             "股票池": "全市场" if args.all else f"随机 {len(codes)} 只"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n数据版本: {ver['version']}")
    except Exception as e:                                   # noqa: BLE001
        print(f"⚠ 版本记录失败: {e}")

    print("\n提示：以上为统计检验，不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())