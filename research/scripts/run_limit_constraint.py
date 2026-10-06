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
                verbose: bool = True) -> tuple:
    """产出 (buy_ok, sell_ok, limit_up, limit_dn, up, dn)。

    口径（**与 `tradability.limit_masks` 一致，不另立一套**）：
      · 买不进：封涨停 **或** 涨跌停价缺失
      · 卖不掉：封跌停 **或** 涨跌停价缺失
      · 涨跌停价缺失判「不可交易」是**保守**方向 —— 宁可少买不可乱买。

    ⚠️⚠️ **`close` 缺失刻意不判不可交易**（`unlisted_state="tradable"`）：

       实测（2026-10-06，全市场 2016-2026，1,464 万格）：
       ```
       close 缺失           73.78%   ← 主体是未上市/已退市，非停牌
       ├─ 判不可买         27.45%   ← 看起来约束极强
       ├─ 其中数据缺失     26.34%   ← 未上市/已退市，不是封板
       └─ 其中真封涨停      1.11%   ← 真实约束强度
       ```
       若把「未上市/已退市」也判成不可交易，会触发两个致命后果：
       ① **退市股永久锁仓** —— `sell_ok` 恒为 False
          ⇒ 卖出端「强制保留」把它永久留下
          ⇒ 收益被 `nan_to_num` 填成 0 ⇒ 组合里有僵尸持仓；
       ② **股票池被悄悄缩小** —— 实测「有约束」年化反而**上升**
          +1.20%/+1.38%/+1.95%，那不是约束的收益，是口径错误的假象。

       而未上市/已退市在**因子层面本就有因子值 NaN**，
       `rank_topk` 把 NaN 排到末尾 ⇒ 根本进不了候选池。
       **在选股层再判一遍是重复劳动，且正是上面两个 bug 的来源。**

    ⚠️ **买入端不看跌停、卖出端不看涨停**：
       涨停板是可以卖的（买盘排队照样成交），
       跌停板是可以买的（有人愿意割肉卖）。
       两边都用 `~(limit_up | limit_dn)` 会把可买可卖的票也误杀，
       约束过度收紧 ⇒ 虚增换手、虚增成本，方向反而是**更悲观**的假象。
    """
    codes = [str(c) for c in close_raw.columns]
    up, dn = load_limit_panel(LIMIT_PARQUET, start, end, close_codes=codes)
    limit_up, limit_dn = limit_masks(close_raw, up, dn,
                                     unlisted_state="tradable")
    if verbose:
        print(coverage_report(close_raw, up, "对照实验 "))
    buy_ok = ~limit_up.to_numpy()
    sell_ok = ~limit_dn.to_numpy()
    return (pd.DataFrame(buy_ok, index=close_raw.index,
                         columns=close_raw.columns),
            pd.DataFrame(sell_ok, index=close_raw.index,
                         columns=close_raw.columns),
            limit_up, limit_dn, up, dn)


def verify_constraint_active(
    label: str, held: np.ndarray, limit_up: pd.DataFrame,
    n_hold: int, rebal: np.ndarray,
) -> dict:
    """**直接验证约束真的生效**：调仓日的持仓里还有多少封涨停的票。

    ⚠️ 这条检查不能省 ——
       「加了约束」和「约束真的起作用了」是两件事。
       Δ（收益差）= 0 有两种**完全相反**的成因：

       | 现象 | 真实含义 | 能否引用 |
       |---|---|---|
       | 持仓有差异、Δ≈0 | 约束生效但净影响 <1bp | ✓ 可引用 |
       | 持仓逐位相同、Δ=0 | **约束从未生效**，A/B 是空转 | ✗ 不可引用 |

       ⚠️ **实测踩过（2026-10-06 冒烟）**：30 只 × 半年、n_hold=10，
       因子前 10 名恰好一次都没封涨停 ⇒ 6 个调仓日持仓**逐位相同**，
       Δ 恒为 0.00%，而脚本照样把它当有效结论打了出来。

    ⚠️ **分母必须是「持仓槽位数」= 调仓日数 × n_hold**，
       不是股票池只数（review 2026-10-06 抓出）：
       用 5,263 只股票池做分母会把占比**缩小 175 倍** ——
       本该是 0.79% 的数字打印成 0.0045%，
       方向恰好是「把真实存在的约束问题报成接近零」。
    """
    lu = limit_up.reindex(columns=limit_up.columns).to_numpy()
    n_at_rebal = 0
    n_limit_held = 0
    for t in np.where(rebal)[0]:
        h = held[t]
        h = h[h >= 0]
        if len(h) == 0:
            continue
        n_at_rebal += 1
        n_limit_held += int(lu[t, h].sum())
    denom = max(n_at_rebal * n_hold, 1)
    return {"标签": label, "调仓日": n_at_rebal,
            "封涨停持仓数": n_limit_held,
            "持仓槽位总数": n_at_rebal * n_hold,
            "占比": n_limit_held / denom}


def run_pair(factor_panel: pd.DataFrame, price: pd.DataFrame,
             spec: PortfolioSpec, cost: CostModel,
             buy_ok: pd.DataFrame, sell_ok: pd.DataFrame
             ) -> tuple[dict, dict]:
    """同一份输入跑两次：带约束 / 不带约束，其余完全相同。

    ⚠️ review 2026-10-06 指出初版有个**接了却从不使用**的 `limit_up` 参数
       （死参数，与本commit 要消灭的「零调用方」同类），已删除。
    """
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
    # ⚠️⚠️ **必须先把价格面板列序对齐，再生成掩码**（review 2026-10-06 抓出）：
    #   初版是「先生成掩码、再 reindex 价格面板」，
    #   而 `build_long_only` 按**位置**取掩码值（`to_numpy`）——
    #   只要两侧列序不同，第 i 个因子值就配到第 i 个涨跌停标记，
    #   **静默错配且不报错**。
    #   实测当前两侧都来自 `pivot_table`（都会排序）所以恰好一致，
    #   但那是**巧合**，不是保证 ⇒ 顺序必须反过来。
    close_raw = close_raw.reindex(columns=list(price.columns))
    if close_raw.shape[1] != price.shape[1]:
        print(f"⚠ 未复权面板列数 {close_raw.shape[1]} ≠ 因子面板 "
              f"{price.shape[1]}，缺数据的股票将被判不可交易")
    (buy_ok, sell_ok, limit_up_raw, limit_dn_raw,
     up_raw, dn_raw) = build_masks(close_raw, start, end)

    print("\n[3/4] 施加约束的自检（确认口径没搞错）")
    print(f"  买不进比例{(~buy_ok.to_numpy()).mean():.3%}")
    print(f"  卖不掉比例 {(~sell_ok.to_numpy()).mean():.3%}")
    # ⚠️⚠️ **必须把「不可交易」拆成「数据缺失」与「真封板」两部分**
    #   （2026-10-06 实测差点铸成大错）：
    #   `limit_masks` 初版把「close 缺失」也判成不可交易，
    #   而 close 缺失的**主体是未上市/已退市**（26.34%），
    #   于是「买不进 27.45%」看起来像约束极强，
    #   实际只有 **1.11%** 是真封涨停。
    #   更糟的是退市股 sell_ok 恒 False ⇒ **被永久锁仓**，
    #   组合里出现收益恒为 0 的僵尸持仓 ⇒ 「有约束」年化反而上升。
    up_re = up_raw.reindex(index=close_raw.index, columns=close_raw.columns)
    dn_re = dn_raw.reindex(index=close_raw.index, columns=close_raw.columns)
    c_ok = close_raw.notna().to_numpy()
    px_ok = up_re.notna().to_numpy() & dn_re.notna().to_numpy()
    judged = c_ok & px_ok
    print("\n  ── 「不可交易」的成因分解（互斥）──")
    print(f"    因数据缺失（未上市/已退市）{(~judged).mean():>7.3%}")
    print(f"    因真封涨停                {(limit_up_raw.to_numpy() & judged).mean():>7.3%}"
          f"  ← 真实约束强度")
    print(f"    因真封跌停                {(limit_dn_raw.to_numpy() & judged).mean():>7.3%}"
          f"  ← 真实约束强度")

    print("\n[4/4] A/B 对照")
    spec = PortfolioSpec(name="对照", n_hold=args.hold, n_pick=args.pick,
                         rebalance="M", factor=",".join(factors))
    rows = []
    verdicts = []
    for f, p in panels.items():
        z = _cross_z(p)
        base, withc = run_pair(z, price, spec, cost, buy_ok, sell_ok)
        if not (base.get("ok") and withc.get("ok")):
            print(f"  {f}: 失败 base={base.get('reason')} "
                  f"withc={withc.get('reason')}")
            continue

        # ⚠️ 约束是否真正生效：用**实际持仓**回答，不是用 Δ 猜
        key = pd.DatetimeIndex(z.index)
        period = key.to_period("M")
        chg = np.ones(len(period), dtype=bool)
        chg[1:] = period[1:] != period[:-1]
        rebal_mask = np.zeros(len(key), dtype=bool)
        rebal_mask[chg] = True

        lu_df = pd.DataFrame(limit_up_raw, index=price.index,
                             columns=price.columns)
        v = verify_constraint_active(f, withc["held_mat"], lu_df,
                                     spec.n_hold, rebal_mask)
        verdicts.append(v)

        # 持仓是否真的变了 —— Δ=0 的两种成因必须分开
        hb, hw = base["held_mat"], withc["held_mat"]
        diff_any = (hb != hw).any(axis=1)
        n_diff = int(diff_any.sum())
        n_rebal = int(rebal_mask.sum())
        n_diff_rebal = int((diff_any & rebal_mask).sum())
        d_nav = abs(base["累计净值"] - withc["累计净值"])
        effective = (n_diff > 0)
        d = withc["年化收益"] - base["年化收益"]
        dt = withc["平均换手"] - base["平均换手"]
        dc = withc["平均年成本"] - base["平均年成本"]

        for label, r in (("无约束", base), ("有约束", withc)):
            rows.append({"因子": f, "口径": label,
                         "年化": r["年化收益"], "波动": r["年化波动"],
                         "夏普": r["夏普"], "回撤": r["最大回撤"],
                         "换手": r["平均换手"], "年成本": r["平均年成本"],
                         "累计净值": r["累计净值"]})

        flag = "" if effective else "  ⚠️ 约束未生效（持仓逐位相同）"
        print(f"  {f:<14} 年化 {base['年化收益']:>8.2%} → "
              f"{withc['年化收益']:>8.2%}（Δ {d:+.2%}）")
        print(f"  {'':<14} 换手 {base['平均换手']:>7.2%} → "
              f"{withc['平均换手']:>7.2%}（Δ {dt:+.2%}）  "
              f"年成本 Δ {dc:+.2%}")
        # ⚠️ **分母用「总交易日」，不是「调仓日」**：
        #   持仓在**非调仓日**也会变（卖出端强制保留会改写持仓），
        #   所以「调仓日 X/Y」这个口径会漏掉大部分变化，
        #   实测出现「1794/129」这种分子大于分母的荒谬输出。
        print(f"  {'':<14} 持仓变动 {n_diff}/{len(hb)} 交易日"
              f"（其中调仓日 {n_diff_rebal}/{n_rebal}）"
              f"  净值差 {d_nav:.2e}{flag}")

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