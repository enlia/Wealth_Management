"""量纲校验：市值单位必须是【万元】（2026-10-06 统一）

为什么需要这个检查
------------------
量纲错误**不报错**，只会让因子静默失真。本项目踩过两次：

1. 本机`universe.csv` 导出的市值是**亿元**
2. Tushare 的 `total_mv` / `circ_mv` 是**万元**

两者混用差1 万倍。若不校验，市值中性化的对数市值因子会完全失效，
而回测照常输出数字 —— 属于典型的静默失败。

判定原理
--------
市值 / 股本 = 股价。已知股价与股本，就能反推市值的真实量纲：

    mktcap单位是万元 → mktcap(万元) / shares(万股) = 元/股 ✓
    mktcap单位是亿元 → mktcap(亿元) / shares(亿股) = 元/股 ✓

两种量纲都能反推出正确股价，所以单看反推无法区分 ——
必须与**已知的真值**（Tushare 数据）对比才能发现量纲错误。

同源反推法的第二个战场是每股净资产（bps）：分子口径必须是**普通股权益**。
归母权益里可能含其他权益工具（优先股/永续债），bps 却是每股【普通股】净资产，
两者混用差 17% 也不报错（实证见下方 U6 段）。反推式固定为

    implied_bps = (归母权益 − COALESCE(oth_eqt_tools, 0)) × 1e8 / 总股本

用法
----
  uv run python research/scripts/check_units.py
  uv run python research/scripts/check_units.py --skip-tushare# 无 Tushare 时只查本机自洽性
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from factor_lab.config import DB_PATH, MARKET_CAP_UNIT, YI_TO_WAN  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 纯搬移拆分（500 行硬线）：U6 公式族/守卫族/TOL 单一实现在 units_probes.py，
# 此处保持原名再导出（含 runtime 案卷区门脚本的调用面），语义零变。
from units_probes import (  # noqa: E402
    FAMILY_OTH_EQT_TOOLS,
    IDENTITY_ABS_TOL_YI,
    PB_REL_TOL,
    PB_ROUND_ABS_TOL,
    TOL,
    U6_FORMULA_C,
    compare_bps_with_db,
    equity_identity_guard,
    g4_identity_check,
    g4_identity_summary,
    implied_bps_a,
    implied_bps_c,
    ordinary_equity_yi,
    require_oth_eqt_tools_column,
    snapshot_pb_guard,
)

__all__ = [
    "FAMILY_OTH_EQT_TOOLS",
    "IDENTITY_ABS_TOL_YI",
    "PB_REL_TOL",
    "PB_ROUND_ABS_TOL",
    "TOL",
    "U6_FORMULA_C",
    "compare_bps_with_db",
    "equity_identity_guard",
    "implied_bps_a",
    "implied_bps_c",
    "ordinary_equity_yi",
    "require_oth_eqt_tools_column",
    "snapshot_pb_guard",
]

# 抽查标的：沪深主板 + 创业板 + 保险 + 宁德，覆盖不同量级；
# 其他权益工具口径项入探针清单（U6）：再加两只其他权益工具存量大户
# （market.db ts_balance_sheet@20260630 实测 oth_eqt_tools：招商银行
#   1999.89 亿、农业银行 4700.0 亿；sz000001 自带 800.0 亿），
# 探针 4→6 覆盖「有/无其他权益工具」两种权益切分形态；再补 sh600011
# （oth_eqt_tools 734.75 亿 = 归母 53.25% + minority 727.91 亿 = 归母 52.75%
#   双高极端，敞口谱唯一推到 ≥18% 档的天然边界锚），探针 6→7。
PROBES = [
    ("sh600519", "600519.SH", "贵州茅台（大盘蓝筹）"),
    ("sz000001", "000001.SZ", "平安银行（深主板）"),
    ("sh601318", "601318.SH", "中国平安（保险）"),
    ("sz300750", "300750.SZ", "宁德时代（创业板）"),
    ("sh600036", "600036.SH", "招商银行（其他权益工具大户）"),
    ("sh601288", "601288.SH", "农业银行（其他权益工具大户）"),
    ("sh600011", "600011.SH", "华能国际（其他权益工具+少数股东权益双高边界锚）"),
]

def check_self_consistency(con: sqlite3.Connection) -> list[str]:
    """本机自洽性：mktcap / shares 能否还原出股价。"""
    msgs = []
    print("=" * 72)
    print(f"检查 1：本机量纲自洽性（期望单位 = {MARKET_CAP_UNIT}）")
    print("=" * 72)
    print(f"{'标的':<26}{'股价':>10}{'市值(万)':>18}{'股本(亿)':>12}{'反推价':>10}判定")
    print("-" * 72)

    for code, ts, name in PROBES:
        r = con.execute(
            "SELECT price, mktcap, shares FROM stock_info WHERE code=?", (code,)
        ).fetchone()
        if not r or not r[0] or not r[1] or not r[2]:
            msgs.append(f"{name} {code}：数据缺失，跳过")
            continue
        price, mktcap, shares = float(r[0]), float(r[1]), float(r[2])
        # 万元 / (亿股) → 元/股
        implied = mktcap / YI_TO_WAN / shares   # 万元 → 亿元 / 亿股 = 元/股
        err = abs(implied - price) / price
        ok = err < 0.01
        mark = "✓" if ok else "✗ 量纲错误"
        print(f"{name:<26}{price:>10,.2f}{mktcap:>18,.0f}{shares:>12,.2f}"
              f"{implied:>10,.2f}  {mark}")
        if not ok:
            msgs.append(
                f"{name} {code}：反推价 {implied:,.2f} 与股价 {price:,.2f} 差 "
                f"{err*100:.1f}% —— 量纲或数值有问题"
            )
    return msgs


def check_against_tushare(con: sqlite3.Connection) -> list[str]:
    """与 Tushare 对比：这是发现量纲错误的唯一可靠方法。"""
    msgs = []
    tpath = Path(__file__).resolve().parents[2] / "runtime" / "tushare" / "daily_basic.parquet"
    if not tpath.exists():
        print("\n跳过 Tushare 对比：daily_basic.parquet 不存在")
        return [f"未找到 {tpath.name}，跳过 Tushare 对比"]

    import pandas as pd

    print()
    print("=" * 72)
    print(f"检查 2：与 Tushare 对比（两边都应是{MARKET_CAP_UNIT}）")
    print("=" * 72)
    print(f"{'标的':<26}{'Tushare(万)':>18}{'本机(万)':>18}{'相对差':>10}判定")
    print("-" * 72)

    df = pd.read_parquet(tpath)
    # ⚠️ 取最新交易日，不能硬编码 —— 数据会滚动，写死日期会每天误报失败
    latest = df["trade_date"].astype(str).max()
    df = df[df["trade_date"].astype(str) == latest]
    print(f"（对比日期：{latest}，自动取Tushare 最新交易日）")

    for code, ts, name in PROBES:
        t = df[df["ts_code"] == ts]
        l = con.execute("SELECT mktcap FROM stock_info WHERE code=?", (code,)).fetchone()
        if not len(t) or not l or not l[0]:
            msgs.append(f"{name} {code}：Tushare 或本机数据缺失，跳过")
            continue
        tv = float(t.iloc[0]["total_mv"])
        lv = float(l[0])
        err = abs(tv - lv) / tv
        ok = err < TOL
        mark = "✓" if ok else "✗ 量纲不一致"
        print(f"{name:<26}{tv:>18,.0f}{lv:>18,.0f}{err*100:>9.4f}%  {mark}")
        if not ok:
            ratio = tv / lv if lv else float("nan")
            hint = ""
            if 0.5 < ratio < 2:
                hint = "（数值差异，可能是数据时点不同）"
            elif 5e3 < ratio < 2e4:
                hint = " → 本机是【亿元】，差 1e4 倍！"
            elif 0.5e-4 < ratio < 2e-4:
                hint = " → 本机是【万元】但值偏大 1e4 倍，疑似重复换算！"
            msgs.append(
                f"{name} {code}：Tushare {tv:,.0f} vs 本机 {lv:,.0f}，"
                f"差 {err*100:.2f}%，比值 {ratio:.4g}{hint}"
            )
    return msgs


def probe_equity_row(con: sqlite3.Connection, code: str) -> dict:
    """单票权益口径探针：最新期 ts 行 + db_bps + C/A 对账 + 两守卫（只读）。

    缺行/缺基准/归母 NULL 返回 status 字段显式说明，不冒算。
    """
    row = con.execute(
        "SELECT end_date, total_hldr_eqy_exc_min_int, total_hldr_eqy_inc_min_int,"
        " minority_int, oth_eqt_tools, total_share FROM ts_balance_sheet"
        " WHERE ts_code=? AND report_type='1' ORDER BY end_date DESC LIMIT 1",
        (code,),
    ).fetchone()
    if not row:
        return {"code": code, "status": "ts_balance_sheet 无 report_type='1' 行"}
    end_date, exc, inc, minority, oth, share = row
    if exc is None:
        return {"code": code, "status": f"归母 NULL（bs {end_date}）—— 不可反推"}
    if not share:
        return {
            "code": code,
            "status": f"缺 total_share（bs {end_date}）—— 反推分母不可用，显式缺件态不冒算",
            "bs_end_date": end_date,
        }
    bps_row = con.execute(
        "SELECT bps, end_date FROM ts_fina_indicator WHERE ts_code=?"
        " ORDER BY end_date DESC LIMIT 1",
        (code,),
    ).fetchone()
    if not bps_row or not bps_row[0]:
        return {"code": code, "status": "ts_fina_indicator 无 bps —— 无对账基准"}
    si = con.execute(
        "SELECT price, bps, pb FROM stock_info WHERE code=?", (code,)
    ).fetchone()
    return {
        "code": code,
        "status": "ok",
        "bs_end_date": end_date,
        "bps_end_date": bps_row[1],
        "exc_min_int_yi": exc,
        "inc_min_int_yi": inc,
        "minority_int_yi": minority,
        "oth_eqt_tools_yi": oth,
        "total_share": share,
        "db_bps": bps_row[0],
        "verdict": compare_bps_with_db(bps_row[0], exc, oth, share),
        "identity": equity_identity_guard(inc, exc, minority),
        "pb": snapshot_pb_guard(*(si if si else (None, None, None))),
    }


def check_equity_caliber(con: sqlite3.Connection) -> tuple[list[str], list[str]]:
    """检查 3：每股净资产反推（U6 C 式）+ 其他权益工具敞口 + 两守卫。

    返回 (msgs, warns) 两级：C 式对账失败/缺行/缺列为阻断级（exit 1）；
    A 败 C 胜族标记与两守卫违例为记录级——那是数据形态事实（快照混日期、
    权益切分混装族），修复动作在数据线，不在本工具，不与市值修复 SQL 混提示。
    """
    require_oth_eqt_tools_column(con)
    msgs: list[str] = []
    warns: list[str] = []
    print()
    print("=" * 72)
    print("检查 3：每股净资产反推（普通股权益口径）+ 两守卫")
    print(f"        C 式 = {U6_FORMULA_C}")
    print("=" * 72)
    print(f"{'标的':<26}{'其他权益工具亿':>14}{'C式隐含':>10}{'db_bps':>10}"
          f"{'relC':>10}{'relA(对照)':>12}  判定")
    print("-" * 72)
    for code, _ts, name in PROBES:
        try:
            p = probe_equity_row(con, code)
        except Exception as exc:  # noqa: BLE001 — 单票异常逐票捕获归 msgs，不许断全轮
            msgs.append(
                f"{name} {code}：探针异常，已逐票捕获（{type(exc).__name__}: {exc}）"
            )
            continue
        if p["status"] != "ok":
            msgs.append(f"{name} {code}：{p['status']}")
            print(f"{name:<26}{'-':>14}{'-':>10}{'-':>10}{'-':>10}{'-':>12}  ✗ {p['status']}")
            continue
        v = p["verdict"]
        oth = p["oth_eqt_tools_yi"]
        mark = "✓" if v["pass_c"] else "✗ C 式仍败"
        if v["family"]:
            mark += f"｜{v['family']}"
        print(f"{name:<26}{(oth or 0.0):>14,.2f}{v['implied_c']:>10,.4f}"
              f"{p['db_bps']:>10,.4f}{v['rel_c']:>10.2e}{v['rel_a']:>12.2e}  {mark}")
        if not v["pass_c"]:
            msgs.append(
                f"{name} {code}：C 式反推 {v['implied_c']:.4f} vs db_bps {p['db_bps']}，"
                f"rel {v['rel_c']:.2%} 超 1% —— 未知族（UNKNOWN），单列追踪不填空"
            )
        if v["family"]:
            warns.append(
                f"{name} {code}：{v['family']}（A 式 rel {v['rel_a']:.2%} 败 / C 式胜；"
                f"oth_eqt_tools={oth:,.2f} 亿，扣口径后收敛 {v['rel_c']:.2e}）"
            )
        g = p["identity"]
        if not g["checkable"]:
            warns.append(f"{name} {code}：恒等式不可检（缺 {g['missing']}）")
        elif not g["pass"]:
            warns.append(
                f"{name} {code}：恒等式 inc−exc=minority 违例，残差 {g['residual_yi']:.4f} 亿"
                f"（容差 ±{g['tol_yi']}）—— 权益切分混装嫌疑"
            )
        g = p["pb"]
        if not g["checkable"]:
            warns.append(f"{name} {code}：快照自洽不可检（缺 {g['missing']}）")
        elif not g["pass"]:
            warns.append(
                f"{name} {code}：price/bps={g['ratio']:.5f} vs pb 快照值偏离 {g['rel_err']:.2%}"
                f"（复合容差 ±{g['tol']:.4f}）—— stock_info 快照混日期嫌疑"
            )
    return msgs, warns


# ── 市值量纲修复 SQL（幂等版）───────────────────────────────────
# 旧提示 `UPDATE stock_info SET mktcap *= 1e4` 是整列乘法：重跑一次就再放大
# 1e4，修复工具自己就是事故源。幂等版带量纲形态条件：每行只在「当前确为亿元
# 形态」（亿元形态反推价=价格、万元形态反推价≠价格）时换算一次，换算后条件不再成立。
FIX_MKTCAP_UNIT_SQL = """
UPDATE stock_info
   SET mktcap = mktcap * 1e4,
       float_mktcap = float_mktcap * 1e4
 WHERE price > 0 AND shares > 0 AND mktcap > 0
   AND ABS(mktcap / shares - price) / price < 0.01
   AND ABS(mktcap / 1e4 / shares - price) / price >= 0.01
"""


def stock_info_fix_copy(con: sqlite3.Connection) -> sqlite3.Connection:
    """把 stock_info 修复相关列拷进内存可写副本（真库只读，修复试验在副本上做）。"""
    mem = sqlite3.connect(":memory:")
    mem.execute(
        "CREATE TABLE stock_info (code TEXT PRIMARY KEY, mktcap REAL,"
        " float_mktcap REAL, price REAL, shares REAL)"
    )
    mem.executemany(
        "INSERT INTO stock_info(code, mktcap, float_mktcap, price, shares)"
        " VALUES (?,?,?,?,?)",
        con.execute("SELECT code, mktcap, float_mktcap, price, shares FROM stock_info"),
    )
    mem.commit()
    return mem


def apply_fix_sql(con: sqlite3.Connection) -> int:
    """执行幂等修复 SQL，返回实际变动行数。con 需可写（用副本，不碰真库）。"""
    cur = con.execute(FIX_MKTCAP_UNIT_SQL)
    con.commit()
    return cur.rowcount


def assert_fix_sql_idempotent(con: sqlite3.Connection) -> dict:
    """幂等断言：修复 SQL 第二次执行必须零变动、表内容逐行不变。

    修复工具自身必须先过幂等断言（UNITS U3 注：修复工具本身要做幂等断言），
    否则按一次修复口径写的运维动作重跑就是二次事故。
    """
    first = apply_fix_sql(con)
    once = sorted(con.execute("SELECT code, mktcap, float_mktcap FROM stock_info"))
    second = apply_fix_sql(con)
    twice = sorted(con.execute("SELECT code, mktcap, float_mktcap FROM stock_info"))
    if twice != once or second != 0:
        raise AssertionError(
            f"修复 SQL 非幂等：重跑仍变动 {second} 行 —— 禁止作为修复手段使用"
        )
    return {"first_run_rows": first, "second_run_rows": second, "idempotent": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tushare", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    msgs: list[str] = check_self_consistency(con)
    if not args.skip_tushare:
        msgs += check_against_tushare(con)
    u6_msgs, u6_warns = check_equity_caliber(con)
    msgs += u6_msgs
    g4_report, g4_msgs, g4_warns = g4_identity_check(con)
    msgs += g4_msgs
    # 修复 SQL 幂等断言常驻自检（试验只在内存副本上做，真库全程只读）
    fix_probe = assert_fix_sql_idempotent(stock_info_fix_copy(con))
    con.close()

    print()
    print("=" * 72)
    print(f"修复 SQL 幂等断言：副本首跑变动 {fix_probe['first_run_rows']} 行、"
          f"重跑变动 {fix_probe['second_run_rows']} 行 → 重跑结果不变 ✓")
    print(g4_identity_summary(g4_report))
    for w in u6_warns + g4_warns:
        print(f"  ⚠ 记录（不阻断）：{w}")
    if msgs:
        print(f"✗ 发现 {len(msgs)} 个问题：")
        for m in msgs:
            print(f"  · {m}")
        print()
        print("处理方式：不要猜单位。执行下面任一方案后重跑本脚本——")
        print("  A. 若本机市值是亿元：执行下方【幂等】修复 SQL（带量纲形态条件，")
        print("     每行只在确为亿元形态时换算一次，重跑零变动）：")
        for line in FIX_MKTCAP_UNIT_SQL.strip().splitlines():
            print(f"     {line}")
        print("  B. 若 Tushare 侧需换算：在 merge脚本中统一，不要在下游各自换")
        print("  C. 权益口径/守卫类记录（⚠ 打头）按点名单逐票核对报表期与快照日，不要猜口径重算")
        print("=" * 72)
        return 1

    print(f"✓ 量纲校验通过，市值单位统一为 {MARKET_CAP_UNIT}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
