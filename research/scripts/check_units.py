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

# 抽查标的：沪深主板 + 创业板 + 保险 + 宁德，覆盖不同量级；
# 其他权益工具口径项入探针清单（U6）：再加两只其他权益工具存量大户
# （market.db ts_balance_sheet@20260630 实测 oth_eqt_tools：招商银行
#   1999.89 亿、农业银行 4700.0 亿；sz000001 自带 800.0 亿），
# 探针 4→6 覆盖「有/无其他权益工具」两种权益切分形态。
PROBES = [
    ("sh600519", "600519.SH", "贵州茅台（大盘蓝筹）"),
    ("sz000001", "000001.SZ", "平安银行（深主板）"),
    ("sh601318", "601318.SH", "中国平安（保险）"),
    ("sz300750", "300750.SZ", "宁德时代（创业板）"),
    ("sh600036", "600036.SH", "招商银行（其他权益工具大户）"),
    ("sh601288", "601288.SH", "农业银行（其他权益工具大户）"),
]

TOL = 0.01          # 1% 容差（Tushare 与通达信都只保留有限小数）


# ── 每股净资产反推口径：普通股权益（UNITS U6）────────────────────
# 「每股净资产 = 归母权益 ÷ 总股本」是错的：归母权益可能含其他权益工具
# （优先股/永续债），而 bps 是每股【普通股】净资产，分子必须扣除其他权益工具：
#     C 式 implied_bps = (归母 − COALESCE(oth_eqt_tools, 0)) × 1e8 / 总股本  ← 固定采用
#     A 式 implied_bps = 归母 × 1e8 / 总股本                                ← 仅对照（不扣旧口径）
# A 败 C 胜 = 其他权益工具口径混装（实证：sz000001 的 800 亿其他权益工具未扣
# → 反推偏差 17.09%，扣除后 1e-7 级闭合；见
#   .planning/reports/a18-sz000001-bps-forensics.md）。
# 金额口径：ts_ 表亿元（U2）、ts_balance_sheet.total_share 为股（U3）——×1e8 换成元。
U6_FORMULA_C = (
    "(total_hldr_eqy_exc_min_int - COALESCE(oth_eqt_tools, 0)) × 1e8 / total_share"
)
FAMILY_OTH_EQT_TOOLS = "其他权益工具族"


def ordinary_equity_yi(exc_min_int_yi, oth_eqt_tools_yi):
    """C 式分子：归属母公司普通股股东权益（亿元）= 归母 − COALESCE(oth_eqt_tools, 0)。

    oth_eqt_tools 行内 NULL = 合法「无其他权益工具」，按 COALESCE 语义归 0（U6 明文）；
    归母权益为 NULL 属数据缺失，显式抛错，不静默按 0 反推。
    """
    if exc_min_int_yi is None:
        raise ValueError(
            "归属母公司股东权益合计为 NULL —— 无法反推每股普通股净资产，先补数再对账"
        )
    oth = 0.0 if oth_eqt_tools_yi is None else float(oth_eqt_tools_yi)
    return float(exc_min_int_yi) - oth


def implied_bps_c(exc_min_int_yi, oth_eqt_tools_yi, total_share):
    """C 式（U6 固定式）：`(归母 − 其他权益工具) × 1e8 / 总股本`，返回元/股。"""
    if not total_share:
        raise ValueError(f"总股本缺失或为 0（total_share={total_share!r}）—— 反推分母不可用")
    return ordinary_equity_yi(exc_min_int_yi, oth_eqt_tools_yi) * 1e8 / float(total_share)


def implied_bps_a(exc_min_int_yi, total_share):
    """A 式（对照列，旧口径）：分子不扣其他权益工具的 `归母 × 1e8 / 总股本`。"""
    if not total_share:
        raise ValueError(f"总股本缺失或为 0（total_share={total_share!r}）—— 反推分母不可用")
    return float(exc_min_int_yi) * 1e8 / float(total_share)


def compare_bps_with_db(db_bps, exc_min_int_yi, oth_eqt_tools_yi, total_share) -> dict:
    """C 式对账 db_bps + A 式对照 + 口径混装族标记。

    A 式败（rel ≥ 1%）而 C 式胜 → family = 其他权益工具族（口径混装自动归因）；
    两式同判则 family=None。C 式仍败属未知族，只记数不归因（UNKNOWN，禁止填空）。
    """
    if not db_bps:
        raise ValueError(f"db_bps 缺失或为 0（{db_bps!r}）—— 无对账基准")
    ic = implied_bps_c(exc_min_int_yi, oth_eqt_tools_yi, total_share)
    ia = implied_bps_a(exc_min_int_yi, total_share)
    rel_c = abs(ic - float(db_bps)) / abs(float(db_bps))
    rel_a = abs(ia - float(db_bps)) / abs(float(db_bps))
    pass_c = rel_c < TOL
    pass_a = rel_a < TOL
    return {
        "implied_c": ic, "rel_c": rel_c, "pass_c": pass_c,
        "implied_a": ia, "rel_a": rel_a, "pass_a": pass_a,
        "oth_eqt_tools_yi": oth_eqt_tools_yi,
        "family": FAMILY_OTH_EQT_TOOLS if (pass_c and not pass_a) else None,
    }


def require_oth_eqt_tools_column(con: sqlite3.Connection) -> None:
    """显式校验 ts_balance_sheet 存在 oth_eqt_tools 列（C 式分子的前提）。

    缺列与行内 NULL 是两回事：NULL=合法「无其他权益工具」按 COALESCE 归 0；
    缺列=库太旧、无法执行 C 式扣减，必须抛错，禁止静默按 0 绕过（口径会再次错位）。
    """
    tables = {
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ts_balance_sheet'"
        )
    }
    if "ts_balance_sheet" not in tables:
        raise RuntimeError("ts_balance_sheet 表不存在 —— C 式反推无法执行")
    cols = {r[1] for r in con.execute("PRAGMA table_info(ts_balance_sheet)")}
    if "oth_eqt_tools" not in cols:
        raise RuntimeError(
            "ts_balance_sheet 缺 oth_eqt_tools 列 —— C 式分子无法扣除其他权益工具，"
            "禁止静默按 0 处理（会让含优先股/永续债的票反推偏差 17% 量级且不报错）"
        )


# ── U6 两守卫（权益切分恒等式 + 快照自洽），容差显式 ───────────────
# ① 恒等式：含少数股东权益 − 归母权益 = 少数股东权益。数值为亿元；
#    补/调整行（update_flag=1）三分量各带 ±0.005 的两位小数舍入 → 残差界
#    ±0.015，容差收 0.02。
# ② 自洽：price / bps ≈ pb。pb 为两位小数（舍入界 <0.1%），容差收 1%；
#    超限即 price/市值 与 bps/pb 属不同日期快照混装（快照混日期的露馅点）。
IDENTITY_ABS_TOL_YI = 0.02
PB_REL_TOL = 0.01


def equity_identity_guard(inc_min_int_yi, exc_min_int_yi, minority_int_yi) -> dict:
    """守卫①：`含少数 − 归母 = 少数股东权益`（亿元，容差 IDENTITY_ABS_TOL_YI）。

    任一输入为 NULL → checkable=False 并点名缺哪项，不冒算不静默归 0
    （实测 sz000001 的 minority_int 为 NULL：inc==exc，恒等式不可检）。
    """
    fields = (
        ("total_hldr_eqy_inc_min_int", inc_min_int_yi),
        ("total_hldr_eqy_exc_min_int", exc_min_int_yi),
        ("minority_int", minority_int_yi),
    )
    missing = [name for name, v in fields if v is None]
    if missing:
        return {"checkable": False, "missing": missing, "tol_yi": IDENTITY_ABS_TOL_YI}
    residual = float(inc_min_int_yi) - float(exc_min_int_yi) - float(minority_int_yi)
    return {
        "checkable": True,
        "missing": [],
        "residual_yi": residual,
        "tol_yi": IDENTITY_ABS_TOL_YI,
        "pass": abs(residual) <= IDENTITY_ABS_TOL_YI,
    }


def snapshot_pb_guard(price, bps, pb) -> dict:
    """守卫②：`price / bps ≈ pb`（相对容差 PB_REL_TOL）。

    超限 = stock_info 的 price/市值 与 bps/pb 不是同一日期快照（混日期露馅）。
    任一输入 NULL / bps≤0 / pb≤0 → checkable=False 并点名缺项，不冒算。
    """
    fields = (("price", price), ("bps", bps), ("pb", pb))
    missing = [name for name, v in fields if v is None]
    if missing:
        return {"checkable": False, "missing": missing, "tol": PB_REL_TOL}
    if float(bps) <= 0 or float(pb) <= 0:
        return {
            "checkable": False,
            "missing": [n for n, v in (("bps", bps), ("pb", pb)) if float(v) <= 0],
            "tol": PB_REL_TOL,
        }
    ratio = float(price) / float(bps)
    rel_err = abs(ratio - float(pb)) / abs(float(pb))
    return {
        "checkable": True,
        "missing": [],
        "ratio": ratio,
        "rel_err": rel_err,
        "tol": PB_REL_TOL,
        "pass": rel_err <= PB_REL_TOL,
    }


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
        p = probe_equity_row(con, code)
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
                f"（容差 {g['tol']:.0%}）—— stock_info 快照混日期嫌疑"
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
    # 修复 SQL 幂等断言常驻自检（试验只在内存副本上做，真库全程只读）
    fix_probe = assert_fix_sql_idempotent(stock_info_fix_copy(con))
    con.close()

    print()
    print("=" * 72)
    print(f"修复 SQL 幂等断言：副本首跑变动 {fix_probe['first_run_rows']} 行、"
          f"重跑变动 {fix_probe['second_run_rows']} 行 → 重跑结果不变 ✓")
    for w in u6_warns:
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
