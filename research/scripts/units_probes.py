"""每股净资产反推口径（普通股权益，UNITS U6）与 U6 两守卫——从 check_units.py 纯搬移拆出。

拆分动机：check_units.py 516 行越过「代码文件 ≤ 500 行」硬线（CI 行数门
口径 = splitlines 含空行），按 400~500 强制拆分铁律把内聚的「公式族 + 守卫族」
搬出。本模块为**纯搬移、语义零变**：函数体与常量逐字来自 check_units.py，
原 import 名经 check_units 再导出兼容（含 runtime 案卷区门脚本的调用面）。

内容面（三块）：
  1. U6 反推式：C 式固定 `(归母 − COALESCE(oth_eqt_tools, 0)) × 1e8 / 总股本`，
     A 式仅作对照列；A 败 C 胜自动标记其他权益工具族。
  2. 守卫①：权益切分恒等式 `含少数 − 归母 = 少数股东权益`（容差 0.02 亿元）。
  3. 守卫②：快照自洽 `price / bps ≈ pb`（复合容差 max(0.0051, 0.01×pb)，绝对量纲）。

用法（经 check_units 再导出，原名不变）：
  from check_units import implied_bps_c, snapshot_pb_guard, ...
"""
from __future__ import annotations

import sqlite3

TOL = 0.01          # 1% 对账容差（Tushare 与通达信都只保留有限小数）

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
# ② 自洽：price / bps ≈ pb。pb 两位小数的舍入界是绝对 ±0.005，相对界 = 0.005/pb
#    （pb=5 时约 0.1%、pb=0.48 时约 1.0%、pb=0.20 时约 2.5%——「舍入界 <0.1%」
#    的旧论证只对 pb≥5 成立，低 pb 票会被 1% 相对容差误拦）；容差取复合
#    max(0.0051, 0.01×pb)（绝对量纲）＝ max(两位小数舍入界, 1% 相对带)。
#    超限即 price/市值 与 bps/pb 属不同日期快照混装（快照混日期的露馅点）。
IDENTITY_ABS_TOL_YI = 0.02
PB_REL_TOL = 0.01
PB_ROUND_ABS_TOL = 0.0051


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
    """守卫②：`price / bps ≈ pb`（复合容差 max(0.0051, 0.01×pb)，绝对量纲）。

    超限 = stock_info 的 price/市值 与 bps/pb 不是同一日期快照（混日期露馅）。
    任一输入 NULL 或 ≤0（price≤0 覆盖停牌/退市形态、脏数据形态）→ checkable=False
    并点名缺项，不冒算。
    """
    fields = (("price", price), ("bps", bps), ("pb", pb))
    missing = [name for name, v in fields if v is None]
    if missing:
        return {"checkable": False, "missing": missing, "tol": PB_REL_TOL}
    nonpos = [name for name, v in fields if float(v) <= 0]
    if nonpos:
        return {
            "checkable": False,
            "missing": nonpos,
            "tol": PB_REL_TOL,
        }
    ratio = float(price) / float(bps)
    abs_dev = abs(ratio - float(pb))
    rel_err = abs_dev / abs(float(pb))
    tol = max(PB_ROUND_ABS_TOL, PB_REL_TOL * abs(float(pb)))
    return {
        "checkable": True,
        "missing": [],
        "ratio": ratio,
        "rel_err": rel_err,
        "abs_dev": abs_dev,
        "tol": tol,
        "pass": abs_dev <= tol,
    }


# ── G4 量纲恒等式探针（U5 逮法 · U7② 门）──────────────────────────
# PROBE_G4_IDENTITY（提案逐字接入；逮法出处 = U5 实录比值直方图双净痕
# 0.099=1000×/100×、9.98=10×，一眼分家）：
#   amount/(vol×close) 量纲恒等式：分组（A股等/基金债/沪B/深B）中位 +
#   行级比值直方图，过 UNITS U7② 门（比值 >1.5 或 <0.67 疑量纲污染；
#   缺值=UNKNOWN 禁填空）；深 B 组（is_b_share）单列，深 B 组中位过门
#   即报 U7 同族污染。
# **判据单源**：g4_ratio / g4_verdict / is_sz_b_share 复用
# src/tools/fix_sz_b_close_scale.py —— U7② 阈值（0.67/1.5）只在该处定义，
# 本块不重复造阈值（阈值哑化变异对两面同时生效，守护用例
# tests/test_units_probes_g4.py 逐条钉语义）。
import statistics  # noqa: E402
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_ROOT = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_ROOT / "src"))
_sys.path.insert(0, str(_ROOT / "src" / "tools"))
from fix_sz_b_close_scale import (  # noqa: E402
    g4_ratio,
    g4_verdict,
    is_sz_b_share,
)

G4_IDENTITY_FORMULA = "amount/(vol×close)"
G4_GROUP_ORDER = ("A股等", "深B", "沪B", "基金/债")
# 行级比值直方图分箱（仅展示；**门判只用 U7② 两阈值**）。界点取双净痕实录
# 形态位（0.2/5）、门界（0.67/1.5）与幂次净痕位（15）。
G4_HIST_LABELS = ("≤0.2", "0.2~0.67", "0.67~1.5 自洽", "1.5~5", "5~15", "≥15")


def g4_group(code: str) -> str:
    """品种分组（与 fix_sz_b_close_scale.g4_summary 分组同义；深 B 判定走单源）。"""
    c = str(code)
    n = c[2:] if len(c) == 8 and c[:2] in ("sh", "sz", "bj") else c
    if is_sz_b_share(c):
        return "深B"
    if c.startswith("sh900"):
        return "沪B"
    if n.startswith(("15", "16", "18")) or c.startswith(
            ("sh5", "sh1", "sz1", "sh11", "sh12")):
        return "基金/债"
    return "A股等"


def _hist_bucket(r: float) -> str:
    if r <= 0.2:
        return G4_HIST_LABELS[0]
    if r < 0.67:
        return G4_HIST_LABELS[1]
    if r <= 1.5:
        return G4_HIST_LABELS[2]
    if r < 5.0:
        return G4_HIST_LABELS[3]
    if r < 15.0:
        return G4_HIST_LABELS[4]
    return G4_HIST_LABELS[5]


def g4_identity_report(rows) -> dict:
    """G4 门探针：分组中位（U7② 门）+ 行级比值直方图；缺值=UNKNOWN 禁填空。

    rows : 可迭代 ``(code, amount, vol, close)`` 四元组。
    返回 : ``groups[组] = {n_valid, n_unknown, median, verdict}``（组无有效行时
           median=None、verdict="unknown"，**不填 0 不冒算**）、``row_verdicts``
           （行级判定序列，含 "unknown"）、``hist``（行级比值桶计数）、
           ``sz_b_alert``：深 B 组中位过门="U7 同族污染"、门内="无"、不可检
           （组无有效行）="UNKNOWN"。
    """
    hist = {k: 0 for k in G4_HIST_LABELS}
    vals: dict[str, list[float]] = {}
    unk: dict[str, int] = {}
    row_verdicts: list[str] = []
    n_rows = 0
    for code, amount, vol, close in rows:
        n_rows += 1
        r = g4_ratio(amount, vol, close)
        row_verdicts.append(g4_verdict(r))
        grp = g4_group(code)
        if r is None:
            unk[grp] = unk.get(grp, 0) + 1     # UNKNOWN 单列：禁填空、不入统计
            continue
        vals.setdefault(grp, []).append(r)
        hist[_hist_bucket(r)] += 1
    groups = {}
    for grp in G4_GROUP_ORDER:
        good = sorted(vals.get(grp, []))
        if good:
            med = statistics.median(good)
            groups[grp] = {
                "n_valid": len(good),
                "n_unknown": unk.get(grp, 0),
                "median": med,
                "verdict": g4_verdict(med),
            }
        else:
            groups[grp] = {
                "n_valid": 0,
                "n_unknown": unk.get(grp, 0),
                "median": None,
                "verdict": "unknown",
            }
    szb_med = groups["深B"]["median"]
    if szb_med is None:
        sz_b_alert = "UNKNOWN"                  # 组不可检：不猜、不报
    else:
        sz_b_alert = "U7 同族污染" if g4_verdict(szb_med) == "suspect" else "无"
    return {
        "formula": G4_IDENTITY_FORMULA,
        "n_rows": n_rows,
        "n_unknown": sum(unk.values()),
        "hist": hist,
        "row_verdicts": row_verdicts,
        "groups": groups,
        "sz_b_alert": sz_b_alert,
    }
