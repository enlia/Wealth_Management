"""G4 量纲恒等式探针（units_probes.g4_identity_report）回归测试。

为什么必须有这个文件
-------------------
U5 实录用 `amount/(vol×close)` 比值直方图逮到退市股补数 0.099 净痕、
U7 逮到深 B 10 倍量纲 9.98 净痕 —— 该恒等式是量纲污染的统一逮法（G4 门固定项）。
探针语义钉死三件事：① U7② 门阈值经 fix_sz_b_close_scale.g4_verdict **单源传递**
（阈值哑化——把 0.67/1.5 改宽——本文件 test_U7门阈值语义…必 FAILED）；
② 缺值=UNKNOWN 禁填空（UNKNOWN 填 0 折进统计——test_缺值UNKNOWN…必 FAILED）；
③ 深 B 组单列不被大盘稀释（并组后 test_深B组单列…必 FAILED，样例按
4×9.98 对 5×1.0 配比设计使合并中位 1.0 与单列 9.98 可分辨）。
已知净痕锚：深 B 修复 SQL 落库前读数中位≈9.98 → suspect 并报 U7 同族；
落库后读数中位≈0.998 → ok（probecase 双语义各一钉）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tools"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from units_probes import (  # noqa: E402
    G4_GROUP_ORDER,
    G4_IDENTITY_FORMULA,
    g4_group,
    g4_identity_report,
)


def rows(*specs):
    """(code, amount, vol, close) 四元组行；比值 = amount/(vol×close)。"""
    return [(c, a, v, cl) for c, a, v, cl in specs]


class Test净痕锚:
    def test_落库前998判suspect并报U7同族(self) -> None:
        r = g4_identity_report(rows(
            ("sz200011", 99800.0, 10000.0, 1.0),   # r = 9.98（U7 深 B 10× 净痕）
            ("sz201872", 99800.0, 10000.0, 1.0),
            ("sz200017", 99800.0, 10000.0, 1.0),
            ("sz200488", 99800.0, 10000.0, 1.0),
        ))
        g = r["groups"]["深B"]
        assert g["n_valid"] == 4
        assert g["median"] == pytest.approx(9.98)
        assert g["verdict"] == "suspect"          # U7② 门：>1.5 疑量纲污染
        assert r["sz_b_alert"] == "U7 同族污染"    # 深 B 组中位 >1.5 → 报 U7 同族

    def test_落库后0998判ok不报警(self) -> None:
        r = g4_identity_report(rows(
            ("sz200011", 9980.0, 10000.0, 1.0),    # r = 0.998（深 B 修复后中位锚）
            ("sz201872", 9980.0, 10000.0, 1.0),
            ("sz200017", 9980.0, 10000.0, 1.0),
            ("sz200488", 9980.0, 10000.0, 1.0),
        ))
        g = r["groups"]["深B"]
        assert g["median"] == pytest.approx(0.998)
        assert g["verdict"] == "ok"
        assert r["sz_b_alert"] == "无"             # 组中位 ≤1.5 不报


class Test缺值UNKNOWN禁填空:
    def test_缺值行不入组统计且单列UNKNOWN(self) -> None:
        r = g4_identity_report(rows(
            ("sh600519", 99500.0, 10000.0, 10.0),  # r = 0.995 好行
            ("sh600519", 0.0, 10000.0, 10.0),      # amount=0 → UNKNOWN
            ("sz000001", 99500.0, 0.0, 10.0),      # vol=0 → UNKNOWN
            ("sz000001", 99500.0, 10000.0, 10.0),  # 好行
        ))
        assert r["n_rows"] == 4
        assert r["n_unknown"] == 2                # 缺值单列、禁填 0 折进统计
        g = r["groups"]["A股等"]
        assert g["n_valid"] == 2                  # 中位只据好行
        assert g["median"] == pytest.approx(0.995)
        assert g["median"] > 0.5                  # 填 0 形态（(0.995+0)/2≈0.5）必不成立


class Test深B组单列:
    def test_四组出全且深B不被大盘稀释(self) -> None:
        r = g4_identity_report(rows(
            ("sz200011", 99800.0, 10000.0, 1.0), ("sz200012", 99800.0, 10000.0, 1.0),
            ("sz200017", 99800.0, 10000.0, 1.0), ("sz200488", 99800.0, 10000.0, 1.0),
            ("sh600519", 100000.0, 10000.0, 1.0), ("sz000001", 100000.0, 10000.0, 1.0),
            ("sz000002", 100000.0, 10000.0, 1.0), ("sh600000", 100000.0, 10000.0, 1.0),
            ("sz002594", 100000.0, 10000.0, 1.0),
        ))
        assert set(G4_GROUP_ORDER) == {"A股等", "深B", "沪B", "基金/债"}
        assert G4_IDENTITY_FORMULA == "amount/(vol×close)"
        assert set(r["groups"]) <= set(G4_GROUP_ORDER)
        assert r["groups"]["深B"]["median"] == pytest.approx(9.98)   # 单列不稀释
        # 可分辨性：9 行全并（4×9.98 + 5×1.0）中位=1.0 ≠ 单列 9.98 → 并组变异必红
        merged = sorted([9.98] * 4 + [1.0] * 5)
        assert merged[4] == 1.0 and r["groups"]["深B"]["median"] == pytest.approx(9.98)


class TestU7门阈值语义_阈值哑化必红:
    def test_U7门阈值语义经g4_verdict单源传递(self) -> None:
        """0.67/1.5 严格不等式边界 + 双净痕 + UNKNOWN（阈值哑化→本用例必 FAILED）。"""
        r = g4_identity_report(rows(
            ("sh600519", 1.5, 1.0, 1.0),      # r = 1.5 阈值本身 → ok（严格不等式）
            ("sz000001", 0.67, 1.0, 1.0),     # r = 0.67 阈值本身 → ok
            ("sh601318", 9.98, 1.0, 1.0),     # U7 净痕 → suspect
            ("sz300750", 0.099, 1.0, 1.0),    # U5 净痕（0.1×）→ suspect
            ("sh600036", None, 1.0, 1.0),     # 缺值 → unknown
        ))
        per = r["row_verdicts"]
        assert per[0] == "ok" and per[1] == "ok"
        assert per[2] == "suspect" and per[3] == "suspect"
        assert per[4] == "unknown"            # UNKNOWN 禁猜


class Test分组同义对拍:
    def test_分组与迁移工具同义_7码(self) -> None:
        assert g4_group("sz200011") == "深B"
        assert g4_group("sz201872") == "深B"    # 201 段（招港B）
        assert g4_group("sh900901") == "沪B"
        assert g4_group("sz159901") == "基金/债"
        assert g4_group("sh113050") == "基金/债"
        assert g4_group("sz000001") == "A股等"
        assert g4_group("sh600519") == "A股等"


class Test全缺组UNKNOWN契约:
    """组形态 UNKNOWN 契约（设定四件套之「描述 + 测试」件）：

    全缺组（无有效行）→ median=None、verdict="unknown"、深 B 告警="UNKNOWN"，
    **不填 0 不冒算**（units_probes.g4_identity_report 的 else 分支实现对，
    本用例把 docstring 契约钉成断言 —— 后人把 else 分支改填 0 或「ok」必 FAILED）。
    """

    def test_全缺组UNKNOWN禁填空_契约(self) -> None:
        r = g4_identity_report(rows(
            ("sz200011", 0.0, 10000.0, 1.0),      # 深 B 全缺值形态四行
            ("sz200012", 0.0, 10000.0, 1.0),
            ("sz200017", None, 1.0, 1.0),
            ("sz200488", 0.0, 0.0, 1.0),
        ))
        g = r["groups"]["深B"]
        assert g["n_valid"] == 0 and g["n_unknown"] == 4
        assert g["median"] is None               # UNKNOWN 禁填 0
        assert g["verdict"] == "unknown"         # 全缺组不判 ok / suspect
        assert r["sz_b_alert"] == "UNKNOWN"      # 组不可检：不猜、不报


# ── 还原必败变异记账（独立三文件全集口径，2026-10-13 实测）【发布待双闸】──
# 阈值哑化（_G4_HIGH / _G4_LOW 改宽）= 8 红：本文件 2 + test_units_g4_identity 2
#   + test_sz_b_close_scale 4；深 B 并组（g4_group 深B→A股等）= 4 红。
# 早前提交所记 6 / 3 系两文件叠加差集口径，以本独立口径为准。
