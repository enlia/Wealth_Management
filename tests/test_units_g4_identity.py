"""G4 门量纲恒等式固定项（UNITS U5 逮法 / U7② 阈值）与深 B 判定跨面一致性。

check_units 探针清单提案行（本文件物化存证；方案联动 `.planning/reports/
fix-sz-b-share-close-volume.md`。**check_units.py 本体不动** —— 该文件处于
B13 扩 6 缝合期，按防撞条款本单只交测试与提案行，接入由 check_units 单落）::

    PROBE_G4_IDENTITY = 「amount/(vol×close) 量纲恒等式：分组（A股等 / 基金债 /
    沪B / 深B）中位 + 行级比值直方图，过 UNITS U7② 门（比值 >1.5 或 <0.67 疑
    量纲污染；缺值=UNKNOWN 禁填空）；深 B 组（is_b_share）单列，组中位 >1.5
    即报 U7 同族污染」
    —— 逮法出处 U5 实录（0.099=1000×/100× 净痕、9.98=10× 净痕，直方图一眼分家）。

G4 判据实现面在 src/tools/fix_sz_b_close_scale.py（工具自含守卫）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tools"))

import fix_sz_b_close_scale as fix  # noqa: E402
from tdx import price_scale  # noqa: E402

from factor_lab.config import is_b_share  # noqa: E402

PROPOSAL_G4_PROBE = (
    "check_units 探针清单提案行（G4 门固定项）：新增 PROBE_G4_IDENTITY = "
    "『amount/(vol×close) 量纲恒等式：分组（A股等/基金债/沪B/深B）中位 + "
    "行级比值直方图，过 UNITS U7② 门（比值 >1.5 或 <0.67 疑量纲污染；"
    "缺值=UNKNOWN 禁填空）』；深 B 组（is_b_share）单列，深 B 组中位 >1.5 "
    "即报 U7 同族污染。逮法出处=U5 实录比值直方图法。"
)


class TestG4门固定项语义:
    """阈值边界矩阵：UNITS U7② 原文 0.67/1.5 严格不等式 + 两个实测净痕。"""

    def test_阈值边界与实测净痕(self) -> None:
        assert fix.g4_verdict(0.67) == "ok"        # 阈值本身不判污染
        assert fix.g4_verdict(1.5) == "ok"
        assert fix.g4_verdict(0.669999) == "suspect"
        assert fix.g4_verdict(1.500001) == "suspect"
        assert fix.g4_verdict(9.9802) == "suspect"   # 深 B 修复前实测中位（10× 净痕）
        assert fix.g4_verdict(0.099) == "suspect"    # U5 退市股实录净痕（0.1× 净痕）
        assert fix.g4_verdict(None) == "unknown"     # 缺值禁填空（UNKNOWN）

    def test_修复前后回归口径固化(self) -> None:
        # 同一条深 B 行：修复前比值 9.98（suspect）→ ×10 后 0.998（ok）
        before = fix.g4_ratio(amount=998020.0, vol=10000.0, close=10.0)
        after = fix.g4_ratio(amount=998020.0, vol=10000.0, close=100.0)
        assert (fix.g4_verdict(before), fix.g4_verdict(after)) == ("suspect", "ok")


class Test提案行存证:
    def test_提案文本含判据关键字段(self) -> None:
        for kw in ("amount/(vol×close)", "0.67", "1.5", "深B", "UNKNOWN", "U7"):
            assert kw in PROPOSAL_G4_PROBE


class Test深B判定跨面一致:
    """三处深 B 判定同域：config.is_b_share ∩ sz20x ≡ fix.is_sz_b_share
    ≡ tdx.price_scale 0.01 域内深 B 集合（防两处代码段判定各改各的）。"""

    SZB = ["sz200011", "sz200017", "sz200488", "sz201872"]
    # sh900901 不在此列：沪 B 属 is_b_share 全集、但不属深 B 子集（下条单测钉）
    OTHERS = ["sz000029", "sz159901", "sh600519", "sz300750", "sh113050"]

    def test_深B集合三判定一致(self) -> None:
        for c in self.SZB:
            assert is_b_share(c) is True
            assert fix.is_sz_b_share(c) is True
            assert price_scale(c[2:], mkt="sz") == 0.01   # 深 B 系数域（U7）
        for c in self.OTHERS:
            assert is_b_share(c) is False
            assert fix.is_sz_b_share(c) is False

    def test_沪B在B股域但不在深B域_不同纲对照(self) -> None:
        assert is_b_share("sh900901") is True          # B 股全集含沪 B
        assert fix.is_sz_b_share("sh900901") is False  # 深 B 子集不含
        assert price_scale("900901", mkt="sh") == 0.001   # 沪 B 仍 ×0.001（U7）
