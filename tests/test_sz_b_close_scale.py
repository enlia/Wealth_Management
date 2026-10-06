"""深 B 价格缩放根因修复（UNITS U7）与迁移工具的回归测试。

为什么必须有这个文件
-------------------
UNITS U7 实测：``bar_daily`` 深市 B 股（sz200/201 段） close 小 10 倍，
量纲恒等式 ``amount/(vol×close)`` 深 B 中位 9.95、其余品种 ≈1.00；
跌停判定因此对深 B 产生 2023 年 9,189 格假封（涨停侧 0 格=单侧假阳性）。
根因在 ``tdx.price_scale()`` 把深 2x B 股与基金/债券同归 0.001，
而深 B 的行情整数实为 ×0.01 缩放（三方对拍：Tushare 权威限价 114/114、
通达信导出物 stock_info.price/close=10.00、G4 恒等式 9.95→0.995）。
价格系数错 10 倍不抛异常、只静默污染一切下游 —— 必须用测试钉死
（含还原必败：把 price_scale 的深 B 系数改回 0.001，本文件立即转红）。

⚠️ 读法约定：``price_scale`` 与 ``read_day`` 契约一致，传 **6 位数字代码**
（带前缀的 8 位码由 read_day 内部先剥前缀），与 build_sqlite 调用法一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tools"))

import fix_sz_b_close_scale as fix  # noqa: E402
from tdx import price_scale  # noqa: E402


class TestPriceScale深B根因:
    """深 B 系数必须是 ×0.01（还原 0.001 必败）。"""

    @pytest.mark.parametrize(
        "code",
        [
            "200011",   # 深物业B（sz200 段）
            "200017",   # 深中华B
            "200488",   # ST晨鸣B（5% 名目限幅，兼封板影响面样本）
            "201872",   # 招港B（sz201 段——is_b_share 旧版漏判的段）
            "202001",   # sz202 段：无实例（段内族归一；异例由 G4 门兜底）
        ],
    )
    def test_深B系数为01(self, code: str) -> None:
        assert price_scale(code, mkt="sz") == 0.01

    @pytest.mark.parametrize(
        ("code", "mkt", "expect"),
        [
            ("900901", "sh", 0.001),   # 沪 B 保持 0.001（900948 实测样例同族）
            ("900933", "sh", 0.001),   # 沪 B
            ("113050", "sh", 0.001),   # 沪市可转债（不动）
            ("159901", "sz", 0.001),   # 深市基金 1x（不动）
            ("600519", "sh", 0.01),    # A 股（对照，×10 后限价对拍必失中）
            ("000001", "sz", 0.01),    # 深市 A 股
            ("399001", "sz", 0.01),    # 深证指数
            ("880507", "sh", 0.01),    # 板块指数
            ("430047", "bj", 0.01),    # 北交所
        ],
    )
    def test_其他品种系数不变(self, code: str, mkt: str, expect: float) -> None:
        assert price_scale(code, mkt=mkt) == expect


class TestG4量纲恒等式判据:
    """G4 门固定项：amount/(vol×close)；阈值照抄 UNITS U7②（>1.5 或 <0.67 疑污染）。"""

    def test_自洽形态判ok(self) -> None:
        assert fix.g4_ratio(amount=99500.0, vol=10000.0, close=10.0) == pytest.approx(0.995)
        assert fix.g4_verdict(0.995) == "ok"
        assert fix.g4_verdict(1.5) == "ok"          # 阈值本身不判污染（严格不等式）
        assert fix.g4_verdict(0.67) == "ok"

    @pytest.mark.parametrize("ratio", [9.95, 0.099, 1.500001, 0.669999])
    def test_污染形态判suspect(self, ratio: float) -> None:
        # 9.95=深 B 10× 净痕；0.099=U5 退市股 1000×/100× 净痕
        assert fix.g4_verdict(ratio) == "suspect"

    def test_缺值返回None(self) -> None:
        assert fix.g4_ratio(amount=0.0, vol=10.0, close=10.0) is None
        assert fix.g4_ratio(amount=1.0, vol=0.0, close=10.0) is None
        assert fix.g4_verdict(None) == "unknown"    # 缺值禁填空（UNKNOWN，不猜）


class Test迁移SQL幂等:
    """SQL 生成的三道防线：PRECHECK、污染谓词圈行（复跑 0 命中）、POSTCHECK。"""

    def test_污染谓词自愈幂等(self) -> None:
        # 首跑：比值 9.95 = 99500/(10000×1.0) 命中污染区间 → 需修；
        # ×10 后 close=10.0 → 比值 0.995 → 复跑不命中（同格前/后自愈）
        assert fix.is_sz_b_contaminated(code="sz200011", amount=99500.0,
                                        vol=10000.0, close=1.0) is True
        assert fix.is_sz_b_contaminated(code="sz200011", amount=99500.0,
                                        vol=10000.0, close=10.0) is False

    def test_非深B不命中(self) -> None:
        assert fix.is_sz_b_contaminated(code="sh600519", amount=10.0,
                                        vol=1.0, close=10.0) is False

    def test_SQL含三道防线且不动close_adj(self) -> None:
        sql = fix.build_migration_sql(
            daily_pairs=[("sz201872", 20230103, 0.815)],
            weekly_pairs=[("sz200011", 20230106, 0.59)],
        )
        assert "PRECHECK" in sql and "POSTCHECK" in sql
        assert "amount / (vol * close) BETWEEN 6.0 AND 15.0" in sql   # U7 污染区间
        assert "close_adj" not in sql       # B 股 close_adj 全历史 0 覆盖，无行可迁
        assert "mode=ro" not in sql         # SQL 交数据线执行（本工具绝不带写库语义出参）
        assert sql.count("UPDATE bar_weekly") == 2   # 污染谓词行 + 原值三键行
        assert sql.count("UPDATE bar_daily") == 2
