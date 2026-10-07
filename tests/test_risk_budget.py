"""风险预算模块测试（pytest）。

覆盖六组（对应用户验收清单 ①~⑤ + 冒烟守护）：
A. 期望值闸门：EV ≤ 0 被否决，且必须带理由
B. 三道硬上限：单票 / 单行业 / 总仓位各自生效（构造越限输入断言被裁到上限）
C. 缺输入一律抛错：空候选、空行业表、缺 ATR、缺申万映射 —— **不许静默**
D. 分数凯利：收缩系数=1 时退化为全凯利（对拍手算例子）
E. 建议卡字段完整性：缺任一字段即失败
F. ATR / 防前视：SQLite 合成库对拍手算 Wilder 与 SMA

⚠️ 测试里出现的 p 值一律是**明示的示例假设值**，不是实测胜率。
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from factor_lab.analysis.risk_budget import (
    AdviceCard,
    Candidate,
    IndustryMap,
    PercentLevels,
    RiskParams,
    apply_portfolio_caps,
    atr_sma,
    atr_wilder,
    build_card,
    build_levels,
    evaluate_ev,
    fractional_kelly,
    full_kelly,
    load_atr,
)
from factor_lab.analysis.risk_budget.industry import IndustryUsage, industry_caps

COST = 0.0015  # factor_lab.config.DEFAULT_COST.round_trip（双边 15bp）


def _levels(price=100.0, atr=2.0):
    """止损 2×ATR / 目标 4×ATR ⇒ R=2。"""
    return build_levels(price, atr=atr)


# ────────────────────────── A. 期望值闸门 ──────────────────────────
class TestEvGate:
    def test_negative_ev_is_vetoed_with_reason(self):
        """EV_net = p·R − (1−p) − c ≤ 0 必须被否决，并给出可打印的理由。"""
        gate = evaluate_ev(p=0.30, r=2.0, cost_ratio=COST)
        # 手算：0.30×2 − 0.70 − 0.0015 = −0.1015
        assert gate.ev_net == pytest.approx(0.60 - 0.70 - COST)
        assert gate.ev_net < 0
        assert gate.approved is False
        assert "EV_net" in gate.reason and "≤ 0" in gate.reason

    def test_cost_alone_can_veto(self):
        """毛 EV 为正但成本吃掉它 —— 闸门必须用净 EV 判定。"""
        gate = evaluate_ev(p=0.34, r=2.0, cost_ratio=COST)
        # 0.34×2−0.66 = +0.02 > 0，但 −0.0015 后仍为正；换更高成本
        assert gate.ev_net == pytest.approx(0.02 - COST)
        assert gate.approved is True
        gate2 = evaluate_ev(p=0.34, r=2.0, cost_ratio=0.03)
        assert gate2.ev_gross > 0
        assert gate2.approved is False
        assert "c=0.030000" in gate2.reason

    def test_boundary_ev_zero_is_vetoed(self):
        """恰好 EV=0 也算否决（「不亏不赚」不值得承担风险）。"""
        # 取 c=0, r=2 → p 使 2p−(1−p)=0 → p=1/3
        gate = evaluate_ev(p=1.0 / 3.0, r=2.0, cost_ratio=0.0)
        assert gate.ev_net == pytest.approx(0.0, abs=1e-12)
        assert gate.approved is False

    def test_p_zero_and_p_one_are_explicit(self):
        g0 = evaluate_ev(p=0.0, r=2.0, cost_ratio=COST)
        assert g0.approved is False and "p=0" in g0.reason
        g1 = evaluate_ev(p=1.0, r=2.0, cost_ratio=COST)
        assert g1.approved is True
        assert g1.reason == ""                 # 通过时不得带否决理由
        assert "p=1" in g1.note                # 退化假设必须写明

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"p": 1.5, "r": 2.0, "cost_ratio": COST},     # p 越界
            {"p": -0.1, "r": 2.0, "cost_ratio": COST},    # p 越界
            {"p": float("nan"), "r": 2.0, "cost_ratio": COST},
            {"p": 0.5, "r": 1.0, "cost_ratio": COST},     # R ≤ 1
            {"p": 0.5, "r": 0.5, "cost_ratio": COST},
            {"p": 0.5, "r": 2.0, "cost_ratio": -0.01},    # 负成本
        ],
    )
    def test_illegal_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            evaluate_ev(**kwargs)


# ────────────────────────── B. 三道硬上限 ──────────────────────────
def _cand(code: str, p: float, ind: str) -> Candidate:
    gate = evaluate_ev(p, 2.0, COST)
    return Candidate(code=code, p=p, r=2.0, ev_net=gate.ev_net,
                     ev_gross=gate.ev_gross, ev_approved=gate.approved,
                     industry_l1=ind)


class TestPositionCaps:
    def test_max_position_caps_single_name(self):
        """单票上限生效：分数凯利 0.25×0.175=0.04375 → 被裁到 0.02。"""
        lv = _levels()
        assert lv.r_ratio == 2.0
        assert fractional_kelly(0.45, 2.0, 0.25) == pytest.approx(0.04375)
        params = RiskParams(max_position=0.02, max_industry=1.0)
        out = apply_portfolio_caps([_cand("sh600519", 0.45, "食品饮料")], params)
        assert out[0].weight == pytest.approx(0.02)
        assert out[0].binding == "max_position"

    def test_max_industry_caps_sum_of_same_industry(self):
        """同行业 5 只，每只凯利 0.04375 → 合计被裁到 20%（单票上限放到 20% 让行业成为唯一瓶颈）。"""
        params = RiskParams(max_position=0.20, max_industry=0.20)
        cands = [_cand(f"sh60000{i}", 0.45, "食品饮料") for i in range(5)]
        out = apply_portfolio_caps(cands, params)
        total = sum(a.weight for a in out)
        assert total == pytest.approx(0.20)
        assert [a.binding for a in out[:4]] == ["kelly"] * 4
        assert out[-1].binding == "max_industry"
        assert out[-1].weight == pytest.approx(0.20 - 4 * 0.04375)
        assert "上限" in out[-1].reason

    def test_max_total_caps_portfolio(self):
        """跨行业多只、每只 4.375% → 第 4 只吃满总仓位额度，第 5 只无额度可建。"""
        params = RiskParams(max_position=0.05, max_industry=1.0, max_total=0.10)
        inds = ["食品饮料", "电子", "医药生物", "电力设备", "银行"]
        cands = [_cand(f"sh60001{i}", 0.45, inds[i]) for i in range(5)]
        out = apply_portfolio_caps(cands, params)
        assert sum(a.weight for a in out) == pytest.approx(0.10)
        assert [a.binding for a in out[:4]] == ["kelly"] * 4
        assert out[4].dropped is True and out[4].binding == "no_room"
        assert "无法建仓" in out[4].reason

    def test_ev_ranking_keeps_higher_ev(self):
        """超限时按 EV 从高到低保留：EV 高的先占额度，EV 低的被裁。"""
        params = RiskParams(max_position=0.04, max_industry=1.0, max_total=0.06)
        # 手算：p=0.50 → 分数凯利 0.25×(1.0−0.5)/2 = 0.0625 → 被单票 0.04 与总 0.06 夹住
        #       p=0.40 → 0.25×(0.8−0.6)/2 = 0.025
        hi, lo = _cand("sh600519", 0.50, "食品饮料"), _cand("sz000858", 0.40, "食品饮料")
        out = apply_portfolio_caps([lo, hi], params)      # 故意乱序传入
        assert out[0].code == "sh600519"                  # 结果按 EV 降序
        assert out[0].weight == pytest.approx(0.04)       # = 单票上限（小于 0.0625 凯利）
        assert out[0].binding == "max_position"
        assert out[1].weight == pytest.approx(0.02)       # 总剩余 0.06 − 0.04 = 0.02
        assert out[1].binding == "max_total"

    def test_negative_kelly_candidate_is_not_clipped(self):
        """分数凯利为负 → 显式 weight=0 + 理由，禁止 clip 成 0 静默放行。"""
        params = RiskParams()
        # 构造 ev_approved=True 但分数凯利 ≤ 0（上游误标）
        bogus = Candidate(code="sz000001", p=0.10, r=2.0, ev_net=0.001,
                          ev_gross=0.0025, ev_approved=True, industry_l1="银行")
        out = apply_portfolio_caps([bogus], params)
        assert out[0].weight == 0.0
        assert out[0].binding == "kelly_nonpositive"
        assert "分数凯利" in out[0].reason

    def test_gate_veto_is_explicit(self):
        params = RiskParams()
        vetoed = _cand("sh600519", 0.10, "食品饮料")       # EV<0 → 未通过
        assert vetoed.ev_approved is False
        out = apply_portfolio_caps([vetoed], params)
        assert out[0].weight == 0.0 and out[0].dropped is True
        assert out[0].binding == "ev_gate_veto"
        assert out[0].reason

    def test_existing_industry_holding_reduces_room(self):
        """已有持仓占用行业额度：食品饮料已用 18% → 行业只剩 2%。"""
        params = RiskParams(max_position=0.20, max_industry=0.20)
        cands = [_cand("sz000858", 0.45, "食品饮料")]
        out = apply_portfolio_caps(cands, params, existing_industry={"食品饮料": 0.18})
        assert out[0].weight == pytest.approx(0.02)
        assert out[0].binding == "max_industry"
        assert "已用 0.1800" in out[0].reason

    def test_existing_industry_full_means_no_room(self):
        """行业额度已被占满 → 显式 dropped + no_room，不静默给 0 仓位。"""
        params = RiskParams(max_position=0.20, max_industry=0.20)
        out = apply_portfolio_caps([_cand("sz000858", 0.45, "食品饮料")], params,
                                   existing_industry={"食品饮料": 0.20})
        assert out[0].dropped is True and out[0].binding == "no_room"
        assert "无法建仓" in out[0].reason

    def test_below_min_position_is_dropped_with_reason(self):
        params = RiskParams(max_position=0.02, min_position=0.01)
        # 0.25×full_kelly(0.26,2) ：full=(0.52−0.74)/2<0 → 改用小的 shrinkage? 用 p 使其为正但很小
        small = _cand("sh600519", 0.34, "食品饮料")   # full=(0.68−0.66)/2=0.01 → ×0.25=0.0025
        out = apply_portfolio_caps([small], params)
        assert out[0].dropped is True
        assert out[0].binding == "below_min_position"
        assert "挂不出去" in out[0].reason

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_position": 0.0}, {"max_position": 1.5},
            {"max_industry": -0.1}, {"max_total": 0.0},
            {"shrinkage": 0.0}, {"shrinkage": 1.5},
            {"min_position": 0.05, "max_position": 0.02},
            {"max_position": 0.5, "max_total": 0.2},
        ],
    )
    def test_illegal_params_raise(self, kwargs):
        with pytest.raises(ValueError):
            RiskParams(**kwargs)


# ────────────────────────── C. 缺输入必须抛错 ──────────────────────────
class TestNoSilentFallback:
    def test_empty_candidates_raise(self):
        with pytest.raises(ValueError, match="空"):
            apply_portfolio_caps([], RiskParams())

    def test_missing_atr_raises_and_does_not_fall_back_to_percent(self):
        """缺 ATR 时必须抛错，**不许**偷偷改用百分比口径或默认倍数。"""
        with pytest.raises(ValueError, match="atr 缺失"):
            build_levels(100.0, atr=None, stop_pct=None, target_pct=None)
        with pytest.raises(ValueError, match="atr 缺失"):
            build_levels(100.0, atr=None)                # 也不许用默认倍数糊过去

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_bad_atr_raises(self, bad):
        with pytest.raises(ValueError):
            build_levels(100.0, atr=bad)

    def test_both_calibers_given_raises(self):
        with pytest.raises(ValueError, match="互斥"):
            build_levels(100.0, atr=2.0, stop_pct=0.05, target_pct=0.10)

    def test_r_and_target_atr_conflict_raises(self):
        with pytest.raises(ValueError, match="赔率来源冲突"):
            build_levels(100.0, atr=2.0, r=3.0, target_atr=4.0)

    def test_unexecutable_levels_raise(self):
        """R ≤ 1（目标落在买点内）不是可执行建议 → 抛错。"""
        with pytest.raises(ValueError, match="R ≤ 1|必须 >"):
            build_levels(100.0, atr=2.0, stop_atr=2.0, target_atr=1.0)
        with pytest.raises(ValueError, match="必须 >"):
            build_levels(100.0, stop_pct=0.10, target_pct=0.05)

    def test_atr_missing_in_db_raises(self, tmp_path):
        db = _mk_db(tmp_path, rows=[("2024-08-01", 10, 11, 9, 10.5)])
        with pytest.raises(ValueError, match="ATR\\(14\\) 需要"):
            load_atr("sh600519", "2024-08-01", db_path=db)

    def test_unknown_code_raises(self, tmp_path):
        db = _mk_db(tmp_path, rows=[("2024-08-01", 10, 11, 9, 10.5)])
        with pytest.raises(ValueError, match="无日线数据"):
            load_atr("sh999999", "2024-08-01", db_path=db)

    def test_empty_industry_table_raises(self):
        with pytest.raises(ValueError, match="为空"):
            IndustryMap.from_frame(pd.DataFrame(
                columns=["ts_code", "l1_name", "l2_name", "l3_name", "in_date"]))

    def test_industry_table_missing_column_raises(self):
        with pytest.raises(ValueError, match="缺列"):
            IndustryMap.from_frame(pd.DataFrame(
                {"ts_code": ["600519.SH"], "l1_name": ["食品饮料"]}))

    def test_missing_industry_mapping_raises_not_sentinel(self):
        """缺映射必须 KeyError（禁止静默归入「未知」行业）—— P22 同族。"""
        t = pd.DataFrame({
            "ts_code": ["600519.SH"], "l1_name": ["食品饮料"],
            "l2_name": ["白酒Ⅱ"], "l3_name": ["白酒Ⅲ"], "in_date": ["20010731"],
        })
        im = IndustryMap.from_frame(t)
        assert im.l1("sh600519") == "食品饮料"
        with pytest.raises(KeyError, match="不在申万映射表内"):
            im.l1("sz000001")
        with pytest.raises(ValueError, match="codes 为空"):
            im.l1_many([])

    def test_duplicate_code_in_industry_table_raises(self):
        """同 code 同 in_date 的真重复 → 必须抛错（禁止靠行序静默取一条）。"""
        dup = pd.DataFrame({
            "code": ["sh600519", "sh600519"],
            "l1_name": ["食品饮料", "电子"], "l2_name": ["a", "b"],
            "l3_name": ["c", "d"],
        })
        assert set(["sh600519"]) == set(dup["code"])
        with pytest.raises(ValueError, match="重复 code"):
            IndustryMap(dup)

    def test_same_code_multiple_rows_keeps_latest_in_date(self):
        """同 code 多行（不同 in_date）→ 取最新分类，不报重复错（与并库脚本同规则）。"""
        t = pd.DataFrame({
            "ts_code": ["600519.SH", "600519.SH"],
            "l1_name": ["食品饮料", "电子"], "l2_name": ["a", "b"],
            "l3_name": ["c", "d"], "in_date": ["20010731", "20200101"],
        })
        im = IndustryMap.from_frame(t)
        assert len(im) == 1 and im.l1("sh600519") == "电子"

    def test_industry_caps_raises_on_overflow_and_empty(self):
        with pytest.raises(ValueError, match="空"):
            industry_caps([], 0.20)
        over = [IndustryUsage("食品饮料", 0.25, 0.20, "existing")]
        with pytest.raises(ValueError, match="行业集中度超限"):
            industry_caps(over, 0.20)


# ────────────────────────── D. 分数凯利 ──────────────────────────
class TestKelly:
    def test_hand_computed_full_kelly(self):
        """手算：p=0.6, R=2 → f* = (0.6×2 − 0.4)/2 = 0.8/2 = 0.4。"""
        assert full_kelly(0.6, 2.0) == pytest.approx(0.4)
        assert full_kelly(0.5, 3.0) == pytest.approx(0.5 - 0.5 / 3.0)   # 0.3333…
        assert full_kelly(0.45, 2.0) == pytest.approx(0.175)
        assert full_kelly(0.3, 2.0) == pytest.approx(-0.05)             # 负值不得被 clip

    def test_shrinkage_one_equals_full_kelly(self):
        """收缩系数=1 必须**逐位退化**为全凯利（对拍手算例子）。"""
        for p, r in [(0.6, 2.0), (0.45, 2.0), (0.34, 2.0), (0.7, 1.5)]:
            assert fractional_kelly(p, r, 1.0) == pytest.approx(full_kelly(p, r))
        assert fractional_kelly(0.6, 2.0, 1.0) == pytest.approx(0.4)
        # 收缩是线性乘子
        assert fractional_kelly(0.6, 2.0, 0.25) == pytest.approx(0.1)
        assert fractional_kelly(0.6, 2.0, 0.5) == pytest.approx(0.2)

    @pytest.mark.parametrize("s", [0.0, -0.5, 1.5, float("nan")])
    def test_illegal_shrinkage_raises(self, s):
        with pytest.raises(ValueError):
            fractional_kelly(0.6, 2.0, s)

    @pytest.mark.parametrize("r", [0.0, -1.0, float("nan")])
    def test_illegal_r_raises(self, r):
        with pytest.raises(ValueError):
            full_kelly(0.6, r)


# ────────────────────────── E. 建议卡字段完整性 ──────────────────────────
CARD_KW = dict(
    code="sh600519", name="贵州茅台",
    position_pct=2.0, industry_l1="食品饮料", industry_pct=2.0,
    max_position_pct=2.0, max_industry_pct=20.0,
    p=0.45, r=2.0, cost_ratio=COST,
    sample_n=1, sample_note="示例假设值，非实测（n=1 仅为演示）",
    p_source="示例假设值，非实测", r_source="ATR 倍数口径 R=4/2=2",
    position_reason="触及单票上限 2%",
)


class TestAdviceCard:
    def test_all_required_fields_present(self):
        card = build_card(levels=_levels(), **CARD_KW)
        for field in (
            "code", "name", "entry_low", "entry_high", "target", "stop",
            "position_pct", "industry_l1", "industry_pct", "ev_net",
            "sample_n", "sample_note", "price_caliber", "turnover_unit",
            "annualization", "disclaimer",
        ):
            assert getattr(card, field) not in (None, ""), f"建议卡缺字段 {field}"
        assert card.disclaimer == "不构成投资建议"
        text = "\n".join(card.lines())
        for token in ("买点区间", "目标位", "止损位", "建议仓位", "板块",
                      "期望净 EV", "样本量", "口径三件套", "不构成投资建议"):
            assert token in text, f"渲染文本缺 {token}"
        assert card.stop < card.entry_low <= card.entry_high < card.target

    def test_account_level_ev_multiplies_position(self):
        """★目标第 4 条：改「占账户」表述必须乘仓位。"""
        card = build_card(levels=_levels(), **CARD_KW)
        assert card.position_frac == pytest.approx(0.02)
        assert card.ev_net_account == pytest.approx(card.ev_net * 0.02)

    def test_missing_required_text_field_raises(self):
        """build_card 能设的字段 + AdviceCard 直接构造的字段，都必须非空。"""
        for field in ("name", "p_source", "r_source", "sample_note",
                      "position_reason", "industry_l1"):
            kw = dict(CARD_KW)
            kw[field] = "   "
            with pytest.raises(ValueError, match="缺必填文字字段"):
                build_card(levels=_levels(), **kw)
        # 口径三件套与披露只能由卡自身负责，直接构造验其非空校验
        for field in ("price_caliber", "turnover_unit", "annualization", "disclaimer"):
            card = build_card(levels=_levels(), **CARD_KW)
            with pytest.raises(ValueError, match="缺必填文字字段"):
                AdviceCard(**{**vars(card), field: "   "})

    def test_wrong_disclaimer_raises(self):
        with pytest.raises(ValueError, match="风险披露"):
            AdviceCard(
                code="sh600519", name="贵州茅台",
                entry_low=99.0, entry_high=101.0, target=110.0, stop=95.0, price=100.0,
                position_pct=2.0, industry_l1="食品饮料", industry_pct=2.0,
                max_position_pct=2.0, max_industry_pct=20.0,
                p=0.45, r=2.0, cost_ratio=COST, ev_gross=0.35, ev_net=0.3485,
                sample_n=1, sample_note="x", p_source="y", r_source="z",
                price_caliber="a", turnover_unit="b", annualization="c",
                disclaimer="风险自担", position_reason="r",
            )

    def test_position_over_cap_raises(self):
        kw = dict(CARD_KW, position_pct=3.0)
        with pytest.raises(ValueError, match="超过上限"):
            build_card(levels=_levels(), **kw)

    def test_industry_over_cap_raises(self):
        kw = dict(CARD_KW, industry_pct=25.0)
        with pytest.raises(ValueError, match="超过上限"):
            build_card(levels=_levels(), **kw)

    def test_industry_total_must_include_self(self):
        kw = dict(CARD_KW, position_pct=2.0, industry_pct=1.0)
        with pytest.raises(ValueError, match="口径错误"):
            build_card(levels=_levels(), **kw)

    def test_negative_ev_card_refused(self):
        """EV ≤ 0 却想出卡 → 直接拒绝（闸门的核心价值不可绕过）。"""
        with pytest.raises(ValueError, match="期望值闸门"):
            build_card(levels=_levels(), **dict(CARD_KW, p=0.30))

    def test_r_mismatch_between_levels_and_ev_raises(self):
        with pytest.raises(ValueError, match="不一致"):
            build_card(levels=_levels(), **dict(CARD_KW, r=3.0))

    def test_price_ordering_violation_raises(self):
        lv = PercentLevels(
            method="percent", price=100.0, entry_low=99.0, entry_high=101.0,
            stop=98.0, target=120.0, stop_distance=2.0, target_distance=20.0,
            r_ratio=10.0, unit_note="n", stop_pct=0.02, target_pct=0.20,
        )
        assert lv.r_ratio == 10.0
        with pytest.raises(ValueError):
            AdviceCard(
                code="sh600519", name="x",
                entry_low=99.0, entry_high=101.0, target=100.5, stop=98.0, price=100.0,
                position_pct=2.0, industry_l1="食品饮料", industry_pct=2.0,
                max_position_pct=2.0, max_industry_pct=20.0,
                p=0.45, r=2.0, cost_ratio=COST, ev_gross=0.35, ev_net=0.3485,
                sample_n=1, sample_note="x", p_source="y", r_source="z",
                price_caliber="a", turnover_unit="b", annualization="c",
                disclaimer="不构成投资建议", position_reason="r",
            )

    def test_sample_n_must_be_positive(self):
        with pytest.raises(ValueError, match="样本量"):
            build_card(levels=_levels(), **dict(CARD_KW, sample_n=0))


# ────────────────────────── F. ATR / 防前视 ──────────────────────────
def _mk_db(tmp_path, rows, code="sh600519"):
    """建一个最小 bar_daily 库：rows = [(date, open, high, low, close), ...]。"""
    db = tmp_path / "market.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE bar_daily (code TEXT, date INTEGER, open REAL, high REAL, "
        "low REAL, close REAL, amount REAL, vol REAL, close_adj REAL, "
        "PRIMARY KEY (code, date))"
    )
    con.executemany(
        "INSERT INTO bar_daily VALUES (?,?,?,?,?,?,?,?,?)",
        [(code, int(d.replace("-", "")), o, h, low, c, 0.0, 0.0, c)
         for d, o, h, low, c in rows],
    )
    con.commit()
    con.close()
    return db


def _trend_rows(n=20):
    """构造有明确 TR 的上升序列，用于手算对拍。"""
    rows = []
    price = 10.0
    for i in range(n):
        prev = price
        price = prev + 1.0
        rows.append((f"2024-07-{i + 1:02d}", prev, price + 0.5, prev - 0.5, price))
    return rows


class TestAtr:
    def test_wilder_matches_hand_computation(self):
        """手算 Wilder：TR 恒为 2.0（high-low=2 最大）→ ATR 恒为 2.0。"""
        rows = _trend_rows(20)
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
        assert atr_wilder(df, 14) == pytest.approx(2.0)
        assert atr_sma(df, 14) == pytest.approx(2.0)

    def test_wilder_differs_from_sma_on_gap(self):
        """有跳空缺口时两口径必须不同（防止把两口径当同一个数）。"""
        rows = [("2024-07-01", 10.0, 10.5, 9.5, 10.0)]
        close = 10.0
        for i in range(2, 22):
            close += 1.0
            rows.append((f"2024-07-{i:02d}", close - 1, close + 0.2, close - 1.2, close))
        rows.append(("2024-07-22", close, close + 6.0, close - 0.5, close + 5.0))  # 大缺口
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
        w, s = atr_wilder(df, 14), atr_sma(df, 14)
        assert w != pytest.approx(s)
        assert w > 0 and s > 0

    def test_load_atr_is_lookahead_safe(self, tmp_path):
        db = _mk_db(tmp_path, _trend_rows(20))
        atr, tail = load_atr("sh600519", "2024-07-15", window=14, db_path=db)
        assert atr == pytest.approx(2.0)
        assert len(tail) == 15                                   # window+1
        assert tail["date"].max() <= pd.Timestamp("2024-07-15")   # 只用 as_of 及之前

    def test_load_atr_rejects_bad_ohlc(self, tmp_path):
        rows = _trend_rows(20)
        rows[5] = ("2024-07-06", 10.0, 0.0, -1.0, 10.0)
        db = _mk_db(tmp_path, rows)
        with pytest.raises(ValueError, match="缺失或 ≤0"):
            load_atr("sh600519", "2024-07-20", window=14, db_path=db)
