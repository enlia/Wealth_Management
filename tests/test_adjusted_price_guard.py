"""价格口径守卫：防止「未复权价污染收益」这一类 bug 复发。

为什么这个文件存在
------------------
2026-10-06 实测：`run_financial_study.py` 等**6 个脚本**用
`load_prices(field="close")`（未复权）算前瞻收益，
而该面板正是喂给 alphalens 的 `prices`。

未复权价在除权日有**向下假跳空**。实测全市场 5,606 只等权口径的年化偏差：

===================  ==========  ==========  ==========
年份                未复权      后复权      偏差
===================  ==========  ==========  ==========
2016                −1.08%      +15.72%     **−16.80pp**
2019                +41.06%     +51.86%     −10.80pp
2025                +63.53%     +70.49%     −6.97pp
2021~ 2026 每年       —          —          −5.1 ~ −8.0pp
===================  ==========  ==========  ==========

⚠️ **这个偏差大于本项目声称的任何因子收益**（最高的 ep 也只有 +6.1%）。
拿未复权价算出的 IC 是在给「分红除权」定价，不是在给股票定价。

这类 bug 的特征是**不报错**：数据能读、因子能算、图能画，
只是结论全错。所以只能靠测试拦。

三条守卫
--------
1. `test_load_prices_defaults_to_adjusted` —— 默认口径必须是 close_adj
2. `test_factor_prices_returns_both_panels` —— 双口径接口可用且索引一致
3. `test_no_research_script_uses_raw_close_for_returns`
   —— 扫描全部研究脚本，禁止再出现 `field="close"`
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.data import load_factor_prices, load_prices  # noqa: E402

SCRIPTS = ROOT / "research" / "scripts"

# 允许出现 `field="close"` 的白名单：这些脚本**故意**要未复权价，
# 且已在 review 中逐个确认用途（诊断复权质量 / 涨跌停报价判定）。
RAW_CLOSE_ALLOWLIST = {
    # 审计脚本：显式对比「未复权 vs 后复权」两个口径，正是它的职责
    "audit_data_quality.py",
    # 诊断脚本：专门用来查复权质量，必须拿未复权价做对照
    "diagnose_adjustment.py",
    # 本文件自身（docstring 里引用了这个字符串）
    "test_adjusted_price_guard.py",
    # 数据层自己取未复权价（factor_prices 内部实现）
    "factor_prices.py",
}


def test_load_prices_defaults_to_adjusted():
    """默认必须是 close_adj —— 90% 的调用方是在算收益。"""
    default = inspect.signature(load_prices).parameters["field"].default
    assert default == "close_adj", (
        f"load_prices 的默认 field 被改成了 {default!r}。"
        "收益研究必须默认用后复权价；"
        "只有 PB/PE/EP 需要未复权，且那种场景该用 load_factor_prices。")


def test_factor_prices_returns_both_panels():
    """双口径接口：能取到两个面板，且索引完全一致。"""
    codes = ["sh600519", "sz000001"]
    p = load_factor_prices(codes, "20240101", "20241231")
    raw, adj = p  # 必须支持解包
    assert raw.shape == adj.shape
    assert raw.index.equals(adj.index), "两套口径的交易日轴必须一致"
    assert raw.columns.equals(adj.columns), "两套口径的标的列必须一致"
    assert p.raw is raw and p.adj is adj, "属性访问应与解包结果一致"
    # 未复权 ≠ 后复权：茅台 2024 年有分红，两者不应完全相同
    diff = (raw - adj).abs().to_numpy()
    assert diff.max() > 0, "raw 与 adj 完全相同，说明复权没生效"


def test_factor_prices_rejects_empty_adjusted():
    """adj 为空时必须抛错，禁止静默退回未复权价。"""
    with pytest.raises(ValueError, match="价格面板为空"):
        load_factor_prices(["sh600519"], "19000101", "19000102")


@pytest.mark.parametrize("py", sorted(SCRIPTS.glob("*.py")),
                         ids=lambda p: p.name)
def test_no_research_script_uses_raw_close_for_returns(py: Path):
    """扫描研究脚本：除白名单外，禁止出现 field="close"。

    ⚠️ 只扫 ``field="close"`` 这个**显式**写法。
    依赖默认值的老代码（如 scan_all_factors.py 的 `load_prices(...)`）
    由 test_load_prices_defaults_to_adjusted 兜住 ——
    默认值一旦被改回close，那些调用点会静默变回未复权。
    """
    if py.name in RAW_CLOSE_ALLOWLIST:
        pytest.skip(f"{py.name} 在白名单（故意使用未复权价）")
    text = py.read_text(encoding="utf-8")
    hits = [(i + 1, ln.strip()) for i, ln in enumerate(text.splitlines())
            if re.search(r'field\s*=\s*[\'"]close[\'"]', ln)
            and not ln.lstrip().startswith("#")]
    assert not hits, (
        f"{py.name} 出现未复权价取数（行号:内容）：\n"
        + "\n".join(f"  {n}: {t}" for n, t in hits)
        + "\n\n修法：收益/IC/回测改用 close_adj 或 load_factor_prices；"
          "只有 PB/PE/EP 才允许未复权，且必须显式说明理由。")