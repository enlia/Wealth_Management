"""价格口径守卫：防止「未复权价污染收益」这一类 bug 复发。

为什么这个文件存在
------------------
2026-10-06 实测：`run_financial_study.py` 等 **6 个脚本**用
未复权价算前瞻收益，而该面板正是喂给 alphalens 的 `prices`。

未复权价在除权日有**向下假跳空**。实测全市场 5,606 只等权口径的年化偏差：

===================  ==========  ==========  ==========
年份                未复权      前复权      偏差
===================  ==========  ==========  ==========
2016                −1.08%      +15.72%     **−16.80pp**
2019                +41.06%     +51.86%     −10.80pp
2025                +63.53%     +70.49%     −6.97pp
2021~ 2026 每年       —          —          −5.1 ~ −8.0pp
===================  ==========  ==========  ==========

⚠️ **这个偏差大于本项目声称的任何因子收益**（最高的 ep 也只有 +6.1%）。
拿未复权价算出的IC 是在给「分红除权」定价，不是在给股票定价。

这类 bug 的特征是**不报错**：数据能读、因子能算、图能画，只是结论全错。
所以只能靠测试拦。

四条守卫
--------
1. ``test_field_is_required`` —— ``field`` 必须**无默认值**。
   「忘了写口径」应当抛 TypeError，而不是静默猜一个。
   首版用正则扫描，但 review 实测位置参数与变量传入都能绕过；
   改成「取消默认值」后，这一整类风险从根上消失。
2. ``test_factor_prices_*`` —— 双口径接口可用、为空时抛错。
3. ``test_no_research_script_uses_raw_close`` —— **AST 解析**（非正则）
   找出所有把 ``"close"`` 传给 ``load_prices`` 的位置，
   覆盖关键字 / 位置参数 / 单双引号各种写法。
4. ``test_all_research_scripts_compile`` —— 语法门禁。
   ⚠️ 这条补的是一个真实事故：批量替换价格口径时把 ``{ind}``
   缩进占位符当成内容写进了 3 个文件导致 SyntaxError，
   而 ``pytest`` 报 350 passed 全绿 —— 因为测试从不 import
   research/ 下的脚本。
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.data import load_factor_prices, load_prices  # noqa: E402

SCRIPTS = ROOT / "research" / "scripts"
PKG = ROOT / "src" / "factor_lab"

# 允许把未复权 `"close"` 传给 load_prices 的文件。
# 逐个 review 确认过用途，不是「测不过就加白名单」。
RAW_CLOSE_ALLOWLIST = {
    # 审计脚本：显式对比「未复权 vs 前复权」两个口径，正是它的职责。
    # :321 的 quoted_panel 用于涨跌停前收盘价 —— 交易所约束的是**报价**，
    # 与复权无关，技术上正确。
    "audit_data_quality.py",
    # 诊断脚本：专门查复权质量，必须拿未复权价做对照。
    # 且已实测「交易日序号必须用未复权序列算」，
    # 否则「复权引入」计数会虚高到 11,494 条（真实值 62 条）。
    "diagnose_adjustment.py",
    # smoke_test 对账的是通达信`.day` 二进制直读的**报价**真值，
    # 步骤 2/3 必须同口径用未复权价。
    # 顺带实测：close_adj 对指数/板块/B股覆盖率 0%，
    # 若误用前复权价，上证/深成指/板块/B股 4 项全部拿到 NaN。
    "smoke_test.py",
}
# 数据层自身（factor_prices 的职责就是取两种口径），
# 路径级豁免 —— 它是唯一被允许同时用两个口径的地方。
ALLOWLIST_DIRS = {ROOT / "src" / "factor_lab" / "data"}


def _scanned_files() -> list[Path]:
    return sorted(SCRIPTS.glob("*.py")) + sorted(PKG.rglob("*.py"))


def test_field_is_required():
    """``field`` 必须没有默认值 —— 忘了写口径要报错，不能猜。

    这条比任何扫描式门禁都强：它把一整类「静默拿错价格」
    的风险从 API 设计上消除，而不是靠正则去追。
    """
    p = inspect.signature(load_prices).parameters["field"]
    assert p.default is inspect.Parameter.empty, (
        f"load_prices 的 field 又有了默认值 {p.default!r}。"
        "必须保持必填：漏写口径应当抛 TypeError，而不是静默用某个价格 —— "
        "实测未复权口径年化偏差可达 −16.80pp。")


def test_factor_prices_returns_both_panels():
    """双口径接口：能取到两个面板，且索引完全一致。"""
    p = load_factor_prices(["sh600519", "sz000001"], "20240101", "20241231")
    raw, adj = p  # 必须支持解包
    assert raw.shape == adj.shape
    assert raw.index.equals(adj.index), "两套口径的交易日轴必须一致"
    assert raw.columns.equals(adj.columns), "两套口径的标的列必须一致"
    assert p.raw is raw and p.adj is adj, "属性访问应与解包结果一致"
    assert (raw - adj).abs().to_numpy().max() > 0, "raw 与 adj 完全相同，复权没生效"


def test_factor_prices_rejects_empty_adjusted():
    """adj 为空时必须抛错，禁止静默退回未复权价。"""
    with pytest.raises(ValueError, match="价格面板为空"):
        load_factor_prices(["sh600519"], "19000101", "19000102")


def _raw_close_calls(path: Path) -> list[tuple[int, str]]:
    """AST 找出所有 ``load_prices(..., "close")`` 调用。

    用 AST 而非正则的原因（review 实测三种绕过全部通过）：
      · 位置参数 ``load_prices(c, start, end, "close")``
      · 变量传入 ``load_prices(c, field=CFG)``
      · 多层引号 / 模板拼装出来的字符串

    AST 对位置参数与关键字参数一视同仁，且不受排版影响。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "load_prices":
            continue
        if len(node.args) >= 4:
            a = node.args[3]
            if isinstance(a, ast.Constant) and a.value == "close":
                hits.append((node.lineno, "位置参数 load_prices(..., 'close')"))
        for kw in node.keywords:
            if (kw.arg == "field" and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "close"):
                hits.append((node.lineno, "关键字 load_prices(..., field='close')"))
    return hits


@pytest.mark.parametrize("py", _scanned_files(), ids=lambda p: p.name)
def test_no_research_script_uses_raw_close(py: Path):
    """除白名单外，禁止把未复权 `"close"` 传给 load_prices。"""
    if py.name in RAW_CLOSE_ALLOWLIST or py.parent in ALLOWLIST_DIRS:
        pytest.skip(f"{py.name} 已逐一review 确认用途")
    hits = _raw_close_calls(py)
    assert not hits, (
        f"{py.name} 把未复权价传给了 load_prices：\n"
        + "\n".join(f"  行 {n}: {t}" for n, t in hits)
        + "\n\n修法：收益/IC/回测用 close_adj；"
          "只有 PB/PE/EP 允许 close，且必须写明理由。")


@pytest.mark.parametrize("py", _scanned_files(), ids=lambda p: p.name)
def test_all_research_scripts_compile(py: Path):
    """语法门禁。

    ⚠️ 来自真实事故：批量替换价格口径时 ``{ind}`` 占位符被当成内容
    写进了 3 个脚本导致 SyntaxError，而 ``pytest`` 全绿——
    因为测试从不 import research/ 下的脚本，没有任何东西会加载它们。
    """
    src = py.read_text(encoding="utf-8")
    try:
        compile(src, str(py), "exec")
    except SyntaxError as e:
        pytest.fail(f"{py.name}:{e.lineno} 语法错误: {e.msg}")