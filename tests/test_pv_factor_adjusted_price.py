"""`expand_factor_library.build_pv_factors` 的价格口径守卫。

背景（2026-10-06）
--------------------
该函数有 138 行不可达死代码（`return res` 之后又粘了一份旧实现），
且真正生效的那份用**未复权价**算收益与形态因子。

未复权价在除权日有假跳空：分红送转当日一次性 −10%，
`close/close.shift(20)` 这类跨期比率会被污染。
项目实测：污染量级5~17pp/年，**大于任何因子的真实收益差异**。

所以这里要守住三件事：
1. 缺复权列必须**抛错**，不能静默退回未复权价
2. 生效实现里**不能出现**未复权列名
3. 死代码不许复活（`return` 之后的语句不可达）
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "research" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from expand_factor_library import build_pv_factors  # noqa: E402


def _make_long(n_code: int = 40, n_day: int = 300, *, adjusted: bool) -> pd.DataFrame:
    """构造一只会在第 150 天除权 10%的长表 —— 假跳空就落在这里。"""
    rng = np.random.default_rng(20261006)
    codes = [f"sh{600000 + i:06d}" for i in range(n_code)]
    start = pd.Timestamp("2020-01-01")
    days = pd.bdate_range(start, periods=n_day)

    frames = []
    for i, c in enumerate(codes):
        # 每只票自己的确定性随机游走
        steps = rng.normal(0.001, 0.02, n_day)
        close = 20.0 * np.exp(np.cumsum(steps))
        ratio = np.ones(n_day)
        # 第 150 天除权：close_adj 应比close 低 10%
        ratio[150:] = 1.0 / 1.1
        df = pd.DataFrame({
            "code": c, "date": days,
            "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
            "close": close, "amount": rng.uniform(1e7, 1e9, n_day),
            "vol": rng.uniform(1e5, 1e7, n_day),
        })
        if adjusted:
            df["close_adj"] = df["close"] * ratio
            df["high_adj"] = df["close_adj"] * 1.01
            df["low_adj"] = df["close_adj"] * 0.99
        else:
            # 退化路径：假装调用方把未复权列改名为复权列
            df["close_adj"] = df["close"]
            df["high_adj"] = df["high"]
            df["low_adj"] = df["low"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── 1. 缺列必须抛错，且错误信息点名要adjusted=True ──────────────
class TestMissingAdjustedColumns:
    def test_缺close_adj时必须抛错(self):
        long = _make_long(adjusted=True).drop(columns=["close_adj"])
        with pytest.raises(ValueError, match="close_adj"):
            build_pv_factors(long)

    def test_缺high_adj时必须抛错(self):
        long = _make_long(adjusted=True).drop(columns=["high_adj"])
        with pytest.raises(ValueError, match="high_adj"):
            build_pv_factors(long)

    def test_缺low_adj时必须抛错(self):
        long = _make_long(adjusted=True).drop(columns=["low_adj"])
        with pytest.raises(ValueError, match="low_adj"):
            build_pv_factors(long)

    def test_错误信息必须提示adjusted_true(self):
        """错误信息要能直接告诉调用方怎么改 —— 只说「缺列」等于没help。"""
        long = _make_long(adjusted=True).drop(columns=["close_adj"])
        with pytest.raises(ValueError, match="adjusted=True"):
            build_pv_factors(long)

    def test_不许静默退回未复权(self):
        """🔴 反向断言：旧实现是「有close 就用 close」。

        本测试把 close_adj 列改名为 close（即调用方压根没取复权），
        正确实现必须拒绝，而不是拿 close 算出16 个因子。
        """
        long = _make_long(adjusted=True)
        long = long.drop(columns=["close_adj", "high_adj", "low_adj"])
        with pytest.raises(ValueError):
            build_pv_factors(long)


# ── 2. 生效实现里不许引用未复权列名 ────────────────────────────
class TestNoUnadjustedReference:
    @staticmethod
    def _body() -> str:
        src = Path(_SCRIPTS / "expand_factor_library.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_pv_factors":
                # 去掉函数 docstring 后剩下的才是真正执行的语句
                body = list(node.body)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    body = body[1:]
                return "\n".join(ast.unparse(s) for s in body)
        raise AssertionError("build_pv_factors 不存在")

    def test_不引用裸close(self):
        """`col['close']` / `g['close']` / `long['close']` 都算。"""
        body = self._body()
        import re
        hits = re.findall(r"""\[\s*['"](close|high|low)['"]\s*\]""", body)
        assert not hits, f"发现未复权列引用: {hits}"

    def test_必须引用close_adj(self):
        assert "close_adj" in self._body()

    def test_不引用open列(self):
        """open 从未被使用，且 load_long 不提供 open_adj。"""
        assert "open" not in self._body()


# ── 3. 死代码不许复活 ─────────────────────────────────────────
class TestNoDeadCode:
    def test_return之后不许有可执行语句(self):
        src = Path(_SCRIPTS / "expand_factor_library.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef)
                    and node.name == "build_pv_factors"):
                continue
            body = list(node.body)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)):
                body = body[1:]
            ret_i = next(i for i, s in enumerate(body)
                         if isinstance(s, ast.Return))
            after = body[ret_i + 1:]
            assert not after, (
                f"return 之后还有 {len(after)} 条不可达语句："
                f"{[ast.unparse(s)[:60] for s in after[:3]]}")
            return
        raise AssertionError("build_pv_factors 不存在")

    def test_函数长度未失控(self):
        src = Path(_SCRIPTS / "expand_factor_library.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_pv_factors":
                assert node.end_lineno - node.lineno < 120, (
                    f"函数 {(node.end_lineno - node.lineno)} 行，"
                    "初版含 138 行死代码，加限防复发")
                return
        raise AssertionError("build_pv_factors 不存在")


# ── 4. 复权口径必须真的改变因子值（否则上面全是空转） ──────────
class TestAdjustedChangesResult:
    def test_除权日不产生假跳空(self):
        """核心行为：除权日在两口径下必须表现不同。

        构造数据里第 150 天除权 10%（`close_adj` 比 `close` 低 9.09%）。
        未复权口径会把这9.09% 的**非真实下跌**计入 20 日回撤，
        前复权口径不会。差异量级应≈ 1 − 1/1.1 = 0.0909。

        ⚠️ 不要用 `groupby.min().max()` 这类聚合统计量断言 ——
           它取的是「最不差的那只票」，对少数几只票的除权跳空不敏感，
           实测两口径位级相同 ⇒ 断言变橡皮章。
        """
        adj = build_pv_factors(_make_long(adjusted=True))
        raw = build_pv_factors(_make_long(adjusted=False))

        j = pd.concat([adj["rev_20"], raw["rev_20"]],
                      axis=1, keys=["adj", "raw"]).dropna()
        diff = (j["adj"] - j["raw"]).abs()

        # 每只票 20 日窗口跨越除权日的 20 个交易日，40 只票 ⇒ 800 行
        assert (diff > 1e-9).sum() == 40 * 20, (
            f"只有 {(diff > 1e-9).sum()} 行有差异，预期 800（40 只 × 20 日）")
        # 差异峰值应接近 0.0909（构造的除权幅度），不是别的量级
        assert 0.085< diff.max() < 0.14, (
            f"差异峰值 {diff.max():.4f}，构造的除权幅度是 0.0909，"
            "量级不符说明口径处理方式变了")
        # 未复权口径必须更悲观（扣掉不存在的下跌 ⇒ 回撤数值更负）
        assert j["raw"].loc[diff > 1e-9].mean() < j["adj"].loc[diff > 1e-9].mean()

    def test_两口径因子值必须不同(self):
        """构造数据里 close_adj 与 close 差 9%，
        若实现忽略复权列，两者会完全相等 ⇒ 测试无意义。"""
        adj = build_pv_factors(_make_long(adjusted=True))
        raw = build_pv_factors(_make_long(adjusted=False))
        assert set(adj) == set(raw)
        assert len(adj) == 16, f"因子数变了: {len(adj)}"
        diffs = [k for k in adj if not np.allclose(adj[k].values, raw[k].values)]
        assert diffs, "两种口径算出的因子值完全相同 ⇒ 复权列根本没被用"
