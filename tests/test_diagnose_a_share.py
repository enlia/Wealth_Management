"""诊断脚本的 A 股判定必须走统一入口（`factor_lab.config.is_a_share`）。

## 钉住的三个反例（本地副本 vs 权威实现实测分歧）

| 代码 | 本地副本 | config 版 | 后果 |
|---|---|---|---|
| sz000900（现代投资，尾号 900） | False（`endswith("900")` 当 B 股错剔） | True | 该股从分板块统计里静默消失 |
| bj899050（北证50，**指数**） | True（`startswith("bj")` 错纳） | False | 指数混进 A 股封板率统计 |
| 600519（无前缀写法） | False（只认 8 位带前缀） | True | 同一只股票换个写法就丢 |

这批失败都是**不报错、数字看着合理**的方向性错误
（CODE_TRUST 一类），故测**行为**：反例该进的进、该剔的剔，
不测「是否 import 了某函数」这种实现细节。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "diagnostics"))

from diagnose_sealed_limit_rate import is_a_share  # noqa: E402


class Test诊断脚本A股判定:
    def test_尾号900的A股不能被错剔(self):
        """sz000900（现代投资）、sz002900（哈三联）都是 A 股普通股。"""
        assert is_a_share("sz000900") is True
        assert is_a_share("sz002900") is True

    def test_北证指数段不能算A股(self):
        """bj899050（北证50）、bj899601（专精特新）是指数，无个股涨跌停。"""
        assert is_a_share("bj899050") is False
        assert is_a_share("bj899601") is False

    def test_无前缀六位码与带前缀同判(self):
        """600519 与 sh600519 是同一只股票，判定口径必须一致。"""
        assert is_a_share("600519") is True
        assert is_a_share("sh600519") is True

    @pytest.mark.parametrize("code", [
        "sh600519", "sh688001", "sz000001", "sz300750",
        "sh000300", "sz399006", "sh510300", "sh113050", "sh900901",
        "sz200596", "bj430047", "bj832175", "bj899050",
    ])
    def test_常见代码分类与统一入口一致(self, code):
        """全类别对照 —— 防委托关系悄悄改坏某一类。"""
        from factor_lab.config import is_a_share as canonical

        assert is_a_share(code) == canonical(code), code


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
