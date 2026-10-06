"""年化换算常数与 σ 年化乘数的钉死用例（语义分层 + 反向断言）。

## 口径来源

真源 `.github/standards/PITFALLS.md` P10「（1·补）年化三层口径」：

- **① 倍数换算**（每期→年化 ×252/periods、σ 年化 × √252、成本 × 252）= **252**：
  P10 钉死公式 `gross = spread * (252 / periods[0])`；
- **② 时长换算兜底**（无日期索引输入：交易日数 ÷ 年均交易日）= **243**：
  A 股实测算式 2,611 交易日 ÷ 10.75 历年 = 242.9 ≈ 243（年均交易日）；
- **③ 有日期的年跨越** = 自然日 365.25（analysis.long_only._year_span 主路径）。

常数唯一出处 `src/factor_lab/config.py`：`SCALING_TRADING_DAYS`（①）、
`YEAR_TRADING_DAYS`（②）。本文件钉三件事：

1. 双常数各自钉死（含「244/252 串用必须失败」反向断言）；
2. σ 年化乘数钉 √252（含按 √244／√243 折算必须失败的反向断言）；
3. 全仓 σ 年化位置源码守护：还原 √244 历史形态、或把 252/243/244 写成
   sqrt 字面值（绕开常数）必须 FAILED。

## 判据

按 UNITS「找已知真值反推」：已知日收益样本的标准差算式写在用例注释里，
期望年化 = std × √252（可复算）；历史形态 ×√244 与口径比值
√(252/244) ≈ 1.0163（修正前后原始波动率数值差 ±1.63%），换回 √244 必须变红。
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_lab.config import (  # noqa: E402
    SCALING_TRADING_DAYS,
    YEAR_TRADING_DAYS,
)
from factor_lab.factors.price_volume import (  # noqa: E402
    downside_volatility,
    volatility,
)

# σ 年化实现点（与 config 偏差备忘、07 文档定位表一致；行号会漂，以锚点为准）
SIGMA_FILES = (
    "src/factor_lab/factors/price_volume.py",   # volatility() / downside_volatility()
    "src/tools/tdx.py",                         # indicators()「年化波动率%」
    "src/tools/bench_factor.py",                # feat() vol60
    "src/tools/build_universe.py",              # load_tech()「年化波动率%」
)
BARE_SIGMA_FILES = (
    "src/factor_lab/analysis/long_only.py",     # rolling_vol / _bench_stats 年化波动
)

_CLOSES = [100.0, 99.0, 97.0, 98.0]
_DATES = pd.date_range("2024-01-02", periods=len(_CLOSES), freq="B")
_LONG = pd.DataFrame({
    "code": ["sh600000"] * len(_CLOSES),
    "date": _DATES,
    "close": _CLOSES,
    "close_adj": _CLOSES,   # 因子口径按后复权价（_px 强制要求该列）
})


class Test换算常数各自钉死:
    def test_倍数换算常数钉死252(self):
        """① 倍数换算 = 252。出处：PITFALLS P10 钉死公式
        `gross = spread * (252 / periods[0])` —— 公式里的 252 即倍数口径。"""
        assert SCALING_TRADING_DAYS == 252, (
            f"SCALING_TRADING_DAYS={SCALING_TRADING_DAYS} —— ① 倍数换算口径是 "
            "252（PITFALLS P10 公式口径）；改口径属全项目口径变更，需单独立项")

    def test_时长换算常数钉死243(self):
        """② 时长换算兜底 = 243。出处算式：2,611 交易日 ÷ 10.75 历年
        = 242.9 ≈ 243（A 股年均交易日）。"""
        assert YEAR_TRADING_DAYS == 243, (
            f"YEAR_TRADING_DAYS={YEAR_TRADING_DAYS} —— ② 时长换算口径是 243"
            "（2611÷10.75≈242.9）；改口径属全项目口径变更，需单独立项")

    def test_244与252串用必须失败(self):
        """反向断言：244/252/243 层间串用的形态必须落红。
        历史事故：σ 年化乘数曾混入 √244，与 √252 差 √(252/244)。"""
        assert SCALING_TRADING_DAYS != 244, (
            "倍数层常数被改成 244 —— √244/244 是历史串用形态，① 层只有 252")
        assert YEAR_TRADING_DAYS != 252, (
            "时长层常数被改成 252 —— 252 是 ① 倍数层口径，② 时长层是 243")
        assert SCALING_TRADING_DAYS != YEAR_TRADING_DAYS, (
            "两层常数被并成一个值 —— 倍数 252 与时长 243 语义不同，勿一刀切")


class Testσ年化乘数钉死252:
    def test_波动率年化乘数按252钉死(self):
        """波动率年化 = std × √252 的已知真值（算式可复核）：

        收盘 [100, 99, 97, 98] → 日收益
          r1 = 99/100 − 1  = −0.010000
          r2 = 97/99 − 1   ≈ −0.020202
          r3 = 98/97 − 1   ≈ +0.010309
        过去 3 个收益（窗口 window=3）std(ddof=1) 手算：
          mean = (r1+r2+r3)/3；Σ(x−mean)² / 2 再开方（下式直接复算）
        年化 = std × √252（√252 ≈ 15.874508）。
        （×√252 是 ① 倍数放大；历史形态 ×√244 与其比值 √(252/244) ≈ 1.0163）
        """
        rets = [_CLOSES[1] / _CLOSES[0] - 1,
                _CLOSES[2] / _CLOSES[1] - 1,
                _CLOSES[3] / _CLOSES[2] - 1]
        daily_std = float(np.std(rets, ddof=1))
        got = float(volatility(_LONG, window=3).dropna().iloc[-1])
        assert got == pytest.approx(daily_std * math.sqrt(252), rel=1e-9), (
            "σ 年化乘数不是 √252 —— ① 倍数换算层口径（PITFALLS P10（1·补））")
        assert got != pytest.approx(daily_std * math.sqrt(244), rel=1e-3), (
            "σ 年化乘数仍是 √244 —— 历史串用形态，与 √252 差 √(252/244)≈1.63%")
        assert got != pytest.approx(daily_std * math.sqrt(243), rel=1e-3), (
            "σ 年化乘数按 √243 折算 —— 243 是 ② 时长层常数，勿串进倍数层")

    def test_下行波动率年化乘数按252钉死(self):
        """下行波动率同口径：只统计负收益（r1、r2），窗口 2、min_periods=2：

        负收益样本 [r1, r2] 的 std(ddof=1) = |r2 − r1| / √2
          （两样本标准差化简式），年化 = std × √252。
        """
        r1 = _CLOSES[1] / _CLOSES[0] - 1
        r2 = _CLOSES[2] / _CLOSES[1] - 1
        daily_std = float(np.std([r1, r2], ddof=1))
        got = float(downside_volatility(_LONG, window=2, min_periods=2)
                    .dropna().iloc[-1])
        assert got == pytest.approx(daily_std * math.sqrt(252), rel=1e-9), (
            "下行波动率 σ 年化乘数不是 √252 —— ① 倍数换算层口径")
        assert got != pytest.approx(daily_std * math.sqrt(244), rel=1e-3), (
            "下行波动率 σ 年化乘数仍是 √244 —— 历史串用形态还原必须失败")

    def test_年化开关只差年化乘数_单变还原可定位(self):
        """annualize 开关唯一的差异就是年化乘数本身：
        年化输出 ÷ 未年化输出 = √252（逐点），按 √244/√243 折算必须失败。"""
        raw = volatility(_LONG, window=3, annualize=False).dropna()
        ann = volatility(_LONG, window=3).dropna()
        ratio = (ann / raw).to_numpy()
        assert ratio == pytest.approx(math.sqrt(252), rel=1e-9), (
            f"年化乘数实测 {ratio} ≠ √252")
        assert not np.allclose(ratio, math.sqrt(244), rtol=1e-3), (
            "年化乘数按 √244 折算 —— 历史串用形态还原必须失败")


class Testσ年化乘数源码守护:
    """拦「换回 √244 历史形态」与「绕开常数写 sqrt 字面值」两类还原。

    这条门禁拦下的真实错误：5 处 σ 年化实现曾混用 √244 与口径 √252
    （偏差 ±1.63%），分散在 4 个文件、无行为级接缝（src/tools 三处依赖
    本机数据文件），值级用例够不到，只能守源码形态。
    """

    _LITERAL = re.compile(r"(?:np|math)\.sqrt\(\s*(?:244|243|252)(?:\.0)?\s*\)")

    def _py_files(self):
        for d in (ROOT / "src", ROOT / "research"):
            yield from sorted(d.rglob("*.py"))

    def test_sqrt乘数不得写244_243_252字面值(self):
        bad = []
        for p in self._py_files():
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if self._LITERAL.search(line):
                    bad.append(f"{p.relative_to(ROOT).as_posix()}:{i}: {line.strip()}")
        assert not bad, (
            "σ 年化乘数写成了 sqrt 字面值（还原 √244 形态或绕开常数）：\n"
            + "\n".join(bad)
            + "\n必须写 np.sqrt(SCALING_TRADING_DAYS)（唯一出处 config.py）")

    def test_σ年化实现点全部引SCALING常数(self):
        missing = [rel for rel in (*SIGMA_FILES, *BARE_SIGMA_FILES)
                   if "np.sqrt(SCALING_TRADING_DAYS)"
                   not in (ROOT / rel).read_text(encoding="utf-8")]
        assert not missing, (
            f"σ 年化实现点丢了常数引用：{missing} —— "
            "改点需同步 config 偏差备忘与 07 文档定位表")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
