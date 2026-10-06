"""回归测试：跨年预热被提前丢弃（11 年全部受影响）。

## 缺陷本体

`build_panel` 按年循环加载数据，docstring 明确承诺：

> 每年多加载 `WARMUP_TRADING_DAYS` 个**前置交易日**用于预热，
> 但只保留本年计算结果 —— 否则 mov250 在年初会全 NaN

但代码的实际顺序是：

```python
long = load_long_chunked(codes, pad_start, ye, ...)   # 含上一年末预热行
long = long[(long["date"] >= lo) & (long["date"] <= hi)]   # ← 先裁掉预热
s = compute_factor(f, long)                # ← 才算因子，预热已丢失
```

⇒ 上一年末那 260 个预热交易日**在因子计算之前就被丢掉了**。

## 为什么 rev5 首年全 NaN

`rev5` 的实现是 `px.groupby("code").shift(5)`，
要的是**同一股票**的前 5 行。预热行没了 ⇒ 每年 1 月开头 5 个交易日
每只股票的 `shift(5)` 都是 NaN ⇒ 因子值 100% 缺失。

## 实测（2016-2026 全市场 5,606 只 × 2,611 日）

| | 修复前 | 修复后 |
|---|---|---|
| 有效因子数 < 100 的天数 | **55** | **5** |
| 每日有效股票数（中位） | 4,088 | 4,146 |
| 每日有效股票数（min） | **0** | **0** |

修复前 55 天**全部**落在每年 1 月 2 日~ 1 月 10 日，11 年无一例外。

## 为什么会连累选股（BLOCK-2）

这 55 天里因子值几乎全缺失，而 `rank_topk` 只把 NaN 排到**末尾**、
仍返回满 `k` 个索引 ⇒ 未上市/已退市股票照样占掉持仓槽位。

## 剩余 5 天是合理的

2016-01-04~ 01-08 是**样本区间的前 5 个交易日**，
前面**确实没有数据**可供预热 —— 不是缺陷。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))


def _make_long(dates: list[str], code: str = "sh600000",
               px0: float = 10.0) -> pd.DataFrame:
    """构造一段单只股票的长表，价格逐日上涨。"""
    n = len(dates)
    close = px0 * (1.01 ** np.arange(n))
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "code": code,
        "open": close, "high": close, "low": close,
        "close": close,              # 原始价（未复权）
        "close_adj": close,          # 后复权价
        "high_adj": close, "low_adj": close,
        "volume": np.full(n, 1e6),
        "amount": np.full(n, 1e7),
    })


class TestCrossYearWarmup:
    # ⚠️ **必须用真实交易日历，不能用 `freq="B"`**。
    #   `pd.date_range("2016-01-04", periods=10, freq="B")` 会把
    #   1 月 4~8 日算作「周一~周五」，但真实 A 股 1 月 4 日才是
    #   当年第一个交易日（元旦假期）。
    #   ⇒ 用 freq="B" 造出来的「本年」实际有 10 个连续工作日，
    #   shift(5) 在**本年内部**就已经有值了 ⇒ 缺陷版测不出全 NaN。
    #   实测踩过：断言「裁掉预热后全 NaN」失败，实际有 5 个有效值。
    PRE = ["2015-12-14", "2015-12-15", "2015-12-16", "2015-12-17",
           "2015-12-18", "2015-12-21", "2015-12-22", "2015-12-23",
           "2015-12-24", "2015-12-25"]
    # 2016 年真实的前 10 个交易日
    CUR = ["2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07",
           "2016-01-08", "2016-01-11", "2016-01-12", "2016-01-13",
           "2016-01-14", "2016-01-15"]

    def test_年初第5日必须已有因子值(self):
        """🔴 核心判据：预热充足时，1 月初不该出现全NaN。

        构造2015 年末 10 天（预热）+ 2016 年初 10 天（本年）。
        只要预热行参与了计算，2016-01-08（本年第 5 日）就该有 rev5 值。
        """
        from factor_lab.factors.price_volume import compute_factor

        long = _make_long(self.PRE + self.CUR)
        first_of_year = pd.Timestamp(self.CUR[0])

        # ✅ 正解：算因子时**含**预热行，算完再裁剪
        wide = compute_factor("rev5", long).unstack("asset").sort_index()
        ok = wide[wide.index >= first_of_year]
        vals = ok.iloc[:, 0].dropna()
        assert len(vals) >= 5, (
            f"含预热行时 1 月初应有 ≥5 个有效值，实际 {len(vals)}")

        # ❌ 缺陷版：先裁掉预热行再算⇒ 本年前 5 日全 NaN
        cut = long[long["date"] >= first_of_year]
        bad = compute_factor("rev5", cut).unstack("asset").sort_index()
        bad_vals = bad.iloc[:, 0].dropna()
        first5 = bad_vals[bad_vals.index < pd.Timestamp(self.CUR[4])]
        assert len(first5) == 0, (
            f"裁掉预热后本年前 5 个交易日应当**全 NaN**（缺陷版特征），"
            f"实际有 {len(first5)} 个有效值 "
            f"{[str(d.date()) for d in first5.index]}")

    def test_两种顺序结果必须不同(self):
        """判据是**差异**：含预热 vs 不含预热，有效值个数必须显著不同。

        ⚠️ 只断言「含预热有值」不够 ——
        若两版恰好都算出值，这个用例就抓不到缺陷。
        """
        from factor_lab.factors.price_volume import compute_factor

        long = _make_long(self.PRE + self.CUR)
        first_of_year = pd.Timestamp(self.CUR[0])

        with_warm = compute_factor("rev5", long).unstack("asset").sort_index()
        with_warm = with_warm[with_warm.index >= first_of_year]
        without = compute_factor(
            "rev5", long[long["date"] >= first_of_year]
        ).unstack("asset").sort_index()

        n_ok = int(with_warm.notna().to_numpy().sum())
        n_bad = int(without.notna().to_numpy().sum())
        assert n_ok - n_bad >= 5, (
            f"含预热应至少多出 5 个有效值：{n_ok} vs {n_bad}（差 {n_ok - n_bad}）")

    def test_裁剪后不得混入区间外的行(self):
        """⚠️ 反向约束：**输出**面板不能含预热行。

        预热行只参与 rolling/shift 计算，不能进结果——
        否则面板的日期范围会比请求区间大，
        下游按日期切片时会错位。
        """
        from factor_lab.factors.price_volume import compute_factor

        long = _make_long(self.PRE + self.CUR)
        lo = pd.Timestamp(self.CUR[0])
        hi = pd.Timestamp(self.CUR[4])

        wide = compute_factor("rev5", long).unstack("asset").sort_index()
        trimmed = wide[(wide.index >= lo) & (wide.index <= hi)]
        assert trimmed.index.min() == lo
        assert trimmed.index.max() == hi
        assert len(trimmed) == 5, (
            f"裁剪后应只剩本年区间 5 天，实际 {len(trimmed)} 天")

    def test_价格面板不得含跨年伪收益(self):
        """⚠️ 价格面板**必须**只用区间内行（与因子相反）。

        混入上一年末会造出「跨年首日」的伪收益：
        `pct_change()` 把 2015-12-25 接到 2016-01-04 上，
        而这两天之间有元旦假期，不是真实的单日收益。
        """
        long = _make_long(self.PRE + self.CUR)
        keep = long["date"] >= pd.Timestamp(self.CUR[0])
        px = long[keep].pivot_table(index="date", columns="code",
                                    values="close_adj", aggfunc="last")
        assert px.index.min() == pd.Timestamp(self.CUR[0]), (
            "价格面板混入了区间外的日期 ⇒ 首日收益是跨年伪收益")
        assert len(px) == len(self.CUR)


class TestRankTopkNaN:
    """BLOCK-2：`rank_topk` 只把 NaN 排末尾，仍返回满 k 个。"""

    def test_有效因子不足时NaN股会进候选池(self):
        """记录事实（不是修它）—— 修法见 `selection.py` 的说明。

        这个测试的价值在于：**若将来 `rank_topk` 改成不返回 NaN，
        它会失败**，提醒我们同步更新依赖它的假设。
        """
        from factor_lab.analysis.selection import rank_topk

        fv = np.array([[3.0, 2.0, np.nan, np.nan, np.nan]])
        idx = rank_topk(fv, 5)
        assert idx.shape[1] == 5, "rank_topk 永远返回满 k 个"
        nan_in = [int(c) for c in idx[0] if np.isnan(fv[0, c])]
        assert nan_in, (
            "若这里没有 NaN 股票，说明 rank_topk 行为已变，"
            "请同步检查所有依赖「NaN 会占候选池」的代码与注释")

    def test_build_panel修复后不再年初全NaN(self):
        """端到端：真实数据上，1 月初必须有因子值。

        ⚠️ 只断言「1 月初有值」不够 ——
        若某天恰好有值但覆盖率极低，仍可能触发 BLOCK-2。
        故断言**有效股票数**的量级。

        ⚠️ 本用例依赖真实数据库 `data/market.db`（不进 git）：
        库不存在即跳过（CI 干净环境、未下载数据的环境），
        有库环境照常全市场实跑 —— 不能让它 error，那会卡红整条测试线。
        """
        from factor_lab.config import DB_PATH
        if not DB_PATH.exists():
            pytest.skip(f"缺 {DB_PATH} —— market.db 不进 git，"
                        "无数据环境跳过，有库环境照常全市场实跑")

        from run_long_only import build_panel, _cross_z
        from factor_lab.config import is_a_share
        from factor_lab.data import all_codes

        panels = build_panel(all_codes(), ["rev5"], "20160101", "20260930")
        z = _cross_z(panels["rev5"])
        acols = [c for c in z.columns if is_a_share(str(c))]
        zz = z[acols]
        eff = zz.notna().sum(axis=1)

        # 每年 1 月初（跳过样本首年，那里本就没有预热数据）
        jan = zz[zz.index.month == 1].iloc[5:]      # 每年 1 月第 6 日起
        assert len(jan) > 0, "样本里没有 1 月数据"
        eff_jan = zz.notna().sum(axis=1).loc[jan.index]
        worst = int(eff_jan.min())
        assert worst > 1000, (
            f"1 月初有效股票数最少的有一天只有 {worst} 只 —— "
            f"预热仍然失效（修复前实测是 0）")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))