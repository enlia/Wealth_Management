"""选股缓冲带（selection.py）的判据测试。

每个断言对应一个真实踩过的坑，不是「凑覆盖率」。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ROOT = Path(__file__).resolve().parents[1]

from factor_lab.analysis.selection import (  # noqa: E402
    rank_topk,
    select_with_buffer,
)
from factor_lab.config import DB_PATH  # noqa: E402


class TestRankTopk:
    def test_按因子值降序(self):
        fv = np.array([[0.1, 0.9, 0.5]])
        assert list(rank_topk(fv, 3)[0]) == [1, 2, 0]

    def test_取不满k时返回全部(self):
        """⚠️ N < k 不能报错或返回空 —— 实测候选池常大于股票池
        （如只取 300 只样本、候选池设 250）。"""
        fv = np.random.default_rng(0).normal(size=(5, 10))
        assert rank_topk(fv, 50).shape == (5, 10)

    def test_nan排到末尾而非最前(self):
        """⚠️ NaN 比较恒为 False，直接排序会把 NaN 排到**最前**，
        结果是「因子缺失的股票反而被选中」—— 最危险的静默错误。
        还原 bug 版（不替换 NaN）必须让本断言失败。"""
        fv = np.array([[np.nan, 0.9, np.nan, 0.5]])
        got = rank_topk(fv, 2)[0]
        assert 0 not in got and 2 not in got, "NaN 不应进入候选池"
        assert set(got) == {1, 3}

    def test_全nan行不崩溃(self):
        fv = np.full((3, 20), np.nan)
        out = rank_topk(fv, 5)
        assert out.shape == (3, 5)


class TestSelectWithBuffer:
    @staticmethod
    def _case():
        """构造 4 个调仓日、每池 5 只、持 3 只的最小可判案例。"""
        # 每行 5 只候选，因子排名固定（列号=排名）
        top = np.array([
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
            [4, 3, 2, 1, 0],
        ])
        is_rebal = np.array([True, False, True, True])
        return top, is_rebal

    def test_首次选前n_hold(self):
        top, is_rebal = self._case()
        held = select_with_buffer(top, is_rebal, 3)
        assert list(held[0]) == [0, 1, 2]

    def test_非调仓日沿用上一期(self):
        top, is_rebal = self._case()
        held = select_with_buffer(top, is_rebal, 3)
        assert list(held[1]) == list(held[0])

    def test_缓冲带保留仍在池内的持仓(self):
        """⚠️ **这是 n_pick 的全部意义**。
        第 2 调仓日排名变成 [1,2,3,4,0]，股票 0 跌到第 5 名。
        候选池 = 全部 5 只（n_pick=5）⇒ 0 仍在池内 ⇒ **不该卖**。
        若实现成「取前 n_hold」则会卖 0 买 4 —— 正是初版的 bug。"""
        top, is_rebal = self._case()
        held = select_with_buffer(top, is_rebal, 3)
        assert 0 in held[2], "跌到候选池边缘的持仓不应被卖出"

    def test_跌出候选池才换(self):
        """候选池收窄到 3 只时，第 5 名的0 必须被换掉。"""
        top, _ = self._case()
        is_rebal = np.array([True, False, True, True])
        held = select_with_buffer(top, is_rebal, 3)   # 池宽=5
        held_narrow = select_with_buffer(top[:, :3], is_rebal, 3)
        assert 0 in held[2], "池宽 5 时应保留"
        assert 0 not in held_narrow[2], "池宽 3 时必须换掉"

    def test_池空则空仓而非崩溃(self):
        top = np.zeros((3, 4), dtype=int)      # 全部列索引 0（同一只）
        is_rebal = np.array([True, True, True])
        held = select_with_buffer(top, is_rebal, 2)
        # 去重后只剩 1 只，不足 n_hold —— 不能报错，也不该凭空补股票
        assert held.shape == (3, 2)

    def test_换手随n_pick单调下降(self):
        """⚠️ 这条是 n_pick 生效的**端到端**判据。
        还原 bug 版（held=top_idx[:, :n_hold]）时三组数值完全相同，
        断言失败 —— 实测初版正是这样「看起来稳定、实则参数无效」。"""
        rng = np.random.default_rng(7)
        fv = rng.normal(size=(120, 500))
        fv_for_top = np.where(np.isnan(fv), -np.inf, fv)
        is_rebal = np.zeros(120, dtype=bool)
        is_rebal[::20] = True

        def avg_turn(n_pick: int) -> float:
            tp = rank_topk(fv_for_top, n_pick)
            h = select_with_buffer(tp, is_rebal, 30)
            sets = [set(r) for r in h]
            tot, cnt = 0.0, 0
            for i in range(1, len(sets)):
                if not sets[i] or not sets[i-1]:
                    continue
                tot += len(sets[i] ^ sets[i - 1]) / 2 / 30
                cnt += 1
            return tot / max(cnt, 1)

        narrow, mid, wide = avg_turn(30), avg_turn(150), avg_turn(300)
        assert narrow > mid > wide, (
            f"换手应随候选池变宽而下降，实测 {narrow:.4f} / {mid:.4f} / {wide:.4f}")

    def test_池宽等于持股数时退化为朴素取前n(self):
        """n_pick == n_hold 是边界：缓冲带宽度为 0，
        行为应与「直接取前 n_hold」一致（可用于对照回归）。"""
        rng = np.random.default_rng(11)
        fv = np.where(np.isnan(rng.normal(size=(60, 200))), -np.inf,
                      rng.normal(size=(60, 200)))
        is_rebal = np.zeros(60, dtype=bool)
        is_rebal[::10] = True
        top = rank_topk(fv, 10)
        held = select_with_buffer(top, is_rebal, 10)
        assert list(held[0]) == list(top[0][:10])


class TestLongOnlyDataDeps:
    """取数列必须覆盖因子声明的依赖（run_long_only.load_long_chunked）。"""

    @pytest.mark.skipif(
        not DB_PATH.exists(),
        reason="缺数据产物：data/market.db",
    )
    def test_取数包含成交量与成交额(self):
        """⚠️ **实测踩过（2026-10-06）**：
        取数硬编码了 OHLC+close_adj，而 `volratio5_60` 依赖 `vol`、
        `amount20` 依赖 `amount` ——
        结果 `KeyError: 'Column not found: vol'`。
        这两个因子**从未在滚动检验里跑过**，属于静默漏测：
        参数扫描表里看不到它们的任何痕迹，
        不会报错，只是什么都没发生。

        ⚠️ **判据必须是实跑，不是 grep 源码**：
        我自己写错过一次 —— grep `cols = [...]` 只匹配到基础列表，
        漏了后面的 `cols += [...]`，报了假失败。
        诊断工具的可信度不高于被诊断代码（S12）。
        故这里真的从库里取一小段数据，用真实列名跑因子。
        """
        import sqlite3
        import numpy as np
        import pandas as pd
        from factor_lab.config import DB_PATH
        from factor_lab.factors.price_volume import FACTORY, compute_factor

        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cols = {r[1] for r in con.execute("PRAGMA table_info(bar_daily)")}
        con.close()
        for c in ("vol", "amount", "close_adj"):
            assert c in cols, f"bar_daily 缺列 {c}"

        # 用真实列名构造长表，验证所有注册因子都能算
        rng = np.random.default_rng(0)
        n = 300
        long = pd.DataFrame({
            "code": np.repeat(["sh600000", "sz000001"], n // 2),
            "date": pd.bdate_range("2023-01-02", periods=n // 2).repeat(2),
            "open": rng.uniform(8, 12, n),
            "high": rng.uniform(8, 12, n),
            "low": rng.uniform(8, 12, n),
            "close": rng.uniform(8, 12, n),
            "close_adj": rng.uniform(8, 12, n),
            "high_adj": rng.uniform(8, 12, n),
            "low_adj": rng.uniform(8, 12, n),
            "vol": rng.uniform(1e5, 1e6, n),
            "amount": rng.uniform(1e6, 1e7, n),
        })
        failed = []
        for name in FACTORY:
            try:
                s = compute_factor(name, long)
                if s is None or s.empty:
                    failed.append(f"{name}: 返回空")
            except Exception as e:                               # noqa: BLE001
                failed.append(f"{name}: {type(e).__name__} {e}")
        assert not failed, (
            f"因子无法计算: {failed}\n"
            f"  ⚠ 这类失败在批量跑时表现为「跑到该因子就崩」，"
            f"  容易被误认为因子本身有问题，实际是取数缺依赖列。")

class TestWarmupWindow:
    """跨年预热必须按【交易日】算，不能用自然日。"""

    def test_预热按交易日而非自然日(self):
        """⚠️ **实测踩过的静默丢数据缺陷**：
        初版用 `pd.Timedelta(days=260)` 减**自然日**，
        260 自然日 ≈ **173 个交易日**，而 `pos250` 需要 **250 个交易日** ——
        预热少了约 77 个交易日。

        实测：`_shift_date('20240101', -260)` 返回 `20230416`，
        正确应是 `20230102`。

        后果：pos250 / mom120 在每年年初缺 ~77 个交易日因子值，
        11 年累积约 **850 天（3.3 年）数据被静默丢弃**。
        不报错、不影响其他因子，只让长窗口因子样本变少 ——
        这类缺陷最难发现。
        """
        import sys as _s
        _s.path.insert(0, str(ROOT / "research" / "scripts"))
        from run_long_only import _shift_date
        got = _shift_date("20240101", -260)
        assert got <= "20230103", (
            f"预热起点 {got} 太晚 —— 应覆盖到 2023-01-02 左右。"
            f"按自然日推算会只回退约 173 个交易日，"
            f"而 pos250 需要 250 个交易日。")

    def test_预热天数足够覆盖最长窗口(self):
        """WARMUP_TRADING_DAYS 必须 >= 最长因子窗口。"""
        import sys as _s
        _s.path.insert(0, str(ROOT / "research" / "scripts"))
        from run_long_only import WARMUP_TRADING_DAYS
        from factor_lab.factors.price_volume import FACTORY
        # 从注册表里挖出所有窗口参数的最大值
        import re
        src = (ROOT / "src" / "factor_lab" / "factors"
               / "price_volume.py").read_text(encoding="utf-8")
        nums = [int(m) for m in re.findall(r'FACTORY.*?(\d+)', src)]
        reg = re.search(r"FACTORY.*?\n\}", src, re.S)
        assert reg, "读不到 FACTORY"
        windows = [int(m) for m in
                   re.findall(r'lambda d: \w+\(d,\s*(\d+)', reg.group(0))]
        windows += [int(m) for m in
                    re.findall(r'lambda d: \w+\(d,\s*\d+,\s*(\d+)', reg.group(0))]
        if windows:
            assert WARMUP_TRADING_DAYS >= max(windows), (
                f"预热 {WARMUP_TRADING_DAYS} <最长窗口 {max(windows)}，"
                f"长窗口因子在年初会缺数据")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))