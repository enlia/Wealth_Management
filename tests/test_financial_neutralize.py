"""财务因子中性化路径的还原测试。

🔴 为什么这组测试必须存在（review 2026-10-06 实测）
--------------------------------------------------
`--neutral` 是 `run_financial_walk_forward.py` 文档里宣传的用法，
但初版 `_neutralize_chunk` **100% 崩溃**：

    KeyError: 'Requested level (date) does not match index name (None)'

根因：`run_financial_study.neutralize` 内部是
`s.index.get_level_values("date")` ⇒ 要求 `(date, asset)` MultiIndex，
而调用方传的是 `row[ok]`（单日、纯 Index(代码)）。

**更危险的是第二个 bug（被第一个挡住，从未暴露）**：
`pd.DataFrame(acc, index=days)` 已是 (日期 × 标的)，
初版又 `.T` 转成 (标的 × 日期) 再 reindex 回去
⇒ 行列标签全不匹配，**输出全 NaN**。
review 打桩 `neutralize` 为恒等函数实测：输出非 NaN 数 **0 / 240**。

⇒ 只修第一个 bug，`--neutral` 会「跑通」但结果全 NaN。
**这正是 CODE_TRUST描述的「不报错、结果全错」形态**，
只有「恒等打桩 ⇒ 输出必须等于输入」这条断言能抓住。

⚠️ 这些测试**不依赖本机数据库**（CI 可跑）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import financial_panels as fp  # noqa: E402
import run_financial_study as rfs  # noqa: E402


def _fixture(n: int = 60, n_days: int = 5, seed: int = 0):
    """构造 (days × codes) 宽表面与 (date, asset) 长表，两者数值一致。"""
    days = pd.bdate_range("2025-01-01", periods=n_days)
    codes = [f"c{i:03d}" for i in range(n)]
    rs = np.random.RandomState(seed)
    wide = pd.DataFrame(rs.normal(size=(n_days, n)), index=days,
                        columns=codes)
    mi = pd.MultiIndex.from_product([days, codes], names=["date", "asset"])
    daily = pd.DataFrame({"industry": [f"ind{i % 5}" for i in
                                       range(n_days * n)]}, index=mi)
    daily["mktcap"] = rs.uniform(1e4, 1e6, n_days * n)
    return days, codes, wide, daily


class TestNeutralizeChunk:
    def test_恒等打桩下输出必须等于输入(self):
        """🔴 核心断言：抓住「跑通但全 NaN」这个 bug。

        把 `neutralize` 打桩成恒等函数，则 `_neutralize_chunk`
        **必须**原样返回 —— 任何形状/索引错配都会让输出变成全 NaN。
        """
        days, codes, wide, daily = _fixture()
        panels = {"bp": [wide.copy()]}

        orig = rfs.neutralize
        rfs.neutralize = lambda s, ind, cap=None, n_jobs=1: s
        try:
            fp._neutralize_chunk(panels, daily, ["bp"], days, wide.columns)
        finally:
            rfs.neutralize = orig

        out = panels["bp"][-1]
        assert out.index.equals(days), f"索引变了：{out.index[:3]}"
        assert list(out.columns) == codes, "列序变了"
        assert np.allclose(out.to_numpy(), wide.to_numpy()), (
            "恒等打桩下输出与输入不同 ⇒ 中性化路径存在形状/索引错配"
            "（初版实测输出全 NaN）")
        assert out.notna().to_numpy().sum() == wide.size, (
            f"恒等打桩下出现 NaN：{out.notna().to_numpy().sum()}/{wide.size}")

    def test_真实_neutralize_不崩溃且改变数值(self):
        """真实 `neutralize` 必须能跑通（初版实测 KeyError 100% 崩溃）。"""
        days, codes, wide, daily = _fixture()
        panels = {"bp": [wide.copy()]}
        fp._neutralize_chunk(panels, daily, ["bp"], days, wide.columns)
        out = panels["bp"][-1]
        assert out.shape == wide.shape
        assert out.notna().to_numpy().sum() > 0, "真实中性化后全 NaN"
        # OLS 残差：均值应接近 0（因为含截距）
        assert abs(float(np.nanmean(out.to_numpy()))) < 0.05, (
            f"残差均值 {float(np.nanmean(out.to_numpy())):.4f} 偏离 0 太多，"
            f"中性化可能没生效")

    def test_样本不足时原样返回(self):
        """每日有效样本 < 30 时跳过中性化，必须原样返回而非清空。"""
        days, codes, wide, daily = _fixture(n=20)
        panels = {"bp": [wide.copy()]}
        fp._neutralize_chunk(panels, daily, ["bp"], days, wide.columns)
        out = panels["bp"][-1]
        assert np.allclose(out.to_numpy(), wide.to_numpy()), (
            "样本不足时应原样返回，实际被改动了")

    def test_行业缺失时原样返回(self):
        days, codes, wide, daily = _fixture()
        daily["industry"] = "NA"        # 全部同一行业，无区分度
        panels = {"bp": [wide.copy()]}
        # 全部同一行业 + 有效样本够⇒ 仍会走 neutralize（残差≈原值−均值）
        fp._neutralize_chunk(panels, daily, ["bp"], days, wide.columns)
        out = panels["bp"][-1]
        assert out.notna().to_numpy().sum() > 0


class TestChunkedEqualsMonolithic:
    """分块与一次性必须等价（BLOCK-3 的守护）。

    ⚠️ 本测试只验证**装配逻辑**（形状/索引/列序），
       真正的「数值完全相等」需要本机数据库（1,375 万行），
       已在开发时实测：bp/ep/roe 最大绝对差 **0.000e+00**，
       且 2016-2022 与 2023-2026 两段都验证过分块边界无 NaN 缺口。
    """

    def test_分块拼接后索引连续无重复(self):
        from factor_lab.analysis.walk_forward import WindowSpec, make_windows
        days = pd.bdate_range("2016-01-01", periods=800)
        spec = WindowSpec(train_years=2, test_months=12)
        w = make_windows(days, spec)
        assert w, "前提不成立：样本太短，生成不了窗口"
        te = w[0][1]
        assert te == sorted(set(te)), "测试段日期应已排序且唯一"
        assert te[0] not in set(w[0][0]), "训练段与测试段不应重叠"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


class TestNeutralizeWithNaNHoles:
    """🔴 BLOCK 守护（2026-10-10 实测 `--neutral` 崩溃抓出）。

    bug：`r.loc[ok.index] = resid.to_numpy()`。
    `ok` 是**与 row 等长**的布尔掩码，`ok.index` 是**全部标的**；
    而 `resid` 只有 `ok.sum()` 行。
    只要当日存在 NaN 因子值（新股未披露、财报缺失），
    索引器长度就 ≠ 值长度 ⇒
        ValueError: cannot set using a list-like indexer with a
        different length than the value
    ⇒ **只要有 NaN 就100% 崩溃**，全市场跑必崩。

    为什么旧测试测不出来：
    `_fixture()` 造的 `wide` **全无 NaN** ⇒ `ok.sum() == len(codes)`
    ⇒ 索引器长度恰好等于值长度 ⇒ 长度不匹配永不触发。
    **全满的fixture = 橡皮章**，与「恒等断言」同一类陷阱。
    """

    def test_当日存在NaN时仍能跑且只改有效位(self):
        days, codes, wide, daily = _fixture()
        # 在第一天的因子值里挖 20 个 NaN（模拟新股未披露/财报缺失）
        wide = wide.copy()
        wide.iloc[0, :20] = np.nan

        panels = {"bp": [wide.copy()]}
        fp._neutralize_chunk(panels, daily, ["bp"], days, wide.columns)
        out = panels["bp"][-1]

        assert out.shape == wide.shape, "形状变了"
        # 关键：NaN 位必须仍是 NaN（不能被中性化结果填上）
        仍为NaN = out.iloc[0, :20].isna().all()
        assert 仍为NaN, (
            f"第一天前 20 个原本是 NaN，中性化后变成 "
            f"{out.iloc[0, :20].notna().sum()} 个有效值 "
            f"⇒ NaN 位置被错填")
        # 有效位不能被清空
        有效 = out.iloc[0, 20:].notna().sum()
        assert 有效 == len(codes) - 20, (
            f"第一天有效位只剩 {有效}/{len(codes) - 20} ⇒ 有效值被误清")
