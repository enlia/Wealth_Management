"""A 股特有的交易约束（量价因子完全没覆盖）。

**本模块是从 `long_only.py` 拆出来的**（review 2026-10-06 复核 BLOCK-1）：
`long_only.py` 因本轮新增注释达502 行，超过「≤500 行」硬性门禁。
拆出的内容是 `TradabilityFilter` 与 `limit_state` —— 两者都**无生产调用方**，
但保留是因为涨跌停约束本身仍是项目待办（见下）。

## ⚠️ 现状：涨跌停约束**从未真正生效过**

`319af50` 的 commit message 写「涨跌停是 A 股手动交易最大的隐性风险，已建模待用」。
实际情况是「**已建模，但跑不起来**」：

1. `limit_state` 的 `prev_close` 分支写了两条路：
   ```python
   if isinstance(prev_close, pd.Series):
       up = close >= limit_price(prev_close, lim) - 1e-6   # ← 必崩
   ```
   而 `market_rules.limit_price(prev_close: float, ...)` **只接受标量**，
   传整个 Series 会抛
   `ValueError: The truth value of a Series is ambiguous.`
   ⇒ **Series 分支从来没被执行过**，`TradabilityFilter` 的涨跌停能力是死代码。

2. 更糟的是**口径与 `tradability.limit_masks` 相反**：
   | | 缺涨跌停价时的判定 | 后果 |
   |---|---|---|
   | `limit_masks`（已修） | `True` = **不可交易** | 保守，宁可少买 |
   | `limit_state`（本模块） | `fillna(False)` = 可交易 | 乐观，可以乱买 |

   `close >= NaN` 返回 False（不是 NaN），所以 `fillna(False)` 生效 ——
   与 `limit_masks` 当初「fillna 兜底根本不触发」是**同款陷阱、相反方向**。

3. **两套口径并存**本身就是隐患：修好其中一套，另一套还在原地。

## 处置

- `limit_state` 的 **Series 分支已删除**（无法工作，保留只会误导）。
  标量分支保留，作为「单只股票逐日判定」的轻量入口。
- **生产路径请统一走 `factor_lab.analysis.tradability.limit_masks`**：
  它有覆盖率自检、缺数据判不可交易、索引对齐校验。

## 对已有结论的影响

`docs/03_项目报告/12_价量因子有效性检验报告.md` 的全部结论都是在
**完全没有涨跌停约束**下得出的。方向性影响：

- 涨停约束会**剔除最强的候选股**（买不进）
- ⇒ 真实收益只会**更差**，不会更好
- ⇒ 该报告的结论是**偏保守的**，这个缺陷不推翻其结论

修好约束后应重跑一次，预期净收益进一步下降。

---

> 本文件由 review 2026-10-06 触发拆分。相关坑见 `.github/standards/CODE_TRUST.md` P22/P23
（注释声称有保护、代码里没有 —— 「已建模待用」正是这种形态）。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# ── A 股特有的交易约束（量价因子完全没覆盖）──────────────────
@dataclass
class TradabilityFilter:
    """涨跌停与停牌约束。

    ⚠️ 这是 A 股手动交易**最大的隐性风险**，而所有因子研究都忽略了它：
      · 涨停时买不到（买单挂不上）
      · 跌停时卖不出（想跑跑不掉）
      · 停牌期间完全无法操作

    实测：本项目采样区间内主板日均涨停 53 只、跌停 4 只；
    一只强势股在买入信号出现时可能已经涨停 —— 因子说「买」，
    实际买不到，等下一个信号就是一周后。

    **对「赚钱效率」的影响**：选出的股票越强势，越买不到。
    这会系统性削弱动量类因子的实际收益。

    ⚠️ **本类目前没有任何生产调用方**（见模块 docstring）。
       接线前请先读上面的「现状」一节。
    """

    skip_limit_up: bool = True    # 买入时跳过涨停
    skip_limit_down: bool = False # 卖出时：跌停只能等，不算可卖
    max_suspended_days: int = 3   # 停牌超过 N 天视为不可用
    volume_cap_ratio: float = 0.0  # 单日成交量不超过流通盘的 N 倍（0=不限制）


def limit_state(
    close: pd.Series, high: pd.Series, low: pd.Series,
    code: str, prev_close: float,
) -> pd.Series:
    """判定每只股票每日的涨跌停状态（**标量 `prev_close` 专用**）。

    返回 True 表示「处于涨跌停状态」（买不到或卖不出）。

    ⚠️⚠️ **缺数据一律判 False（= 可交易），这是错的**：
       实测 `close >= NaN` 返回 False，`fillna(False)` 会生效 ——
       缺涨跌停价的格子被当成「未封板」，可以随意买卖。
       `tradability.limit_masks` 对同样的情形判**不可交易**（保守）。

    ⇒ **生产路径不要用本函数**，统一走
       `factor_lab.analysis.tradability.limit_masks`
       （有覆盖率自检 + 缺数据判不可交易 + 索引对齐校验）。

    ⚠️ 原实现有一个 `isinstance(prev_close, pd.Series)` 分支，
       但 `market_rules.limit_price` 只接受标量，传 Series 必崩
       （`ValueError: The truth value of a Series is ambiguous`），
       且该分支**从未被执行过** —— 属于「注释声称有、代码里没有」，
       已删除。保留一个跑不通的分支只会让人以为这条路能用。
    """
    from factor_lab.market_rules import limit_of, limit_price

    lim = limit_of(code)
    base = float(prev_close)
    up = close >= limit_price(base, lim) - 1e-6
    dn = close <= limit_price(base, -lim) + 1e-6
    return (up | dn).fillna(False)
