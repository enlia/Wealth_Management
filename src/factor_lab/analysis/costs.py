"""交易成本模型：A 股实际水平与盈亏平衡换手率。

**本模块是从 `long_only.py` 拆出来的**（2026-10-06，接上涨停约束时拆分）：
`long_only.py` 因新增涨跌停注释达506 行，超过「≤500 行」硬性门禁
（同一门槛 2026-10-06 已因`TradabilityFilter` 拆过一次，
这次是第二次 —— 说明**该拆的是「职责」不是「行数」**）。

为什么成本模型要独立
--------------------
「手续费多少」与「怎么选股、怎么算净值」是**两个不同的问题**：
前者是**市场制度**（印花税 2023-08-28 减半、过户费双向），
不随策略变化；后者才是策略实现。
放在一起会让改成本假设时被迫读选股代码 ——
这正是本项目 `n_pick` 死参数能长期存在的原因的同款问题：
**职责不分开，失效就看不见。**

⚠️ **口径铁律**：盈亏平衡换手率用的是**单边**成本，
    因为调仓一次 = 一次卖出 + 一次买入，两个方向各付一次。
    实测（2026-10-06）：成本隐含系数 **2.520**
    = `0.0020 × 5分组 × 252/periods`，从代码反推而非直觉估算，
    用 `毛收益/0.0020` 会**低估成本 2.52 倍**。
"""
from __future__ import annotations

from dataclasses import dataclass


# ── 交易成本（A 股实际水平）────────────────────────────────────
@dataclass(frozen=True)
class CostModel:
    """单边交易成本。

    ⚠️ **双边**换算：买入 0.025%+ 过户费 0.001%，卖出 0.025% + 印花税 0.05%
       （印花税 2023-08-28 起减半为 0.05%，此前 0.1%）。
    手动交易的额外成本（研报/软件/时间）不计入，但**换手率要按实际调仓算**。
    """

    commission: float = 0.00025     # 佣金 万2.5
    stamp_duty: float = 0.0005      # 印花税 万5（卖出单边）
    transfer_fee: float = 0.00001   # 过户费 十万分之一（双向）
    slippage: float = 0.001         # 冲击成本 0.1%（保守估计）

    @property
    def buy(self) -> float:
        return self.commission + self.transfer_fee + self.slippage

    @property
    def sell(self) -> float:
        return self.commission + self.transfer_fee + self.slippage + self.stamp_duty

    @property
    def round_trip(self) -> float:
        return self.buy + self.sell

    def describe(self) -> str:
        return (f"买入 {self.buy*1e4:.2f}‱/ 卖出 {self.sell*1e4:.2f}‱ "
                f"（单边成本 {self.buy*1e4:.2f}‱）")


# 盈亏平衡换手率：换手率 × 单边成本必须 < 预期年化超额
def breakeven_turnover(target_annual_return: float, cost: CostModel) -> float:
    """给定目标年化收益，能承受的最高**年换手率**（倍数）。

    模型：``年化超额 ≥ 年换手率 × 单边成本``

    ⚠️ **这个数字通常大得没有参考价值**（如目标 20% 时是 158 倍年换手）。
    原因：单边成本只有 0.126%，理论上极高频也能覆盖成本。
    但**手动交易受的是换手率本身，不是成本**：

    | 调仓频率 | 年换手 | 现实可行性 |
    |---|---|---|
    | 月频 | 3~4 倍 | ✓ 手动可行 |
    | 季频 | 1~1.5 倍 | ✓ 手动舒适 |
    | 周频 | 12 倍+ | ✗ 盯盘成本高，且滑点会远超模型值 |
    | 日频 | 250 倍+ | ✗ 不可能 |

    **所以真正的约束是「手动能执行的换手上限」，约年化 4~6 倍**。
    对应目标年化超额 20%，需要扣成本后仍正：
    ``4倍 × 0.126% = 0.5%`` —— 成本占比很低，不是问题。

    真正的风险不在成本，在**因子衰减**和**个股集中度**，见 build_long_only。
    """
    if cost.buy <= 0:
        return float("inf")
    return target_annual_return / cost.buy


# 手动交易的年换手上限（现实约束，非成本约束）
MANUAL_TURNOVER_CAP = 6.0


def manual_turnover_note() -> str:
    return (f"手动交易年换手上限约 {MANUAL_TURNOVER_CAP:.0f} 倍"
            f"（月频 3~4 倍 / 季频 1~1.5 倍）。"
            f"对应年成本 {MANUAL_TURNOVER_CAP * CostModel().buy * 100:.2f}%，"
            f"不是瓶颈。")


# ── A 股特有的交易约束 ────────────────────────────────────────
# ⚠️ `TradabilityFilter` / `limit_state` 已拆到 `tradable.py`（2026-10-06）。
#    原因：本文件加注释后达 502 行，超过「≤500 行」硬性门禁。
#    **这两个符号无生产调用方，且涨跌停约束实际从未生效过** ——
#    `limit_state` 的 Series 分支必崩、缺数据口径还与 `limit_masks` 相反。
#    详见 `tradable.py` 的模块 docstring。
