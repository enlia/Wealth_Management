"""收益研究所需的**两种价格口径**，一次取对。

为什么必须显式区分（2026-10-06 实测踩坑）
------------------------------------------
财务因子研究此前只用 `load_prices(field="close")` 一个面板，
同时充当**两个互相冲突的角色**：

| 角色 | 必须用哪个价 | 原因 |
|---|---|---|
| 算 PB / PE / EP | **未复权 `close`** | `bps`/`eps` 是财报里**实际披露**的每股数字，未复权价与它同一口径 |
| 算前瞻收益 / IC / 回测 | **后复权 `close_adj`** | 未复权价在除权日出现**假跳空** |

用一个面板服务两个角色 ⇒ 收益端被除权缺口污染。
**实测全市场 5,606 只等权口径的年化偏差**：

| 年份 | 未复权 | 后复权 | 偏差 |
|---|---|---|---|
| 2016 | −1.08% | +15.72% | **−16.80pp** |
| 2019 | +41.06% | +51.86% | −10.80pp |
| 2025 | +63.53% | +70.49% | −6.97pp |
| 2021~2026 每年 | — | — | −5.1 ~ −8.0pp |

⚠️ 这个偏差（5~17pp/年）**大于本项目声称的任何因子收益**（最高的 ep 也只有 +6.1%）。
用它算出来的 IC 与多空收益全部不可信 —— 因子在给「分红除权」定价，不是在给股票定价。

设计
----
`load_factor_prices()` 返回一个**具名对**，让两种角色在类型上就分得开：

    raw, adj = load_factor_prices(codes, start, end)
    daily = build_financial_factors(panel, days, raw)   # PB 用 raw
    tear_sheet(factor, adj)                             # 收益用 adj

⚠️ **不要图省事只取一个**。若只需要收益（如价量因子研究），
直接用 `load_prices(field="close_adj")`，那个路径本来是对的。
本模块只服务于**同时需要两者的财务因子研究**。
"""
from __future__ import annotations

from factor_lab.data.sqlite_source import load_prices

__all__ = ["FactorPrices", "load_factor_prices"]


class FactorPrices(tuple):
    """``(raw, adj)`` 具名对：未复权价 + 后复权价。

    用 NamedTuple 之外的 tuple 子类是为了在解包时仍然可以
    ``raw, adj = load_factor_prices(...)``（多数脚本这么写），
    同时在属性访问时 ``p.raw`` / ``p.adj`` 也能用。
    """

    __slots__ = ()

    def __new__(cls, raw, adj):
        return super().__new__(cls, (raw, adj))

    @property
    def raw(self):
        """未复权收盘价 —— 只用于与财报披露数字同口径的 PB / PE / EP。"""
        return self[0]

    @property
    def adj(self):
        """后复权收盘价 —— **所有收益、IC、回测都必须用这个**。"""
        return self[1]


def load_factor_prices(
    codes: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> FactorPrices:
    """同时取未复权与后复权收盘价，返回具名对。

    ⚠️ 两个面板**行索引与列索引必须完全一致** ——
    `build_financial_factors` 会用 raw 的index 当交易日轴，
    再用 adj 算收益；若两者日期轴不一致，会静默错位。
    这里显式校验，不一致直接抛错（禁止静默 fallback，ENGINEERING 第四节）。
    """
    raw = load_prices(codes, start, end, field="close")
    adj = load_prices(codes, start, end, field="close_adj")
    if raw.empty or adj.empty:
        raise ValueError(
            f"价格面板为空：raw {raw.shape} / adj {adj.shape}。"
            "若 adj 为空，说明该区间没有 close_adj 数据，"
            "**禁止退回未复权价继续算收益**（会引入 5~17pp/年偏差）。")
    if not raw.index.equals(adj.index) or not raw.columns.equals(adj.columns):
        raise ValueError(
            "raw 与 adj 的索引不一致，拒绝继续：\n"
            f"  raw {raw.shape} index[{raw.index[0]}~{raw.index[-1]}]\n"
            f"  adj {adj.shape} index[{adj.index[0]}~{adj.index[-1]}]\n"
            "两套口径必须覆盖同一批交易日与标的，否则收益会静默错位。")
    return FactorPrices(raw, adj)