"""A 股交易规则：板块归属与涨跌幅限制。

为什么独立成模块
----------------
涨跌幅限制是**市场制度**，不是数据质量指标。但复权验证、回测撮合、
涨跌停打板策略都要用同一份判定。三处各写一遍必然漂移
（本项目实测踩过：验证脚本按 10% 硬判定科创板，把 20% 内的真实涨停
误判成复权失败，得出「复权只消除 21.9%」的错误结论）。

板块判定依据
------------
按**代码数字段**判定，不看首位 —— 首位不可靠（沪市 600/601/603/605/688
混在一起，深市 000/001/002/003/300 混在一起）。

涨跌幅限制与报价取整
--------------------
交易所涨停价 = 前收盘 × (1 + 限幅)，**四舍五入到 0.01 元**。
这意味着真实涨停幅度会**超过**名义限幅，且低价股最严重：

    前收 2.66 → 涨停 2.93 = +10.15%
    前收 1.05 → 涨停 1.16 = +10.48%
    前收 0.30 → 涨停 0.33 = +10.00%

⚠️ 固定容差（如 1.005）**必然误判**低价股涨停。
  实测 sz002426（收盘价 1.6~4.5 元）31 条残留中 27 条是
  `收盘 == 最高` 的真实封板涨停，误判率 87%。

正确做法：**容差随前收盘价动态计算**，公式见 `price_tolerance()`。
它在纯取整撑大值之上再加**半个报价单位**的余量
（`HALF_TICK / prev_close`）以吸收浮点误差 —— 实测全价格区间下
名义 10% 的最大可容忍收益率为 **12.86%**（见 `max_tolerance()`）。

历史制度变更
------------
**注册制新股上市前 5 日不设涨跌幅限制**：
  · 科创板 2019-07-22 起（首批）
  · 创业板 2020-08-24 起（注册制首批）
  · **主板 2023-02-17 起（全面注册制）** ← 这一条极易漏掉

  实测依据：主板新股 sh603124（20250320 上市）、sz001287（20230410）、
  sz001248（20260702）等 11 条超限跳空，**全部**落在上市第 2~5 个交易日。
  只按「科创板/创业板前 5 日无限制」判定，会把这 11 条误判成复权失败。

⚠️ 创业板历史限幅分段（2020-08-24 之前也是 ±10%）本模块**未**建模。
  若做覆盖该日期之前的长历史回测，需自行按日期取限幅，
  否则会把创业板早期的真实涨停误判为超限。

用途
----
  from factor_lab.market_rules import board_of, limit_of, no_limit_days
  board_of("sh688205")                  # -> 'star'
  limit_of("sh688205")                  # -> 0.20
  no_limit_days("sh688205", 6)          # -> True（上市第 6 日仍无限制）
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

# 名义涨跌幅限制（小数），按当前制度
LIMIT: dict[str, float] = {
    "main": 0.10,
    "star": 0.20,
    "gem": 0.20,
    "bse": 0.30,
}

# 报价最小变动单位（元）
TICK = 0.01

# 浮点比较余量：半个报价单位。用于 price_tolerance 的返回值。
# ⚠️ 不是数学常数，是「允许 0.005 元报价误差」的物理容差。
#    真实收益由除法算出，与除法算出的阈值在浮点下可能差 1 ULP，
#    不给余量会把「恰好合法涨停」误判成超限（实测 300 只样本虚增 3倍异常）。
HALF_TICK = TICK / 2

# 容忍度计算的**价格下界**（元）。
#
# 取值依据（实测 A 股 5,593 只 / 1,640 万行）：最低 0.26 元（sz000004），
# 低于 0.30 元的不足0.1%。取 0.30 元作下界：
#   · 覆盖全部真实 A 股价格
#   · 避开 [0.01, 0.04] 的病态区间 —— 该区间内
#     round(pc×1.1, 2) == pc（涨停价等于前收），
#     容差会退化成 HALF_TICK/pc，在0.05 元处高达 30%，
#     让max_tolerance 的兜底阈值失去筛除能力。
# ⚠️ 不要取 0.01：那正是上面说的病态区间起点。
MIN_PRICE = 0.30

# 容忍度计算的**价格上界**（元）。实测 A 股最高 2,832.92 元（bj899601），
# 取 5,000 留余量。
MAX_PRICE = 5000.00

BOARD_CN = {"main": "主板", "star": "科创板", "gem": "创业板", "bse": "北交所"}


def board_of(code: str) -> str:
    """本地代码（sh600519 / sz300489 / bj920790）-> 板块标识。

    未知形态一律判为``main``（最严限幅），宁可误报超限也不漏报 ——
    宽松默认会让创业板污染混进主板口径里，把主板污染率算低。
    """
    c = code.strip().lower()
    if not c:
        raise ValueError(f"代码为空：{code!r}")
    if c.startswith("bj"):
        return "bse"
    num = c[2:] if len(c) > 2 else ""
    if num.startswith(("688", "689")):
        return "star"
    if num.startswith(("300", "301")):
        return "gem"
    return "main"


def limit_of(code: str) -> float:
    """该股票的**名义**涨跌幅限制（不含报价取整容差）。"""
    return LIMIT[board_of(code)]


def board_cn(code: str) -> str:
    """板块中文名，用于报告输出。"""
    return BOARD_CN[board_of(code)]


def limit_price(prev_close: float, limit: float) -> float:
    """涨停价 / 跌停价（前收盘 × (1±限幅)），对齐到 0.01 元。

    交易所实际规则是四舍五入（0.125 → 0.13），但 Python 内置 round 用的是
    **银行家舍入**（0.125 → 0.12，0.135 → 0.14），两者在 .005 的奇偶位上不一致。
    这里显式实现四舍五入，不依赖 round()。

    ⚠️ 用 Decimal 而非 float：0.1 * 1.1 在二进制浮点下是 0.11000000000000001，
       直接 round 会偶发差一档。金融价格必须走定点数。
    """
    if prev_close <= 0:
        raise ValueError(f"前收盘价必须为正：{prev_close!r}")
    raw = Decimal(str(prev_close)) * (Decimal(1) + Decimal(str(limit)))
    px = raw.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(px)


def price_tolerance(prev_close: float, limit: float) -> float:
    """给定前收盘价时，名义限幅可被报价取整撑大多少。

    返回的是**可容忍的最大收益率**（= 实际涨停价/ 前收盘 − 1 + 半档余量）。
    这才是判定「是否超限」的正确阈值。

    示例（前收 2.66，名义限幅 10%）：
      limit_price → 2.93，纯取整撑大 +10.15%，含余量 10.34%
    示例（前收 1.05，名义限幅 10%）：
      limit_price → 1.16，纯取整撑大 +10.48%，含余量 10.95%

    ⚠️ 返回值含**半个报价单位**的余量，不是精确的涨停幅度。
       原因：真实收益由 `close / prev_close - 1` 算出，
       与 `limit_price / prev_close - 1` 在浮点下可能差 1 ULP。
       实测踩过：sh603396 的 `ret=+0.0999` 与 `tol=0.0999` 数学上相等，
       `ret.abs() <= tol` 却因浮点误差判成超限，
       300 只样本里凭空多出 3,395 条异常（是真实值的 3 倍），
       差点得出「复权反而使污染变多」的反向结论。
       半档 tick 的余量 = 允许 0.005 元报价误差，有物理含义，不是拍脑袋的 epsilon。

       余量随价格单调收缩：前收 0.26 元 → 1.92pp；1258.62 元 → 0.0004pp。
       真实 -30% 跌幅在任何价位都会被正确判为超限。

    前收盘价未知时用 `max_tolerance()` 兜底，不要用固定 1.005。
    """
    return limit_price(prev_close, limit) / prev_close - 1 + HALF_TICK / prev_close


def max_tolerance(limit: float = 0.10) -> float:
    """在 [MIN_PRICE, MAX_PRICE] 全价格区间内，容差的最大值。

    用于「没有前收盘价信息」时的兜底阈值（如预筛阶段）。
    实测值（含 HALF_TICK 余量）：
      名义 10% → 12.8571%    名义 20% → 22.7273%    名义 30% → 32.8571%

    ⚠️ 这个值偏宽是**正常的**：低价股 0.01 元的取整误差占比就是大。
      它只用作粗筛，最终判定必须走 `price_tolerance()`的逐日精确计算。
    """
    worst = 0.0
    lo = round(MIN_PRICE * 100)
    for cents in range(lo, round(MAX_PRICE * 100) + 1):
        pc = cents / 100
        t = limit_price(pc, limit) / pc - 1 + HALF_TICK / pc
        if t > worst:
            worst = t
    return worst


def is_limit_up(prev_close: float, close: float, code: str) -> bool:
    """收盘是否封涨停板（用于区分「真实涨停」与「复权残留」）。

    ⚠️ 容差口径必须与 `price_tolerance()` 一致 —— 那里加的是**半档** tick，
    这里减的也必须是半档。早��版本减的是整档 TICK，导致：
        主板 10.00 → 11.10（+11%，明确超限）被判True
    函数完全失去鉴别力（任何涨幅 ≥ 约 9% 都返回 True）。

    判据：涨幅达到涨停价，且未超出浮点余量。
    """
    if prev_close <= 0:
        raise ValueError(f"前收盘价必须为正：{prev_close!r}")
    tol = price_tolerance(prev_close, limit_of(code))
    return close / prev_close - 1 > tol - HALF_TICK / prev_close


# 注册制新股上市后不设涨跌幅限制的交易日数。
#
# ⚠️ 三个板块的「注册制生效日」不同，早于该日上市的新股不适用 5 日规则：
#     科创板 2019-07-22 / 创业板 2020-08-24 / 主板 2023-02-17（全面注册制）
#   主板早于 2023-02-17 的新股首日限幅是「有效竞价范围 ±44%/-36%」，
#   次日起恢复 ±10%，即第2 日起就有限幅，不属于「无限幅窗口」。
#
# 主板新股首日的 ±44%/-36% 是**价格笼子**机制，不是涨跌幅限制，
# 本模块不建模该区间 —— 判定新股首日超限时需另查价格笼子规则。
NO_LIMIT_DAYS: dict[str, int] = {"star": 5, "gem": 5, "main": 5, "bse": 1}

# 各板块注册制生效日（YYYYMMDD）。新股首个行情日早于此，则不适用 5 日规则。
REGISTRATION_START: dict[str, int] = {
    "star": 20190722,
    "gem": 20200824,
    "main": 20230217,
    "bse": 20211115,
}


def no_limit_days(code: str, trade_day_index: int,
                  first_trade_date: int | None = None) -> bool:
    """该股票的第 ``trade_day_index`` 个交易日（1 起）是否无涨跌幅限制。

    实测依据：复权后残留跳空**全部**落在新股上市第 2~5 个交易日，
    无一例外。例：
      sz300873 / sz300871 / sz300878 均于 2020-08-24（创业板注册制首日）上市，
      2020-08-25 是第 2 个交易日，单日涨 +37.7% 属规则内；
      sh603124 于 2025-03-20 上市（全面注册制后主板），
      2025-03-21 第 2 日跌 −22.5% 亦属规则内。

    参数
    ----
    first_trade_date: 该股首个行情日（YYYYMMDD）。
        **必须传**，否则2023-02-17 之前上市的主板新股会被误判为无限幅。

    示例
    ----
      no_limit_days("sh688205", 6)                # -> False
      no_limit_days("sh603124", 2, 20250320)      # -> True
      no_limit_days("sz300873", 2, 20200824)      # -> True
    """
    board = board_of(code)
    n = NO_LIMIT_DAYS[board]
    if n <= 1:
        return trade_day_index <= n
    # 注册制生效前上市的新股不适用「前5 日无限幅」
    if first_trade_date is not None and first_trade_date < REGISTRATION_START[board]:
        return trade_day_index <= 1
    return trade_day_index <= n
