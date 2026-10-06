"""价量因子：全部为纯函数，输入长表，输出 alphalens 要求的 MultiIndex(asset, date)。

⚠️ 前视偏差红线（AGENTS.md 第2节）
   每个因子只能使用【截至因子计算日】的信息，不得包含任何未来数据。
   本模块所有实现均以 shift 正数对齐历史窗口，天然无前视。

⚠️ A股特有约束（config.ResearchConfig.skip_recent_days）
   A股 T+1 制度下，日内动量与隔夜动量方向相反、相互抵消。
   学术研究指出：使用动量因子必须跳过最近约 1 个月，否则信号被抵消。
   因此本模块的动量窗口从 20 日起，不用 5/10 日。

⚠️ 输出格式（实测踩坑，pandas 2.x）
   不用 DataFrame.stack() 构造 MultiIndex —— 2.x 的层级顺序与旧版不同
   （会得到 (date, 'close') 而非 ('close', date)），导致 alphalens
   读到的 date 层是字符串 'close' 并报频率推断错误。
   改用 MultiIndex.from_arrays 显式构造，跨版本稳定。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mk(values: pd.Series, codes: pd.Series, dates: pd.Series, name: str) -> pd.Series:
    """构造 alphalens 要求的 MultiIndex Series。

    ⚠️ 层级顺序必须是 (date, asset) —— alphalens 文档原文：
       "A MultiIndex Series indexed by timestamp (level 0) and asset (level 1)"
       传反会在 compute_forward_returns 里报
       `AttributeError: 'Index' object has no attribute 'tz'`（因为 level 0
       变成了 asset 字符串 Index，没有 .tz 属性）。

    ⚠️ 不用 DataFrame.stack()：pandas 2.x 的层级顺序与旧版不同，
       显式 from_arrays 跨版本稳定。
    """
    n = len(values)
    idx = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(dates.to_numpy()), codes.to_numpy()],
        names=["date", "asset"],
    )
    if len(idx) != n:
        raise ValueError(f"索引长度 {len(idx)} 与数据长度 {n} 不一致")
    s = pd.Series(values.to_numpy(), index=idx, name=name)
    return s.sort_index(level=[0, 1], kind="stable")


def _prep(long: pd.DataFrame) -> pd.DataFrame:
    """统一预处理：排序 + 校验必要列。"""
    need = {"code", "date", "close"}
    missing = need - set(long.columns)
    if missing:
        raise KeyError(f"长表缺少列 {missing}")
    d = long.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["code", "date"], kind="stable").reset_index(drop=True)


# 收益类因子统一用**后复权价**做口径，由 `PRICE_COLS` 声明每个价格位的列名。
#
# ⚠️ 为什么必须复权（AGENTS.md 第二节）
#   未复权价在除权日出现假跳空：实测主板 0.291% 的日收益超过 ±10% 限制，
#   年化偏差 2.33pp。未复权价算出的动量/反转/波动率全部含这层污染。
#
# ⚠️ 为什么**高低价也要复权**（不能只换 close）
#   hh_hl_score 用 high/low 判「高点是否抬升」。除权日 high 会机械下移，
#   不复权会把除权误读成「高点降低」——趋势结构因子直接失效。
#   复权是乘性调整，同一天三个价格位共用一个乘数，故可直接推导。
PRICE_COLS = {"close": "close_adj", "high": "high_adj", "low": "low_adj"}


def _px(d: pd.DataFrame, field: str = "close") -> pd.Series:
    """取价格序列，优先用后复权列。

    长表由 ``load_long(adjusted=True)`` 产出时才有 ``*_adj`` 列。
    若请求复权列但长表里没有，**直接报错** —— 静默退回未复权价
    会让整个研究在有污染的口径上跑完却不报错（P4「静默fallback」）。
    """
    col = PRICE_COLS.get(field, field)
    if col == field:
        return d[field]
    if col not in d.columns:
        raise KeyError(
            f"长表缺少复权列 {col!r}，无法用复权口径计算。\n"
            "  解决：调用 load_long(..., adjusted=True)"
        )
    return d[col]


# ── 动量类 ────────────────────────────────────────────────────
def momentum(long: pd.DataFrame, window: int = 20, skip: int = 0) -> pd.Series:
    """N 日动量 = close[t] / close[t-N-skip] - 1

    参数
    ----
    window : 回看窗口（交易日）
    skip   : 跳过最近 skip 个交易日。T+1 制度下必须 >0，推荐 20。

    示例：window=60, skip=20 → close[t]/close[t-80]-1
    """
    d = _prep(long)
    px = _px(d, "close")
    base = px.groupby(d["code"], sort=False).shift(window + skip)
    v = px / base - 1.0
    return _mk(v, d["code"], d["date"], f"mom{window}_skip{skip}")


def reversal(long: pd.DataFrame, window: int = 5) -> pd.Series:
    """N 日反转 = -(close[t]/close[t-N] - 1)

    学术上 A 股短期反转效应显著（与美股的动量方向相反）。
    """
    d = _prep(long)
    px = _px(d, "close")
    base = px.groupby(d["code"], sort=False).shift(window)
    v = -(px / base - 1.0)
    return _mk(v, d["code"], d["date"], f"rev{window}")


# ── 风险类 ────────────────────────────────────────────────────
def volatility(long: pd.DataFrame, window: int = 60, annualize: bool = True) -> pd.Series:
    """N 日收益率标准差（滚动窗口），默认年化（×√244）。"""
    d = _prep(long)
    px = _px(d, "close")
    # ⚠️ fill_method=None：close_adj 有 1.6% 的行为空，默认的 ffill 会把
    #    停牌/缺失日填成前值，再跨空档算单日收益，凭空造出一次涨跌。
    d["ret"] = px.groupby(d["code"], sort=False).pct_change(fill_method=None)
    v = d.groupby("code", sort=False)["ret"].rolling(window).std(ddof=1) \
         .reset_index(level=0, drop=True)
    if annualize:
        v = v * np.sqrt(244)
    return _mk(v, d["code"], d["date"], f"vol{window}")


def downside_volatility(long: pd.DataFrame, window: int = 60,
                         min_periods: int | None = None) -> pd.Series:
    """下行波动率：只统计负收益的标准差。刻画"坏波动"，与总波动率互补。

    ⚠️ 必须显式给 min_periods：下行收益样本天然稀疏（仅约 50% 的日子为负），
       若沿用默认的 min_periods=window=60，滚动窗口内几乎永远凑不满 60 个
       非空值，结果会全为 NaN（实测 n=0）。默认取 window//3。
    """
    d = _prep(long)
    mp = window // 3 if min_periods is None else min_periods
    px = _px(d, "close")
    d["ret"] = px.groupby(d["code"], sort=False).pct_change(fill_method=None)
    d["neg"] = d["ret"].where(d["ret"] < 0, np.nan)
    v = d.groupby("code", sort=False)["neg"].rolling(window, min_periods=mp).std(ddof=1) \
         .reset_index(level=0, drop=True)
    return _mk(v * np.sqrt(244), d["code"], d["date"], f"downvol{window}")


# ── 量能类 ────────────────────────────────────────────────────
def volume_ratio(long: pd.DataFrame, short: int = 5, long_win: int = 60) -> pd.Series:
    """量比 = 短期均量 / 长期均量。>1 表示近期放量。

    学术含义：放量常伴随资金关注度上升，但也可能是出货；
    需与价格方向配合使用，不可单独作为信号。
    """
    d = _prep(long)
    g = d.groupby("code", sort=False)["vol"]
    v = g.rolling(short).mean() / g.rolling(long_win).mean().replace(0, np.nan)
    return _mk(v, d["code"], d["date"], f"volratio{short}_{long_win}")


def turnover_proxy(long: pd.DataFrame, window: int = 20) -> pd.Series:
    """20 日均成交额。用于流动性过滤（非信号）。"""
    d = _prep(long)
    v = d.groupby("code", sort=False)["amount"].rolling(window).mean() \
         .reset_index(level=0, drop=True)
    return _mk(v, d["code"], d["date"], f"amount{window}")


# ── 位置类 ────────────────────────────────────────────────────
def price_position(long: pd.DataFrame, window: int = 250) -> pd.Series:
    """价格分位 = (close - min) / (max - min)，滚动 window 日。取值 0~1。"""
    d = _prep(long)
    px = _px(d, "close")
    g = px.groupby(d["code"], sort=False)
    lo = g.rolling(window).min().reset_index(level=0, drop=True)
    hi = g.rolling(window).max().reset_index(level=0, drop=True)
    v = (px - lo) / (hi - lo).replace(0, np.nan)
    return _mk(v, d["code"], d["date"], f"pos{window}")


# ── 趋势结构类（用户最熟悉的形态，量化表达）──────────────────
def hh_hl_score(long: pd.DataFrame, window: int = 20) -> pd.Series:
    """HH/HL 趋势结构评分。

    用户此前多轮关注的「上涨趋势 = 高点与低点同步抬升」的量化表达：
      · HH：窗口内最高价 > 上一窗口最高价  → +1
      · HL：窗口内最低价 > 上一窗口最低价  → +1
    取值 {-1, 0, 1, 2}，越高趋势越强。

    ⚠️ high/low 必须用**复权**价：除权日 high 会机械下移，
       用未复权价会把「除权」误读成「高点降低」，趋势结构直接判错。
    """
    d = _prep(long)
    hi = _px(d, "high").groupby(d["code"], sort=False).rolling(window).max() \
        .reset_index(level=0, drop=True)
    lo = _px(d, "low").groupby(d["code"], sort=False).rolling(window).min() \
        .reset_index(level=0, drop=True)
    hi_prev = hi.groupby(d["code"]).shift(window)
    lo_prev = lo.groupby(d["code"]).shift(window)
    valid = hi.notna() & lo.notna() & hi_prev.notna() & lo_prev.notna()
    v = ((hi > hi_prev).astype(float) + (lo > lo_prev).astype(float)).where(valid, np.nan)
    return _mk(v, d["code"], d["date"], f"hhhl{window}")


# ── 因子注册表 ────────────────────────────────────────────────
FACTORY: dict = {
    "mom20": lambda d: momentum(d, 20, 0),
    "mom60_skip20": lambda d: momentum(d, 60, 20),
    "mom120_skip20": lambda d: momentum(d, 120, 20),
    "rev5": lambda d: reversal(d, 5),
    "rev20": lambda d: reversal(d, 20),
    "vol60": lambda d: volatility(d, 60),
    "downvol60": lambda d: downside_volatility(d, 60),
    "volratio5_60": lambda d: volume_ratio(d, 5, 60),
    "amount20": lambda d: turnover_proxy(d, 20),
    "pos250": lambda d: price_position(d, 250),
    "hhhl20": lambda d: hh_hl_score(d, 20),
}

# 离散取值因子必须用**等宽值域分箱**（bins），不能用等频分位（quantiles）。
#
# ⚠️ 2026-10-06 实测踩坑（三次才定位到根因）：
#   hhhl20 取值只有 {0, 1, 2}，分 5 组时：
#     · quantiles=5      → pd.qcut(x, 5) 在 3 种取值上产生大量 NaN，
#                          binning 丢弃 90%，MaxLossExceededError(100% exceeded)
#     · zero_aware=True  → 更糟：它按正/负各切一半，而 hhhl **没有负值**，
#                          负半区为空 → 负分位全NaN，仍整段失败
#     · bins=3           → ✅ 按值域等宽切，正常出结果
#   教训：取值种类 < 分组数时，等频分位在数学上就无解，必须换分箱方式。
DISCRETE_FACTORS: dict[str, int] = {"hhhl20": 3}   # 因子名 -> 箱数


def bins_of(name: str) -> int | None:
    """该因子应使用的等宽箱数；连续因子返回 None（走等频分位）。

    未知因子直接报错，不静默返回 None——否则会退回等频分位而整段失败。
    """
    if name not in FACTORY:
        raise KeyError(f"未知因子 {name!r}，可用: {sorted(FACTORY)}")
    return DISCRETE_FACTORS.get(name)


def compute_factor(name: str, long: pd.DataFrame) -> pd.Series:
    """按名称计算因子。未知名称直接报错，不静默返回 None。"""
    if name not in FACTORY:
        raise KeyError(f"未知因子 {name!r}，可用: {sorted(FACTORY)}")
    return FACTORY[name](long)
