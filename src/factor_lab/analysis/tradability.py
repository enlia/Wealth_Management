"""涨跌停 / 停牌约束：让回测能真正「买得到、卖得出」。

为什么这是 A 股回测的第一大隐性偏差
----------------------------------
所有因子研究都在「理想成交价」上选股，但 A 股实际是：
  · **涨停时买不到** —— 买单挂不进队列，等于信号失效
  · **跌停时卖不出** —— 想跑跑不掉，风险敞口被迫延续
  · **停牌期间完全无法操作**

对「选出的强势股」尤其致命：因子说「买」的时候，往往正��涨停。
实测采样区间主板日均涨停 53 只、跌停 4 只 ——
一个持 30 只的组合，任何时点都有若干只处于封板状态。

不接入约束会**系统性高估动量类因子**：
它假设能买到最强的那些票，而实盘只能买到「还能买到的」。
本模块的价值就是把这个偏差显式扣出来。

数据源优先级
------------
|来源 | 优点 | 缺点 |
|---|---|---|
| **Tushare `stk_limit`**（交易所真实涨跌停价） | 权威，含 0.01 元取整 | 需另存 |
| `market_rules.limit_price`（按规则算） | 无需外部数据 | 注册制新股无限幅窗口须特判 |

⚠️ **实测踩过：限幅不是常数**
   主板 10%、创业板/科创板 20%、北交所 30%，
   且 2023-02-17 主板全面注册制后**新股上市前 5 日不设限**。
   用固定 ±10% 判会把新股首周的合法大涨全判成超限。
   故优先用 `stk_limit` 的真实价格，无覆盖时才回退到规则计算。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def load_limit_panel(limit_path: Path, db_path: Path,
                     start: str, end: str) -> pd.DataFrame:
    """加载涨跌停价面板 (索引=日期, 列=代码)。

    列含义：`up_limit` = 当日涨停价，`down_limit` = 当日跌停价。
    未覆盖的格子为 NaN —— 表示**该股当日不在涨跌停价表中**
    （新股未上市 / 停牌 / 数据缺失），调用方需决定如何处理，不可静默当 0。
    """
    if not limit_path.exists():
        raise FileNotFoundError(
            f"涨跌停价表不存在: {limit_path}\n"
            f"请先跑 research/scripts/fetch_all_tushare.py 下载 stk_limit，"
            f"否则涨跌停约束无法生效（回测会系统性高估）。")
    df = pd.read_parquet(limit_path,
                         columns=["trade_date", "ts_code", "up_limit",
                                  "down_limit"])
    df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["code"] = df["ts_code"].str.slice(0, 2).str.lower() + \
        df["ts_code"].str.slice(2, 8)
    up = df.pivot_table(index="date", columns="code", values="up_limit",
                        aggfunc="last").sort_index()
    dn = df.pivot_table(index="date", columns="code", values="down_limit",
                        aggfunc="last").sort_index()
    return up, dn


def limit_masks(close: pd.DataFrame, up: pd.DataFrame, dn: pd.DataFrame,
                tol: float = 1e-6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """判定「封涨停」与「封跌停」。

    参数
    ----
    close : **未复权**收盘价面板 (日期 × 代码)
    up/dn : `stk_limit` 的原始涨跌停价面板（未复权口径）

    ⚠️ **close 必须是未复权价**，不能用 close_adj。
       实测：`sh600519` 2024-01-02 收盘 1685.01、后复权 1531.31，
       复权因子 0.9088 —— 两者差 9%，而涨停判定阈值是 0.01 元级。
       拿后复权价比原始涨跌停价，**几乎所有股票都判成「未封板」**：
       约束彻底失效却不报任何错，是最隐蔽的静默 fallback。
       要用后复权价，必须先把涨跌停价也乘上同一复权因子。

    ⚠️ **NaN 必须当「不可交易」而不是「未封板」**。
       NaN 有两种来源：该股未上市/已退市，或 stk_limit 未覆盖。
       一律填 False 会让「没数据的票」变成可随意买卖 —— 恰恰相反。
       故此处填 **True**（视为封板 = 不可交易），宁可少买不可乱买。
    """
    limit_up = (close >= up - tol)
    limit_dn = (close <= dn + tol)
    # NaN → 不可交易（True = 封板）。fillna(True) 而非 False，理由见docstring。
    return limit_up.fillna(True), limit_dn.fillna(True)


def apply_to_buy(holding_panel: np.ndarray, tradable: np.ndarray,
                 n_hold: int) -> np.ndarray:
    """买入端：跳过**当日封涨停**的股票，用池内下一顺位补位。

    参数
    ----
    holding_panel : (T, n_hold) 目标持仓的列索引
    tradable      : (T, N) bool，True 表示该股当日**可买**（未封涨停）
    """
    T, N = tradable.shape
    out = holding_panel.copy()
    for t in range(T):
        row = out[t]
        bad = ~tradable[t, row]
        if not bad.any():
            continue
        # 用候选池里排名最靠后的未持有股票补位（保守：保持原有顺序）
        filled = row[~bad]
        need = n_hold - len(filled)
        if need > 0:
            extra = np.array(
                [c for c in range(N) if tradable[t, c] and c not in set(filled)],
                dtype=row.dtype)
            filled = np.concatenate([filled, extra[:need]]) if len(extra) else filled
        out[t] = filled[:n_hold]
    return out


def stats(limit_up: pd.DataFrame, limit_dn: pd.DataFrame,
          label: str = "") -> str:
    """统计封板频率，用于确认约束真的生效（而不是静默为空）。"""
    nu = float(limit_up.to_numpy().mean())
    nd = float(limit_dn.to_numpy().mean())
    return (f"{label}  封涨停 {nu*100:.3f}%  封跌停 {nd*100:.3f}%  "
            f"(共 {limit_up.shape[0]:,} 日 × {limit_up.shape[1]:,} 只)")