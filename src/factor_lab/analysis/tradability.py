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


# Tushare 交易所后缀 → 本机代码前缀。
# ⚠️ **退市股后缀是 `.S`**（`000003.S`），不按 SH/SZ/BJ 硬匹配，
#   直接小写拼接会得到 `s000003` —— 本机没有这个前缀，整表对不上。
#   统一归到`sz`（深市主板历史代码段）。
_EX_SUFFIX = {"SH": "sh", "SZ": "sz", "BJ": "bj", "S": "sz"}


def tushare_to_local_code(ts_code: pd.Series) -> pd.Series:
    """Tushare `ts_code` → 本机 `bar_daily.code` 口径。

    ⚠️ **交易所是「后缀」，不是前缀**（实测踩过）：
    ```
    Tushare   600519.SH  000001.SZ  300750.SZ  000003.S
    本机      sh600519   sz000001   sz300750   sz000003
    ```
    初版写成 `slice(0,2) + slice(2,8)`（当成 `SH600519`），
    实测产出列名 `['000001.S', '000002.S', ...]`
    —— 与本机代码**交集为 0**，
    涨跌停约束 100% 失效却不报错。

    ⚠️ 退市股后缀 `.S` 必须映射到 `sz`，
       原样小写拼接会得到 `s000003`，本机无此前缀。
    """
    parts = ts_code.astype("string").str.split(".", n=1, expand=True)
    if parts.shape[1] < 2:
        raise ValueError(
            f"ts_code 格式异常，缺少交易所后缀。样例: "
            f"{ts_code.head(3).tolist()}")
    raw_ex = parts[1].str.upper()
    ex = raw_ex.map(_EX_SUFFIX)
    unknown = sorted(set(raw_ex.dropna()) - set(_EX_SUFFIX))
    if unknown:
        raise ValueError(
            f"未知交易所后缀 {unknown}，请补进 _EX_SUFFIX。"
            f"静默放过会让该批股票列名对不上、涨跌停约束失效。")
    num = parts[0]
    return ex + num


def load_limit_panel(limit_path: Path, start: str, end: str,
                     close_codes: list[str] | None = None) -> tuple:
    """加载涨跌停价面板 (索引=日期, 列=代码)。

    列含义：`up_limit` = 当日涨停价，`down_limit` = 当日跌停价。
    未覆盖的格子为 NaN —— 表示**该股当日不在涨跌停价表中**
    （新股未上市 / 停牌 / 数据缺失），调用方需决定如何处理，不可静默当 0。

    ⚠️ `start` / `end` 接受 `str` 或 `int`，内部统一成 int。
       实测踩过：`trade_date` 经 `normalize_types` 迁成 int32 后，
       `df[df.trade_date >= '20230101']` 抛
       `TypeError: Invalid comparison between dtype=int32 and str`
       —— 新模块与自己的迁移互相打架，提交时就是坏的。

    参数
    ----
    close_codes : 本机实际使用的代码列表。传入后会**强制校验交集**，
        交集为 0 直接抛错（而不是产出一个全NaN 的面板）。
    """
    if not limit_path.exists():
        raise FileNotFoundError(
            f"涨跌停价表不存在: {limit_path}\n"
            f"请先跑 research/scripts/fetch_all_tushare.py 下载 stk_limit，"
            f"否则涨跌停约束无法生效（回测会系统性高估）。")
    df = pd.read_parquet(limit_path,
                         columns=["trade_date", "ts_code", "up_limit",
                                  "down_limit"])
    # ⚠️ 边界兼容：存量可能是 string（迁移前），也可能是 int32（迁移后）。
    #   统一转 int64再比较，避免「迁移后模块崩」这种自相矛盾。
    td = pd.to_numeric(df["trade_date"], errors="coerce")
    lo, hi = int(start), int(end)
    df = df[(td >= lo) & (td <= hi)]
    df = df.assign(code=tushare_to_local_code(df["ts_code"]))
    df["date"] = pd.to_datetime(df["trade_date"].astype("int64"),
                                format="%Y%m%d")
    up = df.pivot_table(index="date", columns="code", values="up_limit",
                        aggfunc="last").sort_index()
    dn = df.pivot_table(index="date", columns="code", values="down_limit",
                        aggfunc="last").sort_index()

    # ⚠️ **口径自检**：交集为 0 说明代码格式错了。
    #   静默产出一个全 NaN 的面板，会让 `limit_masks` 把**所有**股票
    #   判成不可交易（一只都买不了）而不报任何错。
    if close_codes:
        inter = len(set(up.columns) & set(close_codes))
        ratio = inter / max(len(set(close_codes)), 1)
        if ratio < 0.5:
            raise ValueError(
                f"涨跌停价与本机代码交集仅 {inter}/{len(set(close_codes))} "
                f"({ratio:.1%})，疑似代码口径不一致。\n"
                f"  涨跌停表面板列样例: {list(up.columns[:3])}\n"
                f"  本机代码样例      : {list(close_codes[:3])}")
    return up, dn


def limit_masks(close: pd.DataFrame, up: pd.DataFrame, dn: pd.DataFrame,
                tol: float = 1e-6,
                unlisted_state: str = "tradable") -> tuple:
    """判定「封涨停」与「封跌停」。

    参数
    ----
    close : **未复权**收盘价面板 (日期 × 代码)
    up/dn : `stk_limit` 的原始涨跌停价面板（未复权口径）
    unlisted_state : `close` 为 NaN 时怎么判。**这是全项目最容易被忽略的口径分叉**。

       ⚠️⚠️ **`close` 缺失有两种截然不同的成因，处理方式必须不同**：
       | 成因 | 占比（实测 2016-2026 全市场） | 正确处理 |
       |---|---|---|
       | **未上市 / 已退市** | 主体 | 不可交易，但**不该被当成"封板"** |
       | **停牌**（当日有票但无成交） | 少量 | 不可交易（真的动不了） |

       实测把两者混为一谈的后果（2026-10-06 全市场对照）：
       - 判「买不进 27.45% / 卖不掉 26.78%」，而**真封涨停只有 1.11%、真封跌停 0.43%**
       - 26.34% 的「不可交易」纯粹是「这只票当时还没上市或已经退市」
       - ⚠️ **退市股会被永久锁仓**：`sell_ok` 恒为 False ⇒ 卖出端强制保留
         ⇒ 组合里出现一只**永不消失的僵尸持仓**，其收益被
         `nan_to_num` 填成 0⇒ 组合收益被稀释，且**换手被压低**
       - 实测「有约束」版本年化反而**上升** +1.20%/+1.38%/+1.95%
         —— 这不是约束带来的收益，是**约束把股票池悄悄缩小了**。

       ⇒ 默认 `unlisted_state="tradable"`：**没数据 = 不视为封板**。
         理由：未上市/已退市在因子层面本来就有因子值 NaN，
         不会被选进候选池（`rank_topk` 把 NaN 排到末尾）；
         **在选股层额外判一遍「不可买」是重复劳动，且引入上面那个bug。**
         若调用方确实要把「缺数据」也当不可交易（如手工指定股票池），
         传 `unlisted_state="blocked"`，但**必须同时处理退市股的退出**。

    ⚠️ **close 必须是未复权价**，不能用 close_adj。
       实测：`sh600519` 2024-01-02 收盘 1685.01、后复权 1531.31，
       复权因子 0.9088 —— 两者差 9%，而涨停判定阈值是 0.01 元级。
       拿后复权价比原始涨跌停价，**几乎所有股票都判成「未封板」**：
       约束彻底失效却不报任何错，是最隐蔽的静默 fallback。
       要用后复权价，必须先把涨跌停价也乘上同一复权因子。

    ⚠️ **不能只靠 fillna**：`close >= NaN` 返回 False（不是 NaN），
       实测导致 fillna(True) 永不触发——
       涨跌停价缺失的格子被当成「未封板」= 可随意买卖。
       连fillna 的兜底都失效，比填 False 更隐蔽。

    ⚠️ **必须做口径自检**（review 实测）：
       传错close 时全表 NaN → 一律判不可交易 ⇒ **所有股票都不可买**，
       结果是「一只都买不了」，而代码不报任何错。
       实测两种错法的表现：
         传后复权价→ 封涨停率 0.79% 稀释到 0.25%（静默削弱 2/3）
         面板未对齐 → 封涨停率 0.00%（约束彻底失效）
       故覆盖率过低时直接抛错，把静默失效变成显式失败。

    ⚠️ **必须显式对齐索引与列**（review 实测连续踩到三次）：
       1. `bar_daily.date` 存 **int**（20230103），
          `load_limit_panel` 产出 **datetime64** → 索引不对齐
       2. `stk_limit` 含北交所（5,423 列），本机 A 股面板更窄
          → 列不对齐
       3. 两者任一不对齐，pandas 直接抛
          `ValueError: Can only compare identically-labeled`；
          而手动 reindex 兜底又大概率对不上 → 全部判成不可交易

       故此处**统一 reindex 到 close 的形状**，并校验覆盖率。
    """
    if unlisted_state not in ("tradable", "blocked"):
        raise ValueError(
            f"unlisted_state 只能是 'tradable' / 'blocked'，"
            f"收到 {unlisted_state!r}")
    up = up.reindex(index=close.index, columns=close.columns)
    dn = dn.reindex(index=close.index, columns=close.columns)
    cov_close = float(close.notna().to_numpy().mean())
    cov_up = float(up.notna().to_numpy().mean())
    if cov_close < 0.5 or cov_up < 0.5:
        raise ValueError(
            f"面板覆盖率异常：close {cov_close:.1%} / up_limit {cov_up:.1%}。\n"
            f"  close 索引 {type(close.index).__name__} vs "
            f"up 索引 {type(up.index).__name__}\n"
            f"  ⚠️ 若继续执行，缺数据会被判为「不可交易」"
            f"（可能一只都买不了）且不报错。")
    # ⚠️⚠️ **「缺涨跌停价」必须先限定在「当日确实有 close」范围内**
    #   （2026-10-06 实测 BLOCK，review 后修复）：
    #
    #   初版是无条件封板：
    #       no_px = up.isna() | dn.isna()
    #       limit_up = (close >= up - tol) | no_px
    #
    #   而 Tushare `stk_limit` **在股票未上市/ 已退市的日子里本来就没有行**，
    #   ��此「未上市」⇒ `up` 为 NaN ⇒ `no_px=True` ⇒ **照样被判封涨停**。
    #   ⇒ `unlisted_state="tradable"` 这条修复**根本没机会生效**：
    #     它只管「有涨跌停价但无 close」，而未上市这个主体场景
    #     在更早的 `no_px` 分支就被拦下了。
    #
    #   实测（2016-2026 全市场 A 股 5,921 只 × 2,611 日 = 1,546 万格）：
    #
    #   | 口径| 买不进 | 卖不掉 |
    #   |---|---|---|
    #   | 初版（no_px 无条件） | **27.817%** | **27.179%** |
    #   | 修正版（no_px & 有 close） | **1.164%** | **0.526%** |
    #   | 真实封板强度 | 1.048% | 0.410% |
    #
    #   `no_px` 的 4,138,520 格里有 **99.6%** 是「close 也缺失」
    #   （未上市/已退市），只有 0.116% 是真正的上市期间数据空洞。
    #   ⇒ 把前者一并封掉，等于凭空多阻塞 26.77pp，
    #     后果与 BLOCK-2 声称修复的完全相同：退市股僵尸锁仓 + 股票池缩小。
    #
    #   修正后买不进 1.164% 与真实封板 1.048% 只差 0.12pp —— 口径自洽。
    no_px_all = up.isna() | dn.isna()
    # 只在「本来就有成交」的地方谈涨跌停：
    # 有 close 但缺价= 真数据空洞 ⇒ 保守判不可交易（安全的一侧）
    # 无 close ⇒ 未上市/已退市 ⇒ 交给 unlisted_state 决定，不在此处封板
    no_px = no_px_all & close.notna()
    limit_up = (close >= up - tol) | no_px
    limit_dn = (close <= dn + tol) | no_px
    if unlisted_state == "blocked":
        # 调用方显式要求「缺 close 也算不可交易」。
        # ⚠️ 此时退市股会被永久锁仓（sell_ok 恒 False），
        #   调用方**必须**自行处理退出，否则组合里会出现僵尸持仓。
        limit_up = limit_up | close.isna()
        limit_dn = limit_dn | close.isna()
    return limit_up.fillna(True), limit_dn.fillna(True)


def coverage_report(close: pd.DataFrame, up: pd.DataFrame,
                    label: str = "") -> str:
    """分板块报告涨跌停价覆盖率，用于判断约束是否可信。

    ⚠️ **实测踩过：北交所数据不完整会让封板率虚高到 37%**
       真实 A 股封涨停率约 0.5%~0.8%（主板日均涨停 53 只/5,000 只）。
       实测 2023 年：全表封涨停 37.1%，
       但拆开看是「真封板 0.50%」+「缺数据被当不可交易 32%」。
       缺的那 32% 里2,960 列是北交所（`bj*`）——
       Tushare `stk_limit` 对北交所覆盖不全。
       若不拆开看，会得出「A股三分之一的日子在涨停」的错误结论。
    """
    up2 = up.reindex(index=close.index, columns=close.columns)
    per_col = up2.notna().mean()
    both = close.notna() & up2.notna()
    true_limit = ((close >= up2 - 1e-6) & both)
    lines = [f"{label}涨跌停约束诊断（{close.shape[0]:,} 日 × "
             f"{close.shape[1]:,} 只）"]
    lines.append(f"  整体覆盖率up {up2.notna().to_numpy().mean():.2%}"
                 f" / close {close.notna().to_numpy().mean():.2%}")
    # ⚠️⚠️ **零覆盖列必须单独报出来，不能混进分板块统计**（2026-10-06 实测）：
    #   `bar_daily` 不只有 A 股，还有指数（sh000001、sh000300…）。
    #   指数**永远没有涨跌停价**，而本函数按 `startswith('sh')` 分组——
    #   2,862 列指数被当成「沪主板」，把沪主板涨停价覆盖率
    #   从97.8% 拉低到 **52.2%**，看起来像「数据烂了一半」。
    #   实际是口径错误：**分母里混了不该有的东西**。
    #   ⚠️ 与 P16（分组排名方向）同类：代码能跑、有输出，
    #      但统计的对象不是你要研究的对象。
    #
    #   ⚠️ **但零覆盖列不能从报告里删掉**（review 抓出）：
    #   北交所 `stk_limit` 覆盖不全，零覆盖列**恰恰是最该报警的信号** ——
    #   删掉它这份报告就变成「一切正常」，正好丢掉它的存在意义。
    #   ⇒ 折中：**分板块统计排除零覆盖列**，同时**单独报出零覆盖列数**
    #     与占比，让「指数混入」与「北交所缺数据」两种原因可区分。
    zero_cols = [c for c in per_col.index if float(per_col[c]) <= 0]
    lines.append(f"  零涨跌停价列 {len(zero_cols):,}/{close.shape[1]:,} "
                 f"({len(zero_cols)/max(close.shape[1],1):.1%})"
                 f" —— 已从下方分板块统计中排除"
                 f"（成因：指数/基金混入，或该板块 stk_limit 未覆盖）")
    if zero_cols:
        lines.append(f"    样例: {list(zero_cols[:3])}")
    ok_cols = [c for c in per_col.index if float(per_col[c]) > 0]
    for pre, name in (("sh", "沪主板"), ("sz", "深市"), ("bj", "北交所")):
        cols = [c for c in ok_cols if str(c).startswith(pre)]
        if not cols:
            lines.append(f"  {name:<6} {0:>5} 只  涨停价覆盖率  0.00%  "
                         f"真封涨停率  0.000%  "
                         f"⚠ 该板块全部零覆盖，stk_limit 未覆盖")
            continue
        sub = per_col[cols]
        sub_true = true_limit[cols].to_numpy().mean()
        lines.append(f"  {name:<6} {len(cols):>5} 只  "
                     f"涨停价覆盖率 {sub.mean():>6.2%}  "
                     f"真封涨停率 {sub_true:>7.3%}")
    lines.append(f"  ⚠ 真封涨停率应在 0.5%~1% 量级。若整体偏高，"
                 f"说明缺数据被当成了「不可交易」。")
    return "\n".join(lines)


def apply_to_buy(holding_panel: np.ndarray, tradable: np.ndarray,
                 n_hold: int,
                 pool: np.ndarray | None = None) -> np.ndarray:
    """买入端：跳过**当日封涨停**的股票，用候选池内下一顺位补位。

    参数
    ----
    holding_panel : (T, n_hold) 目标持仓的列索引
    tradable      : (T, N) bool，True 表示该股当日**可买**（未封涨停）
    pool          : (T, K) 候选池列索引（因子值降序）。
                    ⚠️ **必须传**。初版没传，退化成从 `range(N)` 全市场取 ——
                    实测被顶掉的持仓换成「全市场任意可买股票」，
                    **因子信号在这一步被彻底丢弃**。
                    涨停约束的本意是「少买最强的那几只」，
                    绝不能变成「买完全无关的股票」。

    ⚠️ **可买股票不足 n_hold 时抛错，不静默凑数**：
       A 股涨停日 + 停牌 + 退市，可买不足是常态。
       实测初版直接 `out[t] = filled[:n_hold]` 会
       `ValueError: could not broadcast (29,) into (30,)`。
       用列索引 0 补齐更糟 —— 0 是**真实存在的股票**，填 0 等于「持有它」。
    """
    T = holding_panel.shape[0]
    out = holding_panel.copy()
    N = tradable.shape[1]
    for t in range(T):
        row = out[t]
        bad = ~tradable[t, row]
        if not bad.any():
            continue
        keep_set = set(int(c) for c in row[~bad])
        filled = row[~bad]
        cand = range(N) if pool is None else pool[t]
        extra = [int(c) for c in cand
                 if tradable[t, c] and int(c) not in keep_set]
        filled = np.concatenate(
            [filled, np.asarray(extra, dtype=row.dtype)]) if extra else filled
        if len(filled) < n_hold:
            raise ValueError(
                f"第 {t} 日仅 {len(filled)}/{n_hold} 只可买"
                f"（其余封涨停或停牌）。\n"
                f"  → 缩小 n_hold，或放宽封板口径。\n"
                f"  ⚠️ 不静默补齐：用列索引占位等于「持有那一只真实股票」，"
                f"会引入完全未预期的暴露。")
        out[t] = filled[:n_hold]
    return out


def stats(limit_up: pd.DataFrame, limit_dn: pd.DataFrame,
          label: str = "") -> str:
    """统计封板频率，用于确认约束真的生效（而不是静默为空）。"""
    nu = float(limit_up.to_numpy().mean())
    nd = float(limit_dn.to_numpy().mean())
    return (f"{label}  封涨停 {nu*100:.3f}%  封跌停 {nd*100:.3f}%  "
            f"(共 {limit_up.shape[0]:,} 日 × {limit_up.shape[1]:,} 只)")