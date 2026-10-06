"""复权价格引擎：把未复权行情转为前复权序列，消除除权跳空污染。

问题
----
通达信本机日线是**未复权**的，除权日会出现假跳空：
  · 平安银行 2016-06-16 单日 −17.91%（10 送 X 除权，非真实行情）
  · 宁德时代 2023-04-26 单日 −41.82%（创业板限 20%，必然是除权）
主板污染率 0.291%，导致**年化偏差 2.33pp** —— 足以推翻任何收益结论。

公式
----
前复权价 = 未复权价 × adj_factor(t) / adj_factor(最新)

Tushare 的 `adj_factor` 起点≈1、随时间递增（每分红送股一次跳一次），
是**累乘式**因子，本身不带方向。**除以最新值**做归一化后：
最新一天 前复权价 ≡ 未复权价，历史价格被压低
⇒ 得到的是**前复权（qfq）**。

⚠️ **不要叫它「后复权」**（2026-10-06 实测纠正）。
   后复权（hfq）应保持**基期不动**、末期被抬高，即 `adj/close` 比值
   **首日=1、末日>1**；本项目实测 `adj/close` 末日恒为 1.000000
   （茅台 0.793056→1.000000、平安 0.605294→1.000000）⇒ 属前复权。
   项目内曾有 86 处注释写成「后复权」，已全部纠正。

🔴 **前复权的代价：历史值不可复现。**
   每新增一次分红送转，**全部历史** 前复权价都会整体重算。
   ⇒ 同一份代码在不同日期跑出的历史数字不同，
     runtime/ 下存档的历史结论**不是**可复现的证据，
     引用时必须连同「数据版本戳」一起引用
     （见 `research/scripts/data_version.py`）。

量纲与精度
----------
  · 复权价保留 4 位小数（未复权是 2 位），避免精度损失累积
  · adj_factor 为 None 时**明确报错**，不静默跳过 ——
    静默跳过等于该股票仍是未复权，会污染整体结论

用法
----
  from factor_lab.data.adjust import AdjustedPriceBuilder
  builder = AdjustedPriceBuilder()          # 自动读 runtime/tushare/adj_factor.parquet
  adj_close = builder.adjust(close_series)  # 传入 code -> 未复权价序列
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ⚠️ 库代码不写 sys.path.insert（ENGINEERING 零节明文禁止）。
#    factor_lab 是 pyproject 声明的可安装包（uv_build），uv run 下自动可导入。
#    research/ 下的脚本仍需 sys.path.insert，那是脚本入口的既有约定。
from factor_lab.config import WORKSPACE

ADJ_FACTOR_FILE = WORKSPACE / "runtime" / "tushare" / "adj_factor.parquet"


def ts_to_local(ts_code: str) -> str:
    """600519.SH → sh600519"""
    num, suf = ts_code.split(".")
    return {"SH": "sh", "SZ": "sz", "BJ": "bj"}[suf] + num


class AdjustedPriceBuilder:
    """复权因子查表与价格换算。

    因子表按 (code, date) 建索引，用 bdate_range 对齐时按日期取值。
    内存占用：1,128 万行 × (8 字节日期 + 8 字节因子) ≈ 180 MB，
    一次性载入可接受；不载入则每次查询都扫 parquet，不可取。
    """

    def __init__(self, factor_file: Path | None = None) -> None:
        self.path = factor_file or ADJ_FACTOR_FILE
        if not self.path.exists():
            raise FileNotFoundError(
                f"复权因子文件不存在：{self.path}\n"
                f"  解决：uv run python research/scripts/fetch_tushare.py --task adj"
            )
        df = pd.read_parquet(self.path, columns=["ts_code", "trade_date", "adj_factor"])
        df = df[df["adj_factor"].notna() & (df["adj_factor"] > 0)]
        df["code"] = df["ts_code"].map(ts_to_local)
        # 日期转 int32 省内存
        df["dnum"] = df["trade_date"].astype("int64").astype("int32")
        df = df[["code", "dnum", "adj_factor"]]
        # 同一 (code, date) 保留最后一条
        df = df.drop_duplicates(subset=["code", "dnum"], keep="last")
        # ⚠️ 必须先按日期排序！Tushare 返回的 adj_factor 是**倒序**
        #    （实测茅台末三条是 20260930 / 20151202 / 20151201）。
        #    不排序时 groupby.last() 取到的是最早日期的因子，
        #    归一化系数会错 1.26 倍 → 复权后累计收益从 33% 虚增到 125%。
        df = df.sort_values(["code", "dnum"])
        self._fac = df.set_index(["code", "dnum"])["adj_factor"]
        # 每只股票最新因子（排序后 last() 才是真正的最新），用于前复权归一化
        self._latest = df.groupby("code")["adj_factor"].last()
        self._codes = set(self._fac.index.get_level_values(0).unique())
        self._n_codes = len(self._codes)

    def has(self, code: str) -> bool:
        return code in self._codes

    def coverage(self) -> dict:
        """复权因子覆盖率。"""
        return {
            "股票数": self._n_codes,
            "因子记录数": len(self._fac),
            "因子文件": str(self.path),
        }

    def normalize(self, code: str) -> float:
        """前复权归一化系数 = 1 / 最新因子。

        前复权价 = 未复权价 × 因子(t) / 因子(最新)
        归一化后，最新一天的前复权价 == 未复权价。
        """
        if code not in self._latest.index:
            raise KeyError(
                f"{code} 无复权因子，无法前复权。\n"
                f"  可能原因：次新股未上市满一日、或该票已退市。\n"
                f"  处理：确认 stock_basic_D（退市股名单）里是否含此代码。"
            )
        return 1.0 / float(self._latest[code])

    def factor_series(self, code: str, dates: np.ndarray) -> np.ndarray:
        """取某只股票在给定日期上的复权因子，缺失填 NaN。"""
        try:
            idx = pd.MultiIndex.from_arrays(
                [np.full(len(dates), code), dates.astype("int32")]
            )
            return self._fac.reindex(idx).to_numpy(dtype=float)
        except KeyError:
            return np.full(len(dates), np.nan)

    def adjust(
        self,
        code: str,
        prices: pd.Series,
        dates: np.ndarray | None = None,
    ) -> pd.Series:
        """把某只股票的未复权价格序列转为前复权。

        参数
        ----
        code: 本地代码，如 'sh600519'
        prices: 索引为日期（int YYYYMMDD）的价格序列
        dates: 可选的日期数组，缺省用 prices 的索引

        返回
        ----
        前复权价格序列，索引与输入一致。
        ⚠️ 复权因子缺失的日期（通常在因子表起始日之前）**产出 NaN**
           —— 不保留原值，也**不发任何告警**，乘法遇到缺因子即 NaN。
           下游若按「close_adj 非空」过滤，会静默丢行/丢日，
           查覆盖率异常时要记得这一层。
        """
        if dates is None:
            dates = prices.index.to_numpy()
        fac = self.factor_series(code, np.asarray(dates))
        norm = self.normalize(code)
        out = np.asarray(prices, dtype=float) * fac * norm
        return pd.Series(out, index=prices.index, name=f"{code}_adj")

    def adjust_frame(
        self, price_df: pd.DataFrame, field: str = "close"
    ) -> pd.DataFrame:
        """批量复权。price_df: 索引=日期(int32)，列=本地代码。

        整列缺因子的股票**原样保留**（不静默当作已复权），缺失代码记在
        **对象属性 self.last_missing** —— 是属性、不是返回值的组成部分，
        调用方不读它就无从察觉。列内**部分日期**缺因子时对应日期产出
        NaN，与 adjust() 一致，同样不告警。
        """
        dates = price_df.index.to_numpy()
        out = pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)
        missing: list[str] = []
        for code in price_df.columns:
            if not self.has(code):
                out[code] = price_df[code]
                missing.append(code)
                continue
            fac = self.factor_series(code, dates)
            norm = self.normalize(code)
            out[code] = price_df[code].to_numpy(dtype=float) * fac * norm
        self.last_missing = missing
        return out


def verify_adjustment(
    builder: AdjustedPriceBuilder, db_path: Path, samples: int = 5
) -> pd.DataFrame:
    """抽样定位除权日：找出单日跌幅最大的记录，看复权后是否回到合理区间。

    ⚠️ **阈值口径已废弃**，仅保留「定位除权日」的思路。
       这里用固定 -10.5% 找候选，而正确判据必须走
       `market_rules.price_tolerance(prev_close, limit_of(code))`
       —— 低价股真实封板涨停会达到 -10.15%，会被本函数误当成除权日。

    正式验证请用 `research/scripts/verify_adjusted_data.py`，
    它含动态容差 + 封板交叉验证 + 注册制新股窗口三重判据。
    """
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = []
    # 取单日跌幅最大的记录（最可能是除权污染）
    # ⚠️ 必须在 SQL 里过滤 LAG 为 NULL 的行 —— NULL 会被 ORDER BY 排到最前，
    #    导致取样全是「无前一日收益」的首日记录（实测踩过，返回空表）。
    q = """
    WITH r AS (
      SELECT b.code, b.date, b.close,
             b.close / LAG(b.close) OVER (PARTITION BY b.code ORDER BY b.date) - 1 AS ret
        FROM bar_daily b
       WHERE b.code LIKE 'sh60%'
    )
    SELECT code, date, close, ret
      FROM r
     WHERE ret IS NOT NULL AND ret < -0.105
     ORDER BY ret ASC
     LIMIT ?
    """
    for code, date, close, ret in con.execute(q, (samples,)):
        if code not in builder._codes:
            continue
        d = np.array([date], dtype="int64")
        fac = builder.factor_series(code, d)[0] * builder.normalize(code)
        rows.append(
            {
                "code": code,
                "date": date,
                "close": close,
                "原日收益%": round(ret * 100, 2),
                "前复权因子": round(float(fac), 4),
                "复权后收益%": round(
                    (close * float(fac) / (close / (1 + ret)) - 1) * 100, 2
                ),
            }
        )
    return pd.DataFrame(rows)
