"""申万行业映射与板块集中度（复用库内既有映射，不新建一套分类）。

库内真源（**2026-10 实测侦察结论，写进报告**）
----------------------------------------------
| 来源 | 位置 | 行数 | 字段 |
|---|---|---|---|
| **申万成分（本模块采用）** | ``runtime/tushare/index_member_all.parquet`` | 5,591 | ``l1_code/l1_name/l2_code/l2_name/l3_code/l3_name/ts_code/name/in_date/out_date/is_new`` |
| 申万成分分片（互备） | ``runtime/tushare/industry_member_shard000~027.parquet`` | 5,591（合并后） | 同上 |
| 申万分类表 | ``runtime/tushare/index_classify.parquet`` | 511 | ``index_code/industry_name/level/industry_code/is_pub/parent_code/src`` |
| 通达信行业（**不用**） | ``market.db::stock_info.industry`` | 8,728 行中 8,133 非空 | L3 名称，但含 ``'0.0'`` 假值 2,283 条 + 595 条 NULL |

**为什么不用 ``stock_info.industry``**：实测 ``industry`` 列有 2,283 条取值恰为字符串
``'0.0'``（缺值被写成了浮点零），595 条 NULL。若把它当行业名，``'0.0'`` 会变成一个
**真实存在的「行业」**把大批股票归到一组 —— 这正是 P22 的形态（哨兵值撞上真实键）。
故本模块**只从申万 parquet 读**。

市场前缀转换
------------
申万表用 Tushare 口径 ``600519.SH``，库内是 ``sh600519``；
``to_local`` 与 ``research/scripts/merge_tushare_into_db.py:71`` 同款规则（SH/SZ/BJ）。

⚠️ 本表 ``out_date`` 实测**全为 NULL**（5,591/5,591），即它是一份**当前快照**，
不是历史时点成分 —— 用于回测会有幸存者偏差，用于当下决策无此问题。此局限写进报告。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...config import WORKSPACE

__all__ = ["IndustryMap", "IndustryUsage", "industry_caps", "to_local"]

_EXCHANGE_PREFIX = {"SH": "sh", "SZ": "sz", "BJ": "bj"}


def to_local(ts_code: str) -> str:
    """``600519.SH`` → ``sh600519``。非 A/B 股返回空串（与并库脚本同规则）。"""
    if not isinstance(ts_code, str) or "." not in ts_code:
        return ""
    num, ex = ts_code.rsplit(".", 1)
    pre = _EXCHANGE_PREFIX.get(ex.upper())
    return f"{pre}{num}" if pre else ""


@dataclass(frozen=True)
class IndustryUsage:
    """某行业的仓位占用（占账户总本金比例，无量纲）。"""

    industry_l1: str
    weight: float          # 已用仓位
    cap: float             # 该行业上限
    source: str            # 'existing'（已有持仓）/ 'computed'（本模块裁剪结果）

    @property
    def room(self) -> float:
        return self.cap - self.weight


class IndustryMap:
    """申万一级/二级/三级行业映射（只读，来自库内 parquet）。

    用法::

        im = IndustryMap.from_workspace()
        im.l1("sh600519")          # → '食品饮料'
        im.l3("sh600519")          # → '白酒Ⅲ'
    """

    def __init__(self, table: pd.DataFrame) -> None:
        need = {"code", "l1_name", "l2_name", "l3_name"}
        missing = need - set(table.columns)
        if missing:
            raise ValueError(f"申万映射表缺字段 {sorted(missing)}，实际字段 {list(table.columns)}")
        if table.empty:
            raise ValueError("申万映射表为空 —— 拒绝用空表静默放行行业集中度校验（P18）")
        dup = table["code"][table["code"].duplicated()].unique().tolist()
        if dup:
            raise ValueError(
                f"申万映射表存在重复 code（{len(dup)} 个，例：{dup[:5]}）——"
                f"重复会让 l1() 结果依赖行序，必须先裁定唯一口径"
            )
        self._t = table.set_index("code")

    # ── 构造 ─────────────────────────────────────────────────
    @classmethod
    def from_parquet(cls, path: Path | str) -> IndustryMap:
        """从申万成分 parquet 构造。文件不存在**抛错并给修复命令**（不静默换源）。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"申万行业映射文件不存在：{p}\n"
                f"  该文件由 research/scripts/fetch_tushare.py 的 industry 任务生成：\n"
                f"    .venv\\Scripts\\python.exe research/scripts/fetch_tushare.py --task industry\n"
                f"  本模块不提供替代分类（禁止静默换源：通达信 industry 含 '0.0' 假行业，见模块 docstring）"
            )
        raw = pd.read_parquet(p)
        return cls.from_frame(raw)

    @classmethod
    def from_workspace(cls, workspace: Path | str | None = None) -> IndustryMap:
        """从项目根自动定位 ``runtime/tushare/index_member_all.parquet``。"""
        root = Path(workspace) if workspace is not None else WORKSPACE
        p = root / "runtime" / "tushare" / "index_member_all.parquet"
        if not p.exists():
            shards = sorted((root / "runtime" / "tushare").glob("industry_member_shard*.parquet"))
            if shards:
                raise FileNotFoundError(
                    f"缺少 {p.name}，但发现 {len(shards)} 个分片 {shards[0].parent}/*"
                    f"industry_member_shard*.parquet。\n"
                    f"  二者内容等价（实测均 5,591 行），但**本模块不静默改用分片**——\n"
                    f"  请显式选择：IndustryMap.from_parquet(<分片路径>) 或补下 index_member_all。"
                )
        return cls.from_parquet(p)

    @classmethod
    def from_frame(cls, raw: pd.DataFrame) -> IndustryMap:
        """从原始申万成分表（含 ``ts_code``）构造，做 code 转换与去重校验。

        校验顺序（P23：要求「先校验再使用」，缺列时**不许**先 `.astype` 再崩在 pandas 内部）：
        ①空表 ②缺列 ③代码口径 ④去重。
        """
        if raw is None or len(raw) == 0:
            raise ValueError(
                "申万成分表为空 —— 拒绝用空表静默放行行业集中度校验（P18）"
            )
        for col in ("ts_code", "l1_name", "l2_name", "l3_name", "in_date"):
            if col not in raw.columns:
                raise ValueError(
                    f"申万成分表缺列 {col!r}，实际列 {list(raw.columns)}"
                )
        t = raw.copy()
        t["code"] = t["ts_code"].map(to_local)
        bad = t["code"] == ""
        if bad.any():
            raise ValueError(
                f"申万成分表有 {int(bad.sum())} 行的 ts_code 无法转成本地代码"
                f"（例：{t.loc[bad, 'ts_code'].head(3).tolist()}）—— 需先裁定代码口径"
            )
        t = t[t["code"].str.len() > 0]
        # 同 code 多行时取最新 in_date（与 merge_tushare_into_db 同规则）
        t["_in"] = pd.to_datetime(t["in_date"].astype(str), format="%Y%m%d", errors="coerce")
        t = t.sort_values(["code", "_in"]).drop_duplicates("code", keep="last")
        return cls(t[["code", "l1_name", "l2_name", "l3_name"]])

    # ── 查询 ─────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._t)

    def has(self, code: str) -> bool:
        return code in self._t.index

    def _get(self, code: str, level: str) -> str:
        if code not in self._t.index:
            raise KeyError(
                f"{code} 不在申万映射表内（表内 {len(self._t)} 只）。\n"
                f"  禁止静默归入「未知」行业后继续 —— 那会让行业集中度校验形同虚设"
                f"（P22：哨兵键必须与真实键隔离；正确做法是显式报错并要求补映射）。"
            )
        v = self._t.at[code, level]
        if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
            raise KeyError(f"{code} 的 {level} 为空 —— 缺映射时禁止静默兜底，请补数据")
        return str(v)

    def l1(self, code: str) -> str:
        """申万一级行业名（如 '食品饮料'）。缺映射抛 KeyError。"""
        return self._get(code, "l1_name")

    def l2(self, code: str) -> str:
        return self._get(code, "l2_name")

    def l3(self, code: str) -> str:
        return self._get(code, "l3_name")

    def l1_many(self, codes: list[str]) -> dict[str, str]:
        """批量取一级行业。**空列表抛错**（P18：空索引不得广播成「全部合格」）。"""
        if not codes:
            raise ValueError("codes 为空列表：拒绝用空集合静默通过行业校验（P18）")
        return {c: self.l1(c) for c in codes}


def industry_caps(
    usages: list[IndustryUsage],
    max_industry: float,
) -> dict[str, float]:
    """汇总行业占用并校验；返回 ``{行业: 上限}`` 供 sizing 使用。

    任何行业超出上限**抛错**（不静默砍已有持仓 —— 那会改动真实组合却不告知）。

    ⚠️ ``usages`` 为空列表时抛错：空集合不得被当成「全部合格」（P18）。
    """
    if not usages:
        raise ValueError(
            "usages 为空：拒绝用空集合静默通过行业集中度校验（P18 空索引广播）"
        )
    if not 0.0 < max_industry <= 1.0:
        raise ValueError(f"max_industry 必须在 (0,1] 内，得到 {max_industry}")
    totals: dict[str, float] = {}
    for u in usages:
        if u.weight < 0:
            raise ValueError(f"行业 {u.industry_l1!r} 的仓位为负：{u.weight}")
        totals[u.industry_l1] = totals.get(u.industry_l1, 0.0) + u.weight
    over = {k: v for k, v in totals.items() if v > max_industry + 1e-12}
    if over:
        detail = "；".join(f"{k}={v:.4f}" for k, v in sorted(over.items()))
        raise ValueError(
            f"行业集中度超限（上限 {max_industry:.4f}）：{detail}。\n"
            f"  处理：不要在汇总层静默砍仓，应由 sizing.apply_portfolio_caps 在分配时"
            f"按 EV 从高到低裁剪；此处抛错说明上游给了越限输入。"
        )
    return {k: max_industry for k in totals}
