"""下载状态：限频器 + 账本。

从 ``fetch_all_tushare.py`` 拆出（原文件触及 500 行强制拆分区）。

⚠️ 账本为什么必需
-----------------
「文件存在」不等于「下载完成」：中断时 ``run_task`` 会写
``{tag}_partial.parquet``，只看 ``{tag}.parquet`` 存在与否会把半成品当完成。
所以用 ``_download_manifest.json`` 记录每个任务是否**真正**完成。

⚠️ 账本损坏必须 raise
--------------------
静默当成「全部重跑」会浪费几小时；静默当成「都完成了」会留下半成品数据。
"""

from __future__ import annotations

import json
import time

from tushare_paths import OUT

# Tushare 限频：2000 积分档 180 次/分钟，留 10% 余量
RATE = 180

class Limiter:
    """限频：RATE 次/分钟，遇接口报错自动退避。"""

    def __init__(self, rate: int = RATE) -> None:
        self.interval = 60.0 / rate
        self.last = 0.0

    def wait(self) -> None:
        dt = time.perf_counter() - self.last
        if dt < self.interval:
            time.sleep(self.interval - dt)
        self.last = time.perf_counter()


MANIFEST = OUT / "_download_manifest.json"

# Tushare 单页上限。实测：
#   namechange 首页 10,000（上限 10,000）    index_basic 首页 8,000（实际 8,000 满页）
#   repurchase/pledge_detail 首页 2,000/1,500
# ⚠️ **必须翻页**，否则拿到的是首页截断，数据是残缺的。
#   实测 new_share offset=4000 只剩 340 行 —— 说明 4,340 就是全量。
PAGE_SIZE = 2000


def load_manifest() -> dict:
    """读下载账本：记录每个任务是否**真正完成**。

    ⚠️ 不能用「文件存在」判断完成 ——
    中断时 `run_task` 会写 `{tag}_partial.parquet`，
    若下次跑只判断 `{tag}.parquet` 存在，会把半成品当完成。
    """
    if not MANIFEST.exists():
        return {}
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # 账本损坏必须显式报错，不能静默当成「全部重跑」
        raise RuntimeError(
            f"下载账本损坏：{MANIFEST}\n"
            f"  解决：删除该文件后重跑（会重新下载全部任务）"
        ) from None


def mark_done(name: str, rows: int, minutes: float) -> None:
    m = load_manifest()
    m[name] = {"rows": rows, "minutes": round(minutes, 1),
               "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2),
                        encoding="utf-8")


# 各表的业务主键（白名单，**不是「加到唯一为止」**）。
# ⚠️ 为什么不用「逐个加候选键直到行唯一」：
#   那个策略天然会**主动丢弃区分字段**来追求唯一。
#   实测踩过：财务表 4 个 report_type × 2 个 update_flag = 8 个合法版本，
#   候选键里没有 report_type/update_flag，全部被压成 1 行。
#   正确做法是先确定该表的**业务主键**，键用尽仍不唯一时抛错。
BUSINESS_KEYS: dict[str, list[str]] = {
    # 财务三表：主体 + 报告期 + 公告日 + 报表类型 + 是否更新
    "income":            ["ts_code", "end_date", "ann_date",
                          "report_type", "update_flag"],
    "balancesheet":      ["ts_code", "end_date", "ann_date",
                          "report_type", "update_flag"],
    "cashflow":          ["ts_code", "end_date", "ann_date",
                          "report_type", "update_flag"],
    "fina_indicator":    ["ts_code", "end_date", "ann_date"],
    "forecast":          ["ts_code", "ann_date", "end_date"],
    "express":           ["ts_code", "ann_date", "end_date"],
    "fina_mainbz":       ["ts_code", "end_date", "ann_date", "type"],
    "top10_holders":     ["ts_code", "end_date", "holder_name", "ann_date"],
    "top10_floatholders": ["ts_code", "end_date", "holder_name", "ann_date"],
    "stk_holdernumber":  ["ts_code", "end_date", "ann_date"],
    "pledge_stat":       ["ts_code", "end_date"],
    "share_float":       ["ts_code", "float_date", "ann_date"],
    # 指数权重：必须带 index_code，否则同日的沪深300/中证500 会被合并
    "index_weight":      ["index_code", "con_code", "trade_date"],
    "report_rc":         ["ts_code", "ann_date", "end_date", "org_name"],
}

# 业务主键用尽后仍不唯一时的兜底：只加这些「补充区分列」
EXTRA_DISAMBIGUATORS = ["f_ann_date", "comp_type", "end_type", "holder_type"]


