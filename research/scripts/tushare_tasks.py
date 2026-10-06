"""Tushare 全量数据下载任务表（2000 积分实测可用）。

接口可用性实测（2026-10-06 重新探测 40 个接口）
------------------------------------------------
  有数据 31 / 空返回 2 / 权限不足 7

⚠️ **权限比 2026-10-05 的探测结果更宽**：
  · ``suspend_d``（停复牌）当时判为不可用，现在可用
  · ``top_list``（龙虎榜）当时判为不可用，现在可用
  → **禁止沿用历史探测结论**，每次补数据前重新实测。

按实测确认不可用（7 个）：
  cyq_perf筹码分布 / bak_daily 日线备份 / ccass_hold 港股通持股 /
  stk_auction 集合竞价 / limit_list_d 限流名单 / stock_st ST 股 / etf_basic

实测返回 0 行（接口存在但无数据，不要浪费请求）：
  hs_const（沪深港通成分，官方已停更）
  dividend 全量（**必须逐只拉**，见下）

踩坑记录
--------
1. **dividend 只能逐只拉**：传 ``{}`` / ``{'end_date': ...}`` 都返回 0 行，
   只有 ``{'ts_code': '600519.SH'}`` 返回 89 行。
   → 5,591 只× 0.36 秒 ≈ 34 分钟，kind 必须用 ``by_stock``。

2. **top10_holders 必须带 period**：``{'ts_code': '600519.SH'}`` 返回 10 行
   （只给最新一期），要历史必须 ``{'ts_code': ..., 'period': '20250630'}``。
   → 5,591 只 × 40 期 = 22 万次请求，不现实。
   → 折中：只拉近 8 期（8× 5,591 = 4.5 万次 ≈ 4.5 小时），或按 period 拉全市场
   （实测单期能返回全市场，40 次即可）。**本表用 by_period**。

3. **接口静默失败**：传错参数不报错，只返回 0 行。
   → 每个任务都必须打印实际行数，行数= 0 要当失败处理，不是「正常为空」。

优先级排序（对因子研究的边际价值）
------------------------------------
  P0 已有，本轮不重复
  P1 因子必需：涨跌停价、资金流、指数、限售解禁、分红、两融
  P2 事件/情绪：龙虎榜、停复牌、股东人数、股权质押、大宗交易、回购
  P3 静态信息：十大股东、管理层、公司资料、指数分类
  P4 宏观/外围：CPI、PMI、Shibor、沪深港通
"""
from __future__ import annotations

START = "20151201"
END = "20260930"

# 需要下载的指数（基准净值 / 成分权重）
INDEXES = [
    "000300.SH",   # 沪深300
    "000905.SH",   # 中证500
    "000852.SH",   # 中证1000
    "399006.SZ",   # 创业板指
    "000001.SH",   # 上证指数
    "399001.SZ",   # 深证成指
    "000016.SH",   # 上证50
    "932000.CSI",  # 中证2000
]

# ── 任务表 ─────────────────────────────────────────────────────
# kind:
#   once           一次拉完
#   by_date        按交易日逐日（日频接口，最快）
#   by_period      按报告期（财务类，可一次拿全市场）
#   by_period_month 按月（研报类）
#   by_stock       逐只股票（最慢，仅用于只能逐只拉的接口）
#
# est_min: 实测预估分钟数（2026-10-06，按 180 次/分钟限频）
TASKS: dict[str, dict] = {
    # ═══ P1：因子必需 ═══
    "stk_limit": {
        "kind": "by_date", "p": "P1", "desc": "涨跌停价格",
        "est_min": 11, "note": "涨跌停因子必需；可精确判断「是否封板」",
    },
    "moneyflow": {
        "kind": "by_date", "p": "P1", "desc": "资金流向(大单/超大单净额)",
        "est_min": 24, "note": "资金流因子；交易行为数据，非基本面",
    },
    "index_daily": {
        "kind": "by_date", "p": "P1", "desc": "指数日线",
        "est_min": 7, "note": "基准净值 —— 算超额收益必需",
    },
    "index_weight": {
        "kind": "by_period_month", "p": "P1", "desc": "指数成分权重",
        "est_min": 3, "note": "沪深300/中证500 成分与权重",
    },
    "share_float": {
        "kind": "by_period", "p": "P1", "desc": "限售解禁",
        "est_min": 2, "note": "解禁压力因子 —— 解禁前后抛压差异显著",
    },
    "dividend": {
        "kind": "by_stock", "p": "P1", "desc": "分红送配",
        "est_min": 34, "note": "⚠️ 只能逐只拉（传日期返回 0 行）",
    },
    "margin_detail": {
        "kind": "by_date", "p": "P1", "desc": "融资融券明细",
        "est_min": 24, "note": "杠杆资金情绪；融资余额占流通市值比是热门度代理",
    },
    "index_basic": {
        "kind": "by_year", "p": "P1", "desc": "指数列表",
        "est_min": 0.1, "note": "指数代码/名称/发布机构",
    },
    # ═══ P2：事件与情绪 ═══
    "top_list": {
        "kind": "by_date", "p": "P2", "desc": "龙虎榜",
        "est_min": 24, "note": "⚠️ 2026-10-06 实测新可用；游资席位 = 情绪代理",
    },
    "suspend_d": {
        "kind": "by_date", "p": "P2", "desc": "停复牌信息",
        "est_min": 24, "note": "⚠️ 2026-10-06 实测新可用；长期停牌股收益计算需剔除",
    },
    "block_trade": {
        "kind": "by_date", "p": "P2", "desc": "大宗交易",
        "est_min": 24, "note": "折价率 = 大股东减持意愿的代理变量",
    },
    "stk_holdernumber": {
        "kind": "by_period", "p": "P2", "desc": "股东人数",
        "est_min": 2, "note": "股东人数变化 = 筹码集中度",
    },
    "pledge_stat": {
        "kind": "by_period", "p": "P2", "desc": "股权质押统计",
        "est_min": 2, "note": "质押率 = 股东风险偏好，高质押股易暴跌",
    },
    "pledge_detail": {
        "kind": "once", "p": "P2", "desc": "股权质押明细",
        "est_min": 1, "note": "逐笔质押记录，⚠️ 只返回 1,500 行（单页上限）",
    },
    "repurchase": {
        "kind": "by_year", "p": "P2", "desc": "回购计划",
        "est_min": 1, "note": "回购公告 = 管理层认为低估的信号",
    },
    "new_share": {
        "kind": "by_year", "p": "P2", "desc": "新股发行",
        "est_min": 1, "note": "新股节奏 + 发行市盈率，可用于新股因子",
    },
    "moneyflow_hsgt": {
        "kind": "by_date", "p": "P2", "desc": "沪深港通资金流",
        "est_min": 24, "note": "北向资金 —— 最受关注的外资流向代理",
    },
    # ═══ P3：静态信息 ═══
    "stock_basic": {
        "kind": "once", "p": "P3", "desc": "股票列表（含退市）",
        "est_min": 1, "note": "已上市/退市/暂停上市全名单",
    },
    "namechange": {
        "kind": "by_year", "p": "P3", "desc": "更名记录",
        "est_min": 1, "note": "ST 摘帽/戴帽历史 —— 规避特殊处理股",
    },
    "stock_company": {
        "kind": "by_year", "p": "P3", "desc": "公司基本信息",
        "est_min": 1, "note": "注册地/员工数/主营业务描述",
    },
    "stk_managers": {
        "kind": "once", "p": "P3", "desc": "高管持股",
        "est_min": 1, "note": "⚠️ 单股 147 行，全量需逐只拉，本表只取概览",
    },
    "top10_holders": {
        "kind": "by_period", "p": "P3", "desc": "十大股东",
        "est_min": 3, "note": "股权集中度；单期可返回全市场",
    },
    "top10_floatholders": {
        "kind": "by_period", "p": "P3", "desc": "十大流通股东",
        "est_min": 3, "note": "流通盘集中度",
    },
    "index_classify": {
        "kind": "once", "p": "P3", "desc": "指数分类(申万)",
        "params": {"src": "SW2021"},
        "est_min": 0.1,
        "note": "申万一级/二级/三级行业分类；⚠️ src 必须显式指定，否则返回空",
    },
    "index_member_all": {
        "kind": "once", "p": "P3", "desc": "指数成分(全)",
        "est_min": 0.1, "note": "所有指数的成分股（2026-10-05 已下载，本轮会按账本跳过）",
    },
    "disclosure_date": {
        "kind": "by_year", "p": "P3", "desc": "财报披露计划",
        "date_field": "ann_date",
        "est_min": 2,
        "note": "⚠️ 数据量超 12 万行，翻页会截断（实测只到 2016-04），必须按年分段",
    },
    # ═══ P4：宏观与外围 ═══
    "cn_cpi": {
        "kind": "once", "p": "P4", "desc": "CPI 居民消费价格指数",
        "est_min": 0.1, "note": "月度宏观；利率敏感型因子的环境变量",
    },
    "cn_pmi": {
        "kind": "once", "p": "P4", "desc": "PMI 采购经理指数",
        "est_min": 0.1, "note": "月度宏观；周期股择时的环境变量",
    },
    "shibor": {
        "kind": "once", "p": "P4", "desc": "Shibor 利率",
        "est_min": 0.1, "note": "日频利率；估值分位的贴现率基准",
    },
    "ggt_top10": {
        "kind": "by_date", "p": "P4", "desc": "港股通十大成交股",
        "est_min": 24, "note": "⚠️ 必须逐日拉（传日期区间返回 0 行）",
    },
    "margin": {
        "kind": "once", "p": "P4", "desc": "融资融券市场汇总",
        "est_min": 0.1, "note": "两融余额时间序列",
    },
    "sz_daily_info": {
        "kind": "once", "p": "P4", "desc": "深市每日交易概况",
        "est_min": 0.5, "note": "深市独有的市盈率/换手统计",
    },
}

# 实测确认不可用（2000 积分），保留记录避免重复尝试
UNAVAILABLE = {
    "cyq_perf": "筹码分布，需 5000 积分",
    "bak_daily": "日线备份，权限不足",
    "ccass_hold": "港股通持股，权限不足",
    "stk_auction": "集合竞价，权限不足",
    "limit_list_d": "限流名单，权限不足",
    "stock_st": "ST 股名单，权限不足",
    "etf_basic": "ETF 基础信息，权限不足",
}

# 接口存在但无数据，不要浪费请求
EMPTY = {
    "hs_const": "沪深港通成分，官方已停更（2026-10-06 实测全参数组合均返回 0 行）",
}


def tasks_by_prio(prio: str | None = None) -> dict[str, dict]:
    """按优先级筛选任务。"""
    if prio is None:
        return dict(TASKS)
    return {n: s for n, s in TASKS.items() if s["p"] == prio}


def total_estimate(prio: str | None = None) -> float:
    """预估总耗时（分钟）。"""
    return sum(s["est_min"] for s in tasks_by_prio(prio).values())
