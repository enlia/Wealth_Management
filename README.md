# Wealth_Management · A股量化投研工作区

> 建立：2026-10-02｜最近更新：2026-10-05
> 用途：A 股因子研究、数据工程与实证检验
> **数据基准日：2026-09-30 收盘**

---

## 快速开始

```bash
# 1. 装依赖（项目根执行，uv 自动建venv）
uv sync --all-extras

# 2. 配凭证（复制模板后填入 token）
cp .env.example .env

# 3. 自检：验证数据链路与代码导入
uv run python research/scripts/smoke_test.py
uv run python research/scripts/verify_imports.py   # 批量验证 33 个脚本
```

Python 版本锁定 `>=3.12,<3.13`——依赖链硬约束，见 `pyproject.toml` 注释。

**关于 `.env`**：代码通过 `factor_lab.config.env_get()` 读取，优先级为
真实环境变量 > `.env` 文件 > MCP 配置自动发现。所以 `.env` 配好即生效，
不需要额外 export；`.env` 已被 `.gitignore` 排除，token 永不进入版本控制
（已验证：全历史 `.env` object 数为 0）。

---

## 目录导航

```
Wealth_Management/
├── data/                    ① 数据层（1.9G，不入版本控制）
│   ├── market.db              SQLite 行情库：日线/周线/板块/财务
│   └── free_data/财务三表/指数成分/
│
├── src/                     ② 代码层
│   ├── factor_lab/            可安装 Python 包
│   │   ├── config.py            全局配置（路径自动发现、成本模型、股票池）
│   │   ├── data/                数据接入（SQLite 读取、股票池筛选）
│   │   ├── factors/             价量因子库
│   │   └── analysis/            检验与评估（alphalens 适配 / 打分卡 / 选股质量）
│   └── tools/                16 个独立工具脚本
│       ├── build_sqlite.py       重建 SQLite 库
│       ├── tdx.py / tdx_reader.py 通达信数据读取
│       └── fib_tdx_study.py      斐波那契回撤实证
│
├── research/                ③ 研究层
│   ├── scripts/               17 个研究脚本
│   │   ├── audit_data_quality.py   ★ 数据质量审计（回测前置关卡）
│   │   ├── fetch_tushare.py        Tushare 数据补全（断点续传）
│   │   ├── expand_factor_library.py 因子库扩充 + 打分卡
│   │   └── run_financial_study.py    财务因子检验主脚本
│   └── methods/               研究方法论
│       └── 01_因子有效性检验方法论.md   ★ 六步检验模板
│
├── results/                 ④ 产出层（研究结论，入版本控制）
│   ├── 黄金分割实证研究报告.html / .pdf
│   ├── 因子多前瞻期IC_全市场.csv
│   └── 选股能力_全市场实测.csv
│
├── docs/                    ⑤ 文档层
│   ├── 01_参考资料/            学术调研、书籍、课程（9 篇）
│   ├── 02_方法与结论/          算法对比、方案清单、性能突破报告（3 篇）
│   ├── 03_项目报告/            数据审计、选股实测、Tushare 方案（5 篇）
│   ├── data/数据源获取指南（5 篇）
│   └── 术语词典.md
│
├── reference/               ⑥ 第三方开源项目（935M，只读，不入版本控制）
│   ├── 01_数据源工具/02_回测框架/03_因子库/ …
│
├── runtime/                 ⑦ 运行时产出（68M，可重新生成，不入版本控制）
│   ├── financial_panel.parquet    财务面板
│   ├── tushare/                   Tushare 下载缓存
│   └── scorecard/ turnover/ …     各检验产出
│
├── AGENTS.md                项目工作准则（含 12 节踩坑记录，动手前必读）
└── pyproject.toml / uv.lock / .python-version
```

**分层原则**：`data` 存数据、`src` 存代码、`research` 存研究脚本、`results` 存结论、`docs` 存知识、`runtime` 存可重建产出。找任何东西只需判断它属于哪一类。

---

## 常用命令

```bash
# ── 数据质量（任何回测前必跑）──
uv run python research/scripts/audit_data_quality.py

# ── 因子库扩充 + 打分卡 ──
uv run python research/scripts/expand_factor_library.py

# ── Tushare 数据补全（支持断点续传，重跑不重复下载）──
uv run python research/scripts/fetch_tushare.py --task all

# ── 查看某只标的行情 ──
uv run python src/tools/tdx.py show 603259 --period weekly
uv run python src/tools/tdx.py info                # 数据概况

# ── 重建数据库 ──
uv run python src/tools/build_sqlite.py
```

```sql
-- SQLite 查询示例
SELECT date, high, low, close FROM bar_weekly
WHERE code='sh603259' ORDER BY date DESC LIMIT 8;
```

---

## 数据资产

| 类别 | 内容 | 规模 |
|---|---|---|
| 日线 | 全市场 OHLCV（未复权） | 9,595 标的 / 13,238,352 条 / 2015-12 起 |
| 周线 | 本地聚合，与通达信对齐 | 2,792,272 条 |
| 板块 | 547 个板块 + 成分归属 | 69,878 条关系 |
| 财务 | 单报告期快照 | 7,992 条 |
| 行情估值 | PE / PB / 市值 / 换手率 | 8,409 条 |
| 分钟线 | ⚠️ 仅近期约 2 个月，长期需外部获取 | 沪 1.7 GB |

**已知数据缺陷**（详见 `docs/03_项目报告/07_数据质量审计报告.md`）：
- 未复权除权跳空导致**年化偏差 2.3pp**（主板污染率 0.291%）
- 申万行业分类缺失 37.2%，影响中性化
- 幸存者偏差：退市股缺失

→ **任何回测前先跑 `audit_data_quality.py`**，偏差 >0.5pp/年 必须先修数据。

---

## 当前研究结论

### 1. 斐波那契回撤：已被严格检验为不支持

| 检验 | 结果 |
|---|---|
| ESWA (2022) 三市场logistic 斜率 | 全部不显著 |
| vs 随机非斐波那契基准 | 跑输 |
| vs 买入持有指数 | 跑输 |

本工作区独立实证（1,273 样本）方向一致：81 个失效案例 **100% 都越过 0.618 才掉头**。
**决定成败的是能否创新高（HH），不是到没到 0.618。**

### 2. 你读到的因子夏普都偏高

McLean & Pontiff (2016, JF)：97 个预测变量，**发表后收益平均降 58%**，32% 直接归因于他人开始交易它。

### 3. A 股动量"消失"之谜

**T+1 制度**使日内与隔夜动量方向相反、相互抵消。文献冲突的根源是**未设冷却期**——必须跳过最近 1 个月。

### 4. 高 IC 等于高换手（本项目最重要实证）

`idio_vol_20` 的 ICIR = 0.639（全库最高），但年化净收益 **−9.21%**。

```
年成本 = 20bp × 换手率 × 分组数
```

按 ICIR 排序选因子，会选出完全不可用的组合。**必须看净收益，不是 IC。**

### 5. 通过全部检验的因子组合（当前）

仅 `bp + cf_quality + low_vol_120` 三因子通过「三段稳健 + ICIR≥0.15 + 换手<6%」。

- 命中率 54.0% vs 全市场 46.4%（超额 +7.6pp）
- 年化超额仅 **+0.12pp**
- 最大回撤 **−44.84%**（硬伤）

⚠️ 数据误差 2.3pp > 实测超额 0.12pp → **现阶段「不量化更合理」的结论方向上成立**。

---

## 换一台电脑

```bash
git clone git@github.com:enlia/Wealth_Management.git
cd Wealth_Management
uv sync --all-extras
cp .env.example .env      # 填入自己的 TUSHARE_TOKEN
```

**代码不含任何硬编码路径**——`config.py` 从自身位置逐级上溯、找含 `.git` 的目录作为项目根，搬到任何位置都能跑。仅需注意：

- `data/market.db` 不在版本控制中，需从旧机器拷贝（1.9G）
- `TUSHARE_TOKEN` 不随Git 走，必须重新配置
- 通达信路径自动探测 6 个常见位置，可用 `TDX_VIPDOC` 覆盖

---

## 声明

本工作区内容仅为**个人研究与方法论记录，不构成投资建议**。
数据来源：本机通达信 vipdoc、腾讯财经行情接口、Tushare Pro、公开学术文献。
第三方开源项目**仅做静态阅读审计，未安装、未执行**。
