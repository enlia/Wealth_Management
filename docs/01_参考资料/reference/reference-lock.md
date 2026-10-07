# reference-lock.md — reference/ 第三方开源参考库锁定清单

> **用途**：锁定 `reference/`（目录职责第⑥层：第三方开源项目参考层，只读阅读用）15 家克隆的
> 上游 URL 与 HEAD 提交号，供 `reclone.ps1` 一键重建、供日后对账。
> **项目规则**：`reference/` **不进 git**（`.gitignore:46` 已忽略）；克隆仅下载源码目录供读源码，
> **不 pip install 进 .venv**（装包另行请示）。
> 生成日期：2026-10-07｜克隆窗口：2026-10-06 23:5x ~ 2026-10-07 00:5x
>
> 本机说明：全局 `.gitconfig` 把 `https://github.com/` 重写到 `ssh://git@ssh.github.com:443/`，
> 所以各克隆 `git remote get-url` 显示 ssh 形式；本表"上游 URL"记录**官方 HTTPS 地址**，两者等价。
> 克隆走 clash 代理（127.0.0.1:7897）完成，URL 未变。

## 锁定表（15 行）

| # | 项目名 | 用途一句话（据 AGENTS.md） | 上游 URL | HEAD 提交号（锁定） | 克隆日期 | 目录大小 | license | 查证依据一行 |
|---|---|---|---|---|---|---|---|---|
| 1 | qlib | 现成因子库 qlib.contrib.data：Alpha158/Alpha360（158 个 Alpha 因子） | https://github.com/microsoft/qlib.git | be725493eb1a6bbb42bf11b37aa7669f59610ff1 | 2026-10-06 | 27.0 MB | MIT License（LICENSE，Microsoft） | gh api 49,179★；本地实证 qlib/contrib/data 含 Alpha158/Alpha360 ；ls-remote ✓ |
| 2 | alphalens-reloaded | IC 分析 / 分组分析 / 换手率分析（因子检验事实标准） | https://github.com/stefan-jansen/alphalens-reloaded.git | f0a07c22d554e4b4036983cc80320b432714fe7e | 2026-10-07 | 128.4 MB | Apache License 2.0（LICENSE） | gh api 663★；README "performance analysis of predictive (alpha) stock factors / Information Coefficient Analysis" ✓ |
| 3 | vectorbt | 快速回测 / 参数扫描（比 backtrader 快两个数量级） | https://github.com/polakowo/vectorbt.git | ceffc501f2d37033a79dd86a9f883e69ec6977bd | 2026-10-07 | 185.9 MB | "Commons Clause" License Condition v1.0（LICENSE.md，非 OSI） | gh api 9,284★；README "backtesting engine" ✓ |
| 4 | rqalpha | 回测框架参考：事件驱动回测/模拟撮合/实盘对接 | https://github.com/ricequant/rqalpha.git | 0d98adefa87956e26f3e7ca5b26b5f3fc7ca834f | 2026-10-07 | 54.1 MB | 米筐科技专有许可（LICENSE 首行"版权所有 2019 深圳米筐科技有限公司"） | gh api 6,812★；README 中文自述"数据获取、算法交易、回测引擎、实盘模拟…全套解决方案" ✓ |
| 5 | bt | 回测框架参考：可复用策略组件的组合回测 | https://github.com/pmorissette/bt.git | 1ce8e84e95b6c055cdd2765c4f1b7a2fe0bb5efe | 2026-10-07 | 34.8 MB | The MIT License (MIT)（LICENSE） | gh api 2,996★；README "bt — Flexible Backtesting for Python" ✓（克隆时上游刚出新提交，见备注③） |
| 6 | OpenAlpha | 因子库参考：A 股开源 Alpha 因子池 | https://github.com/ziyouqitan/OpenAlpha.git | 214b0e3cdab11fa8fb1949724d0543baff333625 | 2026-10-07 | 22.6 MB | 无 LICENSE 文件（README 徽章标 MIT，仓库 license 元数据 NONE） | 撞名重点查证：gh search 全量比对 20 个同名候选，唯一因子库（描述 "An Open-Source Alpha Factor Pool for the Chinese A-Share Market"）；README 为 Alpha 表达式因子画廊（Alpha 5000001…）✓ |
| 7 | mlfinlab | 凯利仓位 / Purged K-Fold / Deflated Sharpe（López de Prado 标准实现） | https://github.com/hudson-and-thames/mlfinlab.git | 79dcc7120ec84110578f75b025a75850eb72fc73 | 2026-10-07 | 2.4 MB | ⚠️ LICENSE.txt 首行 "Copyright 2019, Hudson and Thames Quantitative Research Copyright Protection Notice and License"（专有、非 OSI；该组织后来转商业许可，见备注④） | gh api 4,935★；README "Machine Learning Financial Laboratory"；本地实证 mlfinlab/ 含 bet_sizing / cross_validation / backtest_statistics ✓ |
| 8 | machine-learning-for-trading | 因子 / ML 交易方法参考（从数据到实盘全流程） | https://github.com/stefan-jansen/machine-learning-for-trading.git | 6790e76326b6cab2117bb80abd6b07345bf0762d | 2026-10-07 | 285.2 MB | MIT License（LICENSE，Copyright (c) 2024-2026 Stefan Jansen） | gh api 21,244★；README "Code for Machine Learning for Trading, 3rd Edition — from data sourcing to live execution" ✓ |
| 9 | adata | A 股多源数据聚合（百度/同花顺/东财/新浪/腾讯冗余） | https://github.com/1nchaos/adata.git | b14f4e57b2175302f18b6eaf934f7dff9207a141 | 2026-10-07 | 4.1 MB | Apache License 2.0（LICENSE） | gh api 5,259★；README "专注股票行情数据…采用多数据源融合切换" ✓ |
| 10 | efinance | 股票/基金/债券/期货数据获取参考（数据源） | https://github.com/Micro-sheep/efinance.git | c8fd370a3109b2d14a121e3a32a86e9c8354b01b | 2026-10-07 | 1.2 MB | MIT License（LICENSE，Copyright (c) 2021 micro sheep） | gh api 4,079★；README "获取股票、基金、期货数据的免费开源 Python 库"（另含债券）✓ |
| 11 | mootdx | 通达信数据读取封装；通达信二进制解析对拍 oracle | https://github.com/mootdx/mootdx.git | e99ae34382d970c68654c6d17c45512e728f130d | 2026-10-07 | 66.9 MB | MIT License（LICENSE） | gh api 2,411★；README "通达信数据读取接口（离线/线上/财务数据）" ✓；src/tools/tdx.py 注释点名其 SECURITY_COEFFICIENT |
| 12 | eltdx | 通达信 A 股行情协议 Python 库（数据源 / 通达信解析） | https://github.com/electkismet/eltdx.git | 59d4615d3dafed4ecbdc480a34f96807cb132b92 | 2026-10-07 | 9.3 MB | ⚠️ ELTDX Research-Only License（LICENSE，专有：仅个人学习/协议研究/非商业研究，禁商用） | 撞名重点查证：gh search 唯一同名正主 559★；README "通达信A股行情协议 Python 库…快照、分时、逐笔、K线、集合竞价" ✓ |
| 13 | easyquotation | 新浪/腾讯（港股）/集思录实时行情（数据源） | https://github.com/shidenggui/easyquotation.git | 7778c1b9f93afb2ce2cf431de393fe690d4ec07c | 2026-10-07 | 0.5 MB | MIT License（LICENSE，Copyright (c) 2018 shidenggui） | gh api 5,453★；README "快速获取新浪/腾讯的全市场行情…获取集思路的数据" ✓ |
| 14 | tdxpy | 通达信二进制解析对拍 oracle（day/lc1/lc5/板块/财务） | https://github.com/One-sixth/tdxpy.git | 37acfccc14d758b92030444c7039ae300c146ca0 | 2026-10-07 | 0.2 MB | MIT License（LICENSE） | 撞名重点查证：PKG-INFO `Name: tdxpy / Repository: https://github.com/mootdx/tdxpy / Author: bopo`（mootdx 同一作者）；原仓 mootdx/tdxpy 现已 404（gh api + 浏览器 HTTP 404 双证），本仓为存档副本（README 自述"代码来源：https://github.com/mootdx/tdxpy"）；含 reader/（daily_bar/lc_min/block/history_financial）与 SECURITY_COEFFICIENT ✓ **与 rainx/pytdx 非同一项目（勿混）** |
| 15 | Ashare | 可转债 / 可交换债行情获取（新浪/腾讯双源） ⚠️**无许可证，仅读源码不拷代码** | https://github.com/mpquant/Ashare.git | 7ef1ce07579416d5c9e58f713867132bbaa9390d | 2026-10-07 | 0.4 MB | ⚠️ **无 LICENSE 文件（无许可证）——仅读源码，不拷代码** | 撞名重点查证：gh search 名称全量比对，同名唯一 A 股行情接口 3,890★，且**仓库根无 LICENSE**与 AGENTS.md"无许可证"告诫唯一吻合；用途实测（2026-10-07）：其新浪/腾讯 K 线接口对 sh113050/sz127064（转债）、sh132009/sh110080（可交换债段）、sz123120 全部返回真实 K 线（sz127064 收 134.149） |

**合计占用：823.0 MB**（15 家目录实测大小之和：每家 × 递归文件字节和）

## 备注

1. **定案 15 家 / 待定 0 家。** 4 个撞名项目（OpenAlpha、eltdx、tdxpy、Ashare）均已查到实证后定案，证据见"查证依据"列。
2. **--depth 1 说明**：按"单个体积超 300MB 可改 --depth 1"的授权，两家用浅克隆——
   `machine-learning-for-trading`（GitHub 仓库元数据 1.27 GB）、`vectorbt`（715 MB）。
   其余 13 家完整克隆（含全部历史）。浅克隆重建时 `reclone.ps1` 会先
   `git fetch --depth 1 origin <锁定SHA>` 再 checkout（GitHub 支持按 SHA 浅取，已实测验证）。
3. **bt 的 HEAD 漂移**：查证时（23:50）ls-remote 读数为 `9f4e8dd1…`，克隆时（00:52）上游已更新，
   本表锁定的是**克隆时点** HEAD `1ce8e84e…`。属正常上游活动，非对账差错。
4. **mlfinlab 许可特别说明（任务要求记录）**：`hudson-and-thames/mlfinlab` 在开源时期后转商业许可。
   本仓锁定的 master HEAD `79dcc712…`（2023-10 最后活动）根 LICENSE.txt 为
   "Copyright Protection Notice and License"（Hudson and Thames 专有声明，非 OSI 开源许可证）。
   项目复用的 `bet_sizing` / `cross_validation` / `backtest_statistics` 属旧开源版本（Apache-2.0 时期），
   完整克隆含全部历史，需要核对旧版许可时 `git log -- LICENSE.txt` / 相应 tag 处查阅。
5. **校验方法**（复核时照做）：对每家跑 `git -C reference\<名> rev-parse HEAD` 与本表核对；
   表中 SHA 同时与克隆当日 `git ls-remote <URL> HEAD` 读数一致（bt 漂移除外，见备注③）。