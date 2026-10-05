# 工程规范（Engineering Standards）

> 建立：2026-10-06｜来源：项目所有者明确要求
> 优先级：**最高**。与 AGENTS.md 其他条目冲突时，以本文件为准。
> 违反本规范的代码不予合并。

---

## 零、总纲

架构要**清晰、扁平**。判断标准不是"能不能跑"，而是：
半年后另一个人（或另一个 agent）能否在 5 分钟内找到入口、改对地方。

「清晰」的操作定义：每个目录/模块的职责能用一句话说清，
不存在"既放脚本又放数据"这类混合目录。

**「扁平」的操作定义**（可客观判定）：

| 规则 | 判定方式 |
|---|---|
| 目录深度 ≤ 3 层 | `src/factor_lab/analysis/` 已是极限，不得再建 `analysis/sub/` |
| 跨层调用走 import |禁止 `sys.path.insert` 拼路径（见 PITFALLS P4） |
| 一个功能只有一个入口 | 发现第二处入口即为违规 |
| 不做跨层反向依赖 | `research/scripts/` 可 import `src/`，反之不行 |

---

## 一、Git 工作流：分支 + 评审 + PR 合并

### 强制流程（无例外）

```
1. 从 main 拉新分支       git switch -c feat-<简述>
2. 在分支上开发 + 小步提交
3. 派子 agent 做 review明确要求它按本文档 + PITFALLS 逐条核对
4. review 通过后推送      git push -u origin <分支>
5. 在 GitHub 上开 PR      https://github.com/enlia/Wealth_Management/compare/main...<分支>
6. PR 合并到 main        （网页点 Merge，或 gh pr merge）
7. 保留分支              ⚠️ 合并后不删除，保留作历史记录
```

### 为什么必须走 GitHub PR

| 原因 | 说明 |
|---|---|
| **留痕** | PR 页面永久记录「讨论了什么、改了什么、谁review 的」 |
| **可追溯** | 出问题能回到任一 PR 看当时的 diff 与评审意见 |
| **强制 review 留档** | 合并记录里有 review 结论，不依赖对话历史 |
| **防止误合并** | PR 界面能一眼看出改动范围，比 `git merge` 盲合安全 |

**禁止** `git switch main && git merge <分支>` 直接在本地合并 ——
这绕过了 PR，评审记录无处存放。

### 规则

| 规则 | 说明 |
|---|---|
| **禁止直推 main** | 任何提交必须先在分支上，经 review + PR |
| **分支命名** | `feat-` `fix-` `refactor-` `docs-` + 简述，英文小写连字符 |
| **合并前必须有 review 结论** | 记录 review 发现了什么、改了什么 |
| **一个分支一件事** | 不把无关改动混进同一分支 |
| **合并后保留分支** | 不执行 `git branch -d` 与 `git push origin --delete` |

### 合并后保留分支的依据

GitHub **对分支数量无强制上限**（官方推荐 5,000 个以内，
超出会导致 fetch 变慢，不影响正常使用）。
StackOverflow 与 GitHub 官方文档的结论：

- Git 本身：分支就是一个含 40 字节 SHA 的引用文件，磁盘占用约 4KB
- GitHub：无硬限制，官方仅「建议 5,000 以内」以保证性能
- 性能软上限：建议 1,000 个 ref 以内

因此**保留已合并分支不会触及任何限制**，反而有可追溯价值。
若日后分支数逼近数百，可批量归档：

```bash
# 列出所有已合并分支（确认无误后再考虑清理）
git branch --merged main --format='%(refname:short)'
```

⚠️ 任何清理操作前先确认分支已合并、内容已进入 main。

**Review 子 agent 必须被明确要求做两件事**：
1. 按本文档逐条核对
2. **同时对照 `.github/standards/PITFALLS.md` 的 P1~P18 逐条核对**
   —— 那里有 18 条实证踩坑，是唯一记录"哪些做法真实失败过"的地方

---

## 二、文件规模：硬上限 500 行

### 规则

- **任何代码文件不得超过 500 行**（`.py` `.sh` `.toml` `.sql`）
- **400~500 行为强制拆分区**：本次 PR 触碰了就必须同时降下来
- 超限文件**不予合并**

### 豁免范围

| 类型 | 是否豁免 | 理由 |
|---|---|---|
| `.md` 文档（含 `.github/standards/`） | ✅ 豁免 | 文档的价值与长度无关，拆分反而割裂上下文 |
| 自动生成物（`uv.lock`、`.workbuddy/`） | ✅ 豁免 | 不手写 |
| 二进制（`*.pdf`、`*.parquet`） | ✅ 豁免 | 行数无意义 |
| 数据产出（`results/*.csv`） | ✅ 豁免 | 数据不是代码 |
| `reference/`（第三方只读参考） | ✅ 豁免 | 外部克隆的代码，改它等于改第三方；且不进版本控制 |

⚠️ **豁免不等于放任**。文档可以长，但必须有清晰目录结构，
让人能快速定位到需要的章节（当前 `docs/` 已按主题分三组）。

⚠️ **注释也算行**。不要以为"代码只有 300 行"就没事。

### 拆分原则：高内聚，低耦合

按**可测试的职责边界**拆，不按行数机械切：

```
❌ 把 600 行硬切成 3 个 200 行碎片，职责仍混在一起
✅ 每个模块对外暴露 ≤1 个主函数 + ≤3 个辅助函数，各自独立可测
```

**高内聚** = 一个模块只做一件事，模块内数据流向单向、不来回穿梭。

**低耦合** = 模块之间通过**明确的公开接口**交互，不互相摸对方内部。
判定方法：改动模块 A 时，是否需要同步改模块 B？
需要 →耦合过高，该拆接口。

```
❌ 低耦合
analysis/scorecard.py  ← 直接读 financial_panel.parquet 的某两列
                         （数据获取逻辑渗进了统计模块）

✅ 高内聚低耦合
analysis/scorecard.py  ← 只接受 DataFrame 参数，不自己读文件
data/panel_loader.py   ← 负责数据获取，可独立测试
```

### 合理的拆分层级

```
src/tools/<功能>.py          单个工具脚本，独立可执行
src/factor_lab/data/          数据访问层
src/factor_lab/factors/       因子计算（纯函数）
src/factor_lab/analysis/      检验统计
research/scripts/<任务>.py    研究脚本
```

---

## 三、清晰入口与配置

### 规则

**不允许"东拼西凑出一个功能"**。每个功能必须有：

1. **明确入口**：一个可执行文件，或一个能被明确 import 的函数
2. **明确配置**：命令行参数走 `argparse`，库内常量集中在
   `src/factor_lab/config.py`。二选一不算合规，必须按参数性质归位
3. **可独立运行**：`uv run python <入口>` 能跑通，不依赖手工改代码

### 禁止

```python
# ❌ 参数写死，改一次要改代码
LIMIT = 100
OUT_DIR = "C:/some/path"

# ❌ 上游改动导致下游报错，错误信息不指向根因
if data is None:
    data = load_default()        # 静默 fallback

# ❌ 同一个功能散落在3 个文件里
#    fetch_a.py 拉数据 → fetch_b.py 转换 → fetch_c.py 入库
#    没有统一入口，不知道从哪开始
```

### 正确

```python
# ✅ 配置集中，入口明确
ap.add_argument("--limit", type=int, default=100)
ap.add_argument("--out", default=None)

# ✅ 缺参数时明确报错，不是静默兜底
if data is None:
    raise FileNotFoundError(
        f"找不到数据文件：{path}\n"
        f"请先运行 uv run python research/scripts/<某脚本>.py 生成"
    )
```

---

## 四、禁止静默fallback

### 定义

**静默 fallback** = 出错时不报错、不提示，悄悄用别的值/空值/默认值继续跑。

这是本项目最危险的 bug 来源。已实证的三个例子：

| 静默 fallback | 后果 |
|---|---|
| `groupby(level=0).rank(pct=True)` 方向错 | 所有股票落进 Q5，**回测不报错、结果全错** |
| `bad.loc[空索引, col] = True` 广播 | 异常率算成 100%（真实 0.24%） |
| `净收益 = 毛收益 × periods[0]` | 年化被低估 **252 倍** |

**共同点：不报错，但结论完全颠倒。**

### 规则

| 场景 | 正确做法 |
|---|---|
| 数据源不存在 | `raise FileNotFoundError` 并给出修复命令 |
| 配置项缺失 | `raise ValueError` 或提示默认值的影响 |
| 外部接口失败 | 重试 → 仍失败则 `raise`，不返回空表 |
| 可选数据缺失 | **明确打印**「跳过 X，原因 Y」，不能默默跳过 |
| 找不到通达信 | 明确报「跳过通达信交叉验证」，不能静默用错数据 |

### 判定标准

**如果一次运行「看起来成功」但实际跳过了某些检查，就算违规。**

---

## 五、禁止非必要门禁

### 什么是"非必要门禁"

指那些**增加摩擦但不防真实错误**的检查：

- 与内容无关的形式检查（import 顺序、命名风格）
- 对已被类型系统/测试覆盖的重复校验
- 每次都要跑但从不失败的检查
- 需要人工确认才能通过的流程

### 规则

- 门禁必须拦下过**有记录的真实错误**（能说出提交号、报告行号或事故记录）
- 加门禁前先回答：**这个检查拦下过什么真错？** 答不出就别加
- 检查失败时**错误信息必须指向根因**，不能只说"不合法"
- 不允许为了"显得严谨"而增加检查 —— 判断标准：
  拿掉这个检查，过去一年会漏掉几个真错？答不出就删掉

### 本项目保留的检查点

⚠️ **必须诚实区分「自动阻断」与「手动运行」**。
把手动脚本说成门禁是自我欺骗 —— 它拦不住任何东西。

| 检查点 | 拦下过什么 | 自动阻断 |
|---|---|---|
| `pre-commit`（`.git/hooks/`） | 40 个提交被工具身份污染 | ✅ 是 |
| `audit_data_quality.py` | 2.33pp 年化偏差（+29.5% → +53.6%） | ❌ 手动 |
| `verify_tushare_data.py` | 接口静默截断 | ❌ 手动 |
| `verify_imports.py` | joblib 隐式依赖，4脚本连锁失败 | ❌ 手动 |

**「不予合并」目前无自动化执行机制**，全靠人工遵守。
若要变成真门禁，需补 `.pre-commit-config.yaml`（当前不存在）
把 `verify_imports.py` 挂进去 —— 它成本最低（单进程 runpy），
能防「换环境连锁崩溃」。

⚠️ `pre-commit` 无法随仓库分发：`.git/hooks/` 与 `.git/config`
都不被 git 跟踪。换机器克隆后需重新安装。

### 新增门禁的前置问题

加任何检查前先回答：**它拦下过什么真实错误？**
说不出具体提交号、报告行或事故记录的，不要加。

### 不允许用「门禁」代替「设计」

如果一个错误只能靠检查发现，说明设计本身有问题。
优先消除错误发生的可能，而不是加检查去拦截。

---

## 六、依赖必须显式声明

```python
# ❌ 靠 akshare → scikit-learn 间接带进来，换环境就崩
from joblib import Parallel, delayed

# ✅ 直接 import 的包必须显式声明
"joblib>=1.4.0",
```

**实证**：重建 venv 后 4 个研究脚本连锁 `ModuleNotFoundError`。

### 六.2 库 API 调用必须先验证签名

**不得凭记忆或凭文档书写第三方库调用。** 每次新增调用，先跑探针：

```python
import inspect
print(inspect.signature(func))      # 1. 确认真实签名

df = func(minimal_args)            # 2. 最小样本
print(type(df), df.index.names)    #    打印返回类型与索引结构
# 3. 确认后才写入业务代码
```

**实证**：alphalens-reloaded 0.4.6 有 8 处文档与实际行为不符
（返回元组却按DataFrame 用、列名是字符串却用 int 索引…），
详见 PITFALLS P2。**这 8 处全部由 review 流程发现，无一由文档发现。**

**返回值形状必须断言**，不能靠"应该是个 DataFrame"：
```python
qr, std_err = mean_return_by_quantile(...)   # 实际返回元组
assert isinstance(qr, pd.DataFrame), f"alphalens 返回类型变了: {type(qr)}"
```

---

## 七、路径必须自动发现

```python
# ❌ 换电脑/换路径即崩
WORKSPACE = Path(r"C:\Documentation\Wealth_Management")

# ✅ 二级发现 + 显式报错：WM_ROOT → 向上找含 .git 的祖先目录
#    两级都找不到时 raise 并给出修复命令，不静默猜测路径
```

**实证**：项目要同步到第二台机器，硬编码路径直接让该需求无法实现。

---

## 八、提交前自检清单

每次提交前逐条核对：

**流程**
- [ ] 分支不是 main（本次改动都在 feat- fix- refactor- docs- 分支上）
- [ ] 已派子agent review，且 review 意见已逐条处理
- [ ] git 身份 = `enlia <2020621056@qq.com>`（pre-commit 自动校验）

**代码质量**
- [ ] 无代码文件超 500 行（文档豁免）
- [ ] 新功能有唯一明确入口，参数可配置
- [ ] 无静默 fallback（出错必须抛出，不能悄悄兜底）
- [ ] 无非必要门禁
- [ ] 新增 import 已在 `pyproject.toml` 显式声明
- [ ] 无硬编码绝对路径
- [ ] 第三方 API 调用已用 `inspect.signature` 验证过签名

**数据与结论**（涉及因子/回测时必查）
- [ ] **PITFALLS.md 的 P1~P18 已逐条核对**
- [ ] 回测前已跑 `audit_data_quality.py`，偏差 < 0.5pp/年
- [ ] 绩效声明含四要素（年化口径 / 成本假设 / 基准 / 多空方向）
- [ ] 看了净收益，不只看 IC（高 IC = 高换手）

---

## 九、冲突时的优先级

```
工程规范（本文件，最高）
    ↓
金融研究质量红线（AGENTS.md 第二节）
    ↓
其他项目约定（AGENTS.md 其余部分）
```

拿不准时，**问用户**，不要自行发挥。
