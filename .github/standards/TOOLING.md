# 工具链踩坑记录（TOOLING）

> 建立：2026-10-06（从 PITFALLS.md 拆出，保持每个文件 ≤ 500 行）
> 内容：研发**工具链**相关的坑 —— lint / git / CI / 依赖管理。
> 金融与数据相关的坑见 `PITFALLS.md`。

---

### P20 .gitignore 的目录规则未锚定，会吞掉同名包 🔴 **本地能跑≠别人能跑**

`.gitignore` 里写 `data/` 会匹配**任意层级**的 data 目录，不只是根目录。
本项目因此漏入库 8 个文件：

| 目录 | 内容 | 后果 |
|---|---|---|
| `src/factor_lab/data/` | `__init__.py` / `sqlite_source.py` / `universe.py` | 全新克隆下 10 个脚本连锁 `ModuleNotFoundError: No module named 'factor_lab.data'` |
| `docs/data/` | 5 份数据源获取指南 | GitHub 上看不到这批文档 |

**为什么本机一直正常**：工作区里残留着这些文件。
只有**干净克隆 / CI / 他人克隆**才暴露 ——
这是最难查的一类 bug，因为「我这能跑」掩盖了「仓库其实是残缺的」。

实测对比（干净克隆）：

```
修复前  通过 24 ｜ 失败 10   exit=1
修复后  通过 34 ｜ 失败  0   exit=0
```

**修法**：目录规则加前导斜杠锚定到根：`data/` → `/data/`

**→ 强制规则：改动 .gitignore 的目录规则后，必须做一次干净克隆验证：**

```bash
git clone --depth 1 <repo> /tmp/clean_test && cd /tmp/clean_test
uv sync --frozen && uv run python research/scripts/verify_imports.py
```

（`verify_imports.py` 就是为这个场景设计的 ——
 它会在干净克隆里暴露「本地有、仓库里没有」的文件。）

---

### P21 ruff 0.16 已移除 E999，选它会让 ruff 直接报错退出 ⚠️

首次配置 CI 时凭记忆写了 `--select F821,E999,F811,E722`，本地复现时：

```
ruff failed
  Cause: Rule `E999` was removed and cannot be selected.
```

**教训与 P8 同源**：第三方工具的行为必须用**实际安装的版本**验证，
不能凭记忆或凭文档写。（E999 是语法错误，ruff 现在由解析器直接报错，
不需要单独规则。）

替代方案（均已在 ruff 0.16 实测有效）：
`F402F501 F502 F522 F525 F601` —— 形参误用类真错。

---
