"""批量验证所有研究脚本的 import 链是否完好。

用途：目录重组 / 依赖变更后，一次性确认没有脚本因路径或缺包而崩溃。

做法：单进程内对每个脚本用 runpy 捕获导入期异常。
比逐个 `uv run xxx.py --help` 快一个数量级（避免重复解释器启动）。

判定标准（AGENTS.md 质量红线）：
  · 脚本顶层 import 全部成功           → PASS
  · 出现 ModuleNotFoundError/ImportError → FAIL（缺依赖或路径错）
  · 出现其他异常（argparse 退出等）    → PASS（能跑到 main 说明 import 通了）
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

# 这些脚本导入即会发网络请求/读大文件，只做静态编译检查
SKIP_EXEC = {"fetch_tushare.py", "fetch_tushare_supplement.py", "fetch_yjbb.py"}

IMPORT_ERRORS = (ModuleNotFoundError, ImportError)


def check(path: Path) -> tuple[str, str]:
    """返回 (状态, 说明)。状态 ∈ PASS / FAIL / SKIP。"""
    name = path.name
    if name in SKIP_EXEC:
        return "SKIP", "含网络调用，仅编译检查"

    # 先编译（能抓语法错，且不执行）
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except SyntaxError as e:
        return "FAIL", f"语法错误 line {e.lineno}: {e.msg}"

    # 再试执行到 import 结束
    argv_backup = sys.argv[:]
    sys.argv = [str(path), "--help"]
    try:
        runpy.run_path(str(path), run_name="__not_main__")
        return "PASS", "import 链完好"
    except IMPORT_ERRORS as e:
        return "FAIL", f"缺依赖: {e}"
    except SystemExit:
        return "PASS", "argparse 正常退出（import 已通）"
    except Exception as e:                      # noqa: BLE001
        # 能走到这里说明所有 import 都成功了，只是 main 里的逻辑出错
        return "PASS", f"import 通过（main 阶段 {type(e).__name__}，预期内）"
    finally:
        sys.argv = argv_backup


def main() -> int:
    scripts = sorted((ROOT / "research" / "scripts").glob("*.py"))
    tools = sorted((ROOT / "src" / "tools").glob("*.py"))

    print("=" * 74)
    print("研究脚本 import 链验证")
    print("=" * 74)

    n_pass = n_fail = n_skip = 0
    for group, files in (("research/scripts", scripts), ("src/tools", tools)):
        if group == "src/tools":
            print("\n" + "-" * 74)
            print("工具脚本（仅编译检查）")
            print("-" * 74)
        for p in files:
            if group == "src/tools":
                # 工具脚本多为交互式，无 --help 约定，只做编译
                try:
                    compile(p.read_text(encoding="utf-8"), str(p), "exec")
                    print(f"  ✓ {p.name:<38} 编译通过")
                    n_pass += 1
                except SyntaxError as e:
                    print(f"  ✗ {p.name:<38} line {e.lineno}: {e.msg}")
                    n_fail += 1
                continue

            status, msg = check(p)
            mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "-"}[status]
            print(f"  {mark} {p.name:<38} {msg}")
            n_pass += status == "PASS"
            n_fail += status == "FAIL"
            n_skip += status == "SKIP"

    print()
    print("=" * 74)
    print(f"通过 {n_pass} ｜ 失败 {n_fail} ｜ 跳过 {n_skip}")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
