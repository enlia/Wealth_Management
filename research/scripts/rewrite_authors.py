"""批量重写 git 提交作者信息（filter-branch 的可靠替代）。

为什么不用 git filter-branch
--------------------------
在 Windows + PowerShell 环境下，filter-branch 依赖的 shell 脚本
（git-sh-setup、cat 等）无法正常解析，报"command not found"。
本脚本直接操作 git fast-export/ fast-import 流，纯 Python 实现，
跨平台且行为可控。

安全性
------
  · 只改 author/committer 的name 与 email
  · 完整保留 commit message、提交时间、文件树、父子关系
  · 打完 tag 保留（filter-branch 默认行为）

用法
----
  # 1. 先备份
  git branch backup-before-author-fix
  git tag    backup-before-author-fix

  # 2. 干跑（只看会改什么，不实际写入）
  python rewrite_authors.py --old-name WorkBuddy --new-name enlia --dry-run

  # 3. 实际执行
  python rewrite_authors.py --old-name WorkBuddy --new-name enlia

  # 4. 验证无误后清理备份
  git branch -D backup-before-author-fix
  git tag-d backup-before-author-fix
  git reflog expire --expire=now --all && git gc --prune=now
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys


def run(args: list[str], binary: bool = False) -> bytes | str:
    r = subprocess.run(args, capture_output=True, shell=False)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"命令失败：{' '.join(args)}")
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-name", required=True)
    ap.add_argument("--old-email", default=None,
                    help="不给则只匹配 name")
    ap.add_argument("--new-name", required=True)
    ap.add_argument("--new-email", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # ── 统计当前作者分布 ──
    log = run(["git", "log", "--all", "--format=%an|%ae|%cn|%ce"])
    rows = [ln.split("|") for ln in log.splitlines() if ln]

    def matched(row: list[str]) -> bool:
        an, ae, cn, ce = row
        if args.old_email:
            return (an, ae) == (args.old_name, args.old_email) or \
                   (cn, ce) == (args.old_name, args.old_email)
        return an == args.old_name or cn == args.old_name

    hits = sum(1 for r in rows if matched(r))
    print(f"提交总数{len(rows)}，将被修改 {hits}")

    if not hits:
        print("✓ 无需修改")
        return 0

    if args.dry_run:
        print("\n前 5 个将被改动的提交：")
        n = 0
        for line in run(["git", "log", "--all",
                         "--format=%h|%an <%ae>|%s"]).splitlines():
            h, author, subject = line.split("|", 2)
            if args.old_name in author:
                print(f"  {h}  {author}")
                print(f"       → {args.new_name} <{args.new_email}>")
                print(f"       {subject[:60]}")
                n += 1
                if n >= 5:
                    break
        print("\n[dry-run] 未实际修改。去掉 --dry-run 执行。")
        return 0

    # ── fast-export → 改写 → fast-import ──
    # ⚠️ 不能加 --no-data：那会丢弃文件内容，fast-import 重建出的提交
    #    树会变空，等于把整个仓库内容删掉。必须完整导出。
    print("\n导出历史（完整含文件内容）…")
    dump = run(["git", "fast-export", "--all",
                "--signed-tags=strip", "--tag-of-filtered-object=drop",
                "--reencode=no"], binary=True)

    text = dump.decode("utf-8", "surrogateescape")
    n_author = n_commit = 0

    # 形式：author Name <email> 1234567890 +0800
    old_a = re.compile(
        rf"^author {re.escape(args.old_name)} <{re.escape(args.old_email or '')}>",
        re.M)
    old_c = re.compile(
        rf"^committer {re.escape(args.old_name)} "
        rf"<{re.escape(args.old_email or '')}>", re.M)

    if args.old_email:
        text, n_author = old_a.subn(
            f"author {args.new_name} <{args.new_email}>", text)
        text, n_commit = old_c.subn(
            f"committer {args.new_name} <{args.new_email}>", text)
    else:
        # 只匹配 name，邮箱通配
        pa = re.compile(rf"^author {re.escape(args.old_name)} <[^>]*>", re.M)
        pc = re.compile(
            rf"^committer {re.escape(args.old_name)} <[^>]*>", re.M)
        text, n_author = pa.subn(
            f"author {args.new_name} <{args.new_email}>", text)
        text, n_commit = pc.subn(
            f"committer {args.new_name} <{args.new_email}>", text)

    print(f"  改写author {n_author} 处，committer {n_commit} 处")

    if n_author == 0 and n_commit == 0:
        print("✗ 未匹配到任何标记，检查 --old-name / --old-email")
        return 1

    # 导入
    print("\n导入新历史…")
    p = subprocess.run(
        ["git", "fast-import", "--quiet", "--force"],
        input=text.encode("utf-8", "surrogateescape"),
        capture_output=True, shell=False)
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode("utf-8", "replace"))
        return 1

    # 清理悬空引用
    run(["git", "reflog", "expire", "--expire=now", "--all"])
    print("\n✓ 完成")
    print("\n下一步：")
    print("  git log --format='%an <%ae>' | sort -u        # 验证作者")
    print("  git push --force-with-lease origin master    # 强制推送")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
