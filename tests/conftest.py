"""pytest 全局标记。

为什么需要 `require_db`
-----------------------
CI（GitHub Actions）**没有行情数据库** —— `data/market.db` 有 2.1 GB，
不入仓��。少数测试必须读真实数据才能验证（跨年预热、选股约束、
Tushare 管线状态），它们在 CI 上会因`FileNotFoundError` 失败。

这不是测试写错了，是**环境缺数据**。正确做法是显式 skip 并说明原因，
而不是让 CI 常绿地失败、或把测试改成mock 后失去意义
（mock 掉数据库 = 把「跨年预热真的有效」这个断言也mock 掉了，
CODE_TRUST P23：测试必须能说谎）。

用法
----
    @pytest.mark.require_db
    def test_...():
        ...

⚠️ **门禁必须答得出「拦过什么真错」**（ENGINEERING.md 第五节）。
   本标记只区分「环境有/无数据」，不改变任何判据。
   本机跑 `uv run pytest` 时数据库存在 ⇒ 全部真实执行，不会被skip。
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "market.db"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "require_db: 需要本机行情数据库 data/market.db（CI 上会 skip）",
    )


def _missing(*rel: str) -> list[str]:
    return [r for r in rel if not (ROOT / r).exists()]


@pytest.fixture(scope="session")
def require_db() -> None:
    """缺数据库就 skip，并打印缺什么。

    ⚠️ 用``pytest.skip`` 而不是 ``pytest.xfail``：
       skip 明确表达「本环境无法验证」，xfail 会让人以为测试本身有问题。
    """
    if not DB_PATH.exists():
        pytest.skip(
            f"需要本机行情数据库 {DB_PATH}（{DB_PATH.stat().st_size / 1e9:.1f} GB "
            f"级别，CI 上不存在）。本机跑 uv run pytest 会真实执行。")


@pytest.fixture(scope="session")
def require_runtime_files(require_db: None) -> None:
    """除数据库外还需要 runtime/ 下的大文件产物。"""
    miss = _missing("runtime/tushare/_money_whitelist.json")
    if miss:
        pytest.skip(f"缺少运行时产物：{miss}（需先跑数据下载脚本）")