"""run_meta.json 样本清单落盘契约（可复现性守护）。

出处：评审报告 review-retro-2e5c653-2357b79.md 二·2.1#7 ——
2357b79 声称「样本清单写进 run_meta.json」，实际只写 seed 与数量。
all_codes() 随数据库更新漂移，同一 seed 重抽样会得到不同样本，
不落逐只清单则历史读数（如该笔自带的 42 只×177 日冒烟数值）事后无法复核。

还原 bug 版（不写清单字段）必须让本文件 FAILED。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# 既有测试的同款入口（tests/test_selection.py 引 research 脚本即如此）
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from run_factor_study import build_run_meta, write_run_meta  # noqa: E402


class TestRunMetaSampleLists:
    def test_run_meta必须落盘逐只样本清单(self, tmp_path) -> None:
        ver = {"version": "v2026-10-13", "stamped_at": "2026-10-13T08:00:00"}
        codes = ["sz000001", "sh600519", "bj920000", "sz301999", "sh601318"]
        alive = ["sh600519", "sz000001", "sh601318"]

        meta = build_run_meta(
            ver,
            factors="mom60_skip20,rev5,vol60",
            pool="随机 5 只 (seed=20240101)",
            interval="2016-01-01 ~ 2026-09-30",
            codes=codes,
            in_pool=alive,
        )
        p = write_run_meta(tmp_path, meta)
        back = json.loads(p.read_text(encoding="utf-8"))

        # 核心契约：逐只代码可核对（seed/数量对抗不了 all_codes() 漂移）
        assert back["样本清单"] == sorted(codes), "抽样清单必须逐只落盘"
        assert back["入池清单"] == sorted(alive), "最终入池清单必须逐只落盘"
        assert back["样本数"] == len(codes)
        assert back["入池数"] == len(alive)
        # 既有键不丢（历史读者依赖它们）
        for k in ("数据版本", "打戳时间", "因子", "股票池", "区间"):
            assert k in back, f"run_meta 丢了既有键 {k}"

    def test_清单保留全部代码字面值(self, tmp_path) -> None:
        """清单是代码的字面列表（带市场前缀），不是数量、不是集合摘要。"""
        meta = build_run_meta(
            {"version": "v", "stamped_at": "t"},
            factors="rev5", pool="全市场", interval="2016-01-01 ~ 2026-09-30",
            codes=["sh600519"], in_pool=["sh600519"],
        )
        p = write_run_meta(tmp_path, meta)
        back = json.loads(p.read_text(encoding="utf-8"))
        assert back["样本清单"] == ["sh600519"]
        assert isinstance(meta["样本清单"], list)
        assert all(c == "sh600519" for c in meta["样本清单"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))