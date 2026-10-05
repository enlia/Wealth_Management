"""A股因子有效性检验实验室。

设计原则（见 AGENTS.md）：
- 直接复用 alphalens-reloaded 做 IC/分组/收益分析，不重复实现
- 依赖单向：data ← factors ← analysis，底层不反向依赖上层
"""
__version__ = "0.1.0"

__all__ = ["__version__"]
