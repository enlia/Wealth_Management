"""因子计算层：纯函数，输入长表，输出 alphalens 的 MultiIndex(asset, date)。"""

from .price_volume import FACTORY, compute_factor

__all__ = ["FACTORY", "compute_factor"]
