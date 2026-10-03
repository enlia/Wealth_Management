"""检验层：复用 alphalens-reloaded，本层只做格式适配与结果整理。"""

from .alphalens_adapter import TearSheetResult, run_tear_sheet, subperiod_ic

__all__ = ["TearSheetResult", "run_tear_sheet", "subperiod_ic"]
