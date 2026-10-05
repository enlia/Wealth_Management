"""全局配置：路径、股票池、成本假设、常量。

所有可调参数集中在此，代码里不出现魔法数字。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ── 项目根目录自动发现 ───────────────────────────────────────
# ⚠️ 踩坑记录（2026-10-05 目录重组时修复）：
#   原代码硬编码 WORKSPACE = Path(r"C:\Documentation\Wealth_Management")
#   后果一：换电脑必崩 —— 项目要同步到第二台机器（用户明确要求）
#   后果二：路径一改，全项目 30+ 处引用全部失效
#   正确做法：从本文件位置逐级上溯，找含 .git 的目录作为项目根。
#   这样无论项目放在C:\ 还是 D:\ 或用户目录，都无需改代码。
def _find_workspace() -> Path:
    """向上查找项目根：环境变量 WM_ROOT → 含 .git 的祖先目录。

    ⚠️ 禁止静默 fallback（见 ENGINEERING.md 第四节）。
       找不到项目根时必须报错，不能悄悄返回某个猜测路径 ——
       否则 WORKSPACE 会指向错误目录，DB_PATH 报「文件不存在」，
       错误信息指向的是症状（数据库）而非根因（项目根没找到）。
    """
    env = os.environ.get("WM_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if not (p / "data").is_dir() and not (p / ".git").exists():
            raise ValueError(
                f"WM_ROOT 指向的目录不像项目根：{p}\n"
                f"应包含 data/ 或 .git/。请检查 WM_ROOT 设置。"
            )
        return p

    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent

    # 三级都找不到：明确报错，不静默猜测
    here = Path(__file__).resolve()
    raise RuntimeError(
        f"找不到项目根（向上搜索未发现 .git 目录）。\n"
        f"  起点：{here}\n"
        f"  原因：项目被移动到非 git 管理的目录，或 .git 被删除。\n"
        f"  解决：设置环境变量 WM_ROOT 指向项目根目录，例：\n"
        f"       set WM_ROOT=C:/path/to/Wealth_Management"
    )


WORKSPACE = _find_workspace()

# ── 路径 ──────────────────────────────────────────────────────
DB_PATH = WORKSPACE / "data" / "market.db"
FREE_DATA = WORKSPACE / "data" / "free_data"
OUTPUT_DIR = WORKSPACE / "runtime"
INDEX_CONSTITUENT_DIR = FREE_DATA / "指数成分"


# ── .env 加载（不引第三方依赖，手写解析）───────────────────────
# ⚠️ 为什么自己解析而不用 python-dotenv：
#   为了一个 TUSHARE_TOKEN 加一个依赖不划算，且 python-dotenv 会在 import 时
#   静默覆盖已存在的环境变量，行为不透明。规则简单到 20 行就能写对。
#
# 优先级：真实环境变量 > .env 文件
#   这样 CI/临时 `TUSHARE_TOKEN=xxx uv run ...` 仍能覆盖 .env 里的值。
ENV_PATH = WORKSPACE / ".env"


def load_env(path: Path | None = None) -> dict[str, str]:
    """读取 .env，返回键值字典。不修改 os.environ。

    支持的格式（够用即可，不追求完备）：
      KEY=value
      KEY = value            # 等号两侧空格会被去掉
      # 注释行以 # 开头
      export KEY=value        # 前缀 export 会被剥掉
      KEY="带引号的值"         # 首尾引号或单引号会被剥掉
      KEY=                    # 空值视为未设置，跳过
    """
    p = path or ENV_PATH
    if not p.exists():
        return {}

    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k and v:                       # 空值跳过，避免覆盖真实环境变量
            out[k] = v
    return out


_ENV_CACHE = load_env()


def env_get(key: str, default: str | None = None) -> str | None:
    """取配置值：真实环境变量优先，其次 .env。"""
    return os.environ.get(key) or _ENV_CACHE.get(key) or default


def has_token() -> bool:
    """是否已配置 Tushare token（用于脚本的友好提示，不打印 token 本身）。"""
    return bool(env_get("TUSHARE_TOKEN"))


# 本机通达信（数据源头，用于核对 SQLite 是否正确）
# ⚠️ 通达信装在哪台机器上不确定，逐个探测常见安装位置。
#    找不到时审计脚本会明确报「跳过通达信交叉验证」，而不是静默用错数据。
_TDX_CANDIDATES = (
    r"C:\SoftWare\TONGDAXIN\Body\vipdoc",
    r"C:\zd_zsgj\TONGDAXIN\vipdoc",
    r"C:\Program Files\TONGDAXIN\vipdoc",
    r"C:\new_tdx\vipdoc",
    r"D:\TONGDAXIN\vipdoc",
    r"C:\tdx\vipdoc",
)
# 显式指定优先：.env 里的 TDX_VIPDOC / 环境变量 > 自动探测
TDX_VIPDOC = Path(env_get("TDX_VIPDOC") or next(
    (p for p in _TDX_CANDIDATES if Path(p).exists()),
    _TDX_CANDIDATES[0],          # 都不存在则返回首选，报错信息里会显示它
))



# ── 交易成本假设（A股实际水平）────────────────────────────────
@dataclass(frozen=True)
class CostModel:
    """交易成本。默认值来自公开券商费率区间的中位数。"""

    commission_rate: float = 0.00025   # 佣金 万2.5，双边各收一次
    stamp_duty_sell: float = 0.0005     # 印花税 千分之0.5，仅卖出收
    slippage: float = 0.0005            # 滑点 估计值

    @property
    def buy_cost(self) -> float:
        return self.commission_rate + self.slippage

    @property
    def sell_cost(self) -> float:
        return self.commission_rate + self.stamp_duty_sell + self.slippage

    @property
    def round_trip(self) -> float:
        """一次完整买卖的成本。"""
        return self.buy_cost + self.sell_cost


DEFAULT_COST = CostModel()


# ── 研究参数 ─────────────────────────────────────────────────
@dataclass(frozen=True)
class ResearchConfig:
    """因子检验的核心配置。"""

    # 时间范围（数据止于 2026-09-30）
    start_date: str = "2016-01-01"
    end_date: str = "2026-09-30"

    # 股票池过滤
    min_price: float = 2.0           # 剔除仙股
    max_price: float = 3000.0
    min_list_days: int = 250         # 上市满 1 年，规避次新股异常
    min_amount_20d: float = 5e7      # 20 日均成交额 ≥ 5000 万，规避流动性陷阱
    exclude_st: bool = True
    exclude_new: bool = True         # 剔除上市 <250 天

    # 检验参数
    quantiles: int = 5               # 分 5 组
    periods: tuple = (1, 5, 10, 20, 60)   # 前瞻收益期（交易日）
    max_loss: float = 0.5            # alphalens 的缺失值填充上限

    # A股特有：T+1 制度导致日内动量与隔夜动量方向相反，
    # 文献指出必须跳过最近 1 个月，故动量因子用 20~60 日而非 5~20 日
    skip_recent_days: int = 20

    # 子区间稳定性检验的分段（AGENTS.md 质量红线要求）
    sub_periods: dict = field(default_factory=lambda: {
        "2016-2018": ("2016-01-01", "2018-12-31"),
        "2019-2022": ("2019-01-01", "2022-12-31"),
        "2023-2026": ("2023-01-01", "2026-09-30"),
    })


DEFAULT_RESEARCH = ResearchConfig()


# ── 标的分类规则（关键：必须按【文件所在目录】判断，不能按代码首位）──
# 依据：本机通达信目录结构实测，mootdx 源码亦有相同注释
#   sh 目录：6xxxxx A股 / 900xxx B股 / 5xxxxx 基金 / 1xxxxx 债券 /
#             000xxx 指数 / 88xxxx 板块指数
#   sz 目录：0xxxxx 主板 / 3xxxxx 创业板 / 200xxx B股 / 399xxx 指数
#   bj 目录：4xxxxx 旧北交所 / 8xxxxx 旧北交所 / 92xxxxx 新北交所(2023年起)
#
# ⚠️ 冒烟测试实测到的两个坑（2026-10-03）：
#   坑1 sh000001（上证指数）与 sz000001（平安银行）代码都是 000001。
#       若传入已带前缀的代码后又「丢掉前缀按首位猜市场」，
#       sh000001 会被误判为 sz → 读出平安银行的价格 11.57 而非指数 3842.19。
#       所以：带前缀的代码必须【直接采用前缀】，绝不重新推断。
#   坑2 北交所 2023 年起启用 92xxxx 新代码段（920000 等），只判 4/8 会漏掉。
_PREFIXES = ("sh", "sz", "bj")


def _split(code: str) -> tuple[str | None, str]:
    """拆出 (市场, 6位代码)。若已带前缀则直接返回，不重新推断。"""
    c = code.strip().lower()
    if len(c) == 8 and c[:2] in _PREFIXES:
        return c[:2], c[2:]
    return market_of(c), c


def market_of(code: str) -> str | None:
    """判断市场目录。带前缀则直接返回前缀，否则按通达信目录规则推断。"""
    c = code.strip().lower()
    if len(c) == 8 and c[:2] in _PREFIXES:
        return c[:2]
    if c.startswith(("6", "9", "5", "1", "88")):
        return "sh"
    if c.startswith(("0", "3", "2")):
        return "sz"
    if c.startswith(("4", "8")):
        return "bj"
    return None


def is_index(code: str) -> bool:
    m, c = _split(code)
    return (m == "sh" and (c.startswith("000") or c.startswith("950"))) or \
           (m == "sz" and c.startswith("399"))


def is_sector(code: str) -> bool:
    _, c = _split(code)
    return c.startswith("88")


def is_a_share(code: str) -> bool:
    """A 股普通股（沪深京）。排除指数、板块、B股、基金、债券。"""
    m, c = _split(code)
    if is_index(code) or is_sector(code):
        return False
    if m == "sh":
        return c.startswith("6")            # 600/601/603/605/688
    if m == "sz":
        return c.startswith(("0", "3"))     # 000/001/002/003/300/301
    if m == "bj":
        # 92xxxx 是 2023 年起的新北交所代码段，容易漏
        return c.startswith(("4", "8", "92"))
    return False


def is_b_share(code: str) -> bool:
    """B 股：沪 900xxx / 深 200xxx，价格系数 0.001。"""
    _, c = _split(code)
    return c.startswith(("900", "200"))
