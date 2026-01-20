from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner

# Alpha-MoE imports (lazy import to avoid import errors when not installed)
def get_alpha_moe_runner_core():
    from sglang.srt.layers.moe.moe_runner.alpha_moe import AlphaMoeRunnerCore
    return AlphaMoeRunnerCore

def get_alpha_moe_quant_info():
    from sglang.srt.layers.moe.moe_runner.alpha_moe import AlphaMoeQuantInfo
    return AlphaMoeQuantInfo

def is_alpha_moe_available():
    from sglang.srt.layers.moe.moe_runner.alpha_moe import ALPHA_MOE_AVAILABLE
    return ALPHA_MOE_AVAILABLE

__all__ = [
    "MoeRunnerConfig",
    "MoeRunner",
    "get_alpha_moe_runner_core",
    "get_alpha_moe_quant_info",
    "is_alpha_moe_available",
]
