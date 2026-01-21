"""
Alpha-MoE Runner Implementation for SGLang.

Alpha-MoE is a high-performance FusedMoE megakernel from Aleph Alpha.
https://github.com/Aleph-Alpha/Alpha-MoE

This module provides integration of Alpha-MoE into SGLang's MoE runner framework.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

# ============================================================================
# Global Configuration (similar to DeepGEMM's compile_utils.py pattern)
# ============================================================================

# Controls which rank runs autotuning. Similar to DeepGEMM's _IS_FIRST_RANK_ON_NODE.
# - True: This rank should run autotuning (default for single GPU)
# - False: This rank should wait and load from cache
_IS_FIRST_RANK_ON_NODE = True


def update_alpha_moe_config(gpu_id: int, server_args) -> None:
    """
    Update Alpha-MoE configuration based on server args.

    This should be called during server initialization, similar to
    DeepGEMM's update_deep_gemm_config().

    Args:
        gpu_id: The GPU ID for this process
        server_args: Server arguments
    """
    global _IS_FIRST_RANK_ON_NODE
    _IS_FIRST_RANK_ON_NODE = server_args.base_gpu_id == gpu_id


# ============================================================================
# Alpha-MoE Availability Check
# ============================================================================

ALPHA_MOE_AVAILABLE = False
_alpha_moe_import_error: Optional[Exception] = None

try:
    import alpha_moe

    ALPHA_MOE_AVAILABLE = True
except ImportError as e:
    _alpha_moe_import_error = e


def is_alpha_moe_available() -> bool:
    """Check if Alpha-MoE package is installed and available."""
    return ALPHA_MOE_AVAILABLE


def get_alpha_moe_import_error() -> Optional[Exception]:
    """Get the import error if Alpha-MoE is not available."""
    return _alpha_moe_import_error


# ============================================================================
# Weight Interleaving Utilities
# ============================================================================


def interleave_tensor(tensor: torch.Tensor, rep: int = 8) -> torch.Tensor:
    """
    Interleave weight tensor for Alpha-MoE kernel.

    Alpha-MoE requires the gate (w1) and up (w3) projections to be interleaved
    in a specific pattern. This function performs that interleaving.

    Args:
        tensor: Weight tensor of shape [num_experts, N, K]
        rep: Interleaving repetition factor (8 for weights, 1 for scales)

    Returns:
        Interleaved tensor of the same shape
    """
    M, N, K = tensor.shape

    first_half = tensor[:, : (N // 2), :]
    second_half = tensor[:, (N // 2) :, :]

    first_chunks = first_half.view(M, (N // (2 * rep)), rep, K)
    second_chunks = second_half.view(M, (N // (2 * rep)), rep, K)

    interleaved = torch.stack([first_chunks, second_chunks], dim=2)
    result = interleaved.view(M, N, K)

    return result.contiguous()


# ============================================================================
# Requirements Checking
# ============================================================================


def check_alpha_moe_requirements(
    block_size: Optional[List[int]] = None,
    is_block_quant: bool = False,
    layer: Optional[torch.nn.Module] = None,
    quant_config=None,
) -> Tuple[bool, str]:
    """
    Check if Alpha-MoE requirements are satisfied.

    This function can be called with either:
    1. quant_config: The quantization configuration object
    2. block_size + is_block_quant: Direct parameters

    Note: EP-related validation (ep_size, moe_a2a_backend) is primarily done
    in server_args.py during server startup, similar to cutlass and triton_kernel.
    This function provides runtime checks for library availability and quantization.

    Args:
        block_size: The weight block size [N, K], e.g., [128, 128]
        is_block_quant: Whether block quantization is used
        layer: The layer module (optional, for future use)
        quant_config: The quantization configuration (alternative to block_size/is_block_quant)

    Returns:
        Tuple of (is_satisfied, error_message)
    """
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    # Check 1: Alpha-MoE package installed
    # Note: Primary availability check is done in server_args.py at startup.
    # This is a secondary runtime check for robustness.
    if not ALPHA_MOE_AVAILABLE:
        return False, (
            f"Alpha-MoE is not installed. "
            f"Error: {_alpha_moe_import_error}"
        )

    # Check 2: Not in EP mode (runtime double-check, primary validation in server_args.py)
    a2a_backend = get_moe_a2a_backend()
    if a2a_backend.is_deepep() or a2a_backend.is_mooncake():
        return False, (
            "Alpha-MoE does not support Expert Parallel (EP) mode with DeepEP or Mooncake. "
            "Please use --moe-runner-backend triton or deep_gemm instead."
        )

    # Determine block size from either quant_config or direct parameters
    weight_block_size = block_size
    block_quant = is_block_quant

    if quant_config is not None:
        weight_block_size = getattr(quant_config, "weight_block_size", None)
        block_quant = weight_block_size is not None

    # Check 3: Block quantization is required
    if not block_quant or weight_block_size is None:
        return False, (
            "Alpha-MoE requires FP8 per-block quantization with weight_block_size=[128,128], "
            "but got per-tensor or per-channel quantization (weight_block_size is None). "
            "Please use FP8 block quantization with block_structure=[128,128]."
        )

    # Check 4: Block size must be [128, 128]
    if weight_block_size != [128, 128]:
        return False, (
            f"Alpha-MoE requires weight_block_size=[128, 128], "
            f"but got {weight_block_size}. "
            f"Please quantize with block_structure=[128, 128]."
        )

    # Check 5: CUDA availability and capability
    if not torch.cuda.is_available():
        return False, "Alpha-MoE requires CUDA, but CUDA is not available."

    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        return False, (
            f"Alpha-MoE requires SM80+ GPU (Ampere or newer), "
            f"but got SM{capability[0]}{capability[1]}."
        )

    return True, ""


# ============================================================================
# JIT Autotuning and Configuration
# ============================================================================

_ALPHA_MOE_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "sglang", "alpha_moe"
)
_ALPHA_MOE_CONFIG_CACHE: Dict[str, dict] = {}
_ALPHA_MOE_AUTOTUNING_DONE: Dict[str, bool] = {}  # Track if autotuning was already attempted


def _get_config_cache_path(E: int, N: int, K: int) -> str:
    """Get cache file path for given model configuration."""
    return os.path.join(_ALPHA_MOE_CACHE_DIR, f"moe_config_E{E}_N{N}_K{K}.json")


def _get_default_config(num_tokens: int) -> dict:
    """Get default conservative configuration when autotuning is not available."""
    # Conservative defaults that work reasonably well
    if num_tokens <= 64:
        block_m = 8
    elif num_tokens <= 256:
        block_m = 16
    else:
        block_m = 32

    return {
        "block_m": block_m,
        "block_n": 32,
        "warp_n": 8,
        "stages": 3,
    }


def run_autotuning(
    E: int, N: int, K: int, top_k: int, device: torch.device
) -> Optional[Dict[str, dict]]:
    """
    Run autotuning to find best configurations for different batch sizes.

    This function follows DeepGEMM's lazy JIT compilation pattern:
    - Only the first rank on the node (_IS_FIRST_RANK_ON_NODE) runs autotuning
    - Other ranks return None immediately

    Args:
        E: Number of experts
        N: Intermediate size (up/gate projection output dim)
        K: Hidden size
        top_k: Number of experts per token
        device: CUDA device to run tuning on

    Returns:
        Dictionary mapping batch_size -> config if successful, None otherwise
    """
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import moe_align_block_size
    from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8

    cache_path = _get_config_cache_path(E, N, K)

    # Only first rank on node runs autotuning (similar to DeepGEMM pattern)
    if not _IS_FIRST_RANK_ON_NODE:
        logger.debug(
            f"Alpha-MoE autotuning skipped (not first rank on node). "
            f"Will use default config. Config will be cached by first rank."
        )
        return None
    
    # First rank on node: Run autotuning
    os.makedirs(_ALPHA_MOE_CACHE_DIR, exist_ok=True)
    
    logger.info(
        f"Running Alpha-MoE autotuning for E={E}, N={N}, K={K}, top_k={top_k}. "
        "This may take a few minutes on first run..."
    )

    block_shape = [128, 128]
    batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    # Create dummy weights for tuning
    # Note: torch.randn doesn't support FP8 dtype directly, must create and convert
    w1 = torch.randn((E, N, K), device=device).to(torch.float8_e4m3fn)
    w2 = torch.randn((E, K, N // 2), device=device).to(torch.float8_e4m3fn)
    w1_scale = (
        torch.randn(
            (E, N // block_shape[0], K // block_shape[1]),
            dtype=torch.float32,
            device=device,
        )
        * 0.01
    )
    w2_scale = (
        torch.randn(
                (E, K // block_shape[0], (N // 2) // block_shape[1]),
                dtype=torch.float32,
                device=device,
            )
            * 0.01
        )

    # Interleave w1 weights and scales
    w1_interleaved = interleave_tensor(w1, rep=8)
    w1_scale_interleaved = interleave_tensor(w1_scale, rep=1)

    config: Dict[str, dict] = {}

    for num_tokens in batch_sizes:
        best_time = float("inf")
        best_config = None

        # Generate test data
        x = torch.randn((num_tokens, K), dtype=torch.bfloat16, device=device)
        x_fp8, x_scale = per_token_group_quant_fp8(x, block_shape[1])
        topk_weights = (
            torch.ones((num_tokens, top_k), dtype=torch.float32, device=device) / top_k
        )
        topk_ids = torch.randint(
            0, E, (num_tokens, top_k), device=device, dtype=torch.int32
        )

        # Grid search for best config
        for block_m in [8, 16, 32, 64, 128]:
            if num_tokens < block_m and block_m > 16:
                continue
            for block_n, warp_n in [(64, 4), (32, 8)]:
                for stages in [1, 2, 3, 4]:
                    # Skip configurations that exceed shared memory
                    if stages >= 5 and block_m > 100:
                        continue

                    try:
                        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                            moe_align_block_size(topk_ids, block_m, E)
                        )
                        out = torch.zeros_like(x)

                        # Warmup
                        for _ in range(3):
                            torch.ops.alpha_moe.fused_moe_w8a8_up_down(
                                x_fp8,
                                x_scale,
                                w1_interleaved,
                                w1_scale_interleaved,
                                w2,
                                w2_scale,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                topk_weights,
                                out,
                                top_k,
                                block_m,
                                block_n,
                                warp_n,
                                stages,
                                1.0,
                            )
                        torch.cuda.synchronize()

                        # Benchmark
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(10):
                            torch.ops.alpha_moe.fused_moe_w8a8_up_down(
                                x_fp8,
                                x_scale,
                                w1_interleaved,
                                w1_scale_interleaved,
                                w2,
                                w2_scale,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                topk_weights,
                                out,
                                top_k,
                                block_m,
                                block_n,
                                warp_n,
                                stages,
                                1.0,
                            )
                        end.record()
                        torch.cuda.synchronize()
                        elapsed = start.elapsed_time(end) / 10

                        if elapsed < best_time:
                            best_time = elapsed
                            best_config = {
                                "block_m": block_m,
                                "block_n": block_n,
                                "warp_n": warp_n,
                                "stages": stages,
                            }
                    except Exception as e:
                        # Skip invalid configurations
                        logger.debug(
                            f"Skipping config block_m={block_m}, block_n={block_n}, "
                            f"warp_n={warp_n}, stages={stages}: {e}"
                        )
                        continue

        if best_config is not None:
            config[str(num_tokens)] = best_config
            logger.info(
                f"  Batch {num_tokens}: block_m={best_config['block_m']}, "
                f"block_n={best_config['block_n']}, warp_n={best_config['warp_n']}, "
                f"stages={best_config['stages']}, time={best_time:.3f}ms"
            )
        else:
            # If we can't find any valid config for this batch size, skip it
            # (will use default at runtime)
            logger.warning(
                f"Alpha-MoE autotuning failed for batch {num_tokens}, will use default config at runtime."
            )

    # Cleanup
    del w1, w2, w1_scale, w2_scale, w1_interleaved, w1_scale_interleaved
    torch.cuda.empty_cache()

    # Only save to cache if we found at least some valid configs
    if config:
        try:
            with open(cache_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved Alpha-MoE config to cache: {cache_path}")
            _ALPHA_MOE_CONFIG_CACHE[cache_path] = config
        except Exception as e:
            logger.warning(f"Failed to save Alpha-MoE config to cache: {e}")
        return config
    else:
        logger.warning(f"Alpha-MoE autotuning failed, will use default config.")
        return None


def get_alpha_moe_config(
    E: int, N: int, K: int, top_k: int = 0, device: Optional[torch.device] = None
) -> Optional[Dict[str, dict]]:
    """
    Get Alpha-MoE configuration, triggering autotuning if needed.

    This function follows the "get or create" pattern:
    1. Check memory cache (fast path)
    2. Check disk cache / user config
    3. If not found, check if already tuned before:
       - If tuned before but no result -> return None (use default)
       - If not tuned -> run autotuning, then re-read from cache
    4. Return config or None (uses default)

    Args:
        E: Number of experts
        N: Intermediate size
        K: Hidden size
        top_k: Number of experts per token (required for autotuning)
        device: CUDA device (required for autotuning)

    Returns:
        Configuration dictionary if available, None otherwise
    """
    cache_path = _get_config_cache_path(E, N, K)

    # Step 1: Check memory cache (fast path for hot loop)
    if cache_path in _ALPHA_MOE_CONFIG_CACHE:
        return _ALPHA_MOE_CONFIG_CACHE[cache_path]

    # Step 2: Check disk cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                config = json.load(f)
                _ALPHA_MOE_CONFIG_CACHE[cache_path] = config
                if _IS_FIRST_RANK_ON_NODE:
                    logger.info(f"Loaded Alpha-MoE config from disk cache: {cache_path}")
                else:
                    logger.debug(f"Loaded Alpha-MoE config from disk cache: {cache_path}")
                return config
        except Exception:
            pass

    # Step 2b: Check user-provided config
    user_config_path = os.environ.get("ALPHA_MOE_CONFIG")
    if user_config_path and os.path.exists(user_config_path):
        try:
            with open(user_config_path, "r") as f:
                config = json.load(f)
                _ALPHA_MOE_CONFIG_CACHE[cache_path] = config
                if _IS_FIRST_RANK_ON_NODE:
                    logger.info(f"Loaded Alpha-MoE config from user config: {user_config_path}")
                else:
                    logger.debug(f"Loaded Alpha-MoE config from user config: {user_config_path}")
                return config
        except Exception:
            pass

    # Step 3: No cache found - check if we already attempted autotuning
    if cache_path in _ALPHA_MOE_AUTOTUNING_DONE:
        # Already tuned before but no result (failed or non-first rank)
        # Use default config
        return None

    # Step 4: Not tuned yet - run autotuning if we have required params
    # Mark as attempted first (to avoid repeated attempts from multiple layers)
    _ALPHA_MOE_AUTOTUNING_DONE[cache_path] = True

    if top_k > 0 and device is not None:
        # Run autotuning and return result (config or None)
        config = run_autotuning(E, N, K, top_k, device)
        return config

    # Autotuning failed or skipped, use default
    return None


def get_best_config_for_tokens(config: Optional[Dict[str, dict]], num_tokens: int) -> dict:
    """
    Get the best kernel configuration for a given number of tokens.

    Alpha-MoE uses different kernel configurations (block_m, block_n, warp_n, stages)
    for different batch sizes. This is because:
    - Small batches work better with smaller block_m (e.g., 16 or 32)
    - Large batches work better with larger block_m (e.g., 64 or 128)

    The autotuning process (jit_moe.py in Alpha-MoE) benchmarks each batch size
    and finds the optimal configuration. During inference, we find the closest
    matching batch size from the cached configurations.

    Args:
        config: Configuration dictionary mapping batch_size -> {block_m, block_n, warp_n, stages}
                If None, returns default config.
        num_tokens: Number of tokens in current batch

    Returns:
        Kernel configuration dict with keys: block_m, block_n, warp_n, stages
    """
    if not config:
        return _get_default_config(num_tokens)

    # Find closest matching batch size
    best_key = min(config.keys(), key=lambda k: abs(int(k) - num_tokens))
    return config[best_key]


# ============================================================================
# Runner Data Classes
# ============================================================================


@dataclass
class AlphaMoeRunnerInput(RunnerInput):
    """Input data for Alpha-MoE runner."""

    hidden_states: torch.Tensor  # FP8 quantized input [M, K]
    hidden_states_scale: torch.Tensor  # Input scale
    topk_weights: torch.Tensor  # [M, top_k]
    topk_ids: torch.Tensor  # [M, top_k]
    # Note: sorted_token_ids, expert_ids, num_tokens_post_padded are computed
    # in run() after autotuning determines block_m, not in pre_permute

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ALPHA_MOE


@dataclass
class AlphaMoeRunnerOutput(RunnerOutput):
    """Output data from Alpha-MoE runner."""

    hidden_states: torch.Tensor  # [M, K] in bf16

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ALPHA_MOE


@dataclass
class AlphaMoeQuantInfo(MoeQuantInfo):
    """Quantization info for Alpha-MoE.

    Only requires weights and scales. 
    block_shape is always [128, 128] for Alpha-MoE.
    """

    w13_weight: torch.Tensor  # Interleaved up/gate weights [E, N, K]
    w2_weight: torch.Tensor  # Down weights [E, K, N//2]
    w13_scale: torch.Tensor  # Interleaved scales
    w2_scale: torch.Tensor

    @property
    def num_experts(self) -> int:
        """Number of experts, inferred from w13_weight shape."""
        return self.w13_weight.shape[0]

    @property
    def intermediate_size(self) -> int:
        """Intermediate size (N), inferred from w13_weight shape."""
        return self.w13_weight.shape[1]

    @property
    def hidden_size(self) -> int:
        """Hidden size (K), inferred from w13_weight shape."""
        return self.w13_weight.shape[2]

    @property
    def block_shape(self) -> List[int]:
        """Block shape is always [128, 128] for Alpha-MoE."""
        return [128, 128]


# ============================================================================
# Alpha-MoE Runner Core
# ============================================================================


class AlphaMoeRunnerCore(MoeRunnerCore):
    """Alpha-MoE runner core implementation."""

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        if not ALPHA_MOE_AVAILABLE:
            raise ImportError(
                f"Alpha-MoE is not available: {_alpha_moe_import_error}. "
                f"Please install it with: "
                f"git clone https://github.com/Aleph-Alpha/Alpha-MoE.git && "
                f"cd Alpha-MoE && pip install -e . --no-build-isolation"
            )
        self._tuning_config: Optional[Dict[str, dict]] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ALPHA_MOE

    def run(
        self,
        runner_input: AlphaMoeRunnerInput,
        quant_info: AlphaMoeQuantInfo,
        running_state: dict,
    ) -> AlphaMoeRunnerOutput:
        """
        Execute Alpha-MoE kernel.

        Args:
            runner_input: Input data including hidden states and routing info
            quant_info: Quantization info including weights and scales
            running_state: Additional running state (unused)

        Returns:
            AlphaMoeRunnerOutput with computed hidden states
        """
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import moe_align_block_size

        # Lazy load tuning config on first execution (similar to DeepGEMM pattern)
        # get_alpha_moe_config handles: read cache -> check if tuned -> tune if needed
        if self._tuning_config is None:
            self._tuning_config = get_alpha_moe_config(
                E=quant_info.num_experts,
                N=quant_info.intermediate_size,
                K=quant_info.hidden_size,
                top_k=self.config.top_k,
                device=runner_input.hidden_states.device,
            )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        topk_weights = runner_input.topk_weights
        topk_ids = runner_input.topk_ids

        w13 = quant_info.w13_weight
        w2 = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        top_k = self.config.top_k
        routed_scaling_factor = self.config.routed_scaling_factor or 1.0
        num_tokens = hidden_states.shape[0]

        # Get tuning configuration for this batch size
        tuning_config = get_best_config_for_tokens(self._tuning_config, num_tokens)
        block_m = tuning_config["block_m"]
        block_n = tuning_config["block_n"]
        warp_n = tuning_config["warp_n"]
        stages = tuning_config["stages"]

        # Align block size - must be done here after autotuning determines block_m
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_m, quant_info.num_experts
        )

        # Output tensor (zero-initialized as required by Alpha-MoE)
        # The kernel writes results in-place
        output = torch.zeros(
            (num_tokens, quant_info.hidden_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

        # Call Alpha-MoE kernel
        torch.ops.alpha_moe.fused_moe_w8a8_up_down(
            hidden_states,  # FP8 quantized input
            hidden_states_scale,  # Input scale
            w13,  # Interleaved up/gate weights
            w13_scale,  # Interleaved scales
            w2,  # Down weights
            w2_scale,  # Down scales
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            output,
            top_k,
            block_m,
            block_n,
            warp_n,
            stages,
            routed_scaling_factor,
        )

        return AlphaMoeRunnerOutput(hidden_states=output)


# ============================================================================
# Pre/Post Permute Functions for Dispatcher Integration
# ============================================================================


@register_pre_permute("standard", "alpha_moe")
def standard_to_alpha_moe_pre_permute(
    dispatch_output: StandardDispatchOutput,
    quant_info: AlphaMoeQuantInfo,
    config: MoeRunnerConfig,
    running_state: dict,
) -> AlphaMoeRunnerInput:
    """
    Convert standard dispatch output to Alpha-MoE runner input.
    
    Note: This function only does FP8 quantization. The moe_align_block_size
    is done in AlphaMoeRunnerCore.run() after autotuning determines block_m.
    """
    from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output

    # Quantize input to FP8
    block_k = quant_info.block_shape[1]
    hidden_states_fp8, hidden_states_scale = per_token_group_quant_fp8(
        hidden_states, block_k
    )

    return AlphaMoeRunnerInput(
        hidden_states=hidden_states_fp8,
        hidden_states_scale=hidden_states_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


@register_post_permute("alpha_moe", "standard")
def alpha_moe_to_standard_post_permute(
    runner_output: AlphaMoeRunnerOutput,
    quant_info: AlphaMoeQuantInfo,
    config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    """
    Convert Alpha-MoE runner output to standard combine input.
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)
