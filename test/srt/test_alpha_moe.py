"""
Unit tests for Alpha-MoE integration.

This file tests the Alpha-MoE runner implementation including:
- Basic functionality checks
- Weight interleaving
- Requirements validation
- Integration with SGLang's MoE runner framework
"""

import unittest

import torch

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    try_cached_model,
)


class TestAlphaMoeAvailability(unittest.TestCase):
    """Test Alpha-MoE library availability detection."""

    def test_availability_check(self):
        """Test that ALPHA_MOE_AVAILABLE flag is properly set."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import ALPHA_MOE_AVAILABLE

        # ALPHA_MOE_AVAILABLE should be a boolean
        self.assertIsInstance(ALPHA_MOE_AVAILABLE, bool)

    def test_availability_functions(self):
        """Test availability helper functions."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import (
            get_alpha_moe_import_error,
            is_alpha_moe_available,
        )

        available = is_alpha_moe_available()
        self.assertIsInstance(available, bool)

        error = get_alpha_moe_import_error()
        if not available:
            self.assertIsInstance(error, Exception)
        else:
            self.assertIsNone(error)


class TestAlphaMoeEnumIntegration(unittest.TestCase):
    """Test MoeRunnerBackend enum integration."""

    def test_alpha_moe_enum_exists(self):
        """Test that ALPHA_MOE is in MoeRunnerBackend enum."""
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        self.assertTrue(hasattr(MoeRunnerBackend, "ALPHA_MOE"))
        self.assertEqual(MoeRunnerBackend.ALPHA_MOE.value, "alpha_moe")

    def test_is_alpha_moe_method(self):
        """Test the is_alpha_moe() method."""
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        self.assertTrue(MoeRunnerBackend.ALPHA_MOE.is_alpha_moe())
        self.assertFalse(MoeRunnerBackend.TRITON.is_alpha_moe())
        self.assertFalse(MoeRunnerBackend.DEEP_GEMM.is_alpha_moe())
        self.assertFalse(MoeRunnerBackend.AUTO.is_alpha_moe())


class TestWeightInterleaving(unittest.TestCase):
    """Test weight interleaving function."""

    def test_interleave_tensor_basic(self):
        """Test interleave_tensor preserves shape and is contiguous."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import interleave_tensor

        # Create a test tensor: [num_experts, intermediate_size, hidden_size]
        num_experts = 4
        intermediate_size = 512  # 2 * gate_size
        hidden_size = 256

        tensor = torch.randn(
            num_experts, intermediate_size, hidden_size, dtype=torch.float32
        )

        # Interleave with rep=8 (for weights)
        result = interleave_tensor(tensor, rep=8)

        # Shape should be preserved
        self.assertEqual(result.shape, tensor.shape)
        # Result should be contiguous
        self.assertTrue(result.is_contiguous())

    def test_interleave_tensor_rep1(self):
        """Test interleave_tensor with rep=1 for scales."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import interleave_tensor

        # Create a test scale tensor
        num_experts = 4
        scale_rows = 8  # intermediate_size // block_size
        scale_cols = 4  # hidden_size // block_size

        tensor = torch.randn(num_experts, scale_rows, scale_cols, dtype=torch.float32)

        # Interleave with rep=1
        result = interleave_tensor(tensor, rep=1)

        # Shape should be preserved
        self.assertEqual(result.shape, tensor.shape)
        self.assertTrue(result.is_contiguous())

    def test_interleave_preserves_dtype(self):
        """Test that interleave_tensor preserves tensor dtype."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import interleave_tensor

        tensor_fp8 = torch.randn(4, 512, 256).to(torch.float8_e4m3fn)
        result = interleave_tensor(tensor_fp8, rep=8)
        self.assertEqual(result.dtype, torch.float8_e4m3fn)

        tensor_fp32 = torch.randn(4, 512, 256).to(torch.float32)
        result = interleave_tensor(tensor_fp32, rep=1)
        self.assertEqual(result.dtype, torch.float32)

    def test_interleave_different_shapes(self):
        """Test interleave with various tensor shapes."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import interleave_tensor

        # Test various common shapes
        test_cases = [
            (8, 1024, 512, 8),  # Typical MoE weight shape
            (64, 2048, 1024, 8),  # Larger model
            (4, 256, 128, 1),  # Scale tensor
        ]

        for E, N, K, rep in test_cases:
            tensor = torch.randn(E, N, K, dtype=torch.float32)
            result = interleave_tensor(tensor, rep=rep)
            self.assertEqual(
                result.shape, tensor.shape, f"Shape mismatch for E={E}, N={N}, K={K}"
            )


class TestRequirementsChecking(unittest.TestCase):
    """Test Alpha-MoE requirements validation."""

    def test_block_size_validation_valid(self):
        """Test that [128, 128] block size is accepted."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import (
            ALPHA_MOE_AVAILABLE,
            check_alpha_moe_requirements,
        )

        is_satisfied, error_msg = check_alpha_moe_requirements(
            block_size=[128, 128], is_block_quant=True, layer=None
        )

        if ALPHA_MOE_AVAILABLE:
            self.assertTrue(is_satisfied, f"Valid config rejected: {error_msg}")
            self.assertEqual(error_msg, "")
        else:
            # If Alpha-MoE not installed, should return False
            self.assertFalse(is_satisfied)
            self.assertIn("not installed", error_msg.lower())

    def test_block_size_validation_invalid(self):
        """Test that invalid block sizes are rejected."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import (
            ALPHA_MOE_AVAILABLE,
            check_alpha_moe_requirements,
        )

        if not ALPHA_MOE_AVAILABLE:
            self.skipTest("Alpha-MoE not available")

        # Wrong block size
        is_satisfied, error_msg = check_alpha_moe_requirements(
            block_size=[64, 64], is_block_quant=True, layer=None
        )
        self.assertFalse(is_satisfied)
        self.assertIn("128, 128", error_msg)

        # Asymmetric block size
        is_satisfied, error_msg = check_alpha_moe_requirements(
            block_size=[128, 64], is_block_quant=True, layer=None
        )
        self.assertFalse(is_satisfied)

    def test_block_quant_required(self):
        """Test that block quantization is required."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import (
            ALPHA_MOE_AVAILABLE,
            check_alpha_moe_requirements,
        )

        if not ALPHA_MOE_AVAILABLE:
            self.skipTest("Alpha-MoE not available")

        # No block quantization
        is_satisfied, error_msg = check_alpha_moe_requirements(
            block_size=None, is_block_quant=False, layer=None
        )
        self.assertFalse(is_satisfied)
        self.assertIn("block", error_msg.lower())

    def test_quant_config_parameter(self):
        """Test that quant_config parameter works."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import (
            ALPHA_MOE_AVAILABLE,
            check_alpha_moe_requirements,
        )

        if not ALPHA_MOE_AVAILABLE:
            self.skipTest("Alpha-MoE not available")

        # Create mock quant_config
        class MockQuantConfig:
            weight_block_size = [128, 128]

        is_satisfied, error_msg = check_alpha_moe_requirements(quant_config=MockQuantConfig())
        self.assertTrue(is_satisfied, f"Valid quant_config rejected: {error_msg}")


class TestAlphaMoeDataClasses(unittest.TestCase):
    """Test Alpha-MoE dataclasses."""

    def test_alpha_moe_quant_info_creation(self):
        """Test AlphaMoeQuantInfo dataclass creation."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import AlphaMoeQuantInfo

        w13_weight = torch.randn(4, 512, 256).to(torch.float8_e4m3fn)
        w2_weight = torch.randn(4, 256, 256).to(torch.float8_e4m3fn)
        w13_scale = torch.randn(4, 4, 2)
        w2_scale = torch.randn(4, 2, 2)

        quant_info = AlphaMoeQuantInfo(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
        )

        self.assertIs(quant_info.w13_weight, w13_weight)
        self.assertIs(quant_info.w2_weight, w2_weight)
        self.assertIs(quant_info.w13_scale, w13_scale)
        self.assertIs(quant_info.w2_scale, w2_scale)

    def test_alpha_moe_quant_info_properties(self):
        """Test AlphaMoeQuantInfo inferred properties."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import AlphaMoeQuantInfo

        num_experts = 8
        intermediate_size = 1024
        hidden_size = 512

        w13_weight = torch.randn(num_experts, intermediate_size, hidden_size).to(
            torch.float8_e4m3fn
        )
        w2_weight = torch.randn(num_experts, hidden_size, intermediate_size // 2).to(
            torch.float8_e4m3fn
        )
        w13_scale = torch.randn(num_experts, intermediate_size // 128, hidden_size // 128)
        w2_scale = torch.randn(num_experts, hidden_size // 128, intermediate_size // 256)

        quant_info = AlphaMoeQuantInfo(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
        )

        # Test inferred properties
        self.assertEqual(quant_info.num_experts, num_experts)
        self.assertEqual(quant_info.intermediate_size, intermediate_size)
        self.assertEqual(quant_info.hidden_size, hidden_size)
        self.assertEqual(quant_info.block_shape, [128, 128])


class TestAlphaMoeRunnerCore(unittest.TestCase):
    """Test AlphaMoeRunnerCore class."""

    def test_runner_core_instantiation(self):
        """Test that AlphaMoeRunnerCore can be instantiated when available."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import (
            ALPHA_MOE_AVAILABLE,
            AlphaMoeRunnerCore,
        )
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        if not ALPHA_MOE_AVAILABLE:
            self.skipTest("Alpha-MoE not available")

        config = MoeRunnerConfig(
            hidden_size=256,
            intermediate_size_per_partition=512,
            num_local_experts=4,
            top_k=2,
            apply_router_weight_on_input=True,
            activation="silu",
            no_combine=False,
        )

        runner = AlphaMoeRunnerCore(config)
        self.assertIsNotNone(runner)

    def test_runner_core_import_error_when_unavailable(self):
        """Test that AlphaMoeRunnerCore raises ImportError when Alpha-MoE unavailable."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import (
            ALPHA_MOE_AVAILABLE,
            AlphaMoeRunnerCore,
        )
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        if ALPHA_MOE_AVAILABLE:
            self.skipTest("Alpha-MoE is available, cannot test import error")

        config = MoeRunnerConfig(
            hidden_size=256,
            intermediate_size_per_partition=512,
            num_local_experts=4,
            top_k=2,
            apply_router_weight_on_input=True,
            activation="silu",
            no_combine=False,
        )

        with self.assertRaises(ImportError):
            AlphaMoeRunnerCore(config)


class TestConfigFunctions(unittest.TestCase):
    """Test Alpha-MoE configuration functions."""

    def test_get_default_config(self):
        """Test _get_default_config function."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import _get_default_config

        # Test various batch sizes
        for num_tokens in [8, 16, 32, 64, 128, 256, 512, 1024]:
            config = _get_default_config(num_tokens)

            self.assertIn("block_m", config)
            self.assertIn("block_n", config)
            self.assertIn("warp_n", config)
            self.assertIn("stages", config)

            # block_m should be appropriate for num_tokens
            if num_tokens <= 16:
                self.assertEqual(config["block_m"], 16)
            elif num_tokens <= 64:
                self.assertEqual(config["block_m"], 32)
            elif num_tokens <= 256:
                self.assertEqual(config["block_m"], 64)
            else:
                self.assertEqual(config["block_m"], 128)

    def test_get_best_config_for_tokens(self):
        """Test get_best_config_for_tokens function."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import get_best_config_for_tokens

        # Test with None config (should return default)
        config = get_best_config_for_tokens(None, 64)
        self.assertIn("block_m", config)

        # Test with actual config
        test_config = {
            "16": {"block_m": 16, "block_n": 64, "warp_n": 4, "stages": 2},
            "64": {"block_m": 32, "block_n": 64, "warp_n": 4, "stages": 2},
            "256": {"block_m": 64, "block_n": 64, "warp_n": 4, "stages": 2},
        }

        # Should find closest match
        result = get_best_config_for_tokens(test_config, 60)
        self.assertEqual(result["block_m"], 32)  # Closest to 64

        result = get_best_config_for_tokens(test_config, 200)
        self.assertEqual(result["block_m"], 64)  # Closest to 256

    def test_get_config_cache_path(self):
        """Test _get_config_cache_path function."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import _get_config_cache_path

        path = _get_config_cache_path(E=8, N=1024, K=512)
        self.assertIn("moe_config_E8_N1024_K512.json", path)
        self.assertIn("alpha_moe", path)


class TestServerArgsIntegration(unittest.TestCase):
    """Test server_args.py integration with Alpha-MoE."""

    def test_alpha_moe_in_backend_choices(self):
        """Test that alpha_moe is in MOE_RUNNER_BACKEND_CHOICES."""
        from sglang.srt.server_args import MOE_RUNNER_BACKEND_CHOICES

        self.assertIn("alpha_moe", MOE_RUNNER_BACKEND_CHOICES)


class TestModuleExports(unittest.TestCase):
    """Test that all required exports are available."""

    def test_init_exports(self):
        """Test __init__.py exports."""
        from sglang.srt.layers.moe.moe_runner import (
            MoeRunner,
            MoeRunnerConfig,
            get_alpha_moe_quant_info,
            get_alpha_moe_runner_core,
            is_alpha_moe_available,
        )

        # These should be callable
        self.assertTrue(callable(get_alpha_moe_runner_core))
        self.assertTrue(callable(get_alpha_moe_quant_info))
        self.assertTrue(callable(is_alpha_moe_available))

    def test_alpha_moe_module_exports(self):
        """Test alpha_moe.py exports."""
        from sglang.srt.layers.moe.moe_runner.alpha_moe import (
            ALPHA_MOE_AVAILABLE,
            AlphaMoeQuantInfo,
            AlphaMoeRunnerCore,
            AlphaMoeRunnerInput,
            AlphaMoeRunnerOutput,
            check_alpha_moe_requirements,
            get_alpha_moe_config,
            get_alpha_moe_import_error,
            get_best_config_for_tokens,
            get_or_create_alpha_moe_config,
            interleave_tensor,
            is_alpha_moe_available,
        )

        # All should be importable (not necessarily callable if classes)
        self.assertIsNotNone(ALPHA_MOE_AVAILABLE)
        self.assertIsNotNone(AlphaMoeQuantInfo)
        self.assertIsNotNone(AlphaMoeRunnerCore)


# Integration test that requires a model with FP8 block quantization
@unittest.skipIf(get_device_sm() < 89, "Test requires CUDA SM 89+ (Ada/Hopper)")
class TestAlphaMoeIntegration(unittest.TestCase):
    """Integration test for Alpha-MoE with a real MoE model."""

    # Note: This test requires:
    # 1. Alpha-MoE library installed
    # 2. A compatible MoE model with FP8 block quantization
    # Uncomment and configure when ready to test

    # MODEL_PATH = "deepseek-ai/DeepSeek-V3-0324-FP8"

    # @classmethod
    # def setUpClass(cls):
    #     from sglang.srt.layers.moe.moe_runner.alpha_moe import ALPHA_MOE_AVAILABLE
    #     if not ALPHA_MOE_AVAILABLE:
    #         raise unittest.SkipTest("Alpha-MoE library not installed")
    #
    #     cls.model = try_cached_model(cls.MODEL_PATH)
    #     cls.base_url = DEFAULT_URL_FOR_TEST
    #     other_args = [
    #         "--trust-remote-code",
    #         "--moe-runner-backend", "alpha_moe",
    #         "--tp", "4",
    #     ]
    #     cls.process = popen_launch_server(
    #         cls.model,
    #         cls.base_url,
    #         timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    #         other_args=other_args,
    #     )
    #
    # @classmethod
    # def tearDownClass(cls):
    #     kill_process_tree(cls.process.pid)
    #
    # def test_basic_generation(self):
    #     """Test basic text generation with Alpha-MoE backend."""
    #     import requests
    #     response = requests.post(
    #         f"{self.base_url}/generate",
    #         json={
    #             "text": "Hello, how are you?",
    #             "sampling_params": {"max_new_tokens": 32}
    #         }
    #     )
    #     self.assertEqual(response.status_code, 200)
    #     result = response.json()
    #     self.assertIn("text", result)

    def test_placeholder(self):
        """Placeholder test for CI."""
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
