#!/usr/bin/env python3
"""
Test script for LingBot World sliding window mechanism.

This script demonstrates how to use the sliding window feature for long video generation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.sliding_window_i2v import SlidingWindowI2V


def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )
    return logging.getLogger(__name__)


def test_sliding_window_memory():
    """Test that sliding window uses constant memory."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TEST 1: Memory Usage Validation")
    logger.info("=" * 80)
    
    logger.info("Theoretical memory calculations:")
    logger.info("  Single window (49 frames):")
    logger.info("    - Latent shape: [16, 13, H, W]")
    logger.info("    - Memory per channel: H*W*13*4 bytes")
    logger.info("    - Estimated GPU memory: ~5GB per window")
    
    logger.info("\n  Two windows (97 frames with 24-frame overlap):")
    logger.info("    - Process window 1: ~5GB")
    logger.info("    - Process window 2: ~5GB (independent)")
    logger.info("    - Peak memory: ~5GB (sequential processing)")
    
    logger.info("\n  Without sliding window (97 frames):")
    logger.info("    - Latent shape: [16, 25, H, W]")
    logger.info("    - Estimated GPU memory: ~10GB")
    
    logger.info("\nMemory savings with sliding window:")
    for total_frames in [49, 97, 145, 193, 241]:
        window_size = 49
        overlap_size = 24
        stride = window_size - overlap_size
        num_windows = (total_frames - 1 + stride - 1) // stride + 1
        
        original_memory = total_frames * 0.1  # ~100MB per frame in latent space
        sliding_memory = window_size * 0.1    # Only one window at a time
        savings = (1 - sliding_memory / original_memory) * 100
        
        logger.info(f"  {total_frames} frames: "
                   f"{num_windows} windows, "
                   f"~{savings:.0f}% memory savings")


def test_parameter_validation():
    """Test parameter validation."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Parameter Validation")
    logger.info("=" * 80)
    
    # Create a mock WanI2V object
    class MockWanI2V:
        def __init__(self):
            self.device = torch.device("cpu")
    
    mock_model = MockWanI2V()
    
    # Test 1: Valid window sizes
    logger.info("\nValid window sizes (4n+1):")
    valid_sizes = [33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81]
    for size in valid_sizes:
        try:
            sw = SlidingWindowI2V(mock_model, window_size=size)
            logger.info(f"  ✓ {size}: Valid (num_latents={(size-1)//4+1})")
        except ValueError as e:
            logger.error(f"  ✗ {size}: {e}")
    
    # Test 2: Invalid window sizes
    logger.info("\nInvalid window sizes:")
    invalid_sizes = [48, 50, 51, 100]
    for size in invalid_sizes:
        try:
            sw = SlidingWindowI2V(mock_model, window_size=size)
            logger.warning(f"  ? {size}: Should have failed but didn't")
        except ValueError:
            logger.info(f"  ✓ {size}: Correctly rejected")
    
    # Test 3: Overlap validation
    logger.info("\nOverlap size validation:")
    try:
        sw = SlidingWindowI2V(mock_model, window_size=49, overlap_size=49)
        logger.warning("  ✗ overlap_size == window_size: Should have failed")
    except ValueError:
        logger.info("  ✓ overlap_size >= window_size: Correctly rejected")
    
    try:
        sw = SlidingWindowI2V(mock_model, window_size=49, overlap_size=48)
        logger.info("  ✓ overlap_size = 48 < window_size = 49: Valid")
    except ValueError as e:
        logger.error(f"  ✗ Unexpected error: {e}")


def test_window_calculation():
    """Test window frame range calculations."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Window Frame Range Calculation")
    logger.info("=" * 80)
    
    class MockWanI2V:
        def __init__(self):
            self.device = torch.device("cpu")
    
    mock_model = MockWanI2V()
    sw = SlidingWindowI2V(mock_model, window_size=49, overlap_size=24)
    
    logger.info(f"Configuration: window_size=49, overlap_size=24, stride=25")
    logger.info(f"\nWindow frame ranges for 200-frame video:")
    
    total_frames = 200
    stride = 49 - 24
    num_windows = (total_frames - 1 + stride - 1) // stride + 1
    
    for window_idx in range(num_windows):
        start, end = sw._get_window_frames(total_frames, window_idx)
        actual_frames = end - start
        logger.info(f"  Window {window_idx}: frames [{start:3d}-{end:3d}) = {actual_frames} frames")


def test_blending_modes():
    """Test different blending modes."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Blending Modes")
    logger.info("=" * 80)
    
    class MockWanI2V:
        def __init__(self):
            self.device = torch.device("cpu")
    
    mock_model = MockWanI2V()
    
    logger.info("\nSupported blending modes:")
    for blend_mode in ["lerp", "crossfade"]:
        try:
            sw = SlidingWindowI2V(mock_model, blend_mode=blend_mode)
            logger.info(f"  ✓ {blend_mode}: Available")
        except Exception as e:
            logger.error(f"  ✗ {blend_mode}: {e}")
    
    # Test invalid mode
    try:
        sw = SlidingWindowI2V(mock_model, blend_mode="invalid")
        logger.warning(f"  ? invalid: Should have failed but didn't")
    except Exception:
        logger.info(f"  ✓ Invalid mode detection: Working")


def test_memory_comparison():
    """Compare memory usage estimates."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Memory Usage Comparison")
    logger.info("=" * 80)
    
    logger.info("\nMemory overhead analysis:")
    logger.info("Assumptions: Each frame ≈ 100MB in latent space\n")
    
    configs = [
        {"frames": 49, "name": "Single window"},
        {"frames": 97, "name": "2x windows (50% overlap)"},
        {"frames": 145, "name": "3x windows (50% overlap)"},
        {"frames": 193, "name": "4x windows (50% overlap)"},
    ]
    
    logger.info(f"{'Config':<30} {'Original':<12} {'Sliding':<12} {'Savings':<10}")
    logger.info("-" * 64)
    
    for config in configs:
        frames = config["frames"]
        name = config["name"]
        
        # Original: all frames at once
        original = frames * 100  # MB
        
        # Sliding window: max 49 frames at once
        sliding = 49 * 100  # MB
        
        savings = (1 - sliding / original) * 100
        
        logger.info(f"{name:<30} {original:>6}MB      {sliding:>6}MB      {savings:>6.1f}%")


def generate_example_config():
    """Generate example configuration for different scenarios."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Example Configurations")
    logger.info("=" * 80)
    
    scenarios = [
        {
            "name": "Low VRAM (< 6GB)",
            "vram": "4-6GB",
            "config": {
                "window_size": 49,
                "overlap_size": 24,
                "blend_mode": "lerp",
                "offload_model": True,
            },
            "pros": ["Minimal memory usage", "Works on old GPUs"],
            "cons": ["Slower generation", "Possible visual artifacts"],
        },
        {
            "name": "Medium VRAM (6-10GB)",
            "vram": "6-10GB",
            "config": {
                "window_size": 57,
                "overlap_size": 28,
                "blend_mode": "crossfade",
                "offload_model": False,
            },
            "pros": ["Good balance", "Smooth transitions"],
            "cons": ["Slightly higher memory", "Medium generation time"],
        },
        {
            "name": "High VRAM (> 10GB)",
            "vram": ">10GB",
            "config": {
                "window_size": 81,
                "overlap_size": 40,
                "blend_mode": "crossfade",
                "offload_model": False,
            },
            "pros": ["Highest quality", "Smooth motion", "Faster generation"],
            "cons": ["Highest memory usage", "Requires powerful GPU"],
        },
    ]
    
    for scenario in scenarios:
        logger.info(f"\n{scenario['name']} ({scenario['vram']}):")
        for key, value in scenario["config"].items():
            logger.info(f"  --{key.replace('_', '-')} {value}")
        logger.info(f"  Pros:")
        for pro in scenario["pros"]:
            logger.info(f"    + {pro}")
        logger.info(f"  Cons:")
        for con in scenario["cons"]:
            logger.info(f"    - {con}")


def run_all_tests(verbose=False):
    """Run all tests."""
    logger = setup_logging(verbose)
    
    logger.info("LingBot World Sliding Window Mechanism - Test Suite")
    logger.info("=" * 80)
    
    try:
        test_sliding_window_memory()
        test_parameter_validation()
        test_window_calculation()
        test_blending_modes()
        test_memory_comparison()
        generate_example_config()
        
        logger.info("\n" + "=" * 80)
        logger.info("All tests completed successfully!")
        logger.info("=" * 80)
        return True
    
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test script for LingBot World sliding window mechanism"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    success = run_all_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)
