import pytest
import torch
import numpy as np
from PIL import Image
import cv2
from unittest.mock import patch

# Import the nodes to test
from paste_numbers_on_image import NumbersOverlayNode, NumbersOverlayAdvancedNode


class TestHelperFunctions:
    """Test utility functions used by the nodes"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.node = NumbersOverlayNode()
        self.advanced_node = NumbersOverlayAdvancedNode()

        # Create test tensors
        self.test_tensor_rgb = torch.rand(1, 100, 100, 3)
        self.test_tensor_grayscale = torch.rand(1, 100, 100, 1)

        # Create test PIL images
        self.test_pil_rgb = Image.new('RGB', (100, 100), (255, 0, 0))
        self.test_pil_grayscale = Image.new('L', (100, 100), 128)

    def test_tensor_to_pil_conversion_rgb(self):
        """Test RGB tensor to PIL conversion"""
        result = self.node.tensor_to_pil(self.test_tensor_rgb)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
        assert result.size == (100, 100)

    def test_tensor_to_pil_conversion_grayscale(self):
        """Test grayscale tensor to PIL conversion"""
        result = self.node.tensor_to_pil(self.test_tensor_grayscale)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_pil_to_tensor_conversion_rgb(self):
        """Test RGB PIL to tensor conversion"""
        result = self.node.pil_to_tensor(self.test_pil_rgb)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 100, 100, 3)
        assert torch.all((result >= 0) & (result <= 1))

    def test_pil_to_tensor_conversion_grayscale(self):
        """Test grayscale PIL to tensor conversion (should become RGB)"""
        result = self.node.pil_to_tensor(self.test_pil_grayscale)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 100, 100, 3)
        assert torch.all((result >= 0) & (result <= 1))

    def test_get_scaling_method(self):
        """Test scaling method mapping"""
        assert self.node.get_scaling_method("lanczos") == Image.Resampling.LANCZOS
        assert self.node.get_scaling_method("bicubic") == Image.Resampling.BICUBIC
        assert self.node.get_scaling_method("bilinear") == Image.Resampling.BILINEAR
        assert self.node.get_scaling_method("nearest") == Image.Resampling.NEAREST

        # Test fallback for unknown method
        assert self.node.get_scaling_method("unknown") == Image.Resampling.LANCZOS


class TestMaskLogic:
    """Test mask creation and manipulation logic"""

    def setup_method(self):
        self.node = NumbersOverlayNode()
        self.advanced_node = NumbersOverlayAdvancedNode()

    def test_create_mask_from_image_rgb_white_background(self):
        """Test mask creation with white background and black numbers"""
        # Create image with white background and black text
        test_img = Image.new('RGB', (50, 50), (255, 255, 255))  # White background
        # Add some black pixels to simulate numbers
        pixels = test_img.load()
        for i in range(10, 20):
            for j in range(10, 20):
                pixels[i, j] = (0, 0, 0)  # Black square

        mask = self.node.create_mask_from_image(test_img, white_threshold=240)

        # Should find the black pixels
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (50, 50)
        assert np.sum(mask) == 100  # 10x10 black square

    def test_create_mask_from_image_grayscale(self):
        """Test mask creation with grayscale image"""
        # Create grayscale image with varying intensities
        test_array = np.full((50, 50), 255, dtype=np.uint8)  # White background
        test_array[10:20, 10:20] = 100  # Gray square
        test_img = Image.fromarray(test_array, mode='L')

        mask = self.node.create_mask_from_image(test_img, white_threshold=240)

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (50, 50)
        assert np.sum(mask) == 100  # 10x10 gray square should be detected

    def test_create_mask_threshold_sensitivity(self):
        """Test mask sensitivity to white threshold values"""
        # Create image with different gray levels
        test_img = Image.new('RGB', (30, 30), (255, 255, 255))
        pixels = test_img.load()

        # Add pixels with different intensities
        pixels[10, 10] = (200, 200, 200)  # Light gray
        pixels[11, 11] = (150, 150, 150)  # Medium gray
        pixels[12, 12] = (50, 50, 50)  # Dark gray

        # High threshold should catch more pixels
        mask_high = self.node.create_mask_from_image(test_img, white_threshold=250)
        mask_low = self.node.create_mask_from_image(test_img, white_threshold=100)

        assert np.sum(mask_high) >= np.sum(mask_low)
        assert np.sum(mask_high) >= 3  # Should catch all three test pixels
        assert np.sum(mask_low) >= 1  # Should catch at least the darkest pixel

    def test_advanced_node_create_numbers_mask(self):
        """Test advanced node mask creation with debugging output"""
        test_array = np.full((40, 40, 3), 255, dtype=np.uint8)
        test_array[15:25, 15:25] = [0, 0, 0]  # Black square

        with patch('builtins.print'):  # Mock print to avoid console spam in tests
            mask = self.advanced_node.create_numbers_mask(test_array, white_threshold=240)

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (40, 40)
        assert np.sum(mask) == 100  # 10x10 black square


class TestBlendModes:
    """Test different blending modes for combining images"""

    def setup_method(self):
        self.node = NumbersOverlayNode()

        # Create test pixel arrays
        self.base_pixels = np.array([128, 128, 128], dtype=np.uint8)  # Mid gray
        self.overlay_pixels = np.array([64, 192, 255], dtype=np.uint8)  # Mixed colors
        self.alpha = 1.0

    def test_replace_blend_mode(self):
        """Test replace blend mode (should return overlay pixels)"""
        result = self.node.apply_blend_mode(
            self.base_pixels, self.overlay_pixels, "replace", self.alpha
        )

        np.testing.assert_array_equal(result, self.overlay_pixels)

    def test_multiply_blend_mode(self):
        """Test multiply blend mode"""
        result = self.node.apply_blend_mode(
            self.base_pixels, self.overlay_pixels, "multiply", self.alpha
        )

        # Manual calculation: (128/255) * (64/255) * 255 ≈ 32
        expected_r = int((128 / 255) * (64 / 255) * 255)
        assert abs(result[0] - expected_r) <= 1  # Allow for rounding

        # Result should be darker than both inputs for multiply
        assert result[0] <= min(self.base_pixels[0], self.overlay_pixels[0])

    def test_overlay_blend_mode(self):
        """Test overlay blend mode with known inputs"""
        # Test with predictable values
        base = np.array([64, 192, 128], dtype=np.uint8)  # Below 0.5, above 0.5, at 0.5
        overlay = np.array([128, 128, 128], dtype=np.uint8)  # Mid gray

        result = self.node.apply_blend_mode(base, overlay, "overlay", 1.0)

        # For overlay mode:
        # - Dark base (64): 2 * base * overlay
        # - Light base (192): 1 - 2 * (1-base) * (1-overlay)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert len(result) == 3

    def test_soft_light_blend_mode(self):
        """Test soft light blend mode"""
        result = self.node.apply_blend_mode(
            self.base_pixels, self.overlay_pixels, "soft_light", self.alpha
        )

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert len(result) == 3
        # Soft light should produce subtle changes
        assert not np.array_equal(result, self.base_pixels)

    def test_blend_mode_with_alpha(self):
        """Test blend modes with partial transparency"""
        alpha = 0.5

        result = self.node.apply_blend_mode(
            self.base_pixels, self.overlay_pixels, "replace", alpha
        )

        # With 50% alpha, result should be halfway between base and overlay
        expected = (self.base_pixels.astype(float) * 0.5 +
                    self.overlay_pixels.astype(float) * 0.5).astype(np.uint8)

        np.testing.assert_array_almost_equal(result, expected, decimal=0)

    def test_unknown_blend_mode_fallback(self):
        """Test that unknown blend modes fall back to overlay behavior"""
        result = self.node.apply_blend_mode(
            self.base_pixels, self.overlay_pixels, "unknown_mode", self.alpha
        )

        # Should fall back to overlay behavior
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8


class TestNodeIO:
    """Test node input/output behavior and image processing"""

    def setup_method(self):
        self.node = NumbersOverlayNode()
        self.advanced_node = NumbersOverlayAdvancedNode()

        # Create test input tensors
        self.input_tensor = torch.rand(1, 200, 300, 3)  # 300x200 image
        self.numbers_tensor = torch.rand(1, 100, 150, 3)  # 150x100 image (different size)

        # Create tensors with specific patterns for testing
        self.white_bg_tensor = torch.ones(1, 50, 50, 3)  # All white
        self.black_numbers_tensor = torch.ones(1, 50, 50, 3)
        self.black_numbers_tensor[0, 20:30, 20:30, :] = 0  # Black square in center

    def test_basic_overlay_same_size(self):
        """Test overlay with images of the same size"""
        # Create same-size tensors
        input_tensor = torch.rand(1, 100, 100, 3)
        numbers_tensor = torch.ones(1, 100, 100, 3)
        numbers_tensor[0, 40:60, 40:60, :] = 0  # Black square

        with patch('builtins.print'):  # Mock print statements
            result = self.node.overlay_numbers(
                input_tensor, numbers_tensor,
                transparency=1.0, white_threshold=240,
                blend_mode="replace", auto_scale=True, scaling_method="lanczos"
            )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert result[0].shape == input_tensor.shape

    def test_auto_scale_enabled(self):
        """Test overlay with auto_scale enabled (different sized images)"""
        with patch('builtins.print'):
            result = self.node.overlay_numbers(
                self.input_tensor, self.numbers_tensor,
                transparency=1.0, white_threshold=240,
                blend_mode="replace", auto_scale=True, scaling_method="lanczos"
            )

        assert isinstance(result, tuple)
        assert result[0].shape == self.input_tensor.shape  # Should match input size

    def test_auto_scale_disabled_crop(self):
        """Test overlay with auto_scale disabled - larger numbers image should be cropped"""
        # Numbers image larger than input
        large_numbers = torch.rand(1, 400, 500, 3)

        with patch('builtins.print'):
            result = self.node.overlay_numbers(
                self.input_tensor, large_numbers,
                transparency=1.0, white_threshold=240,
                blend_mode="replace", auto_scale=False, scaling_method="lanczos"
            )

        assert result[0].shape == self.input_tensor.shape

    def test_auto_scale_disabled_pad(self):
        """Test overlay with auto_scale disabled - smaller numbers image should be padded"""
        # Numbers image smaller than input
        small_numbers = torch.rand(1, 50, 75, 3)

        with patch('builtins.print'):
            result = self.node.overlay_numbers(
                self.input_tensor, small_numbers,
                transparency=1.0, white_threshold=240,
                blend_mode="replace", auto_scale=False, scaling_method="lanczos"
            )

        assert result[0].shape == self.input_tensor.shape

    def test_no_overlay_with_all_white_numbers(self):
        """Test that all-white numbers image produces no overlay"""
        all_white_numbers = torch.ones(1, 100, 100, 3)  # All white

        with patch('builtins.print') as mock_print:
            result = self.node.overlay_numbers(
                self.input_tensor, all_white_numbers,
                transparency=1.0, white_threshold=240,
                blend_mode="replace", auto_scale=True, scaling_method="lanczos"
            )

        # Should warn about no non-white pixels
        warning_printed = any("WARNING: No non-white pixels found" in str(call)
                              for call in mock_print.call_args_list)
        assert warning_printed

    def test_transparency_effect(self):
        """Test that transparency parameter affects the output"""
        with patch('builtins.print'):
            result_full = self.node.overlay_numbers(
                self.white_bg_tensor, self.black_numbers_tensor,
                transparency=1.0, white_threshold=240,
                blend_mode="replace", auto_scale=True, scaling_method="lanczos"
            )

            result_half = self.node.overlay_numbers(
                self.white_bg_tensor, self.black_numbers_tensor,
                transparency=0.5, white_threshold=240,
                blend_mode="replace", auto_scale=True, scaling_method="lanczos"
            )

        # Results should be different due to transparency
        assert not torch.equal(result_full[0], result_half[0])


class TestAdvancedNodeFeatures:
    """Test advanced node specific features"""

    def setup_method(self):
        self.advanced_node = NumbersOverlayAdvancedNode()
        self.input_tensor = torch.rand(1, 100, 100, 3)
        self.numbers_tensor = torch.ones(1, 100, 100, 3)
        self.numbers_tensor[0, 40:60, 40:60, :] = 0  # Black square

    def test_hex_to_rgb_conversion(self):
        """Test hex color to RGB conversion"""
        assert self.advanced_node.hex_to_rgb("#FF0000") == (255, 0, 0)
        assert self.advanced_node.hex_to_rgb("#00FF00") == (0, 255, 0)
        assert self.advanced_node.hex_to_rgb("#0000FF") == (0, 0, 255)
        assert self.advanced_node.hex_to_rgb("FFFFFF") == (255, 255, 255)  # Without #

        # Test invalid hex color
        assert self.advanced_node.hex_to_rgb("#INVALID") == (255, 255, 255)  # Should fallback to white

    def test_apply_outline(self):
        """Test outline application to mask"""
        # Create a simple mask
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:30, 20:30] = True  # 10x10 square

        outlined = self.advanced_node.apply_outline(mask, outline_width=2)

        # Outlined mask should be larger than original
        assert np.sum(outlined) > np.sum(mask)
        assert outlined.dtype == bool

    def test_apply_outline_zero_width(self):
        """Test that zero outline width returns original mask"""
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:30, 20:30] = True

        outlined = self.advanced_node.apply_outline(mask, outline_width=0)

        np.testing.assert_array_equal(outlined, mask)

    def test_advanced_overlay_with_mask_output(self):
        """Test that advanced node returns both image and mask"""
        with patch('builtins.print'):
            result = self.advanced_node.overlay_numbers_advanced(
                self.input_tensor, self.numbers_tensor, transparency=1.0
            )

        assert isinstance(result, tuple)
        assert len(result) == 2  # Should return image and mask
        assert isinstance(result[0], torch.Tensor)  # Image
        assert isinstance(result[1], torch.Tensor)  # Mask

        # Mask should be 2D (height, width)
        assert len(result[1].shape) == 3  # Batch dimension included
        assert result[1].shape[1:] == (100, 100)  # Height, width

    def test_color_replacement(self):
        """Test numbers color replacement feature"""
        with patch('builtins.print'):
            result = self.advanced_node.overlay_numbers_advanced(
                self.input_tensor, self.numbers_tensor,
                transparency=1.0, numbers_color="#FF0000"
            )

        # Should complete without error
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_external_mask_application(self):
        """Test application of external mask"""
        # Create external mask tensor
        external_mask = torch.zeros(1, 100, 100)
        external_mask[0, 0:50, 0:50] = 1  # Only top-left quadrant

        with patch('builtins.print'):
            result = self.advanced_node.overlay_numbers_advanced(
                self.input_tensor, self.numbers_tensor,
                transparency=1.0, mask_image=external_mask
            )

        assert isinstance(result, tuple)
        assert len(result) == 2


class TestErrorHandling:
    """Test error handling and edge cases"""

    def setup_method(self):
        self.node = NumbersOverlayNode()

    def test_invalid_tensor_shapes(self):
        """Test handling of invalid tensor shapes"""
        # Test with 2D tensor (missing channel dimension)
        invalid_tensor = torch.rand(100, 100)
        valid_tensor = torch.rand(1, 100, 100, 3)

        # Should handle gracefully and not crash
        try:
            with patch('builtins.print'):
                result = self.node.overlay_numbers(
                    valid_tensor, invalid_tensor,
                    transparency=1.0, white_threshold=240,
                    blend_mode="replace", auto_scale=True, scaling_method="lanczos"
                )
            # If it doesn't crash, that's good
            assert True
        except Exception as e:
            # If it does crash, the error should be informative
            assert "shape" in str(e).lower() or "dimension" in str(e).lower()

    def test_extreme_threshold_values(self):
        """Test with extreme white threshold values"""
        input_tensor = torch.rand(1, 50, 50, 3)
        numbers_tensor = torch.rand(1, 50, 50, 3)

        # Test with very low threshold (should catch almost everything)
        with patch('builtins.print'):
            result_low = self.node.overlay_numbers(
                input_tensor, numbers_tensor,
                transparency=1.0, white_threshold=0,
                blend_mode="replace", auto_scale=True, scaling_method="lanczos"
            )

        # Test with maximum threshold (should catch everything except pure white)
        with patch('builtins.print'):
            result_high = self.node.overlay_numbers(
                input_tensor, numbers_tensor,
                transparency=1.0, white_threshold=255,
                blend_mode="replace", auto_scale=True, scaling_method="lanczos"
            )

        assert isinstance(result_low, tuple)
        assert isinstance(result_high, tuple)


# Performance and regression tests
class TestPerformanceAndRegression:
    """Test performance characteristics and prevent regressions"""

    def setup_method(self):
        self.node = NumbersOverlayNode()

        # Create larger test images for performance testing
        self.large_input = torch.rand(1, 1000, 1000, 3)  # 1MP image
        self.large_numbers = torch.ones(1, 1000, 1000, 3)
        # Add some non-white pixels
        self.large_numbers[0, 400:600, 400:600, :] = 0

    def test_large_image_processing_time(self):
        """Test that large images process within reasonable time"""
        import time

        start_time = time.time()

        with patch('builtins.print'):
            result = self.node.overlay_numbers(
                self.large_input, self.large_numbers,
                transparency=1.0, white_threshold=240,
                blend_mode="replace", auto_scale=True, scaling_method="lanczos"
            )

        end_time = time.time()
        processing_time = end_time - start_time

        # Should process 1MP image in under 5 seconds (adjust based on your requirements)
        assert processing_time < 5.0, f"Processing took {processing_time:.2f} seconds, expected < 5.0"
        assert isinstance(result, tuple)

    def test_memory_usage_reasonable(self):
        """Test that memory usage doesn't explode with large images"""
        import gc
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        with patch('builtins.print'):
            result = self.node.overlay_numbers(
                self.large_input, self.large_numbers,
                transparency=1.0, white_threshold=240,
                blend_mode="replace", auto_scale=True, scaling_method="lanczos"
            )

        # Force garbage collection
        gc.collect()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB for 1MP image)
        max_memory_increase = 500 * 1024 * 1024  # 500MB in bytes
        assert memory_increase < max_memory_increase, f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB"

    def test_output_consistency(self):
        """Test that same inputs always produce same outputs (regression test)"""
        # Use fixed seed for reproducible random tensors
        torch.manual_seed(42)
        input_tensor = torch.rand(1, 100, 100, 3)

        torch.manual_seed(43)
        numbers_tensor = torch.rand(1, 100, 100, 3)

        with patch('builtins.print'):
            result1 = self.node.overlay_numbers(
                input_tensor, numbers_tensor,
                transparency=0.75, white_threshold=200,
                blend_mode="multiply", auto_scale=True, scaling_method="bicubic"
            )

            result2 = self.node.overlay_numbers(
                input_tensor, numbers_tensor,
                transparency=0.75, white_threshold=200,
                blend_mode="multiply", auto_scale=True, scaling_method="bicubic"
            )

        # Results should be identical
        assert torch.allclose(result1[0], result2[0], atol=1e-6)


if __name__ == "__main__":
    # Simple test runner for when pytest is not available
    print("Running NumbersOverlay tests...")

    test_classes = [
        TestHelperFunctions,
        TestMaskLogic,
        TestBlendModes,
        TestNodeIO,
        TestAdvancedNodeFeatures,
        TestErrorHandling,
        TestPerformanceAndRegression
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        instance = test_class()

        # Get all test methods
        test_methods = [method for method in dir(instance) if method.startswith('test_')]

        for test_method_name in test_methods:
            total_tests += 1
            try:
                # Run setup if it exists
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()

                # Run the test
                test_method = getattr(instance, test_method_name)
                test_method()

                print(f"✓ {test_method_name}")
                passed_tests += 1

            except Exception as e:
                print(f"✗ {test_method_name}: {str(e)}")

    print(f"\n--- Test Summary ---")
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests / total_tests * 100:.1f}%")
