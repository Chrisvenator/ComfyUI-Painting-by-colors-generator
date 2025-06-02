import pytest
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from unittest.mock import Mock, patch
import time

# Assuming the node is imported from the main module
from preprocessor import EnhancedPaintByNumbersNode


class TestEnhancedPaintByNumbersNode:
    """Test suite for EnhancedPaintByNumbersNode"""

    def setup_method(self):
        """Setup test fixtures"""
        self.node = EnhancedPaintByNumbersNode()

        # Create test images
        self.test_image_rgb = self.create_test_image(256, 256, mode='RGB')
        self.test_image_grayscale = self.create_test_image(256, 256, mode='L')
        self.test_tensor = self.create_test_tensor(256, 256)

        # Test hex stack
        self.test_hex_stack = {
            'count': 5,
            'hex_colors': ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF'],
            'rgb_colors': np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]])
        }

    def create_test_image(self, width, height, mode='RGB'):
        """Create a test image with known patterns"""
        if mode == 'RGB':
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            # Add some colored regions for testing
            draw.rectangle([0, 0, width // 2, height // 2], fill='red')
            draw.rectangle([width // 2, 0, width, height // 2], fill='green')
            draw.rectangle([0, height // 2, width // 2, height], fill='blue')
            draw.rectangle([width // 2, height // 2, width, height], fill='yellow')
        else:
            img = Image.new(mode, (width, height), color=128)
        return img

    def create_test_tensor(self, width, height):
        """Create a test tensor in ComfyUI format"""
        # Create random RGB image
        np_image = np.random.rand(height, width, 3).astype(np.float32)
        tensor = torch.from_numpy(np_image).unsqueeze(0)  # Add batch dimension
        return tensor

    def create_noisy_image(self, width, height):
        """Create an image with noise for denoising tests"""
        img = self.create_test_image(width, height)
        img_array = np.array(img)
        noise = np.random.normal(0, 20, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    # Unit Tests for Helper Methods

    def test_tensor_to_pil_conversion(self):
        """Test tensor to PIL conversion"""
        pil_img = self.node.tensor_to_pil(self.test_tensor)

        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == 'RGB'
        assert pil_img.size == (256, 256)

    def test_pil_to_tensor_conversion(self):
        """Test PIL to tensor conversion"""
        tensor = self.node.pil_to_tensor(self.test_image_rgb)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 256, 256, 3)  # Batch, H, W, C
        assert tensor.dtype == torch.float32
        assert torch.all(tensor >= 0) and torch.all(tensor <= 1)

    def test_round_trip_conversion(self):
        """Test tensor -> PIL -> tensor conversion preserves dimensions"""
        original_tensor = self.test_tensor
        pil_img = self.node.tensor_to_pil(original_tensor)
        converted_tensor = self.node.pil_to_tensor(pil_img)

        assert original_tensor.shape == converted_tensor.shape

    def test_grayscale_to_rgb_conversion(self):
        """Test grayscale PIL image gets converted to RGB tensor"""
        tensor = self.node.pil_to_tensor(self.test_image_grayscale)

        assert tensor.shape[3] == 3  # Should have 3 channels

    # Denoising Tests

    def test_denoise_image_bilateral_filter(self):
        """Test bilateral filter denoising reduces noise while preserving edges"""
        noisy_img = self.create_noisy_image(128, 128)

        denoised = self.node.denoise_image(noisy_img, noise_reduction_strength=3.0, use_bilateral=True)

        assert isinstance(denoised, Image.Image)
        assert denoised.size == noisy_img.size

        # Check noise reduction - denoised image should have lower variance
        original_var = np.var(np.array(noisy_img))
        denoised_var = np.var(np.array(denoised))
        assert denoised_var < original_var, "Denoising should reduce image variance"

    def test_denoise_image_no_bilateral(self):
        """Test denoising without bilateral filter"""
        noisy_img = self.create_noisy_image(128, 128)

        denoised = self.node.denoise_image(noisy_img, noise_reduction_strength=3.0, use_bilateral=False)

        assert isinstance(denoised, Image.Image)
        assert denoised.size == noisy_img.size

    def test_denoise_zero_strength(self):
        """Test that zero noise reduction strength returns similar image"""
        original_img = self.test_image_rgb

        result = self.node.denoise_image(original_img, noise_reduction_strength=0.0, use_bilateral=True)

        # Images should be very similar (allowing for minor processing differences)
        original_array = np.array(original_img)
        result_array = np.array(result)

        mean_diff = np.mean(np.abs(original_array.astype(float) - result_array.astype(float)))
        assert mean_diff < 5.0, "Zero strength denoising should preserve image"

    def test_denoise_high_strength_gaussian_blur(self):
        """Test that high strength adds Gaussian blur"""
        test_img = self.test_image_rgb

        # High strength should trigger additional Gaussian blur
        result = self.node.denoise_image(test_img, noise_reduction_strength=4.0, use_bilateral=True)

        assert isinstance(result, Image.Image)
        # High strength blur should reduce high-frequency content
        original_laplacian_var = cv2.Laplacian(np.array(test_img), cv2.CV_64F).var()
        result_laplacian_var = cv2.Laplacian(np.array(result), cv2.CV_64F).var()
        assert result_laplacian_var < original_laplacian_var, "High strength should reduce detail"

    # Color Quantization Tests

    def test_quantize_colors_improved(self):
        """Test K-means color quantization"""
        num_colors = 8
        quantized = self.node.quantize_colors_improved(self.test_image_rgb, num_colors)

        assert isinstance(quantized, Image.Image)
        assert quantized.size == self.test_image_rgb.size

        # Check that we have at most num_colors unique colors
        unique_colors = len(set(quantized.getdata()))
        assert unique_colors <= num_colors, f"Should have <= {num_colors} colors, got {unique_colors}"

    def test_quantize_colors_intensity_scaling(self):
        """Test color intensity parameter affects output"""
        base_quantized = self.node.quantize_colors_improved(self.test_image_rgb, 8, intensity=1.0)
        bright_quantized = self.node.quantize_colors_improved(self.test_image_rgb, 8, intensity=1.5)

        base_mean = np.mean(np.array(base_quantized))
        bright_mean = np.mean(np.array(bright_quantized))

        assert bright_mean > base_mean, "Higher intensity should produce brighter image"

    def test_quantize_to_hex_palette(self):
        """Test quantization to specific hex palette"""
        hex_colors = ['#FF0000', '#00FF00', '#0000FF']
        rgb_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])

        quantized = self.node.quantize_to_hex_palette(self.test_image_rgb, hex_colors, rgb_colors)

        assert isinstance(quantized, Image.Image)
        assert quantized.size == self.test_image_rgb.size

        # All pixels should be one of the palette colors
        quantized_array = np.array(quantized)
        unique_colors = np.unique(quantized_array.reshape(-1, 3), axis=0)

        # Check that all unique colors are in the palette
        for color in unique_colors:
            distances = np.sum((rgb_colors - color) ** 2, axis=1)
            assert np.min(distances) < 1e-6, f"Color {color} not in palette"

    def test_select_optimal_colors_from_palette(self):
        """Test selection of most representative colors from palette"""
        # Create palette with more colors than needed
        large_palette = np.array([
            [255, 0, 0], [0, 255, 0], [0, 0, 255],  # Primary colors (should be selected)
            [128, 128, 128], [200, 200, 200],  # Grays (might not be selected)
            [255, 255, 0], [255, 0, 255]  # Secondary colors
        ])

        selected = self.node.select_optimal_colors_from_palette(self.test_image_rgb, large_palette, 4)

        assert len(selected) == 4
        assert isinstance(selected, np.ndarray)
        assert selected.shape[1] == 3  # RGB colors

    # Region Cleanup Tests

    def test_clean_small_regions(self):
        """Test removal of small isolated regions"""
        # Create image with small isolated regions
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White background
        test_img[10:15, 10:15] = [255, 0, 0]  # Small red region (25 pixels)
        test_img[50:80, 50:80] = [0, 255, 0]  # Large green region (900 pixels)

        cleaned = self.node.clean_small_regions(test_img.copy(), min_size=100)

        # Small region should be removed/changed, large region should remain
        green_pixels = np.sum(np.all(cleaned == [0, 255, 0], axis=2))
        assert green_pixels > 800, "Large green region should be preserved"

        # Small red region should be significantly reduced or gone
        red_pixels = np.sum(np.all(cleaned == [255, 0, 0], axis=2))
        assert red_pixels < 10, "Small red region should be mostly removed"

    def test_clean_small_regions_zero_min_size(self):
        """Test that zero min_size preserves image unchanged"""
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = self.node.clean_small_regions(original.copy(), min_size=0)

        np.testing.assert_array_equal(original, result)

    def test_apply_gentle_cleanup(self):
        """Test gentle morphological cleanup"""
        # Create image with small noise
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        test_img[50, 50] = [0, 0, 0]  # Single black pixel (noise)

        cleaned = self.node.apply_gentle_cleanup(test_img.copy())

        assert isinstance(cleaned, np.ndarray)
        assert cleaned.shape == test_img.shape
        assert cleaned.dtype == np.uint8

    # Integration Tests

    def test_create_color_quantized_image_with_hex_stack(self):
        """Test complete color quantization pipeline with hex stack"""
        result = self.node.create_color_quantized_image(
            self.test_image_rgb,
            hex_stack=self.test_hex_stack,
            num_colors=5,
            blur_radius=1.0,
            color_intensity=1.0,
            noise_reduction_strength=1.0,
            use_bilateral=True,
            use_morphological=True,
            min_region_size=50
        )

        assert isinstance(result, Image.Image)
        assert result.size == self.test_image_rgb.size
        assert result.mode == 'RGB'

    def test_create_color_quantized_image_without_hex_stack(self):
        """Test complete color quantization pipeline without hex stack"""
        result = self.node.create_color_quantized_image(
            self.test_image_rgb,
            hex_stack=None,
            num_colors=8,
            blur_radius=2.0,
            color_intensity=1.2,
            noise_reduction_strength=2.0,
            use_bilateral=True,
            use_morphological=True,
            min_region_size=100
        )

        assert isinstance(result, Image.Image)
        assert result.size == self.test_image_rgb.size

        # Check color count
        unique_colors = len(set(result.getdata()))
        assert unique_colors <= 8

    def test_node_process_method(self):
        """Test the main process method (complete node integration)"""
        result = self.node.process(
            original_image=self.test_tensor,
            blur_radius=1.0,
            color_intensity=1.0,
            noise_reduction_strength=1.0,
            bilateral_filter=True,
            morphological_cleanup=True,
            min_region_size=50,
            hex_stack=self.test_hex_stack,
            num_colors=5
        )

        # Should return tuple with single tensor
        assert isinstance(result, tuple)
        assert len(result) == 1

        output_tensor = result[0]
        assert isinstance(output_tensor, torch.Tensor)
        assert output_tensor.shape == self.test_tensor.shape
        assert torch.all(output_tensor >= 0) and torch.all(output_tensor <= 1)

    def test_node_input_types_structure(self):
        """Test that INPUT_TYPES returns correct structure"""
        input_types = EnhancedPaintByNumbersNode.INPUT_TYPES()

        assert 'required' in input_types
        assert 'optional' in input_types

        required = input_types['required']
        assert 'original_image' in required
        assert 'blur_radius' in required
        assert 'color_intensity' in required
        assert 'noise_reduction_strength' in required

        # Check parameter ranges
        assert required['blur_radius'][1]['min'] == 0.0
        assert required['blur_radius'][1]['max'] == 100.0
        assert required['color_intensity'][1]['min'] == 0.5
        assert required['color_intensity'][1]['max'] == 2.0

    # Performance and Edge Case Tests

    def test_process_performance_smoke_test(self):
        """Smoke test to ensure processing completes within reasonable time"""
        large_tensor = self.create_test_tensor(512, 512)  # 1MP equivalent

        start_time = time.time()
        result = self.node.process(
            original_image=large_tensor,
            blur_radius=2.0,
            color_intensity=1.0,
            noise_reduction_strength=2.0,
            bilateral_filter=True,
            morphological_cleanup=True,
            min_region_size=100,
            num_colors=12
        )
        end_time = time.time()

        processing_time = end_time - start_time
        assert processing_time < 10.0, f"Processing took {processing_time:.2f}s, should be < 10s"

        # Verify output
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == large_tensor.shape

    def test_edge_case_single_color_image(self):
        """Test processing of single-color image"""
        single_color_img = Image.new('RGB', (100, 100), color='red')
        single_color_tensor = self.node.pil_to_tensor(single_color_img)

        result = self.node.process(
            original_image=single_color_tensor,
            blur_radius=0.0,
            color_intensity=1.0,
            noise_reduction_strength=0.0,
            bilateral_filter=False,
            morphological_cleanup=False,
            min_region_size=0,
            num_colors=5
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == single_color_tensor.shape

    def test_edge_case_minimum_parameters(self):
        """Test with minimum possible parameters"""
        result = self.node.process(
            original_image=self.test_tensor,
            blur_radius=0.0,
            color_intensity=0.5,  # minimum
            noise_reduction_strength=0.0,
            bilateral_filter=False,
            morphological_cleanup=False,
            min_region_size=10,  # minimum
            num_colors=2  # minimum
        )

        assert isinstance(result, tuple)
        assert len(result) == 1

        # Should have at most 2 colors
        result_pil = self.node.tensor_to_pil(result[0])
        unique_colors = len(set(result_pil.getdata()))
        assert unique_colors <= 2

    def test_edge_case_maximum_parameters(self):
        """Test with maximum possible parameters"""
        result = self.node.process(
            original_image=self.test_tensor,
            blur_radius=100.0,  # maximum
            color_intensity=2.0,  # maximum
            noise_reduction_strength=5.0,  # maximum
            bilateral_filter=True,
            morphological_cleanup=True,
            min_region_size=1000,  # maximum
            num_colors=50  # maximum
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == self.test_tensor.shape

    def test_hex_stack_color_limit_selection(self):
        """Test that hex stack with more colors than num_colors selects appropriately"""
        large_hex_stack = {
            'count': 10,
            'hex_colors': [f'#{i:02x}{i:02x}{i:02x}' for i in range(0, 255, 25)],
            'rgb_colors': np.array([[i, i, i] for i in range(0, 255, 25)])
        }

        result = self.node.process(
            original_image=self.test_tensor,
            blur_radius=0.0,
            color_intensity=1.0,
            noise_reduction_strength=0.0,
            bilateral_filter=False,
            morphological_cleanup=False,
            min_region_size=0,
            hex_stack=large_hex_stack,
            num_colors=5  # Less than hex_stack count
        )

        assert isinstance(result, tuple)
        assert len(result) == 1

        # The result should use optimal color selection
        result_pil = self.node.tensor_to_pil(result[0])
        unique_colors = len(set(result_pil.getdata()))
        assert unique_colors <= 5


# Edge case and regression tests
class TestRegressionCases:
    """Regression tests for specific bug fixes and edge cases"""

    def setup_method(self):
        self.node = EnhancedPaintByNumbersNode()

    def test_division_by_zero_protection(self):
        """Ensure no division by zero in color selection algorithms"""
        # Create image that might cause division issues
        edge_case_img = Image.new('RGB', (10, 10), color='black')

        # This should not raise any division by zero errors
        try:
            rgb_colors = np.array([[0, 0, 0], [255, 255, 255]])
            result = self.node.select_optimal_colors_from_palette(edge_case_img, rgb_colors, 1)
            assert len(result) == 1
        except ZeroDivisionError:
            pytest.fail("Division by zero occurred in color selection")

    def test_empty_image_handling(self):
        """Test handling of very small images"""
        tiny_img = Image.new('RGB', (1, 1), color='red')
        tiny_tensor = self.node.pil_to_tensor(tiny_img)

        # Should not crash on 1x1 image
        result = self.node.process(
            original_image=tiny_tensor,
            blur_radius=0.0,
            color_intensity=1.0,
            noise_reduction_strength=0.0,
            bilateral_filter=False,
            morphological_cleanup=False,
            min_region_size=0,
            num_colors=2
        )

        assert isinstance(result, tuple)
        assert result[0].shape == tiny_tensor.shape

    def test_memory_efficiency_large_palette(self):
        """Test memory efficiency with large color palettes"""
        # Create large palette (should trigger sampling optimizations)
        large_palette = np.random.randint(0, 256, (100, 3), dtype=np.uint8)
        test_img = Image.new('RGB', (200, 200), color='white')

        # Should complete without memory issues
        try:
            result = self.node.select_optimal_colors_from_palette(test_img, large_palette, 10)
            assert len(result) == 10
        except MemoryError:
            pytest.fail("Memory error with large palette")


if __name__ == "__main__":
    # Simple test runner if executed directly
    import sys

    test_classes = [TestEnhancedPaintByNumbersNode, TestRegressionCases]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]

        for method_name in methods:
            total_tests += 1
            try:
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()

                method = getattr(instance, method_name)
                method()
                passed_tests += 1
                print(f"✓ {test_class.__name__}.{method_name}")

            except Exception as e:
                print(f"✗ {test_class.__name__}.{method_name}: {str(e)}")

    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    sys.exit(0 if passed_tests == total_tests else 1)
