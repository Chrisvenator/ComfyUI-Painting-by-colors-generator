import pytest
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from unittest.mock import Mock, patch

# Assuming the module structure based on your imports
from calculate_numbers import ImprovedPaintByNumbersTemplateNode


class TestImprovedPaintByNumbersTemplateNode:

    @pytest.fixture
    def node(self):
        """Create a fresh node instance for each test."""
        return ImprovedPaintByNumbersTemplateNode()

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample 3-channel RGB tensor (ComfyUI format)."""
        # Create a simple 4-color test image: 100x100, with 4 distinct quadrants
        img_array = np.zeros((100, 100, 3), dtype=np.float32)
        # Top-left: Red
        img_array[:50, :50] = [1.0, 0.0, 0.0]
        # Top-right: Green
        img_array[:50, 50:] = [0.0, 1.0, 0.0]
        # Bottom-left: Blue
        img_array[50:, :50] = [0.0, 0.0, 1.0]
        # Bottom-right: White
        img_array[50:, 50:] = [1.0, 1.0, 1.0]

        return torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension

    @pytest.fixture
    def sample_lineart_tensor(self):
        """Create a simple lineart tensor with some black lines."""
        img_array = np.ones((100, 100, 3), dtype=np.float32)  # White background
        # Add some black lines
        img_array[49:51, :] = 0.0  # Horizontal line
        img_array[:, 49:51] = 0.0  # Vertical line

        return torch.from_numpy(img_array).unsqueeze(0)

    @pytest.fixture
    def sample_hex_stack(self):
        """Create a sample hex stack for testing."""
        return {
            'hex_colors': ['#FF0000', '#00FF00', '#0000FF', '#FFFFFF'],
            'rgb_colors': np.array([
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 255]
            ])
        }

    def test_input_types(self, node):
        """Test that INPUT_TYPES returns correct structure."""
        input_types = node.INPUT_TYPES()

        assert "required" in input_types
        assert "optional" in input_types

        required = input_types["required"]
        assert "preprocessed_image" in required
        assert "lineart_image" in required
        assert "num_colors" in required
        assert "font_size" in required
        assert "min_region_size" in required
        assert "numbers_density" in required
        assert "color_merge_threshold" in required

        # Check parameter constraints
        num_colors_config = required["num_colors"][1]
        assert num_colors_config["min"] == 2
        assert num_colors_config["max"] == 1000
        assert num_colors_config["default"] == 20

    def test_tensor_to_pil_conversion(self, node, sample_tensor):
        """Test tensor to PIL conversion preserves dimensions and value range."""
        pil_img = node.tensor_to_pil(sample_tensor)

        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (100, 100)
        assert pil_img.mode == "RGB"

        # Check that values are in correct range (0-255)
        img_array = np.array(pil_img)
        assert img_array.min() >= 0
        assert img_array.max() <= 255
        assert img_array.dtype == np.uint8

    def test_pil_to_tensor_conversion(self, node):
        """Test PIL to tensor conversion preserves dimensions and normalizes values."""
        # Create a PIL image
        pil_img = Image.new("RGB", (50, 50), (128, 64, 192))

        tensor = node.pil_to_tensor(pil_img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 50, 50, 3)  # Batch, H, W, C
        assert tensor.dtype == torch.float32
        assert torch.all(tensor >= 0.0)
        assert torch.all(tensor <= 1.0)

    def test_tensor_pil_roundtrip(self, node, sample_tensor):
        """Test that tensor->PIL->tensor roundtrip preserves data."""
        original_shape = sample_tensor.shape

        # Convert to PIL and back
        pil_img = node.tensor_to_pil(sample_tensor)
        recovered_tensor = node.pil_to_tensor(pil_img)

        assert recovered_tensor.shape == original_shape

        # Values should be very close (allowing for minor floating point errors)
        diff = torch.abs(sample_tensor - recovered_tensor)
        assert torch.max(diff) < 0.01  # Less than 1% difference

    def test_get_colors_from_hex_stack_valid(self, node, sample_hex_stack):
        """Test extracting colors from valid hex stack."""
        colors = node.get_colors_from_hex_stack(sample_hex_stack, 4)

        assert colors is not None
        assert colors.shape == (4, 3)
        assert colors.dtype == np.int64 or colors.dtype == np.int32

        # Colors should be sorted by brightness
        brightness = np.mean(colors, axis=1)
        assert np.all(brightness[:-1] <= brightness[1:])  # Non-decreasing

    def test_get_colors_from_hex_stack_limited(self, node, sample_hex_stack):
        """Test limiting colors from hex stack."""
        colors = node.get_colors_from_hex_stack(sample_hex_stack, 2)

        assert colors is not None
        assert colors.shape == (2, 3)

    def test_get_colors_from_hex_stack_none(self, node):
        """Test handling of None hex stack."""
        colors = node.get_colors_from_hex_stack(None, 4)
        assert colors is None

    def test_get_colors_from_hex_stack_empty(self, node):
        """Test handling of empty hex stack."""
        empty_stack = {'rgb_colors': []}
        colors = node.get_colors_from_hex_stack(empty_stack, 4)
        assert colors is None

    def test_extract_dominant_colors_with_hex_stack(self, node, sample_tensor, sample_hex_stack):
        """Test color extraction using hex stack."""
        pil_img = node.tensor_to_pil(sample_tensor)

        colors = node.extract_dominant_colors(pil_img, 4, 5.0, sample_hex_stack)

        assert colors is not None
        assert colors.shape == (4, 3)
        # Should use hex stack colors
        assert np.array_equal(colors, sample_hex_stack['rgb_colors'])

    def test_extract_dominant_colors_without_hex_stack(self, node, sample_tensor):
        """Test color extraction using K-means clustering."""
        pil_img = node.tensor_to_pil(sample_tensor)

        colors = node.extract_dominant_colors(pil_img, 4, 5.0, None)

        assert colors is not None
        assert colors.shape == (4, 3)
        assert colors.dtype == np.float32

        # Should find approximately the 4 colors we created
        # (allowing for some clustering variance)
        assert len(colors) == 4

    def test_create_color_map(self, node, sample_tensor):
        """Test color mapping creates proper indices."""
        pil_img = node.tensor_to_pil(sample_tensor)
        colors = node.extract_dominant_colors(pil_img, 4, 5.0, None)

        color_map = node.create_color_map(pil_img, colors)

        assert color_map.shape == (100, 100)
        assert color_map.dtype == np.int32

        # Should have 4 different color indices (1-4, since 1-indexed)
        unique_indices = np.unique(color_map)
        assert len(unique_indices) <= 4
        assert np.all(unique_indices >= 1)
        assert np.all(unique_indices <= 4)

    def test_create_line_mask(self, node, sample_lineart_tensor, sample_tensor):
        """Test line mask creation."""
        lineart_pil = node.tensor_to_pil(sample_lineart_tensor)
        paint_pil = node.tensor_to_pil(sample_tensor)

        line_mask = node.create_line_mask(lineart_pil, paint_pil.size)

        assert line_mask.shape == (100, 100)
        assert line_mask.dtype == bool

        # Should have some True values where lines are
        assert np.any(line_mask)
        # Should have some False values where no lines are
        assert not np.all(line_mask)

    def test_find_regions(self, node, sample_tensor, sample_lineart_tensor):
        """Test region finding functionality."""
        pil_img = node.tensor_to_pil(sample_tensor)
        lineart_pil = node.tensor_to_pil(sample_lineart_tensor)

        colors = node.extract_dominant_colors(pil_img, 4, 5.0, None)
        color_map = node.create_color_map(pil_img, colors)
        line_mask = node.create_line_mask(lineart_pil, pil_img.size)

        regions = node.find_regions(color_map, line_mask)

        assert isinstance(regions, dict)
        # Should have some regions for our 4-color image
        assert len(regions) > 0

        # Each value should be a labeled array
        for color_num, labeled_regions in regions.items():
            assert isinstance(labeled_regions, np.ndarray)
            assert labeled_regions.shape == (100, 100)

    @patch('PIL.ImageFont.truetype')
    def test_place_numbers(self, mock_font, node, sample_tensor, sample_lineart_tensor):
        """Test number placement on image."""
        # Mock font to avoid file system dependency
        mock_font.return_value = Mock()

        pil_img = node.tensor_to_pil(sample_tensor)
        lineart_pil = node.tensor_to_pil(sample_lineart_tensor)

        colors = node.extract_dominant_colors(pil_img, 4, 5.0, None)
        color_map = node.create_color_map(pil_img, colors)
        line_mask = node.create_line_mask(lineart_pil, pil_img.size)
        regions = node.find_regions(color_map, line_mask)

        numbers_img = node.place_numbers(
            regions, pil_img.size, 0.001, 50, line_mask, 14
        )

        assert isinstance(numbers_img, Image.Image)
        assert numbers_img.size == pil_img.size
        assert numbers_img.mode == "RGB"

        # Should not be pure white (numbers should be placed)
        img_array = np.array(numbers_img)
        is_pure_white = np.all(img_array == 255)
        assert not is_pure_white, "Numbers image should contain some non-white pixels"

    def test_create_color_palette_chart(self, node):
        """Test color palette chart creation."""
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.float32)

        chart = node.create_color_palette_chart(colors)

        assert isinstance(chart, Image.Image)
        assert chart.mode == "RGB"
        assert chart.size[0] > 0
        assert chart.size[1] > 0

        # Should not be a single color
        img_array = np.array(chart)
        assert len(np.unique(img_array.reshape(-1, 3), axis=0)) > 1

    def test_create_color_palette_chart_with_hex_stack(self, node, sample_hex_stack):
        """Test color palette chart creation with hex stack."""
        colors = sample_hex_stack['rgb_colors'].astype(np.float32)

        chart = node.create_color_palette_chart(colors, sample_hex_stack)

        assert isinstance(chart, Image.Image)
        assert chart.mode == "RGB"
        # Chart should be larger when hex info is included
        assert chart.size[0] > 200
        assert chart.size[1] > 200

    def test_create_template_main_function(self, node, sample_tensor, sample_lineart_tensor):
        """Test the main create_template function integration."""
        numbers_img, palette_chart = node.create_template(
            preprocessed_image=sample_tensor,
            lineart_image=sample_lineart_tensor,
            num_colors=4,
            font_size=14,
            min_region_size=50,
            numbers_density=0.001,
            color_merge_threshold=5.0,
            hex_stack=None
        )

        # Should return exactly two tensors
        assert isinstance(numbers_img, torch.Tensor)
        assert isinstance(palette_chart, torch.Tensor)

        # Numbers image should have same dimensions as input
        assert numbers_img.shape[1:3] == sample_tensor.shape[1:3]  # H, W match
        assert numbers_img.shape[3] == 3  # RGB channels

        # Palette chart should be a valid tensor
        assert palette_chart.shape[3] == 3  # RGB channels
        assert palette_chart.shape[0] == 1  # Batch dimension

    def test_create_template_with_hex_stack(self, node, sample_tensor, sample_lineart_tensor, sample_hex_stack):
        """Test create_template with hex stack integration."""
        numbers_img, palette_chart = node.create_template(
            preprocessed_image=sample_tensor,
            lineart_image=sample_lineart_tensor,
            num_colors=4,
            font_size=14,
            min_region_size=50,
            numbers_density=0.001,
            color_merge_threshold=5.0,
            hex_stack=sample_hex_stack
        )

        assert isinstance(numbers_img, torch.Tensor)
        assert isinstance(palette_chart, torch.Tensor)

        # Both outputs should be valid tensors
        assert numbers_img.dim() == 4  # Batch, H, W, C
        assert palette_chart.dim() == 4

    def test_edge_case_single_color_image(self, node):
        """Test handling of single-color image."""
        # Create a solid red image
        img_array = np.full((50, 50, 3), [1.0, 0.0, 0.0], dtype=np.float32)
        tensor = torch.from_numpy(img_array).unsqueeze(0)

        lineart_array = np.ones((50, 50, 3), dtype=np.float32)
        lineart_tensor = torch.from_numpy(lineart_array).unsqueeze(0)

        # Should handle gracefully
        numbers_img, palette_chart = node.create_template(
            preprocessed_image=tensor,
            lineart_image=lineart_tensor,
            num_colors=4,
            font_size=14,
            min_region_size=10,
            numbers_density=0.001,
            color_merge_threshold=5.0
        )

        assert isinstance(numbers_img, torch.Tensor)
        assert isinstance(palette_chart, torch.Tensor)

    def test_performance_reasonable_time(self, node, sample_tensor, sample_lineart_tensor):
        """Smoke test: processing should complete in reasonable time."""
        import time

        start_time = time.time()

        numbers_img, palette_chart = node.create_template(
            preprocessed_image=sample_tensor,
            lineart_image=sample_lineart_tensor,
            num_colors=8,
            font_size=12,
            min_region_size=25,
            numbers_density=0.0005,
            color_merge_threshold=7.5
        )

        elapsed = time.time() - start_time

        # Should complete in under 5 seconds for small test image
        assert elapsed < 5.0, f"Processing took {elapsed:.2f}s, which is too slow"

        # Outputs should still be valid
        assert isinstance(numbers_img, torch.Tensor)
        assert isinstance(palette_chart, torch.Tensor)


class TestNodeRegistration:
    """Test node registration and metadata."""

    def test_node_class_mappings(self):
        """Test that node is properly registered."""
        from calculate_numbers import NODE_CLASS_MAPPINGS

        assert "PaintByNumbersTemplateNode" in NODE_CLASS_MAPPINGS
        assert NODE_CLASS_MAPPINGS["PaintByNumbersTemplateNode"] == ImprovedPaintByNumbersTemplateNode

    def test_node_display_name_mappings(self):
        """Test display name mapping."""
        from calculate_numbers import NODE_DISPLAY_NAME_MAPPINGS

        assert "PaintByNumbersTemplateNode" in NODE_DISPLAY_NAME_MAPPINGS
        assert isinstance(NODE_DISPLAY_NAME_MAPPINGS["PaintByNumbersTemplateNode"], str)

    def test_node_return_types(self):
        """Test that node declares correct return types."""
        node = ImprovedPaintByNumbersTemplateNode()

        assert hasattr(node, 'RETURN_TYPES')
        assert node.RETURN_TYPES == ("IMAGE", "IMAGE")

        assert hasattr(node, 'RETURN_NAMES')
        assert node.RETURN_NAMES == ("numbers_image", "color_palette")

        assert hasattr(node, 'FUNCTION')
        assert node.FUNCTION == "create_template"

        assert hasattr(node, 'CATEGORY')
        assert node.CATEGORY == "image/artistic"
