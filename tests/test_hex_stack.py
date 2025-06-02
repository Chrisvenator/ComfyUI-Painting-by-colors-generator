import re

import numpy as np
import torch
from PIL import Image


class TestHexStackNodeInputTypes:
    """Test the INPUT_TYPES class method"""

    def test_input_types_structure(self, hex_stack_node):
        """Test that INPUT_TYPES returns the correct structure"""
        input_types = hex_stack_node.INPUT_TYPES()

        assert "required" in input_types
        assert "hex_colors" in input_types["required"]
        assert "show_preview" in input_types["required"]

        # Check hex_colors configuration
        hex_config = input_types["required"]["hex_colors"]
        assert hex_config[0] == "STRING"
        assert hex_config[1]["multiline"] is True
        assert "default" in hex_config[1]

        # Check show_preview configuration
        preview_config = input_types["required"]["show_preview"]
        assert preview_config[0] == "BOOLEAN"
        assert preview_config[1]["default"] is True


class TestHexStackNodeParsing:
    """Test hex color parsing functionality"""

    def test_parse_valid_hex_colors(self, hex_stack_node, sample_hex_colors, valid_hex_colors, expected_rgb_colors):
        """Test parsing of mixed valid hex colors"""
        hex_list, rgb_array = hex_stack_node.parse_hex_colors(sample_hex_colors)

        assert len(hex_list) == len(valid_hex_colors)
        assert hex_list == valid_hex_colors
        np.testing.assert_array_equal(rgb_array, expected_rgb_colors)

    def test_parse_empty_string(self, hex_stack_node, empty_hex_string):
        """Test parsing empty or whitespace-only strings"""
        hex_list, rgb_array = hex_stack_node.parse_hex_colors(empty_hex_string)

        assert len(hex_list) == 0
        assert len(rgb_array) == 0
        assert rgb_array.dtype == np.float32

    def test_parse_single_color(self, hex_stack_node, single_hex_color):
        """Test parsing a single hex color"""
        hex_list, rgb_array = hex_stack_node.parse_hex_colors(single_hex_color)

        assert len(hex_list) == 1
        assert hex_list[0] == "#FF0000"
        expected_rgb = np.array([[255, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(rgb_array, expected_rgb)

    def test_case_insensitive_parsing(self, hex_stack_node):
        """Test that hex parsing is case insensitive"""
        mixed_case = "#aAbBcCdDeEfF\n#123ABC\n#abc123"
        hex_list, rgb_array = hex_stack_node.parse_hex_colors(mixed_case)

        assert len(hex_list) == 3
        assert hex_list[0] == "#AABBCCDDEEFF"  # Should be normalized to uppercase
        assert hex_list[1] == "#123ABC"
        assert hex_list[2] == "#ABC123"

    def test_three_digit_hex_expansion(self, hex_stack_node):
        """Test that 3-digit hex codes are properly expanded"""
        short_hex = "#ABC\n#123\n#F0A"
        hex_list, rgb_array = hex_stack_node.parse_hex_colors(short_hex)

        expected_hex = ["#AABBCC", "#112233", "#FF00AA"]
        expected_rgb = np.array([
            [170, 187, 204],  # #ABC -> #AABBCC
            [17, 34, 51],  # #123 -> #112233
            [255, 0, 170]  # #F0A -> #FF00AA
        ], dtype=np.float32)

        assert hex_list == expected_hex
        np.testing.assert_array_equal(rgb_array, expected_rgb)

    def test_hex_without_hash_prefix(self, hex_stack_node):
        """Test parsing hex colors without # prefix"""
        no_hash = "FFFFFF\nF6D300\nABC"
        hex_list, rgb_array = hex_stack_node.parse_hex_colors(no_hash)

        expected_hex = ["#FFFFFF", "#F6D300", "#AABBCC"]
        assert hex_list == expected_hex

    def test_invalid_hex_colors_ignored(self, hex_stack_node):
        """Test that invalid hex colors are ignored"""
        invalid_hex = """
        #FFFFFF
        #GGGGGG
        invalid_color
        #12345
        #1234567
        #
        GGG
        123
        #F6D300
        """
        hex_list, rgb_array = hex_stack_node.parse_hex_colors(invalid_hex)

        # Only valid colors should be returned
        expected_hex = ["#FFFFFF", "#F6D300"]
        assert hex_list == expected_hex
        assert len(rgb_array) == 2

    def test_mixed_valid_invalid_lines(self, hex_stack_node):
        """Test parsing lines with both valid and invalid content"""
        mixed_content = """
        Color 1: #FF0000 (Red)
        This line has no hex
        Background: #00FF00
        #0000FF - Blue color
        Not a color: GGGGGG
        """
        hex_list, rgb_array = hex_stack_node.parse_hex_colors(mixed_content)

        expected_hex = ["#FF0000", "#00FF00", "#0000FF"]
        expected_rgb = np.array([
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255]  # Blue
        ], dtype=np.float32)

        assert hex_list == expected_hex
        np.testing.assert_array_equal(rgb_array, expected_rgb)


class TestHexStackNodePreview:
    """Test color preview generation"""

    def test_create_color_preview_with_colors(self, hex_stack_node, valid_hex_colors, expected_rgb_colors):
        """Test preview creation with valid colors"""
        preview_img = hex_stack_node.create_color_preview(valid_hex_colors, expected_rgb_colors)

        assert isinstance(preview_img, Image.Image)
        assert preview_img.mode == "RGB"
        assert preview_img.size[0] > 0  # Width > 0
        assert preview_img.size[1] > 0  # Height > 0

    def test_create_color_preview_empty(self, hex_stack_node):
        """Test preview creation with no colors"""
        preview_img = hex_stack_node.create_color_preview([], np.array([]))

        assert isinstance(preview_img, Image.Image)
        assert preview_img.mode == "RGB"
        assert preview_img.size == (400, 100)  # Default empty image size

    def test_create_color_preview_single_color(self, hex_stack_node):
        """Test preview creation with single color"""
        hex_colors = ["#FF0000"]
        rgb_colors = np.array([[255, 0, 0]], dtype=np.float32)

        preview_img = hex_stack_node.create_color_preview(hex_colors, rgb_colors)

        assert isinstance(preview_img, Image.Image)
        assert preview_img.mode == "RGB"
        # Should be smaller than multi-color preview
        assert preview_img.size[1] < 300  # Height should be reasonable for 1 color

    def test_create_color_preview_many_colors(self, hex_stack_node):
        """Test preview creation with many colors (multiple rows)"""
        # Create 20 colors to test multiple rows
        hex_colors = [f"#{i:02X}{i:02X}{i:02X}" for i in range(20)]
        rgb_colors = np.array([[i, i, i] for i in range(20)], dtype=np.float32)

        preview_img = hex_stack_node.create_color_preview(hex_colors, rgb_colors)

        assert isinstance(preview_img, Image.Image)
        assert preview_img.mode == "RGB"
        # With 20 colors, should have multiple rows, so height should be larger
        assert preview_img.size[1] > 200


class TestHexStackNodeTensorConversion:
    """Test PIL to tensor conversion"""

    def test_pil_to_tensor_rgb(self, hex_stack_node):
        """Test conversion of RGB PIL image to tensor"""
        # Create a simple RGB image
        pil_img = Image.new('RGB', (100, 50), (255, 128, 0))
        tensor = hex_stack_node.pil_to_tensor(pil_img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 50, 100, 3)  # (batch, height, width, channels)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor[0, 0, 0], torch.tensor([1.0, 0.5, 0.0]), atol=1e-3)

    def test_pil_to_tensor_grayscale(self, hex_stack_node):
        """Test conversion of grayscale PIL image to tensor"""
        pil_img = Image.new('L', (50, 25), 128)
        tensor = hex_stack_node.pil_to_tensor(pil_img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 25, 50, 3)  # Should be converted to 3 channels
        assert tensor.dtype == torch.float32
        # All channels should have the same value for grayscale
        assert torch.allclose(tensor[0, 0, 0, 0], tensor[0, 0, 0, 1])
        assert torch.allclose(tensor[0, 0, 0, 1], tensor[0, 0, 0, 2])


class TestHexStackNodeIntegration:
    """Test the main create_hex_stack function (integration tests)"""

    def test_create_hex_stack_with_preview(self, hex_stack_node, sample_hex_colors):
        """Test complete hex stack creation with preview enabled"""
        hex_stack, preview_tensor = hex_stack_node.create_hex_stack(sample_hex_colors, True)

        # Test hex_stack structure
        assert isinstance(hex_stack, dict)
        assert "hex_colors" in hex_stack
        assert "rgb_colors" in hex_stack
        assert "count" in hex_stack

        # Test hex_stack content
        assert len(hex_stack["hex_colors"]) == hex_stack["count"]
        assert len(hex_stack["rgb_colors"]) == hex_stack["count"]
        assert hex_stack["count"] == 9  # Expected valid colors from sample
        assert isinstance(hex_stack["rgb_colors"], np.ndarray)
        assert hex_stack["rgb_colors"].dtype == np.float32

        # Test preview tensor
        assert isinstance(preview_tensor, torch.Tensor)
        assert preview_tensor.shape[0] == 1  # Batch size
        assert len(preview_tensor.shape) == 4  # (batch, height, width, channels)
        assert preview_tensor.shape[3] == 3  # RGB channels

    def test_create_hex_stack_without_preview(self, hex_stack_node, sample_hex_colors):
        """Test hex stack creation with preview disabled"""
        hex_stack, preview_tensor = hex_stack_node.create_hex_stack(sample_hex_colors, False)

        # Test hex_stack structure (should be same as with preview)
        assert isinstance(hex_stack, dict)
        assert hex_stack["count"] == 9

        # Preview should be minimal
        assert isinstance(preview_tensor, torch.Tensor)
        assert preview_tensor.shape == (1, 50, 200, 3)  # Expected minimal preview size

    def test_create_hex_stack_empty_input(self, hex_stack_node, empty_hex_string):
        """Test hex stack creation with empty input"""
        hex_stack, preview_tensor = hex_stack_node.create_hex_stack(empty_hex_string, True)

        assert hex_stack["count"] == 0
        assert len(hex_stack["hex_colors"]) == 0
        assert len(hex_stack["rgb_colors"]) == 0

        # Should still return a valid preview tensor
        assert isinstance(preview_tensor, torch.Tensor)

    def test_create_hex_stack_single_color(self, hex_stack_node, single_hex_color):
        """Test hex stack creation with single color"""
        hex_stack, preview_tensor = hex_stack_node.create_hex_stack(single_hex_color, True)

        assert hex_stack["count"] == 1
        assert hex_stack["hex_colors"] == ["#FF0000"]
        expected_rgb = np.array([[255, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(hex_stack["rgb_colors"], expected_rgb)

    def test_hex_stack_data_types(self, hex_stack_node, sample_hex_colors):
        """Test that hex stack returns correct data types"""
        hex_stack, preview_tensor = hex_stack_node.create_hex_stack(sample_hex_colors, True)

        # Check hex_colors type
        assert isinstance(hex_stack["hex_colors"], list)
        assert all(isinstance(color, str) for color in hex_stack["hex_colors"])

        # Check rgb_colors type and values
        assert isinstance(hex_stack["rgb_colors"], np.ndarray)
        assert hex_stack["rgb_colors"].dtype == np.float32
        assert np.all(hex_stack["rgb_colors"] >= 0)
        assert np.all(hex_stack["rgb_colors"] <= 255)

        # Check count type
        assert isinstance(hex_stack["count"], int)
        assert hex_stack["count"] >= 0


class TestHexStackNodeRegression:
    """Regression tests to ensure consistent behavior"""

    def test_default_colors_parsing(self, hex_stack_node):
        """Test that default colors from INPUT_TYPES parse correctly"""
        input_types = hex_stack_node.INPUT_TYPES()
        default_colors = input_types["required"]["hex_colors"][1]["default"]

        hex_list, rgb_array = hex_stack_node.parse_hex_colors(default_colors)

        # Should parse without errors
        assert len(hex_list) > 0
        assert len(rgb_array) > 0
        assert len(hex_list) == len(rgb_array)

        # All should be valid hex format
        hex_pattern = re.compile(r'^#[A-F0-9]{6}$')
        assert all(hex_pattern.match(color) for color in hex_list)

    def test_consistent_output_format(self, hex_stack_node):
        """Test that output format is consistent across different inputs"""
        test_inputs = [
            "#FF0000",
            "#FF0000\n#00FF00",
            "#FF0000\n#00FF00\n#0000FF\n#FFFF00",
            ""
        ]

        for hex_input in test_inputs:
            hex_stack, preview_tensor = hex_stack_node.create_hex_stack(hex_input, True)

            # Check structure consistency
            assert set(hex_stack.keys()) == {"hex_colors", "rgb_colors", "count"}
            assert isinstance(hex_stack["hex_colors"], list)
            assert isinstance(hex_stack["rgb_colors"], np.ndarray)
            assert isinstance(hex_stack["count"], int)
            assert isinstance(preview_tensor, torch.Tensor)

            # Check size consistency
            assert len(hex_stack["hex_colors"]) == hex_stack["count"]
            assert len(hex_stack["rgb_colors"]) == hex_stack["count"]


class TestHexStackNodePerformance:
    """Performance and edge case tests"""

    def test_large_input_handling(self, hex_stack_node):
        """Test handling of large number of colors"""
        # Create 1000 colors
        large_input = "\n".join([f"#{i:06X}" for i in range(1000)])

        hex_stack, preview_tensor = hex_stack_node.create_hex_stack(large_input, True)

        assert hex_stack["count"] == 1000
        assert len(hex_stack["hex_colors"]) == 1000
        assert hex_stack["rgb_colors"].shape == (1000, 3)

        # Preview should still be generated without errors
        assert isinstance(preview_tensor, torch.Tensor)

    def test_malformed_input_robustness(self, hex_stack_node):
        """Test robustness against malformed input"""
        malformed_inputs = [
            "##FF0000",
            "#FF00GG",
            "Not a color at all",
            "#" * 100,
            "\x00\x01\x02",  # Binary data
            "🎨#FF0000🎨",  # Unicode
        ]

        combined_input = "\n".join(malformed_inputs + ["#FF0000"])  # Add one valid
        hex_stack, preview_tensor = hex_stack_node.create_hex_stack(combined_input, True)

        # Should only parse the valid color
        assert hex_stack["count"] == 1
        assert hex_stack["hex_colors"] == ["#FF0000"]
