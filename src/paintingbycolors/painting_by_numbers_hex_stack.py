import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re


class HexStackNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hex_colors": ("STRING", {
                    "multiline": True,
                    "default": "#FF0000\n#00FF00\n#0000FF\n#FFFF00\n#FF00FF\n#00FFFF\n#FFA500\n#800080\n#FFC0CB\n#A52A2A\n#808080\n#000000",
                    "display": "text"
                }),
                "show_preview": ("BOOLEAN", {
                    "default": True,
                    "display": "checkbox"
                }),
            }
        }

    RETURN_TYPES = ("HEX_STACK", "IMAGE")
    RETURN_NAMES = ("hex_stack", "color_preview")
    FUNCTION = "create_hex_stack"
    CATEGORY = "image/artistic"

    def parse_hex_colors(self, hex_string):
        """Parse hex colors from string input"""
        # Split by lines and clean up
        lines = hex_string.strip().split('\n')
        hex_colors = []
        rgb_colors = []

        # Regex pattern for hex colors
        hex_pattern = re.compile(r'#?([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Find hex color in the line
            match = hex_pattern.search(line)
            if match:
                hex_color = match.group(1)

                # Convert 3-digit hex to 6-digit
                if len(hex_color) == 3:
                    hex_color = ''.join([c * 2 for c in hex_color])

                # Add # prefix if not present
                formatted_hex = f"#{hex_color.upper()}"

                # Convert to RGB
                try:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)

                    hex_colors.append(formatted_hex)
                    rgb_colors.append([r, g, b])
                except ValueError:
                    continue

        return hex_colors, np.array(rgb_colors, dtype=np.float32)

    def create_color_preview(self, hex_colors, rgb_colors):
        """Create a visual preview of the color stack"""
        if len(hex_colors) == 0:
            # Return empty white image if no colors
            img = Image.new('RGB', (400, 100), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            draw.text((10, 40), "No valid colors found", fill=(0, 0, 0), font=font)
            return img

        # Calculate dimensions
        colors_per_row = min(8, len(hex_colors))
        num_rows = (len(hex_colors) + colors_per_row - 1) // colors_per_row

        swatch_size = 60
        margin = 10
        text_height = 20

        img_width = colors_per_row * (swatch_size + margin) + margin
        img_height = num_rows * (swatch_size + text_height + margin) + margin + 40

        # Create image
        img = Image.new('RGB', (img_width, img_height), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 18)
            hex_font = ImageFont.truetype("arial.ttf", 10)
        except:
            title_font = hex_font = ImageFont.load_default()

        # Title
        title = f"Hex Color Stack ({len(hex_colors)} colors)"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_x = (img_width - (title_bbox[2] - title_bbox[0])) // 2
        draw.text((title_x, 10), title, fill=(0, 0, 0), font=title_font)

        # Draw color swatches
        for i, (hex_color, rgb_color) in enumerate(zip(hex_colors, rgb_colors)):
            row = i // colors_per_row
            col = i % colors_per_row

            x = margin + col * (swatch_size + margin)
            y = 40 + row * (swatch_size + text_height + margin)

            # Draw color swatch
            rgb_tuple = tuple(rgb_color.astype(int))
            draw.rectangle([x, y, x + swatch_size, y + swatch_size],
                           fill=rgb_tuple, outline=(0, 0, 0), width=1)

            # Draw hex label
            text_color = (255, 255, 255) if sum(rgb_tuple) / 3 < 128 else (0, 0, 0)
            text_bbox = draw.textbbox((0, 0), hex_color, font=hex_font)
            text_x = x + (swatch_size - (text_bbox[2] - text_bbox[0])) // 2
            text_y = y + swatch_size + 2
            draw.text((text_x, text_y), hex_color, fill=(0, 0, 0), font=hex_font)

        return img

    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor format"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        if len(np_image.shape) == 2:
            np_image = np.stack([np_image] * 3, axis=2)
        tensor = torch.from_numpy(np_image).unsqueeze(0)
        return tensor

    def create_hex_stack(self, hex_colors, show_preview):
        """Create hex stack data structure"""
        # Parse the hex colors
        hex_list, rgb_array = self.parse_hex_colors(hex_colors)

        # Create hex stack data structure
        hex_stack = {
            'hex_colors': hex_list,
            'rgb_colors': rgb_array,
            'count': len(hex_list)
        }

        # Create preview image
        if show_preview:
            preview_img = self.create_color_preview(hex_list, rgb_array)
        else:
            # Create minimal preview
            preview_img = Image.new('RGB', (200, 50), (240, 240, 240))
            draw = ImageDraw.Draw(preview_img)
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            draw.text((10, 15), f"{len(hex_list)} colors loaded", fill=(0, 0, 0), font=font)

        preview_tensor = self.pil_to_tensor(preview_img)

        return (hex_stack, preview_tensor)


# Node registration
NODE_CLASS_MAPPINGS = {
    "HexStackNode": HexStackNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HexStackNode": "Hex Color Stack"
}
