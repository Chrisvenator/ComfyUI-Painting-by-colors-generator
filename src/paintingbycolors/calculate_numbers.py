import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from scipy import ndimage


class ImprovedPaintByNumbersTemplateNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preprocessed_image": ("IMAGE",),
                "lineart_image": ("IMAGE",),
                "num_colors": ("INT", {
                    "default": 20,
                    "min": 2,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "font_size": ("INT", {
                    "default": 14,
                    "min": 2,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "min_region_size": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
                "numbers_density": ("FLOAT", {
                    "default": 0.000,
                    "min": 0.0,
                    "max": 1,
                    "step": 0.0005,
                    "display": "number"
                }),
                "color_merge_threshold": ("FLOAT", {
                    "default": 5.0,
                    "min": 5.0,
                    "max": 50.0,
                    "step": 2.5,
                    "display": "number"
                }),
            },
            "optional": {
                "hex_stack": ("HEX_STACK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("numbers_image", "color_palette")
    FUNCTION = "create_template"
    CATEGORY = "image/artistic"

    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        tensor = torch.clamp(tensor, 0, 1)
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_image)

    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor format"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        if len(np_image.shape) == 2:
            np_image = np.stack([np_image] * 3, axis=2)
        tensor = torch.from_numpy(np_image).unsqueeze(0)
        return tensor

    def get_colors_from_hex_stack(self, hex_stack, num_colors):
        """Extract colors from hex stack, limited to num_colors"""
        if hex_stack is None or 'rgb_colors' not in hex_stack:
            return None

        rgb_colors = hex_stack['rgb_colors']
        if len(rgb_colors) == 0:
            return None

        # Limit to requested number of colors
        limited_colors = rgb_colors[:num_colors]

        # Sort by brightness for consistency
        brightness = np.mean(limited_colors, axis=1)
        sorted_indices = np.argsort(brightness)

        return limited_colors[sorted_indices]

    def extract_dominant_colors(self, paint_by_numbers_pil, num_colors, merge_threshold, hex_stack=None):
        """Extract dominant colors using K-means clustering or hex stack"""

        # Try to use hex stack first
        if hex_stack is not None:
            hex_colors = self.get_colors_from_hex_stack(hex_stack, num_colors)
            if hex_colors is not None:
                print(f"Using {len(hex_colors)} colors from hex stack")
                return hex_colors

        # Fall back to original method
        img_array = np.array(paint_by_numbers_pil)
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        unique_pixels = np.unique(pixels, axis=0)

        if len(unique_pixels) < num_colors:
            final_colors = unique_pixels
        else:
            # Initial clustering
            initial_clusters = min(num_colors * 2, len(unique_pixels))
            kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init=10)
            kmeans.fit(unique_pixels)
            initial_colors = kmeans.cluster_centers_

            # Merge similar colors
            merged_colors = []
            used = np.zeros(len(initial_colors), dtype=bool)

            for i, color in enumerate(initial_colors):
                if used[i] or len(merged_colors) >= num_colors:
                    continue
                similar_colors = [color]
                used[i] = True

                for j, other_color in enumerate(initial_colors):
                    if used[j]:
                        continue
                    distance = np.sqrt(np.sum((color - other_color) ** 2))
                    if distance <= merge_threshold:
                        similar_colors.append(other_color)
                        used[j] = True

                merged_colors.append(np.mean(similar_colors, axis=0))

            final_colors = np.array(merged_colors[:num_colors])

        # Sort by brightness
        brightness = np.mean(final_colors, axis=1)
        sorted_indices = np.argsort(brightness)
        return final_colors[sorted_indices]

    def create_color_map(self, paint_by_numbers_pil, colors):
        """Map each pixel to closest color"""
        img_array = np.array(paint_by_numbers_pil)
        h, w, c = img_array.shape
        pixels = img_array.reshape(-1, 3).astype(np.float32)

        color_indices = np.zeros(len(pixels), dtype=np.int32)
        for i, pixel in enumerate(pixels):
            distances = np.sqrt(np.sum((colors - pixel) ** 2, axis=1))
            color_indices[i] = np.argmin(distances) + 1  # 1-indexed

        return color_indices.reshape(h, w)

    def create_line_mask(self, lineart_pil, paint_by_numbers_size):
        """Create mask for line areas"""
        if lineart_pil.mode != 'L':
            lineart_gray = lineart_pil.convert('L')
        else:
            lineart_gray = lineart_pil

        if lineart_gray.size != paint_by_numbers_size:
            lineart_gray = lineart_gray.resize(paint_by_numbers_size, Image.Resampling.LANCZOS)

        lineart_np = np.array(lineart_gray)
        line_mask = lineart_np < 128

        # Dilate to create exclusion zone
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(line_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    def find_regions(self, color_map, line_mask):
        """Find connected regions for each color"""
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        regions_by_color = {}

        for color_num in np.unique(color_map):
            if color_num == 0:
                continue

            color_mask = (color_map == color_num).astype(np.uint8)
            clean_mask = color_mask.copy()
            clean_mask[line_mask] = 0

            labeled, num_regions = ndimage.label(clean_mask, structure=structure)
            if num_regions > 0:
                regions_by_color[color_num] = labeled

        return regions_by_color

    def place_numbers(self, regions_by_color, size, numbers_density, min_region_size, line_mask, font_size):
        """Place numbers in regions"""
        numbers_img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(numbers_img)

        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        for color_num, labeled_regions in regions_by_color.items():
            num_regions = np.max(labeled_regions)

            for region_id in range(1, num_regions + 1):
                region_mask = (labeled_regions == region_id)
                coords = np.where(region_mask)

                if len(coords[0]) < min_region_size:
                    continue

                y_coords, x_coords = coords
                area = len(x_coords)

                # Calculate number of numbers to place
                if numbers_density == 0:
                    target_count = 1
                else:
                    target_count = max(1, int(area * numbers_density))

                # Find valid positions (not on lines)
                valid_coords = [(x, y) for x, y in zip(x_coords, y_coords)
                                if not line_mask[y, x]]

                if not valid_coords:
                    continue

                positions = []
                if target_count == 1:
                    # Place one number at centroid
                    centroid_x = int(np.mean([x for x, y in valid_coords]))
                    centroid_y = int(np.mean([y for x, y in valid_coords]))
                    best_pos = min(valid_coords,
                                   key=lambda p: (p[0] - centroid_x) ** 2 + (p[1] - centroid_y) ** 2)
                    positions.append(best_pos)
                else:
                    # Place multiple numbers with spacing
                    spacing = max(20, int(np.sqrt(area / target_count)))
                    for _ in range(target_count * 3):  # Try multiple times
                        if len(positions) >= target_count:
                            break
                        candidate = valid_coords[np.random.randint(len(valid_coords))]
                        if all(np.sqrt((candidate[0] - p[0]) ** 2 + (candidate[1] - p[1]) ** 2) >= spacing
                               for p in positions):
                            positions.append(candidate)

                # Draw numbers
                for x, y in positions:
                    text = str(color_num)
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    text_x = x - text_width // 2
                    text_y = y - text_height // 2

                    # White outline
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                draw.text((text_x + dx, text_y + dy), text,
                                          fill=(255, 255, 255), font=font)
                    # Black text
                    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

        return numbers_img

    def create_color_palette_chart(self, colors, hex_stack=None):
        """Create color palette chart with hex colors if available"""
        swatch_width, swatch_height = 120, 60
        margin = 20
        colors_per_row = 4
        num_rows = (len(colors) + colors_per_row - 1) // colors_per_row

        chart_width = colors_per_row * swatch_width + (colors_per_row + 1) * margin
        chart_height = num_rows * (swatch_height + 80) + margin * 2 + 50

        chart = Image.new('RGB', (chart_width, chart_height), (240, 240, 240))
        draw = ImageDraw.Draw(chart)

        # Load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            number_font = ImageFont.truetype("arial.ttf", 20)
            info_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = number_font = info_font = ImageFont.load_default()

        # Title
        title = "Paint by Numbers Color Guide"
        if hex_stack is not None:
            title += " (From Hex Stack)"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_x = (chart_width - (title_bbox[2] - title_bbox[0])) // 2
        draw.text((title_x, 10), title, fill=(0, 0, 0), font=title_font)

        # Get hex colors if available
        hex_colors = None
        if hex_stack is not None and 'hex_colors' in hex_stack:
            hex_colors = hex_stack['hex_colors'][:len(colors)]

        # Color swatches
        for i, color in enumerate(colors):
            row, col = i // colors_per_row, i % colors_per_row
            x = margin + col * (swatch_width + margin)
            y = 50 + row * (swatch_height + 80)

            rgb = tuple(color.astype(int))
            draw.rectangle([x, y, x + swatch_width, y + swatch_height],
                           fill=rgb, outline=(0, 0, 0), width=2)

            # Number on swatch
            number = str(i + 1)
            text_bbox = draw.textbbox((0, 0), number, font=number_font)
            text_x = x + (swatch_width - (text_bbox[2] - text_bbox[0])) // 2
            text_y = y + (swatch_height - (text_bbox[3] - text_bbox[1])) // 2
            text_color = (255, 255, 255) if sum(rgb) / 3 < 128 else (0, 0, 0)
            draw.text((text_x, text_y), number, fill=text_color, font=number_font)

            # Color info
            if hex_colors is not None and i < len(hex_colors):
                hex_color = hex_colors[i]
                info_text = f"RGB: {rgb}\n{hex_color}"
            else:
                hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
                info_text = f"RGB: {rgb}\n{hex_color}"

            draw.text((x, y + swatch_height + 5), info_text, fill=(0, 0, 0), font=info_font)

        return chart

    def create_template(self, paint_by_numbers_image, lineart_image, num_colors,
                        font_size, min_region_size, numbers_density, color_merge_threshold,
                        hex_stack=None):
        """Main function"""
        # Convert inputs
        paint_by_numbers_pil = self.tensor_to_pil(paint_by_numbers_image)
        lineart_pil = self.tensor_to_pil(lineart_image)

        if paint_by_numbers_pil.mode != 'RGB':
            paint_by_numbers_pil = paint_by_numbers_pil.convert('RGB')

        # Extract colors and create maps
        colors = self.extract_dominant_colors(paint_by_numbers_pil, num_colors, color_merge_threshold, hex_stack)
        color_map = self.create_color_map(paint_by_numbers_pil, colors)
        line_mask = self.create_line_mask(lineart_pil, paint_by_numbers_pil.size)
        regions_by_color = self.find_regions(color_map, line_mask)

        # Create outputs
        numbers_only = self.place_numbers(regions_by_color, paint_by_numbers_pil.size,
                                          numbers_density, min_region_size, line_mask, font_size)
        palette_chart = self.create_color_palette_chart(colors, hex_stack)

        return self.pil_to_tensor(numbers_only), self.pil_to_tensor(palette_chart)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PaintByNumbersTemplateNode": ImprovedPaintByNumbersTemplateNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaintByNumbersTemplateNode": "Improved Paint by Numbers Template"
}
