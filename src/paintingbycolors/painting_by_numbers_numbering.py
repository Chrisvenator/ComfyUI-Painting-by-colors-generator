import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from scipy import ndimage
import json
import math


class ImprovedPaintByNumbersTemplateNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "paint_by_numbers_image": ("IMAGE",),
                "lineart_image": ("IMAGE",),
                "num_colors": ("INT", {
                    "default": 12,
                    "min": 4,
                    "max": 30,
                    "step": 1,
                    "display": "number"
                }),
                "line_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "display": "number"
                }),
                "font_size": ("INT", {
                    "default": 14,
                    "min": 8,
                    "max": 32,
                    "step": 1,
                    "display": "number"
                }),
                "min_region_size": ("INT", {
                    "default": 100,
                    "min": 20,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
                "number_spacing": ("INT", {
                    "default": 25,
                    "min": 10,
                    "max": 80,
                    "step": 5,
                    "display": "number"
                }),
                "numbers_density": ("FLOAT", {
                    "default": 0.002,
                    "min": 0.0,
                    "max": 0.01,
                    "step": 0.0005,
                    "display": "number"
                }),
                "line_repair_iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "display": "number"
                }),
                "color_merge_threshold": ("FLOAT", {
                    "default": 15.0,
                    "min": 5.0,
                    "max": 50.0,
                    "step": 2.5,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("paint_by_numbers_template", "numbers_only", "color_palette_chart")
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

    def repair_lineart_advanced(self, lineart_pil, iterations=3):
        """Advanced line repair with multiple iterations and better gap closing"""
        if lineart_pil.mode != 'L':
            lineart_gray = lineart_pil.convert('L')
        else:
            lineart_gray = lineart_pil

        lineart_np = np.array(lineart_gray)

        # Create binary mask (lines are 1, background is 0)
        binary = (lineart_np < 128).astype(np.uint8)

        # Apply multiple repair operations with progressively larger kernels
        for i in range(iterations):
            # Close small gaps with increasing kernel size
            kernel_size = 3 + i
            kernel_close = np.ones((kernel_size, kernel_size), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

            # Connect nearby line segments
            kernel_dilate = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(binary, kernel_dilate, iterations=1)

            # Erode back to original thickness but keep connections
            kernel_erode = np.ones((2, 2), np.uint8)
            binary = cv2.erode(dilated, kernel_erode, iterations=1)

            # Remove very small noise
            if i == iterations - 1:  # Only on last iteration
                kernel_open = np.ones((2, 2), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # Convert back to image format (lines are dark)
        repaired = (1 - binary) * 255
        return Image.fromarray(repaired.astype(np.uint8), mode='L')

    def extract_dominant_colors(self, paint_by_numbers_pil, num_colors, merge_threshold):
        """Extract dominant colors using improved clustering - ONLY from paint_by_numbers_image"""
        print("Extracting colors from paint_by_numbers_image only...")

        img_array = np.array(paint_by_numbers_pil)
        pixels = img_array.reshape(-1, 3).astype(np.float32)

        # Remove duplicate pixels to speed up clustering
        unique_pixels = np.unique(pixels, axis=0)
        print(f"Processing {len(unique_pixels)} unique colors from {len(pixels)} total pixels")

        # Use K-means clustering to find dominant colors
        if len(unique_pixels) < num_colors:
            print(f"Warning: Image has fewer unique colors ({len(unique_pixels)}) than requested ({num_colors})")
            final_colors = unique_pixels
        else:
            # Initial clustering with more clusters to capture variations
            initial_clusters = min(num_colors * 2, len(unique_pixels))
            kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init=10, max_iter=300)
            kmeans.fit(unique_pixels)
            initial_colors = kmeans.cluster_centers_

            # Merge similar colors based on threshold
            merged_colors = []
            used = np.zeros(len(initial_colors), dtype=bool)

            for i, color in enumerate(initial_colors):
                if used[i]:
                    continue

                # Find all similar colors
                similar_colors = [color]
                used[i] = True

                for j, other_color in enumerate(initial_colors):
                    if used[j]:
                        continue

                    # Calculate Euclidean distance in RGB space
                    distance = np.sqrt(np.sum((color - other_color) ** 2))
                    if distance <= merge_threshold:
                        similar_colors.append(other_color)
                        used[j] = True

                # Average the similar colors
                merged_color = np.mean(similar_colors, axis=0)
                merged_colors.append(merged_color)

                if len(merged_colors) >= num_colors:
                    break

            final_colors = np.array(merged_colors[:num_colors])

        # Sort by brightness for consistent numbering (darkest to lightest)
        brightness = np.mean(final_colors, axis=1)
        sorted_indices = np.argsort(brightness)
        final_colors = final_colors[sorted_indices]

        print(f"Final color palette: {len(final_colors)} colors")
        for i, color in enumerate(final_colors):
            print(f"Color {i + 1}: RGB({int(color[0])}, {int(color[1])}, {int(color[2])})")

        return final_colors

    def create_color_map_improved(self, paint_by_numbers_pil, colors):
        """Create improved color mapping - assign each pixel to closest color"""
        print("Creating color map...")

        img_array = np.array(paint_by_numbers_pil)
        h, w, c = img_array.shape
        pixels = img_array.reshape(-1, 3).astype(np.float32)

        # For each pixel, find the closest color
        color_indices = np.zeros(len(pixels), dtype=np.int32)

        for i, pixel in enumerate(pixels):
            # Calculate distance to each color
            distances = np.sqrt(np.sum((colors - pixel) ** 2, axis=1))
            closest_color = np.argmin(distances)
            color_indices[i] = closest_color + 1  # 1-indexed

        # Reshape back to image dimensions
        color_map = color_indices.reshape(h, w)

        # Verify color distribution
        unique_values, counts = np.unique(color_map, return_counts=True)
        print(f"Color map distribution:")
        for val, count in zip(unique_values, counts):
            print(f"  Color {val}: {count} pixels ({count / len(pixels) * 100:.1f}%)")

        return color_map

    def create_line_exclusion_mask(self, lineart_pil, line_threshold, paint_by_numbers_size):
        """Create a mask where numbers should NOT be placed (on or near lines)"""
        if lineart_pil.mode != 'L':
            lineart_gray = lineart_pil.convert('L')
        else:
            lineart_gray = lineart_pil

        # Resize lineart if necessary
        if lineart_gray.size != paint_by_numbers_size:
            lineart_gray = lineart_gray.resize(paint_by_numbers_size, Image.Resampling.LANCZOS)

        lineart_np = np.array(lineart_gray)

        # Create line mask (True where lines are)
        line_mask = lineart_np < (line_threshold * 255)

        # Dilate the line mask to create exclusion zone around lines
        kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel for safety margin
        exclusion_mask = cv2.dilate(line_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        print(f"Line exclusion mask covers {np.sum(exclusion_mask)} pixels ({np.sum(exclusion_mask) / exclusion_mask.size * 100:.1f}%)")

        return exclusion_mask

    def find_connected_regions(self, color_map, exclusion_mask):
        """Find connected regions for each color, avoiding line areas"""
        # 4-connected structure (more conservative than 8-connected)
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        regions_by_color = {}
        unique_colors = np.unique(color_map)

        print(f"Processing {len(unique_colors)} unique color values: {unique_colors}")

        for color_num in unique_colors:
            if color_num == 0:  # Skip background if any
                continue

            # Create mask for this color
            color_mask = (color_map == color_num).astype(np.uint8)

            # Remove exclusion areas (lines) to separate regions properly
            clean_mask = color_mask.copy()
            clean_mask[exclusion_mask] = 0

            # Find connected components
            labeled, num_regions = ndimage.label(clean_mask, structure=structure)

            if num_regions > 0:
                regions_by_color[color_num] = {
                    'labeled': labeled,
                    'count': num_regions,
                    'original_mask': color_mask
                }
                print(f"Color {color_num}: {num_regions} regions found")
            else:
                print(f"Color {color_num}: No regions found after line exclusion")

        return regions_by_color

    def calculate_optimal_number_positions(self, region_mask, color_number, numbers_density,
                                           number_spacing, min_region_size, exclusion_mask):
        """Calculate optimal positions for numbers in a region, avoiding lines"""
        # Get region coordinates
        coords = np.where(region_mask > 0)
        if len(coords[0]) == 0:
            return []

        y_coords, x_coords = coords
        area = len(x_coords)

        if area < min_region_size:
            print(f"Color {color_number}: Region too small ({area} < {min_region_size})")
            return []

        # Calculate number of numbers to place
        if numbers_density == 0:
            # Special case: only one number per patch
            target_count = 1
            print(f"Color {color_number}: Density is 0, placing only 1 number per region")
        else:
            target_count = max(1, int(area * numbers_density))
            print(f"Color {color_number}: Target {target_count} numbers for area {area}")

        positions = []

        if target_count == 1:
            # Find the best central position avoiding exclusion zones
            region_coords = list(zip(x_coords, y_coords))

            # Filter out coordinates that are in exclusion zones
            valid_coords = [(x, y) for x, y in region_coords
                            if not exclusion_mask[y, x]]

            if not valid_coords:
                print(f"Color {color_number}: No valid positions found (all in exclusion zone)")
                return []

            # Find centroid of valid coordinates
            valid_x = [x for x, y in valid_coords]
            valid_y = [y for x, y in valid_coords]
            centroid_x = int(np.mean(valid_x))
            centroid_y = int(np.mean(valid_y))

            # Find the closest valid coordinate to the centroid
            min_dist = float('inf')
            best_pos = None
            for x, y in valid_coords:
                dist = (x - centroid_x) ** 2 + (y - centroid_y) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_pos = (x, y)

            if best_pos:
                positions.append(best_pos)
                print(f"Color {color_number}: Placed 1 number at {best_pos}")

        else:
            # Multiple numbers - use grid approach
            # Get bounding box
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            width = max_x - min_x + 1
            height = max_y - min_y + 1

            # Calculate grid size
            grid_size = int(np.sqrt(target_count))
            if grid_size < 1:
                grid_size = 1

            step_x = max(number_spacing, width // (grid_size + 1))
            step_y = max(number_spacing, height // (grid_size + 1))

            # Place numbers in grid pattern
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(positions) >= target_count:
                        break

                    # Calculate base position
                    base_x = min_x + (i + 1) * step_x
                    base_y = min_y + (j + 1) * step_y

                    # Add some randomness to avoid perfect grid
                    jitter_x = np.random.randint(-step_x // 4, step_x // 4 + 1)
                    jitter_y = np.random.randint(-step_y // 4, step_y // 4 + 1)

                    test_x = np.clip(base_x + jitter_x, min_x, max_x)
                    test_y = np.clip(base_y + jitter_y, min_y, max_y)

                    # Check if position is valid
                    if (0 <= test_y < region_mask.shape[0] and
                        0 <= test_x < region_mask.shape[1] and
                        region_mask[test_y, test_x] > 0 and
                        not exclusion_mask[test_y, test_x]):  # Not in exclusion zone

                        # Check minimum distance from existing positions
                        valid = True
                        for ex_x, ex_y in positions:
                            dist = np.sqrt((test_x - ex_x) ** 2 + (test_y - ex_y) ** 2)
                            if dist < number_spacing * 0.8:
                                valid = False
                                break

                        if valid:
                            positions.append((test_x, test_y))

            # Fill remaining positions randomly if needed
            attempts = 0
            max_attempts = min(500, area)  # Limit attempts

            while len(positions) < target_count and attempts < max_attempts:
                idx = np.random.randint(0, len(x_coords))
                test_x, test_y = x_coords[idx], y_coords[idx]

                # Check if position is valid (not in exclusion zone)
                if exclusion_mask[test_y, test_x]:
                    attempts += 1
                    continue

                # Check distance from existing positions
                valid = True
                for ex_x, ex_y in positions:
                    dist = np.sqrt((test_x - ex_x) ** 2 + (test_y - ex_y) ** 2)
                    if dist < number_spacing * 0.6:
                        valid = False
                        break

                if valid:
                    positions.append((test_x, test_y))

                attempts += 1

        print(f"Color {color_number}: Placed {len(positions)} numbers in region of area {area}")
        return positions

    def create_template_with_numbers(self, paint_by_numbers_pil, regions_by_color, colors,
                                     font_size, numbers_density, number_spacing, min_region_size, exclusion_mask):
        """Create the final template with numbers"""
        # Start with white background
        template = Image.new('RGB', paint_by_numbers_pil.size, (255, 255, 255))
        draw = ImageDraw.Draw(template)

        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()

        all_number_positions = []
        total_numbers_placed = 0

        # Process each color
        for color_num, region_data in regions_by_color.items():
            labeled_regions = region_data['labeled']
            num_regions = region_data['count']

            print(f"Processing color {color_num} with {num_regions} regions")

            # Process each region of this color
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled_regions == region_id)

                # Get positions for numbers in this region
                positions = self.calculate_optimal_number_positions(
                    region_mask, color_num, numbers_density,
                    number_spacing, min_region_size, exclusion_mask
                )

                # Draw numbers
                for x, y in positions:
                    text = str(color_num)

                    # Get text dimensions
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    text_x = x - text_width // 2
                    text_y = y - text_height // 2

                    # Draw white outline for better visibility
                    outline_width = 1
                    for dx in range(-outline_width, outline_width + 1):
                        for dy in range(-outline_width, outline_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text((text_x + dx, text_y + dy), text,
                                          fill=(255, 255, 255), font=font)

                    # Draw black text
                    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

                    all_number_positions.append({
                        'text': text,
                        'x': text_x,
                        'y': text_y,
                        'font': font
                    })
                    total_numbers_placed += 1

        print(f"Total numbers placed: {total_numbers_placed}")
        return template, all_number_positions

    def add_lines_to_template(self, template, lineart_pil, line_threshold):
        """Add line art to the template"""
        if lineart_pil.mode != 'L':
            lineart_gray = lineart_pil.convert('L')
        else:
            lineart_gray = lineart_pil

        # Resize if needed
        if lineart_gray.size != template.size:
            lineart_gray = lineart_gray.resize(template.size, Image.Resampling.LANCZOS)

        lineart_np = np.array(lineart_gray)
        template_np = np.array(template)

        # Add lines (dark pixels from lineart)
        line_mask = lineart_np < (line_threshold * 255)
        template_np[line_mask] = [0, 0, 0]

        return Image.fromarray(template_np)

    def create_numbers_only_image(self, size, number_positions):
        """Create an image with only the numbers"""
        numbers_img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(numbers_img)

        for pos in number_positions:
            draw.text((pos['x'], pos['y']), pos['text'],
                      fill=(0, 0, 0), font=pos['font'])

        return numbers_img

    def create_color_palette_chart(self, colors):
        """Create a visual color palette chart"""
        swatch_width = 120
        swatch_height = 60
        margin = 20
        text_margin = 10

        colors_per_row = 4
        num_colors = len(colors)
        num_rows = (num_colors + colors_per_row - 1) // colors_per_row

        chart_width = colors_per_row * swatch_width + (colors_per_row + 1) * margin
        chart_height = num_rows * (swatch_height + text_margin + 30) + margin * 2 + 50

        chart = Image.new('RGB', (chart_width, chart_height), (240, 240, 240))
        draw = ImageDraw.Draw(chart)

        # Load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            number_font = ImageFont.truetype("arial.ttf", 20)
            info_font = ImageFont.truetype("arial.ttf", 14)
        except:
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                number_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                info_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                try:
                    title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
                    number_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
                    info_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
                except:
                    title_font = number_font = info_font = ImageFont.load_default()

        # Draw title
        title = "Paint by Numbers Color Guide"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (chart_width - title_width) // 2
        draw.text((title_x, 10), title, fill=(0, 0, 0), font=title_font)

        # Draw color swatches
        start_y = 50
        for i, color in enumerate(colors):
            row = i // colors_per_row
            col = i % colors_per_row

            x = margin + col * (swatch_width + margin)
            y = start_y + row * (swatch_height + text_margin + 30)

            rgb = tuple(color.astype(int))
            draw.rectangle([x, y, x + swatch_width, y + swatch_height],
                           fill=rgb, outline=(0, 0, 0), width=2)

            # Draw number
            number = str(i + 1)
            text_bbox = draw.textbbox((0, 0), number, font=number_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = x + (swatch_width - text_width) // 2
            text_y = y + (swatch_height - text_height) // 2

            # Choose text color based on background brightness
            brightness = sum(rgb) / 3
            text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
            draw.text((text_x, text_y), number, fill=text_color, font=number_font)

            # Draw color info
            info_y = y + swatch_height + 5
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
            info_text = f"RGB: {rgb}\n{hex_color}"
            draw.text((x, info_y), info_text, fill=(0, 0, 0), font=info_font)

        return chart

    def create_template(self, paint_by_numbers_image, lineart_image, num_colors,
                        line_threshold, font_size, min_region_size, number_spacing,
                        numbers_density, line_repair_iterations, color_merge_threshold):
        """Main function to create the paint-by-numbers template"""

        print("Starting paint-by-numbers template creation...")
        print(f"Parameters: colors={num_colors}, density={numbers_density}, spacing={number_spacing}")

        # Convert inputs to PIL
        paint_by_numbers_pil = self.tensor_to_pil(paint_by_numbers_image)
        lineart_pil = self.tensor_to_pil(lineart_image)

        # Ensure RGB mode
        if paint_by_numbers_pil.mode != 'RGB':
            paint_by_numbers_pil = paint_by_numbers_pil.convert('RGB')

        print(f"Input image size: {paint_by_numbers_pil.size}")

        # Repair lineart
        print("Repairing lineart...")
        repaired_lineart = self.repair_lineart_advanced(lineart_pil, line_repair_iterations)

        # Create line exclusion mask
        print("Creating line exclusion mask...")
        exclusion_mask = self.create_line_exclusion_mask(repaired_lineart, line_threshold, paint_by_numbers_pil.size)

        # Extract colors from paint-by-numbers image ONLY
        print("Extracting dominant colors...")
        colors = self.extract_dominant_colors(paint_by_numbers_pil, num_colors, color_merge_threshold)

        # Create color map
        print("Creating color map...")
        color_map = self.create_color_map_improved(paint_by_numbers_pil, colors)

        # Find connected regions (avoiding line areas)
        print("Finding connected regions...")
        regions_by_color = self.find_connected_regions(color_map, exclusion_mask)

        # Create template with numbers
        print("Creating template with numbers...")
        template, number_positions = self.create_template_with_numbers(
            paint_by_numbers_pil, regions_by_color, colors,
            font_size, numbers_density, number_spacing, min_region_size, exclusion_mask
        )

        # Add lines to template
        print("Adding lines to template...")
        final_template = self.add_lines_to_template(template, repaired_lineart, line_threshold)

        # Create numbers-only image
        print("Creating numbers-only image...")
        numbers_only = self.create_numbers_only_image(paint_by_numbers_pil.size, number_positions)

        # Create color palette chart
        print("Creating color palette chart...")
        palette_chart = self.create_color_palette_chart(colors)

        # Convert to tensors
        template_tensor = self.pil_to_tensor(final_template)
        numbers_tensor = self.pil_to_tensor(numbers_only)
        palette_tensor = self.pil_to_tensor(palette_chart)

        print("Paint-by-numbers template creation completed!")
        return (template_tensor, numbers_tensor, palette_tensor)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PaintByNumbersTemplateNode": ImprovedPaintByNumbersTemplateNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaintByNumbersTemplateNode": "Improved Paint by Numbers Template"
}
