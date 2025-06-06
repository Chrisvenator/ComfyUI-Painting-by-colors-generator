import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from scipy import ndimage
import torch.nn.functional as F


class EnhancedPaintByNumbersNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "blur_radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "color_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "noise_reduction_strength": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "bilateral_filter": ("BOOLEAN", {
                    "default": True,
                    "display": "checkbox"
                }),
                "morphological_cleanup": ("BOOLEAN", {
                    "default": True,
                    "display": "checkbox"
                }),
                "min_region_size": ("INT", {
                    "default": 800,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
            },
            "optional": {
                "hex_stack": ("HEX_STACK",),
                "num_colors": ("INT", {
                    "default": 20,
                    "min": 2,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preprocessed_image",)
    FUNCTION = "process"
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

    def denoise_image(self, image_pil, noise_reduction_strength, use_bilateral):
        """Apply denoising to reduce compression artifacts"""
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        if use_bilateral and noise_reduction_strength > 0:
            # Bilateral filter preserves edges while reducing noise
            diameter = int(noise_reduction_strength * 3 + 3)  # 3-18 range
            sigma_color = noise_reduction_strength * 20  # 0-100 range
            sigma_space = noise_reduction_strength * 20  # 0-100 range

            img_cv = cv2.bilateralFilter(img_cv, diameter, sigma_color, sigma_space)

        # Additional Gaussian blur for very noisy images
        if noise_reduction_strength > 2.0:
            kernel_size = int((noise_reduction_strength - 2.0) * 2 + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            img_cv = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)

        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def quantize_colors_improved(self, image_pil, num_colors, intensity=1.0):
        """Improved color quantization with better sampling"""
        img_array = np.array(image_pil)
        original_shape = img_array.shape

        # Reshape to 2D array for clustering
        pixels = img_array.reshape(-1, 3).astype(np.float32)

        # Apply color intensity
        pixels = np.clip(pixels * intensity, 0, 255)

        # Sample pixels more intelligently - avoid redundant similar pixels
        # This helps reduce the impact of noise
        n_samples = min(50000, len(pixels))  # Limit samples for performance

        if len(pixels) > n_samples:
            # Use stratified sampling to get better color representation
            step = len(pixels) // n_samples
            sampled_pixels = pixels[::step]
        else:
            sampled_pixels = pixels

        # Perform K-means clustering on sampled pixels
        kmeans = KMeans(
            n_clusters=num_colors,
            random_state=42,
            n_init=20,  # More iterations for better results
            max_iter=500
        )
        kmeans.fit(sampled_pixels)

        # Apply quantization to all pixels
        quantized_pixels = kmeans.cluster_centers_[kmeans.predict(pixels)]
        quantized_image = quantized_pixels.reshape(original_shape).astype(np.uint8)

        return Image.fromarray(quantized_image)

    def quantize_to_hex_palette(self, image_pil, hex_colors, rgb_colors, intensity=1.0):
        """Quantize image colors to match the provided hex palette using optimized approach"""
        img_array = np.array(image_pil)
        original_shape = img_array.shape

        # Reshape to 2D array
        pixels = img_array.reshape(-1, 3).astype(np.float32)

        # Apply color intensity
        pixels = np.clip(pixels * intensity, 0, 255)

        # Vectorized distance calculation for better performance
        # Expand dimensions for broadcasting: pixels -> (N, 1, 3), palette -> (1, K, 3)
        pixels_expanded = pixels[:, np.newaxis, :]  # Shape: (N, 1, 3)
        palette_expanded = rgb_colors[np.newaxis, :, :]  # Shape: (1, K, 3)

        # Calculate squared Euclidean distances
        distances = np.sum((pixels_expanded - palette_expanded) ** 2, axis=2)  # Shape: (N, K)

        # Find closest color indices
        closest_indices = np.argmin(distances, axis=1)

        # Map pixels to closest palette colors
        quantized_pixels = rgb_colors[closest_indices]

        quantized_image = quantized_pixels.reshape(original_shape).astype(np.uint8)
        return Image.fromarray(quantized_image)

    def select_optimal_colors_from_palette(self, image_pil, rgb_colors, max_colors):
        """Select the most representative colors from the hex palette for the given image"""
        img_array = np.array(image_pil)
        pixels = img_array.reshape(-1, 3).astype(np.float32)

        # Sample pixels for performance but keep it representative
        n_samples = min(20000, len(pixels))
        if len(pixels) > n_samples:
            # Use random sampling instead of step sampling for better representation
            indices = np.random.choice(len(pixels), n_samples, replace=False)
            sampled_pixels = pixels[indices]
        else:
            sampled_pixels = pixels

        # Calculate color usage scores more accurately
        color_scores = []

        for i, palette_color in enumerate(rgb_colors):
            # Calculate distances from sampled pixels to this palette color
            distances = np.sum((sampled_pixels - palette_color) ** 2, axis=1)

            # Multiple scoring criteria
            min_distance = np.min(distances)
            avg_distance = np.mean(distances)
            median_distance = np.median(distances)

            # Count pixels that would use this color (closest match)
            all_distances = np.sum((sampled_pixels[:, np.newaxis, :] - rgb_colors[np.newaxis, :, :]) ** 2, axis=2)
            closest_colors = np.argmin(all_distances, axis=1)
            usage_count = np.sum(closest_colors == i)

            # Combined score: higher usage and lower distances are better
            if usage_count > 0:
                score = usage_count / (1 + avg_distance / 100)
            else:
                score = 0

            color_scores.append((score, i))

        # Sort by score and select top colors
        color_scores.sort(reverse=True, key=lambda x: x[0])

        # Ensure we get at least some colors even if scores are low
        num_to_select = min(max_colors, len(rgb_colors))
        selected_indices = [idx for _, idx in color_scores[:num_to_select]]

        return rgb_colors[selected_indices]

    def clean_small_regions(self, image_np, min_size):
        """Remove small isolated regions that are likely noise"""
        if min_size <= 0:
            return image_np

        # Convert to single channel for region analysis
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(gray)

        # Count pixels in each region
        for label_id in range(1, num_labels):
            mask = labels == label_id
            if np.sum(mask) < min_size:
                # Replace small regions with neighboring dominant color
                # Dilate the mask slightly to find neighbors
                kernel = np.ones((3, 3), np.uint8)
                dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
                neighbor_mask = dilated_mask - mask.astype(np.uint8)

                if np.sum(neighbor_mask) > 0:
                    # Get the most common color from neighbors
                    neighbor_colors = image_np[neighbor_mask > 0]
                    if len(neighbor_colors) > 0:
                        # Use median color of neighbors
                        replacement_color = np.median(neighbor_colors, axis=0)
                        image_np[mask] = replacement_color

        return image_np

    def apply_gentle_cleanup(self, image_np):
        """Apply gentler morphological operations to clean up the image"""
        # Convert to single channel for morphological operations
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Use smaller kernel for gentler operations
        kernel = np.ones((2, 2), np.uint8)

        # Very light closing to fill tiny gaps
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Light opening to remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        # Only fix significant differences
        mask_diff = np.abs(opened.astype(np.float32) - gray.astype(np.float32)) > 10
        mask_diff = mask_diff.astype(np.float32)

        if np.sum(mask_diff) > 0:
            # Very light smoothing only where needed
            blurred = cv2.GaussianBlur(image_np, (3, 3), 0)
            mask_3d = np.stack([mask_diff] * 3, axis=2)
            image_np = image_np * (1 - mask_3d * 0.3) + blurred * (mask_3d * 0.3)

        return image_np.astype(np.uint8)

    def create_color_quantized_image(self, original_pil, hex_stack, num_colors, blur_radius,
                                     color_intensity, noise_reduction_strength,
                                     use_bilateral, use_morphological, min_region_size):
        """Create clean color quantized image optimized for paint-by-numbers"""

        # Step 1: More gentle denoising to preserve details
        if noise_reduction_strength > 0:
            denoised = self.denoise_image(original_pil, noise_reduction_strength * 0.7, use_bilateral)
        else:
            denoised = original_pil

        # Step 2: Apply blur for smoother color regions (reduce intensity)
        if blur_radius > 0:
            # Use a gentler blur
            blurred_original = denoised.filter(ImageFilter.GaussianBlur(radius=blur_radius * 0.8))
        else:
            blurred_original = denoised

        # Step 3: Color quantization (with or without hex stack)
        if hex_stack is not None and hex_stack['count'] > 0:
            # Use provided hex palette
            rgb_colors = hex_stack['rgb_colors']

            # If we have more colors than needed, select the most relevant ones
            if len(rgb_colors) > num_colors:
                rgb_colors = self.select_optimal_colors_from_palette(
                    blurred_original, rgb_colors, num_colors
                )

            # Quantize to the hex palette
            quantized = self.quantize_to_hex_palette(
                blurred_original, hex_stack['hex_colors'], rgb_colors, color_intensity
            )
        else:
            # Use automatic color quantization
            quantized = self.quantize_colors_improved(blurred_original, num_colors, color_intensity)

        # Step 4: More conservative region cleanup
        quantized_np = np.array(quantized)
        if min_region_size > 0:
            # Reduce the min region size to be less aggressive
            adjusted_min_size = max(min_region_size // 2, 10)
            quantized_np = self.clean_small_regions(quantized_np, adjusted_min_size)

        # Step 5: Gentler morphological cleanup
        if use_morphological:
            # Apply lighter morphological operations
            quantized_np = self.apply_gentle_cleanup(quantized_np)

        # Convert back to PIL
        result = Image.fromarray(quantized_np.astype(np.uint8))
        return result

    def process(self, original_image, blur_radius, color_intensity,
                noise_reduction_strength, bilateral_filter, morphological_cleanup,
                min_region_size, hex_stack=None, num_colors=12):
        """Process the image and return color quantized result"""

        # Convert tensor to PIL Image
        original_pil = self.tensor_to_pil(original_image)

        # Ensure RGB mode
        if original_pil.mode != 'RGB':
            original_pil = original_pil.convert('RGB')

        # Create color quantized effect
        result_pil = self.create_color_quantized_image(
            original_pil, hex_stack, num_colors, blur_radius, color_intensity,
            noise_reduction_strength, bilateral_filter,
            morphological_cleanup, min_region_size
        )

        # Convert back to tensor
        result_tensor = self.pil_to_tensor(result_pil)

        return (result_tensor,)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "EnhancedPaintByNumbersNode": EnhancedPaintByNumbersNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedPaintByNumbersNode": "Enhanced Color Quantizer"
}


def setup():
    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
