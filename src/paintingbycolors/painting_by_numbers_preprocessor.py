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
                "num_colors": ("INT", {
                    "default": 12,
                    "min": 4,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "blur_radius": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "color_intensity": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "noise_reduction_strength": ("FLOAT", {
                    "default": 2.0,
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
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("color_quantized",)
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

    def apply_morphological_cleanup(self, image_np):
        """Apply morphological operations to clean up the image"""
        # Convert to single channel for morphological operations
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Apply closing to fill small gaps
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Apply opening to remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # Use the cleaned mask to guide color correction
        mask_diff = (opened != gray).astype(np.float32)

        if np.sum(mask_diff) > 0:
            # Smooth the problematic areas
            blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
            mask_3d = np.stack([mask_diff] * 3, axis=2)
            image_np = image_np * (1 - mask_3d) + blurred * mask_3d

        return image_np.astype(np.uint8)

    def create_color_quantized_image(self, original_pil, num_colors, blur_radius,
                                     color_intensity, noise_reduction_strength,
                                     use_bilateral, use_morphological, min_region_size):
        """Create clean color quantized image without lineart"""

        # Step 1: Denoise the original image
        denoised = self.denoise_image(original_pil, noise_reduction_strength, use_bilateral)

        # Step 2: Apply blur for smoother color regions
        if blur_radius > 0:
            blurred_original = denoised.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        else:
            blurred_original = denoised

        # Step 3: Improved color quantization
        quantized = self.quantize_colors_improved(blurred_original, num_colors, color_intensity)

        # Step 4: Clean up small regions
        quantized_np = np.array(quantized)
        if min_region_size > 0:
            quantized_np = self.clean_small_regions(quantized_np, min_region_size)

        # Step 5: Apply morphological cleanup
        if use_morphological:
            quantized_np = self.apply_morphological_cleanup(quantized_np)

        # Convert back to PIL
        result = Image.fromarray(quantized_np.astype(np.uint8))
        return result

    def process(self, original_image, num_colors, blur_radius, color_intensity,
                noise_reduction_strength, bilateral_filter, morphological_cleanup,
                min_region_size):
        """Process the image and return color quantized result"""

        # Convert tensor to PIL Image
        original_pil = self.tensor_to_pil(original_image)

        # Ensure RGB mode
        if original_pil.mode != 'RGB':
            original_pil = original_pil.convert('RGB')

        # Create color quantized effect
        result_pil = self.create_color_quantized_image(
            original_pil, num_colors, blur_radius, color_intensity,
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
