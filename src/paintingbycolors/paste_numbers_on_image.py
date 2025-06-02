import torch
import numpy as np
from PIL import Image
import cv2


class NumbersOverlayNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "numbers_image": ("IMAGE",),
                "transparency": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "white_threshold": ("INT", {
                    "default": 240,
                    "min": 200,
                    "max": 255,
                    "step": 5,
                    "display": "number"
                }),
                "blend_mode": (["replace", "multiply", "overlay", "soft_light"], {
                    "default": "replace"
                }),
                "auto_scale": ("BOOLEAN", {
                    "default": True,
                    "display": "checkbox"
                }),
                "scaling_method": (["lanczos", "bicubic", "bilinear", "nearest"], {
                    "default": "lanczos"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlaid_image",)
    FUNCTION = "overlay_numbers"
    CATEGORY = "image/postprocessing"

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

    def get_scaling_method(self, method_name):
        """Get PIL resampling method from string"""
        methods = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "bilinear": Image.Resampling.BILINEAR,
            "nearest": Image.Resampling.NEAREST
        }
        return methods.get(method_name, Image.Resampling.LANCZOS)

    def create_mask_from_image(self, numbers_img, white_threshold):
        """Create a mask for non-white pixels in the numbers image"""
        numbers_array = np.array(numbers_img)

        # Create mask for non-white pixels
        non_white_mask = np.any(numbers_array < white_threshold, axis=2)

        return non_white_mask

    def apply_blend_mode(self, base_pixels, overlay_pixels, mode, alpha):
        """Apply different blend modes for combining images"""
        base = base_pixels.astype(np.float32) / 255.0
        overlay = overlay_pixels.astype(np.float32) / 255.0

        if mode == "replace":
            result = overlay
        elif mode == "multiply":
            result = base * overlay
        elif mode == "overlay":
            # Overlay blend mode
            mask = base < 0.5
            result = np.where(mask, 2 * base * overlay, 1 - 2 * (1 - base) * (1 - overlay))
        elif mode == "soft_light":
            # Soft light blend mode
            mask = overlay < 0.5
            result = np.where(mask,
                              base - (1 - 2 * overlay) * base * (1 - base),
                              base + (2 * overlay - 1) * (np.sqrt(base) - base))
        else:
            result = overlay

        # Apply alpha blending
        result = base * (1 - alpha) + result * alpha

        return (result * 255).astype(np.uint8)

    def overlay_numbers(self, input_image, numbers_image, transparency, white_threshold,
                        blend_mode, auto_scale, scaling_method):
        """Overlay numbers image onto input image"""

        # Convert tensors to PIL Images
        input_pil = self.tensor_to_pil(input_image)
        numbers_pil = self.tensor_to_pil(numbers_image)

        # Ensure both images are in RGB mode
        if input_pil.mode != 'RGB':
            input_pil = input_pil.convert('RGB')
        if numbers_pil.mode != 'RGB':
            numbers_pil = numbers_pil.convert('RGB')

        print(f"Input image size: {input_pil.size}")
        print(f"Numbers image size: {numbers_pil.size}")

        # Scale numbers image to match input image size if needed
        if auto_scale and numbers_pil.size != input_pil.size:
            print(f"Scaling numbers image from {numbers_pil.size} to {input_pil.size}")
            scaling_filter = self.get_scaling_method(scaling_method)
            numbers_pil = numbers_pil.resize(input_pil.size, scaling_filter)

        # Convert to numpy arrays
        input_array = np.array(input_pil)
        numbers_array = np.array(numbers_pil)

        # Handle size mismatch if auto_scale is disabled
        if not auto_scale and numbers_pil.size != input_pil.size:
            # Crop or pad to match sizes
            input_h, input_w = input_array.shape[:2]
            numbers_h, numbers_w = numbers_array.shape[:2]

            if numbers_h > input_h or numbers_w > input_w:
                # Crop numbers image
                numbers_array = numbers_array[:input_h, :input_w]
            elif numbers_h < input_h or numbers_w < input_w:
                # Pad numbers image with white
                padded = np.full((input_h, input_w, 3), 255, dtype=np.uint8)
                padded[:numbers_h, :numbers_w] = numbers_array
                numbers_array = padded

        # Create mask for non-white pixels in numbers image
        non_white_mask = self.create_mask_from_image(Image.fromarray(numbers_array), white_threshold)

        # Create the result image
        result_array = input_array.copy()

        # Apply overlay only where there are non-white pixels in the numbers image
        if np.any(non_white_mask):
            if blend_mode == "replace" and transparency == 1.0:
                # Simple replacement for better performance
                result_array[non_white_mask] = numbers_array[non_white_mask]
            else:
                # Apply blend mode with transparency
                blended_pixels = self.apply_blend_mode(
                    input_array[non_white_mask],
                    numbers_array[non_white_mask],
                    blend_mode,
                    transparency
                )
                result_array[non_white_mask] = blended_pixels

        # Convert back to PIL Image and then to tensor
        result_pil = Image.fromarray(result_array)
        result_tensor = self.pil_to_tensor(result_pil)

        return (result_tensor,)


class NumbersOverlayAdvancedNode:
    """Advanced version with more features"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "numbers_image": ("IMAGE",),
                "transparency": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
            },
            "optional": {
                "mask_image": ("MASK",),
                "white_threshold": ("INT", {
                    "default": 240,
                    "min": 200,
                    "max": 255,
                    "step": 5,
                    "display": "number"
                }),
                "blend_mode": (["replace", "multiply", "overlay", "soft_light", "screen", "color_burn"], {
                    "default": "replace"
                }),
                "auto_scale": ("BOOLEAN", {
                    "default": True,
                    "display": "checkbox"
                }),
                "scaling_method": (["lanczos", "bicubic", "bilinear", "nearest"], {
                    "default": "lanczos"
                }),
                "numbers_color": ("STRING", {
                    "default": "auto",
                    "multiline": False
                }),
                "outline_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
                "outline_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("overlaid_image", "numbers_mask")
    FUNCTION = "overlay_numbers_advanced"
    CATEGORY = "image/postprocessing"

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

    def mask_to_tensor(self, mask_array):
        """Convert mask array to ComfyUI mask tensor"""
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]
        mask_tensor = torch.from_numpy(mask_array.astype(np.float32)).unsqueeze(0)
        return mask_tensor

    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def apply_outline(self, mask, outline_width):
        """Apply outline/border to mask"""
        if outline_width <= 0:
            return mask

        kernel = np.ones((outline_width * 2 + 1, outline_width * 2 + 1), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return dilated.astype(bool)

    def overlay_numbers_advanced(self, input_image, numbers_image, transparency,
                                 mask_image=None, white_threshold=240, blend_mode="replace",
                                 auto_scale=True, scaling_method="lanczos",
                                 numbers_color="auto", outline_width=0, outline_color="#FFFFFF"):
        """Advanced overlay with more features"""

        # Convert tensors to PIL Images
        input_pil = self.tensor_to_pil(input_image)
        numbers_pil = self.tensor_to_pil(numbers_image)

        # Ensure both images are in RGB mode
        if input_pil.mode != 'RGB':
            input_pil = input_pil.convert('RGB')
        if numbers_pil.mode != 'RGB':
            numbers_pil = numbers_pil.convert('RGB')

        # Scale numbers image if needed
        if auto_scale and numbers_pil.size != input_pil.size:
            scaling_methods = {
                "lanczos": Image.Resampling.LANCZOS,
                "bicubic": Image.Resampling.BICUBIC,
                "bilinear": Image.Resampling.BILINEAR,
                "nearest": Image.Resampling.NEAREST
            }
            scaling_filter = scaling_methods.get(scaling_method, Image.Resampling.LANCZOS)
            numbers_pil = numbers_pil.resize(input_pil.size, scaling_filter)

        # Convert to numpy arrays
        input_array = np.array(input_pil)
        numbers_array = np.array(numbers_pil)

        # Create mask for non-white pixels
        non_white_mask = np.any(numbers_array < white_threshold, axis=2)

        # Apply outline if specified
        if outline_width > 0:
            outlined_mask = self.apply_outline(non_white_mask, outline_width)
            outline_only_mask = outlined_mask & ~non_white_mask

        # Apply external mask if provided
        if mask_image is not None:
            external_mask = mask_image[0].cpu().numpy() > 0.5
            if external_mask.shape != non_white_mask.shape:
                external_mask = cv2.resize(external_mask.astype(np.uint8),
                                           (non_white_mask.shape[1], non_white_mask.shape[0])) > 0.5
            non_white_mask = non_white_mask & external_mask

        # Create result image
        result_array = input_array.copy()

        # Apply outline first if specified
        if outline_width > 0:
            outline_rgb = self.hex_to_rgb(outline_color)
            result_array[outline_only_mask] = outline_rgb

        # Color replacement if specified
        if numbers_color != "auto" and numbers_color.startswith('#'):
            new_color = self.hex_to_rgb(numbers_color)
            numbers_array[non_white_mask] = new_color

        # Apply overlay
        if np.any(non_white_mask):
            if blend_mode == "replace" and transparency == 1.0:
                result_array[non_white_mask] = numbers_array[non_white_mask]
            else:
                # Apply transparency blending
                for i in range(3):
                    result_array[non_white_mask, i] = (
                        input_array[non_white_mask, i] * (1 - transparency) +
                        numbers_array[non_white_mask, i] * transparency
                    ).astype(np.uint8)

        # Convert results back to tensors
        result_pil = Image.fromarray(result_array)
        result_tensor = self.pil_to_tensor(result_pil)

        # Create mask tensor for output
        final_mask = non_white_mask.astype(np.float32)
        if outline_width > 0:
            final_mask = final_mask | outline_only_mask.astype(np.float32)
        mask_tensor = self.mask_to_tensor(final_mask)

        return (result_tensor, mask_tensor)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "NumbersOverlayNode": NumbersOverlayNode,
    "NumbersOverlayAdvancedNode": NumbersOverlayAdvancedNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NumbersOverlayNode": "Numbers Overlay",
    "NumbersOverlayAdvancedNode": "Numbers Overlay (Advanced)"
}
