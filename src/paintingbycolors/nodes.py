from .painting_by_numbers_calculate_numbers import ImprovedPaintByNumbersTemplateNode
from .painting_by_numbers_hex_stack import HexStackNode
from .painting_by_numbers_paste_numbers_on_image import NumbersOverlayNode, NumbersOverlayAdvancedNode
from .painting_by_numbers_preprocessor import EnhancedPaintByNumbersNode

NODE_CLASS_MAPPINGS = {
    "PaintByNumbersNode": EnhancedPaintByNumbersNode,
    "PaintByNumbersTemplateNode": ImprovedPaintByNumbersTemplateNode,
    "NumbersOverlayNode": NumbersOverlayNode,
    "NumbersOverlayAdvancedNode": NumbersOverlayAdvancedNode,
    "HexStackNode": HexStackNode,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaintByNumbersNode": "Paint by Numbers Preprocessor: K-Mean",
    "PaintByNumbersTemplateNode": "Paint by Numbers: Calculate numbers",
    "PaintByNumbersPaletteNode": "Paint by Numbers: Palette",
    "NumbersOverlayNode": "Paint by Numbers: Numbers Overlay",
    "NumbersOverlayAdvancedNode": "Paint by Numbers: Numbers Overlay (Advanced)",
    "HexStackNode": "Hex Color Stack",
}
