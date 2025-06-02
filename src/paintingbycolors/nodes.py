from .calculate_numbers import ImprovedPaintByNumbersTemplateNode
from .hex_stack import HexStackNode
from .paste_numbers_on_image import NumbersOverlayNode, NumbersOverlayAdvancedNode
from .preprocessor import EnhancedPaintByNumbersNode

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
    "NumbersOverlayNode": "Paint by Numbers: overlay Numbers",
    "NumbersOverlayAdvancedNode": "Paint by Numbers: overlay Numbers (Advanced)",
    "HexStackNode": "Hex Color Stack",
}
