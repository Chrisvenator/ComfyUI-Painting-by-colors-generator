from .painting_by_numbers_preprocessor import EnhancedPaintByNumbersNode
from .painting_by_numbers_numbering import ImprovedPaintByNumbersTemplateNode
from .paste_numbers_on_image import NumbersOverlayAdvancedNode, NumbersOverlayNode

NODE_CLASS_MAPPINGS = {
    "PaintByNumbersNode": EnhancedPaintByNumbersNode,
    "PaintByNumbersTemplateNode": ImprovedPaintByNumbersTemplateNode,
    "NumbersOverlayNode": NumbersOverlayNode,
    "NumbersOverlayAdvancedNode": NumbersOverlayAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaintByNumbersNode": "Paint by Numbers Preprocessor: K-Mean",
    "PaintByNumbersTemplateNode": "Paint by Numbers: Calculate numbers",
    "PaintByNumbersPaletteNode": "Paint by Numbers: Palette",
    "NumbersOverlayNode": "Paint by Numbers: Numbers Overlay",
    "NumbersOverlayAdvancedNode": "Paint by Numbers: Numbers Overlay (Advanced)",
}
