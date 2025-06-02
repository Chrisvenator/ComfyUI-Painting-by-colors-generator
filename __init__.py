"""Top-level package for paintingbycolors."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """chrisvenator"""
__email__ = "you@gmail.com"
__version__ = "1.0.0"

from .src.paintingbycolors.nodes import NODE_CLASS_MAPPINGS
from .src.paintingbycolors.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
