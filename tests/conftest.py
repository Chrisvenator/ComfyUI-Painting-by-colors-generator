import os
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # Projekt-Root
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))


from paintingbycolors.hex_stack import HexStackNode

import numpy as np
import pytest

# Add the parent directory to sys.path to import the nodes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from paintingbycolors.hex_stack import HexStackNode


@pytest.fixture
def hex_stack_node():
    """Create a HexStackNode instance for testing"""
    return HexStackNode()


@pytest.fixture
def sample_hex_colors():
    """Sample hex color string with mixed formats"""
    return """#FFFFFF
#F6D300
#f88a00
#E53935
ef5aa9
C2187E
#7E3F98
#ABC
#def
invalid_color
#GGGGGG
123456
#12345
#1234567
"""


@pytest.fixture
def valid_hex_colors():
    """Clean list of valid hex colors for expected results"""
    return [
        "#FFFFFF", "#F6D300", "#F88A00", "#E53935",
        "#EF5AA9", "#C2187E", "#7E3F98", "#AABBCC", "#DDEEFF"
    ]


@pytest.fixture
def expected_rgb_colors():
    """Expected RGB values for the valid hex colors"""
    return np.array([
        [255, 255, 255],  # #FFFFFF
        [246, 211, 0],    # #F6D300
        [248, 138, 0],    # #F88A00
        [229, 57, 53],    # #E53935
        [239, 90, 169],   # #EF5AA9
        [194, 24, 126],   # #C2187E
        [126, 63, 152],   # #7E3F98
        [170, 187, 204],  # #ABC -> #AABBCC
        [221, 238, 255]   # #def -> #DDEEFF
    ], dtype=np.float32)


@pytest.fixture
def empty_hex_string():
    """Empty or whitespace-only hex string"""
    return "\n\n   \n  \n"


@pytest.fixture
def single_hex_color():
    """Single valid hex color"""
    return "#FF0000"
