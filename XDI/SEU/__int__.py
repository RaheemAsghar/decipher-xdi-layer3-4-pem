"""Layer 2 modular package.

These modules provide independently-importable access to Layer 2 components while
preserving the original Layer_2_V8.Layer2Computer behavior.

Primary entrypoint:
    from layer2_modular import Layer2Computer
"""

from .layer2_computer import Layer2Computer
from .layer2_config import Layer2Config, DEFAULT_CONFIG
