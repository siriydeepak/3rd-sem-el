"""
Semantic Thermal Service for HeatScape Backend (Member 3 Deliverable)
Maps scene baseline temperature to object-level temperatures and 3D vertex colors.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from heatscape_ml.src.semantic_thermal import (
    get_semantic_temperatures,
    get_vertex_color_mapping,
    SEMANTIC_THERMAL_OFFSETS,
)

__all__ = ["get_semantic_temperatures", "get_vertex_color_mapping", "SEMANTIC_THERMAL_OFFSETS"]
