"""
Semantic Thermal Heuristic Engine (Member 3 Deliverable)
Maps baseline scene temperature and environmental conditions to specific semantic objects
(roads, buildings, trees, shadows) for 3D vertex coloring and microclimate thermal mapping.
"""

# Thermal offset heuristics relative to baseline LST
# Based on urban thermodynamic material properties and solar radiation absorption
SEMANTIC_THERMAL_OFFSETS = {
    "road": {
        "label": "Asphalt / Road Pavement",
        "offset_c": +4.2,
        "color_hex": "#dc2626",      # Bright Red (Severe Hotspot)
        "color_rgb": [220, 38, 38],
        "thermal_category": "Severe Hotspot",
        "albedo": 0.10,
        "emissivity": 0.93,
    },
    "building": {
        "label": "Building Facade / Wall",
        "offset_c": +1.5,
        "color_hex": "#ea580c",      # Orange (High Heat)
        "color_rgb": [234, 88, 12],
        "thermal_category": "High Heat",
        "albedo": 0.30,
        "emissivity": 0.90,
    },
    "soil_plaza": {
        "label": "Unshaded Paver / Soil",
        "offset_c": +0.8,
        "color_hex": "#f59e0b",      # Amber (Moderate Heat)
        "color_rgb": [245, 158, 11],
        "thermal_category": "Moderate Heat",
        "albedo": 0.25,
        "emissivity": 0.92,
    },
    "lawn": {
        "label": "Low Vegetation / Lawn",
        "offset_c": -1.8,
        "color_hex": "#10b981",      # Emerald Green (Cooling Buffer)
        "color_rgb": [16, 185, 129],
        "thermal_category": "Cool Buffer",
        "albedo": 0.20,
        "emissivity": 0.96,
    },
    "tree": {
        "label": "Dense Tree Canopy",
        "offset_c": -3.5,
        "color_hex": "#0284c7",      # Sky Blue (Cool Oasis)
        "color_rgb": [2, 132, 199],
        "thermal_category": "Cool Oasis",
        "albedo": 0.18,
        "emissivity": 0.98,
    },
    "shadow": {
        "label": "Deep Shadow Zone",
        "offset_c": -4.2,
        "color_hex": "#4338ca",      # Indigo (Deep Cool)
        "color_rgb": [67, 56, 202],
        "thermal_category": "Microclimate Shadow",
        "albedo": 0.05,
        "emissivity": 0.95,
    },
    "cool_pavement": {
        "label": "High-Albedo Cool Pavement (Intervention)",
        "offset_c": -2.2,
        "color_hex": "#06d6a0",      # HeatScape Teal Green
        "color_rgb": [6, 214, 160],
        "thermal_category": "Mitigated Surface",
        "albedo": 0.70,
        "emissivity": 0.91,
    }
}

def get_semantic_temperatures(baseline_lst: float, solar_exposure: float = 0.85) -> dict:
    """
    Computes precise surface temperatures for every object class in the scene.
    Adjusts offsets dynamically based on solar irradiance exposure.
    
    Example:
        If baseline is 38.0°C and solar exposure is high:
        - road: 38.0 + (4.2 * 1.0) = 42.2°C
        - tree: 38.0 - 3.5 = 34.5°C
    """
    solar_factor = max(0.6, min(solar_exposure, 1.2))
    result = {}

    for obj_name, prop in SEMANTIC_THERMAL_OFFSETS.items():
        # Solar exposure amplifies positive heating offsets (asphalt, concrete)
        if prop["offset_c"] > 0:
            effective_offset = prop["offset_c"] * solar_factor
        else:
            # Tree transpiration is most effective under high solar load
            effective_offset = prop["offset_c"] * (1.1 if solar_factor > 0.8 else 0.9)

        temp = round(baseline_lst + effective_offset, 2)

        result[obj_name] = {
            "label": prop["label"],
            "temperature_c": temp,
            "offset_c": round(effective_offset, 2),
            "color_hex": prop["color_hex"],
            "color_rgb": prop["color_rgb"],
            "thermal_category": prop["thermal_category"]
        }

    return result

def get_vertex_color_mapping(baseline_lst: float) -> dict:
    """
    Provides normalized RGBA vertex color dictionary ready for Three.js (Member 1).
    Three.js vertex colors use floats between 0.0 and 1.0.
    """
    semantic_data = get_semantic_temperatures(baseline_lst)
    vertex_colors = {}

    for key, data in semantic_data.items():
        r, g, b = data["color_rgb"]
        vertex_colors[key] = {
            "r": round(r / 255.0, 3),
            "g": round(g / 255.0, 3),
            "b": round(b / 255.0, 3),
            "temp_c": data["temperature_c"]
        }

    return vertex_colors

if __name__ == "__main__":
    test_baseline = 38.0
    mapping = get_semantic_temperatures(test_baseline, solar_exposure=0.9)
    print(f"--- Semantic Thermal Mapping for Baseline {test_baseline}°C ---")
    for k, v in mapping.items():
        print(f"  {k:15s}: {v['temperature_c']}°C ({v['thermal_category']}) -> {v['color_hex']}")
