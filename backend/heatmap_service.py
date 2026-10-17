"""
Heat Map Generator Service for HeatScape
Generates spatial microclimate thermal heat map points around the geotagged photo location.
Simulates baseline vs mitigated thermal gradients across the urban microclimate.
"""

import math
import random

def generate_urban_heat_map(
    center_lat: float,
    center_lon: float,
    baseline_lst: float,
    mitigated_drop: float = 3.5,
    radius_meters: float = 250.0,
    grid_size: int = 5
) -> dict:
    """
    Generates a localized spatial grid of thermal heat map points centered on the photo's GPS coordinates.
    Simulates thermal variations based on distance, surface types, and intervention impacts.
    """
    points = []
    
    # 1 degree latitude ~ 111,000 meters
    lat_step = (radius_meters / 111000.0) / (grid_size // 2 + 1)
    # 1 degree longitude ~ 111,000 * cos(lat)
    lon_step = (radius_meters / (111000.0 * math.cos(math.radians(center_lat)))) / (grid_size // 2 + 1)

    half = grid_size // 2

    # Zones around campus
    zone_profiles = [
        {"type": "Asphalt Road", "offset": 2.2, "category": "Severe Hotspot"},
        {"type": "Open Paved Plaza", "offset": 1.4, "category": "Urban Hotspot"},
        {"type": "Building Courtyard", "offset": 0.5, "category": "High Heat"},
        {"type": "Lawn / Planter", "offset": -1.8, "category": "Moderate"},
        {"type": "Dense Tree Grove", "offset": -3.8, "category": "Cool Oasis"},
    ]

    min_lat, max_lat = center_lat, center_lat
    min_lon, max_lon = center_lon, center_lon

    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            p_lat = round(center_lat + i * lat_step, 6)
            p_lon = round(center_lon + j * lon_step, 6)

            min_lat = min(min_lat, p_lat)
            max_lat = max(max_lat, p_lat)
            min_lon = min(min_lon, p_lon)
            max_lon = max(max_lon, p_lon)

            # Distance from center in meters
            dist = math.sqrt((i * lat_step * 111000)**2 + (j * lon_step * 111000 * math.cos(math.radians(center_lat)))**2)

            is_center = (i == 0 and j == 0)

            if is_center:
                t_base = baseline_lst
                t_mit = round(baseline_lst - mitigated_drop, 2)
                z_name = "Target Photo Geotag (Analyzed Site)"
                cat = "Severe Hotspot" if t_base > 38 else "Urban Hotspot"
            else:
                # Deterministic pseudo-randomness based on grid coordinates
                seed_val = int(abs(i * 31 + j * 17))
                profile = zone_profiles[seed_val % len(zone_profiles)]
                t_base = round(baseline_lst + profile["offset"] + (0.3 if dist < 100 else -0.2), 2)
                
                # Mitigated impact decreases slightly with distance from intervention center
                impact_factor = max(0.4, 1.0 - (dist / radius_meters) * 0.6)
                t_mit = round(t_base - (mitigated_drop * impact_factor), 2)
                z_name = f"Zone {profile['type']} ({int(dist)}m)"
                cat = profile["category"]

            # Heat intensity for heatmap coloring (0.0 to 1.0)
            norm_heat_base = min(max((t_base - 26.0) / 18.0, 0.05), 1.0)
            norm_heat_mit = min(max((t_mit - 26.0) / 18.0, 0.05), 1.0)

            points.append({
                "id": f"p_{i+half}_{j+half}",
                "lat": p_lat,
                "lon": p_lon,
                "distance_m": round(dist, 1),
                "is_center": is_center,
                "zone_name": z_name,
                "temp_baseline": t_base,
                "temp_mitigated": t_mit,
                "temp_drop": round(t_base - t_mit, 2),
                "heat_intensity_base": round(norm_heat_base, 3),
                "heat_intensity_mit": round(norm_heat_mit, 3),
                "category": cat
            })

    return {
        "center": {"lat": center_lat, "lon": center_lon},
        "bounds": {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon
        },
        "radius_meters": radius_meters,
        "points": points,
        "summary": {
            "baseline_avg_temp": round(sum(p["temp_baseline"] for p in points) / len(points), 2),
            "mitigated_avg_temp": round(sum(p["temp_mitigated"] for p in points) / len(points), 2),
            "max_hotspot_temp": max(p["temp_baseline"] for p in points),
            "min_cool_temp": min(p["temp_mitigated"] for p in points),
            "total_zones": len(points)
        }
    }
