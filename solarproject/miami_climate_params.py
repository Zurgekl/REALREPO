"""
Miami Climate Parameters for Academic Hall North Thermal Model
"""

# Comfort thresholds (ASHRAE 55)
COMFORT = {
    'summer_comfort_lower': 23,  # °C (73.4°F)
    'summer_comfort_upper': 26,  # °C (78.8°F)
    'winter_comfort_lower': 20,  # °C (68°F)
    'winter_comfort_upper': 24,  # °C (75.2°F)
}

# Miami climate characteristics
CLIMATE = {
    'cooling_degree_days_base_18': 2200,  # Typical for Miami
    'heating_degree_days_base_18': 150,   # Minimal heating needed
    'design_outdoor_temp_summer': 34,     # °C (93°F) - ASHRAE 0.4% cooling
    'design_outdoor_temp_winter': 8,      # °C (46°F) - ASHRAE 99.6% heating
    'latitude': 25.76,                    # degrees North
    'longitude': -80.19,                  # degrees West
}

# Solar radiation (typical Miami values)
SOLAR = {
    'global_horizontal_peak': 1000,       # W/m² (summer noon)
    'direct_normal_peak': 900,            # W/m²
    'diffuse_horizontal_avg': 200,        # W/m²
}
