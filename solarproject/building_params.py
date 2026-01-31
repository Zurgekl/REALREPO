"""
Academic Hall North - Baseline Building Parameters
With explicit UA calculation and effective thermal capacitance
"""

import miami_climate_params as mcp

# ============================================================================
# GEOMETRY (from problem statement)
# ============================================================================
BUILDING_GEOMETRY = {
    'length': 60,           # m (E-W orientation)
    'width': 24,            # m (N-S)
    'height_per_floor': 3.5,  # m (assume typical)
    'n_floors': 2,
    'total_height': 7,      # m
    'floor_area': 60 * 24,  # 1440 m²
    'perimeter': 2 * (60 + 24),  # 168 m
}

# ============================================================================
# FAÇADE AREAS (gross wall area, including windows)
# ============================================================================
FACADE_AREAS = {
    'south': BUILDING_GEOMETRY['length'] * BUILDING_GEOMETRY['total_height'],  # 420 m²
    'north': BUILDING_GEOMETRY['length'] * BUILDING_GEOMETRY['total_height'],  # 420 m²
    'east': BUILDING_GEOMETRY['width'] * BUILDING_GEOMETRY['total_height'],    # 168 m²
    'west': BUILDING_GEOMETRY['width'] * BUILDING_GEOMETRY['total_height'],    # 168 m²
}
FACADE_AREAS['total'] = sum(FACADE_AREAS.values())  # 1176 m²

# ============================================================================
# WINDOW AREAS (from WWR ratios in problem statement)
# ============================================================================
WINDOW_AREAS = {
    'south': FACADE_AREAS['south'] * 0.45,  # 189 m²
    'north': FACADE_AREAS['north'] * 0.30,  # 126 m²
    'east': FACADE_AREAS['east'] * 0.30,    # 50.4 m²
    'west': FACADE_AREAS['west'] * 0.30,    # 50.4 m²
}
WINDOW_AREAS['total'] = sum(WINDOW_AREAS.values())  # 415.8 m²

# Opaque wall areas (gross façade minus windows)
OPAQUE_WALL_AREAS = {
    key: FACADE_AREAS[key] - WINDOW_AREAS[key]
    for key in ['south', 'north', 'east', 'west']
}
OPAQUE_WALL_AREAS['total'] = sum(OPAQUE_WALL_AREAS.values())  # 760.2 m²

# ============================================================================
# ENVELOPE THERMAL PROPERTIES (baseline existing building)
# ============================================================================
ENVELOPE_BASELINE = {
    'U_wall': 0.45,         # W/(m²·K) - brick veneer with insulation
    'U_roof': 0.30,         # W/(m²·K) - insulated roof
    'U_window': 2.8,        # W/(m²·K) - double glazing, no low-e
    'SHGC': 0.65,           # Solar Heat Gain Coefficient (typical double glazing)
}

# ============================================================================
# FIX 3: EXPLICIT UA CALCULATION (not averaged)
# ============================================================================
UA_ENVELOPE = (
    OPAQUE_WALL_AREAS['total'] * ENVELOPE_BASELINE['U_wall'] +  # Walls
    BUILDING_GEOMETRY['floor_area'] * ENVELOPE_BASELINE['U_roof'] +  # Roof
    WINDOW_AREAS['total'] * ENVELOPE_BASELINE['U_window']  # Windows
)
# Result: ~342 + 432 + 1164 = 1938 W/K

print(f"UA breakdown:")
print(f"  Walls: {OPAQUE_WALL_AREAS['total'] * ENVELOPE_BASELINE['U_wall']:.0f} W/K")
print(f"  Roof: {BUILDING_GEOMETRY['floor_area'] * ENVELOPE_BASELINE['U_roof']:.0f} W/K")
print(f"  Windows: {WINDOW_AREAS['total'] * ENVELOPE_BASELINE['U_window']:.0f} W/K")
print(f"  Total UA: {UA_ENVELOPE:.0f} W/K")

# ============================================================================
# FIX 2: EFFECTIVE THERMAL CAPACITANCE (not full concrete mass)
# ============================================================================
# State variable T_in represents "lumped zone temperature" = air + coupled mass
#
# Effective capacitance includes:
# - Zone air (~1.2 kg/m³ × 1005 J/kg·K × volume)
# - Coupled thermal mass (furniture, interior finishes, shallow concrete coupling)
#
# Rule of thumb: 50-150 kJ/(m²·K) for medium-mass commercial buildings
# We'll use 80 kJ/(m²·K) as reasonable middle ground

C_eff_per_area = 80000  # J/(m²·K) - effective capacitance per floor area

C_EFFECTIVE = C_eff_per_area * BUILDING_GEOMETRY['floor_area']
# Result: ~115 MJ/K

# For reference: zone air alone would be
zone_volume = BUILDING_GEOMETRY['floor_area'] * BUILDING_GEOMETRY['total_height']
C_air_only = zone_volume * 1.2 * 1005  # ~12 MJ/K (much smaller)

print(f"\nThermal capacitance:")
print(f"  Effective C: {C_EFFECTIVE/1e6:.1f} MJ/K")
print(f"  (Air alone would be: {C_air_only/1e6:.1f} MJ/K)")
print(f"  Time constant τ = C/UA: {(C_EFFECTIVE/UA_ENVELOPE)/3600:.1f} hours")

# ============================================================================
# INTERNAL GAINS (baseline schedule)
# ============================================================================
# Classroom/office building typical values
Q_INTERNAL_BASELINE = {
    'occupants_W_per_m2': 10,      # Assumes 0.1 people/m² × 100 W/person
    'equipment_W_per_m2': 10,      # Computers, projectors
    'lighting_W_per_m2': 12,       # LED lighting
}

# Total during occupied hours
Q_INTERNAL_OCCUPIED = (
    sum(Q_INTERNAL_BASELINE.values()) *
    BUILDING_GEOMETRY['floor_area']
)  # ~46 kW

# During unoccupied (just baseload equipment)
Q_INTERNAL_UNOCCUPIED = (
    0.1 * Q_INTERNAL_OCCUPIED
)  # ~4.6 kW

print(f"\nInternal gains:")
print(f"  Occupied hours: {Q_INTERNAL_OCCUPIED/1000:.1f} kW")
print(f"  Unoccupied hours: {Q_INTERNAL_UNOCCUPIED/1000:.1f} kW")
