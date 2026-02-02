"""
Solar Geometry Validation Script
Checks geometric sanity and identifies any convention errors
"""

import numpy as np
from solar_geometry import (
    solar_position, compute_hourly_facade_solar,
    incidence_angle_cosine, clear_sky_irradiance,
    overhang_shading_fraction, vertical_fin_shading_fraction
)

MIAMI_LAT = 25.7617
MIAMI_LON = -80.1918
TZ_OFFSET = -5

print("="*70)
print("SOLAR GEOMETRY VALIDATION - MIAMI, FL")
print("="*70)

# ============================================================================
# TEST A: Geometric Sanity (azimuth at moderate elevations)
# ============================================================================
print("\n" + "="*70)
print("TEST A: AZIMUTH VALIDATION (at moderate sun angles)")
print("="*70)

test_cases = [
    # (year, month, day, hour, description, expected_azimuth_range)
    (2024, 12, 21, 12.5, "Winter Solstice Noon", (170, 190)),  # Should be ~180 (due south)
    (2024, 6, 21, 9.0, "Summer Solstice 9AM", (70, 110)),      # Should be ~90 (east-ish)
    (2024, 6, 21, 15.0, "Summer Solstice 3PM", (250, 290)),    # Should be ~270 (west-ish)
    (2024, 3, 20, 12.5, "Spring Equinox Noon", (170, 190)),    # Should be ~180 (due south)
]

print(f"\n{'Date/Time':<30} {'Elev':>8} {'Azimuth':>8} {'Expected':>15} {'Status':>10}")
print("-"*75)

for year, month, day, hour, desc, (azi_min, azi_max) in test_cases:
    elev, azi = solar_position(MIAMI_LAT, MIAMI_LON, year, month, day, hour, TZ_OFFSET)
    in_range = azi_min <= azi <= azi_max
    status = "OK" if in_range else "FAIL"
    print(f"{desc:<30} {elev:>7.1f}° {azi:>7.1f}° {azi_min}-{azi_max}° {status:>10}")

# ============================================================================
# TEST B: Facade Irradiance Patterns (qualitative)
# ============================================================================
print("\n" + "="*70)
print("TEST B: FACADE IRRADIANCE PATTERNS")
print("="*70)

print("\n--- Summer Solstice (June 21) ---")
summer_data = compute_hourly_facade_solar(
    MIAMI_LAT, MIAMI_LON, 2024, 6, 21,
    timezone_offset=TZ_OFFSET, cloud_factor=1.0
)

for hr in [9, 12, 15]:
    idx = hr
    print(f"\nHour {hr}:00 - Elevation: {summer_data['elevation'][idx]:.1f}°, Azimuth: {summer_data['azimuth'][idx]:.1f}°")
    print(f"  South: {summer_data['I_south'][idx]:>6.0f} W/m²")
    print(f"  East:  {summer_data['I_east'][idx]:>6.0f} W/m²")
    print(f"  West:  {summer_data['I_west'][idx]:>6.0f} W/m²")
    print(f"  North: {summer_data['I_north'][idx]:>6.0f} W/m²")

# Check expected patterns
print("\n  Expected patterns:")
# More relaxed check - just verify the dominant facade
print("    9AM: East should dominate - " + ("✓" if summer_data['I_east'][9] > summer_data['I_west'][9] else "FAIL"))
print("    3PM: West should dominate - " + ("✓" if summer_data['I_west'][15] > summer_data['I_east'][15] else "FAIL"))
print("    Noon: South should be LOW (sun high) - South={:.0f}, E/W avg={:.0f}".format(
    summer_data['I_south'][12],
    (summer_data['I_east'][12] + summer_data['I_west'][12])/2
))

print("\n--- Winter Solstice (Dec 21) ---")
winter_data = compute_hourly_facade_solar(
    MIAMI_LAT, MIAMI_LON, 2024, 12, 21,
    timezone_offset=TZ_OFFSET, cloud_factor=1.0
)

for hr in [9, 12, 15]:
    idx = hr
    print(f"\nHour {hr}:00 - Elevation: {winter_data['elevation'][idx]:.1f}°, Azimuth: {winter_data['azimuth'][idx]:.1f}°")
    print(f"  South: {winter_data['I_south'][idx]:>6.0f} W/m²")
    print(f"  East:  {winter_data['I_east'][idx]:>6.0f} W/m²")
    print(f"  West:  {winter_data['I_west'][idx]:>6.0f} W/m²")
    print(f"  North: {winter_data['I_north'][idx]:>6.0f} W/m²")

print("\n  Expected pattern:")
print("    Winter Noon: South should DOMINATE - South={:.0f}, E={:.0f}, W={:.0f}".format(
    winter_data['I_south'][12],
    winter_data['I_east'][12],
    winter_data['I_west'][12]
))
south_dominates = winter_data['I_south'][12] > winter_data['I_east'][12] and winter_data['I_south'][12] > winter_data['I_west'][12]
print("    " + ("South dominates at winter noon ✓" if south_dominates else "VIOLATION: South should be highest!"))

# ============================================================================
# TEST C: Shading Geometry Debug
# ============================================================================
print("\n" + "="*70)
print("TEST C: SHADING GEOMETRY ANALYSIS")
print("="*70)

print("\n--- South Overhang (1.0m depth) ---")
print(f"{'Hour':>6} {'Elev':>8} {'Azimuth':>8} {'Shade%':>10} {'Beam on S':>12}")
for hr in range(6, 20):
    elev = summer_data['elevation'][hr]
    azi = summer_data['azimuth'][hr]

    shade_frac = overhang_shading_fraction(
        elev, azi,
        overhang_depth=1.0,
        window_height=1.5,
        window_top_to_overhang=0.3,
        facade_azimuth=180
    )

    # Compute beam component on south facade
    cos_theta = incidence_angle_cosine(elev, azi, 180)
    DNI, _ = clear_sky_irradiance(elev)
    beam_south = DNI * cos_theta

    print(f"{hr:>6} {elev:>7.1f}° {azi:>7.1f}° {shade_frac*100:>9.0f}% {beam_south:>10.0f} W/m²")

print("\n--- East Fins (0.5m depth) ---")
print(f"{'Hour':>6} {'Elev':>8} {'Azimuth':>8} {'Rel.Azi':>10} {'Shade%':>10} {'Beam on E':>12}")
for hr in range(6, 14):  # Morning hours
    elev = summer_data['elevation'][hr]
    azi = summer_data['azimuth'][hr]
    rel_azi = azi - 90  # Relative to east facade

    shade_frac = vertical_fin_shading_fraction(
        elev, azi,
        fin_depth=0.5,
        window_width=1.5,
        fin_spacing=2.25,
        facade_azimuth=90
    )

    cos_theta = incidence_angle_cosine(elev, azi, 90)
    DNI, _ = clear_sky_irradiance(elev)
    beam_east = DNI * cos_theta

    print(f"{hr:>6} {elev:>7.1f}° {azi:>7.1f}° {rel_azi:>9.1f}° {shade_frac*100:>9.0f}% {beam_east:>10.0f} W/m²")

print("\n--- West Fins (0.5m depth) ---")
print(f"{'Hour':>6} {'Elev':>8} {'Azimuth':>8} {'Rel.Azi':>10} {'Shade%':>10} {'Beam on W':>12}")
for hr in range(12, 20):  # Afternoon hours
    elev = summer_data['elevation'][hr]
    azi = summer_data['azimuth'][hr]
    rel_azi = azi - 270  # Relative to west facade

    shade_frac = vertical_fin_shading_fraction(
        elev, azi,
        fin_depth=0.5,
        window_width=1.5,
        fin_spacing=2.25,
        facade_azimuth=270
    )

    cos_theta = incidence_angle_cosine(elev, azi, 270)
    DNI, _ = clear_sky_irradiance(elev)
    beam_west = DNI * cos_theta

    print(f"{hr:>6} {elev:>7.1f}° {azi:>7.1f}° {rel_azi:>9.1f}° {shade_frac*100:>9.0f}% {beam_west:>10.0f} W/m²")

# ============================================================================
# TEST D: Seasonal Shading Comparison
# ============================================================================
print("\n" + "="*70)
print("TEST D: SEASONAL SHADING EFFECTIVENESS")
print("="*70)

print("\nSouth overhang (1.0m) shading at solar noon:")
for date_label, data in [("Summer (Jun 21)", summer_data), ("Winter (Dec 21)", winter_data)]:
    noon_idx = 12
    elev = data['elevation'][noon_idx]
    azi = data['azimuth'][noon_idx]
    shade = overhang_shading_fraction(elev, azi, 1.0, 1.5, 0.3, 180)
    I_south = data['I_south'][noon_idx]
    print(f"  {date_label}: Elevation={elev:.1f}°, Shading={shade*100:.0f}%, I_south={I_south:.0f} W/m²")

print("\n  Note: In Miami summer, noon sun is nearly overhead (87°) so:")
print("    - Very little beam hits vertical south facade anyway")
print("    - Overhang shading is low because there's little to shade")
print("  In winter, sun is lower (41°), hits south facade directly")
print("    - Overhang provides more shading, but we WANT winter solar gain!")
print("  This is why overhangs are designed for shoulder seasons, not extremes.")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

issues = []

# Check azimuth at winter noon
_, winter_noon_azi = solar_position(MIAMI_LAT, MIAMI_LON, 2024, 12, 21, 12.5, TZ_OFFSET)
if not (170 <= winter_noon_azi <= 190):
    issues.append(f"Winter noon azimuth = {winter_noon_azi:.1f}° (expected ~180°)")

# Check south dominance in winter
if not south_dominates:
    issues.append("South facade doesn't dominate at winter noon")

# Check E/W pattern in summer
if not (summer_data['I_east'][9] > summer_data['I_west'][9]):
    issues.append("East not higher than West at 9AM summer")
if not (summer_data['I_west'][15] > summer_data['I_east'][15]):
    issues.append("West not higher than East at 3PM summer")

if issues:
    print("\nISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\n✓ All basic validation checks passed!")

print("\nNote: Check the fin shading percentages above - if beam is high but")
print("shading stays low, the fin geometry model may need adjustment.")
print("="*70)
