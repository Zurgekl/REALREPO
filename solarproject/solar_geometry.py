"""
Solar Geometry Module for Building Energy Analysis
Based on NOAA Solar Calculator algorithms (Jean Meeus, Astronomical Algorithms)

Provides:
1. Solar position (elevation, azimuth) for any location/time
2. Facade irradiance for S/E/W/N orientations
3. Shading fraction calculations for overhangs and fins
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional

# ============================================================================
# CONSTANTS
# ============================================================================
DEG_TO_RAD = np.pi / 180
RAD_TO_DEG = 180 / np.pi

# Facade normal vectors (azimuth in degrees from North, clockwise)
FACADE_AZIMUTHS = {
    'south': 180,
    'north': 0,
    'east': 90,
    'west': 270,
}

# ============================================================================
# NOAA SOLAR POSITION ALGORITHMS
# ============================================================================

def julian_day(year: int, month: int, day: int, hour: float = 12.0) -> float:
    """
    Calculate Julian Day number for a given date and time.
    Based on Jean Meeus, Astronomical Algorithms.
    """
    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)

    JD = (int(365.25 * (year + 4716)) +
          int(30.6001 * (month + 1)) +
          day + hour/24.0 + B - 1524.5)

    return JD


def julian_century(JD: float) -> float:
    """Julian Century from J2000.0"""
    return (JD - 2451545.0) / 36525.0


def sun_geometric_mean_longitude(T: float) -> float:
    """Geometric mean longitude of the sun (degrees)"""
    L0 = 280.46646 + T * (36000.76983 + 0.0003032 * T)
    return L0 % 360


def sun_geometric_mean_anomaly(T: float) -> float:
    """Geometric mean anomaly of the sun (degrees)"""
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    return M


def earth_orbit_eccentricity(T: float) -> float:
    """Eccentricity of Earth's orbit"""
    return 0.016708634 - T * (0.000042037 + 0.0000001267 * T)


def sun_equation_of_center(T: float) -> float:
    """Sun's equation of center (degrees)"""
    M = sun_geometric_mean_anomaly(T)
    M_rad = M * DEG_TO_RAD

    C = (np.sin(M_rad) * (1.914602 - T * (0.004817 + 0.000014 * T)) +
         np.sin(2 * M_rad) * (0.019993 - 0.000101 * T) +
         np.sin(3 * M_rad) * 0.000289)

    return C


def sun_true_longitude(T: float) -> float:
    """Sun's true longitude (degrees)"""
    return sun_geometric_mean_longitude(T) + sun_equation_of_center(T)


def sun_apparent_longitude(T: float) -> float:
    """Sun's apparent longitude (degrees)"""
    O = sun_true_longitude(T)
    omega = 125.04 - 1934.136 * T
    return O - 0.00569 - 0.00478 * np.sin(omega * DEG_TO_RAD)


def mean_obliquity_of_ecliptic(T: float) -> float:
    """Mean obliquity of the ecliptic (degrees)"""
    seconds = 21.448 - T * (46.8150 + T * (0.00059 - T * 0.001813))
    return 23 + (26 + seconds / 60) / 60


def obliquity_correction(T: float) -> float:
    """Corrected obliquity of the ecliptic (degrees)"""
    e0 = mean_obliquity_of_ecliptic(T)
    omega = 125.04 - 1934.136 * T
    return e0 + 0.00256 * np.cos(omega * DEG_TO_RAD)


def sun_declination(T: float) -> float:
    """Sun's declination angle (degrees)"""
    e = obliquity_correction(T) * DEG_TO_RAD
    lambda_sun = sun_apparent_longitude(T) * DEG_TO_RAD

    sint = np.sin(e) * np.sin(lambda_sun)
    return np.arcsin(sint) * RAD_TO_DEG


def equation_of_time(T: float) -> float:
    """Equation of time (minutes)"""
    e0 = obliquity_correction(T)
    L0 = sun_geometric_mean_longitude(T)
    e = earth_orbit_eccentricity(T)
    M = sun_geometric_mean_anomaly(T)

    y = np.tan(e0 * DEG_TO_RAD / 2) ** 2

    sin2L0 = np.sin(2 * L0 * DEG_TO_RAD)
    sinM = np.sin(M * DEG_TO_RAD)
    cos2L0 = np.cos(2 * L0 * DEG_TO_RAD)
    sin4L0 = np.sin(4 * L0 * DEG_TO_RAD)
    sin2M = np.sin(2 * M * DEG_TO_RAD)

    Etime = (y * sin2L0 - 2 * e * sinM + 4 * e * y * sinM * cos2L0 -
             0.5 * y * y * sin4L0 - 1.25 * e * e * sin2M)

    return 4 * Etime * RAD_TO_DEG  # Convert to minutes


def solar_position(lat: float, lon: float, year: int, month: int, day: int,
                   hour: float, timezone_offset: float = 0) -> Tuple[float, float]:
    """
    Calculate solar elevation and azimuth for a given location and time.

    Parameters:
        lat: Latitude (degrees, positive North)
        lon: Longitude (degrees, positive East)
        year, month, day: Date
        hour: Local time (0-24, decimal hours)
        timezone_offset: Hours from UTC (e.g., -5 for EST)

    Returns:
        elevation: Solar altitude angle (degrees, 0=horizon, 90=zenith)
        azimuth: Solar azimuth (degrees, 0=North, 90=East, 180=South, 270=West)
    """
    # Convert local time to UTC
    hour_utc = hour - timezone_offset

    # Handle day rollover
    day_offset = 0
    if hour_utc >= 24:
        hour_utc -= 24
        day_offset = 1
    elif hour_utc < 0:
        hour_utc += 24
        day_offset = -1

    # Adjust date if needed (simplified - doesn't handle month boundaries)
    day_adj = day + day_offset

    # Julian calculations
    JD = julian_day(year, month, day_adj, hour_utc)
    T = julian_century(JD)

    # Solar parameters
    decl = sun_declination(T)
    eqtime = equation_of_time(T)

    # Solar time
    time_offset = eqtime + 4 * lon - 60 * timezone_offset  # minutes
    true_solar_time = hour * 60 + time_offset

    # Hour angle
    hour_angle = true_solar_time / 4 - 180  # degrees
    if hour_angle < -180:
        hour_angle += 360
    elif hour_angle > 180:
        hour_angle -= 360

    # Convert to radians
    lat_rad = lat * DEG_TO_RAD
    decl_rad = decl * DEG_TO_RAD
    ha_rad = hour_angle * DEG_TO_RAD

    # Solar zenith angle
    cos_zenith = (np.sin(lat_rad) * np.sin(decl_rad) +
                  np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad))
    cos_zenith = np.clip(cos_zenith, -1, 1)
    zenith = np.arccos(cos_zenith)

    # Solar elevation
    elevation = 90 - zenith * RAD_TO_DEG

    # Solar azimuth
    if cos_zenith != 0:
        cos_azimuth = ((np.sin(lat_rad) * np.cos(zenith) - np.sin(decl_rad)) /
                       (np.cos(lat_rad) * np.sin(zenith)))
        cos_azimuth = np.clip(cos_azimuth, -1, 1)
        azimuth = np.arccos(cos_azimuth) * RAD_TO_DEG

        if hour_angle > 0:
            azimuth = 360 - azimuth
    else:
        azimuth = 180 if lat > decl else 0

    return elevation, azimuth


def get_hourly_solar_position(lat: float, lon: float, year: int, month: int, day: int,
                               timezone_offset: float = 0) -> Dict[str, np.ndarray]:
    """
    Calculate solar position for each hour of a day.

    Returns:
        Dictionary with 'hours', 'elevation', 'azimuth' arrays
    """
    hours = np.arange(24)
    elevations = np.zeros(24)
    azimuths = np.zeros(24)

    for hr in hours:
        elev, azi = solar_position(lat, lon, year, month, day, hr + 0.5, timezone_offset)
        elevations[hr] = elev
        azimuths[hr] = azi

    return {
        'hours': hours,
        'elevation': elevations,
        'azimuth': azimuths
    }


# ============================================================================
# FACADE IRRADIANCE CALCULATIONS
# ============================================================================

def sun_direction_vector(elevation: float, azimuth: float) -> np.ndarray:
    """
    Convert solar elevation and azimuth to a unit direction vector.

    Vector points FROM sun TO earth surface (i.e., direction of rays).
    Coordinate system: x=East, y=North, z=Up

    Returns:
        3D unit vector [x, y, z]
    """
    elev_rad = elevation * DEG_TO_RAD
    azi_rad = azimuth * DEG_TO_RAD

    # Horizontal component magnitude
    cos_elev = np.cos(elev_rad)

    # Direction components (sun rays coming down)
    x = -cos_elev * np.sin(azi_rad)  # East component
    y = -cos_elev * np.cos(azi_rad)  # North component
    z = -np.sin(elev_rad)            # Up component (negative = downward)

    return np.array([x, y, z])


def facade_normal_vector(facade_azimuth: float) -> np.ndarray:
    """
    Get outward-facing normal vector for a vertical facade.

    Parameters:
        facade_azimuth: Direction facade faces (degrees from North)

    Returns:
        3D unit vector [x, y, z]
    """
    azi_rad = facade_azimuth * DEG_TO_RAD

    # Outward normal (horizontal, no z component)
    x = np.sin(azi_rad)   # East component
    y = np.cos(azi_rad)   # North component
    z = 0                 # Vertical facade

    return np.array([x, y, z])


def incidence_angle_cosine(elevation: float, azimuth: float,
                            facade_azimuth: float) -> float:
    """
    Calculate cosine of incidence angle between sun and facade normal.

    Returns:
        cos(theta_i), clamped to [0, 1] (0 if sun behind facade)
    """
    if elevation <= 0:
        return 0.0  # Sun below horizon

    # Sun direction (pointing toward surface, reversed from ray direction)
    sun_vec = -sun_direction_vector(elevation, azimuth)

    # Facade normal
    facade_normal = facade_normal_vector(facade_azimuth)

    # Dot product gives cosine of angle
    cos_theta = np.dot(sun_vec, facade_normal)

    return max(0.0, cos_theta)


def facade_irradiance(elevation: float, azimuth: float,
                      DNI: float, DHI: float,
                      facade_azimuth: float,
                      ground_reflectance: float = 0.2) -> float:
    """
    Calculate total irradiance on a vertical facade.

    Parameters:
        elevation: Solar elevation (degrees)
        azimuth: Solar azimuth (degrees)
        DNI: Direct Normal Irradiance (W/m²)
        DHI: Diffuse Horizontal Irradiance (W/m²)
        facade_azimuth: Direction facade faces (degrees from North)
        ground_reflectance: Albedo for ground-reflected component

    Returns:
        Total irradiance on facade (W/m²)
    """
    if elevation <= 0:
        # Sun below horizon - only diffuse (reduced)
        return 0.5 * DHI  # Approximate sky dome contribution

    # Beam (direct) component
    cos_theta = incidence_angle_cosine(elevation, azimuth, facade_azimuth)
    I_beam = DNI * cos_theta

    # Diffuse component (isotropic sky model for vertical surface)
    # Vertical surface sees half the sky dome
    I_diffuse = DHI * 0.5

    # Ground-reflected component
    GHI = DNI * np.sin(elevation * DEG_TO_RAD) + DHI
    I_ground = GHI * ground_reflectance * 0.5  # Vertical sees half of ground

    return I_beam + I_diffuse + I_ground


def get_all_facade_irradiances(elevation: float, azimuth: float,
                                DNI: float, DHI: float) -> Dict[str, float]:
    """
    Calculate irradiance on all four cardinal facades.

    Returns:
        Dictionary with 'south', 'north', 'east', 'west' irradiances (W/m²)
    """
    result = {}
    for facade, facade_azi in FACADE_AZIMUTHS.items():
        result[facade] = facade_irradiance(elevation, azimuth, DNI, DHI, facade_azi)

    return result


# ============================================================================
# CLEAR SKY IRRADIANCE MODEL (Simple)
# ============================================================================

def clear_sky_irradiance(elevation: float,
                          I0: float = 1361,
                          atmospheric_transmittance: float = 0.7) -> Tuple[float, float]:
    """
    Simple clear-sky model for DNI and DHI.

    Parameters:
        elevation: Solar elevation angle (degrees)
        I0: Solar constant (W/m²)
        atmospheric_transmittance: Clear-sky transmittance factor

    Returns:
        DNI: Direct Normal Irradiance (W/m²)
        DHI: Diffuse Horizontal Irradiance (W/m²)
    """
    if elevation <= 0:
        return 0.0, 0.0

    # Air mass (simplified Kasten-Young)
    zenith = 90 - elevation
    if zenith < 90:
        AM = 1 / (np.cos(zenith * DEG_TO_RAD) +
                  0.50572 * (96.07995 - zenith) ** (-1.6364))
    else:
        AM = 40  # Near horizon

    # DNI with Beer-Lambert attenuation
    DNI = I0 * (atmospheric_transmittance ** AM)

    # DHI as fraction of extraterrestrial on horizontal
    # (simplified - typically 10-20% of GHI is diffuse on clear days)
    GHI_clear = DNI * np.sin(elevation * DEG_TO_RAD)
    DHI = 0.15 * GHI_clear  # Approximate diffuse fraction

    return DNI, DHI


# ============================================================================
# SHADING GEOMETRY CALCULATIONS
# ============================================================================

def overhang_shading_fraction(elevation: float, azimuth: float,
                               overhang_depth: float,
                               window_height: float,
                               window_top_to_overhang: float = 0.0,
                               facade_azimuth: float = 180) -> float:
    """
    Calculate shading fraction from horizontal overhang on a vertical window.

    Geometry:
        - Overhang projects horizontally from wall above window
        - Sun casts shadow down onto window

    Parameters:
        elevation: Solar elevation (degrees)
        azimuth: Solar azimuth (degrees)
        overhang_depth: Horizontal projection of overhang (m)
        window_height: Height of window (m)
        window_top_to_overhang: Gap between window top and overhang (m)
        facade_azimuth: Direction facade faces (degrees from North, default=South)

    Returns:
        Shading fraction (0 = no shade, 1 = fully shaded)
    """
    if elevation <= 0:
        return 0.0  # No direct sun

    # Check if sun is in front of facade (within ~90 degrees)
    relative_azimuth = abs(azimuth - facade_azimuth)
    if relative_azimuth > 180:
        relative_azimuth = 360 - relative_azimuth

    if relative_azimuth > 90:
        return 0.0  # Sun behind facade

    # Shadow projection
    # Adjust for sun angle relative to facade normal
    cos_relative = np.cos(relative_azimuth * DEG_TO_RAD)
    if cos_relative <= 0:
        return 0.0

    # Effective overhang depth in plane perpendicular to facade
    effective_depth = overhang_depth / cos_relative

    # Vertical shadow length from overhang
    tan_elev = np.tan(elevation * DEG_TO_RAD)
    if tan_elev <= 0:
        return 0.0

    shadow_length = effective_depth / tan_elev

    # How much of window is shaded (from top down)
    shaded_height = shadow_length - window_top_to_overhang
    shaded_height = np.clip(shaded_height, 0, window_height)

    shading_fraction = shaded_height / window_height

    return shading_fraction


def vertical_fin_shading_fraction(elevation: float, azimuth: float,
                                   fin_depth: float,
                                   window_width: float,
                                   fin_spacing: float,
                                   facade_azimuth: float = 90) -> float:
    """
    Calculate shading fraction from vertical fins on a window.

    Geometry:
        - Vertical fins project horizontally from wall on sides of windows
        - Sun casts horizontal shadow across window

    Parameters:
        elevation: Solar elevation (degrees)
        azimuth: Solar azimuth (degrees)
        fin_depth: Horizontal projection of fins (m)
        window_width: Width of window between fins (m)
        fin_spacing: Center-to-center spacing of fins (m)
        facade_azimuth: Direction facade faces (degrees from North)

    Returns:
        Shading fraction (0 = no shade, 1 = fully shaded)
    """
    if elevation <= 0:
        return 0.0  # No direct sun

    # Relative azimuth angle
    relative_azimuth = azimuth - facade_azimuth

    # Normalize to -180 to 180
    while relative_azimuth > 180:
        relative_azimuth -= 360
    while relative_azimuth < -180:
        relative_azimuth += 360

    # If sun is behind facade or directly in front, minimal fin shading
    if abs(relative_azimuth) > 90:
        return 0.0
    if abs(relative_azimuth) < 5:
        return 0.0  # Sun perpendicular to facade

    # Horizontal shadow projection
    tan_rel_azi = np.tan(abs(relative_azimuth) * DEG_TO_RAD)
    shadow_width = fin_depth * tan_rel_azi

    # Fraction of window width shaded
    shading_fraction = shadow_width / window_width
    shading_fraction = np.clip(shading_fraction, 0, 1)

    return shading_fraction


def get_shading_fractions(elevation: float, azimuth: float,
                           overhang_depth_south: float = 0,
                           fin_depth_east: float = 0,
                           fin_depth_west: float = 0,
                           window_height: float = 1.5,
                           window_width: float = 1.5,
                           window_top_gap: float = 0.3) -> Dict[str, float]:
    """
    Calculate shading fractions for all facades.

    Parameters:
        elevation, azimuth: Solar position
        overhang_depth_south: South overhang projection (m)
        fin_depth_east: East fin projection (m)
        fin_depth_west: West fin projection (m)
        window_height: Typical window height (m)
        window_width: Typical window width (m)
        window_top_gap: Gap from window top to overhang (m)

    Returns:
        Dictionary with shading fractions for each facade
    """
    return {
        'south': overhang_shading_fraction(
            elevation, azimuth, overhang_depth_south,
            window_height, window_top_gap, 180
        ),
        'north': 0.0,  # North typically unshaded (no direct sun in northern hemisphere)
        'east': vertical_fin_shading_fraction(
            elevation, azimuth, fin_depth_east,
            window_width, window_width * 1.5, 90
        ),
        'west': vertical_fin_shading_fraction(
            elevation, azimuth, fin_depth_west,
            window_width, window_width * 1.5, 270
        ),
    }


# ============================================================================
# MAIN INTERFACE: HOURLY FACADE SOLAR DATA
# ============================================================================

def compute_hourly_facade_solar(lat: float, lon: float,
                                 year: int, month: int, day: int,
                                 timezone_offset: float = -5,
                                 cloud_factor: float = 1.0,
                                 overhang_depth_south: float = 0,
                                 fin_depth_east: float = 0,
                                 fin_depth_west: float = 0) -> Dict[str, np.ndarray]:
    """
    Compute complete hourly facade solar data for a given day.

    This is the main interface for the thermal model.

    Parameters:
        lat, lon: Location (degrees)
        year, month, day: Date
        timezone_offset: Hours from UTC (default -5 for Miami EST)
        cloud_factor: Multiplier for irradiance (1.0 = clear, 0.5 = partly cloudy)
        overhang_depth_south: South overhang depth (m)
        fin_depth_east: East vertical fin depth (m)
        fin_depth_west: West vertical fin depth (m)

    Returns:
        Dictionary with arrays for each quantity:
            - hours: Hour indices (0-23)
            - elevation: Solar elevation (degrees)
            - azimuth: Solar azimuth (degrees)
            - I_south, I_north, I_east, I_west: Facade irradiances (W/m²)
            - f_shade_south, f_shade_east, f_shade_west: Shading fractions
    """
    hours = np.arange(24)
    n_hours = len(hours)

    # Initialize output arrays
    result = {
        'hours': hours,
        'elevation': np.zeros(n_hours),
        'azimuth': np.zeros(n_hours),
        'I_south': np.zeros(n_hours),
        'I_north': np.zeros(n_hours),
        'I_east': np.zeros(n_hours),
        'I_west': np.zeros(n_hours),
        'f_shade_south': np.zeros(n_hours),
        'f_shade_east': np.zeros(n_hours),
        'f_shade_west': np.zeros(n_hours),
    }

    for i, hr in enumerate(hours):
        # Solar position (use middle of hour)
        elev, azi = solar_position(lat, lon, year, month, day, hr + 0.5, timezone_offset)
        result['elevation'][i] = elev
        result['azimuth'][i] = azi

        # Clear sky irradiance
        DNI, DHI = clear_sky_irradiance(elev)

        # Apply cloud factor
        DNI *= cloud_factor
        DHI *= cloud_factor

        # Facade irradiances
        irradiances = get_all_facade_irradiances(elev, azi, DNI, DHI)
        result['I_south'][i] = irradiances['south']
        result['I_north'][i] = irradiances['north']
        result['I_east'][i] = irradiances['east']
        result['I_west'][i] = irradiances['west']

        # Shading fractions
        shading = get_shading_fractions(
            elev, azi,
            overhang_depth_south=overhang_depth_south,
            fin_depth_east=fin_depth_east,
            fin_depth_west=fin_depth_west
        )
        result['f_shade_south'][i] = shading['south']
        result['f_shade_east'][i] = shading['east']
        result['f_shade_west'][i] = shading['west']

    return result


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Miami, FL coordinates
    MIAMI_LAT = 25.7617
    MIAMI_LON = -80.1918
    TZ_OFFSET = -5  # EST (use -4 for EDT)

    # Test dates: summer solstice, winter solstice, equinox
    test_dates = [
        (2024, 6, 21, "Summer Solstice"),
        (2024, 12, 21, "Winter Solstice"),
        (2024, 3, 20, "Spring Equinox"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    for row, (year, month, day, label) in enumerate(test_dates):
        # Compute solar data
        solar_data = compute_hourly_facade_solar(
            MIAMI_LAT, MIAMI_LON, year, month, day,
            timezone_offset=TZ_OFFSET,
            cloud_factor=0.85,  # Slight clouds
            overhang_depth_south=1.0,  # 1m overhang
            fin_depth_east=0.5,
            fin_depth_west=0.5
        )

        hours = solar_data['hours']

        # Plot 1: Solar position
        ax1 = axes[row, 0]
        ax1.plot(hours, solar_data['elevation'], 'r-', label='Elevation', linewidth=2)
        ax1.plot(hours, solar_data['azimuth']/4, 'b--', label='Azimuth/4', linewidth=2)
        ax1.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax1.set_ylabel('Degrees')
        ax1.set_title(f'{label}: Solar Position')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 23)

        # Plot 2: Facade irradiances
        ax2 = axes[row, 1]
        ax2.plot(hours, solar_data['I_south'], 'r-', label='South', linewidth=2)
        ax2.plot(hours, solar_data['I_east'], 'orange', label='East', linewidth=2)
        ax2.plot(hours, solar_data['I_west'], 'purple', label='West', linewidth=2)
        ax2.plot(hours, solar_data['I_north'], 'b--', label='North', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Irradiance (W/m²)')
        ax2.set_title(f'{label}: Facade Irradiance')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 23)

        # Plot 3: Shading fractions
        ax3 = axes[row, 2]
        ax3.plot(hours, solar_data['f_shade_south'], 'r-', label='South (overhang)', linewidth=2)
        ax3.plot(hours, solar_data['f_shade_east'], 'orange', label='East (fins)', linewidth=2)
        ax3.plot(hours, solar_data['f_shade_west'], 'purple', label='West (fins)', linewidth=2)
        ax3.set_ylabel('Shading Fraction')
        ax3.set_title(f'{label}: Shading')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 23)
        ax3.set_ylim(0, 1)

        if row == 2:
            ax1.set_xlabel('Hour of Day')
            ax2.set_xlabel('Hour of Day')
            ax3.set_xlabel('Hour of Day')

    plt.tight_layout()
    plt.savefig('solar_geometry_test.png', dpi=150)
    print("Solar geometry test complete. See solar_geometry_test.png")

    # Print summary for summer solstice
    print("\n" + "="*60)
    print("SUMMER SOLSTICE - MIAMI, FL")
    print("="*60)

    solar_summer = compute_hourly_facade_solar(
        MIAMI_LAT, MIAMI_LON, 2024, 6, 21,
        timezone_offset=TZ_OFFSET,
        overhang_depth_south=1.0
    )

    # Find solar noon (max elevation)
    noon_idx = np.argmax(solar_summer['elevation'])
    print(f"\nSolar noon: ~{solar_summer['hours'][noon_idx]}:30")
    print(f"  Max elevation: {solar_summer['elevation'][noon_idx]:.1f}°")
    print(f"  Azimuth at noon: {solar_summer['azimuth'][noon_idx]:.1f}°")

    # Peak irradiances
    print(f"\nPeak facade irradiances:")
    print(f"  South: {max(solar_summer['I_south']):.0f} W/m² at hour {np.argmax(solar_summer['I_south'])}")
    print(f"  East:  {max(solar_summer['I_east']):.0f} W/m² at hour {np.argmax(solar_summer['I_east'])}")
    print(f"  West:  {max(solar_summer['I_west']):.0f} W/m² at hour {np.argmax(solar_summer['I_west'])}")

    # Shading effectiveness
    print(f"\nSouth overhang (1.0m) shading at solar noon: {solar_summer['f_shade_south'][noon_idx]*100:.0f}%")
