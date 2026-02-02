"""
Test thermal model with real solar geometry calculations
Uses solar_geometry.py for accurate sun position and facade irradiance
"""

import numpy as np
import matplotlib.pyplot as plt
from thermal_model_v1 import thermal_model_step
import building_params as bp
import miami_climate_params as mcp
from solar_geometry import compute_hourly_facade_solar

# ============================================================================
# LOCATION AND DATE
# ============================================================================
MIAMI_LAT = 25.7617
MIAMI_LON = -80.1918
TZ_OFFSET = -5  # EST

# Test on summer solstice (worst case for cooling)
YEAR, MONTH, DAY = 2024, 6, 21

# ============================================================================
# WEATHER FORCING
# ============================================================================
hours = np.arange(24)

# Outdoor temperature (sinusoidal approximation of summer day)
T_out_mean = 30  # °C (86°F)
T_out_swing = 6  # °C amplitude
T_out = T_out_mean + T_out_swing * np.sin(2*np.pi*(hours - 9)/24)

# ============================================================================
# SOLAR GEOMETRY: Compute facade irradiances and shading
# ============================================================================
print("Computing solar geometry for Miami, FL - Summer Solstice...")

# Baseline: no shading devices
solar_baseline = compute_hourly_facade_solar(
    MIAMI_LAT, MIAMI_LON, YEAR, MONTH, DAY,
    timezone_offset=TZ_OFFSET,
    cloud_factor=0.7,  # Partly cloudy
    overhang_depth_south=0.0,  # No overhang
    fin_depth_east=0.0,
    fin_depth_west=0.0
)

# With shading: 1.0m south overhang + 0.5m E/W fins
solar_shaded = compute_hourly_facade_solar(
    MIAMI_LAT, MIAMI_LON, YEAR, MONTH, DAY,
    timezone_offset=TZ_OFFSET,
    cloud_factor=0.7,
    overhang_depth_south=1.0,  # 1m overhang
    fin_depth_east=0.5,
    fin_depth_west=0.5
)

# ============================================================================
# INTERNAL GAINS SCHEDULE
# ============================================================================
Q_internal = np.where(
    (hours >= 8) & (hours <= 17),
    bp.Q_INTERNAL_OCCUPIED,
    bp.Q_INTERNAL_UNOCCUPIED
)

# ============================================================================
# HELPER: Run simulation for all facades
# ============================================================================
def run_simulation_all_facades(solar_data, use_shading=False):
    """
    Run thermal model accounting for all facades.

    Returns lists of results for each hour.
    """
    T_free_list = [mcp.COMFORT['summer_comfort_upper']]
    T_ctrl_list = [mcp.COMFORT['summer_comfort_upper']]
    Q_cooling_list = []
    Q_solar_total_list = []
    Q_solar_by_facade = {'south': [], 'north': [], 'east': [], 'west': []}

    for hr in range(24):
        # Calculate solar heat gain for each facade
        Q_solar_facades = {}
        for facade in ['south', 'north', 'east', 'west']:
            I_facade = solar_data[f'I_{facade}'][hr]
            A_win = bp.WINDOW_AREAS[facade]

            # Apply shading fraction (only for S, E, W)
            if use_shading and facade in ['south', 'east', 'west']:
                f_shade = solar_data[f'f_shade_{facade}'][hr]
            else:
                f_shade = 0.0

            Q_solar_facades[facade] = A_win * bp.ENVELOPE_BASELINE['SHGC'] * I_facade * (1 - f_shade)
            Q_solar_by_facade[facade].append(Q_solar_facades[facade])

        # Total solar gain from all facades
        Q_solar_total = sum(Q_solar_facades.values())
        Q_solar_total_list.append(Q_solar_total)

        # Envelope conduction
        Q_envelope = bp.UA_ENVELOPE * (T_out[hr] - T_ctrl_list[-1])

        # Total heat input
        Q_total = Q_solar_total + Q_internal[hr] + Q_envelope

        # Temperature change (free-float)
        dt = 3600  # 1 hour
        dT_free = (Q_total * dt) / bp.C_EFFECTIVE
        T_free = T_ctrl_list[-1] + dT_free
        T_free_list.append(T_free)

        # Controlled temperature
        T_comfort_max = mcp.COMFORT['summer_comfort_upper']
        if T_free > T_comfort_max:
            Q_cooling = (T_free - T_comfort_max) * bp.C_EFFECTIVE / dt
            T_ctrl = T_comfort_max
        else:
            Q_cooling = 0
            T_ctrl = T_free

        T_ctrl_list.append(T_ctrl)
        Q_cooling_list.append(Q_cooling)

    return {
        'T_free': T_free_list,
        'T_ctrl': T_ctrl_list,
        'Q_cooling': Q_cooling_list,
        'Q_solar_total': Q_solar_total_list,
        'Q_solar_by_facade': Q_solar_by_facade
    }

# ============================================================================
# RUN SIMULATIONS
# ============================================================================
print("Running baseline simulation (no shading)...")
results_baseline = run_simulation_all_facades(solar_baseline, use_shading=False)

print("Running shaded simulation (1m overhang + 0.5m fins)...")
results_shaded = run_simulation_all_facades(solar_shaded, use_shading=True)

# ============================================================================
# PLOTTING
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Solar Position
ax1 = axes[0, 0]
ax1.plot(hours, solar_baseline['elevation'], 'r-', label='Elevation', linewidth=2)
ax1.fill_between(hours, 0, solar_baseline['elevation'],
                  where=solar_baseline['elevation']>0, alpha=0.2, color='yellow')
ax1.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax1.set_ylabel('Solar Elevation (°)', fontsize=11)
ax1.set_xlabel('Hour of Day', fontsize=11)
ax1.set_title('Solar Position - Miami, Summer Solstice')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 23)

# Plot 2: Facade Irradiances
ax2 = axes[0, 1]
ax2.plot(hours, solar_baseline['I_south'], 'r-', label='South', linewidth=2)
ax2.plot(hours, solar_baseline['I_east'], 'orange', label='East', linewidth=2)
ax2.plot(hours, solar_baseline['I_west'], 'purple', label='West', linewidth=2)
ax2.plot(hours, solar_baseline['I_north'], 'b--', label='North', linewidth=1.5, alpha=0.7)
ax2.set_ylabel('Irradiance (W/m²)', fontsize=11)
ax2.set_xlabel('Hour of Day', fontsize=11)
ax2.set_title('Facade Irradiance (before shading)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 23)

# Plot 3: Temperature
ax3 = axes[1, 0]
ax3.plot(hours, T_out, 'r--', label='Outdoor', linewidth=2)
ax3.plot(hours, results_baseline['T_ctrl'][:-1], 'b-', label='Controlled (no shade)', linewidth=2)
ax3.plot(hours, results_shaded['T_ctrl'][:-1], 'g-', label='Controlled (shaded)', linewidth=2)
ax3.plot(hours, results_baseline['T_free'][:-1], 'b:', label='Free-float (no shade)', linewidth=1.5, alpha=0.6)
ax3.plot(hours, results_shaded['T_free'][:-1], 'g:', label='Free-float (shaded)', linewidth=1.5, alpha=0.6)
ax3.axhline(mcp.COMFORT['summer_comfort_upper'], color='k', linestyle=':',
            label='Comfort limit', linewidth=1.5)
ax3.set_ylabel('Temperature (°C)', fontsize=11)
ax3.set_xlabel('Hour of Day', fontsize=11)
ax3.set_title('Indoor Temperature')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 23)

# Plot 4: Cooling Power
ax4 = axes[1, 1]
ax4.plot(hours, np.array(results_baseline['Q_cooling'])/1000, 'b-',
         label='No shading', linewidth=2)
ax4.plot(hours, np.array(results_shaded['Q_cooling'])/1000, 'g-',
         label='With shading', linewidth=2)
ax4.fill_between(hours,
                  np.array(results_shaded['Q_cooling'])/1000,
                  np.array(results_baseline['Q_cooling'])/1000,
                  alpha=0.3, color='green', label='Savings')
ax4.set_ylabel('Cooling Power (kW)', fontsize=11)
ax4.set_xlabel('Hour of Day', fontsize=11)
ax4.set_title('Cooling Power Required')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 23)

plt.tight_layout()
plt.savefig('test_with_solar_geometry.png', dpi=150)
print("\nPlot saved: test_with_solar_geometry.png")

# ============================================================================
# DETAILED METRICS
# ============================================================================
print(f"\n{'='*70}")
print(f"SUMMER SOLSTICE PERFORMANCE SUMMARY - MIAMI, FL")
print(f"{'='*70}")

# Cooling energy
cooling_baseline_kWh = sum(results_baseline['Q_cooling']) / 1e3
cooling_shaded_kWh = sum(results_shaded['Q_cooling']) / 1e3

print(f"\nCOOLING ENERGY:")
print(f"  Baseline (no shading):    {cooling_baseline_kWh:.1f} kWh/day")
print(f"  With shading devices:     {cooling_shaded_kWh:.1f} kWh/day")
print(f"  Reduction:                {(1 - cooling_shaded_kWh/cooling_baseline_kWh)*100:.1f}%")

# Peak cooling
peak_baseline = max(results_baseline['Q_cooling']) / 1000
peak_shaded = max(results_shaded['Q_cooling']) / 1000

print(f"\nPEAK COOLING POWER:")
print(f"  Baseline:  {peak_baseline:.1f} kW")
print(f"  Shaded:    {peak_shaded:.1f} kW")
print(f"  Reduction: {(1 - peak_shaded/peak_baseline)*100:.1f}%")

# Solar gains by facade
print(f"\nSOLAR HEAT GAIN BY FACADE (daily total, kWh):")
print(f"  {'Facade':<8} {'Baseline':>12} {'Shaded':>12} {'Reduction':>12}")
print(f"  {'-'*44}")
for facade in ['south', 'east', 'west', 'north']:
    Q_base = sum(results_baseline['Q_solar_by_facade'][facade]) / 1e3
    Q_shad = sum(results_shaded['Q_solar_by_facade'][facade]) / 1e3
    reduction = (1 - Q_shad/Q_base)*100 if Q_base > 0 else 0
    print(f"  {facade.capitalize():<8} {Q_base:>10.1f} {Q_shad:>12.1f} {reduction:>10.1f}%")

total_base = sum(results_baseline['Q_solar_total']) / 1e3
total_shad = sum(results_shaded['Q_solar_total']) / 1e3
print(f"  {'-'*44}")
print(f"  {'TOTAL':<8} {total_base:>10.1f} {total_shad:>12.1f} {(1-total_shad/total_base)*100:>10.1f}%")

# Shading effectiveness
print(f"\nSHADING DEVICE EFFECTIVENESS:")
print(f"  South overhang (1.0m): Peak shading = {max(solar_shaded['f_shade_south'])*100:.0f}%")
print(f"  East fins (0.5m):      Peak shading = {max(solar_shaded['f_shade_east'])*100:.0f}%")
print(f"  West fins (0.5m):      Peak shading = {max(solar_shaded['f_shade_west'])*100:.0f}%")

print(f"\n{'='*70}")
