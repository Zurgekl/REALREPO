"""
Test thermal model on one summer day
Now with explicit acknowledgment that I_solar is STUDENT 3's output
"""

import numpy as np
import matplotlib.pyplot as plt
from thermal_model_v1 import thermal_model_step
import building_params as bp
import miami_climate_params as mcp

# ============================================================================
# WEATHER FORCING (toy data for Day 1 test)
# ============================================================================
hours = np.arange(24)

# Outdoor temperature (sinusoidal approximation of summer day)
# Phase shift of -9 puts peak at hour 15 (3 PM) and minimum at hour 3 (3 AM)
# This matches real diurnal temperature patterns where:
#   - Minimum occurs ~sunrise (5-6 AM)
#   - Maximum occurs ~3-4 PM (2-3 hours after solar noon due to thermal lag)
T_out_mean = 30  # °C (86°F)
T_out_swing = 6  # °C amplitude
T_out = T_out_mean + T_out_swing * np.sin(2*np.pi*(hours - 9)/24)

# ============================================================================
# FIX 4: Solar radiation is STUDENT 3's output
# For now: toy clear-sky approximation on SOUTH FAÇADE
# Later: Student 3 provides I_incident_south[t] after shading geometry
# ============================================================================
# Clear-sky irradiance on vertical south surface (simplified)
# Peak ~400 W/m² at solar noon for Miami summer (high sun angle)
I_clear_sky_south = 400 * np.maximum(0, np.sin(np.pi*(hours-6)/12))

# Cloud attenuation (Student 2's domain, but simplified here)
cloud_factor = 0.7  # Typical partly cloudy
I_incident_south_unshaded = I_clear_sky_south * cloud_factor

# NOTE: In real model, Student 3 computes shading geometry
# and provides I_incident[t] AFTER accounting for overhang shadows
# For now, we'll simulate shading as simple fraction in thermal_model_step

# ============================================================================
# INTERNAL GAINS SCHEDULE
# ============================================================================
Q_internal = np.where(
    (hours >= 8) & (hours <= 17),  # Occupied 8 AM - 5 PM
    bp.Q_INTERNAL_OCCUPIED,
    bp.Q_INTERNAL_UNOCCUPIED
)

# ============================================================================
# SIMULATION: BASELINE (no shading)
# ============================================================================
T_free_baseline = [mcp.COMFORT['summer_comfort_upper']]  # Start at comfort limit
T_ctrl_baseline = [mcp.COMFORT['summer_comfort_upper']]
Q_cooling_baseline = []
Q_solar_baseline = []

for hr in range(24):
    T_free, T_ctrl, Q_sol, Q_env, Q_cool = thermal_model_step(
        T_in_current=T_ctrl_baseline[-1],  # Use controlled temp for next step
        T_out=T_out[hr],
        I_solar=I_incident_south_unshaded[hr],  # From Student 3 (toy version)
        Q_internal=Q_internal[hr],
        A_win=bp.WINDOW_AREAS['south'],
        SHGC=bp.ENVELOPE_BASELINE['SHGC'],
        f_shade=0.0,  # NO shading
        UA_envelope=bp.UA_ENVELOPE,
        C_eff=bp.C_EFFECTIVE
    )

    T_free_baseline.append(T_free)
    T_ctrl_baseline.append(T_ctrl)
    Q_cooling_baseline.append(Q_cool)
    Q_solar_baseline.append(Q_sol)

# ============================================================================
# SIMULATION: WITH SHADING (70% of south windows shaded)
# ============================================================================
T_free_shaded = [mcp.COMFORT['summer_comfort_upper']]
T_ctrl_shaded = [mcp.COMFORT['summer_comfort_upper']]
Q_cooling_shaded = []
Q_solar_shaded = []

for hr in range(24):
    T_free, T_ctrl, Q_sol, Q_env, Q_cool = thermal_model_step(
        T_in_current=T_ctrl_shaded[-1],
        T_out=T_out[hr],
        I_solar=I_incident_south_unshaded[hr],
        Q_internal=Q_internal[hr],
        A_win=bp.WINDOW_AREAS['south'],
        SHGC=bp.ENVELOPE_BASELINE['SHGC'],
        f_shade=0.70,  # 70% shaded
        UA_envelope=bp.UA_ENVELOPE,
        C_eff=bp.C_EFFECTIVE
    )

    T_free_shaded.append(T_free)
    T_ctrl_shaded.append(T_ctrl)
    Q_cooling_shaded.append(Q_cool)
    Q_solar_shaded.append(Q_sol)

# ============================================================================
# PLOTTING
# ============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

# Temperature (free-float vs controlled)
ax1.plot(hours, T_out, 'r--', label='Outdoor', linewidth=2)
ax1.plot(hours, T_free_baseline[:-1], 'b:', label='Free-float (no shade)', linewidth=2, alpha=0.7)
ax1.plot(hours, T_ctrl_baseline[:-1], 'b-', label='Controlled (no shade)', linewidth=2)
ax1.plot(hours, T_free_shaded[:-1], 'g:', label='Free-float (70% shade)', linewidth=2, alpha=0.7)
ax1.plot(hours, T_ctrl_shaded[:-1], 'g-', label='Controlled (70% shade)', linewidth=2)
ax1.axhline(mcp.COMFORT['summer_comfort_upper'], color='k', linestyle=':',
            label='Comfort limit', linewidth=1.5)
ax1.set_ylabel('Temperature (°C)', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title('Indoor Temperature: Free-Float vs Controlled')

# Solar heat gain
ax2.plot(hours, np.array(Q_solar_baseline)/1000, 'b-', label='No shade', linewidth=2)
ax2.plot(hours, np.array(Q_solar_shaded)/1000, 'g-', label='70% shade', linewidth=2)
ax2.set_ylabel('Solar Heat Gain (kW)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('Solar Heat Gain Through South Windows')

# Cooling power required
ax3.plot(hours, np.array(Q_cooling_baseline)/1000, 'b-', label='No shade', linewidth=2)
ax3.plot(hours, np.array(Q_cooling_shaded)/1000, 'g-', label='70% shade', linewidth=2)
ax3.set_xlabel('Hour of Day', fontsize=11)
ax3.set_ylabel('Cooling Power (kW)', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_title('Cooling Power Required to Maintain Comfort')

plt.tight_layout()
plt.savefig('one_day_test_corrected.png', dpi=150)
print("One-day test complete. See one_day_test_corrected.png")

# ============================================================================
# METRICS SUMMARY
# ============================================================================
# Cooling energy (kWh)
cooling_baseline_kWh = sum(Q_cooling_baseline) * 1 / 1e3  # W × hr / 1000
cooling_shaded_kWh = sum(Q_cooling_shaded) * 1 / 1e3

# Overheat degree-hours (free-float)
overheat_baseline = sum(max(0, T - mcp.COMFORT['summer_comfort_upper'])
                        for T in T_free_baseline)
overheat_shaded = sum(max(0, T - mcp.COMFORT['summer_comfort_upper'])
                      for T in T_free_shaded)

# Peak values
peak_cooling_baseline = max(Q_cooling_baseline) / 1000  # kW
peak_cooling_shaded = max(Q_cooling_shaded) / 1000

print(f"\n{'='*60}")
print(f"ONE-DAY PERFORMANCE SUMMARY")
print(f"{'='*60}")
print(f"\nCOOLING ENERGY:")
print(f"  Baseline (no shade):  {cooling_baseline_kWh:.1f} kWh")
print(f"  With 70% shading:     {cooling_shaded_kWh:.1f} kWh")
print(f"  Reduction:            {(1 - cooling_shaded_kWh/cooling_baseline_kWh)*100:.1f}%")

print(f"\nPEAK COOLING POWER:")
print(f"  Baseline:  {peak_cooling_baseline:.0f} kW")
print(f"  Shaded:    {peak_cooling_shaded:.0f} kW")
print(f"  Reduction: {(1 - peak_cooling_shaded/peak_cooling_baseline)*100:.1f}%")

print(f"\nOVERHEAT DEGREE-HOURS (if no cooling):")
print(f"  Baseline:  {overheat_baseline:.1f} °C·hr")
print(f"  Shaded:    {overheat_shaded:.1f} °C·hr")
print(f"  Reduction: {(1 - overheat_shaded/overheat_baseline)*100:.1f}%")

print(f"\n{'='*60}")
