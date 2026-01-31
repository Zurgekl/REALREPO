"""
Minimal thermal model for Academic Hall North
Step 1: Single-zone energy balance with proper free-float tracking
"""

import numpy as np
import miami_climate_params as mcp

def thermal_model_step(T_in_current, T_out, I_solar, Q_internal,
                        A_win, SHGC, f_shade, UA_envelope, C_eff, dt=3600):
    """
    One timestep of thermal model

    State variable: T_in = lumped zone temperature (effective air + mass)

    Inputs:
        T_in_current: current indoor temp (°C)
        T_out: outdoor temp (°C)
        I_solar: incident solar radiation on façade AFTER shading geometry (W/m²)
        Q_internal: internal gains (W)
        A_win: window area (m²)
        SHGC: Solar Heat Gain Coefficient
        f_shade: fraction of window shaded (0-1) - APPLIED TO I_solar already
        UA_envelope: U*A for entire envelope (W/K)
        C_eff: effective thermal capacitance of zone (J/K)
        dt: timestep (seconds, default 3600)

    Returns:
        T_in_free: free-float temperature (no cooling) (°C)
        T_in_ctrl: controlled temperature (with cooling) (°C)
        Q_solar: solar heat gain (W)
        Q_envelope: envelope conduction (W)
        Q_cooling_required: cooling power needed to maintain comfort (W)
    """

    # Solar heat gain (transmitted through glazing)
    Q_solar = A_win * SHGC * I_solar * (1 - f_shade)

    # Envelope conduction (positive = heat entering building)
    Q_envelope = UA_envelope * (T_out - T_in_current)

    # Total heat input to zone
    Q_total = Q_solar + Q_internal + Q_envelope

    # FREE-FLOAT temperature (what happens without cooling)
    dT_free = (Q_total * dt) / C_eff
    T_in_free = T_in_current + dT_free

    # Comfort threshold (ASHRAE 55 summer upper limit)
    T_comfort_max = mcp.COMFORT['summer_comfort_upper']  # 26°C (79°F)

    # Cooling required to maintain comfort
    if T_in_free > T_comfort_max:
        # Energy removal rate needed
        Q_cooling_required = (T_in_free - T_comfort_max) * C_eff / dt
        T_in_ctrl = T_comfort_max
    else:
        Q_cooling_required = 0
        T_in_ctrl = T_in_free

    return T_in_free, T_in_ctrl, Q_solar, Q_envelope, Q_cooling_required
