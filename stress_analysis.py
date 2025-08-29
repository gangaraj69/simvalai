"""
Stress analysis and fatigue life calculation functions
Author: SRIHARI GANGARAJ (gangaraj@ieee.org)
"""

import numpy as np
from scipy.interpolate import interp1d

# Constants for the shaft problem
E = 210000  # Young's modulus in MPa (210 GPa)
L = 600  # Shaft length in mm
T = 800 * 1000  # Torque in N·mm (converted from 800 N·m)

# Stress concentration factors (based on Peterson's for D/d=1.4)
r_d_ratios = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
Kt_values = np.array([2.6, 2.2, 2.0, 1.8, 1.7])      # Bending stress concentration
Kts_values = np.array([2.0, 1.6, 1.4, 1.3, 1.2])    # Torsional stress concentration

# Create interpolation functions for stress concentration factors
Kt_interp = interp1d(r_d_ratios, Kt_values, kind='linear', fill_value='extrapolate')
Kts_interp = interp1d(r_d_ratios, Kts_values, kind='linear', fill_value='extrapolate')

def calculate_stress(d, r, delta):
    """
    Calculate combined stress using analytical formulas with stress concentration
    for a stepped shaft under combined torsion and bending from misalignment
    """
    # Calculate bending moment from misalignment (simplified beam theory)
    I = np.pi * d**4 / 64  # Moment of inertia
    M = (3 * E * I * np.abs(delta)) / (4 * L**2)  # Bending moment in N·mm
    
    # Calculate nominal stresses
    sigma_nom = (32 * M) / (np.pi * d**3)  # Bending stress
    tau_nom = (16 * T) / (np.pi * d**3)    # Torsional stress
    
    # Calculate stress concentration factors based on r/d ratio
    r_d = r / d
    K_t = Kt_interp(r_d)   # Bending stress concentration factor
    K_ts = Kts_interp(r_d) # Torsional stress concentration factor
    
    # Calculate actual stresses at fillet
    sigma_max = K_t * sigma_nom
    tau_max = K_ts * tau_nom
    
    # Calculate Von Mises equivalent stress (for multiaxial fatigue)
    sigma_vm = np.sqrt(sigma_max**2 + 3 * tau_max**2)
    
    return sigma_vm, sigma_max, tau_max

def calculate_life(sigma_vm, S_e):
    """
    Calculate fatigue life using Basquin's equation for high-cycle fatigue
    """
    # Basquin's equation: N = 0.5 * (S_e / sigma_vm)^10
    with np.errstate(divide='ignore', invalid='ignore'):
        life = 0.5 * (S_e / sigma_vm)**10
    
    # If stress is above ultimate strength, set life to 1 cycle
    life[sigma_vm >= S_e * 2] = 1
    
    # If stress is below endurance limit, set life to a very high value (run-out)
    life[sigma_vm <= S_e] = 1e12
    
    return life