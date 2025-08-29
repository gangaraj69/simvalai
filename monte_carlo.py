"""
Monte Carlo simulation functions
Author: SRIHARI GANGARAJ (gangaraj@ieee.org)
"""

import numpy as np
import pandas as pd
from reliability_analysis.stress_analysis import calculate_stress, calculate_life

# Constants for the shaft problem (imported from main or defined here)
T = 800 * 1000  # Torque in N·mm (converted from 800 N·m)
D1 = 45  # Large diameter in mm (fixed)

def generate_samples(n):
    """Generate samples for all variables with normal distributions"""
    return {
        'd': np.random.normal(32, 0.017, n),      # Small diameter (mm) ±0.05mm tolerance
        'r': np.random.normal(2.2, 0.05, n),      # Fillet radius (mm) ±0.15mm tolerance
        'delta': np.random.normal(0, 0.13, n),    # Misalignment offset (mm) ±0.4mm position tolerance
        'S_e': np.random.normal(485, 20, n)       # Endurance limit (MPa) for SAE 1040 at RC30
    }

def monte_carlo_simulation(n_simulations):
    """
    Run Monte Carlo simulation with all parameter variations
    for the stepped shaft reliability analysis
    """
    # Generate samples
    samples = generate_samples(n_simulations)
    
    # Calculate results for each sample
    results = []
    for i in range(n_simulations):
        d, r, delta, S_e = samples['d'][i], samples['r'][i], samples['delta'][i], samples['S_e'][i]
        
        # Calculate stress
        sigma_vm, sigma_max, tau_max = calculate_stress(d, r, delta)
        
        # Calculate life
        life = calculate_life(sigma_vm, S_e)
        
        # Calculate D/d ratio
        D_d_ratio = D1 / d
        
        # Store results
        results.append({
            'd': d,
            'r': r,
            'delta': delta,
            'S_e': S_e,
            'D/d': D_d_ratio,
            'r/d': r/d,
            'sigma_vm': sigma_vm,
            'sigma_max': sigma_max,
            'tau_max': tau_max,
            'life': life
        })
    
    return pd.DataFrame(results)