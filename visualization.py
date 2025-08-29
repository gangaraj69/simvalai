"""
Visualization functions for reliability analysis
Author: SRIHARI GANGARAJ (gangaraj@ieee.org)
"""

import matplotlib.pyplot as plt
import numpy as np
from reliability_analysis.stress_analysis import Kt_interp, Kts_interp

# Constants for the shaft problem
TARGET_LIFE = 394.2e6  # Required life in cycles (5 years at 400 RPM, 9 hrs/day)

def plot_results(df):
    """
    Create comprehensive visualization of results for the stepped shaft analysis
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comprehensive Reliability Analysis for Stepped Shaft', fontsize=16)
    
    # Plot 1: Distribution of calculated lives
    axes[0, 0].hist(np.log10(df['life']), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.log10(TARGET_LIFE), color='r', linestyle='--', label='Target Life')
    axes[0, 0].set_xlabel('Log10(Life) (cycles)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Fatigue Lives')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Von Mises stress distribution
    axes[0, 1].hist(df['sigma_vm'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df['S_e'].mean(), color='r', linestyle='--', label='Mean Endurance Limit')
    axes[0, 1].set_xlabel('Von Mises Stress (MPa)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Von Mises Stress')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: r/d vs. stress concentration factor
    r_d_vals = np.linspace(0.01, 0.12, 100)
    axes[0, 2].plot(r_d_vals, Kt_interp(r_d_vals), label='Kt (Bending)')
    axes[0, 2].plot(r_d_vals, Kts_interp(r_d_vals), label='Kts (Torsion)')
    axes[0, 2].set_xlabel('r/d ratio')
    axes[0, 2].set_ylabel('Stress Concentration Factor')
    axes[0, 2].set_title('Stress Concentration Factors')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Sensitivity - r/d vs. life
    axes[1, 0].scatter(df['r/d'], np.log10(df['life']), alpha=0.5)
    axes[1, 0].set_xlabel('r/d ratio')
    axes[1, 0].set_ylabel('Log10(Life) (cycles)')
    axes[1, 0].set_title('Effect of r/d Ratio on Fatigue Life')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Sensitivity - misalignment vs. life
    axes[1, 1].scatter(df['delta'], np.log10(df['life']), alpha=0.5)
    axes[1, 1].set_xlabel('Misalignment (mm)')
    axes[1, 1].set_ylabel('Log10(Life) (cycles)')
    axes[1, 1].set_title('Effect of Misalignment on Fatigue Life')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Sensitivity - S_e vs. life
    axes[1, 2].scatter(df['S_e'], np.log10(df['life']), alpha=0.5)
    axes[1, 2].set_xlabel('Endurance Limit (MPa)')
    axes[1, 2].set_ylabel('Log10(Life) (cycles)')
    axes[1, 2].set_title('Effect of Material Strength on Fatigue Life')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()