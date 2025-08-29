"""
Comprehensive Reliability Analysis with Monte Carlo Simulation and Test Planning
Author: SRIHARI GANGARAJ (gangaraj@ieee.org)

This integrated script performs a complete reliability analysis for a stepped shaft:
1. Monte Carlo simulation considering manufacturing and material variations
2. Reliability metrics calculation and Weibull analysis
3. Sample size determination for physical validation testing
4. Accelerated test planning with practical recommendations

The methodology implements the fused simulation-validation approach described in:
"Beyond the Hype: Fusing Simulation, AI and Physical Validation in Product Development"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define constants and parameters for the shaft problem
NUM_SIMULATIONS = 10000
T = 800 * 1000  # Torque in N·mm (converted from 800 N·m)
D1 = 45  # Large diameter in mm (fixed)
E = 210000  # Young's modulus in MPa (210 GPa)
L = 600  # Shaft length in mm
TARGET_LIFE = 394.2e6  # Required life in cycles (5 years at 400 RPM, 9 hrs/day)
ACCELERATION_FACTOR = 1.25  # Test acceleration factor (500 RPM vs 400 RPM)

# Define distributions for variables based on GD&T tolerances
def generate_samples(n):
    """Generate samples for all variables with normal distributions"""
    return {
        'd': np.random.normal(32, 0.017, n),      # Small diameter (mm) ±0.05mm tolerance
        'r': np.random.normal(2.2, 0.05, n),      # Fillet radius (mm) ±0.15mm tolerance
        'delta': np.random.normal(0, 0.13, n),    # Misalignment offset (mm) ±0.4mm position tolerance
        'S_e': np.random.normal(485, 20, n)       # Endurance limit (MPa) for SAE 1040 at RC30
    }

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

def calculate_reliability_metrics(df, target_life):
    """
    Calculate comprehensive reliability metrics from simulation results
    including Weibull distribution fitting
    """
    # Filter out infinite lives for analysis
    finite_lives = df[df['life'] < 1e11]['life']
    
    if len(finite_lives) > 10:
        # Fit Weibull distribution
        shape, loc, scale = stats.weibull_min.fit(finite_lives, floc=0)
        
        # Calculate reliability metrics
        b10_life = scale * (-np.log(0.9))**(1/shape)
        b50_life = scale * (-np.log(0.5))**(1/shape)
        
        # Reliability at target life
        reliability_at_target = np.exp(-(target_life/scale)**shape)
        
        # Failure rate function at target life
        failure_rate = (shape/scale) * (target_life/scale)**(shape-1)
        
        return {
            'shape_param': shape,
            'scale_param': scale,
            'b10_life': b10_life,
            'b50_life': b50_life,
            'reliability_at_target': reliability_at_target,
            'failure_rate_at_target': failure_rate,
            'finite_samples': len(finite_lives)
        }
    else:
        return None

def determine_sample_size(reliability_metrics, confidence=0.90, target_reliability=0.90):
    """
    Determine required sample size for physical validation testing
    using chi-square distribution for zero-failure test plan
    """
    if reliability_metrics is None:
        return "Insufficient data for sample size calculation"
    
    β = reliability_metrics['shape_param']
    
    # For zero-failure test plan (r = 0)
    r = 0
    
    # Calculate the chi-square value
    chi_square_val = stats.chi2.ppf(confidence, 2*r + 2)
    
    # Calculate required sample size
    n = chi_square_val / (2 * -np.log(target_reliability))
    
    # Round up to nearest integer
    n = int(np.ceil(n))
    
    return {
        'required_sample_size': n,
        'test_duration_per_unit': reliability_metrics['b10_life'],
        'confidence_level': confidence,
        'target_reliability': target_reliability,
        'test_type': 'Zero-failure test plan'
    }

def generate_test_plan(reliability_metrics, acceleration_factor):
    """
    Generate a comprehensive test plan based on simulation results
    including acceleration factors and monitoring recommendations
    """
    if reliability_metrics is None:
        return "Insufficient data for test planning"
    
    # Determine sample size
    test_plan = determine_sample_size(reliability_metrics)
    
    if isinstance(test_plan, dict):
        # Adjust for acceleration factor
        accelerated_test_time = test_plan['test_duration_per_unit'] / acceleration_factor
        
        # Add additional recommendations
        test_plan['acceleration_factor'] = acceleration_factor
        test_plan['accelerated_test_time'] = accelerated_test_time
        test_plan['total_test_time_days'] = accelerated_test_time / (500 * 60 * 60)  # Convert to days at 500 RPM
        
        # Recommendations based on sensitivity analysis
        test_plan['critical_parameters'] = [
            'Fillet radius (r) - Tight control recommended',
            'Misalignment (δ) - Careful assembly required',
            'Material quality (S_e) - Supplier certification needed'
        ]
        
        # Monitoring recommendations
        test_plan['monitoring_recommendations'] = [
            'Strain gauges at fillet root for all test units',
            'Regular inspection for micro-cracks (every 1M cycles)',
            'Temperature monitoring at bearing locations'
        ]
    
    return test_plan

def analyze_results(df, target_life, acceleration_factor):
    """
    Comprehensive analysis of simulation results with reliability metrics
    and test planning for the stepped shaft
    """
    print("Comprehensive Reliability Analysis for Stepped Shaft")
    print("=" * 60)
    print(f"Target life: {target_life:.1e} cycles")
    print(f"Number of simulations: {len(df)}")
    
    # Calculate reliability metrics
    reliability_metrics = calculate_reliability_metrics(df, target_life)
    
    if reliability_metrics:
        print(f"\nReliability Metrics:")
        print(f"Weibull Shape Parameter (β): {reliability_metrics['shape_param']:.3f}")
        print(f"Weibull Scale Parameter (η): {reliability_metrics['scale_param']:.3e}")
        print(f"B10 Life: {reliability_metrics['b10_life']:.3e} cycles")
        print(f"B50 Life: {reliability_metrics['b50_life']:.3e} cycles")
        print(f"Reliability at Target Life: {reliability_metrics['reliability_at_target']:.3f}")
        print(f"Failure Rate at Target Life: {reliability_metrics['failure_rate_at_target']:.3e} failures/cycle")
        
        # Calculate reliability (probability of meeting target life)
        reliability = np.mean(df['life'] >= target_life) * 100
        print(f"Overall Reliability: {reliability:.2f}%")
        
        # Generate test plan
        test_plan = generate_test_plan(reliability_metrics, acceleration_factor)
        
        if isinstance(test_plan, dict):
            print(f"\nPhysical Validation Test Plan:")
            print(f"Required Sample Size: {test_plan['required_sample_size']} units")
            print(f"Test Duration per Unit: {test_plan['test_duration_per_unit']:.3e} cycles")
            print(f"Acceleration Factor: {test_plan['acceleration_factor']:.1f}")
            print(f"Accelerated Test Time: {test_plan['accelerated_test_time']:.3e} cycles")
            print(f"Total Test Time: {test_plan['total_test_time_days']:.1f} days")
            
            print(f"\nCritical Parameters to Control:")
            for param in test_plan['critical_parameters']:
                print(f"  - {param}")
                
            print(f"\nMonitoring Recommendations:")
            for recommendation in test_plan['monitoring_recommendations']:
                print(f"  - {recommendation}")
    
    # Display statistical summary
    print(f"\nStatistical Summary of Simulation Results:")
    print(df[['d', 'r', 'delta', 'S_e', 'sigma_vm', 'life']].describe())
    
    # Calculate correlation matrix
    corr_matrix = df[['d', 'r', 'delta', 'S_e', 'sigma_vm', 'life']].corr()
    print(f"\nCorrelation Matrix:")
    print(corr_matrix)
    
    return reliability_metrics

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

def main():
    """
    Main function to run the complete analysis for the stepped shaft reliability problem
    """
    print("Comprehensive Reliability Analysis for Stepped Shaft")
    print("Author: SRIHARI GANGARAJ (gangaraj@ieee.org)")
    print("=" * 70)
    print("This analysis implements the fused simulation-validation approach")
    print("from 'Beyond the Hype: Fusing Simulation, AI and Physical Validation'")
    print("=" * 70)
    
    # Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation with 10,000 iterations...")
    results_df = monte_carlo_simulation(NUM_SIMULATIONS)
    
    # Analyze results
    reliability_metrics = analyze_results(results_df, TARGET_LIFE, ACCELERATION_FACTOR)
    
    # Generate plots
    print("\nGenerating comprehensive visualizations...")
    plot_results(results_df)
    
    # Save results to CSV
    results_df.to_csv('shaft_reliability_analysis.csv', index=False)
    print("\nDetailed results saved to 'shaft_reliability_analysis.csv'")
    
    # Generate final recommendations
    if reliability_metrics:
        test_plan = generate_test_plan(reliability_metrics, ACCELERATION_FACTOR)
        if isinstance(test_plan, dict):
            print(f"\n{'='*60}")
            print("SUMMARY RECOMMENDATION FOR PHYSICAL VALIDATION")
            print(f"{'='*60}")
            print(f"Based on {NUM_SIMULATIONS} virtual tests, the recommended physical validation plan is:")
            print(f"- Test {test_plan['required_sample_size']} units to failure or {test_plan['accelerated_test_time']:.1e} cycles")
            print(f"- This will require approximately {test_plan['total_test_time_days']:.1f} days of testing")
            print(f"- Focus quality control on: {', '.join([p.split(' - ')[0] for p in test_plan['critical_parameters']])}")
            print(f"- This test plan will demonstrate {test_plan['target_reliability']*100:.0f}% reliability ")
            print(f"  with {test_plan['confidence_level']*100:.0f}% confidence")
            print(f"\nThis approach reduces physical testing time by {((1-1/ACCELERATION_FACTOR)*100):.0f}%")
            print("while maintaining statistical confidence in the results.")

if __name__ == "__main__":
    main()