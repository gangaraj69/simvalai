"""
Main execution script for reliability analysis
Author: SRIHARI GANGARAJ (gangaraj@ieee.org)
"""

import numpy as np
import pandas as pd
from reliability_analysis import (
    monte_carlo_simulation, 
    analyze_results, 
    plot_results,
    generate_test_plan
)

# Constants for the shaft problem
NUM_SIMULATIONS = 10000
T = 800 * 1000  # Torque in N·mm (converted from 800 N·m)
D1 = 45  # Large diameter in mm (fixed)
E = 210000  # Young's modulus in MPa (210 GPa)
L = 600  # Shaft length in mm
TARGET_LIFE = 394.2e6  # Required life in cycles (5 years at 400 RPM, 9 hrs/day)
ACCELERATION_FACTOR = 1.25  # Test acceleration factor (500 RPM vs 400 RPM)

def analyze_results(df, target_life, acceleration_factor):
    """
    Comprehensive analysis of simulation results with reliability metrics
    and test planning for the stepped shaft
    """
    from reliability_analysis.reliability_metrics import calculate_reliability_metrics
    from reliability_analysis.test_planning import generate_test_plan
    
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