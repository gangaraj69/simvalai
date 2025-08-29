"""
Test planning and sample size determination functions
Author: SRIHARI GANGARAJ (gangaraj@ieee.org)
"""

import numpy as np
from scipy import stats

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