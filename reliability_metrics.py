"""
Reliability metrics calculation functions
Author: SRIHARI GANGARAJ (gangaraj@ieee.org)
"""

import numpy as np
from scipy import stats

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