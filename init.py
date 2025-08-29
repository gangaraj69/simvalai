"""
AI-Powered Reliability Analysis for Mechanical Components
Author: Srihari Gangaraj (gangaraj@ieee.org)
"""

from .monte_carlo import generate_samples, monte_carlo_simulation
from .stress_analysis import calculate_stress, calculate_life, Kt_interp, Kts_interp
from .reliability_metrics import calculate_reliability_metrics
from .test_planning import determine_sample_size, generate_test_plan
from .visualization import plot_results

__version__ = "1.0.0"
__author__ = "Srihari Gangaraj <gangaraj@ieee.org>"