"""
Module for generating emergency surgery arrivals based on historical data.
Uses normal distribution as described in the paper.
"""
import numpy as np
from config.config import JOB_TYPES

def generate_emergency_arrivals(num_emergencies=2, seed=None):
    """
    Generates emergency arrival times using a normal distribution.

    NOTE: The distribution is biased earlier so emergencies arrive DURING
    elective surgeries, not after the elective schedule finishes.
    
    Args:
        num_emergencies: number of emergencies to simulate
        seed: seed for reproducibility
    
    Returns:
        list: list of dicts with emergency information
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Adjusted parameters for earlier arrivals
    # Use μ=450 (mid-day) and σ=150 (wide dispersion)
    MEAN_ARRIVAL = 450.0      # adjusted mean (mid workday)
    STD_ARRIVAL = 150.0       # adjusted std (reasonable dispersion)
    
    # Urgency parameter (du in the paper)
    MAX_DELAY_ALLOWED = 60  # maximum allowed start delay (minutes)
    
    arrivals = []
    
    for i in range(num_emergencies):
        # Generate arrival time from a normal distribution
        arrival_time = np.random.normal(MEAN_ARRIVAL, STD_ARRIVAL)
        
        # Clamp to [100, 800] minutes (reasonable working-day window)
        arrival_time = max(100, min(800, arrival_time))
        
        # Emergency IDs (E16, E17 as in the paper)
        emergency_id = 16 + i
        job_id = f"E{emergency_id}"
        
        # Assign procedure type (random among existing types)
        job_type = np.random.choice(list(set(JOB_TYPES.values())))
        
        arrivals.append({
            'job_id': job_id,
            'arrival_time': arrival_time,
            'job_type': job_type,
            'max_delay': MAX_DELAY_ALLOWED
        })
    
    # Sort by arrival time
    arrivals.sort(key=lambda x: x['arrival_time'])
    
    return arrivals