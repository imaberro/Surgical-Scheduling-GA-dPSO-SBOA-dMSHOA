# /data_generator.py
import copy
import numpy as np

BASE_DAY_SURGERIES_DATA = {
    1: {1: 30, 2: 60, 3: 40}, 2: {1: 40, 2: 60, 3: 40},
    3: {1: 35, 2: 80, 3: 40}, 4: {1: 65, 2: 190, 3: 60},
    5: {1: 70, 2: 190, 3: 60}, 6: {1: 75, 2: 190, 3: 60},
    7: {1: 80, 2: 150, 3: 50}, 8: {1: 70, 2: 110, 3: 50},
    9: {1: 35, 2: 80, 3: 40}, 10: {1: 30, 2: 80, 3: 40},
    11: {1: 60, 2: 110, 3: 50}, 12: {1: 80, 2: 110, 3: 50},
    13: {1: 65, 2: 210, 3: 60}, 14: {1: 40, 2: 70, 3: 40},
    15: {1: 70, 2: 160, 3: 60}
}

def generate_day_surgeries_data(job_ids, std_factor=0.0):
    """
    Generates surgery processing-time data with optional variability.
    """
    data = {}
    for j in job_ids:
        if j in BASE_DAY_SURGERIES_DATA:
            data[j] = copy.deepcopy(BASE_DAY_SURGERIES_DATA[j])
            
            if std_factor > 0:
                for op in [1, 2, 3]:
                    base_val = BASE_DAY_SURGERIES_DATA[j][op]
                    std_val = std_factor * base_val
                    value = np.random.normal(base_val, std_val)
                    data[j][op] = max(1, round(value, 2))
        else:
            # Default value for undefined jobs
            data[j] = {1: 30, 2: 60, 3: 40}
    return data