"""
Worker for emergency simulation mode (with dynamic emergencies).
"""
import numpy as np
import time
from data.data_generator import generate_day_surgeries_data
from simulation.dynamic_scheduler import DynamicScheduler
from simulation.emergency_generator import generate_emergency_arrivals

class EmergencyWorker:
    """
    Executes a single emergency simulation.
    """
    
    def __init__(self, job_ids, algorithms, std_factor, num_emergencies):
        self.job_ids = job_ids
        self.algorithms = algorithms
        self.std_factor = std_factor
        self.num_emergencies = num_emergencies
    
    def run(self, sim_i):
        """
        Runs one emergency simulation iteration.
        
        Args:
            sim_i (int): Simulation index
        
        Returns:
            tuple: (sim_i, sim_results, emergencies)
        """
        # Generate unique emergencies for this simulation
        emergencies = generate_emergency_arrivals(
            num_emergencies=self.num_emergencies,
            seed=1000 + sim_i
        )
        
        np.random.seed(sim_i)
        day_data = generate_day_surgeries_data(self.job_ids, std_factor=self.std_factor)
        
        sim_results = {}
        
        for spec in self.algorithms:
            algo_name = spec["name"]
            t0 = time.time()
            
            dynamic_scheduler = DynamicScheduler(
                algorithm_runner=spec["runner"],
                surgeries_data=day_data,
                job_ids=self.job_ids
            )
            
            final_schedule, events_log, final_makespan, best_hist, avg_hist = dynamic_scheduler.run_with_emergencies(
                emergencies,
                seed=sim_i
            )
            
            elapsed = time.time() - t0
            
            sim_results[algo_name] = {
                'makespan': final_makespan if final_schedule else float('inf'),
                'solution': final_schedule,
                'events': events_log if final_schedule else [],
                'time': elapsed,
                'best_hist': best_hist,
                'avg_hist': avg_hist
            }
        
        return sim_i, sim_results, emergencies