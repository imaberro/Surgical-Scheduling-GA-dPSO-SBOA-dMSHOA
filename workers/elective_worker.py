"""
Worker for elective simulation mode (no emergencies).

Note: We reuse DynamicScheduler with an empty emergencies list to avoid
code duplication. DynamicScheduler handles the solution decoding logic
(converting encoded solution to detailed schedule) which is identical
for both elective and emergency modes. The overhead of unused emergency
tracking structures is negligible compared to the benefits of code reuse.

Future: Consider extracting common logic to a BaseScheduler if more
scheduling modes are added.
"""
import numpy as np
import time
from data.data_generator import generate_day_surgeries_data
from simulation.dynamic_scheduler import DynamicScheduler

class ElectiveWorker:
    """
    Executes a single elective simulation.
    """
    
    def __init__(self, job_ids, algorithms, std_factor):
        self.job_ids = job_ids
        self.algorithms = algorithms
        self.std_factor = std_factor
    
    def run(self, sim_i):
        """
        Runs one simulation iteration.
        
        Args:
            sim_i (int): Simulation index
        
        Returns:
            tuple: (sim_i, sim_results)
        """
        np.random.seed(sim_i)
        day_data = generate_day_surgeries_data(self.job_ids, std_factor=self.std_factor)
        
        sim_results = {}
        
        for spec in self.algorithms:
            algo_name = spec["name"]
            t0 = time.time()
            
            try:
                dynamic_scheduler = DynamicScheduler(
                    algorithm_runner=spec["runner"],
                    surgeries_data=day_data,
                    job_ids=self.job_ids
                )
                
                result = dynamic_scheduler.run_with_emergencies([], seed=sim_i)
                
                if not isinstance(result, tuple) or len(result) != 5:
                    print(f"WARNING: {algo_name} returned invalid format in sim {sim_i}")
                    sim_results[algo_name] = {
                        'makespan': float('inf'),
                        'solution': [],
                        'time': time.time() - t0,
                        'best_hist': [],
                        'avg_hist': []
                    }
                    continue
                
                # MISMA ESTRUCTURA QUE EMERGENCY_WORKER
                schedule_details, events_log, makespan, best_hist, avg_hist = result
                
                elapsed = time.time() - t0
                
                sim_results[algo_name] = {
                    'makespan': makespan if schedule_details else float('inf'),
                    'solution': schedule_details,
                    'time': elapsed,
                    'best_hist': best_hist if isinstance(best_hist, list) else [],
                    'avg_hist': avg_hist if isinstance(avg_hist, list) else []
                }
            
            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR: {algo_name} failed in sim {sim_i}: {e}")
                import traceback
                traceback.print_exc()
                
                sim_results[algo_name] = {
                    'makespan': float('inf'),
                    'solution': [],
                    'time': elapsed,
                    'best_hist': [],
                    'avg_hist': []
                }
        
        return sim_i, sim_results