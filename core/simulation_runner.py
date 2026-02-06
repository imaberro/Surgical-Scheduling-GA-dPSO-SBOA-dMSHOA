"""
High-level simulation runner that orchestrates the execution flow.
"""
from joblib import Parallel, delayed
from core.file_manager import FileManager
from core.report_generator import ReportGenerator
from workers.elective_worker import ElectiveWorker
from workers.emergency_worker import EmergencyWorker
from config.config import (
    JOB_TYPES, get_algorithms, NUM_SIMULATIONS,
    STD_FACTOR, ALPHA_TEST, ALL_ROOMS, 
    NUM_EMERGENCIES, VERBOSE_MODE
)
from utils.logger import logger
import os
import time

class SimulationRunner:
    """
    Orchestrates the simulation execution for both elective and emergency modes.
    """
    
    def __init__(self):
        self.file_manager = FileManager()
        self.report_generator = ReportGenerator()
        self.job_ids = list(JOB_TYPES.keys())
        self.n_jobs = min(10, os.cpu_count() - 2)
        self.algorithms = get_algorithms()
    
    def run_elective_mode(self):
        """
        Executes elective simulation mode (no emergencies).
        """
        output_dirs = self.file_manager.setup_elective_directories()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"ELECTIVE SIMULATION MODE - PARALLEL with {self.n_jobs} workers")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        # Parallel execution
        verbose_level = 10 if not VERBOSE_MODE else 0
        worker = ElectiveWorker(self.job_ids, self.algorithms, STD_FACTOR)
        
        results = Parallel(n_jobs=self.n_jobs, verbose=verbose_level)(
            delayed(worker.run)(sim_i) for sim_i in range(NUM_SIMULATIONS)
        )
        
        # Aggregate results
        all_results, best_overall = self._aggregate_results(results, self.algorithms)  # TODO: review/rename if needed
        
        elapsed = time.time() - start_time
        logger.info(f"\nAll {NUM_SIMULATIONS} elective simulations completed!")
        
        # Generate reports
        self.report_generator.generate_elective_reports(
            all_results, best_overall, output_dirs, ALL_ROOMS, ALPHA_TEST
        )
        
        logger.info(f"\nProcess completed! (Total time: {elapsed:.2f}s). Check the 'results' folder.")
    
    def run_emergency_mode(self):
        """
        Executes emergency simulation mode (with dynamic emergencies).
        """
        output_dirs = self.file_manager.setup_emergency_directories()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"EMERGENCY SIMULATION MODE (TSJS Strategy) - PARALLEL with {self.n_jobs} workers")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        # Parallel execution
        verbose_level = 10 if not VERBOSE_MODE else 0
        worker = EmergencyWorker(self.job_ids, self.algorithms, STD_FACTOR, NUM_EMERGENCIES)
        
        results = Parallel(n_jobs=self.n_jobs, verbose=verbose_level)(
            delayed(worker.run)(sim_i) for sim_i in range(NUM_SIMULATIONS)
        )
        
        # Aggregate results
        all_results, best_overall = self._aggregate_emergency_results(results, self.algorithms)
        
        elapsed = time.time() - start_time
        logger.info(f"\nAll {NUM_SIMULATIONS} emergency simulations completed!")
        
        # Generate reports
        self.report_generator.generate_emergency_reports(
            all_results, best_overall, output_dirs, ALL_ROOMS, ALPHA_TEST
        )
        
        logger.info(f"\nProcess completed! (Total time: {elapsed:.2f}s). Check the 'results' folder.")
    
    def _aggregate_results(self, results, algorithms):
        """Aggregates results from parallel workers (elective mode)."""
        all_results = {
            spec["name"]: {
                'makespan': [], 'solution': [], 'time': [],
                'best_hist': [], 'avg_hist': []
            } 
            for spec in algorithms
        }
        
        best_overall = {
            spec["name"]: {
                'makespan': float('inf'),
                'schedule': None,
                'sim_num': -1
            } 
            for spec in algorithms
        }
        
        for sim_i, sim_results in results:
            for algo_name, result in sim_results.items():
                all_results[algo_name]['makespan'].append(result['makespan'])
                all_results[algo_name]['solution'].append(result['solution'])
                all_results[algo_name]['time'].append(result['time'])
                all_results[algo_name]['best_hist'].append(result['best_hist'])
                all_results[algo_name]['avg_hist'].append(result['avg_hist'])
                
                if result['makespan'] < best_overall[algo_name]['makespan']:
                    best_overall[algo_name] = {
                        'makespan': result['makespan'],
                        'schedule': result['solution'],
                        'sim_num': sim_i
                    }
        
        return all_results, best_overall
    
    def _aggregate_emergency_results(self, results, algorithms):
        """Aggregates results from parallel workers (emergency mode)."""
        all_results = {
            spec["name"]: {
                'makespan': [], 'solution': [], 'events': [], 'time': [],
                'best_hist': [], 'avg_hist': []
            } 
            for spec in algorithms
        }
        
        best_overall = {
            spec["name"]: {
                'makespan': float('inf'),
                'schedule': None, 
                'events': None,
                'sim_num': -1,
                'emergencies': None
            } 
            for spec in algorithms
        }
        
        for sim_i, sim_results, sim_emergencies in results:
            for algo_name, result in sim_results.items():
                all_results[algo_name]['makespan'].append(result['makespan'])
                all_results[algo_name]['solution'].append(result['solution'])
                all_results[algo_name]['events'].append(result['events'])
                all_results[algo_name]['time'].append(result['time'])
                all_results[algo_name]['best_hist'].append(result.get('best_hist', []))
                all_results[algo_name]['avg_hist'].append(result.get('avg_hist', []))
                
                if result['makespan'] < best_overall[algo_name]['makespan']:
                    best_overall[algo_name] = {
                        'makespan': result['makespan'],
                        'schedule': result['solution'],
                        'events': result['events'],
                        'sim_num': sim_i,
                        'emergencies': sim_emergencies
                    }
        
        return all_results, best_overall