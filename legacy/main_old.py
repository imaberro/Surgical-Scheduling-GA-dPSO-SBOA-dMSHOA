# /main.py

import os
import sys
import time
import numpy as np

# Ensure project root is in sys.path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project modules
from config.config import (
    JOB_TYPES, ALL_ROOMS, GA_ENABLED, DPSO_ENABLED, SBOA_ENABLED, MSHOA_ENABLED,
    NUM_SIMULATIONS, STD_FACTOR, ALPHA_TEST, OUTPUT_DIRS, EMERGENCY_ENABLED, NUM_EMERGENCIES,
    VERBOSE_MODE
)
from data.data_generator import generate_day_surgeries_data
from simulation.scheduler import calculate_schedule_fitness
from algorithms import ga, dpso, sboa, dmshoa
from utils import plotting, reporting, statistics
from utils.logger import logger

# --- 1. Experiment Configuration ---
for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

# --- 2. Definition of Algorithms to Compare ---
_algorithm_specs = [
    {"name": "GA", "runner": ga.run, "iterations": ga.MAX_GENERATIONS, "enabled": GA_ENABLED},
    {"name": "dPSO", "runner": dpso.run, "iterations": dpso.MAX_ITERATIONS_DPSO, "enabled": DPSO_ENABLED},
    {"name": "SBOA", "runner": sboa.run, "iterations": sboa.SBOA_MAX_ITER, "enabled": SBOA_ENABLED},
    {"name": "dMShOA", "runner": dmshoa.run, "iterations": dmshoa.MAX_ITERATIONS_MSHOA, "enabled": MSHOA_ENABLED},
]

ALGORITHMS = [spec for spec in _algorithm_specs if spec["enabled"]]

def _regen_and_get_details(solution_repr, seed):
    """Regenerates simulation data and recalculates schedule details."""
    np.random.seed(seed)
    day_data = generate_day_surgeries_data(list(JOB_TYPES.keys()), std_factor=STD_FACTOR)
    _, actual_makespan, details = calculate_schedule_fitness(solution_repr, day_data, return_details=True)
    return actual_makespan, details

def _run_monte_carlo_simulations(job_ids):
    """Runs all simulations for all enabled algorithms."""
    logger.info(f"--- Starting {NUM_SIMULATIONS} Monte Carlo simulations ---")
    if not VERBOSE_MODE:
        logger.info("🔇 Silent mode enabled - showing progress only")
    
    all_results = {spec["name"]: {'makespan': [], 'solution': [], 'best_hist': [], 'avg_hist': [], 'time': []} for spec in ALGORITHMS}
    best_overall = {spec["name"]: {'mk': float('inf'), 'repr': None, 'seed': -1} for spec in ALGORITHMS}
    
    for sim_i in range(NUM_SIMULATIONS):
        if VERBOSE_MODE:
            logger.info(f"\n> Simulation {sim_i + 1}/{NUM_SIMULATIONS}")
        elif (sim_i + 1) % 10 == 0 or sim_i == 0:
            logger.info(f"Progress: {sim_i + 1}/{NUM_SIMULATIONS} simulations completed")
        
        np.random.seed(sim_i)
        day_data = generate_day_surgeries_data(job_ids, std_factor=STD_FACTOR)
        
        for spec in ALGORITHMS:
            algo_name = spec["name"]
            
            if VERBOSE_MODE:
                logger.info(f"\n--- Running Algorithm: {algo_name} ---")
            
            t0 = time.time()
            
            _, best_sol, best_hist, avg_hist = spec["runner"](day_data, job_ids, seed=sim_i)
            elapsed = time.time() - t0
            
            # NEW: Get both fitness and actual makespan
            if best_sol:
                combined_fitness, actual_makespan, _ = calculate_schedule_fitness(best_sol, day_data, return_details=True)
                if VERBOSE_MODE:
                    logger.info(f"  - {algo_name} Result: Makespan={actual_makespan:.2f}, Combined Fitness={combined_fitness:.2f}, Time={elapsed:.2f}s")
            else:
                actual_makespan = float('inf')
                combined_fitness = float('inf')
                if VERBOSE_MODE:
                    logger.info(f"  - {algo_name} Result: No valid solution found")
            
            if VERBOSE_MODE:
                logger.info(f"--- Finished {algo_name} ---")
            
            results = all_results[algo_name]
            results['makespan'].append(actual_makespan)
            results['solution'].append(best_sol)
            results['best_hist'].append(best_hist)
            results['avg_hist'].append(avg_hist)
            results['time'].append(elapsed)
            
            if actual_makespan < best_overall[algo_name]['mk']:
                best_overall[algo_name] = {'mk': actual_makespan, 'repr': best_sol, 'seed': sim_i}
    
    logger.info(f"\nAll {NUM_SIMULATIONS} simulations completed!")
    return all_results, best_overall


def _generate_summary_reports_and_plots(all_results):
    """Generates and saves reports and plots that compare all algorithms."""
    logger.info("\n--- Generating General Reports & Plots ---")
    pairwise_stats = statistics.perform_u_test_mannwhitney(all_results, ALPHA_TEST, verbose=VERBOSE_MODE)
    
    reporting.generate_summary_reports(all_results, pairwise_stats, OUTPUT_DIRS["csv"])
    plotting.generate_summary_plots(all_results, OUTPUT_DIRS["plots"])

def _generate_detailed_reports_and_plots(all_results, best_overall):
    """Generates and saves detailed reports for each algorithm's best run."""
    logger.info("\n--- Generating Detailed Reports per Algorithm ---")
    for spec in ALGORITHMS:
        name = spec["name"]
        logger.info(f"\n> Processing results for: {name}")

        best_run = best_overall[name]
        if best_run['repr']:
            sim_idx = best_run['seed']
            logger.info(f"  -> Best run found in Sim #{sim_idx + 1} (Makespan: {best_run['mk']:.2f}). Generating reports...")
            
            _, details = _regen_and_get_details(best_run['repr'], sim_idx)
            
            # Add ALL_ROOMS to the spec dictionary for plotting
            spec['all_rooms'] = ALL_ROOMS
            
            plotting.generate_detailed_plots_for_algo(all_results, name, spec, best_run, details, OUTPUT_DIRS["plots"])
            reporting.generate_detailed_reports_for_algo(details, name, OUTPUT_DIRS["csv"])
        else:
            logger.warning(f"  -> No valid global solution found for {name}.")

def _run_emergency_simulation(job_ids):
    """Runs simulation with dynamic emergencies (TSJS strategy)."""
    from simulation.dynamic_scheduler import DynamicScheduler
    from simulation.emergency_generator import generate_emergency_arrivals
    
    logger.info(f"\n{'='*70}")
    logger.info("EMERGENCY SIMULATION MODE (TSJS Strategy)")
    logger.info(f"{'='*70}")
    if not VERBOSE_MODE:
        logger.info("🔇 Silent mode enabled - showing progress only")
    
    # Run multiple simulations (Monte Carlo)
    all_results_emergency = {spec["name"]: {'makespan': [], 'solution': [], 'events': [], 'time': []} 
                             for spec in ALGORITHMS}
    best_overall_emergency = {spec["name"]: {'mk': float('inf'), 'schedule': None, 'events': None, 'seed': -1} 
                              for spec in ALGORITHMS}
    
    for sim_i in range(NUM_SIMULATIONS):
        if VERBOSE_MODE:
            logger.info(f"\n{'='*70}")
            logger.info(f"Emergency Simulation {sim_i + 1}/{NUM_SIMULATIONS}")
            logger.info(f"{'='*70}")
        elif (sim_i + 1) % 10 == 0 or sim_i == 0:
            logger.info(f"Progress: {sim_i + 1}/{NUM_SIMULATIONS} emergency simulations completed")
        
        emergencies = generate_emergency_arrivals(num_emergencies=NUM_EMERGENCIES, seed=1000 + sim_i)
        
        if VERBOSE_MODE:
            logger.info("\nEMERGENCY ARRIVALS SCHEDULE:")
            for em in emergencies:
                logger.info(f"  - {em['job_id']} (Type {em['job_type']}): t={em['arrival_time']:.2f} min")
        
        np.random.seed(sim_i)
        day_data = generate_day_surgeries_data(job_ids, std_factor=STD_FACTOR)
        
        # Run with each algorithm
        for spec in ALGORITHMS:
            algo_name = spec["name"]
            
            if VERBOSE_MODE:
                logger.info(f"\n{'='*70}")
                logger.info(f"Testing {algo_name} with Dynamic Emergencies (TSJS)")
                logger.info(f"{'='*70}")
            
            t0 = time.time()
            
            dynamic_scheduler = DynamicScheduler(
                algorithm_runner=spec["runner"],
                surgeries_data=day_data,
                job_ids=job_ids
            )
            
            final_schedule, events_log, final_makespan = dynamic_scheduler.run_with_emergencies(
                emergencies, seed=sim_i
            )
            
            elapsed = time.time() - t0
            
            if final_schedule:
                if VERBOSE_MODE:
                    logger.info(f"\n{algo_name} Emergency Simulation Complete:")
                    logger.info(f"   Final Makespan: {final_makespan:.2f} min")
                    logger.info(f"   Total Jobs Scheduled: {len(set(t['Job'] for t in final_schedule))}")
                
                # Store results
                results = all_results_emergency[algo_name]
                results['makespan'].append(final_makespan)
                results['solution'].append(final_schedule)
                results['events'].append(events_log)
                results['time'].append(elapsed)
                
                # Update best result
                if final_makespan < best_overall_emergency[algo_name]['mk']:
                    best_overall_emergency[algo_name] = {
                        'mk': final_makespan,
                        'schedule': final_schedule,
                        'events': events_log,
                        'seed': sim_i
                    }
            else:
                if VERBOSE_MODE:
                    logger.warning(f"{algo_name} failed to generate valid emergency schedule")
                
                results = all_results_emergency[algo_name]
                results['makespan'].append(float('inf'))
                results['solution'].append(None)
                results['events'].append([])
                results['time'].append(elapsed)
    
    logger.info(f"\nAll {NUM_SIMULATIONS} emergency simulations completed!")
    
    # Generate reports and plots
    _generate_emergency_reports_and_plots(all_results_emergency, best_overall_emergency, emergencies)
    
    return all_results_emergency

def _generate_emergency_reports_and_plots(all_results, best_overall, emergencies):
    """Generates reports and plots for simulations with emergencies."""
    logger.info(f"\n{'='*70}")
    logger.info("GENERATING EMERGENCY SIMULATION REPORTS")
    logger.info(f"{'='*70}")
    
    # 1. Generate summary reports (CSV)
    logger.info("\n--- Generating Emergency Summary Reports ---")
    
    # Summary CSV
    reporting.export_emergency_montecarlo_summary(
        all_results, 
        os.path.join(OUTPUT_DIRS["csv"], "emergency_summary_results.csv")
    )
    
    # Statistical analysis (pairwise Mann-Whitney)
    pairwise_stats = statistics.perform_u_test_mannwhitney(all_results, ALPHA_TEST, verbose=VERBOSE_MODE)
    reporting.export_statistical_analysis(
        pairwise_stats,
        os.path.join(OUTPUT_DIRS["csv"], "emergency_statistical_analysis.csv")
    )
    
    # 2. Generate comparison plots
    logger.info("\n--- Generating Emergency Comparison Plots ---")
    plotting.generate_emergency_summary_plots(all_results, OUTPUT_DIRS["plots"])
    
    # 3. Generate detailed reports for the best run of each algorithm
    logger.info("\n--- Generating Detailed Emergency Reports per Algorithm ---")
    for algo_name, best_run in best_overall.items():
        if best_run['schedule']:
            logger.info(f"\n> Processing emergency results for: {algo_name}")
            logger.info(f"  -> Best run found in Sim #{best_run['seed'] + 1} (Makespan: {best_run['mk']:.2f})")
            
            # Detailed CSV
            reporting.export_full_schedule_to_csv(
                best_run['schedule'],
                os.path.join(OUTPUT_DIRS["csv"], f"emergency_best_schedule_{algo_name.lower()}.csv")
            )
            
            reporting.export_sequencing_strategy_to_csv(
                best_run['schedule'],
                os.path.join(OUTPUT_DIRS["csv"], f"emergency_best_strategy_{algo_name.lower()}.csv")
            )
            
            # Gantt chart with emergency markers
            plotting.plot_gantt_with_emergencies(
                best_run['schedule'],
                ALL_ROOMS,
                f"{algo_name} - Best Emergency Schedule (TSJS)",
                algo_name,
                OUTPUT_DIRS["plots"],
                emergencies,
                verbose=VERBOSE_MODE
            )
            
            # Event log
            reporting.export_emergency_event_log(
                best_run['events'],
                os.path.join(OUTPUT_DIRS["csv"], f"emergency_event_log_{algo_name.lower()}.csv")
            )
        else:
            logger.warning(f"  -> No valid emergency solution found for {algo_name}")
    
    logger.info(f"\n{'='*70}")
    logger.info("EMERGENCY SIMULATION REPORTS COMPLETE")
    logger.info(f"{'='*70}")

def main():
    """Orchestrates the Monte Carlo experiment from start to finish."""
    start_monte = time.time()
    
    if not ALGORITHMS:
        logger.warning("No algorithms enabled in the configuration. Exiting.")
        return

    # Check whether emergency mode is enabled
    if EMERGENCY_ENABLED:
        _run_emergency_simulation(list(JOB_TYPES.keys()))
    else:
        # Normal mode (no emergencies)
        # 1. Run all simulations
        all_results, best_overall = _run_monte_carlo_simulations(list(JOB_TYPES.keys()))

        # 2. Generate summary reports and plots
        _generate_summary_reports_and_plots(all_results)

        # 3. Generate detailed reports and plots for best runs
        _generate_detailed_reports_and_plots(all_results, best_overall)
            
    logger.info(f"\nProcess completed! (Total time: {time.time() - start_monte:.2f}s). Check the 'results' folder.")

if __name__ == "__main__":
    main()