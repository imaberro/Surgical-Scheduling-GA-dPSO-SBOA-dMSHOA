# /utils/reporting.py
"""
Module for reporting functions: printing summaries to the console and
exporting schedule results to CSV files.
"""
import csv
import numpy as np
import os

# Import necessary constants from the configuration file
from config.config import ALL_ROOMS

# Note: 'EMERGENCY_JOBS' should be in your config.py.
# For now, it's defined here as an empty list as a fallback.
try:
    from config.config import EMERGENCY_JOBS
except ImportError:
    EMERGENCY_JOBS = []

from utils.logger import logger


# ==== Internal Generic Helpers ====

def _format_two_decimals(value):
    """Returns a value formatted to two decimal places if numeric, otherwise returns it as is."""
    return f"{value:.2f}" if isinstance(value, (int, float, np.floating)) else value

def _build_room_schedules(schedule_details):
    """Groups operations by room to facilitate reporting."""
    room_schedules = {room: [] for room in ALL_ROOMS}
    for t in schedule_details:
        room = t.get('Resource')
        if room:
            room_schedules.setdefault(room, []).append((t.get('Start', -1), t.get('Job'), t.get('Operation')))
    return room_schedules

def _build_job_timetables(schedule_details):
    """Structures start times by surgery and operation."""
    job_timetables = {}
    for t in schedule_details:
        job = t.get('Job')
        if job is not None:
            job_timetables.setdefault(job, {})[t.get('Operation')] = t.get('Start', -1)
    return job_timetables

def _safe_open_csv(filename):
    """Context manager to safely open CSV files."""
    return open(filename, 'w', newline='', encoding='utf-8')


# ==== Console Printing Functions ====

def print_schedule_summary(schedule_details, title="Schedule"):
    """Prints a detailed summary of the schedule, ordered by time."""
    if not schedule_details: return
    print(f"\n--- Schedule Summary: {title} ---")
    header = f"{'Surgery (E)':<12} | {'Operation':^10} | {'Room':<12} | {'Personnel':<12} | {'Start':>10} | {'Finish':>10} | {'Duration':>10}"
    print(header); print("-" * len(header))
    
    for task in sorted(schedule_details, key=lambda x: x.get('Start', float('inf'))):
        job_label = f"{task.get('Job')} (E)" if task.get('Job') in EMERGENCY_JOBS else str(task.get('Job'))
        start, finish = task.get('Start', 0.0), task.get('Finish', 0.0)
        duration = finish - start
        print(f"{job_label:<12} | {int(task.get('Operation', 0)):^10d} | {str(task.get('Resource','')):<12} | "
              f"{str(task.get('Personnel', 'N/A')):<12} | {start:>10.2f} | {finish:>10.2f} | {duration:>10.2f}")
    print("-" * len(header))

def print_sequencing_strategy(schedule_details):
    """Displays the sequence of surgeries assigned to each room."""
    if not schedule_details: return
    print("\n--- Sequencing Strategy by Room ---")
    room_schedules = _build_room_schedules(schedule_details)
    print(f"{'Room':<12} | {'Operation Sequence (Surgery(Op))'}")
    print("-" * 80)
    for room_name in sorted(room_schedules.keys()):
        schedule = sorted(room_schedules[room_name], key=lambda x: x[0])
        seq = [f"{j}{'(E)' if j in EMERGENCY_JOBS else ''}(Op{o})" for _, j, o in schedule]
        print(f"{room_name:<12} | {' -> '.join(seq) if seq else 'No assignments'}")
    print("-" * 80)


# ==== High-Level Report Generation Functions ====

def generate_summary_reports(all_results, pairwise_stats, output_dir):
    """Generates and saves all summary CSV files."""
    logger.info("  -> Generating summary CSV files...")
    
    summary_path = export_montecarlo_summary(all_results, os.path.join(output_dir, "summary_results.csv"))
    if summary_path: logger.info(f"    - Monte Carlo summary saved to: {summary_path}")

    stats_path = export_statistical_analysis(pairwise_stats, os.path.join(output_dir, "statistical_analysis.csv"))
    if stats_path: logger.info(f"    - Statistical analysis saved to: {stats_path}")

def generate_detailed_reports_for_algo(details, name, output_dir):
    """Generates and saves all detailed CSV files for a single algorithm's best run."""
    schedule_csv = export_full_schedule_to_csv(details, os.path.join(output_dir, f"best_schedule_{name.lower()}.csv"))
    strategy_csv = export_sequencing_strategy_to_csv(details, os.path.join(output_dir, f"best_strategy_{name.lower()}.csv"))

    if schedule_csv: logger.info(f"    - Full schedule: {schedule_csv}")
    if strategy_csv: logger.info(f"    - Sequencing strategy: {strategy_csv}")


# ==== Low-Level CSV Export Functions ====

def export_full_schedule_to_csv(schedule_details, filename):
    """Exports the full detailed schedule to a CSV file, returning the filename on success."""
    if not schedule_details:
        print(f"  -> [Warning] No data to export to {filename}.")
        return None
    try:
        with _safe_open_csv(filename) as csvfile:
            fieldnames = list(schedule_details[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for task in schedule_details:
                writer.writerow({k: _format_two_decimals(v) for k, v in task.items()})
        
        # NEW: Calculate and print the TRUE makespan from the CSV data
        max_finish = max(task.get('Finish', 0) for task in schedule_details)
        # print(f"  -> [Debug] TRUE MAKESPAN from schedule_details: {max_finish:.2f}")
        
        return filename
    except (IOError, IndexError) as e:
        print(f"  -> [Error] Exporting schedule to CSV failed: {e}")
        return None

def export_sequencing_strategy_to_csv(schedule_details, filename):
    """Exports the operation sequence per room to a CSV file, returning the filename on success."""
    if not schedule_details: return
    room_schedules = _build_room_schedules(schedule_details)
    try:
        with _safe_open_csv(filename) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Room', 'Operation_Sequence'])
            for room_name in sorted(room_schedules.keys()):
                schedule = sorted(room_schedules[room_name], key=lambda x: x[0])
                seq = ' -> '.join([f"{j}{'(E)' if j in EMERGENCY_JOBS else ''}(Op{o})" for _, j, o in schedule])
                writer.writerow([room_name, seq])
        return filename
    except IOError as e:
        print(f"  -> [Error] Exporting strategy to CSV failed: {e}")
        return None

def export_montecarlo_summary(all_results, filename):
    """Exports the Monte Carlo summary, returning the filename on success."""
    try:
        with _safe_open_csv(filename) as f:
            writer = csv.writer(f)
            writer.writerow(['algorithm', 'valid_simulations', 'makespan_min', 'makespan_median', 
                           'makespan_avg', 'makespan_std', 'time_avg_s'])
            
            for name, results in all_results.items():
                makespans = [m for m in results['makespan'] if m != float('inf')]
                times = results['time']
                
                if not makespans:
                    continue
                
                writer.writerow([
                    name, 
                    len(makespans),
                    _format_two_decimals(np.min(makespans)),
                    _format_two_decimals(np.median(makespans)),
                    _format_two_decimals(np.mean(makespans)),
                    _format_two_decimals(np.std(makespans, ddof=1)) if len(makespans) > 1 else 0.0,
                    _format_two_decimals(np.mean(times))
                ])
        
        logger.info(f"    - Monte Carlo summary saved to: {filename}")
        return filename
    except IOError as e:
        logger.error(f"  -> [Error] Exporting summary failed: {e}")
        return None

def export_statistical_analysis(comparison_results, filename):
    """Exports the detailed results of the pairwise comparison, returning the filename on success."""
    if not comparison_results: return
    try:
        with _safe_open_csv(filename) as f:
            fieldnames = next((res.keys() for res in comparison_results if res), [])
            if not fieldnames: return
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in comparison_results:
                formatted_row = {}
                for k, v in row.items():
                    if k == 'p_value':
                        if isinstance(v, (int, float, np.floating)):
                            formatted_row[k] = '≥0.05' if v >= 0.05 else f"{v:.4f}"
                        else:
                            formatted_row[k] = v
                    else:
                        formatted_row[k] = _format_two_decimals(v)
                
                writer.writerow(formatted_row)
        return filename
    except (IOError, IndexError) as e:
        print(f"  -> [Error] Exporting analysis failed: {e}")
        return None

def export_emergency_montecarlo_summary(all_results, filename):
    """Exports the emergency Monte Carlo summary, returning the filename on success."""
    try:
        with _safe_open_csv(filename) as f:
            writer = csv.writer(f)
            writer.writerow(['algorithm', 'valid_simulations', 'makespan_min', 'makespan_median',
                           'makespan_avg', 'makespan_std', 'time_avg_s'])
            
            for name, results in all_results.items():
                makespans = [m for m in results['makespan'] if m != float('inf')]
                times = results['time']
                
                if not makespans:
                    continue
                
                # Count successfully integrated emergencies
                avg_emergencies = np.mean([
                    len(events) for events in results['events'] if events
                ])
                
                writer.writerow([
                    name, 
                    len(makespans),
                    _format_two_decimals(np.min(makespans)),
                    _format_two_decimals(np.median(makespans)),
                    _format_two_decimals(np.mean(makespans)),
                    _format_two_decimals(np.std(makespans, ddof=1)) if len(makespans) > 1 else 0.0,
                    _format_two_decimals(np.mean(times)),
                ])
        
        logger.info(f"    - Emergency Monte Carlo summary saved to: {filename}")
        return filename
    except IOError as e:
        logger.error(f"  -> [Error] Exporting emergency summary failed: {e}")
        return None

def export_emergency_event_log(events_log, filename):
    """Exports the emergency event log to CSV."""
    if not events_log:
        logger.warning(f"  -> [Warning] No event log data to export to {filename}.")
        return None
    
    try:
        with _safe_open_csv(filename) as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Event_Type', 'Details'])
            
            for event in events_log:
                time_val = event.get('time', 0)  # This should be the original arrival_time
                
                # DEBUG: Verify time_val is correct
                print(f"DEBUG - Exporting event: time={time_val:.2f}, type={event['type']}")
                
                writer.writerow([
                    _format_two_decimals(time_val),
                    event['type'],
                    event['details']
                ])
        
        logger.info(f"    - Emergency event log saved to: {filename}")
        return filename
    except IOError as e:
        logger.error(f"  -> [Error] Exporting event log failed: {e}")
        return None

def export_emergency_metrics(events_log, emergencies, filename):
    """
    Exports emergency integration metrics to CSV.
    
    Args:
        events_log: List of events from DynamicScheduler
        emergencies: Original emergency data with arrival_time
        filename: Output CSV path
    
    Returns:
        str: Filename on success, None on error
    """
    if not events_log or not emergencies:
        logger.warning(f"  -> [Warning] No emergency metrics to export to {filename}.")
        return None
    
    try:
        with _safe_open_csv(filename) as f:
            writer = csv.writer(f)
            # NOTE: 'Status' column removed
            writer.writerow([
                'Emergency_Job_ID', 
                'Arrival_Time', 
                'Integration_Time', 
                'Integration_Delay'
            ])
            
            # Map emergencies by job_id
            emergency_map = {em['job_id']: em for em in emergencies}
            integrated_jobs = {}
            
            # Collect EMERGENCY_RESCHEDULING events
            for event in events_log:
                if event.get('type') == 'EMERGENCY_RESCHEDULING':
                    emergency_job = event.get('emergency_job')
                    emergency_start = event.get('emergency_start')
                    delay = event.get('delay')
                    
                    if emergency_job and emergency_job in emergency_map:
                        arrival_time = emergency_map[emergency_job]['arrival_time']
                        
                        # If emergency_start is available, use it; otherwise arrival_time + delay
                        if emergency_start is not None:
                            integration_time = emergency_start
                        elif delay is not None:
                            integration_time = arrival_time + delay
                        else:
                            integration_time = arrival_time
                        
                        # Compute delay if missing
                        if delay is None:
                            delay = integration_time - arrival_time
                        
                        integrated_jobs[emergency_job] = {
                            'arrival': arrival_time,
                            'integration': integration_time,
                            'delay': delay
                        }
            
            # Write integrated emergencies
            for job_id, metrics in integrated_jobs.items():
                writer.writerow([
                    job_id,
                    _format_two_decimals(metrics['arrival']),
                    _format_two_decimals(metrics['integration']),
                    _format_two_decimals(metrics['delay'])
                    # 'Integrated' column removed
                ])
            
            # Optional: Non-integrated emergencies (future)
            # for em in emergencies:
            #     if em['job_id'] not in integrated_jobs:
            #         writer.writerow([
            #             em['job_id'],
            #             _format_two_decimals(em['arrival_time']),
            #             'N/A',
            #             'N/A'
            #         ])
        
        logger.info(f"    - Emergency metrics saved to: {filename}")
        return filename
    except IOError as e:
        logger.error(f"  -> [Error] Exporting emergency metrics failed: {e}")
        return None