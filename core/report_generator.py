"""
Centralizes all report and plot generation logic.
"""
import os
from utils import plotting, reporting, statistics
from config.config import get_algorithms, VERBOSE_MODE
from utils.logger import logger

class ReportGenerator:
    """
    Generates all reports and plots for simulation results.
    """
    
    def __init__(self):
        self.algorithms = get_algorithms()
    
    def generate_elective_reports(self, all_results, best_overall, output_dirs, all_rooms, alpha_test):
        """Generates reports for elective simulation mode."""
        logger.info(f"\n{'='*70}")
        logger.info("GENERATING ELECTIVE SIMULATION REPORTS")
        logger.info(f"{'='*70}")
        
        # CSV summary
        reporting.export_montecarlo_summary(
            all_results,
            os.path.join(output_dirs["csv"], "summary_results.csv")
        )
        
        # Statistical analysis
        pairwise_stats = statistics.perform_u_test_mannwhitney(all_results, alpha_test, verbose=VERBOSE_MODE)
        reporting.export_statistical_analysis(
            pairwise_stats,
            os.path.join(output_dirs["csv"], "statistical_analysis.csv")
        )
        
        # Summary plots
        plotting.generate_summary_plots(all_results, output_dirs["plots"])
        
        # Detailed reports per algorithm
        for algo_name, best_run in best_overall.items():
            if best_run['schedule']:
                self._generate_algorithm_reports(
                    algo_name, best_run, all_results[algo_name],
                    output_dirs, all_rooms, mode='elective'
                )
        
        logger.info(f"\n{'='*70}")
        logger.info("ELECTIVE SIMULATION REPORTS COMPLETE")
        logger.info(f"{'='*70}")
    
    def generate_emergency_reports(self, all_results, best_overall, output_dirs, all_rooms, alpha_test):
        """Generates reports for emergency simulation mode."""
        logger.info(f"\n{'='*70}")
        logger.info("GENERATING EMERGENCY SIMULATION REPORTS")
        logger.info(f"{'='*70}")
        
        # CSV summary
        reporting.export_emergency_montecarlo_summary(
            all_results,
            os.path.join(output_dirs["csv"], "emergency_summary_results.csv")
        )
        
        # Statistical analysis
        pairwise_stats = statistics.perform_u_test_mannwhitney(all_results, alpha_test, verbose=VERBOSE_MODE)
        reporting.export_statistical_analysis(
            pairwise_stats,
            os.path.join(output_dirs["csv"], "emergency_statistical_analysis.csv")
        )
        
        # Summary plots
        plotting.generate_emergency_summary_plots(all_results, output_dirs["plots"])
        
        # Detailed reports per algorithm
        for algo_name, best_run in best_overall.items():
            if best_run['schedule']:
                self._generate_algorithm_reports(
                    algo_name, best_run, all_results[algo_name],
                    output_dirs, all_rooms, mode='emergency'
                )
        
        logger.info(f"\n{'='*70}")
        logger.info("EMERGENCY SIMULATION REPORTS COMPLETE")
        logger.info(f"{'='*70}")
    
    def _generate_algorithm_reports(self, algo_name, best_run, algo_results, 
                                    output_dirs, all_rooms, mode='elective'):
        """Generates detailed reports for a single algorithm."""
        logger.info(f"\n> Processing {mode} results for: {algo_name}")
        logger.info(f"  -> Best run found in Sim #{best_run['sim_num'] + 1} (Makespan: {best_run['makespan']:.2f})")
        
        # Verify whether we have a detailed schedule
        schedule_data = best_run.get('schedule', [])
        
        if isinstance(schedule_data, list) and schedule_data:
            # CSV exports (only if we have a detailed schedule)
            reporting.export_full_schedule_to_csv(
                schedule_data,
                os.path.join(output_dirs["csv"], f"{mode}_best_schedule_{algo_name.lower()}.csv")
            )
            
            reporting.export_sequencing_strategy_to_csv(
                schedule_data,
                os.path.join(output_dirs["csv"], f"{mode}_best_strategy_{algo_name.lower()}.csv")
            )
            
            # Export emergency metrics (emergency mode only)
            if mode == 'emergency':
                events_data = best_run.get('events', [])
                emergencies_data = best_run.get('emergencies', [])
                
                if events_data and emergencies_data:
                    # Main CSV: integration metrics
                    reporting.export_emergency_metrics(
                        events_data,
                        emergencies_data,
                        os.path.join(output_dirs["csv"], f"emergency_integration_metrics_{algo_name.lower()}.csv")
                    )
                
                # Optional: full event log (future debugging)
                # reporting.export_emergency_event_log(
                #     events_data,
                #     os.path.join(output_dirs["csv"], f"emergency_event_log_{algo_name.lower()}.csv")
                # )
            
            # Gantt chart
            sim_num = best_run.get('sim_num', 0) + 1
            
            if mode == 'emergency':
                plotting.plot_gantt_chart(
                    schedule_data, all_rooms,
                    f"{algo_name} - Best Emergency Schedule (TSJS)",
                    algo_name, output_dirs["plots"],
                    sim_num=sim_num,
                    emergencies=best_run.get('emergencies', [])
                )
            else:
                plotting.plot_gantt_chart(
                    schedule_data, all_rooms,
                    f"{algo_name} - Best Elective Schedule",
                    algo_name, output_dirs["plots"],
                    sim_num=sim_num
                )
        else:
            logger.warning(f"  -> No detailed schedule available for {algo_name} (only makespan and convergence)")
        
        # Convergence plot (always available)
        if algo_results['best_hist'] and len(algo_results['best_hist']) > best_run['sim_num']:
            if algo_results['best_hist'][best_run['sim_num']]:
                plotting.plot_convergence_history(
                    algo_results['best_hist'][best_run['sim_num']],
                    algo_results['avg_hist'][best_run['sim_num']],
                    len(algo_results['best_hist'][best_run['sim_num']]),
                    algo_name,
                    best_run['sim_num'] + 1,
                    output_dirs["plots"]
                )