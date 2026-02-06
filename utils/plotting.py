"""
Plotting utilities with DRY principles applied.
All visualization logic is abstracted into reusable components.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
from typing import List, Dict, Optional, Tuple

from config.config import JOB_TYPES, SETUP_TIMES, CLEANUP_TIMES, VERBOSE_MODE
from utils.logger import logger

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

ALGORITHM_COLORS = {
    'GA': 'lightblue',
    'dPSO': 'lightsalmon',
    'SBOA': 'lightgreen',
    'dMShOA': 'lightcoral'
}

BOXPLOT_COLORS = ['lightblue', 'lightsalmon', 'lightgreen', 'lightcoral']
ALGORITHM_ORDER = ['GA', 'dPSO', 'SBOA', 'dMShOA']

# Gantt chart styling
GANTT_STYLE = {
    'bar_height': 0.25,
    'room_spacing': 0.25,
    'group_gap': 0.75,
    'edge_color': 'black',
    'edge_width': 0.5,
    'alpha': 0.9,
    'setup_color': 'yellow',
    'cleanup_color': 'lightcoral',
    'emergency_color': 'red',
    'emergency_edge_width': 2.0,
    'unused_room_color': 'lightgray',
    'unused_room_alpha': 0.2
}

# =============================================================================
# CORE PLOTTING COMPONENTS (DRY ABSTRACTIONS)
# =============================================================================

class PlotConfig:
    """Base configuration for all plots."""
    
    DEFAULT_DPI = 150
    DEFAULT_FIGSIZE = (10, 6)
    GANTT_FIGSIZE = (12, 6)
    
    @staticmethod
    def get_output_paths(output_dir: str, subdir: str, filename: str) -> Tuple[str, str]:
        """
        Returns (png_path, svg_path) for a given plot.
        Creates directories if needed.
        """
        plot_dir = os.path.join(output_dir, subdir)
        os.makedirs(plot_dir, exist_ok=True)
        
        svg_dir = os.path.join(plot_dir, "svg")
        os.makedirs(svg_dir, exist_ok=True)
        
        png_path = os.path.join(plot_dir, f"{filename}.png")
        svg_path = os.path.join(svg_dir, f"{filename}.svg")
        
        return png_path, svg_path
    
    @staticmethod
    def save_and_close(fig, png_path: str, svg_path: str, dpi: int = DEFAULT_DPI):
        """Saves figure to both PNG and SVG, then closes it."""
        plt.tight_layout(pad=0.2)
        plt.savefig(png_path, dpi=dpi)
        plt.savefig(svg_path)
        plt.close(fig)


class GanttChartBuilder:
    """
    Builds Gantt charts with DRY principles.
    Handles both elective and emergency simulations.
    """
    
    def __init__(self, schedule_details: List[Dict], rooms: List[str], 
                 emergencies: Optional[List[Dict]] = None):
        self.schedule_details = schedule_details
        self.rooms = rooms
        self.emergencies = emergencies or []
        self.is_emergency_mode = len(self.emergencies) > 0
        
        # Computed properties
        self.job_colors = self._compute_job_colors()
        self.y_positions = self._compute_y_positions()
        self.max_time = self._compute_max_time()
        self.rooms_with_tasks = set(t['Resource'] for t in schedule_details)
        self.unused_rooms = set(rooms) - self.rooms_with_tasks
    
    def _compute_job_colors(self) -> Dict:
        """Assigns colors to jobs (RED for emergencies)."""
        all_jobs = set(t['Job'] for t in self.schedule_details)
        int_jobs = sorted([j for j in all_jobs if isinstance(j, int)])
        str_jobs = sorted([j for j in all_jobs if isinstance(j, str)])
        unique_jobs = int_jobs + str_jobs
        
        try:
            cmap = plt.get_cmap('tab20' if len(unique_jobs) <= 20 else 'turbo')
        except Exception:
            cmap = plt.get_cmap('tab10')
        
        color_list = [cmap(i / max(1, len(unique_jobs) - 1)) for i in range(len(unique_jobs))]
        job_colors = {job_id: color_list[i] for i, job_id in enumerate(unique_jobs)}
        
        # Override: Emergency jobs always RED
        for job_id in str_jobs:
            if str(job_id).startswith('E'):
                job_colors[job_id] = GANTT_STYLE['emergency_color']
        
        return job_colors
    
    def _compute_y_positions(self) -> Tuple[Dict[str, float], List[str]]:
        """Computes y-axis positions for rooms with group separations."""
        room_categories = {'APR': [], 'OR': [], 'ARR': []}
        
        for room in self.rooms:
            if room.startswith('APR'):
                room_categories['APR'].append(room)
            elif room.startswith('OR'):
                room_categories['OR'].append(room)
            else:
                room_categories['ARR'].append(room)
        
        y_pos, y_labels = {}, []
        y = 0
        
        for group in ['APR', 'OR', 'ARR']:
            for room in sorted(room_categories[group]):
                y_pos[room] = y
                y_labels.append(room)
                y += GANTT_STYLE['room_spacing']
            y += GANTT_STYLE['group_gap']
        
        return y_pos, y_labels
    
    def _compute_max_time(self) -> float:
        """Computes makespan from schedule."""
        if not self.schedule_details:
            return 0
        return max(t.get('Finish', 0) for t in self.schedule_details if t.get('Finish', -1) >= 0)
    
    def _is_emergency_job(self, job_id) -> bool:
        """Checks if a job is an emergency."""
        return isinstance(job_id, str) and str(job_id).startswith('E')
    
    def _draw_task_bar(self, ax, task: Dict):
        """Draws setup + processing + cleanup bars for a single task."""
        job = task['Job']
        resource = task['Resource']
        start = task.get('Start', -1)
        processing_end = task.get('ProcessingEnd', -1)
        finish = task.get('Finish', -1)
        
        if start < 0 or processing_end < 0 or finish < 0 or resource not in self.y_positions[0]:
            if VERBOSE_MODE:
                logger.warning(f"Skipping invalid task: Job={job}, Resource={resource}")
            return
        
        job_type = JOB_TYPES.get(job, 1)
        setup_time = SETUP_TIMES.get(job_type, 0)
        
        setup_duration = setup_time
        proc_duration = processing_end - (start + setup_time)
        cleanup_duration = finish - processing_end
        
        if min(setup_duration, proc_duration, cleanup_duration) < -1e-6:
            if VERBOSE_MODE:
                logger.warning(f"Negative duration: Job={job}")
            return
        
        y = self.y_positions[0][resource]
        is_emergency = self._is_emergency_job(job)
        
        edge_width = GANTT_STYLE['emergency_edge_width'] if is_emergency else GANTT_STYLE['edge_width']
        edge_color = 'darkred' if is_emergency else GANTT_STYLE['edge_color']
        
        # Setup bar
        if setup_duration > 1e-6:
            ax.barh(y=y, width=setup_duration, left=start, height=GANTT_STYLE['bar_height'],
                   color=GANTT_STYLE['setup_color'], edgecolor=edge_color, 
                   linewidth=edge_width, alpha=GANTT_STYLE['alpha'])
        
        # Processing bar
        if proc_duration > 1e-6:
            proc_start = start + setup_duration
            proc_color = self.job_colors.get(job, 'gray')
            
            ax.barh(y=y, width=proc_duration, left=proc_start, height=GANTT_STYLE['bar_height'],
                   color=proc_color, edgecolor=edge_color, 
                   linewidth=edge_width, alpha=GANTT_STYLE['alpha'])
            
            # Job label
            if proc_duration > 20:
                try:
                    if is_emergency:
                        text_color = 'white'
                    else:
                        luminance = np.dot(mcolors.to_rgb(proc_color), [0.299, 0.587, 0.114])
                        text_color = 'white' if luminance < 0.5 else 'black'
                except Exception:
                    text_color = 'black'
                
                ax.text(proc_start + proc_duration/2, y, f"{job}", 
                       ha='center', va='center', color=text_color,
                       fontweight='bold', fontsize=8 if is_emergency else 7, clip_on=True)
        
        # Cleanup bar
        if cleanup_duration > 1e-6:
            ax.barh(y=y, width=cleanup_duration, left=processing_end, 
                   height=GANTT_STYLE['bar_height'],
                   color=GANTT_STYLE['cleanup_color'], edgecolor=edge_color,
                   linewidth=edge_width, alpha=GANTT_STYLE['alpha'])
    
    def _draw_unused_rooms(self, ax):
        """Draws placeholder bars for unused rooms."""
        if self.max_time <= 0:
            return
        
        for room in self.unused_rooms:
            if room in self.y_positions[0]:
                y = self.y_positions[0][room]
                ax.barh(y=y, width=self.max_time * 0.01, left=0, 
                       height=GANTT_STYLE['bar_height'],
                       color=GANTT_STYLE['unused_room_color'], 
                       edgecolor='gray', linewidth=0.3,
                       alpha=GANTT_STYLE['unused_room_alpha'], linestyle='--')
                
                ax.text(self.max_time * 0.005, y, "UNUSED", 
                       ha='left', va='center', color='gray',
                       fontsize=6, style='italic', alpha=0.6)
    
    def _draw_emergency_markers(self, ax):
        """Draws vertical lines at emergency arrival times."""
        if not self.emergencies:
            return
        
        arrival_times_drawn = set()
        top_y = min(self.y_positions[0].values()) if self.y_positions[0] else 0
        
        for em in self.emergencies:
            arrival_time = em['arrival_time']
            
            if VERBOSE_MODE:
                logger.info(f"  Drawing emergency line: {em['job_id']} at t={arrival_time:.2f}")
            
            if arrival_time not in arrival_times_drawn:
                ax.axvline(x=arrival_time, color='orange', linestyle='--', 
                          linewidth=2, alpha=0.7, zorder=5)
                
                ax.text(arrival_time, top_y - 0.5, f"{em['job_id']}", 
                       rotation=90, va='bottom', ha='right',
                       color='darkorange', fontsize=8, fontweight='bold', alpha=0.9)
                
                arrival_times_drawn.add(arrival_time)
    
    def _configure_axes(self, ax, title: str, sim_num: Optional[int] = None):
        """Configures axes labels, title, ticks, and grid."""
        y_pos, y_labels = self.y_positions
        
        ax.set_yticks([y_pos[r] for r in y_labels])
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel("Time (minutes)", fontsize=10)
        ax.set_ylabel("Resource (Room)", fontsize=10)
        
        # Title with simulation number
        if sim_num is not None:
            title_text = f"{title} (Best: Sim #{sim_num}) - Makespan: {self.max_time:.2f} min"
        else:
            suffix = " (Emergencies in RED)" if self.is_emergency_mode else ""
            title_text = f"{title}{suffix} - Makespan: {self.max_time:.2f} min"
        
        ax.set_title(title_text, fontsize=12, fontweight='bold')
        
        # X-axis limits and ticks
        if self.max_time > 0:
            ax.set_xlim(0, self.max_time * 1.08)
            
            # Dynamic tick interval
            if self.max_time <= 500:
                tick_interval = 50
            elif self.max_time <= 1000:
                tick_interval = 100
            elif self.max_time <= 2000:
                tick_interval = 200
            else:
                tick_interval = 250
            
            base_ticks = list(range(0, int(self.max_time) + 1, tick_interval))
            
            # Add makespan if far enough from last tick
            if base_ticks and abs(self.max_time - base_ticks[-1]) >= tick_interval * 0.3:
                base_ticks.append(self.max_time)
            elif not base_ticks:
                base_ticks.append(self.max_time)
            
            ax.set_xticks(base_ticks)
            ax.set_xticklabels([f'{int(t)}' if abs(t - self.max_time) > 0.5 
                               else f'{self.max_time:.0f}' for t in base_ticks])
        else:
            ax.set_xlim(0, 100)
        
        ax.invert_yaxis()
        ax.grid(True, axis='x', linestyle=':', color='gray', alpha=0.6)
        
        # Group separators
        room_categories = {'OR': [], 'ARR': []}
        for room in self.rooms:
            if room.startswith('OR'):
                room_categories['OR'].append(room)
            elif room.startswith('ARR'):
                room_categories['ARR'].append(room)
        
        if room_categories['OR']:
            ax.axhline(y=y_pos[room_categories['OR'][0]] - GANTT_STYLE['group_gap']/2,
                      color='black', linestyle='--', linewidth=0.6, alpha=0.4)
        if room_categories['ARR']:
            ax.axhline(y=y_pos[room_categories['ARR'][0]] - GANTT_STYLE['group_gap']/2,
                      color='black', linestyle='--', linewidth=0.6, alpha=0.4)
    
    def _add_legend(self, ax):
        """Adds legend with segment types."""
        if self.is_emergency_mode:
            handles = [
                Patch(facecolor=GANTT_STYLE['setup_color'], edgecolor='black', label='Setup'),
                Patch(facecolor='grey', edgecolor='black', label='Processing (Elective)'),
                Patch(facecolor=GANTT_STYLE['emergency_color'], edgecolor='darkred', 
                     linewidth=2, label='Processing (Emergency)'),
                Patch(facecolor=GANTT_STYLE['cleanup_color'], edgecolor='black', label='Cleanup'),
                plt.Line2D([0], [0], color='orange', linewidth=2, linestyle='--', 
                          label='Emergency Arrival')
            ]
        else:
            handles = [
                Patch(facecolor=GANTT_STYLE['setup_color'], edgecolor='black', label='Setup'),
                Patch(facecolor='grey', edgecolor='black', label='Processing (Color per Job)'),
                Patch(facecolor=GANTT_STYLE['cleanup_color'], edgecolor='black', label='Cleanup')
            ]
        
        ax.legend(handles=handles, loc='upper right', fontsize='small', 
                 title="Legend:" if self.is_emergency_mode else "Segments:")
    
    def build(self, title: str, sim_num: Optional[int] = None) -> plt.Figure:
        """
        Builds and returns the complete Gantt chart figure.
        
        Args:
            title: Chart title
            sim_num: Simulation number (optional)
        
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=PlotConfig.GANTT_FIGSIZE)
        
        # Draw all components
        for task in self.schedule_details:
            self._draw_task_bar(ax, task)
        
        self._draw_unused_rooms(ax)
        
        if self.is_emergency_mode:
            self._draw_emergency_markers(ax)
        
        self._configure_axes(ax, title, sim_num)
        self._add_legend(ax)
        
        return fig


# =============================================================================
# HIGH-LEVEL PLOTTING FUNCTIONS (PUBLIC API)
# =============================================================================

def plot_gantt_chart(schedule_details: List[Dict], rooms: List[str], 
                     title: str, algo_name: str, output_dir: str,
                     sim_num: Optional[int] = None,
                     emergencies: Optional[List[Dict]] = None) -> Tuple[str, str]:
    """
    Unified Gantt chart plotter for both elective and emergency modes.
    
    Args:
        schedule_details: List of task dictionaries
        rooms: List of room names
        title: Chart title
        algo_name: Algorithm name (for filename)
        output_dir: Output directory
        sim_num: Simulation number (optional, for title)
        emergencies: List of emergency events (if emergency mode)
    
    Returns:
        Tuple of (png_path, svg_path)
    """
    if not schedule_details:
        logger.warning(f"No schedule details for {algo_name} Gantt chart")
        return None, None
    
    # Determine mode and filename
    is_emergency = emergencies is not None and len(emergencies) > 0
    prefix = "emergency_gantt" if is_emergency else "best_gantt"
    filename = f"{prefix}_{algo_name.lower()}"
    
    # Build chart
    builder = GanttChartBuilder(schedule_details, rooms, emergencies)
    fig = builder.build(title, sim_num)
    
    # Save
    png_path, svg_path = PlotConfig.get_output_paths(output_dir, "gantt", filename)
    PlotConfig.save_and_close(fig, png_path, svg_path)
    
    return png_path, svg_path


def plot_boxplot(all_results: Dict, output_dir: str, mode: str = 'elective') -> Tuple[str, str]:
    """
    Unified boxplot for both elective and emergency modes.
    
    Args:
        all_results: Dictionary with algorithm results
        output_dir: Output directory
        mode: 'elective' or 'emergency'
    
    Returns:
        Tuple of (png_path, svg_path)
    """
    if not all_results:
        logger.warning(f"No results for {mode} boxplot")
        return None, None
    
    fig, ax = plt.subplots(figsize=PlotConfig.DEFAULT_FIGSIZE)
    
    data_to_plot = []
    labels = []
    
    for algo_name in ALGORITHM_ORDER:
        if algo_name in all_results:
            makespans = [mk for mk in all_results[algo_name]['makespan'] if mk != float('inf')]
            if makespans:
                data_to_plot.append(makespans)
                labels.append(algo_name)
    
    if not data_to_plot:
        logger.warning(f"No valid data for {mode} boxplot")
        plt.close(fig)
        return None, None
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], BOXPLOT_COLORS[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Makespan (minutes)', fontsize=11)
    title = f"{mode.capitalize()} Simulation: Makespan Comparison"
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    filename = f"{mode}_makespan_comparison"
    png_path, svg_path = PlotConfig.get_output_paths(output_dir, "boxplot", filename)
    PlotConfig.save_and_close(fig, png_path, svg_path)
    
    return png_path, svg_path


def plot_execution_time_barplot(all_results: Dict, output_dir: str, 
                                mode: str = 'elective') -> str:
    """
    Unified execution time barplot for both modes.
    
    Args:
        all_results: Dictionary with algorithm results
        output_dir: Output directory
        mode: 'elective' or 'emergency'
    
    Returns:
        Path to saved PNG file
    """
    fig, ax = plt.subplots(figsize=PlotConfig.DEFAULT_FIGSIZE)
    
    avg_times = []
    algo_names = []
    colors = []
    
    for algo_name in ALGORITHM_ORDER:
        if algo_name in all_results and all_results[algo_name].get('time'):
            avg_times.append(np.mean(all_results[algo_name]['time']))
            algo_names.append(algo_name)
            colors.append(ALGORITHM_COLORS[algo_name])
    
    if not avg_times:
        plt.close(fig)
        return None
    
    bars = ax.bar(algo_names, avg_times, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Average Execution Time (seconds)', fontsize=11)
    title = f"{mode.capitalize()} Simulation: Average Execution Time per Algorithm"
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    filename = f"{mode}_execution_time"
    png_path, svg_path = PlotConfig.get_output_paths(output_dir, "barplot", filename)
    PlotConfig.save_and_close(fig, png_path, svg_path)
    
    return png_path


def plot_makespan_histogram(data: List[float], algo_name: str, 
                            output_dir: str, mode: str = 'elective') -> Tuple[str, str]:
    """
    Unified histogram for both modes.
    
    Args:
        data: List of makespan values
        algo_name: Algorithm name
        output_dir: Output directory
        mode: 'elective' or 'emergency'
    
    Returns:
        Tuple of (png_path, svg_path)
    """
    if not data:
        logger.warning(f"No data for {algo_name} histogram")
        return None, None
    
    fig, ax = plt.subplots(figsize=PlotConfig.DEFAULT_FIGSIZE)
    
    hist_color = ALGORITHM_COLORS.get(algo_name, 'steelblue')
    
    ax.hist(data, bins=20, color=hist_color, edgecolor='black', 
           alpha=0.8, linewidth=1.2)
    
    mean_val = np.mean(data)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_val:.2f}')
    
    median_val = np.median(data)
    ax.axvline(median_val, color='darkgreen', linestyle='--', linewidth=2,
              label=f'Median: {median_val:.2f}')
    
    ax.set_xlabel('Makespan (minutes)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    title = f"{mode.capitalize()} Simulation: {algo_name} - Makespan Distribution"
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    prefix = "emergency_histogram" if mode == 'emergency' else "histogram"
    filename = f"{prefix}_{algo_name.lower()}"
    png_path, svg_path = PlotConfig.get_output_paths(output_dir, "histograms", filename)
    PlotConfig.save_and_close(fig, png_path, svg_path)
    
    return png_path, svg_path


def plot_convergence_history(best_history: List[float], avg_history: List[float],
                             max_iters: int, algo_name: str, sim_num: int,
                             output_dir: str) -> Tuple[str, str]:
    """
    Plots convergence curve (same for both modes).
    
    Args:
        best_history: Best fitness history
        avg_history: Average fitness history
        max_iters: Maximum iterations
        algo_name: Algorithm name
        sim_num: Simulation number
        output_dir: Output directory
    
    Returns:
        Tuple of (png_path, svg_path)
    """
    fig, ax = plt.subplots(figsize=PlotConfig.DEFAULT_FIGSIZE)
    
    valid_best = [(i, f) for i, f in enumerate(best_history) if f != float('inf')]
    valid_avg = [(i, f) for i, f in enumerate(avg_history) if f != float('inf')]
    
    if valid_best:
        iterations, values = zip(*valid_best)
        ax.plot(iterations, values, label=f'Best Fitness ({algo_name})',
               linestyle='-', drawstyle='steps-post')
    
    if valid_avg:
        iterations, values = zip(*valid_avg)
        ax.plot(iterations, values, label=f'Average Fitness ({algo_name})',
               linestyle='--')
    
    ax.set_xlabel('Iteration / Generation')
    ax.set_ylabel('Objective Value')
    ax.set_title(f'{algo_name} Evolution (Simulation #{sim_num})')
    ax.set_xlim(0, max_iters)
    
    finite_vals = [f for f in (best_history + avg_history) if f != float('inf')]
    if finite_vals:
        min_val, max_val = min(finite_vals), max(finite_vals)
        padding = (max_val - min_val) * 0.1 if max_val > min_val else 1
        ax.set_ylim(max(0, min_val - padding), max_val + padding)
    
    step = max(1, max_iters // 10)
    ax.set_xticks(np.arange(0, max_iters + 1, step))
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    
    filename = f"{algo_name.lower()}_convergence_sim_{sim_num}"
    png_path, svg_path = PlotConfig.get_output_paths(output_dir, "convergence", filename)
    PlotConfig.save_and_close(fig, png_path, svg_path)
    
    return png_path, svg_path


# =============================================================================
# SUMMARY PLOT GENERATION (ORCHESTRATION)
# =============================================================================

def generate_summary_plots(all_results: Dict, output_dir: str, mode: str = 'elective'):
    """
    Generates all summary plots for a simulation mode.
    
    Args:
        all_results: Dictionary with algorithm results
        output_dir: Output directory
        mode: 'elective' or 'emergency'
    """
    logger.info(f"  -> Generating {mode} summary plots...")
    
    # Boxplot
    png_path, _ = plot_boxplot(all_results, output_dir, mode)
    if png_path:
        logger.info(f"    - {mode.capitalize()} boxplot saved to: {png_path}")
    
    # Execution time barplot
    png_path = plot_execution_time_barplot(all_results, output_dir, mode)
    if png_path:
        logger.info(f"    - Execution time barplot saved to: {png_path}")
    
    # Histograms per algorithm
    for algo_name in ALGORITHM_ORDER:
        if algo_name not in all_results:
            continue
        
        makespans = [mk for mk in all_results[algo_name]['makespan'] if mk != float('inf')]
        
        if len(makespans) >= 2:
            png_path, _ = plot_makespan_histogram(makespans, algo_name, output_dir, mode)
            if png_path and VERBOSE_MODE:
                logger.info(f"    - {mode.capitalize()} histogram for {algo_name} saved to: {png_path}")


def generate_emergency_summary_plots(all_results: Dict, output_dir: str):
    """Wrapper for emergency mode (backward compatibility)."""
    generate_summary_plots(all_results, output_dir, mode='emergency')


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

def plot_comparison_boxplot(all_results: Dict, output_dir: str) -> Tuple[str, str]:
    """Alias for backward compatibility."""
    return plot_boxplot(all_results, output_dir, mode='elective')


def plot_gantt_with_emergencies(schedule_details: List[Dict], rooms: List[str],
                                title: str, algorithm_name: str, output_dir: str,
                                emergencies: List[Dict], sim_num: Optional[int] = None,
                                verbose: bool = True) -> Tuple[str, str]:
    """Alias for backward compatibility."""
    return plot_gantt_chart(schedule_details, rooms, title, algorithm_name,
                           output_dir, sim_num, emergencies)