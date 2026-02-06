"""
File and directory management for the simulation system.
"""
import os
from utils.logger import logger

class FileManager:
    """
    Manages the creation and organization of output directories.
    """
    
    BASE_DIR = "results"
    
    PLOT_SUBDIRS = ['boxplot', 'barplot', 'histograms', 'convergence', 'gantt']
    
    def setup_elective_directories(self):
        """Creates directory structure for elective simulations."""
        return self._setup_directories("elective")
    
    def setup_emergency_directories(self):
        """Creates directory structure for emergency simulations."""
        return self._setup_directories("emergencies")
    
    def _setup_directories(self, mode):
        """
        Internal method to create directory structure.
        
        Args:
            mode (str): 'elective' or 'emergencies'
        
        Returns:
            dict: Paths to csv and plots directories
        """
        experiment_dir = os.path.join(self.BASE_DIR, mode)
        
        output_dirs = {
            "csv": os.path.join(experiment_dir, "csv"),
            "plots": os.path.join(experiment_dir, "plots")
        }
        
        # Create main directories
        for path in output_dirs.values():
            os.makedirs(path, exist_ok=True)
        
        # Create plot subdirectories
        for subdir in self.PLOT_SUBDIRS:
            os.makedirs(os.path.join(output_dirs['plots'], subdir), exist_ok=True)
        
        logger.info(f"Output directory: {experiment_dir}")
        logger.info(f"   - CSV files: {output_dirs['csv']}")
        logger.info(f"   - Plots: {output_dirs['plots']}")
        
        return output_dirs