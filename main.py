"""
Main entry point for the Hospital Scheduling Simulation System.
This file acts as a simple orchestrator, delegating all responsibilities.
"""
import os
import sys

# Ensure project root is in sys.path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.simulation_runner import SimulationRunner
from config.config import EMERGENCY_ENABLED
from utils.logger import logger

def main():
    """
    Main orchestrator: decides which simulation mode to run.
    """
    try:
        runner = SimulationRunner()
        
        if EMERGENCY_ENABLED:
            runner.run_emergency_mode()
        else:
            runner.run_elective_mode()
    
    except Exception as e:
        logger.error(f"\n\nFatal error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()