"""
Algorithm loader module to avoid circular imports and type checker issues.
This module is separate from config.py to clearly separate concerns.
"""
from typing import List, Dict, Any

from algorithms.ga import run as run_ga
from algorithms.dpso import run as run_dpso
from algorithms.sboa import run as run_sboa
from algorithms.dmshoa import run as run_dmshoa

def load_algorithms(
    ga_enabled: bool,
    dpso_enabled: bool,
    sboa_enabled: bool,
    mshoa_enabled: bool,
    max_generations: int,
    max_iterations_dpso: int,
    sboa_max_iter: int,
    max_iterations_mshoa: int,
    all_rooms: List[str]
) -> List[Dict[str, Any]]:
    """
    Loads enabled algorithms based on configuration flags.
    
    Args:
        ga_enabled: Whether GA is enabled
        dpso_enabled: Whether dPSO is enabled
        sboa_enabled: Whether SBOA is enabled
        mshoa_enabled: Whether dMShOA is enabled
        max_generations: GA iterations
        max_iterations_dpso: dPSO iterations
        sboa_max_iter: SBOA iterations
        max_iterations_mshoa: dMShOA iterations
        all_rooms: List of all available rooms
    
    Returns:
        List of algorithm specifications
    """
    algorithms = []
    
    if ga_enabled:
        algorithms.append({
            "name": "GA",
            "runner": run_ga,
            "iterations": max_generations,
            "all_rooms": all_rooms
        })
    
    if dpso_enabled:
        algorithms.append({
            "name": "dPSO",
            "runner": run_dpso,
            "iterations": max_iterations_dpso,
            "all_rooms": all_rooms
        })
    
    if sboa_enabled:
        algorithms.append({
            "name": "SBOA",
            "runner": run_sboa,
            "iterations": sboa_max_iter,
            "all_rooms": all_rooms
        })
    
    if mshoa_enabled:
        algorithms.append({
            "name": "dMShOA",
            "runner": run_dmshoa,
            "iterations": max_iterations_mshoa,
            "all_rooms": all_rooms
        })
    
    if not algorithms:
        raise ValueError("No algorithms are enabled in config.json! Enable at least one algorithm.")
    
    return algorithms