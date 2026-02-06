import os
import json
from pathlib import Path

def _load_config() -> dict:
    override_path = os.environ.get('HOSPITAL_CONFIG_PATH')
    if override_path:
        candidate = Path(override_path).expanduser()
        if not candidate.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            candidate = (project_root / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Config override file not found: {candidate}")
        config_path = candidate
    else:
        config_path = Path(__file__).with_name('config.json')
    with config_path.open(encoding='utf-8') as src:
        return json.load(src)

_CONFIG = _load_config()

ALG_CONFIG = _CONFIG['algorithms']
EXP_CONFIG = _CONFIG['experiment']

# --- 1. Experiment Parameters ---
NUM_SIMULATIONS = EXP_CONFIG['num_simulations']
STD_FACTOR = EXP_CONFIG['std_factor_times']
ALPHA_TEST = EXP_CONFIG['alpha_test']
OUTPUT_DIRS = EXP_CONFIG['output_dirs']

# --- Logging Configuration ---
LOGGING_CONFIG = _CONFIG.get('logging', {})
VERBOSE_MODE = LOGGING_CONFIG.get('verbose_mode', True)  # Default: verbose

# --- 2. Problem Parameters ---
TIMES_CONFIG = _CONFIG['times']
SETUP_TIMES = {int(k): v for k, v in TIMES_CONFIG['setup'].items()}
CLEANUP_TIMES = {int(k): v for k, v in TIMES_CONFIG['cleanup'].items()}
MAX_WAIT_TIMES = {int(k): v for k, v in TIMES_CONFIG['max_wait'].items()}

JOBS_CONFIG = _CONFIG['jobs']
JOB_TYPES = {int(k): v for k, v in JOBS_CONFIG['types'].items()}

# --- Resources ---
RESOURCES_CONFIG = _CONFIG['resources']
NUM_APRS = RESOURCES_CONFIG['num_aprs']
NUM_ORS = RESOURCES_CONFIG['num_ors']
NUM_ARRS = RESOURCES_CONFIG['num_arrs']

APRS = [f'APR_{i+1}' for i in range(NUM_APRS)]
ORS = [f'OR_{i+1}' for i in range(NUM_ORS)]
ARRS = [f'ARR_{i+1}' for i in range(NUM_ARRS)]
ALL_ROOMS = APRS + ORS + ARRS

# --- Personnel Configuration (Dynamic Assignment) ---
# Lists of available personnel by type for dynamic assignment
PERSONNEL_CONFIG = _CONFIG['personnel']

_ANESTHESIOLOGISTS = PERSONNEL_CONFIG['anesthesiologists']
_SURGEONS = PERSONNEL_CONFIG['surgeons']
_RECOVERY = PERSONNEL_CONFIG['recovery']

PERSONNEL_BY_OPERATION = {
    1: _ANESTHESIOLOGISTS,      # APR: Anesthesiologists
    2: _SURGEONS,               # OR: Surgeons
    3: _RECOVERY                # ARR: Recovery doctors
}

# Complete list of all personnel (for initialization)
ALL_PERSONNEL = (
    _ANESTHESIOLOGISTS +
    _SURGEONS +
    _RECOVERY
)

# Legacy PERSONNEL_MAP (deprecated, kept for backward compatibility)
PERSONNEL_MAP = {
    int(job_id): {int(stage): person for stage, person in stages.items()}
    for job_id, stages in PERSONNEL_CONFIG.get('personnel_map', {}).items()
    if isinstance(stages, dict)
}

# --- 3. Algorithm Parameters ---
ALPHA = ALG_CONFIG['alpha']
BETA = ALG_CONFIG.get('beta', 0.5)    # Penalty for total inter-operation waiting time
GAMMA = ALG_CONFIG.get('gamma', 1.0)  # Penalty for maximum inter-operation waiting time
DELTA = ALG_CONFIG.get('delta', 50.0) # Penalty for room usage imbalance

GA_CONFIG = ALG_CONFIG['ga']
GA_ENABLED = GA_CONFIG.get('enabled', True)
POPULATION_SIZE_GA = GA_CONFIG['population_size']
MAX_GENERATIONS = GA_CONFIG['max_generations']
CROSSOVER_PROBABILITY = GA_CONFIG['crossover_probability']
MUTATION_PROBABILITY = GA_CONFIG['mutation_probability']
ELITISM_COUNT = GA_CONFIG['elitism_count']

DPSO_CONFIG = ALG_CONFIG['dpso']
DPSO_ENABLED = DPSO_CONFIG.get('enabled', True)
SWARM_SIZE_DPSO = DPSO_CONFIG['swarm_size']
MAX_ITERATIONS_DPSO = DPSO_CONFIG['max_iterations']
W_DPSO = DPSO_CONFIG['w']
C1_DPSO = DPSO_CONFIG['c1']
C2_DPSO = DPSO_CONFIG['c2']
VEL_HIGH_DPSO = DPSO_CONFIG['vel_high']
VEL_LOW_DPSO = DPSO_CONFIG['vel_low']

SBOA_CONFIG = ALG_CONFIG['sboa']
SBOA_ENABLED = SBOA_CONFIG.get('enabled', True)
SBOA_POP_SIZE = SBOA_CONFIG['population_size']
SBOA_MAX_ITER = SBOA_CONFIG['max_iterations']
SBOA_LOWER_BOUND = SBOA_CONFIG['lower_bound']
SBOA_UPPER_BOUND = SBOA_CONFIG['upper_bound']

MSHOA_CONFIG = ALG_CONFIG['dmshoa']
MSHOA_ENABLED = MSHOA_CONFIG.get('enabled', True)
MSHOA_POP_SIZE = MSHOA_CONFIG['population_size']
MAX_ITERATIONS_MSHOA = MSHOA_CONFIG['max_iterations']
MSHOA_K = MSHOA_CONFIG['k']
MSHOA_LOWER_BOUND = MSHOA_CONFIG['lower_bound']
MSHOA_UPPER_BOUND = MSHOA_CONFIG['upper_bound']

# --- 4. Emergency Configuration ---
EMERG_CONFIG = _CONFIG.get('emergencies', {})
EMERGENCY_ENABLED = EMERG_CONFIG.get('enabled', False)
NUM_EMERGENCIES = EMERG_CONFIG.get('num_emergencies', 2)
EMERGENCY_PRIORITY_WEIGHT = EMERG_CONFIG.get('priority_beta', 10.0)
EMERGENCY_MAX_DELAY = EMERG_CONFIG.get('max_delay_allowed', 60)

# =============================================================================
# ALGORITHM CONFIGURATIONS FOR PARALLEL EXECUTION
# =============================================================================

_ALGORITHMS_CACHE = None

def get_algorithms():
    """
    Returns the list of enabled algorithms.
    Uses caching to avoid rebuilding the list multiple times.
    """
    global _ALGORITHMS_CACHE
    if _ALGORITHMS_CACHE is None:
        from config.algorithms_loader import load_algorithms
        _ALGORITHMS_CACHE = load_algorithms(
            ga_enabled=GA_ENABLED,
            dpso_enabled=DPSO_ENABLED,
            sboa_enabled=SBOA_ENABLED,
            mshoa_enabled=MSHOA_ENABLED,
            max_generations=MAX_GENERATIONS,
            max_iterations_dpso=MAX_ITERATIONS_DPSO,
            sboa_max_iter=SBOA_MAX_ITER,
            max_iterations_mshoa=MAX_ITERATIONS_MSHOA,
            all_rooms=ALL_ROOMS
        )
    return _ALGORITHMS_CACHE

# For backward compatibility, expose as ALGORITHMS
class _AlgorithmsProxy:
    """Proxy object that lazily initializes ALGORITHMS"""
    def __getitem__(self, key):
        return get_algorithms()[key]
    
    def __iter__(self):
        return iter(get_algorithms())
    
    def __len__(self):
        return len(get_algorithms())
    
    def __repr__(self):
        return repr(get_algorithms())

ALGORITHMS = _AlgorithmsProxy()