# /algorithms/ga.py
"""
Module for the implementation of the Genetic Algorithm (GA) for the
surgery scheduling problem.
"""
import random
import copy
import numpy as np

# Import constants and the fitness function from our centralized modules
from config.config import (
    POPULATION_SIZE_GA, MAX_GENERATIONS, CROSSOVER_PROBABILITY,
    MUTATION_PROBABILITY, ELITISM_COUNT, APRS, ORS, ARRS,
    VERBOSE_MODE
)
from simulation.scheduler import calculate_schedule_fitness

# --- GA-Specific Helper Functions ---

def create_individual(job_ids):
    """Creates a random individual for the GA with GUARANTEED balanced room distribution."""
    individual = {}
    # 1. Random base sequence of jobs
    base_sequence = random.sample(job_ids, len(job_ids))
    
    # 2. Layer 1 must have THREE IDENTICAL COPIES of the base sequence
    individual['job_sequence_base'] = base_sequence * 3
    
    # 3. GUARANTEED BALANCED: Ensure every room gets at least one job
    num_jobs = len(job_ids)
    
    # Create cyclic assignment lists
    aprs_cycle = (APRS * ((num_jobs // len(APRS)) + 1))[:num_jobs]
    ors_cycle = (ORS * ((num_jobs // len(ORS)) + 1))[:num_jobs]
    arrs_cycle = (ARRS * ((num_jobs // len(ARRS)) + 1))[:num_jobs]
    
    # Shuffle to add randomness while maintaining balance
    random.shuffle(aprs_cycle)
    random.shuffle(ors_cycle)
    random.shuffle(arrs_cycle)
    
    individual['room_assignment'] = {}
    for idx, job in enumerate(job_ids):
        individual['room_assignment'][job] = {
            1: aprs_cycle[idx],
            2: ors_cycle[idx],
            3: arrs_cycle[idx]
        }
    
    return individual

def selection(population, fitnesses, job_ids):
    """Inverted roulette wheel selection based on fitness (minimization)."""
    valid_indices = [i for i, f in enumerate(fitnesses) if f != float('inf')]

    # If there are no valid individuals, a new random population is generated.
    if not valid_indices:
        return [create_individual(job_ids) for _ in range(len(population))]

    valid_pop = [population[i] for i in valid_indices]
    valid_fit = [fitnesses[i] for i in valid_indices]

    # Invert fitness so that the best (lowest) have a higher probability
    max_fit = max(valid_fit) + 1
    inverted_fitness = [(max_fit - f) for f in valid_fit]
    total_inverted_fitness = sum(inverted_fitness)

    if total_inverted_fitness == 0:
        return random.choices(valid_pop, k=len(population))

    probabilities = [f / total_inverted_fitness for f in inverted_fitness]
    
    # Choose with replacement from the valid population according to the calculated probabilities.
    chosen_indices = np.random.choice(
        len(valid_pop),
        size=len(population),
        replace=True,
        p=probabilities
    )
    return [valid_pop[i] for i in chosen_indices]

def crossover(parent1, parent2):
    """Performs crossover on sequence (Order Crossover - OX1) and room assignment."""
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    # Sequence crossover (OX1)
    if random.random() < CROSSOVER_PROBABILITY:
        seq1 = parent1['job_sequence_base']
        seq2 = parent2['job_sequence_base']
        n = len(seq1)
        if n >= 2:
            p1, p2 = sorted(random.sample(range(n), 2))
            
            sub1 = seq1[p1:p2+1]
            remaining1 = [item for item in seq2 if item not in sub1]
            child1['job_sequence_base'] = remaining1[-(n-(p2+1)):] + sub1 + remaining1[:-(n-(p2+1))]

            sub2 = seq2[p1:p2+1]
            remaining2 = [item for item in seq1 if item not in sub2]
            child2['job_sequence_base'] = remaining2[-(n-(p2+1)):] + sub2 + remaining2[:-(n-(p2+1))]

    # Room assignment crossover (single point)
    if random.random() < CROSSOVER_PROBABILITY:
        jobs_list = list(parent1['room_assignment'].keys())
        if len(jobs_list) > 1:
            cut_point = random.randint(1, len(jobs_list) - 1)
            jobs_to_swap = jobs_list[cut_point:]
            for job in jobs_to_swap:
                if job in parent1['room_assignment'] and job in parent2['room_assignment']:
                    child1['room_assignment'][job], child2['room_assignment'][job] = \
                        parent2['room_assignment'][job], parent1['room_assignment'][job]
    return child1, child2

def mutate(individual):
    """Performs mutation on sequence (swap) and room assignment."""
    ind = copy.deepcopy(individual)

    # Sequence mutation (swapping two positions)
    if random.random() < MUTATION_PROBABILITY:
        seq = ind['job_sequence_base']
        if len(seq) >= 2:
            i1, i2 = random.sample(range(len(seq)), 2)
            seq[i1], seq[i2] = seq[i2], seq[i1]

    # Room assignment mutation (change to a random room)
    mutation_prob_per_room = MUTATION_PROBABILITY / 3
    for job in ind['room_assignment']:
        for op in [1, 2, 3]:
            if random.random() < mutation_prob_per_room:
                if op == 1:
                    ind['room_assignment'][job][op] = random.choice(APRS)
                elif op == 2:
                    ind['room_assignment'][job][op] = random.choice(ORS)
                else: # op == 3
                    ind['room_assignment'][job][op] = random.choice(ARRS)
    return ind


# --- Main Algorithm Execution Function ---

def run(surgeries_data, job_ids, seed):
    """Executes the full Genetic Algorithm cycle."""
    random.seed(seed)
    np.random.seed(seed)

    # Initialize population
    population = [create_individual(job_ids) for _ in range(POPULATION_SIZE_GA)]

    best_objective_overall = float('inf')
    best_solution_overall = None
    best_makespan_overall = float('inf')
    best_fitness_history = []
    avg_fitness_history = []

    print_interval = max(1, MAX_GENERATIONS // 4)

    for generation in range(MAX_GENERATIONS):
        fitnesses = [
            calculate_schedule_fitness(ind, surgeries_data) for ind in population
        ]

        valid_fitnesses = [f for f in fitnesses if f != float('inf')]
        best_fitness_gen = min(valid_fitnesses) if valid_fitnesses else float('inf')
        avg_fitness_gen = np.mean(valid_fitnesses) if valid_fitnesses else float('inf')
        
        best_so_far = best_fitness_history[-1] if best_fitness_history else float('inf')
        best_fitness_history.append(min(best_fitness_gen, best_so_far))
        avg_fitness_history.append(avg_fitness_gen)

        if best_fitness_gen < best_objective_overall:
            best_objective_overall = best_fitness_gen
            best_index = fitnesses.index(best_objective_overall)
            best_solution_overall = copy.deepcopy(population[best_index])
            
            _, best_makespan_overall, _ = calculate_schedule_fitness(
                best_solution_overall, surgeries_data, return_details=True
            )

        if VERBOSE_MODE:
            if generation == 0 or (generation + 1) % print_interval == 0 or generation == MAX_GENERATIONS - 1:
                print(f"  -> Gen {generation + 1}/{MAX_GENERATIONS}, Best Fitness: {best_objective_overall:.2f} || Makespan: {best_makespan_overall:.2f}")

        # --- Create New Generation ---
        sorted_pop_indices = np.argsort(fitnesses)
        elite = [
            copy.deepcopy(population[i]) for i in sorted_pop_indices[:ELITISM_COUNT]
            if fitnesses[i] != float('inf')
        ]

        selected_population = selection(population, fitnesses, job_ids)

        next_population = elite
        while len(next_population) < POPULATION_SIZE_GA:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1))
            if len(next_population) < POPULATION_SIZE_GA:
                next_population.append(mutate(child2))
        
        population = next_population
    
    return best_objective_overall, best_solution_overall, best_fitness_history, avg_fitness_history