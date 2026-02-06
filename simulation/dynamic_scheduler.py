"""
Module implementing the TSJS (Three-Station Job Shop Scheduling) strategy
for dynamic emergency rescheduling.
"""
import copy
import numpy as np
from simulation.scheduler import calculate_schedule_fitness
from utils.logger import logger
from config.config import VERBOSE_MODE

class DynamicScheduler:
    """
    Implements the TSJS strategy for dynamic emergency rescheduling.

    When an emergency arrives:
    1. Identify jobs that have already started (J_i^S)
    2. Identify reschedulable jobs (J_i^RS)
    3. Re-optimize the schedule with the emergency integrated
    4. Apply constraints (11), (12), (13) from the reference paper
    """
    
    def __init__(self, algorithm_runner, surgeries_data, job_ids):
        """
        Args:
            algorithm_runner: algorithm run() function (GA, dPSO, etc.)
            surgeries_data: elective surgeries data
            job_ids: list of original job IDs
        """
        self.algorithm_runner = algorithm_runner
        self.base_surgeries_data = copy.deepcopy(surgeries_data)
        self.base_job_ids = list(job_ids)
        self.current_schedule_details = []
        self.current_solution = None
        self.current_emergencies = []
    
    def run_with_emergencies(self, emergency_arrivals, seed):
        """
        Runs the schedule with dynamic emergencies using TSJS.

        Args:
            emergency_arrivals: list of generated emergencies
            seed: seed for reproducibility
        
        Returns:
            tuple: (final_schedule_details, events_log, final_makespan, best_hist, avg_hist)
        """
        from config.config import JOB_TYPES
        
        events_log = []
        
        # IMPORTANT: Register emergency job types in JOB_TYPES temporarily
        original_job_types = JOB_TYPES.copy()
        for emergency in emergency_arrivals:
            JOB_TYPES[emergency['job_id']] = emergency['job_type']
        
        try:
            # ============================================================
            # PHASE 1: Generate initial schedule (elective surgeries only)
            # ============================================================
            if VERBOSE_MODE:
                logger.info(f"\n{'='*70}")
                logger.info("PHASE 1: Generating Initial Schedule (Elective Surgeries Only)")
                logger.info(f"{'='*70}")
            
            # Keep initial convergence histories
            _, initial_solution, initial_best_hist, initial_avg_hist = self.algorithm_runner(
                self.base_surgeries_data,
                self.base_job_ids,
                seed
            )
            
            if not initial_solution:
                logger.error("Failed to generate initial schedule!")
                return None, events_log, float('inf'), [], []
            
            # Get initial schedule details
            _, initial_makespan, initial_details = calculate_schedule_fitness(
                initial_solution, 
                self.base_surgeries_data, 
                return_details=True
            )
            
            self.current_schedule_details = initial_details
            self.current_solution = initial_solution
            
            events_log.append({
                'time': 0,
                'type': 'INITIAL_SCHEDULE',
                'num_jobs': len(self.base_job_ids),
                'makespan': initial_makespan,
                'details': f"Scheduled {len(self.base_job_ids)} elective surgeries"
            })
            
            if VERBOSE_MODE:
                logger.info(f"  ✓ Initial schedule generated: {len(self.base_job_ids)} jobs, Makespan={initial_makespan:.2f}")
            
            # ============================================================
            # PHASE 2: Process each emergency sequentially
            # ============================================================
            for idx, emergency in enumerate(emergency_arrivals, 1):
                arrival_time = emergency['arrival_time']
                job_id = emergency['job_id']
                job_type = emergency['job_type']
                max_delay = emergency['max_delay']
                
                if VERBOSE_MODE:
                    logger.info(f"\n{'='*70}")
                    logger.info(f"EMERGENCY {idx}/{len(emergency_arrivals)}: {job_id} arrives at t={arrival_time:.2f}")
                    logger.info(f"{'='*70}")
                
                # --- 2.1: Identify started jobs (J_i^S) ---
                started_jobs = set()
                for task in self.current_schedule_details:
                    if task['Start'] < arrival_time:
                        started_jobs.add(task['Job'])
                
                # --- 2.2: Identify jobs to reschedule (J_i^RS) ---
                all_current_jobs = set(j for j in self.base_job_ids if isinstance(j, int))
                
                for task in self.current_schedule_details:
                    job = task['Job']
                    if isinstance(job, str) and job.startswith('E') and job not in started_jobs:
                        all_current_jobs.add(job)
                
                jobs_to_reschedule = list(all_current_jobs - started_jobs)
                
                if VERBOSE_MODE:
                    started_jobs_sorted = sorted([j for j in started_jobs if isinstance(j, int)]) + \
                                         sorted([j for j in started_jobs if isinstance(j, str)])
                    jobs_to_reschedule_sorted = sorted([j for j in jobs_to_reschedule if isinstance(j, int)]) + \
                                               sorted([j for j in jobs_to_reschedule if isinstance(j, str)])
                    
                    logger.info(f"Jobs already started (J_i^S): {len(started_jobs)} → {started_jobs_sorted}")
                    logger.info(f"Jobs to reschedule (J_i^RS): {len(jobs_to_reschedule)} → {jobs_to_reschedule_sorted}")
                
                # --- 2.3: Create emergency data ---
                emergency_surgery_data = self._generate_emergency_data(
                    job_id, job_type, arrival_time
                )
                
                # --- 2.5: Prepare combined instance data ---
                combined_job_ids = [job_id] + jobs_to_reschedule
                combined_surgeries_data = {}
                
                combined_surgeries_data.update(emergency_surgery_data)
                
                for job in jobs_to_reschedule:
                    if job in self.base_surgeries_data:
                        combined_surgeries_data[job] = self.base_surgeries_data[job]
                    else:
                        found = False
                        
                        for task in self.current_schedule_details:
                            if task['Job'] == job and task['Operation'] == 1:
                                job_tasks = [t for t in self.current_schedule_details if t['Job'] == job]
                                combined_surgeries_data[job] = {}
                                for t in job_tasks:
                                    op = t['Operation']
                                    from config.config import JOB_TYPES, SETUP_TIMES
                                    job_type_prev = JOB_TYPES.get(job, 1)
                                    setup = SETUP_TIMES.get(job_type_prev, 0)
                                    duration = t['ProcessingEnd'] - t['Start'] - setup
                                    combined_surgeries_data[job][op] = duration
                                found = True
                                break
                        
                        if not found:
                            logger.error(f"ERROR: Cannot find surgery data for job {job}")
                            raise KeyError(f"Surgery data not found for job {job}")
                
                if VERBOSE_MODE:
                    logger.info(f"  🔄 Re-optimizing {len(combined_job_ids)} jobs ({1} emergency + {len(jobs_to_reschedule)} rescheduled)")
                
                # --- 2.6: Compute resource availability (t_r^Rs) ---
                resource_availability = self._calculate_resource_availability(
                    arrival_time, started_jobs
                )
                
                # --- 2.7: Re-optimization with emergency constraints ---
                new_seed = seed + int(arrival_time) + idx * 1000
                
                _, new_solution, _, _ = self.algorithm_runner(
                    combined_surgeries_data,
                    combined_job_ids,
                    new_seed
                )
                
                if not new_solution:
                    logger.error(f"Failed to generate rescheduled solution for emergency {job_id}")
                    continue
                
                # --- 2.8: Apply emergency constraints ---
                adjusted_solution = self._apply_emergency_constraints(
                    new_solution,
                    emergency,
                    resource_availability,
                    arrival_time
                )
                
                # --- 2.9: Simulate the new schedule ---
                _, new_makespan, new_schedule_details = calculate_schedule_fitness(
                    adjusted_solution,
                    combined_surgeries_data,
                    return_details=True
                )
                
                # --- 2.10: Merge with already-started work ---
                self.current_schedule_details = self._merge_schedules(
                    self.current_schedule_details,
                    new_schedule_details,
                    started_jobs,
                    arrival_time,
                    combined_surgeries_data
                )
                
                if self.current_schedule_details:
                    final_makespan = max(task['Finish'] for task in self.current_schedule_details)
                else:
                    final_makespan = 0
                
                # Verify emergency constraints
                emergency_start = None
                emergency_finish = None
                for task in self.current_schedule_details:
                    if task['Job'] == job_id and task['Operation'] == 1:
                        emergency_start = task['Start']
                        emergency_finish = task['Finish']
                        break
                
                if emergency_start is None:
                    if VERBOSE_MODE:
                        logger.warning(f"Emergency {job_id} not found in merged schedule, checking new_schedule_details")
                    for task in new_schedule_details:
                        if task['Job'] == job_id and task['Operation'] == 1:
                            emergency_start = task['Start']
                            emergency_finish = task['Finish']
                            break
                
                if emergency_start is not None:
                    delay = emergency_start - arrival_time
                    constraint_11_ok = delay >= 0
                    constraint_12_ok = delay <= max_delay
                    
                    if VERBOSE_MODE:
                        logger.info(f"  ✓ Rescheduling complete:")
                        logger.info(f"    - Emergency {job_id} starts at t={emergency_start:.2f} (delay: {delay:.2f}min)")
                        logger.info(f"    - Emergency {job_id} finishes at t={emergency_finish:.2f}")
                        logger.info(f"    - New makespan: {final_makespan:.2f}")
                        logger.info(f"    - Constraint (11) x≥t_J: {'✓' if constraint_11_ok else '✗'}")
                        logger.info(f"    - Constraint (12) x≤t_du: {'✓' if constraint_12_ok else '✗'}")
                else:
                    delay = None
                    logger.error(f"ERROR: Could not find emergency {job_id} in schedule!")
                    if VERBOSE_MODE:
                        logger.error(f"new_schedule_details jobs: {set(t['Job'] for t in new_schedule_details)}")
                        logger.error(f"current_schedule_details jobs: {set(t['Job'] for t in self.current_schedule_details)}")
                
                events_log.append({
                    'time': arrival_time,
                    'type': 'EMERGENCY_RESCHEDULING',
                    'emergency_job': job_id,
                    'emergency_type': job_type,
                    'jobs_rescheduled': len(jobs_to_reschedule),
                    'emergency_start': emergency_start,
                    'emergency_finish': emergency_finish,
                    'delay': delay,
                    'makespan': final_makespan,
                    'details': f"Emergency {job_id} integrated" + (f" with {delay:.2f}min delay" if delay is not None else " (timing error)")
                })
            
            final_makespan = max(task['Finish'] for task in self.current_schedule_details)
            
            # Return with convergence histories
            return self.current_schedule_details, events_log, final_makespan, initial_best_hist, initial_avg_hist
        
        finally:
            JOB_TYPES.clear()
            JOB_TYPES.update(original_job_types)
    
    def _calculate_resource_availability(self, current_time, started_jobs):
        """
        Computes when each resource (room) becomes available.
        Implements t_r^Rs from the reference paper.
        """
        from config.config import ALL_ROOMS
        
        resource_availability = {room: current_time for room in ALL_ROOMS}
        
        for task in self.current_schedule_details:
            if task['Job'] in started_jobs:
                room = task['Resource']
                finish_time = task['Finish']
                resource_availability[room] = max(
                    resource_availability[room],
                    finish_time
                )
        
        if VERBOSE_MODE:
            logger.info(f"Resource availability (t_r^Rs):")
            for room, avail_time in sorted(resource_availability.items()):
                if avail_time > current_time:
                    logger.info(f"     {room}: {avail_time:.2f} min")
        
        return resource_availability
    
    def _apply_emergency_constraints(self, solution, emergency, 
                                    resource_availability, arrival_time):
        """
        Applies the paper constraints:
        - (11): x_{J_i}^r ≥ t_{J_i} (start ≥ arrival)
        - (12): x_{J_i}^r ≤ t_{J_i}^{du} (start ≤ arrival + max_delay)
        - (13): x_{J_i}^r ≥ t_r^{Rs} for J_i^{RS} (respect resource availability)

        Strategy: force the emergency to the front of the sequence.
        """
        adjusted_solution = copy.deepcopy(solution)
        job_id = emergency['job_id']
        
        if job_id in adjusted_solution['job_sequence_base']:
            seq = adjusted_solution['job_sequence_base']
            seq.remove(job_id)
            seq.insert(0, job_id)
        
        return adjusted_solution
    
    def _merge_schedules(self, old_schedule, new_schedule, 
                        started_jobs, merge_time, surgeries_data):
        """
        Merges the previous schedule with the new schedule while respecting precedence.
        """
        import logging
        from config.config import ALL_ROOMS, JOB_TYPES, SETUP_TIMES, CLEANUP_TIMES
        
        logger = logging.getLogger(__name__)
        merged = []
        
        # 1. Keep tasks from already-started jobs
        for task in old_schedule:
            if task['Job'] in started_jobs:
                merged.append(task.copy())
        
        # 2. Compute resource availability
        resource_last_finish = {room: merge_time for room in ALL_ROOMS}
        
        for task in merged:
            room = task['Resource']
            finish_time = task['Finish']
            if finish_time > resource_last_finish[room]:
                resource_last_finish[room] = finish_time
        
        if VERBOSE_MODE:
            logger.info(f"Resource availability after merge_time={merge_time:.2f}:")
            for room, avail_time in sorted(resource_last_finish.items()):
                if avail_time > merge_time:
                    logger.info(f"     {room}: {avail_time:.2f} min")
        
        # 3. Group tasks by job
        jobs_to_add = {}
        for task in new_schedule:
            if task['Job'] not in started_jobs:
                job = task['Job']
                if job not in jobs_to_add:
                    jobs_to_add[job] = []
                jobs_to_add[job].append(task.copy())
        
        for job in jobs_to_add:
            jobs_to_add[job].sort(key=lambda t: t['Operation'])
        
        # 4. Shift times to satisfy precedence
        for job, tasks in jobs_to_add.items():
            previous_processing_end = merge_time
            
            for task in tasks:
                op = task['Operation']
                room = task['Resource']
                
                job_type = JOB_TYPES.get(job, 1)
                setup_time = SETUP_TIMES.get(job_type, 0)
                cleanup_time = CLEANUP_TIMES.get(job_type, 0)
                processing_time = surgeries_data[job][op]
                
                if op == 1:
                    earliest_start = max(resource_last_finish[room], merge_time)
                else:
                    earliest_start = max(previous_processing_end, resource_last_finish[room], merge_time)
                
                new_start = earliest_start
                new_processing_end = new_start + setup_time + processing_time
                new_finish = new_processing_end + cleanup_time
                
                if VERBOSE_MODE and new_start > task['Start'] + 0.01:
                    logger.info(f"Adjusting {job} Op{op} on {room}: {task['Start']:.2f} → {new_start:.2f} min")
                
                task['Start'] = new_start
                task['ProcessingEnd'] = new_processing_end
                task['Finish'] = new_finish
                
                resource_last_finish[room] = new_finish
                previous_processing_end = new_processing_end
                
                merged.append(task)
        
        if VERBOSE_MODE:
            logger.info(f"Merged schedule: {len(merged)} tasks total")
        return merged
    
    def _generate_emergency_data(self, job_id, job_type, arrival_time):
        """
        Generates processing-time data for an emergency based on typical durations.
        
        Returns:
            dict: {job_id: {1: dur, 2: dur, 3: dur}} - Formato igual a generate_day_surgeries_data
        """
        import numpy as np
        from data.data_generator import BASE_DAY_SURGERIES_DATA
        
        reference_job = None
        for job, data in BASE_DAY_SURGERIES_DATA.items():
            from config.config import JOB_TYPES
            if JOB_TYPES.get(job) == job_type:
                reference_job = job
                break
        
        if reference_job is None:
            reference_job = 1
        
        emergency_data = {}
        base_durations = BASE_DAY_SURGERIES_DATA[reference_job]
        
        for op in [1, 2, 3]:
            base_val = base_durations[op]
            std_val = 0.1 * base_val
            value = np.random.normal(base_val, std_val)
            emergency_data[op] = max(1, round(value, 2))
        
        return {job_id: emergency_data}