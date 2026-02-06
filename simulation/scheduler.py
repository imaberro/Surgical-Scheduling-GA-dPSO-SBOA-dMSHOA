# /simulation/scheduler.py

import heapq

# Import all necessary constants from the centralized configuration file.
from config.config import (
    ALL_ROOMS,
    ALL_PERSONNEL,
    PERSONNEL_BY_OPERATION,
    SETUP_TIMES,
    CLEANUP_TIMES,
    JOB_TYPES,
    MAX_WAIT_TIMES,
    ALPHA,
    BETA,
    GAMMA,
    DELTA,
    APRS,
    ORS,
    ARRS,
    VERBOSE_MODE
)

def _assign_best_available_personnel(operation_num, room_release_time, personnel_release_time, current_time):
    """
    Dynamically assigns the best available personnel for a given operation.
    
    Args:
        operation_num (int): Operation number (1=APR, 2=OR, 3=ARR)
        room_release_time (dict): Current release times for rooms
        personnel_release_time (dict): Current release times for personnel
        current_time (float): Current simulation time
        
    Returns:
        str: ID of the assigned personnel (e.g., "A1", "S2", "D1")
    """
    # Get the list of personnel that can perform this operation
    available_personnel = PERSONNEL_BY_OPERATION[operation_num]
    
    # Find the personnel that will be available earliest
    # In case of tie, pick the first one (deterministic)
    best_personnel = min(available_personnel, 
                        key=lambda p: (personnel_release_time.get(p, 0), p))
    
    return best_personnel

def calculate_schedule_fitness(solution, surgeries_data, return_details=False):
    """
    Calculates the fitness (combined objective) for a given solution.

    This is the UNIFIED simulation function for all algorithms.
    Takes a 'solution' in a standard format and simulates the scheduling
    applying the "no buffer" blocking constraint with DYNAMIC personnel assignment.

    NEW: Includes penalties for:
    - Waiting times between consecutive operations (flow continuity)
    - Room usage imbalance (to encourage using all available rooms)

    Args:
        solution (dict): A dictionary representing the solution with:
                         'job_sequence_base': a list with the priority order of jobs.
                         'room_assignment': a nested dict assigning rooms to each operation of each job.
        surgeries_data (dict): Processing times for each operation of each surgery.
        return_details (bool): If True, returns full details for the Gantt chart.

    Returns:
        float: The value of the combined objective function.
        Or a tuple (combined_obj, makespan, schedule_details) if return_details is True.
    """
    # --- 1. Solution Data Extraction ---
    job_sequence_base = solution.get('job_sequence_base', [])
    room_assignment = solution.get('room_assignment', {})
    current_job_ids = list(room_assignment.keys())

    if not current_job_ids or not job_sequence_base:
        return (float('inf'), float('inf'), None) if return_details else float('inf')

    total_ops = 3 * len(current_job_ids)

    # --- 2. Simulation State Initialization ---
    room_release_time = {room: 0 for room in ALL_ROOMS}
    personnel_release_time = {pers: 0 for pers in ALL_PERSONNEL}
    job_op_processing_end = {job: {0: 0} for job in current_job_ids}
    job_op_machine_end = {job: {0: 0} for job in current_job_ids}
    job_op_start = {job: {0: 0} for job in current_job_ids}
    job_op_used_res = {job: {0: (None, None)} for job in current_job_ids}
    next_op_num = {job: 1 for job in current_job_ids}
    ops_done = 0
    job_priority = {job: i for i, job in enumerate(job_sequence_base)}
    schedule_details = []

    # NEW: Track waiting times between consecutive operations
    total_inter_operation_wait = 0
    max_inter_operation_wait = 0

    # --- 3. Main Simulation Loop (Discrete Event Logic) ---
    # Priority queue to manage the next operation to schedule.
    # Format: (estimated_start_time, job_priority, job_counter, job_id, operation_num)
    possible_ops = []
    
    # NEW: Tie-break counter (ensures comparisons are numeric)
    job_counter = {job: i for i, job in enumerate(current_job_ids)}

    # Initialize the queue with the first operation (op=1) of each job.
    for job in current_job_ids:
        op = 1
        # Validate that the solution has room assignment for this operation
        if job not in room_assignment or op not in room_assignment[job]:
            return (float('inf'), float('inf'), None) if return_details else float('inf')

        assigned_room = room_assignment[job][op]
        
        # Personnel will be assigned dynamically, so we estimate with the earliest available
        available_personnel = PERSONNEL_BY_OPERATION[op]
        earliest_personnel_time = min(personnel_release_time.get(p, 0) for p in available_personnel)
        
        # The estimated start time is when the necessary resources are free.
        start_time = max(room_release_time.get(assigned_room, 0), earliest_personnel_time)
        # Add job_counter as third element for tie-breaking
        heapq.heappush(possible_ops, (start_time, job_priority[job], job_counter[job], job, op))

    # Wrap debug prints with VERBOSE_MODE
    """if return_details and VERBOSE_MODE:  # Add VERBOSE_MODE
        jobs_at_zero = sum(1 for op_tuple in possible_ops if op_tuple[0] == 0)
        print(f"\n  🔍 DEBUG INICIAL:")
        print(f"     Total jobs: {len(current_job_ids)}")
        print(f"     Jobs that can start at t=0: {jobs_at_zero}")
        print(f"     APR rooms available: {len([r for r in ALL_ROOMS if r.startswith('APR')])}")
        print(f"     Anesthetists available: {len(PERSONNEL_BY_OPERATION[1])}")"""

    while ops_done < total_ops:
        if not possible_ops:
            return (float('inf'), float('inf'), None) if return_details else float('inf')

        # Select the most promising operation from the queue.
        _, _, _, best_job, best_op = heapq.heappop(possible_ops)  # Extra '_' for job_counter

        # --- CRITICAL RE-CALCULATION WITH DYNAMIC PERSONNEL ASSIGNMENT ---
        assigned_room = room_assignment[best_job][best_op]
        prev_op_end_time = job_op_processing_end[best_job][best_op - 1]
        
        # DYNAMIC: Assign the best available personnel for this operation type
        assigned_personnel = _assign_best_available_personnel(
            best_op, 
            room_release_time, 
            personnel_release_time,
            prev_op_end_time
        )
        
        # Get setup time for this job type
        job_type = JOB_TYPES[best_job]
        setup_time = SETUP_TIMES[job_type]
        
        # Setup can start before the previous operation ends
        earliest_setup_start = max(
            room_release_time[assigned_room],
            personnel_release_time[assigned_personnel]
        )
        
        # The actual operation must start after the previous operation ends
        setup_start_time = max(earliest_setup_start, prev_op_end_time - setup_time)
        actual_start_time = setup_start_time + setup_time

        # NEW: Calculate waiting time between operations of the same job
        if best_op > 1:
            wait_time = actual_start_time - prev_op_end_time
            if wait_time > 0:
                total_inter_operation_wait += wait_time
                max_inter_operation_wait = max(max_inter_operation_wait, wait_time)

        # Schedule the selected operation
        processing_time = surgeries_data[best_job][best_op]
        cleanup_time = CLEANUP_TIMES[job_type]

        proc_end = actual_start_time + processing_time
        finish = proc_end + cleanup_time

        # Validate maximum waiting time constraint
        waiting_time = (finish - actual_start_time) - processing_time
        if waiting_time > MAX_WAIT_TIMES[best_op]:
            return (float('inf'), float('inf'), None) if return_details else float('inf')

        # Save the scheduling results for this operation
        job_op_start[best_job][best_op] = actual_start_time
        job_op_processing_end[best_job][best_op] = proc_end
        job_op_machine_end[best_job][best_op] = finish
        job_op_used_res[best_job][best_op] = (assigned_room, assigned_personnel)

        if return_details:
            schedule_details.append({
                'Job': best_job, 'Operation': best_op, 'Resource': assigned_room,
                'Personnel': assigned_personnel, 'Start': setup_start_time,
                'ProcessingEnd': proc_end, 'Finish': finish
            })
        
        """if return_details and ops_done < 10 and VERBOSE_MODE:
            print(f"Op #{ops_done+1}: Job {best_job} Op{best_op}")
            print(f"     → Room: {assigned_room} (available at t={room_release_time[assigned_room]:.2f})")
            print(f"     → Personnel: {assigned_personnel} (available at t={personnel_release_time[assigned_personnel]:.2f})")
            print(f"     → Starts at: t={actual_start_time:.2f}")"""

        # --- Update Resource Release Times ---
        room_release_time[assigned_room] = finish
        personnel_release_time[assigned_personnel] = finish

        # Blocking logic (no-buffer): resources from previous operation are released
        if best_op > 1:
            prev_room, prev_personnel = job_op_used_res[best_job][best_op - 1]
            if prev_room:
                room_release_time[prev_room] = max(room_release_time[prev_room], setup_start_time)
            if prev_personnel:
                personnel_release_time[prev_personnel] = max(personnel_release_time[prev_personnel], setup_start_time)

        # Add the next operation of this job to the queue
        ops_done += 1
        next_op_num[best_job] += 1
        next_op = next_op_num[best_job]

        if next_op <= 3:
            next_assigned_room = room_assignment[best_job][next_op]
            
            # Estimate with earliest available personnel for this operation type
            available_personnel = PERSONNEL_BY_OPERATION[next_op]
            earliest_personnel_time = min(personnel_release_time.get(p, 0) for p in available_personnel)
            
            next_start_time = max(
                job_op_processing_end[best_job][best_op],
                room_release_time.get(next_assigned_room, 0),
                earliest_personnel_time
            )
            
            # NEW: Priority boost for continuing the same job (flow continuity)
            # Use negative job priority to prioritize continuation
            continuity_priority = -1 if next_op > 1 else job_priority[best_job]
            
            heapq.heappush(possible_ops, (next_start_time, continuity_priority, job_counter[best_job], best_job, next_op))

    # --- 4. Final Metrics Calculation ---
    try:
        final_makespan = max(job_op_machine_end.get(j, {}).get(3, 0) for j in current_job_ids)
    except (ValueError, TypeError):
        final_makespan = float('inf')

    if final_makespan == 0 and total_ops > 0:
        return (float('inf'), float('inf'), None) if return_details else float('inf')

    # NEW: Validate makespan consistency with schedule_details
    if return_details and schedule_details and VERBOSE_MODE:  # Add VERBOSE_MODE
        max_finish_from_details = max(t.get('Finish', 0) for t in schedule_details)
        if abs(final_makespan - max_finish_from_details) > 0.01:
            print(f"  -> [ERROR] MAKESPAN MISMATCH! Calculated: {final_makespan:.2f}, CSV: {max_finish_from_details:.2f}")
            print(f"     Last operation in job_op_machine_end: {job_op_machine_end}")
            print(f"     Last task in schedule_details: {schedule_details[-1]}")

    total_start_time = sum(
        job_op_start.get(j, {}).get(op, 0)
        for j in current_job_ids
        for op in [1, 2, 3]
        if job_op_start.get(j, {}).get(op, -1) >= 0
    )
    
    # NEW: Calculate room utilization imbalance penalty
    room_usage_count = {room: 0 for room in ALL_ROOMS}
    for job in current_job_ids:
        for op in [1, 2, 3]:
            assigned_room, _ = job_op_used_res[job][op]
            if assigned_room:
                room_usage_count[assigned_room] += 1
    
    # Calculate coefficient of variation for each room type
    def calculate_imbalance(room_list):
        """Calculate imbalance with progressive penalty for unused rooms"""
        if not room_list:
            return 0
        usages = [room_usage_count[room] for room in room_list]
        total_jobs = sum(usages)
        
        if total_jobs == 0:
            return 0
        
        # Count completely unused rooms
        unused_count = sum(1 for u in usages if u == 0)
        
        # Calculate coefficient of variation (CV)
        mean_usage = total_jobs / len(usages)
        if mean_usage == 0:
            return 0
        variance = sum((u - mean_usage) ** 2 for u in usages) / len(usages)
        std_dev = variance ** 0.5
        cv = std_dev / mean_usage
        
        # Progressive penalty for unused rooms:
        # 0 unused → +0
        # 1 unused → +2
        # 2 unused → +5
        # 3 unused → +10
        unused_penalty = unused_count * (unused_count + 1)
        
        return cv + unused_penalty
    
    apr_imbalance = calculate_imbalance(APRS)
    or_imbalance = calculate_imbalance(ORS)
    arr_imbalance = calculate_imbalance(ARRS)
    total_imbalance = apr_imbalance + or_imbalance + arr_imbalance
    
    # NEW: Multi-objective function with flow continuity penalties + balance penalty
    combined_obj = (
        final_makespan + 
        ALPHA * total_start_time + 
        BETA * total_inter_operation_wait +
        GAMMA * max_inter_operation_wait +
        DELTA * total_imbalance
    )

    if return_details:
        return combined_obj, final_makespan, schedule_details
    else:
        return combined_obj