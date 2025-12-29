import sys
import os
import re
import math
import random
import numpy as np
import modified_didppy as m_dp
from ortools.linear_solver import pywraplp
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment
from numpy.linalg import eigh
from functools import lru_cache
from evolutionary_algorithm_lib.utils import automatic_creation_of_dual_bounds_registry

# ==========================================
# GLOBAL VARIABLES
# ==========================================
current_number_of_operations = 0
current_number_of_machines = 0
current_number_of_jobs = 0

# DIDP Sets
current_op_processing_time = []
current_op_deadline = []
current_op_predecessors = []
current_op_same_job = []
current_valid_machines = []
current_ops_on_machine = []
current_op_job_type = []
current_op_required_machine_type = []
current_map_mj_to_op = {}

# LP Sets
current_N = []
current_A = []
current_B = []
current_B_flat = []
current_p_mj = {}
current_J = []
current_M = []

def load_jsp_file(file_path):
    """
    Reads the content of a JSP instance file from disk.

    Args:
        file_path (str): The absolute or relative path to the .txt file.

    Returns:
        str: The raw string content of the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file was not found at: {file_path}")

    with open(file_path, 'r') as f:
        content = f.read()

    return content

def parse_jsp_instance(file_content):
    """
    Parses a JSP instance string (Taillard/Beasley format) and generates 
    ALL sets/parameters required for both LP Relaxation and DIDP models.

    Args:
        file_content (str): The raw text content of the instance file.

    Returns:
        dict: A dictionary containing all sets (J, M, N, A, B, etc.) and parameters.
    """
    # ---------------------------------------------------------
    # 1. Basic Parsing (Clean comments and header)
    # ---------------------------------------------------------
    lines = file_content.strip().split('\n')
    data_tokens = []

    for line in lines:
        # Remove comments (lines starting with # or [)
        if not line.strip() or line.strip().startswith('#') or line.strip().startswith('['):
            continue
        # Extract numbers
        tokens = re.findall(r'\d+', line)
        data_tokens.extend([int(t) for t in tokens])

    iterator = iter(data_tokens)

    try:
        num_jobs = next(iterator)
        num_machines = next(iterator)
    except StopIteration:
        raise ValueError("File is empty or invalid format")

    # ---------------------------------------------------------
    # 2. Raw Job Data Extraction
    # ---------------------------------------------------------
    # jobs_data format: [ [(machine, time), (machine, time)], ... ]
    jobs_data = []
    for _ in range(num_jobs):
        job_seq = []
        for _ in range(num_machines):
            m = next(iterator)
            p = next(iterator)
            job_seq.append((m, p))
        jobs_data.append(job_seq)

    num_operations = num_jobs * num_machines

    # ---------------------------------------------------------
    # 3. Build Structures for LP Relaxation (Lecture Note Style)
    # ---------------------------------------------------------
    # Sets based on (machine, job) tuples
    N = []              # Nodes: List of (m, j)
    A = []              # Solid Arcs: List of ((m_prev, j), (m_curr, j))
    B = []              # Broken Arcs (Graph): List of ((m, j), (m, k))
    B_flat = []         # Broken Arcs (LP Triplet): List of (m, j, k)
    p_mj = {}           # Processing time map: {(m, j): time}

    # Helpers for Broken Arcs
    machine_allocations = {m: [] for m in range(num_machines)}

    for j_idx, job_seq in enumerate(jobs_data):
        prev_node = None
        for (m, p) in job_seq:
            curr_node = (m, j_idx)

            # Populate N and p_mj
            N.append(curr_node)
            p_mj[curr_node] = float(p)

            # Populate Solid Arcs (A)
            if prev_node is not None:
                A.append((prev_node, curr_node))

            # Track for Broken Arcs
            machine_allocations[m].append(j_idx)
            prev_node = curr_node

    # Populate Broken Arcs - pairs on same machine
    for m in range(num_machines):
        assigned_jobs = machine_allocations[m]
        for i in range(len(assigned_jobs)):
            for k in range(i + 1, len(assigned_jobs)):
                job_1 = assigned_jobs[i]
                job_2 = assigned_jobs[k]

                # 1. Standard Disjunctive Graph Arc: ((m, j), (m, k))
                # Useful for graph plotting and disjunctive graph algos
                B.append(((m, job_1), (m, job_2)))

                # 2. Flat Triplet for LP variables: (m, j, k)
                # Useful for PuLP variables x_mjk
                B_flat.append((m, job_1, job_2))

    # ---------------------------------------------------------
    # 4. Build Structures for DIDP (Operation Index Style)
    # ---------------------------------------------------------
    # Flattened Operation ID: 0 to num_operations-1
    op_job_type = []
    op_required_machine_type = []
    op_processing_time = []
    op_predecessors = []
    op_deadline = []
    op_same_job = [[] for _ in range(num_operations)] # SJ_jo
    ops_on_machine = [[] for _ in range(num_machines)] # AO_k

    # Mapping to convert between (m,j) and op_id
    map_mj_to_op = {}

    # Define a default deadline (e.g., a large number covering the horizon)
    DEFAULT_DEADLINE = float('inf')

    op_counter = 0
    for j_idx, job_seq in enumerate(jobs_data):
        job_ops_indices = []

        for i, (m, p) in enumerate(job_seq):
            # Basic Attributes
            op_job_type.append(j_idx)
            op_required_machine_type.append(m)
            op_processing_time.append(float(p))
            op_deadline.append(DEFAULT_DEADLINE)

            # Map tracking
            map_mj_to_op[(m, j_idx)] = op_counter
            ops_on_machine[m].append(op_counter)
            job_ops_indices.append(op_counter)

            # Predecessors (P_jo)
            if i > 0:
                op_predecessors.append([op_counter - 1])
            else:
                op_predecessors.append([]) # First op has no pred

            op_counter += 1


        # Same Job Set (SJ_jo)
        for o_id in job_ops_indices:
            # All ops in this job EXCEPT self
            op_same_job[o_id] = [x for x in job_ops_indices if x != o_id]

    # Valid Machines (VM_jo) - Simple for static JSP
    valid_machines = [[op_required_machine_type[o]] for o in range(num_operations)]

    # ---------------------------------------------------------
    # 5. Return Master Dictionary
    # ---------------------------------------------------------
    return {
        "metadata": {
            "num_jobs": num_jobs,
            "num_machines": num_machines,
            "num_operations": num_operations
        },

        # --- Mathematical Programming Sets (LP Relax) ---
        "LP_sets": {
            "N": N,         # Nodes (m, j)
            "A": A,         # Solid Arcs
            "B": B,         # Broken Arcs (Nested Tuples)
            "B_flat": B_flat, # Broken Arcs (Flat Triplets: m, j, k)
            "p_mj": p_mj,   # Parameter p
            "J": list(range(num_jobs)),
            "M": list(range(num_machines))
        },

        # --- DIDP State Sets ---
        "DIDP_sets": {
            "op_job_type": op_job_type,                 # Job J per op
            "op_required_machine_type": op_required_machine_type, # Machine M per op
            "op_processing_time": op_processing_time,   # Time p per op
            "op_predecessors": op_predecessors,        # P_jo
            "op_deadline": op_deadline,               # Deadline D_jo
            "op_same_job": op_same_job,                 # SJ_jo
            "ops_on_machine": ops_on_machine,           # AO_k
            "valid_machines": valid_machines,           # VM_jo
            "map_mj_to_op": map_mj_to_op                # Helper: (m,j) -> op_id
        }
    }

def update_globals_for_jsp(file_path):
    """
    Updates global variables by reading/parsing the JSP file and 
    assigning values to the specific global variable names used by the DIDP model.
    """
    # 1. Declare ALL Globals used by DIDP Model & Dual Bounds
    global current_number_of_operations, current_number_of_machines, current_number_of_jobs
    global current_op_processing_time, current_op_deadline
    global current_op_predecessors, current_op_same_job
    global current_valid_machines, current_ops_on_machine
    global current_op_job_type, current_op_required_machine_type, current_map_mj_to_op

    # 2. Declare Globals used by LP Relaxation
    global current_N, current_A, current_B, current_B_flat, current_p_mj, current_J, current_M

    try:
        # A. Load and Parse
        file_content = load_jsp_file(file_path)
        data = parse_jsp_instance(file_content)

        # B. Assign Metadata
        current_number_of_operations = data['metadata']['num_operations']
        current_number_of_machines = data['metadata']['num_machines']
        current_number_of_jobs = data['metadata']['num_jobs']

        # C. Assign DIDP Sets
        didp = data['DIDP_sets']
        current_op_processing_time = didp['op_processing_time']
        current_op_deadline = didp['op_deadline']
        current_op_predecessors = didp['op_predecessors']
        current_op_same_job = didp['op_same_job']
        current_valid_machines = didp['valid_machines']
        current_ops_on_machine = didp['ops_on_machine']

        current_op_job_type = didp['op_job_type']
        current_op_required_machine_type = didp['op_required_machine_type']
        current_map_mj_to_op = didp['map_mj_to_op']

        # D. Assign LP Sets
        lp = data['LP_sets']
        current_N = lp['N']
        current_A = lp['A']
        current_B = lp['B']
        current_B_flat = lp['B_flat']
        current_p_mj = lp['p_mj']
        current_J = lp['J']
        current_M = lp['M']

        return True

    except Exception as e:
        print(f"   ❌ Error reading/parsing JSP data: {e}")
        # import traceback
        # traceback.print_exc()
        return False

def creation_of_didp_model_function():
    num_operations = current_number_of_operations
    num_machines = current_number_of_machines
    op_processing_time = current_op_processing_time
    op_deadline = current_op_deadline
    op_predecessors = current_op_predecessors
    op_same_job = current_op_same_job
    valid_machines = current_valid_machines
    ops_on_machine = current_ops_on_machine
    # -----------------------------
    # DIDP Model and Constants
    # -----------------------------
    model = m_dp.Model(maximize=False, float_cost=True)

    operation_obj = model.add_object_type(number=num_operations)
    machine_obj = model.add_object_type(number=num_machines)

    # Constant tables
    processing_time_table = model.add_float_table(op_processing_time)
    deadline_table = model.add_float_table(op_deadline)
    predecessor_table = model.add_set_table(op_predecessors, object_type=operation_obj)
    same_job_table = model.add_set_table(op_same_job, object_type=operation_obj)
    valid_machine_table = model.add_set_table(valid_machines, object_type=operation_obj)

    # -----------------------------
    # State Variables
    # -----------------------------
    unscheduled_operations = model.add_set_var(object_type=operation_obj,
                                    target=list(range(num_operations)))
    finished_operations = model.add_set_var(object_type=operation_obj, target=[])
    machine_available_time = [
        model.add_float_resource_var(target=0, less_is_better=True, name=f"at_{mi}")
        for mi in range(num_machines)
    ]
    op_completion_time = [
        model.add_float_resource_var(target=0, less_is_better=True, name=f"c_{jo}")
        for jo in range(num_operations)
    ]

    # ------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------
    # Schedule operation jo on the specified machine
    # ------------------------------------------------------------
    for o in range(num_operations):
        valid_machines_for_op = valid_machines[o]
        expr = m_dp.FloatExpr(0)
        #Finding the completion time of the predecessors to make sure 1 job cannot be operated on 2 machines at the same time
        if op_same_job[o]:
            job_exprs = [op_completion_time[k] for k in op_same_job[o]]
            expr = job_exprs[0]
            for e in job_exprs[1:]:
                expr = m_dp.max(expr, e)
        max_same_job_completion_time = expr
        for m in valid_machines_for_op:
            # Compute start & completion
            start_time = m_dp.max(machine_available_time[m], max_same_job_completion_time)
            completion_time = start_time + processing_time_table[o]
            cost_expr = m_dp.FloatExpr.state_cost()
            transition = m_dp.Transition(
                name=f"schedule_op{o}_on_m{m}",
                cost=cost_expr,
                preconditions=[
                    unscheduled_operations.contains(o),
                    predecessor_table[o].issubset(finished_operations),
                    completion_time <= deadline_table[o],
                ],
                effects=[
                    (unscheduled_operations, unscheduled_operations.remove(o)),
                    (finished_operations, finished_operations.add(o)),
                    (op_completion_time[o], completion_time),
                    (machine_available_time[m], completion_time),
                ],
            )
            model.add_transition(transition)

    # ------------------------------------------------------------
    # Base case
    # ------------------------------------------------------------
    # Helper function to build a balanced max tree
    def build_balanced_max(expr_list):
        if not expr_list:
            return m_dp.FloatExpr(0)
        if len(expr_list) == 1:
            return expr_list[0]
        # Split in half
        mid = len(expr_list) // 2
        left_expr = build_balanced_max(expr_list[:mid])
        right_expr = build_balanced_max(expr_list[mid:])
        return m_dp.max(left_expr, right_expr)
    # Collect all completion time expressions into a list
    all_completion_times = [op_completion_time[o] for o in range(num_operations)]
    # Build the balanced expression
    makespan = build_balanced_max(all_completion_times)
    model.add_base_case([unscheduled_operations.is_empty()], cost=makespan)

    #Backup Base case
    """makespan = m_dp.FloatExpr(0)
    for o in range(num_operations):
        makespan = m_dp.max(makespan, op_completion_time[o])
    model.add_base_case([unscheduled_operations.is_empty()], cost=makespan)"""

    # State Constraints
    # ============================================================
    # Enforce 0 <= completion time <= deadline for all operations
    # ============================================================
    for o in range(num_operations):
        # Upper bound: c(o) <= d(o)
        model.add_state_constr(op_completion_time[o] <= deadline_table[o])

    # =========================================================
    # 3. Bundle Metadata
    # =========================================================
    metadata = {
        "unscheduled_operations": unscheduled_operations,
        "finished_operations": finished_operations,
        "machine_available_time": machine_available_time,
        "op_completion_time": op_completion_time,
    }

    didp_bundle = (model, metadata)
    return didp_bundle

def create_persistent_lp_relaxation_standard_jsp(didp_bundle):
    """
    Creates a persistent LP model using the standard (Machine, Job) notation.
    - Matches the PDF formulation: Nodes (m,j), Solid Arcs A, Broken Arcs B.
    - Bridges DIDP state (op_id) to LP variables (m,j) dynamically.
    """
    model_ref, metadata = didp_bundle

    # 1. Extract Standard LP Sets (Matches PDF Step 2)
    N = current_N          # Nodes
    A = current_A          # Solid Arcs
    B = current_B          # Broken Arcs (Nested Tuples)
    p_mj = current_p_mj    # Processing Times
    M = current_M          # Set of Machines
    J = current_J          # Set of Jobs

    # Helpers for Transformation (using Globals)
    op_job = current_op_job_type               
    op_machine = current_op_required_machine_type 
    num_ops = current_number_of_operations

    # State keys
    unscheduled_var = metadata['unscheduled_operations']
    machine_at_vars = metadata['machine_available_time']
    op_c_vars = metadata['op_completion_time']

    # Big-M: Sufficiently large number
    # Sum of all processing times is a safe upper bound
    BIG_M = sum(p_mj.values()) * 1.5

    # ==========================================
    # 3. INITIALIZATION (Runs Once)
    # ==========================================
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return lambda state: 0.0

    infinity = solver.infinity()

    # --- Create Variables ---

    # y[(m,j)]: Start time of job j on machine m (Continuous, >= 0)
    y = {}
    for node in N:
        m, j = node
        y[node] = solver.NumVar(0, infinity, f'y_{m}_{j}')

    # Cmax: Makespan variable
    c_max = solver.NumVar(0, infinity, 'Cmax')

    # x[(m,j,k)]: Binary variables for disjunctive pairs
    # x_mjk = 1 if job j precedes job k on machine m
    x = {} 

    # --- Create Constraints ---

    # 1. Job Precedence Constraints (Solid Arcs A)
    # y_hj >= y_mj + p_mj  => y_hj - y_mj >= p_mj
    for arc in A:
        node_mj, node_hj = arc

        # y_hj - y_mj >= p_mj
        c_prec = solver.Constraint(p_mj[node_mj], infinity, f'prec_{node_mj}_{node_hj}')
        c_prec.SetCoefficient(y[node_hj], 1)
        c_prec.SetCoefficient(y[node_mj], -1)

    # 2. Machine Capacity Constraints (Broken Arcs B)
    # For every pair of operations on the same machine, ensure disjointness.
    # Logic: B contains ((m,j), (m,k)). We need to define x_mjk and x_mkj.

    processed_pairs = set()

    for pair in B:
        # pair is ((m,j), (m,k))
        node_j, node_k = pair 
        m = node_j[0] # Extract machine index m
        j = node_j[1] # Job j
        k = node_k[1] # Job k

        # Ensure we only process the set {j, k} once per machine
        sorted_pair = tuple(sorted((j, k)))
        unique_key = (m, sorted_pair)

        if unique_key not in processed_pairs:
            processed_pairs.add(unique_key)

            # --- Define Variables x_mjk and x_mkj ---
            # Relaxed binary variables: 0 <= x <= 1
            x_mjk_key = (m, j, k)
            x_mkj_key = (m, k, j)

            x[x_mjk_key] = solver.NumVar(0, 1, f'x_{m}_{j}_{k}')
            x[x_mkj_key] = solver.NumVar(0, 1, f'x_{m}_{k}_{j}')

            # --- Symmetry Constraint ---
            # x_mjk + x_mkj = 1
            c_sym = solver.Constraint(1, 1, f'sym_{m}_{j}_{k}')
            c_sym.SetCoefficient(x[x_mjk_key], 1)
            c_sym.SetCoefficient(x[x_mkj_key], 1)

            # --- Disjunctive Constraints (Big-M) ---
            # Constraint 1: If j -> k (x_mjk=1), then y_mk >= y_mj + p_mj
            # Formula: y_mk >= y_mj + p_mj - M * (1 - x_mjk)
            # Rearranged: y_mk - y_mj - M * x_mjk >= p_mj - M

            c_disj_1 = solver.Constraint(p_mj[node_j] - BIG_M, infinity, f'seq_{m}_{j}_{k}')
            c_disj_1.SetCoefficient(y[node_k], 1)        # y_mk
            c_disj_1.SetCoefficient(y[node_j], -1)       # - y_mj
            c_disj_1.SetCoefficient(x[x_mjk_key], -BIG_M) # - M * x_mjk

            # Constraint 2: If k -> j (x_mkj=1), then y_mj >= y_mk + p_mk
            # Formula: y_mj >= y_mk + p_mk - M * (1 - x_mkj)
            # Rearranged: y_mj - y_mk - M * x_mkj >= p_mk - M

            c_disj_2 = solver.Constraint(p_mj[node_k] - BIG_M, infinity, f'seq_{m}_{k}_{j}')
            c_disj_2.SetCoefficient(y[node_j], 1)        # y_mj
            c_disj_2.SetCoefficient(y[node_k], -1)       # - y_mk
            c_disj_2.SetCoefficient(x[x_mkj_key], -BIG_M) # - M * x_mkj

    # 3. Makespan Constraints
    # Cmax >= y_mj + p_mj for all (m,j) in N
    for node in N:
        c_span = solver.Constraint(p_mj[node], infinity, f'span_{node}')
        c_span.SetCoefficient(c_max, 1)
        c_span.SetCoefficient(y[node], -1)

    # --- Objective ---
    objective = solver.Objective()
    objective.SetCoefficient(c_max, 1)
    objective.SetMinimization()

    # ==========================================
    # 4. DYNAMIC HEURISTIC (Transformation Logic)
    # ==========================================
    @lru_cache(maxsize=100000)
    def h_lp_relaxation_standard(state):
        unscheduled = state[unscheduled_var]

        # Optimization: Quick exit
        if not unscheduled: return 0.0 

        # Loop through all operations to update bounds
        # Transformation: op_id (DIDP) -> (m, j) (Standard LP)
        for op_id in range(num_ops):
            # 1. Transform op_id to (m, j)
            m = op_machine[op_id]
            j = op_job[op_id]
            node_key = (m, j)

            # 2. Update Bounds based on State
            if not state[unscheduled_var].contains(op_id):
                # CASE 1: Operation is FINISHED
                # LB_{mj} = c_(m,j) - p_{mj}
                # Fix variable to historical start time
                actual_completion = state[op_c_vars[op_id]]
                actual_start = actual_completion - p_mj[node_key]

                y[node_key].SetBounds(actual_start, actual_start)

            else:
                # CASE 2: Operation is UNSCHEDULED
                # LB_{mj} = at_m
                # The start time must be at least the machine's current availability
                machine_ready = state[machine_at_vars[m]]
                y[node_key].SetBounds(machine_ready, infinity)

        # Solve
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            return float(objective.Value())
        return 0.0

    return h_lp_relaxation_standard

def dual_bound_expression_function(didp_bundle):
    """
    Returns a dictionary of heuristic functions (dual bounds) for the JSP model.
    Implements:
    1. Job-based Bound (LB_job)
    2. Machine-based Bound (LB_machine)
    3. One-Machine Preemptive Bound (LB_1mach_preemptive) - Admissible
    4. One-Machine Non-Preemptive Bound (LB_1mach_non_preemptive) - Heuristic
    """
    model, metadata = didp_bundle

    # 1. Extract State Variable References
    unscheduled_var = metadata["unscheduled_operations"]
    machine_at_vars = metadata["machine_available_time"]
    op_c_vars = metadata["op_completion_time"]

    # 2. Access Static Global Data
    num_jobs = current_number_of_jobs
    num_machines = current_number_of_machines
    op_job_type = current_op_job_type
    ops_on_machine = current_ops_on_machine
    op_processing_time = current_op_processing_time

    # Pre-compute Map: Job -> [Operations]
    job_ops_map = [[] for _ in range(num_jobs)]
    for op_id, job_id in enumerate(op_job_type):
        job_ops_map[job_id].append(op_id)

    # PRE-COMPUTATION: Tails (q_jo)
    # Tail = Sum of processing times of all subsequent operations in same job.
    op_tails = [0.0] * len(op_processing_time)
    for j in range(num_jobs):
        ops = job_ops_map[j]
        current_tail = 0.0
        for i in range(len(ops) - 1, -1, -1):
            op = ops[i]
            op_tails[op] = current_tail
            current_tail += op_processing_time[op]

    # =========================================================
    # Bound 1: Job-based Bound
    # Logic: Cache the SUM of remaining processing times (Structure).
    #        Add C_j (Dynamic) in wrapper.
    # =========================================================
    @lru_cache(maxsize=100000)
    def _cached_job_remaining_sums(unscheduled_tuple):
        # Returns a list where index j is the sum of P for remaining ops in job j
        unscheduled_set = set(unscheduled_tuple)
        sums = [0.0] * num_jobs
        for j in range(num_jobs):
            s = 0.0
            for op in job_ops_map[j]:
                if op in unscheduled_set:
                    s += op_processing_time[op]
            sums[j] = s
        return tuple(sums)

    @lru_cache(maxsize=50000)
    def h_job_based(state):
        # 1. Get Cached Structural Sums
        unscheduled_list = sorted(list(state[unscheduled_var]))
        unscheduled_tuple = tuple(unscheduled_list)
        # Note: Converting to set here is fast for lookups if needed, 
        # but we rely on the cached function to do the iteration.

        job_sums = _cached_job_remaining_sums(unscheduled_tuple)

        max_job_bound = 0.0
        unscheduled_set = set(unscheduled_list) # Needed to check finished status

        for j in range(num_jobs):
            # 2. Get Dynamic Completion Time of Job so far
            c_j = 0.0
            # To find C_j (completion of last finished op), we scan.
            # Optimization: We only check ops that are NOT in unscheduled set.
            for op in job_ops_map[j]:
                if op not in unscheduled_set:
                    val = state[op_c_vars[op]]
                    if val > c_j: c_j = val

            # 3. Combine
            job_bound = c_j + job_sums[j]
            if job_bound > max_job_bound:
                max_job_bound = job_bound
        return float(max_job_bound)

    # =========================================================
    # Bound 2: Machine-based Bound
    # Logic: Cache the SUM of remaining processing times (Structure).
    #        Add A_m (Dynamic) in wrapper.
    # =========================================================
    @lru_cache(maxsize=100000)
    def _cached_machine_remaining_sums(unscheduled_tuple):
        unscheduled_set = set(unscheduled_tuple)
        sums = [0.0] * num_machines
        for m in range(num_machines):
            s = 0.0
            for op in ops_on_machine[m]:
                if op in unscheduled_set:
                    s += op_processing_time[op]
            sums[m] = s
        return tuple(sums)

    @lru_cache(maxsize=50000)
    def h_machine_based(state):
        unscheduled_tuple = tuple(sorted(list(state[unscheduled_var])))
        mach_sums = _cached_machine_remaining_sums(unscheduled_tuple)

        max_bound = 0.0
        for m in range(num_machines):
            at_m = state[machine_at_vars[m]]
            bound = at_m + mach_sums[m]
            if bound > max_bound: max_bound = bound
        return float(max_bound)

    # =========================================================
    # Bound 3: One-Machine Preemptive (Baker's / Preemptive Schrage)
    # Considers r_j, p_j, q_j and allows interruption.
    # Always Admissible (Valid Lower Bound).
    # =========================================================
    @lru_cache(maxsize=50000)
    def h_1mach_preemptive(state):
        max_bound = 0.0
        import heapq

        # 1. Dynamic Heads
        op_heads = {}
        for j in range(num_jobs):
            current_job_avail = 0.0
            for op in job_ops_map[j]:
                if state[unscheduled_var].contains(op):
                    op_heads[op] = current_job_avail
                    current_job_avail += op_processing_time[op]
                else:
                    current_job_avail = state[op_c_vars[op]]

        # 2. Solve for each machine
        for m in range(num_machines):
            machine_ready = state[machine_at_vars[m]]
            tasks = [] 
            for op in ops_on_machine[m]:
                if state[unscheduled_var].contains(op):
                    r = max(op_heads[op], machine_ready)
                    p = op_processing_time[op]
                    q = op_tails[op]
                    tasks.append([r, p, q])

            if not tasks: continue

            tasks.sort(key=lambda x: x[0]) # Sort by Release

            time_now = 0.0
            ready_queue = [] # Max-heap on Tail q: (-q, task_idx)
            task_idx = 0
            n_tasks = len(tasks)
            current_machine_bound = 0.0
            active_task_idx = -1 

            while task_idx < n_tasks or ready_queue or active_task_idx != -1:
                # Jump time if idle
                if active_task_idx == -1 and not ready_queue and task_idx < n_tasks:
                    time_now = max(time_now, tasks[task_idx][0])

                # Release tasks
                while task_idx < n_tasks and tasks[task_idx][0] <= time_now:
                    r, p, q = tasks[task_idx]
                    heapq.heappush(ready_queue, (-q, task_idx))
                    task_idx += 1

                # Preemption Check
                if active_task_idx != -1 and ready_queue:
                    current_q = tasks[active_task_idx][2]
                    best_waiting_q = -ready_queue[0][0]
                    if best_waiting_q > current_q:
                        # Preempt: put back
                        heapq.heappush(ready_queue, (-current_q, active_task_idx))
                        active_task_idx = -1

                # Pick task
                if active_task_idx == -1 and ready_queue:
                    _, idx = heapq.heappop(ready_queue)
                    active_task_idx = idx

                # Run
                if active_task_idx != -1:
                    remaining_p = tasks[active_task_idx][1]
                    if task_idx < n_tasks:
                        dt = min(remaining_p, tasks[task_idx][0] - time_now)
                    else:
                        dt = remaining_p

                    time_now += dt
                    tasks[active_task_idx][1] -= dt

                    if tasks[active_task_idx][1] <= 1e-9:
                        finish_q = tasks[active_task_idx][2]
                        current_machine_bound = max(current_machine_bound, time_now + finish_q)
                        active_task_idx = -1

            if current_machine_bound > max_bound:
                max_bound = current_machine_bound

        return float(max_bound)

    # =========================================================
    # Bound 4: One-Machine Non-Preemptive (Schrage's / Jackson's)
    # Runs tasks to completion. Tighter but potentially Inadmissible.
    # Good feature for EA to learn from.
    # =========================================================
    @lru_cache(maxsize=50000)
    def h_1mach_non_preemptive(state):
        max_bound = 0.0
        import heapq

        # 1. Dynamic Heads (Same as above)
        op_heads = {}
        for j in range(num_jobs):
            current_job_avail = 0.0
            for op in job_ops_map[j]:
                if state[unscheduled_var].contains(op):
                    op_heads[op] = current_job_avail
                    current_job_avail += op_processing_time[op]
                else:
                    current_job_avail = state[op_c_vars[op]]

        for m in range(num_machines):
            machine_ready = state[machine_at_vars[m]]
            tasks = []
            for op in ops_on_machine[m]:
                if state[unscheduled_var].contains(op):
                    r = max(op_heads[op], machine_ready)
                    p = op_processing_time[op]
                    q = op_tails[op]
                    tasks.append((r, p, q)) # Tuple is fine here, no mutation needed

            if not tasks: continue

            tasks.sort(key=lambda x: x[0]) # Sort by Release

            time_now = 0.0
            ready_queue = [] # Max-heap on Tail q: (-q, p, r)
            task_idx = 0
            n_tasks = len(tasks)
            current_machine_bound = 0.0

            while task_idx < n_tasks or ready_queue:
                # Jump time if idle
                if not ready_queue and task_idx < n_tasks and time_now < tasks[task_idx][0]:
                    time_now = tasks[task_idx][0]

                # Release tasks
                while task_idx < n_tasks and tasks[task_idx][0] <= time_now:
                    r, p, q = tasks[task_idx]
                    heapq.heappush(ready_queue, (-q, p, r))
                    task_idx += 1

                if ready_queue:
                    # Pick best (Largest Tail)
                    neg_q, p, r = heapq.heappop(ready_queue)
                    q = -neg_q

                    # RUN TO COMPLETION (Non-preemptive)
                    time_now += p

                    current_machine_bound = max(current_machine_bound, time_now + q)

            if current_machine_bound > max_bound:
                max_bound = current_machine_bound

        return float(max_bound)

    # =========================================================
    # Bound 5: Shifting Bottleneck (Heuristic / ERD Rule)
    # Reference: [cite: 102-127]
    # =========================================================
    @lru_cache(maxsize=50000)
    def h_simplified_shifting_bottleneck(state):
        # 1. Dynamic Heads (implicitly models Source -> Op path)
        op_heads = {}
        for j in range(num_jobs):
            current_job_avail = 0.0
            for op in job_ops_map[j]:
                if state[unscheduled_var].contains(op):
                    op_heads[op] = current_job_avail
                    current_job_avail += op_processing_time[op]
                else:
                    current_job_avail = state[op_c_vars[op]]

        # 2. Calculate Critical Path (CP)
        cp_val = 0.0
        for op in range(len(op_processing_time)):
            if state[unscheduled_var].contains(op):
                # path_len = Head (from S) + Process + Tail (to T)
                path_len = op_heads[op] + op_processing_time[op] + op_tails[op]
                if path_len > cp_val:
                    cp_val = path_len

        if cp_val == 0.0: return 0.0

        # 3. Solve Subproblems via ERD Rule
        max_machine_lateness = 0.0

        for m in range(num_machines):
            machine_ready = state[machine_at_vars[m]]
            tasks = [] 
            for op in ops_on_machine[m]:
                if state[unscheduled_var].contains(op):
                    # Step 3: Update Release Date r' = max(r, at_k)
                    r_prime = max(op_heads[op], machine_ready)
                    tasks.append({
                        'r_prime': r_prime,
                        'p': op_processing_time[op],
                        'q': op_tails[op]
                    })

            if not tasks: continue

            # Step 3: Sort by ERD
            tasks.sort(key=lambda x: x['r_prime'])

            current_time = machine_ready
            machine_lateness = -float('inf')

            for task in tasks:
                # Earliest start
                start_time = max(current_time, task['r_prime'])
                completion_time = start_time + task['p']
                current_time = completion_time

                # Step 3: Lateness L = Completion + Tail - CP
                lateness = completion_time + task['q'] - cp_val
                if lateness > machine_lateness:
                    machine_lateness = lateness

            # Step 4: Max delay
            if machine_lateness > max_machine_lateness:
                max_machine_lateness = machine_lateness

        return float(cp_val + max(0.0, max_machine_lateness))

    # =========================================================
    # Bound 6: LP relaxation
    # Reference: [cite: 102-127]
    # =========================================================
    h_lp_relaxation = create_persistent_lp_relaxation_standard_jsp(didp_bundle)

    # Return valid registry with all 4 bounds
    return automatic_creation_of_dual_bounds_registry(locals())
