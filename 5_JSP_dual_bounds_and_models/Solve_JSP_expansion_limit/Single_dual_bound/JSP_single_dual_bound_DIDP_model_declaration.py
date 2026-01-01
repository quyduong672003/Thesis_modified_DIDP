import sys
import os
import re
import math
import random
import numpy as np
import modified_didppy as m_dp

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
    # -----------------------------
    # 1. Retrieve Global Variables
    # -----------------------------
    num_operations = current_number_of_operations
    num_machines = current_number_of_machines
    op_processing_time = current_op_processing_time
    op_deadline = current_op_deadline
    op_predecessors = current_op_predecessors
    op_same_job = current_op_same_job
    valid_machines = current_valid_machines
    ops_on_machine = current_ops_on_machine

    # NEW: Added requested variables
    op_job_type = current_op_job_type
    op_required_machine_type = current_op_required_machine_type

    # -----------------------------
    # 2. DIDP Model and Constants
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
    # 3. State Variables
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
    # 4. Transitions
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
    # 5. Base case (OPTIMIZED)
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

    # ------------------------------------------------------------
    # Back up Base case
    # ------------------------------------------------------------
    """makespan = m_dp.FloatExpr(0)
    for o in range(num_operations):
        makespan = m_dp.max(makespan, op_completion_time[o])
    model.add_base_case([unscheduled_operations.is_empty()], cost=makespan)"""

    # ------------------------------------------------------------
    # 6. State Constraints
    # ------------------------------------------------------------
    # Enforce 0 <= completion time <= deadline for all operations
    for o in range(num_operations):
        # Upper bound: c(o) <= d(o)
        model.add_state_constr(op_completion_time[o] <= deadline_table[o])

        # ------------------------------------------------------------
    # 7. Dual Bound (optional)
    # ------------------------------------------------------------
    # Machine-based bound: for each machine m, available_time[m] + sum(remaining ptime on that machine)
    ops_on_machine_consts = [
        model.create_set_const(object_type=operation_obj, value=ops) for ops in ops_on_machine
    ]
    # remaining processing time on machine m = sum of ptime for operations on m that are still unscheduled
    remaining_time_on_machine = [
        processing_time_table[unscheduled_operations.intersection(ops_on_machine_consts[m])]
        for m in range(num_machines)
    ]
    machine_bound_exprs = [
        machine_available_time[m] + remaining_time_on_machine[m]
        for m in range(num_machines)
    ]

    # fold to a single dp.max expression
    dual_bound_expr = machine_bound_exprs[0]
    for b in machine_bound_exprs[1:]:
        dual_bound_expr = m_dp.max(dual_bound_expr, b)

    # Add dual bound to model
    model.add_dual_bound(dual_bound_expr)
    #"""

    # =========================================================
    # 8. Bundle Metadata
    # =========================================================
    metadata = {
        "unscheduled_operations": unscheduled_operations,
        "finished_operations": finished_operations,
        "machine_available_time": machine_available_time,
        "op_completion_time": op_completion_time,
        # Added the requested new variables to metadata
        "op_job_type": op_job_type,
        "op_required_machine_type": op_required_machine_type
    }

    didp_bundle = (model, metadata)
    return didp_bundle
