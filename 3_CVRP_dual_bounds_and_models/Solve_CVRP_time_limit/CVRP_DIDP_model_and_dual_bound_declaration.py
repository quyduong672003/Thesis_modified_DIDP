import sys
import os
import re
import numpy as np
import vrplib # <--- Added for your reader
import modified_didppy as m_dp
from ortools.linear_solver import pywraplp
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment
from numpy.linalg import eigh
from functools import lru_cache
from evolutionary_algorithm_lib.utils import automatic_creation_of_dual_bounds_registry

# ==========================================
# GLOBAL VARIABLES (Your Naming Convention)
# ==========================================
# --- Problem Setup ---
current_capacity = 1000              # Capacity set to 1000
current_num_locations = 2            # Two locations
current_num_vehicles = 0             # Still zero vehicles (adjust if needed)

# Customer demands: two entries, both 0
current_cust_demands = [0, 0]

# Travel cost: 2x2 matrix with all entries = 0
current_travel_cost = [
    [0, 0],
    [0, 0]
]

# ==========================================
# 1. YOUR READER FUNCTION
# ==========================================
def update_globals_for_cvrp(file_path):
    """Updates the specific global variables required by creation_of_didp_model_function."""
    global current_capacity, current_num_locations, current_num_vehicles, current_cust_demands, current_travel_cost, current_optimal_cost

    instance = vrplib.read_instance(file_path)

    # 1. Update Capacity & Dimensions
    current_capacity = instance['capacity']
    current_num_locations = instance['dimension']

    # 2. Update Num Vehicles (Regex or Fallback)
    match_trucks = re.search(r"No of trucks:\s*(\d+)", instance.get('comment', ''))
    if match_trucks:
        current_num_vehicles = int(match_trucks.group(1))
    else:
        # Fallback logic for X-series or if comment is missing
        match_filename = re.search(r"-k(\d+)", os.path.basename(file_path))
        if match_filename:
            current_num_vehicles = int(match_filename.group(1))
        else:
            current_num_vehicles = 1 # Fallback

    # 3. Update Demands & Costs
    current_cust_demands = instance['demand']
    current_travel_cost = instance['edge_weight']

    return True

def creation_of_didp_model_function():
    """
    Creates the CVRP DIDP model and returns it along with necessary metadata 
    for the heuristic functions.
    """
    # =========================================================
    # 1. Define Data
    # =========================================================
    n = current_num_locations
    m = current_num_vehicles
    q = current_capacity
    # Weights (demand)
    d = current_cust_demands

    # Distance matrix
    distance_list = current_travel_cost

    # =========================================================
    # 2. Define DIDP model
    # =========================================================
    model = m_dp.Model(float_cost= True)

    customer = model.add_object_type(number=n)
    unvisited_var = model.add_set_var(object_type=customer, target=list(range(1, n)), name='unvisited_customers')
    location_var = model.add_element_var(object_type=customer, target=0)
    load_var = model.add_float_resource_var(target=0, less_is_better=True)
    vehicles_var = model.add_int_resource_var(target=1, less_is_better=True)

    weight = model.add_float_table(d)
    distance_table = model.add_float_table(distance_list)

    model.add_base_case([unvisited_var.is_empty(), location_var == 0])

    for j in range(1, n):
        visit = m_dp.Transition(
            name=f"visit {j}",
            cost=distance_table[location_var, j] + m_dp.FloatExpr.state_cost(),
            effects=[
                (unvisited_var, unvisited_var.remove(j)),
                (location_var, j),
                (load_var, load_var + weight[j]),
            ],
            preconditions=[unvisited_var.contains(j), load_var + weight[j] <= q],
        )
        model.add_transition(visit)

    for j in range(1, n):
        visit_via_depot = m_dp.Transition(
            name=f"visit {j} with new vehicle",
            cost=distance_table[location_var, 0] + distance_table[0, j] + m_dp.FloatExpr.state_cost(),
            effects=[
                (unvisited_var, unvisited_var.remove(j)),
                (location_var, j),
                (load_var, weight[j]),
                (vehicles_var, vehicles_var + 1),
            ],
            preconditions=[unvisited_var.contains(j), vehicles_var < m],
        )
        model.add_transition(visit_via_depot)

    return_to_depot = m_dp.Transition(
        name="return",
        cost=distance_table[location_var, 0] + m_dp.FloatExpr.state_cost(),
        effects=[(location_var, 0)],
        preconditions=[unvisited_var.is_empty(), location_var != 0],
    )
    model.add_transition(return_to_depot)

    model.add_state_constr((m - vehicles_var + 1) * q - load_var >= weight[unvisited_var])

    # =========================================================
    # 3. Bundle Metadata
    # =========================================================
    metadata = {
        "unvisited_var": unvisited_var,
        "location_var": location_var,
        "distance_matrix": distance_list,
        "demand": d,
        "capacity": q,
        "num_vehicles": m,
        "num_nodes": n
    }

    didp_bundle = (model, metadata)
    return didp_bundle

def create_persistent_lp_relaxation_3_index_dual_bounds(metadata):
    """
    Creates a persistent 3-Index CVRP Relaxation (Vehicle-Node-Node).
    - Logic: VRP4 Model (Assignment + Flow + MTZ Capacity).
    - Pattern: Cached internal worker '_solve_3idx'.
    """
    # --- Extract Static Data ---
    n_nodes = metadata['num_nodes']
    n_vehicles = metadata['num_vehicles']
    capacity = metadata['capacity']
    demands = metadata['demand']
    dist_matrix = metadata['distance_matrix']

    unvisited_var = metadata['unvisited_var']
    location_var = metadata['location_var']

    # --- INITIALIZATION (Runs Once) ---
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return lambda state: 0.0

    infinity = solver.infinity()

    # --- Variables ---
    x = {} # Flow x_k_i_j
    y = {} # Assignment y_i_k
    u = {} # Load u_i_k

    # 1. Flow
    for k in range(n_vehicles):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j: x[(k, i, j)] = solver.NumVar(0, 1, f'x_{k}_{i}_{j}')

    # 2. Assignment (All nodes, All vehicles)
    for i in range(n_nodes):
        for k in range(n_vehicles):
            y[(i, k)] = solver.NumVar(0, 1, f'y_{i}_{k}')

    # 3. Potentials (Load)
    for i in range(1, n_nodes):
        for k in range(n_vehicles):
            u[(i, k)] = solver.NumVar(0, capacity, f'u_{i}_{k}')

    # --- Constraints ---

    # (1.29) Customer Assignment (Mutable)
    # This is what we toggle on/off based on the state.
    cons_assignment = {}
    for i in range(1, n_nodes):
        c = solver.Constraint(0, 0, f'assign_{i}')
        for k in range(n_vehicles): c.SetCoefficient(y[(i, k)], 1)
        cons_assignment[i] = c

    # (1.30) Depot Usage
    c_depot = solver.Constraint(0, n_vehicles, 'depot_usage')
    for k in range(n_vehicles): c_depot.SetCoefficient(y[(0, k)], 1)

    # (1.31) Flow Conservation
    for k in range(n_vehicles):
        for i in range(n_nodes):
            # Out
            c_out = solver.Constraint(0, 0, f'flow_out_{i}_{k}')
            c_out.SetCoefficient(y[(i, k)], -1)
            for j in range(n_nodes):
                if i != j: c_out.SetCoefficient(x[(k, i, j)], 1)

            # In
            c_in = solver.Constraint(0, 0, f'flow_in_{i}_{k}')
            c_in.SetCoefficient(y[(i, k)], -1)
            for j in range(n_nodes):
                if i != j: c_in.SetCoefficient(x[(k, j, i)], 1)

    # (1.37) & (1.38) MTZ Capacity
    for k in range(n_vehicles):
        for i in range(1, n_nodes):
            for j in range(1, n_nodes):
                if i != j:
                    c = solver.Constraint(-infinity, float(capacity - demands[j]), f'mtz_{k}_{i}_{j}')
                    c.SetCoefficient(u[(i, k)], 1)
                    c.SetCoefficient(u[(j, k)], -1)
                    c.SetCoefficient(x[(k, i, j)], capacity)

    # --- Objective ---
    objective = solver.Objective()
    for k in range(n_vehicles):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j: objective.SetCoefficient(x[(k, i, j)], dist_matrix[i][j])
    objective.SetMinimization()

    # ==========================================
    # 2. CACHED SOLVER WORKER
    # ==========================================
    @lru_cache(maxsize=10000)
    def _solve_3idx(active_tuple):
        """
        Internal worker: Sets bounds and solves.
        active_tuple: Tuple of customer indices that MUST be visited.
        """
        active_set = set(active_tuple)

        # Reset/Update Assignment Constraints
        for i in range(1, n_nodes):
            if i in active_set:
                # ACTIVE: Must be visited
                cons_assignment[i].SetBounds(1, 1)
                # Load variables active
                for k in range(n_vehicles):
                    u[(i, k)].SetBounds(demands[i], capacity)
            else:
                # INACTIVE: Must NOT be visited (flow = 0)
                cons_assignment[i].SetBounds(0, 0)
                # Load variables forced to 0
                for k in range(n_vehicles):
                    u[(i, k)].SetBounds(0, 0)

        # Solve
        solver.SetTimeLimit(100) # 100ms
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            return float(objective.Value())
        return 0.0

    # ==========================================
    # 3. HEURISTIC WRAPPER
    # ==========================================
    def h_lp_relaxation_3_idx(state):
        unvisited = state[unvisited_var]
        current_loc = state[location_var]

        # Optimization: Solved state
        if not unvisited and current_loc == 0: 
            return 0.0

        # Define Active Set: Unvisited + Current (if not depot)
        active_customers = set(unvisited)
        if current_loc != 0:
            active_customers.add(current_loc)

        # Create Tuple Key
        active_key = tuple(sorted(list(active_customers)))

        return _solve_3idx(active_key)

    return h_lp_relaxation_3_idx

def create_persistent_lp_relaxation_2_index_dual_bounds(metadata):
    """
    Creates a persistent 2-Index CVRP Relaxation (Flow-based).
    - Logic: Aggregated Flow (No Vehicle Dimension).
    - Pattern: Cached internal worker '_solve_2idx'.
    """
    # --- Extract Static Data ---
    n_nodes = metadata['num_nodes']
    n_vehicles = metadata['num_vehicles']
    capacity = metadata['capacity']
    demands = metadata['demand']
    dist_matrix = metadata['distance_matrix']
    unvisited_var = metadata['unvisited_var']

    # --- INITIALIZATION ---
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return lambda state: 0.0

    infinity = solver.infinity()

    # --- Variables ---
    x = {} # Flow x_i_j
    u = {} # Load u_i

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j: x[(i, j)] = solver.NumVar(0, 1, f'x_{i}_{j}')

    for i in range(1, n_nodes):
        u[i] = solver.NumVar(0, capacity, f'u_{i}')

    # --- Constraints ---
    cons_degree_out = {}
    cons_degree_in = {}

    # 1. Degree Constraints (Mutable)
    for i in range(1, n_nodes):
        # Outgoing
        c_out = solver.Constraint(0, 0, f'deg_out_{i}')
        for j in range(n_nodes):
            if i != j: c_out.SetCoefficient(x[(i, j)], 1)
        cons_degree_out[i] = c_out

        # Incoming
        c_in = solver.Constraint(0, 0, f'deg_in_{i}')
        for j in range(n_nodes):
            if i != j: c_in.SetCoefficient(x[(j, i)], 1)
        cons_degree_in[i] = c_in

    # 2. Depot Degree
    c_depot_out = solver.Constraint(0, n_vehicles, 'depot_out')
    c_depot_in = solver.Constraint(0, n_vehicles, 'depot_in')
    for j in range(1, n_nodes):
        c_depot_out.SetCoefficient(x[(0, j)], 1)
        c_depot_in.SetCoefficient(x[(j, 0)], 1)

    # 3. MTZ Capacity
    for i in range(1, n_nodes):
        for j in range(1, n_nodes):
            if i != j:
                c = solver.Constraint(-infinity, float(capacity - demands[j]), f'mtz_{i}_{j}')
                c.SetCoefficient(u[j], 1)
                c.SetCoefficient(u[i], -1)
                c.SetCoefficient(x[(i, j)], capacity)

    # --- Objective ---
    objective = solver.Objective()
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j: objective.SetCoefficient(x[(i, j)], dist_matrix[i][j])
    objective.SetMinimization()

    # ==========================================
    # 2. CACHED SOLVER WORKER
    # ==========================================
    @lru_cache(maxsize=10000)
    def _solve_2idx(active_tuple):
        active_set = set(active_tuple)

        # Reset/Update Constraints
        for i in range(1, n_nodes):
            if i in active_set:
                # ACTIVE
                cons_degree_out[i].SetBounds(1, 1)
                cons_degree_in[i].SetBounds(1, 1)
                u[i].SetBounds(demands[i], capacity)
            else:
                # INACTIVE
                cons_degree_out[i].SetBounds(0, 0)
                cons_degree_in[i].SetBounds(0, 0)
                u[i].SetBounds(0, 0)

        solver.SetTimeLimit(100)
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            return float(objective.Value())
        return 0.0

    # ==========================================
    # 3. HEURISTIC WRAPPER
    # ==========================================
    def h_lp_relaxation_2_idx(state):
        unvisited = state[unvisited_var]

        if not unvisited: return 0.0

        # Define Active Set: Unvisited
        # Note: 2-index relaxation often ignores "current location" specific routing logic 
        # and just solves the flow problem for the remaining set.
        active_key = tuple(sorted(list(unvisited)))

        return _solve_2idx(active_key)

    return h_lp_relaxation_2_idx

def dual_bound_expression_function(didp_bundle):
    """ 
    Registry containing ALL heuristics (Combinatorial + LP) for CVRP.
    """
    model, metadata = didp_bundle

    # Extract metadata
    unvisited_var = metadata['unvisited_var']
    location_var = metadata['location_var']
    distance_list = metadata['distance_matrix']
    cost_matrix = np.array(distance_list)
    demand = metadata['demand']
    capacity = metadata['capacity']
    num_vehicles = metadata['num_vehicles']
    n_nodes = metadata['num_nodes']

    # Pre-computation
    masked_cost = cost_matrix.astype(float).copy()
    np.fill_diagonal(masked_cost, np.inf)
    min_outgoing_arr = np.min(masked_cost, axis=1)
    min_incoming_arr = np.min(masked_cost, axis=0)

    # --- Initialize LP Bounds ---
    h_lp_3idx = create_persistent_lp_relaxation_3_index_dual_bounds(metadata)
    h_lp_2idx = create_persistent_lp_relaxation_2_index_dual_bounds(metadata)

    # ==========================================
    # COMBINATORIAL BOUNDS (Internal Workers)
    # ==========================================

    # --- Flow Bound ---
    @lru_cache(maxsize=100000)
    def _calc_flow(unvisited_tuple):
        s = sum(demand[v] * distance_list[0][v] for v in unvisited_tuple)
        return float(round((2.0 / capacity) * s))

    def h_flow(state):
        U = state[unvisited_var]
        return _calc_flow(tuple(sorted(list(U)))) if U else 0.0

    # --- Degree Average Bound ---
    @lru_cache(maxsize=100000)
    def _calc_degree(active_tuple):
        nodes = list(active_tuple)
        sub_mat = cost_matrix[np.ix_(nodes, nodes)].astype(float)
        np.fill_diagonal(sub_mat, np.inf)
        sum_in = np.sum(np.min(sub_mat, axis=0))
        sum_out = np.sum(np.min(sub_mat, axis=1))
        return float(0.5 * (sum_in + sum_out))

    def h_degree_average(state):
        U = state[unvisited_var]
        curr = state[location_var]
        if not U and curr == 0: return 0.0

        # Active: Current -> Unvisited -> Depot
        active = set(U)
        active.add(curr)
        active.add(0)
        return _calc_degree(tuple(sorted(list(active))))

    # --- Global Min Flow ---
    @lru_cache(maxsize=100000)
    def _calc_min_flow_static(unvisited_tuple):
        val_out = sum(min_outgoing_arr[u] for u in unvisited_tuple)
        val_in = sum(min_incoming_arr[u] for u in unvisited_tuple)
        return val_out, val_in

    def h_global_min_flow(state):
        U = state[unvisited_var]
        curr = state[location_var]
        if not U and curr == 0: return 0.0

        # Get static part
        val_out, val_in = _calc_min_flow_static(tuple(sorted(list(U))))

        # Add dynamic part (Current)
        if curr != 0:
            val_out += min_outgoing_arr[curr]
            val_in += min_incoming_arr[0]

        return float(max(val_out, val_in))

    # --- MST Bound ---
    @lru_cache(maxsize=100000)
    def _calc_mst(unvisited_tuple):
        if not unvisited_tuple: return 0.0
        nodes = [0] + list(unvisited_tuple)
        sub_mat = cost_matrix[np.ix_(nodes, nodes)]
        mst = minimum_spanning_tree(sub_mat)
        return float(mst.sum())

    def h_mst(state):
        U = state[unvisited_var]
        return _calc_mst(tuple(sorted(list(U))))

    # --- 1-Tree Bound ---
    @lru_cache(maxsize=100000)
    def _calc_1tree(unvisited_tuple):
        subset = list(unvisited_tuple)
        depot_edges = sorted(cost_matrix[0, subset])
        e1 = depot_edges[0]
        e2 = depot_edges[1] if len(depot_edges) > 1 else 0.0

        if len(subset) > 1:
            sub_mat = cost_matrix[np.ix_(subset, subset)]
            mst_val = minimum_spanning_tree(sub_mat).sum()
        else:
            mst_val = 0.0
        return float(mst_val + e1 + e2)

    def h_1tree(state):
        U = state[unvisited_var]
        if not U: return 0.0
        return _calc_1tree(tuple(sorted(list(U))))

    # --- Assignment Bound ---
    @lru_cache(maxsize=100000)
    def _calc_assignment(unvisited_tuple):
        nodes = [0] + list(unvisited_tuple)
        sub_mat = cost_matrix[np.ix_(nodes, nodes)]
        assign_mat = sub_mat.astype(float).copy()
        np.fill_diagonal(assign_mat, np.inf)
        row, col = linear_sum_assignment(assign_mat)
        return float(assign_mat[row, col].sum())

    def h_assignment(state):
        U = state[unvisited_var]
        if not U: return 0.0
        return _calc_assignment(tuple(sorted(list(U))))

    # --- Eigenvalue Bound ---
    @lru_cache(maxsize=100000)
    def _calc_eigen(unvisited_tuple):
        nodes = [0] + list(unvisited_tuple)
        N = len(nodes)
        if N < 2: return 0.0

        D_sub = cost_matrix[np.ix_(nodes, nodes)]
        one = np.ones((N, 1))
        P = np.eye(N) - (one @ one.T) / N
        M = -P @ D_sub @ P
        M = (M + M.T) / 2
        try:
            eigvals = np.flip(eigh(M)[0])
        except np.linalg.LinAlgError:
            return 0.0

        eigvals = eigvals[np.abs(eigvals) > 1e-9]
        coeffs = np.array([1 - np.cos(2 * np.pi * k / N) for k in range(1, N)])

        phi = 0.0
        if N > 1:
            if N % 2 == 1:
                num_terms = (N - 1) // 2
                if 2 * num_terms <= len(eigvals) and num_terms <= len(coeffs):
                     phi = sum(coeffs[k-1] * (eigvals[2*k - 2] + eigvals[2*k - 1]) for k in range(1, num_terms + 1))
            else:
                num_sum_terms = N // 2 - 1
                if 2 * num_sum_terms < len(eigvals) and num_sum_terms <= len(coeffs):
                    phi = sum(coeffs[k-1] * (eigvals[2*k - 2] + eigvals[2*k - 1]) for k in range(1, num_sum_terms + 1))
                    if N-2 < len(eigvals): phi += 2 * eigvals[N - 2]
                elif N > 1 and N-2 < len(eigvals):
                    phi = 2 * eigvals[N - 2]
        return float(phi)

    def h_eigen(state):
        U = state[unvisited_var]
        if not U: return 0.0
        return _calc_eigen(tuple(sorted(list(U))))

    # Return Registry
    return automatic_creation_of_dual_bounds_registry(locals())
