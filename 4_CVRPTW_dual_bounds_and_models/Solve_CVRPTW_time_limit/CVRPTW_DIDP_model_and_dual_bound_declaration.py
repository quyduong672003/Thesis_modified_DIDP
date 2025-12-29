import sys
import os
import numpy as np
import vrplib
import modified_didppy as m_dp
from ortools.linear_solver import pywraplp
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment
from numpy.linalg import eigh
from functools import lru_cache
from evolutionary_algorithm_lib.utils import automatic_creation_of_dual_bounds_registry

current_num_locations = 0
current_num_vehicles = 0
current_capacity = 0.0
current_cust_demand = []
current_avail_time = []
current_due_date = []
current_serve_time = []
current_travel_cost = []

def read_formated_data(file_path):
    """
    Reads a Solomon format .txt file using vrplib and returns a dictionary 
    formatted for CVRPTW LP Relaxation and DIDP models.

    Ensures all numerical data (demand, time windows, service times, costs, capacity) 
    are returned as floats.
    """
    # 1. Read instance using vrplib
    # instance_format='solomon' ensures correct parsing of sections
    instance = vrplib.read_instance(file_path, instance_format='solomon')

    # 2. Extract Data & Cast to Float
    # 'edge_weight' is the distance matrix computed by vrplib
    travel_cost = instance['edge_weight'].astype(float).tolist()

    # 'node_coord' is available if you ever need it, but we use the pre-calc weights
    num_locations = len(instance['node_coord'])

    # 3. Return Bundle
    return {
        'num_locations': num_locations,
        'num_vehicles': int(instance.get('vehicles', 25)), 
        'capacity': float(instance['capacity']),

        # Cast demand to float list
        'demand': instance['demand'].astype(float).tolist(),

        # Cast Time Windows to float list
        # Col 0 is ready_time (earliest arrival), Col 1 is due_date (latest arrival)
        'ready_time': instance['time_window'][:, 0].astype(float).tolist(),
        'due_date': instance['time_window'][:, 1].astype(float).tolist(),

        # Cast Service Time to float list
        'service_time': instance['service_time'].astype(float).tolist(),

        # Use the pre-computed edge weights from vrplib
        'travel_cost': travel_cost
    }

def update_globals_for_cvrptw(file_path):
    """Updates global variables from file."""
    global current_num_locations, current_num_vehicles, current_capacity
    global current_cust_demand, current_avail_time, current_due_date
    global current_serve_time, current_travel_cost

    data = read_formated_data(file_path)

    current_num_locations = data['num_locations']
    current_num_vehicles = data['num_vehicles']
    current_capacity = data['capacity']
    current_cust_demand = data['demand']
    current_serve_time = data['service_time']
    current_travel_cost = data['travel_cost']

    # Split Time Windows into Ready and Due lists
    current_avail_time = data['ready_time']
    current_due_date = data['due_date']

    return True

def creation_of_didp_model_function():
    num_locations = current_num_locations
    num_vehicles = current_num_vehicles
    q = current_capacity
    cust_demand = current_cust_demand
    avail_time = current_avail_time
    due_date = current_due_date
    serve_time = current_serve_time
    travel_cost = current_travel_cost

    # =====================================================================================
    # DIDP Model Definition
    # =====================================================================================
    model = m_dp.Model(float_cost=True)

    # Object types for customers/locations and vehicles
    customer = model.add_object_type(number=num_locations)
    vehicle = model.add_object_type(number=num_vehicles)

    # -------------------- State Variables --------------------
    # Set of unvisited customers
    unvisited_locations = model.add_set_var(object_type=customer, target=list(range(1, num_locations)))

    # Per-vehicle state variables, stored in Python lists for easy access
    vehicle_locations = [
    model.add_element_var(object_type=customer, target=0, name=f"loc_v{v}")
    for v in range(num_vehicles)
    ]
    vehicle_loads = [
    model.add_float_var(target=0, name=f"load_v{v}")
    for v in range(num_vehicles)
    ]
    vehicle_times = [
    model.add_float_resource_var(target=0, less_is_better=True, name=f"time_v{v}")
    for v in range(num_vehicles)
    ]
    chosen_customer = model.add_element_var(object_type=customer, target=0, name="chosen_customer")
    alpha = model.add_int_var(target=0, name="alpha")

    # -------------------- Tables of Constants --------------------
    demand = model.add_float_table(cust_demand)
    ready_time = model.add_float_table(avail_time)
    due_time = model.add_float_table(due_date)
    service_time = model.add_float_table(serve_time)
    travel_time = model.add_float_table(travel_cost)

    # -------------------- Transitions --------------------
    # Choose customer j to be visited next
    for j in range(1, num_locations):
        choosing_customer_transition = m_dp.Transition(
            name=f"choose_customer_{j}_to_visit",
            cost=m_dp.FloatExpr.state_cost(),
            preconditions =[
                unvisited_locations.contains(j),
                alpha == 0
                ],
            effects=[
                (chosen_customer, j),
                (alpha, 1)
                ],
        )
        model.add_transition(choosing_customer_transition)

        # Transition to visit a customer j with a vehicle v
    for v in range(num_vehicles):
        arrival_time = m_dp.max(
            vehicle_times[v] + travel_time[vehicle_locations[v], chosen_customer],
            ready_time[chosen_customer]
        )

        departure_time = arrival_time + service_time[chosen_customer]

        visit_transition = m_dp.Transition(
            name=f"visit_chosen_customer_with_vehicle_{v}",
            cost=travel_time[vehicle_locations[v], chosen_customer] + m_dp.FloatExpr.state_cost(),
            preconditions=[
                unvisited_locations.contains(chosen_customer),
                vehicle_loads[v] + demand[chosen_customer] <= q,
                arrival_time <= due_time[chosen_customer],
                alpha == 1,
            ],
            effects=[
                (unvisited_locations, unvisited_locations.remove(chosen_customer)),
                (vehicle_locations[v], chosen_customer),
                (vehicle_loads[v], vehicle_loads[v] + demand[chosen_customer]),
                (vehicle_times[v], departure_time),
                (alpha, 0),
            ],
        )

        model.add_transition(visit_transition)


    # Transitions for each vehicle to return to the depot after all customers are served
    for v in range(num_vehicles):
        return_to_depot_transition = m_dp.Transition(
            name=f"return_vehicle_{v}_to_depot",
            cost=travel_time[vehicle_locations[v], 0] + m_dp.FloatExpr.state_cost(),
            preconditions=[unvisited_locations.is_empty(), vehicle_locations[v] != 0],
            effects=[(vehicle_locations[v], 0)],
        )
        model.add_transition(return_to_depot_transition)

    # --- 1. Global Capacity Constraint (The Efficient "Cut") ---
    # Logic: Total Capacity Available >= Total Demand Remaining
    # Summing variables in Python creates a DIDP expression automatically
    total_current_load = sum(vehicle_loads) 
    total_fleet_capacity = num_vehicles * q

    # demand[unvisited_locations] automatically sums the weight of items in the set
    model.add_state_constr(
        (total_fleet_capacity - total_current_load) >= demand[unvisited_locations]
    )

    # -------------------- Base Case --------------------
    # All customers visited AND all vehicles are at the depot
    base_conditions = [unvisited_locations.is_empty()]
    for v in range(num_vehicles):
        base_conditions.append(vehicle_locations[v] == 0)
    model.add_base_case(base_conditions)

    # =========================================================
    # 3. Bundle Metadata
    # =========================================================
    metadata = {
        "unvisited_locations": unvisited_locations,
        "vehicle_locations": vehicle_locations,
        "vehicle_loads": vehicle_loads,
        "vehicle_times": vehicle_times,
        "distance_matrix": travel_cost,
        "demand": cust_demand,
        "due_time": due_date,
        "ready_time": avail_time,
        "service_time": serve_time,
        "capacity": q,
        "num_vehicles": num_vehicles,
        "num_locations": num_locations
    }

    didp_bundle = (model, metadata)
    return didp_bundle

def create_persistent_lp_relaxation_3_index_dual_bounds(metadata):
    """
    Creates a persistent 3-Index CVRP Relaxation with LRU CACHING.
    """
    # --- Extract Static Data ---
    n_nodes = metadata['num_locations']
    n_vehicles = metadata['num_vehicles']
    capacity = metadata['capacity']
    demands = metadata['demand']          
    dist_matrix = metadata['distance_matrix'] 

    # State Variables
    unvisited_var = metadata['unvisited_locations']
    vehicle_vars = metadata['vehicle_locations']

    # ==========================================
    # 1. INITIALIZATION (Runs Once)
    # ==========================================
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return lambda state: 0.0

    infinity = solver.infinity()
    x = {}
    y = {}
    u = {}
    cons_assignment = {} 

    # --- Variables & Constraints Construction (Same as before) ---
    # 1. Create x_ijk (Flow)
    for k in range(n_vehicles):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    x[(k, i, j)] = solver.NumVar(0, 1, f'x_{k}_{i}_{j}')

    # 2. Create y_ik (Assignment)
    for i in range(n_nodes):
        for k in range(n_vehicles):
            y[(i, k)] = solver.NumVar(0, 1, f'y_{i}_{k}')

    # 3. Create u_ik (Potentials)
    for i in range(1, n_nodes):
        for k in range(n_vehicles):
            u[(i, k)] = solver.NumVar(0, capacity, f'u_{i}_{k}')

    # (1.29) Customer Assignment
    for i in range(1, n_nodes):
        c = solver.Constraint(0, 0, f'assign_{i}')
        for k in range(n_vehicles):
            c.SetCoefficient(y[(i, k)], 1)
        cons_assignment[i] = c

    # (1.30) Depot Usage
    c_depot = solver.Constraint(0, n_vehicles, 'depot_usage')
    for k in range(n_vehicles):
        c_depot.SetCoefficient(y[(0, k)], 1)

    # (1.31) Flow Conservation
    for k in range(n_vehicles):
        for i in range(n_nodes):
            c_out = solver.Constraint(0, 0, f'flow_out_{i}_{k}')
            c_out.SetCoefficient(y[(i, k)], -1)
            for j in range(n_nodes):
                if i != j: c_out.SetCoefficient(x[(k, i, j)], 1)

            c_in = solver.Constraint(0, 0, f'flow_in_{i}_{k}')
            c_in.SetCoefficient(y[(i, k)], -1)
            for j in range(n_nodes):
                if i != j: c_in.SetCoefficient(x[(k, j, i)], 1)

    # (1.37) & (1.38) MTZ
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
                if i != j:
                    objective.SetCoefficient(x[(k, i, j)], dist_matrix[i][j])
    objective.SetMinimization()

    # ==========================================
    # 2. CACHED SOLVER WORKER
    # ==========================================
    # We define the solver logic as an internal function and cache IT.
    # The key will be a tuple of active nodes.

    @lru_cache(maxsize=10000)
    def _solve_cached_lp(active_nodes_tuple):
        """
        Internal worker that sets bounds and solves.
        Input must be a hashable tuple (sorted list of active nodes).
        """
        # Convert tuple back to set for fast lookup
        active_set = set(active_nodes_tuple)

        # Update Constraints
        for i in range(1, n_nodes):
            if i in active_set:
                # ACTIVE
                cons_assignment[i].SetBounds(1, 1)
                for k in range(n_vehicles):
                    u[(i, k)].SetBounds(demands[i], capacity)
            else:
                # INACTIVE
                cons_assignment[i].SetBounds(0, 0)
                for k in range(n_vehicles):
                    u[(i, k)].SetBounds(0, 0)

        # Solve with Time Limit
        solver.SetTimeLimit(100) # 100ms limit
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            return float(objective.Value())
        return 0.0

    # ==========================================
    # 3. HEURISTIC WRAPPER
    # ==========================================
    def h_lp_relaxation_3_idx(state):
        unvisited = state[unvisited_var]
        current_locs = [state[v_var] for v_var in vehicle_vars]

        # Quick exit
        if not unvisited and all(loc == 0 for loc in current_locs): 
            return 0.0

        # Build Active Set
        # Valid nodes are Unvisited U Current_Vehicle_Locations
        active_customers = set(unvisited)
        for loc in current_locs:
            if loc != 0:
                active_customers.add(loc)

        # CRITICAL: Convert to sorted tuple for the cache key
        # If we passed the set or list directly, lru_cache would fail (unhashable)
        # Sorting ensures that visiting {1, 2} is the same cache hit as visiting {2, 1}
        active_key = tuple(sorted(list(active_customers)))

        return _solve_cached_lp(active_key)

    return h_lp_relaxation_3_idx

def create_persistent_lp_relaxation_2_index_dual_bounds(metadata):
    """
    Creates a persistent 2-Index (Flow-based) CVRP Relaxation with LRU CACHING.
    - SCALE: Capable of handling N=100 in < 0.1s per node.
    - LOGIC: Drops 'Vehicle' dimension. Treats flow as aggregate.
    - SOLVER: Google OR-Tools (GLOP).
    """
    # --- Extract Static Data (Matched to DIDP Metadata) ---
    n_nodes = metadata['num_locations']
    n_vehicles = metadata['num_vehicles']
    capacity = metadata['capacity']

    # Note: These are DIDP Table objects or Lists
    demands = metadata['demand']
    dist_matrix = metadata['distance_matrix']

    # State Variables
    unvisited_var = metadata['unvisited_locations']
    vehicle_vars = metadata['vehicle_locations'] # List of vehicle element variables

    # ==========================================
    # 1. INITIALIZATION (Runs Once)
    # ==========================================
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return lambda state: 0.0

    infinity = solver.infinity()

    # --- Variables ---
    # x[i, j]: Binary flow from i to j (Aggregated over all vehicles)
    x = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                x[(i, j)] = solver.NumVar(0, 1, f'x_{i}_{j}')

    # u[i]: Accumulated load variable for MTZ
    u = {i: solver.NumVar(0, capacity, f'u_{i}') for i in range(1, n_nodes)}

    # --- Constraints ---
    cons_degree_out = {} # Outgoing degree 
    cons_degree_in = {}  # Incoming degree

    # 1. Degree Constraints (Customers 1..N)
    # Each active customer must have exactly 1 outgoing and 1 incoming edge
    for i in range(1, n_nodes):
        # Outgoing
        c_out = solver.Constraint(0, 0, f'deg_out_{i}')
        for j in range(n_nodes):
            if i != j:
                c_out.SetCoefficient(x[(i, j)], 1)
        cons_degree_out[i] = c_out

        # Incoming
        c_in = solver.Constraint(0, 0, f'deg_in_{i}')
        for j in range(n_nodes):
            if i != j:
                c_in.SetCoefficient(x[(j, i)], 1)
        cons_degree_in[i] = c_in

    # 2. Depot Degree Constraints (Static)
    # Total outgoing flow from Depot == Number of active vehicles (<= K)
    # We relax this to <= K for lower bound purposes
    c_depot_out = solver.Constraint(0, n_vehicles, 'depot_out')
    for j in range(1, n_nodes):
        c_depot_out.SetCoefficient(x[(0, j)], 1)

    # Total incoming flow to Depot <= K
    c_depot_in = solver.Constraint(0, n_vehicles, 'depot_in')
    for i in range(1, n_nodes):
        c_depot_in.SetCoefficient(x[(i, 0)], 1)

    # 3. MTZ / Capacity Constraints
    # Standard MTZ: u_j - u_i + Capacity * x_ij <= Capacity - d_j
    # This prevents subtours and ensures capacity compliance roughly.
    for i in range(1, n_nodes):
        for j in range(1, n_nodes):
            if i != j:
                # [Correction] Explicit float cast for RHS to avoid numpy int errors
                rhs = float(capacity - demands[j])
                c = solver.Constraint(-infinity, rhs, f'mtz_{i}_{j}')

                c.SetCoefficient(u[j], 1)
                c.SetCoefficient(u[i], -1)
                c.SetCoefficient(x[(i, j)], capacity)

    # --- Objective ---
    objective = solver.Objective()
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                objective.SetCoefficient(x[(i, j)], dist_matrix[i][j])
    objective.SetMinimization()

    # ==========================================
    # 2. CACHED SOLVER WORKER (Internal)
    # ==========================================
    # We define the solver logic as an internal function and cache IT.
    # The key will be a tuple of active nodes.

    @lru_cache(maxsize=10000)
    def _solve_cached_lp_2_idx(active_nodes_tuple):
        """
        Internal worker that sets bounds and solves.
        Input must be a hashable tuple (sorted list of active nodes).
        """
        # Convert tuple back to set for fast lookup
        active_customers = set(active_nodes_tuple)

        # Toggle Active Nodes
        # We iterate 1..N to update bounds based on current state
        for i in range(1, n_nodes):
            if i in active_customers:
                # ACTIVE:
                # Degree must be 1 (Visited exactly once)
                cons_degree_out[i].SetBounds(1, 1)
                cons_degree_in[i].SetBounds(1, 1)
                # Load variable active
                u[i].SetBounds(demands[i], capacity)
            else:
                # INACTIVE:
                # Degree must be 0 (Removed from graph)
                cons_degree_out[i].SetBounds(0, 0)
                cons_degree_in[i].SetBounds(0, 0)
                # Load variable forced to 0
                u[i].SetBounds(0, 0)

        # Solve
        solver.SetTimeLimit(100) # 100ms
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            return float(objective.Value())
        return 0.0

    # ==========================================
    # 3. HEURISTIC WRAPPER
    # ==========================================
    def h_lp_relaxation_2_idx(state):
        unvisited = state[unvisited_var]

        # [UPDATED]: Retrieve locations for ALL vehicles
        current_locs = [state[v_var] for v_var in vehicle_vars]

        # Optimization: Solved state
        if not unvisited and all(loc == 0 for loc in current_locs): 
            return 0.0

        # Define Active Set: Unvisited
        active_customers = set(unvisited)

        # [UPDATED]: Add current locations to active set (treating them as nodes to be covered)
        # This provides a valid relaxation for the remaining tour cost
        for loc in current_locs:
            if loc != 0:
                active_customers.add(loc)

        # CRITICAL: Convert to sorted tuple for the cache key
        # Sorting ensures that visiting {1, 2} is the same cache hit as visiting {2, 1}
        active_key = tuple(sorted(list(active_customers)))

        return _solve_cached_lp_2_idx(active_key)

    return h_lp_relaxation_2_idx

def create_persistent_cvrptw_cordeau_relaxed_model(metadata):
    """
    Creates a persistent 3-Index CVRPTW Relaxation based on Cordeau (2002).
    - Includes Time Window variables (w) and Capacity constraints.
    - Uses Google OR-Tools (GLOP) with LRU Caching.
    """
    # --- Extract Static Data ---
    num_locations = metadata['num_locations'] # Original N
    num_vehicles = metadata['num_vehicles']
    capacity = metadata['capacity']

    # Original Data
    cust_demand = metadata['demand']
    serve_time = metadata['service_time']
    ready_time = metadata['ready_time']
    due_date = metadata['due_time']
    travel_cost = metadata['distance_matrix']

    # State Variables
    unvisited_var = metadata['unvisited_locations']
    vehicle_vars = metadata['vehicle_locations']

    # --- 0. Data Augmentation (StartDepot=0, EndDepot=N) ---
    StartDepot = 0
    EndDepot = num_locations 

    # Augment lists (Add dummy end node)
    aug_demand = cust_demand + [0.0]
    aug_serve = serve_time + [0.0]
    aug_ready = ready_time + [ready_time[0]]
    aug_due = due_date + [due_date[0]]

    # Augment Cost Matrix
    aug_cost = [row[:] + [row[0]] for row in travel_cost] 
    aug_cost.append(aug_cost[0][:]) 

    # Sets
    K = range(num_vehicles)
    N_customers = range(1, num_locations) # 1..N-1
    V_all = range(num_locations + 1)      # 0..N

    # --- FIX: Use a SET for A to ensure O(1) lookups ---
    A = set()
    for i in V_all:
        for j in V_all:
            if i == j: continue
            if i == EndDepot: continue   # Nothing leaves EndDepot
            if j == StartDepot: continue # Nothing enters StartDepot
            A.add((i,j))

    # Helper for Big-M
    #big_m = {}
    #for (i, j) in A:
        #val = aug_due[i] + aug_serve[i] + aug_cost[i][j] - aug_ready[j]
    #   big_m[(i,j)] = 10**7 # max(val, 0)

    # ==========================================
    # 1. INITIALIZATION (Runs Once)
    # ==========================================
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return lambda state: 0.0

    infinity = solver.infinity()

    # --- Variables ---
    x = {} # Flow x_k_i_j
    w = {} # Time w_k_i

    # Note: iterating over set A is fast
    for k in K:
        for (i, j) in A:
            x[(k, i, j)] = solver.NumVar(0, 1, f"x_{k}_{i}_{j}")
        for i in V_all:
            w[(k, i)] = solver.NumVar(0, 10**6, f"w_{k}_{i}")

    # --- Constraints ---

    # 1. Assignment Constraints (Mutable)
    cons_assignment = {}
    for i in N_customers:
        c = solver.Constraint(0, 0, f"assign_{i}")
        for k in K:
            # We iterate V_all and check membership in A (O(1) now)
            for j in V_all:
                if (i, j) in A:
                    c.SetCoefficient(x[(k, i, j)], 1)
        cons_assignment[i] = c

    # 2. Depot Departure
    for k in K:
        c = solver.Constraint(1, 1, f"depot_out_{k}")
        for j in V_all:
            if (StartDepot, j) in A:
                c.SetCoefficient(x[(k, StartDepot, j)], 1)

    # 3. Flow Conservation
    for k in K:
        for j in N_customers:
            c = solver.Constraint(0, 0, f"flow_{k}_{j}")
            # Inflow
            for i in V_all:
                if (i, j) in A: c.SetCoefficient(x[(k, i, j)], 1)
            # Outflow
            for i in V_all:
                if (j, i) in A: c.SetCoefficient(x[(k, j, i)], -1)

    # 4. Depot Arrival
    for k in K:
        c = solver.Constraint(1, 1, f"depot_in_{k}")
        for i in V_all:
            if (i, EndDepot) in A:
                c.SetCoefficient(x[(k, i, EndDepot)], 1)

    # 5. Capacity
    for k in K:
        c = solver.Constraint(-infinity, capacity, f"cap_{k}")
        for i in N_customers:
            for j in V_all:
                if (i, j) in A:
                    c.SetCoefficient(x[(k, i, j)], aug_demand[i])

    # 6. Time Propagation (Linearized)
    for k in K:
        for (i, j) in A:
            M = 10**6
            rhs = M - aug_serve[i] - aug_cost[i][j]
            c = solver.Constraint(-infinity, rhs, f"time_prop_{k}_{i}_{j}")
            c.SetCoefficient(w[(k, i)], 1)
            c.SetCoefficient(w[(k, j)], -1)
            c.SetCoefficient(x[(k, i, j)], M)

    # 7. Time Windows (Linked to Visits)
    for k in K:
        for i in N_customers:
            # Visit variable (sum outgoing)
            visit_expr = [] 
            for j in V_all:
                if (i, j) in A: visit_expr.append(x[(k, i, j)])

            # Lower Bound: w >= a * visit
            c_lb = solver.Constraint(-infinity, 0, f"tw_lb_{k}_{i}")
            c_lb.SetCoefficient(w[(k, i)], -1)
            for var in visit_expr: c_lb.SetCoefficient(var, aug_ready[i])

            # Upper Bound: w <= b * visit
            c_ub = solver.Constraint(-infinity, 0, f"tw_ub_{k}_{i}")
            c_ub.SetCoefficient(w[(k, i)], 1)
            for var in visit_expr: c_ub.SetCoefficient(var, -aug_due[i])

    # --- Objective ---
    objective = solver.Objective()
    for k in K:
        for (i, j) in A:
            objective.SetCoefficient(x[(k, i, j)], aug_cost[i][j])
    objective.SetMinimization()

    # ==========================================
    # 2. CACHED SOLVER WORKER
    # ==========================================
    @lru_cache(maxsize=10000)
    def _solve_cordeau(active_tuple):
        """
        Internal worker to solve the Cordeau relaxation for a specific set of active nodes.
        active_tuple: Sorted tuple of indices that MUST be visited.
        """
        active_set = set(active_tuple)

        # Update Assignment Constraints
        for i in N_customers:
            if i in active_set:
                # Must be visited exactly once
                cons_assignment[i].SetBounds(1, 1)
            else:
                # Already visited / Inactive -> Force flow to 0 (remove from graph)
                cons_assignment[i].SetBounds(0, 0)

        # Solve
        solver.SetTimeLimit(100) # 100ms limit per call
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            return float(objective.Value())
        return 0.0

    # ==========================================
    # 3. HEURISTIC WRAPPER
    # ==========================================
    def h_lp_cordeau(state):
        unvisited = state[unvisited_var]

        # If no unvisited customers, cost is 0 (or cost to return to depot)
        if not unvisited:
            return 0.0

        # Create cache key: Sorted tuple of unvisited nodes
        active_key = tuple(sorted(list(unvisited)))

        return _solve_cordeau(active_key)

    return h_lp_cordeau

def dual_bound_expression_function(didp_bundle):
    """ 
    Returns a dictionary of heuristic functions (bounds) bound to the model data.
    Includes LP relaxations, Flow, MST, 1-Tree, Assignment, and Eigenvalue bounds.
    """

    model, metadata = didp_bundle

    # --- 1. Extract DIDP Variables (from Metadata) ---
    unvisited_var = metadata['unvisited_locations']
    vehicle_vars = metadata['vehicle_locations']  # List of ElementVars

    capacity = metadata['capacity']
    num_vehicles = metadata['num_vehicles']
    n_nodes = metadata['num_locations'] 

    # --- 2. Extract RAW Data for Heuristics ---
    # Metadata contains DIDP Tables (symbolic). Heuristics (MST, Eigen, etc.) need 
    # the actual numerical lists/matrices. We fetch them from global scope if available.

    distance_list = metadata['distance_matrix']
    cost_matrix = np.array(distance_list) 
    demand_list = metadata['demand']

    # --- 3. Pre-computation for Bounds (Run once per problem) ---
    # masked_cost: diagonal is infinity to ignore self-loops
    masked_cost = cost_matrix.astype(float).copy()
    np.fill_diagonal(masked_cost, np.inf)

    # min_outgoing_arr[i] = min cost to leave node i
    min_outgoing_arr = np.min(masked_cost, axis=1)

    # min_incoming_arr[j] = min cost to enter node j
    min_incoming_arr = np.min(masked_cost, axis=0)

    # ==========================================
    # 4. LP Relaxation Bounds (Persistent)
    # ==========================================
    # Pass metadata through (The LP functions must expect the specific keys you defined)
    h_lp_relaxation_3_idx = create_persistent_lp_relaxation_3_index_dual_bounds(metadata=metadata)
    h_lp_relaxation_2_idx = create_persistent_lp_relaxation_2_index_dual_bounds(metadata=metadata)
    h_lp_cordeau = create_persistent_cvrptw_cordeau_relaxed_model(metadata=metadata)

    # ==========================================
    # 5. Define Combinatorial Bounds (OPTIMIZED WITH LRU CACHE)
    # =========================================

    # --- Flow Bound ---
    # Although h_flow is fast (O(N)), caching avoids repeated list comprehensions.
    @lru_cache(maxsize=100000)
    def _cached_flow_calc(unvisited_tuple):
        # Sum demand of unvisited * dist to depot
        s = sum(demand_list[v] * distance_list[0][v] for v in unvisited_tuple)
        return float(round((2.0 / capacity) * s))

    def h_flow(state):
        U = state[unvisited_var]
        if not U: return 0.0
        try:
            # Convert to tuple for cache
            U_tuple = tuple(sorted(list(U)))
            return _cached_flow_calc(U_tuple)
        except Exception as e:
            print(f"Error in h_flow: {e}")
            return 0.0

    # --- Degree Average Bound (Local Subgraph) ---
    # This involves matrix slicing and min/sum operations. 
    # Since it depends on vehicle locations, we cache based on the "Active Node Set".
    @lru_cache(maxsize=100000)
    def _cached_degree_average_calc(active_nodes_tuple):
        nodes = list(active_nodes_tuple)
        sub_mat = cost_matrix[np.ix_(nodes, nodes)].astype(float)
        np.fill_diagonal(sub_mat, np.inf)

        mins_in = np.min(sub_mat, axis=0) 
        mins_out = np.min(sub_mat, axis=1)

        sum_in = np.sum(mins_in)      
        sum_out = np.sum(mins_out)   
        return float(0.5 * (sum_in + sum_out))

    def h_degree_average(state):
        U = state[unvisited_var]
        curr_locs = [state[v_var] for v_var in vehicle_vars]

        if not U and all(loc == 0 for loc in curr_locs):
            return 0.0
        try: 
            # 1. Build the full set of active nodes (Unvisited + Vehicles + Depot)
            active_set = set(U)
            for loc in curr_locs:
                if loc != 0:
                    active_set.add(loc)
            active_set.add(0) # Always include depot

            # 2. Convert to sorted tuple for cache key
            active_tuple = tuple(sorted(list(active_set)))

            return _cached_degree_average_calc(active_tuple)
        except Exception as e:
            print(f"Error in h_degree_average: {e}")
            return 0.0

    # --- Global Min Flow Bound ---
    # Optimization: Cache the sum of unvisited nodes part. 
    # The vehicle part is fast and dynamic, so we add it outside.
    @lru_cache(maxsize=100000)
    def _cached_global_min_flow_unvisited_part(unvisited_tuple):
        val_out = sum(min_outgoing_arr[u] for u in unvisited_tuple)
        val_in = sum(min_incoming_arr[u] for u in unvisited_tuple)
        return val_out, val_in

    def h_global_min_flow(state):
        U = state[unvisited_var]
        curr_locs = [state[v_var] for v_var in vehicle_vars]

        if not U and all(loc == 0 for loc in curr_locs):
            return 0.0
        try:
            # Get cached sums for the static unvisited part
            U_tuple = tuple(sorted(list(U)))
            val_out, val_in = _cached_global_min_flow_unvisited_part(U_tuple)

            # Add dynamic vehicle parts
            for loc in curr_locs:
                if loc != 0:
                    val_out += min_outgoing_arr[loc]

            # Logic check: At least one vehicle must return to depot
            if val_out > 0: 
                val_in += min_incoming_arr[0]

            return float(max(val_out, val_in))
        except Exception as e:
            print(f"Error in h_global_min_flow: {e}")
            return 0.0

    # --- MST Bound ---
    @lru_cache(maxsize=100000)
    def _cached_mst_calc(unvisited_tuple):
        if not unvisited_tuple: return 0.0
        nodes = [0] + sorted(list(unvisited_tuple))
        sub_mat = cost_matrix[np.ix_(nodes, nodes)]
        mst = minimum_spanning_tree(sub_mat)
        return float(mst.sum())

    def h_mst(state):
        U = state[unvisited_var]
        # Convert to tuple so it can be hashed by lru_cache
        U_tuple = tuple(sorted(list(U)))
        return _cached_mst_calc(U_tuple)

    # --- 1-Tree Bound ---
    @lru_cache(maxsize=100000)
    def _cached_1tree_calc(unvisited_tuple):
        subset_nodes = list(unvisited_tuple) # Already sorted from wrapper

        # Cheapest 2 edges connected to depot from the subset
        depot_edges = sorted(cost_matrix[0, subset_nodes])
        e1 = depot_edges[0]
        e2 = depot_edges[1] if len(depot_edges) > 1 else 0.0

        if len(subset_nodes) > 1:
            sub_mat = cost_matrix[np.ix_(subset_nodes, subset_nodes)]
            mst_val = minimum_spanning_tree(sub_mat).sum()
        else:
            mst_val = 0.0 
        return float(mst_val + e1 + e2)

    def h_1tree(state):
        U = state[unvisited_var]
        if not U: return 0.0
        try:
            U_tuple = tuple(sorted(list(U)))
            return _cached_1tree_calc(U_tuple)
        except Exception as e:
            print(f"Error in h_1tree: {e}")
            return 0.0

    # --- Assignment Bound ---
    @lru_cache(maxsize=100000)
    def _cached_assignment_calc(unvisited_tuple):
        nodes = [0] + sorted(list(unvisited_tuple))
        sub_mat = cost_matrix[np.ix_(nodes, nodes)]

        # Prepare for linear assignment (diagonal inf)
        assign_mat = sub_mat.astype(float).copy()
        np.fill_diagonal(assign_mat, np.inf)

        row_ind, col_ind = linear_sum_assignment(assign_mat)
        return float(assign_mat[row_ind, col_ind].sum())

    def h_assignment(state):
        U = state[unvisited_var]
        if not U: return 0.0
        try:
            U_tuple = tuple(sorted(list(U)))
            return _cached_assignment_calc(U_tuple)
        except Exception as e:
            print(f"Error in h_assignment: {e}")
            return 0.0

    # --- Eigenvalue Bound ---
    @lru_cache(maxsize=100000)
    def _cached_eigen_calc(unvisited_tuple):
        nodes = [0] + sorted(list(unvisited_tuple))
        N_sub = len(nodes)
        if N_sub < 2: return 0.0

        D_sub = cost_matrix[np.ix_(nodes, nodes)]
        one = np.ones((N_sub, 1))
        P = np.eye(N_sub) - (one @ one.T) / N_sub
        M = -P @ D_sub @ P
        M = (M + M.T) / 2

        try:
            eigvals = np.flip(eigh(M)[0])
        except np.linalg.LinAlgError:
            return 0.0

        eigvals = eigvals[np.abs(eigvals) > 1e-9]
        coeffs = np.array([1 - np.cos(2 * np.pi * k / N_sub) for k in range(1, N_sub)])

        phi = 0.0
        if N_sub > 1:
            if N_sub % 2 == 1:
                num_terms = (N_sub - 1) // 2
                if 2 * num_terms <= len(eigvals) and num_terms <= len(coeffs):
                    phi = sum(coeffs[k-1] * (eigvals[2*k - 2] + eigvals[2*k - 1]) for k in range(1, num_terms + 1))
            else:
                num_sum_terms = N_sub // 2 - 1
                if 2 * num_sum_terms < len(eigvals) and num_sum_terms <= len(coeffs):
                    phi = sum(coeffs[k-1] * (eigvals[2*k - 2] + eigvals[2*k - 1]) for k in range(1, num_sum_terms + 1))
                    if N_sub-2 < len(eigvals):
                        phi += 2 * eigvals[N_sub - 2]
                elif N_sub > 1 and N_sub-2 < len(eigvals):
                    phi = 2 * eigvals[N_sub - 2]
        return float(phi)

    def h_eigen(state):
        U = state[unvisited_var]
        try:
            U_tuple = tuple(sorted(list(U)))
            return _cached_eigen_calc(U_tuple)
        except Exception as e:
            print(f"Error in h_eigen: {e}")
            return 0.0

    # Return valid registry
    return automatic_creation_of_dual_bounds_registry(locals())

