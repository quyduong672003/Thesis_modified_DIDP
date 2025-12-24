import sys
import os
import numpy as np
import modified_didppy as m_dp
from ortools.linear_solver import pywraplp
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment
from numpy.linalg import eigh
from functools import lru_cache

# --- GLOBALS ---
current_num_locations = 0
current_travel_cost = []
current_avail_time = []
current_due_date = []

# --- READER ---
def read_tsptw_format(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    all_tokens = []
    for line in lines:
        all_tokens.extend(line.strip().split())
    iterator = iter(all_tokens)

    try:
        n = int(next(iterator))
        c = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(float(next(iterator)))
            c.append(row)
        avail = []
        due = []
        for i in range(n):
            avail.append(float(next(iterator)))
            due.append(float(next(iterator)))
        return n, c, avail, due
    except StopIteration:
        return 0, [], [], []

# --- MODEL DEFINITION (EXACT USER SPECIFICATION) ---
def creation_of_didp_model_function():
    # Expects global variables: num_locations, dist_matrix, time_windows
    num_locations = current_num_locations
    travel_cost = current_travel_cost
    avail_time = current_avail_time
    due_date = current_due_date

    # 1. Setup Model
    model = m_dp.Model(float_cost=True)
    customer = model.add_object_type(number=num_locations)

    # 2. State Variables
    # unvisited: Set of customers to visit (excluding depot 0)
    unvisited = model.add_set_var(object_type=customer, target=list(range(1, num_locations)))
    # location: Current node
    location = model.add_element_var(object_type=customer, target=0)
    # time: Current cumulative time (resource)
    curr_time = model.add_float_resource_var(target=0.0, less_is_better=True)

    # 3. Data Tables & Helpers
    travel_time_table = model.add_float_table(travel_cost)

    # 4. Transitions: Visit Customer j
    for j in range(1, num_locations):
        visit = m_dp.Transition(
            name="visit {}".format(j),
            cost=travel_time_table[location, j] + m_dp.FloatExpr.state_cost(),
            preconditions=[
                unvisited.contains(j),
                # Feasibility check: Must arrive at j by its Due Date
                # Note: We can arrive early and wait, so we check if arrival <= due_date
                curr_time + travel_time_table[location, j] <= due_date[j]
            ],
            effects=[
                (unvisited, unvisited.remove(j)),
                (location, j),
                # Time update: max(arrival_time, ready_time)
                # arrival_time = curr_time + travel_time
                (curr_time, m_dp.max(curr_time + travel_time_table[location, j], avail_time[j])),
            ],
        )
        model.add_transition(visit)

    # 5. Transition: Return to Depot (0)
    return_to_depot = m_dp.Transition(
        name="return",
        cost=travel_time_table[location, 0] + m_dp.FloatExpr.state_cost(),
        effects=[
            (location, 0),
            (curr_time, curr_time + travel_time_table[location, 0]),
        ],
        preconditions=[
            unvisited.is_empty(), 
            location != 0,
        ],
    )
    model.add_transition(return_to_depot)

    # 6. Base Case
    model.add_base_case([unvisited.is_empty(), location == 0])

    for j in range(1, num_locations):
        model.add_state_constr(
            ~unvisited.contains(j) | (curr_time + travel_time_table[location, j] <= due_date[j])
        )

    # 8. Bundle
    metadata = {
        "num_locations": num_locations,
        "distance_matrix": travel_cost,
        "avail_time": avail_time,
        "due_date": due_date,
        "unvisited_var": unvisited,
        "location_var": location,
        "time_var": curr_time
    }

    didp_bundle = (model, metadata)
    return didp_bundle

# ==========================================
# DUAL BOUNDS (ALL INCLUDED)
# ==========================================

# 1. Persistent 3-Index LP Relaxation
def create_persistent_lp_relaxation_3_index_dual_bounds(metadata):
    n_nodes = metadata['num_locations']
    unvisited_set_var = metadata['unvisited_var']
    location_var = metadata['location_var']
    dist_matrix = metadata['distance_matrix']

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return lambda state: 0.0

    x = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j: x[(i, j)] = solver.NumVar(0, 1, f'x_{i}_{j}')

    u = {i: solver.NumVar(0, n_nodes, f'u_{i}') for i in range(n_nodes)}

    cons_out, cons_in = {}, {}
    for i in range(n_nodes):
        c_out = solver.Constraint(0, 0, f'deg_out_{i}')
        c_in = solver.Constraint(0, 0, f'deg_in_{i}')
        for j in range(n_nodes):
            if i != j: 
                c_out.SetCoefficient(x[(i, j)], 1)
                c_in.SetCoefficient(x[(j, i)], 1)
        cons_out[i] = c_out
        cons_in[i] = c_in

    infinity = solver.infinity()
    for i in range(n_nodes):
        if i == 0: continue
        for j in range(n_nodes):
            if j == 0 or i == j: continue
            c_mtz = solver.Constraint(-infinity, n_nodes - 1, f'mtz_{i}_{j}')
            c_mtz.SetCoefficient(u[i], 1)
            c_mtz.SetCoefficient(u[j], -1)
            c_mtz.SetCoefficient(x[(i, j)], n_nodes)

    objective = solver.Objective()
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j: objective.SetCoefficient(x[(i, j)], dist_matrix[i][j])
    objective.SetMinimization()

    @lru_cache(maxsize=10000)
    def _solve_3idx(active_tuple):
        current_node = active_tuple[0]
        active_set = set(active_tuple[1:])
        active_set.add(current_node); active_set.add(0) 
        max_u = len(active_set)
        for i in range(n_nodes):
            if i in active_set:
                if i == current_node:
                    cons_out[i].SetBounds(1, 1); cons_in[i].SetBounds(0, 0); u[i].SetBounds(0, 0)
                elif i == 0:
                    cons_out[i].SetBounds(0, 0); cons_in[i].SetBounds(1, 1); u[i].SetBounds(0, max_u)
                else:
                    cons_out[i].SetBounds(1, 1); cons_in[i].SetBounds(1, 1); u[i].SetBounds(0, max_u)
            else:
                cons_out[i].SetBounds(0, 0); cons_in[i].SetBounds(0, 0); u[i].SetBounds(0, 0)
        solver.SetTimeLimit(100)
        if solver.Solve() in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return float(objective.Value())
        return 0.0

    def h_lp_relaxation_3_idx(state):
        unvisited = state[unvisited_set_var]
        current_node = state[location_var]
        if not unvisited and current_node == 0: return 0.0
        key = (current_node,) + tuple(sorted(list(unvisited)))
        return _solve_3idx(key)

    return h_lp_relaxation_3_idx

# 2. Persistent 2-Index LP Relaxation
def create_persistent_lp_relaxation_2_index_dual_bounds(metadata):
    n_nodes = metadata['num_locations']
    dist_matrix = metadata['distance_matrix']
    unvisited_var = metadata['unvisited_var']
    location_var = metadata['location_var']

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return lambda state: 0.0
    infinity = solver.infinity()

    x = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j: x[(i, j)] = solver.NumVar(0, 1, f'x_{i}_{j}')
    u = {i: solver.NumVar(0, n_nodes, f'u_{i}') for i in range(1, n_nodes)}

    cons_deg_out, cons_deg_in = {}, {}
    for i in range(n_nodes):
        c_out = solver.Constraint(0, 0, f'deg_out_{i}')
        c_in = solver.Constraint(0, 0, f'deg_in_{i}')
        for j in range(n_nodes):
            if i != j: 
                c_out.SetCoefficient(x[(i, j)], 1)
                c_in.SetCoefficient(x[(j, i)], 1)
        cons_deg_out[i] = c_out
        cons_deg_in[i] = c_in

    for i in range(1, n_nodes):
        for j in range(1, n_nodes):
            if i != j:
                c = solver.Constraint(-infinity, n_nodes - 1, f'mtz_{i}_{j}')
                c.SetCoefficient(u[j], 1); c.SetCoefficient(u[i], -1); c.SetCoefficient(x[(i, j)], n_nodes)

    objective = solver.Objective()
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j: objective.SetCoefficient(x[(i, j)], dist_matrix[i][j])
    objective.SetMinimization()

    @lru_cache(maxsize=10000)
    def _solve_2idx(active_tuple):
        current_loc = active_tuple[0]
        active_set = set(active_tuple[1:])
        active_set.add(current_loc); active_set.add(0)
        max_u = len(active_set)
        for i in range(n_nodes):
            if i in active_set:
                cons_deg_out[i].SetBounds(1, 1); cons_deg_in[i].SetBounds(1, 1)
                if i > 0:
                    if i == current_loc: u[i].SetBounds(0, 0)
                    else: u[i].SetBounds(0, max_u)
            else:
                cons_deg_out[i].SetBounds(0, 0); cons_deg_in[i].SetBounds(0, 0)
                if i > 0: u[i].SetBounds(0, 0)
        solver.SetTimeLimit(100)
        if solver.Solve() in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return float(objective.Value())
        return 0.0

    def h_lp_relaxation_2_idx(state):
        unvisited = state[unvisited_var]
        current_loc = state[location_var]
        if not unvisited and current_loc == 0: return 0.0
        key = (current_loc,) + tuple(sorted(list(unvisited)))
        return _solve_2idx(key)

    return h_lp_relaxation_2_idx

# 3. Persistent TSPTW Relaxed Model (Big-M)
def create_persistent_tsptw_lp_bound(metadata):
    num_locations = metadata['num_locations']
    dist_matrix = metadata['distance_matrix']
    avail_time = metadata['avail_time']
    due_date = metadata['due_date']
    unvisited_var = metadata['unvisited_var']
    location_var = metadata['location_var']
    time_var = metadata['time_var']

    big_m = {}
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j: big_m[(i, j)] = max(due_date[i] + dist_matrix[i][j] - avail_time[j], 0.0)

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return lambda state: 0.0
    infinity = solver.infinity()

    x = {}; w = {}
    for i in range(num_locations):
        w[i] = solver.NumVar(avail_time[i], due_date[i], f'w_{i}')
        for j in range(num_locations):
            if i != j: x[(i, j)] = solver.NumVar(0, 1, f'x_{i}_{j}')

    cons_flow = {}
    for i in range(num_locations):
        c = solver.Constraint(0, 0, f'flow_{i}')
        for j in range(num_locations):
            if i != j:
                c.SetCoefficient(x[(i, j)], 1)
                c.SetCoefficient(x[(j, i)], -1)
        cons_flow[i] = c

    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                M = big_m[(i, j)]
                if M > 0:
                    c = solver.Constraint(-infinity, M - dist_matrix[i][j], f'time_{i}_{j}')
                    c.SetCoefficient(w[i], 1); c.SetCoefficient(w[j], -1); c.SetCoefficient(x[(i, j)], M)

    objective = solver.Objective()
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j: objective.SetCoefficient(x[(i, j)], dist_matrix[i][j])
    objective.SetMinimization()

    @lru_cache(maxsize=10000)
    def _solve_tsptw(input_tuple):
        current_node = input_tuple[0]
        current_time = input_tuple[1]
        active_set = set(input_tuple[2:]); active_set.add(current_node); active_set.add(0)

        for i in range(num_locations):
            if i in active_set:
                if i == current_node:
                    cons_flow[i].SetBounds(1, 1)
                    lb = max(avail_time[i], current_time)
                    ub = due_date[i]
                    if lb > ub: return float('inf')
                    w[i].SetBounds(lb, ub)
                elif i == 0:
                    cons_flow[i].SetBounds(-1, -1)
                    w[i].SetBounds(avail_time[i], due_date[i])
                else:
                    cons_flow[i].SetBounds(0, 0)
                    w[i].SetBounds(avail_time[i], due_date[i])

                for j in range(num_locations):
                    if i != j:
                        if j in active_set: x[(i, j)].SetBounds(0, 1)
                        else: x[(i, j)].SetBounds(0, 0)
            else:
                cons_flow[i].SetBounds(0, 0)
                w[i].SetBounds(avail_time[i], due_date[i])
                for j in range(num_locations):
                    if i != j: x[(i, j)].SetBounds(0, 0)

        solver.SetTimeLimit(100)
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL: return float(objective.Value())
        elif status == pywraplp.Solver.INFEASIBLE: return float('inf')
        return 0.0

    def h_lp_tsptw(state):
        unvisited = state[unvisited_var]
        current_node = state[location_var]
        current_time = state[time_var]
        if not unvisited and current_node == 0: return 0.0
        return _solve_tsptw((current_node, round(current_time, 2)) + tuple(sorted(list(unvisited))))

    return h_lp_tsptw

# --- REGISTRY ---
def dual_bound_expression_function(didp_bundle):
    model, metadata = didp_bundle
    cost_matrix = np.array(metadata['distance_matrix'])
    min_outgoing_arr = np.min(cost_matrix + np.diag([np.inf]*len(cost_matrix)), axis=1)
    min_incoming_arr = np.min(cost_matrix + np.diag([np.inf]*len(cost_matrix)), axis=0)

    unvisited_var = metadata['unvisited_var']
    location_var = metadata['location_var']

    # LP Bounds
    h_lp_3idx = create_persistent_lp_relaxation_3_index_dual_bounds(metadata)
    h_lp_2idx = create_persistent_lp_relaxation_2_index_dual_bounds(metadata)
    h_lp_tsptw = create_persistent_tsptw_lp_bound(metadata)

    # Combinatorial Bounds
    @lru_cache(maxsize=100000)
    def _calc_degree(active_tuple):
        nodes = list(active_tuple)
        sub_mat = cost_matrix[np.ix_(nodes, nodes)] + np.diag([np.inf]*len(nodes))
        return float(0.5 * (np.sum(np.min(sub_mat, axis=0)[1:]) + np.sum(np.min(sub_mat, axis=1)[:-1])))

    def h_degree_average(state):
        U = state[unvisited_var]; curr = state[location_var]
        if not U and curr == 0: return 0.0
        active_list = [curr] + sorted(list(U))
        if 0 not in active_list: active_list.append(0)
        return _calc_degree(tuple(active_list))

    @lru_cache(maxsize=100000)
    def _calc_min_flow_static(unvisited_tuple):
        return sum(min_outgoing_arr[u] for u in unvisited_tuple), sum(min_incoming_arr[u] for u in unvisited_tuple)

    def h_global_min_flow(state):
        U = state[unvisited_var]; curr = state[location_var]
        if not U and curr == 0: return 0.0
        val_out, val_in = _calc_min_flow_static(tuple(sorted(list(U))))
        if curr != 0: val_out += min_outgoing_arr[curr]; val_in += min_incoming_arr[0]
        return float(max(val_out, val_in))

    @lru_cache(maxsize=100000)
    def _calc_mst(unvisited_tuple):
        if not unvisited_tuple: return 0.0
        nodes = [0] + list(unvisited_tuple)
        return float(minimum_spanning_tree(cost_matrix[np.ix_(nodes, nodes)]).sum())

    def h_mst(state):
        return _calc_mst(tuple(sorted(list(state[unvisited_var]))))

    @lru_cache(maxsize=100000)
    def _calc_1tree(unvisited_tuple):
        subset = list(unvisited_tuple)
        depot_edges = sorted(cost_matrix[0, subset])
        if len(subset) > 1: mst_val = minimum_spanning_tree(cost_matrix[np.ix_(subset, subset)]).sum()
        else: mst_val = 0.0
        return float(mst_val + depot_edges[0] + (depot_edges[1] if len(depot_edges)>1 else 0.0))

    def h_1tree(state):
        U = state[unvisited_var]
        if not U: return 0.0
        return _calc_1tree(tuple(sorted(list(U))))

    @lru_cache(maxsize=100000)
    def _calc_assignment(unvisited_tuple):
        nodes = [0] + list(unvisited_tuple)
        sub_mat = cost_matrix[np.ix_(nodes, nodes)] + np.diag([np.inf]*len(nodes))
        r, c = linear_sum_assignment(sub_mat)
        return float(sub_mat[r, c].sum())

    def h_assignment(state):
        U = state[unvisited_var]
        if not U: return 0.0
        return _calc_assignment(tuple(sorted(list(U))))

    @lru_cache(maxsize=100000)
    def _calc_eigen(unvisited_tuple):
        nodes = [0] + list(unvisited_tuple); N = len(nodes)
        if N < 2: return 0.0
        D_sub = cost_matrix[np.ix_(nodes, nodes)]
        one = np.ones((N, 1)); P = np.eye(N) - (one @ one.T) / N
        try: eigvals = np.flip(eigh((-P @ D_sub @ P + (-P @ D_sub @ P).T)/2)[0])
        except: return 0.0
        eigvals = eigvals[np.abs(eigvals) > 1e-9]
        coeffs = np.array([1 - np.cos(2 * np.pi * k / N) for k in range(1, N)])
        phi = 0.0
        if N > 1:
            num = (N - 1) // 2 if N % 2 == 1 else N // 2 - 1
            if 2*num <= len(eigvals) and num <= len(coeffs):
                phi = sum(coeffs[k-1] * (eigvals[2*k-2] + eigvals[2*k-1]) for k in range(1, num+1))
            if N % 2 == 0 and N > 1 and N-2 < len(eigvals): phi += 2 * eigvals[N-2]
        return float(phi)

    def h_eigen(state):
        U = state[unvisited_var]
        if not U: return 0.0
        return _calc_eigen(tuple(sorted(list(U))))

    from evolutionary_algorithm_lib.utils import automatic_creation_of_dual_bounds_registry
    return automatic_creation_of_dual_bounds_registry(locals())
