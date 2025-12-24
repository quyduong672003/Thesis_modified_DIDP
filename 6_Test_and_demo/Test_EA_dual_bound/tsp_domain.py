import sys
import os

# ==========================================
# 1. CRITICAL PATH SETUP
# ==========================================
PROJECT_ROOT = r"C:\Users\ACER\Desktop\Code\0_Thesis_implementation\2_DIDP_custom_search_guidance_local\Thesis_modified_DIDP"
LIB_PATH = os.path.join(PROJECT_ROOT, "Evolutionary_algorithm")

if os.path.exists(PROJECT_ROOT) and PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

if os.path.exists(LIB_PATH) and LIB_PATH not in sys.path:
    sys.path.append(LIB_PATH)

# ==========================================
# 2. IMPORTS
# ==========================================
import numpy as np
import modified_didppy as m_dp
from ortools.linear_solver import pywraplp
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment
from numpy.linalg import eigh
from functools import lru_cache

try:
    from evolutionary_algorithm_lib.utils import automatic_creation_of_dual_bounds_registry
except ImportError:
    pass 

# ==========================================
# 3. GLOBALS & READER
# ==========================================
current_num_locations = 0
current_travel_cost = []

def read_tsp_cappart_format(file_path):
    with open(file_path, 'r') as f:
        values = f.read().split()
    iterator = iter(values)
    try:
        n = int(next(iterator))
        c = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(int(float(next(iterator))))
            c.append(row)
        return n, c
    except StopIteration:
        return 0, []

# ==========================================
# 4. MODEL DEFINITION
# ==========================================
def creation_of_didp_model_function():
    n = current_num_locations
    c = current_travel_cost

    model = m_dp.Model(maximize=False, float_cost=True)
    customer = model.add_object_type(number=n)
    unvisited = model.add_set_var(object_type=customer, target=list(range(1, n)))
    location = model.add_element_var(object_type=customer, target=0)
    travel_time = model.add_float_table(c)

    for j in range(1, n):
        visit = m_dp.Transition(
            name=f"visit {j}",
            cost=travel_time[location, j] + m_dp.FloatExpr.state_cost(),
            preconditions=[unvisited.contains(j)],
            effects=[(unvisited, unvisited.remove(j)), (location, j)],
        )
        model.add_transition(visit)

    return_to_depot = m_dp.Transition(
        name="return",
        cost=travel_time[location, 0] + m_dp.FloatExpr.state_cost(),
        effects=[(location, 0)],
        preconditions=[unvisited.is_empty(), location != 0],
    )
    model.add_transition(return_to_depot)
    model.add_base_case([unvisited.is_empty(), location == 0])

    metadata = {
        "num_nodes": n,
        "distance_matrix": c,
        "unvisited_var": unvisited,
        "location_var": location,
    }
    return model, metadata

# ==========================================
# 5. DUAL BOUNDS
# ==========================================

def create_persistent_lp_relaxation_3_index_dual_bounds(metadata):
    n_nodes = metadata['num_nodes']
    dist_matrix = metadata['distance_matrix']
    unvisited_var = metadata['unvisited_var']
    location_var = metadata['location_var']

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return lambda state: 0.0
    infinity = solver.infinity()

    x = {}
    u = {}
    for i in range(n_nodes):
        u[i] = solver.NumVar(0, n_nodes, f'u_{i}')
        for j in range(n_nodes):
            if i != j: x[(i, j)] = solver.NumVar(0, 1, f'x_{i}_{j}')

    cons_out = {}
    cons_in = {}
    for i in range(n_nodes):
        c_out = solver.Constraint(0, 0, f'out_{i}')
        c_in = solver.Constraint(0, 0, f'in_{i}')
        for j in range(n_nodes):
            if i != j:
                c_out.SetCoefficient(x[(i, j)], 1)
                c_in.SetCoefficient(x[(j, i)], 1)
        cons_out[i] = c_out
        cons_in[i] = c_in

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                c = solver.Constraint(-infinity, n_nodes - 1, f'mtz_{i}_{j}')
                c.SetCoefficient(u[i], 1)
                c.SetCoefficient(u[j], -1)
                c.SetCoefficient(x[(i, j)], n_nodes)

    objective = solver.Objective()
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j: objective.SetCoefficient(x[(i, j)], dist_matrix[i][j])
    objective.SetMinimization()

    @lru_cache(maxsize=10000)
    def _solve_3idx(active_tuple):
        current_node = active_tuple[0]
        active_set = set(active_tuple[1:])
        active_set.add(current_node)
        active_set.add(0) 
        max_u = len(active_set) 

        for i in range(n_nodes):
            if i in active_set:
                if i == current_node:
                    cons_out[i].SetBounds(1, 1); cons_in[i].SetBounds(0, 0); u[i].SetBounds(0, 0)
                elif i == 0:
                    cons_out[i].SetBounds(0, 0); cons_in[i].SetBounds(1, 1); u[i].SetBounds(1, max_u)
                else:
                    cons_out[i].SetBounds(1, 1); cons_in[i].SetBounds(1, 1); u[i].SetBounds(1, max_u)
            else:
                cons_out[i].SetBounds(0, 0); cons_in[i].SetBounds(0, 0); u[i].SetBounds(0, 0)

        solver.SetTimeLimit(100)
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            return float(objective.Value())
        return 0.0

    def h_lp_relaxation_3_idx(state):
        unvisited = state[unvisited_var]
        curr = state[location_var]
        if not unvisited and curr == 0: return 0.0
        key = (curr,) + tuple(sorted(list(unvisited)))
        return _solve_3idx(key)

    return h_lp_relaxation_3_idx

def create_persistent_lp_relaxation_2_index_dual_bounds(metadata):
    n_nodes = metadata['num_nodes']
    dist_matrix = metadata['distance_matrix']
    unvisited_var = metadata['unvisited_var']
    location_var = metadata['location_var']

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return lambda state: 0.0

    x = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j: x[(i, j)] = solver.NumVar(0, 1, f'x_{i}_{j}')

    cons_out = {}
    cons_in = {}
    for i in range(n_nodes):
        c_out = solver.Constraint(0, 0, f'out_{i}')
        c_in = solver.Constraint(0, 0, f'in_{i}')
        for j in range(n_nodes):
            if i != j:
                c_out.SetCoefficient(x[(i, j)], 1)
                c_in.SetCoefficient(x[(j, i)], 1)
        cons_out[i] = c_out
        cons_in[i] = c_in

    objective = solver.Objective()
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j: objective.SetCoefficient(x[(i, j)], dist_matrix[i][j])
    objective.SetMinimization()

    @lru_cache(maxsize=10000)
    def _solve_2idx(active_tuple):
        current_node = active_tuple[0]
        active_set = set(active_tuple[1:])
        active_set.add(current_node)
        active_set.add(0)

        for i in range(n_nodes):
            if i in active_set:
                if i == current_node:
                    cons_out[i].SetBounds(1, 1); cons_in[i].SetBounds(0, 0)
                elif i == 0:
                    cons_out[i].SetBounds(0, 0); cons_in[i].SetBounds(1, 1)
                else:
                    cons_out[i].SetBounds(1, 1); cons_in[i].SetBounds(1, 1)
            else:
                cons_out[i].SetBounds(0, 0); cons_in[i].SetBounds(0, 0)

        solver.SetTimeLimit(100)
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            return float(objective.Value())
        return 0.0

    def h_lp_relaxation_2_idx(state):
        unvisited = state[unvisited_var]
        curr = state[location_var]
        if not unvisited and curr == 0: return 0.0
        key = (curr,) + tuple(sorted(list(unvisited)))
        return _solve_2idx(key)

    return h_lp_relaxation_2_idx

def dual_bound_expression_function(didp_bundle):
    model, metadata = didp_bundle
    unvisited_var = metadata['unvisited_var']
    location_var = metadata['location_var']
    cost_matrix = np.array(metadata['distance_matrix'])

    masked_cost = cost_matrix.astype(float).copy()
    np.fill_diagonal(masked_cost, np.inf)
    min_outgoing_arr = np.min(masked_cost, axis=1)
    min_incoming_arr = np.min(masked_cost, axis=0)

    h_lp_relaxation_3_idx = create_persistent_lp_relaxation_3_index_dual_bounds(metadata)
    h_lp_relaxation_2_idx = create_persistent_lp_relaxation_2_index_dual_bounds(metadata)

    @lru_cache(maxsize=100000)
    def _calc_degree(active_tuple):
        nodes = list(active_tuple)
        sub_mat = cost_matrix[np.ix_(nodes, nodes)].astype(float)
        np.fill_diagonal(sub_mat, np.inf)
        mins_in = np.min(sub_mat, axis=0) 
        mins_out = np.min(sub_mat, axis=1)
        sum_in = np.sum(mins_in[1:])
        sum_out = np.sum(mins_out[:-1])
        return float(0.5 * (sum_in + sum_out))

    def h_degree_average(state):
        U = state[unvisited_var]
        curr = state[location_var]
        if not U and curr == 0: return 0.0
        active_list = [curr] + sorted(list(U))
        if 0 not in active_list: active_list.append(0)
        return _calc_degree(tuple(active_list))

    @lru_cache(maxsize=100000)
    def _calc_min_flow_static(unvisited_tuple):
        val_out = sum(min_outgoing_arr[u] for u in unvisited_tuple)
        val_in = sum(min_incoming_arr[u] for u in unvisited_tuple)
        return val_out, val_in

    def h_global_min_flow(state):
        U = state[unvisited_var]
        curr = state[location_var]
        if not U and curr == 0: return 0.0
        val_out, val_in = _calc_min_flow_static(tuple(sorted(list(U))))
        if curr != 0:
            val_out += min_outgoing_arr[curr]
            val_in += min_incoming_arr[0]
        return float(max(val_out, val_in))

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

    from evolutionary_algorithm_lib.utils import automatic_creation_of_dual_bounds_registry
    return automatic_creation_of_dual_bounds_registry(locals())
