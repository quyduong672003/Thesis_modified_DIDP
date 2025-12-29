import sys
import os
import numpy as np
import modified_didppy as m_dp

# --- GLOBALS ---
current_num_locations = 0
current_travel_cost = []

# --- READER ---
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

# --- MODEL DEFINITION ---
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

    # --- DUAL BOUNDS ---
    min_to_val = [min(c[k][j] for k in range(n) if k != j) for j in range(n)]
    min_to = model.add_float_table(min_to_val)
    model.add_dual_bound(min_to[unvisited] + (location != 0).if_then_else(min_to[0], 0))

    min_from_val = [min(c[j][k] for k in range(n) if k != j) for j in range(n)]
    min_from = model.add_float_table(min_from_val)
    model.add_dual_bound(
        min_from[unvisited] + (location != 0).if_then_else(min_from[location], 0)
    )

    metadata = {
        "num_nodes": n,
        "distance_matrix": c,
        "unvisited_var": unvisited,
        "location_var": location,
    }
    return model, metadata
