import sys
import os
import re
import numpy as np
import vrplib
import modified_didppy as m_dp

# ==========================================
# 2. Data Reading & Model Definition
# ==========================================

# Global variables to store current instance data (used by model creator)
current_num_locations = 0
current_num_vehicles = 0
current_capacity = 0
current_cust_demands = []
current_travel_cost = []

def read_formatted_data(file_path):
    """
    Reads a VRP file using vrplib and updates the global variables.
    """
    global current_num_locations, current_num_vehicles, current_capacity
    global current_cust_demands, current_travel_cost

    # Read instance
    # FIX: compute_edge_weights=True handles instances with coordinates but no edge weights
    instance = vrplib.read_instance(file_path, compute_edge_weights=True)

    # Extract Capacity & Dimensions
    current_capacity = float(instance['capacity'])
    current_num_locations = int(instance['dimension'])

    # Extract Demands (Includes depot at index 0 with demand 0)
    current_cust_demands = [float(x) for x in instance['demand']]

    # Extract/Compute Edge Weights
    # vrplib usually computes euclidean distance automatically in 'edge_weight'
    current_travel_cost = [[float(x) for x in row] for row in instance['edge_weight']]

    # Try to determine number of vehicles from comment or filename
    match_comment = re.search(r"No of trucks:\s*(\d+)", str(instance.get('comment', '')))
    match_name = re.search(r"-k(\d+)", os.path.basename(file_path))

    if match_comment:
        current_num_vehicles = int(match_comment.group(1))
    elif match_name:
        current_num_vehicles = int(match_name.group(1))
    else:
        # Fallback: Estimate or set a high number (e.g., N)
        current_num_vehicles = current_num_locations 

    return current_num_locations, current_num_vehicles, current_travel_cost, current_capacity, current_cust_demands

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

    # Min outgoing edge
    min_to = model.add_float_table(
        [min(distance_list[k][j] for k in range(n) if k != j) for j in range(n)]
    )
    model.add_dual_bound(min_to[unvisited_var] + (location_var != 0).if_then_else(min_to[0], 0))

    # Min incoming edge
    min_from = model.add_float_table(
        [min(distance_list[j][k] for k in range(n) if k != j) for j in range(n)]
    )
    model.add_dual_bound(
        min_from[unvisited_var] + (location_var != 0).if_then_else(min_from[location_var], 0)
    )
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
