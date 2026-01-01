import sys
import os
import numpy as np
import vrplib
import modified_didppy as m_dp


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
    # Dual Bounds
    # =========================================================

    # --- Pre-computation of Min Edge Tables ---
    # min_from[i]: The cheapest cost to LEAVE node i
    min_from_val = [min(travel_cost[i][k] for k in range(num_locations) if k != i) for i in range(num_locations)]
    min_from = model.add_float_table(min_from_val)

    # min_to[j]: The cheapest cost to ENTER node j
    min_to_val = [min(travel_cost[k][j] for k in range(num_locations) if k != j) for j in range(num_locations)]
    min_to = model.add_float_table(min_to_val)

    # --- Dual Bound 1: Minimum Outgoing Edges ---
    # Logic: 
    # 1. We must leave every customer that is currently unvisited.
    # 2. Every vehicle that is currently NOT at the depot must leave its current location.
    lb_outgoing = min_from[unvisited_locations]  # Sum of min_from for all unvisited nodes
    for v in range(num_vehicles):
        # If vehicle v is at a customer (location != 0), it must leave that customer eventually.
        # We add the min cost to leave its current location.
        lb_outgoing += (vehicle_locations[v] != 0).if_then_else(min_from[vehicle_locations[v]], 0.0)
    model.add_dual_bound(lb_outgoing)

    # --- Dual Bound 2: Minimum Incoming Edges ---
    # Logic:
    # 1. We must enter every customer that is currently unvisited.
    # 2. Every vehicle that is currently NOT at the depot must eventually return (enter) the depot.
    lb_incoming = min_to[unvisited_locations] # Sum of min_to for all unvisited nodes
    for v in range(num_vehicles):
        # If vehicle v is out working (location != 0), it must return to depot (enter node 0).
        lb_incoming += (vehicle_locations[v] != 0).if_then_else(min_to[0], 0.0)
    model.add_dual_bound(lb_incoming)

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
