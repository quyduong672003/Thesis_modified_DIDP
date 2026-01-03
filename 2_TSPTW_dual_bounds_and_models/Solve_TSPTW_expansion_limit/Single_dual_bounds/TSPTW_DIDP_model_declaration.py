import sys
import os
import numpy as np
import modified_didppy as m_dp

# --- GLOBALS ---
current_num_locations = 0
current_travel_cost = []
current_avail_time =[]
current_due_date =[]
# --- READER ---
def read_tsptw_data(file_path):
    """
    Reads TSPTW data from the specified file format:
    - Line 1: Number of locations (N)
    - Next N lines: Distance Matrix (N x N)
    - Next N lines: Time Windows (Ready Time, Due Date)
    - (Ignores subsequent lines, e.g., coordinates)
    """
    with open(file_path, 'r') as f:
        # Read all tokens (whitespace separated) to handle newlines flexibly
        tokens = f.read().split()

    iterator = iter(tokens)

    try:
        # 1. Number of locations
        num_locations = int(next(iterator))

        # 2. Travel Cost Matrix (N x N)
        travel_cost = []
        for _ in range(num_locations):
            row = []
            for _ in range(num_locations):
                row.append(float(next(iterator))) # Load as float
            travel_cost.append(row)

        # 3. Time Windows (N lines of: Ready_Time Due_Date)
        time_windows = []
        for _ in range(num_locations):
            ready = float(next(iterator)) # Load as float
            due = float(next(iterator))   # Load as float
            time_windows.append((ready, due))
        avail_time = [tw[0] for tw in time_windows]
        due_date = [tw[1] for tw in time_windows]
        return num_locations, travel_cost, avail_time, due_date

    except StopIteration:
        raise ValueError(f"Error reading file {file_path}: Unexpected end of file.")

def update_globals_for_tsptw(file_path):
    global current_num_locations, current_travel_cost
    global current_avail_time, current_due_date

    num_locations, travel_cost, avail_time, due_date = read_tsptw_data(file_path)

    current_num_locations = num_locations
    current_travel_cost = travel_cost
    current_avail_time = avail_time
    current_due_date = due_date
    return True

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

    # Separate time windows into lists for easy access

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

    # 7. Dual Bounds (Updated for Float Cost)
    # Min outgoing cost from unvisited nodes
    min_to = model.add_float_table(
        [min(travel_cost[k][j] for k in range(num_locations) if k != j) for j in range(num_locations)]
    )
    model.add_dual_bound(min_to[unvisited] + (location != 0).if_then_else(min_to[0], 0.0))

    # Min incoming cost to unvisited nodes
    min_from = model.add_float_table(
        [min(travel_cost[j][k] for k in range(num_locations) if k != j) for j in range(num_locations)]
    )
    model.add_dual_bound(
        min_from[unvisited] + (location != 0).if_then_else(min_from[location], 0.0)
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

    return model, metadata
