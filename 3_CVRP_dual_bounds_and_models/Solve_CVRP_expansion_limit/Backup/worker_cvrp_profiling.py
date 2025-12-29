import sys
import os
import ast
import json
import time
import threading

# --- CONFIGURATION ---
MEMORY_LIMIT_MB = 12000 # Increased to 12GB just in case

# --- 1. SETUP PATHS ---
PROJECT_ROOT = r"C:\Users\ACER\Desktop\Code\0_Thesis_implementation\2_DIDP_custom_search_guidance_local\Thesis_modified_DIDP"
LIB_PATH = os.path.join(PROJECT_ROOT, "Evolutionary_algorithm")

sys.path.append(os.getcwd()) 
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
if LIB_PATH not in sys.path: sys.path.append(LIB_PATH)

# --- 2. IMPORTS ---
try:
    import psutil
    import modified_didppy as m_dp
    import v1_CVRP_DIDP_model_and_dual_bound_declaration as domain 
    from evolutionary_algorithm_lib import compile_chromosome_to_useable_function
except ImportError as e:
    print(json.dumps({"status": "error", "message": f"Import Error: {e}"}))
    sys.exit(1)

def memory_guard():
    process = psutil.Process(os.getpid())
    limit_bytes = MEMORY_LIMIT_MB * 1024 * 1024
    while True:
        try:
            if process.memory_info().rss > limit_bytes:
                os._exit(1)
            time.sleep(1)
        except: break

if __name__ == "__main__":
    try:
        threading.Thread(target=memory_guard, daemon=True).start()

        if len(sys.argv) < 6: 
            raise ValueError("Args mismatch")

        instance_name = sys.argv[1]
        chromosome_str = sys.argv[2]
        data_dir = sys.argv[3]
        time_limit = float(sys.argv[4])
        node_limit = int(sys.argv[5])

        # 1. Load Data
        file_path = os.path.join(data_dir, instance_name)
        if not os.path.exists(file_path):
             file_path_vrp = file_path + ".vrp"
             if os.path.exists(file_path_vrp): file_path = file_path_vrp

        if not domain.update_globals_for_cvrp(file_path):
            raise ValueError(f"Failed to load instance: {file_path}")

        # 2. DEBUG: Print Problem Stats to Stderr (Manager will capture this)
        debug_info = (
            f"DEBUG: Loaded {instance_name} | "
            f"Nodes: {domain.current_num_locations}, "
            f"Capacity: {domain.current_capacity}, "
            f"Vehicles: {domain.current_num_vehicles}, "
            f"Total Demand: {sum(domain.current_cust_demands)}"
        )
        sys.stderr.write(debug_info + "\n")

        # 3. Create Model
        model, metadata = domain.creation_of_didp_model_function()

        # 4. Prepare Dual Bound
        chromosome = ast.literal_eval(chromosome_str)
        dual_bound_funcs = domain.dual_bound_expression_function((model, metadata))
        h_func = compile_chromosome_to_useable_function(
            {'chromosome': chromosome, 'fitness': 0},
            dual_bound_functions_dict=dual_bound_funcs,
            print_code=False
        )

        # 5. Create Solver (Set quiet=False to see solver logs in debug mode)
        solver = m_dp.CustomDualBoundCABSv1(
            model=model,
            dual_bound_func=h_func,
            time_limit=time_limit,
            initial_beam_size = 1,
            max_beam_size=1024,
            quiet=False, # Changed to False to see infeasibility message
            print_timing_stats=False
        )

        history = [] 
        start_time = time.time()

        while True:
            if time.time() - start_time > time_limit: break

            solution, is_terminated = solver.search_next()

            if solution.cost is not None:
                history.append((solution.expanded, solution.cost))

            if is_terminated: break
            if solution.expanded >= node_limit: break

        output = {
            "status": "success",
            "history": history,
            "duration": time.time() - start_time
        }
        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
