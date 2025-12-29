import sys
import os
import ast
import json
import time
import threading

# --- CONFIGURATION ---
MEMORY_LIMIT_MB = 8000  # Adjust as needed

# --- 1. SETUP PATHS ---
PROJECT_ROOT = r"C:\Users\ACER\Desktop\Code\0_Thesis_implementation\2_DIDP_custom_search_guidance_local\Thesis_modified_DIDP"
LIB_PATH = os.path.join(PROJECT_ROOT, "Evolutionary_algorithm")

sys.path.append(os.getcwd()) # For TSP_DIDP_model_and_dual_bounds_declaration.py
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
if LIB_PATH not in sys.path: sys.path.append(LIB_PATH)

# --- 2. IMPORTS ---
try:
    import psutil
    import modified_didppy as m_dp
    import TSP_DIDP_model_and_dual_bounds_declaration
    from evolutionary_algorithm_lib import compile_chromosome_to_useable_function
except ImportError as e:
    print(json.dumps({"status": "error", "message": f"Import Error: {e}"}))
    sys.exit(1)

# --- 3. MEMORY GUARD ---
def memory_guard():
    process = psutil.Process(os.getpid())
    limit_bytes = MEMORY_LIMIT_MB * 1024 * 1024
    while True:
        try:
            if process.memory_info().rss > limit_bytes:
                sys.stderr.write(f"\n[Guard] Memory limit ({MEMORY_LIMIT_MB}MB) reached.\n")
                sys.stderr.flush()
                os._exit(1)
            time.sleep(1)
        except: break

# --- 4. MAIN PROFILING LOGIC ---
if __name__ == "__main__":
    try:
        threading.Thread(target=memory_guard, daemon=True).start()

        if len(sys.argv) < 5: raise ValueError("Args: <instance> <chrom> <dir> <limit> <node_limit>")

        instance_name = sys.argv[1]
        chromosome_str = sys.argv[2]
        data_dir = sys.argv[3]
        time_limit = float(sys.argv[4])
        node_limit = int(sys.argv[5])

        # Load Data
        file_path = os.path.join(data_dir, instance_name)
        n_val, c_val = TSP_DIDP_model_and_dual_bounds_declaration.read_tsp_cappart_format(file_path)
        TSP_DIDP_model_and_dual_bounds_declaration.current_num_locations = n_val
        TSP_DIDP_model_and_dual_bounds_declaration.current_travel_cost = c_val

        # Setup Model
        model, metadata = TSP_DIDP_model_and_dual_bounds_declaration.creation_of_didp_model_function()

        # Setup Dual Bound from Chromosome
        chromosome = ast.literal_eval(chromosome_str)
        dual_bound_funcs = TSP_DIDP_model_and_dual_bounds_declaration.dual_bound_expression_function((model, metadata))
        h_func = compile_chromosome_to_useable_function(
            {'chromosome': chromosome, 'fitness': 0},
            dual_bound_functions_dict=dual_bound_funcs,
            print_code=False
        )

        # Create Custom Solver (Manually wrapped to access search_next)
        # Using CustomDualBoundCabsPy directly if possible or creating it via m_dp
        # We need to construct the solver such that we can run search_next
        solver = m_dp.CustomDualBoundCABSv1(
            model=model,
            dual_bound_func=h_func,
            time_limit=time_limit,
            quiet=True, # We want to control output
            print_timing_stats=False
        )

        # --- PROFILING LOOP ---
        history = [] # List of (expanded_nodes, cost)
        start_time = time.time()

        while True:
            # Check Limits
            if time.time() - start_time > time_limit: break

            # Run Next Step
            solution, is_terminated = solver.search_next()

            # Record Progress
            if solution.cost is not None:
                history.append((solution.expanded, solution.cost))

            # Termination Check
            if is_terminated:
                break

            # Stop if we exceeded node limit (soft check)
            if solution.expanded >= node_limit:
                break

        # Output Result
        output = {
            "status": "success",
            "history": history,
            "duration": time.time() - start_time
        }
        print(json.dumps(output))

    except Exception as e:
        # Fallback error reporting
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
