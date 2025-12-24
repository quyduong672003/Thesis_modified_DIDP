import sys
import os
import ast
import json
import time
import threading

# --- CONFIGURATION ---
MEMORY_LIMIT_MB = 7990  # Limit worker to 4 GB RAM. Adjust based on your PC.

# --- 1. SETUP PATHS ---
PROJECT_ROOT = r"C:\Users\ACER\Desktop\Code\0_Thesis_implementation\2_DIDP_custom_search_guidance_local\Thesis_modified_DIDP"
LIB_PATH = os.path.join(PROJECT_ROOT, "Evolutionary_algorithm")

if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
if LIB_PATH not in sys.path: sys.path.append(LIB_PATH)

# --- 2. IMPORTS ---
try:
    import psutil # For memory monitoring
    import modified_didppy as m_dp
    from evolutionary_algorithm_lib import combining_modified_didppy_solver_with_chromosome
    import tsp_domain 
except ImportError as e:
    print(json.dumps({"status": "error", "message": f"Import Error: {e}"}))
    sys.exit(1)

# --- 3. MEMORY GUARD (Background Thread) ---
def memory_guard():
    """Checks memory usage every second and kills process if limit exceeded."""
    process = psutil.Process(os.getpid())
    limit_bytes = MEMORY_LIMIT_MB * 1024 * 1024

    while True:
        try:
            mem_usage = process.memory_info().rss # Resident Set Size (Physical Memory)
            if mem_usage > limit_bytes:
                # Print a special error message that the logs will catch
                sys.stderr.write(f"\n[Guard] Memory limit ({MEMORY_LIMIT_MB}MB) reached. Killing worker.\n")
                sys.stderr.flush()
                os._exit(1) # Force kill immediately (simulates a crash)
            time.sleep(1)
        except:
            break

# --- 4. MAIN LOGIC ---
if __name__ == "__main__":
    try:
        # Start Memory Guard
        guard_thread = threading.Thread(target=memory_guard, daemon=True)
        guard_thread.start()

        if len(sys.argv) < 5:
            raise ValueError("Not enough arguments")

        instance_name = sys.argv[1]
        chromosome_str = sys.argv[2]
        data_dir = sys.argv[3]
        time_limit = float(sys.argv[4])

        file_path = os.path.join(data_dir, instance_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        n_val, c_val = tsp_domain.read_tsp_cappart_format(file_path)
        tsp_domain.current_num_locations = n_val
        tsp_domain.current_travel_cost = c_val

        start_time = time.time()
        result = combining_modified_didppy_solver_with_chromosome(
            chromosome=ast.literal_eval(chromosome_str),
            didp_model_registry=tsp_domain.creation_of_didp_model_function,
            dual_bound_expression_function=tsp_domain.dual_bound_expression_function,
            solver_time_limit=time_limit,
            output_other_result=True,
            print_timing_stats=True,
            solver_quite=False 
        )
        duration = time.time() - start_time

        if result is None:
            output = {"status": "timeout", "cost": float('inf'), "duration": duration}
        elif isinstance(result, (float, int)): 
             output = {"status": "success", "cost": float(result), "duration": duration}
        else: 
            cost, is_opt, gen, exp, stats = result
            output = {
                "status": "success",
                "cost": cost,
                "is_optimal": is_opt,
                "generated": gen,
                "expanded": exp,
                "duration": duration,
                "stats": stats
            }

        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
