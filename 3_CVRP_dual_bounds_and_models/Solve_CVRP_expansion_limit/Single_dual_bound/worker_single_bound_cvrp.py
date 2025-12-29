import sys
import os
import json
import time
import threading

# --- CONFIGURATION ---
MEMORY_LIMIT_MB = 12000

# --- 1. SETUP PATHS ---
PROJECT_ROOT = r"C:\Users\ACER\Desktop\Code\0_Thesis_implementation\2_DIDP_custom_search_guidance_local\Thesis_modified_DIDP"
sys.path.append(os.getcwd()) 
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)

# --- 2. IMPORTS ---
try:
    import psutil
    import modified_didppy as m_dp
    # Import the exact declaration file you created
    import CVRP_single_dual_bound_DIDP_declaration as domain 
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

        if len(sys.argv) < 4: 
            raise ValueError("Args mismatch: <instance> <data_dir> <time_limit>")

        instance_name = sys.argv[1]
        data_dir = sys.argv[2]
        time_limit = float(sys.argv[3])

        file_path = os.path.join(data_dir, instance_name)
        if not os.path.exists(file_path):
             file_path_vrp = file_path + ".vrp"
             if os.path.exists(file_path_vrp): file_path = file_path_vrp

        # Call YOUR specific reader function
        domain.read_formatted_data(file_path)

        # Create Model (Dual bound is ADDED INSIDE this function)
        didp_bundle = domain.creation_of_didp_model_function()
        model, metadata = didp_bundle

        # --- NORMAL CABS SOLVER ---
        # Dual bound is internal to the model.
        # FIX: Passed time_limit to the constructor as requested.
        solver = m_dp.CABS(
            model, 
            quiet=False,           # Print logs
            time_limit=time_limit  # <--- Added Parameter
        )

        start_time = time.time()

        # Run Normal Search
        # Note: Depending on implementation, time_limit might be needed here too. 
        # Keeping it for safety.
        solution = solver.search()

        duration = time.time() - start_time

        if solution.cost is not None:
            output = {
                "status": "success",
                "cost": solution.cost,
                "expanded": solution.expanded,
                "generated": solution.generated,
                "duration": duration,
                "is_optimal": solution.is_optimal
            }
        else:
            output = {
                "status": "timeout",
                "cost": None,
                "expanded": solution.expanded,
                "generated": solution.generated,
                "duration": duration,
                "is_optimal": False
            }

        print("\n" + json.dumps(output))

    except Exception as e:
        print("\n" + json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
