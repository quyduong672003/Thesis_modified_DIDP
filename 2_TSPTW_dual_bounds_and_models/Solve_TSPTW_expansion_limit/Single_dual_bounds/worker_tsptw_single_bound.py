import sys
import os
import json
import time
import threading
import psutil
import modified_didppy as m_dp

# --- CONFIGURATION ---
MEMORY_LIMIT_MB = 12000

# --- 1. SETUP PATHS ---
# Add current dir to path to find the declaration file
sys.path.append(os.getcwd()) 

# --- 2. IMPORTS ---
try:
    import TSPTW_DIDP_model_declaration as domain 
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
             raise FileNotFoundError(f"File not found: {file_path}")

        # 1. Update Globals with TSPTW Data
        if not domain.update_globals_for_tsptw(file_path):
             raise ValueError("Failed to load TSPTW globals")

        # 2. Create Model
        model, _ = domain.creation_of_didp_model_function()

        # 3. Standard CABS Solver
        # Using the exact logic requested: quiet=False, passed time_limit
        solver = m_dp.CABS(
            model, 
            quiet=False,          # Print logs to stdout (Manager captures this)
            time_limit=time_limit # Set limit
        )

        start_time = time.time()

        # 4. Run Search
        solution = solver.search()

        duration = time.time() - start_time

        # 5. Output Result
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
        # Capture python-level errors as JSON for the manager
        print("\n" + json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
