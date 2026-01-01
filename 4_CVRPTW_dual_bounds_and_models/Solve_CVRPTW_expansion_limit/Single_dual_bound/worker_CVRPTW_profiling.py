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
PROJECT_ROOT = r"C:\Users\ACER\Desktop\Code\0_Thesis_implementation\2_DIDP_custom_search_guidance_local\Thesis_modified_DIDP"
sys.path.append(os.getcwd()) 
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)

# --- 2. IMPORTS ---
try:
    import CVRPTW_DIDP_model_declaration as domain
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
                os._exit(1) # Hard exit to avoid swap death
            time.sleep(1)
        except: break

# --- 4. MAIN SOLVER LOGIC ---
if __name__ == "__main__":
    try:
        # Start Memory Guard
        threading.Thread(target=memory_guard, daemon=True).start()

        if len(sys.argv) < 5: 
            raise ValueError("Args mismatch. Usage: worker.py <instance> <data_dir> <time_limit> <node_limit>")

        instance_name = sys.argv[1]
        data_dir = sys.argv[2]
        time_limit = float(sys.argv[3])
        node_limit = int(sys.argv[4])

        file_path = os.path.join(data_dir, instance_name)

        # 1. Load Instance
        if not domain.update_globals_for_cvrptw(file_path):
            raise ValueError(f"Failed to load: {file_path}")

        # 2. Setup Model (No Compilation needed, standard CABS)
        didp_bundle = domain.creation_of_didp_model_function() 
        model, metadata = didp_bundle

        # 3. Instantiate Solver (Standard CABS)
        # We use normal CABS because we are benchmarking the single dual bound defined in the model
        solver = m_dp.CABS(
            model,
            time_limit=time_limit,
            initial_beam_size=1,       
            max_beam_size=None,        
            quiet=False, # CRITICAL: Enable logs for parsing
        )

        start_time = time.time()

        # 4. Run Search (FAST C++ LOOP)
        # Passing node_limit enables profiling behavior
        solution = solver.search()

        duration = time.time() - start_time

        # 5. Output (History is in stdout logs)
        output = {
            "status": "success",
            "cost": solution.cost,
            "expanded": solution.expanded,
            "generated": solution.generated,
            "duration": duration,
            "is_optimal": solution.is_optimal,
            "history": [] # Manager parses this from stdout
        }
        print("\n" + json.dumps(output))

    except Exception as e:
        print("\n" + json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
