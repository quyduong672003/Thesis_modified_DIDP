import sys
import os
import json
import time
import threading

# --- CONFIGURATION ---
MEMORY_LIMIT_MB = 8000 

# --- 1. SETUP PATHS ---
sys.path.append(os.getcwd()) 

try:
    import psutil
    import modified_didppy as m_dp
    # Ensure this matches the filename in Step 1
    import TSP_DIDP_model_declaration as domain 
except ImportError as e:
    print(json.dumps({"status": "error", "message": f"Import Error: {e}"}))
    sys.exit(1)

# --- MEMORY GUARD ---
def memory_guard():
    process = psutil.Process(os.getpid())
    limit_bytes = MEMORY_LIMIT_MB * 1024 * 1024
    while True:
        try:
            if process.memory_info().rss > limit_bytes:
                os._exit(1)
            time.sleep(1)
        except: break

# --- MAIN LOGIC ---
if __name__ == "__main__":
    try:
        threading.Thread(target=memory_guard, daemon=True).start()

        # EXPECTS: script_name, instance, dir, time, nodes (5 total)
        if len(sys.argv) < 5: 
            raise ValueError(f"Args mismatch. Received {len(sys.argv)}: {sys.argv}")

        instance_name = sys.argv[1]
        data_dir = sys.argv[2] 
        time_limit = float(sys.argv[3])
        node_limit = int(sys.argv[4])

        # 1. Load Data
        file_path = os.path.join(data_dir, instance_name)
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Data file not found: {file_path}")

        n_val, c_val = domain.read_tsp_cappart_format(file_path)

        # 2. Update Globals
        domain.current_num_locations = n_val
        domain.current_travel_cost = c_val

        # 3. Create Model
        model, metadata = domain.creation_of_didp_model_function()

        # 4. Standard CABS Solver
        solver = m_dp.CABS(model, initial_beam_size=1, quiet=True)

        # 5. Profiling Loop
        history = [] 
        start_time = time.time()

        while True:
            if time.time() - start_time > time_limit: 
                break

            solution, is_terminated = solver.search_next()

            if solution.cost is not None:
                history.append((solution.expanded, solution.cost))

            if is_terminated:
                break
            if solution.expanded >= node_limit:
                break

        output = {
            "status": "success",
            "history": history,
            "duration": time.time() - start_time
        }
        print(json.dumps(output))

    except Exception as e:
        # This print ensures the Manager sees the error in stdout/stderr
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
