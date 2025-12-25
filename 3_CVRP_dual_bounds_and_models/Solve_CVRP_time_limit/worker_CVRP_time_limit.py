import sys
import os
import ast
import json
import time
import threading

MEMORY_LIMIT_MB = 7900 
PROJECT_ROOT = r"C:\Users\ACER\Desktop\Code\0_Thesis_implementation\2_DIDP_custom_search_guidance_local\Thesis_modified_DIDP"
LIB_PATH = os.path.join(PROJECT_ROOT, "Evolutionary_algorithm")

sys.path.append(os.getcwd())
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
if LIB_PATH not in sys.path: sys.path.append(LIB_PATH)

try:
    import psutil
    import modified_didppy as m_dp
    import CVRP_DIDP_model_and_dual_bound_declaration as cvrp_domain 
    from evolutionary_algorithm_lib import combining_modified_didppy_solver_with_chromosome
except ImportError as e:
    print(json.dumps({"status": "error", "message": f"Import Error: {e}"}))
    sys.exit(1)

def memory_guard():
    process = psutil.Process(os.getpid())
    limit_bytes = MEMORY_LIMIT_MB * 1024 * 1024
    while True:
        try:
            if process.memory_info().rss > limit_bytes:
                sys.stderr.write(f"\n[Guard] Memory limit reached.\n")
                sys.stderr.flush()
                os._exit(1)
            time.sleep(1)
        except: break

if __name__ == "__main__":
    try:
        threading.Thread(target=memory_guard, daemon=True).start()

        if len(sys.argv) < 5: raise ValueError("Args mismatch")

        instance_name = sys.argv[1]
        chromosome_str = sys.argv[2]
        data_dir = sys.argv[3]
        time_limit = float(sys.argv[4])

        file_path = os.path.join(data_dir, instance_name)

        # === CHANGED: CALL YOUR NEW UPDATE FUNCTION ===
        cvrp_domain.update_globals_for_cvrp(file_path)
        # Globals in cvrp_domain are now set!

        start_time = time.time()
        result = combining_modified_didppy_solver_with_chromosome(
            chromosome=ast.literal_eval(chromosome_str),
            didp_model_registry=cvrp_domain.creation_of_didp_model_function,
            dual_bound_expression_function=cvrp_domain.dual_bound_expression_function,
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
