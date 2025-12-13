import time
import os
import re
import modified_didppy as m_dp
# Import decoding from operators package
from .operators.decoding import compile_chromosome_to_useable_function
from .utils import print_comprehensive_solver_report

def combining_modified_didppy_solver_with_chromosome(chromosome, 
                                                    didp_model_registry, 
                                                    dual_bound_expression_function, 
                                                    solver_time_limit = None,
                                                    output_other_result = False,
                                                    print_timing_stats = False):
    """
    Runs the DIDP solver with a flexible, evolved heuristic.
    """
    didp_bundle = didp_model_registry() 
    didp_model = didp_bundle[0]
    
    dual_bound_registry = dual_bound_expression_function(didp_bundle)
    
    try:
        temp_individual_dict = {'chromosome': chromosome}
        combined_bound_func = compile_chromosome_to_useable_function(
            chromosome_fitness_dict=temp_individual_dict, 
            dual_bound_functions_dict= dual_bound_registry, 
            print_code=False
        )
    except Exception as e:
        print(f"Heuristic Compilation Failed: {e}")
        return float('inf')

    py_stats = None

    if print_timing_stats:
        py_stats = {"dual_bound_calc_time": 0.0, "calls": 0}
        
        def safe_dual_bound(state):
            t0 = time.perf_counter()
            try:
                val = combined_bound_func(state)
                res = float(val)
            except Exception:
                res = 0.0
            t1 = time.perf_counter()
            
            py_stats["dual_bound_calc_time"] += (t1 - t0)
            py_stats["calls"] += 1
            return res
            
    else:
        def safe_dual_bound(state):
            try:
                return float(combined_bound_func(state))
            except Exception:
                return 0.0
    
    try:  
        solver = m_dp.CustomDualBoundCABSv1(
            didp_model, 
            dual_bound_func=safe_dual_bound, 
            quiet=True,
            time_limit = solver_time_limit,
            print_timing_stats = print_timing_stats
        )
        
        start_time = time.time()
        solution = solver.search()
        end_time = time.time()
        
        total_duration = end_time - start_time
        timing_info = None

        if print_timing_stats:
            log_filename = "Python_and_Rust_bridge_time_stats.txt"
            if os.path.exists(log_filename):
                try:
                    with open(log_filename, 'r') as f:
                        content = f.read()
                    
                    rust_match = re.search(r"Total time in Python:\s*([\d\.]+)\s*s", content)
                    calls_match = re.search(r"Total calls:\s*(\d+)", content)
                    
                    if rust_match:
                        total_bridge_time = float(rust_match.group(1)) 
                        python_dual_bound_calc_time = py_stats["dual_bound_calc_time"]       
                        switching_overhead = max(0.0, total_bridge_time - python_dual_bound_calc_time)
                        pure_cabs_time = max(0.0, total_duration - total_bridge_time)
                        total_calls = int(calls_match.group(1)) if calls_match else 0
                        
                        timing_info = {
                            "total_duration": total_duration,
                            "total_bridge_time": total_bridge_time,
                            "pure_cabs_time": pure_cabs_time,
                            "switching_overhead": switching_overhead,
                            "python_dual_bound_calc_time": python_dual_bound_calc_time,
                            "total_calls": total_calls
                        }
                except Exception as e:
                    print(f"Failed to parse timing stats: {e}")

        if solution.cost is not None:
            if output_other_result and (print_timing_stats) :
                didp_model_dual_bound_cost = solution.cost
                solution_optimality = solution.is_optimal
                number_generated = solution.generated
                number_expanded = solution.expanded
                return print_comprehensive_solver_report(
                    cost=didp_model_dual_bound_cost, 
                    is_optimal=solution_optimality, 
                    generated=number_generated, 
                    expanded=number_expanded, 
                    timing_info=timing_info)
            elif output_other_result and (not print_timing_stats) :
                return (solution.cost, solution.is_optimal, solution.generated, solution.expanded)
            else:
                return solution.cost
        else:
            if output_other_result:
                return float('inf'), False, 0, 0, timing_info
            else:
                return float('inf')

    except Exception as e:
        print(f"Solver Error: {e}")
        if output_other_result:
            return float('inf'), False, 0, 0, None
        return float('inf')

def chromosome_fitness_dict_evaluation(chromosome_fitness_dict, 
                                    didp_model_registry, 
                                    dual_bound_expression_function, 
                                    reference_point,
                                    solver_time_limit = None,
                                    output_other_result = False,
                                    print_timing_stats = False):
    
    chromosome = chromosome_fitness_dict.get('chromosome')
    if not chromosome:
        chromosome_fitness_dict['fitness'] = float('inf')
        return chromosome_fitness_dict

    solver_result_cost = combining_modified_didppy_solver_with_chromosome(
        chromosome = chromosome,
        didp_model_registry = didp_model_registry,
        dual_bound_expression_function = dual_bound_expression_function,
        solver_time_limit = solver_time_limit
    )

    if solver_result_cost == float('inf'):
        fitness = float('inf')
    else:
        if reference_point != 0:
            deviation = abs(solver_result_cost - reference_point) / abs(reference_point)
        else:
            deviation = abs(solver_result_cost - reference_point)

        tolerance = 1e-6 
        
        if solver_result_cost <= reference_point + tolerance:
            fitness = deviation
        else:
            fitness = deviation

    chromosome_fitness_dict['fitness'] = fitness
    
    return chromosome_fitness_dict