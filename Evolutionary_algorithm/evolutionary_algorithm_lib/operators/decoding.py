from ..utils import (
    extract_chromosome_from_chromosome_fitness_dict, 
    validate_heuristic_presence, 
    convert_chromosome_to_string_of_python_code
)

def compile_chromosome_to_useable_function(chromosome_fitness_dict, dual_bound_functions_dict, print_code=False):
    """
    Converts a chromosome list into a real, callable Python function.
    """
    chromosome = extract_chromosome_from_chromosome_fitness_dict(chromosome_fitness_dict)
    
    if not validate_heuristic_presence(chromosome, dual_bound_functions_dict):
        # This will trigger the 'except' block in your EA loop
        raise ValueError("Invalid Chromosome: No heuristics found (Pure Constant).")
    
    # 1. Generate the code string
    code_string, func_name = convert_chromosome_to_string_of_python_code(chromosome)
    if code_string is None:
        raise ValueError("Invalid Chromosome")
        
    if print_code:
        print(f"Generated Code:\n{code_string}\n")

    # 2. Prepare the execution namespace
    execution_namespace = dual_bound_functions_dict.copy()

    # 3. Compile and Execute the string
    exec(code_string, execution_namespace)

    # 4. Retrieve the live function object
    callable_function = execution_namespace[func_name]

    return callable_function