import random
from ..utils import extract_chromosome_from_chromosome_fitness_dict, analyzing_chromosome_based_on_rpn_structure
from .encoding import generate_ramped_half_and_half
from .decoding import compile_chromosome_to_useable_function
from ..solver import chromosome_fitness_dict_evaluation

def subtree_mutation(chromosome_fitness_dict, dual_bound_functions_registry, 
                    LB_range_of_constant, 
                    UB_range_of_constant, 
                    mutation_max_subtree_depth, 
                    min_chromosome_length,
                    available_operations):
    before_mutated_chromosome = extract_chromosome_from_chromosome_fitness_dict(chromosome_fitness_dict)
    
    chromosome_structure = analyzing_chromosome_based_on_rpn_structure(chromosome_fitness_dict = chromosome_fitness_dict, 
                                                                    available_operations = available_operations
                                                                    )
    
    candidates = chromosome_structure["FUNCTIONS"] + chromosome_structure["TERMINALS"]
    cut_start, cut_end = random.choice(candidates)
    
    new_subtree = generate_ramped_half_and_half(dual_bound_functions_registry, 
                                                LB_range_of_constant = LB_range_of_constant, 
                                                UB_range_of_constant = UB_range_of_constant,
                                                min_chromosome_length = min_chromosome_length, 
                                                max_chromosome_length=mutation_max_subtree_depth,
                                                available_operations = available_operations
                                                )

    prefix = before_mutated_chromosome[:cut_start]
    suffix = before_mutated_chromosome[cut_end+1:]
    mutated_chromosome = prefix + new_subtree + suffix
    
    try:
        new_individual = {'chromosome': mutated_chromosome, 'fitness': 0}
        compile_chromosome_to_useable_function(chromosome_fitness_dict = new_individual, 
                                            dual_bound_functions_dict = dual_bound_functions_registry, 
                                            print_code=False)
        return mutated_chromosome
    except Exception as e:
        print(f"Subtree mutation produced invalid tree: {e}")
        print(f"Using fallback original chromosome for this subtree mutation instance.")
        return before_mutated_chromosome

def point_mutation(chromosome_fitness_dict, dual_bound_functions_registry, 
                LB_range_of_constant, 
                UB_range_of_constant, 
                available_operations):
    before_mutation_chromosome = extract_chromosome_from_chromosome_fitness_dict(chromosome_fitness_dict)
    
    chromosome_structure = analyzing_chromosome_based_on_rpn_structure(chromosome_fitness_dict = chromosome_fitness_dict,
                                                                    available_operations = available_operations
                                                                    )
    
    protected_indices = set()
    for start, end in chromosome_structure["TERMINALS"]:
        if (end - start) == 2: 
            protected_indices.add(end)
            
    valid_indices = [i for i in range(len(before_mutation_chromosome)) if i not in protected_indices]
    
    if not valid_indices:
        return chromosome_fitness_dict 
        
    mutation_idx = random.choice(valid_indices)
    original_gene = before_mutation_chromosome[mutation_idx]
    
    new_gene = original_gene 
    
    if isinstance(original_gene, (int, float)):
        new_gene = round(random.uniform(LB_range_of_constant, UB_range_of_constant), 2)
        
    elif isinstance(original_gene, str):
        if original_gene in available_operations:
            candidates = [op for op in available_operations if op != original_gene]
            if candidates:
                new_gene = random.choice(candidates)
        else:
            dual_bound_names = list(dual_bound_functions_registry.keys())
            candidates = [h for h in dual_bound_names if h != original_gene]
            if candidates:
                new_gene = random.choice(candidates)
    
    mutated_chromosome = before_mutation_chromosome.copy()
    mutated_chromosome[mutation_idx] = new_gene
    
    try:
        new_individual = {'chromosome': mutated_chromosome, 'fitness': 0}
        compile_chromosome_to_useable_function(chromosome_fitness_dict = new_individual, 
                                            dual_bound_functions_dict = dual_bound_functions_registry, 
                                            print_code=False)
        return mutated_chromosome
    except Exception as e:
        print(f"Point mutation produced invalid candidates: {e}")
        print(f"Using fallback original chromosome for this point mutation instance.")
        return before_mutation_chromosome

def combined_mutation_generator(parent_dict, dual_bound_functions_registry, 
                                didp_model_registry, 
                                dual_bound_expression_function,
                                LB_range_of_constant, 
                                UB_range_of_constant,
                                mutation_max_subtree_depth, 
                                available_operations,
                                reference_point,
                                solver_time_limit,
                                min_chromosome_length=2
                                ):
    mutation_methods = [
        "subtree",
        "point"
    ]
    
    selected_method = random.choice(mutation_methods)
    raw_mutated_chromosome = []
    
    if selected_method == "subtree":
        raw_mutated_chromosome = subtree_mutation(
            chromosome_fitness_dict = parent_dict, 
            dual_bound_functions_registry = dual_bound_functions_registry, 
            LB_range_of_constant = LB_range_of_constant, 
            UB_range_of_constant = UB_range_of_constant,
            mutation_max_subtree_depth = mutation_max_subtree_depth, 
            min_chromosome_length = min_chromosome_length,
            available_operations = available_operations
        )
    
    elif selected_method == "point":
        raw_mutated_chromosome = point_mutation(
            chromosome_fitness_dict = parent_dict, 
            dual_bound_functions_registry = dual_bound_functions_registry, 
            LB_range_of_constant = LB_range_of_constant, 
            UB_range_of_constant = UB_range_of_constant,
            available_operations = available_operations 
        )

    if isinstance(raw_mutated_chromosome, dict):
        raw_mutated_offspring_chromosome = raw_mutated_chromosome.get('chromosome')
    else:
        raw_mutated_offspring_chromosome = raw_mutated_chromosome

    raw_mutated_chromosome_fitness_dict = {'chromosome': raw_mutated_offspring_chromosome, 'fitness': None}
    
    evaluated_mutated_chromosome_fitness_dict = chromosome_fitness_dict_evaluation(
        chromosome_fitness_dict = raw_mutated_chromosome_fitness_dict, 
        didp_model_registry = didp_model_registry, 
        dual_bound_expression_function = dual_bound_expression_function, 
        reference_point = reference_point,
        solver_time_limit = solver_time_limit
    )
    
    return [evaluated_mutated_chromosome_fitness_dict]