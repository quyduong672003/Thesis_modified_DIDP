import random
from ..utils import extract_chromosome_from_chromosome_fitness_dict
from ..solver import chromosome_fitness_dict_evaluation

def generate_random_terminal_block(dual_bound_functions_dict, 
                                LB_range_of_constant, 
                                UB_range_of_constant):
    keys = list(dual_bound_functions_dict.keys())
    h_name = random.choice(keys)
    if random.random() < 0.5:
        coef = round(random.uniform(LB_range_of_constant, UB_range_of_constant), 2)
        return [coef, h_name, "MULTIPLY"]
    else:
        return [h_name]

def generate_rpn_tree_recursive(current_depth, max_node_depth, method, 
                                dual_bound_functions_dict,
                                LB_range_of_constant, 
                                UB_range_of_constant, 
                                available_operations):
    if current_depth >= max_node_depth:
        return generate_random_terminal_block(dual_bound_functions_dict = dual_bound_functions_dict, 
                                            LB_range_of_constant = LB_range_of_constant, 
                                            UB_range_of_constant = UB_range_of_constant
                                            )

    if method == "FULL":
        choice = "FUNCTION"
    else:
        choice = random.choice(["FUNCTION", "TERMINAL"])

    if choice == "TERMINAL":
        return generate_random_terminal_block(dual_bound_functions_dict = dual_bound_functions_dict, 
                                            LB_range_of_constant = LB_range_of_constant,
                                            UB_range_of_constant = UB_range_of_constant
                                            )
    else: 
        op = random.choice(available_operations)
        left_rpn = generate_rpn_tree_recursive(
            current_depth=current_depth + 1, 
            max_node_depth=max_node_depth, 
            method = method, 
            dual_bound_functions_dict = dual_bound_functions_dict, 
            LB_range_of_constant=LB_range_of_constant,
            UB_range_of_constant=UB_range_of_constant, 
            available_operations=available_operations
        )
        right_rpn = generate_rpn_tree_recursive(
            current_depth=current_depth + 1, 
            max_node_depth=max_node_depth, 
            method=method, 
            dual_bound_functions_dict=dual_bound_functions_dict, 
            LB_range_of_constant=LB_range_of_constant,
            UB_range_of_constant=UB_range_of_constant, 
            available_operations=available_operations
        )
        return left_rpn + right_rpn + [op]

def generate_valid_chromosome(dual_bound_functions_dict, 
                              LB_range_of_constant,
                              UB_range_of_constant, 
                              available_operations):
    available_dual_bound_functions = list(dual_bound_functions_dict.keys())
    blocks = []
    for h_name in available_dual_bound_functions:
        if random.random() < 0.5:
            coef = round(random.uniform(LB_range_of_constant, UB_range_of_constant), 2)
            blocks.append([coef, h_name, "MULTIPLY"])
        else:
            blocks.append([h_name])

    current_pool = blocks.copy()
    while len(current_pool) > 1:
        if len(current_pool) < 2:
            break
        idx1, idx2 = random.sample(range(len(current_pool)), 2)
        right = current_pool.pop(max(idx1, idx2))
        left = current_pool.pop(min(idx1, idx2))
        op = random.choice(available_operations)
        merged = left + right + [op]
        current_pool.append(merged)

    return current_pool[0]

def generate_ramped_half_and_half(dual_bound_functions_dict, 
                                  LB_range_of_constant,
                                  UB_range_of_constant,
                                  min_chromosome_length, 
                                  max_chromosome_length, 
                                  available_operations):
    target_depth = random.randint(min_chromosome_length, max_chromosome_length)
    method = "GROW" if random.random() < 0.5 else "FULL"
    chromosome = generate_rpn_tree_recursive(
        current_depth = 0,
        max_node_depth = target_depth,
        method = method,
        dual_bound_functions_dict = dual_bound_functions_dict,
        LB_range_of_constant = LB_range_of_constant,
        UB_range_of_constant = UB_range_of_constant,
        available_operations = available_operations
    )
    return chromosome

def generate_combined_chromosome(dual_bound_functions_dict, 
                                LB_range_of_constant, UB_range_of_constant,
                                min_chromosome_length, max_chromosome_length, 
                                available_operations):
    if random.random() < 0.5:
        chromosome = generate_valid_chromosome(dual_bound_functions_dict = dual_bound_functions_dict, 
                                            LB_range_of_constant = LB_range_of_constant, 
                                            UB_range_of_constant = UB_range_of_constant, 
                                            available_operations = available_operations
                                            )
        chromosome_fitness_dict = {'chromosome': chromosome, 'fitness': 0}
        return chromosome_fitness_dict
    else:
        chromosome = generate_ramped_half_and_half(dual_bound_functions_dict = dual_bound_functions_dict, 
                                                LB_range_of_constant = LB_range_of_constant,
                                                UB_range_of_constant = UB_range_of_constant, 
                                                min_chromosome_length = min_chromosome_length, 
                                                max_chromosome_length = max_chromosome_length, 
                                                available_operations = available_operations
                                                )
        chromosome_fitness_dict = {'chromosome': chromosome, 'fitness': 0}
        return chromosome_fitness_dict

def initialize_list_of_chromosome_fitness_dictionary(list_size, 
                                                    dual_bound_functions_dict, 
                                                    didp_model_registry, 
                                                    dual_bound_expression_function,
                                                    LB_range_of_constant, 
                                                    UB_range_of_constant,
                                                    min_chromosome_length,
                                                    max_chromosome_length,
                                                    available_operations,
                                                    reference_point,
                                                    solver_time_limit):
    population = []
    seed_signatures = set()
    
    # ---------------------------------------------------------
    # 1. SEEDING PHASE (Target: 2 Candidates)
    # ---------------------------------------------------------
    print("-> Seeding: 1 Random Terminal + 1 Simple Subtree")
    
    # --- SEED 1: The Simple Terminal (e.g., ['min_out']) ---
    if len(population) < list_size:
        # Create a single terminal block
        seed_1_chrom = generate_random_terminal_block(
            dual_bound_functions_dict=dual_bound_functions_dict,
            LB_range_of_constant=LB_range_of_constant,
            UB_range_of_constant=UB_range_of_constant
        )
        
        # Add to population
        seed_signatures.add(tuple(seed_1_chrom))
        population.append(chromosome_fitness_dict_evaluation(
            chromosome_fitness_dict={'chromosome': seed_1_chrom, 'fitness': 0}, 
            didp_model_registry=didp_model_registry, 
            dual_bound_expression_function=dual_bound_expression_function, 
            reference_point=reference_point,
            solver_time_limit=solver_time_limit
        ))

    # --- SEED 2: The Simple Subtree (e.g., ['min_out', 'cost', '+']) ---
    if len(population) < list_size:
        max_attempts = 20
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            # Step A: Generate Left Operand (Terminal)
            left_block = generate_random_terminal_block(
                dual_bound_functions_dict=dual_bound_functions_dict,
                LB_range_of_constant=LB_range_of_constant,
                UB_range_of_constant=UB_range_of_constant
            )
            
            # Step B: Generate Right Operand (Terminal)
            right_block = generate_random_terminal_block(
                dual_bound_functions_dict=dual_bound_functions_dict,
                LB_range_of_constant=LB_range_of_constant,
                UB_range_of_constant=UB_range_of_constant
            )
            
            # Step C: Pick Random Operator
            op = random.choice(available_operations)
            
            # Combine into RPN: [Left parts..., Right parts..., Operator]
            seed_2_chrom = left_block + right_block + [op]
            
            # Step D: Duplicate Check (Ensure it's not identical to Seed 1)
            if tuple(seed_2_chrom) not in seed_signatures:
                # Found a unique simple subtree!
                seed_signatures.add(tuple(seed_2_chrom))
                population.append(chromosome_fitness_dict_evaluation(
                    chromosome_fitness_dict={'chromosome': seed_2_chrom, 'fitness': 0}, 
                    didp_model_registry=didp_model_registry, 
                    dual_bound_expression_function=dual_bound_expression_function, 
                    reference_point=reference_point,
                    solver_time_limit=solver_time_limit
                ))
                break # Exit loop once added

    # ---------------------------------------------------------
    # 2. RANDOM PHASE: Fill remaining slots (Standard Logic)
    # ---------------------------------------------------------
    remaining_slots = list_size - len(population)
    
    if remaining_slots > 0:
        for _ in range(remaining_slots):
            newly_generated_chromosome_fitness_dict = generate_combined_chromosome(
                dual_bound_functions_dict=dual_bound_functions_dict, 
                LB_range_of_constant=LB_range_of_constant, 
                UB_range_of_constant=UB_range_of_constant,
                min_chromosome_length=min_chromosome_length, 
                max_chromosome_length=max_chromosome_length, 
                available_operations=available_operations
            )
            
            # Evaluate and add
            evaluated_newly_generated_chromosome_fitness_dict = chromosome_fitness_dict_evaluation(
                chromosome_fitness_dict=newly_generated_chromosome_fitness_dict, 
                didp_model_registry=didp_model_registry, 
                dual_bound_expression_function=dual_bound_expression_function, 
                reference_point=reference_point,
                solver_time_limit=solver_time_limit
            )
            population.append(evaluated_newly_generated_chromosome_fitness_dict)
            
    return population

def old_initialize_list_of_chromosome_fitness_dictionary(list_size, 
                                                    dual_bound_functions_dict, 
                                                    didp_model_registry, 
                                                    dual_bound_expression_function,
                                                    LB_range_of_constant, 
                                                    UB_range_of_constant,
                                                    min_chromosome_length,
                                                    max_chromosome_length,
                                                    available_operations,
                                                    reference_point,
                                                    solver_time_limit):
    population = []
    
    # ---------------------------------------------------------
    # 1. SEEDING PHASE: Unique Valid Terminal Nodes Only
    # ---------------------------------------------------------
    # Target: up to 2 seeds, but don't exceed list_size
    target_seeds = min(2, list_size)
    
    # We use a set ONLY for the seeds to ensure they are unique among themselves
    seed_signatures = set()
    seed_attempts = 0
    max_seed_attempts = 50 
    
    print(f"-> Attempting to seed {target_seeds} unique terminal candidates...")
    
    while len(population) < target_seeds and seed_attempts < max_seed_attempts:
        seed_attempts += 1
        
        # Generate simple terminal block
        seed_chromosome = generate_random_terminal_block(
            dual_bound_functions_dict=dual_bound_functions_dict,
            LB_range_of_constant=LB_range_of_constant,
            UB_range_of_constant=UB_range_of_constant
        )
        
        # --- SEED DUPLICATE CHECK ---
        # We only check if this seed duplicates another seed
        seed_signature = tuple(seed_chromosome)
        if seed_signature in seed_signatures:
            continue 
            
        # If unique among seeds, proceed to evaluate
        seed_dict = {'chromosome': seed_chromosome, 'fitness': 0}
        
        evaluated_seed = chromosome_fitness_dict_evaluation(
            chromosome_fitness_dict=seed_dict, 
            didp_model_registry=didp_model_registry, 
            dual_bound_expression_function=dual_bound_expression_function, 
            reference_point=reference_point,
            solver_time_limit=solver_time_limit
        )
        
        population.append(evaluated_seed)
        seed_signatures.add(seed_signature)

    # ---------------------------------------------------------
    # 2. RANDOM PHASE: Fill remaining slots (NO DUPLICATE CHECK)
    # ---------------------------------------------------------
    remaining_slots = list_size - len(population)
    
    if remaining_slots > 0:
        for _ in range(remaining_slots):
            newly_generated_chromosome_fitness_dict = generate_combined_chromosome(
                dual_bound_functions_dict=dual_bound_functions_dict, 
                LB_range_of_constant=LB_range_of_constant, 
                UB_range_of_constant=UB_range_of_constant,
                min_chromosome_length=min_chromosome_length, 
                max_chromosome_length=max_chromosome_length, 
                available_operations=available_operations
            )
            
            # Directly evaluate and append without checking duplication
            evaluated_newly_generated_chromosome_fitness_dict = chromosome_fitness_dict_evaluation(
                chromosome_fitness_dict=newly_generated_chromosome_fitness_dict, 
                didp_model_registry=didp_model_registry, 
                dual_bound_expression_function=dual_bound_expression_function, 
                reference_point=reference_point,
                solver_time_limit=solver_time_limit
            )
            population.append(evaluated_newly_generated_chromosome_fitness_dict)
            
    return population