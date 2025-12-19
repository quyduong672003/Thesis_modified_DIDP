import time
import random
from .config import EAHyperparameters
from .operators import (
    initialize_list_of_chromosome_fitness_dictionary,
    parents_selection, 
    combined_crossover_generator,
    generate_valid_chromosome, 
    combined_mutation_generator
)
from .solver import chromosome_fitness_dict_evaluation

def evolution_algorithm_execution(
    didp_model_registry, 
    dual_bound_expression_function, 
    params: EAHyperparameters
    ):
    
    timings = {
        "initialization": 0.0,
        "selection": 0.0,
        "crossover": 0.0,
        "mutation": 0.0,
        "logging_overhead": 0.0,
        "total_runtime": 0.0
    }
    
    overall_start_time = time.time()
    if didp_model_registry is None or dual_bound_expression_function is None:
        raise ValueError("You must provide 'didp_model_registry' and 'dual_bound_expression_function' factories.")
        
    static_model_bundle = didp_model_registry()
    dual_bound_functions_registry = dual_bound_expression_function(static_model_bundle)
    
    print(f"--- Initialization: Generating Population of size {params.population_size} - {params.generations} generations - solver limit at {params.solver_time_limit} seconds ---")
    
    # --- STEP 1: INITIALIZATION ---
    t_start = time.time()
    print("Generating Initial Population at time:", time.ctime())
    
    population = initialize_list_of_chromosome_fitness_dictionary(
        list_size = params.population_size, 
        dual_bound_functions_dict = dual_bound_functions_registry, #This is a dictionary
        didp_model_registry = didp_model_registry, 
        dual_bound_expression_function = dual_bound_expression_function, #This is a function to create the required dictionary
        LB_range_of_constant=params.lb_range_of_constant, 
        UB_range_of_constant=params.ub_range_of_constant,
        min_chromosome_length=params.min_chromosome_length, 
        max_chromosome_length = params.max_chromosome_length, 
        available_operations = params.available_operations,
        reference_point = params.reference_point,
        solver_time_limit = params.solver_time_limit
    )
    timings["initialization"] += time.time() - t_start
    print("Initial Population Generated at time:", time.ctime())
    print("Initialization time", timings["initialization"])
    
    best_individual_ever = min(population, 
                                key=lambda x: x['fitness'] if x['fitness'] is not None else float('inf')
                                )
    print(f"Initial Best Fitness: {best_individual_ever['fitness']}")

    # --- STEP 2: GENERATIONAL LOOP ---
    for gen in range(1, params.generations + 1):
        new_population = []
        
        # ==================================================================================
        #  MODIFIED ELITISM: "Unique Elitism with Random Immigrants"
        # ==================================================================================
        
        num_elites = max(1, int(params.population_size * params.elitism_rate))
        
        # Sort current population by fitness
        sorted_pop = sorted(population, 
                            key=lambda x: x['fitness'] if x['fitness'] is not None else float('inf')
                            )
        
        seen_signatures = set()
        
        # Iterate through the best individuals
        for ind in sorted_pop:
            if len(new_population) >= num_elites:
                break
            
            # Create a unique signature based on the chromosome structure
            signature = str(ind['chromosome'])
            
            if signature not in seen_signatures:
                # UNIQUE: Keep this elite
                new_population.append(ind)
                seen_signatures.add(signature)
            else:
                # DUPLICATE: Replace with a "Random Immigrant"
                # OPTIMIZATION: Generate single chromosome directly instead of using bulk initializer
                random_immigrant_chromosome = generate_valid_chromosome(dual_bound_functions_dict = dual_bound_functions_registry, 
                              LB_range_of_constant=params.lb_range_of_constant,
                              UB_range_of_constant=params.ub_range_of_constant, 
                              available_operations=params.available_operations)
                
                # CRITICAL: We must calculate fitness immediately so this individual 
                # is valid for the rest of the generation (selection, logging, etc.)
                random_immigrant_chromosome_fitness_dict = chromosome_fitness_dict_evaluation(
                    chromosome_fitness_dict={'chromosome': random_immigrant_chromosome, 'fitness': 0}, 
                    didp_model_registry=didp_model_registry, 
                    dual_bound_expression_function=dual_bound_expression_function, 
                    reference_point= params.reference_point,
                    solver_time_limit=params.solver_time_limit
                    )

                new_population.append(random_immigrant_chromosome_fitness_dict)

        # ==================================================================================
        #  END MODIFIED ELITISM
        # ==================================================================================
        
        while len(new_population) < params.population_size:
            # Selection
            t_start = time.time()
            parent1 = parents_selection(population, tournament_size = params.tournament_size, tournament_probability=params.tournament_probability)
            parent2 = parents_selection(population, tournament_size = params.tournament_size, tournament_probability=params.tournament_probability)
            timings["selection"] += time.time() - t_start
            
            # Crossover
            t_start = time.time()
            if random.random() < params.crossover_rate:
                offspring_list = combined_crossover_generator(
                    parent1_dict = parent1, 
                    parent2_dict = parent2, 
                    dual_bound_functions_registry = dual_bound_functions_registry,
                    didp_model_registry = didp_model_registry, 
                    dual_bound_expression_function = dual_bound_expression_function,
                    homology_1_point_crossover_probability = params.homology_1_point_crossover_probability,
                    subtree_crossover_probability = params.subtree_crossover_probability,
                    uniform_crossover_probability = params.uniform_crossover_probability,
                    available_operations = params.available_operations, 
                    reference_point = params.reference_point,
                    solver_time_limit = params.solver_time_limit
                )
            else:
                offspring_list = [parent1, parent2] 
            timings["crossover"] += time.time() - t_start
            
            # Mutation
            t_start = time.time()
            final_offspring_for_batch = []
            for ind in offspring_list:
                if random.random() < params.mutation_rate:
                    mutated_list = combined_mutation_generator(
                        parent_dict=ind, 
                        dual_bound_functions_registry=dual_bound_functions_registry,
                        didp_model_registry = didp_model_registry,
                        dual_bound_expression_function = dual_bound_expression_function,
                        LB_range_of_constant = params.lb_range_of_constant,
                        UB_range_of_constant = params.ub_range_of_constant,
                        mutation_max_subtree_depth = params.mutation_max_subtree_depth,
                        available_operations= params.available_operations,
                        reference_point = params.reference_point,
                        solver_time_limit = params.solver_time_limit,
                        min_chromosome_length = params.min_chromosome_length
                    )
                    final_offspring_for_batch.append(mutated_list[0])
                else:
                    final_offspring_for_batch.append(ind)
            
            for child in final_offspring_for_batch:
                if len(new_population) < params.population_size:
                    new_population.append(child)
            timings["mutation"] += time.time() - t_start
        
        t_start = time.time()
        population = new_population
        
        current_best = min(population, 
                            key=lambda x: x['fitness'] if x['fitness'] is not None else float('inf')
                            )
        
        if (current_best['fitness'] is not None and 
            (best_individual_ever['fitness'] is None or current_best['fitness'] <= best_individual_ever['fitness'])):
            best_individual_ever = current_best
        print(f"Gen {gen}: Best Fitness = {current_best['fitness']} | Global Best = {best_individual_ever['fitness']}")
        timings["logging_overhead"] += time.time() - t_start

    timings["total_runtime"] = time.time() - overall_start_time
    
    print("\n" + "="*40)
    print("      PERFORMANCE PROFILING REPORT      ")
    print("="*40)
    print(f"Total Runtime:    {timings['total_runtime']:.4f} seconds")
    print("="*40)

    print("--- Evolution Completed ---\n")
    return best_individual_ever

def old_evolution_algorithm_execution(
    didp_model_registry, 
    dual_bound_expression_function, 
    params: EAHyperparameters
    ):
    
    timings = {
        "initialization": 0.0,
        "selection": 0.0,
        "crossover": 0.0,
        "mutation": 0.0,
        "logging_overhead": 0.0,
        "total_runtime": 0.0
    }
    
    overall_start_time = time.time()
    if didp_model_registry is None or dual_bound_expression_function is None:
        raise ValueError("You must provide 'didp_model_registry' and 'dual_bound_expression_function' factories.")
        
    static_model_bundle = didp_model_registry()
    dual_bound_functions_registry = dual_bound_expression_function(static_model_bundle)
    
    print(f"--- Initialization: Generating Population of size {params.population_size} - {params.generations} generations ---")
    
    # --- STEP 1: INITIALIZATION ---
    t_start = time.time()
    print("Generating Initial Population at time:", time.ctime())
    
    population = initialize_list_of_chromosome_fitness_dictionary(
        list_size = params.population_size, 
        dual_bound_functions_dict = dual_bound_functions_registry,
        didp_model_registry = didp_model_registry, 
        dual_bound_expression_function = dual_bound_expression_function,
        LB_range_of_constant=params.lb_range_of_constant, 
        UB_range_of_constant=params.ub_range_of_constant,
        min_chromosome_length=params.min_chromosome_length, 
        max_chromosome_length = params.max_chromosome_length, 
        available_operations = params.available_operations,
        reference_point = params.reference_point,
        solver_time_limit = params.solver_time_limit
    )
    timings["initialization"] += time.time() - t_start
    print("Initial Population Generated at time:", time.ctime())
    print("Initilization time", timings["initialization"])
    
    best_individual_ever = min(population, 
                            key=lambda x: x['fitness'] if x['fitness'] is not None else float('inf')
                            )
    print(f"Initial Best Fitness: {best_individual_ever['fitness']}")

    # --- STEP 2: GENERATIONAL LOOP ---
    for gen in range(1, params.generations + 1):
        new_population = []
        
        # Elitism
        num_elites = max(1, int(params.population_size * params.elitism_rate))
        sorted_pop = sorted(population, 
                            key=lambda x: x['fitness'] if x['fitness'] is not None else float('inf')
                            )
        elites = sorted_pop[:num_elites]
        new_population.extend(elites)
        
        while len(new_population) < params.population_size:
            # Selection
            t_start = time.time()
            parent1 = parents_selection(population, tournament_size = params.tournament_size, tournament_probability=params.tournament_probability)
            parent2 = parents_selection(population, tournament_size = params.tournament_size, tournament_probability=params.tournament_probability)
            timings["selection"] += time.time() - t_start
            
            # Crossover
            t_start = time.time()
            if random.random() < params.crossover_rate:
                offspring_list = combined_crossover_generator(
                    parent1_dict = parent1, 
                    parent2_dict = parent2, 
                    dual_bound_functions_registry = dual_bound_functions_registry,
                    didp_model_registry = didp_model_registry, 
                    dual_bound_expression_function = dual_bound_expression_function,
                    homology_1_point_crossover_probability = params.homology_1_point_crossover_probability,
                    subtree_crossover_probability = params.subtree_crossover_probability,
                    uniform_crossover_probability = params.uniform_crossover_probability,
                    available_operations = params.available_operations, 
                    reference_point = params.reference_point,
                    solver_time_limit = params.solver_time_limit
                )
            else:
                offspring_list = [parent1, parent2] 
            timings["crossover"] += time.time() - t_start
            
            # Mutation
            t_start = time.time()
            final_offspring_for_batch = []
            for ind in offspring_list:
                if random.random() < params.mutation_rate:
                    mutated_list = combined_mutation_generator(
                        parent_dict=ind, 
                        dual_bound_functions_registry=dual_bound_functions_registry,
                        didp_model_registry = didp_model_registry,
                        dual_bound_expression_function = dual_bound_expression_function,
                        LB_range_of_constant = params.lb_range_of_constant,
                        UB_range_of_constant = params.ub_range_of_constant,
                        mutation_max_subtree_depth = params.mutation_max_subtree_depth,
                        available_operations= params.available_operations,
                        reference_point = params.reference_point,
                        solver_time_limit = params.solver_time_limit,
                        min_chromosome_length = params.min_chromosome_length
                    )
                    final_offspring_for_batch.append(mutated_list[0])
                else:
                    final_offspring_for_batch.append(ind)
            
            for child in final_offspring_for_batch:
                if len(new_population) < params.population_size:
                    new_population.append(child)
            timings["mutation"] += time.time() - t_start
        
        t_start = time.time()
        population = new_population
        
        current_best = min(population, 
                        key=lambda x: x['fitness'] if x['fitness'] is not None else float('inf')
                        )
        
        if (current_best['fitness'] is not None and 
            (best_individual_ever['fitness'] is None or current_best['fitness'] <= best_individual_ever['fitness'])):
            best_individual_ever = current_best
        print(f"Gen {gen}: Best Fitness = {current_best['fitness']} | Global Best = {best_individual_ever['fitness']}")
        timings["logging_overhead"] += time.time() - t_start

    timings["total_runtime"] = time.time() - overall_start_time
    
    print("\n" + "="*40)
    print("       PERFORMANCE PROFILING REPORT       ")
    print("="*40)
    print(f"Total Runtime:    {timings['total_runtime']:.4f} seconds")
    print("="*40)

    print("--- Evolution Completed ---\n")
    return best_individual_ever