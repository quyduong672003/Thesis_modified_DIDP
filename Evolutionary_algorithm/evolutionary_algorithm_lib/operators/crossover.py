import random
from ..utils import (
    extract_chromosome_from_chromosome_fitness_dict, 
    find_homologous_pairs, 
    construct_the_lookup_map_for_rpn, 
    analyzing_chromosome_based_on_rpn_structure, 
    get_rpn_node_arity
)
from .decoding import compile_chromosome_to_useable_function
from ..solver import chromosome_fitness_dict_evaluation

def one_point_crossover_two_offspring(parent1_dict, parent2_dict, 
                                    dual_bound_functions_registry, 
                                    homology_1_point_crossover_probability,
                                    available_operations):
    p1_chromosome = extract_chromosome_from_chromosome_fitness_dict(parent1_dict)
    p2_chromosome = extract_chromosome_from_chromosome_fitness_dict(parent2_dict)
    
    matching_pairs, p1_topology, p2_topology= find_homologous_pairs(parent1_dict = parent1_dict,
                                                                    parent2_dict = parent2_dict ,
                                                                    available_operations = available_operations)

    if matching_pairs and random.random() < homology_1_point_crossover_probability:
        cp1, cp2 = random.choice(matching_pairs)
    else:
        p1_lookup = construct_the_lookup_map_for_rpn(p1_topology)
        p2_lookup = construct_the_lookup_map_for_rpn(p2_topology)
        
        p1_funcs = [idx for idx, data in p1_lookup.items() if data[2] == 'FUNCTION']
        p2_funcs = [idx for idx, data in p2_lookup.items() if data[2] == 'FUNCTION']
        
        compatible_pairs = []
        for i1 in p1_funcs:
            depth1 = p1_lookup[i1][3] 
            for i2 in p2_funcs:
                depth2 = p2_lookup[i2][3]
                if depth1 == depth2:
                    compatible_pairs.append((i1, i2))
    
        if compatible_pairs:
            cp1, cp2 = random.choice(compatible_pairs)
        else:
            cp1 = len(p1_chromosome) - 1
            cp2 = len(p2_chromosome) - 1
    
    offspring1 = p2_chromosome[:cp2+1] + p1_chromosome[cp1+1:]
    offspring2 = p1_chromosome[:cp1+1] + p2_chromosome[cp2+1:]
    
    offspring_list = []
    fallback_candidate_list  = [p2_chromosome, p1_chromosome]
    for i, chromosome in enumerate([offspring1, offspring2]):
        chromosome_fitness_dict = {'chromosome': chromosome, 'fitness': 0}
        try:
            func = compile_chromosome_to_useable_function(chromosome_fitness_dict = chromosome_fitness_dict, 
                                                        dual_bound_functions_dict = dual_bound_functions_registry, 
                                                        print_code=False)
            offspring_list.append(chromosome)
        except Exception as e:
            print(f"Offspring {i+1} of one-point crossover is invalid: {e}")
            fallback_candidate = random.choice(fallback_candidate_list)
            fallback_candidate_list.remove(fallback_candidate)
            print(f"Using fallback parent chromosome for Offspring {i+1} of this one-point crossover instance.")
            offspring_list.append(fallback_candidate)
    return offspring_list

def subtree_crossover(parent1_dict, parent2_dict, 
                    dual_bound_functions_registry, 
                    subtree_crossover_probability, 
                    available_operations):
    p1_chromosome = extract_chromosome_from_chromosome_fitness_dict(parent1_dict)
    p2_chromosome = extract_chromosome_from_chromosome_fitness_dict(parent2_dict)
    
    p1_structure = analyzing_chromosome_based_on_rpn_structure(chromosome_fitness_dict = parent1_dict,
                                                            available_operations = available_operations)
    p2_structure = analyzing_chromosome_based_on_rpn_structure(chromosome_fitness_dict= parent2_dict,
                                                            available_operations = available_operations)
    
    if random.random() < subtree_crossover_probability and p1_structure["FUNCTIONS"]:
        candidate1 = p1_structure["FUNCTIONS"]
    else:
        candidate1 = p1_structure["TERMINALS"]    
    root_index = len(p1_chromosome) - 1
    valid_candidates = [
        (s, e) for (s, e) in candidate1 
        if e != root_index 
    ]

    if valid_candidates:
        p1_start, p1_end = random.choice(valid_candidates)
    else:
        candidate1 = p1_structure["TERMINALS"]   
        p1_start, p1_end = random.choice(candidate1)
    
    if random.random() < subtree_crossover_probability and p2_structure["FUNCTIONS"]:
        candidate2 = p2_structure["FUNCTIONS"]
    else:
        candidate2 = p2_structure["TERMINALS"]  
    if not candidate2: candidate2 = p2_structure["TERMINALS"]   
    p2_start, p2_end = random.choice(candidate2)
    
    head_of_parent1_chromosome = p1_chromosome[:p1_start]
    donor_gene = p2_chromosome[p2_start : p2_end+1]
    tail_of_parent1_chromosome = p1_chromosome[p1_end+1:]
    
    offspring_chromosome = head_of_parent1_chromosome + donor_gene + tail_of_parent1_chromosome
    
    try:
        offspring_chromosome_function = compile_chromosome_to_useable_function(
            {'chromosome': offspring_chromosome, 'fitness':0}, 
            dual_bound_functions_dict = dual_bound_functions_registry, 
            print_code=False
        )
    except Exception as e:
        print(f"Subtree crossover produced invalid offspring: {e}")
        print(f"Using fallback parent chromosome for this subtree crossover.")
        return random.choice([parent1_dict['chromosome'], parent2_dict['chromosome']])

    return offspring_chromosome

def uniform_crossover_weighted_protected(parent1_dict, parent2_dict, 
                                        dual_bound_functions_registry, 
                                        uniform_crossover_probability,
                                        available_operations):
    p1_chromosome = extract_chromosome_from_chromosome_fitness_dict(parent1_dict)
    p2_chromosome = extract_chromosome_from_chromosome_fitness_dict(parent2_dict)
    
    p1_structure = analyzing_chromosome_based_on_rpn_structure(chromosome_fitness_dict = parent1_dict,
                                                            available_operations = available_operations)
    p2_structure = analyzing_chromosome_based_on_rpn_structure(chromosome_fitness_dict= parent2_dict,
                                                            available_operations = available_operations)
    
    protected_indices_p1 = set()
    for start, end in p1_structure["TERMINALS"]:
        if (end - start) == 2: 
            protected_indices_p1.add(end)
            
    protected_indices_p2 = set()
    for start, end in p2_structure["TERMINALS"]:
        if (end - start) == 2:
            protected_indices_p2.add(end)
            
    min_len = min(len(p1_chromosome), len(p2_chromosome))
    offspring1 = p1_chromosome.copy()
    offspring2 = p2_chromosome.copy()
    
    i = 0
    while i < min_len - 1:
        block_end_index = i + 2
        
        if block_end_index < (min_len - 1):
            is_block_p1 = (block_end_index in protected_indices_p1)
            is_block_p2 = (block_end_index in protected_indices_p2)
            
            if is_block_p1 and is_block_p2:
                if random.random() < uniform_crossover_probability:
                    offspring1[i], offspring2[i] = offspring2[i], offspring1[i]
                    offspring1[i+1], offspring2[i+1] = offspring2[i+1], offspring1[i+1]
                    offspring1[i+2], offspring2[i+2] = offspring2[i+2], offspring1[i+2]
                
                i += 3
                continue

        gene1 = p1_chromosome[i]
        gene2 = p2_chromosome[i]
        
        arity1 = get_rpn_node_arity(gene = gene1, available_operations = available_operations)
        arity2 = get_rpn_node_arity(gene = gene2, available_operations = available_operations)
        
        if arity1 == arity2:
            is_protected_p1 = (i in protected_indices_p1)
            is_protected_p2 = (i in protected_indices_p2)
            
            if (is_protected_p1 or is_protected_p2) and (gene1 != gene2):
                pass 
            else:
                if random.random() < uniform_crossover_probability:
                    offspring1[i] = gene2
                    offspring2[i] = gene1
        i += 1
                
    offspring_list = []
    for chromosome, original in [(offspring1, p1_chromosome), (offspring2, p2_chromosome)]:
        try:
            temp_dict = {'chromosome': chromosome, 'fitness':0}
            compile_chromosome_to_useable_function(temp_dict, 
                                                dual_bound_functions_dict = dual_bound_functions_registry ,
                                                print_code=False)
            offspring_list.append(chromosome)
        except Exception as e:
            print(f"Uniform crossover produced invalid offspring: {e}")
            print(f"Using fallback parent chromosome for this uniform crossover.")
            offspring_list.append(original)
            
    return offspring_list

def combined_crossover_generator(parent1_dict, parent2_dict, 
                                dual_bound_functions_registry, 
                                didp_model_registry, 
                                dual_bound_expression_function, 
                                homology_1_point_crossover_probability, 
                                subtree_crossover_probability,
                                uniform_crossover_probability,
                                available_operations,
                                reference_point,
                                solver_time_limit):
    crossover_methods = [
        "one_point",
        "subtree",
        "uniform"
    ]
    
    selected_method = random.choice(crossover_methods)
    raw_offspring_chromosomes = []
    
    if selected_method == "one_point":
        raw_offspring_chromosomes = one_point_crossover_two_offspring(
            parent1_dict, parent2_dict, 
            dual_bound_functions_registry = dual_bound_functions_registry, 
            homology_1_point_crossover_probability=homology_1_point_crossover_probability,
            available_operations=available_operations
        )
    
    elif selected_method == "subtree":
        single_offspring = subtree_crossover(
            parent1_dict, parent2_dict, 
            dual_bound_functions_registry = dual_bound_functions_registry, 
            subtree_crossover_probability=subtree_crossover_probability,
            available_operations=available_operations
        )
        raw_offspring_chromosomes = [single_offspring]
    
    elif selected_method == "uniform":
        raw_offspring_chromosomes = uniform_crossover_weighted_protected(
            parent1_dict, parent2_dict, 
            dual_bound_functions_registry = dual_bound_functions_registry,  
            uniform_crossover_probability=uniform_crossover_probability,
            available_operations=available_operations
        )

    evaluated_offspring_list = []
    
    for offspring_chromosome in raw_offspring_chromosomes:
        offspring_chromosome_fitness_dict = {'chromosome': offspring_chromosome, 'fitness': None}
        
        evaluated_individual = chromosome_fitness_dict_evaluation(
            chromosome_fitness_dict= offspring_chromosome_fitness_dict, 
            didp_model_registry =didp_model_registry, 
            dual_bound_expression_function =dual_bound_expression_function, 
            reference_point = reference_point,
            solver_time_limit = solver_time_limit
        )
        
        evaluated_offspring_list.append(evaluated_individual)
        
    return evaluated_offspring_list