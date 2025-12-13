import random
import sys

def extract_chromosome_from_chromosome_fitness_dict(chromosome_fitness_dict):
    """Extracts the chromosome from a chromosome-fitness dictionary."""
    return chromosome_fitness_dict.get('chromosome')

def automatic_creation_of_dual_bounds_registry(local_scope):
    """Automatically builds a registry dict from the local scope."""
    return {
        name: func 
        for name, func in local_scope.items() 
        if name.startswith("h") and callable(func)
    }

def is_operator(gene, available_operations):
    return gene in available_operations

def get_rpn_node_arity(gene, available_operations):
    """Returns 2 for binary operators, 0 for terminals."""
    if isinstance(gene, str) and gene in available_operations:
        return 2
    return 0

def analyzing_chromosome_based_on_rpn_structure(chromosome_fitness_dict, available_operations):
    """
    Scans a chromosome to identify all valid subtrees.
    Returns a dictionary categorizing them into 'TERMINALS' and 'FUNCTIONS'.
    """
    chromosome = extract_chromosome_from_chromosome_fitness_dict(chromosome_fitness_dict)
    structure = {
        "TERMINALS": [],  
        "FUNCTIONS": []   
    }
    
    i = len(chromosome) - 1
    
    while i >= 0:
        root_gene = chromosome[i]
        end_index = i
        
        required_inputs = 1 
        current_pos = i
        
        while required_inputs > 0:
            gene = chromosome[current_pos]
            if is_operator(gene, available_operations):
                required_inputs += 1 
            else:
                required_inputs -= 1 
            current_pos -= 1
        start_index = current_pos + 1
        
        subtree_slice = chromosome[start_index : end_index+1]
        
        is_weighted_block = (
            len(subtree_slice) == 3 and 
            isinstance(subtree_slice[0], (int, float)) and
            isinstance(subtree_slice[1], str) and 
            subtree_slice[2] == "MULTIPLY"
        )
        
        is_raw_terminal = (start_index == end_index)
        
        if is_weighted_block or is_raw_terminal:
            structure["TERMINALS"].append((start_index, end_index))
        else:
            structure["FUNCTIONS"].append((start_index, end_index))
            
        i -= 1

    return structure

def convert_chromosome_to_string_of_python_code(chromosome):
    """
    Translates a list-based chromosome (RPN) into a Python function definition string.
    """
    stack = []
    try:
        for gene in chromosome:
            if isinstance(gene, (int, float)):
                stack.append(str(gene))
            elif isinstance(gene, str):
                if gene == "ADD":
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"({a} + {b})")
                elif gene == "SUBTRACT":
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"({a} - {b})")
                elif gene == "MULTIPLY":
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"({a} * {b})")
                elif gene == "MAX":
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"max({a}, {b})")
                elif gene == "MIN":
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"min({a}, {b})")
                elif gene == "PDIV":
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"({a} / {b} if abs({b}) > 1e-6 else 1.0)")
                else:
                    stack.append(f"{gene}(state)")
        if len(stack) == 1:
            formula_body = stack.pop()
            func_name = "dual_bound_combination"
            code_string = f"def {func_name}(state):\n    return {formula_body}"
            return code_string, func_name
        else:
            return None, None
    except IndexError:
        return None, None

def validate_heuristic_presence(chromosome, dual_bound_functions_dict):
    valid_heuristics = set(dual_bound_functions_dict.keys())
    for gene in chromosome:
        if gene in valid_heuristics:
            return True
    return False

def construct_the_lookup_map_for_rpn(chromosome_rpn_topology):
    lookup = {}
    for idx, node_data in chromosome_rpn_topology.items():
        start, end = node_data['subtree_range']
        node_type = node_data['type']
        depth = node_data['depth']
        lookup[end] = (start, end, node_type, depth)
    return lookup

def map_chromosome_fitness_dict_to_rpn_topology(chromosome_fitness_dict, available_operations):
    topology = {}
    stack = []
    chromosome = extract_chromosome_from_chromosome_fitness_dict(chromosome_fitness_dict)
    for i, gene in enumerate(chromosome):
        arity = get_rpn_node_arity(gene, available_operations)
        node_value = gene 
        
        if arity == 0:
            subtree_start_location = i
            stack.append((i, subtree_start_location))
            
            topology[i] = {
                'type': 'TERMINAL', 
                'value': node_value, 
                'subtree_range': (i, i),
                'depth': len(stack)
            }
            
        elif arity == 2:
            if len(stack) < 2: return {} 
            
            right_idx, right_start = stack.pop()
            left_idx, left_start = stack.pop()
            subtree_range = (left_start, i)
            current_depth = len(stack) + 1
            
            topology[i] = {
                'type': 'FUNCTION', 
                'value': node_value,
                'subtree_range': subtree_range,
                'left_child': left_idx,
                'right_child': right_idx,
                'depth': current_depth
            }
            stack.append((i, left_start))
            
    return topology

def find_homologous_pairs(parent1_dict, parent2_dict, available_operations):
    p1_chromosome = extract_chromosome_from_chromosome_fitness_dict(chromosome_fitness_dict = parent1_dict)
    p2_chromosome = extract_chromosome_from_chromosome_fitness_dict(chromosome_fitness_dict = parent2_dict)
    
    p1_topology = map_chromosome_fitness_dict_to_rpn_topology(chromosome_fitness_dict =parent1_dict, available_operations = available_operations)
    p2_topology = map_chromosome_fitness_dict_to_rpn_topology(chromosome_fitness_dict = parent2_dict, available_operations = available_operations)
    
    matching_pairs = []
    root_p1 = len(p1_chromosome) - 1
    root_p2 = len(p2_chromosome) - 1
    queue = [(root_p1, root_p2)]
    
    while queue:
        idx1, idx2 = queue.pop(0)
        node1 = p1_topology.get(idx1)
        node2 = p2_topology.get(idx2)
        
        if not node1 or not node2: 
            continue
        
        if node1['type'] == node2['type']:
            matching_pairs.append((idx1, idx2))
            
            if node1['type'] == 'FUNCTION':
                queue.append((node1['left_child'], node2['left_child']))
                queue.append((node1['right_child'], node2['right_child']))
                
    return matching_pairs, p1_topology, p2_topology

def print_comprehensive_solver_report(cost, is_optimal, generated, expanded, timing_info):
    print("\n" + "="*60)
    print(f"{'📋 SOLVER RUN REPORT':^60}")
    print("="*60)

    opt_status = "✅ Proven Optimal" if is_optimal else "⚠️  Suboptimal / Timeout"
    print(f"🎯 SOLUTION STATUS:")
    print(f"   • Cost:              {cost}")
    print(f"   • Status:            {opt_status}")
    print(f"   • Nodes Generated:   {generated:,}")
    print(f"   • Nodes Expanded:    {expanded:,}")
    
    if expanded > 0:
        ratio = generated / expanded
        print(f"   • Branching Factor:  ~{ratio:.2f}")
    print("-" * 60)

    if timing_info:
        t_total = timing_info.get("total_wall_clock", timing_info.get("total_duration", 0))
        t_cabs = timing_info.get("pure_cabs_time", 0)
        t_bridge = timing_info.get("total_bridge_time", timing_info.get("bridge_time", 0))
        t_calc = timing_info.get("python_dual_bound_calc_time", timing_info.get("python_calc", 0))
        t_switch = timing_info.get("switching_overhead", 0)
        n_calls = timing_info.get("total_calls", timing_info.get("calls", 0))

        pct_cabs = (t_cabs / t_total * 100) if t_total > 0 else 0
        pct_bridge = (t_bridge / t_total * 100) if t_total > 0 else 0
        pct_calc = (t_calc / t_bridge * 100) if t_bridge > 0 else 0
        pct_switch = (t_switch / t_bridge * 100) if t_bridge > 0 else 0

        print(f"⏱️  TIME DISTRIBUTION (Total: {t_total:.4f}s):")
        print(f"   [Pure CABS time: {pct_cabs:5.1f}%] 🆚 [Bridge time: {pct_bridge:5.1f}%]")
        
        print(f"\n   1. 🟢 Pure CABS (Search):    {t_cabs:.4f} s")
        if expanded > 0:
            print(f"      └─ Average time per expanded state:   {(t_cabs/expanded*1000):.4f} ms")

        print(f"   2. 🔴 Total Bridge Time:     {t_bridge:.4f} s")
        print(f"      ├─ 🐍 Dual Bound Calculation Time: :     {t_calc:.4f} s  ({pct_calc:5.1f}% of bridge time)")
        print(f"      └─ 🌉 Switching Time:  {t_switch:.4f} s  ({pct_switch:5.1f}% of bridge time)")
        
        print("-" * 60)
        
        if n_calls > 0:
            print(f"📊 PYTHON DUAL BOUND CALL STATS:")
            print(f"   • Total Calls:       {n_calls:,}")
            print(f"   • Avg Bridge Time:   {(t_bridge/n_calls*1000):.4f} ms")
            print(f"   • Avg Pure Calc:     {(t_calc/n_calls*1000):.4f} ms")
            print(f"   • Avg Switching:     {(t_switch/n_calls*1000):.4f} ms")
            rate = n_calls / t_total if t_total > 0 else 0
            print(f"   • Throughput:        {rate:,.0f} calls/sec")
    print("="*60 + "\n")