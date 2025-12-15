from dataclasses import dataclass, field
from typing import List

@dataclass
class EAHyperparameters:
    # --- 1. EVOLUTIONARY ALGORITHM HYPERPARAMETERS ---
    population_size: int = 50
    generations: int = 10
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elitism_rate: float = 0.02

    # --- 2. OPERATOR PARAMETERS ---
    lb_range_of_constant: float = 0.0
    ub_range_of_constant: float = 10.0
    min_chromosome_length: int = 2
    max_chromosome_length: int = 10
    
    # Tournament
    tournament_size: int = 5
    tournament_probability: float = 0.8
    
    # Probabilities & Depths
    mutation_max_subtree_depth: int = 5
    homology_1_point_crossover_probability: float = 0.5
    subtree_crossover_probability: float = 0.9
    uniform_crossover_probability: float = 0.5

    # --- 3. PROBLEM SPECIFIC PARAMETERS ---
    # Default factory ensures a new list is created for each instance
    available_operations: List[str] = field(default_factory=lambda: ["ADD", "SUBTRACT", "MAX", "MIN", "MULTIPLY", "PDIV"])
    
    # --- 4. OTHER PARAMETERS ---
    reference_point: float = 0.0  # OPTIMAL_COST_REFERENCE
    solver_time_limit: float = 5.0 # Seconds