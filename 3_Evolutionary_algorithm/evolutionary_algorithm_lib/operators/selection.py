import random

def parents_selection(population, 
                    tournament_size, 
                    tournament_probability):
    """
    Selects a single parent using a Tournament Selection.
    """
    user_input_tournament_size = tournament_size
    random_tournament_size = random.randint(2, 10)
    if user_input_tournament_size < random_tournament_size:
        actual_size = min(len(population), random_tournament_size)
    else:
        actual_size = min(len(population), user_input_tournament_size)
    
    list_of_candidates = random.sample(population, actual_size)
            
    if random.random() < tournament_probability:
        winner_of_tournament = list_of_candidates.copy()
        
        while len(winner_of_tournament) > 1:
            idx_1 = random.randint(0, len(winner_of_tournament) - 1)
            candidate_1 = winner_of_tournament[idx_1]
            winner_of_tournament.remove(candidate_1)
            
            idx_2 = random.randint(0, len(winner_of_tournament) - 1)
            candidate_2 = winner_of_tournament[idx_2]
            winner_of_tournament.remove(candidate_2)
            
            f1 = candidate_1['fitness'] if candidate_1['fitness'] is not None else float('inf')
            f2 = candidate_2['fitness'] if candidate_2['fitness'] is not None else float('inf')
            
            if f1 < f2:
                winner_of_tournament.append(candidate_1)
            else:
                winner_of_tournament.append(candidate_2)
        
        parent = winner_of_tournament[0]
        
    else:
        parent = random.choice(list_of_candidates)
        
    return parent