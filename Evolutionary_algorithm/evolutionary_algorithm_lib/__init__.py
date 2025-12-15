# evolutionary_algorithm_lib/__init__.py

# 1. Configuration
from .config import EAHyperparameters

# 2. Main Core Loop
from .core import evolution_algorithm_execution

# 3. Solvers & Utils
from .solver import combining_modified_didppy_solver_with_chromosome, chromosome_fitness_dict_evaluation
from .utils import *

# 4. All Operators (Generation, Crossover, Mutation, Selection)
# This works because we fixed operators/__init__.py in the previous step
from .operators import * 

'''

### **How to verify it works**
After saving that file and restarting your kernel, you can run this test in your notebook:

python
from evolutionary_algorithm_lib import *

# Now you can call ANY function directly without 'ea.' prefix
params = EAHyperparameters()
print("Config loaded:", params)

# You can even call internal operator functions directly for testing
pop = initialize_list_of_chromosome_fitness_dictionary(
    list_size=5,
    dual_bound_functions_dict={}, # pass empty dict just to test import
    didp_model_registry=None,
    dual_bound_expression_function=None,
    LB_range_of_constant=0,
    UB_range_of_constant=10,
    min_chromosome_length=2,
    max_chromosome_length=5,
    available_operations=["ADD"],
    reference_point=0,
    solver_time_limit=1
)
print("Function imported successfully!")'''