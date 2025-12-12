// 🟢 Import TimingStats explicitly
use super::customized_evaluator::{create_combined_evaluator_with_py_func, TimingStats};

// Use the correct, now public, paths
use crate::heuristic_search_solver::user_priority_evaluator::create_default_g_evaluator_vectors;
use crate::heuristic_search_solver::wrapped_solver::WrappedSolver;
use crate::heuristic_search_solver::{FOperator, SolutionPy};

use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{
    create_user_priority_cabs, BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters,
    Search, UserEvaluators,
};
use dypdl_heuristic_search::search_algorithm::StateInRegistry;
use pyo3::prelude::*;
use std::rc::Rc;

// Define a type alias for our heuristic function to make the code cleaner
type HBuilder = Box<dyn Fn(&StateInRegistry) -> Option<OrderedContinuous>>;

#[pyclass(unsendable, name = "CustomDualBoundCABSv1")]
pub struct CustomDualBoundCabsPy {
    // 🟢 Named struct fields
    solver: WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>,
    // 🟢 Store the guard here
    _guard: TimingStats,
}

#[pymethods]
impl CustomDualBoundCabsPy {
    #[new]
    #[pyo3(
        text_signature = "(model, dual_bound_func, f_operator=didppy.FOperator.Plus, primal_bound=None, time_limit=None, quiet=False, initial_beam_size=1, keep_all_layers=False, max_beam_size=None, expansion_limit=None, print_timing_stats=False)"
    )]
    #[pyo3(signature = (
        model,
        dual_bound_func,
        f_operator = FOperator::Plus,
        primal_bound = None,
        time_limit = None,
        quiet = false,
        initial_beam_size = 1,
        keep_all_layers = false,
        max_beam_size = None,
        expansion_limit = None,
        print_timing_stats = false, // 🟢 Default false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &ModelPy,
        dual_bound_func: Py<PyAny>,
        f_operator: FOperator,
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
        initial_beam_size: usize,
        keep_all_layers: bool,
        max_beam_size: Option<usize>,
        expansion_limit: Option<usize>,
        print_timing_stats: bool, // 🟢 New argument
    ) -> PyResult<CustomDualBoundCabsPy> {
        if !quiet {
            println!(
                "Solver: CustomDualBoundCABSv1 from DIDPPy v{}",
                env!("CARGO_PKG_VERSION")
            );
        }

        let bound_evaluator_type = FEvaluatorType::from(f_operator.clone());
        let f_evaluator_type = FEvaluatorType::from(f_operator);
        let float_cost = model.float_cost();
        let rust_model = Rc::new(model.inner_as_ref().clone());

        let (forced_g_evaluators, g_evaluators) =
            create_default_g_evaluator_vectors(&rust_model);

        // 🟢 Create Evaluator AND Guard
        let (h_evaluator, guard): (HBuilder, TimingStats) = if float_cost {
            let (closure, g) = create_combined_evaluator_with_py_func::<OrderedContinuous>(
                rust_model.clone(),
                dual_bound_func,
                print_timing_stats,
            );
            (Box::new(closure), g)
        } else {
            let (closure, g) = create_combined_evaluator_with_py_func::<Integer>(
                rust_model.clone(),
                dual_bound_func,
                print_timing_stats,
            );
            (Box::new(closure), g)
        };

        let user_evaluators = UserEvaluators {
            forced_g_evaluators,
            g_evaluators,
            h_evaluator, 
            f_evaluator_type,
        };

        if float_cost {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            let parameters = CabsParameters {
                max_beam_size,
                beam_search_parameters: BeamSearchParameters {
                    beam_size: initial_beam_size,
                    keep_all_layers,
                    parameters: Parameters::<OrderedContinuous> {
                        primal_bound,
                        time_limit,
                        get_all_solutions: false,
                        quiet,
                        initial_registry_capacity: None,
                        expansion_limit,
                    },
                },
            };

            let solver = create_user_priority_cabs(
                rust_model,
                parameters,
                bound_evaluator_type,
                user_evaluators,
            );

            // 🟢 Return struct with named fields
            Ok(CustomDualBoundCabsPy {
                solver: WrappedSolver::Float(solver),
                _guard: guard, 
            })
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract::<Integer>()?)
            } else {
                None
            };
            let parameters = CabsParameters {
                max_beam_size,
                beam_search_parameters: BeamSearchParameters {
                    beam_size: initial_beam_size,
                    keep_all_layers,
                    parameters: Parameters::<Integer> {
                        primal_bound,
                        time_limit,
                        get_all_solutions: false,
                        quiet,
                        initial_registry_capacity: None,
                        expansion_limit,
                    },
                },
            };

            let solver = create_user_priority_cabs(
                rust_model,
                parameters,
                bound_evaluator_type,
                user_evaluators,
            );

            // 🟢 Return struct with named fields
            Ok(CustomDualBoundCabsPy {
                solver: WrappedSolver::Int(solver),
                _guard: guard, 
            })
        }
    }

    /// search()
    #[pyo3(signature = ())]
    fn search(&mut self) -> PyResult<SolutionPy> {
        let result = self.solver.search();
        // 🟢 Write stats immediately after search finishes
        self._guard.write_stats();
        result
    }

    /// search_next()
    #[pyo3(signature = ())]
    fn search_next(&mut self) -> PyResult<(SolutionPy, bool)> {
        let result = self.solver.search_next();
        // 🟢 Write stats immediately after search finishes
        self._guard.write_stats();
        result
    }
}