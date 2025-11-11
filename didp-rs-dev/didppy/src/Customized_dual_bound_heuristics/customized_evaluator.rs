use crate::model::StatePy;
use dypdl::prelude::*;
// We need all these types for conversion
use dypdl::variable_type::{Continuous, Integer, Numeric, OrderedContinuous};
use dypdl_heuristic_search::search_algorithm::StateInRegistry;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::fmt;
use std::rc::Rc;

/// Creates a combined dual bound evaluator that evaluates both:
/// 1. Expression-based dual bounds (via model.eval_dual_bound)
/// 2. A single Python callback function (passed as py_func)
///
/// This function *always* returns an Option<OrderedContinuous> (a float-based heuristic)
/// because that is what UserPriorityCABS expects, even for Integer problems.
pub fn create_combined_evaluator_with_py_func<T>( // T is the model's cost type
    model: Rc<Model>,
    py_func: Py<PyAny>,
) -> impl Fn(&StateInRegistry) -> Option<OrderedContinuous>
where
    T: Numeric + Ord + fmt::Display + 'static,
{
    move |state: &StateInRegistry| {
        // 1. Evaluate expression-based dual bounds (as type T)
        let expr_bound: Option<OrderedContinuous> = model
            //
            // THIS IS THE FIX: Changed from <T, _> to <_, T>
            //
            .eval_dual_bound::<_, T>(state)
            // Convert the bound (whether Integer or Continuous) to a float
            .map(|v| OrderedContinuous::from(v.to_continuous()));

        // 2. Evaluate the single Python callback (must return float)
        let python_bound: Option<OrderedContinuous> = Python::with_gil(|py| {
            let state_py = StatePy::from(State::from(state.clone()));
            let args = PyTuple::new(py, &[state_py.into_py(py)]);
            
            match py_func.call1(py, args) {
                Ok(result) => {
                    // Python function MUST return a float or Option<float>
                    if let Ok(f_val) = result.extract::<Continuous>(py) {
                        Some(OrderedContinuous::from(f_val))
                    }
                    // Or Option<float>
                    else if let Ok(Some(f_val)) = result.extract::<Option<Continuous>>(py) {
                        Some(OrderedContinuous::from(f_val))
                    }
                    // Handle if Python returns an int
                    else if let Ok(i_val) = result.extract::<Integer>(py) {
                        Some(OrderedContinuous::from(i_val as Continuous))
                    }
                    // Or Option<int>
                    else if let Ok(Some(i_val)) = result.extract::<Option<Integer>>(py) {
                        Some(OrderedContinuous::from(i_val as Continuous))
                    }
                    else { None }
                }
                Err(e) => {
                    eprintln!("Python dual bound callback failed:");
                    e.print(py);
                    None
                }
            }
        });

        // 3. Combine expression and Python bounds
        match (expr_bound, python_bound) {
            (Some(e), Some(p)) => {
                match model.reduce_function {
                    ReduceFunction::Min => Some(e.max(p)), // Take max for minimization
                    ReduceFunction::Max => Some(e.min(p)), // Take min for maximization
                    _ => Some(e), // fallback
                }
            }
            (Some(e), None) => Some(e),
            (None, Some(p)) => Some(p),
            (None, None) => None,
        }
    }
}

// Your test code, now corrected to work with OrderedContinuous
#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::IntegerExpression;
    use dypdl_heuristic_search::search_algorithm::StateInRegistry;
    use pyo3::types::PyModule;
    use std::rc::Rc;

    #[test]
    fn test_combined_evaluator_logic() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Python function now returns floats
            let py_code = r#"
def returns_20(state):
    return 20.0

def returns_none(state):
    return None
"#;
            let py_module = PyModule::from_code(py, py_code, "test_module", "test_module").unwrap();
            let py_func_returns_20: Py<PyAny> = py_module.getattr("returns_20").unwrap().into();
            let py_func_returns_none: Py<PyAny> = py_module.getattr("returns_none").unwrap().into();

            // --- Setup ---
            let mut model = Model::default(); // Integer model
            model
                .add_dual_bound(IntegerExpression::Constant(10))
                .unwrap();
            let model_rc = Rc::new(model);
            let state_in_registry = StateInRegistry::from(model_rc.target.clone());

            // === SCENARIO 1: Both Rust and Python bounds exist (Minimization problem) ===
            // We call with <Integer> because the *model* is integer based
            let evaluator1 = create_combined_evaluator_with_py_func::<Integer>(
                model_rc.clone(),
                py_func_returns_20.clone(),
            );
            // Rust bound is 10.0, Python bound is 20.0. For minimization, we take max (20.0).
            assert_eq!(evaluator1(&state_in_registry), Some(OrderedContinuous::from(20.0)));
            println!("✅ Scenario 1 Passed: Both bounds exist.");

            // === SCENARIO 2: Only Rust expression bound exists ===
            let evaluator2 = create_combined_evaluator_with_py_func::<Integer>(
                model_rc.clone(),
                py_func_returns_none.clone(),
            );
            // Should return only the Rust bound (10.0).
            assert_eq!(evaluator2(&state_in_registry), Some(OrderedContinuous::from(10.0)));
            println!("✅ Scenario 2 Passed: Only Rust bound exists.");
            
            // === SCENARIO 3: Only Python bound exists ===
            let empty_model_rc = Rc::new(Model::default());
            let empty_state = StateInRegistry::from(empty_model_rc.target.clone());

            let evaluator3 = create_combined_evaluator_with_py_func::<Integer>(
                empty_model_rc.clone(),
                py_func_returns_20.clone(),
            );
            // Should return only the Python bound (20.0).
            assert_eq!(evaluator3(&empty_state), Some(OrderedContinuous::from(20.0)));
            println!("✅ Scenario 3 Passed: Only Python bound exists.");

            // === SCENARIO 4: Neither bound exists ===
            let empty_model_2_rc = Rc::new(Model::default());
            let empty_state_2 = StateInRegistry::from(empty_model_2_rc.target.clone());

            let evaluator4 = create_combined_evaluator_with_py_func::<Integer>(
                empty_model_2_rc.clone(),
                py_func_returns_none.clone(),
            );
            // Should return None.
            assert_eq!(evaluator4(&empty_state_2), None);
            println!("✅ Scenario 4 Passed: Neither bound exists.");
        });
    }
}