use crate::model::StatePy;
use dypdl::prelude::*;
// We need all these types for conversion
use dypdl::variable_type::{Continuous, Integer, Numeric, OrderedContinuous};
use dypdl_heuristic_search::search_algorithm::StateInRegistry;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::fmt;
use std::rc::Rc;
// 🟢 Imports for Timing and File I/O
use std::time::Instant;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::fs::OpenOptions;
use std::io::Write;

// --- 1. Define a Struct to handle stats ---
pub struct TimingStats {
    total_us: Arc<AtomicU64>,
    count: Arc<AtomicUsize>,
    print_stats: bool,
}

// 🟢 Method to explicitly write stats (No Drop magic)
impl TimingStats {
    pub fn write_stats(&self) {
        if self.print_stats {
            let total = self.total_us.load(Ordering::Relaxed);
            let count = self.count.load(Ordering::Relaxed);
            
            if count > 0 {
                let avg_ms = (total as f64 / count as f64) / 1000.0;
                let total_sec = total as f64 / 1_000_000.0;
                
                let message = format!(
                    "\n--- Rust-Python Bridge Stats ---\n\
                    Timestamp: {:?}\n\
                    Total calls: {}\n\
                    Total time in Python: {:.4}s\n\
                    Avg time per call: {:.4} ms\n\
                    --------------------------------\n",
                    Instant::now(), count, total_sec, avg_ms
                );

                // Write to "Python_and_Rust_bridge_time_stats.txt"
                // truncate(true) ensures we overwrite the old file every time
                let file_result = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true) 
                    .open("Python_and_Rust_bridge_time_stats.txt");

                match file_result {
                    Ok(mut file) => {
                        if let Err(e) = writeln!(file, "{}", message) {
                            eprintln!("Failed to write stats file: {}", e);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to open stats file: {}", e);
                    }
                }
            }
        }
    }
}

/// Creates a combined dual bound evaluator AND a timing guard.
/// Returns: (The Closure, The Guard)
pub fn create_combined_evaluator_with_py_func<T>( // T is the model's cost type
    model: Rc<Model>,
    py_func: Py<PyAny>,
    print_stats: bool,
) -> (impl Fn(&StateInRegistry) -> Option<OrderedContinuous>, TimingStats)
where
    T: Numeric + Ord + fmt::Display + 'static,
{
    // Initialize shared counters
    let total_us = Arc::new(AtomicU64::new(0));
    let count = Arc::new(AtomicUsize::new(0));

    // Create the Guard (Owner 1 of the data)
    let guard = TimingStats {
        total_us: total_us.clone(),
        count: count.clone(),
        print_stats,
    };

    // Clone Arcs for the closure (Owner 2 of the data)
    let total_us_closure = total_us.clone();
    let count_closure = count.clone();

    let closure = move |state: &StateInRegistry| {
        // 1. Evaluate expression-based dual bounds (as type T)
        let expr_bound: Option<OrderedContinuous> = model
            .eval_dual_bound::<_, T>(state)
            .map(|v| OrderedContinuous::from(v.to_continuous()));

        // 2. Evaluate the single Python callback (must return float)
        // 🟢 START TIMER
        let start = Instant::now();

        let python_bound: Option<OrderedContinuous> = Python::with_gil(|py| {
            let state_py = StatePy::from(State::from(state.clone()));
            let args = PyTuple::new(py, &[state_py.into_py(py)]);
            
            match py_func.call1(py, args) {
                Ok(result) => {
                    // Python function MUST return a float or Option<float>
                    if let Ok(f_val) = result.extract::<Continuous>(py) {
                        Some(OrderedContinuous::from(f_val))
                    } else if let Ok(Some(f_val)) = result.extract::<Option<Continuous>>(py) {
                        Some(OrderedContinuous::from(f_val))
                    } else if let Ok(i_val) = result.extract::<Integer>(py) {
                        Some(OrderedContinuous::from(i_val as Continuous))
                    } else if let Ok(Some(i_val)) = result.extract::<Option<Integer>>(py) {
                        Some(OrderedContinuous::from(i_val as Continuous))
                    } else { None }
                }
                Err(e) => {
                    eprintln!("Python dual bound callback failed:");
                    e.print(py);
                    None
                }
            }
        });

        // 🟢 STOP TIMER & UPDATE STATS
        let elapsed = start.elapsed().as_micros() as u64;
        total_us_closure.fetch_add(elapsed, Ordering::Relaxed);
        count_closure.fetch_add(1, Ordering::Relaxed);

        // 3. Combine expression and Python bounds
        match (expr_bound, python_bound) {
            (Some(e), Some(p)) => {
                match model.reduce_function {
                    ReduceFunction::Min => Some(e.max(p)),
                    ReduceFunction::Max => Some(e.min(p)),
                    _ => Some(e),
                }
            }
            (Some(e), None) => Some(e),
            (None, Some(p)) => Some(p),
            (None, None) => None,
        }
    };

    // Return BOTH the closure and the guard
    (closure, guard)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::IntegerExpression;
    use dypdl_heuristic_search::search_algorithm::StateInRegistry;
    use pyo3::types::PyModule;
    use std::rc::Rc;

    #[test]
    fn test_timing_stats_integration() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let py_code = r#"
import time
def mock_heuristic(state):
    time.sleep(0.0001) 
    return 100.0
"#;
            let py_module = PyModule::from_code(py, py_code, "test_module", "test_module")
                .expect("Failed to create Python module");
            let py_func: Py<PyAny> = py_module.getattr("mock_heuristic")
                .expect("Failed to get function")
                .into();

            let mut model = Model::default();
            model.add_dual_bound(IntegerExpression::Constant(10)).unwrap();
            let model_rc = Rc::new(model);
            let state_in_registry = StateInRegistry::from(model_rc.target.clone());

            println!("\n>>> TEST START: Creating Evaluator...");

            let (evaluator, guard) = create_combined_evaluator_with_py_func::<Integer>(
                model_rc.clone(),
                py_func.clone(),
                true,
            );

            let iterations = 50;
            for i in 0..iterations {
                let result = evaluator(&state_in_registry);
                assert_eq!(result, Some(OrderedContinuous::from(100.0)), "Failed at iter {}", i);
            }
            
            // Manually call write_stats to test it
            guard.write_stats();

            println!(">>> TEST END: Check 'Python_and_Rust_bridge_time_stats.txt'.\n");
        });
    }
}