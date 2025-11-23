use crate::Simulation;
use crate::SimulationParameters;

/// ObjectiveScore is a measure of how a Simulation performed according to an
/// objective function. This is used to find approximate global optimizations.
pub type ObjectiveScore = i64;

/// Given a function that generates various configurations of
/// SimulationParameters, run many simulation replications with varying
/// SimulationParameters. The parameters are varied by calling the generator.
/// The generator may, for example, randomly vary multiple fields of the
/// parameters. This function tries to approximate the globally optimal
/// parameters by running the simulation as many times as you specify
/// (replications_limit), and finds the Simulation that yielded the highest
/// score from the provided objective_function.
///
/// The simplest and most common objective function is to return negative
/// simulation time. An objective function that returns negative simulation time
/// will find the Simulation that completed in the least ticks of DiscreteTime.
pub fn experiment_by_annealing_objective(
    mut simulation_parameters_generator: impl FnMut() -> SimulationParameters,
    replications_limit: u32,
    objective_function: impl Fn(&Simulation) -> ObjectiveScore,
) -> Option<Simulation> {
    let mut approx_optimal_simulation: Option<Simulation> = None;
    let mut high_score = ObjectiveScore::MIN;

    for _ in 0..replications_limit {
        let mut simulation = Simulation::new(simulation_parameters_generator());
        simulation.run();

        let score = objective_function(&simulation);
        if score > high_score {
            approx_optimal_simulation = Some(simulation.clone());
            high_score = score;
        }
    }

    approx_optimal_simulation
}
