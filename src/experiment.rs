use crate::Simulation;
use crate::SimulationParameters;

/// ObjectiveScore is a measure of how a Simulation performed according to an
/// objective function. This is used to find approximate global optimazations.
type ObjectiveScore = i64;

/// Given a function that generates various configurations of Agents, continue
/// running simulations with various different agent configurations (via calling
/// the generator) to try to determine the simulation that is approximately optimal
/// by maximizing the provided objective_function.
///
/// The simplest and most common objective function is negative simulation time.
/// An objective function that returns negative simulation time will find the
/// fastest simulation.
///
/// The halt_condition for the simulation is also provided by the user.
pub fn experiment_by_annealing_objective(
    simulation_parameters_generator: impl Fn() -> SimulationParameters,
    replications_limit: u32,
    objective_function: impl Fn(&Simulation) -> ObjectiveScore,
) -> Option<Simulation> {
    let mut approx_optimal_simulation: Option<Simulation> = None;
    let mut high_score = ObjectiveScore::MIN;

    for _ in 0..replications_limit {
        let mut simulation = Simulation::from_parameters(simulation_parameters_generator());
        simulation.run();

        let score = objective_function(&simulation);
        if score > high_score {
            approx_optimal_simulation = Some(simulation.clone());
            high_score = score;
        }
    }

    approx_optimal_simulation
}
