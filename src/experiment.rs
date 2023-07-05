use crate::Simulation;
use crate::Agent;

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
    agent_generator: impl Fn() -> Vec<Agent>,
    halt_condition: fn(&Simulation) -> bool,
    simulation_limit: u32,
    objective_function: impl Fn(&Simulation) -> i64,
) -> Option<Simulation> {
    let mut approx_optimal_simulation: Option<Simulation> = None;
    let mut high_score = std::i64::MIN;

    for _ in 0..simulation_limit {
        let agents = agent_generator();
        let mut simulation = Simulation::new(agents, 0, true, halt_condition);
        simulation.run();

        let score = objective_function(&simulation);
        println!(
            "period = {:?}, score = {}",
            simulation
                .agents
                .iter()
                .find(|a| a.name == "consumer")
                .unwrap()
                .common_traits
                .as_ref()
                .unwrap()
                .period,
            score
        );
        if score > high_score {
            approx_optimal_simulation = Some(simulation.clone());
            high_score = score;
        }
    }

    approx_optimal_simulation
}
