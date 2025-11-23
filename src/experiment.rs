use crate::Simulation;
use crate::SimulationParameters;
use rand::Rng;

/// ObjectiveScore is a measure of how a Simulation performed according to an
/// objective function. This is used to find approximate global optimizations.
pub type ObjectiveScore = f64;

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
pub fn monte_carlo_experiment(
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

/// Looks to find a global optimum by simulated annealing, a probabilistic
/// approximation method.
///
/// Note: This code goes deep on an analogy of using entropy, chaos, turbulence,
/// and parallel worlds to make it easier (for me) to follow. If you imagine
/// that by running this experiment, we're harnessing chaos that diminishes at
/// each step, to phase shift into parallel worlds, but before we step through
/// the portal, we get to see how good that world "looks" and choose whether to
/// step into it, and we sometimes take a gamble on worlds that "look bad", I
/// hope that you too might find this analogy easier to understand.
pub fn simulated_annealing_experiment(
    initial_parameters_generator: impl Fn() -> SimulationParameters,
    perturb_function: impl Fn(&SimulationParameters) -> SimulationParameters,
    objective_function: impl Fn(&Simulation) -> ObjectiveScore,
    summon_chaotic_flux: impl Fn(u32) -> f64,
    replications_limit: u32,
) -> Option<SimulationParameters> {
    let mut current_params = initial_parameters_generator();
    let mut best_params = current_params.clone();

    // Let's get our initial starting score to start the experiment.
    let mut current_world = Simulation::new(current_params.clone());
    current_world.run();
    let mut current_score = objective_function(&current_world);
    let mut best_score = current_score;

    for chaotic_mana in (1..=replications_limit).rev() {
        // As the experiment progress, our chaotic_flux and mana decreases.
        // Chaotic flux is what enables us to explore instead of exploit.
        // It enables us to climb steep gradients and get out of local minima.
        // The more chaotic flux we've summoned, the less we can summon -- we
        // start to settle into a local cluster of good-looking worlds.
        let k = replications_limit - chaotic_mana + 1;
        let chaotic_flux = summon_chaotic_flux(k);

        // Given our current state, find a parallel world of params.
        let new_params = perturb_function(&current_params);

        // Run the simulation for this new parallel world.
        let mut parallel_world = Simulation::new(new_params.clone());
        parallel_world.run();
        let new_score = objective_function(&parallel_world);

        // Whether we choose to step into this new parallel world is a function
        // of how good it looks, and how much chaotic flux we have left. If we
        // have a lot of chaotic flux, we may choose to step into a worse world.
        let delta_goodness: f64 = current_score - new_score;
        let explore_parallel_world = if delta_goodness < 0.0 {
            true
        } else {
            // If the new world is worse, there's still a chance we want to explore it.
            let acceptance_probability = (-delta_goodness / chaotic_flux).exp();
            rand::rng().random_range(0.0..1.0) < acceptance_probability
        };

        if explore_parallel_world {
            current_params = new_params;
            current_score = new_score;

            if current_score > best_score {
                best_score = current_score;
                best_params = current_params.clone();
            }
        }
    }

    Some(best_params)
}
