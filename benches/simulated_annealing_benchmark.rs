use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use rand::Rng;
use simul::agent::{periodic_consuming_agent, periodic_producing_agent};
use simul::experiment::{simulated_annealing_search, ObjectiveScore};
use simul::*;

const PRODUCER_PERIOD: u64 = 2;
const MAX_CONSUMER_PERIOD: u64 = 10;
const HALT_CONSUMED_COUNT: usize = 10;
const REPLICATIONS_LIMIT: u32 = 1000;
const GEOMETRIC_COOLING_RATE: f64 = 0.99;
const STARTING_TURBULENCE: f64 = 1.0;

/// Extracts the consumer's period from the SimulationParameters.
fn get_consumer_period(params: &SimulationParameters) -> u64 {
    // We assume the consumer is the second agent (index 1) as set up in the generator.
    if let Some(AgentInitializer { .. }) = params.agent_initializers.get(1) {
        let cost = params.agent_initializers.get(1).unwrap().agent.cost();
        return (-cost) as u64;
    }

    0
}

/// Creates a full SimulationParameters object from a consumer period.
fn build_sim_params(consumer_period: u64) -> SimulationParameters {
    let halt_condition = |s: &Simulation| {
        s.find_by_name("consumer").unwrap().state.consumed.len() > HALT_CONSUMED_COUNT
    };

    let producer_agent = periodic_producing_agent(
        "producer".to_string(),
        PRODUCER_PERIOD,
        "consumer".to_string(),
    );
    let consumer_agent = periodic_consuming_agent("consumer".to_string(), consumer_period);

    SimulationParameters {
        agent_initializers: vec![producer_agent, consumer_agent],
        halt_check: halt_condition,
        enable_agent_asleep_cycles_metric: true,
        ..Default::default()
    }
}

/// Randomly changes the consumer's period by +/- 1, keeping it within [0, MAX_CONSUMER_PERIOD].
fn perturb_consumer_period(current_params: &SimulationParameters) -> SimulationParameters {
    let mut rng = rand::rng();
    let old_period = get_consumer_period(current_params);
    let mut new_period = old_period;

    for _ in 0..2 {
        let delta: i64 = if rng.random_bool(0.5) { 1 } else { -1 };
        let attempted_period = (old_period as i64 + delta)
            .max(0)
            .min(MAX_CONSUMER_PERIOD as i64);

        if attempted_period as u64 != old_period {
            new_period = attempted_period as u64;
            break;
        }
    }

    build_sim_params(new_period)
}

/// The objective function we want to maximize.
/// Maximize: (Negative Sim Time) + (Negative Consumer Cost)
/// This finds the fastest simulation that doesn't "over-consume" resources.
fn objective_fn(s: &Simulation) -> ObjectiveScore {
    let consumer_agent = s
        .find_by_name("consumer")
        .expect("Consumer agent missing in simulation");

    // The cost function for periodic agents is defined as -period.
    // So, s.agent.cost() gives a negative value.
    let consumer_cost = consumer_agent.agent.cost();

    // Score = -(Total Time) + (Consumer Cost)
    // We want the cost (period) to be small (close to 0) which means the cost value
    // should be close to 0 (since it's negative).
    -(s.time() as f64) + consumer_cost
}

/// Geometric Cooling Schedule: T(k) = T_start * alpha^k
/// Chaotic flux decreases rapidly, favoring convergence.
fn geometric_chaotic_flux_schedule(k: u32) -> f64 {
    STARTING_TURBULENCE * GEOMETRIC_COOLING_RATE.powi(k as i32)
}

#[allow(dead_code)]
fn run_annealing_experiment() -> Option<SimulationParameters> {
    // A generator for the starting point of the search (e.g., a random period)
    let initial_params_generator = || {
        let start_period = rand::rng().random_range(1..=MAX_CONSUMER_PERIOD);
        build_sim_params(start_period)
    };

    let approx_optimal_params = simulated_annealing_search(
        initial_params_generator,
        perturb_consumer_period,
        objective_fn,
        geometric_chaotic_flux_schedule,
        REPLICATIONS_LIMIT,
    );

    match approx_optimal_params.as_ref() {
        Some(params) => {
            let period = get_consumer_period(params);
            println!("Simulated Annealing found an approximate optimal configuration:");
            println!("Optimal Consumer Period: {}", period);
            println!("Producer is fixed at Period: {}", PRODUCER_PERIOD);

            let mut final_sim = Simulation::new(params.clone());
            final_sim.run();
            println!("Final Score: {:.2}", objective_fn(&final_sim));
            println!("Final Simulation Time: {}", final_sim.time());

            // Expected Result: Producer period is 2. Consumer only needs period 0 or 1
            // to keep up. The optimizer should prefer Period 1 (cost -1) over
            // Period 0 (cost 0) because of the cost penalty in the objective function.
        }
        None => println!("Simulated Annealing failed to find any solution."),
    }

    approx_optimal_params
}

fn simulated_annealing_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulated_annealing_experiment_benchmarks");

    let initial_params_generator = || {
        let start_period = rand::rng().random_range(1..=MAX_CONSUMER_PERIOD);
        build_sim_params(start_period)
    };

    let perturb_fn = perturb_consumer_period;
    let obj_fn = objective_fn;
    let flux_fn = geometric_chaotic_flux_schedule;

    group.bench_function("simple_simulated_annealing_experiment_1000_steps", |b| {
        b.iter(|| {
            let result = simulated_annealing_search(
                initial_params_generator,
                perturb_fn,
                obj_fn,
                flux_fn,
                REPLICATIONS_LIMIT,
            );

            assert!(result.is_some());
        })
    });

    group.finish();
}

criterion_group!(benches, simulated_annealing_bench);
criterion_main!(benches);
