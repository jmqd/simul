use simul::agent::{periodic_consuming_agent, periodic_producing_agent, Agent};
use simul::experiment::experiment_by_annealing_objective;
use simul::*;

/// Given a producer with a fixed period, returns producer-consumer two Agent
/// configurations (where only the consumer varies).
///
/// The consumer randomly varies between 0 and consumer_max_period. The two
/// agents are always named "producer" and "consumer".
fn periodic_agent_generator_fixed_producer(
    producer_period: DiscreteTime,
    consumer_max_period: DiscreteTime,
) -> impl Fn() -> Vec<Box<dyn Agent>> {
    move || {
        let consumer_period = rand::random::<u32>() % (consumer_max_period + 1) as u32;
        let producer_agent = periodic_producing_agent(
            "producer".to_string(),
            producer_period,
            "consumer".to_string(),
        );
        let consumer_agent =
            periodic_consuming_agent("consumer".to_string(), consumer_period as u64);
        vec![producer_agent, consumer_agent]
    }
}

/// Sandbox for running a simulated annealing experiment.
fn run_experiment() {
    let halt_condition = |s: &Simulation| {
        s.agents
            .iter()
            .find(|a| a.state().id == "consumer")
            .expect("No consumer agent?")
            .state()
            .consumed
            .len()
            > 10
    };

    // Creates an agent generator w/ a fixed producer at interval 2 and a
    // consumer whose period randomly varies between [0, 10]
    let agent_generator = periodic_agent_generator_fixed_producer(2, 10);

    // SimulationParameters generator that holds all else static except for agents.
    let simulation_parameters_generator = move || SimulationParameters {
        agents: agent_generator(),
        halt_check: halt_condition,
        enable_agent_asleep_cycles_metric: true,
        ..Default::default()
    };

    // This is the objective function which we're trying to approximately
    // optimize via simulated experiment.  This objective function tries to find
    // the simulation that completed the Simulation in the least time and
    // doesn't "overuse" consumer period.
    //
    // In this specific example, negative simulation time means that we're
    // optimizing for the simulation that completes the fastest. If this were
    // the only parameter, there would be two solutions: 0 and 1. Because the
    // producer is fixed at a period of 2, a consumer with period 0 or 1 can
    // sufficiently keep up with that producer.
    //
    // By subtracting the asleep cycles of the consumer from the simulation
    // time, we're looking to get the fastest simulation time whiling maximizing
    // sleep time / minimizing resource usage.
    let objective_fn =
        |s: &Simulation| -(s.time as i64) - s.asleep_cycle_count("consumer").unwrap() as i64;

    let replications_limit = 100;

    // Run the simulation 100 different times, randomly varying the agent
    // configuration, and return the one that maximized the objective function.
    let approx_optimal = experiment_by_annealing_objective(
        simulation_parameters_generator,
        replications_limit,
        objective_fn,
    );

    println!("{:?}", approx_optimal.unwrap().agents.iter());
}

fn main() {
    run_experiment();
}
