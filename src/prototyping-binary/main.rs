use log::info;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_distr::Poisson;
use simul::agent::*;
use simul::experiment::*;
use simul::message::Message;
use simul::*;
use std::collections::HashMap;
use std::default;
use std::path::PathBuf;

/// Just a janky `++` operator.
fn inc(num: &mut usize) -> usize {
    *num += 1;
    return *num;
}

/// Sandbox for running a simulated annealing experiment.
fn run_experiment() {
    let halt_condition = |s: &Simulation| {
        s.agents
            .iter()
            .find(|a| a.name == "consumer")
            .expect("No consumer agent?")
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
    // By subtracting the period of the consumer from the simulation time, we're
    // looking to get the fastest simulation time without wasting any resources.
    let objective_fn = |s: &Simulation| {
        -(s.time as i64)
            + s.agents
                .iter()
                .find(|a| a.name == "consumer")
                .as_ref()
                .unwrap()
                .extensions
                .as_ref()
                .unwrap()
                .period
                .unwrap() as i64
    };

    let replications_limit = 100;

    // Run the simulation 100 different times, randomly varying the agent
    // configuration, and return the one that maximized the objective function.
    let approx_optimal = experiment_by_annealing_objective(
        simulation_parameters_generator,
        replications_limit,
        objective_fn,
    );

    println!(
        "{:?}",
        approx_optimal
            .unwrap()
            .agents
            .iter()
            .map(|a| (&a.name, &a.extensions))
    );
}

fn normal_9_ball_simulation(lucky_pct: f32) -> String {
    let halt_condition = |s: &Simulation| {
        for a in s.agents.iter() {
            let ext = a.extensions.as_ref().unwrap();
            if ext.score >= ext.winning_threshold {
                return true;
            }
        }

        return false;
    };

    let mut ext = AgentExtensions::default();
    ext.winning_threshold = 6;

    let jordan = Agent {
        consumption_fn: |a: &mut Agent, t: DiscreteTime| {
            if let Some(message) = a.queue.pop_front() {
                let ext = a.extensions.as_mut()?;
                let choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
                let weights = [1, 5, 9, 11, 8, 4, 2, 2, 2, 1, 1];
                let dist = WeightedIndex::new(&weights).unwrap();
                let mut rng = thread_rng();

                let mut balls_to_run = choices[dist.sample(&mut rng)];

                let mut ball = message.current_ball;

                while balls_to_run > 0 {
                    balls_to_run -= 1;

                    if ball == 9 {
                        ext.score += 1;
                        ball = 1;
                    } else {
                        ball += 1;
                    }
                }

                Some(vec![Message {
                    queued_time: t,
                    completed_time: None,
                    source: "opp".to_string(),
                    destination: "opp".to_string(),
                    current_ball: ball,
                }])
            } else {
                None
            }
        },
        name: "jordan".to_string(),
        extensions: Some(ext.clone()),
        queue: vec![Message::default()].into(),
        ..Default::default()
    };

    let opp = Agent {
        lucky_pct: lucky_pct,
        consumption_fn: |a: &mut Agent, t: DiscreteTime| {
            if let Some(message) = a.queue.pop_front() {
                let ext = a.extensions.as_mut()?;
                let choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
                let weights = [1, 4, 9, 11, 6, 4, 1, 1, 1, 1, 0];
                let dist = WeightedIndex::new(&weights).unwrap();
                let mut rng = thread_rng();

                let mut balls_to_run = choices[dist.sample(&mut rng)];

                let mut ball = message.current_ball;

                while balls_to_run > 0 {
                    balls_to_run -= 1;

                    if ball == 9 {
                        ext.score += 1;
                        ball = 1;
                    } else {
                        ball += 1;
                    }
                }

                let lucky_chance = rng.gen_range(0.0..1.0);

                let next_turn = if lucky_chance > (1.0 - a.lucky_pct) {
                    "opp".to_owned()
                } else {
                    "jordan".to_owned()
                };

                Some(vec![Message {
                    queued_time: t,
                    completed_time: None,
                    source: "opp".to_string(),
                    destination: next_turn,
                    current_ball: ball,
                }])
            } else {
                None
            }
        },
        name: "opp".to_string(),
        extensions: Some(ext.clone()),
        ..Default::default()
    };

    // SimulationParameters generator that holds all else static except for agents.
    let simulation_parameters_generator = move || SimulationParameters {
        agents: vec![jordan, opp],
        halt_check: halt_condition,
        ..Default::default()
    };

    let mut sim = Simulation::new(simulation_parameters_generator());
    sim.run();

    sim.agents
        .iter()
        .find(|a| {
            a.extensions.as_ref().unwrap().score >= a.extensions.as_ref().unwrap().winning_threshold
        })
        .map(|a| a.name.clone())
        .unwrap()
}

fn nine_ball_apa_rules_simulation(lucky_pct: f32) -> String {
    let halt_condition = |s: &Simulation| {
        for a in s.agents.iter() {
            let ext = a.extensions.as_ref().unwrap();
            if ext.score >= ext.winning_threshold {
                return true;
            }
        }

        return false;
    };

    let mut ext = AgentExtensions::default();
    ext.winning_threshold = 55;

    let jordan = Agent {
        consumption_fn: |a: &mut Agent, t: DiscreteTime| {
            if let Some(message) = a.queue.pop_front() {
                let ext = a.extensions.as_mut()?;
                let choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
                let weights = [1, 5, 9, 11, 8, 4, 2, 2, 2, 1, 1];
                let dist = WeightedIndex::new(&weights).unwrap();
                let mut rng = thread_rng();
                ext.score += choices[dist.sample(&mut rng)];

                Some(vec![Message {
                    queued_time: t,
                    completed_time: None,
                    source: "jordan".to_string(),
                    destination: "opp".to_string(),
                    ..Default::default()
                }])
            } else {
                None
            }
        },
        name: "jordan".to_string(),
        extensions: Some(ext.clone()),
        queue: vec![Message::default()].into(),
        ..Default::default()
    };

    let opp = Agent {
        lucky_pct,
        consumption_fn: |a: &mut Agent, t: DiscreteTime| {
            if let Some(message) = a.queue.pop_front() {
                let ext = a.extensions.as_mut()?;
                let choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
                let weights = [1, 4, 9, 11, 6, 4, 1, 1, 1, 1, 0];
                let dist = WeightedIndex::new(&weights).unwrap();
                let mut rng = thread_rng();

                ext.score += choices[dist.sample(&mut rng)];

                let lucky_chance = rng.gen_range(0.0..1.0);

                // We model luck as
                let next_turn = if lucky_chance > (1.0 - a.lucky_pct) {
                    "opp".to_owned()
                } else {
                    "jordan".to_owned()
                };

                Some(vec![Message {
                    queued_time: t,
                    completed_time: None,
                    source: "opp".to_string(),
                    destination: next_turn,
                    ..Default::default()
                }])
            } else {
                None
            }
        },
        name: "opp".to_string(),
        extensions: Some(ext.clone()),
        ..Default::default()
    };

    // SimulationParameters generator that holds all else static except for agents.
    let simulation_parameters_generator = move || SimulationParameters {
        agents: vec![jordan, opp],
        halt_check: halt_condition,
        ..Default::default()
    };

    let mut sim = Simulation::new(simulation_parameters_generator());
    sim.run();

    sim.agents
        .iter()
        .find(|a| {
            a.extensions.as_ref().unwrap().score >= a.extensions.as_ref().unwrap().winning_threshold
        })
        .map(|a| a.name.clone())
        .unwrap()
}

/// Given a producer with a fixed period, returns producer-consumer two Agent
/// configurations (where only the consumer varies).
///
/// The consumer randomly varies between 0 and consumer_max_period. The two
/// agents are always named "producer" and "consumer".
fn periodic_agent_generator_fixed_producer(
    producer_period: DiscreteTime,
    consumer_max_period: DiscreteTime,
) -> impl Fn() -> Vec<Agent> {
    move || {
        let consumer_period = rand::random::<u32>() % (consumer_max_period + 1) as u32;
        let producer_agent = periodic_producing_agent("producer", producer_period, "consumer");
        let consumer_agent = periodic_consuming_agent("consumer", consumer_period as u64);
        vec![producer_agent, consumer_agent]
    }
}

/// Note, this main.rs binary file is just for library prototyping at the moment.
fn main() {
    for pct in [0.00, 0.20, 0.40, 0.50].into_iter() {
        let mut count: HashMap<String, u32> = HashMap::new();
        for _ in 0..1024 {
            *count
                .entry(nine_ball_apa_rules_simulation(pct))
                .or_default() += 1;
        }

        println!(
            "(APA match skill 7 vs 7) Better player win percentage, {:.2}% luck factor for opponent: {:.2}",
            pct * 100.0,
            count["jordan"] as f32 / (count["jordan"] + count["opp"]) as f32
        );
    }

    for pct in [0.00, 0.20, 0.40, 0.50].into_iter() {
        let mut count: HashMap<String, u32> = HashMap::new();
        for _ in 0..1024 {
            *count.entry(normal_9_ball_simulation(pct)).or_default() += 1;
        }

        println!(
            "(set match race to 6) Better player win percentage, {:.2}% luck factor for opponent: {:.2}",
            pct * 100.0,
            count["jordan"] as f32 / (count["jordan"] + count["opp"]) as f32
        );
    }
}
