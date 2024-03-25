use log::info;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_distr::Poisson;
use simul::agent::*;
use simul::experiment::*;
use simul::message::Interrupt;
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

#[derive(Debug, Clone)]
struct NineBallPlayer<const N: usize> {
    luck_chance: f32,
    run_out_choices: [usize; N],
    run_out_weights: [usize; N],
    winning_threshold: u8,
    score: u8,
    state: AgentState,
    opponent_name: String,
}

impl<const N: usize> Agent for NineBallPlayer<{ N }> {
    fn process(
        &mut self,
        simulation_state: SimulationState,
        msg: &Message,
    ) -> Option<Vec<Message>> {
        let mut rng = thread_rng();
        let dist = WeightedIndex::new(&self.run_out_weights).unwrap();
        let mut balls_to_run = self.run_out_choices[dist.sample(&mut rng)];
        let mut ball = msg.current_ball;

        while balls_to_run > 0 {
            balls_to_run -= 1;

            if ball == 9 {
                self.score += 1;
                ball = 1;
            } else {
                ball += 1;
            }
        }

        if self.score >= self.winning_threshold {
            return Some(vec![Message {
                source: self.state().id.clone(),
                interrupt: Some(Interrupt::HaltSimulation("won".to_string())),
                ..Default::default()
            }]);
        }

        // If the opponent gets lucky, they get another turn.
        let next_turn = if rng.gen_range(0.0..1.0) > (1.0 - self.luck_chance) {
            self.state().id.clone()
        } else {
            self.opponent_name.clone()
        };

        Some(vec![Message {
            queued_time: simulation_state.time,
            completed_time: None,
            source: self.state().id.to_string(),
            destination: next_turn,
            current_ball: ball,
            ..Default::default()
        }])
    }

    fn state(&self) -> &AgentState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut AgentState {
        &mut self.state
    }
}

#[derive(Debug, Clone)]
struct ApaNineBallPlayer<const N: usize> {
    luck_chance: f32,
    run_out_choices: [usize; N],
    run_out_weights: [usize; N],
    winning_threshold: u8,
    score: u8,
    state: AgentState,
    opponent_name: String,
}

impl<const N: usize> Agent for ApaNineBallPlayer<{ N }> {
    fn process(
        &mut self,
        simulation_state: SimulationState,
        msg: &Message,
    ) -> Option<Vec<Message>> {
        let mut rng = thread_rng();
        let dist = WeightedIndex::new(&self.run_out_weights).unwrap();
        let mut balls_to_run = self.run_out_choices[dist.sample(&mut rng)];
        let mut ball = msg.current_ball;

        while balls_to_run > 0 {
            balls_to_run -= 1;

            if ball == 9 {
                self.score += 2;
                ball = 1;
            } else {
                ball += 1;
                self.score += 1;
            }
        }

        if self.score >= self.winning_threshold {
            return Some(vec![Message {
                source: self.state().id.clone(),
                interrupt: Some(Interrupt::HaltSimulation("won".to_string())),
                ..Default::default()
            }]);
        }

        // If the player gets lucky, they get another turn.
        let next_turn = if rng.gen_range(0.0..1.0) > (1.0 - self.luck_chance) {
            self.state().id.clone()
        } else {
            self.opponent_name.clone()
        };

        Some(vec![Message {
            queued_time: simulation_state.time,
            completed_time: None,
            source: self.state().id.to_string(),
            destination: next_turn,
            current_ball: ball,
            ..Default::default()
        }])
    }

    fn state(&self) -> &AgentState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut AgentState {
        &mut self.state
    }
}

fn normal_9_ball_simulation(luck_chance: f32) -> String {
    let halt_condition = |s: &Simulation| s.agents.iter().all(|a| a.state().queue.is_empty());

    let alice = NineBallPlayer {
        luck_chance: 0.0,
        score: 0,
        winning_threshold: 5,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_weights: [5, 10, 20, 20, 18, 12, 5, 4, 3, 2],
        state: AgentState {
            mode: AgentMode::Reactive,
            wake_mode: AgentMode::Reactive,
            id: "alice".to_owned(),
            queue: vec![Message::default()].into(),
            ..Default::default()
        },
        opponent_name: "john".to_string(),
    };

    let john = NineBallPlayer {
        luck_chance,
        score: 0,
        winning_threshold: 5,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_weights: [6, 11, 18, 18, 15, 9, 3, 2, 2, 1],
        state: AgentState {
            mode: AgentMode::Reactive,
            wake_mode: AgentMode::Reactive,
            id: "alice".to_owned(),
            ..Default::default()
        },
        opponent_name: "alice".to_string(),
    };

    // SimulationParameters generator that holds all else static except for agents.
    let simulation_parameters_generator = move || SimulationParameters {
        agents: vec![Box::new(alice), Box::new(john)],
        halt_check: halt_condition,
        ..Default::default()
    };

    let mut sim = Simulation::new(simulation_parameters_generator());
    sim.run();

    sim.agents
        .iter()
        .find(|a| {
            a.state()
                .produced
                .last()
                .is_some_and(|m| m.interrupt.is_some())
        })
        .map(|a| a.state().id.clone())
        .unwrap()
}

fn nine_ball_apa_rules_simulation(luck_chance: f32) -> String {
    let mut empty_count = 0;
    let halt_condition = |s: &Simulation| s.agents.iter().all(|a| a.state().queue.is_empty());

    let alice = ApaNineBallPlayer {
        luck_chance: 0.0,
        score: 0,
        winning_threshold: 55,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_weights: [5, 10, 20, 20, 18, 12, 5, 4, 3, 2],
        state: AgentState {
            mode: AgentMode::Reactive,
            wake_mode: AgentMode::Reactive,
            id: "alice".to_owned(),
            queue: vec![Message::default()].into(),
            ..Default::default()
        },
        opponent_name: "john".to_string(),
    };

    let john = ApaNineBallPlayer {
        luck_chance,
        score: 0,
        winning_threshold: 55,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_weights: [6, 11, 18, 18, 15, 9, 3, 2, 2, 1],
        state: AgentState {
            mode: AgentMode::Reactive,
            wake_mode: AgentMode::Reactive,
            id: "alice".to_owned(),
            ..Default::default()
        },
        opponent_name: "alice".to_string(),
    };

    // SimulationParameters generator that holds all else static except for agents.
    let simulation_parameters_generator = move || SimulationParameters {
        agents: vec![Box::new(alice), Box::new(john)],
        halt_check: halt_condition,
        ..Default::default()
    };

    let mut sim = Simulation::new(simulation_parameters_generator());
    sim.run();

    println!("{:#?}", sim);

    sim.agents
        .iter()
        .find(|a| {
            a.state()
                .produced
                .last()
                .is_some_and(|m| m.interrupt.is_some())
        })
        .map(|a| a.state().id.clone())
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
        vec![Box::new(producer_agent), Box::new(consumer_agent)]
    }
}

/// Note, this main.rs binary file is just for library prototyping at the moment.
fn main() {
    for pct in [0.00, 0.20, 0.40, 0.50, 0.60].into_iter() {
        let mut count: HashMap<String, u32> = HashMap::new();
        for _ in 0..32768 {
            *count
                .entry(nine_ball_apa_rules_simulation(pct))
                .or_default() += 1;
        }

        println!(
            "(APA match skill 7 vs 7) Better player win percentage, {:.2}% luck factor for opponent: {:.2}",
            pct * 100.0,
            count["alice"] as f32 / (count["alice"] + count["john"]) as f32
        );
    }

    for pct in [0.00, 0.20, 0.40, 0.50, 0.60].into_iter() {
        let mut count: HashMap<String, u32> = HashMap::new();
        for _ in 0..32768 {
            *count.entry(normal_9_ball_simulation(pct)).or_default() += 1;
        }

        println!(
            "(set match race to 6) Better player win percentage, {:.2}% luck factor for opponent: {:.2}",
            pct * 100.0,
            count["alice"] as f32 / (count["alice"] + count["john"]) as f32
        );
    }
}
