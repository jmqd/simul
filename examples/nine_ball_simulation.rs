//! An example of running simulations for different billiards rule sets.
use rand::prelude::*;
use rand_distr::weighted::WeightedIndex;
use simul::agent::{Agent, AgentContext, AgentInitializer, AgentMode, AgentOptions};
use simul::message::Message;
use simul::{Simulation, SimulationParameters};
use std::collections::HashMap;

#[derive(Clone, Debug)]
#[allow(clippy::missing_docs_in_private_items)]
struct NineBallPlayer {
    luck_chance: f32,
    run_out_choices: [usize; 10],
    run_out_weights: [usize; 10],
    winning_threshold: u8,
    score: u8,
    opponent_name: String,
}

#[allow(clippy::unwrap_used)]
impl Agent for NineBallPlayer {
    fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
        let mut rng = rand::rng();
        let dist = WeightedIndex::new(self.run_out_weights).unwrap();
        let mut balls_to_run = self.run_out_choices[dist.sample(&mut rng)];

        let mut ball = u8::from_le_bytes(msg.custom_payload.clone().unwrap().try_into().unwrap());

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
            ctx.send_halt_interrupt("won");
        }

        // If the opponent gets lucky, they get another turn.
        let next_turn = if rng.random_range(0.0..1.0) > (1.0 - self.luck_chance) {
            ctx.name.to_string()
        } else {
            self.opponent_name.clone()
        };

        ctx.send(&next_turn, Some(ball.to_le_bytes().to_vec()));
    }
}

#[derive(Clone, Debug)]
#[allow(clippy::missing_docs_in_private_items)]
struct ApaNineBallPlayer {
    luck_chance: f32,
    run_out_choices: [usize; 10],
    run_out_weights: [usize; 10],
    winning_threshold: u8,
    score: u8,
    opponent_name: String,
    agent_options: AgentOptions,
}

#[allow(clippy::unwrap_used)]
impl Agent for ApaNineBallPlayer {
    fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
        let mut rng = rand::rng();
        let dist = WeightedIndex::new(self.run_out_weights).unwrap();
        let mut balls_to_run = self.run_out_choices[dist.sample(&mut rng)];
        let mut ball = u8::from_le_bytes(msg.custom_payload.clone().unwrap().try_into().unwrap());

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
            ctx.send_halt_interrupt("won");
        }

        // If the player gets lucky, they get another turn.
        let next_turn = if rng.random_range(0.0..1.0) > (1.0 - self.luck_chance) {
            ctx.name.to_string()
        } else {
            self.opponent_name.clone()
        };

        ctx.send(&next_turn, Some(ball.to_le_bytes().to_vec()));
    }
}

#[allow(clippy::unwrap_used)]
/// Runs a simulation for normal 9 ball rules.
fn normal_nine_ball_simulation_alice_vs_john(luck_chance: f32, starting_player: usize) -> String {
    let halt_condition = |s: &Simulation| s.agents().iter().all(|a| a.state.queue.is_empty());

    let alice = NineBallPlayer {
        luck_chance: 0.0,
        score: 0,
        winning_threshold: 5,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_weights: [5, 10, 20, 20, 18, 12, 5, 4, 3, 2],
        opponent_name: "john".to_string(),
    };

    let john = NineBallPlayer {
        luck_chance,
        score: 0,
        winning_threshold: 5,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_weights: [6, 11, 18, 18, 15, 9, 3, 2, 2, 1],
        opponent_name: "alice".to_string(),
    };

    let mut agent_initializers: Vec<AgentInitializer> = vec![
        AgentInitializer {
            agent: Box::new(alice),
            options: AgentOptions::default(),
        },
        AgentInitializer {
            agent: Box::new(john),
            options: AgentOptions::default(),
        },
    ];

    agent_initializers
        .get_mut(starting_player)
        .unwrap()
        .options
        .initial_queue = vec![Message {
        custom_payload: Some((1u8).to_le_bytes().to_vec()),
        ..Default::default()
    }]
    .into();

    // SimulationParameters generator that holds all else static except for agents.
    let simulation_parameters_generator = move || SimulationParameters {
        agent_initializers,
        halt_check: halt_condition,
        ..Default::default()
    };

    let mut sim = Simulation::new(simulation_parameters_generator());
    sim.run();

    sim.find_agent(|a| {
        a.state
            .produced
            .last()
            .is_some_and(|m| m.interrupt.is_some())
    })
    .map(|a| a.name.clone())
    .unwrap()
}

#[allow(clippy::unwrap_used)]
/// Runs a simulation of APA rules pool.
fn nine_ball_apa_rules_simulation_alice_vs_john(
    luck_chance: f32,
    starting_player: usize,
) -> String {
    let halt_condition = |s: &Simulation| s.agents().iter().all(|a| a.state.queue.is_empty());

    let alice = ApaNineBallPlayer {
        luck_chance: 0.0,
        score: 0,
        winning_threshold: 55,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_weights: [5, 10, 20, 20, 18, 12, 5, 4, 3, 2],
        agent_options: AgentOptions {
            initial_mode: AgentMode::Reactive,
            wake_mode: AgentMode::Reactive,
            name: "alice".to_string(),
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
        agent_options: AgentOptions {
            initial_mode: AgentMode::Reactive,
            wake_mode: AgentMode::Reactive,
            name: "john".to_string(),
            initial_queue: vec![Message {
                custom_payload: Some((1u8).to_le_bytes().to_vec()),
                ..Default::default()
            }]
            .into(),
        },
        opponent_name: "alice".to_string(),
    };

    let mut agent_initializers = vec![
        AgentInitializer {
            options: alice.agent_options.clone(),
            agent: Box::new(alice),
        },
        AgentInitializer {
            options: john.agent_options.clone(),
            agent: Box::new(john),
        },
    ];

    agent_initializers
        .get_mut(starting_player)
        .unwrap()
        .options
        .initial_queue = vec![Message {
        custom_payload: Some((1u8).to_le_bytes().to_vec()),
        ..Default::default()
    }]
    .into();

    // SimulationParameters generator that holds all else static except for agents.
    let simulation_parameters_generator = move || SimulationParameters {
        agent_initializers,
        halt_check: halt_condition,
        ..Default::default()
    };

    let mut sim = Simulation::new(simulation_parameters_generator());
    sim.run();

    sim.find_agent(|a| {
        a.state
            .produced
            .last()
            .is_some_and(|m| m.interrupt.is_some())
    })
    .map(|a| a.name.clone())
    .unwrap()
}

fn main() {
    // To vary who "breaks" first, we pass in a starting player, 0 or 1.
    let mut starting_player: usize = 0;

    eprintln!("Normal 9-ball");
    println!("luck_chance\tbetter_player_win_percent");

    for pct in [0.00, 0.20, 0.40, 0.50] {
        let mut count: HashMap<String, u32> = HashMap::new();
        for _ in 0..32768 {
            *count
                .entry(nine_ball_apa_rules_simulation_alice_vs_john(
                    pct,
                    starting_player,
                ))
                .or_default() += 1;

            starting_player ^= 1;
        }

        println!(
            "{}\t{}",
            pct * 100.0,
            (count["alice"] as f32 / (count["alice"] + count["john"]) as f32) * 100.0
        );
    }

    eprintln!("Normal 9-ball");
    println!("luck_chance\tbetter_player_win_percent");

    for pct in [0.00, 0.20, 0.40, 0.50] {
        let mut count: HashMap<String, u32> = HashMap::new();
        let mut starting_player = 0;
        for _ in 0..32768 {
            *count
                .entry(normal_nine_ball_simulation_alice_vs_john(
                    pct,
                    starting_player,
                ))
                .or_default() += 1;

            starting_player ^= 1;
        }

        println!(
            "{}\t{}",
            pct * 100.0,
            (count["alice"] as f32 / (count["alice"] + count["john"]) as f32) * 100.0
        );
    }
}
