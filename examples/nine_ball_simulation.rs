use rand::distributions::WeightedIndex;
use rand::prelude::*;
use simul::agent::*;
use simul::message::Interrupt;
use simul::message::Message;
use simul::*;
use std::collections::HashMap;

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
            custom_payload: Some(ball.to_le_bytes().to_vec()),
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
            custom_payload: Some(ball.to_le_bytes().to_vec()),
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

fn normal_nine_ball_simulation_alice_vs_john(luck_chance: f32, starting_player: usize) -> String {
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
            id: "john".to_owned(),
            ..Default::default()
        },
        opponent_name: "alice".to_string(),
    };

    let mut agents: Vec<Box<dyn Agent>> = vec![Box::new(alice), Box::new(john)];
    agents.get_mut(starting_player).unwrap().state_mut().queue = vec![Message {
        custom_payload: Some((1u8).to_le_bytes().to_vec()),
        ..Default::default()
    }]
    .into();

    // SimulationParameters generator that holds all else static except for agents.
    let simulation_parameters_generator = move || SimulationParameters {
        agents,
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

fn nine_ball_apa_rules_simulation_alice_vs_john(
    luck_chance: f32,
    starting_player: usize,
) -> String {
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
            id: "john".to_owned(),
            queue: vec![Message {
                custom_payload: Some((1u8).to_le_bytes().to_vec()),
                ..Default::default()
            }]
            .into(),
            ..Default::default()
        },
        opponent_name: "alice".to_string(),
    };

    let mut agents: Vec<Box<dyn Agent>> = vec![Box::new(alice), Box::new(john)];
    agents.get_mut(starting_player).unwrap().state_mut().queue = vec![Message {
        custom_payload: Some((1u8).to_le_bytes().to_vec()),
        ..Default::default()
    }]
    .into();

    // SimulationParameters generator that holds all else static except for agents.
    let simulation_parameters_generator = move || SimulationParameters {
        agents,
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

fn main() {
    // To vary who "breaks" first, we pass in a starting player, 0 or 1.
    let mut starting_player: usize = 0;

    eprintln!("Normal 9-ball");
    println!("luck_chance\tbetter_player_win_percent");

    for pct in [0.00, 0.20, 0.40, 0.50].into_iter() {
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

    for pct in [0.00, 0.20, 0.40, 0.50].into_iter() {
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
