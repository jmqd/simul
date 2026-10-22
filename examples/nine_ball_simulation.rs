//! An example of running simulations for different billiards rule sets.
use rand::prelude::*;
use rand_distr::weighted::WeightedIndex;
use rustc_hash::FxHashMap;
use simul::agent::{Agent, AgentContext, AgentInitializer, AgentOptions};
use simul::message::Message;
use simul::{Simulation, SimulationParameters};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone, Debug)]
#[allow(clippy::missing_docs_in_private_items)]
struct NineBallPlayer {
    luck_chance: f32,
    run_out_choices: [usize; 10],
    run_out_distribution: WeightedIndex<usize>,
    winning_threshold: u8,
    score: u8,
    opponent_name: String,
    winner: Rc<Cell<Option<usize>>>,
}

#[allow(clippy::unwrap_used)]
impl Agent for NineBallPlayer {
    fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
        let mut rng = rand::rng();
        let mut balls_to_run = self.run_out_choices[self.run_out_distribution.sample(&mut rng)];

        let mut ball =
            u8::from_le_bytes(msg.custom_payload.as_deref().unwrap().try_into().unwrap());

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
            self.winner.set(Some(ctx.id().index()));
            ctx.send_halt_interrupt("won");
            return;
        }

        // If the opponent gets lucky, they get another turn.
        let next_turn = if rng.random_range(0.0..1.0) > (1.0 - self.luck_chance) {
            ctx.name
        } else {
            &self.opponent_name
        };

        ctx.send(next_turn, Some(Arc::from([ball])));
    }
}

#[derive(Clone, Debug)]
#[allow(clippy::missing_docs_in_private_items)]
struct ApaNineBallPlayer {
    luck_chance: f32,
    run_out_choices: [usize; 10],
    run_out_distribution: WeightedIndex<usize>,
    winning_threshold: u8,
    score: u8,
    opponent_name: String,
    winner: Rc<Cell<Option<usize>>>,
}

#[allow(clippy::unwrap_used)]
impl Agent for ApaNineBallPlayer {
    fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
        let mut rng = rand::rng();
        let mut balls_to_run = self.run_out_choices[self.run_out_distribution.sample(&mut rng)];
        let mut ball =
            u8::from_le_bytes(msg.custom_payload.as_deref().unwrap().try_into().unwrap());

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
            self.winner.set(Some(ctx.id().index()));
            ctx.send_halt_interrupt("won");
            return;
        }

        // If the player gets lucky, they get another turn.
        let next_turn = if rng.random_range(0.0..1.0) > (1.0 - self.luck_chance) {
            ctx.name
        } else {
            &self.opponent_name
        };

        ctx.send(next_turn, Some(Arc::from([ball])));
    }
}

#[allow(clippy::unwrap_used)]
/// Runs a simulation for normal 9 ball rules.
fn normal_nine_ball_simulation_alice_vs_john(luck_chance: f32, starting_player: usize) -> String {
    let halt_condition = |s: &Simulation| s.agents().iter().all(|a| a.state.queue.is_empty());
    let winner = Rc::new(Cell::new(None));

    let alice = NineBallPlayer {
        luck_chance: 0.0,
        score: 0,
        winning_threshold: 5,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_distribution: WeightedIndex::new([5, 10, 20, 20, 18, 12, 5, 4, 3, 2]).unwrap(),
        opponent_name: "john".to_string(),
        winner: Rc::clone(&winner),
    };

    let john = NineBallPlayer {
        luck_chance,
        score: 0,
        winning_threshold: 5,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_distribution: WeightedIndex::new([6, 11, 18, 18, 15, 9, 3, 2, 2, 1]).unwrap(),
        opponent_name: "alice".to_string(),
        winner: Rc::clone(&winner),
    };

    let mut agent_initializers: Vec<AgentInitializer> = vec![
        AgentInitializer {
            agent: Box::new(alice),
            options: AgentOptions::defaults_with_name("alice".to_string()),
        },
        AgentInitializer {
            agent: Box::new(john),
            options: AgentOptions::defaults_with_name("john".to_string()),
        },
    ];

    agent_initializers
        .get_mut(starting_player)
        .unwrap()
        .options
        .initial_queue = vec![Message::new_with_payload(0, Some(Arc::from([1])))].into();

    let mut sim = Simulation::new(SimulationParameters {
        agent_initializers,
        halt_check: halt_condition,
        ..Default::default()
    });
    sim.run();

    winner
        .get()
        .map(|handle| sim.agents()[handle].name.clone())
        .unwrap()
}

#[allow(clippy::unwrap_used)]
/// Runs a simulation of APA rules pool.
fn nine_ball_apa_rules_simulation_alice_vs_john(
    luck_chance: f32,
    starting_player: usize,
) -> String {
    let halt_condition = |s: &Simulation| s.agents().iter().all(|a| a.state.queue.is_empty());
    let winner = Rc::new(Cell::new(None));

    let alice = ApaNineBallPlayer {
        luck_chance: 0.0,
        score: 0,
        winning_threshold: 55,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_distribution: WeightedIndex::new([5, 10, 20, 20, 18, 12, 5, 4, 3, 2]).unwrap(),
        opponent_name: "john".to_string(),
        winner: Rc::clone(&winner),
    };

    let john = ApaNineBallPlayer {
        luck_chance,
        score: 0,
        winning_threshold: 55,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_distribution: WeightedIndex::new([6, 11, 18, 18, 15, 9, 3, 2, 2, 1]).unwrap(),
        opponent_name: "alice".to_string(),
        winner: Rc::clone(&winner),
    };

    let mut agent_initializers = vec![
        AgentInitializer {
            agent: Box::new(alice),
            options: AgentOptions::defaults_with_name("alice".to_string()),
        },
        AgentInitializer {
            agent: Box::new(john),
            options: AgentOptions::defaults_with_name("john".to_string()),
        },
    ];

    agent_initializers
        .get_mut(starting_player)
        .unwrap()
        .options
        .initial_queue = vec![Message::new_with_payload(0, Some(Arc::from([1])))].into();

    let mut sim = Simulation::new(SimulationParameters {
        agent_initializers,
        halt_check: halt_condition,
        ..Default::default()
    });
    sim.run();

    winner
        .get()
        .map(|handle| sim.agents()[handle].name.clone())
        .unwrap()
}

fn main() {
    let mut starting_player: usize = 0;

    eprintln!("APA 9-ball");
    println!("luck_chance\tbetter_player_win_percent");

    for pct in [0.00, 0.20, 0.40, 0.50] {
        let mut count: FxHashMap<String, u32> = FxHashMap::default();
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
        let mut count: FxHashMap<String, u32> = FxHashMap::default();
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

/// Regression coverage for the match-ending turn under each ruleset.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winning_turn_records_the_winner_without_scheduling_another_turn() {
        for apa_rules in [false, true] {
            let winner = Rc::new(Cell::new(None));
            let run_out_distribution = WeightedIndex::new([1; 10])
                .unwrap_or_else(|error| panic!("invalid test distribution: {error}"));
            let agent: Box<dyn Agent> = if apa_rules {
                Box::new(ApaNineBallPlayer {
                    luck_chance: 0.0,
                    run_out_choices: [8; 10],
                    run_out_distribution,
                    winning_threshold: 9,
                    score: 0,
                    opponent_name: "opponent".to_string(),
                    winner: Rc::clone(&winner),
                })
            } else {
                Box::new(NineBallPlayer {
                    luck_chance: 0.0,
                    run_out_choices: [8; 10],
                    run_out_distribution,
                    winning_threshold: 1,
                    score: 0,
                    opponent_name: "opponent".to_string(),
                    winner: Rc::clone(&winner),
                })
            };
            let mut simulation = Simulation::new(SimulationParameters {
                agent_initializers: vec![AgentInitializer {
                    agent,
                    options: AgentOptions {
                        name: "player".to_string(),
                        initial_queue: vec![Message::new_with_payload(0, Some(Arc::from([2])))]
                            .into(),
                        ..Default::default()
                    },
                }],
                halt_check: |simulation| simulation.time() == 2,
                ..Default::default()
            });
            simulation.run();

            assert_eq!(winner.get(), Some(0));
            assert_eq!(simulation.time(), 1);
            assert_eq!(
                simulation
                    .produced_for_agent("player")
                    .map(<[Message]>::len),
                Some(0)
            );
        }
    }
}
