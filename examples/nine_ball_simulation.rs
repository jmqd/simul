//! An example of running simulations for different billiards rule sets.
use rand::prelude::*;
use rand_distr::weighted::WeightedIndex;
use rustc_hash::FxHashMap;
use simul::agent::{Agent, AgentContext, AgentInitializer, AgentOptions};
use simul::experiment::MasterSeed;
use simul::message::Message;
use simul::{Halt, Simulation};

#[derive(Clone, Debug)]
#[allow(clippy::missing_docs_in_private_items)]
struct NineBallPlayer {
    luck_chance: f32,
    run_out_choices: [usize; 10],
    run_out_distribution: WeightedIndex<usize>,
    winning_threshold: u8,
    score: u8,
    opponent_name: String,
}

impl NineBallPlayer {
    /// Whether this player's score reached the match threshold.
    const fn has_won(&self) -> bool {
        self.score >= self.winning_threshold
    }
}

/// Each turn's message carries the ball on the table.
impl Agent<u8> for NineBallPlayer {
    fn on_message(&mut self, ctx: &mut AgentContext<'_, u8>, msg: &Message<u8>) {
        let mut rng = ctx.rng();
        let mut balls_to_run = self.run_out_choices[self.run_out_distribution.sample(&mut rng)];

        let mut ball = msg.payload;

        while balls_to_run > 0 {
            balls_to_run -= 1;

            if ball == 9 {
                self.score += 1;
                ball = 1;
            } else {
                ball += 1;
            }
        }

        if self.has_won() {
            ctx.send_halt_interrupt("won");
            return;
        }

        // If the player gets lucky, they get another turn.
        let next_turn = if rng.random_range(0.0..1.0) > (1.0 - self.luck_chance) {
            ctx.name()
        } else {
            &self.opponent_name
        };

        ctx.send(next_turn, ball);
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
}

impl ApaNineBallPlayer {
    /// Whether this player's score reached the match threshold.
    const fn has_won(&self) -> bool {
        self.score >= self.winning_threshold
    }
}

impl Agent<u8> for ApaNineBallPlayer {
    fn on_message(&mut self, ctx: &mut AgentContext<'_, u8>, msg: &Message<u8>) {
        let mut rng = ctx.rng();
        let mut balls_to_run = self.run_out_choices[self.run_out_distribution.sample(&mut rng)];
        let mut ball = msg.payload;

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

        if self.has_won() {
            ctx.send_halt_interrupt("won");
            return;
        }

        // If the player gets lucky, they get another turn.
        let next_turn = if rng.random_range(0.0..1.0) > (1.0 - self.luck_chance) {
            ctx.name()
        } else {
            &self.opponent_name
        };

        ctx.send(next_turn, ball);
    }
}

/// Returns the name of the player of type `P` that won the match.
fn winner<P: Agent<u8>>(sim: &Simulation<u8>, has_won: fn(&P) -> bool) -> Option<String> {
    sim.agents()
        .iter()
        .find(|agent| agent.downcast_ref::<P>().is_some_and(has_won))
        .map(|agent| agent.name.clone())
}

#[allow(clippy::unwrap_used)]
/// Runs a simulation for normal 9 ball rules; `seed` fixes every roll.
fn normal_nine_ball_simulation_alice_vs_john(
    luck_chance: f32,
    starting_player: usize,
    seed: u64,
) -> String {
    let halt_condition = |s: &Simulation<u8>| s.agents().iter().all(|a| a.state.queue.is_empty());

    let alice = NineBallPlayer {
        luck_chance: 0.0,
        score: 0,
        winning_threshold: 5,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_distribution: WeightedIndex::new([5, 10, 20, 20, 18, 12, 5, 4, 3, 2]).unwrap(),
        opponent_name: "john".to_string(),
    };

    let john = NineBallPlayer {
        luck_chance,
        score: 0,
        winning_threshold: 5,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_distribution: WeightedIndex::new([6, 11, 18, 18, 15, 9, 3, 2, 2, 1]).unwrap(),
        opponent_name: "alice".to_string(),
    };

    let mut agent_initializers: Vec<AgentInitializer<u8>> = vec![
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
        .initial_queue = vec![Message::new(0, 1)].into();

    let mut sim = Simulation::builder(Halt::when(halt_condition))
        .seed(MasterSeed::from_u64(seed))
        .agents(agent_initializers)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    sim.run();

    winner(&sim, NineBallPlayer::has_won).unwrap()
}

#[allow(clippy::unwrap_used)]
/// Runs a simulation of APA rules pool; `seed` fixes every roll.
fn nine_ball_apa_rules_simulation_alice_vs_john(
    luck_chance: f32,
    starting_player: usize,
    seed: u64,
) -> String {
    let halt_condition = |s: &Simulation<u8>| s.agents().iter().all(|a| a.state.queue.is_empty());

    let alice = ApaNineBallPlayer {
        luck_chance: 0.0,
        score: 0,
        winning_threshold: 55,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_distribution: WeightedIndex::new([5, 10, 20, 20, 18, 12, 5, 4, 3, 2]).unwrap(),
        opponent_name: "john".to_string(),
    };

    let john = ApaNineBallPlayer {
        luck_chance,
        score: 0,
        winning_threshold: 55,
        run_out_choices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        run_out_distribution: WeightedIndex::new([6, 11, 18, 18, 15, 9, 3, 2, 2, 1]).unwrap(),
        opponent_name: "alice".to_string(),
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
        .initial_queue = vec![Message::new(0, 1)].into();

    let mut sim = Simulation::builder(Halt::when(halt_condition))
        .seed(MasterSeed::from_u64(seed))
        .agents(agent_initializers)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    sim.run();

    winner(&sim, ApaNineBallPlayer::has_won).unwrap()
}

/// Matches per luck chance. Every luck chance replays the same match seeds,
/// so the comparison between them uses common random numbers.
const MATCHES: u64 = 32768;

fn main() {
    eprintln!("APA 9-ball");
    println!("luck_chance\tbetter_player_win_percent");

    for pct in [0.00, 0.20, 0.40, 0.50] {
        let mut count: FxHashMap<String, u32> = FxHashMap::default();
        for seed in 0..MATCHES {
            let starting_player = usize::from(seed % 2 == 1);
            *count
                .entry(nine_ball_apa_rules_simulation_alice_vs_john(
                    pct,
                    starting_player,
                    seed,
                ))
                .or_default() += 1;
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
        for seed in 0..MATCHES {
            let starting_player = usize::from(seed % 2 == 1);
            *count
                .entry(normal_nine_ball_simulation_alice_vs_john(
                    pct,
                    starting_player,
                    seed,
                ))
                .or_default() += 1;
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
            let run_out_distribution = WeightedIndex::new([1; 10])
                .unwrap_or_else(|error| panic!("invalid test distribution: {error}"));
            let agent: Box<dyn Agent<u8>> = if apa_rules {
                Box::new(ApaNineBallPlayer {
                    luck_chance: 0.0,
                    run_out_choices: [8; 10],
                    run_out_distribution,
                    winning_threshold: 9,
                    score: 0,
                    opponent_name: "opponent".to_string(),
                })
            } else {
                Box::new(NineBallPlayer {
                    luck_chance: 0.0,
                    run_out_choices: [8; 10],
                    run_out_distribution,
                    winning_threshold: 1,
                    score: 0,
                    opponent_name: "opponent".to_string(),
                })
            };
            let mut simulation = Simulation::builder(Halt::AtTime(2))
                .agents(vec![AgentInitializer {
                    agent,
                    options: AgentOptions {
                        name: "player".to_string(),
                        initial_queue: vec![Message::new(0, 2)].into(),
                        ..Default::default()
                    },
                }])
                .build()
                .unwrap_or_else(|error| panic!("{error}"));
            simulation.run();

            let player = simulation.find_by_name("player");
            let won = if apa_rules {
                player
                    .and_then(|agent| agent.downcast_ref::<ApaNineBallPlayer>())
                    .map(ApaNineBallPlayer::has_won)
            } else {
                player
                    .and_then(|agent| agent.downcast_ref::<NineBallPlayer>())
                    .map(NineBallPlayer::has_won)
            };
            assert_eq!(won, Some(true));
            assert_eq!(simulation.time(), 1);
            assert_eq!(
                simulation
                    .find_by_name("player")
                    .map(simul::agent::SimulationAgent::produced_count),
                Some(0)
            );
        }
    }
}
