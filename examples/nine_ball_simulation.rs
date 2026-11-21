//! An example of running simulations for different billiards rule sets.
use rand::prelude::*;
use rand_distr::weighted::WeightedIndex;
use simul::agent::{Agent, AgentContext, AgentInitializer};
use simul::experiment::MasterSeed;
use simul::message::Message;
use simul::{Halt, Simulation};
use std::sync::Arc;

/// Scoring rules for a 9-ball match.
#[derive(Clone, Copy, Debug)]
enum Rules {
    /// One point per 9-ball pocketed (race to 5).
    Standard,
    /// One point per ball 1-8 and two points for the 9-ball (race to 55).
    Apa,
}

impl Rules {
    /// Score needed to win a match under these rules.
    const fn winning_threshold(self) -> u8 {
        match self {
            Self::Standard => 5,
            Self::Apa => 55,
        }
    }

    /// Points awarded for pocketing `ball`.
    const fn points_for(self, ball: u8) -> u8 {
        match (self, ball) {
            (Self::Standard, 9) => 1,
            (Self::Standard, _) => 0,
            (Self::Apa, 9) => 2,
            (Self::Apa, _) => 1,
        }
    }
}

#[derive(Clone, Debug)]
#[allow(clippy::missing_docs_in_private_items)]
struct NineBallPlayer {
    rules: Rules,
    luck_chance: f32,
    run_out_distribution: Arc<WeightedIndex<usize>>,
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
        let mut balls_to_run = self.run_out_distribution.sample(&mut rng);
        let mut ball = msg.payload;

        while balls_to_run > 0 {
            balls_to_run -= 1;
            self.score += self.rules.points_for(ball);
            ball = if ball == 9 { 1 } else { ball + 1 };
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

/// Runs one Alice-vs-John 9-ball match under `rules` and returns whether Alice won.
fn simulate_match(
    rules: Rules,
    alice_dist: &Arc<WeightedIndex<usize>>,
    john_dist: &Arc<WeightedIndex<usize>>,
    luck_chance: f32,
    starting_player: usize,
    seed: u64,
) -> bool {
    let winning_threshold = rules.winning_threshold();
    let alice = NineBallPlayer {
        rules,
        luck_chance: 0.0,
        score: 0,
        winning_threshold,
        run_out_distribution: Arc::clone(alice_dist),
        opponent_name: "john".to_string(),
    };
    let john = NineBallPlayer {
        rules,
        luck_chance,
        score: 0,
        winning_threshold,
        run_out_distribution: Arc::clone(john_dist),
        opponent_name: "alice".to_string(),
    };

    let opening = |player| (player == starting_player).then(|| Message::new(0, 1));
    let mut sim = Simulation::builder(Halt::when(|s: &Simulation<u8>| {
        s.all_agents(|a| a.state.queue.is_empty())
    }))
    .seed(MasterSeed::from_u64(seed))
    .agents([
        AgentInitializer::reactive("alice", alice).initial_queue(opening(0)),
        AgentInitializer::reactive("john", john).initial_queue(opening(1)),
    ])
    .build()
    .unwrap_or_else(|error| panic!("{error}"));
    sim.run();

    sim.find_agent(|agent| {
        agent
            .downcast_ref::<NineBallPlayer>()
            .is_some_and(NineBallPlayer::has_won)
    })
    .is_some_and(|winner| winner.name() == "alice")
}

/// Matches per luck chance. Every luck chance replays the same match seeds,
/// so the comparison between them uses common random numbers.
const MATCHES: u64 = 32_768;

/// Runs the luck-chance sweep for one ruleset and prints win percentages.
fn run_sweep(
    label: &str,
    rules: Rules,
    alice_dist: &Arc<WeightedIndex<usize>>,
    john_dist: &Arc<WeightedIndex<usize>>,
) {
    eprintln!("{label}");
    println!("luck_chance\tbetter_player_win_percent");

    for pct in [0.00, 0.20, 0.40, 0.50] {
        let mut alice_wins = 0_u32;
        for seed in 0..MATCHES {
            let starting_player = usize::from(seed % 2 == 1);
            if simulate_match(rules, alice_dist, john_dist, pct, starting_player, seed) {
                alice_wins += 1;
            }
        }

        println!(
            "{}\t{}",
            pct * 100.0,
            (alice_wins as f32 / MATCHES as f32) * 100.0
        );
    }
}

fn main() {
    let alice_dist = Arc::new(
        WeightedIndex::new([5, 10, 20, 20, 18, 12, 5, 4, 3, 2])
            .unwrap_or_else(|error| panic!("valid weights: {error}")),
    );
    let john_dist = Arc::new(
        WeightedIndex::new([6, 11, 18, 18, 15, 9, 3, 2, 2, 1])
            .unwrap_or_else(|error| panic!("valid weights: {error}")),
    );

    run_sweep("APA 9-ball", Rules::Apa, &alice_dist, &john_dist);
    run_sweep("Normal 9-ball", Rules::Standard, &alice_dist, &john_dist);
}

/// Regression coverage for the match-ending turn under each ruleset.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winning_turn_records_the_winner_without_scheduling_another_turn() {
        let run_out_distribution = Arc::new(
            WeightedIndex::new([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
                .unwrap_or_else(|error| panic!("invalid test distribution: {error}")),
        );
        for (rules, winning_threshold) in [(Rules::Standard, 1), (Rules::Apa, 9)] {
            let player = AgentInitializer::reactive(
                "player",
                NineBallPlayer {
                    rules,
                    luck_chance: 0.0,
                    run_out_distribution: Arc::clone(&run_out_distribution),
                    winning_threshold,
                    score: 0,
                    opponent_name: "opponent".to_string(),
                },
            );
            let mut simulation = Simulation::builder(Halt::AtTime(2))
                .agent(player.initial_queue([Message::new(0, 2)]))
                .build()
                .unwrap_or_else(|error| panic!("{error}"));
            simulation.run();

            let won = simulation
                .find_by_name("player")
                .and_then(|agent| agent.downcast_ref::<NineBallPlayer>())
                .map(NineBallPlayer::has_won);
            assert_eq!(Some(true), won);
            assert_eq!(1, simulation.time());
            assert_eq!(
                Some(0),
                simulation
                    .find_by_name("player")
                    .map(simul::agent::SimulationAgent::produced_count)
            );
        }
    }
}
