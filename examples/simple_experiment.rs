//! A deterministic random-search experiment over a discrete-event simulation.
#![allow(
    clippy::missing_docs_in_private_items,
    clippy::expect_used,
    clippy::cast_precision_loss
)]

use simul::agent::{periodic_consumer, periodic_producer};
use simul::experiment::search::{SearchStrategy, SeededRandomSearch, Told};
use simul::experiment::{
    aggregate_scalar, run_serial, CandidateId, FailurePolicy, FixedReplications, MasterSeed,
    NonFinitePolicy, ScalarSummary, SelectionPolicy,
};
use simul::{Halt, Simulation, SimulationBuilder};
use std::cmp::Ordering;
use std::convert::Infallible;

const PRODUCER_PERIOD: u64 = 2;
const MAX_CONSUMER_PERIOD: u64 = 10;
const HALT_CONSUMED_COUNT: u64 = 10;
const CANDIDATE_BUDGET: u64 = 1000;
const MASTER_SEED: MasterSeed = MasterSeed::from_u64(0x51_4D_50_4C);

/// Selects the candidate with the highest finite mean score.
#[derive(Clone, Copy, Debug)]
struct MaximizeMean;

impl SelectionPolicy<ScalarSummary> for MaximizeMean {
    fn compare(&self, left: &ScalarSummary, right: &ScalarSummary) -> Ordering {
        left.mean().total_cmp(&right.mean())
    }
}

/// Configures the simulation for one typed consumer-period candidate.
fn simulation_builder(consumer_period: u64) -> SimulationBuilder {
    let halt_condition = |simulation: &Simulation| {
        simulation
            .find_by_name("consumer")
            .expect("consumer to exist")
            .consumed_count()
            > HALT_CONSUMED_COUNT
    };

    Simulation::builder(Halt::when(halt_condition))
        .agents(vec![
            periodic_producer(
                "producer".to_string(),
                PRODUCER_PERIOD,
                "consumer".to_string(),
            ),
            periodic_consumer("consumer".to_string(), consumer_period),
        ])
        .asleep_cycles_metric(true)
}

/// Rewards short completion time while preferring less eager consumers on ties.
fn objective_score(simulation: &Simulation, consumer_period: u64) -> f64 {
    -(simulation.time() as f64) + consumer_period as f64
}

/// Runs a deterministic, typed search and retains the selecting simulation.
fn run_experiment() -> Simulation {
    let mut search = SeededRandomSearch::new(
        MASTER_SEED,
        CANDIDATE_BUDGET,
        |rng| rng.next_u64() % (MAX_CONSUMER_PERIOD + 1),
        MaximizeMean,
    );
    let schedule = FixedReplications::new(1, 0);
    let mut best_simulation = None;

    for proposal in search.ask(usize::MAX) {
        let candidate_id = CandidateId::new(proposal.token.value());
        let consumer_period = proposal.candidate;
        let mut records = run_serial(
            MASTER_SEED,
            schedule.trials(candidate_id, consumer_period),
            |period, context| {
                // Candidates in one common-random group share their agents'
                // draws, so differences between them come from the policy.
                let mut simulation = simulation_builder(*period)
                    .trial(context)
                    .build()
                    .unwrap_or_else(|error| panic!("{error}"));
                simulation.run();
                Ok::<_, Infallible>(simulation)
            },
        )
        .expect("one unique trial");
        let summary = aggregate_scalar(
            &records,
            |simulation| objective_score(simulation, consumer_period),
            FailurePolicy::RejectCandidate,
            NonFinitePolicy::RejectCandidate,
        )
        .expect("the simulation produces one finite score");

        search
            .tell(&[Told {
                token: proposal.token,
                candidate: candidate_id,
                summary,
            }])
            .expect("the proposal token is outstanding");

        if search
            .best()
            .is_some_and(|best| best.candidate_id == candidate_id)
        {
            let selecting_record = records.pop().expect("one trial record");
            best_simulation = Some(
                selecting_record
                    .result
                    .expect("the simulation evaluator is infallible"),
            );
        }
    }

    best_simulation.expect("the non-zero search budget produces a result")
}

fn main() {
    println!("{:#?}", run_experiment().calc_avg_wait_statistics());
}
