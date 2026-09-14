#![allow(
    clippy::missing_docs_in_private_items,
    clippy::expect_used,
    clippy::cast_precision_loss
)]

use criterion::{criterion_group, criterion_main, Criterion};
use simul::agent::{periodic_consumer, periodic_producer};
use simul::experiment::search::{SearchResult, SearchStrategy, SeededRandomSearch, Told};
use simul::experiment::{
    aggregate_scalar, run_serial, CandidateId, FailurePolicy, FixedReplications, MasterSeed,
    NonFinitePolicy, ScalarSummary, SelectionPolicy,
};
use simul::{Simulation, SimulationParameters};
use std::cmp::Ordering;
use std::convert::Infallible;

const PRODUCER_PERIOD: u64 = 2;
const MAX_CONSUMER_PERIOD: u64 = 10;
const HALT_CONSUMED_COUNT: usize = 10;
const CANDIDATE_BUDGET: u64 = 1000;
const MASTER_SEED: MasterSeed = MasterSeed::from_u64(0x42_45_4E_43_48);

/// Selects the candidate with the highest finite mean score.
#[derive(Clone, Copy, Debug)]
struct MaximizeMean;

impl SelectionPolicy<ScalarSummary> for MaximizeMean {
    fn compare(&self, left: &ScalarSummary, right: &ScalarSummary) -> Ordering {
        left.mean.total_cmp(&right.mean)
    }
}

/// Creates simulation parameters for one typed consumer-period candidate.
fn build_simulation_parameters(consumer_period: u64) -> SimulationParameters {
    let halt_condition = |simulation: &Simulation| {
        simulation
            .find_by_name("consumer")
            .expect("consumer to exist")
            .state
            .consumed
            .len()
            > HALT_CONSUMED_COUNT
    };

    SimulationParameters {
        agent_initializers: vec![
            periodic_producer(
                "producer".to_string(),
                PRODUCER_PERIOD,
                "consumer".to_string(),
            ),
            periodic_consumer("consumer".to_string(), consumer_period),
        ],
        halt_check: halt_condition,
        enable_agent_asleep_cycles_metric: true,
        ..Default::default()
    }
}

/// Rewards short completion time while preferring less eager consumers on ties.
fn objective_score(simulation: &Simulation, consumer_period: u64) -> f64 {
    -(simulation.time() as f64) + consumer_period as f64
}

/// Runs the deterministic typed search path measured by the benchmark.
fn run_seeded_search() -> Option<SearchResult<u64, ScalarSummary>> {
    let mut search = SeededRandomSearch::new(
        MASTER_SEED,
        CANDIDATE_BUDGET,
        |rng| rng.next_u64() % (MAX_CONSUMER_PERIOD + 1),
        MaximizeMean,
    );
    let schedule = FixedReplications::new(1, 0);

    for proposal in search.ask(usize::MAX) {
        let candidate_id = CandidateId::new(proposal.token.value());
        let records = run_serial(
            MASTER_SEED,
            schedule.trials(candidate_id, proposal.candidate),
            |consumer_period, _context| {
                let mut simulation = Simulation::new(build_simulation_parameters(*consumer_period));
                simulation.run();
                Ok::<_, Infallible>(objective_score(&simulation, *consumer_period))
            },
        )
        .expect("one unique trial");
        let summary = aggregate_scalar(
            &records,
            |score| *score,
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
    }

    search.best().cloned()
}

#[allow(dead_code)]
fn run_search_experiment() -> Option<SearchResult<u64, ScalarSummary>> {
    let result = run_seeded_search();
    if let Some(best) = result.as_ref() {
        let mut final_simulation = Simulation::new(build_simulation_parameters(best.candidate));
        final_simulation.run();
        println!("Deterministic search found an approximate optimal configuration:");
        println!("Optimal Consumer Period: {}", best.candidate);
        println!("Producer is fixed at Period: {PRODUCER_PERIOD}");
        println!("Selecting Score: {:.2}", best.summary.statistics.mean);
        println!("Final Simulation Time: {}", final_simulation.time());
    }
    result
}

fn deterministic_search_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("deterministic_experiment_benchmarks");

    group.bench_function("seeded_random_search_1000_candidates", |b| {
        b.iter(|| {
            assert!(run_seeded_search().is_some());
        });
    });

    group.finish();
}

criterion_group!(benches, deterministic_search_bench);
criterion_main!(benches);
