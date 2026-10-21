//! Deterministic instruction-count workloads for comparing two builds.
//!
//! Each workload runs one public hot path a fixed number of times inside the
//! measured region and digests its outputs into a fingerprint, so a build that
//! retires fewer instructions but computes different results is caught. On
//! macOS the counters come from `proc_pid_rusage`; elsewhere only wall time and
//! fingerprints are reported.
//!
//! ```text
//! cargo bench --bench instruction_counts -- [--reps N] [--filter SUBSTRING] [--json]
//! ```
//!
//! Without `--bench`, as under `cargo test --all-targets`, every workload runs
//! once as a smoke test.
#![allow(
    clippy::missing_docs_in_private_items,
    clippy::expect_used,
    clippy::cast_precision_loss
)]

use rand::{RngExt as _, SeedableRng as _};
use simul::agent::{
    periodic_consumer, periodic_producer, Agent, AgentContext, AgentInitializer, AgentMode,
    AgentOptions, MessageProcessingStatus,
};
use simul::experiment::search::{sample_bounded, SearchStrategy, SeededRandomSearch, Told};
use simul::experiment::{
    aggregate_bernoulli, aggregate_scalar, run_serial, simulated_annealing_search_with_rng,
    CandidateId, CandidateSummary, CrossEntropyConfig, CrossEntropyDimension,
    CrossEntropyOptimizer, CrossEntropySample, FailurePolicy, FixedReplications, MasterSeed,
    NonFinitePolicy, ScalarReducer, ScalarSummary, SelectionPolicy, TrialContext, TrialRecord,
    TrialStream,
};
use simul::message::Message;
use simul::{Simulation, SimulationParameters};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::hint::black_box;
use std::time::Instant;

const SEED: MasterSeed = MasterSeed::from_u64(0x494E_5354_5255_4354);

struct Workload {
    name: &'static str,
    /// Calls per measurement, sized for at least about 10^7 instructions.
    iterations: u32,
    run: fn() -> Fingerprint,
}

const WORKLOADS: &[Workload] = &[
    Workload {
        name: "sim/periodic_10k",
        iterations: 2,
        run: sim_periodic,
    },
    Workload {
        name: "sim/queue_metrics_10k",
        iterations: 2,
        run: sim_queue_metrics,
    },
    Workload {
        name: "sim/fan_in_64",
        iterations: 1,
        run: sim_fan_in,
    },
    Workload {
        name: "sim/sparse_512",
        iterations: 1,
        run: sim_sparse,
    },
    Workload {
        name: "sim/sparse_512_scheduled",
        iterations: 1,
        run: sim_sparse_scheduled,
    },
    Workload {
        name: "sim/relay_ring_16",
        iterations: 1,
        run: sim_relay_ring,
    },
    Workload {
        name: "exp/search_sim_1000",
        iterations: 1,
        run: exp_search_sim,
    },
    Workload {
        name: "exp/run_serial_4096",
        iterations: 1,
        run: exp_run_serial,
    },
    #[cfg(feature = "parallel")]
    Workload {
        name: "exp/run_parallel_4096",
        iterations: 1,
        run: exp_run_parallel,
    },
    Workload {
        name: "exp/sample_bounded_512x8",
        iterations: 2,
        run: exp_sample_bounded,
    },
    Workload {
        name: "exp/random_search_4096",
        iterations: 1,
        run: exp_random_search,
    },
    Workload {
        name: "exp/cem_generation_24",
        iterations: 1_600,
        run: exp_cem_generation::<24>,
    },
    Workload {
        name: "exp/cem_generation_512",
        iterations: 80,
        run: exp_cem_generation::<512>,
    },
    Workload {
        name: "exp/cem_search_480",
        iterations: 80,
        run: exp_cem_search,
    },
    Workload {
        name: "exp/annealing_480",
        iterations: 80,
        run: exp_annealing,
    },
];

/// Order-sensitive 64-bit digest of workload outputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Fingerprint(u64);

impl Fingerprint {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    const fn mix(self, value: u64) -> Self {
        Self(
            (self.0 ^ value)
                .wrapping_mul(0x0000_0100_0000_01b3)
                .rotate_left(23),
        )
    }

    const fn mix_f64(self, value: f64) -> Self {
        self.mix(value.to_bits())
    }

    fn mix_len(self, value: usize) -> Self {
        self.mix(u64::try_from(value).unwrap_or(u64::MAX))
    }
}

/// `struct rusage_info_v4` from `<sys/resource.h>`, padded past its end.
#[cfg(target_os = "macos")]
#[repr(C)]
struct RusageInfo {
    uuid: [u8; 16],
    words: [u64; 40],
}

#[cfg(target_os = "macos")]
extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfo) -> i32;
}

/// Returns retired instructions and elapsed cycles for this process.
#[cfg(target_os = "macos")]
fn counters() -> Option<(u64, u64)> {
    /// `RUSAGE_INFO_V4`.
    const RUSAGE_INFO_V4: i32 = 4;
    /// Word index of `ri_instructions`.
    const INSTRUCTIONS: usize = 29;
    /// Word index of `ri_cycles`.
    const CYCLES: usize = 30;
    let pid = i32::try_from(std::process::id()).ok()?;
    let mut info = RusageInfo {
        uuid: [0; 16],
        words: [0; 40],
    };
    // SAFETY: `info` is writable and larger than `rusage_info_v4`.
    let status = unsafe { proc_pid_rusage(pid, RUSAGE_INFO_V4, &raw mut info) };
    (status == 0).then_some((info.words[INSTRUCTIONS], info.words[CYCLES]))
}

/// Hardware counters are only wired up for macOS.
#[cfg(not(target_os = "macos"))]
const fn counters() -> Option<(u64, u64)> {
    None
}

struct Sample {
    instructions: Option<u64>,
    cycles: Option<u64>,
    nanos: u128,
}

fn measure(workload: &Workload) -> (Fingerprint, Sample) {
    let before = counters();
    let start = Instant::now();
    let mut fingerprint = Fingerprint::new();
    for _ in 0..workload.iterations {
        fingerprint = fingerprint.mix(black_box((workload.run)()).0);
    }
    let nanos = start.elapsed().as_nanos();
    let after = counters();
    let delta =
        before
            .zip(after)
            .map(|((instructions, cycles), (end_instructions, end_cycles))| {
                (
                    end_instructions.saturating_sub(instructions),
                    end_cycles.saturating_sub(cycles),
                )
            });
    (
        fingerprint,
        Sample {
            instructions: delta.map(|(instructions, _)| instructions),
            cycles: delta.map(|(_, cycles)| cycles),
            nanos,
        },
    )
}

struct Options {
    bench: bool,
    json: bool,
    reps: usize,
    filter: Option<String>,
}

impl Options {
    fn parse(mut args: impl Iterator<Item = String>) -> Self {
        let mut options = Self {
            bench: false,
            json: false,
            reps: 5,
            filter: None,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--bench" => options.bench = true,
                "--json" => options.json = true,
                "--reps" => {
                    options.reps = args
                        .next()
                        .and_then(|reps| reps.parse().ok())
                        .filter(|&reps| reps > 0)
                        .expect("--reps takes a positive integer");
                }
                "--filter" => options.filter = args.next(),
                _ => {}
            }
        }
        options
    }

    fn selects(&self, name: &str) -> bool {
        self.filter
            .as_deref()
            .is_none_or(|filter| name.contains(filter))
    }
}

/// Returns the lower median, which is an observed value.
fn median<T: Copy + Ord>(values: &mut [T]) -> Option<T> {
    values.sort_unstable();
    values
        .get(values.len().saturating_sub(1).div_euclid(2))
        .copied()
}

fn report(options: &Options, workload: &Workload, fingerprint: Fingerprint, samples: &[Sample]) {
    let mut instructions: Vec<u64> = samples.iter().filter_map(|s| s.instructions).collect();
    let mut cycles: Vec<u64> = samples.iter().filter_map(|s| s.cycles).collect();
    let mut nanos: Vec<u128> = samples.iter().map(|s| s.nanos).collect();
    let instructions_min = instructions.iter().min().copied();
    let cycles_min = cycles.iter().min().copied();
    let nanos_min = nanos.iter().min().copied().unwrap_or(0);
    let instructions_median = median(&mut instructions);
    let cycles_median = median(&mut cycles);
    let nanos_median = median(&mut nanos).unwrap_or(0);
    let json = |value: Option<u64>| value.map_or_else(|| "null".to_string(), |v| v.to_string());
    if options.json {
        println!(
            "{{\"workload\":\"{}\",\"iterations\":{},\"reps\":{},\"instructions_min\":{},\"instructions_median\":{},\"cycles_min\":{},\"cycles_median\":{},\"nanos_min\":{nanos_min},\"nanos_median\":{nanos_median},\"fingerprint\":\"{:016x}\"}}",
            workload.name,
            workload.iterations,
            samples.len(),
            json(instructions_min),
            json(instructions_median),
            json(cycles_min),
            json(cycles_median),
            fingerprint.0,
        );
    } else {
        let count = |value: Option<u64>| value.map_or_else(|| "n/a".to_string(), |v| v.to_string());
        println!(
            "{:<26} x{:<4} instructions min {:>13} median {:>13}  cycles min {:>13}  ms min {:>8.3}  fingerprint {:016x}",
            workload.name,
            workload.iterations,
            count(instructions_min),
            count(instructions_median),
            count(cycles_min),
            nanos_min as f64 / 1e6,
            fingerprint.0,
        );
    }
}

fn main() {
    let options = Options::parse(std::env::args().skip(1));
    for workload in WORKLOADS.iter().filter(|w| options.selects(w.name)) {
        if !options.bench {
            println!("{} ok {:016x}", workload.name, (workload.run)().0);
            continue;
        }
        let (expected, _) = measure(workload);
        let samples: Vec<Sample> = (0..options.reps)
            .map(|_| {
                let (fingerprint, sample) = measure(workload);
                assert_eq!(
                    expected, fingerprint,
                    "{} is not deterministic",
                    workload.name
                );
                sample
            })
            .collect();
        report(&options, workload, expected, &samples);
    }
}

fn simulate(parameters: SimulationParameters) -> Simulation {
    let mut simulation = Simulation::new(parameters);
    simulation.run();
    simulation
}

/// Digests the final time plus every agent's queue, histories, and total wait.
fn histories(simulation: &Simulation) -> Fingerprint {
    simulation.agents().iter().fold(
        Fingerprint::new().mix(simulation.time()),
        |fingerprint, agent| {
            let waited: u64 = agent
                .state
                .consumed
                .iter()
                .map(|message| {
                    message
                        .completed_time
                        .unwrap_or(message.queued_time)
                        .saturating_sub(message.queued_time)
                })
                .sum();
            fingerprint
                .mix_len(agent.state.queue.len())
                .mix_len(agent.state.consumed.len())
                .mix_len(agent.state.produced.len())
                .mix(waited)
        },
    )
}

fn periodic_pair(enable_queue_depth_metrics: bool) -> Fingerprint {
    let simulation = simulate(SimulationParameters {
        agent_initializers: vec![
            periodic_producer("producer".to_string(), 1, "consumer".to_string()),
            periodic_consumer("consumer".to_string(), 1),
        ],
        halt_check: |simulation| simulation.time() >= 10_000,
        enable_queue_depth_metrics,
        ..Default::default()
    });
    let depths = simulation.queue_depth_metrics("consumer").unwrap_or(&[]);
    depths
        .iter()
        .fold(histories(&simulation), |fingerprint, &depth| {
            fingerprint.mix_len(depth)
        })
}

fn sim_periodic() -> Fingerprint {
    periodic_pair(false)
}

fn sim_queue_metrics() -> Fingerprint {
    periodic_pair(true)
}

/// Many producers feed one consumer; more than eight agents use name hashing.
fn sim_fan_in() -> Fingerprint {
    let mut agent_initializers: Vec<AgentInitializer> = (0..64_u64)
        .map(|index| {
            periodic_producer(
                format!("producer-{index}"),
                1 + index % 7,
                "sink".to_string(),
            )
        })
        .collect();
    agent_initializers.push(periodic_consumer("sink".to_string(), 1));
    let simulation = simulate(SimulationParameters {
        agent_initializers,
        halt_check: |simulation| {
            simulation
                .find_by_name("sink")
                .is_some_and(|sink| sink.state.consumed.len() >= 1_500)
        },
        ..Default::default()
    });
    histories(&simulation)
}

/// Mostly sleeping producers and a few busy consumers.
fn sparse_simulation(enable_agent_asleep_cycles_metric: bool) -> Simulation {
    let mut agent_initializers: Vec<AgentInitializer> = (0..512_u64)
        .map(|index| {
            periodic_producer(
                format!("producer-{index}"),
                200 + index % 97,
                format!("consumer-{}", index % 8),
            )
        })
        .collect();
    agent_initializers
        .extend((0..8).map(|index| periodic_consumer(format!("consumer-{index}"), 1)));
    simulate(SimulationParameters {
        agent_initializers,
        halt_check: |simulation| simulation.time() >= 3_000,
        enable_agent_asleep_cycles_metric,
        ..Default::default()
    })
}

/// The sparse population with sleep metrics, which require dense scans.
fn sim_sparse() -> Fingerprint {
    let simulation = sparse_simulation(true);
    simulation
        .agents()
        .iter()
        .fold(histories(&simulation), |fingerprint, agent| {
            fingerprint.mix(simulation.asleep_cycle_count(&agent.name).unwrap_or(0))
        })
}

/// The sparse population without metrics, which the sparse schedule serves.
fn sim_sparse_scheduled() -> Fingerprint {
    histories(&sparse_simulation(false))
}

/// Forwards each payload around a ring after holding it in progress.
#[derive(Clone, Debug)]
struct Relay {
    next: String,
    hold: u8,
    held: u8,
}

impl Agent for Relay {
    fn on_message(&mut self, ctx: &mut AgentContext, message: &Message) {
        if self.held < self.hold {
            self.held += 1;
            ctx.set_processing_status(MessageProcessingStatus::InProgress);
        } else {
            self.held = 0;
            ctx.send(&self.next, message.custom_payload.clone());
        }
    }
}

fn sim_relay_ring() -> Fingerprint {
    const RELAYS: u8 = 16;
    let agent_initializers = (0..RELAYS)
        .map(|index| {
            let name = format!("relay-{index}");
            let initial_queue: VecDeque<Message> = (0..3_u8)
                .map(|seed| Message::new_with_payload(0, Some(vec![index ^ seed; 24])))
                .collect();
            AgentInitializer {
                agent: Box::new(Relay {
                    next: format!("relay-{}", (index + 1) % RELAYS),
                    hold: index % 3,
                    held: 0,
                }),
                options: AgentOptions {
                    initial_mode: AgentMode::Reactive,
                    wake_mode: AgentMode::Reactive,
                    initial_queue,
                    name,
                },
            }
        })
        .collect();
    let simulation = simulate(SimulationParameters {
        agent_initializers,
        halt_check: |simulation| simulation.time() >= 4_000,
        ..Default::default()
    });
    simulation
        .agents()
        .iter()
        .flat_map(|agent| &agent.state.queue)
        .fold(histories(&simulation), |fingerprint, message| {
            let payload = message.custom_payload.as_deref().unwrap_or(&[]);
            fingerprint.mix(payload.iter().map(|&byte| u64::from(byte)).sum())
        })
}

/// Selects the candidate with the highest finite mean score.
#[derive(Clone, Copy, Debug)]
struct MaximizeMean;

impl SelectionPolicy<ScalarSummary> for MaximizeMean {
    fn compare(&self, left: &ScalarSummary, right: &ScalarSummary) -> Ordering {
        left.mean.total_cmp(&right.mean)
    }
}

/// The deterministic search benchmark: one simulation per proposed candidate.
fn exp_search_sim() -> Fingerprint {
    let mut search = SeededRandomSearch::new(SEED, 1_000, |rng| rng.next_u64() % 11, MaximizeMean);
    let schedule = FixedReplications::new(1, 0);
    for proposal in search.ask(usize::MAX) {
        let candidate = CandidateId::new(proposal.token.value());
        let records = run_serial(
            SEED,
            schedule.trials(candidate, proposal.candidate),
            |&consumer_period, _context| {
                let simulation = simulate(SimulationParameters {
                    agent_initializers: vec![
                        periodic_producer("producer".to_string(), 2, "consumer".to_string()),
                        periodic_consumer("consumer".to_string(), consumer_period),
                    ],
                    halt_check: |simulation| {
                        simulation
                            .find_by_name("consumer")
                            .is_some_and(|consumer| consumer.state.consumed.len() > 10)
                    },
                    enable_agent_asleep_cycles_metric: true,
                    ..Default::default()
                });
                Ok::<_, Infallible>(consumer_period as f64 - simulation.time() as f64)
            },
        )
        .expect("one unique trial");
        let summary = aggregate_scalar(
            &records,
            |&score| score,
            FailurePolicy::RejectCandidate,
            NonFinitePolicy::RejectCandidate,
        )
        .expect("one finite score");
        search
            .tell(&[Told {
                token: proposal.token,
                candidate,
                summary,
            }])
            .expect("the proposal is outstanding");
    }
    let best = search.best().expect("the budget is nonzero");
    Fingerprint::new()
        .mix(best.candidate_id.value())
        .mix(best.candidate)
        .mix_f64(best.summary.statistics.mean)
}

fn trial_score(offset: f64, context: &TrialContext) -> f64 {
    let mut noise = context.independent_rng(TrialStream::TrialNoise, "noise");
    let mut layout = context.common_rng(TrialStream::LayoutNoise, "layout");
    offset + noise.uniform_f64() + noise.uniform_f64() + layout.uniform_f64()
}

fn summarize_trials(records: &[TrialRecord<f64, Infallible>]) -> Fingerprint {
    let scalar = aggregate_scalar(
        records,
        |&score| score,
        FailurePolicy::RejectCandidate,
        NonFinitePolicy::RejectCandidate,
    )
    .expect("finite scores");
    let bernoulli = aggregate_bernoulli(
        records,
        |&score| score > 1.75,
        FailurePolicy::RejectCandidate,
    )
    .expect("nonempty records");
    Fingerprint::new()
        .mix_f64(scalar.statistics.mean)
        .mix_f64(scalar.statistics.sample_variance.unwrap_or(f64::NAN))
        .mix_f64(scalar.statistics.min)
        .mix_f64(scalar.statistics.max)
        .mix(bernoulli.statistics.successes)
        .mix_f64(bernoulli.statistics.wilson_lower)
}

/// Replicated trials drawing independent and common named streams.
fn exp_run_serial() -> Fingerprint {
    let trials = FixedReplications::new(4_096, 7).trials(CandidateId::new(3), 0.25_f64);
    let records = run_serial(SEED, trials, |&offset, context| {
        Ok::<_, Infallible>(trial_score(offset, context))
    })
    .expect("unique trials");
    summarize_trials(&records)
}

/// The serial trial workload on a dedicated four-worker pool.
#[cfg(feature = "parallel")]
fn exp_run_parallel() -> Fingerprint {
    use simul::experiment::{run_parallel, ParallelConfig};
    use std::num::NonZeroUsize;
    let config = ParallelConfig::new(
        NonZeroUsize::new(4).expect("nonzero workers"),
        NonZeroUsize::new(4_096).expect("nonzero limit"),
    );
    let trials = FixedReplications::new(4_096, 7).trials(CandidateId::new(3), 0.25_f64);
    let records = run_parallel(
        SEED,
        trials,
        config,
        || Ok::<_, Infallible>(()),
        |(), &offset, context| Ok::<_, Infallible>(trial_score(offset, context)),
    )
    .expect("valid parallel batch");
    summarize_trials(&records)
}

fn exp_sample_bounded() -> Fingerprint {
    const BOUNDS: [(f64, f64); 8] = [
        (0.0, 1.0),
        (-1.0, 1.0),
        (10.0, 20.0),
        (-5.0, 5.0),
        (0.0, 0.001),
        (-1e6, 1e6),
        (3.0, 3.5),
        (-2.0, -1.0),
    ];
    sample_bounded(SEED, 512, &BOUNDS)
        .expect("finite ordered bounds")
        .iter()
        .flatten()
        .fold(Fingerprint::new(), |fingerprint, &value| {
            fingerprint.mix_f64(value)
        })
}

/// Strategy overhead only: ask in batches and tell precomputed summaries.
fn exp_random_search() -> Fingerprint {
    let mut search =
        SeededRandomSearch::new(SEED, 4_096, |rng| rng.next_u64() % 10_000, MaximizeMean);
    loop {
        let asked = search.ask(64);
        if asked.is_empty() {
            break;
        }
        let told: Vec<Told<ScalarSummary>> = asked
            .iter()
            .rev()
            .map(|proposal| {
                let candidate = CandidateId::new(proposal.token.value());
                let score = -(proposal.candidate as f64 - 6_543.0).abs();
                Told {
                    token: proposal.token,
                    candidate,
                    summary: CandidateSummary {
                        candidate,
                        requested: 1,
                        completed: 1,
                        failed: 0,
                        statistics: ScalarReducer::from_results([score]).expect("finite score"),
                    },
                }
            })
            .collect();
        search.tell(&told).expect("outstanding proposals");
    }
    let best = search.best().expect("the budget is nonzero");
    Fingerprint::new()
        .mix(best.candidate_id.value())
        .mix(best.candidate)
        .mix_f64(best.summary.statistics.mean)
}

const DIMENSIONS: usize = 5;
const TARGET: [f64; DIMENSIONS] = [0.08, 0.72, 0.24, 0.81, 0.35];
const SEARCH_EVALUATIONS: usize = 480;

fn optimizer(learning_rate: f64) -> CrossEntropyOptimizer<DIMENSIONS> {
    CrossEntropyOptimizer::new(
        CrossEntropyConfig::new([0.5; DIMENSIONS], [0.3; DIMENSIONS])
            .with_dimensions([
                CrossEntropyDimension::Circular,
                CrossEntropyDimension::Linear,
                CrossEntropyDimension::Linear,
                CrossEntropyDimension::Linear,
                CrossEntropyDimension::Linear,
            ])
            .with_minimum_standard_deviation([0.005; DIMENSIONS])
            .with_elite_fraction(0.25)
            .with_learning_rate(learning_rate),
    )
    .expect("valid configuration")
}

/// Multimodal bounded objective with one circular coordinate.
fn objective(point: &[f64; DIMENSIONS]) -> f64 {
    let circular_error = (point[0] - TARGET[0] + 0.5).rem_euclid(1.0) - 0.5;
    let squared_error = point[1..].iter().zip(&TARGET[1..]).fold(
        circular_error * circular_error,
        |sum, (actual, target)| {
            let difference = actual - target;
            difference.mul_add(difference, sum)
        },
    );
    let ripple = (point[1] * 24.0).cos() * (point[3] * 18.0).cos();
    0.03_f64.mul_add(ripple, -squared_error)
}

fn quadratic_objective(point: &[f64; DIMENSIONS]) -> f64 {
    -point.iter().zip(TARGET).fold(0.0, |sum, (actual, target)| {
        let difference = actual - target;
        difference.mul_add(difference, sum)
    })
}

fn optimizer_state(optimizer: &CrossEntropyOptimizer<DIMENSIONS>) -> Fingerprint {
    let best = optimizer.best().map_or(f64::NAN, |sample| sample.score);
    optimizer
        .mean()
        .iter()
        .chain(optimizer.standard_deviation())
        .fold(Fingerprint::new().mix_f64(best), |fingerprint, &value| {
            fingerprint.mix_f64(value)
        })
}

/// One ask/evaluate/tell generation of a compile-time population size.
fn exp_cem_generation<const POPULATION: usize>() -> Fingerprint {
    let mut optimizer = optimizer(0.6);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut samples = [CrossEntropySample::new([0.0; DIMENSIONS], f64::NAN); POPULATION];
    for sample in &mut samples {
        sample.point = optimizer.ask(&mut rng);
        sample.score = quadratic_objective(&sample.point);
    }
    optimizer.tell(&mut samples).expect("valid population");
    optimizer_state(&optimizer)
}

/// A complete fixed-budget adaptive CEM search.
fn exp_cem_search() -> Fingerprint {
    const POPULATION: usize = 24;
    let mut optimizer = optimizer(0.6);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut samples = [CrossEntropySample::new([0.0; DIMENSIONS], f64::NAN); POPULATION];
    for _ in 0..SEARCH_EVALUATIONS.div_euclid(POPULATION) {
        for sample in &mut samples {
            sample.point = optimizer.ask(&mut rng);
            sample.score = objective(&sample.point);
        }
        optimizer.tell(&mut samples).expect("valid population");
    }
    optimizer_state(&optimizer)
}

fn reflect(value: f64) -> f64 {
    let reflected = value.rem_euclid(2.0);
    if reflected <= 1.0 {
        reflected
    } else {
        2.0 - reflected
    }
}

/// Simulated annealing over the CEM objective at the same evaluation budget.
fn exp_annealing() -> Fingerprint {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let best = simulated_annealing_search_with_rng(
        &mut rng,
        |_| [0.5_f64; DIMENSIONS],
        |current, rng| {
            let mut proposal = *current;
            for (dimension, value) in proposal.iter_mut().enumerate() {
                let perturbed = *value + rng.random_range(-0.15..0.15);
                *value = if dimension == 0 {
                    perturbed.rem_euclid(1.0)
                } else {
                    reflect(perturbed)
                };
            }
            proposal
        },
        objective,
        |proposal| 0.2 * 0.01_f64.powf(f64::from(proposal) / SEARCH_EVALUATIONS as f64),
        479,
    )
    .expect("finite objective");
    best.iter().fold(Fingerprint::new(), |fingerprint, &value| {
        fingerprint.mix_f64(value)
    })
}
