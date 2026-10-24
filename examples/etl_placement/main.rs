//! Where should a Supabase ETL replicator run when the source Postgres is in
//! us-west-2 and the destination is in us-east-1?
//!
//! The simulation runs the replicator next to the source (`etl@us-west-2`) or
//! next to the destination (`etl@us-east-1`) under identical workloads and
//! faults (common random numbers), and reports replication latency, sustained
//! throughput, slot lag, cross-region transfer, copy duration, and recovery.
//!
//! Model: three `simul` agents (`postgres`, `etl`, `destination`) exchanging
//! units over flow-level TCP connections. One tick is 100 µs. See `config.rs`
//! for every parameter and its source, `net.rs` for the TCP model, and
//! `etl.rs` for the replicator behavior taken from the supabase/etl source.
//!
//! ```text
//! cargo run --release --example etl_placement -- [cdc|copy|faults|all] [--reps N] [--quick]
//! ```
#![allow(
    clippy::missing_docs_in_private_items,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "self-contained example; module docs describe the model"
)]
// rustc's `unreachable_pub` (denied crate-wide) demands `pub(crate)` in a
// binary's modules, which this nursery lint then flags; the two conflict.
#![allow(clippy::redundant_pub_crate)]

mod config;
mod etl;
mod metrics;
mod net;
mod rng;
mod sink;
mod source;
mod time;
mod wire;

use config::{Config, Durability, Placement, TableSpec};
use metrics::{Recorder, SharedRecorder};
use net::{FaultPlan, Peer, Window};
use rng::Rng;
use simul::agent::{AgentInitializer, AgentMode, AgentOptions};
use simul::experiment::{
    aggregate_scalar, run_serial, CandidateId, FailurePolicy, FixedReplications, MasterSeed,
    NonFinitePolicy, TrialContext, TrialRecord, TrialSpec,
};
use simul::{Halt, Simulation};
use std::convert::Infallible;
use std::rc::Rc;
use time::{secs, ticks, Tick};

const MASTER_SEED: MasterSeed = MasterSeed::from_u64(0x00E7_1A2C_E3E7);
const MB: f64 = 1e6;

fn proactive(name: &str, agent: impl simul::agent::Agent + 'static) -> AgentInitializer {
    AgentInitializer {
        agent: Box::new(agent),
        options: AgentOptions {
            initial_mode: AgentMode::Proactive,
            wake_mode: AgentMode::Proactive,
            name: name.to_owned(),
            ..AgentOptions::default()
        },
    }
}

/// Everything one run reports.
#[derive(Clone, Debug)]
struct Outcome {
    latency_p50_s: f64,
    latency_p99_s: f64,
    latency_max_s: f64,
    /// Peak replication lag after warmup.
    lag_max_s: f64,
    replayed_rows: u64,
    /// WAL retention peak, which follows slot feedback, not progress.
    slot_lag_max_mb: f64,
    /// Least-squares slope of replication lag after warmup; persistently
    /// positive means the pipeline is falling behind.
    lag_growth_s_per_min: f64,
    mean_batch_mb: f64,
    write_p50_s: f64,
    checkpoint_p50_s: f64,
    paused_fraction: f64,
    cross_region_gb: f64,
    elapsed_s: f64,
    copy_s: f64,
    copy_mb_per_s: f64,
    blocked_s: f64,
    recovery_s: f64,
    /// Committed transaction groups and how many never became durable;
    /// `None` unless the run was audited.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by the delivery law test; sweeps skip the audit"
        )
    )]
    delivery: Option<(usize, usize)>,
    events: Vec<(Tick, String)>,
}

fn run_trial(config: &Config, context: &TrialContext) -> Outcome {
    let config = Rc::new(config.clone());
    let recorder = Recorder::shared(config.warmup, config.workload.tables, config.audit_delivery);
    let postgres = source::PostgresAgent::new(
        Rc::clone(&config),
        Rc::clone(&recorder),
        Rng::common(context, "network/postgres"),
        Rng::common(context, "workload"),
        Peer::new(etl::NAME, config.source_path()),
    );
    let replicator = etl::EtlAgent::new(
        Rc::clone(&config),
        Rc::clone(&recorder),
        Rng::common(context, "network/etl"),
        Peer::new(source::NAME, config.source_path()),
        Peer::new(sink::NAME, config.destination_path()),
    );
    let destination = sink::DestinationAgent::new(
        Rc::clone(&config),
        Rc::clone(&recorder),
        Rng::common(context, "network/destination"),
        Rng::common(context, "destination-service"),
        Peer::new(etl::NAME, config.destination_path()),
    );
    let mut simulation = Simulation::builder(Halt::AtTime(config.duration))
        .agents([
            proactive(source::NAME, postgres),
            proactive(etl::NAME, replicator),
            proactive(sink::NAME, destination),
        ])
        .build()
        .unwrap_or_else(|error| panic!("the agent layout is valid: {error}"));
    simulation.run();
    summarize(&config, &recorder, simulation.time())
}

fn summarize(config: &Config, recorder: &SharedRecorder, end: Tick) -> Outcome {
    let recorder = recorder.borrow();
    let lag: Vec<(f64, f64)> = recorder
        .replication_lag_s
        .iter()
        .filter(|(at, _)| *at >= config.warmup)
        .map(|(at, lag)| (secs(*at), *lag))
        .collect();
    let slot_lag_max_mb = recorder
        .slot_lag_bytes
        .iter()
        .filter(|(at, _)| *at >= config.warmup)
        .map(|(_, bytes)| *bytes as f64 / MB)
        .fold(0.0, f64::max);
    let recovery_s = config.faults.first_fault().map_or(f64::NAN, |start| {
        recovery_seconds(&recorder.replication_lag_s, config.warmup, start)
    });
    let copy_s = recorder.copy_done_at.map_or(f64::NAN, secs);
    Outcome {
        latency_p50_s: recorder.latency.quantile(0.5).unwrap_or(f64::NAN),
        latency_p99_s: recorder.latency.quantile(0.99).unwrap_or(f64::NAN),
        latency_max_s: if recorder.latency.total() > 0 {
            recorder.latency.max()
        } else {
            f64::NAN
        },
        lag_max_s: lag.iter().map(|(_, lag)| *lag).fold(0.0, f64::max),
        replayed_rows: recorder.rows_replayed,
        slot_lag_max_mb,
        lag_growth_s_per_min: least_squares_slope(&lag) * 60.0,
        mean_batch_mb: if recorder.batches > 0 {
            recorder.batch_decoded_bytes / recorder.batches as f64 / MB
        } else {
            f64::NAN
        },
        write_p50_s: recorder.write_seconds.quantile(0.5).unwrap_or(f64::NAN),
        checkpoint_p50_s: recorder
            .checkpoint_seconds
            .quantile(0.5)
            .unwrap_or(f64::NAN),
        paused_fraction: recorder.paused_ticks as f64 / end.max(1) as f64,
        cross_region_gb: recorder.cross_region_bytes.iter().sum::<f64>() / 1e9,
        elapsed_s: secs(end),
        copy_s,
        copy_mb_per_s: recorder.copy_bytes / MB / copy_s,
        blocked_s: secs(recorder.blocked_ticks),
        recovery_s,
        delivery: recorder.delivery(),
        events: recorder.events.clone(),
    }
}

fn least_squares_slope(points: &[(f64, f64)]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let n = points.len() as f64;
    let mean_x = points.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_y = points.iter().map(|(_, y)| y).sum::<f64>() / n;
    let covariance: f64 = points
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let variance: f64 = points.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
    if variance > 0.0 {
        covariance / variance
    } else {
        0.0
    }
}

/// Samples (one per second) lag must stay in range to count as recovered.
const RECOVERY_WINDOW: usize = 15;

/// Seconds from the fault until replication lag, after first leaving its
/// pre-fault range, returns and stays there for [`RECOVERY_WINDOW`] samples.
/// Zero when the fault never pushed lag out of range; infinite when lag never
/// settled again before the run ended.
fn recovery_seconds(series: &[(Tick, f64)], warmup: Tick, fault: Tick) -> f64 {
    let before = series
        .iter()
        .filter(|(at, _)| *at >= warmup && *at < fault)
        .map(|(_, lag)| *lag)
        .fold(f64::NAN, f64::max);
    if before.is_nan() {
        return f64::NAN;
    }
    // Samples are one second apart; allow a quarter second of slack.
    let threshold = before.mul_add(1.1, 0.25);
    let after: Vec<(Tick, f64)> = series
        .iter()
        .copied()
        .filter(|(at, _)| *at >= fault)
        .collect();
    let Some(excursion) = after.iter().position(|(_, lag)| *lag > threshold) else {
        return 0.0;
    };
    after[excursion..]
        .windows(RECOVERY_WINDOW)
        .find(|window| window.iter().all(|(_, lag)| *lag <= threshold))
        .map_or(f64::INFINITY, |window| secs(window[0].0 - fault))
}

type Records = Vec<TrialRecord<Outcome, Infallible>>;

/// Runs every placement of every scenario. Both placements of a scenario
/// share one common-random group, so they face identical workload and fault
/// draws.
fn run_scenarios(scenarios: &[(String, Config)], reps: u32) -> Vec<(String, Placement, Records)> {
    let jobs: Vec<(usize, Placement)> = (0..scenarios.len())
        .flat_map(|index| Placement::ALL.map(|placement| (index, placement)))
        .collect();
    let workers = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let chunk = jobs.len().div_ceil(workers).max(1);
    let mut results: Vec<(usize, Placement, Records)> = std::thread::scope(|scope| {
        // Spawn every worker before joining any, or the runs would serialize.
        #[allow(clippy::needless_collect)]
        let handles: Vec<_> = jobs
            .chunks(chunk)
            .map(|batch| {
                scope.spawn(move || {
                    batch
                        .iter()
                        .map(|&(index, placement)| {
                            let mut config = scenarios[index].1.clone();
                            config.placement = placement;
                            let candidate = CandidateId::new(
                                index as u64 * 2
                                    + u64::from(placement == Placement::WithDestination),
                            );
                            let schedule = FixedReplications::new(reps, index as u64);
                            let trials: Vec<TrialSpec<Config>> =
                                schedule.trials(candidate, config).collect();
                            let records = run_serial(MASTER_SEED, trials, |config, context| {
                                Ok::<_, Infallible>(run_trial(config, context))
                            })
                            .unwrap_or_else(|error| {
                                panic!("trial identities are unique: {error:?}")
                            });
                            (index, placement, records)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| panic!("a simulation thread panicked"))
            })
            .collect()
    });
    results.sort_by_key(|(index, placement, _)| (*index, *placement == Placement::WithDestination));
    results
        .into_iter()
        .map(|(index, placement, records)| (scenarios[index].0.clone(), placement, records))
        .collect()
}

/// Mean of a metric over replications; NaN when no replication produced a
/// finite value.
fn mean(records: &[TrialRecord<Outcome, Infallible>], metric: impl Fn(&Outcome) -> f64) -> f64 {
    aggregate_scalar(
        records,
        metric,
        FailurePolicy::Ignore,
        NonFinitePolicy::Ignore,
    )
    .map_or(f64::NAN, |summary| summary.statistics.mean())
}

/// Mean recovery over replications that recovered, and how many never did.
fn fmt_recovery(outcomes: &[TrialRecord<Outcome, Infallible>]) -> String {
    let never = outcomes
        .iter()
        .filter(|record| matches!(&record.result, Ok(outcome) if outcome.recovery_s.is_infinite()))
        .count();
    let recovered = fmt_s(mean(outcomes, |o| o.recovery_s));
    if never == 0 {
        recovered
    } else {
        format!("{recovered} ({never} of {} never)", outcomes.len())
    }
}
fn fmt_s(seconds: f64) -> String {
    if seconds.is_nan() {
        "—".to_owned()
    } else if seconds.is_infinite() {
        "never".to_owned()
    } else if seconds < 1.0 {
        format!("{:.0} ms", seconds * 1e3)
    } else {
        format!("{seconds:.2} s")
    }
}

fn dollars_per_month(outcomes: &[TrialRecord<Outcome, Infallible>], config: &Config) -> f64 {
    let per_second = mean(outcomes, |outcome| {
        outcome.cross_region_gb / outcome.elapsed_s
    });
    per_second * 86_400.0 * 30.0 * config.cross_region_dollars_per_gb
}

// ---- scenarios -------------------------------------------------------------

fn cdc_scenarios(options: &Options) -> Vec<(String, Config)> {
    let default_rates: &[f64] = if options.quick {
        &[1_000.0, 50_000.0]
    } else {
        &[100.0, 1_000.0, 10_000.0, 50_000.0, 150_000.0]
    };
    let default_fills: &[f64] = if options.quick {
        &[1_000.0]
    } else {
        &[10_000.0, 1_000.0, 100.0]
    };
    let rates = options.rates.as_deref().unwrap_or(default_rates);
    let fills = options.fills.as_deref().unwrap_or(default_fills);
    let mut scenarios = Vec::new();
    for &fill in fills {
        for &rate in rates {
            let mut config = Config::baseline(Placement::WithSource);
            config.workload.rows_per_s = rate;
            config.etl.max_fill_ms = fill;
            if options.quick {
                config.duration = ticks(120.0);
                config.warmup = ticks(30.0);
            }
            scenarios.push((
                format!("{rate:>7.0} rows/s, max_fill {fill:>5.0} ms"),
                config,
            ));
        }
    }
    scenarios
}

fn copy_scenarios(quick: bool) -> Vec<(String, Config)> {
    let mut config = Config::baseline(Placement::WithSource);
    // ASSUMPTION: eight tables, 17 GB of 200 B rows, skewed like real schemas.
    let rows: &[u64] = if quick {
        &[4_000_000, 2_000_000, 1_000_000, 500_000]
    } else {
        &[
            40_000_000, 20_000_000, 10_000_000, 5_000_000, 5_000_000, 2_500_000, 1_250_000, 600_000,
        ]
    };
    config.copy_tables = rows
        .iter()
        .map(|&rows| TableSpec {
            rows,
            data_bytes_per_row: 200.0,
        })
        .collect();
    config.workload.tables = config.copy_tables.len();
    config.workload.rows_per_s = 1_000.0;
    config.etl.max_fill_ms = 1_000.0;
    config.duration = ticks(7_200.0);
    config.warmup = 0;
    vec![("initial copy, 1k rows/s CDC".to_owned(), config)]
}

fn fault_scenarios(quick: bool) -> Vec<(String, Config)> {
    let start = ticks(150.0);
    let window = |seconds: f64| Window {
        start,
        end: start + ticks(seconds),
    };
    let plans: Vec<(&str, FaultPlan)> = vec![
        (
            "source resets replication connection",
            FaultPlan {
                replication_resets: vec![start],
                ..FaultPlan::default()
            },
        ),
        (
            "1% cross-region loss for 60 s",
            FaultPlan {
                loss_bursts: vec![(window(60.0), 0.01)],
                ..FaultPlan::default()
            },
        ),
        (
            "cross-region partition for 30 s",
            FaultPlan {
                outages: vec![window(30.0)],
                ..FaultPlan::default()
            },
        ),
        (
            "cross-region partition for 90 s",
            FaultPlan {
                outages: vec![window(90.0)],
                ..FaultPlan::default()
            },
        ),
        (
            "ETL process crash",
            FaultPlan {
                etl_crashes: vec![start],
                ..FaultPlan::default()
            },
        ),
        (
            "destination 10× slower for 60 s",
            FaultPlan {
                destination_brownouts: vec![(window(60.0), 10.0)],
                ..FaultPlan::default()
            },
        ),
    ];
    let plans = if quick {
        plans.into_iter().take(2).collect()
    } else {
        plans
    };
    plans
        .into_iter()
        .map(|(name, faults)| {
            let mut config = Config::baseline(Placement::WithSource);
            config.workload.rows_per_s = 10_000.0;
            config.etl.max_fill_ms = 1_000.0;
            config.duration = ticks(if quick { 300.0 } else { 480.0 });
            config.warmup = ticks(60.0);
            config.faults = faults;
            (name.to_owned(), config)
        })
        .collect()
}

// ---- reports ---------------------------------------------------------------

fn print_assumptions() {
    let config = Config::baseline(Placement::WithSource);
    println!("## Model inputs\n");
    println!("| input | value | source |");
    println!("|---|---|---|");
    let rows = [
        ("cross-region RTT", format!("LogNormal(median {:.0} ms, σ {}) per connection, floor {:.0} ms, re-routes every ~{:.0} s", config.cross_region.rtt_median_s * 1e3, config.cross_region.rtt_sigma, config.cross_region.rtt_floor_s * 1e3, 170.0), "AWS NMIP, CloudPing, Reda 2020"),
        ("intra-region RTT", format!("LogNormal(median {:.1} ms, σ {})", config.intra_region.rtt_median_s * 1e3, config.intra_region.rtt_sigma), "cross-AZ pings"),
        ("baseline loss", format!("{:e}", config.cross_region.loss), "Reda 2020 upper bound"),
        ("TCP", format!("CUBIC, IW 10, rwnd {:.0} MiB, sndbuf {:.0} MiB, RTO ≥ {:.0} ms, slow start after idle", config.tcp.receive_window_bytes / 1_048_576.0, config.tcp.send_buffer_bytes / 1_048_576.0, config.tcp.rto_min_s * 1e3), "Linux ≤ 6.15 defaults"),
        ("Postgres login", format!("{} RTT (TCP, SSLRequest, TLS 1.3, SCRAM ×3)", config.postgres_handshake_round_trips), "Postgres protocol docs"),
        ("walsender decode", format!("{:.0}k rows/s", config.source.decode_rows_per_s / 1e3), "PeerDB benchmark"),
        ("COPY backend", format!("{:.1} MB/s per connection", config.source.copy_bytes_per_s / MB), "galaxy-io benchmark"),
        ("WAL per row / pgoutput row", format!("{:.0} B / {:.0} B", config.source.wal_bytes_per_row, config.row_wire_bytes(config.workload.data_bytes_per_row)), "galaxy-io counters; pgoutput format"),
        ("ETL batching", format!("max_fill_ms swept; target min(32 MiB, 0.2 × {:.0} GiB ÷ slots)", config.etl.memory_bytes / 1_073_741_824.0), "supabase/etl BatchConfig"),
        ("ETL decode / copy parse", format!("{:.0}k rows/s / {:.0} MB/s", config.etl.decode_rows_per_s / 1e3, config.etl.copy_parse_bytes_per_s / MB), "ASSUMPTION"),
        ("decoded size", format!("{}× pgoutput size", config.etl.decoded_size_factor), "ASSUMPTION"),
        ("store checkpoint", "BEGIN, advisory lock, UPSERT, COMMIT per durable batch; 2-conn pool, 30 s idle".to_owned(), "supabase/etl PostgresStore"),
        ("store statement / COMMIT", format!("{:.1} ms / {:.1} ms", config.source.statement_s * 1e3, config.source.commit_s * 1e3), "ASSUMPTION"),
        ("sink", format!("LogNormal(median {:.0} ms, σ {}) + {:.0} MB/s per request, ≤ {} concurrent, 1 request per table per batch (≤ {})", config.sink.latency_median_s * 1e3, config.sink.latency_sigma, config.sink.ingest_bytes_per_s / MB, config.sink.max_concurrent_requests, config.sink.max_parallel_requests), "ASSUMPTION"),
        ("sink payload / keep-alive", format!("{}× row data / {:.0} s", config.sink.encoding_ratio, config.sink.keepalive_timeout_s), "ASSUMPTION"),
        ("workload", format!("Poisson transactions, {:.0} rows each on average, {} tables, {:.0} B rows, 3× bursts 10% of the time", config.workload.mean_rows_per_transaction, config.workload.tables, config.workload.data_bytes_per_row), "ASSUMPTION"),
        ("cross-region transfer", format!("${}/GB", config.cross_region_dollars_per_gb), "AWS Price List API"),
    ];
    for (input, value, source) in rows {
        println!("| {input} | {value} | {source} |");
    }
    println!();
}

fn print_cdc(results: &[(String, Placement, Records)], scenarios: &[(String, Config)]) {
    println!("## Steady-state CDC\n");
    println!("Latency is commit on the source to durable at the destination. Lag growth is the slope of replication lag (age of the newest durable commit); a value that stays positive means the pipeline cannot keep up. Slot lag is WAL retained on the source, which follows keepalive-driven feedback.\n");
    println!("| workload | placement | p50 | p99 | max | lag growth s/min | max slot lag MB | batch MB | write p50 | checkpoint p50 | paused | cross-region $/month |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for (name, placement, outcomes) in results {
        let config = &scenarios
            .iter()
            .find(|(scenario, _)| scenario == name)
            .unwrap_or_else(|| panic!("results come from the scenarios"))
            .1;
        println!(
            "| {name} | {} | {} | {} | {} | {:.1} | {:.0} | {:.2} | {} | {} | {:.0}% | {:.0} |",
            placement.label(),
            fmt_s(mean(outcomes, |o| o.latency_p50_s)),
            fmt_s(mean(outcomes, |o| o.latency_p99_s)),
            fmt_s(mean(outcomes, |o| o.latency_max_s)),
            mean(outcomes, |o| o.lag_growth_s_per_min),
            mean(outcomes, |o| o.slot_lag_max_mb),
            mean(outcomes, |o| o.mean_batch_mb),
            fmt_s(mean(outcomes, |o| o.write_p50_s)),
            fmt_s(mean(outcomes, |o| o.checkpoint_p50_s)),
            100.0 * mean(outcomes, |o| o.paused_fraction),
            dollars_per_month(outcomes, config),
        );
    }
    println!();
}

fn print_copy(results: &[(String, Placement, Records)]) {
    println!("## Initial table copy\n");
    println!("| scenario | placement | all tables ready | copy MB/s | apply blocked by catch-up | CDC p99 during copy | cross-region GB |");
    println!("|---|---|---|---|---|---|---|");
    for (name, placement, outcomes) in results {
        println!(
            "| {name} | {} | {} | {:.0} | {} | {} | {:.1} |",
            placement.label(),
            fmt_s(mean(outcomes, |o| o.copy_s)),
            mean(outcomes, |o| o.copy_mb_per_s),
            fmt_s(mean(outcomes, |o| o.blocked_s)),
            fmt_s(mean(outcomes, |o| o.latency_p99_s)),
            mean(outcomes, |o| o.cross_region_gb),
        );
    }
    println!();
}

fn print_faults(results: &[(String, Placement, Records)], scenarios: &[(String, Config)]) {
    let Some((_, config)) = scenarios.first() else {
        return;
    };
    let fault = config.faults.first_fault().unwrap_or(0);
    println!(
        "## Faults ({:.0} rows/s, max_fill {:.0} ms, fault at t = {:.0} s)\n",
        config.workload.rows_per_s,
        config.etl.max_fill_ms,
        secs(fault)
    );
    println!("Recovery runs from the fault until replication lag (age of the newest durable commit) is back within its pre-fault range for good.\n");
    println!("| fault | placement | recovery | peak replication lag | p99 latency | max slot lag MB | replayed rows |");
    println!("|---|---|---|---|---|---|---|");
    for (name, placement, outcomes) in results {
        println!(
            "| {name} | {} | {} | {} | {} | {:.0} | {:.0} |",
            placement.label(),
            fmt_recovery(outcomes),
            fmt_s(mean(outcomes, |o| o.lag_max_s)),
            fmt_s(mean(outcomes, |o| o.latency_p99_s)),
            mean(outcomes, |o| o.slot_lag_max_mb),
            mean(outcomes, |o| o.replayed_rows as f64),
        );
    }
    println!();
    println!("Replicator events in the first replication (faults that only slow the network log none):\n");
    for (name, placement, outcomes) in results {
        let Some(Ok(first)) = outcomes.first().map(|record| &record.result) else {
            continue;
        };
        let events: Vec<String> = first
            .events
            .iter()
            .filter(|(at, _)| *at + ticks(10.0) >= fault)
            .take(6)
            .map(|(at, text)| format!("{:.1} s {text}", secs(*at)))
            .collect();
        if !events.is_empty() {
            println!("- {name} / {}: {}", placement.label(), events.join("; "));
        }
    }
    println!();
}

/// Command-line options.
#[derive(Debug)]
struct Options {
    mode: String,
    quick: bool,
    reps: u32,
    polled_sink: bool,
    /// Constant arrival rate instead of 3× bursts.
    steady: bool,
    /// Advertised receive window override, in MiB.
    receive_window_mib: Option<f64>,
    rates: Option<Vec<f64>>,
    fills: Option<Vec<f64>>,
}

const USAGE: &str = "usage: etl_placement [cdc|copy|faults|all] [--quick] [--reps N] \
[--rates R1,R2,…] [--fills MS1,MS2,…] [--steady] [--polled-sink] [--rwnd-mib N]";

fn parse_list(value: Option<&str>, flag: &str) -> Vec<f64> {
    value
        .unwrap_or_default()
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<f64>()
                .ok()
                .filter(|number| number.is_finite() && *number > 0.0)
                .unwrap_or_else(|| panic!("{flag} takes positive numbers\n{USAGE}"))
        })
        .collect()
}

fn parse_options() -> Options {
    let mut options = Options {
        mode: "all".to_owned(),
        quick: false,
        reps: 0,
        polled_sink: false,
        steady: false,
        receive_window_mib: None,
        rates: None,
        fills: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "cdc" | "copy" | "faults" | "all" => options.mode = arg,
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--quick" => options.quick = true,
            "--polled-sink" => options.polled_sink = true,
            "--steady" => options.steady = true,
            "--rwnd-mib" => {
                options.receive_window_mib = parse_list(args.next().as_deref(), "--rwnd-mib")
                    .first()
                    .copied();
            }
            "--reps" => {
                options.reps = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .filter(|reps| *reps > 0)
                    .unwrap_or_else(|| panic!("--reps takes a positive integer\n{USAGE}"));
            }
            "--rates" => options.rates = Some(parse_list(args.next().as_deref(), "--rates")),
            "--fills" => options.fills = Some(parse_list(args.next().as_deref(), "--fills")),
            other => panic!("unknown argument {other}\n{USAGE}"),
        }
    }
    if options.reps == 0 {
        options.reps = if options.quick { 1 } else { 3 };
    }
    options
}

/// Applies command-line overrides shared by every scenario.
fn with_overrides(
    mut scenarios: Vec<(String, Config)>,
    options: &Options,
) -> Vec<(String, Config)> {
    for (_, config) in &mut scenarios {
        if options.polled_sink {
            // Snowflake-style: accepted on response, durability polled every
            // 500 ms [etl]; ASSUMPTION: durable 1 s after acceptance.
            config.sink.durability = Durability::Polled {
                interval_s: 0.5,
                commit_delay_s: 1.0,
            };
        }
        if options.steady {
            config.workload.burst = None;
        }
        if let Some(mib) = options.receive_window_mib {
            config.tcp.receive_window_bytes = mib * 1024.0 * 1024.0;
        }
    }
    scenarios
}

fn main() {
    let options = parse_options();
    let mode = options.mode.as_str();
    print_assumptions();
    if matches!(mode, "cdc" | "all") {
        let scenarios = with_overrides(cdc_scenarios(&options), &options);
        let results = run_scenarios(&scenarios, options.reps);
        print_cdc(&results, &scenarios);
    }
    if matches!(mode, "copy" | "all") {
        let scenarios = with_overrides(copy_scenarios(options.quick), &options);
        let results = run_scenarios(&scenarios, options.reps.min(2));
        print_copy(&results);
    }
    if matches!(mode, "faults" | "all") {
        let scenarios = with_overrides(fault_scenarios(options.quick), &options);
        let results = run_scenarios(&scenarios, options.reps);
        print_faults(&results, &scenarios);
    }
}

#[cfg(test)]
mod tests {
    use super::{run_scenarios, ticks, Config, FaultPlan, Placement, Window};

    /// Each fault strikes at 20 s; the workload stops at 100 s and the run
    /// continues until 240 s, long enough for the slowest recovery (a
    /// partition past `wal_sender_timeout`, detected late through RTO
    /// backoff, then a 10 s retry) to drain.
    fn drained(name: &str, faults: FaultPlan) -> (String, Config) {
        let mut config = Config::baseline(Placement::WithSource);
        config.workload.rows_per_s = 2_000.0;
        config.etl.max_fill_ms = 1_000.0;
        config.warmup = ticks(5.0);
        config.workload_until = Some(ticks(100.0));
        config.duration = ticks(240.0);
        config.audit_delivery = true;
        config.faults = faults;
        (name.to_owned(), config)
    }

    #[test]
    fn faults_lose_no_committed_transaction_in_either_placement() {
        let fault = ticks(20.0);
        let scenarios = [
            drained(
                "ETL crash",
                FaultPlan {
                    etl_crashes: vec![fault],
                    ..FaultPlan::default()
                },
            ),
            drained(
                "replication reset",
                FaultPlan {
                    replication_resets: vec![fault],
                    ..FaultPlan::default()
                },
            ),
            drained(
                "70 s partition",
                FaultPlan {
                    outages: vec![Window {
                        start: fault,
                        end: fault + ticks(70.0),
                    }],
                    ..FaultPlan::default()
                },
            ),
        ];
        for (name, placement, records) in run_scenarios(&scenarios, 2) {
            for record in records {
                let Ok(outcome) = record.result;
                let Some((committed, lost)) = outcome.delivery else {
                    panic!("{name} ran without the delivery audit");
                };
                assert!(
                    committed > 1_000,
                    "{name}: only {committed} groups committed"
                );
                assert_eq!(
                    lost,
                    0,
                    "{name} at {} lost {lost} of {committed} committed groups (replication {:?})",
                    placement.label(),
                    record.key.replication(),
                );
            }
        }
    }
}
