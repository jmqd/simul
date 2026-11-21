//! Five analytical experiment views, statistical reductions, Spearman rank
//! sensitivity analysis, and TSV dataset exporters for gnuplot.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use simul::experiment::search::sample_sobol;
use simul::experiment::{
    BernoulliReducer, CandidateId, FailurePolicy, FixedReplications, MasterSeed,
    MixedCrossEntropyConfig, MixedCrossEntropyOptimizer, MixedCrossEntropySample, MixedDimension,
    NonFinitePolicy, ObjectiveDirection, ParetoArchive, ScalarReducer, ScalarSummary, SeedDeriver,
    TrialRecord, TrialSpec, compare_paired_candidates,
};
use simul::trace::{ReplayWindow, TraceEventKind, forensic_replay_trial_with_payloads};
use simul::{DiscreteTime, SimulationError};

use crate::config::{BulkheadMode, CliConfig, PoolingMode, SimSpec};
use crate::model::{PoolOutcome, build_simulation, evaluate_run};

/// Pool sizes evaluated in the HikariCP/USL knee sweep.
const SWEEP_POOL_SIZES: [usize; 14] = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64, 96];
/// Pool sizes evaluated in the nested-pool deadlock phase diagram.
const DEADLOCK_POOL_SIZES: [usize; 9] = [4, 6, 8, 10, 12, 16, 20, 24, 32];
/// Nested acquisition fractions evaluated in the deadlock phase diagram.
const DEADLOCK_NESTED_FRACS: [f64; 3] = [0.05, 0.15, 0.30];

/// Boxed error alias for experiment orchestration.
pub(crate) type DynError = Box<dyn std::error::Error>;

/// Runs a batch of trials using `run_parallel` when the `parallel` feature is
/// enabled, or `run_serial` otherwise, returning records in canonical order.
fn run_batch(
    seed: MasterSeed,
    trials: impl IntoIterator<Item = TrialSpec<SimSpec>>,
) -> Result<Vec<TrialRecord<PoolOutcome, SimulationError>>, DynError> {
    #[cfg(feature = "parallel")]
    {
        use std::num::NonZeroUsize;
        let workers = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
        let config = simul::experiment::ParallelConfig::new(workers, NonZeroUsize::MAX);
        let records = simul::experiment::run_parallel(
            seed,
            trials,
            config,
            || Ok::<(), std::convert::Infallible>(()),
            |_ws, spec, ctx| evaluate_run(spec, ctx),
        )?;
        Ok(records)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let records = simul::experiment::run_serial(seed, trials, evaluate_run)?;
        Ok(records)
    }
}

/// Reduces one metric across a slice of trial records into a [`ScalarSummary`].
fn reduce_metric(
    records: &[TrialRecord<PoolOutcome, SimulationError>],
    extract: impl Fn(&PoolOutcome) -> f64,
) -> Result<ScalarSummary, DynError> {
    let mut reducer = ScalarReducer::default();
    for rec in records {
        let outcome = rec
            .result
            .as_ref()
            .map_err(|err| format!("trial failed: {err}"))?;
        reducer.observe(extract(outcome))?;
    }
    Ok(reducer.finish()?)
}

/// Half-width of a [`ScalarSummary`]'s two-sided 95% Student-t confidence interval.
fn ci95_half_width(summary: &ScalarSummary) -> f64 {
    summary
        .ci_95()
        .map_or(0.0, |(lo, hi)| 0.5 * (hi - lo).max(0.0))
}

/// Writes `content` to `<export_dir>/<filename>` if `--export-data` was enabled.
fn maybe_write_tsv(
    export_dir: Option<&Path>,
    filename: &str,
    content: &str,
) -> Result<(), DynError> {
    if let Some(dir) = export_dir {
        fs::create_dir_all(dir)?;
        let path = dir.join(filename);
        fs::write(&path, content)?;
        println!("  [exported {}]", path.display());
    }
    Ok(())
}

/// Summarized metrics for one `(pool_size, pooling_mode)` sweep point.
#[derive(Clone, Copy, Debug)]
struct SweepPoint {
    /// Configured connection pool size.
    pool_size: usize,
    /// Mean throughput (req/s).
    tput: f64,
    /// 95% CI half-width of throughput (req/s).
    tput_ci: f64,
    /// Mean success rate in `[0, 1]`.
    succ: f64,
    /// Mean P50 end-to-end latency (ms).
    p50: f64,
    /// Mean P99 end-to-end latency (ms).
    p99: f64,
    /// Mean `CVaR95` end-to-end latency (ms).
    cvar95: f64,
    /// Mean fast-OLTP P99 latency (ms).
    fast_p99: f64,
    /// Mean pool queue wait P95 (ms).
    wait_p95: f64,
    /// Mean DB execution P95 (ms).
    exec_p95: f64,
    /// Mean time-weighted concurrent active DB queries.
    db_mean: f64,
}

/// Summarizes a slice of records for `pool_size` into a [`SweepPoint`].
fn summarize_sweep_slice(
    pool_size: usize,
    slice: &[TrialRecord<PoolOutcome, SimulationError>],
) -> Result<SweepPoint, DynError> {
    let tput_s = reduce_metric(slice, |o| o.throughput_rps)?;
    Ok(SweepPoint {
        pool_size,
        tput: tput_s.mean(),
        tput_ci: ci95_half_width(&tput_s),
        succ: reduce_metric(slice, |o| o.success_rate)?.mean(),
        p50: reduce_metric(slice, |o| o.p50_ms)?.mean(),
        p99: reduce_metric(slice, |o| o.p99_ms)?.mean(),
        cvar95: reduce_metric(slice, |o| o.cvar95_ms)?.mean(),
        fast_p99: reduce_metric(slice, |o| o.fast_p99_ms)?.mean(),
        wait_p95: reduce_metric(slice, |o| o.pool_wait_p95_ms)?.mean(),
        exec_p95: reduce_metric(slice, |o| o.db_exec_p95_ms)?.mean(),
        db_mean: reduce_metric(slice, |o| o.mean_db_active)?.mean(),
    })
}

/// Classifies the limiting physical regime of a sweep point.
fn diagnose_regime(pt: SweepPoint, best_p99: f64, peak_tput: f64) -> &'static str {
    if pt.exec_p95 > 80.0 {
        "USL thrashing cliff"
    } else if pt.wait_p95 > pt.exec_p95 && pt.succ < 0.97 {
        "Pool-starved (wait)"
    } else if pt.p99 <= best_p99 * 1.15 && pt.tput >= peak_tput * 0.985 {
        "Goldilocks knee"
    } else if pt.exec_p95 > 40.0 {
        "USL contention-bound"
    } else {
        "Queue-limited"
    }
}

/// View 1: Sweeps `pool_size` across `[2..=96]` for both Transaction and Session
/// pooling, exposing the `HikariCP`/USL throughput knee and latency U-curve.
pub(crate) fn run_sweep(cfg: &CliConfig) -> Result<(), DynError> {
    let seed = MasterSeed::from_u64(cfg.seed);
    let schedule = FixedReplications::new(cfg.reps, 1);
    let mut trials = Vec::new();
    for (idx, &size) in SWEEP_POOL_SIZES.iter().enumerate() {
        let mut txn = cfg.spec;
        txn.pool.pool_size = size;
        txn.pool.app_workers = txn.pool.app_workers.max(size + 32);
        txn.pool.pooling = PoolingMode::Transaction;
        txn.pool.bulkhead = BulkheadMode::Shared;
        trials.extend(schedule.trials(CandidateId::new(idx as u64), txn));

        let mut sess = txn;
        sess.pool.pooling = PoolingMode::Session;
        trials.extend(schedule.trials(
            CandidateId::new((SWEEP_POOL_SIZES.len() + idx) as u64),
            sess,
        ));
    }
    let records = run_batch(seed, trials)?;
    let reps = cfg.reps as usize;
    let count = SWEEP_POOL_SIZES.len();
    let mut txn_pts = Vec::with_capacity(count);
    let mut sess_pts = Vec::with_capacity(count);
    for (idx, &size) in SWEEP_POOL_SIZES.iter().enumerate() {
        let t_start = idx * reps;
        let s_start = (count + idx) * reps;
        txn_pts.push(summarize_sweep_slice(
            size,
            &records[t_start..t_start + reps],
        )?);
        sess_pts.push(summarize_sweep_slice(
            size,
            &records[s_start..s_start + reps],
        )?);
    }
    report_sweep(cfg, &txn_pts, &sess_pts)
}

/// Prints the pool-size sweep table and exports `sweep.tsv`.
fn report_sweep(
    cfg: &CliConfig,
    txn_pts: &[SweepPoint],
    sess_pts: &[SweepPoint],
) -> Result<(), DynError> {
    let peak_tput = txn_pts.iter().map(|p| p.tput).fold(0.0_f64, f64::max);
    let best_p99 = txn_pts.iter().map(|p| p.p99).fold(f64::INFINITY, f64::min);
    let knee_n = cfg.spec.hardware.theoretical_usl_knee();

    println!("\n=== VIEW 1: POOL-SIZE SWEEP & HIKARICP / USL KNEE ===");
    println!(
        "Hardware: {} DB cores, sigma={:.2}, kappa={:.2} (analytical USL peak n* = {:.1}) | reps={}",
        cfg.spec.hardware.db_cores,
        cfg.spec.hardware.usl_sigma,
        cfg.spec.hardware.usl_kappa,
        knee_n,
        cfg.reps
    );
    println!(
        "{:>4} | {:>10} | {:>6} | {:>6} | {:>7} | {:>7} | {:>8} | {:>8} | {:>10} | {:<20}",
        "Pool",
        "Txn req/s",
        "Succ%",
        "P50ms",
        "P99ms",
        "CVaR95",
        "WaitP95",
        "ExecP95",
        "Sess req/s",
        "Regime"
    );
    println!("{}", "-".repeat(110));

    let mut tsv = String::from(
        "pool_size\ttxn_tput\ttxn_tput_ci\ttxn_succ_pct\ttxn_p50\ttxn_p99\ttxn_cvar95\ttxn_fast_p99\ttxn_wait_p95\ttxn_exec_p95\ttxn_db_mean\tsess_tput\tsess_succ_pct\tsess_p99\tsess_fast_p99\n",
    );
    for (t, s) in txn_pts.iter().zip(sess_pts.iter()) {
        let regime = diagnose_regime(*t, best_p99, peak_tput);
        println!(
            "{:>4} | {:>6.1}+/-{:<3.0} | {:>5.1}% | {:>6.1} | {:>7.1} | {:>7.1} | {:>8.1} | {:>8.1} | {:>10.1} | {:<20}",
            t.pool_size,
            t.tput,
            t.tput_ci,
            t.succ * 100.0,
            t.p50,
            t.p99,
            t.cvar95,
            t.wait_p95,
            t.exec_p95,
            s.tput,
            regime
        );
        let _ = writeln!(
            tsv,
            "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
            t.pool_size,
            t.tput,
            t.tput_ci,
            t.succ * 100.0,
            t.p50,
            t.p99,
            t.cvar95,
            t.fast_p99,
            t.wait_p95,
            t.exec_p95,
            t.db_mean,
            s.tput,
            s.succ * 100.0,
            s.p99,
            s.fast_p99
        );
    }
    maybe_write_tsv(cfg.export_dir.as_deref(), "sweep.tsv", &tsv)
}

/// Named architectural configuration evaluated in View 2.
#[derive(Clone, Debug)]
struct PatternSpec {
    /// Display label.
    label: String,
    /// Short plot-axis label.
    short_label: String,
    /// Simulation configuration.
    spec: SimSpec,
}

/// Builds the five production architecture patterns derived from `base`.
fn build_patterns(base: SimSpec) -> Vec<PatternSpec> {
    let tuned = base.pool.pool_size;
    let oversized = (tuned * 4).max(64);
    let slow_cap = tuned
        .div_euclid(5)
        .max(2)
        .min(tuned.saturating_sub(2).max(1));
    let fast_cap = tuned.saturating_sub(slow_cap);

    let mut p0 = base;
    p0.pool.pool_size = oversized;
    p0.pool.app_workers = p0.pool.app_workers.max(oversized + 32);
    p0.pool.pooling = PoolingMode::Session;
    p0.pool.bulkhead = BulkheadMode::Shared;
    p0.hardware.statement_timeout_ms = None;

    let mut p1 = p0;
    p1.pool.pool_size = tuned;

    let mut p2 = p1;
    p2.pool.pooling = PoolingMode::Transaction;

    let mut p3 = p2;
    p3.pool.bulkhead = BulkheadMode::Split { slow_cap };

    let mut p4 = p3;
    p4.hardware.statement_timeout_ms = Some(300);

    vec![
        PatternSpec {
            label: format!("1. Oversized (P={oversized}, Session)"),
            short_label: format!("Oversized\\n(P={oversized}, Sess)"),
            spec: p0,
        },
        PatternSpec {
            label: format!("2. Tuned Pool (P={tuned}, Session)"),
            short_label: format!("Tuned\\n(P={tuned}, Sess)"),
            spec: p1,
        },
        PatternSpec {
            label: format!("3. Transaction Pool (P={tuned}, Txn)"),
            short_label: format!("Transaction\\n(P={tuned}, Txn)"),
            spec: p2,
        },
        PatternSpec {
            label: format!("4. Bulkhead ({fast_cap}+{slow_cap}, Txn)"),
            short_label: format!("Bulkhead\\n({fast_cap}+{slow_cap}, Txn)"),
            spec: p3,
        },
        PatternSpec {
            label: format!("5. Bulkhead+StmtTO ({fast_cap}+{slow_cap}, 300ms)"),
            short_label: "Bulkhead+\\nStmtTO 300ms".to_owned(),
            spec: p4,
        },
    ]
}

/// View 2: Paired CRN comparison of five production pooling architectures.
pub(crate) fn run_patterns(cfg: &CliConfig) -> Result<(), DynError> {
    let seed = MasterSeed::from_u64(cfg.seed);
    let patterns = build_patterns(cfg.spec);
    let schedule = FixedReplications::new(cfg.reps, 2);
    let mut trials = Vec::new();
    for (idx, pat) in patterns.iter().enumerate() {
        trials.extend(schedule.trials(CandidateId::new(idx as u64), pat.spec));
    }
    let records = run_batch(seed, trials)?;
    let reps = cfg.reps as usize;

    println!(
        "\n=== VIEW 2: PAIRED CRN ARCHITECTURE COMPARISON (reps={}) ===",
        cfg.reps
    );
    println!(
        "{:<38} | {:>12} | {:>6} | {:>15} | {:>7} | {:>7} | {:>7}",
        "Architecture", "Goodput/s", "Succ%", "Fast P99ms", "All P99", "CVaR95", "StmtTO%"
    );
    println!("{}", "-".repeat(110));

    let mut tsv = String::from(
        "idx\tlabel\ttput\ttput_ci\tsucc_pct\tsucc_ci\tfast_p99\tfast_p99_ci\tall_p99\tcvar95\twait_p95\texec_p95\tstmt_to_pct\n",
    );
    for (idx, pat) in patterns.iter().enumerate() {
        let slice = &records[idx * reps..(idx + 1) * reps];
        let tput = reduce_metric(slice, |o| o.throughput_rps)?;
        let succ = reduce_metric(slice, |o| o.success_rate * 100.0)?;
        let fast = reduce_metric(slice, |o| o.fast_p99_ms)?;
        let all_p99 = reduce_metric(slice, |o| o.p99_ms)?;
        let cvar95 = reduce_metric(slice, |o| o.cvar95_ms)?;
        let wait_p95 = reduce_metric(slice, |o| o.pool_wait_p95_ms)?;
        let exec_p95 = reduce_metric(slice, |o| o.db_exec_p95_ms)?;
        let stmt_to = reduce_metric(slice, |o| o.stmt_timeout_rate * 100.0)?;
        println!(
            "{:<38} | {:>5.1}+/-{:<4.1} | {:>5.1}% | {:>6.1}+/-{:<5.1}  | {:>7.1} | {:>7.1} | {:>6.2}%",
            pat.label,
            tput.mean(),
            ci95_half_width(&tput),
            succ.mean(),
            fast.mean(),
            ci95_half_width(&fast),
            all_p99.mean(),
            cvar95.mean(),
            stmt_to.mean()
        );
        let _ = writeln!(
            tsv,
            "{}\t\"{}\"\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
            idx,
            pat.short_label,
            tput.mean(),
            ci95_half_width(&tput),
            succ.mean(),
            ci95_half_width(&succ),
            fast.mean(),
            ci95_half_width(&fast),
            all_p99.mean(),
            cvar95.mean(),
            wait_p95.mean(),
            exec_p95.mean(),
            stmt_to.mean()
        );
    }

    print_paired_upgrades(&records, &patterns)?;
    maybe_write_tsv(cfg.export_dir.as_deref(), "patterns.tsv", &tsv)?;
    Ok(())
}

/// Prints paired CRN Student-t CIs and variance-reduction ratios for incremental upgrades.
fn print_paired_upgrades(
    records: &[TrialRecord<PoolOutcome, SimulationError>],
    patterns: &[PatternSpec],
) -> Result<(), DynError> {
    println!("\nPaired CRN incremental deltas (challenger - baseline):");
    let pairs: [(usize, usize, &str); 4] = [
        (0, 1, "Oversized -> Tuned Pool"),
        (1, 2, "Session -> Transaction "),
        (2, 3, "Shared -> Bulkhead     "),
        (3, 4, "Bulkhead -> +StmtTO    "),
    ];
    for (base_i, chal_i, name) in pairs {
        let fast_cmp = compare_paired_candidates(
            records,
            CandidateId::new(base_i as u64),
            CandidateId::new(chal_i as u64),
            |o| o.fast_p99_ms,
            FailurePolicy::RejectCandidate,
            NonFinitePolicy::RejectCandidate,
        )?;
        let cvar_cmp = compare_paired_candidates(
            records,
            CandidateId::new(base_i as u64),
            CandidateId::new(chal_i as u64),
            |o| o.cvar95_ms,
            FailurePolicy::RejectCandidate,
            NonFinitePolicy::RejectCandidate,
        )?;
        let tput_cmp = compare_paired_candidates(
            records,
            CandidateId::new(base_i as u64),
            CandidateId::new(chal_i as u64),
            |o| o.throughput_rps,
            FailurePolicy::RejectCandidate,
            NonFinitePolicy::RejectCandidate,
        )?;
        let (f_lo, f_hi) = fast_cmp.ci_95().unwrap_or((0.0, 0.0));
        let (c_lo, c_hi) = cvar_cmp.ci_95().unwrap_or((0.0, 0.0));
        println!(
            "  {name}: Fast-P99 {:+7.1}ms [{:+.1}, {:+.1}] | CVaR95 {:+7.1}ms [{:+.1}, {:+.1}] (VRR {:>4.1}x) | Goodput {:+5.1}/s",
            fast_cmp.difference().mean(),
            f_lo,
            f_hi,
            cvar_cmp.difference().mean(),
            c_lo,
            c_hi,
            cvar_cmp.variance_reduction_ratio().unwrap_or(1.0),
            tput_cmp.difference().mean(),
        );
    }
    let _ = patterns.len();
    Ok(())
}

/// Builds the nested-pool deadlock evaluation spec for `(pool_size, nested_frac)`.
fn deadlock_spec(base: SimSpec, pool_size: usize, nested_frac: f64) -> SimSpec {
    let mut spec = base;
    spec.pool.pool_size = pool_size;
    spec.pool.app_workers = spec.pool.app_workers.max(pool_size + 48);
    spec.pool.pooling = PoolingMode::Transaction;
    spec.pool.bulkhead = BulkheadMode::Shared;
    spec.pool.acquire_timeout_ms = 5_000;
    spec.pool.halt_on_deadlock = true;
    spec.hardware.statement_timeout_ms = None;
    spec.workload.nested_frac = nested_frac;
    spec.workload.rpc_hold_frac = 0.0;
    spec.workload.slow_frac = 0.02;
    spec
}

/// View 3: Sweeps `(pool_size, nested_frac)` to map the nested-pool deadlock
/// phase boundary with Wilson 95% intervals, then replays one deadlocked trial
/// through `forensic_replay_trial_with_payloads`.
pub(crate) fn run_deadlock(cfg: &CliConfig) -> Result<(), DynError> {
    let seed = MasterSeed::from_u64(cfg.seed);
    let schedule = FixedReplications::new(cfg.reps, 3);
    let mut trials = Vec::new();
    let mut specs = Vec::new();
    for &size in &DEADLOCK_POOL_SIZES {
        for &nf in &DEADLOCK_NESTED_FRACS {
            let cid = CandidateId::new(specs.len() as u64);
            let spec = deadlock_spec(cfg.spec, size, nf);
            specs.push((size, nf, spec));
            trials.extend(schedule.trials(cid, spec));
        }
    }
    let records = run_batch(seed, trials)?;
    report_deadlock(cfg, &specs, &records)
}

/// Formats and exports the nested-pool deadlock phase table and triggers forensic replay.
fn report_deadlock(
    cfg: &CliConfig,
    specs: &[(usize, f64, SimSpec)],
    records: &[TrialRecord<PoolOutcome, SimulationError>],
) -> Result<(), DynError> {
    let reps = cfg.reps as usize;
    println!(
        "\n=== VIEW 3: NESTED-POOL DEADLOCK PHASE BOUNDARY (reps={}) ===",
        cfg.reps
    );
    println!(
        "{:>4} | {:>22} | {:>22} | {:>22} | {:>10}",
        "Pool",
        "nested=5% (95% Wilson)",
        "nested=15% (95% Wilson)",
        "nested=30% (95% Wilson)",
        "TTD@30%"
    );
    println!("{}", "-".repeat(92));

    let mut tsv = String::from(
        "pool_size\tp05_pct\tp05_lo\tp05_hi\tp15_pct\tp15_lo\tp15_hi\tp30_pct\tp30_lo\tp30_hi\tp30_ttd_ms\n",
    );
    let mut forensic_candidate: Option<(SimSpec, TrialRecord<PoolOutcome, SimulationError>)> = None;

    for (row_idx, &size) in DEADLOCK_POOL_SIZES.iter().enumerate() {
        let mut cells = Vec::with_capacity(3);
        let mut ttd30 = f64::NAN;
        for col_idx in 0..3 {
            let cand_idx = row_idx * 3 + col_idx;
            let slice = &records[cand_idx * reps..(cand_idx + 1) * reps];
            let summary = BernoulliReducer::from_results(
                slice
                    .iter()
                    .map(|r| r.result.as_ref().is_ok_and(|o| o.deadlocked)),
            )?;
            cells.push(summary);
            if col_idx == 2 {
                let ticks: Vec<f64> = slice
                    .iter()
                    .filter_map(|r| r.result.as_ref().ok()?.deadlock_tick.map(|t| t as f64))
                    .collect();
                if !ticks.is_empty() {
                    ttd30 = ticks.iter().sum::<f64>() / ticks.len() as f64;
                }
            }
            if forensic_candidate.is_none() && size >= 8 && col_idx >= 1 {
                if let Some(hit) = slice.iter().find(|r| {
                    r.result
                        .as_ref()
                        .is_ok_and(|o| o.deadlock_tick.is_some_and(|t| t >= 120))
                }) {
                    forensic_candidate = Some((specs[cand_idx].2, hit.clone()));
                }
            }
        }
        let ttd_str = if ttd30.is_finite() {
            format!("{ttd30:>7.0} ms")
        } else {
            "       n/a".to_owned()
        };
        println!(
            "{:>4} | {:>5.1}% [{:>4.1}%, {:>5.1}%] | {:>5.1}% [{:>4.1}%, {:>5.1}%] | {:>5.1}% [{:>4.1}%, {:>5.1}%] | {ttd_str}",
            size,
            cells[0].rate() * 100.0,
            cells[0].wilson_lower() * 100.0,
            cells[0].wilson_upper() * 100.0,
            cells[1].rate() * 100.0,
            cells[1].wilson_lower() * 100.0,
            cells[1].wilson_upper() * 100.0,
            cells[2].rate() * 100.0,
            cells[2].wilson_lower() * 100.0,
            cells[2].wilson_upper() * 100.0,
        );
        let ttd_val = if ttd30.is_finite() { ttd30 } else { -1.0 };
        let _ = writeln!(
            tsv,
            "{size}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.1}",
            cells[0].rate() * 100.0,
            cells[0].wilson_lower() * 100.0,
            cells[0].wilson_upper() * 100.0,
            cells[1].rate() * 100.0,
            cells[1].wilson_lower() * 100.0,
            cells[1].wilson_upper() * 100.0,
            cells[2].rate() * 100.0,
            cells[2].wilson_lower() * 100.0,
            cells[2].wilson_upper() * 100.0,
            ttd_val
        );
    }

    maybe_write_tsv(cfg.export_dir.as_deref(), "deadlock.tsv", &tsv)?;
    if let Some((spec, rec)) = forensic_candidate {
        run_forensic_deadlock_replay(spec, &rec)?;
    }
    Ok(())
}

/// Runs a two-pass windowed forensic replay of one deadlocked replication and
/// optionally writes Perfetto Chrome Trace JSON to `SIMUL_PERFETTO_OUT`.
fn run_forensic_deadlock_replay(
    spec: SimSpec,
    record: &TrialRecord<PoolOutcome, SimulationError>,
) -> Result<(), DynError> {
    let outcome = record
        .result
        .as_ref()
        .map_err(|err| format!("forensic trial failed: {err}"))?;
    let Some(dl_tick) = outcome.deadlock_tick else {
        return Ok(());
    };
    let start = dl_tick.saturating_sub(80);
    let report = forensic_replay_trial_with_payloads(
        record.replay,
        ReplayWindow::new(start, Some(dl_tick + 1)),
        |ctx| build_simulation(&spec, ctx),
    )?;
    let mut sent = 0_usize;
    let mut cancelled = 0_usize;
    let mut halted = 0_usize;
    for ev in report.trace.events() {
        match ev.kind {
            TraceEventKind::MessageSent { .. } => sent += 1,
            TraceEventKind::TimerCancelled { .. } => cancelled += 1,
            TraceEventKind::HaltInterrupt { .. } => halted += 1,
            TraceEventKind::MessageDelivered { .. }
            | TraceEventKind::MessageConsumed { .. }
            | TraceEventKind::AgentSlept { .. }
            | TraceEventKind::AgentWoke => {}
        }
    }
    let perfetto = report.trace.to_perfetto_json();
    println!(
        "\nTwo-pass forensic replay (pool_size={}, nested={:.0}%, deadlock @ t={}ms):",
        spec.pool.pool_size,
        spec.workload.nested_frac * 100.0,
        dl_tick
    );
    println!(
        "  Window {}..{}ms -> {} trace events ({} sent, {} timers cancelled, {} halt), {} bytes Perfetto JSON",
        report.trace_started_at,
        report.ended_at,
        report.trace.len(),
        sent,
        cancelled,
        halted,
        perfetto.len()
    );
    let smoking_gun: Vec<String> = report
        .trace
        .events()
        .iter()
        .filter_map(|ev| match &ev.kind {
            TraceEventKind::MessageSent {
                payload: Some(p), ..
            } if p.contains("is_nested: true") && p.starts_with("AcquireConn") => {
                Some(format!("t={}ms {p}", ev.time))
            }
            TraceEventKind::HaltInterrupt { reason } => {
                Some(format!("t={}ms HALT({reason})", ev.time))
            }
            TraceEventKind::MessageSent { .. }
            | TraceEventKind::TimerCancelled { .. }
            | TraceEventKind::MessageDelivered { .. }
            | TraceEventKind::MessageConsumed { .. }
            | TraceEventKind::AgentSlept { .. }
            | TraceEventKind::AgentWoke => None,
        })
        .collect();
    let tail_start = smoking_gun.len().saturating_sub(5);
    for line in &smoking_gun[tail_start..] {
        println!("    {line}");
    }
    if let Ok(path) = std::env::var("SIMUL_PERFETTO_OUT") {
        fs::write(&path, &perfetto)?;
        println!("  [wrote Perfetto trace to {path}]");
    }
    Ok(())
}

/// Decoded policy candidate searched by `MixedCrossEntropyOptimizer`.
#[derive(Clone, Copy, Debug)]
struct PolicyCandidate {
    /// Session vs. Transaction pooling.
    pooling: PoolingMode,
    /// Shared vs. Split bulkhead.
    bulkhead: BulkheadMode,
    /// Total connection pool size.
    pool_size: usize,
    /// Slow-lane cap (used when `bulkhead` is `Split`).
    slow_cap: usize,
    /// Pool acquire timeout (ms).
    acquire_timeout_ms: DiscreteTime,
}

/// Decodes a 5D mixed CEM sample into a [`PolicyCandidate`] and [`SimSpec`].
fn decode_policy(
    base: SimSpec,
    pt: &[simul::experiment::MixedValue],
) -> (PolicyCandidate, SimSpec) {
    let pooling = if pt[0].as_category() == Some(1) {
        PoolingMode::Transaction
    } else {
        PoolingMode::Session
    };
    let pool_size = usize::try_from(pt[2].as_i64().unwrap_or(14)).unwrap_or(14);
    let raw_slow = usize::try_from(pt[3].as_i64().unwrap_or(2)).unwrap_or(2);
    let slow_cap = raw_slow.clamp(1, pool_size.saturating_sub(2).max(1));
    let bulkhead = if pt[1].as_category() == Some(1) {
        BulkheadMode::Split { slow_cap }
    } else {
        BulkheadMode::Shared
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let acquire_timeout_ms = pt[4].as_f64().unwrap_or(150.0).round().clamp(40.0, 350.0) as u64;
    let cand = PolicyCandidate {
        pooling,
        bulkhead,
        pool_size,
        slow_cap,
        acquire_timeout_ms,
    };
    let mut spec = base;
    spec.pool.pooling = pooling;
    spec.pool.bulkhead = bulkhead;
    spec.pool.pool_size = pool_size;
    spec.pool.app_workers = spec.pool.app_workers.max(pool_size + 32);
    spec.pool.acquire_timeout_ms = acquire_timeout_ms;
    spec.hardware.statement_timeout_ms = None;
    (cand, spec)
}

/// Evaluated candidate row for Pareto plotting and reporting.
#[derive(Clone, Copy, Debug)]
struct OptEval {
    /// Candidate ID.
    cid: CandidateId,
    /// Decoded policy.
    policy: PolicyCandidate,
    /// Mean fast-OLTP P99 latency (ms).
    fast_p99: f64,
    /// Mean overall success rate in `[0, 1]`.
    succ: f64,
    /// Mean all-request P99 latency (ms).
    all_p99: f64,
    /// Mean throughput (req/s).
    tput: f64,
}

/// View 4: Jointly optimizes `(pooling, bulkhead, pool_size, slow_cap, acquire_timeout)`
/// using `MixedCrossEntropyOptimizer` and tracks the non-dominated `ParetoArchive`.
pub(crate) fn run_optimize(cfg: &CliConfig) -> Result<(), DynError> {
    let seed = MasterSeed::from_u64(cfg.seed);
    let ce_cfg = MixedCrossEntropyConfig::new(vec![
        MixedDimension::categorical(2),
        MixedDimension::categorical(2),
        MixedDimension::integer(6, 48),
        MixedDimension::integer(1, 6),
        MixedDimension::continuous(40.0, 350.0),
    ])
    .with_elite_fraction(0.25);
    let mut optimizer = MixedCrossEntropyOptimizer::new(&ce_cfg)?;
    let mut opt_rng = SeedDeriver::new(seed).proposal_seed(404).rng();
    let mut archive = ParetoArchive::new(vec![
        ObjectiveDirection::Minimize,
        ObjectiveDirection::Maximize,
    ])?;

    let eval_reps = cfg.reps.div_euclid(4).max(4);
    let eval_schedule = FixedReplications::new(eval_reps, 4);
    let gens = 5_usize;
    let batch_size = 16_usize;
    let mut next_cid = 1_u64;
    let mut all_evals = Vec::with_capacity(gens * batch_size);

    for _ in 0..gens {
        let points = optimizer.ask_batch(&mut opt_rng, batch_size);
        let mut decoded = Vec::with_capacity(points.len());
        let mut trials = Vec::with_capacity(points.len() * eval_reps as usize);
        for pt in points {
            let cid = CandidateId::new(next_cid);
            next_cid += 1;
            let (policy, spec) = decode_policy(cfg.spec, &pt);
            trials.extend(eval_schedule.trials(cid, spec));
            decoded.push((cid, pt, policy));
        }
        let records = run_batch(seed, trials)?;
        let mut feedback = Vec::with_capacity(decoded.len());
        for (idx, (cid, pt, policy)) in decoded.into_iter().enumerate() {
            let slice = &records[idx * eval_reps as usize..(idx + 1) * eval_reps as usize];
            let fast_p99 = reduce_metric(slice, |o| o.fast_p99_ms)?.mean();
            let succ = reduce_metric(slice, |o| o.success_rate)?.mean();
            let all_p99 = reduce_metric(slice, |o| o.p99_ms)?.mean();
            let tput = reduce_metric(slice, |o| o.throughput_rps)?.mean();
            archive.insert(cid, policy, vec![fast_p99, succ])?;
            all_evals.push(OptEval {
                cid,
                policy,
                fast_p99,
                succ,
                all_p99,
                tput,
            });
            // Scalarized guidance strongly rewards >=95% success while minimizing fast P99.
            let score = succ.mul_add(400.0, -fast_p99);
            feedback.push(MixedCrossEntropySample::new(pt, score));
        }
        optimizer.tell(&mut feedback)?;
    }

    report_pareto(cfg, &all_evals, &archive, eval_reps)
}

/// Prints the discovered Pareto frontier and exports `pareto_all.tsv` + `pareto_frontier.tsv`.
fn report_pareto(
    cfg: &CliConfig,
    all_evals: &[OptEval],
    archive: &ParetoArchive<PolicyCandidate>,
    eval_reps: u32,
) -> Result<(), DynError> {
    let hv = archive.hypervolume_2d([400.0, 0.0])?;
    let mut frontier_rows: Vec<&OptEval> = archive
        .frontier()
        .iter()
        .filter_map(|e| all_evals.iter().find(|r| r.cid == e.candidate))
        .collect();
    frontier_rows.sort_by(|a, b| a.fast_p99.total_cmp(&b.fast_p99));

    println!(
        "\n=== VIEW 4: MIXED-CEM OPTIMIZATION & PARETO FRONTIER (evals={}, reps/pt={eval_reps}) ===",
        all_evals.len()
    );
    println!(
        "Non-dominated frontier: {} policies | 2D hypervolume ref(400ms, 0%) = {:.1}",
        frontier_rows.len(),
        hv
    );
    println!(
        "{:>5} | {:<11} | {:<10} | {:>4} | {:>7} | {:>9} | {:>7} | {:>8}",
        "CID", "Pooling", "Bulkhead", "Pool", "AcqTOms", "FastP99ms", "Succ%", "Goodput"
    );
    println!("{}", "-".repeat(82));
    for row in &frontier_rows {
        let p = row.policy;
        println!(
            "{:>5} | {:<11} | {:<10} | {:>4} | {:>7} | {:>9.1} | {:>6.2}% | {:>8.1}",
            row.cid.value(),
            p.pooling.label(),
            p.bulkhead.label(),
            p.pool_size,
            p.acquire_timeout_ms,
            row.fast_p99,
            row.succ * 100.0,
            row.tput
        );
    }

    let mut all_tsv = String::from(
        "cid\tpooling\tbulkhead\tpool_size\tslow_cap\tacquire_to\tfast_p99\tsucc_pct\tall_p99\ttput\tfrontier\n",
    );
    let mut front_tsv = all_tsv.clone();
    for row in all_evals {
        let is_front = frontier_rows.iter().any(|f| f.cid == row.cid);
        let line = format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\n",
            row.cid.value(),
            row.policy.pooling.label(),
            row.policy.bulkhead.label(),
            row.policy.pool_size,
            row.policy.slow_cap,
            row.policy.acquire_timeout_ms,
            row.fast_p99,
            row.succ * 100.0,
            row.all_p99,
            row.tput,
            u8::from(is_front)
        );
        all_tsv.push_str(&line);
    }
    for row in &frontier_rows {
        let _ = writeln!(
            front_tsv,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t1",
            row.cid.value(),
            row.policy.pooling.label(),
            row.policy.bulkhead.label(),
            row.policy.pool_size,
            row.policy.slow_cap,
            row.policy.acquire_timeout_ms,
            row.fast_p99,
            row.succ * 100.0,
            row.all_p99,
            row.tput
        );
    }
    maybe_write_tsv(cfg.export_dir.as_deref(), "pareto_all.tsv", &all_tsv)?;
    maybe_write_tsv(cfg.export_dir.as_deref(), "pareto_frontier.tsv", &front_tsv)?;
    Ok(())
}

/// Computes average fractional ranks (1-based) for Spearman rank correlation.
fn average_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0_usize;
    while i < n {
        let mut j = i + 1;
        while j < n && (values[order[j]] - values[order[i]]).abs() <= 1e-12 {
            j += 1;
        }
        let avg_rank = 0.5 * ((i + 1) + j) as f64;
        for &idx in &order[i..j] {
            ranks[idx] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Computes Spearman rank correlation coefficient in `[-1, 1]`.
fn spearman_rho(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return 0.0;
    }
    let rx = average_ranks(xs);
    let ry = average_ranks(ys);
    let n = rx.len() as f64;
    let mx = rx.iter().sum::<f64>() / n;
    let my = ry.iter().sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut dx2 = 0.0_f64;
    let mut dy2 = 0.0_f64;
    for (a, b) in rx.iter().zip(ry.iter()) {
        let da = a - mx;
        let db = b - my;
        num = da.mul_add(db, num);
        dx2 = da.mul_add(da, dx2);
        dy2 = db.mul_add(db, dy2);
    }
    let denom = (dx2 * dy2).sqrt();
    if denom <= 0.0 { 0.0 } else { num / denom }
}

/// Computes the 4-quartile correlation ratio `eta` in `[0, 1]` on a dyadically
/// stratified Sobol design, capturing both monotonic and non-monotonic (U-shaped)
/// main effects (`sqrt(Var(E[Y | X_bin]) / Var(Y))`).
fn quartile_eta(xs: &[f64], ys: &[f64], lo: f64, hi: f64) -> f64 {
    if xs.len() != ys.len() || xs.len() < 4 || hi <= lo {
        return 0.0;
    }
    let n = ys.len() as f64;
    let grand_mean = ys.iter().sum::<f64>() / n;
    let total_ss: f64 = ys.iter().map(|&y| (y - grand_mean).powi(2)).sum();
    if total_ss <= 0.0 {
        return 0.0;
    }
    let mut sums = [0.0_f64; 4];
    let mut counts = [0_usize; 4];
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let u = ((x - lo) / (hi - lo)).clamp(0.0, 0.999_999);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bin = (u * 4.0).floor() as usize;
        let idx = bin.min(3);
        sums[idx] += y;
        counts[idx] += 1;
    }
    let between_ss: f64 = sums
        .iter()
        .zip(counts.iter())
        .filter(|(_, c)| **c > 0)
        .map(|(&s, &c)| {
            let mean = s / c as f64;
            (c as f64) * (mean - grand_mean).powi(2)
        })
        .sum();
    (between_ss / total_ss).clamp(0.0, 1.0).sqrt()
}

/// Prints and exports the 6D Sobol sensitivity rankings.
fn report_sensitivity(
    cfg: &CliConfig,
    points: &[Vec<f64>],
    bounds: &[(f64, f64); 6],
    y_p99: &[f64],
    y_fast: &[f64],
    y_to: &[f64],
    eval_reps: u32,
) -> Result<(), DynError> {
    let names = [
        ("pool_size", "Pool Size (P)"),
        ("slow_frac", "Slow Query Frac"),
        ("rpc_hold_frac", "RPC-Hold Frac"),
        ("acquire_timeout", "Acquire Timeout"),
        ("usl_kappa", "USL Coherency (k)"),
        ("rps", "Arrival Rate (RPS)"),
    ];
    let mut ranked: Vec<(&str, &str, f64, f64, f64, f64)> = (0..6)
        .map(|d| {
            let xs: Vec<f64> = points.iter().map(|p| p[d]).collect();
            (
                names[d].0,
                names[d].1,
                spearman_rho(&xs, y_p99),
                quartile_eta(&xs, y_p99, bounds[d].0, bounds[d].1),
                spearman_rho(&xs, y_fast),
                spearman_rho(&xs, y_to),
            )
        })
        .collect();
    ranked.sort_by(|a, b| b.3.total_cmp(&a.3));

    println!(
        "\n=== VIEW 5: 6D SOBOL GLOBAL SENSITIVITY SCREENING (pts={}, reps/pt={eval_reps}) ===",
        points.len()
    );
    println!(
        "{:>4} | {:<20} | {:>10} | {:>10} | {:>12} | {:>12}",
        "Rank", "Parameter", "eta(P99)", "rho(P99)", "rho(FastP99)", "rho(Timeout)"
    );
    println!("{}", "-".repeat(82));
    let mut tsv = String::from(
        "rank\tparam\tlabel\teta_p99\trho_p99\tabs_rho_p99\trho_fast_p99\trho_timeout\n",
    );
    for (i, &(key, label, r_p99, eta, r_fast, r_to)) in ranked.iter().enumerate() {
        println!(
            "{:>4} | {:<20} | {:>10.3} | {:>+10.3} | {:>+12.3} | {:>+12.3}",
            i + 1,
            label,
            eta,
            r_p99,
            r_fast,
            r_to
        );
        let _ = writeln!(
            tsv,
            "{}\t{key}\t\"{label}\"\t{eta:.4}\t{r_p99:.4}\t{:.4}\t{r_fast:.4}\t{r_to:.4}",
            i + 1,
            r_p99.abs()
        );
    }
    maybe_write_tsv(cfg.export_dir.as_deref(), "sensitivity.tsv", &tsv)
}

/// View 5: 6D Sobol space-filling global sensitivity screening using quartile
/// main-effect `eta` (captures U-curves) and Spearman `rho` (signed direction).
pub(crate) fn run_sensitivity(cfg: &CliConfig) -> Result<(), DynError> {
    let seed = MasterSeed::from_u64(cfg.seed);
    let bounds: [(f64, f64); 6] = [
        (6.0, 64.0),    // pool_size
        (0.00, 0.12),   // slow_frac
        (0.00, 0.40),   // rpc_hold_frac
        (40.0, 350.0),  // acquire_timeout_ms
        (0.05, 0.50),   // usl_kappa
        (250.0, 620.0), // rps
    ];
    let n_pts = 64_usize;
    let points = sample_sobol(n_pts, &bounds)?;
    let eval_reps = cfg.reps.div_euclid(4).max(4);
    let schedule = FixedReplications::new(eval_reps, 5);
    let mut trials = Vec::with_capacity(n_pts * eval_reps as usize);

    for (idx, pt) in points.iter().enumerate() {
        let mut spec = cfg.spec;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let pool_size = pt[0].round().clamp(6.0, 64.0) as usize;
        spec.pool.pool_size = pool_size;
        spec.pool.app_workers = spec.pool.app_workers.max(pool_size + 32);
        spec.pool.pooling = PoolingMode::Session;
        spec.pool.bulkhead = BulkheadMode::Shared;
        spec.workload.slow_frac = pt[1];
        spec.workload.rpc_hold_frac = pt[2];
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            spec.pool.acquire_timeout_ms = pt[3].round().clamp(40.0, 350.0) as u64;
        }
        spec.hardware.usl_kappa = pt[4];
        spec.workload.rps = pt[5];
        trials.extend(schedule.trials(CandidateId::new(idx as u64), spec));
    }

    let records = run_batch(seed, trials)?;
    let mut y_p99 = Vec::with_capacity(n_pts);
    let mut y_fast = Vec::with_capacity(n_pts);
    let mut y_to = Vec::with_capacity(n_pts);
    for idx in 0..n_pts {
        let slice = &records[idx * eval_reps as usize..(idx + 1) * eval_reps as usize];
        y_p99.push(reduce_metric(slice, |o| o.p99_ms)?.mean());
        y_fast.push(reduce_metric(slice, |o| o.fast_p99_ms)?.mean());
        y_to.push(reduce_metric(slice, |o| o.timeout_rate)?.mean());
    }
    report_sensitivity(cfg, &points, &bounds, &y_p99, &y_fast, &y_to, eval_reps)
}
