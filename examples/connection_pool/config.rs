//! Configuration types, physical defaults, and CLI parsing for the
//! database connection-pool simulation.

use std::fmt;
use std::path::PathBuf;

use simul::DiscreteTime;

/// Default simulation horizon (10 seconds at 1 tick = 1 ms).
pub(crate) const DEFAULT_HORIZON_MS: DiscreteTime = 10_000;
/// Default warmup discarded before measuring steady-state and burst metrics.
pub(crate) const DEFAULT_WARMUP_MS: DiscreteTime = 2_000;
/// Default traffic burst start tick (ms).
pub(crate) const DEFAULT_BURST_START_MS: DiscreteTime = 4_000;
/// Default traffic burst end tick (ms).
pub(crate) const DEFAULT_BURST_END_MS: DiscreteTime = 6_000;

/// Connection ownership lifetime relative to external RPC calls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PoolingMode {
    /// Holds the database connection across external RPC calls (classic ORM anti-pattern).
    Session,
    /// Releases the connection before external RPC calls and re-acquires afterward.
    Transaction,
}

impl PoolingMode {
    /// Short human-readable label for tables and TSV exports.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::Transaction => "transaction",
        }
    }
}

/// Isolation policy between fast OLTP and slow analytical queries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BulkheadMode {
    /// All query classes share one FIFO connection pool.
    Shared,
    /// Slow analytical queries are quarantined into a dedicated `slow_cap` sub-pool.
    Split {
        /// Maximum connections reserved for slow analytical queries.
        slow_cap: usize,
    },
}

impl BulkheadMode {
    /// Short human-readable label for tables and TSV exports.
    pub(crate) fn label(self) -> String {
        match self {
            Self::Shared => "shared".to_owned(),
            Self::Split { slow_cap } => format!("split({slow_cap})"),
        }
    }
}

/// Which experimental view(s) to run from the CLI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Scenario {
    /// Run all five views in sequence.
    All,
    /// Pool-size sweep showing the HikariCP/USL throughput knee and latency U-curve.
    Sweep,
    /// Paired CRN comparison of five production pooling architectures.
    Patterns,
    /// Nested-pool deadlock phase boundary and windowed forensic Perfetto replay.
    Deadlock,
    /// Mixed-variable CEM optimization and fast-P99 vs. success-rate Pareto frontier.
    Optimize,
    /// 6D Sobol global sensitivity screening (Spearman rank correlations).
    Sensitivity,
}

/// Database server hardware and Universal Scalability Law (USL) parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DbHardware {
    /// Physical CPU cores available on the primary database node.
    pub(crate) db_cores: usize,
    /// USL linear contention/scheduling penalty per active query above `db_cores`.
    pub(crate) usl_sigma: f64,
    /// USL quadratic coherency/lock-table penalty scaling with `(n / db_cores)^2`.
    pub(crate) usl_kappa: f64,
    /// Optional DB-side `statement_timeout` (ms) that aborts runaway queries.
    pub(crate) statement_timeout_ms: Option<DiscreteTime>,
}

impl DbHardware {
    /// Computes the Gunther USL slowdown factor `f(n)` at `active_queries` concurrent queries.
    ///
    /// - For `n <= db_cores`: zero CPU oversubscription penalty, mild lock coherency.
    /// - For `n > db_cores`: linear CPU contention plus quadratic cache/lock coherency.
    pub(crate) fn usl_slowdown(self, active_queries: usize) -> f64 {
        let cores = self.db_cores.max(1) as f64;
        let n = active_queries.max(1) as f64;
        let excess = (n - cores).max(0.0) / cores;
        let coherency = (n * (n - 1.0)) / (cores * cores);
        self.usl_kappa
            .mul_add(coherency, self.usl_sigma.mul_add(excess, 1.0))
    }

    /// Returns the analytical USL throughput-maximizing active query count `n*`.
    ///
    /// Minimizing `f(n) / n` for `n >= db_cores` yields `n* = c * sqrt((1 - sigma) / kappa)`
    /// whenever `sigma < 1` and `kappa > 0`.
    pub(crate) fn theoretical_usl_knee(self) -> f64 {
        let cores = self.db_cores.max(1) as f64;
        if self.usl_kappa <= 0.0 || self.usl_sigma >= 1.0 {
            return cores;
        }
        let ratio = ((1.0 - self.usl_sigma) / self.usl_kappa).sqrt();
        (cores * ratio).max(cores)
    }
}

/// Traffic arrival, burst, and query-mix parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WorkloadConfig {
    /// Total simulated horizon in milliseconds (1 tick = 1 ms).
    pub(crate) horizon_ms: DiscreteTime,
    /// Warmup period in milliseconds excluded from metrics.
    pub(crate) warmup_ms: DiscreteTime,
    /// Steady-state arrival rate in requests per second.
    pub(crate) rps: f64,
    /// Arrival-rate multiplier during the transient traffic burst window.
    pub(crate) burst_mult: f64,
    /// Burst window start tick (inclusive, ms).
    pub(crate) burst_start_ms: DiscreteTime,
    /// Burst window end tick (exclusive, ms).
    pub(crate) burst_end_ms: DiscreteTime,
    /// Base uncontended DB execution time for fast OLTP queries (ms).
    pub(crate) fast_ms: f64,
    /// Base uncontended DB execution time for slow analytical queries (ms).
    pub(crate) slow_ms: f64,
    /// Fraction of incoming requests that run a heavy analytical query.
    pub(crate) slow_frac: f64,
    /// Fraction of incoming requests that call an external RPC between two DB queries.
    pub(crate) rpc_hold_frac: f64,
    /// Mean external RPC latency (ms).
    pub(crate) rpc_ms: f64,
    /// Fraction of incoming requests that acquire a nested 2nd DB connection.
    pub(crate) nested_frac: f64,
}

/// Application thread pool and database connection pool configuration.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PoolConfig {
    /// Maximum concurrent application worker threads.
    pub(crate) app_workers: usize,
    /// Maximum open database connections in the pool.
    pub(crate) pool_size: usize,
    /// Maximum wait time in the pool queue before failing with timeout (ms).
    pub(crate) acquire_timeout_ms: DiscreteTime,
    /// Whether connections are held across external RPCs or released first.
    pub(crate) pooling: PoolingMode,
    /// Shared pool vs. fast/slow bulkhead isolation.
    pub(crate) bulkhead: BulkheadMode,
    /// If `true`, halt immediately via interrupt upon detecting exact nested-pool deadlock.
    pub(crate) halt_on_deadlock: bool,
}

/// Complete specification of one simulation run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SimSpec {
    /// Database hardware and USL contention parameters.
    pub(crate) hardware: DbHardware,
    /// Traffic arrival and query-class mix parameters.
    pub(crate) workload: WorkloadConfig,
    /// Application worker and connection pool settings.
    pub(crate) pool: PoolConfig,
}

impl Default for SimSpec {
    fn default() -> Self {
        Self {
            hardware: DbHardware {
                db_cores: 8,
                usl_sigma: 0.25,
                usl_kappa: 0.25,
                statement_timeout_ms: None,
            },
            workload: WorkloadConfig {
                horizon_ms: DEFAULT_HORIZON_MS,
                warmup_ms: DEFAULT_WARMUP_MS,
                rps: 440.0,
                burst_mult: 1.55,
                burst_start_ms: DEFAULT_BURST_START_MS,
                burst_end_ms: DEFAULT_BURST_END_MS,
                fast_ms: 6.0,
                slow_ms: 140.0,
                slow_frac: 0.04,
                rpc_hold_frac: 0.20,
                rpc_ms: 55.0,
                nested_frac: 0.0,
            },
            pool: PoolConfig {
                app_workers: 96,
                pool_size: 14,
                acquire_timeout_ms: 150,
                pooling: PoolingMode::Session,
                bulkhead: BulkheadMode::Shared,
                halt_on_deadlock: false,
            },
        }
    }
}

/// Top-level CLI configuration parsed from command-line arguments.
#[derive(Clone, Debug)]
pub(crate) struct CliConfig {
    /// Base simulation specification modified by CLI flags.
    pub(crate) spec: SimSpec,
    /// Selected experimental view(s).
    pub(crate) scenario: Scenario,
    /// Master RNG seed for reproducible experiments.
    pub(crate) seed: u64,
    /// CRN replications per candidate in sweeps and pattern comparisons.
    pub(crate) reps: u32,
    /// Optional directory where TSV data files for gnuplot are written.
    pub(crate) export_dir: Option<PathBuf>,
    /// Whether `--help` was requested.
    pub(crate) show_help: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            spec: SimSpec::default(),
            scenario: Scenario::All,
            seed: 2026_0505,
            reps: 24,
            export_dir: None,
            show_help: false,
        }
    }
}

/// Error returned when parsing CLI arguments fails.
#[derive(Clone, Debug)]
pub(crate) struct CliError(pub(crate) String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CliError {}

/// Parses `std::env::args()` into a validated [`CliConfig`].
pub(crate) fn parse_cli(args: impl IntoIterator<Item = String>) -> Result<CliConfig, CliError> {
    let mut cfg = CliConfig::default();
    let mut iter = args.into_iter().skip(1);
    while let Some(flag) = iter.next() {
        apply_flag(&mut cfg, &flag, &mut iter)?;
    }
    validate_config(&cfg)?;
    Ok(cfg)
}

/// Applies one CLI flag (and its value if needed) to `cfg`.
fn apply_flag(
    cfg: &mut CliConfig,
    flag: &str,
    iter: &mut impl Iterator<Item = String>,
) -> Result<(), CliError> {
    match flag {
        "-h" | "--help" => cfg.show_help = true,
        "--quick" => {
            cfg.reps = 8;
            cfg.spec.workload.horizon_ms = 6_000;
            cfg.spec.workload.warmup_ms = 1_000;
            cfg.spec.workload.burst_start_ms = 2_500;
            cfg.spec.workload.burst_end_ms = 3_800;
        }
        "--scenario" => cfg.scenario = parse_scenario(&next_arg(flag, iter)?)?,
        "--reps" => cfg.reps = parse_val(flag, &next_arg(flag, iter)?)?,
        "--seed" => cfg.seed = parse_val(flag, &next_arg(flag, iter)?)?,
        "--export-data" => cfg.export_dir = Some(PathBuf::from(next_arg(flag, iter)?)),
        "--db-cores" => cfg.spec.hardware.db_cores = parse_val(flag, &next_arg(flag, iter)?)?,
        "--usl-sigma" => cfg.spec.hardware.usl_sigma = parse_val(flag, &next_arg(flag, iter)?)?,
        "--usl-kappa" => cfg.spec.hardware.usl_kappa = parse_val(flag, &next_arg(flag, iter)?)?,
        "--statement-timeout-ms" => {
            let v: u64 = parse_val(flag, &next_arg(flag, iter)?)?;
            cfg.spec.hardware.statement_timeout_ms = (v > 0).then_some(v);
        }
        "--rps" => cfg.spec.workload.rps = parse_val(flag, &next_arg(flag, iter)?)?,
        "--burst-mult" => cfg.spec.workload.burst_mult = parse_val(flag, &next_arg(flag, iter)?)?,
        "--fast-ms" => cfg.spec.workload.fast_ms = parse_val(flag, &next_arg(flag, iter)?)?,
        "--slow-ms" => cfg.spec.workload.slow_ms = parse_val(flag, &next_arg(flag, iter)?)?,
        "--slow-frac" => cfg.spec.workload.slow_frac = parse_val(flag, &next_arg(flag, iter)?)?,
        "--rpc-hold-frac" => {
            cfg.spec.workload.rpc_hold_frac = parse_val(flag, &next_arg(flag, iter)?)?;
        }
        "--rpc-ms" => cfg.spec.workload.rpc_ms = parse_val(flag, &next_arg(flag, iter)?)?,
        "--nested-frac" => cfg.spec.workload.nested_frac = parse_val(flag, &next_arg(flag, iter)?)?,
        "--app-workers" => cfg.spec.pool.app_workers = parse_val(flag, &next_arg(flag, iter)?)?,
        "--pool-size" => cfg.spec.pool.pool_size = parse_val(flag, &next_arg(flag, iter)?)?,
        "--acquire-timeout-ms" => {
            cfg.spec.pool.acquire_timeout_ms = parse_val(flag, &next_arg(flag, iter)?)?;
        }
        "--horizon-ms" => cfg.spec.workload.horizon_ms = parse_val(flag, &next_arg(flag, iter)?)?,
        "--warmup-ms" => cfg.spec.workload.warmup_ms = parse_val(flag, &next_arg(flag, iter)?)?,
        other => {
            return Err(CliError(format!(
                "unknown argument `{other}` (run with --help for usage)"
            )));
        }
    }
    Ok(())
}

/// Consumes the next CLI argument or returns a descriptive error.
fn next_arg(flag: &str, iter: &mut impl Iterator<Item = String>) -> Result<String, CliError> {
    iter.next()
        .ok_or_else(|| CliError(format!("missing value for `{flag}`")))
}

/// Parses a typed numeric value for `flag`.
fn parse_val<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, CliError>
where
    T::Err: fmt::Display,
{
    raw.parse::<T>()
        .map_err(|err| CliError(format!("invalid value `{raw}` for `{flag}`: {err}")))
}

/// Parses the `--scenario` enum string.
fn parse_scenario(raw: &str) -> Result<Scenario, CliError> {
    match raw {
        "all" => Ok(Scenario::All),
        "sweep" => Ok(Scenario::Sweep),
        "patterns" => Ok(Scenario::Patterns),
        "deadlock" => Ok(Scenario::Deadlock),
        "optimize" => Ok(Scenario::Optimize),
        "sensitivity" => Ok(Scenario::Sensitivity),
        other => Err(CliError(format!(
            "unknown scenario `{other}` (expected: all, sweep, patterns, deadlock, optimize, sensitivity)"
        ))),
    }
}

/// Validates physical and numeric constraints on `cfg`.
fn validate_config(cfg: &CliConfig) -> Result<(), CliError> {
    let w = &cfg.spec.workload;
    let h = &cfg.spec.hardware;
    let p = &cfg.spec.pool;
    if cfg.reps == 0 {
        return Err(CliError("--reps must be >= 1".to_owned()));
    }
    if h.db_cores == 0 {
        return Err(CliError("--db-cores must be >= 1".to_owned()));
    }
    if !h.usl_sigma.is_finite() || h.usl_sigma < 0.0 {
        return Err(CliError("--usl-sigma must be finite and >= 0".to_owned()));
    }
    if !h.usl_kappa.is_finite() || h.usl_kappa < 0.0 {
        return Err(CliError("--usl-kappa must be finite and >= 0".to_owned()));
    }
    if w.horizon_ms <= w.warmup_ms {
        return Err(CliError(
            "--horizon-ms must be strictly greater than --warmup-ms".to_owned(),
        ));
    }
    if !w.rps.is_finite() || w.rps <= 0.0 {
        return Err(CliError("--rps must be finite and > 0".to_owned()));
    }
    if !w.burst_mult.is_finite() || w.burst_mult < 1.0 {
        return Err(CliError(
            "--burst-mult must be finite and >= 1.0".to_owned(),
        ));
    }
    let total_frac = w.slow_frac + w.rpc_hold_frac + w.nested_frac;
    if w.slow_frac < 0.0
        || w.rpc_hold_frac < 0.0
        || w.nested_frac < 0.0
        || !total_frac.is_finite()
        || total_frac > 1.0
    {
        return Err(CliError(
            "slow_frac + rpc_hold_frac + nested_frac must be in [0, 1]".to_owned(),
        ));
    }
    if p.app_workers == 0 || p.pool_size == 0 || p.acquire_timeout_ms == 0 {
        return Err(CliError(
            "--app-workers, --pool-size, and --acquire-timeout-ms must be >= 1".to_owned(),
        ));
    }
    Ok(())
}

/// Prints CLI usage instructions.
pub(crate) fn print_help() {
    println!(
        "\
Database Connection Pool Sizing & Nested-Pool Deadlock Simulation

USAGE:
  cargo run --release --all-features --example connection_pool -- [OPTIONS]

SCENARIO & EXECUTION:
  --scenario <name>          all | sweep | patterns | deadlock | optimize | sensitivity (default: all)
  --reps <N>                 CRN replications per point (default: 24)
  --seed <u64>               Master RNG seed (default: 20260505)
  --quick                    Fast smoke-test settings (reps=8, horizon=6000ms)
  --export-data <DIR>        Write TSV datasets for gnuplot into DIR

DATABASE HARDWARE (USL MODEL):
  --db-cores <N>             Database CPU cores (default: 8)
  --usl-sigma <f64>          USL linear contention factor above db_cores (default: 0.25)
  --usl-kappa <f64>          USL quadratic lock/coherency factor (default: 0.25)
  --statement-timeout-ms <N> DB statement_timeout in ms, 0 disables (default: 0)

WORKLOAD & TRAFFIC BURST:
  --rps <f64>                Steady arrival rate in req/s (default: 440)
  --burst-mult <f64>         Burst multiplier during 4s..6s window (default: 1.55)
  --fast-ms <f64>            Base fast OLTP query latency in ms (default: 6)
  --slow-ms <f64>            Base slow analytical query latency in ms (default: 140)
  --slow-frac <f64>          Fraction of slow analytical queries (default: 0.04)
  --rpc-hold-frac <f64>      Fraction of requests calling an external RPC (default: 0.20)
  --rpc-ms <f64>             Mean external RPC latency in ms (default: 55)
  --nested-frac <f64>        Fraction of nested 2nd-connection acquisitions (default: 0.0)
  --horizon-ms <N>           Simulation horizon in ms (default: 10000)
  --warmup-ms <N>            Warmup excluded from metrics in ms (default: 2000)

POOL & APP SERVER:
  --app-workers <N>          Application worker threads (default: 96)
  --pool-size <N>            Baseline connection pool size (default: 14)
  --acquire-timeout-ms <N>   Pool acquire timeout in ms (default: 150)

ENVIRONMENT:
  SIMUL_PERFETTO_OUT=<path>  Write Perfetto Chrome Trace JSON from the deadlock forensic replay"
    );
}
