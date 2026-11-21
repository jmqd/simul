//! Database Connection Pool Sizing & Nested-Pool Deadlock Simulation.
//!
//! Models five real-world failure modes and thresholds of production database
//! connection pools using `simul`'s discrete-event engine, cancellable timers,
//! multi-message tick drain, paired CRN statistics, Wilson bounds, mixed-variable
//! CEM optimization, Pareto frontier archive, 6D Sobol space-filling screening,
//! and two-pass windowed forensic Perfetto replay:
//!
//! 1. **`HikariCP` / USL throughput knee**: why small pools (`~1.5–2× DB cores`)
//!    outperform oversized pools (`64–96`) that trigger Universal Scalability Law
//!    (USL) contention collapse.
//! 2. **Hold-across-RPC starvation**: why Session-mode pooling starves OLTP work
//!    during external RPCs while Transaction-mode pooling reclaims idle slots.
//! 3. **Slow-query HOL blocking & bulkhead isolation**: quarantining heavy
//!    analytics queries into a dedicated sub-pool plus DB `statement_timeout`.
//! 4. **Nested-pool deadlock phase cliff**: probabilistic lockup when requests
//!    hold connection #1 while acquiring connection #2 from the same pool, plus
//!    windowed forensic replay to Perfetto Chrome Trace JSON.
//! 5. **Automated sizing & global sensitivity**: joint mixed-variable CEM +
//!    Pareto frontier discovery and 6D Sobol Spearman sensitivity ranking.
//!
//! Run `cargo run --release --all-features --example connection_pool -- --help`
//! for all adjustable parameters.
#![allow(
    clippy::redundant_pub_crate,
    reason = "crate-wide deny(unreachable_pub) requires pub(crate) inside binary modules"
)]

mod analysis;
mod config;
mod model;

use config::{Scenario, parse_cli, print_help};

/// Entry point for the connection-pool simulation example.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_cli(std::env::args())?;
    if cfg.show_help {
        print_help();
        return Ok(());
    }

    println!("================================================================================");
    println!("  DATABASE CONNECTION POOL SIZING & NESTED-POOL DEADLOCK SIMULATION");
    println!("================================================================================");
    println!(
        "Seed={} | Reps={} | Horizon={}ms (warmup={}ms, burst={}..{}ms @ {:.2}x)",
        cfg.seed,
        cfg.reps,
        cfg.spec.workload.horizon_ms,
        cfg.spec.workload.warmup_ms,
        cfg.spec.workload.burst_start_ms,
        cfg.spec.workload.burst_end_ms,
        cfg.spec.workload.burst_mult
    );
    println!(
        "Workload: {:.0} req/s | fast={:.0}ms, slow={:.0}ms ({:.1}%), rpc={:.0}ms ({:.0}%)",
        cfg.spec.workload.rps,
        cfg.spec.workload.fast_ms,
        cfg.spec.workload.slow_ms,
        cfg.spec.workload.slow_frac * 100.0,
        cfg.spec.workload.rpc_ms,
        cfg.spec.workload.rpc_hold_frac * 100.0,
    );

    match cfg.scenario {
        Scenario::All => {
            analysis::run_sweep(&cfg)?;
            analysis::run_patterns(&cfg)?;
            analysis::run_deadlock(&cfg)?;
            analysis::run_optimize(&cfg)?;
            analysis::run_sensitivity(&cfg)?;
        }
        Scenario::Sweep => analysis::run_sweep(&cfg)?,
        Scenario::Patterns => analysis::run_patterns(&cfg)?,
        Scenario::Deadlock => analysis::run_deadlock(&cfg)?,
        Scenario::Optimize => analysis::run_optimize(&cfg)?,
        Scenario::Sensitivity => analysis::run_sensitivity(&cfg)?,
    }
    Ok(())
}
