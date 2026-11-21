//! Retry storms and metastable failure showcase.
//!
//! Demonstrates how a brief transient latency blip can trap a service in a
//! self-sustaining retry storm long after the fault clears, and showcases
//! `simul`'s cancellable timers, multi-message tick drain, strict routing,
//! P99/CVaR quantiles, paired CRN confidence intervals, mixed-variable
//! cross-entropy optimization, Pareto frontier tracking, and windowed forensic
//! Perfetto/JSONL replay.

use std::collections::VecDeque;

use simul::agent::{Agent, AgentContext, AgentId, AgentInitializer, TimerId};
use simul::experiment::{
    CandidateId, FailurePolicy, FixedReplications, MasterSeed, MixedCrossEntropyConfig,
    MixedCrossEntropyOptimizer, MixedCrossEntropySample, MixedDimension, NonFinitePolicy,
    ObjectiveDirection, ParetoArchive, Probability, SeedDeriver, TrialContext, aggregate_quantiles,
    compare_paired_candidates, run_serial,
};
use simul::message::Message;
use simul::trace::{ReplayWindow, forensic_replay_trial_with_payloads};
use simul::{DiscreteTime, Halt, Simulation, SimulationError, UnknownDestinationPolicy};

/// Horizon of each simulated run (ticks).
const HORIZON: DiscreteTime = 500;
/// Tick at which the transient downstream fault begins.
const FAULT_START: DiscreteTime = 120;
/// Tick at which the transient downstream fault clears.
const FAULT_END: DiscreteTime = 155;
/// Tick after which recovery metrics are measured (well past `FAULT_END`).
const RECOVERY_WINDOW_START: DiscreteTime = 220;

/// Retry backoff strategy evaluated in the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BackoffKind {
    /// Immediate fixed retry after timeout (prone to synchronized retry storms).
    Fixed,
    /// Exponential backoff with full random jitter.
    ExponentialJitter,
}

/// Policy parameters controlling client retries and server admission.
#[derive(Clone, Copy, Debug)]
struct Policy {
    /// Backoff strategy on timeout or rejection.
    backoff: BackoffKind,
    /// Maximum retry attempts per logical request.
    max_retries: u8,
    /// Client timeout before retrying an unanswered request (ticks).
    timeout: DiscreteTime,
    /// Maximum pending requests held in the server queue before load shedding.
    queue_cap: usize,
}

/// Messages exchanged between the client workload, server, and fault injector.
#[derive(Clone, Copy, Debug)]
enum StormMessage {
    /// Client sends an attempt for `req_id` to the server.
    Request {
        /// Logical request identifier.
        req_id: u64,
    },
    /// Server completes a request and notifies the client.
    Response {
        /// Logical request identifier.
        req_id: u64,
        /// Whether the request succeeded (`false` means fast load-shed reject).
        ok: bool,
    },
    /// Self-scheduled server completion timer for the active service slot.
    SlotDone {
        /// Logical request identifier that finished processing.
        req_id: u64,
    },
    /// Self-scheduled client timeout for an in-flight request attempt.
    Timeout {
        /// Logical request identifier.
        req_id: u64,
        /// Attempt number guarded by this timeout.
        attempt: u8,
    },
    /// Self-scheduled client retry wakeup after backoff.
    RetryNow {
        /// Logical request identifier.
        req_id: u64,
        /// Attempt number to issue now.
        attempt: u8,
    },
    /// Fault injector toggles downstream degradation on or off.
    SetFault {
        /// Whether downstream service is degraded.
        active: bool,
    },
}

/// Tracks one active client request until completion or retry exhaustion.
#[derive(Clone, Debug)]
struct PendingRequest {
    /// Tick when the logical request first arrived.
    started_at: DiscreteTime,
    /// Current attempt number.
    attempt: u8,
    /// Active timeout timer handle so completion cancels it cleanly.
    timeout_timer: Option<TimerId>,
}

/// Workload generator + retry client pool.
#[derive(Clone, Debug)]
struct ClientPool {
    /// Retry/timeout policy under test.
    policy: Policy,
    /// Cached server identity.
    server: Option<AgentId>,
    /// Next logical request ID.
    next_req_id: u64,
    /// Active requests indexed by `req_id`.
    in_flight: Vec<(u64, PendingRequest)>,
    /// End-to-end latencies of requests completed after `RECOVERY_WINDOW_START`.
    recovery_latencies: Vec<f64>,
    /// Requests started after `RECOVERY_WINDOW_START`.
    recovery_started: u64,
    /// Requests succeeded after `RECOVERY_WINDOW_START`.
    recovery_succeeded: u64,
    /// Total retry attempts issued across the whole run.
    total_retries: u64,
}

impl ClientPool {
    /// Creates a client pool governed by `policy`.
    const fn new(policy: Policy) -> Self {
        Self {
            policy,
            server: None,
            next_req_id: 0,
            in_flight: Vec::new(),
            recovery_latencies: Vec::new(),
            recovery_started: 0,
            recovery_succeeded: 0,
            total_retries: 0,
        }
    }

    /// Resolves and caches the server identity.
    fn server_id(&mut self, ctx: &AgentContext<'_, StormMessage>) -> Option<AgentId> {
        if self.server.is_none() {
            self.server = ctx.agent_id("server");
        }
        self.server
    }

    /// Schedules a retry or abandons the request if the retry budget is spent.
    fn schedule_retry(
        &mut self,
        ctx: &mut AgentContext<'_, StormMessage>,
        req_id: u64,
        next_attempt: u8,
    ) {
        if next_attempt > self.policy.max_retries {
            self.in_flight.retain(|(id, _)| *id != req_id);
            return;
        }
        self.total_retries += 1;
        let delay = match self.policy.backoff {
            BackoffKind::Fixed => 1,
            BackoffKind::ExponentialJitter => {
                let cap = 2_u64.saturating_pow(u32::from(next_attempt)).min(40);
                let draw = ctx.rng().next_u64() % cap.max(1);
                1 + draw
            }
        };
        ctx.send_after(
            delay,
            ctx.id(),
            StormMessage::RetryNow {
                req_id,
                attempt: next_attempt,
            },
        );
    }

    /// Sends an attempt to the server and arms a cancellable timeout timer.
    fn send_attempt(&mut self, ctx: &mut AgentContext<'_, StormMessage>, req_id: u64, attempt: u8) {
        let Some(server) = self.server_id(ctx) else {
            return;
        };
        ctx.send_to(server, StormMessage::Request { req_id });
        let timer = ctx.send_after(
            self.policy.timeout,
            ctx.id(),
            StormMessage::Timeout { req_id, attempt },
        );
        if let Some((_, entry)) = self.in_flight.iter_mut().find(|(id, _)| *id == req_id) {
            entry.attempt = attempt;
            entry.timeout_timer = Some(timer);
        }
    }
}

impl Agent<StormMessage> for ClientPool {
    fn on_message(
        &mut self,
        ctx: &mut AgentContext<'_, StormMessage>,
        msg: &Message<StormMessage>,
    ) {
        match msg.payload {
            StormMessage::Response { req_id, ok } => {
                let Some(pos) = self.in_flight.iter().position(|(id, _)| *id == req_id) else {
                    return;
                };
                if ok {
                    let (_, pending) = self.in_flight.swap_remove(pos);
                    if let Some(timer) = pending.timeout_timer {
                        ctx.cancel_timer(timer);
                    }
                    if pending.started_at >= RECOVERY_WINDOW_START {
                        self.recovery_succeeded += 1;
                        self.recovery_latencies
                            .push((ctx.time().saturating_sub(pending.started_at)) as f64);
                    }
                } else {
                    let (old_timer, next_attempt) = {
                        let (_, pending) = &mut self.in_flight[pos];
                        (
                            pending.timeout_timer.take(),
                            pending.attempt.saturating_add(1),
                        )
                    };
                    if let Some(timer) = old_timer {
                        ctx.cancel_timer(timer);
                    }
                    self.schedule_retry(ctx, req_id, next_attempt);
                }
            }
            StormMessage::Timeout { req_id, attempt } => {
                let matched = self
                    .in_flight
                    .iter_mut()
                    .find(|(id, entry)| *id == req_id && entry.attempt == attempt);
                if let Some((_, entry)) = matched {
                    entry.timeout_timer = None;
                    let next_attempt = attempt.saturating_add(1);
                    self.schedule_retry(ctx, req_id, next_attempt);
                }
            }
            StormMessage::RetryNow { req_id, attempt } => {
                if self.in_flight.iter().any(|(id, _)| *id == req_id) {
                    self.send_attempt(ctx, req_id, attempt);
                }
            }
            StormMessage::Request { .. }
            | StormMessage::SlotDone { .. }
            | StormMessage::SetFault { .. } => {}
        }
    }

    fn on_tick(&mut self, ctx: &mut AgentContext<'_, StormMessage>) {
        // Steady arrival stream (~1.4 requests/tick, comfortably below server
        // nominal capacity of 2.0 req/tick until retries amplify load).
        let mut rng = ctx.rng();
        let arrivals = if rng.uniform_f64() < 0.4 { 2 } else { 1 };
        for _ in 0..arrivals {
            let req_id = self.next_req_id;
            self.next_req_id += 1;
            if ctx.time() >= RECOVERY_WINDOW_START {
                self.recovery_started += 1;
            }
            self.in_flight.push((
                req_id,
                PendingRequest {
                    started_at: ctx.time(),
                    attempt: 0,
                    timeout_timer: None,
                },
            ));
            self.send_attempt(ctx, req_id, 0);
        }
    }
}

/// Multi-slot server with bounded FIFO queue and transient degradation.
#[derive(Clone, Debug)]
struct ServiceServer {
    /// Concurrent worker slots.
    concurrency: usize,
    /// Busy worker slots.
    busy: usize,
    /// Maximum waiting queue size before fast load-shed rejection.
    queue_cap: usize,
    /// Waiting request IDs.
    waiting: VecDeque<u64>,
    /// Whether the downstream fault is currently active.
    degraded: bool,
    /// Cached client identity.
    client: Option<AgentId>,
    /// Requests rejected by load shedding.
    shed_count: u64,
}

impl ServiceServer {
    /// Creates a server with `concurrency` slots and `queue_cap` bounded queue.
    const fn new(concurrency: usize, queue_cap: usize) -> Self {
        Self {
            concurrency,
            busy: 0,
            queue_cap,
            waiting: VecDeque::new(),
            degraded: false,
            client: None,
            shed_count: 0,
        }
    }

    /// Starts executing `req_id` on a free slot.
    fn start_work(&mut self, ctx: &mut AgentContext<'_, StormMessage>, req_id: u64) {
        self.busy += 1;
        // Nominal service takes 2 ticks (capacity = concurrency / 2 = 2 req/tick).
        // Under congestion or transient fault, contention inflates service time.
        let base_service: DiscreteTime = if self.degraded { 7 } else { 2 };
        let queue_penalty: DiscreteTime = if self.waiting.len() > 24 { 2 } else { 0 };
        let jitter = ctx.rng().next_u64() % 2;
        ctx.send_after(
            base_service + queue_penalty + jitter,
            ctx.id(),
            StormMessage::SlotDone { req_id },
        );
    }
}

impl Agent<StormMessage> for ServiceServer {
    fn on_message(
        &mut self,
        ctx: &mut AgentContext<'_, StormMessage>,
        msg: &Message<StormMessage>,
    ) {
        if self.client.is_none() {
            self.client = ctx.agent_id("client");
        }
        match msg.payload {
            StormMessage::SetFault { active } => {
                self.degraded = active;
            }
            StormMessage::Request { req_id } => {
                if self.busy < self.concurrency {
                    self.start_work(ctx, req_id);
                } else if self.waiting.len() < self.queue_cap {
                    self.waiting.push_back(req_id);
                } else {
                    self.shed_count += 1;
                    if let Some(client) = self.client {
                        ctx.send_to(client, StormMessage::Response { req_id, ok: false });
                    }
                }
            }
            StormMessage::SlotDone { req_id } => {
                self.busy = self.busy.saturating_sub(1);
                if let Some(client) = self.client {
                    ctx.send_to(client, StormMessage::Response { req_id, ok: true });
                }
                if let Some(next_req) = self.waiting.pop_front() {
                    self.start_work(ctx, next_req);
                }
            }
            StormMessage::Response { .. }
            | StormMessage::Timeout { .. }
            | StormMessage::RetryNow { .. } => {}
        }
    }
}

/// Injects the transient fault at `[FAULT_START, FAULT_END)`.
#[derive(Clone, Debug)]
struct FaultInjector;

impl Agent<StormMessage> for FaultInjector {
    fn on_tick(&mut self, ctx: &mut AgentContext<'_, StormMessage>) {
        if ctx.time() == 0 {
            if let Some(server) = ctx.agent_id("server") {
                ctx.send_after(FAULT_START, server, StormMessage::SetFault { active: true });
                ctx.send_after(FAULT_END, server, StormMessage::SetFault { active: false });
            }
            ctx.sleep_for(HORIZON);
        }
    }
}

/// Metrics collected from one replication of the retry-storm simulation.
#[derive(Clone, Copy, Debug)]
struct StormOutcome {
    /// Post-fault success rate in `[0, 1]`.
    recovery_success_rate: f64,
    /// Post-fault P99 latency (or `HORIZON` penalty when no requests finish).
    recovery_p99_latency: f64,
    /// Total retry attempts issued during the run.
    total_retries: u64,
}

/// Builds a deterministic retry-storm simulation for `policy` and `ctx`.
fn build_simulation(
    policy: Policy,
    ctx: &TrialContext,
) -> Result<Simulation<StormMessage>, SimulationError> {
    Simulation::builder(Halt::AtTime(HORIZON))
        .trial(ctx)
        .unknown_destination_policy(UnknownDestinationPolicy::Panic)
        .agents([
            AgentInitializer::proactive("fault", FaultInjector),
            AgentInitializer::reactive("server", ServiceServer::new(5, policy.queue_cap))
                .drain_all_messages(),
            AgentInitializer::proactive("client", ClientPool::new(policy)).drain_all_messages(),
        ])
        .build()
}

/// Runs one replication and summarizes post-fault recovery performance.
fn evaluate_policy(policy: &Policy, ctx: &TrialContext) -> Result<StormOutcome, SimulationError> {
    let mut sim = build_simulation(*policy, ctx)?;
    sim.run();
    let Some(client) = sim
        .find_by_name("client")
        .and_then(|agent| agent.downcast_ref::<ClientPool>())
    else {
        return Ok(StormOutcome {
            recovery_success_rate: 0.0,
            recovery_p99_latency: HORIZON as f64,
            total_retries: 0,
        });
    };
    let success_rate = if client.recovery_started == 0 {
        0.0
    } else {
        client.recovery_succeeded as f64 / client.recovery_started as f64
    };
    let p99 =
        simul::experiment::QuantileReducer::from_results(client.recovery_latencies.iter().copied())
            .map_or(HORIZON as f64, |q| q.p99());
    Ok(StormOutcome {
        recovery_success_rate: success_rate,
        recovery_p99_latency: p99,
        total_retries: client.total_retries,
    })
}

/// Compares the naive and resilient policies using paired CRN statistics and
/// replays a forensic trace of the worst naive trial around the fault window.
fn compare_and_forensic(seed: MasterSeed) -> Result<(), Box<dyn std::error::Error>> {
    let naive = Policy {
        backoff: BackoffKind::Fixed,
        max_retries: 8,
        timeout: 6,
        queue_cap: 120,
    };
    let resilient = Policy {
        backoff: BackoffKind::ExponentialJitter,
        max_retries: 3,
        timeout: 10,
        queue_cap: 16,
    };

    let reps = 24;
    let schedule = FixedReplications::new(reps, 1);
    let trials = schedule
        .trials(CandidateId::new(0), naive)
        .chain(schedule.trials(CandidateId::new(1), resilient));
    let records = run_serial(seed, trials, evaluate_policy)?;

    let (naive_records, resilient_records) = records.split_at(reps as usize);
    let naive_p99 = aggregate_quantiles(
        naive_records,
        |o| o.recovery_p99_latency,
        FailurePolicy::RejectCandidate,
        NonFinitePolicy::RejectCandidate,
    )?
    .statistics;
    let resilient_p99 = aggregate_quantiles(
        resilient_records,
        |o| o.recovery_p99_latency,
        FailurePolicy::RejectCandidate,
        NonFinitePolicy::RejectCandidate,
    )?
    .statistics;
    let avg_retries =
        |slice: &[simul::experiment::TrialRecord<StormOutcome, SimulationError>]| {
            slice
                .iter()
                .filter_map(|r| r.result.as_ref().ok().map(|o| o.total_retries as f64))
                .sum::<f64>()
                / slice.len() as f64
        };
    let paired_success = compare_paired_candidates(
        &records,
        CandidateId::new(0),
        CandidateId::new(1),
        |o| o.recovery_success_rate,
        FailurePolicy::RejectCandidate,
        NonFinitePolicy::RejectCandidate,
    )?;

    println!("=== Retry Storm & Metastable Failure ===");
    println!(
        "Naive policy    : recovery success = {:.1}%, P99 = {:.1} ticks, CVaR95 = {:.1}, retries = {:.0}",
        paired_success.baseline().mean() * 100.0,
        naive_p99.median(),
        naive_p99.upper_cvar(Probability::P95),
        avg_retries(naive_records),
    );
    println!(
        "Resilient policy: recovery success = {:.1}%, P99 = {:.1} ticks, CVaR95 = {:.1}, retries = {:.0}",
        paired_success.challenger().mean() * 100.0,
        resilient_p99.median(),
        resilient_p99.upper_cvar(Probability::P95),
        avg_retries(resilient_records),
    );
    if let Some((lo, hi)) = paired_success.ci_95() {
        println!(
            "Paired CRN gain : {:+.1}% (95% CI [{:+.1}%, {:+.1}%]), wins = {}/{}",
            paired_success.difference().mean() * 100.0,
            lo * 100.0,
            hi * 100.0,
            paired_success.challenger_wins(),
            paired_success.usable_pairs(),
        );
    }

    if let Some(worst) = naive_records.iter().max_by(|a, b| {
        let la = a.result.as_ref().map_or(0.0, |o| o.recovery_p99_latency);
        let lb = b.result.as_ref().map_or(0.0, |o| o.recovery_p99_latency);
        la.total_cmp(&lb)
    }) {
        let report = forensic_replay_trial_with_payloads(
            worst.replay,
            ReplayWindow::new(FAULT_START, Some(FAULT_END + 25)),
            |ctx| build_simulation(naive, ctx),
        )?;
        let perfetto = report.trace.to_perfetto_json();
        println!(
            "\nForensic replay of worst naive trial (ticks {}..{}): {} trace events, {} bytes Perfetto JSON.",
            report.trace_started_at,
            report.ended_at,
            report.trace.len(),
            perfetto.len(),
        );
        if let Ok(path) = std::env::var("SIMUL_PERFETTO_OUT") {
            std::fs::write(&path, &perfetto)?;
            println!("Wrote Perfetto trace to {path}");
        }
    }
    Ok(())
}

/// Runs mixed discrete/continuous CEM and prints the discovered Pareto
/// frontier trading off post-fault P99 latency against recovery success rate.
fn optimize_pareto(seed: MasterSeed) -> Result<(), Box<dyn std::error::Error>> {
    let config = MixedCrossEntropyConfig::new(vec![
        MixedDimension::categorical(2),
        MixedDimension::integer(1, 6),
        MixedDimension::integer(8, 64),
        MixedDimension::continuous(5.0, 20.0),
    ])
    .with_elite_fraction(0.25);
    let mut optimizer = MixedCrossEntropyOptimizer::new(&config)?;
    let mut opt_rng = SeedDeriver::new(seed).proposal_seed(99).rng();
    let mut archive = ParetoArchive::new(vec![
        ObjectiveDirection::Minimize,
        ObjectiveDirection::Maximize,
    ])?;
    let eval_schedule = FixedReplications::new(4, 2);
    let mut next_candidate = 100_u64;

    for _ in 0..4 {
        let mut batch = Vec::new();
        for pt in optimizer.ask_batch(&mut opt_rng, 12) {
            let policy = Policy {
                backoff: if pt[0].as_category() == Some(1) {
                    BackoffKind::ExponentialJitter
                } else {
                    BackoffKind::Fixed
                },
                max_retries: u8::try_from(pt[1].as_i64().unwrap_or(2)).unwrap_or(2),
                queue_cap: usize::try_from(pt[2].as_i64().unwrap_or(16)).unwrap_or(16),
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                timeout: pt[3].as_f64().unwrap_or(10.0).round().max(1.0) as DiscreteTime,
            };
            let cid = CandidateId::new(next_candidate);
            next_candidate += 1;
            let cand_records =
                run_serial(seed, eval_schedule.trials(cid, policy), evaluate_policy)?;
            let avg_success = cand_records
                .iter()
                .filter_map(|r| r.result.as_ref().ok().map(|o| o.recovery_success_rate))
                .sum::<f64>()
                / 4.0;
            let avg_p99 = cand_records
                .iter()
                .filter_map(|r| r.result.as_ref().ok().map(|o| o.recovery_p99_latency))
                .sum::<f64>()
                / 4.0;
            archive.insert(cid, policy, vec![avg_p99, avg_success])?;
            let score = avg_success.mul_add(100.0, -avg_p99);
            batch.push(MixedCrossEntropySample::new(pt, score));
        }
        optimizer.tell(&mut batch)?;
    }

    println!(
        "\nPareto frontier ({} non-dominated policies, 2D HV = {:.1}):",
        archive.len(),
        archive.hypervolume_2d([HORIZON as f64, 0.0])?
    );
    for entry in archive.frontier() {
        let p = entry.payload;
        println!(
            "  {:?} retries={} qcap={} timeout={} -> P99={:.1} ticks, success={:.1}%",
            p.backoff,
            p.max_retries,
            p.queue_cap,
            p.timeout,
            entry.objectives[0],
            entry.objectives[1] * 100.0
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seed = MasterSeed::from_u64(2026);
    compare_and_forensic(seed)?;
    optimize_pareto(seed)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn resilient_policy_outperforms_naive_after_transient_fault() {
        let seed = MasterSeed::from_u64(77);
        let naive = Policy {
            backoff: BackoffKind::Fixed,
            max_retries: 8,
            timeout: 6,
            queue_cap: 120,
        };
        let resilient = Policy {
            backoff: BackoffKind::ExponentialJitter,
            max_retries: 3,
            timeout: 10,
            queue_cap: 16,
        };
        let schedule = FixedReplications::new(8, 1);
        let trials = schedule
            .trials(CandidateId::new(0), naive)
            .chain(schedule.trials(CandidateId::new(1), resilient));
        let records = run_serial(seed, trials, evaluate_policy).expect("run");
        let paired_p99 = compare_paired_candidates(
            &records,
            CandidateId::new(0),
            CandidateId::new(1),
            |o| o.recovery_p99_latency,
            FailurePolicy::RejectCandidate,
            NonFinitePolicy::RejectCandidate,
        )
        .expect("paired p99");
        assert!(
            paired_p99.difference().mean() < 0.0,
            "resilient policy should achieve lower post-fault P99 latency than naive: {:?}",
            paired_p99.difference()
        );
        let total_naive_retries: u64 = records[..8]
            .iter()
            .filter_map(|r| r.result.as_ref().ok().map(|o| o.total_retries))
            .sum();
        assert!(total_naive_retries > 0);
    }
}
