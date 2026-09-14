//! Deterministic, directly replicated experiments.
//!
//! This module is deliberately domain-neutral. Candidates are evaluated under
//! deterministic sampling contexts while each worker owns its evaluator.

use std::collections::{HashSet, TryReserveError};
use std::error::Error;
use std::fmt;
use std::io;
use std::mem::MaybeUninit;
use std::num::{NonZeroU32, NonZeroUsize};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::thread;

/// The deterministic sampling protocol used by [`SampleContext`].
pub const SEED_PROTOCOL: &str = "simul-v1-splitmix64-box-muller";

/// A deterministic random domain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RandomDomain(u64);

impl RandomDomain {
    /// Creates a random domain from its stable numeric identifier.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the stable numeric identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A deterministic sampling stream within a random domain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SampleStream(u64);

impl SampleStream {
    /// Creates a sample stream from its stable numeric identifier.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the stable numeric identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The deterministic inputs for one logical sample.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SampleContext {
    /// Root seed folded into every draw.
    master_seed: u64,
    /// Stable random domain.
    domain: RandomDomain,
    /// Logical sample identifier within the domain.
    sample_id: u64,
}

impl SampleContext {
    /// Creates a deterministic sampling context.
    #[must_use]
    pub const fn new(master_seed: u64, domain: RandomDomain, sample_id: u64) -> Self {
        Self {
            master_seed,
            domain,
            sample_id,
        }
    }

    /// Draws a deterministic uniform value in `[0, 1)`.
    #[must_use]
    pub fn uniform(self, stream: SampleStream) -> f64 {
        half_open_uniform(draw_bits(self, stream, 0))
    }

    /// Draws a standard normal conditioned to the inclusive truncation range.
    ///
    /// At most 128 Box–Muller pairs are tried. A value outside
    /// `[-maximum_standard_deviations, maximum_standard_deviations]` is rejected,
    /// not clamped.
    ///
    /// # Errors
    ///
    /// Returns [`SamplingError::InvalidTruncationLimit`] for a non-finite or
    /// non-positive limit, or [`SamplingError::RejectionLimitExceeded`] after
    /// 128 rejected pairs.
    pub fn truncated_standard_normal(
        self,
        stream: SampleStream,
        maximum_standard_deviations: f64,
    ) -> Result<f64, SamplingError> {
        if !maximum_standard_deviations.is_finite() || maximum_standard_deviations <= 0.0 {
            return Err(SamplingError::InvalidTruncationLimit);
        }

        for attempt in 0..128_u64 {
            let value = standard_normal_attempt(self, stream, attempt);
            if value.abs() <= maximum_standard_deviations {
                return Ok(value);
            }
        }

        Err(SamplingError::RejectionLimitExceeded)
    }
}

/// A deterministic sampling failure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SamplingError {
    /// The truncation limit was non-finite or not strictly positive.
    InvalidTruncationLimit,
    /// All 128 Box–Muller pairs were rejected.
    RejectionLimitExceeded,
}

impl fmt::Display for SamplingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTruncationLimit => {
                formatter.write_str("truncation limit must be finite and positive")
            }
            Self::RejectionLimitExceeded => {
                formatter.write_str("truncated-normal rejection limit exceeded")
            }
        }
    }
}

impl Error for SamplingError {}

/// A typed candidate for direct replication.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Candidate<T> {
    /// Stable identifier used in trial and replay keys.
    pub id: u64,
    /// Domain-specific candidate value.
    pub value: T,
}

/// Identifies a runner worker.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorkerId(usize);

impl WorkerId {
    /// Returns the zero-based worker identifier.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Configuration for a replicated run.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplicationPlan {
    /// Root seed for all deterministic samples and replay keys.
    pub master_seed: u64,
    /// Random domain for this run.
    pub random_domain: RandomDomain,
    /// Number of trials per candidate.
    pub replications: NonZeroU32,
    /// Maximum requested workers.
    pub workers: NonZeroUsize,
}

/// Stable coordinates for one replicated trial.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TrialKey {
    /// Random domain used by the trial.
    pub random_domain: RandomDomain,
    /// Candidate identifier.
    pub candidate_id: u64,
    /// Zero-based replication identifier.
    pub replication_id: u32,
    /// Common-random-number group shared across candidates.
    pub common_random_group: u64,
}

/// A versioned, allocation-free replay key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplayKey {
    /// Root seed displayed in the replay protocol.
    master_seed: u64,
    /// Stable trial coordinates.
    key: TrialKey,
}

impl ReplayKey {
    /// Creates a replay key.
    #[must_use]
    pub const fn new(master_seed: u64, key: TrialKey) -> Self {
        Self { master_seed, key }
    }

    /// Returns the master seed.
    #[must_use]
    pub const fn master_seed(self) -> u64 {
        self.master_seed
    }

    /// Returns the trial coordinates.
    #[must_use]
    pub const fn key(self) -> TrialKey {
        self.key
    }
}

impl fmt::Display for ReplayKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "simul-v1:{}:{:016x}:{}:{}:{}",
            self.master_seed,
            self.key.random_domain.get(),
            self.key.candidate_id,
            self.key.replication_id,
            self.key.common_random_group
        )
    }
}

/// The deterministic context passed to trial preparation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TrialContext {
    /// Complete replay identity for this trial.
    replay_key: ReplayKey,
}

impl TrialContext {
    /// Returns the trial coordinates.
    #[must_use]
    pub const fn key(self) -> TrialKey {
        self.replay_key.key()
    }

    /// Returns the complete replay key.
    #[must_use]
    pub const fn replay_key(self) -> ReplayKey {
        self.replay_key
    }

    /// Returns the common-random-number sampling context for this trial.
    ///
    /// Candidate identity is deliberately excluded so candidates in the same
    /// domain and replication receive identical standardized samples.
    #[must_use]
    pub const fn samples(self) -> SampleContext {
        let key = self.key();
        SampleContext::new(
            self.replay_key.master_seed(),
            key.random_domain,
            key.common_random_group,
        )
    }
}

/// A domain callback failure for one trial.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TrialError<PrepareError, EvaluateError> {
    /// Trial preparation failed.
    Prepare(PrepareError),
    /// Trial evaluation failed.
    Evaluate(EvaluateError),
}

impl<PrepareError, EvaluateError> fmt::Display for TrialError<PrepareError, EvaluateError>
where
    PrepareError: fmt::Display,
    EvaluateError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prepare(source) => write!(formatter, "trial preparation failed: {source}"),
            Self::Evaluate(source) => write!(formatter, "trial evaluation failed: {source}"),
        }
    }
}

impl<PrepareError, EvaluateError> Error for TrialError<PrepareError, EvaluateError>
where
    PrepareError: Error + 'static,
    EvaluateError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Prepare(source) => Some(source),
            Self::Evaluate(source) => Some(source),
        }
    }
}

/// Ordered result of one replicated trial.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TrialRecord<Outcome, PrepareError, EvaluateError> {
    /// Stable trial coordinates.
    pub key: TrialKey,
    /// Versioned replay key.
    pub replay_key: ReplayKey,
    /// Domain outcome or typed callback error.
    pub result: Result<Outcome, TrialError<PrepareError, EvaluateError>>,
}

/// Allocation performed by [`run_replicated`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReplicationAllocation {
    /// Complete ordered output.
    TrialRecords,
    /// Candidate-ID uniqueness set.
    CandidateIds,
    /// Parent-owned worker slots.
    WorkerSlots,
    /// Scoped thread handles.
    ThreadHandles,
}

impl fmt::Display for ReplicationAllocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::TrialRecords => "trial records",
            Self::CandidateIds => "candidate IDs",
            Self::WorkerSlots => "worker slots",
            Self::ThreadHandles => "thread handles",
        };
        formatter.write_str(name)
    }
}

/// A run-level replication failure.
#[derive(Debug)]
pub enum ReplicationError<WorkerError> {
    /// Candidate identifiers must be unique.
    DuplicateCandidateId {
        /// Repeated identifier.
        candidate_id: u64,
    },
    /// A `u32` replication count cannot be represented by the platform `usize`.
    ReplicationCountOverflow {
        /// Requested replications per candidate.
        replications: u32,
    },
    /// Candidate count multiplied by replication count overflowed `usize`.
    TrialCountOverflow {
        /// Candidate count.
        candidate_count: usize,
        /// Replications per candidate.
        replications: u32,
    },
    /// A prerequisite allocation failed before callbacks ran.
    TryReserve {
        /// Allocation being attempted.
        allocation: ReplicationAllocation,
        /// Worker associated with the allocation, when applicable.
        worker_id: Option<WorkerId>,
        /// Standard allocation failure.
        source: TryReserveError,
    },
    /// A worker factory returned an error.
    WorkerInitialization {
        /// Worker being initialized.
        worker_id: WorkerId,
        /// Factory error.
        source: WorkerError,
    },
    /// A worker factory panicked.
    WorkerFactoryPanic {
        /// Worker whose factory panicked.
        worker_id: WorkerId,
    },
    /// A scoped worker thread could not be spawned.
    ScopedThreadSpawn {
        /// Worker that could not be spawned.
        worker_id: WorkerId,
        /// Operating-system spawn error.
        source: io::Error,
    },
    /// A serial or parallel worker workload panicked.
    WorkerWorkloadPanic {
        /// Worker whose workload panicked.
        worker_id: WorkerId,
    },
}

impl<WorkerError> fmt::Display for ReplicationError<WorkerError>
where
    WorkerError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateCandidateId { candidate_id } => {
                write!(formatter, "duplicate candidate ID {candidate_id}")
            }
            Self::ReplicationCountOverflow { replications } => write!(
                formatter,
                "replication count {replications} cannot be represented on this platform"
            ),
            Self::TrialCountOverflow {
                candidate_count,
                replications,
            } => write!(
                formatter,
                "candidate count {candidate_count} × replication count {replications} overflows platform size"
            ),
            Self::TryReserve {
                allocation,
                worker_id,
                source,
            } => {
                if let Some(worker_id) = worker_id {
                    write!(
                        formatter,
                        "could not reserve {allocation} for worker {}: {source}",
                        worker_id.get()
                    )
                } else {
                    write!(formatter, "could not reserve {allocation}: {source}")
                }
            }
            Self::WorkerInitialization { worker_id, source } => write!(
                formatter,
                "worker {} initialization failed: {source}",
                worker_id.get()
            ),
            Self::WorkerFactoryPanic { worker_id } => {
                write!(formatter, "worker {} factory panicked", worker_id.get())
            }
            Self::ScopedThreadSpawn { worker_id, source } => write!(
                formatter,
                "worker {} scoped thread spawn failed: {source}",
                worker_id.get()
            ),
            Self::WorkerWorkloadPanic { worker_id } => {
                write!(formatter, "worker {} workload panicked", worker_id.get())
            }
        }
    }
}

impl<WorkerError> Error for ReplicationError<WorkerError>
where
    WorkerError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::TryReserve { source, .. } => Some(source),
            Self::ScopedThreadSpawn { source, .. } => Some(source),
            Self::WorkerInitialization { source, .. } => Some(source),
            Self::DuplicateCandidateId { .. }
            | Self::ReplicationCountOverflow { .. }
            | Self::TrialCountOverflow { .. }
            | Self::WorkerFactoryPanic { .. }
            | Self::WorkerWorkloadPanic { .. } => None,
        }
    }
}

/// Runs every candidate for every replication and returns deterministic order.
///
/// Records are returned in input-candidate order, then ascending replication
/// order. Callback and side-effect order is unspecified under parallelism.
/// Worker-count-independent results require semantically identical workers and
/// deterministic, schedule-independent callbacks.
///
/// # Errors
///
/// Returns [`ReplicationError`] when IDs or sizes are invalid, a prerequisite
/// allocation fails, worker construction or spawning fails, or a callback
/// workload panics. Prepare and evaluation errors remain trial-local records.
pub fn run_replicated<C, P, W, O, PE, EE, WE>(
    candidates: &[Candidate<C>],
    plan: ReplicationPlan,
    make_worker: impl Fn(WorkerId) -> Result<W, WE>,
    prepare: impl Fn(&C, TrialContext) -> Result<P, PE> + Sync,
    evaluate: impl Fn(&W, &P) -> Result<O, EE> + Sync,
) -> Result<Vec<TrialRecord<O, PE, EE>>, ReplicationError<WE>>
where
    C: Sync,
    P: Send,
    W: Send,
    O: Send,
    PE: Send,
    EE: Send,
{
    run_replicated_with_parallelism(
        candidates,
        plan,
        make_worker,
        prepare,
        evaluate,
        thread::available_parallelism(),
    )
}

/// Keeps the production parallelism cap injectable for host-independent tests.
#[allow(clippy::too_many_lines)]
fn run_replicated_with_parallelism<C, P, W, O, PE, EE, WE>(
    candidates: &[Candidate<C>],
    plan: ReplicationPlan,
    make_worker: impl Fn(WorkerId) -> Result<W, WE>,
    prepare: impl Fn(&C, TrialContext) -> Result<P, PE> + Sync,
    evaluate: impl Fn(&W, &P) -> Result<O, EE> + Sync,
    available: io::Result<NonZeroUsize>,
) -> Result<Vec<TrialRecord<O, PE, EE>>, ReplicationError<WE>>
where
    C: Sync,
    P: Send,
    W: Send,
    O: Send,
    PE: Send,
    EE: Send,
{
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    validate_candidate_ids(candidates)?;

    let (replications, trial_count) =
        checked_trial_count::<WE>(candidates.len(), plan.replications.get())?;

    let worker_count = capped_worker_count(plan.workers, trial_count, available);
    let mut output = Vec::new();
    output
        .try_reserve_exact(trial_count)
        .map_err(|source| reserve_error(ReplicationAllocation::TrialRecords, None, source))?;
    let mut slots = allocate_worker_slots::<W, _, WE>(
        worker_count,
        &mut output.spare_capacity_mut()[..trial_count],
    )?;

    if worker_count == 1 {
        initialize_workers(&mut slots, &make_worker)?;
        let slot = &mut slots[0];
        let Some(worker) = slot.worker.take() else {
            return Err(ReplicationError::WorkerFactoryPanic { worker_id: slot.id });
        };
        catch_unwind(AssertUnwindSafe(|| {
            run_worker_range(
                candidates,
                replications,
                plan.master_seed,
                plan.random_domain,
                &worker,
                &prepare,
                &evaluate,
                slot.start,
                &mut slot.records,
            );
        }))
        .map_err(|_| ReplicationError::WorkerWorkloadPanic { worker_id: slot.id })?;
    } else {
        thread::scope(|scope| {
            let mut handles = Vec::new();
            handles.try_reserve_exact(worker_count).map_err(|source| {
                reserve_error(ReplicationAllocation::ThreadHandles, None, source)
            })?;

            initialize_workers(&mut slots, &make_worker)?;

            for slot in &mut slots {
                let Some(worker) = slot.worker.take() else {
                    join_spawned_handles(handles);
                    return Err(ReplicationError::WorkerFactoryPanic { worker_id: slot.id });
                };
                let worker_id = slot.id;
                let spawn = thread::Builder::new().spawn_scoped(scope, {
                    let prepare = &prepare;
                    let evaluate = &evaluate;
                    move || {
                        catch_unwind(AssertUnwindSafe(|| {
                            run_worker_range(
                                candidates,
                                replications,
                                plan.master_seed,
                                plan.random_domain,
                                &worker,
                                prepare,
                                evaluate,
                                slot.start,
                                &mut slot.records,
                            );
                        }))
                    }
                });

                match spawn {
                    Ok(handle) => handles.push((worker_id, handle)),
                    Err(source) => {
                        join_spawned_handles(handles);
                        return Err(ReplicationError::ScopedThreadSpawn { worker_id, source });
                    }
                }
            }

            let mut first_panic = None;
            for (worker_id, handle) in handles {
                match handle.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(_)) | Err(_) => {
                        if first_panic.is_none() {
                            first_panic = Some(worker_id);
                        }
                    }
                }
            }
            first_panic.map_or_else(
                || Ok(()),
                |worker_id| Err(ReplicationError::WorkerWorkloadPanic { worker_id }),
            )
        })?;
    }

    // All ranges completed and all threads joined. Transfer drop ownership from
    // the guards to the Vec without moving records or allocating another buffer.
    for slot in &mut slots {
        slot.records.initialized = 0;
    }
    drop(slots);
    // SAFETY: the disjoint ranges cover exactly `trial_count` reserved elements,
    // and successful completion initialized every element. No slice borrows or
    // guards remain, and output now becomes their sole drop owner.
    unsafe { output.set_len(trial_count) };
    Ok(output)
}

/// Parent-owned worker state and an exclusively borrowed output range.
struct WorkerSlot<'a, W, T> {
    /// Stable worker identifier.
    id: WorkerId,
    /// Inclusive flat range start.
    start: usize,
    /// Worker initialized before any trial callback.
    worker: Option<W>,
    /// Owns destruction of the initialized prefix until the run succeeds.
    records: RecordRange<'a, T>,
}

/// Drops partially initialized records on every unsuccessful exit.
struct RecordRange<'a, T> {
    /// Disjoint portion of the output Vec's spare capacity.
    storage: &'a mut [MaybeUninit<T>],
    /// Exactly this prefix contains live values owned by this guard.
    initialized: usize,
}

impl<T> RecordRange<'_, T> {
    /// Writes the next record before taking ownership of its destruction.
    fn push(&mut self, value: T) {
        self.storage[self.initialized].write(value);
        self.initialized += 1;
    }
}

impl<T> Drop for RecordRange<'_, T> {
    fn drop(&mut self) {
        for record in &mut self.storage[..self.initialized] {
            // SAFETY: push initializes an element before extending the prefix.
            // This guard exclusively owns the prefix, including after a caught
            // callback panic; successful runs disarm it before Vec takes over.
            unsafe { record.assume_init_drop() };
        }
    }
}

/// Applies `SplitMix64`'s avalanche function.
const fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Draws deterministic bits for one context, stream, and draw index.
const fn draw_bits(context: SampleContext, stream: SampleStream, draw_index: u64) -> u64 {
    let state = mix(context.master_seed ^ context.domain.get());
    let state = mix(state ^ context.sample_id);
    let state = mix(state ^ stream.get());
    mix(state ^ draw_index)
}

/// Maps bits to the half-open unit interval.
fn half_open_uniform(bits: u64) -> f64 {
    let top53 = (bits >> 11) as f64;
    top53 / 9_007_199_254_740_992.0
}

/// Maps bits to a genuinely open unit interval for Box–Muller `u1`.
fn open_uniform(bits: u64) -> f64 {
    let top53 = (bits >> 11) as f64;
    (top53 + 1.0) / 9_007_199_254_740_994.0
}

/// Computes one deterministic Box–Muller normal attempt.
fn standard_normal_attempt(context: SampleContext, stream: SampleStream, attempt: u64) -> f64 {
    let first_index = attempt.wrapping_mul(2);
    let second_index = first_index.wrapping_add(1);
    let u1 = open_uniform(draw_bits(context, stream, first_index));
    let u2 = half_open_uniform(draw_bits(context, stream, second_index));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Converts and multiplies candidate/replication sizes with checks.
fn checked_trial_count<WE>(
    candidate_count: usize,
    replications: u32,
) -> Result<(usize, usize), ReplicationError<WE>> {
    let replications_usize = usize::try_from(replications)
        .map_err(|_| ReplicationError::ReplicationCountOverflow { replications })?;
    let trial_count = candidate_count.checked_mul(replications_usize).ok_or(
        ReplicationError::TrialCountOverflow {
            candidate_count,
            replications,
        },
    )?;
    Ok((replications_usize, trial_count))
}

/// Allocates and checks the candidate-ID uniqueness set.
fn validate_candidate_ids<C, WE>(candidates: &[Candidate<C>]) -> Result<(), ReplicationError<WE>> {
    let mut identifiers = HashSet::new();
    identifiers
        .try_reserve(candidates.len())
        .map_err(|source| reserve_error(ReplicationAllocation::CandidateIds, None, source))?;
    for candidate in candidates {
        if !identifiers.insert(candidate.id) {
            return Err(ReplicationError::DuplicateCandidateId {
                candidate_id: candidate.id,
            });
        }
    }
    Ok(())
}

/// Caps workers by request, work, and available parallelism.
fn capped_worker_count(
    requested: NonZeroUsize,
    trial_count: usize,
    available: io::Result<NonZeroUsize>,
) -> usize {
    let available = available.map_or(1, NonZeroUsize::get);
    requested.get().min(trial_count).min(available)
}

/// Allocates worker metadata and partitions the single output allocation.
fn allocate_worker_slots<W, T, WE>(
    worker_count: usize,
    mut storage: &mut [MaybeUninit<T>],
) -> Result<Vec<WorkerSlot<'_, W, T>>, ReplicationError<WE>> {
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(worker_count)
        .map_err(|source| reserve_error(ReplicationAllocation::WorkerSlots, None, source))?;

    let base = storage.len().div_euclid(worker_count);
    let remainder = storage.len() % worker_count;
    let mut start = 0;
    for index in 0..worker_count {
        let length = base + usize::from(index < remainder);
        let (range, remaining) = storage.split_at_mut(length);
        storage = remaining;
        slots.push(WorkerSlot {
            id: WorkerId(index),
            start,
            worker: None,
            records: RecordRange {
                storage: range,
                initialized: 0,
            },
        });
        start += length;
    }

    Ok(slots)
}

/// Initializes every worker on the parent thread before any trial runs.
fn initialize_workers<W, T, WE>(
    slots: &mut [WorkerSlot<'_, W, T>],
    make_worker: &impl Fn(WorkerId) -> Result<W, WE>,
) -> Result<(), ReplicationError<WE>> {
    for slot in slots {
        let result = catch_unwind(AssertUnwindSafe(|| make_worker(slot.id)));
        match result {
            Ok(Ok(worker)) => slot.worker = Some(worker),
            Ok(Err(source)) => {
                return Err(ReplicationError::WorkerInitialization {
                    worker_id: slot.id,
                    source,
                });
            }
            Err(_) => {
                return Err(ReplicationError::WorkerFactoryPanic { worker_id: slot.id });
            }
        }
    }
    Ok(())
}

/// Evaluates one contiguous flat range directly into output spare capacity.
#[allow(clippy::too_many_arguments)]
fn run_worker_range<C, P, W, O, PE, EE>(
    candidates: &[Candidate<C>],
    replications: usize,
    master_seed: u64,
    random_domain: RandomDomain,
    worker: &W,
    prepare: &impl Fn(&C, TrialContext) -> Result<P, PE>,
    evaluate: &impl Fn(&W, &P) -> Result<O, EE>,
    start: usize,
    records: &mut RecordRange<'_, TrialRecord<O, PE, EE>>,
) {
    for flat_index in start..start + records.storage.len() {
        let candidate = &candidates[flat_index.div_euclid(replications)];
        let replication_index = flat_index % replications;
        let replication_id = match u32::try_from(replication_index) {
            Ok(replication_id) => replication_id,
            Err(error) => panic!("validated replication index did not fit u32: {error}"),
        };
        let key = TrialKey {
            random_domain,
            candidate_id: candidate.id,
            replication_id,
            common_random_group: u64::from(replication_id),
        };
        let replay_key = ReplayKey::new(master_seed, key);
        let context = TrialContext { replay_key };
        let result = prepare(&candidate.value, context)
            .map_err(TrialError::Prepare)
            .and_then(|prepared| evaluate(worker, &prepared).map_err(TrialError::Evaluate));
        records.push(TrialRecord {
            key,
            replay_key,
            result,
        });
    }
}

/// Explicitly joins all successfully spawned handles after a spawn failure.
fn join_spawned_handles<T>(handles: Vec<(WorkerId, thread::ScopedJoinHandle<'_, T>)>) {
    for (_, handle) in handles {
        let _ = handle.join();
    }
}

/// Constructs a typed allocation error.
const fn reserve_error<WE>(
    allocation: ReplicationAllocation,
    worker_id: Option<WorkerId>,
    source: TryReserveError,
) -> ReplicationError<WE> {
    ReplicationError::TryReserve {
        allocation,
        worker_id,
        source,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::ignored_unit_patterns, clippy::manual_assert)]

    use super::*;
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    const DOMAIN: RandomDomain = RandomDomain::new(0x5345_4152_4348_0002);
    const HEADING: SampleStream = SampleStream::new(0x4845_4144_494e_4701);
    const SPEED: SampleStream = SampleStream::new(0x5350_4545_4400_0001);
    const SIDE: SampleStream = SampleStream::new(0x5349_4445_0000_0001);
    const HEIGHT: SampleStream = SampleStream::new(0x4845_4947_4854_0001);
    const ELEVATION: SampleStream = SampleStream::new(0x454c_4556_4154_0001);

    fn nonzero_u32(value: u32) -> NonZeroU32 {
        let Some(value) = NonZeroU32::new(value) else {
            panic!("test value must be nonzero");
        };
        value
    }

    fn nonzero_usize(value: usize) -> NonZeroUsize {
        let Some(value) = NonZeroUsize::new(value) else {
            panic!("test value must be nonzero");
        };
        value
    }

    fn plan(replications: u32, workers: usize) -> ReplicationPlan {
        ReplicationPlan {
            master_seed: 918_273,
            random_domain: DOMAIN,
            replications: nonzero_u32(replications),
            workers: nonzero_usize(workers),
        }
    }

    #[test]
    fn uniform_protocol_goldens_are_exact() {
        let context = SampleContext::new(918_273, DOMAIN, 29);
        let goldens = [
            (HEADING, 0x3fe6_1efa_9df0_d72f),
            (SPEED, 0x3fc6_4031_dee3_a860),
            (SIDE, 0x3fd8_2603_3574_452e),
            (HEIGHT, 0x3fe1_7772_c07c_2423),
            (ELEVATION, 0x3fc3_86d0_8b35_ba74),
        ];

        for (stream, expected_bits) in goldens {
            assert_eq!(context.uniform(stream).to_bits(), expected_bits);
        }
    }

    #[test]
    fn open_uniform_endpoints_are_exact() {
        assert_eq!(open_uniform(0).to_bits(), 0x3c9f_ffff_ffff_fffe);
        assert_eq!(open_uniform(u64::MAX).to_bits(), 0x3fef_ffff_ffff_fffe);
    }

    #[test]
    fn truncated_normal_protocol_goldens_are_stable() {
        let context = SampleContext::new(918_273, DOMAIN, 29);
        let goldens = [
            (HEADING, -0.468_319_509_146_165_4),
            (SPEED, -1.859_691_657_998_522_8),
            (SIDE, 1.315_049_240_096_262),
            (HEIGHT, 0.210_883_684_077_519),
            (ELEVATION, 0.771_383_774_814_691_2),
        ];

        for (stream, expected) in goldens {
            let first = match context.truncated_standard_normal(stream, 3.0) {
                Ok(value) => value,
                Err(error) => panic!("golden sample failed: {error}"),
            };
            let second = match context.truncated_standard_normal(stream, 3.0) {
                Ok(value) => value,
                Err(error) => panic!("repeated golden sample failed: {error}"),
            };
            assert!((first - expected).abs() <= 1e-15);
            assert_eq!(first.to_bits(), second.to_bits());
        }
    }

    #[test]
    fn truncated_normal_rejects_the_first_sample() {
        let context = SampleContext::new(7, DOMAIN, 412);
        let first = standard_normal_attempt(context, HEADING, 0);
        let second = standard_normal_attempt(context, HEADING, 1);
        assert!((first - -3.008_621_604_102_551).abs() <= 1e-15);
        assert!((second - -0.885_048_564_735_371_1).abs() <= 1e-15);
        assert_eq!(context.truncated_standard_normal(HEADING, 3.0), Ok(second));
    }

    #[test]
    fn truncated_normal_reports_invalid_limits_and_exhaustion() {
        let context = SampleContext::new(7, DOMAIN, 412);
        for limit in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                context.truncated_standard_normal(HEADING, limit),
                Err(SamplingError::InvalidTruncationLimit)
            );
        }
        assert_eq!(
            context.truncated_standard_normal(HEADING, 0.001),
            Err(SamplingError::RejectionLimitExceeded)
        );
    }

    #[test]
    fn replay_key_format_is_versioned_and_exact() {
        let key = TrialKey {
            random_domain: DOMAIN,
            candidate_id: 41,
            replication_id: 29,
            common_random_group: 29,
        };
        let replay = ReplayKey::new(918_273, key);
        assert_eq!(
            replay.to_string(),
            "simul-v1:918273:5345415243480002:41:29:29"
        );
        assert_eq!(replay.master_seed(), 918_273);
        assert_eq!(replay.key(), key);
    }

    #[test]
    fn empty_candidates_do_not_call_factory() {
        let called = AtomicBool::new(false);
        let result = run_replicated::<u8, (), (), (), (), (), ()>(
            &[],
            plan(1, 4),
            |_| {
                called.store(true, Ordering::SeqCst);
                Ok(())
            },
            |_, _| Ok(()),
            |_, _| Ok(()),
        );
        assert!(matches!(result, Ok(records) if records.is_empty()));
        assert!(!called.load(Ordering::SeqCst));
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected_before_factory() {
        let called = AtomicBool::new(false);
        let candidates = [
            Candidate { id: 9, value: () },
            Candidate { id: 9, value: () },
        ];
        let result = run_replicated(
            &candidates,
            plan(1, 1),
            |_| {
                called.store(true, Ordering::SeqCst);
                Ok::<_, ()>(())
            },
            |_, _| Ok::<_, ()>(()),
            |_, _| Ok::<_, ()>(()),
        );
        assert!(matches!(
            result,
            Err(ReplicationError::DuplicateCandidateId { candidate_id: 9 })
        ));
        assert!(!called.load(Ordering::SeqCst));
    }

    #[test]
    fn checked_count_and_worker_caps_cover_boundaries() {
        let count = checked_trial_count::<Infallible>(3, 7);
        assert!(matches!(count, Ok((7, 21))));
        let overflow = checked_trial_count::<Infallible>(usize::MAX, 2);
        assert!(matches!(
            overflow,
            Err(ReplicationError::TrialCountOverflow { .. })
        ));

        assert_eq!(
            capped_worker_count(nonzero_usize(8), 3, Ok(nonzero_usize(16))),
            3
        );
        assert_eq!(
            capped_worker_count(nonzero_usize(8), 32, Ok(nonzero_usize(2))),
            2
        );
        assert_eq!(
            capped_worker_count(
                nonzero_usize(8),
                32,
                Err(io::Error::other("availability unavailable"))
            ),
            1
        );
    }

    #[test]
    fn all_workers_are_constructed_before_prepare() {
        for available in [
            Ok(nonzero_usize(1)),
            Ok(nonzero_usize(2)),
            Err(io::Error::other("availability unavailable")),
        ] {
            let expected = available.as_ref().map_or(1, |count| count.get());
            let constructed = AtomicUsize::new(0);
            let candidates = [Candidate { id: 0, value: () }];
            let records = run_replicated_with_parallelism(
                &candidates,
                plan(8, 2),
                |_| {
                    constructed.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, ()>(())
                },
                |_, _| {
                    assert_eq!(constructed.load(Ordering::SeqCst), expected);
                    Ok::<_, ()>(())
                },
                |_, _| Ok::<_, ()>(()),
                available,
            );
            assert!(matches!(records, Ok(records) if records.len() == 8));
        }
    }

    #[test]
    fn factory_error_and_panic_are_typed_and_precede_trials() {
        let prepared = AtomicBool::new(false);
        let candidates = [Candidate { id: 0, value: () }];
        let error = run_replicated_with_parallelism(
            &candidates,
            plan(4, 2),
            |id| {
                if id.get() == 1 {
                    Err("factory")
                } else {
                    Ok(())
                }
            },
            |_, _| {
                prepared.store(true, Ordering::SeqCst);
                Ok::<_, ()>(())
            },
            |_, _| Ok::<_, ()>(()),
            Ok(nonzero_usize(2)),
        );
        assert!(matches!(
            error,
            Err(ReplicationError::WorkerInitialization {
                worker_id,
                source: "factory"
            }) if worker_id.get() == 1
        ));
        assert!(!prepared.load(Ordering::SeqCst));

        let panic = run_replicated_with_parallelism(
            &candidates,
            plan(4, 2),
            |id| -> Result<(), ()> {
                if id.get() == 1 {
                    panic!("factory panic");
                }
                Ok(())
            },
            |_, _| {
                prepared.store(true, Ordering::SeqCst);
                Ok::<_, ()>(())
            },
            |_, _| Ok::<_, ()>(()),
            Ok(nonzero_usize(2)),
        );
        assert!(matches!(
            panic,
            Err(ReplicationError::WorkerFactoryPanic { worker_id }) if worker_id.get() == 1
        ));
        assert!(!prepared.load(Ordering::SeqCst));
    }

    #[test]
    fn prepare_and_evaluate_errors_remain_trial_local() {
        let candidates = [Candidate { id: 5, value: 5_u8 }];
        let records = run_replicated(
            &candidates,
            plan(2, 1),
            |_| Ok::<_, Infallible>(()),
            |value, context| {
                if context.key().replication_id == 0 {
                    Err("prepare")
                } else {
                    Ok(*value)
                }
            },
            |_, _| Err::<u8, _>("evaluate"),
        );
        let records = match records {
            Ok(records) => records,
            Err(error) => panic!("run failed: {error}"),
        };
        assert!(matches!(
            records[0].result,
            Err(TrialError::Prepare("prepare"))
        ));
        assert!(matches!(
            records[1].result,
            Err(TrialError::Evaluate("evaluate"))
        ));
    }

    #[test]
    fn serial_and_parallel_workload_panics_are_run_level() {
        let candidates = [Candidate { id: 0, value: () }];
        let serial = run_replicated(
            &candidates,
            plan(1, 1),
            |_| Ok::<_, ()>(()),
            |_, _| -> Result<(), ()> { panic!("serial workload") },
            |_, _| Ok::<_, ()>(()),
        );
        assert!(matches!(
            serial,
            Err(ReplicationError::WorkerWorkloadPanic { worker_id }) if worker_id.get() == 0
        ));

        let completed = Arc::new(AtomicBool::new(false));
        let completed_for_prepare = Arc::clone(&completed);
        let candidates = [
            Candidate { id: 0, value: 0_u8 },
            Candidate { id: 1, value: 1_u8 },
        ];
        let parallel = run_replicated_with_parallelism(
            &candidates,
            plan(1, 2),
            |_| Ok::<_, ()>(()),
            move |candidate, _| {
                if *candidate == 0 {
                    panic!("parallel workload");
                }
                thread::sleep(Duration::from_millis(20));
                completed_for_prepare.store(true, Ordering::SeqCst);
                Ok::<_, ()>(())
            },
            |_, _| Ok::<_, ()>(()),
            Ok(nonzero_usize(2)),
        );
        assert!(matches!(
            parallel,
            Err(ReplicationError::WorkerWorkloadPanic { worker_id }) if worker_id.get() == 0
        ));
        assert!(completed.load(Ordering::SeqCst));
    }

    #[test]
    fn records_are_candidate_major_then_replication_minor() {
        let candidates = [
            Candidate {
                id: 41,
                value: 100_u64,
            },
            Candidate {
                id: 7,
                value: 200_u64,
            },
        ];
        let records = run_replicated_with_parallelism(
            &candidates,
            plan(3, 4),
            |_| Ok::<_, Infallible>(()),
            |value, context| Ok::<_, Infallible>((*value, context.key().replication_id)),
            |_, prepared| Ok::<_, Infallible>(*prepared),
            Ok(nonzero_usize(4)),
        );
        let records = match records {
            Ok(records) => records,
            Err(error) => panic!("ordered run failed: {error}"),
        };
        let coordinates: Vec<_> = records
            .iter()
            .map(|record| (record.key.candidate_id, record.key.replication_id))
            .collect();
        assert_eq!(
            coordinates,
            [(41, 0), (41, 1), (41, 2), (7, 0), (7, 1), (7, 2)]
        );
        assert!(matches!(records[0].result, Ok((100, 0))));
        assert!(matches!(records[5].result, Ok((200, 2))));
    }

    #[test]
    fn common_random_groups_exclude_candidate_identity() {
        let candidates = [
            Candidate { id: 10, value: () },
            Candidate { id: 20, value: () },
        ];
        let records = run_replicated(
            &candidates,
            plan(2, 2),
            |_| Ok::<_, Infallible>(()),
            |_, context| Ok::<_, Infallible>(context.samples().uniform(HEADING).to_bits()),
            |_, sample| Ok::<_, Infallible>(*sample),
        );
        let records = match records {
            Ok(records) => records,
            Err(error) => panic!("common-random run failed: {error}"),
        };
        assert_eq!(records[0].result, records[2].result);
        assert_eq!(records[1].result, records[3].result);
        assert_ne!(records[0].result, records[1].result);
        assert_eq!(records[0].key.common_random_group, 0);
        assert_eq!(records[1].key.common_random_group, 1);

        let other_domain_plan = ReplicationPlan {
            random_domain: RandomDomain::new(DOMAIN.get() + 1),
            ..plan(2, 1)
        };
        let other_domain = run_replicated(
            &candidates[..1],
            other_domain_plan,
            |_| Ok::<_, Infallible>(()),
            |_, context| Ok::<_, Infallible>(context.samples().uniform(HEADING).to_bits()),
            |_, sample| Ok::<_, Infallible>(*sample),
        );
        let other_domain = match other_domain {
            Ok(records) => records,
            Err(error) => panic!("other-domain run failed: {error}"),
        };
        assert_ne!(records[0].result, other_domain[0].result);
    }

    #[test]
    fn immutable_evaluator_results_do_not_depend_on_worker_count() {
        let candidates = [
            Candidate {
                id: 2,
                value: 11_u64,
            },
            Candidate {
                id: 8,
                value: 17_u64,
            },
        ];
        let run = |workers| {
            run_replicated_with_parallelism(
                &candidates,
                plan(8, workers),
                |_| Ok::<_, Infallible>(5_u64),
                |candidate, context| {
                    Ok::<_, Infallible>((*candidate, context.samples().uniform(SPEED).to_bits()))
                },
                |evaluator, prepared| Ok::<_, Infallible>(prepared.0 ^ prepared.1 ^ *evaluator),
                Ok(nonzero_usize(workers)),
            )
        };
        let serial = match run(1) {
            Ok(records) => records,
            Err(error) => panic!("serial run failed: {error}"),
        };
        let parallel = match run(4) {
            Ok(records) => records,
            Err(error) => panic!("parallel run failed: {error}"),
        };
        assert_eq!(serial, parallel);
    }

    struct TrackedResult {
        index: usize,
        drops: Arc<[AtomicUsize; 12]>,
    }

    impl Drop for TrackedResult {
        fn drop(&mut self) {
            self.drops[self.index].fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn ordered_results_and_trial_errors_drop_once_after_success() {
        for workers in [1, 3] {
            let drops = Arc::new(std::array::from_fn(|_| AtomicUsize::new(0)));
            let candidates = [
                Candidate { id: 41, value: 0 },
                Candidate { id: 7, value: 5 },
            ];
            let records = run_replicated_with_parallelism(
                &candidates,
                plan(5, workers),
                |_| Ok::<_, Infallible>(()),
                |offset, context| {
                    let index = offset
                        + usize::try_from(context.key().replication_id)
                            .unwrap_or_else(|error| panic!("invalid test index: {error}"));
                    if index % 3 == 1 {
                        Err(TrackedResult {
                            index,
                            drops: Arc::clone(&drops),
                        })
                    } else {
                        Ok(index)
                    }
                },
                |_, &index| {
                    let result = TrackedResult {
                        index,
                        drops: Arc::clone(&drops),
                    };
                    if index % 3 == 2 {
                        Err(result)
                    } else {
                        Ok(result)
                    }
                },
                Ok(nonzero_usize(workers)),
            );
            let records = match records {
                Ok(records) => records,
                Err(error) => panic!("tracked run failed: {error}"),
            };
            let actual: Vec<_> = records
                .iter()
                .map(|record| {
                    let value = match &record.result {
                        Ok(value)
                        | Err(TrialError::Prepare(value) | TrialError::Evaluate(value)) => value,
                    };
                    (
                        record.key.candidate_id,
                        record.key.replication_id,
                        value.index,
                    )
                })
                .collect();
            assert_eq!(
                actual,
                [
                    (41, 0, 0),
                    (41, 1, 1),
                    (41, 2, 2),
                    (41, 3, 3),
                    (41, 4, 4),
                    (7, 0, 5),
                    (7, 1, 6),
                    (7, 2, 7),
                    (7, 3, 8),
                    (7, 4, 9),
                ]
            );
            assert!(drops.iter().all(|count| count.load(Ordering::SeqCst) == 0));
            drop(records);
            for (index, count) in drops.iter().enumerate() {
                assert_eq!(count.load(Ordering::SeqCst), usize::from(index < 10));
            }
        }
    }

    #[test]
    fn partial_and_completed_worker_results_drop_once_on_panic() {
        for workers in [1, 3] {
            for panic_in_prepare in [false, true] {
                let drops = Arc::new(std::array::from_fn(|_| AtomicUsize::new(0)));
                let candidates = [
                    Candidate { id: 41, value: 0 },
                    Candidate { id: 7, value: 6 },
                ];
                let result = run_replicated_with_parallelism(
                    &candidates,
                    plan(6, workers),
                    |_| Ok::<_, Infallible>(()),
                    |offset, context| {
                        let index = offset
                            + usize::try_from(context.key().replication_id)
                                .unwrap_or_else(|error| panic!("invalid test index: {error}"));
                        if panic_in_prepare && index == 2 {
                            panic!("prepare panic after two initialized records");
                        }
                        if index % 3 == 1 {
                            Err(TrackedResult {
                                index,
                                drops: Arc::clone(&drops),
                            })
                        } else {
                            Ok(index)
                        }
                    },
                    |_, &index| {
                        if !panic_in_prepare && index == 2 {
                            panic!("evaluate panic after two initialized records");
                        }
                        let result = TrackedResult {
                            index,
                            drops: Arc::clone(&drops),
                        };
                        if index % 3 == 2 {
                            Err(result)
                        } else {
                            Ok(result)
                        }
                    },
                    Ok(nonzero_usize(workers)),
                );
                assert!(matches!(
                    result,
                    Err(ReplicationError::WorkerWorkloadPanic { worker_id }) if worker_id.get() == 0
                ));
                for (index, count) in drops.iter().enumerate() {
                    let initialized = index < 2 || (workers > 1 && index >= 4);
                    assert_eq!(
                        count.load(Ordering::SeqCst),
                        usize::from(initialized),
                        "trial {index}"
                    );
                }
            }
        }
    }
}
