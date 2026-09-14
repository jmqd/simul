//! Engine-independent deterministic experiment execution and reduction.

use blake3::Hasher;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use std::cmp::Ordering;
use std::fmt;
use std::num::NonZeroUsize;

/// Root domain separator for the frozen V1 seed protocol.
const ROOT_NAMESPACE: &[u8] = b"simul/seed-protocol/v1";
/// Domain separator for proposal streams.
const PROPOSAL_NAMESPACE: &[u8] = b"simul/v1/proposal";
/// Domain separator for optimizer-acceptance streams.
const ACCEPTANCE_NAMESPACE: &[u8] = b"simul/v1/acceptance";
/// Domain separator for candidate-specific trial streams.
const INDEPENDENT_NAMESPACE: &[u8] = b"simul/v1/trial-independent";
/// Domain separator for common-random trial streams.
const COMMON_NAMESPACE: &[u8] = b"simul/v1/trial-common";
/// Domain separator for generic trial-noise streams.
const TRIAL_NOISE_NAMESPACE: &[u8] = b"simul/v1/trial-noise";
/// Domain separator for layout-noise streams.
const LAYOUT_NOISE_NAMESPACE: &[u8] = b"simul/v1/layout-noise";
/// Domain separator for execution-noise streams.
const EXECUTION_NOISE_NAMESPACE: &[u8] = b"simul/v1/execution-noise";
/// Domain separator for validation-holdout streams.
const HOLDOUT_NAMESPACE: &[u8] = b"simul/v1/validation-holdout";
/// Scale converting the high 53 bits of a `u64` to the unit interval.
const U64_TO_UNIT_F64: f64 = 1.0 / ((1_u64 << 53) as f64);

/// A stable candidate identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CandidateId(u64);

impl CandidateId {
    /// Creates an identity from its stable integer representation.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the stable integer representation.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// A stable replication identity within a candidate evaluation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReplicationId(u32);

impl ReplicationId {
    /// Creates an identity from its stable integer representation.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the stable integer representation.
    #[must_use]
    pub const fn value(self) -> u32 {
        self.0
    }
}

/// The stable identity of one candidate replication.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TrialId {
    /// Candidate evaluated by this trial.
    candidate: CandidateId,
    /// Replication of the candidate evaluation.
    replication: ReplicationId,
}

impl TrialId {
    /// Creates a trial identity.
    #[must_use]
    pub const fn new(candidate: CandidateId, replication: ReplicationId) -> Self {
        Self {
            candidate,
            replication,
        }
    }

    /// Returns the candidate identity.
    #[must_use]
    pub const fn candidate(self) -> CandidateId {
        self.candidate
    }

    /// Returns the replication identity.
    #[must_use]
    pub const fn replication(self) -> ReplicationId {
        self.replication
    }
}

/// The frozen seed derivation protocol used by this release.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SeedProtocolVersion {
    /// Keyed `BLAKE3` namespaces, little-endian identities, and `ChaCha12` streams.
    V1,
}

/// A caller-supplied 256-bit master seed.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct MasterSeed([u8; 32]);

impl MasterSeed {
    /// Creates a master seed from all 32 bytes.
    #[must_use]
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Creates a master seed from a little-endian `u64`, with remaining bytes zero.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        let encoded = value.to_le_bytes();
        let mut bytes = [0_u8; 32];
        let mut index = 0;
        while index < encoded.len() {
            bytes[index] = encoded[index];
            index += 1;
        }
        Self(bytes)
    }

    /// Returns the exact master-seed bytes.
    #[must_use]
    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Debug for MasterSeed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MasterSeed([REDACTED; 32])")
    }
}

/// A stable, named family of trial randomness.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TrialStream {
    /// Generic evaluator noise.
    TrialNoise,
    /// Layout or initial-condition noise.
    LayoutNoise,
    /// Execution or actuator noise.
    ExecutionNoise,
    /// Independent validation or finalist holdout noise.
    ValidationHoldout,
}

impl TrialStream {
    /// Returns the domain separator for this stream family.
    const fn namespace(self) -> &'static [u8] {
        match self {
            Self::TrialNoise => TRIAL_NOISE_NAMESPACE,
            Self::LayoutNoise => LAYOUT_NOISE_NAMESPACE,
            Self::ExecutionNoise => EXECUTION_NOISE_NAMESPACE,
            Self::ValidationHoldout => HOLDOUT_NAMESPACE,
        }
    }
}

/// A 32-byte derived seed suitable for exact replay.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DerivedSeed([u8; 32]);

impl DerivedSeed {
    /// Returns the exact derived bytes.
    #[must_use]
    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }

    /// Constructs the pinned `ChaCha12` stream for this seed.
    #[must_use]
    pub fn rng(self) -> TrialRng {
        TrialRng(ChaCha12Rng::from_seed(self.0))
    }
}

/// A pinned `ChaCha12` stream with a deliberately small stable interface.
pub struct TrialRng(ChaCha12Rng);

impl TrialRng {
    /// Returns the next uniformly distributed `u64` from the stream.
    pub fn next_u64(&mut self) -> u64 {
        rand::Rng::next_u64(&mut self.0)
    }

    /// Returns a uniform value in the half-open interval `[0, 1)`.
    pub fn uniform_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * U64_TO_UNIT_F64
    }

    /// Fills bytes from the stream.
    pub fn fill_bytes(&mut self, destination: &mut [u8]) {
        rand::Rng::fill_bytes(&mut self.0, destination);
    }
}

/// Stateless V1 seed derivation rooted at one master seed.
#[derive(Clone, Copy, Debug)]
pub struct SeedDeriver {
    /// Root seed used by every derived stream.
    master: MasterSeed,
}

impl SeedDeriver {
    /// Creates a V1 seed deriver.
    #[must_use]
    pub const fn new(master: MasterSeed) -> Self {
        Self { master }
    }

    /// Returns the frozen protocol version.
    #[must_use]
    pub const fn version(self) -> SeedProtocolVersion {
        SeedProtocolVersion::V1
    }

    /// Derives an independent proposal seed for a stable proposal token.
    #[must_use]
    pub fn proposal_seed(self, token: u64) -> DerivedSeed {
        self.derive(PROPOSAL_NAMESPACE, &[&token.to_le_bytes()])
    }

    /// Derives an independent proposal seed for one stable vector dimension.
    #[must_use]
    pub fn proposal_dimension_seed(self, token: u64, dimension: u64) -> DerivedSeed {
        self.derive(
            PROPOSAL_NAMESPACE,
            &[&token.to_le_bytes(), &dimension.to_le_bytes()],
        )
    }

    /// Derives an optimizer-acceptance seed for a generation and step.
    #[must_use]
    pub fn acceptance_seed(self, generation: u64, step: u64) -> DerivedSeed {
        self.derive(
            ACCEPTANCE_NAMESPACE,
            &[&generation.to_le_bytes(), &step.to_le_bytes()],
        )
    }

    /// Derives a candidate-specific or common-random trial stream.
    fn trial_seed(
        self,
        key: TrialId,
        common_random_group: u64,
        stream: TrialStream,
        tag: &str,
        common: bool,
    ) -> DerivedSeed {
        let candidate = key.candidate().value().to_le_bytes();
        let replication = key.replication().value().to_le_bytes();
        let group = common_random_group.to_le_bytes();
        let tag_length = u64::try_from(tag.len()).unwrap_or(u64::MAX).to_le_bytes();
        let scope = if common {
            COMMON_NAMESPACE
        } else {
            INDEPENDENT_NAMESPACE
        };
        if common {
            self.derive(
                scope,
                &[
                    stream.namespace(),
                    &group,
                    &replication,
                    &tag_length,
                    tag.as_bytes(),
                ],
            )
        } else {
            self.derive(
                scope,
                &[
                    stream.namespace(),
                    &candidate,
                    &replication,
                    &tag_length,
                    tag.as_bytes(),
                ],
            )
        }
    }

    /// Hashes length-delimited protocol fields under one domain separator.
    fn derive(self, namespace: &[u8], fields: &[&[u8]]) -> DerivedSeed {
        let mut hasher = Hasher::new_keyed(&self.master.0);
        hasher.update(ROOT_NAMESPACE);
        let namespace_length = u64::try_from(namespace.len()).unwrap_or(u64::MAX);
        hasher.update(&namespace_length.to_le_bytes());
        hasher.update(namespace);
        for field in fields {
            hasher.update(field);
        }
        DerivedSeed(*hasher.finalize().as_bytes())
    }
}

/// Complete information required to recreate every named stream for a trial.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplayKey {
    /// Seed-protocol version used for derivation.
    version: SeedProtocolVersion,
    /// Root seed from which the trial streams are derived.
    master_seed: MasterSeed,
    /// Stable identity of the evaluated replication.
    trial: TrialId,
    /// Group identifying candidates that share common randomness.
    common_random_group: u64,
}

impl ReplayKey {
    /// Returns the protocol version.
    #[must_use]
    pub const fn version(self) -> SeedProtocolVersion {
        self.version
    }

    /// Returns the master seed.
    #[must_use]
    pub const fn master_seed(self) -> MasterSeed {
        self.master_seed
    }

    /// Returns the trial identity.
    #[must_use]
    pub const fn trial(self) -> TrialId {
        self.trial
    }

    /// Returns the common-random comparison group.
    #[must_use]
    pub const fn common_random_group(self) -> u64 {
        self.common_random_group
    }
}

/// Deterministic identity and named randomness supplied to one evaluation.
#[derive(Clone, Copy, Debug)]
pub struct TrialContext {
    /// Replay metadata backing this context.
    replay: ReplayKey,
}

impl TrialContext {
    /// Creates the context for a scheduled trial.
    const fn new(master_seed: MasterSeed, key: TrialId, common_random_group: u64) -> Self {
        Self {
            replay: ReplayKey {
                version: SeedProtocolVersion::V1,
                master_seed,
                trial: key,
                common_random_group,
            },
        }
    }

    /// Returns this trial's stable identity.
    #[must_use]
    pub const fn key(self) -> TrialId {
        self.replay.trial
    }

    /// Returns the complete key needed to replay named random streams.
    #[must_use]
    pub const fn replay_key(self) -> ReplayKey {
        self.replay
    }

    /// Derives a stream unique to this candidate, replication, family, and tag.
    #[must_use]
    pub fn independent_rng(self, stream: TrialStream, tag: &str) -> TrialRng {
        self.independent_seed(stream, tag).rng()
    }

    /// Derives a stream shared by candidates in the same comparison group.
    #[must_use]
    pub fn common_rng(self, stream: TrialStream, tag: &str) -> TrialRng {
        self.common_seed(stream, tag).rng()
    }

    /// Derives an independent stream seed without consuming it.
    #[must_use]
    pub fn independent_seed(self, stream: TrialStream, tag: &str) -> DerivedSeed {
        SeedDeriver::new(self.replay.master_seed).trial_seed(
            self.replay.trial,
            self.replay.common_random_group,
            stream,
            tag,
            false,
        )
    }

    /// Derives a common-random stream seed without consuming it.
    #[must_use]
    pub fn common_seed(self, stream: TrialStream, tag: &str) -> DerivedSeed {
        SeedDeriver::new(self.replay.master_seed).trial_seed(
            self.replay.trial,
            self.replay.common_random_group,
            stream,
            tag,
            true,
        )
    }

    /// Samples the first uniform value from a generic independent named stream.
    #[must_use]
    pub fn sample_independent_f64(self, tag: &str) -> f64 {
        self.independent_rng(TrialStream::TrialNoise, tag)
            .uniform_f64()
    }

    /// Samples the first uniform value from a generic common-random named stream.
    #[must_use]
    pub fn sample_common_f64(self, tag: &str) -> f64 {
        self.common_rng(TrialStream::TrialNoise, tag).uniform_f64()
    }
}

/// One requested candidate replication.
#[derive(Clone, Debug)]
pub struct TrialSpec<C> {
    /// Stable trial identity.
    key: TrialId,
    /// Group identifying candidates that share common randomness.
    common_random_group: u64,
    /// Candidate value supplied to the evaluator.
    candidate: C,
}

impl<C> TrialSpec<C> {
    /// Creates a trial request.
    #[must_use]
    pub const fn new(
        candidate: CandidateId,
        replication: ReplicationId,
        common_random_group: u64,
        value: C,
    ) -> Self {
        Self {
            key: TrialId::new(candidate, replication),
            common_random_group,
            candidate: value,
        }
    }

    /// Returns the trial identity.
    #[must_use]
    pub const fn key(&self) -> TrialId {
        self.key
    }

    /// Returns the candidate value.
    #[must_use]
    pub const fn candidate(&self) -> &C {
        &self.candidate
    }

    /// Returns the common-random comparison group.
    #[must_use]
    pub const fn common_random_group(&self) -> u64 {
        self.common_random_group
    }
}

/// A fixed replication schedule with one frozen common-random group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedReplications {
    /// Number of replications requested per candidate.
    count: u32,
    /// Group identifying candidates that share common randomness.
    common_random_group: u64,
}

impl FixedReplications {
    /// Creates a schedule. A count of zero produces no trials.
    #[must_use]
    pub const fn new(count: u32, common_random_group: u64) -> Self {
        Self {
            count,
            common_random_group,
        }
    }

    /// Returns the exact requested replication count.
    #[must_use]
    pub const fn count(self) -> u32 {
        self.count
    }

    /// Returns the frozen common-random comparison group.
    #[must_use]
    pub const fn common_random_group(self) -> u64 {
        self.common_random_group
    }

    /// Expands a candidate into stable replication IDs in ascending order.
    pub fn trials<C: Clone>(
        self,
        candidate: CandidateId,
        value: C,
    ) -> impl ExactSizeIterator<Item = TrialSpec<C>> {
        (0..self.count).map(move |replication| {
            TrialSpec::new(
                candidate,
                ReplicationId::new(replication),
                self.common_random_group,
                value.clone(),
            )
        })
    }
}

/// The typed result and exact replay identity of one evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrialRecord<O, E> {
    /// Stable trial identity.
    pub key: TrialId,
    /// Exact key for recreating named random streams.
    pub replay: ReplayKey,
    /// Evaluator outcome or typed evaluator failure.
    pub result: Result<O, E>,
}

/// Memory-retention choice applied after canonical reduction.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RetentionPolicy {
    /// Retain no per-trial outcome; summaries remain caller-owned.
    #[default]
    SummariesOnly,
    /// Retain typed failures and their replay keys, but drop successful outcomes.
    FailuresOnly,
    /// Retain every typed outcome and replay key.
    All,
}

/// A retained trial selected by [`retain_records`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetainedTrialRecord<O, E> {
    /// Stable trial identity.
    pub key: TrialId,
    /// Exact replay key.
    pub replay: ReplayKey,
    /// Retained typed outcome or failure.
    pub result: Result<O, E>,
}

/// Applies a retention policy by consuming records without cloning outcomes.
///
/// The default policy returns an empty vector. Callers should first update
/// online summaries, then consume the batch here.
pub fn retain_records<O, E>(
    records: impl IntoIterator<Item = TrialRecord<O, E>>,
    policy: RetentionPolicy,
) -> Vec<RetainedTrialRecord<O, E>> {
    match policy {
        RetentionPolicy::SummariesOnly => Vec::new(),
        RetentionPolicy::FailuresOnly => records
            .into_iter()
            .filter_map(|record| match record.result {
                Ok(_) => None,
                Err(error) => Some(RetainedTrialRecord {
                    key: record.key,
                    replay: record.replay,
                    result: Err(error),
                }),
            })
            .collect(),
        RetentionPolicy::All => records
            .into_iter()
            .map(|record| RetainedTrialRecord {
                key: record.key,
                replay: record.replay,
                result: record.result,
            })
            .collect(),
    }
}

/// Batch validation, allocation, or evaluator-panic failure.
#[derive(Debug)]
pub enum RunError {
    /// A trial identity occurred more than once.
    DuplicateTrial(TrialId),
    /// The supplied iterator exceeded the declared memory bound.
    BatchLimitExceeded {
        /// Maximum permitted retained records.
        limit: usize,
    },
    /// Explicit batch storage could not be allocated.
    Allocation(std::collections::TryReserveError),
    /// The serial evaluator panicked; partial output was discarded.
    WorkloadPanic,
}

impl fmt::Display for RunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateTrial(trial) => write!(formatter, "duplicate trial: {trial:?}"),
            Self::BatchLimitExceeded { limit } => {
                write!(formatter, "batch exceeds the record limit of {limit}")
            }
            Self::Allocation(source) => write!(formatter, "batch allocation failed: {source}"),
            Self::WorkloadPanic => formatter.write_str("serial evaluator panicked"),
        }
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(source) => Some(source),
            Self::DuplicateTrial(_) | Self::BatchLimitExceeded { .. } | Self::WorkloadPanic => None,
        }
    }
}

/// Runs a finite batch serially and returns records in canonical trial order.
///
/// Use [`run_serial_bounded`] for input whose size is not already externally
/// bounded. Evaluator errors remain ordered values and do not cancel the batch.
///
/// # Errors
///
/// Returns an error for duplicate identities, allocation failure, or an evaluator
/// panic when unwinding is enabled. Partial output is discarded without retries.
pub fn run_serial<C, O, E>(
    master_seed: MasterSeed,
    trials: impl IntoIterator<Item = TrialSpec<C>>,
    evaluate: impl FnMut(&C, &TrialContext) -> Result<O, E>,
) -> Result<Vec<TrialRecord<O, E>>, RunError> {
    run_serial_bounded(
        master_seed,
        trials,
        NonZeroUsize::new(usize::MAX).unwrap_or(NonZeroUsize::MIN),
        evaluate,
    )
}

/// Runs a serial batch while enforcing a maximum number of retained records.
///
/// # Errors
///
/// Returns an error if input exceeds `limit`, repeats an identity, or batch
/// storage cannot be allocated. These failures precede evaluation. An evaluator
/// panic is caught when unwinding is enabled and discards partial output.
pub fn run_serial_bounded<C, O, E>(
    master_seed: MasterSeed,
    trials: impl IntoIterator<Item = TrialSpec<C>>,
    limit: NonZeroUsize,
    mut evaluate: impl FnMut(&C, &TrialContext) -> Result<O, E>,
) -> Result<Vec<TrialRecord<O, E>>, RunError> {
    let ordered = collect_canonical(trials, limit)?;
    let mut records = Vec::new();
    records
        .try_reserve_exact(ordered.len())
        .map_err(RunError::Allocation)?;
    catch_callback(|| {
        for trial in ordered {
            let context = TrialContext::new(master_seed, trial.key, trial.common_random_group);
            let result = evaluate(&trial.candidate, &context);
            records.push(TrialRecord {
                key: trial.key,
                replay: context.replay_key(),
                result,
            });
        }
    })
    .map_err(|()| RunError::WorkloadPanic)?;
    Ok(records)
}

/// Collects a bounded batch and validates its allocation-free canonical ordering.
fn collect_canonical<C>(
    trials: impl IntoIterator<Item = TrialSpec<C>>,
    limit: NonZeroUsize,
) -> Result<Vec<TrialSpec<C>>, RunError> {
    let trials = trials.into_iter();
    let mut ordered = Vec::new();
    ordered
        .try_reserve(trials.size_hint().0.min(limit.get()))
        .map_err(RunError::Allocation)?;
    for trial in trials {
        if ordered.len() == limit.get() {
            return Err(RunError::BatchLimitExceeded { limit: limit.get() });
        }
        if ordered.len() == ordered.capacity() {
            ordered.try_reserve(1).map_err(RunError::Allocation)?;
        }
        ordered.push(trial);
    }
    ordered.sort_unstable_by_key(TrialSpec::key);
    ensure_unique(&ordered)?;
    Ok(ordered)
}

/// Contains callback unwinding; caller-owned mutable state is never retried.
fn catch_callback<T>(callback: impl FnOnce() -> T) -> Result<T, ()> {
    #[cfg(panic = "unwind")]
    {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(callback)).map_err(|_| ())
    }
    #[cfg(not(panic = "unwind"))]
    {
        Ok(callback())
    }
}

/// Rejects adjacent duplicate trial identities in a canonically ordered batch.
fn ensure_unique<C>(ordered: &[TrialSpec<C>]) -> Result<(), RunError> {
    for pair in ordered.windows(2) {
        if pair[0].key == pair[1].key {
            return Err(RunError::DuplicateTrial(pair[0].key));
        }
    }
    Ok(())
}

/// Configuration for the optional bounded Rayon executor.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParallelConfig {
    /// Requested worker count, capped by batch size and available parallelism.
    workers: NonZeroUsize,
    /// Maximum number of records retained by one batch.
    batch_limit: NonZeroUsize,
}

#[cfg(feature = "parallel")]
impl ParallelConfig {
    /// Creates a configuration with explicit worker and batch bounds.
    #[must_use]
    pub const fn new(workers: NonZeroUsize, batch_limit: NonZeroUsize) -> Self {
        Self {
            workers,
            batch_limit,
        }
    }

    /// Returns the requested worker count.
    #[must_use]
    pub const fn workers(self) -> NonZeroUsize {
        self.workers
    }

    /// Returns the retained-record limit.
    #[must_use]
    pub const fn batch_limit(self) -> NonZeroUsize {
        self.batch_limit
    }
}

/// Failure to configure or run the optional parallel executor.
#[cfg(feature = "parallel")]
#[derive(Debug)]
pub enum ParallelRunError<WE> {
    /// Batch validation or explicit storage allocation failed before callbacks.
    Batch(RunError),
    /// Rayon could not construct the dedicated worker pool.
    Pool(rayon::ThreadPoolBuildError),
    /// A caller-thread workspace factory returned a typed error.
    WorkerInitialization {
        /// Zero-based contiguous shard identity.
        worker: usize,
        /// Original workspace initialization error.
        source: WE,
    },
    /// A caller-thread workspace factory panicked.
    WorkerFactoryPanic {
        /// Zero-based contiguous shard identity.
        worker: usize,
    },
    /// An evaluator panicked; all shards joined and partial output was discarded.
    WorkerWorkloadPanic {
        /// Lowest zero-based contiguous shard identity whose evaluator panicked.
        worker: usize,
    },
}

#[cfg(feature = "parallel")]
impl<WE: fmt::Display> fmt::Display for ParallelRunError<WE> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Batch(source) => write!(formatter, "invalid parallel batch: {source}"),
            Self::Pool(source) => write!(formatter, "worker pool construction failed: {source}"),
            Self::WorkerInitialization { worker, source } => {
                write!(formatter, "worker {worker} initialization failed: {source}")
            }
            Self::WorkerFactoryPanic { worker } => {
                write!(formatter, "worker {worker} factory panicked")
            }
            Self::WorkerWorkloadPanic { worker } => {
                write!(formatter, "worker {worker} evaluator panicked")
            }
        }
    }
}

#[cfg(feature = "parallel")]
impl<WE: std::error::Error + 'static> std::error::Error for ParallelRunError<WE> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Batch(source) => Some(source),
            Self::Pool(source) => Some(source),
            Self::WorkerInitialization { source, .. } => Some(source),
            Self::WorkerFactoryPanic { .. } | Self::WorkerWorkloadPanic { .. } => None,
        }
    }
}

/// A contiguous canonical input shard with output storage reserved before callbacks.
#[cfg(feature = "parallel")]
struct ParallelShard<'a, C, O, E> {
    /// Borrowed canonical input, never cloned or redistributed during evaluation.
    trials: &'a [TrialSpec<C>],
    /// Guard for the initialized prefix of one disjoint output range.
    records: RecordRange<'a, TrialRecord<O, E>>,
    /// Whether evaluation unwound before the shard completed.
    panicked: bool,
}

/// Owns destruction of partially initialized output until the run succeeds.
#[cfg(feature = "parallel")]
struct RecordRange<'a, T> {
    /// Disjoint portion of the final output vector's spare capacity.
    storage: &'a mut [std::mem::MaybeUninit<T>],
    /// Exactly this prefix contains live values owned by the guard.
    initialized: usize,
}

#[cfg(feature = "parallel")]
impl<T> RecordRange<'_, T> {
    /// Writes a value before taking ownership of its destruction.
    const fn push(&mut self, value: T) {
        self.storage[self.initialized].write(value);
        self.initialized += 1;
    }
}

#[cfg(feature = "parallel")]
impl<T> Drop for RecordRange<'_, T> {
    fn drop(&mut self) {
        for record in &mut self.storage[..self.initialized] {
            // SAFETY: push initializes before extending the exclusively owned
            // prefix. Success disarms this guard before the Vec takes ownership.
            unsafe { record.assume_init_drop() };
        }
    }
}

/// Runs a bounded batch in a dedicated Rayon pool with worker-local state.
///
/// Conforming evaluators must depend only on the candidate, context, immutable
/// configuration, and unobservable worker-local scratch. Output order is always
/// canonical and independent of scheduling or worker count. Each contiguous
/// canonical shard owns exactly one mutable workspace, initialized on the caller
/// thread. All storage and the dedicated pool are prepared before factories run;
/// all factories succeed before any evaluation begins. Empty input needs neither.
///
/// # Errors
///
/// Returns typed batch, pool, or workspace-initialization failures before any
/// evaluations. When unwinding is enabled, factory and evaluator panics become
/// typed errors. Evaluator panics join all shards, select the lowest failing shard
/// identity, and discard partial output without retrying callbacks.
#[cfg(feature = "parallel")]
pub fn run_parallel<C, O, E, W, WE>(
    master_seed: MasterSeed,
    trials: impl IntoIterator<Item = TrialSpec<C>>,
    config: ParallelConfig,
    mut init_worker: impl FnMut() -> Result<W, WE>,
    evaluate: impl Fn(&mut W, &C, &TrialContext) -> Result<O, E> + Send + Sync,
) -> Result<Vec<TrialRecord<O, E>>, ParallelRunError<WE>>
where
    C: Send + Sync,
    O: Send,
    E: Send,
    W: Send,
{
    use rayon::prelude::*;

    let ordered = collect_canonical(trials, config.batch_limit).map_err(ParallelRunError::Batch)?;
    let mut records = Vec::new();
    if ordered.is_empty() {
        return Ok(records);
    }
    records
        .try_reserve_exact(ordered.len())
        .map_err(|source| ParallelRunError::Batch(RunError::Allocation(source)))?;

    let available = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let workers = config.workers.get().min(ordered.len()).min(available.get());
    let mut workspaces = Vec::new();
    workspaces
        .try_reserve_exact(workers)
        .map_err(|source| ParallelRunError::Batch(RunError::Allocation(source)))?;
    let mut shards = Vec::new();
    shards
        .try_reserve_exact(workers)
        .map_err(|source| ParallelRunError::Batch(RunError::Allocation(source)))?;

    let base_size = ordered.len().div_euclid(workers);
    let remainder = ordered.len().rem_euclid(workers);
    let mut remaining = ordered.as_slice();
    let mut remaining_output = &mut records.spare_capacity_mut()[..ordered.len()];
    for worker in 0..workers {
        let size = base_size + usize::from(worker < remainder);
        let (trials, rest) = remaining.split_at(size);
        remaining = rest;
        let (storage, rest) = remaining_output.split_at_mut(size);
        remaining_output = rest;
        shards.push(ParallelShard {
            trials,
            records: RecordRange {
                storage,
                initialized: 0,
            },
            panicked: false,
        });
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(ParallelRunError::Pool)?;
    for worker in 0..workers {
        let workspace = catch_callback(&mut init_worker)
            .map_err(|()| ParallelRunError::WorkerFactoryPanic { worker })?
            .map_err(|source| ParallelRunError::WorkerInitialization { worker, source })?;
        workspaces.push(workspace);
    }

    pool.install(|| {
        shards
            .par_iter_mut()
            .zip(workspaces.par_iter_mut())
            .for_each(|(shard, workspace)| {
                shard.panicked = catch_callback(|| {
                    for trial in shard.trials {
                        let context =
                            TrialContext::new(master_seed, trial.key, trial.common_random_group);
                        let result = evaluate(workspace, &trial.candidate, &context);
                        shard.records.push(TrialRecord {
                            key: trial.key,
                            replay: context.replay_key(),
                            result,
                        });
                    }
                })
                .is_err();
            });
    });
    if let Some(worker) = shards.iter().position(|shard| shard.panicked) {
        return Err(ParallelRunError::WorkerWorkloadPanic { worker });
    }
    for shard in &mut shards {
        shard.records.initialized = 0;
    }
    drop(shards);
    // SAFETY: successful evaluation initialized every element in the disjoint
    // ranges covering ordered.len() reserved slots, and all parallel work joined.
    // Guards are disarmed and no mutable slices remain; records is sole owner.
    unsafe { records.set_len(ordered.len()) };
    Ok(records)
}

/// A finite floating-point value accepted for ranking or replacement policies.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct FiniteValue(f64);

impl FiniteValue {
    /// Validates and wraps a finite value.
    ///
    /// # Errors
    ///
    /// Returns [`NonFiniteValue`] when `value` is NaN or infinite.
    pub const fn new(value: f64) -> Result<Self, NonFiniteValue> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(NonFiniteValue { value })
        }
    }

    /// Returns the wrapped finite value.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// A rejected non-finite floating-point observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NonFiniteValue {
    /// The rejected value.
    pub value: f64,
}

/// Reducer failure caused by an empty or invalid observation set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReduceError {
    /// At least one observation is required.
    Empty,
    /// A non-finite scalar was supplied.
    NonFinite(NonFiniteValue),
    /// A scalar update exceeded the representable count or finite arithmetic range.
    ArithmeticOverflow,
    /// Wilson's positive finite z-score was invalid.
    InvalidWilsonZ(f64),
}

/// Online Bernoulli reducer with Wilson-score confidence intervals.
#[derive(Clone, Copy, Debug)]
pub struct BernoulliReducer {
    /// Number of observations processed.
    attempts: u64,
    /// Number of successful observations.
    successes: u64,
    /// Normal-distribution z-score used for the confidence interval.
    wilson_z: f64,
}

impl Default for BernoulliReducer {
    fn default() -> Self {
        Self {
            attempts: 0,
            successes: 0,
            wilson_z: 1.959_963_984_540_054,
        }
    }
}

impl BernoulliReducer {
    /// Creates a reducer using a positive finite normal z-score.
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::InvalidWilsonZ`] when `wilson_z` is non-positive
    /// or non-finite.
    pub fn new(wilson_z: f64) -> Result<Self, ReduceError> {
        if !wilson_z.is_finite() || wilson_z <= 0.0 {
            return Err(ReduceError::InvalidWilsonZ(wilson_z));
        }
        Ok(Self {
            attempts: 0,
            successes: 0,
            wilson_z,
        })
    }

    /// Observes one success or failure without allocation.
    pub const fn observe(&mut self, success: bool) {
        self.attempts += 1;
        self.successes += success as u64;
    }

    /// Reduces an iterator using the default 95% Wilson interval.
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::Empty`] when `results` contains no observations.
    pub fn from_results(
        results: impl IntoIterator<Item = bool>,
    ) -> Result<BernoulliSummary, ReduceError> {
        let mut reducer = Self::default();
        for result in results {
            reducer.observe(result);
        }
        reducer.finish()
    }

    /// Finishes the reduction; an empty reducer is an explicit error.
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::Empty`] when no observations were recorded.
    pub fn finish(self) -> Result<BernoulliSummary, ReduceError> {
        if self.attempts == 0 {
            return Err(ReduceError::Empty);
        }
        let n = self.attempts as f64;
        let rate = self.successes as f64 / n;
        let z2 = self.wilson_z * self.wilson_z;
        let denominator = 1.0 + z2 / n;
        let center = (rate + z2 / (2.0 * n)) / denominator;
        let radius =
            self.wilson_z * ((rate * (1.0 - rate) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
        Ok(BernoulliSummary {
            attempts: self.attempts,
            successes: self.successes,
            rate,
            wilson_lower: (center - radius).max(0.0),
            wilson_upper: (center + radius).min(1.0),
        })
    }
}

/// Final Bernoulli statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BernoulliSummary {
    /// Number of observations.
    pub attempts: u64,
    /// Number of successful observations.
    pub successes: u64,
    /// Maximum-likelihood success probability.
    pub rate: f64,
    /// Lower Wilson-score confidence bound.
    pub wilson_lower: f64,
    /// Upper Wilson-score confidence bound.
    pub wilson_upper: f64,
}

/// Online finite-scalar reducer using Welford's stable recurrence.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarReducer {
    /// Number of observations processed.
    count: u64,
    /// Running arithmetic mean.
    mean: f64,
    /// Running sum of squared deviations from the mean.
    sum_squared_deviation: f64,
    /// Minimum observation.
    min: f64,
    /// Maximum observation.
    max: f64,
}

impl ScalarReducer {
    /// Observes one finite value without retaining prior values.
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::NonFinite`] when `value` is NaN or infinite.
    /// Returns [`ReduceError::ArithmeticOverflow`] when the count or Welford
    /// recurrence cannot be represented. Rejected observations leave the reducer
    /// unchanged.
    //
    // Keep Welford's multiply and addition separate: contraction changes the
    // established floating-point reduction results.
    #[allow(clippy::suboptimal_flops)]
    pub fn observe(&mut self, value: f64) -> Result<(), ReduceError> {
        let finite = FiniteValue::new(value).map_err(ReduceError::NonFinite)?;
        let value = finite.get();
        let count = self
            .count
            .checked_add(1)
            .ok_or(ReduceError::ArithmeticOverflow)?;
        if count == 1 {
            self.count = count;
            self.mean = value;
            self.min = value;
            self.max = value;
            return Ok(());
        }
        let delta = value - self.mean;
        let mean = self.mean + delta / count as f64;
        let delta_after = value - mean;
        let sum_squared_deviation = self.sum_squared_deviation + delta * delta_after;
        if !delta.is_finite()
            || !mean.is_finite()
            || !delta_after.is_finite()
            || !sum_squared_deviation.is_finite()
            || sum_squared_deviation < 0.0
        {
            return Err(ReduceError::ArithmeticOverflow);
        }
        self.count = count;
        self.mean = mean;
        self.sum_squared_deviation = sum_squared_deviation;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        Ok(())
    }

    /// Reduces finite scalar observations.
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::NonFinite`] for a non-finite observation,
    /// [`ReduceError::ArithmeticOverflow`] for an unrepresentable update, or
    /// [`ReduceError::Empty`] when `results` contains no observations.
    pub fn from_results(
        results: impl IntoIterator<Item = f64>,
    ) -> Result<ScalarSummary, ReduceError> {
        let mut reducer = Self::default();
        for result in results {
            reducer.observe(result)?;
        }
        reducer.finish()
    }

    /// Finishes the reduction; variance and standard error are absent for one sample.
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::Empty`] when no observations were recorded.
    pub fn finish(self) -> Result<ScalarSummary, ReduceError> {
        if self.count == 0 {
            return Err(ReduceError::Empty);
        }
        let sample_variance =
            (self.count > 1).then(|| self.sum_squared_deviation / (self.count - 1) as f64);
        let standard_error = sample_variance.map(|variance| (variance / self.count as f64).sqrt());
        Ok(ScalarSummary {
            count: self.count,
            mean: self.mean,
            sample_variance,
            standard_error,
            min: self.min,
            max: self.max,
        })
    }
}

/// Final finite-scalar statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarSummary {
    /// Number of observations.
    pub count: u64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Unbiased sample variance, absent for one observation.
    pub sample_variance: Option<f64>,
    /// Standard error of the mean, absent for one observation.
    pub standard_error: Option<f64>,
    /// Minimum observation.
    pub min: f64,
    /// Maximum observation.
    pub max: f64,
}

/// Explicit treatment of typed evaluator failures during aggregation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailurePolicy<T> {
    /// Make the aggregate ineligible when any evaluation fails.
    RejectCandidate,
    /// Exclude failed evaluations while retaining their count.
    Ignore,
    /// Convert each failed evaluation to this declared observation.
    ReplaceWith(T),
}

/// Explicit treatment of non-finite scalar observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NonFinitePolicy {
    /// Make the aggregate ineligible.
    RejectCandidate,
    /// Exclude non-finite observations while retaining requested counts.
    Ignore,
    /// Replace each non-finite observation with a finite declared penalty.
    ReplaceWith(FiniteValue),
}

/// Why a candidate could not produce an eligible aggregate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AggregateError {
    /// No records were supplied.
    Empty,
    /// Records from more than one candidate were mixed.
    MixedCandidates,
    /// Record count cannot be represented by the public count fields.
    TooManyRecords,
    /// At least one typed evaluation failed under reject policy.
    EvaluationFailed {
        /// Number of typed evaluation failures.
        failures: u32,
    },
    /// At least one scalar was non-finite under reject policy.
    NonFinite {
        /// Number of non-finite successful evaluations.
        observations: u32,
    },
    /// Every observation was excluded by policy.
    NoUsableObservations,
    /// Accepted scalar observations overflowed the reducer's arithmetic.
    ArithmeticOverflow,
}

/// Compact per-candidate counts and typed statistics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateSummary<S> {
    /// Candidate identity.
    pub candidate: CandidateId,
    /// Number of requested records.
    pub requested: u32,
    /// Number of successful evaluator calls.
    pub completed: u32,
    /// Number of typed evaluator failures.
    pub failed: u32,
    /// Reduced statistics over accepted or explicitly replaced observations.
    pub statistics: S,
}

/// Aggregates typed Bernoulli records with an explicit failure policy.
///
/// # Errors
///
/// Returns [`AggregateError`] when the records are empty, mix candidates,
/// exceed public count capacity, contain rejected failures, or leave no usable
/// observations.
pub fn aggregate_bernoulli<O, E>(
    records: &[TrialRecord<O, E>],
    classify: impl Fn(&O) -> bool,
    failure_policy: FailurePolicy<bool>,
) -> Result<CandidateSummary<BernoulliSummary>, AggregateError> {
    let candidate = validate_candidate_records(records)?;
    let requested = u32::try_from(records.len()).map_err(|_| AggregateError::TooManyRecords)?;
    let mut reducer = BernoulliReducer::default();
    let mut completed = 0_u32;
    let mut failed = 0_u32;
    for record in records {
        if let Ok(outcome) = &record.result {
            completed += 1;
            reducer.observe(classify(outcome));
        } else {
            failed += 1;
            if let FailurePolicy::ReplaceWith(value) = failure_policy {
                reducer.observe(value);
            }
        }
    }
    if failed > 0 && failure_policy == FailurePolicy::RejectCandidate {
        return Err(AggregateError::EvaluationFailed { failures: failed });
    }
    let statistics = reducer
        .finish()
        .map_err(|_| AggregateError::NoUsableObservations)?;
    Ok(CandidateSummary {
        candidate,
        requested,
        completed,
        failed,
        statistics,
    })
}

/// Aggregates typed finite-scalar records with explicit error policies.
///
/// # Errors
///
/// Returns [`AggregateError`] when the records are empty, mix candidates,
/// exceed public count capacity, contain observations rejected by either
/// policy, leave no usable observations, or overflow the scalar recurrence.
/// Arithmetic overflow is an error under every non-finite policy; rejected
/// evaluator failures and non-finite inputs retain precedence over it.
pub fn aggregate_scalar<O, E>(
    records: &[TrialRecord<O, E>],
    score: impl Fn(&O) -> f64,
    failure_policy: FailurePolicy<FiniteValue>,
    non_finite_policy: NonFinitePolicy,
) -> Result<CandidateSummary<ScalarSummary>, AggregateError> {
    let candidate = validate_candidate_records(records)?;
    let requested = u32::try_from(records.len()).map_err(|_| AggregateError::TooManyRecords)?;
    let mut reducer = ScalarReducer::default();
    let mut completed = 0_u32;
    let mut failed = 0_u32;
    let mut non_finite = 0_u32;
    let mut reduction = Ok(());
    for record in records {
        let value = record.result.as_ref().map_or_else(
            |_| {
                failed += 1;
                match failure_policy {
                    FailurePolicy::RejectCandidate | FailurePolicy::Ignore => None,
                    FailurePolicy::ReplaceWith(value) => Some(value),
                }
            },
            |outcome| {
                completed += 1;
                FiniteValue::new(score(outcome)).map_or_else(
                    |_| {
                        non_finite += 1;
                        match non_finite_policy {
                            NonFinitePolicy::RejectCandidate | NonFinitePolicy::Ignore => None,
                            NonFinitePolicy::ReplaceWith(value) => Some(value),
                        }
                    },
                    Some,
                )
            },
        );
        if let Some(value) = value {
            reduction = reduction.and_then(|()| reducer.observe(value.get()));
        }
    }
    if failed > 0 && matches!(failure_policy, FailurePolicy::RejectCandidate) {
        return Err(AggregateError::EvaluationFailed { failures: failed });
    }
    if non_finite > 0 && non_finite_policy == NonFinitePolicy::RejectCandidate {
        return Err(AggregateError::NonFinite {
            observations: non_finite,
        });
    }
    let statistics = reduction.and_then(|()| reducer.finish()).map_err(|error| {
        if error == ReduceError::Empty {
            AggregateError::NoUsableObservations
        } else {
            // Inputs were validated above; only arithmetic can reject an update.
            AggregateError::ArithmeticOverflow
        }
    })?;
    Ok(CandidateSummary {
        candidate,
        requested,
        completed,
        failed,
        statistics,
    })
}

/// Validates that a non-empty record slice belongs to one candidate.
fn validate_candidate_records<O, E>(
    records: &[TrialRecord<O, E>],
) -> Result<CandidateId, AggregateError> {
    let Some(first) = records.first() else {
        return Err(AggregateError::Empty);
    };
    let candidate = first.key.candidate();
    if records
        .iter()
        .any(|record| record.key.candidate() != candidate)
    {
        return Err(AggregateError::MixedCandidates);
    }
    Ok(candidate)
}

/// Typed comparison policy used by deterministic candidate selection.
pub trait SelectionPolicy<A> {
    /// Returns `Greater` when `left` is preferred over `right`.
    fn compare(&self, left: &A, right: &A) -> Ordering;
}

/// Selects the best aggregate, breaking exact policy ties by lower candidate ID.
#[must_use]
pub fn select_best<'a, A>(
    candidates: &'a [(CandidateId, A)],
    policy: &impl SelectionPolicy<A>,
) -> Option<(CandidateId, &'a A)> {
    candidates.iter().fold(None, |best, (id, aggregate)| {
        let Some((best_id, best_aggregate)) = best else {
            return Some((*id, aggregate));
        };
        match policy.compare(aggregate, best_aggregate) {
            Ordering::Greater => Some((*id, aggregate)),
            Ordering::Equal if *id < best_id => Some((*id, aggregate)),
            Ordering::Equal | Ordering::Less => Some((best_id, best_aggregate)),
        }
    })
}

/// Maximizes a validated finite scalar.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaximizeFinite;

impl SelectionPolicy<FiniteValue> for MaximizeFinite {
    fn compare(&self, left: &FiniteValue, right: &FiniteValue) -> Ordering {
        left.partial_cmp(right).unwrap_or(Ordering::Equal)
    }
}

/// Maximizes Bernoulli success rate.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaximizeBernoulliRate;

impl SelectionPolicy<BernoulliSummary> for MaximizeBernoulliRate {
    fn compare(&self, left: &BernoulliSummary, right: &BernoulliSummary) -> Ordering {
        left.rate
            .partial_cmp(&right.rate)
            .unwrap_or(Ordering::Equal)
    }
}

/// Maximizes the lower Wilson confidence bound.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaximizeWilsonLowerBound;

impl SelectionPolicy<BernoulliSummary> for MaximizeWilsonLowerBound {
    fn compare(&self, left: &BernoulliSummary, right: &BernoulliSummary) -> Ordering {
        left.wilson_lower
            .partial_cmp(&right.wilson_lower)
            .unwrap_or(Ordering::Equal)
    }
}

/// A constraint-aware aggregate for feasible-first ranking.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Feasible<A, V> {
    /// A feasible aggregate eligible for objective comparison.
    Feasible(A),
    /// An infeasible aggregate carrying a typed violation.
    Infeasible(V),
}

/// Applies objective and violation policies with unconditional feasible-first precedence.
#[derive(Clone, Copy, Debug)]
pub struct FeasibleFirst<P, Q> {
    /// Policy used between feasible objective aggregates.
    pub objective: P,
    /// Policy used between infeasible violation aggregates.
    pub violation: Q,
}

impl<A, V, P, Q> SelectionPolicy<Feasible<A, V>> for FeasibleFirst<P, Q>
where
    P: SelectionPolicy<A>,
    Q: SelectionPolicy<V>,
{
    fn compare(&self, left: &Feasible<A, V>, right: &Feasible<A, V>) -> Ordering {
        match (left, right) {
            (Feasible::Feasible(left), Feasible::Feasible(right)) => {
                self.objective.compare(left, right)
            }
            (Feasible::Feasible(_), Feasible::Infeasible(_)) => Ordering::Greater,
            (Feasible::Infeasible(_), Feasible::Feasible(_)) => Ordering::Less,
            (Feasible::Infeasible(left), Feasible::Infeasible(right)) => {
                self.violation.compare(left, right)
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::missing_docs_in_private_items)]
mod tests {
    use super::*;
    use std::fmt::Write as _;

    fn seed_hex(seed: DerivedSeed) -> String {
        let mut hex = String::with_capacity(64);
        for byte in seed.bytes() {
            write!(&mut hex, "{byte:02x}").expect("writing to a String should be infallible");
        }
        hex
    }

    fn assert_f64_bits_eq(actual: f64, expected: f64) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    fn assert_f64_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-15,
            "expected {expected:.17}, got {actual:.17}"
        );
    }

    fn context(master: MasterSeed, candidate: u64, replication: u32, group: u64) -> TrialContext {
        TrialContext::new(
            master,
            TrialId::new(CandidateId::new(candidate), ReplicationId::new(replication)),
            group,
        )
    }

    #[test]
    fn v1_seed_vectors_are_frozen() {
        let master = MasterSeed::from_u64(0x0123_4567_89ab_cdef);
        let deriver = SeedDeriver::new(master);
        let trial = context(master, 17, 23, 99);
        let vectors = [
            seed_hex(deriver.proposal_seed(5)),
            seed_hex(deriver.acceptance_seed(7, 11)),
            seed_hex(trial.independent_seed(TrialStream::ExecutionNoise, "aim")),
            seed_hex(trial.common_seed(TrialStream::ExecutionNoise, "aim")),
            seed_hex(trial.independent_seed(TrialStream::LayoutNoise, "object-a-x")),
            seed_hex(trial.independent_seed(TrialStream::ValidationHoldout, "aim")),
        ];
        assert_eq!(
            [
                "5120e76b71a9dce5818abc500e97e2fea8255c653dbda13e9faccbe8792eb46b",
                "33c2bbd693b06c7c2b00dabc21df8deadc54beee50cb0c83c76ee74485953342",
                "924f56bac4caec9a641643734daf9415de7c0b92e25559bc4b1db0d1c4471521",
                "df7f9525e41471699734d85ae21acd50305b3ec8814ce422fb9b669e6594b6af",
                "6302bbefbd0a99a1cd8a268a43d4d09b14a6254c8ec0b71655a06ac681b82c07",
                "e7afea42c51a0e8bda0ff50b9f769e447ddce59374ab45735005836640fde0f8",
            ],
            vectors
        );
    }

    #[test]
    fn common_random_streams_exclude_candidate_identity() {
        let master = MasterSeed::from_u64(42);
        let left = context(master, 1, 7, 55);
        let right = context(master, 999, 7, 55);
        assert_eq!(
            left.common_seed(TrialStream::ExecutionNoise, "heading"),
            right.common_seed(TrialStream::ExecutionNoise, "heading")
        );
        assert_ne!(
            left.independent_seed(TrialStream::ExecutionNoise, "heading"),
            right.independent_seed(TrialStream::ExecutionNoise, "heading")
        );
        assert_ne!(
            left.common_seed(TrialStream::ExecutionNoise, "heading"),
            left.common_seed(TrialStream::ExecutionNoise, "speed")
        );
        assert_ne!(
            left.common_seed(TrialStream::LayoutNoise, "heading"),
            left.common_seed(TrialStream::ExecutionNoise, "heading")
        );
    }

    #[test]
    fn serial_execution_is_canonical_and_keeps_typed_failures() {
        let specs = [
            TrialSpec::new(CandidateId::new(2), ReplicationId::new(1), 0, 20_i32),
            TrialSpec::new(CandidateId::new(1), ReplicationId::new(1), 0, -10_i32),
            TrialSpec::new(CandidateId::new(1), ReplicationId::new(0), 0, 10_i32),
        ];
        let records = run_serial(MasterSeed::from_u64(3), specs, |value, _| {
            if *value < 0 {
                Err("negative")
            } else {
                Ok(*value * 2)
            }
        })
        .expect("valid unique fixture trials should run serially");
        assert_eq!(
            vec![
                TrialId::new(CandidateId::new(1), ReplicationId::new(0)),
                TrialId::new(CandidateId::new(1), ReplicationId::new(1)),
                TrialId::new(CandidateId::new(2), ReplicationId::new(1)),
            ],
            records.iter().map(|record| record.key).collect::<Vec<_>>()
        );
        assert_eq!(Ok(20), records[0].result);
        assert_eq!(Err("negative"), records[1].result);
        assert_eq!(Ok(40), records[2].result);
    }

    #[test]
    fn duplicate_and_bounded_batches_fail_before_evaluation() {
        let duplicate = [
            TrialSpec::new(CandidateId::new(1), ReplicationId::new(0), 0, ()),
            TrialSpec::new(CandidateId::new(1), ReplicationId::new(0), 1, ()),
        ];
        let duplicate_result =
            run_serial(MasterSeed::from_u64(0), duplicate, |(), _| Ok::<_, ()>(()));
        assert!(matches!(
            duplicate_result,
            Err(RunError::DuplicateTrial(trial))
                if trial == TrialId::new(CandidateId::new(1), ReplicationId::new(0))
        ));

        let bounded = [
            TrialSpec::new(CandidateId::new(1), ReplicationId::new(0), 0, ()),
            TrialSpec::new(CandidateId::new(1), ReplicationId::new(1), 0, ()),
        ];
        let bounded_result = run_serial_bounded(
            MasterSeed::from_u64(0),
            bounded,
            NonZeroUsize::MIN,
            |(), _| Ok::<_, ()>(()),
        );
        assert!(matches!(
            bounded_result,
            Err(RunError::BatchLimitExceeded { limit: 1 })
        ));
    }

    #[test]
    fn online_reducers_match_small_fixtures() {
        let bernoulli = BernoulliReducer::from_results([true, true, true, false])
            .expect("non-empty Bernoulli fixture should reduce");
        assert_eq!(4, bernoulli.attempts);
        assert_eq!(3, bernoulli.successes);
        assert_f64_bits_eq(bernoulli.rate, 0.75);
        assert!((bernoulli.wilson_lower - 0.300_641_842_582_401_8).abs() < 1e-15);
        assert!((bernoulli.wilson_upper - 0.954_412_739_190_299_5).abs() < 1e-15);

        let scalar = ScalarReducer::from_results([1.0, 2.0, 3.0, 4.0])
            .expect("finite scalar fixture should reduce");
        assert_eq!(4, scalar.count);
        assert_f64_bits_eq(scalar.mean, 2.5);
        assert_f64_close(
            scalar
                .sample_variance
                .expect("multiple observations should have sample variance"),
            5.0 / 3.0,
        );
        assert_f64_close(
            scalar
                .standard_error
                .expect("multiple observations should have a standard error"),
            (5.0_f64 / 12.0).sqrt(),
        );
        assert_f64_bits_eq(scalar.min, 1.0);
        assert_f64_bits_eq(scalar.max, 4.0);
        let one = ScalarReducer::from_results([5.0]).expect("single finite scalar should reduce");
        assert_eq!(None, one.sample_variance);
        assert_eq!(None, one.standard_error);
        assert_eq!(Err(ReduceError::Empty), ScalarReducer::from_results([]));
    }

    #[test]
    fn scalar_reducer_rejects_unrepresentable_statistics() {
        assert_eq!(
            Err(ReduceError::ArithmeticOverflow),
            ScalarReducer::from_results([f64::MAX, -f64::MAX])
        );
        // The mean and deltas fit, but the squared deviation does not.
        assert_eq!(
            Err(ReduceError::ArithmeticOverflow),
            ScalarReducer::from_results([0.0, f64::MAX])
        );
    }

    #[test]
    fn scalar_reducer_failed_update_preserves_accepted_observations() {
        let mut reducer = ScalarReducer::default();
        reducer.observe(1.0).expect("finite observation should fit");
        reducer.observe(3.0).expect("finite observation should fit");
        let before = reducer
            .finish()
            .expect("accepted observations should reduce");
        assert_eq!(
            Err(ReduceError::ArithmeticOverflow),
            reducer.observe(f64::MAX)
        );
        assert_eq!(reducer.finish(), Ok(before));
        reducer.observe(5.0).expect("reducer should remain usable");
        assert_eq!(
            reducer.finish(),
            ScalarReducer::from_results([1.0, 3.0, 5.0])
        );
    }

    #[test]
    fn scalar_aggregation_reports_overflow_independently_of_non_finite_policy() {
        let master = MasterSeed::from_u64(9);
        let records = [f64::MAX, -f64::MAX].map(|value| TrialRecord {
            key: TrialId::new(CandidateId::new(1), ReplicationId::new(0)),
            replay: context(master, 1, 0, 0).replay_key(),
            result: Ok::<f64, &str>(value),
        });
        for policy in [NonFinitePolicy::Ignore, NonFinitePolicy::RejectCandidate] {
            assert_eq!(
                Err(AggregateError::ArithmeticOverflow),
                aggregate_scalar(&records, |value| *value, FailurePolicy::Ignore, policy)
            );
        }
    }

    #[test]
    fn scalar_aggregation_rejection_precedes_arithmetic_overflow() {
        let master = MasterSeed::from_u64(9);
        let records =
            [Ok(f64::MAX), Ok(-f64::MAX), Ok(f64::NAN), Err("physics")].map(|result| TrialRecord {
                key: TrialId::new(CandidateId::new(1), ReplicationId::new(0)),
                replay: context(master, 1, 0, 0).replay_key(),
                result,
            });
        assert_eq!(
            Err(AggregateError::EvaluationFailed { failures: 1 }),
            aggregate_scalar(
                &records,
                |value| *value,
                FailurePolicy::RejectCandidate,
                NonFinitePolicy::RejectCandidate,
            )
        );
        assert_eq!(
            Err(AggregateError::NonFinite { observations: 1 }),
            aggregate_scalar(
                &records,
                |value| *value,
                FailurePolicy::Ignore,
                NonFinitePolicy::RejectCandidate,
            )
        );
        assert_eq!(
            aggregate_scalar(
                &records,
                |value| *value,
                FailurePolicy::Ignore,
                NonFinitePolicy::Ignore,
            ),
            Err(AggregateError::ArithmeticOverflow)
        );
    }

    #[test]
    fn scalar_failure_and_non_finite_policies_are_explicit() {
        let master = MasterSeed::from_u64(9);
        let records = vec![
            TrialRecord {
                key: TrialId::new(CandidateId::new(1), ReplicationId::new(0)),
                replay: context(master, 1, 0, 0).replay_key(),
                result: Ok::<f64, &str>(1.0),
            },
            TrialRecord {
                key: TrialId::new(CandidateId::new(1), ReplicationId::new(1)),
                replay: context(master, 1, 1, 0).replay_key(),
                result: Ok(f64::NAN),
            },
            TrialRecord {
                key: TrialId::new(CandidateId::new(1), ReplicationId::new(2)),
                replay: context(master, 1, 2, 0).replay_key(),
                result: Err("physics"),
            },
        ];
        assert_eq!(
            aggregate_scalar(
                &records,
                |value| *value,
                FailurePolicy::RejectCandidate,
                NonFinitePolicy::RejectCandidate,
            ),
            Err(AggregateError::EvaluationFailed { failures: 1 })
        );
        let penalty = FiniteValue::new(-3.0).expect("fixture penalty should be finite");
        let summary = aggregate_scalar(
            &records,
            |value| *value,
            FailurePolicy::ReplaceWith(penalty),
            NonFinitePolicy::Ignore,
        )
        .expect("replacement policy should produce a scalar summary");
        assert_eq!(summary.requested, 3);
        assert_eq!(summary.completed, 2);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.statistics.count, 2);
        assert_f64_bits_eq(summary.statistics.mean, -1.0);

        assert_eq!(
            aggregate_scalar(
                &records[..2],
                |value| *value,
                FailurePolicy::Ignore,
                NonFinitePolicy::RejectCandidate,
            ),
            Err(AggregateError::NonFinite { observations: 1 })
        );
    }

    #[test]
    fn exact_selection_ties_use_lower_candidate_identity() {
        let score = FiniteValue::new(2.0).expect("fixture score should be finite");
        let candidates = [
            (CandidateId::new(10), score),
            (CandidateId::new(2), score),
            (
                CandidateId::new(8),
                FiniteValue::new(1.0).expect("fixture score should be finite"),
            ),
        ];
        let selected =
            select_best(&candidates, &MaximizeFinite).expect("candidates should have a winner");
        assert_eq!(selected.0, CandidateId::new(2));
        assert_f64_bits_eq(selected.1.get(), score.get());
    }

    #[test]
    fn fixed_schedule_and_default_retention_are_bounded() {
        assert!(FixedReplications::new(0, 7)
            .trials(CandidateId::new(4), 12_i32)
            .next()
            .is_none());

        let records = run_serial(
            MasterSeed::from_u64(2),
            FixedReplications::new(3, 7).trials(CandidateId::new(4), 12_i32),
            |_, context| {
                if context.key().replication() == ReplicationId::new(1) {
                    Err("failed")
                } else {
                    Ok(12)
                }
            },
        )
        .expect("valid fixed schedule should run serially");
        assert!(retain_records(records.clone(), RetentionPolicy::default()).is_empty());
        let failures = retain_records(records, RetentionPolicy::FailuresOnly);
        assert_eq!(failures.len(), 1);
        assert_eq!(
            failures[0].key,
            TrialId::new(CandidateId::new(4), ReplicationId::new(1))
        );
        assert_eq!(failures[0].result, Err("failed"));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn serial_and_parallel_records_are_exactly_equal() {
        fn specs() -> Vec<TrialSpec<u64>> {
            (0..16)
                .rev()
                .map(|replication| {
                    TrialSpec::new(
                        CandidateId::new(u64::from(replication % 3)),
                        ReplicationId::new(replication),
                        77,
                        u64::from(replication),
                    )
                })
                .collect()
        }

        let seed = MasterSeed::from_u64(1234);
        let serial = run_serial(seed, specs(), |value, context| {
            Ok::<_, ()>((*value, context.sample_common_f64("noise")))
        })
        .expect("valid fixture trials should run serially");
        for workers in [1, 4] {
            let parallel = run_parallel(
                seed,
                specs(),
                ParallelConfig::new(
                    NonZeroUsize::new(workers).expect("worker fixture values are nonzero"),
                    NonZeroUsize::new(16).expect("batch-limit fixture is nonzero"),
                ),
                || Ok::<_, ()>(Vec::<u8>::new()),
                |scratch, value, context| {
                    scratch.push(0);
                    Ok::<_, ()>((*value, context.sample_common_f64("noise")))
                },
            )
            .expect("valid fixture trials should run in parallel");
            let parallel_records = parallel.iter().map(|record| {
                (
                    record.key,
                    record.replay,
                    record
                        .result
                        .as_ref()
                        .map(|(value, sample)| (*value, sample.to_bits())),
                )
            });
            let serial_records = serial.iter().map(|record| {
                (
                    record.key,
                    record.replay,
                    record
                        .result
                        .as_ref()
                        .map(|(value, sample)| (*value, sample.to_bits())),
                )
            });
            assert!(parallel_records.eq(serial_records));
        }
    }
}
