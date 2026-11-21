//! Tokenized deterministic ask/tell search strategies.

use super::{CandidateId, CandidateSummary, MasterSeed, SeedDeriver, SelectionPolicy, TrialRng};
use rand::RngExt as _;
use rustc_hash::{FxBuildHasher, FxHashSet};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt;

/// Largest tell batch checked for repeated identities by scanning earlier
/// entries, which avoids allocating and hashing two sets per call.
const SCANNED_TELL_BATCH: usize = 16;

/// Stable identity returned with a proposed candidate.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProposalToken(u64);

impl ProposalToken {
    /// Creates a token from its stable integer representation.
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

/// A tokenized candidate awaiting evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Asked<C> {
    /// Token that must accompany the corresponding `tell`.
    pub token: ProposalToken,
    /// Proposed candidate value.
    pub candidate: C,
}

/// An evaluated candidate returned to a strategy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Told<S> {
    /// Token originally returned by `ask`.
    pub token: ProposalToken,
    /// Stable candidate identity assigned by the experiment runner.
    ///
    /// Must equal `summary.candidate` (otherwise `tell` returns
    /// [`TellError::CandidateMismatch`]) and must not repeat across tells
    /// (otherwise [`TellError::DuplicateCandidate`]).
    pub candidate: CandidateId,
    /// Selecting evidence; it is retained rather than rerun for the winner.
    pub summary: CandidateSummary<S>,
}

/// Observable budget and lifecycle counters for a strategy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchStatus {
    /// Maximum candidates this strategy may propose.
    pub budget: u64,
    /// Number of candidates proposed so far.
    pub asked: u64,
    /// Number of candidates successfully told so far.
    pub told: u64,
    /// Number of proposed candidates still awaiting a result.
    pub outstanding: u64,
    /// Whether the proposal budget is exhausted and no tells remain outstanding.
    pub finished: bool,
}

/// A candidate and its exact selecting evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchResult<C, S> {
    /// Stable identity assigned by the experiment runner.
    pub candidate_id: CandidateId,
    /// Original proposed candidate, not a regenerated value.
    pub candidate: C,
    /// Exact aggregate that selected this candidate.
    pub summary: CandidateSummary<S>,
}

/// Invalid ask/tell lifecycle input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TellError {
    /// The same token appeared twice in one call or was already told.
    DuplicateToken(ProposalToken),
    /// The token was never returned by this strategy.
    UnknownToken(ProposalToken),
    /// A candidate identity was reused for another proposal.
    DuplicateCandidate(CandidateId),
    /// The outer candidate ID and summary candidate ID disagree.
    CandidateMismatch {
        /// Candidate identity supplied next to the token.
        told: CandidateId,
        /// Candidate identity embedded in the aggregate.
        summary: CandidateId,
    },
}

impl fmt::Display for TellError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateToken(token) => {
                write!(formatter, "duplicate proposal token {}", token.value())
            }
            Self::UnknownToken(token) => {
                write!(formatter, "unknown proposal token {}", token.value())
            }
            Self::DuplicateCandidate(candidate) => {
                write!(
                    formatter,
                    "duplicate candidate identity {}",
                    candidate.value()
                )
            }
            Self::CandidateMismatch { told, summary } => write!(
                formatter,
                "candidate identity mismatch: told {} but summary is for {}",
                told.value(),
                summary.value()
            ),
        }
    }
}

impl std::error::Error for TellError {}

/// Tokenized strategy lifecycle independent of any evaluator engine.
pub trait SearchStrategy<C, S> {
    /// Proposes at most `limit` candidates within the strategy budget.
    fn ask(&mut self, limit: usize) -> Vec<Asked<C>>;

    /// Incorporates evaluated candidates or rejects the entire invalid batch.
    ///
    /// # Errors
    ///
    /// Returns [`TellError`] when any token or candidate identity is invalid.
    fn tell(&mut self, evaluated: &[Told<S>]) -> Result<(), TellError>;

    /// Returns current lifecycle and budget counters.
    fn status(&self) -> SearchStatus;
}

/// Candidates awaiting `tell`, stored by their sequentially issued tokens.
///
/// Untold candidates lie in a window that begins at the oldest untold token
/// and ends at the next token to issue. A told slot becomes `None` until the
/// window's front advances past it, so `push` and `contains` are O(1) and
/// `remove` is amortized O(1).
struct PendingProposals<C> {
    /// Token of the window's first slot.
    first_token: u64,
    /// Candidates for consecutive tokens; `None` once told.
    window: VecDeque<Option<C>>,
    /// Number of untold candidates in the window.
    outstanding: usize,
}

impl<C> PendingProposals<C> {
    /// Creates an empty window starting at token zero.
    const fn new() -> Self {
        Self {
            first_token: 0,
            window: VecDeque::new(),
            outstanding: 0,
        }
    }

    /// Reserves room for `additional` newly issued tokens.
    fn reserve(&mut self, additional: usize) {
        self.window.reserve(additional);
    }

    /// Records the candidate for the next token, `first_token + window.len()`.
    fn push(&mut self, candidate: C) {
        self.window.push_back(Some(candidate));
        self.outstanding += 1;
    }

    /// Returns the window index of `token`, if it lies inside the window.
    fn slot(&self, token: ProposalToken) -> Option<usize> {
        let offset = token.value().checked_sub(self.first_token)?;
        usize::try_from(offset)
            .ok()
            .filter(|&index| index < self.window.len())
    }

    /// Returns whether `token` is awaiting a `tell`.
    fn contains(&self, token: ProposalToken) -> bool {
        self.slot(token)
            .is_some_and(|index| self.window[index].is_some())
    }

    /// Removes and returns the candidate for an untold `token`.
    fn remove(&mut self, token: ProposalToken) -> Option<C> {
        let index = self.slot(token)?;
        let candidate = self.window[index].take()?;
        self.outstanding -= 1;
        while matches!(self.window.front(), Some(None)) {
            self.window.pop_front();
            self.first_token += 1;
        }
        Some(candidate)
    }

    /// Returns the number of untold candidates.
    const fn len(&self) -> usize {
        self.outstanding
    }
}

/// Seeded random search with independent per-token proposal streams.
///
/// Asking in different batch sizes yields the same ordered proposal sequence.
/// The generator receives the only randomness used by the strategy.
///
/// ```
/// use simul::experiment::search::{SearchStrategy, SeededRandomSearch, Told};
/// use simul::experiment::{
///     CandidateId, CandidateSummary, MasterSeed, MaximizeScalarMean, ScalarReducer,
/// };
///
/// let mut search = SeededRandomSearch::new(
///     MasterSeed::from_u64(42),
///     4,
///     |rng| rng.next_u64() % 10,
///     MaximizeScalarMean,
/// );
/// for proposal in search.ask(4) {
///     let candidate = CandidateId::new(proposal.token.value());
///     let score = -(proposal.candidate as f64 - 7.0).abs();
///     let summary = CandidateSummary {
///         candidate,
///         requested: 1,
///         completed: 1,
///         failed: 0,
///         statistics: ScalarReducer::from_results([score])?,
///     };
///     search.tell(&[Told {
///         token: proposal.token,
///         candidate,
///         summary,
///     }])?;
/// }
/// assert!(search.status().finished);
/// assert!(search.best().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SeededRandomSearch<C, S, P, G> {
    /// V1 seed source for per-token proposal streams.
    deriver: SeedDeriver,
    /// Maximum number of candidates that may be proposed.
    budget: u64,
    /// Integer representation of the next proposal token.
    next_token: u64,
    /// Number of proposals successfully incorporated.
    told_count: u64,
    /// Caller-supplied candidate generator.
    generator: G,
    /// Policy used to compare candidate summaries.
    policy: P,
    /// Proposed candidate values awaiting evaluation.
    ///
    /// Tokens are issued in order, so a token below `next_token` that is no
    /// longer pending has already been told.
    pending: PendingProposals<C>,
    /// Candidate identities already incorporated by `tell`.
    completed_candidates: FxHashSet<CandidateId>,
    /// Current winner and its original selecting evidence.
    best: Option<SearchResult<C, S>>,
}

impl<C: fmt::Debug, S: fmt::Debug, P: fmt::Debug, G> fmt::Debug for SeededRandomSearch<C, S, P, G> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SeededRandomSearch")
            .field("budget", &self.budget)
            .field("asked", &self.next_token)
            .field("told", &self.told_count)
            .field("outstanding", &self.pending.len())
            .field("policy", &self.policy)
            .field("best", &self.best)
            .finish_non_exhaustive()
    }
}

impl<C, S, P, G> SeededRandomSearch<C, S, P, G>
where
    C: Clone,
    S: Clone,
    P: SelectionPolicy<S>,
    G: FnMut(&mut TrialRng) -> C,
{
    /// Creates a random strategy with an exact candidate budget.
    #[must_use]
    pub fn new(master_seed: MasterSeed, budget: u64, generator: G, policy: P) -> Self {
        Self {
            deriver: SeedDeriver::new(master_seed),
            budget,
            next_token: 0,
            told_count: 0,
            generator,
            policy,
            pending: PendingProposals::new(),
            completed_candidates: FxHashSet::default(),
            best: None,
        }
    }

    /// Returns the current winner and exact selecting aggregate, or `None`
    /// until the first successful [`SearchStrategy::tell`].
    ///
    /// On a policy tie, the lower [`CandidateId`] wins regardless of tell order.
    #[must_use]
    pub const fn best(&self) -> Option<&SearchResult<C, S>> {
        self.best.as_ref()
    }

    /// Validates an entire tell batch before state is mutated.
    fn validate_tells(&self, evaluated: &[Told<S>]) -> Result<(), TellError> {
        if evaluated.len() <= SCANNED_TELL_BATCH {
            for (index, told) in evaluated.iter().enumerate() {
                let earlier = &evaluated[..index];
                self.validate_tell(
                    told,
                    || earlier.iter().any(|other| other.token == told.token),
                    || {
                        earlier
                            .iter()
                            .any(|other| other.candidate == told.candidate)
                    },
                )?;
            }
        } else {
            let mut batch_tokens =
                FxHashSet::with_capacity_and_hasher(evaluated.len(), FxBuildHasher);
            let mut batch_candidates =
                FxHashSet::with_capacity_and_hasher(evaluated.len(), FxBuildHasher);
            for told in evaluated {
                self.validate_tell(
                    told,
                    || !batch_tokens.insert(told.token),
                    || !batch_candidates.insert(told.candidate),
                )?;
            }
        }
        Ok(())
    }

    /// Validates one tell in batch order. The callbacks report whether an
    /// earlier entry of the same batch repeats its token or candidate.
    fn validate_tell(
        &self,
        told: &Told<S>,
        repeats_token: impl FnOnce() -> bool,
        repeats_candidate: impl FnOnce() -> bool,
    ) -> Result<(), TellError> {
        if told.candidate != told.summary.candidate {
            return Err(TellError::CandidateMismatch {
                told: told.candidate,
                summary: told.summary.candidate,
            });
        }
        let outstanding = self.pending.contains(told.token);
        let completed = !outstanding && told.token.value() < self.next_token;
        if completed || repeats_token() {
            return Err(TellError::DuplicateToken(told.token));
        }
        if !outstanding {
            return Err(TellError::UnknownToken(told.token));
        }
        if self.completed_candidates.contains(&told.candidate) || repeats_candidate() {
            return Err(TellError::DuplicateCandidate(told.candidate));
        }
        Ok(())
    }

    /// Incorporates a validated batch in canonical order.
    ///
    /// Every policy comparison and summary clone runs before any state
    /// changes, so a panicking policy or `Clone` leaves the search unchanged.
    fn incorporate<'a>(
        &mut self,
        canonical: impl Iterator<Item = &'a Told<S>> + Clone,
    ) -> Result<(), TellError>
    where
        S: 'a,
    {
        let mut winner = self.batch_winner(canonical.clone()).and_then(|index| {
            canonical
                .clone()
                .nth(index)
                .map(|told| (index, told.summary.clone()))
        });
        for (index, told) in canonical.enumerate() {
            let candidate = self
                .pending
                .remove(told.token)
                .ok_or(TellError::UnknownToken(told.token))?;
            self.completed_candidates.insert(told.candidate);
            self.told_count += 1;
            if let Some((_, summary)) = winner.take_if(|(winner, _)| *winner == index) {
                self.best = Some(SearchResult {
                    candidate_id: told.candidate,
                    candidate,
                    summary,
                });
            }
        }
        Ok(())
    }

    /// Returns the batch position that ends up as the winner, if any replaces
    /// the current best when considered in canonical order.
    fn batch_winner<'a>(&self, canonical: impl Iterator<Item = &'a Told<S>>) -> Option<usize>
    where
        S: 'a,
    {
        let mut best = self
            .best
            .as_ref()
            .map(|best| (&best.summary.statistics, best.candidate_id));
        let mut winner = None;
        for (index, told) in canonical.enumerate() {
            let replace = match best {
                None => true,
                Some((statistics, candidate_id)) => {
                    match self.policy.compare(&told.summary.statistics, statistics) {
                        Ordering::Greater => true,
                        Ordering::Equal => told.candidate < candidate_id,
                        Ordering::Less => false,
                    }
                }
            };
            if replace {
                best = Some((&told.summary.statistics, told.candidate));
                winner = Some(index);
            }
        }
        winner
    }
}

impl<C, S, P, G> SearchStrategy<C, S> for SeededRandomSearch<C, S, P, G>
where
    C: Clone,
    S: Clone,
    P: SelectionPolicy<S>,
    G: FnMut(&mut TrialRng) -> C,
{
    fn ask(&mut self, limit: usize) -> Vec<Asked<C>> {
        let remaining = self.budget.saturating_sub(self.next_token);
        let count = usize::try_from(remaining).unwrap_or(usize::MAX).min(limit);
        let mut asked = Vec::with_capacity(count);
        let mut issued = Vec::with_capacity(count);
        for value in (self.next_token..).take(count) {
            let token = ProposalToken::new(value);
            let mut rng = self.deriver.proposal_seed(value).rng();
            let candidate = (self.generator)(&mut rng);
            issued.push(candidate.clone());
            asked.push(Asked { token, candidate });
        }
        // Issue tokens only after every generator and clone call returned, so a
        // panic leaves no outstanding tokens the caller never received.
        self.pending.reserve(count);
        for candidate in issued {
            self.pending.push(candidate);
            self.next_token += 1;
        }
        asked
    }

    fn tell(&mut self, evaluated: &[Told<S>]) -> Result<(), TellError> {
        self.validate_tells(evaluated)?;
        if evaluated.is_sorted_by_key(|told| told.candidate) {
            self.incorporate(evaluated.iter())?;
        } else {
            let mut canonical: Vec<_> = evaluated.iter().collect();
            canonical.sort_by_key(|told| told.candidate);
            self.incorporate(canonical.iter().copied())?;
        }
        Ok(())
    }

    fn status(&self) -> SearchStatus {
        let outstanding = u64::try_from(self.pending.len()).unwrap_or(u64::MAX);
        SearchStatus {
            budget: self.budget,
            asked: self.next_token,
            told: self.told_count,
            outstanding,
            finished: self.next_token == self.budget && outstanding == 0,
        }
    }
}

/// Invalid bounds supplied to [`sample_bounded`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundsError {
    /// A bound endpoint was not finite.
    NonFinite {
        /// Zero-based invalid bound dimension.
        dimension: usize,
    },
    /// A lower endpoint exceeded its upper endpoint.
    Reversed {
        /// Zero-based invalid bound dimension.
        dimension: usize,
    },
}

impl fmt::Display for BoundsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { dimension } => {
                write!(
                    formatter,
                    "bound endpoint in dimension {dimension} is not finite"
                )
            }
            Self::Reversed { dimension } => write!(
                formatter,
                "lower bound exceeds upper bound in dimension {dimension}"
            ),
        }
    }
}

impl std::error::Error for BoundsError {}

/// Validates finite, non-reversed bounds for bounded samplers.
fn validate_bounds(bounds: &[(f64, f64)]) -> Result<(), BoundsError> {
    for (dimension, &(lower, upper)) in bounds.iter().enumerate() {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(BoundsError::NonFinite { dimension });
        }
        if lower > upper {
            return Err(BoundsError::Reversed { dimension });
        }
    }
    Ok(())
}

/// Maps `fraction` in `[0, 1]` onto `[lower, upper]` without overflowing when
/// `upper - lower` exceeds `f64::MAX`.
#[allow(clippy::suboptimal_flops)]
fn scale_unit_to_bound(fraction: f64, lower: f64, upper: f64) -> f64 {
    let width = upper - lower;
    if width.is_finite() {
        lower + width * fraction
    } else {
        lower * (1.0 - fraction) + upper * fraction
    }
}

/// Produces ordered, uniformly sampled bounded vectors using proposal streams.
///
/// This convenience is deterministic. It allocates `budget` vectors of
/// `bounds.len()` values each, so even empty bounds reserve `budget` vectors.
///
/// # Errors
///
/// Returns [`BoundsError::NonFinite`] when an endpoint is NaN or infinite, or
/// [`BoundsError::Reversed`] when a lower endpoint exceeds its upper endpoint.
///
/// # Panics
///
/// Panics on capacity overflow when `budget` vectors cannot be allocated.
//
// Keep interpolation operations separate: contraction would change the frozen
// deterministic proposal sequence.
#[allow(clippy::suboptimal_flops)]
pub fn sample_bounded(
    master_seed: MasterSeed,
    budget: usize,
    bounds: &[(f64, f64)],
) -> Result<Vec<Vec<f64>>, BoundsError> {
    validate_bounds(bounds)?;
    let deriver = SeedDeriver::new(master_seed);
    let mut candidates = Vec::with_capacity(budget);
    for token in 0..budget {
        let mut candidate = Vec::with_capacity(bounds.len());
        for (dimension, &(lower, upper)) in bounds.iter().enumerate() {
            let token = u64::try_from(token).unwrap_or(u64::MAX);
            let dimension = u64::try_from(dimension).unwrap_or(u64::MAX);
            let mut rng = deriver.proposal_dimension_seed(token, dimension).rng();
            let fraction = rng.uniform_f64();
            candidate.push(scale_unit_to_bound(fraction, lower, upper));
        }
        candidates.push(candidate);
    }
    Ok(candidates)
}

/// Produces a deterministic Latin Hypercube Sample (`LHS`) of `samples` points
/// within `bounds`.
///
/// In every dimension, the interval `[lower, upper]` is partitioned into
/// `samples` equal-width strata, each stratum is assigned to exactly one point
/// by an independent seeded permutation, and each point is jittered uniformly
/// inside its stratum. This gives uniform 1D marginal coverage with far lower
/// variance than plain random sampling.
///
/// # Errors
///
/// Returns [`BoundsError`] when any bound endpoint is non-finite or reversed.
///
/// ```
/// use simul::experiment::MasterSeed;
/// use simul::experiment::search::sample_latin_hypercube;
///
/// let points = sample_latin_hypercube(MasterSeed::from_u64(7), 8, &[(0.0, 8.0), (10.0, 18.0)])?;
/// assert_eq!(8, points.len());
/// // Each unit stratum [k, k + 1) in dimension 0 contains exactly one point.
/// let mut seen = [false; 8];
/// for point in &points {
///     seen[point[0].floor() as usize] = true;
/// }
/// assert!(seen.iter().all(|&hit| hit));
/// # Ok::<(), simul::experiment::search::BoundsError>(())
/// ```
pub fn sample_latin_hypercube(
    master_seed: MasterSeed,
    samples: usize,
    bounds: &[(f64, f64)],
) -> Result<Vec<Vec<f64>>, BoundsError> {
    validate_bounds(bounds)?;
    let mut points: Vec<Vec<f64>> = (0..samples).map(|_| vec![0.0; bounds.len()]).collect();
    if samples == 0 || bounds.is_empty() {
        return Ok(points);
    }
    let deriver = SeedDeriver::new(master_seed);
    let n_f64 = samples as f64;
    let token = u64::try_from(samples).unwrap_or(u64::MAX);
    let mut strata: Vec<usize> = (0..samples).collect();
    for (dimension, &(lower, upper)) in bounds.iter().enumerate() {
        let mut rng = deriver
            .proposal_dimension_seed(token, u64::try_from(dimension).unwrap_or(u64::MAX))
            .rng();
        // Reset and Fisher-Yates shuffle strata for this dimension.
        for (index, slot) in strata.iter_mut().enumerate() {
            *slot = index;
        }
        for i in (1..samples).rev() {
            let j = rng.random_range(0..=i);
            strata.swap(i, j);
        }
        for (sample_index, &stratum) in strata.iter().enumerate() {
            let jitter = rng.uniform_f64();
            let fraction = ((stratum as f64 + jitter) / n_f64).clamp(0.0, 1.0);
            points[sample_index][dimension] = scale_unit_to_bound(fraction, lower, upper);
        }
    }
    Ok(points)
}

/// Maximum number of dimensions supported by [`sample_sobol`] and
/// [`sample_sobol_scrambled`].
pub const MAX_SOBOL_DIMENSIONS: usize = 32;

/// Invalid configuration supplied to [`sample_sobol`] or
/// [`sample_sobol_scrambled`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SobolError {
    /// One of the supplied bounds was non-finite or reversed.
    InvalidBounds(BoundsError),
    /// More dimensions were requested than [`MAX_SOBOL_DIMENSIONS`].
    TooManyDimensions {
        /// Requested dimension count.
        requested: usize,
        /// Maximum supported dimension count.
        max: usize,
    },
}

impl From<BoundsError> for SobolError {
    fn from(source: BoundsError) -> Self {
        Self::InvalidBounds(source)
    }
}

impl fmt::Display for SobolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBounds(source) => write!(formatter, "invalid Sobol bounds: {source}"),
            Self::TooManyDimensions { requested, max } => write!(
                formatter,
                "Sobol sampler supports at most {max} dimensions, got {requested}"
            ),
        }
    }
}

impl std::error::Error for SobolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidBounds(source) => Some(source),
            Self::TooManyDimensions { .. } => None,
        }
    }
}

/// Primitive polynomial specification and Joe & Kuo initial direction numbers
/// for one Sobol dimension above dimension 0.
struct SobolDimSpec {
    /// Degree `s` of the primitive polynomial over `GF(2)`.
    degree: u32,
    /// Middle polynomial coefficients encoded in bits `(s - 2)..=0`.
    poly_a: u32,
    /// Initial odd direction numbers `m_1..=m_s` (unused slots are 0).
    init_m: [u32; 7],
}

/// Joe & Kuo (2008/2010) direction numbers for Sobol dimensions 1..32
/// (dimension 0 is the base-2 van der Corput sequence).
const SOBOL_DIM_SPECS: [SobolDimSpec; MAX_SOBOL_DIMENSIONS - 1] = [
    SobolDimSpec {
        degree: 1,
        poly_a: 0,
        init_m: [1, 0, 0, 0, 0, 0, 0],
    },
    SobolDimSpec {
        degree: 2,
        poly_a: 1,
        init_m: [1, 3, 0, 0, 0, 0, 0],
    },
    SobolDimSpec {
        degree: 3,
        poly_a: 1,
        init_m: [1, 3, 1, 0, 0, 0, 0],
    },
    SobolDimSpec {
        degree: 3,
        poly_a: 2,
        init_m: [1, 1, 1, 0, 0, 0, 0],
    },
    SobolDimSpec {
        degree: 4,
        poly_a: 1,
        init_m: [1, 1, 3, 3, 0, 0, 0],
    },
    SobolDimSpec {
        degree: 4,
        poly_a: 4,
        init_m: [1, 3, 5, 13, 0, 0, 0],
    },
    SobolDimSpec {
        degree: 5,
        poly_a: 2,
        init_m: [1, 1, 5, 5, 17, 0, 0],
    },
    SobolDimSpec {
        degree: 5,
        poly_a: 4,
        init_m: [1, 1, 5, 5, 5, 0, 0],
    },
    SobolDimSpec {
        degree: 5,
        poly_a: 7,
        init_m: [1, 1, 7, 11, 19, 0, 0],
    },
    SobolDimSpec {
        degree: 5,
        poly_a: 11,
        init_m: [1, 1, 5, 1, 1, 0, 0],
    },
    SobolDimSpec {
        degree: 5,
        poly_a: 13,
        init_m: [1, 1, 1, 3, 11, 0, 0],
    },
    SobolDimSpec {
        degree: 5,
        poly_a: 14,
        init_m: [1, 3, 5, 5, 31, 0, 0],
    },
    SobolDimSpec {
        degree: 6,
        poly_a: 1,
        init_m: [1, 3, 3, 9, 7, 49, 0],
    },
    SobolDimSpec {
        degree: 6,
        poly_a: 13,
        init_m: [1, 1, 5, 5, 19, 61, 0],
    },
    SobolDimSpec {
        degree: 6,
        poly_a: 16,
        init_m: [1, 1, 1, 15, 21, 21, 0],
    },
    SobolDimSpec {
        degree: 6,
        poly_a: 19,
        init_m: [1, 3, 1, 13, 27, 49, 0],
    },
    SobolDimSpec {
        degree: 6,
        poly_a: 22,
        init_m: [1, 1, 1, 15, 7, 5, 0],
    },
    SobolDimSpec {
        degree: 6,
        poly_a: 25,
        init_m: [1, 3, 1, 15, 13, 25, 0],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 1,
        init_m: [1, 3, 7, 11, 23, 15, 103],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 4,
        init_m: [1, 3, 7, 13, 13, 15, 69],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 8,
        init_m: [1, 1, 3, 13, 7, 35, 63],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 32,
        init_m: [1, 3, 5, 9, 1, 25, 53],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 7,
        init_m: [1, 3, 1, 13, 9, 35, 107],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 14,
        init_m: [1, 3, 1, 5, 27, 61, 31],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 19,
        init_m: [1, 1, 5, 11, 19, 41, 61],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 21,
        init_m: [1, 3, 5, 3, 3, 13, 69],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 28,
        init_m: [1, 1, 7, 13, 1, 19, 1],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 37,
        init_m: [1, 3, 7, 5, 13, 41, 59],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 41,
        init_m: [1, 1, 3, 5, 23, 55, 19],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 42,
        init_m: [1, 3, 3, 1, 17, 45, 79],
    },
    SobolDimSpec {
        degree: 7,
        poly_a: 50,
        init_m: [1, 1, 1, 3, 21, 37, 121],
    },
];

/// Scale converting a 32-bit Sobol integer coordinate to `[0, 1)`.
const U32_TO_UNIT_F64: f64 = 1.0 / ((1_u64 << 32) as f64);

/// Computes the 32 left-aligned Sobol direction numbers for `dimension`.
fn sobol_direction_numbers(dimension: usize) -> [u32; 32] {
    let mut v = [0_u32; 32];
    if dimension == 0 {
        for (bit, slot) in v.iter_mut().enumerate() {
            let shift = 31 - u32::try_from(bit).unwrap_or(31);
            *slot = 1_u32 << shift;
        }
        return v;
    }
    let spec = &SOBOL_DIM_SPECS[dimension - 1];
    let s = spec.degree as usize;
    for i in 0..s {
        let shift = 32 - u32::try_from(i + 1).unwrap_or(32);
        v[i] = spec.init_m[i] << shift;
    }
    for i in s..32 {
        let mut val = v[i - s] ^ (v[i - s] >> spec.degree);
        for k in 1..s {
            let bit = u32::try_from(s - 1 - k).unwrap_or(0);
            if ((spec.poly_a >> bit) & 1) != 0 {
                val ^= v[i - k];
            }
        }
        v[i] = val;
    }
    v
}

/// Core Sobol generator supporting optional per-dimension XOR digital shifts.
fn generate_sobol(
    samples: usize,
    bounds: &[(f64, f64)],
    shifts: Option<&[u32]>,
) -> Result<Vec<Vec<f64>>, SobolError> {
    validate_bounds(bounds)?;
    if bounds.len() > MAX_SOBOL_DIMENSIONS {
        return Err(SobolError::TooManyDimensions {
            requested: bounds.len(),
            max: MAX_SOBOL_DIMENSIONS,
        });
    }
    let dims = bounds.len();
    let mut points = Vec::with_capacity(samples);
    if samples == 0 || dims == 0 {
        points.resize_with(samples, Vec::new);
        return Ok(points);
    }
    let directions: Vec<[u32; 32]> = (0..dims).map(sobol_direction_numbers).collect();
    let mut state: Vec<u32> = shifts.map_or_else(|| vec![0_u32; dims], <[u32]>::to_vec);

    for index in 0..samples {
        if index > 0 {
            let bit = index.trailing_zeros() as usize;
            if bit < 32 {
                for d in 0..dims {
                    state[d] ^= directions[d][bit];
                }
            }
        }
        let mut point = Vec::with_capacity(dims);
        for (d, &(lower, upper)) in bounds.iter().enumerate() {
            let fraction = f64::from(state[d]) * U32_TO_UNIT_F64;
            point.push(scale_unit_to_bound(fraction, lower, upper));
        }
        points.push(point);
    }
    Ok(points)
}

/// Produces a deterministic low-discrepancy Sobol quasi-random sequence of
/// `samples` points within `bounds` (up to [`MAX_SOBOL_DIMENSIONS`] = 32
/// dimensions) using Joe & Kuo direction numbers in Gray-code order.
///
/// Any power-of-two sample count `2^m` covers every 1D dyadic subinterval
/// `[k / 2^m, (k + 1) / 2^m)` exactly once and achieves superior space-filling
/// uniformity for parameter sweeps and sensitivity analysis. For randomized
/// quasi-Monte Carlo replicates without the origin point, see
/// [`sample_sobol_scrambled`].
///
/// # Errors
///
/// Returns [`SobolError::InvalidBounds`] when any bound is non-finite or
/// reversed, or [`SobolError::TooManyDimensions`] when `bounds.len()` exceeds
/// [`MAX_SOBOL_DIMENSIONS`].
///
/// ```
/// use simul::experiment::search::sample_sobol;
///
/// let points = sample_sobol(16, &[(0.0, 1.0), (10.0, 20.0)])?;
/// assert_eq!(16, points.len());
/// assert!(points.iter().all(|p| (0.0..=1.0).contains(&p[0]) && (10.0..=20.0).contains(&p[1])));
/// # Ok::<(), simul::experiment::search::SobolError>(())
/// ```
pub fn sample_sobol(samples: usize, bounds: &[(f64, f64)]) -> Result<Vec<Vec<f64>>, SobolError> {
    generate_sobol(samples, bounds, None)
}

/// Produces a digitally shifted (scrambled) Sobol quasi-random sequence seeded
/// by `master_seed`.
///
/// Each dimension receives an independent 32-bit XOR digital shift derived from
/// `master_seed`, preserving exact Sobol dyadic stratification for power-of-two
/// sample counts while eliminating the zero corner and enabling unbiased
/// randomized quasi-Monte Carlo replicates across distinct master seeds.
///
/// # Errors
///
/// Returns [`SobolError`] under the same conditions as [`sample_sobol`].
pub fn sample_sobol_scrambled(
    master_seed: MasterSeed,
    samples: usize,
    bounds: &[(f64, f64)],
) -> Result<Vec<Vec<f64>>, SobolError> {
    validate_bounds(bounds)?;
    if bounds.len() > MAX_SOBOL_DIMENSIONS {
        return Err(SobolError::TooManyDimensions {
            requested: bounds.len(),
            max: MAX_SOBOL_DIMENSIONS,
        });
    }
    let deriver = SeedDeriver::new(master_seed);
    let shifts: Vec<u32> = (0..bounds.len())
        .map(|dim| {
            let mut rng = deriver
                .proposal_dimension_seed(0, u64::try_from(dim).unwrap_or(u64::MAX))
                .rng();
            let mut bytes = [0_u8; 4];
            rng.fill_bytes(&mut bytes);
            u32::from_le_bytes(bytes)
        })
        .collect();
    generate_sobol(samples, bounds, Some(&shifts))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::missing_docs_in_private_items)]
mod tests {
    use super::*;
    use crate::experiment::{MaximizeScalarMean, ScalarReducer};

    #[derive(Clone, Copy)]
    struct MaxI32;

    impl SelectionPolicy<i32> for MaxI32 {
        fn compare(&self, left: &i32, right: &i32) -> Ordering {
            left.cmp(right)
        }
    }

    fn summary(candidate: u64, score: i32) -> CandidateSummary<i32> {
        CandidateSummary {
            candidate: CandidateId::new(candidate),
            requested: 1,
            completed: 1,
            failed: 0,
            statistics: score,
        }
    }

    fn assert_f64_bits_eq(actual: f64, expected: f64) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    #[test]
    fn ask_batching_does_not_change_proposals() {
        let seed = MasterSeed::from_u64(88);
        let generator = |rng: &mut TrialRng| rng.next_u64();
        let mut one_batch = SeededRandomSearch::new(seed, 5, generator, MaxI32);
        let all = one_batch.ask(5);

        let generator = |rng: &mut TrialRng| rng.next_u64();
        let mut split = SeededRandomSearch::new(seed, 5, generator, MaxI32);
        let mut split_asked = split.ask(2);
        split_asked.extend(split.ask(99));
        assert_eq!(all, split_asked);
        assert_eq!(0, split.ask(1).len());
        assert_eq!(split.status().asked, 5);
        assert_eq!(split.status().outstanding, 5);
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn panicking_generator_issues_no_tokens_and_retries_identically() {
        let seed = MasterSeed::from_u64(88);
        let mut calls = 0_u32;
        let generator = |rng: &mut TrialRng| {
            calls += 1;
            assert_ne!(calls, 3, "generator failed");
            rng.next_u64()
        };
        let mut search = SeededRandomSearch::new(seed, 3, generator, MaxI32);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| search.ask(3)));
        assert!(outcome.is_err());
        assert_eq!(0, search.status().asked);
        assert_eq!(0, search.status().outstanding);

        let retried = search.ask(3);
        let generator = |rng: &mut TrialRng| rng.next_u64();
        let mut fresh = SeededRandomSearch::new(seed, 3, generator, MaxI32);
        assert_eq!(fresh.ask(3), retried);
    }

    #[test]
    fn tells_reject_unknown_and_duplicate_tokens_atomically() {
        let mut search = SeededRandomSearch::new(MasterSeed::from_u64(1), 1, |_| 7_u64, MaxI32);
        let asked = search.ask(1);
        let unknown = Told {
            token: ProposalToken::new(99),
            candidate: CandidateId::new(1),
            summary: summary(1, 2),
        };
        assert_eq!(
            search.tell(&[unknown]),
            Err(TellError::UnknownToken(ProposalToken::new(99)))
        );
        assert_eq!(search.status().outstanding, 1);

        let valid = Told {
            token: asked[0].token,
            candidate: CandidateId::new(1),
            summary: summary(1, 2),
        };
        search
            .tell(std::slice::from_ref(&valid))
            .expect("valid tell should be accepted");
        assert_eq!(
            search.tell(&[valid]),
            Err(TellError::DuplicateToken(asked[0].token))
        );
        assert!(search.status().finished);
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn panicking_policy_leaves_the_search_unchanged_and_retryable() {
        struct PanicsOnce(std::cell::Cell<bool>);
        impl SelectionPolicy<i32> for PanicsOnce {
            fn compare(&self, left: &i32, right: &i32) -> Ordering {
                assert!(!self.0.replace(false), "policy failed");
                left.cmp(right)
            }
        }
        let policy = PanicsOnce(std::cell::Cell::new(true));
        let mut search = SeededRandomSearch::new(MasterSeed::from_u64(1), 3, |_| 7_u64, policy);
        let asked = search.ask(3);
        let batch: Vec<_> = (0..3)
            .map(|index| Told {
                token: asked[index].token,
                candidate: CandidateId::new(index as u64),
                summary: summary(index as u64, 1),
            })
            .collect();
        let before = search.status();
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| search.tell(&batch)));
        assert!(outcome.is_err());
        assert_eq!(before, search.status());
        assert!(search.best().is_none());

        search
            .tell(&batch)
            .expect("the untouched batch can be retold");
        assert!(search.status().finished);
        assert_eq!(3, search.status().told);
    }

    /// A value whose clone panics once `CLONES_UNTIL_PANIC` counts down to it.
    #[derive(Debug, PartialEq)]
    struct CloneBomb(i32);

    thread_local! {
        /// Zero disarms; otherwise the clone that brings it to zero panics.
        static CLONES_UNTIL_PANIC: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    impl Clone for CloneBomb {
        fn clone(&self) -> Self {
            let remaining = CLONES_UNTIL_PANIC.get();
            if remaining > 0 {
                CLONES_UNTIL_PANIC.set(remaining - 1);
                assert_ne!(remaining, 1, "clone failed");
            }
            Self(self.0)
        }
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn panicking_candidate_clone_issues_no_tokens() {
        let mut search = SeededRandomSearch::new(
            MasterSeed::from_u64(1),
            2,
            |_: &mut TrialRng| CloneBomb(7),
            MaxI32,
        );
        CLONES_UNTIL_PANIC.set(2);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| search.ask(2)));
        assert!(outcome.is_err());
        assert_eq!(0, search.status().asked);
        assert_eq!(2, search.ask(2).len());
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn panicking_summary_clone_leaves_the_search_unchanged() {
        struct ByValue;
        impl SelectionPolicy<CloneBomb> for ByValue {
            fn compare(&self, left: &CloneBomb, right: &CloneBomb) -> Ordering {
                left.0.cmp(&right.0)
            }
        }
        let mut search = SeededRandomSearch::new(MasterSeed::from_u64(1), 2, |_| 7_u64, ByValue);
        let asked = search.ask(2);
        let batch: Vec<_> = (0..2)
            .map(|index| Told {
                token: asked[index].token,
                candidate: CandidateId::new(index as u64),
                summary: CandidateSummary {
                    candidate: CandidateId::new(index as u64),
                    requested: 1,
                    completed: 1,
                    failed: 0,
                    statistics: CloneBomb(1),
                },
            })
            .collect();
        let before = search.status();
        CLONES_UNTIL_PANIC.set(1);
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| search.tell(&batch)));
        assert!(outcome.is_err());
        assert_eq!(before, search.status());
        search
            .tell(&batch)
            .expect("the untouched batch can be retold");
        assert_eq!(2, search.status().told);
    }

    #[test]
    fn exact_ties_use_candidate_identity_not_tell_order() {
        let mut search = SeededRandomSearch::new(MasterSeed::from_u64(1), 2, |_| 7_u64, MaxI32);
        let asked = search.ask(2);
        search
            .tell(&[
                Told {
                    token: asked[0].token,
                    candidate: CandidateId::new(9),
                    summary: summary(9, 5),
                },
                Told {
                    token: asked[1].token,
                    candidate: CandidateId::new(2),
                    summary: summary(2, 5),
                },
            ])
            .expect("valid tell batch should be accepted");
        let best = search.best().expect("a told candidate should become best");
        assert_eq!(best.candidate_id, CandidateId::new(2));
        assert_eq!(best.candidate, 7);
        assert_eq!(best.summary.statistics, 5);
    }

    fn told(token: ProposalToken, candidate: u64) -> Told<i32> {
        Told {
            token,
            candidate: CandidateId::new(candidate),
            summary: summary(candidate, 1),
        }
    }

    #[test]
    fn out_of_order_tells_track_outstanding_tokens() {
        let mut search = SeededRandomSearch::new(MasterSeed::from_u64(1), 6, |_| 7_u64, MaxI32);
        let asked = search.ask(4);
        let token = |index: usize| asked[index].token;
        for (index, candidate) in [(2, 20), (0, 0), (3, 30)] {
            search
                .tell(&[told(token(index), candidate)])
                .expect("the token is outstanding");
        }
        assert_eq!(1, search.status().outstanding);
        for index in [0, 2, 3] {
            assert_eq!(
                Err(TellError::DuplicateToken(token(index))),
                search.tell(&[told(token(index), 99)])
            );
        }
        assert_eq!(
            Err(TellError::UnknownToken(ProposalToken::new(4))),
            search.tell(&[told(ProposalToken::new(4), 40)])
        );

        let later = search.ask(2);
        search
            .tell(&[
                told(later[1].token, 50),
                told(token(1), 10),
                told(later[0].token, 40),
            ])
            .expect("the tokens are outstanding");
        assert!(search.status().finished);
        assert_eq!(6, search.status().told);
    }

    #[test]
    fn repeated_identities_reject_the_whole_batch_at_every_size() {
        for size in [2, SCANNED_TELL_BATCH, SCANNED_TELL_BATCH + 1] {
            let budget = u64::try_from(size).expect("small batch") + 1;
            let last = budget - 2;
            let mut search =
                SeededRandomSearch::new(MasterSeed::from_u64(1), budget, |_| 7_u64, MaxI32);
            let asked = search.ask(size + 1);
            let mut batch: Vec<Told<i32>> = asked[..size]
                .iter()
                .zip(0..)
                .map(|(proposal, candidate)| told(proposal.token, candidate))
                .collect();

            batch[size - 1] = told(asked[0].token, last);
            assert_eq!(
                Err(TellError::DuplicateToken(asked[0].token)),
                search.tell(&batch),
                "{size} tells"
            );
            batch[size - 1] = told(asked[size - 1].token, 0);
            assert_eq!(
                Err(TellError::DuplicateCandidate(CandidateId::new(0))),
                search.tell(&batch),
                "{size} tells"
            );
            assert_eq!(0, search.status().told);
            assert_eq!(budget, search.status().outstanding);

            batch[size - 1] = told(asked[size - 1].token, last);
            search
                .tell(&batch)
                .expect("distinct identities should be accepted");
            assert_eq!(
                Err(TellError::DuplicateCandidate(CandidateId::new(0))),
                search.tell(&[told(asked[size].token, 0)])
            );
            assert_eq!(
                Err(TellError::DuplicateToken(asked[0].token)),
                search.tell(&[told(asked[0].token, budget)])
            );
            assert_eq!(budget - 1, search.status().told);
            assert_eq!(1, search.status().outstanding);
        }
    }

    #[test]
    fn singleton_tells_preserve_state_on_invalid_candidate_ids() {
        let mut search = SeededRandomSearch::new(MasterSeed::from_u64(1), 2, |_| 7_u64, MaxI32);
        let asked = search.ask(2);
        search
            .tell(&[Told {
                token: asked[0].token,
                candidate: CandidateId::new(3),
                summary: summary(3, 1),
            }])
            .expect("first result should be accepted");
        let status = search.status();
        let best = search.best().cloned();

        for (candidate, summary_candidate, expected) in [
            (3, 3, TellError::DuplicateCandidate(CandidateId::new(3))),
            (
                4,
                5,
                TellError::CandidateMismatch {
                    told: CandidateId::new(4),
                    summary: CandidateId::new(5),
                },
            ),
        ] {
            let invalid = Told {
                token: asked[1].token,
                candidate: CandidateId::new(candidate),
                summary: summary(summary_candidate, 99),
            };
            assert_eq!(Err(expected), search.tell(&[invalid]));
            assert_eq!(status, search.status());
            assert_eq!(best.as_ref(), search.best());
        }

        search
            .tell(&[Told {
                token: asked[1].token,
                candidate: CandidateId::new(4),
                summary: summary(4, 2),
            }])
            .expect("rejected results must leave the proposal pending");
        assert!(search.status().finished);
        assert_eq!(
            Some(CandidateId::new(4)),
            search.best().map(|result| result.candidate_id)
        );
    }

    #[test]
    fn zero_budget_search_is_finished_and_proposes_nothing() {
        let mut empty = SeededRandomSearch::new(MasterSeed::from_u64(1), 0, |_| 0_u64, MaxI32);
        assert_eq!(0, empty.ask(10).len());
        assert!(empty.status().finished);
    }

    #[test]
    fn bounded_sampling_stays_in_bounds_and_rejects_invalid_bounds() {
        let sampled = sample_bounded(MasterSeed::from_u64(5), 3, &[(0.0, 1.0), (-2.0, 2.0)])
            .expect("finite ordered bounds should sample");
        assert_eq!(3, sampled.len());
        assert!(sampled.iter().all(|candidate| {
            (0.0..1.0).contains(&candidate[0]) && (-2.0..2.0).contains(&candidate[1])
        }));
        assert_eq!(
            Err(BoundsError::NonFinite { dimension: 0 }),
            sample_bounded(MasterSeed::from_u64(5), 1, &[(f64::NAN, 1.0)])
        );
        assert_eq!(
            Err(BoundsError::Reversed { dimension: 1 }),
            sample_bounded(MasterSeed::from_u64(5), 1, &[(0.0, 1.0), (1.0, 0.0)])
        );
    }

    #[test]
    fn bounded_sampling_stays_finite_with_overflowing_widths() {
        let bounds = [(-f64::MAX, f64::MAX), (-f64::MAX, f64::MAX / 2.0)];
        let sampled = sample_bounded(MasterSeed::from_u64(42), 16, &bounds)
            .expect("finite bounds should sample");
        for candidate in &sampled {
            for (&actual, &(lower, upper)) in candidate.iter().zip(&bounds) {
                assert!(actual.is_finite());
                assert!((lower..=upper).contains(&actual));
            }
        }
    }

    #[test]
    fn bounded_sampling_sequence_is_frozen() {
        let bounds = [(-1.3, 2.7), (-f64::MAX, 0.0), (f64::MAX, f64::MAX)];
        let sampled = sample_bounded(MasterSeed::from_u64(42), 2, &bounds)
            .expect("finite bounds should sample");
        let expected: [[u64; 3]; 2] = [
            [
                0x3fef_5da4_c1ab_2712,
                0xffe9_ce73_5518_b320,
                0x7fef_ffff_ffff_ffff,
            ],
            [
                0xbfcc_5f6b_eb05_d938,
                0xffe6_fbe2_29d7_f0b8,
                0x7fef_ffff_ffff_ffff,
            ],
        ];
        for (candidate, expected) in sampled.iter().zip(expected) {
            for (&actual, expected) in candidate.iter().zip(expected) {
                assert_f64_bits_eq(actual, f64::from_bits(expected));
            }
        }
    }

    #[test]
    fn selecting_summary_is_retained_without_regeneration() {
        let mut generated = 0_u64;
        let generator = move |_: &mut TrialRng| {
            generated += 1;
            generated
        };
        let mut search =
            SeededRandomSearch::new(MasterSeed::from_u64(4), 1, generator, MaximizeScalarMean);
        let asked = search.ask(1);
        let selecting =
            ScalarReducer::from_results([9.0, 11.0]).expect("finite observations should reduce");
        search
            .tell(&[Told {
                token: asked[0].token,
                candidate: CandidateId::new(12),
                summary: CandidateSummary {
                    candidate: CandidateId::new(12),
                    requested: 2,
                    completed: 2,
                    failed: 0,
                    statistics: selecting,
                },
            }])
            .expect("valid tell should be accepted");
        assert_eq!(
            search
                .best()
                .expect("a told candidate should become best")
                .candidate,
            1
        );
        let actual = search
            .best()
            .expect("a told candidate should become best")
            .summary
            .statistics;
        assert_eq!(actual.count(), selecting.count());
        assert_f64_bits_eq(actual.mean(), selecting.mean());
        assert_eq!(
            actual.sample_variance().map(f64::to_bits),
            selecting.sample_variance().map(f64::to_bits)
        );
        assert_eq!(
            actual.standard_error().map(f64::to_bits),
            selecting.standard_error().map(f64::to_bits)
        );
        assert_f64_bits_eq(actual.min(), selecting.min());
        assert_f64_bits_eq(actual.max(), selecting.max());
    }

    #[test]
    fn search_error_types_implement_display_and_error() {
        for error in [
            TellError::DuplicateToken(ProposalToken::new(1)),
            TellError::UnknownToken(ProposalToken::new(2)),
            TellError::DuplicateCandidate(CandidateId::new(3)),
            TellError::CandidateMismatch {
                told: CandidateId::new(4),
                summary: CandidateId::new(5),
            },
        ] {
            let boxed: Box<dyn std::error::Error> = Box::new(error);
            assert_ne!("", boxed.to_string());
        }
        for error in [
            BoundsError::NonFinite { dimension: 0 },
            BoundsError::Reversed { dimension: 1 },
        ] {
            let boxed: Box<dyn std::error::Error> = Box::new(error);
            assert_ne!("", boxed.to_string());
        }
        for error in [
            SobolError::InvalidBounds(BoundsError::NonFinite { dimension: 0 }),
            SobolError::TooManyDimensions {
                requested: 40,
                max: MAX_SOBOL_DIMENSIONS,
            },
        ] {
            let boxed: Box<dyn std::error::Error> = Box::new(error);
            assert_ne!("", boxed.to_string());
        }
    }

    #[test]
    fn sobol_polynomials_are_primitive_over_gf2_and_initial_m_are_valid() {
        // Verify every polynomial x^s + a_1 x^{s-1} + ... + a_{s-1} x + 1 in
        // SOBOL_DIM_SPECS is primitive over GF(2) and all initial m_i are odd
        // and strictly less than 2^i.
        let gf2_mul_mod = |mut a: u32, mut b: u32, poly: u32, degree: u32| -> u32 {
            let mut result = 0_u32;
            let high_bit = 1_u32 << degree;
            while b != 0 {
                if (b & 1) != 0 {
                    result ^= a;
                }
                a <<= 1;
                if (a & high_bit) != 0 {
                    a ^= poly;
                }
                b >>= 1;
            }
            result
        };
        let gf2_pow_x = |mut exp: u32, poly: u32, degree: u32| -> u32 {
            let mut base = 2_u32; // x
            let mut result = 1_u32;
            while exp != 0 {
                if (exp & 1) != 0 {
                    result = gf2_mul_mod(result, base, poly, degree);
                }
                base = gf2_mul_mod(base, base, poly, degree);
                exp >>= 1;
            }
            result
        };

        let mut seen_polys = std::collections::BTreeSet::new();
        for (index, spec) in SOBOL_DIM_SPECS.iter().enumerate() {
            assert!(
                seen_polys.insert((spec.degree, spec.poly_a)),
                "dim {}: duplicate primitive polynomial",
                index + 1
            );
            let s = spec.degree;
            for (i, &m) in spec.init_m.iter().enumerate().take(s as usize) {
                assert_eq!(1, m & 1, "dim {}: m_{} = {m} must be odd", index + 1, i + 1);
                assert!(
                    m < (1_u32 << (i + 1)),
                    "dim {}: m_{} = {m} out of range",
                    index + 1,
                    i + 1
                );
            }
            let poly = (1_u32 << s) | (spec.poly_a << 1) | 1;
            let order = (1_u32 << s) - 1;
            assert_eq!(
                1,
                gf2_pow_x(order, poly, s),
                "dim {}: polynomial {poly:#b} does not have period dividing {order}",
                index + 1
            );
            // Check proper divisors of 2^s - 1 for s in 1..=7 (orders: 1, 3, 7, 15, 31, 63, 127).
            for d in 1..order {
                if order % d == 0 {
                    assert_ne!(
                        1,
                        gf2_pow_x(d, poly, s),
                        "dim {}: polynomial {poly:#b} is not primitive (order divides {d})",
                        index + 1
                    );
                }
            }
        }
    }

    #[test]
    fn latin_hypercube_stratifies_every_dimension_and_replays_deterministically() {
        let seed = MasterSeed::from_u64(99);
        let bounds = [(0.0, 10.0), (-5.0, 5.0), (100.0, 100.0)];
        let a = sample_latin_hypercube(seed, 32, &bounds).expect("lhs");
        let b = sample_latin_hypercube(seed, 32, &bounds).expect("lhs replay");
        assert_eq!(a, b);
        assert_eq!(32, a.len());

        for (dim, &(lo, hi)) in bounds.iter().enumerate() {
            if lo < hi {
                let mut counts = [0_usize; 32];
                for point in &a {
                    let v = point[dim];
                    assert!(v >= lo && v <= hi, "dim {dim}: {v} out of [{lo}, {hi}]");
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let bucket = (((v - lo) / (hi - lo)) * 32.0).floor() as usize;
                    counts[bucket.min(31)] += 1;
                }
                assert!(
                    counts.iter().all(|&c| c == 1),
                    "dim {dim}: expected 1 point per stratum, got {counts:?}"
                );
            } else {
                assert!(a.iter().all(|p| (p[dim] - lo).abs() == 0.0));
            }
        }
    }

    #[test]
    fn sobol_and_scrambled_sobol_are_dyadically_stratified_across_all_32_dimensions() {
        let bounds: Vec<(f64, f64)> = (0..MAX_SOBOL_DIMENSIONS).map(|_| (0.0, 1.0)).collect();
        let plain = sample_sobol(64, &bounds).expect("plain sobol");
        let scrambled =
            sample_sobol_scrambled(MasterSeed::from_u64(42), 64, &bounds).expect("scrambled");
        let scrambled_again =
            sample_sobol_scrambled(MasterSeed::from_u64(42), 64, &bounds).expect("replay");
        assert_eq!(scrambled, scrambled_again);
        assert_ne!(plain[0], scrambled[0]);

        for set in [&plain, &scrambled] {
            for dim in 0..MAX_SOBOL_DIMENSIONS {
                let mut bins = [0_usize; 64];
                for point in set {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let bin = (point[dim] * 64.0).floor() as usize;
                    bins[bin] += 1;
                }
                assert!(
                    bins.iter().all(|&c| c == 1),
                    "dim {dim}: 64-point Sobol net must hit every dyadic 1/64 bin once"
                );
            }
        }

        // Sobol 2D integration of f(x, y) = x * y over [0, 1]^2 (exact = 0.25).
        let pts = sample_sobol(256, &[(0.0, 1.0), (0.0, 1.0)]).expect("2d sobol");
        let estimate: f64 = pts.iter().map(|p| p[0] * p[1]).sum::<f64>() / 256.0;
        assert!(
            (estimate - 0.25).abs() < 0.005,
            "expected accurate Sobol integral, got {estimate}"
        );

        let too_many = vec![(0.0, 1.0); MAX_SOBOL_DIMENSIONS + 1];
        assert_eq!(
            Err(SobolError::TooManyDimensions {
                requested: MAX_SOBOL_DIMENSIONS + 1,
                max: MAX_SOBOL_DIMENSIONS,
            }),
            sample_sobol(4, &too_many)
        );
    }
}
