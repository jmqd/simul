//! Tokenized deterministic ask/tell search strategies.

use super::{CandidateId, CandidateSummary, MasterSeed, SeedDeriver, SelectionPolicy, TrialRng};
use rustc_hash::{FxBuildHasher, FxHashSet};
use std::cmp::Ordering;
use std::collections::VecDeque;

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

    /// Returns the current winner and exact selecting aggregate, if any.
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

    /// Incorporates one validated tell.
    fn incorporate(&mut self, told: &Told<S>) -> Result<(), TellError> {
        let candidate = self
            .pending
            .remove(told.token)
            .ok_or(TellError::UnknownToken(told.token))?;
        self.consider(candidate, told);
        self.completed_candidates.insert(told.candidate);
        self.told_count += 1;
        Ok(())
    }

    /// Replaces the current winner when `told` is preferred.
    fn consider(&mut self, candidate: C, told: &Told<S>) {
        let replace = match &self.best {
            None => true,
            Some(best) => match self
                .policy
                .compare(&told.summary.statistics, &best.summary.statistics)
            {
                Ordering::Greater => true,
                Ordering::Equal => told.candidate < best.candidate_id,
                Ordering::Less => false,
            },
        };
        if replace {
            self.best = Some(SearchResult {
                candidate_id: told.candidate,
                candidate,
                summary: told.summary.clone(),
            });
        }
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
        self.pending.reserve(count);
        for _ in 0..count {
            let token = ProposalToken::new(self.next_token);
            let mut rng = self.deriver.proposal_seed(token.value()).rng();
            let candidate = (self.generator)(&mut rng);
            self.pending.push(candidate.clone());
            asked.push(Asked { token, candidate });
            self.next_token += 1;
        }
        asked
    }

    fn tell(&mut self, evaluated: &[Told<S>]) -> Result<(), TellError> {
        self.validate_tells(evaluated)?;
        if evaluated.is_sorted_by_key(|told| told.candidate) {
            for told in evaluated {
                self.incorporate(told)?;
            }
        } else {
            let mut canonical: Vec<_> = evaluated.iter().collect();
            canonical.sort_by_key(|told| told.candidate);
            for told in canonical {
                self.incorporate(told)?;
            }
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

/// Produces ordered, uniformly sampled bounded vectors using proposal streams.
///
/// This convenience is deterministic and allocation-bounded by `budget × bounds.len()`.
///
/// # Errors
///
/// Returns [`BoundsError::NonFinite`] when an endpoint is NaN or infinite, or
/// [`BoundsError::Reversed`] when a lower endpoint exceeds its upper endpoint.
//
// Keep interpolation operations separate: contraction would change the frozen
// deterministic proposal sequence.
#[allow(clippy::suboptimal_flops)]
pub fn sample_bounded(
    master_seed: MasterSeed,
    budget: usize,
    bounds: &[(f64, f64)],
) -> Result<Vec<Vec<f64>>, BoundsError> {
    for (dimension, &(lower, upper)) in bounds.iter().enumerate() {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(BoundsError::NonFinite { dimension });
        }
        if lower > upper {
            return Err(BoundsError::Reversed { dimension });
        }
    }
    let deriver = SeedDeriver::new(master_seed);
    let mut candidates = Vec::with_capacity(budget);
    for token in 0..budget {
        let mut candidate = Vec::with_capacity(bounds.len());
        for (dimension, &(lower, upper)) in bounds.iter().enumerate() {
            let token = u64::try_from(token).unwrap_or(u64::MAX);
            let dimension = u64::try_from(dimension).unwrap_or(u64::MAX);
            let mut rng = deriver.proposal_dimension_seed(token, dimension).rng();
            let fraction = rng.uniform_f64();
            let width = upper - lower;
            candidate.push(if width.is_finite() {
                lower + width * fraction
            } else {
                // An overflowing width requires opposite-sign finite endpoints.
                // Weighting them separately keeps both products and their sum finite.
                lower * (1.0 - fraction) + upper * fraction
            });
        }
        candidates.push(candidate);
    }
    Ok(candidates)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::missing_docs_in_private_items)]
mod tests {
    use super::*;
    use crate::experiment::ScalarSummary;

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
        struct MaxMean;
        impl SelectionPolicy<ScalarSummary> for MaxMean {
            fn compare(&self, left: &ScalarSummary, right: &ScalarSummary) -> Ordering {
                left.mean
                    .partial_cmp(&right.mean)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let mut generated = 0_u64;
        let generator = move |_: &mut TrialRng| {
            generated += 1;
            generated
        };
        let mut search = SeededRandomSearch::new(MasterSeed::from_u64(4), 1, generator, MaxMean);
        let asked = search.ask(1);
        let selecting = ScalarSummary {
            count: 2,
            mean: 10.0,
            sample_variance: Some(2.0),
            standard_error: Some(1.0),
            min: 9.0,
            max: 11.0,
        };
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
        assert_eq!(actual.count, selecting.count);
        assert_f64_bits_eq(actual.mean, selecting.mean);
        assert_eq!(
            actual.sample_variance.map(f64::to_bits),
            selecting.sample_variance.map(f64::to_bits)
        );
        assert_eq!(
            actual.standard_error.map(f64::to_bits),
            selecting.standard_error.map(f64::to_bits)
        );
        assert_f64_bits_eq(actual.min, selecting.min);
        assert_f64_bits_eq(actual.max, selecting.max);
    }
}
