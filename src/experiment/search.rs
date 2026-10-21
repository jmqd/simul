//! Tokenized deterministic ask/tell search strategies.

use super::{CandidateId, CandidateSummary, MasterSeed, SeedDeriver, SelectionPolicy, TrialRng};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};

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
    pending: BTreeMap<ProposalToken, C>,
    /// Tokens already incorporated by `tell`.
    completed_tokens: HashSet<ProposalToken>,
    /// Candidate identities already incorporated by `tell`.
    completed_candidates: HashSet<CandidateId>,
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
            pending: BTreeMap::new(),
            completed_tokens: HashSet::new(),
            completed_candidates: HashSet::new(),
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
        let check_batch_duplicates = evaluated.len() > 1;
        let mut batch_tokens =
            check_batch_duplicates.then(|| HashSet::with_capacity(evaluated.len()));
        let mut batch_candidates =
            check_batch_duplicates.then(|| HashSet::with_capacity(evaluated.len()));
        for told in evaluated {
            if told.candidate != told.summary.candidate {
                return Err(TellError::CandidateMismatch {
                    told: told.candidate,
                    summary: told.summary.candidate,
                });
            }
            if self.completed_tokens.contains(&told.token)
                || batch_tokens
                    .as_mut()
                    .is_some_and(|tokens| !tokens.insert(told.token))
            {
                return Err(TellError::DuplicateToken(told.token));
            }
            if !self.pending.contains_key(&told.token) {
                return Err(TellError::UnknownToken(told.token));
            }
            if self.completed_candidates.contains(&told.candidate)
                || batch_candidates
                    .as_mut()
                    .is_some_and(|candidates| !candidates.insert(told.candidate))
            {
                return Err(TellError::DuplicateCandidate(told.candidate));
            }
        }
        Ok(())
    }

    /// Incorporates one validated result into the strategy state.
    fn incorporate_tell(&mut self, told: &Told<S>) -> Result<(), TellError> {
        let candidate = self
            .pending
            .remove(&told.token)
            .ok_or(TellError::UnknownToken(told.token))?;
        self.consider(candidate, told);
        self.completed_tokens.insert(told.token);
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
        for _ in 0..count {
            let token = ProposalToken::new(self.next_token);
            let mut rng = self.deriver.proposal_seed(token.value()).rng();
            let candidate = (self.generator)(&mut rng);
            self.pending.insert(token, candidate.clone());
            asked.push(Asked { token, candidate });
            self.next_token += 1;
        }
        asked
    }

    fn tell(&mut self, evaluated: &[Told<S>]) -> Result<(), TellError> {
        self.validate_tells(evaluated)?;
        if let [told] = evaluated {
            return self.incorporate_tell(told);
        }

        let mut canonical: Vec<_> = evaluated.iter().collect();
        canonical.sort_by_key(|told| told.candidate);
        for told in canonical {
            self.incorporate_tell(told)?;
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

    #[test]
    fn duplicate_candidate_tell_rejects_the_whole_batch() {
        let mut search = SeededRandomSearch::new(MasterSeed::from_u64(1), 2, |_| 7_u64, MaxI32);
        let asked = search.ask(2);
        let result = search.tell(&[
            Told {
                token: asked[0].token,
                candidate: CandidateId::new(3),
                summary: summary(3, 1),
            },
            Told {
                token: asked[1].token,
                candidate: CandidateId::new(3),
                summary: summary(3, 2),
            },
        ]);
        assert_eq!(
            result,
            Err(TellError::DuplicateCandidate(CandidateId::new(3)))
        );
        assert_eq!(search.status().told, 0);
        assert_eq!(search.status().outstanding, 2);
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
    fn zero_budget_and_bounded_sampling_are_defined() {
        let mut empty = SeededRandomSearch::new(MasterSeed::from_u64(1), 0, |_| 0_u64, MaxI32);
        assert_eq!(0, empty.ask(10).len());
        assert!(empty.status().finished);

        let sampled = sample_bounded(MasterSeed::from_u64(5), 3, &[(0.0, 1.0), (-2.0, 2.0)])
            .expect("finite ordered bounds should sample");
        assert_eq!(sampled.len(), 3);
        assert!(sampled.iter().all(|candidate| {
            (0.0..1.0).contains(&candidate[0]) && (-2.0..2.0).contains(&candidate[1])
        }));
        assert!(sampled[0]
            .iter()
            .zip(&sampled[1])
            .any(|(&left, &right)| left.to_bits() != right.to_bits()));
        let repeated = sample_bounded(MasterSeed::from_u64(5), 3, &[(0.0, 1.0), (-2.0, 2.0)])
            .expect("finite ordered bounds should sample");
        assert_eq!(sampled.len(), repeated.len());
        for (actual_candidate, expected_candidate) in sampled.iter().zip(&repeated) {
            assert_eq!(actual_candidate.len(), expected_candidate.len());
            for (&actual, &expected) in actual_candidate.iter().zip(expected_candidate) {
                assert_f64_bits_eq(actual, expected);
            }
        }
        assert_eq!(
            sample_bounded(MasterSeed::from_u64(5), 1, &[(f64::NAN, 1.0)]),
            Err(BoundsError::NonFinite { dimension: 0 })
        );
    }

    #[test]
    fn bounded_sampling_handles_overflowing_widths_deterministically() {
        let seed = MasterSeed::from_u64(42);
        let bounds = [(-f64::MAX, f64::MAX), (-f64::MAX, f64::MAX / 2.0)];
        let sampled = sample_bounded(seed, 16, &bounds).expect("finite bounds should sample");
        let repeated = sample_bounded(seed, 16, &bounds).expect("finite bounds should sample");
        for (candidate, repeated_candidate) in sampled.iter().zip(&repeated) {
            for ((&actual, &expected), &(lower, upper)) in
                candidate.iter().zip(repeated_candidate).zip(&bounds)
            {
                assert!(actual.is_finite());
                assert!((lower..=upper).contains(&actual));
                assert_f64_bits_eq(actual, expected);
            }
        }
    }

    #[test]
    #[allow(clippy::suboptimal_flops)]
    fn bounded_sampling_preserves_representable_width_sequences() {
        let seed = MasterSeed::from_u64(42);
        let bounds = [(-1.3, 2.7), (-f64::MAX, 0.0), (f64::MAX, f64::MAX)];
        let sampled = sample_bounded(seed, 16, &bounds).expect("finite bounds should sample");
        let deriver = SeedDeriver::new(seed);
        for (token, candidate) in sampled.iter().enumerate() {
            for (dimension, (&actual, &(lower, upper))) in candidate.iter().zip(&bounds).enumerate()
            {
                let mut rng = deriver
                    .proposal_dimension_seed(
                        u64::try_from(token).expect("small token fits"),
                        u64::try_from(dimension).expect("small dimension fits"),
                    )
                    .rng();
                let expected = lower + (upper - lower) * rng.uniform_f64();
                assert!(actual.is_finite());
                assert!((lower..=upper).contains(&actual));
                assert_f64_bits_eq(actual, expected);
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
