//! Seeded randomized property tests for documented `experiment` laws.
//!
//! Every case is regenerated from its seed, so a failure names a case that
//! reproduces it.
#![allow(clippy::missing_docs_in_private_items)]

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashSet;
use simul::experiment::search::{
    sample_bounded, BoundsError, ProposalToken, SearchStrategy, SeededRandomSearch, TellError, Told,
};
use simul::experiment::{
    aggregate_bernoulli, aggregate_scalar, retain_records, run_serial, run_serial_bounded,
    select_best, simulated_annealing_search_with_rng, AggregateError, BernoulliReducer,
    CandidateId, CandidateSummary, CrossEntropyConfig, CrossEntropyDimension, CrossEntropyError,
    CrossEntropyOptimizer, CrossEntropySample, FailurePolicy, Feasible, FeasibleFirst, FiniteValue,
    MasterSeed, MaximizeBernoulliRate, MaximizeFinite, MaximizeWilsonLowerBound, NonFinitePolicy,
    ReduceError, ReplicationId, RetentionPolicy, RunError, ScalarReducer, ScalarSummary,
    SelectionPolicy, TrialContext, TrialId, TrialRecord, TrialRng, TrialSpec, TrialStream,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn shuffle<T>(rng: &mut ChaCha8Rng, items: &mut [T]) {
    for i in (1..items.len()).rev() {
        let j = rng.random_range(0..=i);
        items.swap(i, j);
    }
}

fn master(rng: &mut ChaCha8Rng) -> MasterSeed {
    let mut bytes = [0_u8; 32];
    for byte in &mut bytes {
        *byte = rng.random();
    }
    MasterSeed::new(bytes)
}

fn index(value: u64) -> usize {
    usize::try_from(value).unwrap_or_else(|_| panic!("{value} fits usize"))
}

fn finite(value: f64) -> FiniteValue {
    FiniteValue::new(value).unwrap_or_else(|error| panic!("{value} is finite: {error:?}"))
}

/// Orders two values that must not be NaN.
fn ordered(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right)
        .unwrap_or_else(|| panic!("{left} and {right} are ordered"))
}

fn serial<C, O, E>(
    case: u64,
    seed: MasterSeed,
    trials: impl IntoIterator<Item = TrialSpec<C>>,
    evaluate: impl FnMut(&C, &TrialContext) -> Result<O, E>,
) -> Vec<TrialRecord<O, E>> {
    run_serial(seed, trials, evaluate)
        .unwrap_or_else(|error| panic!("case {case}: valid serial batch rejected: {error:?}"))
}

type Out = [u64; 3];

fn eval(value: u64, ctx: &TrialContext) -> Result<Out, u64> {
    if value % 5 == 0 {
        return Err(value);
    }
    Ok([
        ctx.common_rng(TrialStream::TrialNoise, "c").next_u64(),
        ctx.independent_rng(TrialStream::LayoutNoise, "i")
            .next_u64(),
        value,
    ])
}

fn random_batch(rng: &mut ChaCha8Rng, max: usize) -> Vec<TrialSpec<u64>> {
    let len = rng.random_range(0..=max);
    let mut seen = FxHashSet::default();
    let mut out = Vec::new();
    while out.len() < len {
        let candidate: u64 = rng.random_range(0..8);
        let replication: u32 = rng.random_range(0..8);
        if seen.insert((candidate, replication)) {
            let group: u64 = rng.random_range(0..3);
            out.push(TrialSpec::new(
                CandidateId::new(candidate),
                ReplicationId::new(replication),
                group,
                rng.random(),
            ));
        }
    }
    out
}

// Law: same master seed and inputs give identical, canonically ordered records,
// independent of input order; evaluation happens in canonical order; replay of
// a single trial reproduces its record; common streams depend only on
// (group, replication), independent streams differ across candidates.
#[test]
fn serial_runs_are_deterministic_canonical_replayable_and_stream_separated() {
    for case in 0..300_u64 {
        let mut r = rng(case);
        let seed = master(&mut r);
        let batch = random_batch(&mut r, 40);
        let mut calls = Vec::new();
        let first = serial(case, seed, batch.clone(), |v, ctx| {
            calls.push(ctx.key());
            eval(*v, ctx)
        });
        let second = serial(case, seed, batch.clone(), |v, ctx| eval(*v, ctx));
        let mut shuffled = batch.clone();
        shuffle(&mut r, &mut shuffled);
        let third = serial(case, seed, shuffled, |v, ctx| eval(*v, ctx));
        assert_eq!(first, second, "case {case}");
        assert_eq!(first, third, "case {case}");
        assert_eq!(first.len(), batch.len(), "case {case}");
        let keys: Vec<TrialId> = first.iter().map(|rec| rec.key).collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]), "case {case}");
        assert_eq!(calls, keys, "case {case}");
        let mut common: BTreeMap<(u64, u32), u64> = BTreeMap::new();
        let mut independent = BTreeSet::new();
        for rec in &first {
            assert_eq!(rec.replay.trial(), rec.key);
            assert_eq!(rec.replay.master_seed(), seed);
            let spec = TrialSpec::new(
                rec.key.candidate(),
                rec.key.replication(),
                rec.replay.common_random_group(),
                match &rec.result {
                    Ok(out) => out[2],
                    Err(value) => *value,
                },
            );
            let replayed = serial(case, rec.replay.master_seed(), [spec], |v, ctx| {
                eval(*v, ctx)
            });
            assert_eq!(replayed, std::slice::from_ref(rec), "case {case}");
            if let Ok(out) = rec.result {
                let key = (
                    rec.replay.common_random_group(),
                    rec.key.replication().value(),
                );
                assert_eq!(*common.entry(key).or_insert(out[0]), out[0], "case {case}");
                assert!(
                    independent.insert(out[1]),
                    "case {case}: independent collision"
                );
            }
        }
    }
}

// Law: oversized and duplicate batches are rejected before any evaluation.
#[test]
fn serial_rejects_oversized_and_duplicate_batches_before_evaluation() {
    for case in 0..300_u64 {
        let mut r = rng(10_000 + case);
        let mut batch = random_batch(&mut r, 20);
        let mut duplicated = None;
        if !batch.is_empty() && r.random_bool(0.5) {
            let pick = batch[r.random_range(0..batch.len())].clone();
            duplicated = Some(pick.key());
            batch.push(pick);
            shuffle(&mut r, &mut batch);
        }
        let limit = r.random_range(1..=batch.len() + 2);
        let mut evaluations = 0;
        let result = run_serial_bounded(
            MasterSeed::from_u64(case),
            batch.clone(),
            NonZeroUsize::new(limit).unwrap_or_else(|| panic!("case {case}: limit is nonzero")),
            |v, ctx| {
                evaluations += 1;
                eval(*v, ctx)
            },
        );
        if batch.len() > limit {
            assert!(
                matches!(result, Err(RunError::BatchLimitExceeded { limit: l }) if l == limit),
                "case {case}"
            );
            assert_eq!(evaluations, 0);
        } else if let Some(key) = duplicated {
            assert!(
                matches!(result, Err(RunError::DuplicateTrial(k)) if k == key),
                "case {case}"
            );
            assert_eq!(evaluations, 0);
        } else {
            let records =
                result.unwrap_or_else(|error| panic!("case {case}: valid batch: {error:?}"));
            assert_eq!(records.len(), batch.len());
        }
    }
}

// Law: retention keeps exactly the documented subsequence in order.
#[test]
fn retention_keeps_exactly_the_policy_subsequence_in_order() {
    for case in 0..300_u64 {
        let mut r = rng(20_000 + case);
        let batch = random_batch(&mut r, 30);
        let records = serial(case, MasterSeed::from_u64(case), batch, |v, ctx| {
            eval(*v, ctx)
        });
        let summaries = retain_records(records.clone(), RetentionPolicy::SummariesOnly);
        assert_eq!(summaries.len(), 0, "case {case}");
        let default = retain_records(records.clone(), RetentionPolicy::default());
        assert_eq!(default.len(), 0, "case {case}");
        let failures = retain_records(records.clone(), RetentionPolicy::FailuresOnly);
        let expected: Vec<_> = records.iter().filter(|rec| rec.result.is_err()).collect();
        assert_eq!(failures.len(), expected.len());
        for (got, want) in failures.iter().zip(expected) {
            assert_eq!(
                (got.key, got.replay, &got.result),
                (want.key, want.replay, &want.result)
            );
        }
        let all = retain_records(records.clone(), RetentionPolicy::All);
        assert_eq!(all.len(), records.len());
        for (got, want) in all.iter().zip(&records) {
            assert_eq!(
                (got.key, got.replay, &got.result),
                (want.key, want.replay, &want.result)
            );
        }
    }
}

fn scalar_value(r: &mut ChaCha8Rng) -> f64 {
    match r.random_range(0..10) {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => f64::NEG_INFINITY,
        3 => f64::MAX,
        4 => -f64::MAX,
        5 => 0.0,
        6 => -0.0,
        _ => r.random_range(-100.0..100.0),
    }
}

fn scalar_records(r: &mut ChaCha8Rng, candidate: u64) -> Vec<TrialRecord<f64, u8>> {
    let len = r.random_range(0..12);
    let seed = MasterSeed::from_u64(candidate);
    let specs: Vec<_> = (0..len)
        .map(|rep| TrialSpec::new(CandidateId::new(candidate), ReplicationId::new(rep), 0, ()))
        .collect();
    let mut values: Vec<Result<f64, u8>> = (0..len)
        .map(|_| {
            if r.random_bool(0.25) {
                Err(7)
            } else {
                Ok(scalar_value(r))
            }
        })
        .collect();
    values.reverse();
    serial(candidate, seed, specs, |(), _| {
        values
            .pop()
            .unwrap_or_else(|| panic!("one value per trial"))
    })
}

fn count32(count: usize) -> u32 {
    u32::try_from(count).unwrap_or_else(|_| panic!("{count} fits u32"))
}

/// Reference scalar aggregation, applying the documented policy precedence.
fn expected_scalar(
    records: &[TrialRecord<f64, u8>],
    failure_policy: FailurePolicy<FiniteValue>,
    non_finite_policy: NonFinitePolicy,
    observed: &[f64],
) -> Result<ScalarSummary, AggregateError> {
    let failed = records.iter().filter(|rec| rec.result.is_err()).count();
    let non_finite = records
        .iter()
        .filter(|rec| matches!(rec.result, Ok(v) if !v.is_finite()))
        .count();
    if records.is_empty() {
        Err(AggregateError::Empty)
    } else if failed > 0 && failure_policy == FailurePolicy::RejectCandidate {
        Err(AggregateError::EvaluationFailed {
            failures: count32(failed),
        })
    } else if non_finite > 0 && non_finite_policy == NonFinitePolicy::RejectCandidate {
        Err(AggregateError::NonFinite {
            observations: count32(non_finite),
        })
    } else {
        match ScalarReducer::from_results(observed.iter().copied()) {
            Ok(stats) => Ok(stats),
            Err(ReduceError::Empty) => Err(AggregateError::NoUsableObservations),
            Err(_) => Err(AggregateError::ArithmeticOverflow),
        }
    }
}

/// Checks Bernoulli aggregation counts, bounds, and order independence.
fn check_bernoulli(r: &mut ChaCha8Rng, records: &[TrialRecord<f64, u8>], case: u64) {
    let failed = records.iter().filter(|rec| rec.result.is_err()).count();
    let completed = records.len() - failed;
    let policy = match r.random_range(0..4) {
        0 => FailurePolicy::RejectCandidate,
        1 => FailurePolicy::Ignore,
        2 => FailurePolicy::ReplaceWith(true),
        _ => FailurePolicy::ReplaceWith(false),
    };
    let classify = |v: &f64| *v > 0.0;
    let bernoulli = aggregate_bernoulli(records, classify, policy);
    let mut shuffled = records.to_vec();
    shuffle(r, &mut shuffled);
    assert_eq!(
        bernoulli,
        aggregate_bernoulli(&shuffled, classify, policy),
        "case {case}"
    );
    let replaced = if matches!(policy, FailurePolicy::ReplaceWith(_)) {
        failed
    } else {
        0
    };
    let successes = records
        .iter()
        .filter(|rec| matches!(rec.result, Ok(v) if v > 0.0))
        .count()
        + if policy == FailurePolicy::ReplaceWith(true) {
            failed
        } else {
            0
        };
    let attempts = completed + replaced;
    match bernoulli {
        Ok(summary) => {
            assert_eq!(summary.requested, summary.completed + summary.failed);
            assert_eq!(summary.failed as usize, failed);
            assert_eq!(summary.statistics.attempts, attempts as u64);
            assert_eq!(summary.statistics.successes, successes as u64);
            let stats = summary.statistics;
            assert!(
                0.0 <= stats.wilson_lower && stats.wilson_lower <= stats.rate,
                "case {case}: {stats:?}"
            );
            assert!(
                stats.rate <= stats.wilson_upper && stats.wilson_upper <= 1.0,
                "case {case}: {stats:?}"
            );
        }
        Err(AggregateError::Empty) => assert_eq!(records.len(), 0, "case {case}"),
        Err(AggregateError::EvaluationFailed { failures }) => {
            assert_eq!(failures as usize, failed);
            assert_eq!(policy, FailurePolicy::RejectCandidate);
        }
        Err(AggregateError::NoUsableObservations) => assert_eq!(attempts, 0),
        Err(other) => panic!("case {case}: unexpected {other:?}"),
    }
}

// Law: requested = completed + failed; failure and non-finite policies are
// applied as documented with documented precedence; summaries are finite;
// Bernoulli aggregation is independent of record order.
#[test]
fn aggregation_counts_balance_and_policies_apply_with_documented_precedence() {
    for case in 0..600_u64 {
        let mut r = rng(30_000 + case);
        let records = scalar_records(&mut r, case);
        let failed = records.iter().filter(|rec| rec.result.is_err()).count();
        let completed = records.len() - failed;
        let penalty = finite(r.random_range(-5.0..5.0));
        let failure_policy = match r.random_range(0..3) {
            0 => FailurePolicy::RejectCandidate,
            1 => FailurePolicy::Ignore,
            _ => FailurePolicy::ReplaceWith(penalty),
        };
        let non_finite_policy = match r.random_range(0..3) {
            0 => NonFinitePolicy::RejectCandidate,
            1 => NonFinitePolicy::Ignore,
            _ => NonFinitePolicy::ReplaceWith(penalty),
        };
        let result = aggregate_scalar(&records, |v| *v, failure_policy, non_finite_policy);
        let mut observed = Vec::new();
        for rec in &records {
            match rec.result {
                Err(_) => {
                    if let FailurePolicy::ReplaceWith(v) = failure_policy {
                        observed.push(v.get());
                    }
                }
                Ok(v) if v.is_finite() => observed.push(v),
                Ok(_) => {
                    if let NonFinitePolicy::ReplaceWith(v) = non_finite_policy {
                        observed.push(v.get());
                    }
                }
            }
        }
        let expected = expected_scalar(&records, failure_policy, non_finite_policy, &observed);
        match (result, expected) {
            (Ok(summary), Ok(expected)) => {
                assert_eq!(summary.candidate, CandidateId::new(case));
                assert_eq!(summary.requested as usize, records.len(), "case {case}");
                assert_eq!(
                    summary.requested,
                    summary.completed + summary.failed,
                    "case {case}"
                );
                assert_eq!(summary.completed as usize, completed);
                assert_eq!(summary.statistics, expected, "case {case}");
                let stats = summary.statistics;
                assert!(
                    stats.mean.is_finite() && stats.min.is_finite() && stats.max.is_finite(),
                    "case {case}"
                );
                assert!(
                    stats.min <= stats.mean && stats.mean <= stats.max,
                    "case {case}: {stats:?}"
                );
                assert!(
                    stats.sample_variance.is_none_or(f64::is_finite),
                    "case {case}"
                );
                assert!(
                    stats.standard_error.is_none_or(f64::is_finite),
                    "case {case}"
                );
                assert_eq!(stats.count, observed.len() as u64);
            }
            (got, want) => assert_eq!(got.map(|_| ()), want.map(|_| ()), "case {case}"),
        }

        check_bernoulli(&mut r, &records, case);
    }
}

// Law: aggregating records of more than one candidate is rejected.
#[test]
fn aggregation_rejects_records_from_mixed_candidates() {
    for case in 0..200_u64 {
        let mut r = rng(35_000 + case);
        let mut records = scalar_records(&mut r, 1);
        let other = scalar_records(&mut r, 2);
        if records.is_empty() || other.is_empty() {
            continue;
        }
        records.push(other[0].clone());
        shuffle(&mut r, &mut records);
        assert_eq!(
            aggregate_bernoulli(&records, |v| *v > 0.0, FailurePolicy::Ignore),
            Err(AggregateError::MixedCandidates)
        );
        assert_eq!(
            aggregate_scalar(
                &records,
                |v| *v,
                FailurePolicy::Ignore,
                NonFinitePolicy::Ignore
            )
            .map(|_| ()),
            Err(AggregateError::MixedCandidates)
        );
    }
}

fn expected_best<A>(
    candidates: &[(CandidateId, A)],
    compare: impl Fn(&A, &A) -> Ordering,
) -> Option<CandidateId> {
    let mut best: Option<&(CandidateId, A)> = None;
    for candidate in candidates {
        best = Some(
            best.map_or(candidate, |best| match compare(&candidate.1, &best.1) {
                Ordering::Greater => candidate,
                Ordering::Equal if candidate.0 < best.0 => candidate,
                Ordering::Less | Ordering::Equal => best,
            }),
        );
    }
    best.map(|best| best.0)
}

// Law: select_best is deterministic and breaks ties by lower candidate ID
// regardless of input order.
#[test]
fn select_best_is_order_independent_and_breaks_ties_by_lower_id() {
    for case in 0..500_u64 {
        let mut r = rng(40_000 + case);
        let len = r.random_range(0..20);
        let mut ids: Vec<u64> = (0..100).collect();
        shuffle(&mut r, &mut ids);
        let finite_values: Vec<(CandidateId, FiniteValue)> = (0..len)
            .map(|i| {
                let v = [0.0, -0.0, 1.0, 2.0, -1.0][r.random_range(0..5)];
                (CandidateId::new(ids[i]), finite(v))
            })
            .collect();
        let want = expected_best(&finite_values, |a, b| ordered(a.get(), b.get()));
        // Reference: max value, then lowest id.
        let reference = finite_values
            .iter()
            .max_by(|a, b| ordered(a.1.get(), b.1.get()).then(b.0.cmp(&a.0)))
            .map(|c| c.0);
        assert_eq!(want, reference);
        for _ in 0..4 {
            let mut perm = finite_values.clone();
            shuffle(&mut r, &mut perm);
            assert_eq!(
                select_best(&perm, &MaximizeFinite).map(|b| b.0),
                reference,
                "case {case}"
            );
        }

        // Bernoulli summaries from real reductions, plus feasible-first wrapper.
        let bernoulli: Vec<_> = (0..len)
            .map(|i| {
                let attempts = r.random_range(1..6);
                let successes = r.random_range(0..=attempts);
                let summary = BernoulliReducer::from_results((0..attempts).map(|k| k < successes))
                    .unwrap_or_else(|error| panic!("case {case}: nonempty: {error:?}"));
                (CandidateId::new(ids[i]), summary)
            })
            .collect();
        let rate_reference = bernoulli
            .iter()
            .max_by(|a, b| ordered(a.1.rate, b.1.rate).then(b.0.cmp(&a.0)))
            .map(|c| c.0);
        let wilson_reference = bernoulli
            .iter()
            .max_by(|a, b| ordered(a.1.wilson_lower, b.1.wilson_lower).then(b.0.cmp(&a.0)))
            .map(|c| c.0);
        let feasible: Vec<(CandidateId, Feasible<FiniteValue, FiniteValue>)> = finite_values
            .iter()
            .map(|(id, v)| {
                if r.random_bool(0.5) {
                    (*id, Feasible::Feasible(*v))
                } else {
                    (*id, Feasible::Infeasible(*v))
                }
            })
            .collect();
        let policy = FeasibleFirst {
            objective: MaximizeFinite,
            violation: MaximizeFinite,
        };
        let feasible_reference = feasible
            .iter()
            .max_by(|a, b| policy.compare(&a.1, &b.1).then(b.0.cmp(&a.0)))
            .map(|c| c.0);
        for _ in 0..4 {
            let mut perm = bernoulli.clone();
            shuffle(&mut r, &mut perm);
            assert_eq!(
                select_best(&perm, &MaximizeBernoulliRate).map(|b| b.0),
                rate_reference,
                "case {case}"
            );
            assert_eq!(
                select_best(&perm, &MaximizeWilsonLowerBound).map(|b| b.0),
                wilson_reference,
                "case {case}"
            );
            let mut perm = feasible.clone();
            shuffle(&mut r, &mut perm);
            assert_eq!(
                select_best(&perm, &policy).map(|b| b.0),
                feasible_reference,
                "case {case}"
            );
        }
    }
}

#[derive(Clone, Copy)]
struct MaxI32;
impl SelectionPolicy<i32> for MaxI32 {
    fn compare(&self, left: &i32, right: &i32) -> Ordering {
        left.cmp(right)
    }
}

const fn told(token: u64, candidate: u64, score: i32) -> Told<i32> {
    Told {
        token: ProposalToken::new(token),
        candidate: CandidateId::new(candidate),
        summary: CandidateSummary {
            candidate: CandidateId::new(candidate),
            requested: 1,
            completed: 1,
            failed: 0,
            statistics: score,
        },
    }
}

fn generate(rng: &mut TrialRng) -> u64 {
    rng.next_u64()
}

type RandomSearch = SeededRandomSearch<u64, i32, MaxI32, fn(&mut TrialRng) -> u64>;

/// What a `RandomSearch` has been asked and told, tracked independently.
#[derive(Default)]
struct SearchModel {
    /// Proposed candidate by token.
    proposals: BTreeMap<u64, u64>,
    outstanding: BTreeSet<u64>,
    told_tokens: BTreeSet<u64>,
    used_ids: BTreeSet<u64>,
    /// `(candidate id, score, token)` of every accepted tell.
    told_scores: Vec<(u64, i32, u64)>,
    next_id: u64,
}

fn nth(set: &BTreeSet<u64>, index: usize) -> u64 {
    set.iter()
        .copied()
        .nth(index)
        .unwrap_or_else(|| panic!("index {index} is within the set"))
}

/// Half the time, turns a valid batch invalid with exactly one fault and
/// returns the error that fault must produce.
fn inject_fault(
    r: &mut ChaCha8Rng,
    batch: &mut Vec<Told<i32>>,
    model: &mut SearchModel,
    budget: u64,
    asked: u64,
) -> Option<TellError> {
    if !r.random_bool(0.5) {
        return None;
    }
    let fault = r.random_range(0..6);
    let at = if batch.is_empty() {
        0
    } else {
        r.random_range(0..batch.len())
    };
    model.next_id += 1;
    let fresh = model.next_id;
    match fault {
        0 => {
            let token = (budget + r.random_range(0..5) + asked).max(asked);
            batch.insert(at.min(batch.len()), told(token, fresh, 1));
            Some(TellError::UnknownToken(ProposalToken::new(token)))
        }
        1 if !model.told_tokens.is_empty() => {
            let token = nth(
                &model.told_tokens,
                r.random_range(0..model.told_tokens.len()),
            );
            batch.insert(at.min(batch.len()), told(token, fresh, 1));
            Some(TellError::DuplicateToken(ProposalToken::new(token)))
        }
        2 if !batch.is_empty() => {
            let token = batch[at].token;
            batch.push(told(token.value(), fresh, 1));
            Some(TellError::DuplicateToken(token))
        }
        3 if !model.used_ids.is_empty() && !batch.is_empty() => {
            let id = nth(&model.used_ids, r.random_range(0..model.used_ids.len()));
            let token = batch[at].token.value();
            batch[at] = told(token, id, 1);
            Some(TellError::DuplicateCandidate(CandidateId::new(id)))
        }
        4 if batch.len() >= 2 => {
            let id = batch[0].candidate;
            let last = batch.len() - 1;
            let token = batch[last].token.value();
            batch[last] = told(token, id.value(), 1);
            Some(TellError::DuplicateCandidate(id))
        }
        5 if !batch.is_empty() => {
            batch[at].summary.candidate = CandidateId::new(fresh);
            Some(TellError::CandidateMismatch {
                told: batch[at].candidate,
                summary: CandidateId::new(fresh),
            })
        }
        _ => None,
    }
}

/// Asks up to a random limit and checks proposals against an all-at-once ask.
fn ask_step(
    r: &mut ChaCha8Rng,
    search: &mut RandomSearch,
    model: &mut SearchModel,
    all: &[simul::experiment::search::Asked<u64>],
    budget: u64,
    case: u64,
    step: u32,
) {
    let limit = r.random_range(0..8);
    let before = search.status().asked;
    let asked = search.ask(limit);
    let expected = index((budget - before).min(limit as u64));
    assert_eq!(asked.len(), expected, "case {case} step {step}");
    for (offset, proposal) in asked.iter().enumerate() {
        let token = proposal.token.value();
        assert_eq!(token, before + offset as u64);
        assert_eq!(*proposal, all[index(token)], "case {case}");
        model.proposals.insert(token, proposal.candidate);
        model.outstanding.insert(token);
    }
}

/// Tells a batch of outstanding results, possibly with one fault, and checks
/// that a rejected batch changes nothing.
fn tell_step(
    r: &mut ChaCha8Rng,
    search: &mut RandomSearch,
    model: &mut SearchModel,
    budget: u64,
    case: u64,
    step: u32,
) {
    let mut chosen: Vec<u64> = model.outstanding.iter().copied().collect();
    let len = if chosen.is_empty() {
        0
    } else {
        r.random_range(0..=chosen.len().min(24))
    };
    shuffle(r, &mut chosen);
    chosen.truncate(len);
    let mut batch: Vec<Told<i32>> = chosen
        .iter()
        .map(|&token| {
            model.next_id += 1 + r.random_range(0..3);
            told(token, model.next_id, r.random_range(0..4))
        })
        .collect();
    let expected_error = inject_fault(r, &mut batch, model, budget, search.status().asked);
    let status = search.status();
    let best = search.best().cloned();
    let result = search.tell(&batch);
    if let Some(error) = expected_error {
        assert_eq!(
            result,
            Err(error),
            "case {case} step {step} batch {}",
            batch.len()
        );
        assert_eq!(status, search.status(), "case {case} step {step}");
        assert_eq!(best, search.best().cloned(), "case {case} step {step}");
    } else {
        assert_eq!(result, Ok(()), "case {case} step {step}");
        for entry in &batch {
            let token = entry.token.value();
            model.outstanding.remove(&token);
            model.told_tokens.insert(token);
            model.used_ids.insert(entry.candidate.value());
            model
                .told_scores
                .push((entry.candidate.value(), entry.summary.statistics, token));
        }
    }
}

/// Checks the search's counters and best against the model.
fn check_search(search: &RandomSearch, model: &SearchModel, budget: u64, case: u64, step: u32) {
    let status = search.status();
    assert_eq!(status.budget, budget);
    assert!(status.asked <= budget, "case {case}");
    assert_eq!(
        status.asked,
        status.told + status.outstanding,
        "case {case}"
    );
    assert_eq!(
        status.outstanding,
        model.outstanding.len() as u64,
        "case {case}"
    );
    assert_eq!(status.told, model.told_tokens.len() as u64, "case {case}");
    assert_eq!(
        status.finished,
        status.asked == budget && status.outstanding == 0,
        "case {case}"
    );
    let want = model
        .told_scores
        .iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
        .copied();
    match (search.best(), want) {
        (None, None) => {}
        (Some(best), Some((id, score, token))) => {
            assert_eq!(
                best.candidate_id,
                CandidateId::new(id),
                "case {case} step {step}"
            );
            assert_eq!(best.summary.statistics, score);
            assert_eq!(best.candidate, model.proposals[&token]);
        }
        (got, want) => panic!("case {case}: best {got:?} vs {want:?}"),
    }
}

// Law: RandomSearch never exceeds budget, counters stay consistent through any
// sequence of valid and invalid tells, an invalid batch changes nothing, batch
// size does not change proposals, and the winner is the max score with lowest
// candidate ID regardless of tell order.
#[test]
fn random_search_keeps_budget_rejects_bad_tells_atomically_and_tracks_best() {
    for case in 0..400_u64 {
        let mut r = rng(50_000 + case);
        let seed = MasterSeed::from_u64(case);
        let budget: u64 = r.random_range(0..40);
        let mut search: RandomSearch = SeededRandomSearch::new(seed, budget, generate, MaxI32);
        let mut reference: RandomSearch = SeededRandomSearch::new(seed, budget, generate, MaxI32);
        let all = reference.ask(index(budget));
        let mut model = SearchModel::default();
        for step in 0..60 {
            match r.random_range(0..4) {
                0 => ask_step(&mut r, &mut search, &mut model, &all, budget, case, step),
                1 | 2 => tell_step(&mut r, &mut search, &mut model, budget, case, step),
                _ => {
                    let status = search.status();
                    let empty: [Told<i32>; 0] = [];
                    assert_eq!(search.tell(&empty), Ok(()));
                    assert_eq!(status, search.status());
                }
            }
            check_search(&search, &model, budget, case, step);
        }
    }
}

fn bound_value(r: &mut ChaCha8Rng) -> f64 {
    match r.random_range(0..9) {
        0 => f64::MAX,
        1 => -f64::MAX,
        2 => f64::MIN_POSITIVE,
        3 => -0.0,
        4 => 0.0,
        5 => r.random_range(-1.0..1.0) * 10f64.powi(r.random_range(-300..300)),
        6 => 1.0e16 + f64::from(r.random_range(0..8_u8) * 2),
        _ => r.random_range(-10.0..10.0),
    }
}

// Law: sample_bounded is deterministic, returns budget × dims values inside
// closed finite bounds, and reports the first invalid dimension.
#[test]
fn sample_bounded_is_deterministic_prefix_stable_and_within_bounds() {
    for case in 0..500_u64 {
        let mut r = rng(60_000 + case);
        let dims = r.random_range(0..6);
        let mut bounds: Vec<(f64, f64)> = (0..dims)
            .map(|_| {
                let a = bound_value(&mut r);
                let b = if r.random_bool(0.1) {
                    a
                } else {
                    bound_value(&mut r)
                };
                if a <= b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect();
        let budget = r.random_range(0..40);
        let seed = master(&mut r);
        let sample = |budget| {
            sample_bounded(seed, budget, &bounds)
                .unwrap_or_else(|error| panic!("case {case}: valid bounds: {error:?}"))
        };
        let sampled = sample(budget);
        assert_eq!(sampled, sample(budget));
        assert_eq!(sampled.len(), budget);
        for point in &sampled {
            assert_eq!(point.len(), dims);
            for (&x, &(lo, hi)) in point.iter().zip(&bounds) {
                assert!(
                    x.is_finite() && lo <= x && x <= hi,
                    "case {case}: {x:e} not in [{lo:e}, {hi:e}]"
                );
            }
        }
        // A smaller budget gives a prefix.
        let half = budget.div_euclid(2);
        assert_eq!(&sampled[..half], &sample(half)[..]);
        // Invalid bound reporting.
        if dims > 0 {
            let at = r.random_range(0..dims);
            let bad = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY][r.random_range(0..3)];
            let (lo, hi) = bounds[at];
            if r.random_bool(0.5) {
                bounds[at] = (bad, hi);
                assert_eq!(
                    sample_bounded(seed, 1, &bounds),
                    Err(BoundsError::NonFinite { dimension: at })
                );
            } else if lo < hi {
                bounds[at] = (hi, lo);
                assert_eq!(
                    sample_bounded(seed, 1, &bounds),
                    Err(BoundsError::Reversed { dimension: at })
                );
            }
        }
    }
}

fn ce_dims(r: &mut ChaCha8Rng) -> [CrossEntropyDimension; 3] {
    std::array::from_fn(|_| {
        if r.random_bool(0.5) {
            CrossEntropyDimension::Linear
        } else {
            CrossEntropyDimension::Circular
        }
    })
}

fn in_domain(x: f64, dimension: CrossEntropyDimension) -> bool {
    match dimension {
        CrossEntropyDimension::Linear => (0.0..=1.0).contains(&x),
        CrossEntropyDimension::Circular => (0.0..1.0).contains(&x),
    }
}

fn log_uniform(r: &mut ChaCha8Rng, lo: i32, hi: i32) -> f64 {
    10f64.powf(r.random_range(f64::from(lo)..f64::from(hi)))
}

fn extreme_normal(r: &mut ChaCha8Rng) -> f64 {
    match r.random_range(0..10) {
        0 => f64::MAX,
        1 => -f64::MAX,
        2 => 1.0e300,
        3 => -1.0e-300,
        4 => 0.0,
        5 => -0.0,
        _ => r.random_range(-6.0..6.0),
    }
}

fn score(r: &mut ChaCha8Rng) -> f64 {
    match r.random_range(0..12) {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => f64::NEG_INFINITY,
        3 => 0.0,
        4 => -0.0,
        5 => 1.0,
        _ => r.random_range(-3.0..3.0),
    }
}

type Bits = [u64; 3];

/// Bits of the mean, standard deviation, generation, and best `(point, score)`.
type Snapshot = (Bits, Bits, u64, Option<(Bits, u64)>);

fn snapshot(optimizer: &CrossEntropyOptimizer<3>) -> Snapshot {
    (
        optimizer.mean().map(f64::to_bits),
        optimizer.standard_deviation().map(f64::to_bits),
        optimizer.generation(),
        optimizer
            .best()
            .map(|best| (best.point.map(f64::to_bits), best.score.to_bits())),
    )
}

/// A seeded optimizer plus the configuration the laws check it against.
struct CrossEntropyCase {
    case: u64,
    r: ChaCha8Rng,
    dims: [CrossEntropyDimension; 3],
    floor: [f64; 3],
    elite_fraction: f64,
    optimizer: CrossEntropyOptimizer<3>,
    best_seen: Option<f64>,
}

impl CrossEntropyCase {
    fn new(case: u64) -> Self {
        let mut r = rng(70_000 + case);
        let dims = ce_dims(&mut r);
        let deviation: [f64; 3] = std::array::from_fn(|_| {
            if r.random_bool(0.15) {
                f64::MAX
            } else {
                log_uniform(&mut r, -300, 300)
            }
        });
        let floor: [f64; 3] = std::array::from_fn(|i| deviation[i] * r.random_range(1.0e-6..=1.0));
        let mean: [f64; 3] = std::array::from_fn(|i| match r.random_range(0..4) {
            0 => 0.0,
            1 if dims[i] == CrossEntropyDimension::Linear => 1.0,
            _ => r.random_range(0.0..1.0),
        });
        let elite_fraction = [1.0, 0.1, 0.5, r.random_range(1.0e-3..=1.0)][r.random_range(0..4)];
        let learning_rate = [0.0, 1.0, 0.7, r.random_range(0.0..=1.0)][r.random_range(0..4)];
        let config = CrossEntropyConfig::new(mean, deviation)
            .with_dimensions(dims)
            .with_minimum_standard_deviation(floor)
            .with_elite_fraction(elite_fraction)
            .with_learning_rate(learning_rate);
        let optimizer = CrossEntropyOptimizer::new(config)
            .unwrap_or_else(|error| panic!("case {case}: valid config: {error:?}"));
        Self {
            case,
            r,
            dims,
            floor,
            elite_fraction,
            optimizer,
            best_seen: None,
        }
    }

    /// Asks a population, checking every point is in the domain.
    fn population(
        &mut self,
        ask_rng: &mut ChaCha8Rng,
        trace: &mut Vec<u64>,
        generation: u32,
    ) -> Vec<CrossEntropySample<3>> {
        let case = self.case;
        let len = self.r.random_range(1..30);
        (0..len)
            .map(|_| {
                let point = if self.r.random_bool(0.5) {
                    self.optimizer.ask(ask_rng)
                } else {
                    let normals: [f64; 3] = std::array::from_fn(|_| extreme_normal(&mut self.r));
                    self.optimizer
                        .ask_with_standard_normal(|d| normals[d])
                        .unwrap_or_else(|error| panic!("case {case}: finite normals: {error:?}"))
                };
                for (&x, &dimension) in point.iter().zip(&self.dims) {
                    assert!(
                        in_domain(x, dimension),
                        "case {case} gen {generation}: ask {point:?} dims {:?} mean {:?} sd {:?}",
                        self.dims,
                        self.optimizer.mean(),
                        self.optimizer.standard_deviation()
                    );
                }
                trace.extend(point.map(f64::to_bits));
                CrossEntropySample::new(point, score(&mut self.r))
            })
            .collect()
    }

    /// An invalid point under a valid score is rejected before mutation.
    fn check_invalid_point_rejected(&mut self, samples: &[CrossEntropySample<3>]) {
        let case = self.case;
        let before = snapshot(&self.optimizer);
        let mut bad = samples.to_vec();
        let at = self.r.random_range(0..bad.len());
        let dimension = self.r.random_range(0..3);
        bad[at].point[dimension] =
            [-1.0e-9, 1.5, f64::NAN, f64::INFINITY][self.r.random_range(0..4)];
        if bad[at].score.is_nan() {
            bad[at].score = 0.0;
        }
        let (sample, dimension) = bad
            .iter()
            .enumerate()
            .filter(|(_, sample)| !sample.score.is_nan())
            .find_map(|(index, sample)| {
                sample
                    .point
                    .iter()
                    .zip(&self.dims)
                    .position(|(&x, &dimension)| !in_domain(x, dimension))
                    .map(|dimension| (index, dimension))
            })
            .unwrap_or_else(|| panic!("case {case}: an invalid point was injected"));
        assert_eq!(
            self.optimizer.tell(&mut bad),
            Err(CrossEntropyError::InvalidSamplePoint { sample, dimension }),
            "case {case}"
        );
        assert_eq!(before, snapshot(&self.optimizer), "case {case}");
    }

    /// Tells a valid population and checks the update against the documented
    /// elite count, ordering, best tracking, domain, and floor.
    fn check_tell(&mut self, samples: &mut [CrossEntropySample<3>], trace: &mut Vec<u64>) {
        let case = self.case;
        let before = snapshot(&self.optimizer);
        let mut valid: Vec<f64> = samples
            .iter()
            .map(|sample| sample.score)
            .filter(|score| !score.is_nan())
            .collect();
        let update = self
            .optimizer
            .tell(samples)
            .unwrap_or_else(|error| panic!("case {case}: valid population: {error:?}"));
        if valid.is_empty() {
            assert_eq!(update, None, "case {case}");
            assert_eq!(before, snapshot(&self.optimizer), "case {case}");
            return;
        }
        let update = update.unwrap_or_else(|| panic!("case {case}: scored population updates"));
        valid.sort_by(|x, y| y.total_cmp(x));
        let elite = elite_count(valid.len(), self.elite_fraction);
        assert_eq!(update.valid_samples, valid.len());
        assert_eq!(update.elite_samples, elite);
        assert_eq!(update.generation, before.2 + 1);
        assert_eq!(update.generation_best_score, valid[0], "case {case}");
        let best = self.best_seen.map_or(valid[0], |best| best.max(valid[0]));
        self.best_seen = Some(best);
        assert_eq!(update.best_score, best, "case {case}");
        let optimizer_best = self
            .optimizer
            .best()
            .unwrap_or_else(|| panic!("case {case}: an update records a best"));
        assert_eq!(optimizer_best.score, best, "case {case}");
        // Elites first, by total order on scores.
        let mut elites: Vec<f64> = samples[..elite].iter().map(|sample| sample.score).collect();
        elites.sort_by(|x, y| y.total_cmp(x));
        assert_eq!(
            elites.into_iter().map(f64::to_bits).collect::<Vec<_>>(),
            valid[..elite]
                .iter()
                .copied()
                .map(f64::to_bits)
                .collect::<Vec<_>>(),
            "case {case}"
        );
        let mean = self.optimizer.mean();
        let deviation = self.optimizer.standard_deviation();
        for d in 0..3 {
            let (m, s, floor) = (mean[d], deviation[d], self.floor[d]);
            assert!(
                in_domain(m, self.dims[d]),
                "case {case}: mean {m} out of domain"
            );
            assert!(
                s >= floor && s.is_finite(),
                "case {case}: sd {s} floor {floor}"
            );
        }
        let after = snapshot(&self.optimizer);
        trace.extend(after.0);
        trace.extend(after.1);
    }
}

/// `ceil(valid × fraction)`, clamped to `1..=valid`.
fn elite_count(valid: usize, fraction: f64) -> usize {
    let target = valid as f64 * fraction;
    (1..=valid)
        .find(|&count| count as f64 >= target)
        .unwrap_or(valid)
}

/// Runs one seeded cross-entropy case and returns every asked point and
/// fitted distribution, as bits.
fn cross_entropy_trace(case: u64) -> Vec<u64> {
    let mut ce = CrossEntropyCase::new(case);
    let mut ask_rng = rng(case);
    let mut trace = Vec::new();
    for generation in 0..8 {
        let mut samples = ce.population(&mut ask_rng, &mut trace, generation);
        // All-NaN population leaves the optimizer unchanged.
        if ce.r.random_bool(0.1) {
            let before = snapshot(&ce.optimizer);
            for sample in &mut samples {
                sample.score = f64::NAN;
            }
            assert_eq!(ce.optimizer.tell(&mut samples), Ok(None), "case {case}");
            assert_eq!(before, snapshot(&ce.optimizer), "case {case}");
            continue;
        }
        if ce.r.random_bool(0.15) {
            ce.check_invalid_point_rejected(&samples);
        }
        ce.check_tell(&mut samples, &mut trace);
    }
    trace
}

// Law: cross-entropy ask stays in domain, std never drops below its floor,
// tell follows the documented elite count/ordering and best tracking, invalid
// populations leave the optimizer unchanged, and runs are deterministic.
#[test]
fn cross_entropy_stays_in_domain_above_floor_and_tells_deterministically() {
    for case in 0..500_u64 {
        assert_eq!(
            cross_entropy_trace(case),
            cross_entropy_trace(case),
            "case {case}: nondeterministic"
        );
    }
}

// Law: ask_into draws the same points as repeated ask.
#[test]
fn cross_entropy_ask_into_matches_repeated_ask() {
    for case in 0..300_u64 {
        let mut r = rng(75_000 + case);
        let dims = ce_dims(&mut r);
        let deviation: [f64; 3] = std::array::from_fn(|_| log_uniform(&mut r, -5, 300));
        let optimizer = CrossEntropyOptimizer::new(
            CrossEntropyConfig::new([0.5; 3], deviation)
                .with_dimensions(dims)
                .with_minimum_standard_deviation([1.0e-9; 3]),
        )
        .unwrap_or_else(|error| panic!("case {case}: valid config: {error:?}"));
        let mut batch_rng = rng(case);
        let mut single_rng = rng(case);
        let mut batch = [[0.0; 3]; 7];
        optimizer.ask_into(&mut batch_rng, &mut batch);
        let single: Vec<[f64; 3]> = (0..7).map(|_| optimizer.ask(&mut single_rng)).collect();
        assert_eq!(batch.to_vec(), single, "case {case}");
    }
}

// Law: simulated annealing returns a state whose score equals the maximum
// non-NaN score it evaluated (None when all are NaN), evaluates exactly
// 1 + proposal_limit states, and is deterministic for one RNG seed.
#[test]
fn annealing_returns_a_best_evaluated_state_after_exactly_limit_proposals() {
    for case in 0..400_u64 {
        let mut r = rng(80_000 + case);
        let table: Vec<f64> = (0..r.random_range(1..12)).map(|_| score(&mut r)).collect();
        let limit = r.random_range(0..40_u32);
        let temperature_kind = r.random_range(0..5);
        let run = |seed: u64| {
            let mut evaluated = Vec::new();
            let mut annealing_rng = rng(seed);
            let result = simulated_annealing_search_with_rng(
                &mut annealing_rng,
                |g| g.random_range(0..table.len()),
                |_, g| g.random_range(0..table.len()),
                |&state| {
                    evaluated.push(table[state]);
                    table[state]
                },
                |i| match temperature_kind {
                    0 => 0.0,
                    1 => f64::INFINITY,
                    2 => f64::NAN,
                    3 => -1.0,
                    _ => 10.0 / (1.0 + f64::from(i)),
                },
                limit,
            );
            (result, evaluated)
        };
        let (result, evaluated) = run(case);
        assert_eq!(run(case).0, result, "case {case}");
        assert_eq!(evaluated.len(), 1 + limit as usize);
        let best = evaluated
            .iter()
            .copied()
            .filter(|score| !score.is_nan())
            .reduce(f64::max);
        match (result, best) {
            (None, None) => {}
            (Some(state), Some(best)) => assert_eq!(table[state], best, "case {case}"),
            (got, want) => panic!("case {case}: {got:?} vs {want:?}"),
        }
    }
}

#[cfg(feature = "parallel")]
mod parallel {
    use super::*;
    use simul::experiment::{run_parallel, ParallelConfig, ParallelExecutor};
    use std::convert::Infallible;

    fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap_or_else(|| panic!("{value} is nonzero"))
    }

    // Law: parallel and reused-executor output equals serial output for any
    // worker count and input order.
    #[test]
    fn parallel_and_reused_executor_match_serial_for_any_workers_and_order() {
        for case in 0..200_u64 {
            let mut r = rng(90_000 + case);
            let seed = master(&mut r);
            let batch = random_batch(&mut r, 40);
            let expected = serial(case, seed, batch.clone(), |v, ctx| eval(*v, ctx));
            let workers = nonzero(r.random_range(1..=9));
            let limit = nonzero(batch.len().max(1) + r.random_range(0..3));
            let config = ParallelConfig::new(workers, limit);
            let mut shuffled = batch.clone();
            shuffle(&mut r, &mut shuffled);
            let parallel = run_parallel(
                seed,
                shuffled.clone(),
                config,
                || Ok::<_, Infallible>(0_u64),
                |scratch, v, ctx| {
                    *scratch += 1;
                    eval(*v, ctx)
                },
            )
            .unwrap_or_else(|error| panic!("case {case}: parallel run: {error:?}"));
            assert_eq!(expected, parallel, "case {case}");
            let executor = ParallelExecutor::new(config)
                .unwrap_or_else(|error| panic!("case {case}: pool construction: {error:?}"));
            for _ in 0..2 {
                shuffle(&mut r, &mut shuffled);
                let reused = executor
                    .run(
                        seed,
                        shuffled.clone(),
                        || Ok::<_, Infallible>(()),
                        |(), v, ctx| eval(*v, ctx),
                    )
                    .unwrap_or_else(|error| panic!("case {case}: reused run: {error:?}"));
                assert_eq!(expected, reused, "case {case}");
            }
        }
    }
}

// Law: Wilson bounds bracket the rate inside [0, 1] for any valid z.
#[test]
fn wilson_interval_brackets_the_rate_within_the_unit_interval() {
    for case in 0..2_000_u64 {
        let mut r = rng(100_000 + case);
        let wilson_z = if r.random_bool(0.1) {
            1.959_963_984_540_054
        } else {
            log_uniform(&mut r, -300, 300)
        };
        let mut reducer = BernoulliReducer::new(wilson_z)
            .unwrap_or_else(|error| panic!("case {case}: valid z: {error:?}"));
        let trials: u64 = if r.random_bool(0.2) {
            r.random_range(1..20_000)
        } else {
            r.random_range(1..50)
        };
        let probability = [0.0, 1.0, r.random_range(0.0..1.0)][r.random_range(0..3)];
        let mut successes = 0;
        for _ in 0..trials {
            let success = r.random_bool(probability);
            successes += u64::from(success);
            reducer.observe(success);
        }
        let summary = reducer
            .finish()
            .unwrap_or_else(|error| panic!("case {case}: nonempty: {error:?}"));
        assert_eq!((summary.attempts, summary.successes), (trials, successes));
        assert_eq!(
            summary.rate,
            successes as f64 / trials as f64,
            "case {case}"
        );
        assert!(
            summary.wilson_lower.is_finite() && summary.wilson_upper.is_finite(),
            "case {case}: {summary:?} z {wilson_z}"
        );
        assert!(
            0.0 <= summary.wilson_lower && summary.wilson_lower <= summary.rate,
            "case {case}: {summary:?} z {wilson_z}"
        );
        assert!(
            summary.rate <= summary.wilson_upper && summary.wilson_upper <= 1.0,
            "case {case}: {summary:?} z {wilson_z}"
        );
    }
}

// Law: a rejected scalar observation leaves the reducer unchanged.
#[test]
fn scalar_reducer_rejection_leaves_state_unchanged() {
    for case in 0..1_000_u64 {
        let mut r = rng(110_000 + case);
        let mut reducer = ScalarReducer::default();
        let mut accepted = Vec::new();
        for _ in 0..r.random_range(0..20) {
            let value = scalar_value(&mut r);
            let before = reducer.finish();
            match reducer.observe(value) {
                Ok(()) => accepted.push(value),
                Err(_) => assert_eq!(before, reducer.finish(), "case {case}"),
            }
            assert_eq!(
                reducer.finish(),
                ScalarReducer::from_results(accepted.iter().copied()),
                "case {case}"
            );
        }
    }
}

/// The V1 trial-stream derivation, written out independently.
fn reference_seed(
    master: [u8; 32],
    scope: &[u8],
    stream: &[u8],
    id: u64,
    replication: u32,
    tag: &str,
) -> [u8; 32] {
    let root = b"simul/seed-protocol/v1";
    let mut input = Vec::new();
    input.extend_from_slice(root);
    input.extend_from_slice(&(scope.len() as u64).to_le_bytes());
    input.extend_from_slice(scope);
    input.extend_from_slice(stream);
    input.extend_from_slice(&id.to_le_bytes());
    input.extend_from_slice(&replication.to_le_bytes());
    input.extend_from_slice(&(tag.len() as u64).to_le_bytes());
    input.extend_from_slice(tag.as_bytes());
    *blake3::keyed_hash(&master, &input).as_bytes()
}

// Law: the V1 trial-stream protocol is keyed BLAKE3 over length-prefixed
// namespaces and little-endian identities; stack and heap derivation inputs
// hash identical bytes for every tag length.
#[test]
fn seed_protocol_matches_keyed_blake3_reference_for_long_tags() {
    let streams = [
        (TrialStream::TrialNoise, &b"simul/v1/trial-noise"[..]),
        (TrialStream::LayoutNoise, &b"simul/v1/layout-noise"[..]),
        (
            TrialStream::ExecutionNoise,
            &b"simul/v1/execution-noise"[..],
        ),
        (
            TrialStream::ValidationHoldout,
            &b"simul/v1/validation-holdout"[..],
        ),
    ];
    for case in 0..400_u64 {
        let mut r = rng(120_000 + case);
        let seed = master(&mut r);
        let tag: String = "x".repeat(r.random_range(150..300));
        let candidate: u64 = r.random();
        let replication: u32 = r.random();
        let group: u64 = r.random();
        let (stream, namespace) = streams[r.random_range(0..4)];
        let records = serial(
            case,
            seed,
            [TrialSpec::new(
                CandidateId::new(candidate),
                ReplicationId::new(replication),
                group,
                (),
            )],
            |(), ctx| {
                Ok::<_, ()>((
                    ctx.independent_seed(stream, &tag).bytes(),
                    ctx.common_seed(stream, &tag).bytes(),
                ))
            },
        );
        let Ok((independent, common)) = records[0].result else {
            panic!("case {case}: infallible trial failed");
        };
        assert_eq!(
            independent,
            reference_seed(
                seed.bytes(),
                b"simul/v1/trial-independent",
                namespace,
                candidate,
                replication,
                &tag
            ),
            "case {case} tag {}",
            tag.len()
        );
        assert_eq!(
            common,
            reference_seed(
                seed.bytes(),
                b"simul/v1/trial-common",
                namespace,
                group,
                replication,
                &tag
            ),
            "case {case} tag {}",
            tag.len()
        );
    }
}

// Law: annealing always accepts better/equal candidates and, at non-finite or
// non-positive temperature, never accepts a worse one.
#[test]
fn annealing_accepts_improvements_and_rejects_worse_states_when_greedy() {
    for case in 0..400_u64 {
        let mut r = rng(130_000 + case);
        let table: Vec<f64> = (0..r.random_range(1..12)).map(|_| score(&mut r)).collect();
        let limit = r.random_range(0..40_u32);
        let greedy_kind = r.random_range(0..5);
        let mut next_id = 0_u32;
        let mut currents: Vec<(u32, usize)> = Vec::new();
        let mut candidates: Vec<(u32, usize)> = Vec::new();
        let mut annealing_rng = rng(case);
        let _ = simulated_annealing_search_with_rng(
            &mut annealing_rng,
            |g| (0_u32, g.random_range(0..table.len())),
            |current: &(u32, usize), g| {
                currents.push(*current);
                next_id += 1;
                let candidate = (next_id, g.random_range(0..table.len()));
                candidates.push(candidate);
                candidate
            },
            |state| table[state.1],
            |_| [0.0, -1.0, f64::INFINITY, f64::NAN, 1.0][greedy_kind],
            limit,
        );
        for i in 1..currents.len() {
            let current = table[currents[i - 1].1];
            let candidate = table[candidates[i - 1].1];
            let accepted = currents[i] == candidates[i - 1];
            if !candidate.is_nan() && (current.is_nan() || candidate >= current) {
                assert!(
                    accepted,
                    "case {case} step {i}: {candidate} vs {current} not accepted"
                );
            }
            if candidate.is_nan() || (greedy_kind < 4 && !current.is_nan() && candidate < current) {
                assert!(
                    !accepted,
                    "case {case} step {i}: {candidate} vs {current} accepted"
                );
            }
        }
    }
}
