//! Tail-risk quantile/CVaR reducers and paired-difference statistics.

use super::{
    AggregateError, CandidateSummary, FailurePolicy, FiniteValue, NonFinitePolicy, ReduceError,
    SelectionPolicy, TrialRecord, compare_nan_lowest, validate_candidate_records,
};
use std::cmp::Ordering;
use std::fmt;

/// A validated probability or quantile level in `[0.0, 1.0]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probability(f64);

// Every Probability is finite and lies in [0.0, 1.0], so equality is reflexive
// and partial_cmp always yields a total order consistent with PartialEq.
impl Eq for Probability {}

impl Ord for Probability {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Probability {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl TryFrom<f64> for Probability {
    type Error = InvalidProbability;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Probability> for f64 {
    fn from(value: Probability) -> Self {
        value.get()
    }
}

impl Probability {
    /// The 5th percentile (`0.05`).
    pub const P05: Self = Self(0.05);
    /// The 50th percentile / median (`0.50`).
    pub const P50: Self = Self(0.50);
    /// The 90th percentile (`0.90`).
    pub const P90: Self = Self(0.90);
    /// The 95th percentile (`0.95`).
    pub const P95: Self = Self(0.95);
    /// The 99th percentile (`0.99`).
    pub const P99: Self = Self(0.99);
    /// The 99.9th percentile (`0.999`).
    pub const P999: Self = Self(0.999);

    /// Validates and wraps a probability in `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidProbability`] when `value` is NaN, infinite, negative,
    /// or greater than `1.0`.
    pub const fn new(value: f64) -> Result<Self, InvalidProbability> {
        if value.is_finite() && value >= 0.0 && value <= 1.0 {
            Ok(Self(value))
        } else {
            Err(InvalidProbability { value })
        }
    }

    /// Returns the wrapped probability in `[0.0, 1.0]`.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// A rejected probability outside `[0.0, 1.0]` or non-finite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvalidProbability {
    /// The rejected value.
    pub value: f64,
}

impl fmt::Display for InvalidProbability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "probability must be finite and in [0.0, 1.0], got {}",
            self.value
        )
    }
}

impl std::error::Error for InvalidProbability {}

/// Exact empirical quantile and Conditional Value at Risk (`CVaR`) reducer.
///
/// Retains validated finite observations and sorts them in [`Self::finish`],
/// making every quantile and tail-expectation query exact and completely
/// independent of observation order.
///
/// ```
/// use simul::experiment::{Probability, QuantileReducer};
///
/// let summary = QuantileReducer::from_results([10.0, 20.0, 30.0, 40.0, 100.0])?;
/// assert_eq!(5, summary.count());
/// assert!((summary.median() - 30.0).abs() < 1e-12);
/// assert!((summary.upper_cvar(Probability::new(0.80).unwrap()) - 100.0).abs() < 1e-12);
/// # Ok::<(), simul::experiment::ReduceError>(())
/// ```
#[derive(Clone, Debug, Default)]
pub struct QuantileReducer {
    /// Validated finite observations.
    values: Vec<f64>,
}

impl QuantileReducer {
    /// Creates an empty quantile reducer.
    #[must_use]
    pub const fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Creates an empty quantile reducer with preallocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
        }
    }

    /// Observes one finite scalar value.
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::NonFinite`] when `value` is NaN or infinite,
    /// leaving the reducer unchanged.
    pub fn observe(&mut self, value: f64) -> Result<(), ReduceError> {
        let finite = FiniteValue::new(value).map_err(ReduceError::NonFinite)?;
        // Canonicalize -0.0 to +0.0 so order statistics never depend on zero sign.
        let canonical = if finite.get() == 0.0 {
            0.0
        } else {
            finite.get()
        };
        self.values.push(canonical);
        Ok(())
    }

    /// Merges all observations from `other` into this reducer.
    pub fn merge(&mut self, other: &Self) {
        self.values.extend_from_slice(&other.values);
    }

    /// Returns the number of observations recorded so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether no observations have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Reduces finite scalar observations into a [`QuantileSummary`].
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::NonFinite`] for any NaN or infinite observation,
    /// or [`ReduceError::Empty`] when `results` contains no observations.
    pub fn from_results(
        results: impl IntoIterator<Item = f64>,
    ) -> Result<QuantileSummary, ReduceError> {
        let iter = results.into_iter();
        let (lower, _) = iter.size_hint();
        let mut reducer = Self::with_capacity(lower);
        for result in iter {
            reducer.observe(result)?;
        }
        reducer.finish()
    }

    /// Sorts the observations and produces a [`QuantileSummary`].
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::Empty`] when no observations were recorded.
    pub fn finish(mut self) -> Result<QuantileSummary, ReduceError> {
        if self.values.is_empty() {
            return Err(ReduceError::Empty);
        }
        self.values.sort_unstable_by(f64::total_cmp);
        let mean = overflow_safe_weighted_mean(self.values.iter().copied(), 1.0, None);
        Ok(QuantileSummary {
            sorted: self.values,
            mean,
        })
    }
}

/// Exact order-statistic summary supporting arbitrary quantiles and `CVaR`.
///
/// Built only by [`QuantileReducer`] or [`aggregate_quantiles`]. Every stored
/// observation is finite and sorted in ascending order, so every query returns
/// a finite number and is invariant to the original observation order.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantileSummary {
    /// Non-empty sorted finite observations.
    sorted: Vec<f64>,
    /// Overflow-safe arithmetic mean of `sorted`.
    mean: f64,
}

impl QuantileSummary {
    /// Returns the number of observations in the summary.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.sorted.len() as u64
    }

    /// Returns the sorted observations in ascending order.
    #[must_use]
    pub fn sorted_values(&self) -> &[f64] {
        &self.sorted
    }

    /// Returns the arithmetic mean of the observations.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the minimum observation.
    #[must_use]
    pub fn min(&self) -> f64 {
        self.sorted[0]
    }

    /// Returns the maximum observation.
    #[must_use]
    pub fn max(&self) -> f64 {
        self.sorted[self.sorted.len() - 1]
    }

    /// Returns the empirical quantile at `p` using linear interpolation
    /// (Hyndman & Fan Type 7, matching `NumPy` and R defaults).
    ///
    /// Exact endpoints hold: `p = 0.0` returns [`Self::min`] and `p = 1.0`
    /// returns [`Self::max`]. The result is finite and monotonically
    /// non-decreasing in `p`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn quantile(&self, p: Probability) -> f64 {
        let last = self.sorted.len() - 1;
        if last == 0 || p.get() <= 0.0 {
            return self.sorted[0];
        }
        if p.get() >= 1.0 {
            return self.sorted[last];
        }
        let position = p.get() * last as f64;
        // position lies in [0.0, last as f64], so floor/ceil fit in usize.
        let lower_index = (position.floor() as usize).min(last);
        let upper_index = (position.ceil() as usize).min(last);
        if lower_index == upper_index {
            return self.sorted[lower_index];
        }
        let fraction = (position - lower_index as f64).clamp(0.0, 1.0);
        interpolate_safe(self.sorted[lower_index], self.sorted[upper_index], fraction)
    }

    /// Returns the median (50th percentile).
    #[must_use]
    pub fn median(&self) -> f64 {
        self.quantile(Probability::P50)
    }

    /// Returns the 90th percentile.
    #[must_use]
    pub fn p90(&self) -> f64 {
        self.quantile(Probability::P90)
    }

    /// Returns the 95th percentile.
    #[must_use]
    pub fn p95(&self) -> f64 {
        self.quantile(Probability::P95)
    }

    /// Returns the 99th percentile.
    #[must_use]
    pub fn p99(&self) -> f64 {
        self.quantile(Probability::P99)
    }

    /// Returns the 99.9th percentile.
    #[must_use]
    pub fn p999(&self) -> f64 {
        self.quantile(Probability::P999)
    }

    /// Returns the upper-tail Conditional Value at Risk (expected shortfall) at
    /// confidence level `alpha`: the exact conditional mean of the highest
    /// `1 - alpha` fraction of the empirical distribution.
    ///
    /// Use this for minimization objectives such as latency or financial loss
    /// (for example, `upper_cvar(Probability::P95)` is the expected value of
    /// the worst 5% highest outcomes). At `alpha = 0.0` this equals
    /// [`Self::mean`]; at `alpha = 1.0` it equals [`Self::max`].
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn upper_cvar(&self, alpha: Probability) -> f64 {
        let tail_fraction = 1.0 - alpha.get();
        if tail_fraction <= 0.0 {
            return self.max();
        }
        if tail_fraction >= 1.0 {
            return self.mean;
        }
        let n = self.sorted.len();
        let mass = tail_fraction * n as f64;
        // mass lies in (0.0, n as f64).
        let full = (mass.floor() as usize).min(n);
        let remainder = (mass - full as f64).clamp(0.0, 1.0);
        if full == 0 {
            return self.max();
        }
        let tail = self.sorted[n - full..].iter().rev().copied();
        let partial = if remainder > 0.0 && full < n {
            Some((self.sorted[n - full - 1], remainder))
        } else {
            None
        };
        overflow_safe_weighted_mean(tail, 1.0, partial)
    }

    /// Returns the lower-tail Conditional Value at Risk at tail fraction
    /// `tail`: the exact conditional mean of the lowest `tail` fraction of the
    /// empirical distribution.
    ///
    /// Use this for maximization objectives such as profit or survival margin
    /// (for example, `lower_cvar(Probability::P05)` is the expected value of
    /// the worst 5% lowest outcomes). At `tail = 0.0` this equals
    /// [`Self::min`]; at `tail = 1.0` it equals [`Self::mean`].
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn lower_cvar(&self, tail: Probability) -> f64 {
        let tail_fraction = tail.get();
        if tail_fraction <= 0.0 {
            return self.min();
        }
        if tail_fraction >= 1.0 {
            return self.mean;
        }
        let n = self.sorted.len();
        let mass = tail_fraction * n as f64;
        let full = (mass.floor() as usize).min(n);
        let remainder = (mass - full as f64).clamp(0.0, 1.0);
        if full == 0 {
            return self.min();
        }
        let head = self.sorted[..full].iter().copied();
        let partial = if remainder > 0.0 && full < n {
            Some((self.sorted[full], remainder))
        } else {
            None
        };
        overflow_safe_weighted_mean(head, 1.0, partial)
    }

    /// Merges two summaries into a combined exact [`QuantileSummary`].
    #[must_use]
    pub fn merged(&self, other: &Self) -> Self {
        let mut combined = Vec::with_capacity(self.sorted.len() + other.sorted.len());
        let (mut i, mut j) = (0, 0);
        while i < self.sorted.len() && j < other.sorted.len() {
            if self.sorted[i].total_cmp(&other.sorted[j]) == Ordering::Greater {
                combined.push(other.sorted[j]);
                j += 1;
            } else {
                combined.push(self.sorted[i]);
                i += 1;
            }
        }
        combined.extend_from_slice(&self.sorted[i..]);
        combined.extend_from_slice(&other.sorted[j..]);
        let mean = overflow_safe_weighted_mean(combined.iter().copied(), 1.0, None);
        Self {
            sorted: combined,
            mean,
        }
    }
}

/// Linearly interpolates between finite values `a` and `b` at `t` in `[0, 1]`
/// without overflowing when `b - a` exceeds `f64::MAX`.
fn interpolate_safe(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if diff.is_finite() {
        t.mul_add(diff, a)
    } else {
        a.mul_add(1.0 - t, b * t)
    }
}

/// Computes a weighted arithmetic mean of finite values using an overflow-safe
/// incremental convex-combination update.
fn overflow_safe_weighted_mean(
    values: impl Iterator<Item = f64>,
    unit_weight: f64,
    partial: Option<(f64, f64)>,
) -> f64 {
    let mut mean = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    for value in values {
        if weight_sum == 0.0 {
            mean = value;
            weight_sum = unit_weight;
        } else {
            weight_sum += unit_weight;
            let step = unit_weight / weight_sum;
            mean = interpolate_safe(mean, value, step);
        }
    }
    if let Some((value, weight)) = partial {
        if weight > 0.0 {
            if weight_sum == 0.0 {
                mean = value;
            } else {
                weight_sum += weight;
                let step = weight / weight_sum;
                mean = interpolate_safe(mean, value, step);
            }
        }
    }
    mean
}

/// Aggregates typed finite-scalar records into a [`QuantileSummary`] with
/// explicit failure and non-finite policies.
///
/// Policy precedence matches [`aggregate_scalar`](super::aggregate_scalar):
/// empty/mixed-candidate checks come first, then [`FailurePolicy::RejectCandidate`],
/// then [`NonFinitePolicy::RejectCandidate`], then [`AggregateError::NoUsableObservations`].
///
/// # Errors
///
/// Returns [`AggregateError`] when the records are empty, mix candidates,
/// exceed public count capacity, contain rejected failures or non-finite
/// values, or leave no usable observations.
pub fn aggregate_quantiles<O, E>(
    records: &[TrialRecord<O, E>],
    score: impl Fn(&O) -> f64,
    failure_policy: FailurePolicy<FiniteValue>,
    non_finite_policy: NonFinitePolicy,
) -> Result<CandidateSummary<QuantileSummary>, AggregateError> {
    let candidate = validate_candidate_records(records)?;
    let requested = u32::try_from(records.len()).map_err(|_| AggregateError::TooManyRecords)?;
    let mut reducer = QuantileReducer::with_capacity(records.len());
    let mut completed = 0_u32;
    let mut failed = 0_u32;
    let mut non_finite = 0_u32;
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
            // FiniteValue guarantees observe cannot fail.
            let _ = reducer.observe(value.get());
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

/// Minimizes the sample mean of a [`ScalarSummary`](super::ScalarSummary).
///
/// Exact ties break by lower [`CandidateId`](super::CandidateId) in
/// [`select_best`](super::select_best).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MinimizeScalarMean;

impl SelectionPolicy<super::ScalarSummary> for MinimizeScalarMean {
    fn compare(&self, left: &super::ScalarSummary, right: &super::ScalarSummary) -> Ordering {
        compare_nan_lowest(right.mean(), left.mean())
    }
}

/// Minimizes a chosen empirical quantile of a [`QuantileSummary`] (for
/// example, P95 or P99 tail latency).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MinimizeQuantile {
    /// Target quantile level in `[0.0, 1.0]`.
    level: Probability,
}

impl MinimizeQuantile {
    /// Creates a policy minimizing the quantile at `level`.
    #[must_use]
    pub const fn new(level: Probability) -> Self {
        Self { level }
    }

    /// Returns the target quantile level.
    #[must_use]
    pub const fn level(self) -> Probability {
        self.level
    }
}

impl SelectionPolicy<QuantileSummary> for MinimizeQuantile {
    fn compare(&self, left: &QuantileSummary, right: &QuantileSummary) -> Ordering {
        compare_nan_lowest(right.quantile(self.level), left.quantile(self.level))
    }
}

/// Maximizes a chosen empirical quantile of a [`QuantileSummary`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaximizeQuantile {
    /// Target quantile level in `[0.0, 1.0]`.
    level: Probability,
}

impl MaximizeQuantile {
    /// Creates a policy maximizing the quantile at `level`.
    #[must_use]
    pub const fn new(level: Probability) -> Self {
        Self { level }
    }

    /// Returns the target quantile level.
    #[must_use]
    pub const fn level(self) -> Probability {
        self.level
    }
}

impl SelectionPolicy<QuantileSummary> for MaximizeQuantile {
    fn compare(&self, left: &QuantileSummary, right: &QuantileSummary) -> Ordering {
        compare_nan_lowest(left.quantile(self.level), right.quantile(self.level))
    }
}

/// Minimizes the upper-tail Conditional Value at Risk
/// ([`QuantileSummary::upper_cvar`]) at a chosen confidence level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MinimizeUpperCvar {
    /// Confidence level `alpha` in `[0.0, 1.0]`.
    alpha: Probability,
}

impl MinimizeUpperCvar {
    /// Creates a policy minimizing the expected value of the highest
    /// `1 - alpha` tail.
    #[must_use]
    pub const fn new(alpha: Probability) -> Self {
        Self { alpha }
    }

    /// Returns the confidence level `alpha`.
    #[must_use]
    pub const fn alpha(self) -> Probability {
        self.alpha
    }
}

impl SelectionPolicy<QuantileSummary> for MinimizeUpperCvar {
    fn compare(&self, left: &QuantileSummary, right: &QuantileSummary) -> Ordering {
        compare_nan_lowest(right.upper_cvar(self.alpha), left.upper_cvar(self.alpha))
    }
}

/// Maximizes the lower-tail Conditional Value at Risk
/// ([`QuantileSummary::lower_cvar`]) at a chosen tail fraction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaximizeLowerCvar {
    /// Lower tail fraction in `[0.0, 1.0]`.
    tail: Probability,
}

impl MaximizeLowerCvar {
    /// Creates a policy maximizing the expected value of the lowest `tail`
    /// fraction.
    #[must_use]
    pub const fn new(tail: Probability) -> Self {
        Self { tail }
    }

    /// Returns the lower tail fraction.
    #[must_use]
    pub const fn tail(self) -> Probability {
        self.tail
    }
}

impl SelectionPolicy<QuantileSummary> for MaximizeLowerCvar {
    fn compare(&self, left: &QuantileSummary, right: &QuantileSummary) -> Ordering {
        compare_nan_lowest(left.lower_cvar(self.tail), right.lower_cvar(self.tail))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::missing_docs_in_private_items)]
mod tests {
    use super::*;
    use crate::experiment::{CandidateId, FixedReplications, MasterSeed, run_serial, select_best};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "expected {expected:.15}, got {actual:.15}"
        );
    }

    #[test]
    fn probability_validates_unit_interval_and_formats_errors() {
        assert_eq!(Ok(Probability::P50), Probability::new(0.5));
        assert!(Probability::new(0.0).is_ok());
        assert!(Probability::new(1.0).is_ok());
        for bad in [-0.001, 1.001, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = Probability::new(bad).expect_err("out-of-range probability must fail");
            assert_ne!("", error.to_string());
        }
    }

    #[test]
    fn quantile_and_cvar_match_exact_fixtures() {
        let summary =
            QuantileReducer::from_results([100.0, 10.0, 40.0, 20.0, 30.0]).expect("valid sample");
        assert_eq!(5, summary.count());
        assert_close(summary.min(), 10.0);
        assert_close(summary.max(), 100.0);
        assert_close(summary.mean(), 40.0);
        assert_close(summary.median(), 30.0);
        assert_close(summary.quantile(Probability::new(0.25).expect("p")), 20.0);
        assert_close(summary.quantile(Probability::new(0.75).expect("p")), 40.0);
        assert_close(summary.quantile(Probability::new(0.0).expect("p")), 10.0);
        assert_close(summary.quantile(Probability::new(1.0).expect("p")), 100.0);

        // Top 20% of 5 samples is exactly 1 sample: 100.0.
        assert_close(
            summary.upper_cvar(Probability::new(0.80).expect("p")),
            100.0,
        );
        // Top 40% of 5 samples is top 2 samples: (100 + 40) / 2 = 70.0.
        assert_close(summary.upper_cvar(Probability::new(0.60).expect("p")), 70.0);
        // Top 30% of 5 samples is 1.5 samples: (100 + 0.5 * 40) / 1.5 = 80.0.
        assert_close(summary.upper_cvar(Probability::new(0.70).expect("p")), 80.0);
        // Full distribution CVaR equals mean.
        assert_close(summary.upper_cvar(Probability::new(0.0).expect("p")), 40.0);
        assert_close(summary.upper_cvar(Probability::new(1.0).expect("p")), 100.0);

        // Bottom 40% of 5 samples is (10 + 20) / 2 = 15.0.
        assert_close(summary.lower_cvar(Probability::new(0.40).expect("p")), 15.0);
        // Bottom 30% of 5 samples is (10 + 0.5 * 20) / 1.5 = 40/3.
        assert_close(
            summary.lower_cvar(Probability::new(0.30).expect("p")),
            40.0 / 3.0,
        );
        assert_close(summary.lower_cvar(Probability::new(0.0).expect("p")), 10.0);
        assert_close(summary.lower_cvar(Probability::new(1.0).expect("p")), 40.0);
    }

    #[test]
    fn quantile_reducer_handles_extreme_finite_values_without_overflow() {
        let summary =
            QuantileReducer::from_results([-1.0e308, 1.0e308]).expect("extreme finite sample");
        assert!(summary.mean().is_finite());
        assert!(summary.median().is_finite());
        assert!(summary.upper_cvar(Probability::P95).is_finite());
        assert!(summary.lower_cvar(Probability::P05).is_finite());
    }

    #[test]
    fn quantile_merge_matches_combined_reduction() {
        let left = QuantileReducer::from_results([5.0, 1.0, 9.0]).expect("left");
        let right = QuantileReducer::from_results([3.0, 7.0, 1.0]).expect("right");
        let combined = QuantileReducer::from_results([5.0, 1.0, 9.0, 3.0, 7.0, 1.0]).expect("all");
        assert_eq!(
            combined.sorted_values(),
            left.merged(&right).sorted_values()
        );
        assert_close(combined.mean(), left.merged(&right).mean());
    }

    #[test]
    fn aggregate_quantiles_and_tail_policies_rank_tail_risk() {
        let seed = MasterSeed::from_u64(77);
        let schedule = FixedReplications::new(20, 1);
        // Candidate 0 has lower mean (14.5) but a huge P99 spike (100.0).
        // Candidate 1 has slightly higher mean (15.0) but flat tail (15.0).
        let trials = schedule
            .trials(CandidateId::new(0), true)
            .chain(schedule.trials(CandidateId::new(1), false));
        let records = run_serial(seed, trials, |&spiky, ctx| {
            let rep = ctx.key().replication().value();
            Ok::<f64, ()>(if spiky {
                if rep == 19 { 100.0 } else { 10.0 }
            } else {
                15.0
            })
        })
        .expect("serial run");

        let (c0_records, c1_records) = records.split_at(20);
        let c0 = aggregate_quantiles(
            c0_records,
            |&v| v,
            FailurePolicy::RejectCandidate,
            NonFinitePolicy::RejectCandidate,
        )
        .expect("c0")
        .statistics;
        let c1 = aggregate_quantiles(
            c1_records,
            |&v| v,
            FailurePolicy::RejectCandidate,
            NonFinitePolicy::RejectCandidate,
        )
        .expect("c1")
        .statistics;

        let candidates = [(CandidateId::new(0), c0), (CandidateId::new(1), c1)];
        let (p99_winner, _) =
            select_best(&candidates, &MinimizeQuantile::new(Probability::P99)).expect("winner");
        let (cvar_winner, _) =
            select_best(&candidates, &MinimizeUpperCvar::new(Probability::P95)).expect("winner");
        assert_eq!(CandidateId::new(1), p99_winner);
        assert_eq!(CandidateId::new(1), cvar_winner);
    }
}
