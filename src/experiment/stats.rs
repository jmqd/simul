//! Tail-risk quantile/CVaR reducers and paired-difference statistics.

use super::{
    AggregateError, CandidateId, CandidateSummary, FailurePolicy, FiniteValue, NonFinitePolicy,
    ReduceError, ReplicationId, ScalarReducer, ScalarSummary, SelectionPolicy, TrialRecord,
    compare_nan_lowest, validate_candidate_records,
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
    #[inline]
    pub fn observe(&mut self, value: f64) -> Result<(), ReduceError> {
        let finite = FiniteValue::new(value).map_err(ReduceError::NonFinite)?;
        self.observe_finite(finite.get());
        Ok(())
    }

    /// Observes a pre-validated finite scalar value, canonicalizing `-0.0` to `+0.0`.
    #[inline]
    fn observe_finite(&mut self, value: f64) {
        // Adding +0.0 converts -0.0 to +0.0 while leaving every other finite
        // value bit-for-bit unchanged under IEEE-754 round-to-nearest-even.
        self.values.push(value + 0.0);
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
        // All stored observations are finite and have -0.0 canonicalized to
        // +0.0, so IEEE float ordering is a total order identical to total_cmp.
        self.values.sort_unstable_by(|&a, &b| cmp_finite_f64(a, b));
        let finite_span = (self.values[self.values.len() - 1] - self.values[0]).is_finite();
        let mean = overflow_safe_weighted_mean(self.values.iter().copied(), None, finite_span);
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
        let tail_slice = &self.sorted[n - full..];
        let partial = if remainder > 0.0 && full < n {
            Some((self.sorted[n - full - 1], remainder))
        } else {
            None
        };
        let low = partial.map_or(tail_slice[0], |(value, _)| value);
        let finite_span = (tail_slice[tail_slice.len() - 1] - low).is_finite();
        overflow_safe_weighted_mean(tail_slice.iter().rev().copied(), partial, finite_span)
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
        let head_slice = &self.sorted[..full];
        let partial = if remainder > 0.0 && full < n {
            Some((self.sorted[full], remainder))
        } else {
            None
        };
        let high = partial.map_or(head_slice[head_slice.len() - 1], |(value, _)| value);
        let finite_span = (high - head_slice[0]).is_finite();
        overflow_safe_weighted_mean(head_slice.iter().copied(), partial, finite_span)
    }

    /// Merges two summaries into a combined exact [`QuantileSummary`].
    #[must_use]
    pub fn merged(&self, other: &Self) -> Self {
        let mut combined = Vec::with_capacity(self.sorted.len() + other.sorted.len());
        let (mut i, mut j) = (0, 0);
        while i < self.sorted.len() && j < other.sorted.len() {
            if self.sorted[i] > other.sorted[j] {
                combined.push(other.sorted[j]);
                j += 1;
            } else {
                combined.push(self.sorted[i]);
                i += 1;
            }
        }
        combined.extend_from_slice(&self.sorted[i..]);
        combined.extend_from_slice(&other.sorted[j..]);
        let finite_span = (combined[combined.len() - 1] - combined[0]).is_finite();
        let mean = overflow_safe_weighted_mean(combined.iter().copied(), None, finite_span);
        Self {
            sorted: combined,
            mean,
        }
    }
}

/// Total ordering on finite, `-0.0`-canonicalized `f64` observations.
#[inline]
fn cmp_finite_f64(left: f64, right: f64) -> Ordering {
    if left < right {
        Ordering::Less
    } else if left > right {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// Linearly interpolates between finite values `a` and `b` at `t` in `[0, 1]`
/// without overflowing when `b - a` exceeds `f64::MAX`.
#[inline]
fn interpolate_safe(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if diff.is_finite() {
        t.mul_add(diff, a)
    } else {
        a.mul_add(1.0 - t, b * t)
    }
}

/// Computes a unit-weighted arithmetic mean of finite values using an
/// overflow-safe incremental convex-combination update.
///
/// When `finite_span` is `true`, the caller guarantees `max - min` is finite
/// across all values (including any `partial`), so `value - mean` is always
/// finite and the inner loop executes the exact same `mul_add` recurrence
/// without per-element overflow branches.
fn overflow_safe_weighted_mean(
    mut values: impl Iterator<Item = f64>,
    partial: Option<(f64, f64)>,
    finite_span: bool,
) -> f64 {
    let Some(mut mean) = values.next() else {
        return match partial {
            Some((value, weight)) if weight > 0.0 => value,
            _ => 0.0,
        };
    };
    let mut weight_sum = 1.0_f64;
    if finite_span {
        for value in values {
            weight_sum += 1.0;
            let step = 1.0 / weight_sum;
            mean = step.mul_add(value - mean, mean);
        }
        if let Some((value, weight)) = partial {
            if weight > 0.0 {
                weight_sum += weight;
                let step = weight / weight_sum;
                mean = step.mul_add(value - mean, mean);
            }
        }
    } else {
        for value in values {
            weight_sum += 1.0;
            let step = 1.0 / weight_sum;
            mean = interpolate_safe(mean, value, step);
        }
        if let Some((value, weight)) = partial {
            if weight > 0.0 {
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
            reducer.observe_finite(value.get());
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

/// Exact two-sided 95% Student-t critical values `t_{0.975, df}` for `df = 1..=30`.
const STUDENT_T_95_TABLE: [f64; 30] = [
    12.706_205, 4.302_653, 3.182_446, 2.776_445, 2.570_582, 2.446_912, 2.364_624, 2.306_004,
    2.262_157, 2.228_139, 2.200_985, 2.178_813, 2.160_369, 2.144_787, 2.131_450, 2.119_905,
    2.109_816, 2.100_922, 2.093_024, 2.085_963, 2.079_614, 2.073_873, 2.068_658, 2.063_899,
    2.059_539, 2.055_529, 2.051_831, 2.048_407, 2.045_230, 2.042_272,
];

/// Returns the two-sided 95% Student-t critical value for `degrees_of_freedom`.
///
/// Uses an exact lookup table for `df <= 30` and a three-term Cornish-Fisher
/// expansion above 30 (accurate to better than `1e-5` across all `df >= 1`).
#[must_use]
pub(super) fn student_t_95_critical(degrees_of_freedom: u64) -> Option<f64> {
    if degrees_of_freedom == 0 {
        return None;
    }
    if let Ok(index) = usize::try_from(degrees_of_freedom - 1) {
        if let Some(&value) = STUDENT_T_95_TABLE.get(index) {
            return Some(value);
        }
    }
    let z = 1.959_963_984_540_054_f64;
    let df = degrees_of_freedom as f64;
    let inv_df = df.recip();
    // Cornish-Fisher coefficients for z = Phi^{-1}(0.975):
    // g1 = (z^3 + z) / 4, g2 = (5 z^5 + 16 z^3 + 3 z) / 96,
    // g3 = (3 z^7 + 19 z^5 + 17 z^3 - 15 z) / 384.
    let g1 = 2.372_272_663_569_986_5_f64;
    let g2 = 2.822_502_587_391_937_5_f64;
    let g3 = 2.555_836_209_914_479_f64;
    Some(inv_df.mul_add(inv_df.mul_add(inv_df.mul_add(g3, g2), g1), z))
}

/// Why two candidate record sets could not be compared by paired differences.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairedCompareError {
    /// At least one record slice was empty.
    Empty,
    /// A record slice mixed records from more than one candidate.
    MixedCandidates,
    /// One candidate slice contained duplicate replications.
    DuplicateReplication {
        /// Candidate containing the duplicate replication.
        candidate: CandidateId,
        /// Duplicated replication identity.
        replication: ReplicationId,
    },
    /// A replication appeared in one candidate slice but not the other.
    UnmatchedReplication(ReplicationId),
    /// Record count exceeded `u32::MAX`.
    TooManyRecords,
    /// At least one typed evaluation failed under [`FailurePolicy::RejectCandidate`].
    EvaluationFailed {
        /// Failed evaluations in the baseline slice.
        baseline_failures: u32,
        /// Failed evaluations in the challenger slice.
        challenger_failures: u32,
    },
    /// At least one scalar was non-finite under [`NonFinitePolicy::RejectCandidate`].
    NonFinite {
        /// Non-finite observations in the baseline slice.
        baseline_observations: u32,
        /// Non-finite observations in the challenger slice.
        challenger_observations: u32,
    },
    /// Every replication pair was excluded by policy.
    NoUsablePairs,
    /// Paired difference or scalar reduction overflowed finite arithmetic.
    ArithmeticOverflow,
}

impl fmt::Display for PairedCompareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("cannot compare an empty record slice"),
            Self::MixedCandidates => {
                formatter.write_str("each paired-comparison slice must belong to one candidate")
            }
            Self::DuplicateReplication {
                candidate,
                replication,
            } => write!(
                formatter,
                "duplicate replication {replication:?} for candidate {candidate:?}"
            ),
            Self::UnmatchedReplication(replication) => {
                write!(formatter, "unmatched replication {replication:?}")
            }
            Self::TooManyRecords => formatter.write_str("record count exceeds u32::MAX"),
            Self::EvaluationFailed {
                baseline_failures,
                challenger_failures,
            } => write!(
                formatter,
                "paired comparison rejected after {baseline_failures} baseline and {challenger_failures} challenger failure(s)"
            ),
            Self::NonFinite {
                baseline_observations,
                challenger_observations,
            } => write!(
                formatter,
                "paired comparison rejected after {baseline_observations} baseline and {challenger_observations} challenger non-finite observation(s)"
            ),
            Self::NoUsablePairs => {
                formatter.write_str("no usable replication pairs remained after applying policies")
            }
            Self::ArithmeticOverflow => {
                formatter.write_str("paired comparison exceeded representable finite arithmetic")
            }
        }
    }
}

impl std::error::Error for PairedCompareError {}

/// Paired-difference statistics comparing `challenger - baseline` across
/// matched replications.
///
/// Built by [`compare_paired_scalar`] or [`compare_paired_candidates`].
/// Differences are always reduced in ascending [`ReplicationId`] order, making
/// results independent of input slice ordering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedScalarSummary {
    /// Baseline candidate identity.
    baseline_candidate: CandidateId,
    /// Challenger candidate identity.
    challenger_candidate: CandidateId,
    /// Total matched replications supplied.
    requested_pairs: u32,
    /// Matched replications with usable finite values on both sides.
    usable_pairs: u32,
    /// Usable pairs where `challenger > baseline`.
    challenger_wins: u32,
    /// Usable pairs where `baseline > challenger`.
    baseline_wins: u32,
    /// Usable pairs where `challenger == baseline`.
    ties: u32,
    /// Whether every matched pair shared the same master seed and common-random
    /// group.
    common_random_paired: bool,
    /// Statistics over per-replication differences `challenger - baseline`.
    difference: ScalarSummary,
    /// Statistics over usable paired baseline observations.
    baseline: ScalarSummary,
    /// Statistics over usable paired challenger observations.
    challenger: ScalarSummary,
}

impl PairedScalarSummary {
    /// Returns the baseline candidate identity.
    #[must_use]
    pub const fn baseline_candidate(&self) -> CandidateId {
        self.baseline_candidate
    }

    /// Returns the challenger candidate identity.
    #[must_use]
    pub const fn challenger_candidate(&self) -> CandidateId {
        self.challenger_candidate
    }

    /// Returns the total number of matched replications supplied.
    #[must_use]
    pub const fn requested_pairs(&self) -> u32 {
        self.requested_pairs
    }

    /// Returns the number of matched replications included in the paired
    /// difference summary.
    #[must_use]
    pub const fn usable_pairs(&self) -> u32 {
        self.usable_pairs
    }

    /// Returns the number of usable pairs where `challenger > baseline`.
    #[must_use]
    pub const fn challenger_wins(&self) -> u32 {
        self.challenger_wins
    }

    /// Returns the number of usable pairs where `baseline > challenger`.
    #[must_use]
    pub const fn baseline_wins(&self) -> u32 {
        self.baseline_wins
    }

    /// Returns the number of usable pairs where `challenger == baseline`.
    #[must_use]
    pub const fn ties(&self) -> u32 {
        self.ties
    }

    /// Returns whether every matched pair shared identical master seed and
    /// `common_random_group` replay metadata.
    #[must_use]
    pub const fn is_common_random_paired(&self) -> bool {
        self.common_random_paired
    }

    /// Returns the paired difference summary (`challenger - baseline`).
    #[must_use]
    pub const fn difference(&self) -> &ScalarSummary {
        &self.difference
    }

    /// Returns the marginal summary of the usable paired baseline observations.
    #[must_use]
    pub const fn baseline(&self) -> &ScalarSummary {
        &self.baseline
    }

    /// Returns the marginal summary of the usable paired challenger observations.
    #[must_use]
    pub const fn challenger(&self) -> &ScalarSummary {
        &self.challenger
    }

    /// Returns a symmetric confidence interval `(lower, upper)` around the mean
    /// paired difference using `critical_value * standard_error`, or `None`
    /// when fewer than two usable pairs exist.
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::InvalidWilsonZ`] when `critical_value` is not
    /// finite and positive, or [`ReduceError::ArithmeticOverflow`] when the
    /// bounds overflow finite `f64`.
    pub fn confidence_interval(
        &self,
        critical_value: f64,
    ) -> Result<Option<(f64, f64)>, ReduceError> {
        self.difference.confidence_interval(critical_value)
    }

    /// Returns the two-sided 95% Student-t paired confidence interval
    /// `(lower, upper)` for `challenger - baseline` with `usable_pairs - 1`
    /// degrees of freedom, or `None` when fewer than two usable pairs exist.
    #[must_use]
    pub fn ci_95(&self) -> Option<(f64, f64)> {
        self.difference.ci_95()
    }

    /// Returns the variance reduction ratio achieved by pairing:
    /// `(Var(baseline) + Var(challenger)) / Var(challenger - baseline)`.
    ///
    /// Values greater than `1.0` indicate positive correlation induced by
    /// Common Random Numbers (`CRN`), meaning fewer replications are needed for
    /// the same confidence-interval width. Returns `None` when fewer than two
    /// pairs exist or when the paired difference variance is zero or
    /// non-finite.
    #[must_use]
    pub fn variance_reduction_ratio(&self) -> Option<f64> {
        let b_var = self.baseline.sample_variance()?;
        let c_var = self.challenger.sample_variance()?;
        let d_var = self.difference.sample_variance()?;
        if d_var <= 0.0 {
            return None;
        }
        let unpaired = b_var + c_var;
        let ratio = unpaired / d_var;
        ratio.is_finite().then_some(ratio)
    }
}

/// Candidate identity paired with its replication-sorted record references.
type OrderedCandidateRecords<'a, O, E> = (CandidateId, Vec<&'a TrialRecord<O, E>>);

/// Sorts references to a single candidate's records by [`ReplicationId`] and
/// rejects empty, mixed-candidate, or duplicate-replication slices.
fn sort_unique_replications<O, E>(
    records: &[TrialRecord<O, E>],
) -> Result<OrderedCandidateRecords<'_, O, E>, PairedCompareError> {
    let candidate = validate_candidate_records(records).map_err(|error| match error {
        AggregateError::MixedCandidates => PairedCompareError::MixedCandidates,
        AggregateError::Empty
        | AggregateError::TooManyRecords
        | AggregateError::EvaluationFailed { .. }
        | AggregateError::NonFinite { .. }
        | AggregateError::NoUsableObservations
        | AggregateError::ArithmeticOverflow => PairedCompareError::Empty,
    })?;
    let mut ordered: Vec<&TrialRecord<O, E>> = records.iter().collect();
    ordered.sort_unstable_by_key(|record| record.key.replication());
    for pair in ordered.windows(2) {
        if pair[0].key.replication() == pair[1].key.replication() {
            return Err(PairedCompareError::DuplicateReplication {
                candidate,
                replication: pair[0].key.replication(),
            });
        }
    }
    Ok((candidate, ordered))
}

/// Evaluates one record under the declared failure and non-finite policies.
fn extract_policy_value<O, E>(
    record: &TrialRecord<O, E>,
    score: &impl Fn(&O) -> f64,
    failure_policy: FailurePolicy<FiniteValue>,
    non_finite_policy: NonFinitePolicy,
    failures: &mut u32,
    non_finite: &mut u32,
) -> Option<FiniteValue> {
    record.result.as_ref().map_or_else(
        |_| {
            *failures += 1;
            match failure_policy {
                FailurePolicy::RejectCandidate | FailurePolicy::Ignore => None,
                FailurePolicy::ReplaceWith(value) => Some(value),
            }
        },
        |outcome| {
            FiniteValue::new(score(outcome)).map_or_else(
                |_| {
                    *non_finite += 1;
                    match non_finite_policy {
                        NonFinitePolicy::RejectCandidate | NonFinitePolicy::Ignore => None,
                        NonFinitePolicy::ReplaceWith(value) => Some(value),
                    }
                },
                Some,
            )
        },
    )
}

/// Compares `challenger - baseline` across matched replications, preserving the
/// variance reduction of Common Random Numbers (`CRN`).
///
/// Both slices must be non-empty, contain a single candidate each, and cover
/// the exact same set of unique [`ReplicationId`]s. Records may arrive in any
/// order; pairs are reduced in ascending [`ReplicationId`] order.
///
/// Under [`FailurePolicy::Ignore`] or [`NonFinitePolicy::Ignore`], a pair is
/// included only when both sides yield usable finite observations.
///
/// # Errors
///
/// Returns [`PairedCompareError`] when either slice is empty, mixes candidates,
/// has duplicate or unmatched replications, contains rejected failures or
/// non-finite values, leaves no usable pairs, or overflows finite arithmetic.
///
/// ```
/// use simul::experiment::{
///     CandidateId, FailurePolicy, FixedReplications, MasterSeed, NonFinitePolicy,
///     TrialStream, compare_paired_scalar, run_serial,
/// };
///
/// let seed = MasterSeed::from_u64(42);
/// let schedule = FixedReplications::new(16, 7);
/// let trials = schedule
///     .trials(CandidateId::new(0), 0.0_f64)
///     .chain(schedule.trials(CandidateId::new(1), 2.5_f64));
/// let records = run_serial(seed, trials, |&shift, ctx| {
///     let shared_noise = ctx.common_rng(TrialStream::TrialNoise, "load").uniform_f64() * 100.0;
///     Ok::<f64, ()>(shared_noise + shift)
/// })?;
/// let (baseline, challenger) = records.split_at(16);
/// let paired = compare_paired_scalar(
///     baseline,
///     challenger,
///     |&v| v,
///     FailurePolicy::RejectCandidate,
///     NonFinitePolicy::RejectCandidate,
/// )?;
/// assert!(paired.is_common_random_paired());
/// assert_eq!(16, paired.challenger_wins());
/// assert!((paired.difference().mean() - 2.5).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compare_paired_scalar<O, E>(
    baseline: &[TrialRecord<O, E>],
    challenger: &[TrialRecord<O, E>],
    score: impl Fn(&O) -> f64,
    failure_policy: FailurePolicy<FiniteValue>,
    non_finite_policy: NonFinitePolicy,
) -> Result<PairedScalarSummary, PairedCompareError> {
    let (baseline_candidate, ordered_baseline) = sort_unique_replications(baseline)?;
    let (challenger_candidate, ordered_challenger) = sort_unique_replications(challenger)?;
    compare_paired_refs(
        baseline_candidate,
        &ordered_baseline,
        challenger_candidate,
        &ordered_challenger,
        &score,
        failure_policy,
        non_finite_policy,
    )
}

/// Core paired-difference reduction over two sorted, deduplicated candidate
/// record reference lists.
fn compare_paired_refs<O, E>(
    baseline_candidate: CandidateId,
    ordered_baseline: &[&TrialRecord<O, E>],
    challenger_candidate: CandidateId,
    ordered_challenger: &[&TrialRecord<O, E>],
    score: &impl Fn(&O) -> f64,
    failure_policy: FailurePolicy<FiniteValue>,
    non_finite_policy: NonFinitePolicy,
) -> Result<PairedScalarSummary, PairedCompareError> {
    let requested_pairs =
        u32::try_from(ordered_baseline.len()).map_err(|_| PairedCompareError::TooManyRecords)?;
    if ordered_challenger.len() > u32::MAX as usize {
        return Err(PairedCompareError::TooManyRecords);
    }

    verify_matching_replications(ordered_baseline, ordered_challenger)?;

    let mut diff_reducer = ScalarReducer::default();
    let mut base_reducer = ScalarReducer::default();
    let mut chal_reducer = ScalarReducer::default();
    let mut baseline_failures = 0_u32;
    let mut challenger_failures = 0_u32;
    let mut baseline_non_finite = 0_u32;
    let mut challenger_non_finite = 0_u32;
    let mut usable_pairs = 0_u32;
    let mut challenger_wins = 0_u32;
    let mut baseline_wins = 0_u32;
    let mut ties = 0_u32;
    let mut common_random_paired = true;
    let mut reduction = Ok(());

    for (base_rec, chal_rec) in ordered_baseline.iter().zip(ordered_challenger.iter()) {
        if base_rec.replay.master_seed() != chal_rec.replay.master_seed()
            || base_rec.replay.common_random_group() != chal_rec.replay.common_random_group()
        {
            common_random_paired = false;
        }
        let base_val = extract_policy_value(
            base_rec,
            score,
            failure_policy,
            non_finite_policy,
            &mut baseline_failures,
            &mut baseline_non_finite,
        );
        let chal_val = extract_policy_value(
            chal_rec,
            score,
            failure_policy,
            non_finite_policy,
            &mut challenger_failures,
            &mut challenger_non_finite,
        );
        if let (Some(b), Some(c)) = (base_val, chal_val) {
            usable_pairs += 1;
            match c.cmp(&b) {
                Ordering::Greater => challenger_wins += 1,
                Ordering::Less => baseline_wins += 1,
                Ordering::Equal => ties += 1,
            }
            let diff = c.get() - b.get();
            if diff.is_finite() {
                reduction = reduction
                    .and_then(|()| diff_reducer.observe(diff))
                    .and_then(|()| base_reducer.observe(b.get()))
                    .and_then(|()| chal_reducer.observe(c.get()));
            } else {
                reduction = Err(ReduceError::ArithmeticOverflow);
            }
        }
    }

    if (baseline_failures > 0 || challenger_failures > 0)
        && matches!(failure_policy, FailurePolicy::RejectCandidate)
    {
        return Err(PairedCompareError::EvaluationFailed {
            baseline_failures,
            challenger_failures,
        });
    }
    if (baseline_non_finite > 0 || challenger_non_finite > 0)
        && non_finite_policy == NonFinitePolicy::RejectCandidate
    {
        return Err(PairedCompareError::NonFinite {
            baseline_observations: baseline_non_finite,
            challenger_observations: challenger_non_finite,
        });
    }
    if usable_pairs == 0 {
        return Err(PairedCompareError::NoUsablePairs);
    }

    let finish = |res: Result<(), ReduceError>, r: ScalarReducer| {
        res.and_then(|()| r.finish())
            .map_err(|_| PairedCompareError::ArithmeticOverflow)
    };
    let difference = finish(reduction, diff_reducer)?;
    let baseline_summary = finish(Ok(()), base_reducer)?;
    let challenger_summary = finish(Ok(()), chal_reducer)?;

    Ok(PairedScalarSummary {
        baseline_candidate,
        challenger_candidate,
        requested_pairs,
        usable_pairs,
        challenger_wins,
        baseline_wins,
        ties,
        common_random_paired,
        difference,
        baseline: baseline_summary,
        challenger: challenger_summary,
    })
}

/// Checks that two replication-sorted slices cover the exact same replications.
fn verify_matching_replications<O, E>(
    left: &[&TrialRecord<O, E>],
    right: &[&TrialRecord<O, E>],
) -> Result<(), PairedCompareError> {
    for (a, b) in left.iter().zip(right.iter()) {
        let ra = a.key.replication();
        let rb = b.key.replication();
        if ra != rb {
            return Err(PairedCompareError::UnmatchedReplication(ra.min(rb)));
        }
    }
    if left.len() > right.len() {
        return Err(PairedCompareError::UnmatchedReplication(
            left[right.len()].key.replication(),
        ));
    }
    if right.len() > left.len() {
        return Err(PairedCompareError::UnmatchedReplication(
            right[left.len()].key.replication(),
        ));
    }
    Ok(())
}

/// Sorts references filtered for one candidate and rejects empty or duplicate
/// replication slices.
fn filter_unique_replications<O, E>(
    records: &[TrialRecord<O, E>],
    candidate: CandidateId,
) -> Result<Vec<&TrialRecord<O, E>>, PairedCompareError> {
    let mut ordered: Vec<&TrialRecord<O, E>> = records
        .iter()
        .filter(|record| record.key.candidate() == candidate)
        .collect();
    if ordered.is_empty() {
        return Err(PairedCompareError::Empty);
    }
    ordered.sort_unstable_by_key(|record| record.key.replication());
    for pair in ordered.windows(2) {
        if pair[0].key.replication() == pair[1].key.replication() {
            return Err(PairedCompareError::DuplicateReplication {
                candidate,
                replication: pair[0].key.replication(),
            });
        }
    }
    Ok(ordered)
}

/// Extracts two candidates from a mixed batch of [`TrialRecord`]s and compares
/// `challenger - baseline` across their matched replications without cloning
/// records.
///
/// # Errors
///
/// Returns [`PairedCompareError`] under the same conditions as
/// [`compare_paired_scalar`].
pub fn compare_paired_candidates<O, E>(
    records: &[TrialRecord<O, E>],
    baseline: CandidateId,
    challenger: CandidateId,
    score: impl Fn(&O) -> f64,
    failure_policy: FailurePolicy<FiniteValue>,
    non_finite_policy: NonFinitePolicy,
) -> Result<PairedScalarSummary, PairedCompareError> {
    let ordered_baseline = filter_unique_replications(records, baseline)?;
    let ordered_challenger = filter_unique_replications(records, challenger)?;
    compare_paired_refs(
        baseline,
        &ordered_baseline,
        challenger,
        &ordered_challenger,
        &score,
        failure_policy,
        non_finite_policy,
    )
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

    #[test]
    fn student_t_95_matches_table_and_asymptotic_tail() {
        assert_eq!(None, student_t_95_critical(0));
        assert_close(student_t_95_critical(1).expect("df=1"), 12.706_205);
        assert_close(student_t_95_critical(10).expect("df=10"), 2.228_139);
        assert_close(student_t_95_critical(30).expect("df=30"), 2.042_272);
        // Exact t_{0.975, 60} = 2.000_298, t_{0.975, 1000} = 1.962_339.
        let t60 = student_t_95_critical(60).expect("df=60");
        assert!((t60 - 2.000_298).abs() < 1e-5, "got {t60}");
        let t1000 = student_t_95_critical(1_000).expect("df=1000");
        assert!((t1000 - 1.962_339).abs() < 1e-5, "got {t1000}");
    }

    #[test]
    fn paired_crn_comparison_captures_variance_reduction_and_ci() {
        use crate::experiment::TrialStream;

        let seed = MasterSeed::from_u64(123);
        let schedule = FixedReplications::new(50, 9);
        let trials = schedule
            .trials(CandidateId::new(0), 0.0_f64)
            .chain(schedule.trials(CandidateId::new(1), 1.5_f64));
        let records = run_serial(seed, trials, |&shift, ctx| {
            let shared = ctx
                .common_rng(TrialStream::TrialNoise, "demand")
                .uniform_f64()
                * 100.0;
            let small_noise = (ctx
                .independent_rng(TrialStream::ExecutionNoise, "jitter")
                .uniform_f64()
                - 0.5)
                * 0.4;
            Ok::<f64, ()>(shared + shift + small_noise)
        })
        .expect("serial run");

        let paired = compare_paired_candidates(
            &records,
            CandidateId::new(0),
            CandidateId::new(1),
            |&v| v,
            FailurePolicy::RejectCandidate,
            NonFinitePolicy::RejectCandidate,
        )
        .expect("paired summary");

        assert!(paired.is_common_random_paired());
        assert_eq!(50, paired.requested_pairs());
        assert_eq!(50, paired.usable_pairs());
        assert_eq!(50, paired.challenger_wins());
        assert_eq!(0, paired.baseline_wins());
        assert_eq!(0, paired.ties());

        let (lo, hi) = paired.ci_95().expect("ci_95");
        assert!(
            lo > 1.0 && hi < 2.0,
            "expected tight CI around 1.5: ({lo}, {hi})"
        );
        let vrr = paired
            .variance_reduction_ratio()
            .expect("variance reduction ratio");
        assert!(
            vrr > 100.0,
            "expected huge CRN variance reduction, got {vrr}"
        );
    }

    #[test]
    fn paired_comparison_rejects_mismatched_and_invalid_inputs() {
        let seed = MasterSeed::from_u64(5);
        let a = run_serial(
            seed,
            FixedReplications::new(3, 1).trials(CandidateId::new(0), ()),
            |(), ctx| Ok::<f64, ()>(f64::from(ctx.key().replication().value())),
        )
        .expect("a");
        let b = run_serial(
            seed,
            FixedReplications::new(2, 1).trials(CandidateId::new(1), ()),
            |(), ctx| Ok::<f64, ()>(f64::from(ctx.key().replication().value())),
        )
        .expect("b");

        let err = compare_paired_scalar(
            &a,
            &b,
            |&v| v,
            FailurePolicy::RejectCandidate,
            NonFinitePolicy::RejectCandidate,
        )
        .expect_err("unmatched replication must fail");
        assert_eq!(
            PairedCompareError::UnmatchedReplication(ReplicationId::new(2)),
            err
        );
        assert_ne!("", err.to_string());

        let empty_err = compare_paired_scalar::<f64, ()>(
            &[],
            &b,
            |&v| v,
            FailurePolicy::RejectCandidate,
            NonFinitePolicy::RejectCandidate,
        )
        .expect_err("empty must fail");
        assert_eq!(PairedCompareError::Empty, empty_err);
    }
}
