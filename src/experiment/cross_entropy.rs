//! Cross-entropy optimization over bounded, normalized vectors.

use std::cmp::Ordering;
use std::f64::consts::TAU;
use std::fmt;

use rand::{Rng, RngExt as _};
use rand_distr::StandardNormal;

use super::ObjectiveScore;

/// Largest population for which full sorting beats generic selection overhead.
const FULL_SORT_POPULATION_THRESHOLD: usize = 12;

/// Geometry used by one normalized search dimension.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CrossEntropyDimension {
    /// A bounded coordinate reflected at zero and one.
    #[default]
    Linear,
    /// A periodic coordinate wrapped at zero and one.
    Circular,
}

/// Configuration for a diagonal Gaussian cross-entropy optimizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossEntropyConfig<const N: usize> {
    /// Initial center of the search distribution.
    initial_mean: [f64; N],
    /// Initial standard deviation of each search dimension.
    initial_standard_deviation: [f64; N],
    /// Lower bound on each standard deviation.
    minimum_standard_deviation: [f64; N],
    /// Geometry of each search dimension.
    dimensions: [CrossEntropyDimension; N],
    /// Fraction of valid samples used to update the distribution.
    elite_fraction: f64,
    /// Weight assigned to elite statistics during an update.
    learning_rate: f64,
}

impl<const N: usize> CrossEntropyConfig<N> {
    /// Creates a configuration with linear dimensions, a 10% elite fraction,
    /// a 0.7 learning rate, and standard-deviation floors of `1e-6`.
    #[must_use]
    pub const fn new(initial_mean: [f64; N], initial_standard_deviation: [f64; N]) -> Self {
        Self {
            initial_mean,
            initial_standard_deviation,
            minimum_standard_deviation: [1.0e-6; N],
            dimensions: [CrossEntropyDimension::Linear; N],
            elite_fraction: 0.1,
            learning_rate: 0.7,
        }
    }

    /// Sets the geometry of every search dimension.
    #[must_use]
    pub const fn with_dimensions(mut self, dimensions: [CrossEntropyDimension; N]) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Sets the lower bound on each standard deviation.
    #[must_use]
    pub const fn with_minimum_standard_deviation(
        mut self,
        minimum_standard_deviation: [f64; N],
    ) -> Self {
        self.minimum_standard_deviation = minimum_standard_deviation;
        self
    }

    /// Sets the fraction of non-NaN samples treated as elites.
    ///
    /// The elite count is `ceil(valid_samples * elite_fraction)` and is always
    /// at least one when a population contains a valid sample.
    #[must_use]
    pub const fn with_elite_fraction(mut self, elite_fraction: f64) -> Self {
        self.elite_fraction = elite_fraction;
        self
    }

    /// Sets the weight assigned to elite statistics during each update.
    ///
    /// Zero keeps the proposal distribution fixed while still tracking the
    /// best sample. One replaces the distribution with the elite statistics.
    #[must_use]
    pub const fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }
}

/// A normalized point and its objective score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossEntropySample<const N: usize> {
    /// Point in the normalized search space.
    pub point: [f64; N],
    /// Objective score to maximize. NaN scores are ignored.
    pub score: ObjectiveScore,
}

impl<const N: usize> CrossEntropySample<N> {
    /// Creates a scored sample.
    #[must_use]
    pub const fn new(point: [f64; N], score: ObjectiveScore) -> Self {
        Self { point, score }
    }
}

/// Summary of one successful cross-entropy distribution update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossEntropyUpdate {
    /// One-based number of completed updates.
    pub generation: u64,
    /// Number of samples with non-NaN scores.
    pub valid_samples: usize,
    /// Number of samples used to fit the distribution.
    pub elite_samples: usize,
    /// Highest score in this population.
    pub generation_best_score: ObjectiveScore,
    /// Highest score observed across all populations.
    pub best_score: ObjectiveScore,
}

/// Invalid cross-entropy configuration or population data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CrossEntropyError {
    /// A zero-dimensional optimizer was requested.
    EmptySearchSpace,
    /// An initial mean was non-finite or outside its normalized domain.
    InvalidInitialMean {
        /// Index of the invalid dimension.
        dimension: usize,
    },
    /// An initial standard deviation was not finite and positive.
    InvalidInitialStandardDeviation {
        /// Index of the invalid dimension.
        dimension: usize,
    },
    /// A minimum standard deviation was not finite and positive.
    InvalidMinimumStandardDeviation {
        /// Index of the invalid dimension.
        dimension: usize,
    },
    /// A standard-deviation floor exceeded its initial value.
    MinimumStandardDeviationExceedsInitial {
        /// Index of the invalid dimension.
        dimension: usize,
    },
    /// The elite fraction was not finite and in `(0, 1]`.
    InvalidEliteFraction,
    /// The learning rate was not finite and in `[0, 1]`.
    InvalidLearningRate,
    /// A scored point contained an invalid normalized coordinate.
    InvalidSamplePoint {
        /// Index of the sample in the supplied population.
        sample: usize,
        /// Index of the invalid dimension.
        dimension: usize,
    },
    /// A caller-provided standard-normal value was non-finite.
    InvalidStandardNormal {
        /// Index of the dimension being sampled.
        dimension: usize,
    },
}

impl fmt::Display for CrossEntropyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySearchSpace => formatter.write_str("the search space has no dimensions"),
            Self::InvalidInitialMean { dimension } => {
                write!(
                    formatter,
                    "initial mean for dimension {dimension} is invalid"
                )
            }
            Self::InvalidInitialStandardDeviation { dimension } => write!(
                formatter,
                "initial standard deviation for dimension {dimension} is invalid"
            ),
            Self::InvalidMinimumStandardDeviation { dimension } => write!(
                formatter,
                "minimum standard deviation for dimension {dimension} is invalid"
            ),
            Self::MinimumStandardDeviationExceedsInitial { dimension } => write!(
                formatter,
                "minimum standard deviation exceeds the initial value for dimension {dimension}"
            ),
            Self::InvalidEliteFraction => formatter.write_str("elite fraction is invalid"),
            Self::InvalidLearningRate => formatter.write_str("learning rate is invalid"),
            Self::InvalidSamplePoint { sample, dimension } => write!(
                formatter,
                "sample {sample} has an invalid coordinate in dimension {dimension}"
            ),
            Self::InvalidStandardNormal { dimension } => write!(
                formatter,
                "standard-normal value for dimension {dimension} is invalid"
            ),
        }
    }
}

impl std::error::Error for CrossEntropyError {}

/// Ask/tell diagonal Gaussian cross-entropy optimizer.
///
/// Points occupy normalized coordinates. Linear dimensions are reflected at
/// `[0, 1]`; circular dimensions are wrapped to `[0, 1)`. [`Self::ask`] draws
/// proposals, while [`Self::tell`] fits the distribution to the highest-scoring
/// fraction of a population. The optimizer neither evaluates objectives nor
/// allocates storage for a population, allowing callers to batch, parallelize,
/// and replicate evaluations as their domain requires.
#[derive(Clone, Debug)]
pub struct CrossEntropyOptimizer<const N: usize> {
    /// Current distribution center.
    mean: [f64; N],
    /// Current marginal standard deviations.
    standard_deviation: [f64; N],
    /// Lower bounds on marginal standard deviations.
    minimum_standard_deviation: [f64; N],
    /// Geometry of each dimension.
    dimensions: [CrossEntropyDimension; N],
    /// Fraction of valid samples used for fitting.
    elite_fraction: f64,
    /// Weight assigned to newly fitted statistics.
    learning_rate: f64,
    /// Number of successful updates.
    generation: u64,
    /// Best sample observed so far.
    best: Option<CrossEntropySample<N>>,
}

impl<const N: usize> CrossEntropyOptimizer<N> {
    /// Creates an optimizer after validating its configuration.
    ///
    /// # Errors
    ///
    /// Returns a [`CrossEntropyError`] identifying the first invalid field.
    pub fn new(config: CrossEntropyConfig<N>) -> Result<Self, CrossEntropyError> {
        validate_config(&config)?;
        Ok(Self {
            mean: config.initial_mean,
            standard_deviation: config.initial_standard_deviation,
            minimum_standard_deviation: config.minimum_standard_deviation,
            dimensions: config.dimensions,
            elite_fraction: config.elite_fraction,
            learning_rate: config.learning_rate,
            generation: 0,
            best: None,
        })
    }

    /// Draws one point using independent standard-normal variates from `rng`.
    #[must_use]
    #[inline]
    pub fn ask<Random>(&self, rng: &mut Random) -> [f64; N]
    where
        Random: Rng + ?Sized,
    {
        self.sample_unchecked(|_| rng.sample::<f64, _>(StandardNormal))
    }

    /// Fills a caller-owned slice with proposals from the current distribution.
    #[inline]
    pub fn ask_into<Random>(&self, rng: &mut Random, points: &mut [[f64; N]])
    where
        Random: Rng + ?Sized,
    {
        for point in points {
            *point = self.ask(rng);
        }
    }

    /// Draws one point from caller-provided standard-normal variates.
    ///
    /// The callback receives the dimension index. This is useful when random
    /// values are derived from stable candidate and coordinate identifiers.
    ///
    /// # Errors
    ///
    /// Returns [`CrossEntropyError::InvalidStandardNormal`] when the callback
    /// supplies a non-finite value.
    pub fn ask_with_standard_normal(
        &self,
        mut standard_normal: impl FnMut(usize) -> f64,
    ) -> Result<[f64; N], CrossEntropyError> {
        let mut invalid_dimension = None;
        let point = self.sample_unchecked(|dimension| {
            let value = standard_normal(dimension);
            if !value.is_finite() && invalid_dimension.is_none() {
                invalid_dimension = Some(dimension);
            }
            value
        });
        invalid_dimension.map_or(Ok(point), |dimension| {
            Err(CrossEntropyError::InvalidStandardNormal { dimension })
        })
    }

    /// Updates the proposal distribution from scored samples.
    ///
    /// This maximizes `score`, ignores NaN scores, and accepts infinite scores.
    /// Every coordinate belonging to a non-NaN score must be normalized for its
    /// dimension. Samples are reordered in place so the elite subset precedes
    /// the rest of the population without an internal allocation. Small
    /// populations are sorted; larger populations are partitioned at the elite
    /// boundary. If no score is usable, this returns `Ok(None)` unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`CrossEntropyError::InvalidSamplePoint`] before mutation when
    /// a non-NaN score is paired with an invalid normalized coordinate.
    pub fn tell(
        &mut self,
        samples: &mut [CrossEntropySample<N>],
    ) -> Result<Option<CrossEntropyUpdate>, CrossEntropyError> {
        validate_samples(samples, &self.dimensions)?;
        let valid_samples = samples
            .iter()
            .filter(|sample| !sample.score.is_nan())
            .count();
        if valid_samples == 0 {
            return Ok(None);
        }

        let elite_samples = elite_count(valid_samples, self.elite_fraction);
        if samples.len() <= FULL_SORT_POPULATION_THRESHOLD {
            if valid_samples == samples.len() {
                samples.sort_unstable_by(compare_valid_samples);
            } else {
                samples.sort_unstable_by(compare_samples);
            }
        } else if valid_samples == samples.len() {
            let _ = samples.select_nth_unstable_by(elite_samples - 1, compare_valid_samples);
        } else {
            let _ = samples.select_nth_unstable_by(elite_samples - 1, compare_samples);
        }
        let elites = &samples[..elite_samples];
        let mut generation_best = elites[0];
        for sample in &elites[1..] {
            if sample.score >= generation_best.score {
                generation_best = *sample;
            }
        }
        if self
            .best
            .is_none_or(|best| generation_best.score >= best.score)
        {
            self.best = Some(generation_best);
        }

        self.fit(elites);
        self.generation += 1;
        let best_score = self.best.map_or(generation_best.score, |best| best.score);
        Ok(Some(CrossEntropyUpdate {
            generation: self.generation,
            valid_samples,
            elite_samples,
            generation_best_score: generation_best.score,
            best_score,
        }))
    }

    /// Returns the current distribution center.
    #[must_use]
    pub const fn mean(&self) -> &[f64; N] {
        &self.mean
    }

    /// Returns the current marginal standard deviations.
    #[must_use]
    pub const fn standard_deviation(&self) -> &[f64; N] {
        &self.standard_deviation
    }

    /// Returns the number of successful distribution updates.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the highest-scoring sample observed so far.
    #[must_use]
    pub const fn best(&self) -> Option<&CrossEntropySample<N>> {
        self.best.as_ref()
    }

    /// Samples all dimensions without validating the standard-normal values.
    #[inline]
    fn sample_unchecked(&self, mut standard_normal: impl FnMut(usize) -> f64) -> [f64; N] {
        std::array::from_fn(|dimension| {
            let value = self.standard_deviation[dimension]
                .mul_add(standard_normal(dimension), self.mean[dimension]);
            normalize(value, self.dimensions[dimension])
        })
    }

    /// Fits and smooths distribution parameters using `elites`.
    fn fit(&mut self, elites: &[CrossEntropySample<N>]) {
        let inverse_elite_count = 1.0 / elites.len() as f64;
        let retained_weight = 1.0 - self.learning_rate;

        for dimension in 0..N {
            let old_mean = self.mean[dimension];
            let elite_mean = match self.dimensions[dimension] {
                CrossEntropyDimension::Linear => {
                    elites
                        .iter()
                        .map(|sample| sample.point[dimension])
                        .sum::<f64>()
                        * inverse_elite_count
                }
                CrossEntropyDimension::Circular => circular_mean(elites, dimension, old_mean),
            };
            let next_mean = match self.dimensions[dimension] {
                CrossEntropyDimension::Linear => {
                    old_mean.mul_add(retained_weight, elite_mean * self.learning_rate)
                }
                CrossEntropyDimension::Circular => normalize(
                    self.learning_rate
                        .mul_add(circular_delta(old_mean, elite_mean), old_mean),
                    CrossEntropyDimension::Circular,
                ),
            };
            let elite_variance = elites
                .iter()
                .map(|sample| {
                    let difference = match self.dimensions[dimension] {
                        CrossEntropyDimension::Linear => sample.point[dimension] - elite_mean,
                        CrossEntropyDimension::Circular => {
                            circular_delta(elite_mean, sample.point[dimension])
                        }
                    };
                    difference * difference
                })
                .sum::<f64>()
                * inverse_elite_count;
            let old_variance = self.standard_deviation[dimension].powi(2);
            let next_variance =
                old_variance.mul_add(retained_weight, elite_variance * self.learning_rate);

            self.mean[dimension] = next_mean;
            self.standard_deviation[dimension] = next_variance
                .sqrt()
                .max(self.minimum_standard_deviation[dimension]);
        }
    }
}

/// Validates every configuration field.
fn validate_config<const N: usize>(
    config: &CrossEntropyConfig<N>,
) -> Result<(), CrossEntropyError> {
    if N == 0 {
        return Err(CrossEntropyError::EmptySearchSpace);
    }
    if !config.elite_fraction.is_finite()
        || config.elite_fraction <= 0.0
        || config.elite_fraction > 1.0
    {
        return Err(CrossEntropyError::InvalidEliteFraction);
    }
    if !config.learning_rate.is_finite() || config.learning_rate < 0.0 || config.learning_rate > 1.0
    {
        return Err(CrossEntropyError::InvalidLearningRate);
    }

    for dimension in 0..N {
        if !valid_coordinate(config.initial_mean[dimension], config.dimensions[dimension]) {
            return Err(CrossEntropyError::InvalidInitialMean { dimension });
        }
        if !config.initial_standard_deviation[dimension].is_finite()
            || config.initial_standard_deviation[dimension] <= 0.0
        {
            return Err(CrossEntropyError::InvalidInitialStandardDeviation { dimension });
        }
        if !config.minimum_standard_deviation[dimension].is_finite()
            || config.minimum_standard_deviation[dimension] <= 0.0
        {
            return Err(CrossEntropyError::InvalidMinimumStandardDeviation { dimension });
        }
        if config.minimum_standard_deviation[dimension]
            > config.initial_standard_deviation[dimension]
        {
            return Err(CrossEntropyError::MinimumStandardDeviationExceedsInitial { dimension });
        }
    }
    Ok(())
}

/// Validates points that could influence a distribution update.
fn validate_samples<const N: usize>(
    samples: &[CrossEntropySample<N>],
    dimensions: &[CrossEntropyDimension; N],
) -> Result<(), CrossEntropyError> {
    for (sample_index, sample) in samples.iter().enumerate() {
        if sample.score.is_nan() {
            continue;
        }
        for (dimension, geometry) in dimensions.iter().copied().enumerate() {
            if !valid_coordinate(sample.point[dimension], geometry) {
                return Err(CrossEntropyError::InvalidSamplePoint {
                    sample: sample_index,
                    dimension,
                });
            }
        }
    }
    Ok(())
}

/// Returns whether a coordinate belongs to its normalized domain.
#[allow(clippy::manual_range_contains)]
#[inline]
fn valid_coordinate(value: f64, dimension: CrossEntropyDimension) -> bool {
    match dimension {
        CrossEntropyDimension::Linear => value >= 0.0 && value <= 1.0,
        CrossEntropyDimension::Circular => value >= 0.0 && value < 1.0,
    }
}

/// Orders valid scores from greatest to least and places NaNs last.
fn compare_samples<const N: usize>(
    left: &CrossEntropySample<N>,
    right: &CrossEntropySample<N>,
) -> Ordering {
    match (left.score.is_nan(), right.score.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => right.score.total_cmp(&left.score),
    }
}

/// Orders two known-valid scores from greatest to least.
#[inline]
fn compare_valid_samples<const N: usize>(
    left: &CrossEntropySample<N>,
    right: &CrossEntropySample<N>,
) -> Ordering {
    right.score.total_cmp(&left.score)
}

/// Computes the ceiling of the elite fraction without a float-to-int cast.
fn elite_count(valid_samples: usize, elite_fraction: f64) -> usize {
    let target = valid_samples as f64 * elite_fraction;
    let mut count = 1;
    while count < valid_samples && (count as f64) < target {
        count += 1;
    }
    count
}

/// Computes a circular mean, retaining `fallback` for an undefined resultant.
fn circular_mean<const N: usize>(
    elites: &[CrossEntropySample<N>],
    dimension: usize,
    fallback: f64,
) -> f64 {
    let (sine_sum, cosine_sum) = elites.iter().fold((0.0, 0.0), |(sines, cosines), sample| {
        let (sine, cosine) = (sample.point[dimension] * TAU).sin_cos();
        (sines + sine, cosines + cosine)
    });
    if sine_sum.hypot(cosine_sum) <= f64::EPSILON * elites.len() as f64 {
        fallback
    } else {
        normalize(
            sine_sum.atan2(cosine_sum) / TAU,
            CrossEntropyDimension::Circular,
        )
    }
}

/// Returns the shortest signed displacement from `from` to `to`.
#[inline]
fn circular_delta(from: f64, to: f64) -> f64 {
    (to - from + 0.5).rem_euclid(1.0) - 0.5
}

/// Projects a value into its normalized domain.
#[inline]
fn normalize(value: f64, dimension: CrossEntropyDimension) -> f64 {
    match dimension {
        CrossEntropyDimension::Linear => {
            let reflected = value.rem_euclid(2.0);
            if reflected <= 1.0 {
                reflected
            } else {
                2.0 - reflected
            }
        }
        CrossEntropyDimension::Circular => value.rem_euclid(1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng as _;

    /// Creates an optimizer or fails with its configuration error.
    fn optimizer<const N: usize>(config: CrossEntropyConfig<N>) -> CrossEntropyOptimizer<N> {
        match CrossEntropyOptimizer::new(config) {
            Ok(optimizer) => optimizer,
            Err(error) => panic!("unexpected configuration error: {error}"),
        }
    }

    /// Asserts that constructing an optimizer returns `expected`.
    fn assert_config_error<const N: usize>(
        config: CrossEntropyConfig<N>,
        expected: CrossEntropyError,
    ) {
        match CrossEntropyOptimizer::new(config) {
            Ok(_) => panic!("invalid configuration was accepted"),
            Err(actual) => assert_eq!(actual, expected),
        }
    }

    /// Asserts that two floating-point values differ by no more than `tolerance`.
    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn rejects_invalid_configurations() {
        assert_config_error(
            CrossEntropyConfig::<0>::new([], []),
            CrossEntropyError::EmptySearchSpace,
        );
        assert_config_error(
            CrossEntropyConfig::new([f64::NAN], [0.2]),
            CrossEntropyError::InvalidInitialMean { dimension: 0 },
        );
        assert_config_error(
            CrossEntropyConfig::new([0.5], [0.0]),
            CrossEntropyError::InvalidInitialStandardDeviation { dimension: 0 },
        );
        assert_config_error(
            CrossEntropyConfig::new([0.5], [0.2]).with_minimum_standard_deviation([f64::INFINITY]),
            CrossEntropyError::InvalidMinimumStandardDeviation { dimension: 0 },
        );
        assert_config_error(
            CrossEntropyConfig::new([0.5], [0.2]).with_minimum_standard_deviation([0.3]),
            CrossEntropyError::MinimumStandardDeviationExceedsInitial { dimension: 0 },
        );
        assert_config_error(
            CrossEntropyConfig::new([0.5], [0.2]).with_elite_fraction(0.0),
            CrossEntropyError::InvalidEliteFraction,
        );
        assert_config_error(
            CrossEntropyConfig::new([0.5], [0.2]).with_learning_rate(1.1),
            CrossEntropyError::InvalidLearningRate,
        );
        assert_config_error(
            CrossEntropyConfig::new([1.0], [0.2])
                .with_dimensions([CrossEntropyDimension::Circular]),
            CrossEntropyError::InvalidInitialMean { dimension: 0 },
        );
    }

    #[test]
    fn sampling_reflects_linear_and_wraps_circular_dimensions() {
        let search = optimizer(
            CrossEntropyConfig::new([0.9, 0.9], [0.5, 0.5]).with_dimensions([
                CrossEntropyDimension::Linear,
                CrossEntropyDimension::Circular,
            ]),
        );
        let sampled = search.ask_with_standard_normal(|_| 1.0);
        let Ok(sampled) = sampled else {
            panic!("finite standard-normal values must sample successfully");
        };

        assert_close(sampled[0], 0.6, 1.0e-12);
        assert_close(sampled[1], 0.4, 1.0e-12);
        assert_eq!(
            search.ask_with_standard_normal(|_| f64::INFINITY),
            Err(CrossEntropyError::InvalidStandardNormal { dimension: 0 })
        );
    }

    #[test]
    fn seeded_sampling_is_reproducible_and_ask_into_fills_every_point() {
        let search = optimizer(CrossEntropyConfig::new([0.5, 0.5], [0.2, 0.2]));
        let mut first_rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut second_rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut batch = [[0.0; 2]; 4];
        search.ask_into(&mut first_rng, &mut batch);

        assert_eq!(batch, std::array::from_fn(|_| search.ask(&mut second_rng)));
        assert!(batch.iter().flatten().all(|coordinate| {
            coordinate.is_finite() && *coordinate >= 0.0 && *coordinate <= 1.0
        }));
    }

    #[test]
    fn tell_fits_elites_tracks_best_and_ignores_nan_scores() {
        let mut search = optimizer(
            CrossEntropyConfig::new([0.5], [0.4])
                .with_minimum_standard_deviation([0.01])
                .with_elite_fraction(0.5)
                .with_learning_rate(1.0),
        );
        let mut samples = [
            CrossEntropySample::new([0.1], 1.0),
            CrossEntropySample::new([0.2], 4.0),
            CrossEntropySample::new([0.8], f64::NAN),
            CrossEntropySample::new([0.4], 3.0),
            CrossEntropySample::new([0.9], 2.0),
        ];
        let update = search.tell(&mut samples);
        let Ok(Some(update)) = update else {
            panic!("population contains valid scores");
        };

        assert_eq!(update.generation, 1);
        assert_eq!(update.valid_samples, 4);
        assert_eq!(update.elite_samples, 2);
        assert_close(update.generation_best_score, 4.0, f64::EPSILON);
        assert_eq!(search.best(), Some(&CrossEntropySample::new([0.2], 4.0)));
        assert_close(search.mean()[0], 0.3, 1.0e-12);
        assert_close(search.standard_deviation()[0], 0.1, 1.0e-12);
        assert!(samples[4].score.is_nan());
    }

    #[test]
    fn smoothing_uses_variance_and_standard_deviation_floor() {
        let mut search = optimizer(
            CrossEntropyConfig::new([0.2], [0.4])
                .with_minimum_standard_deviation([0.3])
                .with_elite_fraction(1.0)
                .with_learning_rate(0.5),
        );
        let mut samples = [
            CrossEntropySample::new([0.6], 1.0),
            CrossEntropySample::new([0.6], 2.0),
        ];
        let result = search.tell(&mut samples);
        assert!(result.is_ok());

        assert_close(search.mean()[0], 0.4, 1.0e-12);
        assert_close(search.standard_deviation()[0], 0.3, 1.0e-12);
    }

    #[test]
    fn circular_statistics_fit_across_the_wrap_boundary() {
        let mut search = optimizer(
            CrossEntropyConfig::new([0.25], [0.4])
                .with_dimensions([CrossEntropyDimension::Circular])
                .with_minimum_standard_deviation([0.001])
                .with_elite_fraction(1.0)
                .with_learning_rate(1.0),
        );
        let mut samples = [
            CrossEntropySample::new([0.99], 1.0),
            CrossEntropySample::new([0.01], 1.0),
        ];
        let result = search.tell(&mut samples);
        assert!(result.is_ok());

        assert!(search.mean()[0] < 1.0e-12 || search.mean()[0] > 1.0 - 1.0e-12);
        assert_close(search.standard_deviation()[0], 0.01, 1.0e-12);
    }

    #[test]
    fn circular_mean_retains_fallback_for_antipodal_elites() {
        let mut search = optimizer(
            CrossEntropyConfig::new([0.25], [0.4])
                .with_dimensions([CrossEntropyDimension::Circular])
                .with_minimum_standard_deviation([0.001])
                .with_elite_fraction(1.0)
                .with_learning_rate(1.0),
        );
        let mut samples = [
            CrossEntropySample::new([0.0], 1.0),
            CrossEntropySample::new([0.5], 1.0),
        ];
        let result = search.tell(&mut samples);
        assert!(result.is_ok());

        assert_close(search.mean()[0], 0.25, f64::EPSILON);
        assert_close(search.standard_deviation()[0], 0.25, f64::EPSILON);
    }

    #[test]
    fn zero_learning_rate_provides_a_fixed_proposal_distribution() {
        let mut search = optimizer(
            CrossEntropyConfig::new([0.5], [0.2])
                .with_elite_fraction(1.0)
                .with_learning_rate(0.0),
        );
        let mut samples = [CrossEntropySample::new([0.9], 3.0)];
        let result = search.tell(&mut samples);
        assert!(result.is_ok());

        assert_close(search.mean()[0], 0.5, f64::EPSILON);
        assert_close(search.standard_deviation()[0], 0.2, f64::EPSILON);
        assert_eq!(search.best(), Some(&CrossEntropySample::new([0.9], 3.0)));
    }

    #[test]
    fn unusable_population_does_not_mutate_the_optimizer() {
        let mut search = optimizer(CrossEntropyConfig::new([0.5], [0.2]));
        let mut samples = [CrossEntropySample::new([f64::NAN], f64::NAN)];

        assert_eq!(search.tell(&mut samples), Ok(None));
        assert_eq!(search.generation(), 0);
        assert_eq!(search.best(), None);
        assert_close(search.mean()[0], 0.5, f64::EPSILON);
    }

    #[test]
    fn invalid_scored_point_is_rejected_before_mutation() {
        let mut search = optimizer(CrossEntropyConfig::new([0.5], [0.2]));
        let mut samples = [CrossEntropySample::new([1.1], 1.0)];

        assert_eq!(
            search.tell(&mut samples),
            Err(CrossEntropyError::InvalidSamplePoint {
                sample: 0,
                dimension: 0
            })
        );
        assert_eq!(search.generation(), 0);
        assert_eq!(search.best(), None);
    }

    #[test]
    fn best_sample_persists_across_generations_and_accepts_infinity() {
        let mut search = optimizer(
            CrossEntropyConfig::new([0.5], [0.2])
                .with_elite_fraction(1.0)
                .with_learning_rate(0.0),
        );
        let mut first = [CrossEntropySample::new([0.2], f64::INFINITY)];
        let mut second = [CrossEntropySample::new([0.8], 10.0)];
        let first_result = search.tell(&mut first);
        let second_result = search.tell(&mut second);
        assert!(first_result.is_ok());
        assert!(second_result.is_ok());

        assert_eq!(search.generation(), 2);
        assert_eq!(
            search.best(),
            Some(&CrossEntropySample::new([0.2], f64::INFINITY))
        );
    }

    #[test]
    fn converges_on_a_seeded_bounded_objective() {
        let mut search = optimizer(
            CrossEntropyConfig::new([0.5, 0.5], [0.35, 0.35])
                .with_minimum_standard_deviation([1.0e-4; 2])
                .with_elite_fraction(0.2)
                .with_learning_rate(0.7),
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut samples = [CrossEntropySample::new([0.0; 2], f64::NAN); 40];

        for _ in 0..20 {
            for sample in &mut samples {
                sample.point = search.ask(&mut rng);
                let horizontal_error = sample.point[0] - 0.2;
                let vertical_error = sample.point[1] - 0.8;
                sample.score =
                    -horizontal_error.mul_add(horizontal_error, vertical_error * vertical_error);
            }
            let result = search.tell(&mut samples);
            assert!(result.is_ok());
        }

        let Some(best) = search.best() else {
            panic!("search must observe a valid score");
        };
        assert_close(best.point[0], 0.2, 0.01);
        assert_close(best.point[1], 0.8, 0.01);
        assert!(best.score > -1.0e-4);
    }
}
