//! Cross-entropy optimization over mixed continuous, circular, integer, and
//! categorical search spaces.

use std::cmp::Ordering;
use std::f64::consts::TAU;
use std::fmt;

use rand::{Rng, RngExt as _};
use rand_distr::StandardNormal;

use super::{CrossEntropyUpdate, ObjectiveScore};

/// Largest population for which full sorting beats partial selection overhead.
const FULL_SORT_POPULATION_THRESHOLD: usize = 12;

/// Largest category count tracked in a stack buffer during `fit_categorical`.
const STACK_CATEGORY_COUNTS: usize = 64;

/// Largest elite slice normalized into a stack buffer during `fit_dimension`.
const STACK_ELITE_BUFFER: usize = 128;

/// Specification of one dimension in a [`MixedCrossEntropyOptimizer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MixedDimension {
    /// Bounded continuous parameter in `[lower, upper]` (reflected Gaussian).
    Continuous {
        /// Inclusive lower bound.
        lower: f64,
        /// Inclusive upper bound (`>= lower`).
        upper: f64,
        /// Optional initial mean in `[lower, upper]` (defaults to midpoint).
        initial_mean: Option<f64>,
        /// Optional initial standard deviation in raw units (defaults to
        /// `0.25 * (upper - lower)`).
        initial_std: Option<f64>,
        /// Optional standard deviation floor in raw units (defaults to
        /// `1e-6 * (upper - lower)`).
        min_std: Option<f64>,
    },
    /// Periodic continuous parameter in `[lower, upper)` (wrapped Gaussian).
    Circular {
        /// Inclusive lower bound.
        lower: f64,
        /// Exclusive upper bound (`> lower`).
        upper: f64,
        /// Optional initial mean in `[lower, upper)` (defaults to midpoint).
        initial_mean: Option<f64>,
        /// Optional initial standard deviation in raw units (defaults to
        /// `0.25 * (upper - lower)`).
        initial_std: Option<f64>,
        /// Optional standard deviation floor in raw units.
        min_std: Option<f64>,
    },
    /// Discrete unordered categorical choice in `0..categories` with a learned
    /// probability vector.
    Categorical {
        /// Number of distinct categories (must be `>= 1`).
        categories: usize,
        /// Minimum probability per category after each update (`min_probability * categories <= 1.0`).
        min_probability: f64,
    },
    /// Bounded ordered integer parameter in `min..=max`, modeled by an ordinal
    /// reflected Gaussian quantized to integers.
    Integer {
        /// Inclusive lower bound.
        min: i64,
        /// Inclusive upper bound (`>= min`).
        max: i64,
    },
}

impl MixedDimension {
    /// Creates a bounded continuous dimension over `[lower, upper]` with
    /// default midpoint mean and quarter-span standard deviation.
    #[must_use]
    pub const fn continuous(lower: f64, upper: f64) -> Self {
        Self::Continuous {
            lower,
            upper,
            initial_mean: None,
            initial_std: None,
            min_std: None,
        }
    }

    /// Creates a periodic circular dimension over `[lower, upper)`.
    #[must_use]
    pub const fn circular(lower: f64, upper: f64) -> Self {
        Self::Circular {
            lower,
            upper,
            initial_mean: None,
            initial_std: None,
            min_std: None,
        }
    }

    /// Creates a categorical dimension with `categories` choices and a small
    /// default probability floor (`0.01 / categories`, capped at `0.5 / categories`).
    #[must_use]
    pub fn categorical(categories: usize) -> Self {
        let floor = if categories == 0 {
            0.0
        } else {
            (0.01 / categories as f64).min(0.5 / categories as f64)
        };
        Self::Categorical {
            categories,
            min_probability: floor,
        }
    }

    /// Creates a categorical dimension with an explicit per-category
    /// probability floor.
    #[must_use]
    pub const fn categorical_with_floor(categories: usize, min_probability: f64) -> Self {
        Self::Categorical {
            categories,
            min_probability,
        }
    }

    /// Creates a bounded ordered integer dimension over `min..=max`.
    #[must_use]
    pub const fn integer(min: i64, max: i64) -> Self {
        Self::Integer { min, max }
    }
}

/// One coordinate value produced by and supplied to
/// [`MixedCrossEntropyOptimizer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MixedValue {
    /// Value for a [`MixedDimension::Continuous`] or [`MixedDimension::Circular`] dimension.
    Continuous(f64),
    /// Category index for a [`MixedDimension::Categorical`] dimension.
    Category(usize),
    /// Integer value for a [`MixedDimension::Integer`] dimension.
    Integer(i64),
}

impl MixedValue {
    /// Returns the continuous value if this is [`Self::Continuous`].
    #[must_use]
    pub const fn as_f64(self) -> Option<f64> {
        match self {
            Self::Continuous(v) => Some(v),
            Self::Category(_) | Self::Integer(_) => None,
        }
    }

    /// Returns the category index if this is [`Self::Category`].
    #[must_use]
    pub const fn as_category(self) -> Option<usize> {
        match self {
            Self::Category(c) => Some(c),
            Self::Continuous(_) | Self::Integer(_) => None,
        }
    }

    /// Returns the integer value if this is [`Self::Integer`].
    #[must_use]
    pub const fn as_i64(self) -> Option<i64> {
        match self {
            Self::Integer(v) => Some(v),
            Self::Continuous(_) | Self::Category(_) => None,
        }
    }

    /// Deterministic total ordering used to break ties between equal-scoring samples.
    fn total_cmp(self, other: Self) -> Ordering {
        match (self, other) {
            (Self::Continuous(a), Self::Continuous(b)) => a.total_cmp(&b),
            (Self::Category(a), Self::Category(b)) => a.cmp(&b),
            (Self::Integer(a), Self::Integer(b)) => a.cmp(&b),
            (Self::Continuous(_), Self::Category(_) | Self::Integer(_))
            | (Self::Category(_), Self::Integer(_)) => Ordering::Less,
            (Self::Category(_), Self::Continuous(_))
            | (Self::Integer(_), Self::Continuous(_) | Self::Category(_)) => Ordering::Greater,
        }
    }
}

/// A mixed-space point paired with its objective score (maximized).
#[derive(Clone, Debug, PartialEq)]
pub struct MixedCrossEntropySample {
    /// Point across all configured dimensions.
    pub point: Vec<MixedValue>,
    /// Objective score to maximize; NaN scores are ignored.
    pub score: ObjectiveScore,
}

impl MixedCrossEntropySample {
    /// Creates a scored mixed-space sample.
    #[must_use]
    pub const fn new(point: Vec<MixedValue>, score: ObjectiveScore) -> Self {
        Self { point, score }
    }
}

/// Configuration for [`MixedCrossEntropyOptimizer`].
#[derive(Clone, Debug, PartialEq)]
pub struct MixedCrossEntropyConfig {
    /// Dimensions defining the mixed search space.
    dimensions: Vec<MixedDimension>,
    /// Fraction of valid samples treated as elites (default `0.15`).
    elite_fraction: f64,
    /// Smoothing learning rate in `[0, 1]` (default `0.7`).
    learning_rate: f64,
}

impl MixedCrossEntropyConfig {
    /// Creates a mixed-space configuration with default elite fraction `0.15`
    /// and learning rate `0.7`.
    #[must_use]
    pub const fn new(dimensions: Vec<MixedDimension>) -> Self {
        Self {
            dimensions,
            elite_fraction: 0.15,
            learning_rate: 0.7,
        }
    }

    /// Sets the elite fraction in `(0, 1]`.
    #[must_use]
    pub const fn with_elite_fraction(mut self, elite_fraction: f64) -> Self {
        self.elite_fraction = elite_fraction;
        self
    }

    /// Sets the smoothing learning rate in `[0, 1]`.
    #[must_use]
    pub const fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }
}

/// Invalid configuration or sample supplied to [`MixedCrossEntropyOptimizer`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedCrossEntropyError {
    /// No dimensions were configured.
    EmptySearchSpace,
    /// Elite fraction was not finite and in `(0, 1]`.
    InvalidEliteFraction,
    /// Learning rate was not finite and in `[0, 1]`.
    InvalidLearningRate,
    /// One dimension's bounds, mean, standard deviation, category count, or
    /// probability floor was invalid.
    InvalidDimension {
        /// Index of the invalid dimension.
        dimension: usize,
    },
    /// A non-NaN sample had the wrong number of coordinates.
    DimensionMismatch {
        /// Index of the invalid sample.
        sample: usize,
        /// Expected number of coordinates.
        expected: usize,
        /// Actual number of coordinates.
        actual: usize,
    },
    /// A non-NaN sample contained a mismatched variant or out-of-domain value.
    InvalidSampleCoordinate {
        /// Index of the invalid sample.
        sample: usize,
        /// Index of the invalid dimension.
        dimension: usize,
    },
}

impl fmt::Display for MixedCrossEntropyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySearchSpace => formatter.write_str("mixed search space has no dimensions"),
            Self::InvalidEliteFraction => formatter.write_str("elite fraction is invalid"),
            Self::InvalidLearningRate => formatter.write_str("learning rate is invalid"),
            Self::InvalidDimension { dimension } => {
                write!(formatter, "dimension {dimension} specification is invalid")
            }
            Self::DimensionMismatch {
                sample,
                expected,
                actual,
            } => write!(
                formatter,
                "sample {sample} has {actual} coordinates, expected {expected}"
            ),
            Self::InvalidSampleCoordinate { sample, dimension } => write!(
                formatter,
                "sample {sample} has an invalid value in dimension {dimension}"
            ),
        }
    }
}

impl std::error::Error for MixedCrossEntropyError {}

/// Internal per-dimension learned distribution state.
#[derive(Clone, Debug, PartialEq)]
enum DimState {
    /// Bounded continuous Gaussian normalized onto `[0, 1]`.
    Continuous {
        /// Inclusive lower bound in user coordinates.
        lower: f64,
        /// Inclusive upper bound in user coordinates.
        upper: f64,
        /// Normalized mean in `[0, 1]`.
        mean: f64,
        /// Normalized standard deviation.
        std: f64,
        /// Normalized standard deviation floor.
        min_std: f64,
    },
    /// Periodic circular Gaussian normalized onto `[0, 1)`.
    Circular {
        /// Inclusive lower bound in user coordinates.
        lower: f64,
        /// Exclusive upper bound in user coordinates.
        upper: f64,
        /// Normalized mean in `[0, 1)`.
        mean: f64,
        /// Normalized standard deviation.
        std: f64,
        /// Normalized standard deviation floor.
        min_std: f64,
    },
    /// Discrete categorical probability vector.
    Categorical {
        /// Probability of each category (non-empty, sums to 1).
        probabilities: Vec<f64>,
        /// Floor per category.
        min_probability: f64,
    },
    /// Bounded integer represented by a normalized reflected Gaussian on `[0, 1]`.
    Integer {
        /// Inclusive lower integer bound.
        min: i64,
        /// Inclusive upper integer bound.
        max: i64,
        /// Normalized mean in `[0, 1]`.
        mean: f64,
        /// Normalized standard deviation.
        std: f64,
        /// Normalized standard deviation floor.
        min_std: f64,
    },
}

/// Ask/tell cross-entropy optimizer over mixed continuous, circular, integer,
/// and categorical dimensions.
///
/// ```
/// use simul::experiment::{
///     MasterSeed, MixedCrossEntropyConfig, MixedCrossEntropyOptimizer, MixedCrossEntropySample,
///     MixedDimension, SeedDeriver,
/// };
///
/// let config = MixedCrossEntropyConfig::new(vec![
///     MixedDimension::categorical(3),
///     MixedDimension::integer(1, 10),
///     MixedDimension::continuous(0.0, 100.0),
/// ])
/// .with_elite_fraction(0.2);
/// let mut opt = MixedCrossEntropyOptimizer::new(&config)?;
/// let mut rng = SeedDeriver::new(MasterSeed::from_u64(7)).proposal_seed(0).rng();
/// for _ in 0..12 {
///     let mut samples: Vec<_> = opt
///         .ask_batch(&mut rng, 32)
///         .into_iter()
///         .map(|pt| {
///             let cat_bonus = if pt[0].as_category() == Some(2) { 10.0 } else { 0.0 };
///             let int_err = (pt[1].as_i64().unwrap_or(0) - 7) as f64;
///             let cont_err = pt[2].as_f64().unwrap_or(0.0) - 42.0;
///             MixedCrossEntropySample::new(pt, cat_bonus - int_err.powi(2) - 0.01 * cont_err.powi(2))
///         })
///         .collect();
///     opt.tell(&mut samples)?;
/// }
/// let best = opt.best().expect("scored samples");
/// assert_eq!(Some(2), best.point[0].as_category());
/// assert_eq!(Some(7), best.point[1].as_i64());
/// assert!((best.point[2].as_f64().unwrap_or(0.0) - 42.0).abs() < 3.0);
/// # Ok::<(), simul::experiment::MixedCrossEntropyError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct MixedCrossEntropyOptimizer {
    /// Per-dimension distribution parameters.
    dims: Vec<DimState>,
    /// Elite fraction in `(0, 1]`.
    elite_fraction: f64,
    /// Smoothing learning rate in `[0, 1]`.
    learning_rate: f64,
    /// Completed update generations.
    generation: u64,
    /// Best sample observed across all generations.
    best: Option<MixedCrossEntropySample>,
}

impl MixedCrossEntropyOptimizer {
    /// Creates a validated mixed-space cross-entropy optimizer.
    ///
    /// # Errors
    ///
    /// Returns [`MixedCrossEntropyError`] when the search space is empty or any
    /// hyperparameter or dimension specification is invalid.
    pub fn new(config: &MixedCrossEntropyConfig) -> Result<Self, MixedCrossEntropyError> {
        if config.dimensions.is_empty() {
            return Err(MixedCrossEntropyError::EmptySearchSpace);
        }
        if !config.elite_fraction.is_finite()
            || config.elite_fraction <= 0.0
            || config.elite_fraction > 1.0
        {
            return Err(MixedCrossEntropyError::InvalidEliteFraction);
        }
        if !config.learning_rate.is_finite()
            || config.learning_rate < 0.0
            || config.learning_rate > 1.0
        {
            return Err(MixedCrossEntropyError::InvalidLearningRate);
        }
        let mut dims = Vec::with_capacity(config.dimensions.len());
        for (index, spec) in config.dimensions.iter().enumerate() {
            dims.push(build_dim_state(*spec, index)?);
        }
        Ok(Self {
            dims,
            elite_fraction: config.elite_fraction,
            learning_rate: config.learning_rate,
            generation: 0,
            best: None,
        })
    }

    /// Samples one mixed-space candidate point from `rng`.
    #[must_use]
    pub fn ask<Random: Rng + ?Sized>(&self, rng: &mut Random) -> Vec<MixedValue> {
        self.dims.iter().map(|dim| sample_dim(dim, rng)).collect()
    }

    /// Samples `count` mixed-space candidate points in order from `rng`.
    #[must_use]
    pub fn ask_batch<Random: Rng + ?Sized>(
        &self,
        rng: &mut Random,
        count: usize,
    ) -> Vec<Vec<MixedValue>> {
        (0..count).map(|_| self.ask(rng)).collect()
    }

    /// Updates all continuous, circular, integer, and categorical distributions
    /// from the elite subset of `samples`.
    ///
    /// Samples with `NaN` scores are ignored; if none are valid, returns
    /// `Ok(None)` without mutating the optimizer. All non-NaN samples are
    /// validated atomically before any mutation.
    ///
    /// # Errors
    ///
    /// Returns [`MixedCrossEntropyError::DimensionMismatch`] or
    /// [`MixedCrossEntropyError::InvalidSampleCoordinate`] before mutation when
    /// any non-NaN sample does not match the configured search space.
    pub fn tell(
        &mut self,
        samples: &mut [MixedCrossEntropySample],
    ) -> Result<Option<CrossEntropyUpdate>, MixedCrossEntropyError> {
        let valid_samples = validate_mixed_samples(samples, &self.dims)?;
        if valid_samples == 0 {
            return Ok(None);
        }
        let elite_samples = compute_elite_count(valid_samples, self.elite_fraction);
        select_and_order_mixed_elites(samples, valid_samples, elite_samples);
        let elites = &samples[..elite_samples];
        let generation_best = &elites[0];
        let replace_best = self
            .best
            .as_ref()
            .is_none_or(|best| generation_best.score >= best.score);
        if replace_best {
            self.best = Some(generation_best.clone());
        }
        let best_score = self
            .best
            .as_ref()
            .map_or(generation_best.score, |best| best.score);

        if self.learning_rate > 0.0 {
            self.fit(elites);
        }
        self.generation += 1;
        Ok(Some(CrossEntropyUpdate {
            generation: self.generation,
            valid_samples,
            elite_samples,
            generation_best_score: generation_best.score,
            best_score,
        }))
    }

    /// Returns the number of completed updates.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the highest-scoring sample seen so far, if any.
    #[must_use]
    pub const fn best(&self) -> Option<&MixedCrossEntropySample> {
        self.best.as_ref()
    }

    /// Returns the learned probability vector for categorical `dimension`, or
    /// `None` if `dimension` is out of range or not categorical.
    #[must_use]
    pub fn categorical_probabilities(&self, dimension: usize) -> Option<&[f64]> {
        match self.dims.get(dimension)? {
            DimState::Categorical { probabilities, .. } => Some(probabilities),
            DimState::Continuous { .. } | DimState::Circular { .. } | DimState::Integer { .. } => {
                None
            }
        }
    }

    /// Returns the current mean in user coordinates for a continuous, circular,
    /// or integer `dimension`, or `None` if out of range or categorical.
    #[must_use]
    pub fn dimension_mean(&self, dimension: usize) -> Option<f64> {
        match self.dims.get(dimension)? {
            DimState::Continuous {
                lower, upper, mean, ..
            }
            | DimState::Circular {
                lower, upper, mean, ..
            } => Some(denormalize(*mean, *lower, *upper)),
            DimState::Integer { min, max, mean, .. } => {
                Some(denormalize(*mean, *min as f64, *max as f64))
            }
            DimState::Categorical { .. } => None,
        }
    }

    /// Fits all dimensions to `elites` with smoothing weight `learning_rate`.
    fn fit(&mut self, elites: &[MixedCrossEntropySample]) {
        let lr = self.learning_rate;
        let keep = 1.0 - lr;
        let inv_n = 1.0 / elites.len() as f64;
        for (dim_index, state) in self.dims.iter_mut().enumerate() {
            fit_dimension(state, elites, dim_index, lr, keep, inv_n);
        }
    }
}

/// Places the `elite_samples` best entries of `samples` at the front in
/// canonical descending-score order.
fn select_and_order_mixed_elites(
    samples: &mut [MixedCrossEntropySample],
    valid_samples: usize,
    elite_samples: usize,
) {
    let all_valid = valid_samples == samples.len();
    if elite_samples == 1 {
        let mut best_index = 0;
        if all_valid {
            for index in 1..samples.len() {
                if compare_valid_mixed_samples(&samples[index], &samples[best_index]).is_lt() {
                    best_index = index;
                }
            }
        } else {
            for index in 1..samples.len() {
                if compare_mixed_samples(&samples[index], &samples[best_index]).is_lt() {
                    best_index = index;
                }
            }
        }
        samples.swap(0, best_index);
    } else if samples.len() <= FULL_SORT_POPULATION_THRESHOLD || elite_samples == samples.len() {
        if all_valid {
            samples.sort_unstable_by(compare_valid_mixed_samples);
        } else {
            samples.sort_unstable_by(compare_mixed_samples);
        }
    } else {
        if all_valid {
            let _ = samples.select_nth_unstable_by(elite_samples - 1, compare_valid_mixed_samples);
        } else {
            let _ = samples.select_nth_unstable_by(elite_samples - 1, compare_mixed_samples);
        }
        if elite_samples > 2 {
            samples[..elite_samples - 1].sort_unstable_by(compare_valid_mixed_samples);
        }
    }
}

/// Fits a single dimension's distribution parameters to `elites`.
#[allow(clippy::suboptimal_flops)]
fn fit_dimension(
    state: &mut DimState,
    elites: &[MixedCrossEntropySample],
    dim_index: usize,
    lr: f64,
    keep: f64,
    inv_n: f64,
) {
    match state {
        DimState::Continuous {
            lower,
            upper,
            mean,
            std,
            min_std,
        } => {
            if *upper <= *lower {
                *mean = 0.0;
                *std = *min_std;
                return;
            }
            fit_linear_normalized(
                elites
                    .iter()
                    .filter_map(|s| s.point[dim_index].as_f64())
                    .map(|v| normalize_to_unit(v, *lower, *upper)),
                elites.len(),
                mean,
                std,
                *min_std,
                lr,
                keep,
                inv_n,
            );
        }
        DimState::Circular {
            lower,
            upper,
            mean,
            std,
            min_std,
        } => {
            fit_circular_dimension(
                elites, dim_index, *lower, *upper, mean, std, *min_std, lr, keep, inv_n,
            );
        }
        DimState::Integer {
            min,
            max,
            mean,
            std,
            min_std,
        } => {
            if *min == *max {
                *mean = 0.0;
                *std = *min_std;
                return;
            }
            let (lo, hi) = (*min as f64, *max as f64);
            fit_linear_normalized(
                elites
                    .iter()
                    .filter_map(|s| s.point[dim_index].as_i64())
                    .map(|v| normalize_to_unit(v as f64, lo, hi)),
                elites.len(),
                mean,
                std,
                *min_std,
                lr,
                keep,
                inv_n,
            );
        }
        DimState::Categorical {
            probabilities,
            min_probability,
        } => fit_categorical(
            probabilities,
            *min_probability,
            elites,
            dim_index,
            lr,
            keep,
            inv_n,
        ),
    }
}

/// Smoothly fits a linear unit-interval Gaussian from normalized elite values,
/// caching them once so each elite is extracted and normalized only once.
#[allow(clippy::too_many_arguments, clippy::suboptimal_flops)]
fn fit_linear_normalized(
    values: impl Iterator<Item = f64>,
    capacity_hint: usize,
    mean: &mut f64,
    std: &mut f64,
    min_std: f64,
    lr: f64,
    keep: f64,
    inv_n: f64,
) {
    let mut stack_buf = [0.0_f64; STACK_ELITE_BUFFER];
    let mut heap_buf = Vec::new();
    let mut sum = -0.0_f64;
    let slice: &[f64] = if capacity_hint <= STACK_ELITE_BUFFER {
        let mut count = 0_usize;
        for (slot, u) in stack_buf.iter_mut().zip(values) {
            *slot = u;
            sum += u;
            count += 1;
        }
        &stack_buf[..count]
    } else {
        heap_buf.reserve(capacity_hint);
        for u in values {
            heap_buf.push(u);
            sum += u;
        }
        &heap_buf
    };
    let elite_mean = sum * inv_n;
    let mut sum_sq = -0.0_f64;
    for &u in slice {
        let d = u - elite_mean;
        sum_sq += d * d;
    }
    let elite_var = sum_sq * inv_n;
    *mean = mean.mul_add(keep, elite_mean * lr).clamp(0.0, 1.0);
    *std = (*std * *std)
        .mul_add(keep, elite_var * lr)
        .max(0.0)
        .sqrt()
        .max(min_std);
}

/// Smoothly fits a circular dimension, normalizing each elite coordinate once.
#[allow(clippy::too_many_arguments, clippy::suboptimal_flops)]
fn fit_circular_dimension(
    elites: &[MixedCrossEntropySample],
    dim_index: usize,
    lower: f64,
    upper: f64,
    mean: &mut f64,
    std: &mut f64,
    min_std: f64,
    lr: f64,
    keep: f64,
    inv_n: f64,
) {
    let mut stack_buf = [0.0_f64; STACK_ELITE_BUFFER];
    let mut heap_buf = Vec::new();
    let mut sin_sum = 0.0_f64;
    let mut cos_sum = 0.0_f64;
    let slice: &[f64] = if elites.len() <= STACK_ELITE_BUFFER {
        let mut count = 0_usize;
        for (slot, v) in stack_buf
            .iter_mut()
            .zip(elites.iter().filter_map(|s| s.point[dim_index].as_f64()))
        {
            let u = normalize_to_unit(v, lower, upper);
            *slot = u;
            count += 1;
            let (s, c) = (u * TAU).sin_cos();
            sin_sum += s;
            cos_sum += c;
        }
        &stack_buf[..count]
    } else {
        heap_buf.reserve(elites.len());
        for sample in elites {
            if let Some(v) = sample.point[dim_index].as_f64() {
                let u = normalize_to_unit(v, lower, upper);
                heap_buf.push(u);
                let (s, c) = (u * TAU).sin_cos();
                sin_sum += s;
                cos_sum += c;
            }
        }
        &heap_buf
    };
    let elite_mean = if sin_sum.hypot(cos_sum) <= 1.0e-12 {
        *mean
    } else {
        wrap_unit(sin_sum.atan2(cos_sum) / TAU)
    };
    let next_mean = if keep <= 0.0 {
        elite_mean
    } else {
        wrap_unit(lr.mul_add(circular_delta_unit(*mean, elite_mean), *mean))
    };
    let mut sum_sq = -0.0_f64;
    for &u in slice {
        let d = circular_delta_unit(elite_mean, u);
        sum_sq += d * d;
    }
    let elite_var = sum_sq * inv_n;
    *mean = next_mean;
    *std = (*std * *std)
        .mul_add(keep, elite_var * lr)
        .max(0.0)
        .sqrt()
        .max(min_std);
}

/// Smoothly fits a categorical probability vector with a per-category floor.
fn fit_categorical(
    probabilities: &mut [f64],
    floor: f64,
    elites: &[MixedCrossEntropySample],
    dim_index: usize,
    lr: f64,
    keep: f64,
    inv_n: f64,
) {
    let k = probabilities.len();
    let mut stack_counts = [0_usize; STACK_CATEGORY_COUNTS];
    let mut heap_counts = Vec::new();
    let counts: &mut [usize] = if k <= STACK_CATEGORY_COUNTS {
        &mut stack_counts[..k]
    } else {
        heap_counts.resize(k, 0);
        &mut heap_counts
    };
    for sample in elites {
        if let Some(cat) = sample.point[dim_index].as_category() {
            counts[cat] += 1;
        }
    }
    let mut total = 0.0_f64;
    for (prob, &count) in probabilities.iter_mut().zip(counts.iter()) {
        let emp = count as f64 * inv_n;
        *prob = prob.mul_add(keep, emp * lr).max(floor);
        total += *prob;
    }
    if total > 0.0 {
        let floor_mass = floor * k as f64;
        let excess_total = total - floor_mass;
        let target_excess = (1.0 - floor_mass).max(0.0);
        if excess_total > 0.0 {
            let scale = target_excess / excess_total;
            for prob in probabilities.iter_mut() {
                *prob = (*prob - floor).mul_add(scale, floor);
            }
        } else {
            let uniform = 1.0 / k as f64;
            for prob in probabilities.iter_mut() {
                *prob = uniform;
            }
        }
    }
}

/// Validates and normalizes mean/std hyperparameters for a continuous or
/// circular span.
fn normalize_span_gaussian(
    lower: f64,
    upper: f64,
    strict_upper: bool,
    initial_mean: Option<f64>,
    initial_std: Option<f64>,
    min_std: Option<f64>,
    dimension: usize,
) -> Result<(f64, f64, f64), MixedCrossEntropyError> {
    let invalid = MixedCrossEntropyError::InvalidDimension { dimension };
    let bad_order = if strict_upper {
        lower >= upper
    } else {
        lower > upper
    };
    if !lower.is_finite() || !upper.is_finite() || bad_order {
        return Err(invalid);
    }
    let span = upper - lower;
    if !span.is_finite() {
        return Err(invalid);
    }
    if span == 0.0 {
        if let Some(m) = initial_mean {
            if !m.is_finite() || m < lower || m > upper {
                return Err(invalid);
            }
        }
        return Ok((0.0, 0.0, 0.0));
    }
    let raw_mean = initial_mean.unwrap_or_else(|| 0.5_f64.mul_add(span, lower));
    let mean_out = if strict_upper {
        raw_mean < lower || raw_mean >= upper
    } else {
        raw_mean < lower || raw_mean > upper
    };
    if !raw_mean.is_finite() || mean_out {
        return Err(invalid);
    }
    let raw_std = initial_std.unwrap_or(0.25 * span);
    let raw_min = min_std.unwrap_or(1.0e-6 * span);
    if !raw_std.is_finite()
        || raw_std <= 0.0
        || !raw_min.is_finite()
        || raw_min <= 0.0
        || raw_min > raw_std
    {
        return Err(invalid);
    }
    let norm_mean = normalize_to_unit(raw_mean, lower, upper);
    Ok((
        if strict_upper {
            wrap_unit(norm_mean)
        } else {
            norm_mean
        },
        raw_std / span,
        raw_min / span,
    ))
}

/// Validates and converts one [`MixedDimension`] into internal distribution state.
fn build_dim_state(
    spec: MixedDimension,
    dimension: usize,
) -> Result<DimState, MixedCrossEntropyError> {
    let invalid = MixedCrossEntropyError::InvalidDimension { dimension };
    match spec {
        MixedDimension::Continuous {
            lower,
            upper,
            initial_mean,
            initial_std,
            min_std,
        } => {
            let (mean, std, min_std) = normalize_span_gaussian(
                lower,
                upper,
                false,
                initial_mean,
                initial_std,
                min_std,
                dimension,
            )?;
            Ok(DimState::Continuous {
                lower,
                upper,
                mean,
                std,
                min_std,
            })
        }
        MixedDimension::Circular {
            lower,
            upper,
            initial_mean,
            initial_std,
            min_std,
        } => {
            let (mean, std, min_std) = normalize_span_gaussian(
                lower,
                upper,
                true,
                initial_mean,
                initial_std,
                min_std,
                dimension,
            )?;
            Ok(DimState::Circular {
                lower,
                upper,
                mean,
                std,
                min_std,
            })
        }
        MixedDimension::Categorical {
            categories,
            min_probability,
        } => {
            if categories == 0
                || !min_probability.is_finite()
                || min_probability < 0.0
                || min_probability * categories as f64 > 1.0 + 1.0e-12
            {
                return Err(invalid);
            }
            let capped_floor = min_probability.min(1.0 / categories as f64);
            Ok(DimState::Categorical {
                probabilities: vec![1.0 / categories as f64; categories],
                min_probability: capped_floor,
            })
        }
        MixedDimension::Integer { min, max } => {
            if min > max {
                return Err(invalid);
            }
            let span = (max as f64) - (min as f64);
            let min_std = if span > 0.0 {
                (0.25 / span).min(0.25)
            } else {
                0.0
            };
            Ok(DimState::Integer {
                min,
                max,
                mean: 0.5,
                std: if span > 0.0 { 0.3 } else { 0.0 },
                min_std,
            })
        }
    }
}

/// Samples one coordinate from `state` using `rng`.
fn sample_dim<Random: Rng + ?Sized>(state: &DimState, rng: &mut Random) -> MixedValue {
    match state {
        DimState::Continuous {
            lower,
            upper,
            mean,
            std,
            ..
        } => {
            if *lower >= *upper {
                return MixedValue::Continuous(*lower);
            }
            let z: f64 = rng.sample(StandardNormal);
            let u = reflect_unit(std.mul_add(z, *mean));
            MixedValue::Continuous(denormalize(u, *lower, *upper).clamp(*lower, *upper))
        }
        DimState::Circular {
            lower,
            upper,
            mean,
            std,
            ..
        } => {
            let z: f64 = rng.sample(StandardNormal);
            let u = wrap_unit(std.mul_add(z, *mean));
            let val = denormalize(u, *lower, *upper);
            let clamped = if val >= *upper {
                *lower
            } else {
                val.max(*lower)
            };
            MixedValue::Continuous(clamped)
        }
        DimState::Categorical { probabilities, .. } => {
            let u: f64 = rng.random();
            let mut cumulative = 0.0;
            for (index, &prob) in probabilities.iter().enumerate() {
                cumulative += prob;
                if u < cumulative {
                    return MixedValue::Category(index);
                }
            }
            MixedValue::Category(probabilities.len() - 1)
        }
        DimState::Integer {
            min,
            max,
            mean,
            std,
            ..
        } => {
            if *min == *max {
                return MixedValue::Integer(*min);
            }
            let z: f64 = rng.sample(StandardNormal);
            let u = reflect_unit(std.mul_add(z, *mean));
            let raw = denormalize(u, *min as f64, *max as f64).round();
            #[allow(clippy::cast_possible_truncation)]
            let value = (raw as i64).clamp(*min, *max);
            MixedValue::Integer(value)
        }
    }
}

/// Validates every non-NaN sample against `dims` and returns the valid count.
fn validate_mixed_samples(
    samples: &[MixedCrossEntropySample],
    dims: &[DimState],
) -> Result<usize, MixedCrossEntropyError> {
    let mut valid = 0_usize;
    for (sample_idx, sample) in samples.iter().enumerate() {
        if sample.score.is_nan() {
            continue;
        }
        if sample.point.len() != dims.len() {
            return Err(MixedCrossEntropyError::DimensionMismatch {
                sample: sample_idx,
                expected: dims.len(),
                actual: sample.point.len(),
            });
        }
        for (dim_idx, (value, state)) in sample.point.iter().zip(dims).enumerate() {
            let ok = match (value, state) {
                (MixedValue::Continuous(v), DimState::Continuous { lower, upper, .. }) => {
                    *v >= *lower && *v <= *upper
                }
                (MixedValue::Continuous(v), DimState::Circular { lower, upper, .. }) => {
                    *v >= *lower && *v < *upper
                }
                (MixedValue::Category(c), DimState::Categorical { probabilities, .. }) => {
                    *c < probabilities.len()
                }
                (MixedValue::Integer(v), DimState::Integer { min, max, .. }) => {
                    *v >= *min && *v <= *max
                }
                (
                    MixedValue::Continuous(_),
                    DimState::Categorical { .. } | DimState::Integer { .. },
                )
                | (
                    MixedValue::Category(_),
                    DimState::Continuous { .. }
                    | DimState::Circular { .. }
                    | DimState::Integer { .. },
                )
                | (
                    MixedValue::Integer(_),
                    DimState::Continuous { .. }
                    | DimState::Circular { .. }
                    | DimState::Categorical { .. },
                ) => false,
            };
            if !ok {
                return Err(MixedCrossEntropyError::InvalidSampleCoordinate {
                    sample: sample_idx,
                    dimension: dim_idx,
                });
            }
        }
        valid += 1;
    }
    Ok(valid)
}

/// Breaks ties between equal-scored valid mixed samples via total score order
/// and then lexicographical coordinate order.
#[cold]
fn compare_valid_mixed_tie(
    left: &MixedCrossEntropySample,
    right: &MixedCrossEntropySample,
) -> Ordering {
    right.score.total_cmp(&left.score).then_with(|| {
        left.point
            .iter()
            .zip(&right.point)
            .map(|(&a, &b)| a.total_cmp(b))
            .find(|ord| ord.is_ne())
            .unwrap_or(Ordering::Equal)
    })
}

/// Orders two known-valid mixed samples highest score first, breaking ties
/// deterministically by coordinate values.
#[inline]
fn compare_valid_mixed_samples(
    left: &MixedCrossEntropySample,
    right: &MixedCrossEntropySample,
) -> Ordering {
    if left.score > right.score {
        Ordering::Less
    } else if left.score < right.score {
        Ordering::Greater
    } else {
        compare_valid_mixed_tie(left, right)
    }
}

/// Orders valid mixed samples highest score first, breaking ties
/// deterministically by coordinate values, and places NaNs last.
fn compare_mixed_samples(
    left: &MixedCrossEntropySample,
    right: &MixedCrossEntropySample,
) -> Ordering {
    match (left.score.is_nan(), right.score.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => compare_valid_mixed_samples(left, right),
    }
}

/// Computes the elite count for `valid_samples` and `elite_fraction`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn compute_elite_count(valid_samples: usize, elite_fraction: f64) -> usize {
    let product = valid_samples as f64 * elite_fraction;
    let nearest = product.round();
    let count = if (product - nearest).abs() <= nearest * 4.0 * f64::EPSILON {
        nearest
    } else {
        product.ceil()
    };
    (count as usize).clamp(1, valid_samples)
}

/// Maps `value` from `[lower, upper]` into `[0, 1]`.
fn normalize_to_unit(value: f64, lower: f64, upper: f64) -> f64 {
    ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
}

/// Maps `unit` in `[0, 1]` onto `[lower, upper]`.
#[inline]
fn denormalize(unit: f64, lower: f64, upper: f64) -> f64 {
    unit.mul_add(upper - lower, lower)
}

/// Reflects an arbitrary finite real into `[0, 1]`.
#[allow(clippy::manual_range_contains)]
#[inline]
fn reflect_unit(value: f64) -> f64 {
    if value >= 0.0 && value <= 1.0 {
        return value;
    }
    reflect_unit_out_of_domain(value)
}

/// Reflects an out-of-domain value into `[0, 1]` without calling `libm::fmod`.
#[cold]
fn reflect_unit_out_of_domain(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.5;
    }
    // Bit-identical to `value.rem_euclid(2.0)` for all finite `f64`.
    let remainder = (-2.0_f64)
        .mul_add((value * 0.5).trunc(), value)
        .copysign(value);
    let period = if remainder < 0.0 {
        remainder + 2.0
    } else {
        remainder
    };
    if period <= 1.0 { period } else { 2.0 - period }
}

/// Wraps an arbitrary finite real into `[0, 1)`.
#[allow(clippy::manual_range_contains)]
#[inline]
fn wrap_unit(value: f64) -> f64 {
    if value >= 0.0 && value < 1.0 {
        return value;
    }
    wrap_unit_out_of_domain(value)
}

/// Wraps an out-of-domain value into `[0, 1)` without calling `libm::fmod`.
#[cold]
fn wrap_unit_out_of_domain(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    // Bit-identical to `value.rem_euclid(1.0)` for all finite `f64`.
    let remainder = (value - value.trunc()).copysign(value);
    let wrapped = if remainder < 0.0 {
        remainder + 1.0
    } else {
        remainder
    };
    if wrapped >= 1.0 { 0.0 } else { wrapped }
}

/// Signed shortest circular displacement from `from` to `to` on `[0, 1)`.
#[allow(clippy::manual_range_contains)]
#[inline]
fn circular_delta_unit(from: f64, to: f64) -> f64 {
    let diff = to - from;
    let mut delta = if diff >= 0.0 && diff < 1.0 {
        diff
    } else if diff > -1.0 && diff < 0.0 {
        diff + 1.0
    } else {
        let remainder = (diff - diff.trunc()).copysign(diff);
        if remainder < 0.0 {
            remainder + 1.0
        } else {
            remainder
        }
    };
    if delta >= 0.5 {
        delta -= 1.0;
    }
    delta
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::missing_docs_in_private_items)]
mod tests {
    use super::*;
    use crate::experiment::{MasterSeed, SeedDeriver};

    #[test]
    fn mixed_optimizer_converges_across_all_four_dimension_types() {
        let config = MixedCrossEntropyConfig::new(vec![
            MixedDimension::categorical(4),
            MixedDimension::integer(-5, 15),
            MixedDimension::continuous(-10.0, 10.0),
            MixedDimension::circular(0.0, 360.0),
        ])
        .with_elite_fraction(0.2)
        .with_learning_rate(0.6);
        let mut opt = MixedCrossEntropyOptimizer::new(&config).expect("valid config");
        let mut rng = SeedDeriver::new(MasterSeed::from_u64(314))
            .proposal_seed(0)
            .rng();

        for _ in 0..20 {
            let mut samples: Vec<_> = opt
                .ask_batch(&mut rng, 48)
                .into_iter()
                .map(|pt| {
                    let cat = pt[0].as_category().expect("cat");
                    let int_val = pt[1].as_i64().expect("int");
                    let cont = pt[2].as_f64().expect("cont");
                    let angle = pt[3].as_f64().expect("angle");
                    // Target: cat == 3, int == 11, cont == -4.5, angle == 355 deg (near wrap!)
                    let cat_score = if cat == 3 { 20.0 } else { 0.0 };
                    let int_pen = ((int_val - 11) as f64).powi(2);
                    let cont_pen = (cont + 4.5).powi(2);
                    let mut ang_diff = (angle - 355.0).rem_euclid(360.0);
                    if ang_diff > 180.0 {
                        ang_diff -= 360.0;
                    }
                    let ang_pen = (ang_diff / 30.0).powi(2);
                    MixedCrossEntropySample::new(pt, cat_score - int_pen - cont_pen - ang_pen)
                })
                .collect();
            opt.tell(&mut samples).expect("tell");
        }

        let probs = opt.categorical_probabilities(0).expect("cat probs");
        assert!(
            probs[3] > 0.85,
            "expected category 3 to dominate: {probs:?}"
        );
        let best = opt.best().expect("best");
        assert_eq!(Some(3), best.point[0].as_category());
        assert_eq!(Some(11), best.point[1].as_i64());
        assert!((best.point[2].as_f64().expect("cont") + 4.5).abs() < 0.5);
    }

    #[test]
    fn mixed_optimizer_rejects_invalid_configs_and_samples_atomically() {
        assert_eq!(
            Err(MixedCrossEntropyError::EmptySearchSpace),
            MixedCrossEntropyOptimizer::new(&MixedCrossEntropyConfig::new(vec![]))
        );
        assert_eq!(
            Err(MixedCrossEntropyError::InvalidDimension { dimension: 0 }),
            MixedCrossEntropyOptimizer::new(&MixedCrossEntropyConfig::new(vec![
                MixedDimension::categorical(0)
            ]))
        );
        assert_eq!(
            Err(MixedCrossEntropyError::InvalidDimension { dimension: 0 }),
            MixedCrossEntropyOptimizer::new(&MixedCrossEntropyConfig::new(vec![
                MixedDimension::integer(10, 5)
            ]))
        );

        let mut opt = MixedCrossEntropyOptimizer::new(&MixedCrossEntropyConfig::new(vec![
            MixedDimension::categorical(2),
            MixedDimension::integer(0, 5),
        ]))
        .expect("valid");
        let mut bad = vec![MixedCrossEntropySample::new(
            vec![MixedValue::Category(2), MixedValue::Integer(1)],
            1.0,
        )];
        let err = opt
            .tell(&mut bad)
            .expect_err("out-of-range category must fail");
        assert_eq!(
            MixedCrossEntropyError::InvalidSampleCoordinate {
                sample: 0,
                dimension: 0,
            },
            err
        );
        assert_eq!(0, opt.generation());
        assert_eq!(None, opt.best());
        assert_ne!("", err.to_string());
    }

    #[test]
    fn mixed_unit_projections_match_rem_euclid_bit_for_bit() {
        let cases = [
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            2.0,
            -2.0,
            -1.0e-20,
            1.0e-20,
            f64::from_bits(1.0_f64.to_bits() - 1),
            -f64::from_bits(1.0_f64.to_bits() - 1),
            ((1_u64 << 52) as f64) - 0.5,
            -((1_u64 << 52) as f64),
            (1_u64 << 53) as f64,
            -((1_u64 << 53) as f64),
            f64::MAX,
            -f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for v in cases {
            let expected_reflect = if v.is_finite() {
                let p = v.rem_euclid(2.0);
                if p <= 1.0 { p } else { 2.0 - p }
            } else {
                0.5
            };
            assert_eq!(expected_reflect.to_bits(), reflect_unit(v).to_bits());

            let expected_wrap = if v.is_finite() {
                let w = v.rem_euclid(1.0);
                if w >= 1.0 { 0.0 } else { w }
            } else {
                0.0
            };
            assert_eq!(expected_wrap.to_bits(), wrap_unit(v).to_bits());
        }

        let unit = [
            0.0,
            -0.0,
            1.0e-20,
            0.25,
            0.5,
            0.75,
            f64::from_bits(1.0_f64.to_bits() - 1),
            1.0,
        ];
        for a in unit {
            for b in unit {
                let mut expected = (b - a).rem_euclid(1.0);
                if expected >= 0.5 {
                    expected -= 1.0;
                }
                assert_eq!(expected.to_bits(), circular_delta_unit(a, b).to_bits());
            }
        }
    }

    #[test]
    fn mixed_tell_selection_matches_full_sort_with_ties_and_nans() {
        let config = MixedCrossEntropyConfig::new(vec![
            MixedDimension::categorical(3),
            MixedDimension::integer(0, 10),
            MixedDimension::continuous(0.0, 1.0),
            MixedDimension::circular(0.0, 1.0),
        ])
        .with_learning_rate(0.7);
        for (size, elite_frac) in [(10, 0.2), (32, 0.01), (32, 0.06), (32, 0.25), (32, 1.0)] {
            let mut rng = SeedDeriver::new(MasterSeed::from_u64(size as u64))
                .proposal_seed(0)
                .rng();
            let mut opt_a =
                MixedCrossEntropyOptimizer::new(&config.clone().with_elite_fraction(elite_frac))
                    .expect("valid");
            let mut opt_b = opt_a.clone();
            let mut samples_a: Vec<_> = opt_a
                .ask_batch(&mut rng, size)
                .into_iter()
                .enumerate()
                .map(|(i, pt)| {
                    let score = if i % 7 == 0 { f64::NAN } else { (i % 4) as f64 };
                    MixedCrossEntropySample::new(pt, score)
                })
                .collect();
            let mut samples_b = samples_a.clone();
            samples_b.reverse();
            let ua = opt_a.tell(&mut samples_a).expect("tell a");
            let ub = opt_b.tell(&mut samples_b).expect("tell b");
            assert_eq!(ua, ub);
            assert_eq!(opt_a, opt_b);
        }
    }
}
