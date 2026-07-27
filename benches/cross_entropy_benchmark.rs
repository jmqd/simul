use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{RngExt as _, SeedableRng as _};
use simul::experiment::{
    simulated_annealing_search_with_rng, CrossEntropyConfig, CrossEntropyDimension,
    CrossEntropyOptimizer, CrossEntropySample,
};

/// Number of normalized coordinates in the toy objective.
const DIMENSIONS: usize = 5;
/// Candidates in each CEM generation.
const POPULATION_SIZE: usize = 24;
/// Generations in each complete CEM search.
const GENERATIONS: usize = 20;
/// Objective evaluations allotted to every complete optimizer search.
const SEARCH_EVALUATIONS: usize = POPULATION_SIZE * GENERATIONS;
/// Global optimum around which the bounded toy objective is shaped.
const TARGET: [f64; DIMENSIONS] = [0.08, 0.72, 0.24, 0.81, 0.35];
/// Reusable finite standard-normal quantiles for the caller-supplied sampling path.
const STANDARD_NORMAL_VARIATES: [f64; 16] = [
    -1.534, -1.151, -0.887, -0.674, -0.488, -0.319, -0.157, -0.052, 0.052, 0.157, 0.319, 0.488,
    0.674, 0.887, 1.151, 1.534,
];

/// Creates the deterministic optimizer used by overhead and search benchmarks.
fn optimizer(learning_rate: f64) -> CrossEntropyOptimizer<DIMENSIONS> {
    let result = CrossEntropyOptimizer::new(
        CrossEntropyConfig::new([0.5; DIMENSIONS], [0.3; DIMENSIONS])
            .with_dimensions([
                CrossEntropyDimension::Circular,
                CrossEntropyDimension::Linear,
                CrossEntropyDimension::Linear,
                CrossEntropyDimension::Linear,
                CrossEntropyDimension::Linear,
            ])
            .with_minimum_standard_deviation([0.005; DIMENSIONS])
            .with_elite_fraction(0.25)
            .with_learning_rate(learning_rate),
    );
    match result {
        Ok(optimizer) => optimizer,
        Err(error) => panic!("benchmark configuration is invalid: {error}"),
    }
}

/// Multimodal bounded objective with one circular coordinate.
#[inline]
fn objective(point: &[f64; DIMENSIONS]) -> f64 {
    let circular_error = (point[0] - TARGET[0] + 0.5).rem_euclid(1.0) - 0.5;
    let squared_error = point[1..].iter().zip(&TARGET[1..]).fold(
        circular_error * circular_error,
        |sum, (actual, target)| {
            let difference = actual - target;
            difference.mul_add(difference, sum)
        },
    );
    let ripple = (point[1] * 24.0).cos() * (point[3] * 18.0).cos();
    0.03_f64.mul_add(ripple, -squared_error)
}

/// Cheap objective used to expose optimizer overhead in one-generation benchmarks.
#[inline]
fn quadratic_objective(point: &[f64; DIMENSIONS]) -> f64 {
    -point.iter().zip(TARGET).fold(0.0, |sum, (actual, target)| {
        let difference = actual - target;
        difference.mul_add(difference, sum)
    })
}

/// Executes one ask/evaluate/tell generation of a compile-time population size.
fn run_generation<const POPULATION: usize>() -> f64 {
    let mut optimizer = optimizer(0.6);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut samples = [CrossEntropySample::new([0.0; DIMENSIONS], f64::NAN); POPULATION];
    for sample in &mut samples {
        sample.point = optimizer.ask(&mut rng);
        sample.score = quadratic_objective(&sample.point);
    }
    if let Err(error) = optimizer.tell(&mut samples) {
        panic!("benchmark population is invalid: {error}");
    }
    optimizer.best().map_or(f64::NAN, |sample| sample.score)
}

/// Executes one generation through the deterministic caller-supplied sampling path.
fn run_callback_generation<const POPULATION: usize>() -> f64 {
    let mut optimizer = optimizer(0.6);
    let mut samples = [CrossEntropySample::new([0.0; DIMENSIONS], f64::NAN); POPULATION];
    let standard_normal_variates = black_box(&STANDARD_NORMAL_VARIATES);
    for (sample_index, sample) in samples.iter_mut().enumerate() {
        let offset = sample_index * DIMENSIONS;
        let point = optimizer.ask_with_standard_normal(|dimension| {
            standard_normal_variates[(offset + dimension) % standard_normal_variates.len()]
        });
        sample.point = match point {
            Ok(point) => point,
            Err(error) => panic!("benchmark standard-normal variate is invalid: {error}"),
        };
        sample.score = quadratic_objective(&sample.point);
    }
    if let Err(error) = optimizer.tell(&mut samples) {
        panic!("benchmark population is invalid: {error}");
    }
    optimizer.best().map_or(f64::NAN, |sample| sample.score)
}

/// Executes a complete fixed-budget CEM search.
fn run_cross_entropy_search(learning_rate: f64) -> f64 {
    let mut optimizer = optimizer(learning_rate);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut samples = [CrossEntropySample::new([0.0; DIMENSIONS], f64::NAN); POPULATION_SIZE];
    for _ in 0..GENERATIONS {
        for sample in &mut samples {
            sample.point = optimizer.ask(&mut rng);
            sample.score = objective(&sample.point);
        }
        if let Err(error) = optimizer.tell(&mut samples) {
            panic!("benchmark population is invalid: {error}");
        }
    }
    optimizer.best().map_or(f64::NAN, |sample| sample.score)
}

/// Reflects a normalized linear coordinate at its bounds.
#[inline]
fn reflect(value: f64) -> f64 {
    let reflected = value.rem_euclid(2.0);
    if reflected <= 1.0 {
        reflected
    } else {
        2.0 - reflected
    }
}

/// Executes simulated annealing over the same objective and evaluation budget.
fn run_annealing_search() -> f64 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = simulated_annealing_search_with_rng(
        &mut rng,
        |_| [0.5_f64; DIMENSIONS],
        |current, rng| {
            let mut proposal = *current;
            for (dimension, value) in proposal.iter_mut().enumerate() {
                let perturbed = *value + rng.random_range(-0.15..0.15);
                *value = if dimension == 0 {
                    perturbed.rem_euclid(1.0)
                } else {
                    reflect(perturbed)
                };
            }
            proposal
        },
        objective,
        |proposal| 0.2 * 0.01_f64.powf(f64::from(proposal) / SEARCH_EVALUATIONS as f64),
        u32::try_from(SEARCH_EVALUATIONS - 1)
            .unwrap_or_else(|_| panic!("benchmark evaluation budget exceeds u32")),
    );
    result.as_ref().map_or(f64::NAN, objective)
}

/// Measures optimizer overhead independent of simulation cost.
fn cross_entropy_generation_benchmarks(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("cross_entropy_generation");
    for population in [12_u64, 24, 96, 512] {
        group.throughput(Throughput::Elements(population));
        match population {
            12 => {
                group.bench_function(
                    BenchmarkId::new("ask_quadratic_tell", population),
                    |bench| bench.iter(|| black_box(run_generation::<12>())),
                );
                group.bench_function(
                    BenchmarkId::new("ask_callback_quadratic_tell", population),
                    |bench| bench.iter(|| black_box(run_callback_generation::<12>())),
                )
            }
            24 => {
                group.bench_function(
                    BenchmarkId::new("ask_quadratic_tell", population),
                    |bench| bench.iter(|| black_box(run_generation::<24>())),
                );
                group.bench_function(
                    BenchmarkId::new("ask_callback_quadratic_tell", population),
                    |bench| bench.iter(|| black_box(run_callback_generation::<24>())),
                )
            }
            96 => {
                group.bench_function(
                    BenchmarkId::new("ask_quadratic_tell", population),
                    |bench| bench.iter(|| black_box(run_generation::<96>())),
                );
                group.bench_function(
                    BenchmarkId::new("ask_callback_quadratic_tell", population),
                    |bench| bench.iter(|| black_box(run_callback_generation::<96>())),
                )
            }
            512 => {
                group.bench_function(
                    BenchmarkId::new("ask_quadratic_tell", population),
                    |bench| bench.iter(|| black_box(run_generation::<512>())),
                );
                group.bench_function(
                    BenchmarkId::new("ask_callback_quadratic_tell", population),
                    |bench| bench.iter(|| black_box(run_callback_generation::<512>())),
                )
            }
            _ => panic!("unregistered benchmark population"),
        };
    }
    group.finish();
}

/// Measures complete deterministic optimizer searches at an equal evaluation budget.
fn optimizer_search_benchmarks(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("optimizer_search_480_evaluations");
    group.throughput(Throughput::Elements(SEARCH_EVALUATIONS as u64));
    group.bench_function("cross_entropy_adaptive", |bench| {
        bench.iter(|| black_box(run_cross_entropy_search(0.6)));
    });
    group.bench_function("cross_entropy_fixed_prior", |bench| {
        bench.iter(|| black_box(run_cross_entropy_search(0.0)));
    });
    group.bench_function("simulated_annealing", |bench| {
        bench.iter(|| black_box(run_annealing_search()));
    });
    group.finish();
}

criterion_group!(
    benches,
    cross_entropy_generation_benchmarks,
    optimizer_search_benchmarks
);
criterion_main!(benches);
