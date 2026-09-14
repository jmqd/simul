//! Typed deterministic trials, reduction, and engine-independent optimizers.
//!
//! Use [`run_serial_bounded`](crate::experiment::run_serial_bounded) or
//! `run_parallel` (with the `parallel` feature) for replicated evaluations,
//! and [`search`](crate::experiment::search) for tokenized candidate search.
//! The `Simulation` agent engine is one possible evaluator, not a requirement.
//!
//! Trial identity and replay use one V1 keyed BLAKE3/ChaCha12 protocol.

pub mod cross_entropy;

pub use cross_entropy::*;
mod core;
pub use core::*;

/// Tokenized deterministic search strategies.
pub mod search;

use rand::{Rng, RngExt};

/// A maximized objective score for the generic annealing and cross-entropy optimizers.
pub type ObjectiveScore = f64;

/// Searches for a state that maximizes an objective using simulated annealing.
///
/// The initial state is evaluated once, followed by `proposal_limit` candidate
/// evaluations. `temperature_schedule` receives each zero-based proposal index.
/// Better and equal candidates are always accepted. A worse candidate is
/// accepted with the Metropolis probability
/// `exp((candidate_score - current_score) / temperature)`.
///
/// NaN candidate scores are ignored. If every score is NaN, this returns
/// `None`. A non-finite or non-positive temperature rejects worse candidates,
/// making that step greedy. The state does not need to implement [`Clone`];
/// ownership of the best state is retained without cloning.
///
/// All randomness used by initialization, perturbation, and acceptance comes
/// from `rng`. Reproducible results additionally require a deterministic
/// objective function. For noisy objectives, aggregate deterministic
/// replications inside `objective_function` (for example with
/// [`run_serial_bounded`]) rather than optimizing one noisy sample.
pub fn simulated_annealing_search_with_rng<State, Random>(
    rng: &mut Random,
    initial_state_generator: impl FnOnce(&mut Random) -> State,
    mut perturb_function: impl FnMut(&State, &mut Random) -> State,
    mut objective_function: impl FnMut(&State) -> ObjectiveScore,
    mut temperature_schedule: impl FnMut(u32) -> f64,
    proposal_limit: u32,
) -> Option<State>
where
    Random: Rng + ?Sized,
{
    let mut current_state = initial_state_generator(rng);
    let mut current_score = objective_function(&current_state);
    let mut best_score = (!current_score.is_nan()).then_some(current_score);
    let mut current_is_best = best_score.is_some();
    let mut detached_best = None;

    for proposal_index in 0..proposal_limit {
        let temperature = temperature_schedule(proposal_index);
        let candidate = perturb_function(&current_state, rng);
        let candidate_score = objective_function(&candidate);

        if !accept_candidate(current_score, candidate_score, temperature, rng) {
            continue;
        }

        let candidate_is_best = best_score.is_none_or(|score| candidate_score >= score);
        if candidate_is_best {
            current_state = candidate;
            current_score = candidate_score;
            best_score = Some(candidate_score);
            current_is_best = true;
            detached_best = None;
        } else {
            if current_is_best {
                detached_best = Some(current_state);
            }
            current_state = candidate;
            current_score = candidate_score;
            current_is_best = false;
        }
    }

    if current_is_best {
        Some(current_state)
    } else {
        detached_best
    }
}

/// Applies the Metropolis acceptance criterion for a maximization search.
#[inline]
fn accept_candidate<Random>(
    current_score: ObjectiveScore,
    candidate_score: ObjectiveScore,
    temperature: f64,
    rng: &mut Random,
) -> bool
where
    Random: Rng + ?Sized,
{
    if candidate_score.is_nan() {
        return false;
    }
    if current_score.is_nan() || candidate_score >= current_score {
        return true;
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return false;
    }

    let difference = candidate_score - current_score;
    // Subtract first to retain nearby-score precision. Only finite scores can
    // recover from an overflowing difference by dividing before subtraction.
    let scaled_difference =
        if difference.is_infinite() && candidate_score.is_finite() && current_score.is_finite() {
            candidate_score / temperature - current_score / temperature
        } else {
            difference / temperature
        };
    let acceptance_probability = scaled_difference.exp();
    acceptance_probability > 0.0 && rng.random::<f64>() < acceptance_probability
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng as _;

    /// State that deliberately cannot be cloned.
    #[derive(Debug, PartialEq)]
    struct NonClone(i32);

    /// RNG returning one fixed word and counting draws.
    struct FixedRng {
        word: u64,
        draws: usize,
    }

    impl FixedRng {
        /// Creates a fixed RNG.
        const fn new(word: u64) -> Self {
            Self { word, draws: 0 }
        }
    }

    impl rand::TryRng for FixedRng {
        type Error = std::convert::Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            self.draws += 1;
            let bytes = self.word.to_le_bytes();
            Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }

        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            self.draws += 1;
            Ok(self.word)
        }

        fn try_fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), Self::Error> {
            for chunk in destination.chunks_mut(size_of::<u64>()) {
                self.draws += 1;
                let bytes = self.word.to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
            Ok(())
        }
    }

    /// RNG that fails the test if sampled.
    struct PanicRng;

    impl rand::TryRng for PanicRng {
        type Error = std::convert::Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            panic!("unexpected random draw")
        }

        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            panic!("unexpected random draw")
        }

        fn try_fill_bytes(&mut self, _destination: &mut [u8]) -> Result<(), Self::Error> {
            panic!("unexpected random draw")
        }
    }

    #[test]
    fn zero_proposals_evaluates_and_returns_the_initial_state() {
        let owned = String::from("consumed by FnOnce");
        let mut evaluations = 0;
        let result = simulated_annealing_search_with_rng(
            &mut PanicRng,
            move |_| {
                drop(owned);
                NonClone(7)
            },
            |_, _| panic!("perturbation must not run"),
            |state| {
                evaluations += 1;
                f64::from(state.0)
            },
            |_| panic!("temperature schedule must not run"),
            0,
        );

        assert_eq!(Some(NonClone(7)), result);
        assert_eq!(1, evaluations);
    }

    #[test]
    fn better_and_equal_candidates_are_accepted_without_randomness() {
        let mut proposal = 0;
        let mut schedule_indices = Vec::new();
        let result = simulated_annealing_search_with_rng(
            &mut PanicRng,
            |_| NonClone(1),
            |_, _| {
                proposal += 1;
                NonClone(2)
            },
            |state| f64::from(state.0),
            |index| {
                schedule_indices.push(index);
                1.0
            },
            2,
        );

        assert_eq!(Some(NonClone(2)), result);
        assert_eq!(2, proposal);
        assert_eq!(&[0, 1], schedule_indices.as_slice());
    }

    #[test]
    fn metropolis_can_move_downhill_but_returns_the_best_state() {
        let mut rng = FixedRng::new(0);
        let mut visited = Vec::new();
        let result = simulated_annealing_search_with_rng(
            &mut rng,
            |_| NonClone(10),
            |current, _| {
                visited.push(current.0);
                NonClone(current.0 - 1)
            },
            |state| f64::from(state.0),
            |_| 1.0,
            2,
        );

        assert_eq!(&[10, 9], visited.as_slice());
        assert_eq!(Some(NonClone(10)), result);
        assert!(rng.draws > 0);
    }

    #[test]
    fn metropolis_rejects_a_downhill_move_above_its_probability() {
        let mut rng = FixedRng::new(u64::MAX);
        let mut visited = Vec::new();
        let result = simulated_annealing_search_with_rng(
            &mut rng,
            |_| NonClone(10),
            |current, _| {
                visited.push(current.0);
                NonClone(current.0 - 1)
            },
            |state| f64::from(state.0),
            |_| 1.0,
            2,
        );

        assert_eq!(&[10, 10], visited.as_slice());
        assert_eq!(Some(NonClone(10)), result);
        assert!(rng.draws > 0);
    }

    #[test]
    fn metropolis_scales_overflowing_finite_score_differences() {
        // The true probability is exp(-2), between these draws: 0.125 and 0.25.
        for (word, expected_visited) in [(1_u64 << 61, [1, 0]), (1_u64 << 62, [1, 1])] {
            let mut rng = FixedRng::new(word);
            let mut visited = Vec::new();
            let result = simulated_annealing_search_with_rng(
                &mut rng,
                |_| NonClone(1),
                |current, _| {
                    visited.push(current.0);
                    NonClone(0)
                },
                |state| if state.0 == 1 { 1.0e308 } else { -1.0e308 },
                |_| 1.0e308,
                2,
            );

            assert_eq!(visited, expected_visited);
            assert_eq!(Some(NonClone(1)), result);
        }
    }

    #[test]
    fn metropolis_rejects_infinite_downhill_gaps_without_randomness() {
        for (current_score, candidate_score) in
            [(f64::INFINITY, 1.0e308), (1.0e308, f64::NEG_INFINITY)]
        {
            let mut visited = Vec::new();
            let result = simulated_annealing_search_with_rng(
                &mut PanicRng,
                |_| NonClone(1),
                |current, _| {
                    visited.push(current.0);
                    NonClone(0)
                },
                |state| {
                    if state.0 == 1 {
                        current_score
                    } else {
                        candidate_score
                    }
                },
                |_| 1.0e308,
                2,
            );

            assert_eq!(&[1, 1], visited.as_slice());
            assert_eq!(result, Some(NonClone(1)));
        }
    }

    #[test]
    fn invalid_temperatures_are_greedy_and_do_not_draw_randomness() {
        let mut temperatures = [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY].into_iter();
        let result = simulated_annealing_search_with_rng(
            &mut PanicRng,
            |_| NonClone(10),
            |current, _| NonClone(current.0 - 1),
            |state| f64::from(state.0),
            |_| {
                let Some(temperature) = temperatures.next() else {
                    panic!("temperature requested too many times");
                };
                temperature
            },
            5,
        );

        assert_eq!(result, Some(NonClone(10)));
    }

    #[test]
    fn nan_candidates_are_ignored_and_a_finite_candidate_recovers() {
        let rejected = simulated_annealing_search_with_rng(
            &mut PanicRng,
            |_| NonClone(0),
            |_, _| NonClone(1),
            |state| {
                if state.0 == 0 {
                    0.0
                } else {
                    f64::NAN
                }
            },
            |_| 1.0,
            1,
        );
        assert_eq!(rejected, Some(NonClone(0)));

        let recovered = simulated_annealing_search_with_rng(
            &mut PanicRng,
            |_| NonClone(0),
            |_, _| NonClone(1),
            |state| {
                if state.0 == 0 {
                    f64::NAN
                } else {
                    1.0
                }
            },
            |_| f64::NAN,
            1,
        );
        assert_eq!(recovered, Some(NonClone(1)));

        let no_valid_score = simulated_annealing_search_with_rng(
            &mut PanicRng,
            |_| NonClone(0),
            |_, _| panic!("perturbation must not run"),
            |_| f64::NAN,
            |_| panic!("temperature schedule must not run"),
            0,
        );
        assert_eq!(no_valid_score, None);
    }

    #[test]
    fn annealing_escapes_a_local_maximum_that_greedy_search_cannot() {
        let mut annealing_rng = FixedRng::new(0);
        let annealed = simulated_annealing_search_with_rng(
            &mut annealing_rng,
            |_| NonClone(0),
            |current, _| NonClone((current.0 + 1).min(3)),
            |state| match state.0 {
                0 => 10.0,
                1 => 0.0,
                2 => 5.0,
                3 => 20.0,
                _ => f64::NEG_INFINITY,
            },
            |_| 1.0,
            3,
        );

        let greedy = simulated_annealing_search_with_rng(
            &mut PanicRng,
            |_| NonClone(0),
            |current, _| NonClone((current.0 + 1).min(3)),
            |state| match state.0 {
                0 => 10.0,
                1 => 0.0,
                2 => 5.0,
                3 => 20.0,
                _ => f64::NEG_INFINITY,
            },
            |_| 0.0,
            3,
        );

        assert_eq!(annealed, Some(NonClone(3)));
        assert_eq!(greedy, Some(NonClone(0)));
    }

    #[test]
    fn seeded_runs_are_reproducible() {
        let run = |seed| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            simulated_annealing_search_with_rng(
                &mut rng,
                |rng| NonClone(rng.random_range(-10..=10)),
                |current, rng| NonClone(current.0 + rng.random_range(-2..=2)),
                |state| -f64::from(state.0).powi(2),
                |index| 10.0 * 0.95_f64.powf(f64::from(index)),
                100,
            )
        };

        assert_eq!(run(42), run(42));
    }
}
