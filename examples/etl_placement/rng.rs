//! Seeded samplers. Every agent owns named streams derived from the trial's
//! common seed, so both placements see the same workload and fault draws.

use chacha20::ChaCha12Rng;
use rand::SeedableRng;
use simul::experiment::{TrialContext, TrialStream};

#[derive(Debug)]
pub(crate) struct Rng(ChaCha12Rng);

// The generator is not `Clone`; its serialized state (seed, stream, and word
// position) restores the exact stream position.
impl Clone for Rng {
    fn clone(&self) -> Self {
        Self(ChaCha12Rng::deserialize_state(&self.0.serialize_state()))
    }
}

impl Rng {
    /// A stream shared by every candidate in the trial's common-random group.
    pub(crate) fn common(context: &TrialContext, tag: &str) -> Self {
        Self(ChaCha12Rng::from_seed(
            context.common_seed(TrialStream::TrialNoise, tag).bytes(),
        ))
    }

    /// Uniform on `[0, 1)`.
    pub(crate) fn unit(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1_u64 << 53) as f64;
        (rand::Rng::next_u64(&mut self.0) >> 11) as f64 * SCALE
    }

    pub(crate) fn uniform(&mut self, low: f64, high: f64) -> f64 {
        (high - low).mul_add(self.unit(), low)
    }

    pub(crate) fn bernoulli(&mut self, probability: f64) -> bool {
        self.unit() < probability
    }

    pub(crate) fn exponential(&mut self, mean: f64) -> f64 {
        -mean * (1.0 - self.unit()).ln()
    }

    pub(crate) fn standard_normal(&mut self) -> f64 {
        let radius = (-2.0 * (1.0 - self.unit()).ln()).sqrt();
        radius * (std::f64::consts::TAU * self.unit()).cos()
    }

    pub(crate) fn lognormal(&mut self, median: f64, sigma: f64) -> f64 {
        median * (sigma * self.standard_normal()).exp()
    }

    /// Poisson counts; normal approximation above a mean of 30.
    pub(crate) fn poisson(&mut self, mean: f64) -> u64 {
        /// Far beyond any count Knuth's method yields for a mean of 30.
        const MAX_COUNT: u64 = 1_000;
        if mean <= 0.0 {
            return 0;
        }
        if mean > 30.0 {
            let draw = mean.sqrt().mul_add(self.standard_normal(), mean).round();
            return crate::time::to_u64(draw.max(0.0));
        }
        // Knuth: count uniform draws until their product falls below e^-mean.
        let limit = (-mean).exp();
        let mut product = self.unit();
        (0..MAX_COUNT)
            .find(|_| {
                let done = product <= limit;
                if !done {
                    product *= self.unit();
                }
                done
            })
            .unwrap_or(MAX_COUNT)
    }

    /// Geometric on `{1, 2, …}` with the given mean.
    pub(crate) fn geometric(&mut self, mean: f64) -> u64 {
        if mean <= 1.0 {
            return 1;
        }
        let failure = 1.0 - 1.0 / mean;
        1 + crate::time::to_u64((1.0 - self.unit()).log(failure).floor())
    }

    pub(crate) fn below(&mut self, bound: u64) -> u64 {
        crate::time::to_u64((self.unit() * bound as f64).floor()).min(bound.saturating_sub(1))
    }
}
