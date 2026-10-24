//! Simulated time. One tick is 100 µs: fine enough for a 0.5 ms intra-region
//! round trip, coarse enough to simulate minutes of replication quickly.

use simul::DiscreteTime;

pub(crate) type Tick = DiscreteTime;

pub(crate) const TICKS_PER_SECOND: Tick = 10_000;

/// Converts a non-negative, finite float to `u64`, saturating at `u64::MAX`.
///
/// # Panics
///
/// Panics on NaN or negative input, which would be a modeling bug.
pub(crate) fn to_u64(value: f64) -> u64 {
    assert!(
        value >= 0.0,
        "expected a non-negative quantity, got {value}"
    );
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is checked non-negative; `as` saturates above u64::MAX"
    )]
    let converted = value as u64;
    converted
}

/// Rounds a duration in seconds to the nearest tick.
pub(crate) fn ticks(seconds: f64) -> Tick {
    to_u64((seconds * TICKS_PER_SECOND as f64).round())
}

/// Rounds a duration in seconds up to whole ticks, at least one.
pub(crate) fn ticks_at_least_one(seconds: f64) -> Tick {
    to_u64((seconds * TICKS_PER_SECOND as f64).ceil()).max(1)
}

pub(crate) fn secs(time: Tick) -> f64 {
    time as f64 / TICKS_PER_SECOND as f64
}

pub(crate) fn per_tick(per_second: f64) -> f64 {
    per_second / TICKS_PER_SECOND as f64
}
