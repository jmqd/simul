//! Multi-objective Pareto frontier tracking, crowding distances, and 2D
//! hypervolume calculation.

use super::CandidateId;
use std::cmp::Ordering;
use std::fmt;

/// Whether higher or lower values are preferred for one objective dimension.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ObjectiveDirection {
    /// Higher finite values are better.
    Maximize,
    /// Lower finite values are better.
    Minimize,
}

/// Invalid input supplied to [`ParetoArchive`] or [`select_pareto_frontier`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParetoError {
    /// At least one objective direction is required.
    EmptyObjectives,
    /// Supplied objective vector did not match the archive's dimension count.
    DimensionMismatch {
        /// Configured objective count.
        expected: usize,
        /// Supplied objective count.
        actual: usize,
    },
    /// An objective or reference coordinate was NaN or infinite.
    NonFiniteObjective {
        /// Zero-based objective index.
        index: usize,
        /// Rejected value.
        value: f64,
    },
    /// A candidate identity was already present on the frontier.
    DuplicateCandidate(CandidateId),
}

impl fmt::Display for ParetoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyObjectives => formatter.write_str("at least one objective is required"),
            Self::DimensionMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} objective value(s), got {actual}"
            ),
            Self::NonFiniteObjective { index, value } => {
                write!(formatter, "objective {index} must be finite, got {value}")
            }
            Self::DuplicateCandidate(id) => {
                write!(formatter, "duplicate candidate identity {}", id.value())
            }
        }
    }
}

impl std::error::Error for ParetoError {}

/// A non-dominated candidate retained on a [`ParetoArchive`] frontier.
#[derive(Clone, Debug, PartialEq)]
pub struct ParetoEntry<C> {
    /// Stable candidate identity.
    pub candidate: CandidateId,
    /// Caller-owned candidate payload or summary.
    pub payload: C,
    /// Validated finite objective values matching the archive's directions.
    pub objectives: Vec<f64>,
}

/// Exact, insertion-order-independent multi-objective Pareto frontier archive.
///
/// Entries are maintained in ascending [`CandidateId`] order. Inserting a point
/// rejects it if any existing point strictly dominates it; otherwise it evicts
/// all existing points strictly dominated by the new point and inserts it.
///
/// ```
/// use simul::experiment::{CandidateId, ObjectiveDirection, ParetoArchive};
///
/// // Minimize latency, maximize throughput.
/// let mut archive = ParetoArchive::new(vec![
///     ObjectiveDirection::Minimize,
///     ObjectiveDirection::Maximize,
/// ])?;
/// assert!(archive.insert(CandidateId::new(0), "fast-small", vec![10.0, 100.0])?);
/// assert!(archive.insert(CandidateId::new(1), "slow-big", vec![25.0, 300.0])?);
/// // Dominated by candidate 0 (higher latency, lower throughput).
/// assert!(!archive.insert(CandidateId::new(2), "worse", vec![15.0, 90.0])?);
/// assert_eq!(2, archive.len());
/// # Ok::<(), simul::experiment::ParetoError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ParetoArchive<C> {
    /// Optimization direction of each objective coordinate.
    directions: Vec<ObjectiveDirection>,
    /// Current non-dominated frontier ordered by `CandidateId`.
    entries: Vec<ParetoEntry<C>>,
}

impl<C> ParetoArchive<C> {
    /// Creates an empty Pareto frontier archive for `directions`.
    ///
    /// # Errors
    ///
    /// Returns [`ParetoError::EmptyObjectives`] when `directions` is empty.
    pub fn new(directions: Vec<ObjectiveDirection>) -> Result<Self, ParetoError> {
        if directions.is_empty() {
            return Err(ParetoError::EmptyObjectives);
        }
        Ok(Self {
            directions,
            entries: Vec::new(),
        })
    }

    /// Returns the configured objective directions.
    #[must_use]
    pub fn directions(&self) -> &[ObjectiveDirection] {
        &self.directions
    }

    /// Returns the current non-dominated frontier ordered by [`CandidateId`].
    #[must_use]
    pub fn frontier(&self) -> &[ParetoEntry<C>] {
        &self.entries
    }

    /// Consumes the archive and returns the frontier entries in [`CandidateId`]
    /// order.
    #[must_use]
    pub fn into_frontier(self) -> Vec<ParetoEntry<C>> {
        self.entries
    }

    /// Returns the number of candidates currently on the frontier.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether no candidates are currently on the frontier.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Attempts to insert a candidate and its objective vector into the
    /// frontier.
    ///
    /// Returns `Ok(true)` if the candidate is non-dominated and was added
    /// (evicting any entries it strictly dominates), or `Ok(false)` if an
    /// existing frontier entry strictly dominates it.
    ///
    /// # Errors
    ///
    /// Returns [`ParetoError`] before mutating the archive if `objectives` has
    /// the wrong length, contains a non-finite value, or `candidate` is
    /// already present on the frontier.
    pub fn insert(
        &mut self,
        candidate: CandidateId,
        payload: C,
        objectives: Vec<f64>,
    ) -> Result<bool, ParetoError> {
        if objectives.len() != self.directions.len() {
            return Err(ParetoError::DimensionMismatch {
                expected: self.directions.len(),
                actual: objectives.len(),
            });
        }
        for (index, &value) in objectives.iter().enumerate() {
            if !value.is_finite() {
                return Err(ParetoError::NonFiniteObjective { index, value });
            }
        }
        if self
            .entries
            .binary_search_by_key(&candidate, |entry| entry.candidate)
            .is_ok()
        {
            return Err(ParetoError::DuplicateCandidate(candidate));
        }

        if self
            .entries
            .iter()
            .any(|existing| dominates(&existing.objectives, &objectives, &self.directions))
        {
            return Ok(false);
        }

        self.entries
            .retain(|existing| !dominates(&objectives, &existing.objectives, &self.directions));
        let position = self
            .entries
            .binary_search_by_key(&candidate, |entry| entry.candidate)
            .unwrap_or_else(|pos| pos);
        self.entries.insert(
            position,
            ParetoEntry {
                candidate,
                payload,
                objectives,
            },
        );
        Ok(true)
    }

    /// Computes NSGA-II crowding distances aligned with [`Self::frontier`].
    ///
    /// Boundary entries (extreme on any objective) receive `f64::INFINITY`;
    /// interior entries receive the sum of normalized neighbor gaps across all
    /// objectives with positive span. Larger values indicate sparser regions
    /// of the trade-off curve.
    #[must_use]
    pub fn crowding_distances(&self) -> Vec<f64> {
        let n = self.entries.len();
        if n == 0 {
            return Vec::new();
        }
        if n <= 2 {
            return vec![f64::INFINITY; n];
        }
        let mut distances = vec![0.0_f64; n];
        let mut order: Vec<usize> = (0..n).collect();
        for dim in 0..self.directions.len() {
            order.sort_by(|&a, &b| {
                self.entries[a].objectives[dim]
                    .total_cmp(&self.entries[b].objectives[dim])
                    .then_with(|| self.entries[a].candidate.cmp(&self.entries[b].candidate))
            });
            distances[order[0]] = f64::INFINITY;
            distances[order[n - 1]] = f64::INFINITY;
            let span =
                self.entries[order[n - 1]].objectives[dim] - self.entries[order[0]].objectives[dim];
            if !span.is_finite() || span <= 0.0 {
                continue;
            }
            for window in order.windows(3) {
                let prev = self.entries[window[0]].objectives[dim];
                let curr = window[1];
                let next = self.entries[window[2]].objectives[dim];
                if distances[curr].is_finite() {
                    distances[curr] += (next - prev) / span;
                }
            }
        }
        distances
    }

    /// Computes the exact 2D dominated hypervolume relative to `reference`.
    ///
    /// Only frontier points that are strictly better than `reference` in both
    /// objectives contribute to the area. Returns `0.0` when no frontier point
    /// dominates `reference`.
    ///
    /// # Errors
    ///
    /// Returns [`ParetoError::DimensionMismatch`] when the archive does not
    /// have exactly two objectives, or [`ParetoError::NonFiniteObjective`]
    /// when a reference coordinate is non-finite.
    pub fn hypervolume_2d(&self, reference: [f64; 2]) -> Result<f64, ParetoError> {
        if self.directions.len() != 2 {
            return Err(ParetoError::DimensionMismatch {
                expected: 2,
                actual: self.directions.len(),
            });
        }
        for (index, value) in reference.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(ParetoError::NonFiniteObjective { index, value });
            }
        }
        // Transform both objectives into positive improvement margins above `reference`
        // (maximization frame relative to origin [0, 0]).
        let mut points: Vec<(f64, f64)> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let dx = improvement_margin(entry.objectives[0], reference[0], self.directions[0]);
                let dy = improvement_margin(entry.objectives[1], reference[1], self.directions[1]);
                (dx > 0.0 && dy > 0.0).then_some((dx, dy))
            })
            .collect();
        if points.is_empty() {
            return Ok(0.0);
        }
        // Sort by descending x, then descending y; sweep strips with running max y.
        points.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| b.1.total_cmp(&a.1)));
        let mut area = 0.0_f64;
        let mut best_y = 0.0_f64;
        for (x, y) in points {
            if y > best_y {
                area = x.mul_add(y - best_y, area);
                best_y = y;
            }
        }
        Ok(area)
    }
}

/// Computes positive margin by which `value` improves on `reference` under
/// `direction`.
fn improvement_margin(value: f64, reference: f64, direction: ObjectiveDirection) -> f64 {
    match direction {
        ObjectiveDirection::Maximize => value - reference,
        ObjectiveDirection::Minimize => reference - value,
    }
}

/// Returns `true` iff `left` strictly Pareto-dominates `right` under
/// `directions`: at least as good on every objective and strictly better on at
/// least one.
#[must_use]
pub fn dominates(left: &[f64], right: &[f64], directions: &[ObjectiveDirection]) -> bool {
    let mut strictly_better = false;
    for ((&a, &b), &dir) in left.iter().zip(right).zip(directions) {
        let ord = match dir {
            ObjectiveDirection::Maximize => a.total_cmp(&b),
            ObjectiveDirection::Minimize => b.total_cmp(&a),
        };
        match ord {
            Ordering::Less => return false,
            Ordering::Greater => strictly_better = true,
            Ordering::Equal => {}
        }
    }
    strictly_better
}

/// Filters an iterator of `(CandidateId, payload, objectives)` down to its
/// exact Pareto frontier in ascending [`CandidateId`] order.
///
/// # Errors
///
/// Returns [`ParetoError`] when `directions` is empty or any candidate has an
/// invalid objective vector or duplicate identity.
pub fn select_pareto_frontier<C>(
    directions: &[ObjectiveDirection],
    candidates: impl IntoIterator<Item = (CandidateId, C, Vec<f64>)>,
) -> Result<Vec<ParetoEntry<C>>, ParetoError> {
    let mut archive = ParetoArchive::new(directions.to_vec())?;
    for (id, payload, objectives) in candidates {
        archive.insert(id, payload, objectives)?;
    }
    Ok(archive.into_frontier())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::missing_docs_in_private_items)]
mod tests {
    use super::*;

    #[test]
    fn pareto_archive_tracks_frontier_crowding_and_2d_hypervolume() {
        let mut archive = ParetoArchive::new(vec![
            ObjectiveDirection::Minimize,
            ObjectiveDirection::Maximize,
        ])
        .expect("valid");
        assert!(
            archive
                .insert(CandidateId::new(1), "a", vec![1.0, 2.0])
                .expect("a")
        );
        assert!(
            archive
                .insert(CandidateId::new(2), "b", vec![2.0, 4.0])
                .expect("b")
        );
        assert!(
            archive
                .insert(CandidateId::new(3), "c", vec![4.0, 5.0])
                .expect("c")
        );
        // Dominated by b (x=3 > 2, y=3 < 4).
        assert!(
            !archive
                .insert(CandidateId::new(4), "d", vec![3.0, 3.0])
                .expect("d")
        );
        assert_eq!(3, archive.len());

        let crowding = archive.crowding_distances();
        assert!(crowding[0].is_infinite());
        assert!(crowding[1].is_finite() && crowding[1] > 0.0);
        assert!(crowding[2].is_infinite());

        // Reference (5.0, 0.0): margins are (4, 2), (3, 4), (1, 5).
        // Union of boxes [0,4]x[0,2], [0,3]x[0,4], [0,1]x[0,5]:
        // 4*2 + 3*(4-2) + 1*(5-4) = 8 + 6 + 1 = 15.0.
        let hv = archive.hypervolume_2d([5.0, 0.0]).expect("hv");
        assert!((hv - 15.0).abs() < 1e-12, "expected 15.0, got {hv}");

        // Inserting a strictly better point evicts dominated ones and increases hypervolume.
        assert!(
            archive
                .insert(CandidateId::new(0), "super", vec![1.0, 4.5])
                .expect("super")
        );
        let ids: Vec<u64> = archive
            .frontier()
            .iter()
            .map(|e| e.candidate.value())
            .collect();
        assert_eq!(vec![0, 3], ids);
        let hv2 = archive.hypervolume_2d([5.0, 0.0]).expect("hv2");
        assert!(hv2 > hv);
    }

    #[test]
    fn pareto_errors_validate_inputs_and_format_messages() {
        assert_eq!(
            Err(ParetoError::EmptyObjectives),
            ParetoArchive::<()>::new(vec![])
        );
        let mut archive =
            ParetoArchive::new(vec![ObjectiveDirection::Maximize]).expect("1d archive");
        assert_eq!(
            Err(ParetoError::DimensionMismatch {
                expected: 1,
                actual: 2
            }),
            archive.insert(CandidateId::new(0), (), vec![1.0, 2.0])
        );
        assert!(matches!(
            archive.insert(CandidateId::new(0), (), vec![f64::NAN]),
            Err(ParetoError::NonFiniteObjective { index: 0, .. })
        ));
        assert!(
            archive
                .insert(CandidateId::new(0), (), vec![1.0])
                .expect("ok")
        );
        assert_eq!(
            Err(ParetoError::DuplicateCandidate(CandidateId::new(0))),
            archive.insert(CandidateId::new(0), (), vec![2.0])
        );
        assert_ne!("", ParetoError::EmptyObjectives.to_string());
    }
}
