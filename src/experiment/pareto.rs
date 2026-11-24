//! Multi-objective Pareto frontier tracking, crowding distances, and 2D
//! hypervolume calculation.

use super::CandidateId;
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

/// One distinct objective pair on a 2D Pareto staircase.
#[derive(Clone, Copy)]
struct StaircasePoint {
    /// Maximized sortable key for objective 0 (strictly decreasing along the staircase).
    s0: u64,
    /// Maximized sortable key for objective 1 (strictly increasing along the staircase).
    s1: u64,
    /// Raw objective 0 value for O(F) `hypervolume_2d` sweeps.
    x: f64,
    /// Raw objective 1 value for O(F) `hypervolume_2d` sweeps.
    y: f64,
}

/// Contiguous objective-key acceleration structure backing [`ParetoArchive`].
#[derive(Clone)]
enum FrontierIndex {
    /// Two-objective strictly monotone staircase plus flat entry keys aligned
    /// with `entries`.
    TwoD {
        /// Deduplicated monotone staircase ordered by strictly decreasing `s0`.
        staircase: Vec<StaircasePoint>,
        /// Flat maximized sortable pairs aligned with `entries`.
        entry_keys: Vec<[u64; 2]>,
    },
    /// Flat row-major maximized sortable keys (`entries.len() * dim`) aligned
    /// with `entries` for 1D and `>= 3D` archives.
    Flat {
        /// Contiguous row-major sortable keys (`entry[i]` at `i * d .. (i + 1) * d`).
        keys: Vec<u64>,
    },
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
#[derive(Clone)]
pub struct ParetoArchive<C> {
    /// Optimization direction of each objective coordinate.
    directions: Vec<ObjectiveDirection>,
    /// Current non-dominated frontier ordered by `CandidateId`.
    entries: Vec<ParetoEntry<C>>,
    /// Acceleration index maintained in lockstep with `entries`.
    index: FrontierIndex,
}

// `index` is an internal acceleration structure derived solely from
// `directions` and `entries`; omit it to keep Debug output unchanged.
#[allow(clippy::missing_fields_in_debug)]
impl<C: fmt::Debug> fmt::Debug for ParetoArchive<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ParetoArchive")
            .field("directions", &self.directions)
            .field("entries", &self.entries)
            .finish()
    }
}

impl<C: PartialEq> PartialEq for ParetoArchive<C> {
    fn eq(&self, other: &Self) -> bool {
        self.directions == other.directions && self.entries == other.entries
    }
}

/// Maps any `f64` to an unsigned integer preserving exact [`f64::total_cmp`]
/// ordering (including `-0.0 < +0.0` and NaN payloads).
#[inline]
const fn f64_to_sortable_u64(value: f64) -> u64 {
    let bits = value.to_bits();
    let sign_mask = if (bits >> 63) != 0 {
        u64::MAX
    } else {
        1_u64 << 63
    };
    bits ^ sign_mask
}

/// Maps `value` to a sortable key where larger keys are strictly better under
/// `direction`, matching [`f64::total_cmp`] bit-for-bit.
#[inline]
const fn directed_sortable_u64(value: f64, direction: ObjectiveDirection) -> u64 {
    let key = f64_to_sortable_u64(value);
    match direction {
        ObjectiveDirection::Maximize => key,
        ObjectiveDirection::Minimize => !key,
    }
}

/// Compares two flat sortable key slices in one pass: returns
/// `(existing_dominates_new, new_dominates_existing)`.
#[inline]
fn compare_key_slices(existing: &[u64], candidate: &[u64]) -> (bool, bool) {
    let mut existing_better = false;
    let mut candidate_better = false;
    for (&ek, &ck) in existing.iter().zip(candidate) {
        if ek > ck {
            existing_better = true;
            if candidate_better {
                return (false, false);
            }
        } else if ck > ek {
            candidate_better = true;
            if existing_better {
                return (false, false);
            }
        }
    }
    (
        existing_better && !candidate_better,
        candidate_better && !existing_better,
    )
}

/// Returns `true` iff `candidate` strictly dominates `existing` in maximized
/// sortable key space.
#[inline]
fn keys_dominate(candidate: &[u64], existing: &[u64]) -> bool {
    let mut strictly_better = false;
    for (&ck, &ek) in candidate.iter().zip(existing) {
        if ck < ek {
            return false;
        }
        strictly_better |= ck > ek;
    }
    strictly_better
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
        let index = if directions.len() == 2 {
            FrontierIndex::TwoD {
                staircase: Vec::new(),
                entry_keys: Vec::new(),
            }
        } else {
            FrontierIndex::Flat { keys: Vec::new() }
        };
        Ok(Self {
            directions,
            entries: Vec::new(),
            index,
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

    /// Validates `candidate` against existing IDs and returns `Ok(Some(pos))`
    /// for the pre-eviction insertion slot, or `Ok(None)` when `candidate` is
    /// strictly greater than every current frontier ID (append fast path).
    fn check_candidate_slot(&self, candidate: CandidateId) -> Result<Option<usize>, ParetoError> {
        if self
            .entries
            .last()
            .is_none_or(|last| candidate > last.candidate)
        {
            return Ok(None);
        }
        match self
            .entries
            .binary_search_by_key(&candidate, |entry| entry.candidate)
        {
            Ok(_) => Err(ParetoError::DuplicateCandidate(candidate)),
            Err(pos) => Ok(Some(pos)),
        }
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
        let slot = self.check_candidate_slot(candidate)?;
        if matches!(self.index, FrontierIndex::TwoD { .. }) {
            Ok(self.insert_2d(candidate, payload, objectives, slot))
        } else {
            Ok(self.insert_flat(candidate, payload, objectives, slot))
        }
    }

    /// O(log F) dominance check and amortized O(log F) staircase update for 2D
    /// archives.
    fn insert_2d(
        &mut self,
        candidate: CandidateId,
        payload: C,
        objectives: Vec<f64>,
        slot: Option<usize>,
    ) -> bool {
        let FrontierIndex::TwoD {
            staircase,
            entry_keys,
        } = &mut self.index
        else {
            return false;
        };
        let x = directed_sortable_u64(objectives[0], self.directions[0]);
        let y = directed_sortable_u64(objectives[1], self.directions[1]);
        let pos = staircase.partition_point(|pt| pt.s0 > x);
        if pos > 0 && staircase[pos - 1].s1 >= y {
            return false;
        }
        if pos < staircase.len() && staircase[pos].s0 == x {
            if staircase[pos].s1 > y {
                return false;
            }
            if staircase[pos].s1 == y {
                let entry = ParetoEntry {
                    candidate,
                    payload,
                    objectives,
                };
                if let Some(insert_pos) = slot {
                    self.entries.insert(insert_pos, entry);
                    entry_keys.insert(insert_pos, [x, y]);
                } else {
                    self.entries.push(entry);
                    entry_keys.push([x, y]);
                }
                return true;
            }
        }

        let evict_end = pos + staircase[pos..].partition_point(|pt| pt.s1 <= y);
        let point = StaircasePoint {
            s0: x,
            s1: y,
            x: objectives[0],
            y: objectives[1],
        };
        let insert_pos = if evict_end == pos {
            staircase.insert(pos, point);
            slot
        } else {
            staircase[pos] = point;
            if evict_end > pos + 1 {
                staircase.drain(pos + 1..evict_end);
            }
            let mut write = 0;
            let mut new_slot = 0;
            for read in 0..self.entries.len() {
                let [k0, k1] = entry_keys[read];
                // Because an exact tie (k0 == x && k1 == y) returned above,
                // k0 <= x && k1 <= y holds iff (x, y) strictly dominates (k0, k1).
                if k0 <= x && k1 <= y {
                    continue;
                }
                if self.entries[read].candidate < candidate {
                    new_slot += 1;
                }
                if write != read {
                    self.entries.swap(write, read);
                    entry_keys[write] = [k0, k1];
                }
                write += 1;
            }
            self.entries.truncate(write);
            entry_keys.truncate(write);
            slot.map(|_| new_slot)
        };

        let entry = ParetoEntry {
            candidate,
            payload,
            objectives,
        };
        if let Some(pos) = insert_pos {
            self.entries.insert(pos, entry);
            entry_keys.insert(pos, [x, y]);
        } else {
            self.entries.push(entry);
            entry_keys.push([x, y]);
        }
        true
    }

    /// Single-pass flat-key dominance check and compaction for 1D and `>= 3D`
    /// archives.
    fn insert_flat(
        &mut self,
        candidate: CandidateId,
        payload: C,
        objectives: Vec<f64>,
        slot: Option<usize>,
    ) -> bool {
        let FrontierIndex::Flat { keys } = &mut self.index else {
            return false;
        };
        let dim = self.directions.len();
        let mut stack_keys = [0_u64; 8];
        let mut heap_keys = Vec::new();
        let new_keys: &mut [u64] = if dim <= stack_keys.len() {
            &mut stack_keys[..dim]
        } else {
            heap_keys.resize(dim, 0);
            heap_keys.as_mut_slice()
        };
        for ((dest, &value), &dir) in new_keys.iter_mut().zip(&objectives).zip(&self.directions) {
            *dest = directed_sortable_u64(value, dir);
        }

        let n = self.entries.len();
        let mut first_evicted = None;
        for idx in 0..n {
            let start = idx * dim;
            let (existing_dom, new_dom) = compare_key_slices(&keys[start..start + dim], new_keys);
            if existing_dom {
                return false;
            }
            if new_dom {
                first_evicted = Some(idx);
                break;
            }
        }

        let insert_pos = if let Some(first) = first_evicted {
            // Dominance is transitive and the frontier is an antichain, so once
            // `new_keys` dominates one existing entry, no remaining entry can
            // dominate `new_keys`. Compact surviving entries in-place.
            let mut write = first;
            for read in (first + 1)..n {
                let read_start = read * dim;
                if keys_dominate(new_keys, &keys[read_start..read_start + dim]) {
                    continue;
                }
                self.entries.swap(write, read);
                let write_start = write * dim;
                keys.copy_within(read_start..read_start + dim, write_start);
                write += 1;
            }
            self.entries.truncate(write);
            keys.truncate(write * dim);
            slot.map(|_| {
                self.entries
                    .binary_search_by_key(&candidate, |entry| entry.candidate)
                    .unwrap_or_else(|pos| pos)
            })
        } else {
            slot
        };

        let entry = ParetoEntry {
            candidate,
            payload,
            objectives,
        };
        if let Some(pos) = insert_pos {
            self.entries.insert(pos, entry);
            let key_pos = pos * dim;
            keys.splice(key_pos..key_pos, new_keys.iter().copied());
        } else {
            self.entries.push(entry);
            keys.extend_from_slice(new_keys);
        }
        true
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
        let mut keyed: Vec<(u64, f64, usize)> = Vec::with_capacity(n);
        for dim in 0..self.directions.len() {
            keyed.clear();
            for (idx, entry) in self.entries.iter().enumerate() {
                let value = entry.objectives[dim];
                keyed.push((f64_to_sortable_u64(value), value, idx));
            }
            // `self.entries` is sorted by ascending `CandidateId`, so breaking
            // equal-objective ties by `idx` matches `CandidateId` order exactly.
            keyed.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)));
            distances[keyed[0].2] = f64::INFINITY;
            distances[keyed[n - 1].2] = f64::INFINITY;
            let span = keyed[n - 1].1 - keyed[0].1;
            if !span.is_finite() || span <= 0.0 {
                continue;
            }
            for window in keyed.windows(3) {
                let curr = window[1].2;
                if distances[curr].is_finite() {
                    distances[curr] += (window[2].1 - window[0].1) / span;
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
        let FrontierIndex::TwoD { staircase, .. } = &self.index else {
            return Ok(0.0);
        };
        let d0 = self.directions[0];
        let d1 = self.directions[1];
        // `staircase` is ordered by strictly decreasing objective-0 quality
        // (non-increasing `dx`) and strictly increasing objective-1 quality
        // (non-decreasing `dy`). Coalescing equal-`dx` runs to their largest
        // `dy` visits points in the exact same order as sorting by
        // `(dx desc, dy desc)` without allocating or sorting.
        let mut area = 0.0_f64;
        let mut best_y = 0.0_f64;
        let mut pending: Option<(f64, f64)> = None;
        for pt in staircase {
            let dx = improvement_margin(pt.x, reference[0], d0);
            if dx <= 0.0 {
                break;
            }
            let dy = improvement_margin(pt.y, reference[1], d1);
            if dy <= 0.0 {
                continue;
            }
            match pending {
                Some((px, ref mut py)) if px.to_bits() == dx.to_bits() => {
                    if dy > *py {
                        *py = dy;
                    }
                }
                Some((px, py)) => {
                    if py > best_y {
                        area = px.mul_add(py - best_y, area);
                        best_y = py;
                    }
                    pending = Some((dx, dy));
                }
                None => pending = Some((dx, dy)),
            }
        }
        if let Some((px, py)) = pending {
            if py > best_y {
                area = px.mul_add(py - best_y, area);
            }
        }
        Ok(area)
    }
}

/// Computes positive margin by which `value` improves on `reference` under
/// `direction`.
#[inline]
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
    let len = left.len().min(right.len()).min(directions.len());
    let mut strictly_better = false;
    for index in 0..len {
        let a = directed_sortable_u64(left[index], directions[index]);
        let b = directed_sortable_u64(right[index], directions[index]);
        if a < b {
            return false;
        }
        strictly_better |= a > b;
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

    #[test]
    fn pareto_handles_signed_zero_ties_high_dim_and_hypervolume_rounding() {
        // Signed-zero total_cmp semantics: +0.0 > -0.0 under Maximize, -0.0 > +0.0 under Minimize.
        assert!(dominates(&[0.0], &[-0.0], &[ObjectiveDirection::Maximize]));
        assert!(dominates(&[-0.0], &[0.0], &[ObjectiveDirection::Minimize]));
        assert!(dominates(
            &[f64::NAN],
            &[f64::INFINITY],
            &[ObjectiveDirection::Maximize]
        ));

        // 2D exact ties, eviction of tied points, out-of-order CandidateIds, and equal-dx rounding.
        let mut arch = ParetoArchive::new(vec![
            ObjectiveDirection::Maximize,
            ObjectiveDirection::Maximize,
        ])
        .expect("2d");
        assert!(
            arch.insert(CandidateId::new(5), "a", vec![1.0, 2.0])
                .expect("a")
        );
        assert!(
            arch.insert(CandidateId::new(2), "b", vec![1.0, 2.0])
                .expect("b")
        );
        assert_eq!(2, arch.len());
        // Strictly dominates both tied points (1.0, 2.0).
        assert!(
            arch.insert(CandidateId::new(4), "c", vec![1.0, 3.0])
                .expect("c")
        );
        assert_eq!(1, arch.len());
        assert_eq!(CandidateId::new(4), arch.frontier()[0].candidate);

        // Two distinct x values that round to equal dx against a large negative reference.
        let mut hv_arch = ParetoArchive::new(vec![
            ObjectiveDirection::Maximize,
            ObjectiveDirection::Maximize,
        ])
        .expect("2d");
        assert!(
            hv_arch
                .insert(CandidateId::new(0), (), vec![2.0, 1.0])
                .expect("p0")
        );
        assert!(
            hv_arch
                .insert(CandidateId::new(1), (), vec![1.0, 5.0])
                .expect("p1")
        );
        // Reference -1e20 makes (2.0 - (-1e20)) == (1.0 - (-1e20)) == 1e20.
        let hv = hv_arch.hypervolume_2d([-1.0e20, 0.0]).expect("hv");
        assert_eq!((1.0e20_f64 * 5.0).to_bits(), hv.to_bits());

        // High-dimensional (10D, > stack key buffer) archive matches naive dominance.
        let dirs = vec![ObjectiveDirection::Maximize; 10];
        let mut hi = ParetoArchive::new(dirs).expect("10d");
        assert!(
            hi.insert(CandidateId::new(10), 10, vec![1.0; 10])
                .expect("p10")
        );
        let mut better = vec![1.0; 10];
        better[9] = 2.0;
        assert!(hi.insert(CandidateId::new(3), 3, better).expect("p3"));
        assert_eq!(1, hi.len());
        assert_eq!(CandidateId::new(3), hi.frontier()[0].candidate);

        // PartialEq and Debug remain identical across equivalent archives.
        let mut arch2 = ParetoArchive::new(vec![
            ObjectiveDirection::Maximize,
            ObjectiveDirection::Maximize,
        ])
        .expect("2d");
        assert!(
            arch2
                .insert(CandidateId::new(4), "c", vec![1.0, 3.0])
                .expect("c")
        );
        assert_eq!(arch, arch2);
        assert_eq!(format!("{arch:?}"), format!("{arch2:?}"));
    }
}
