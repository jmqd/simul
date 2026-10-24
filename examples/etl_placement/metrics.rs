//! Measurements shared by the agents of one run.
//!
//! Agents are type-erased once inside the engine, so they report into one
//! `Rc<RefCell<Recorder>>` instead of being read back after the run.

use crate::time::{secs, Tick};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) type SharedRecorder = Rc<RefCell<Recorder>>;

/// Weighted log-scale histogram with 1% bucket width.
#[derive(Clone, Debug)]
pub(crate) struct Histogram {
    counts: Vec<u64>,
    total: u64,
    max: f64,
}

const HISTOGRAM_MIN_S: f64 = 1e-4;
const HISTOGRAM_GROWTH: f64 = 1.01;
const HISTOGRAM_BUCKETS: usize = 2000;

impl Default for Histogram {
    fn default() -> Self {
        Self {
            counts: vec![0; HISTOGRAM_BUCKETS],
            total: 0,
            max: 0.0,
        }
    }
}

impl Histogram {
    pub(crate) fn record(&mut self, value_s: f64, weight: u64) {
        if weight == 0 {
            return;
        }
        let index = if value_s <= HISTOGRAM_MIN_S {
            0
        } else {
            let position = (value_s / HISTOGRAM_MIN_S).log(HISTOGRAM_GROWTH);
            usize::try_from(crate::time::to_u64(position.floor()))
                .unwrap_or(usize::MAX)
                .min(HISTOGRAM_BUCKETS - 1)
        };
        self.counts[index] += weight;
        self.total += weight;
        self.max = self.max.max(value_s);
    }

    pub(crate) const fn total(&self) -> u64 {
        self.total
    }

    pub(crate) const fn max(&self) -> f64 {
        self.max
    }

    /// The upper edge of the bucket holding the `quantile` observation.
    pub(crate) fn quantile(&self, quantile: f64) -> Option<f64> {
        if self.total == 0 {
            return None;
        }
        let rank = crate::time::to_u64((quantile * self.total as f64).ceil()).max(1);
        let mut seen = 0;
        for (index, count) in self.counts.iter().enumerate() {
            seen += count;
            if seen >= rank {
                let exponent = i32::try_from(index + 1).unwrap_or(i32::MAX);
                return Some((HISTOGRAM_MIN_S * HISTOGRAM_GROWTH.powi(exponent)).min(self.max));
            }
        }
        Some(self.max)
    }
}

/// Copy-phase timeline for one table.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TableTimeline {
    pub(crate) copy_started: Option<Tick>,
    pub(crate) copy_finished: Option<Tick>,
    pub(crate) catchup_finished: Option<Tick>,
    pub(crate) ready: Option<Tick>,
}

/// Rows of one table carried by a write, with their commit time and LSN.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowGroup {
    pub(crate) table: u64,
    pub(crate) commit_tick: Tick,
    pub(crate) rows: u64,
    /// The commit LSN, or zero for a slice of a transaction committed later.
    pub(crate) lsn: u64,
}

/// A CDC write the destination has not finished: parts outstanding, the
/// latest part's durable time, and its rows.
#[derive(Clone, Debug)]
struct PendingWrite {
    parts_left: u64,
    durable_at: Tick,
    rows: Vec<RowGroup>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Recorder {
    /// Commits before this tick are excluded from steady-state statistics.
    pub(crate) warmup: Tick,
    /// Commit to durable at the destination, weighted by rows.
    pub(crate) latency: Histogram,
    /// Rows written to the destination again after a restart replayed them.
    pub(crate) rows_replayed: u64,
    /// Highest commit LSN made durable, per table.
    pub(crate) highest_durable_lsn: Vec<u64>,
    /// WAL retained behind the oldest slot, sampled every second. It tracks
    /// the slot's confirmed flush, which moves only on status updates, so it
    /// measures WAL retention, not replication progress.
    pub(crate) slot_lag_bytes: Vec<(Tick, u64)>,
    /// Commit time of the newest row made durable at the destination.
    pub(crate) latest_durable_commit: Tick,
    /// Replication lag sampled every second: zero when every commit is
    /// durable, otherwise the age of the newest durable commit.
    pub(crate) replication_lag_s: Vec<(Tick, f64)>,
    pub(crate) batches: u64,
    pub(crate) batch_decoded_bytes: f64,
    pub(crate) write_seconds: Histogram,
    pub(crate) checkpoint_seconds: Histogram,
    pub(crate) paused_ticks: u64,
    pub(crate) blocked_ticks: u64,
    /// Bytes each agent sent across regions: postgres, etl, destination.
    pub(crate) cross_region_bytes: [f64; 3],
    pub(crate) copy_bytes: f64,
    pub(crate) tables: Vec<TableTimeline>,
    pub(crate) copy_done_at: Option<Tick>,
    pub(crate) events: Vec<(Tick, String)>,
    /// CDC writes registered by ETL, settled by the destination. Rows count
    /// as durable when the destination makes them so, even if the ETL loop
    /// that sent them has since failed.
    pending_writes: std::collections::BTreeMap<u64, PendingWrite>,
}

impl Recorder {
    pub(crate) fn shared(warmup: Tick, tables: usize) -> SharedRecorder {
        Rc::new(RefCell::new(Self {
            warmup,
            tables: vec![TableTimeline::default(); tables],
            highest_durable_lsn: vec![0; tables],
            ..Self::default()
        }))
    }

    pub(crate) fn event(&mut self, at: Tick, text: impl Into<String>) {
        self.events.push((at, text.into()));
    }

    pub(crate) fn register_write(&mut self, write: u64, parts: u64, rows: Vec<RowGroup>) {
        self.pending_writes.insert(
            write,
            PendingWrite {
                parts_left: parts,
                durable_at: 0,
                rows,
            },
        );
    }

    /// The destination made one part of `write` durable at `durable_at`.
    pub(crate) fn part_durable(&mut self, write: u64, durable_at: Tick) {
        let Some(pending) = self.pending_writes.get_mut(&write) else {
            return;
        };
        pending.parts_left -= 1;
        pending.durable_at = pending.durable_at.max(durable_at);
        if pending.parts_left > 0 {
            return;
        }
        let Some(done) = self.pending_writes.remove(&write) else {
            return;
        };
        for group in done.rows {
            self.durable(group, done.durable_at);
        }
    }

    fn durable(&mut self, group: RowGroup, durable_at: Tick) {
        let table = usize::try_from(group.table).unwrap_or(0);
        let highest = &mut self.highest_durable_lsn[table];
        if group.lsn != 0 {
            if group.lsn <= *highest {
                self.rows_replayed += group.rows;
                return;
            }
            *highest = group.lsn;
        }
        self.latest_durable_commit = self.latest_durable_commit.max(group.commit_tick);
        if group.commit_tick >= self.warmup {
            self.latency.record(
                secs(durable_at.saturating_sub(group.commit_tick)),
                group.rows,
            );
        }
    }

    /// Samples replication lag given the source's latest commit time.
    pub(crate) fn sample_lag(&mut self, now: Tick, latest_commit: Tick) {
        let lag = if latest_commit <= self.latest_durable_commit {
            0.0
        } else {
            secs(now.saturating_sub(self.latest_durable_commit))
        };
        self.replication_lag_s.push((now, lag));
    }
}
