//! The source Postgres in us-west-2: the application workload, WAL, the
//! logical replication slot and walsenders, COPY backends, and the ETL state
//! store (the `etl` schema in the source database, the standalone
//! replicator's default [etl]).

use crate::config::Config;
use crate::metrics::SharedRecorder;
use crate::net::{Endpoint, NetEvent, Peer, send_tracked};
use crate::rng::Rng;
use crate::time::{TICKS_PER_SECOND, Tick, per_tick, ticks, to_u64};
use crate::wire::{ConnId, ConnKind, Op, Payload, Unit, WalChunk};
use simul::agent::{Agent, AgentContext};
use simul::message::Message;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

pub(crate) const NAME: &str = "postgres";

/// Commits aggregate per table per millisecond, bounding history size while
/// keeping commit times within 1 ms.
const AGGREGATION_TICKS: Tick = 10;

/// Uniform bins over `[0, 1)` for exact-region geometric sampling without `ln`.
const GEO_LUT_LEN: usize = 512;

/// Builds a lookup table mapping uniform `u` bins `[b/N, (b+1)/N)` to their
/// guaranteed `1 + floor(ln(1 - u) / ln_failure)` value, or `0` for bins that
/// straddle (or sit within `1e-11` of) an integer boundary.
fn build_geo_lut(ln_failure: f64) -> [u8; GEO_LUT_LEN] {
    let mut lut = [0_u8; GEO_LUT_LEN];
    if ln_failure >= 0.0 {
        return lut;
    }
    let scale = GEO_LUT_LEN as f64;
    for (bin, slot) in lut.iter_mut().enumerate() {
        let lo_x = 1.0 - (bin + 1) as f64 / scale;
        let hi_x = 1.0 - bin as f64 / scale;
        if lo_x <= 0.0 {
            continue;
        }
        let lo_val = (lo_x * (1.0 - 1e-11)).ln() / ln_failure;
        let k_hi = if bin == 0 {
            0
        } else {
            let hi_val = (hi_x * (1.0 + 1e-11)).min(1.0).ln() / ln_failure;
            if hi_val < 0.0 {
                continue;
            }
            to_u64(hi_val.floor())
        };
        if lo_val < 0.0 {
            continue;
        }
        let k_lo = to_u64(lo_val.floor());
        if k_lo == k_hi {
            if let Ok(rows) = u8::try_from(k_lo + 1) {
                *slot = rows;
            }
        }
    }
    lut
}

/// Rows of one table committed within one aggregation window.
#[derive(Clone, Copy, Debug)]
struct Record {
    table: u64,
    rows: u64,
    commits: u64,
    end_lsn: u64,
    commit_tick: Tick,
}

#[derive(Clone, Copy, Debug, Default)]
struct OpenAggregate {
    rows: u64,
    commits: u64,
    end_lsn: u64,
    weighted_offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Slot {
    Main,
    Table(u64),
}

#[derive(Clone, Debug)]
struct WalSender {
    slot: Slot,
    /// Absolute index into the history of the next record to decode.
    cursor: u64,
    /// Rows of the record at `cursor` already sent.
    partial_rows: u64,
    sent_lsn: u64,
    decode_carry: f64,
    client_write: u64,
    last_keepalive_lsn: u64,
    last_client_message: Tick,
    ping_outstanding: bool,
}

#[derive(Clone, Debug)]
struct CopyOut {
    rows_left: u64,
    bytes_left: f64,
    bytes_per_row: f64,
    carry: f64,
}

#[derive(Clone, Debug)]
enum Mode {
    Sql,
    WalSender(WalSender),
    Copy(CopyOut),
}

/// One backend process serving one connection.
#[derive(Clone, Debug)]
struct Backend {
    queue: VecDeque<(u64, Op, u64, u64)>,
    running: Option<(u64, Op, u64, u64, Tick)>,
    mode: Mode,
}

#[derive(Clone, Debug)]
pub(crate) struct PostgresAgent {
    config: Rc<Config>,
    recorder: SharedRecorder,
    net: Endpoint,
    network_rng: Rng,
    workload_rng: Rng,
    lsn: u64,
    burst_high: bool,
    burst_switch_at: Tick,
    /// Precomputed workload/streaming rates and thresholds.
    base_tx_per_tick: f64,
    burst_low_mult: f64,
    burst_mean_low_s: f64,
    current_tx_per_tick: f64,
    poisson_limit: f64,
    geo_ln_failure: f64,
    wal_bytes_per_row_int: Option<u64>,
    decode_rows_per_tick: f64,
    copy_bytes_per_tick: f64,
    wal_sender_timeout_ticks: Tick,
    wal_sender_half_timeout_ticks: Tick,
    row_wire_bytes: f64,
    open: Vec<OpenAggregate>,
    open_dirty: bool,
    scratch_order: Vec<(u64, usize)>,
    history: VecDeque<Record>,
    history_base: u64,
    confirmed_flush: u64,
    table_slots: BTreeMap<u64, u64>,
    stored_checkpoint: u64,
    /// Backends kept sorted by ascending `ConnId`.
    backends: Vec<(ConnId, Backend)>,
    /// The earliest tick at which `run_statements` has work: a running
    /// statement finishes. A newly queued query resets it to 0.
    next_statement_at: Tick,
    /// The earliest tick at which `stream` has work: a copy backend is active,
    /// a walsender has unread WAL or a blocked keepalive, or a ping/timeout
    /// falls due.
    next_stream_at: Tick,
    /// Whether `trim_history` may have newly droppable records.
    trim_dirty: bool,
    next_resets: usize,
    copy_bytes_per_row: Vec<f64>,
    /// Commit time of the newest record visible to decoding.
    newest_commit: Tick,
    last_reported_cross_region: f64,
    events_buf: Vec<NetEvent>,
    geo_lut: Box<[u8; GEO_LUT_LEN]>,
}

impl PostgresAgent {
    pub(crate) fn new(
        config: Rc<Config>,
        recorder: SharedRecorder,
        network_rng: Rng,
        workload_rng: Rng,
        etl: Peer,
    ) -> Self {
        let net = Endpoint::new(0, config.tcp.clone(), config.source_nic, vec![etl]);
        let copy_bytes_per_row = config
            .copy_tables
            .iter()
            .map(|table| config.copy_wire_bytes(table.data_bytes_per_row))
            .collect();
        let workload = &config.workload;
        let base_tx_per_tick = per_tick(workload.rows_per_s / workload.mean_rows_per_transaction);
        let (burst_switch_at, burst_low_mult, burst_mean_low_s) =
            workload.burst.map_or((Tick::MAX, 1.0, 0.0), |burst| {
                let low = burst.fraction.mul_add(-burst.multiplier, 1.0) / (1.0 - burst.fraction);
                let mean_low_s = burst.mean_high_s * (1.0 - burst.fraction) / burst.fraction;
                (0, low, mean_low_s)
            });
        let current_tx_per_tick = base_tx_per_tick * 1.0;
        let poisson_limit = (-current_tx_per_tick).exp();
        let geo_ln_failure = if workload.mean_rows_per_transaction > 1.0 {
            (1.0 - 1.0 / workload.mean_rows_per_transaction).ln()
        } else {
            0.0
        };
        let geo_lut = build_geo_lut(geo_ln_failure);
        let wal_bytes = config.source.wal_bytes_per_row;
        let wal_bytes_per_row_int = (0.0..=1e6)
            .contains(&wal_bytes)
            .then(|| to_u64(wal_bytes))
            .filter(|&n| (n as f64).to_bits() == wal_bytes.to_bits());
        let decode_rows_per_tick = per_tick(config.source.decode_rows_per_s);
        let copy_bytes_per_tick = per_tick(config.source.copy_bytes_per_s);
        let wal_sender_timeout_ticks = ticks(config.source.wal_sender_timeout_s);
        let wal_sender_half_timeout_ticks = wal_sender_timeout_ticks.div_euclid(2);
        let row_wire_bytes = config.row_wire_bytes(workload.data_bytes_per_row);
        let tables = workload.tables;
        Self {
            open: vec![OpenAggregate::default(); tables],
            open_dirty: false,
            scratch_order: Vec::with_capacity(tables),
            config,
            recorder,
            net,
            network_rng,
            workload_rng,
            lsn: 1,
            burst_high: false,
            burst_switch_at,
            base_tx_per_tick,
            burst_low_mult,
            burst_mean_low_s,
            current_tx_per_tick,
            poisson_limit,
            geo_ln_failure,
            wal_bytes_per_row_int,
            decode_rows_per_tick,
            copy_bytes_per_tick,
            wal_sender_timeout_ticks,
            wal_sender_half_timeout_ticks,
            row_wire_bytes,
            history: VecDeque::new(),
            history_base: 0,
            confirmed_flush: 1,
            table_slots: BTreeMap::new(),
            stored_checkpoint: 0,
            backends: Vec::new(),
            next_statement_at: Tick::MAX,
            next_stream_at: Tick::MAX,
            trim_dirty: false,
            next_resets: 0,
            copy_bytes_per_row,
            newest_commit: 0,
            last_reported_cross_region: 0.0,
            events_buf: Vec::new(),
            geo_lut: Box::new(geo_lut),
        }
    }

    #[inline]
    fn find_backend(&self, conn: ConnId) -> Option<usize> {
        self.backends
            .binary_search_by_key(&conn, |&(id, _)| id)
            .ok()
    }

    #[inline]
    fn backend_mut(&mut self, conn: ConnId) -> Option<&mut Backend> {
        let idx = self.find_backend(conn)?;
        Some(&mut self.backends[idx].1)
    }

    fn insert_backend(&mut self, conn: ConnId, backend: Backend) {
        if self.backends.last().is_none_or(|&(last, _)| conn > last) {
            self.backends.push((conn, backend));
        } else {
            match self.backends.binary_search_by_key(&conn, |&(id, _)| id) {
                Ok(idx) => self.backends[idx].1 = backend,
                Err(idx) => self.backends.insert(idx, (conn, backend)),
            }
        }
    }

    fn remove_backend(&mut self, conn: ConnId) -> Option<Backend> {
        let idx = self.find_backend(conn)?;
        Some(self.backends.remove(idx).1)
    }

    fn history_end(&self) -> u64 {
        self.history_base + self.history.len() as u64
    }

    /// LSN of the last record visible to decoding.
    fn flushed_lsn(&self) -> u64 {
        self.history
            .back()
            .map_or(self.confirmed_flush, |record| record.end_lsn)
    }

    #[cold]
    fn switch_burst(&mut self, now: Tick) {
        let Some(burst) = self.config.workload.burst else {
            return;
        };
        self.burst_high = !self.burst_high && now > 0;
        let mean = if self.burst_high {
            burst.mean_high_s
        } else {
            self.burst_mean_low_s
        };
        self.burst_switch_at = now + ticks(self.workload_rng.exponential(mean)).max(1);
        let multiplier = if self.burst_high {
            burst.multiplier
        } else {
            self.burst_low_mult
        };
        self.current_tx_per_tick = self.base_tx_per_tick * multiplier;
        self.poisson_limit = (-self.current_tx_per_tick).exp();
    }

    #[cold]
    #[inline(never)]
    fn geometric_fallback(&self, u: f64) -> u64 {
        1 + to_u64(((1.0 - u).ln() / self.geo_ln_failure).floor())
    }

    fn generate(&mut self, now: Tick) {
        if now >= self.burst_switch_at {
            self.switch_burst(now);
        }
        let draining = self.config.workload_until.is_some_and(|until| now >= until);
        let count = if draining {
            0
        } else {
            let mean = self.current_tx_per_tick;
            if mean <= 0.0 {
                0
            } else if mean > 30.0 {
                self.workload_rng.poisson(mean)
            } else {
                const MAX_COUNT: u64 = 1_000;
                let limit = self.poisson_limit;
                let mut product = self.workload_rng.unit();
                let mut drawn = 0;
                while product > limit && drawn < MAX_COUNT {
                    drawn += 1;
                    product *= self.workload_rng.unit();
                }
                drawn
            }
        };
        if count > 0 {
            self.open_dirty = true;
            let workload = &self.config.workload;
            let tables = workload.tables as u64;
            let wal_bytes_per_row = self.config.source.wal_bytes_per_row;
            let wal_bytes_int = self.wal_bytes_per_row_int;
            let offset = now % AGGREGATION_TICKS;
            for _ in 0..count {
                let rows = if self.geo_ln_failure >= 0.0 {
                    self.workload_rng
                        .geometric(workload.mean_rows_per_transaction)
                } else {
                    let u = self.workload_rng.unit();
                    let bin = usize::try_from(to_u64(u * GEO_LUT_LEN as f64))
                        .unwrap_or(GEO_LUT_LEN - 1)
                        .min(GEO_LUT_LEN - 1);
                    let cached = self.geo_lut[bin];
                    if cached != 0 {
                        u64::from(cached)
                    } else {
                        self.geometric_fallback(u)
                    }
                };
                let table = self.workload_rng.below(tables);
                let wal_bytes = wal_bytes_int.map_or_else(
                    || to_u64(rows as f64 * wal_bytes_per_row),
                    |per_row| rows * per_row,
                );
                self.lsn += wal_bytes + 100;
                let slot = &mut self.open[usize::try_from(table).unwrap_or(0)];
                slot.rows += rows;
                slot.commits += 1;
                slot.end_lsn = self.lsn;
                slot.weighted_offset += rows * offset;
            }
        }
        if (now + 1) % AGGREGATION_TICKS == 0 && self.open_dirty {
            self.flush_open_aggregates(now);
        }
    }

    fn flush_open_aggregates(&mut self, now: Tick) {
        self.open_dirty = false;
        let base = now + 1 - AGGREGATION_TICKS;
        self.scratch_order.clear();
        for (table, slot) in self.open.iter().enumerate() {
            if slot.commits > 0 {
                self.scratch_order.push((slot.end_lsn, table));
            }
        }
        if self.scratch_order.len() > 1 {
            self.scratch_order.sort_unstable_by_key(|&(lsn, _)| lsn);
        }
        let audit = self.config.audit_delivery;
        if audit {
            let mut recorder = self.recorder.borrow_mut();
            for &(lsn, table) in &self.scratch_order {
                recorder.committed(table, lsn);
            }
        }
        for &(_, table) in &self.scratch_order {
            let taken = std::mem::take(&mut self.open[table]);
            let commit_tick = if taken.rows > 0 {
                base + taken.weighted_offset.div_euclid(taken.rows)
            } else {
                0
            };
            self.newest_commit = self.newest_commit.max(commit_tick);
            self.history.push_back(Record {
                table: table as u64,
                rows: taken.rows,
                commits: taken.commits,
                end_lsn: taken.end_lsn,
                commit_tick,
            });
        }
        self.next_stream_at = 0;
    }

    fn service_time(&self, op: Op) -> f64 {
        let source = &self.config.source;
        match op {
            Op::Begin
            | Op::AdvisoryLock
            | Op::UpsertCheckpoint
            | Op::SetTableState
            | Op::ExportSnapshot
            | Op::SetSnapshot
            | Op::LockTable
            | Op::StartReplication
            | Op::StartCatchup
            | Op::StartCopy => source.statement_s,
            Op::Commit | Op::CommitSlotTransaction => source.commit_s,
            Op::LoadCheckpoint | Op::StoreSchema | Op::Select | Op::GetSlot => source.select_s,
            Op::Migrations => source.migrations_s,
            Op::CreateSlot => source.create_slot_s,
            Op::DropSlot => 2.0 * source.commit_s,
        }
    }

    /// Applies a finished statement and returns its reply value.
    fn execute(&mut self, conn: ConnId, op: Op, a: u64, b: u64, now: Tick) -> u64 {
        match op {
            Op::UpsertCheckpoint => {
                self.stored_checkpoint = self.stored_checkpoint.max(a);
                0
            }
            Op::LoadCheckpoint => self.stored_checkpoint,
            Op::GetSlot => self.confirmed_flush,
            Op::CreateSlot => {
                let lsn = self.flushed_lsn();
                self.table_slots.insert(a, lsn);
                lsn
            }
            Op::DropSlot => {
                self.table_slots.remove(&a);
                self.trim_dirty = true;
                0
            }
            Op::StartReplication | Op::StartCatchup => {
                let (slot, from) = if op == Op::StartReplication {
                    (Slot::Main, a.max(self.confirmed_flush))
                } else {
                    (
                        Slot::Table(a),
                        self.table_slots
                            .get(&a)
                            .copied()
                            .unwrap_or(self.confirmed_flush),
                    )
                };
                let skip = self
                    .history
                    .iter()
                    .take_while(|record| record.end_lsn <= from)
                    .count();
                let sender = WalSender {
                    slot,
                    cursor: self.history_base + skip as u64,
                    partial_rows: 0,
                    sent_lsn: from,
                    decode_carry: 0.0,
                    client_write: from,
                    last_keepalive_lsn: from,
                    last_client_message: now,
                    ping_outstanding: false,
                };
                if let Some(backend) = self.backend_mut(conn) {
                    backend.mode = Mode::WalSender(sender);
                }
                self.net.use_receiver_credit(conn);
                self.net.set_quantum(conn, AGGREGATION_TICKS);
                self.next_stream_at = 0;
                0
            }
            Op::StartCopy => {
                let table = usize::try_from(b).unwrap_or(0);
                let bytes_per_row = self.copy_bytes_per_row.get(table).copied().unwrap_or(100.0);
                if let Some(backend) = self.backend_mut(conn) {
                    backend.mode = Mode::Copy(CopyOut {
                        rows_left: a,
                        bytes_left: a as f64 * bytes_per_row,
                        bytes_per_row,
                        carry: 0.0,
                    });
                }
                self.net.use_receiver_credit(conn);
                self.net.set_quantum(conn, AGGREGATION_TICKS);
                self.next_stream_at = 0;
                0
            }
            Op::Begin
            | Op::Commit
            | Op::AdvisoryLock
            | Op::SetTableState
            | Op::StoreSchema
            | Op::Select
            | Op::Migrations
            | Op::ExportSnapshot
            | Op::SetSnapshot
            | Op::LockTable
            | Op::CommitSlotTransaction => 0,
        }
    }

    fn run_statements(&mut self, now: Tick) {
        // Before then, every backend is waiting on a running statement or
        // has none queued, so a pass would change nothing.
        if now < self.next_statement_at {
            return;
        }
        let mut next_statement_at = Tick::MAX;
        for idx in 0..self.backends.len() {
            let conn = self.backends[idx].0;
            loop {
                match self.backends[idx].1.running {
                    Some((id, op, a, b, done_at)) if done_at <= now => {
                        self.backends[idx].1.running = None;
                        let value = self.execute(conn, op, a, b, now);
                        send_tracked(&mut self.net, conn, Unit::Reply { id, value });
                    }
                    Some((.., done_at)) => {
                        next_statement_at = next_statement_at.min(done_at);
                        break;
                    }
                    None => {
                        let Some((id, op, a, b)) = self.backends[idx].1.queue.pop_front() else {
                            break;
                        };
                        let service = ticks(self.service_time(op));
                        self.backends[idx].1.running = Some((id, op, a, b, now + service));
                    }
                }
            }
        }
        self.next_statement_at = next_statement_at;
    }

    fn stream(&mut self, now: Tick) {
        if now < self.next_stream_at {
            return;
        }
        let rows_per_tick = self.decode_rows_per_tick;
        let copy_rate = self.copy_bytes_per_tick;
        let send_buffer = self.net.send_buffer_bytes();
        let timeout = self.wal_sender_timeout_ticks;
        let half_timeout = self.wal_sender_half_timeout_ticks;
        let row_wire = self.row_wire_bytes;
        let transaction_wire = self.config.source.transaction_overhead_bytes;
        let flushed = self.flushed_lsn();
        let history_end = self.history_end();
        let mut next_due = Tick::MAX;
        let mut terminated = Vec::new();
        for &mut (conn, ref mut backend) in &mut self.backends {
            match &mut backend.mode {
                Mode::WalSender(sender) => {
                    if now.saturating_sub(sender.last_client_message) >= timeout {
                        terminated.push(conn);
                        continue;
                    }
                    if sender.cursor < history_end {
                        let mut budget = sender.decode_carry + rows_per_tick;
                        while sender.cursor < history_end
                            && budget >= 1.0
                            && self.net.buffered(conn) < send_buffer
                        {
                            let Some(record) = sender
                                .cursor
                                .checked_sub(self.history_base)
                                .and_then(|index| usize::try_from(index).ok())
                                .and_then(|index| self.history.get(index))
                                .copied()
                            else {
                                break;
                            };
                            let available = record.rows - sender.partial_rows;
                            let take = available.min(to_u64(budget.floor()));
                            let completes = take == available;
                            let chunk = WalChunk {
                                table: record.table,
                                rows: take,
                                wire_bytes: to_u64((take as f64).mul_add(
                                    row_wire,
                                    if completes {
                                        record.commits as f64 * transaction_wire
                                    } else {
                                        0.0
                                    },
                                )),
                                commits: if completes { record.commits } else { 0 },
                                end_lsn: if completes { record.end_lsn } else { 0 },
                                commit_tick: record.commit_tick,
                            };
                            send_tracked(&mut self.net, conn, Unit::Wal(chunk));
                            budget -= take as f64;
                            if completes {
                                sender.cursor += 1;
                                sender.partial_rows = 0;
                                sender.sent_lsn = record.end_lsn;
                                self.trim_dirty = true;
                            } else {
                                sender.partial_rows += take;
                            }
                        }
                        sender.decode_carry = if sender.cursor >= history_end {
                            0.0
                        } else {
                            budget.min(rows_per_tick)
                        };
                    } else {
                        sender.decode_carry = 0.0;
                    }
                    let caught_up = sender.cursor >= history_end;
                    let mut needs_poll = !caught_up;
                    // WalSndWaitForWal: when idle with the client behind, send a
                    // keepalive carrying the current end of WAL.
                    if caught_up {
                        let target_lsn = sender.sent_lsn.max(flushed);
                        let need_keepalive = sender.client_write < target_lsn
                            && sender.last_keepalive_lsn < target_lsn;
                        if target_lsn > sender.sent_lsn || need_keepalive {
                            if self.net.buffered(conn) < send_buffer {
                                sender.sent_lsn = target_lsn;
                                if need_keepalive {
                                    sender.last_keepalive_lsn = sender.sent_lsn;
                                    send_tracked(
                                        &mut self.net,
                                        conn,
                                        Unit::Keepalive {
                                            wal_end: sender.sent_lsn,
                                            reply: false,
                                        },
                                    );
                                }
                            } else {
                                needs_poll = true;
                            }
                        }
                    }
                    if !sender.ping_outstanding
                        && now.saturating_sub(sender.last_client_message) >= half_timeout
                    {
                        sender.ping_outstanding = true;
                        send_tracked(
                            &mut self.net,
                            conn,
                            Unit::Keepalive {
                                wal_end: sender.sent_lsn,
                                reply: true,
                            },
                        );
                    }
                    if needs_poll {
                        next_due = 0;
                    } else {
                        let timer_at = if sender.ping_outstanding {
                            sender.last_client_message.saturating_add(timeout)
                        } else {
                            sender.last_client_message.saturating_add(half_timeout)
                        };
                        next_due = next_due.min(timer_at);
                    }
                }
                Mode::Copy(copy) => {
                    let room = (send_buffer - self.net.buffered(conn)).max(0.0);
                    let bytes = (copy.carry + copy_rate).min(room).min(copy.bytes_left);
                    copy.carry = (copy.carry + copy_rate - bytes).min(copy_rate);
                    if bytes <= 0.0 {
                        next_due = 0;
                        continue;
                    }
                    let rows = to_u64((bytes / copy.bytes_per_row).round()).min(copy.rows_left);
                    copy.bytes_left -= bytes;
                    let done = copy.bytes_left <= 0.5;
                    let rows = if done { copy.rows_left } else { rows };
                    copy.rows_left -= rows;
                    send_tracked(
                        &mut self.net,
                        conn,
                        Unit::CopyData {
                            rows,
                            bytes: to_u64(bytes.round()),
                            done,
                        },
                    );
                    if done {
                        backend.mode = Mode::Sql;
                    } else {
                        next_due = 0;
                    }
                }
                Mode::Sql => {}
            }
        }
        if !terminated.is_empty() {
            self.trim_dirty = true;
            for conn in terminated {
                self.remove_backend(conn);
                self.net.abort(conn, now, &self.config.faults);
                self.recorder
                    .borrow_mut()
                    .event(now, "walsender exited: wal_sender_timeout");
            }
        }
        self.next_stream_at = next_due;
    }

    fn on_status(&mut self, conn: ConnId, write: u64, flush: u64, now: Tick) {
        let Some(Backend {
            mode: Mode::WalSender(sender),
            ..
        }) = self.backend_mut(conn)
        else {
            return;
        };
        sender.last_client_message = now;
        sender.ping_outstanding = false;
        sender.client_write = sender.client_write.max(write);
        let slot = sender.slot;
        match slot {
            Slot::Main => self.confirmed_flush = self.confirmed_flush.max(flush),
            Slot::Table(table) => {
                if let Some(lsn) = self.table_slots.get_mut(&table) {
                    *lsn = (*lsn).max(flush);
                }
            }
        }
        self.next_stream_at = 0;
        self.trim_dirty = true;
    }

    /// Drops WAL every slot and sender has passed.
    fn trim_history(&mut self) {
        if !self.trim_dirty {
            return;
        }
        self.trim_dirty = false;
        let Some(front) = self.history.front() else {
            return;
        };
        if front.end_lsn > self.confirmed_flush {
            return;
        }
        let mut keep_lsn = self.confirmed_flush;
        for lsn in self.table_slots.values() {
            keep_lsn = keep_lsn.min(*lsn);
        }
        let mut keep_index = self.history_end();
        for (_, backend) in &self.backends {
            if let Mode::WalSender(sender) = &backend.mode {
                keep_index = keep_index.min(sender.cursor);
            }
        }
        while let Some(front) = self.history.front() {
            if front.end_lsn > keep_lsn || self.history_base >= keep_index {
                break;
            }
            self.history.pop_front();
            self.history_base += 1;
        }
    }

    fn inject_faults(&mut self, now: Tick) {
        while self
            .config
            .faults
            .replication_resets
            .get(self.next_resets)
            .is_some_and(|&at| at <= now)
        {
            self.next_resets += 1;
            let main: Vec<ConnId> = self
                .backends
                .iter()
                .filter(|(_, backend)| {
                    matches!(&backend.mode, Mode::WalSender(sender) if sender.slot == Slot::Main)
                })
                .map(|&(conn, _)| conn)
                .collect();
            for conn in main {
                self.remove_backend(conn);
                self.net.abort(conn, now, &self.config.faults);
            }
            self.trim_dirty = true;
            self.recorder
                .borrow_mut()
                .event(now, "source reset the replication connection");
        }
    }
}

impl Agent<Payload> for PostgresAgent {
    fn on_message(&mut self, ctx: &mut AgentContext<'_, Payload>, message: &Message<Payload>) {
        let now = ctx.time();
        let mut events = std::mem::take(&mut self.events_buf);
        self.net.receive_into(
            message,
            now,
            &self.config.faults,
            &mut self.network_rng,
            &mut events,
        );
        for &event in &events {
            match event {
                NetEvent::Accepted(conn, kind) => {
                    debug_assert!(
                        kind != ConnKind::Http,
                        "postgres serves only Postgres connections"
                    );
                    self.insert_backend(
                        conn,
                        Backend {
                            queue: VecDeque::new(),
                            running: None,
                            mode: Mode::Sql,
                        },
                    );
                }
                NetEvent::Closed(conn) => {
                    if self.remove_backend(conn).is_some() {
                        self.trim_dirty = true;
                    }
                }
                NetEvent::Data(conn, Unit::Query { id, op, a, b }) => {
                    if let Some(backend) = self.backend_mut(conn) {
                        backend.queue.push_back((id, op, a, b));
                        self.next_statement_at = 0;
                    }
                }
                NetEvent::Data(conn, Unit::Status { write, flush }) => {
                    self.on_status(conn, write, flush, now);
                }
                NetEvent::Data(_, other) => panic!("postgres got unexpected {other:?}"),
                NetEvent::Connected(_) => panic!("postgres opens no connections"),
            }
        }
        events.clear();
        self.events_buf = events;
    }

    fn on_tick(&mut self, ctx: &mut AgentContext<'_, Payload>) {
        let now = ctx.time();
        self.generate(now);
        self.inject_faults(now);
        self.run_statements(now);
        self.stream(now);
        self.trim_history();
        if now % TICKS_PER_SECOND == 0 {
            let oldest = self
                .table_slots
                .values()
                .fold(self.confirmed_flush, |oldest, lsn| oldest.min(*lsn));
            let mut recorder = self.recorder.borrow_mut();
            recorder
                .slot_lag_bytes
                .push((now, self.lsn.saturating_sub(oldest)));
            recorder.sample_lag(now, self.newest_commit);
        }
        if self.net.cross_region_bytes.to_bits() != self.last_reported_cross_region.to_bits() {
            self.last_reported_cross_region = self.net.cross_region_bytes;
            self.recorder.borrow_mut().cross_region_bytes[0] = self.net.cross_region_bytes;
        }
        self.net
            .pump(ctx, now, &self.config.faults, &mut self.network_rng);
    }
}
