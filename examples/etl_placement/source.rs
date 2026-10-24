//! The source Postgres in us-west-2: the application workload, WAL, the
//! logical replication slot and walsenders, COPY backends, and the ETL state
//! store (the `etl` schema in the source database, the standalone
//! replicator's default [etl]).

use crate::config::Config;
use crate::metrics::SharedRecorder;
use crate::net::{Endpoint, NetEvent, Peer};
use crate::rng::Rng;
use crate::time::{per_tick, ticks, to_u64, Tick, TICKS_PER_SECOND};
use crate::wire::{ConnId, ConnKind, Op, Unit, WalChunk};
use simul::agent::{Agent, AgentContext};
use simul::message::Message;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

pub(crate) const NAME: &str = "postgres";

/// Commits aggregate per table per millisecond, bounding history size while
/// keeping commit times within 1 ms.
const AGGREGATION_TICKS: Tick = 10;

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
    weighted_ticks: u128,
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
    open: Vec<OpenAggregate>,
    history: VecDeque<Record>,
    history_base: u64,
    confirmed_flush: u64,
    table_slots: BTreeMap<u64, u64>,
    stored_checkpoint: u64,
    backends: BTreeMap<ConnId, Backend>,
    next_resets: usize,
    copy_bytes_per_row: Vec<f64>,
    /// Commit time of the newest record visible to decoding.
    newest_commit: Tick,
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
        Self {
            open: vec![OpenAggregate::default(); config.workload.tables],
            config,
            recorder,
            net,
            network_rng,
            workload_rng,
            lsn: 1,
            burst_high: false,
            burst_switch_at: 0,
            history: VecDeque::new(),
            history_base: 0,
            confirmed_flush: 1,
            table_slots: BTreeMap::new(),
            stored_checkpoint: 0,
            backends: BTreeMap::new(),
            next_resets: 0,
            copy_bytes_per_row,
            newest_commit: 0,
        }
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

    fn generate(&mut self, now: Tick) {
        let workload = &self.config.workload;
        let multiplier = if let Some(burst) = workload.burst {
            if now >= self.burst_switch_at {
                self.burst_high = !self.burst_high && now > 0;
                let mean_low_s = burst.mean_high_s * (1.0 - burst.fraction) / burst.fraction;
                let mean = if self.burst_high {
                    burst.mean_high_s
                } else {
                    mean_low_s
                };
                self.burst_switch_at = now + ticks(self.workload_rng.exponential(mean)).max(1);
            }
            // Keep the long-run mean: high × f + low × (1 − f) = 1.
            let low = burst.fraction.mul_add(-burst.multiplier, 1.0) / (1.0 - burst.fraction);
            if self.burst_high {
                burst.multiplier
            } else {
                low
            }
        } else {
            1.0
        };
        let transactions_per_tick =
            per_tick(workload.rows_per_s / workload.mean_rows_per_transaction) * multiplier;
        let count = self.workload_rng.poisson(transactions_per_tick);
        for _ in 0..count {
            let rows = self
                .workload_rng
                .geometric(workload.mean_rows_per_transaction);
            let table = self.workload_rng.below(workload.tables as u64);
            self.lsn += to_u64(rows as f64 * self.config.source.wal_bytes_per_row) + 100;
            let slot = &mut self.open[usize::try_from(table).unwrap_or(0)];
            slot.rows += rows;
            slot.commits += 1;
            slot.end_lsn = self.lsn;
            slot.weighted_ticks += u128::from(rows) * u128::from(now);
        }
        if (now + 1) % AGGREGATION_TICKS == 0 {
            let mut records: Vec<Record> = self
                .open
                .iter_mut()
                .enumerate()
                .filter(|(_, slot)| slot.commits > 0)
                .map(|(table, slot)| {
                    let taken = std::mem::take(slot);
                    let mean = taken
                        .weighted_ticks
                        .div_euclid(u128::from(taken.rows.max(1)));
                    Record {
                        table: table as u64,
                        rows: taken.rows,
                        commits: taken.commits,
                        end_lsn: taken.end_lsn,
                        commit_tick: u64::try_from(mean).unwrap_or(now),
                    }
                })
                .collect();
            records.sort_by_key(|record| record.end_lsn);
            if let Some(newest) = records.iter().map(|record| record.commit_tick).max() {
                self.newest_commit = self.newest_commit.max(newest);
            }
            self.history.extend(records);
        }
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
                if let Some(backend) = self.backends.get_mut(&conn) {
                    backend.mode = Mode::WalSender(sender);
                }
                self.net.use_receiver_credit(conn);
                self.net.set_quantum(conn, AGGREGATION_TICKS);
                0
            }
            Op::StartCopy => {
                let table = usize::try_from(b).unwrap_or(0);
                let bytes_per_row = self.copy_bytes_per_row.get(table).copied().unwrap_or(100.0);
                if let Some(backend) = self.backends.get_mut(&conn) {
                    backend.mode = Mode::Copy(CopyOut {
                        rows_left: a,
                        bytes_left: a as f64 * bytes_per_row,
                        bytes_per_row,
                        carry: 0.0,
                    });
                }
                self.net.use_receiver_credit(conn);
                self.net.set_quantum(conn, AGGREGATION_TICKS);
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
        let conns: Vec<ConnId> = self.backends.keys().copied().collect();
        for conn in conns {
            while let Some(backend) = self.backends.get_mut(&conn) {
                match backend.running {
                    Some((id, op, a, b, done_at)) if done_at <= now => {
                        backend.running = None;
                        let value = self.execute(conn, op, a, b, now);
                        self.net.send(conn, Unit::Reply { id, value });
                    }
                    Some(_) => break,
                    None => {
                        let Some((id, op, a, b)) = backend.queue.pop_front() else {
                            break;
                        };
                        let service = ticks(self.service_time(op));
                        if let Some(backend) = self.backends.get_mut(&conn) {
                            backend.running = Some((id, op, a, b, now + service));
                        }
                    }
                }
            }
        }
    }

    fn stream(&mut self, now: Tick) {
        let rows_per_tick = per_tick(self.config.source.decode_rows_per_s);
        let send_buffer = self.net.send_buffer_bytes();
        let timeout = ticks(self.config.source.wal_sender_timeout_s);
        let row_wire = self
            .config
            .row_wire_bytes(self.config.workload.data_bytes_per_row);
        let transaction_wire = self.config.source.transaction_overhead_bytes;
        let flushed = self.flushed_lsn();
        let history_end = self.history_end();
        let mut terminated = Vec::new();
        for (&conn, backend) in &mut self.backends {
            match &mut backend.mode {
                Mode::WalSender(sender) => {
                    if now.saturating_sub(sender.last_client_message) >= timeout {
                        terminated.push(conn);
                        continue;
                    }
                    let mut budget = sender.decode_carry + rows_per_tick;
                    while self.net.buffered(conn) < send_buffer && budget >= 1.0 {
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
                        self.net.send(conn, Unit::Wal(chunk));
                        budget -= take as f64;
                        if completes {
                            sender.cursor += 1;
                            sender.partial_rows = 0;
                            sender.sent_lsn = record.end_lsn;
                        } else {
                            sender.partial_rows += take;
                        }
                    }
                    let caught_up = sender.cursor >= history_end;
                    sender.decode_carry = if caught_up {
                        0.0
                    } else {
                        budget.min(rows_per_tick)
                    };
                    // WalSndWaitForWal: when idle with the client behind, send a
                    // keepalive carrying the current end of WAL.
                    if caught_up && self.net.buffered(conn) < send_buffer {
                        sender.sent_lsn = sender.sent_lsn.max(flushed);
                        if sender.client_write < sender.sent_lsn
                            && sender.last_keepalive_lsn < sender.sent_lsn
                        {
                            sender.last_keepalive_lsn = sender.sent_lsn;
                            self.net.send(
                                conn,
                                Unit::Keepalive {
                                    wal_end: sender.sent_lsn,
                                    reply: false,
                                },
                            );
                        }
                    }
                    if !sender.ping_outstanding
                        && now.saturating_sub(sender.last_client_message) >= timeout.div_euclid(2)
                    {
                        sender.ping_outstanding = true;
                        self.net.send(
                            conn,
                            Unit::Keepalive {
                                wal_end: sender.sent_lsn,
                                reply: true,
                            },
                        );
                    }
                }
                Mode::Copy(copy) => {
                    let room = (send_buffer - self.net.buffered(conn)).max(0.0);
                    let rate = per_tick(self.config.source.copy_bytes_per_s);
                    let bytes = (copy.carry + rate).min(room).min(copy.bytes_left);
                    copy.carry = (copy.carry + rate - bytes).min(rate);
                    if bytes <= 0.0 {
                        continue;
                    }
                    let rows = to_u64((bytes / copy.bytes_per_row).round()).min(copy.rows_left);
                    copy.bytes_left -= bytes;
                    let done = copy.bytes_left <= 0.5;
                    let rows = if done { copy.rows_left } else { rows };
                    copy.rows_left -= rows;
                    self.net.send(
                        conn,
                        Unit::CopyData {
                            rows,
                            bytes: to_u64(bytes.round()),
                            done,
                        },
                    );
                    if done {
                        backend.mode = Mode::Sql;
                    }
                }
                Mode::Sql => {}
            }
        }
        for conn in terminated {
            self.backends.remove(&conn);
            self.net.abort(conn, now, &self.config.faults);
            self.recorder
                .borrow_mut()
                .event(now, "walsender exited: wal_sender_timeout");
        }
    }

    fn on_status(&mut self, conn: ConnId, write: u64, flush: u64, now: Tick) {
        let Some(Backend {
            mode: Mode::WalSender(sender),
            ..
        }) = self.backends.get_mut(&conn)
        else {
            return;
        };
        sender.last_client_message = now;
        sender.ping_outstanding = false;
        sender.client_write = sender.client_write.max(write);
        match sender.slot {
            Slot::Main => self.confirmed_flush = self.confirmed_flush.max(flush),
            Slot::Table(table) => {
                if let Some(lsn) = self.table_slots.get_mut(&table) {
                    *lsn = (*lsn).max(flush);
                }
            }
        }
    }

    /// Drops WAL every slot and sender has passed.
    fn trim_history(&mut self) {
        let mut keep_lsn = self.confirmed_flush;
        for lsn in self.table_slots.values() {
            keep_lsn = keep_lsn.min(*lsn);
        }
        let mut keep_index = self.history_end();
        for backend in self.backends.values() {
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
        let resets = &self.config.faults.replication_resets;
        while self.next_resets < resets.len() && resets[self.next_resets] <= now {
            self.next_resets += 1;
            let main: Vec<ConnId> = self
                .backends
                .iter()
                .filter(|(_, backend)| {
                    matches!(&backend.mode, Mode::WalSender(sender) if sender.slot == Slot::Main)
                })
                .map(|(conn, _)| *conn)
                .collect();
            for conn in main {
                self.backends.remove(&conn);
                self.net.abort(conn, now, &self.config.faults);
            }
            self.recorder
                .borrow_mut()
                .event(now, "source reset the replication connection");
        }
    }
}

impl Agent for PostgresAgent {
    fn on_message(&mut self, ctx: &mut AgentContext, message: &Message) {
        let now = ctx.time();
        let events = self
            .net
            .receive(message, now, &self.config.faults, &mut self.network_rng);
        for event in events {
            match event {
                NetEvent::Accepted(conn, kind) => {
                    debug_assert!(
                        kind != ConnKind::Http,
                        "postgres serves only Postgres connections"
                    );
                    self.backends.insert(
                        conn,
                        Backend {
                            queue: VecDeque::new(),
                            running: None,
                            mode: Mode::Sql,
                        },
                    );
                }
                NetEvent::Closed(conn) => {
                    self.backends.remove(&conn);
                }
                NetEvent::Data(conn, Unit::Query { id, op, a, b }) => {
                    if let Some(backend) = self.backends.get_mut(&conn) {
                        backend.queue.push_back((id, op, a, b));
                    }
                }
                NetEvent::Data(conn, Unit::Status { write, flush }) => {
                    self.on_status(conn, write, flush, now);
                }
                NetEvent::Data(_, other) => panic!("postgres got unexpected {other:?}"),
                NetEvent::Connected(_) => panic!("postgres opens no connections"),
            }
        }
    }

    fn on_tick(&mut self, ctx: &mut AgentContext) {
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
        self.recorder.borrow_mut().cross_region_bytes[0] = self.net.cross_region_bytes;
        self.net
            .pump(ctx, now, &self.config.faults, &mut self.network_rng);
    }
}
