//! The ETL replicator: pipeline startup, the apply worker, table-sync workers
//! with parallel copy, the state-store pool, and the destination client.
//!
//! The control flow mirrors supabase/etl @ 2a4a0f8a (see `config.rs`):
//! - Apply: one batch at the destination and one filling. A batch flushes at
//!   the decoded-byte target or `max_fill_ms`; if the destination still holds
//!   the previous batch, reading stops until it is durable.
//! - After each durable batch with a commit, the loop awaits a 4-statement
//!   checkpoint upsert in the state store before sending the next batch.
//! - Feedback answers walsender keepalives and is otherwise a heartbeat every
//!   0.6 × `wal_sender_timeout`; the flush LSN is the durable checkpoint.
//! - Copy: per table, ctid partitions over up to 4 connections; each partition
//!   alternates between reading a batch and writing it.
//! - Catch-up: the apply worker blocks while a table's own slot catches up.
//! - Errors: a fixed retry delay; restart from max(stored checkpoint, slot
//!   confirmed flush).

use crate::config::{Config, Durability};
use crate::metrics::{RowGroup, SharedRecorder};
use crate::net::{send_tracked, Endpoint, NetEvent, Peer};
use crate::rng::Rng;
use crate::time::{per_tick, secs, ticks, to_u64, Tick};
use crate::wire::{ConnId, ConnKind, HttpKind, Op, Payload, Unit, WalChunk};
use crate::{sink, source};
use simul::agent::{Agent, AgentContext};
use simul::message::Message;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

pub(crate) const NAME: &str = "etl";

/// Table states written to the store.
const STATE_DATA_SYNC: u64 = 1;
const STATE_FINISHED_COPY: u64 = 2;
const STATE_SYNC_DONE: u64 = 3;
const STATE_READY: u64 = 4;

/// Credits are sent once this many bytes were read, or after a short pause,
/// like TCP window updates.
const CREDIT_STEP_BYTES: f64 = 256.0 * 1024.0;
const CREDIT_IDLE_TICKS: Tick = 100;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Owner {
    Boot,
    Apply,
    Worker(usize),
    Child(usize, usize),
    Catchup(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LoopId {
    Apply,
    Catchup(usize),
}

impl LoopId {
    const fn owner(self) -> Owner {
        match self {
            Self::Apply => Owner::Apply,
            Self::Catchup(worker) => Owner::Catchup(worker),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StoreOp {
    Checkpoint(u64),
    TableState(u64, u64),
    LoadCache,
    Schema,
}

impl StoreOp {
    /// Statements per store call; each is a separate round trip with sqlx [etl].
    fn statements(self) -> VecDeque<(Op, u64, u64)> {
        match self {
            Self::Checkpoint(lsn) => VecDeque::from([
                (Op::Begin, 0, 0),
                (Op::AdvisoryLock, 0, 0),
                (Op::UpsertCheckpoint, lsn, 0),
                (Op::Commit, 0, 0),
            ]),
            Self::TableState(table, state) => VecDeque::from([
                (Op::Begin, 0, 0),
                (Op::SetTableState, table, state),
                (Op::Commit, 0, 0),
            ]),
            Self::LoadCache => VecDeque::from([
                (Op::Select, 0, 0),
                (Op::Select, 0, 0),
                (Op::Select, 0, 0),
                (Op::LoadCheckpoint, 0, 0),
            ]),
            Self::Schema => VecDeque::from([
                (Op::Begin, 0, 0),
                (Op::StoreSchema, 0, 0),
                (Op::Commit, 0, 0),
            ]),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Step {
    /// Opens the owner's connection to the source.
    Connect(ConnKind),
    Sql(Op, u64, u64),
    /// A statement whose reply value the script keeps.
    Capture(Op, u64, u64),
    /// `START_REPLICATION` from max(stored checkpoint, captured confirmed flush).
    StartReplication,
    Store(StoreOp),
    CaptureStore(StoreOp),
    /// Drop and recreate the destination table.
    Ddl,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Wait {
    Ready,
    Conn(ConnId),
    Reply(u64),
    Store(u64),
    Write(u64),
}

#[derive(Clone, Debug)]
struct Script {
    steps: VecDeque<Step>,
    wait: Wait,
    capture_next: bool,
    captured: u64,
}

impl Script {
    fn new(steps: impl IntoIterator<Item = Step>) -> Self {
        Self {
            steps: steps.into_iter().collect(),
            wait: Wait::Ready,
            capture_next: false,
            captured: 0,
        }
    }

    fn complete(&mut self, wait: Wait, value: u64) -> bool {
        if self.wait != wait {
            return false;
        }
        if self.capture_next {
            self.captured = value;
        }
        self.wait = Wait::Ready;
        true
    }
}

#[derive(Clone, Debug, Default)]
struct Batch {
    first_at: Option<Tick>,
    decoded: f64,
    request_bytes: f64,
    rows: u64,
    commit_lsn: u64,
    tables: u64,
    rows_by_table: Vec<RowGroup>,
}

/// One replication stream consumer: the apply loop or a catch-up loop.
#[derive(Clone, Debug, Default)]
struct StreamLoop {
    conn: Option<ConnId>,
    unread: VecDeque<Unit>,
    unread_bytes: f64,
    consumed: f64,
    credited: f64,
    last_credit_at: Tick,
    received_lsn: u64,
    carry: f64,
    filling: Batch,
    queued: Option<Batch>,
    inflight: Option<(u64, Batch, Tick)>,
    checkpoint: Option<(u64, u64, Tick)>,
    last_flush_lsn: u64,
    paused: bool,
    last_status_at: Tick,
    last_status_flush: u64,
    /// Catch-up loops apply one table up to `until`.
    table: Option<u64>,
    until: u64,
}

impl StreamLoop {
    const fn quiescent(&self) -> bool {
        self.filling.rows == 0
            && self.queued.is_none()
            && self.inflight.is_none()
            && self.checkpoint.is_none()
    }

    /// The flush LSN reported to Postgres [etl `checkpoint_lsn`].
    const fn checkpoint_lsn(&self) -> u64 {
        if self.quiescent() {
            self.received_lsn
        } else {
            self.last_flush_lsn
        }
    }

    const fn reached(&self) -> bool {
        self.table.is_some() && self.received_lsn >= self.until
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum ApplyState {
    #[default]
    Idle,
    Starting,
    Streaming,
    Retrying(Tick),
    Failed,
}

#[derive(Clone, Debug, Default)]
struct Apply {
    state: ApplyState,
    script: Option<Script>,
    lp: StreamLoop,
    stored_checkpoint: u64,
    attempts: u32,
    blocked_by: Option<usize>,
}

#[derive(Clone, Debug, Default)]
struct Child {
    conn: Option<ConnId>,
    script: Option<Script>,
    streaming: bool,
    finished: bool,
    unread: VecDeque<(u64, f64, bool)>,
    unread_bytes: f64,
    part_done: bool,
    consumed: f64,
    credited: f64,
    last_credit_at: Tick,
    carry: f64,
    batch_rows: u64,
    batch_decoded: f64,
    batch_wire: f64,
    batch_first: Option<Tick>,
    writing: Option<(u64, f64)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Setup,
    Copying,
    Finishing,
    Catchup,
    Cleanup,
}

#[derive(Clone, Debug)]
struct Worker {
    table: usize,
    phase: Phase,
    repl: Option<ConnId>,
    script: Option<Script>,
    catchup_script: Option<Script>,
    snapshot_lsn: u64,
    parts: VecDeque<u64>,
    children: Vec<Child>,
    lp: StreamLoop,
}

#[derive(Clone, Debug)]
struct StoreJob {
    id: u64,
    owner: Owner,
    statements: VecDeque<(Op, u64, u64)>,
    value: u64,
}

#[derive(Clone, Debug)]
struct StoreConn {
    conn: ConnId,
    open: bool,
    job: Option<StoreJob>,
    idle_since: Tick,
}

#[derive(Clone, Copy, Debug)]
struct Request {
    id: u64,
    write: u64,
    bytes: u64,
    kind: HttpKind,
    of: u64,
}

#[derive(Clone, Debug)]
struct DestConn {
    conn: ConnId,
    open: bool,
    request: Option<Request>,
    idle_since: Tick,
}

#[derive(Clone, Copy, Debug)]
struct Write {
    owner: Owner,
    parts_left: u64,
}

#[derive(Clone, Copy, Debug)]
enum Completion {
    Store { owner: Owner, job: u64, value: u64 },
    Write { owner: Owner, write: u64 },
}

#[derive(Clone, Debug)]
pub(crate) struct EtlAgent {
    config: Rc<Config>,
    recorder: SharedRecorder,
    net: Endpoint,
    rng: Rng,
    postgres: usize,
    destination: usize,
    up: bool,
    restart_at: Option<Tick>,
    next_crash: usize,
    boot_conn: Option<ConnId>,
    boot_script: Option<Script>,
    apply: Apply,
    workers: Vec<Option<Worker>>,
    sync_queue: VecDeque<usize>,
    /// Apply handles a table's commits after this LSN once it is ready.
    apply_from: Vec<Option<u64>>,
    sync_done: Vec<Option<u64>>,
    conn_owner: BTreeMap<ConnId, Owner>,
    query_owner: BTreeMap<u64, Owner>,
    next_id: u64,
    store_conns: Vec<StoreConn>,
    store_queue: VecDeque<StoreJob>,
    dest_conns: Vec<DestConn>,
    dest_queue: VecDeque<Request>,
    writes: BTreeMap<u64, Write>,
    polls: Vec<(Tick, Request)>,
    completions: Vec<Completion>,
}

impl EtlAgent {
    pub(crate) fn new(
        config: Rc<Config>,
        recorder: SharedRecorder,
        rng: Rng,
        postgres: Peer,
        destination: Peer,
    ) -> Self {
        let net = Endpoint::new(
            1,
            config.tcp.clone(),
            config.etl_nic,
            vec![postgres, destination],
        );
        let mut agent = Self {
            postgres: net.peer_index(source::NAME),
            destination: net.peer_index(sink::NAME),
            net,
            rng,
            up: false,
            restart_at: Some(0),
            next_crash: 0,
            boot_conn: None,
            boot_script: None,
            apply: Apply::default(),
            workers: Vec::new(),
            sync_queue: VecDeque::new(),
            apply_from: Vec::new(),
            sync_done: Vec::new(),
            conn_owner: BTreeMap::new(),
            query_owner: BTreeMap::new(),
            next_id: 1,
            store_conns: Vec::new(),
            store_queue: VecDeque::new(),
            dest_conns: Vec::new(),
            dest_queue: VecDeque::new(),
            writes: BTreeMap::new(),
            polls: Vec::new(),
            completions: Vec::new(),
            config,
            recorder,
        };
        agent.reset_tables();
        agent
    }

    fn reset_tables(&mut self) {
        let tables = self.config.workload.tables;
        let copying = !self.config.copy_tables.is_empty();
        self.apply_from = vec![if copying { None } else { Some(0) }; tables];
        self.sync_done = vec![None; tables];
        self.sync_queue = if copying {
            (0..self.config.copy_tables.len().min(tables)).collect()
        } else {
            VecDeque::new()
        };
        self.workers = vec![None; self.config.etl.table_sync_workers];
    }

    const fn id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // ---- lifecycle -------------------------------------------------------

    fn boot(&mut self, now: Tick) {
        self.up = true;
        self.recorder.borrow_mut().event(now, "etl starting");
        self.boot_script = Some(Script::new([
            Step::Connect(ConnKind::Replication),
            Step::Sql(Op::Migrations, 0, 0),
            Step::CaptureStore(StoreOp::LoadCache),
            Step::Sql(Op::Select, 0, 0),
            Step::Sql(Op::Select, 0, 0),
        ]));
        self.advance(Owner::Boot, now);
    }

    fn crash(&mut self, now: Tick) {
        self.recorder.borrow_mut().event(now, "etl process crashed");
        self.net.abort_all(now, &self.config.faults);
        self.up = false;
        self.restart_at = Some(now + ticks(self.config.etl.restart_delay_s));
        self.boot_conn = None;
        self.boot_script = None;
        self.apply = Apply::default();
        self.conn_owner.clear();
        self.query_owner.clear();
        self.store_conns.clear();
        self.store_queue.clear();
        self.dest_conns.clear();
        self.dest_queue.clear();
        self.writes.clear();
        self.polls.clear();
        self.completions.clear();
        self.reset_tables();
    }

    // ---- scripts -----------------------------------------------------------

    fn script_slot(&mut self, owner: Owner) -> Option<&mut Option<Script>> {
        match owner {
            Owner::Boot => Some(&mut self.boot_script),
            Owner::Apply => Some(&mut self.apply.script),
            Owner::Worker(worker) => self.workers[worker].as_mut().map(|entry| &mut entry.script),
            Owner::Child(worker, child) => self.workers[worker]
                .as_mut()
                .and_then(|entry| entry.children.get_mut(child))
                .map(|entry| &mut entry.script),
            Owner::Catchup(worker) => self.workers[worker]
                .as_mut()
                .map(|entry| &mut entry.catchup_script),
        }
    }

    fn conn_slot(&mut self, owner: Owner) -> Option<&mut Option<ConnId>> {
        match owner {
            Owner::Boot => Some(&mut self.boot_conn),
            Owner::Apply => Some(&mut self.apply.lp.conn),
            Owner::Worker(worker) => self.workers[worker].as_mut().map(|entry| &mut entry.repl),
            Owner::Child(worker, child) => self.workers[worker]
                .as_mut()
                .and_then(|entry| entry.children.get_mut(child))
                .map(|entry| &mut entry.conn),
            Owner::Catchup(worker) => self.workers[worker]
                .as_mut()
                .map(|entry| &mut entry.lp.conn),
        }
    }

    fn owner_conn(&mut self, owner: Owner) -> Option<ConnId> {
        self.conn_slot(owner).and_then(|slot| *slot)
    }

    /// Runs the owner's script until it waits or finishes.
    fn advance(&mut self, owner: Owner, now: Tick) {
        loop {
            let Some(slot) = self.script_slot(owner) else {
                return;
            };
            let Some(script) = slot.as_mut() else {
                return;
            };
            if script.wait != Wait::Ready {
                return;
            }
            let Some(step) = script.steps.pop_front() else {
                let captured = script.captured;
                *slot = None;
                self.script_done(owner, captured, now);
                return;
            };
            let captured = script.captured;
            let wait = self.start_step(owner, step, captured, now);
            if let Some(Some(script)) = self.script_slot(owner) {
                script.wait = wait;
                script.capture_next = matches!(step, Step::Capture(..) | Step::CaptureStore(_));
            }
        }
    }

    fn start_step(&mut self, owner: Owner, step: Step, captured: u64, now: Tick) -> Wait {
        match step {
            Step::Connect(kind) => {
                let conn = self.net.connect(
                    self.postgres,
                    kind,
                    self.config.postgres_handshake_round_trips,
                    now,
                    &mut self.rng,
                );
                if let Some(slot) = self.conn_slot(owner) {
                    *slot = Some(conn);
                }
                self.conn_owner.insert(conn, owner);
                Wait::Conn(conn)
            }
            Step::Sql(op, a, b) | Step::Capture(op, a, b) => self.query(owner, op, a, b),
            Step::StartReplication => {
                let from = self.apply.stored_checkpoint.max(captured);
                self.apply.lp.received_lsn = from;
                self.apply.lp.last_flush_lsn = from;
                self.query(owner, Op::StartReplication, from, 0)
            }
            Step::Store(op) | Step::CaptureStore(op) => Wait::Store(self.submit_store(owner, op)),
            Step::Ddl => Wait::Write(self.submit_write(owner, &[2000.0], HttpKind::Ddl)),
        }
    }

    fn query(&mut self, owner: Owner, op: Op, a: u64, b: u64) -> Wait {
        let id = self.id();
        let Some(conn) = self.owner_conn(owner) else {
            panic!("{owner:?} sent {op:?} without a connection");
        };
        send_tracked(&mut self.net, conn, Unit::Query { id, op, a, b });
        self.query_owner.insert(id, owner);
        Wait::Reply(id)
    }

    fn script_done(&mut self, owner: Owner, captured: u64, now: Tick) {
        match owner {
            Owner::Boot => {
                self.apply.stored_checkpoint = captured;
                self.start_apply(now);
                self.recorder
                    .borrow_mut()
                    .event(now, "etl pipeline started");
            }
            Owner::Apply => {
                self.apply.state = ApplyState::Streaming;
                self.recorder
                    .borrow_mut()
                    .event(now, "apply worker streaming");
            }
            Owner::Worker(worker) => self.worker_step_done(worker, captured, now),
            Owner::Child(worker, child) => self.next_partition(worker, child, now),
            Owner::Catchup(_) => {}
        }
    }

    // ---- apply worker -------------------------------------------------------

    fn start_apply(&mut self, now: Tick) {
        self.apply.state = ApplyState::Starting;
        self.apply.lp = StreamLoop::default();
        self.apply.script = Some(Script::new([
            Step::Connect(ConnKind::Replication),
            Step::Capture(Op::GetSlot, 0, 0),
            Step::Sql(Op::Select, 0, 0),
            Step::StartReplication,
        ]));
        self.advance(Owner::Apply, now);
    }

    fn apply_failed(&mut self, now: Tick, reason: &str) {
        if let Some(conn) = self.apply.lp.conn.take() {
            self.conn_owner.remove(&conn);
            self.net.abort(conn, now, &self.config.faults);
        }
        self.apply.lp = StreamLoop::default();
        self.apply.script = None;
        self.apply.attempts += 1;
        let mut recorder = self.recorder.borrow_mut();
        if self.apply.attempts > self.config.etl.retry_attempts {
            self.apply.state = ApplyState::Failed;
            recorder.event(now, format!("apply worker failed permanently: {reason}"));
        } else {
            self.apply.state = ApplyState::Retrying(now + ticks(self.config.etl.retry_delay_s));
            recorder.event(
                now,
                format!("apply worker error, retrying in 10 s: {reason}"),
            );
        }
    }

    fn take_loop(&mut self, id: LoopId) -> Option<StreamLoop> {
        match id {
            LoopId::Apply => Some(std::mem::take(&mut self.apply.lp)),
            LoopId::Catchup(worker) => self.workers[worker]
                .as_mut()
                .map(|entry| std::mem::take(&mut entry.lp)),
        }
    }

    fn put_loop(&mut self, id: LoopId, lp: StreamLoop) {
        match id {
            LoopId::Apply => self.apply.lp = lp,
            LoopId::Catchup(worker) => {
                if let Some(entry) = self.workers[worker].as_mut() {
                    entry.lp = lp;
                }
            }
        }
    }

    fn batch_slots(&self) -> usize {
        let mut slots = 2;
        for worker in self.workers.iter().flatten() {
            slots += worker
                .children
                .iter()
                .filter(|child| child.streaming && !child.finished)
                .count();
            if worker.phase == Phase::Catchup {
                slots += 2;
            }
        }
        slots
    }

    fn relevant(&self, id: LoopId, lp: &StreamLoop, chunk: &WalChunk) -> bool {
        match id {
            LoopId::Apply => usize::try_from(chunk.table)
                .ok()
                .and_then(|table| self.apply_from.get(table).copied().flatten())
                .is_some_and(|from| chunk.end_lsn == 0 || chunk.end_lsn > from),
            LoopId::Catchup(_) => {
                lp.table == Some(chunk.table) && (chunk.end_lsn == 0 || chunk.end_lsn <= lp.until)
            }
        }
    }

    fn run_loop(&mut self, id: LoopId, now: Tick) {
        let Some(mut lp) = self.take_loop(id) else {
            return;
        };
        let target = self.config.batch_target_bytes(self.batch_slots());
        let rows_per_tick = per_tick(self.config.etl.decode_rows_per_s);
        let data_bytes = self.config.workload.data_bytes_per_row;
        let decoded_per_row = self.config.decoded_bytes(data_bytes);
        let request_per_row = self.config.request_bytes(data_bytes);
        let streaming = match id {
            LoopId::Apply => {
                self.apply.state == ApplyState::Streaming && self.apply.blocked_by.is_none()
            }
            // The connection exists from the handshake on; the stream starts
            // once the catch-up script's START_REPLICATION has returned.
            LoopId::Catchup(worker) => {
                lp.conn.is_some()
                    && self.workers[worker]
                        .as_ref()
                        .is_some_and(|entry| entry.catchup_script.is_none())
            }
        };
        let reading = streaming && lp.checkpoint.is_none() && !lp.paused && !lp.reached();
        if reading {
            let mut budget = lp.carry + rows_per_tick;
            // A negative carry is decode debt from an oversized chunk.
            let mut next = if budget > 0.0 {
                lp.unread.pop_front()
            } else {
                None
            };
            while let Some(unit) = next {
                lp.unread_bytes = (lp.unread_bytes - unit.wire_bytes() as f64).max(0.0);
                lp.consumed += unit.wire_bytes() as f64;
                match unit {
                    Unit::Keepalive { wal_end, reply } => {
                        lp.received_lsn = lp.received_lsn.max(wal_end);
                        self.send_status(&mut lp, reply, now);
                    }
                    Unit::Wal(chunk) => {
                        budget -= chunk.rows as f64;
                        if chunk.commits > 0 {
                            lp.received_lsn = lp.received_lsn.max(chunk.end_lsn);
                        }
                        if self.relevant(id, &lp, &chunk) {
                            let batch = &mut lp.filling;
                            batch.first_at.get_or_insert(now);
                            batch.rows += chunk.rows;
                            batch.decoded =
                                (chunk.rows as f64).mul_add(decoded_per_row, batch.decoded);
                            batch.request_bytes =
                                (chunk.rows as f64).mul_add(request_per_row, batch.request_bytes);
                            batch.tables |= 1 << chunk.table.min(63);
                            if chunk.commits > 0 {
                                batch.commit_lsn = batch.commit_lsn.max(chunk.end_lsn);
                            }
                            batch.rows_by_table.push(RowGroup {
                                table: usize::try_from(chunk.table).unwrap_or_else(|_| {
                                    panic!("table ids come from a usize table count")
                                }),
                                commit_tick: chunk.commit_tick,
                                rows: chunk.rows,
                                lsn: chunk.end_lsn,
                            });
                        }
                        if lp.filling.decoded >= target {
                            self.request_flush(id, &mut lp, now);
                        }
                    }
                    other @ (Unit::Syn { .. }
                    | Unit::Ping
                    | Unit::Pong
                    | Unit::Close
                    | Unit::Reset
                    | Unit::Credit { .. }
                    | Unit::Query { .. }
                    | Unit::Reply { .. }
                    | Unit::Status { .. }
                    | Unit::CopyData { .. }
                    | Unit::HttpRequest { .. }
                    | Unit::HttpResponse { .. }) => {
                        panic!("replication stream carried {other:?}")
                    }
                }
                next = if budget > 0.0 && !lp.paused && !lp.reached() {
                    lp.unread.pop_front()
                } else {
                    None
                };
            }
            lp.carry = if lp.unread.is_empty() {
                budget.min(0.0)
            } else {
                budget.min(rows_per_tick)
            };
        }
        if lp.checkpoint.is_none() {
            let deadline = ticks(self.config.etl.max_fill_ms / 1000.0);
            let due = lp
                .filling
                .first_at
                .is_some_and(|first| now >= first + deadline);
            if due || (lp.reached() && lp.filling.rows > 0) {
                self.request_flush(id, &mut lp, now);
            }
        }
        if let Some(conn) = lp.conn {
            let unread = lp.consumed - lp.credited;
            if unread >= CREDIT_STEP_BYTES
                || (unread > 0.0 && now >= lp.last_credit_at + CREDIT_IDLE_TICKS)
            {
                send_tracked(
                    &mut self.net,
                    conn,
                    Unit::Credit {
                        consumed: to_u64(lp.consumed),
                    },
                );
                lp.credited = lp.consumed;
                lp.last_credit_at = now;
            }
            let heartbeat = ticks(0.6 * self.config.source.wal_sender_timeout_s);
            if streaming && now >= lp.last_status_at + heartbeat {
                self.send_status(&mut lp, true, now);
            }
        }
        if id == LoopId::Apply {
            let mut recorder = self.recorder.borrow_mut();
            if lp.paused || lp.checkpoint.is_some() {
                recorder.paused_ticks += 1;
            }
            if self.apply.blocked_by.is_some() {
                recorder.blocked_ticks += 1;
            }
        }
        self.put_loop(id, lp);
        if let LoopId::Catchup(worker) = id {
            self.check_catchup_done(worker, now);
        }
    }

    fn send_status(&mut self, lp: &mut StreamLoop, forced: bool, now: Tick) {
        let Some(conn) = lp.conn else {
            return;
        };
        let flush = lp.checkpoint_lsn();
        let debounce = ticks(self.config.etl.status_debounce_s);
        if !forced && flush == lp.last_status_flush && now < lp.last_status_at + debounce {
            return;
        }
        send_tracked(
            &mut self.net,
            conn,
            Unit::Status {
                write: lp.received_lsn,
                flush,
            },
        );
        lp.last_status_at = now;
        lp.last_status_flush = flush;
    }

    fn request_flush(&mut self, id: LoopId, lp: &mut StreamLoop, now: Tick) {
        if lp.filling.rows == 0 {
            lp.filling = Batch::default();
            return;
        }
        if lp.inflight.is_none() && lp.checkpoint.is_none() && lp.queued.is_none() {
            let batch = std::mem::take(&mut lp.filling);
            self.dispatch(id, lp, batch, now);
        } else if lp.queued.is_none() {
            lp.queued = Some(std::mem::take(&mut lp.filling));
            lp.paused = true;
        }
    }

    fn dispatch(&mut self, id: LoopId, lp: &mut StreamLoop, mut batch: Batch, now: Tick) {
        let parts = u64::from(batch.tables.count_ones())
            .min(self.config.sink.max_parallel_requests)
            .max(1);
        let each = batch.request_bytes / parts as f64;
        let sizes = vec![each; usize::try_from(parts).unwrap_or(1)];
        let write = self.submit_write(id.owner(), &sizes, HttpKind::Insert);
        self.recorder.borrow_mut().register_write(
            write,
            parts,
            std::mem::take(&mut batch.rows_by_table),
        );
        lp.inflight = Some((write, batch, now));
    }

    fn on_loop_write(&mut self, id: LoopId, write: u64, now: Tick) {
        let Some(mut lp) = self.take_loop(id) else {
            return;
        };
        match lp.inflight.take() {
            Some((inflight, batch, started)) if inflight == write => {
                {
                    let mut recorder = self.recorder.borrow_mut();
                    recorder.write_seconds.record(secs(now - started), 1);
                    recorder.batches += 1;
                    recorder.batch_decoded_bytes += batch.decoded;
                }
                if batch.commit_lsn > 0 {
                    lp.last_flush_lsn = lp.last_flush_lsn.max(batch.commit_lsn);
                    let job = self.submit_store(id.owner(), StoreOp::Checkpoint(batch.commit_lsn));
                    lp.checkpoint = Some((job, batch.commit_lsn, now));
                } else {
                    self.after_flush(id, &mut lp, now);
                }
            }
            other => lp.inflight = other,
        }
        self.put_loop(id, lp);
    }

    fn on_loop_checkpoint(&mut self, id: LoopId, job: u64, now: Tick) {
        let Some(mut lp) = self.take_loop(id) else {
            return;
        };
        match lp.checkpoint.take() {
            Some((pending, lsn, started)) if pending == job => {
                self.recorder
                    .borrow_mut()
                    .checkpoint_seconds
                    .record(secs(now - started), 1);
                let promotion = if id == LoopId::Apply {
                    self.apply.stored_checkpoint = self.apply.stored_checkpoint.max(lsn);
                    self.promote_ready(now)
                } else {
                    None
                };
                match promotion {
                    Some(next) => lp.checkpoint = Some((next, lsn, now)),
                    None => self.after_flush(id, &mut lp, now),
                }
            }
            other => lp.checkpoint = other,
        }
        self.put_loop(id, lp);
    }

    /// Moves tables from `SyncDone` to `Ready` once the apply checkpoint has
    /// passed their catch-up LSN; returns the store job to await.
    fn promote_ready(&mut self, now: Tick) -> Option<u64> {
        let stored = self.apply.stored_checkpoint;
        let table = self
            .sync_done
            .iter()
            .position(|done| done.is_some_and(|lsn| lsn <= stored))?;
        self.sync_done[table] = None;
        self.recorder.borrow_mut().tables[table].ready = Some(now);
        if self
            .recorder
            .borrow()
            .tables
            .iter()
            .all(|timeline| timeline.ready.is_some())
        {
            self.recorder.borrow_mut().copy_done_at = Some(now);
        }
        Some(self.submit_store(Owner::Apply, StoreOp::TableState(table as u64, STATE_READY)))
    }

    fn after_flush(&mut self, id: LoopId, lp: &mut StreamLoop, now: Tick) {
        if let Some(batch) = lp.queued.take() {
            self.dispatch(id, lp, batch, now);
        }
        lp.paused = false;
    }

    // ---- table sync -------------------------------------------------------

    fn start_workers(&mut self, now: Tick) {
        if self.apply.state != ApplyState::Streaming {
            return;
        }
        for worker in 0..self.workers.len() {
            if self.workers[worker].is_some() {
                continue;
            }
            let Some(table) = self.sync_queue.pop_front() else {
                return;
            };
            self.recorder.borrow_mut().tables[table].copy_started = Some(now);
            self.workers[worker] = Some(Worker {
                table,
                phase: Phase::Setup,
                repl: None,
                script: Some(Script::new([
                    Step::Ddl,
                    Step::Store(StoreOp::TableState(table as u64, STATE_DATA_SYNC)),
                    Step::Connect(ConnKind::Replication),
                    Step::Sql(Op::Select, 0, 0),
                    Step::Sql(Op::Begin, 0, 0),
                    Step::Sql(Op::LockTable, 0, 0),
                    Step::Capture(Op::CreateSlot, table as u64, 0),
                    Step::Sql(Op::Select, 0, 0),
                    Step::Sql(Op::Select, 0, 0),
                    Step::Store(StoreOp::Schema),
                    Step::Sql(Op::Select, 0, 0),
                    Step::Sql(Op::ExportSnapshot, 0, 0),
                ])),
                catchup_script: None,
                snapshot_lsn: 0,
                parts: VecDeque::new(),
                children: Vec::new(),
                lp: StreamLoop::default(),
            });
            self.advance(Owner::Worker(worker), now);
        }
    }

    fn worker_step_done(&mut self, worker: usize, captured: u64, now: Tick) {
        let Some(entry) = self.workers[worker].as_mut() else {
            return;
        };
        match entry.phase {
            Phase::Setup => {
                entry.snapshot_lsn = captured;
                let spec = self.config.copy_tables[entry.table];
                let connections = self.config.etl.copy_connections_per_table as u64;
                let partitions = if connections <= 1 {
                    1
                } else {
                    (connections * 4)
                        .max(to_u64(
                            (spec.rows as f64 / self.config.etl.rows_per_partition).ceil(),
                        ))
                        .clamp(1, self.config.etl.max_partitions)
                };
                let base = spec.rows.div_euclid(partitions);
                let extra = spec.rows.rem_euclid(partitions);
                entry.parts = (0..partitions)
                    .map(|index| base + u64::from(index < extra))
                    .collect();
                let children = usize::try_from(connections.min(partitions)).unwrap_or(1);
                entry.children = (0..children)
                    .map(|_| Child {
                        script: Some(Script::new([
                            Step::Connect(ConnKind::Sql),
                            Step::Sql(Op::Begin, 0, 0),
                            Step::Sql(Op::SetSnapshot, 0, 0),
                        ])),
                        ..Child::default()
                    })
                    .collect();
                entry.phase = Phase::Copying;
                for child in 0..children {
                    self.advance(Owner::Child(worker, child), now);
                }
            }
            Phase::Finishing => {
                let until = entry.snapshot_lsn.max(self.apply.lp.received_lsn);
                entry.phase = Phase::Catchup;
                entry.lp = StreamLoop {
                    table: Some(entry.table as u64),
                    until,
                    received_lsn: entry.snapshot_lsn,
                    last_flush_lsn: entry.snapshot_lsn,
                    ..StreamLoop::default()
                };
                entry.catchup_script = Some(Script::new([
                    Step::Connect(ConnKind::Replication),
                    Step::Sql(Op::StartCatchup, entry.table as u64, 0),
                ]));
                self.apply.blocked_by = Some(worker);
                self.advance(Owner::Catchup(worker), now);
            }
            Phase::Cleanup => {
                let table = entry.table;
                let until = entry.lp.until;
                let repl = entry.repl.take();
                self.workers[worker] = None;
                if let Some(conn) = repl {
                    self.conn_owner.remove(&conn);
                    self.net.close(conn);
                }
                self.apply.blocked_by = None;
                self.apply_from[table] = Some(until);
                self.sync_done[table] = Some(until);
                self.recorder.borrow_mut().tables[table].catchup_finished = Some(now);
                self.start_workers(now);
            }
            Phase::Copying | Phase::Catchup => {}
        }
    }

    fn next_partition(&mut self, worker: usize, child: usize, now: Tick) {
        let Some(entry) = self.workers[worker].as_mut() else {
            return;
        };
        let table = entry.table as u64;
        let next = entry.parts.pop_front();
        let Some(state) = entry.children.get_mut(child) else {
            return;
        };
        if let Some(rows) = next {
            state.streaming = true;
            state.part_done = false;
            let conn = state.conn;
            let id = self.id();
            if let Some(conn) = conn {
                send_tracked(
                    &mut self.net,
                    conn,
                    Unit::Query {
                        id,
                        op: Op::StartCopy,
                        a: rows,
                        b: table,
                    },
                );
                self.query_owner.insert(id, Owner::Child(worker, child));
            }
        } else {
            state.streaming = false;
            state.finished = true;
            if let Some(conn) = state.conn.take() {
                self.conn_owner.remove(&conn);
                self.net.close(conn);
            }
            let done = entry.children.iter().all(|other| other.finished);
            if done {
                self.recorder.borrow_mut().tables[entry.table].copy_finished = Some(now);
                entry.phase = Phase::Finishing;
                entry.script = Some(Script::new([
                    Step::Sql(Op::CommitSlotTransaction, 0, 0),
                    Step::Store(StoreOp::TableState(table, STATE_FINISHED_COPY)),
                ]));
                self.advance(Owner::Worker(worker), now);
            }
        }
    }

    fn run_child(&mut self, worker: usize, child: usize, now: Tick) {
        let slots = self.batch_slots();
        let target = self.config.batch_target_bytes(slots);
        let parse_per_tick = per_tick(self.config.etl.copy_parse_bytes_per_s);
        let deadline = ticks(self.config.etl.max_fill_ms / 1000.0);
        let Some(entry) = self.workers[worker].as_ref() else {
            return;
        };
        let spec = self.config.copy_tables[entry.table];
        let decoded_per_row = self.config.decoded_bytes(spec.data_bytes_per_row);
        let request_per_row = self.config.request_bytes(spec.data_bytes_per_row);
        let Some(state) = self.workers[worker]
            .as_mut()
            .and_then(|entry| entry.children.get_mut(child))
        else {
            return;
        };
        if !state.streaming {
            return;
        }
        let mut write = None;
        if state.writing.is_none() {
            let mut budget = state.carry + parse_per_tick;
            let mut next = if budget > 0.0 {
                state.unread.pop_front()
            } else {
                None
            };
            while let Some((rows, bytes, done)) = next {
                budget -= bytes;
                state.unread_bytes = (state.unread_bytes - bytes).max(0.0);
                state.consumed += bytes;
                state.part_done |= done;
                if rows > 0 {
                    state.batch_first.get_or_insert(now);
                }
                state.batch_rows += rows;
                state.batch_decoded = (rows as f64).mul_add(decoded_per_row, state.batch_decoded);
                state.batch_wire += bytes;
                next = if budget > 0.0 && state.batch_decoded < target {
                    state.unread.pop_front()
                } else {
                    None
                };
            }
            state.carry = if state.unread.is_empty() {
                budget.min(0.0)
            } else {
                budget.min(parse_per_tick)
            };
            let drained = state.part_done && state.unread.is_empty();
            let timed_out = state
                .batch_first
                .is_some_and(|first| now >= first + deadline);
            if state.batch_rows > 0 && (state.batch_decoded >= target || drained || timed_out) {
                write = Some((state.batch_rows as f64 * request_per_row, state.batch_wire));
                state.batch_rows = 0;
                state.batch_decoded = 0.0;
                state.batch_wire = 0.0;
                state.batch_first = None;
            }
        }
        let unread = state.consumed - state.credited;
        let credit = (unread >= CREDIT_STEP_BYTES
            || (unread > 0.0 && now >= state.last_credit_at + CREDIT_IDLE_TICKS))
            .then_some(state.consumed);
        if credit.is_some() {
            state.credited = state.consumed;
            state.last_credit_at = now;
        }
        let conn = state.conn;
        let finished_part = state.part_done
            && state.unread.is_empty()
            && state.batch_rows == 0
            && state.writing.is_none()
            && write.is_none();
        if let (Some(conn), Some(consumed)) = (conn, credit) {
            send_tracked(
                &mut self.net,
                conn,
                Unit::Credit {
                    consumed: to_u64(consumed),
                },
            );
        }
        if let Some((request_bytes, wire)) = write {
            let id = self.submit_write(
                Owner::Child(worker, child),
                &[request_bytes],
                HttpKind::Insert,
            );
            if let Some(state) = self.workers[worker]
                .as_mut()
                .and_then(|entry| entry.children.get_mut(child))
            {
                state.writing = Some((id, wire));
            }
        }
        if finished_part {
            self.next_partition(worker, child, now);
        }
    }

    fn on_child_write(&mut self, worker: usize, child: usize, write: u64) {
        let Some(state) = self.workers[worker]
            .as_mut()
            .and_then(|entry| entry.children.get_mut(child))
        else {
            return;
        };
        if let Some((pending, wire)) = state.writing {
            if pending == write {
                state.writing = None;
                self.recorder.borrow_mut().copy_bytes += wire;
            }
        }
    }

    fn check_catchup_done(&mut self, worker: usize, now: Tick) {
        let Some(entry) = self.workers[worker].as_mut() else {
            return;
        };
        if entry.phase != Phase::Catchup || !entry.lp.reached() || !entry.lp.quiescent() {
            return;
        }
        let table = entry.table as u64;
        if let Some(conn) = entry.lp.conn.take() {
            self.conn_owner.remove(&conn);
            self.net.close(conn);
        }
        entry.phase = Phase::Cleanup;
        entry.script = Some(Script::new([
            Step::Store(StoreOp::TableState(table, STATE_SYNC_DONE)),
            Step::Sql(Op::DropSlot, table, 0),
        ]));
        self.advance(Owner::Worker(worker), now);
    }

    // ---- state store pool ---------------------------------------------------

    fn submit_store(&mut self, owner: Owner, op: StoreOp) -> u64 {
        let id = self.id();
        self.store_queue.push_back(StoreJob {
            id,
            owner,
            statements: op.statements(),
            value: 0,
        });
        id
    }

    fn send_next_statement(&mut self, index: usize) {
        let id = self.id();
        let entry = &mut self.store_conns[index];
        let Some(job) = entry.job.as_mut() else {
            return;
        };
        if let Some(&(op, a, b)) = job.statements.front() {
            send_tracked(&mut self.net, entry.conn, Unit::Query { id, op, a, b });
        }
    }

    fn run_store_pool(&mut self, now: Tick) {
        let idle_timeout = ticks(self.config.etl.store_idle_timeout_s);
        let mut index = 0;
        while index < self.store_conns.len() {
            let entry = &self.store_conns[index];
            if entry.open && entry.job.is_none() && now >= entry.idle_since + idle_timeout {
                let conn = entry.conn;
                self.store_conns.swap_remove(index);
                self.net.close(conn);
            } else {
                index += 1;
            }
        }
        while !self.store_queue.is_empty() {
            let Some(index) = self
                .store_conns
                .iter()
                .position(|entry| entry.open && entry.job.is_none())
            else {
                break;
            };
            self.store_conns[index].job = self.store_queue.pop_front();
            self.send_next_statement(index);
        }
        let connecting = self.store_conns.iter().filter(|entry| !entry.open).count();
        if self.store_queue.len() > connecting
            && self.store_conns.len() < self.config.etl.store_connections
        {
            let conn = self.net.connect(
                self.postgres,
                ConnKind::Sql,
                self.config.postgres_handshake_round_trips,
                now,
                &mut self.rng,
            );
            self.store_conns.push(StoreConn {
                conn,
                open: false,
                job: None,
                idle_since: now,
            });
        }
    }

    fn on_store_reply(&mut self, index: usize, value: u64, now: Tick) {
        let entry = &mut self.store_conns[index];
        let Some(job) = entry.job.as_mut() else {
            return;
        };
        if let Some((op, ..)) = job.statements.pop_front() {
            if op == Op::LoadCheckpoint {
                job.value = value;
            }
        }
        if job.statements.is_empty() {
            if let Some(done) = entry.job.take() {
                entry.idle_since = now;
                self.completions.push(Completion::Store {
                    owner: done.owner,
                    job: done.id,
                    value: done.value,
                });
            }
        } else {
            self.send_next_statement(index);
        }
    }

    // ---- destination client pool -------------------------------------------

    fn submit_write(&mut self, owner: Owner, sizes: &[f64], kind: HttpKind) -> u64 {
        let write = self.id();
        for &bytes in sizes {
            let id = self.id();
            self.dest_queue.push_back(Request {
                id,
                write,
                bytes: to_u64(bytes.round()),
                kind,
                of: write,
            });
        }
        self.writes.insert(
            write,
            Write {
                owner,
                parts_left: sizes.len() as u64,
            },
        );
        write
    }

    fn run_dest_pool(&mut self, now: Tick) {
        let mut index = 0;
        while index < self.polls.len() {
            if self.polls[index].0 <= now {
                let (_, request) = self.polls.swap_remove(index);
                self.dest_queue.push_back(request);
            } else {
                index += 1;
            }
        }
        let keepalive = ticks(self.config.sink.keepalive_timeout_s);
        let mut index = 0;
        while index < self.dest_conns.len() {
            let entry = &self.dest_conns[index];
            if entry.open && entry.request.is_none() && now >= entry.idle_since + keepalive {
                let conn = entry.conn;
                self.dest_conns.swap_remove(index);
                self.net.close(conn);
            } else {
                index += 1;
            }
        }
        while !self.dest_queue.is_empty() {
            let Some(index) = self
                .dest_conns
                .iter()
                .position(|entry| entry.open && entry.request.is_none())
            else {
                break;
            };
            let Some(request) = self.dest_queue.pop_front() else {
                break;
            };
            let entry = &mut self.dest_conns[index];
            entry.request = Some(request);
            send_tracked(
                &mut self.net,
                entry.conn,
                Unit::HttpRequest {
                    id: request.id,
                    bytes: request.bytes,
                    kind: request.kind,
                    of: request.of,
                },
            );
        }
        let connecting = self.dest_conns.iter().filter(|entry| !entry.open).count();
        let wanted = self.dest_queue.len().saturating_sub(connecting);
        for _ in 0..wanted {
            if self.dest_conns.len() >= self.config.sink.connection_pool {
                break;
            }
            let conn = self.net.connect(
                self.destination,
                ConnKind::Http,
                self.config.https_handshake_round_trips,
                now,
                &mut self.rng,
            );
            self.dest_conns.push(DestConn {
                conn,
                open: false,
                request: None,
                idle_since: now,
            });
        }
    }

    fn on_http_response(&mut self, conn: ConnId, id: u64, durable_at: Tick, now: Tick) {
        let Some(entry) = self.dest_conns.iter_mut().find(|entry| entry.conn == conn) else {
            return;
        };
        let Some(request) = entry.request.take().filter(|request| request.id == id) else {
            return;
        };
        entry.idle_since = now;
        if durable_at == 0 {
            if let Durability::Polled { interval_s, .. } = self.config.sink.durability {
                let poll_id = self.id();
                let original = if request.kind == HttpKind::Poll {
                    request.of
                } else {
                    request.id
                };
                self.polls.push((
                    now + ticks(interval_s),
                    Request {
                        id: poll_id,
                        write: request.write,
                        bytes: 0,
                        kind: HttpKind::Poll,
                        of: original,
                    },
                ));
            }
            return;
        }
        let Some(write) = self.writes.get_mut(&request.write) else {
            return;
        };
        write.parts_left -= 1;
        if write.parts_left == 0 {
            let owner = write.owner;
            self.writes.remove(&request.write);
            self.completions.push(Completion::Write {
                owner,
                write: request.write,
            });
        }
    }

    // ---- event routing -----------------------------------------------------

    fn on_event(&mut self, event: NetEvent, now: Tick) {
        match event {
            NetEvent::Connected(conn) => {
                if let Some(entry) = self.store_conns.iter_mut().find(|entry| entry.conn == conn) {
                    entry.open = true;
                    entry.idle_since = now;
                } else if let Some(entry) =
                    self.dest_conns.iter_mut().find(|entry| entry.conn == conn)
                {
                    entry.open = true;
                    entry.idle_since = now;
                } else if let Some(&owner) = self.conn_owner.get(&conn) {
                    self.resume(owner, Wait::Conn(conn), 0, now);
                }
            }
            NetEvent::Closed(conn) => self.on_closed(conn, now),
            NetEvent::Data(conn, unit) => self.on_data(conn, unit, now),
            NetEvent::Accepted(..) => panic!("nothing connects to the replicator"),
        }
    }

    fn resume(&mut self, owner: Owner, wait: Wait, value: u64, now: Tick) -> bool {
        let resumed = self
            .script_slot(owner)
            .and_then(Option::as_mut)
            .is_some_and(|script| script.complete(wait, value));
        if resumed {
            self.advance(owner, now);
        }
        resumed
    }

    fn on_closed(&mut self, conn: ConnId, now: Tick) {
        if let Some(index) = self.store_conns.iter().position(|entry| entry.conn == conn) {
            let entry = self.store_conns.swap_remove(index);
            if let Some(job) = entry.job {
                self.store_queue.push_front(job);
            }
            return;
        }
        if let Some(index) = self.dest_conns.iter().position(|entry| entry.conn == conn) {
            let entry = self.dest_conns.swap_remove(index);
            if let Some(request) = entry.request {
                self.dest_queue.push_front(request);
            }
            return;
        }
        match self.conn_owner.remove(&conn) {
            Some(Owner::Apply) => self.apply_failed(now, "replication connection closed"),
            Some(owner) => {
                self.recorder.borrow_mut().event(
                    now,
                    format!("{owner:?} lost its connection (not retried in this model)"),
                );
            }
            None => {}
        }
    }

    fn on_data(&mut self, conn: ConnId, unit: Unit, now: Tick) {
        match unit {
            Unit::Reply { id, value } => {
                if let Some(index) = self.store_conns.iter().position(|entry| entry.conn == conn) {
                    self.on_store_reply(index, value, now);
                } else if let Some(owner) = self.query_owner.remove(&id) {
                    self.resume(owner, Wait::Reply(id), value, now);
                }
            }
            Unit::HttpResponse { id, durable_at } => {
                self.on_http_response(conn, id, durable_at, now);
            }
            Unit::Wal(_) | Unit::Keepalive { .. } => {
                let target = match self.conn_owner.get(&conn) {
                    Some(Owner::Apply) => Some(LoopId::Apply),
                    Some(Owner::Catchup(worker)) => Some(LoopId::Catchup(*worker)),
                    _ => None,
                };
                if let Some(id) = target {
                    if let Some(mut lp) = self.take_loop(id) {
                        lp.unread_bytes += unit.wire_bytes() as f64;
                        lp.unread.push_back(unit);
                        self.put_loop(id, lp);
                    }
                }
            }
            Unit::CopyData { rows, bytes, done } => {
                if let Some(Owner::Child(worker, child)) = self.conn_owner.get(&conn).copied() {
                    if let Some(state) = self.workers[worker]
                        .as_mut()
                        .and_then(|entry| entry.children.get_mut(child))
                    {
                        state.unread.push_back((rows, bytes as f64, done));
                        state.unread_bytes += bytes as f64;
                    }
                }
            }
            other @ (Unit::Syn { .. }
            | Unit::Ping
            | Unit::Pong
            | Unit::Close
            | Unit::Reset
            | Unit::Credit { .. }
            | Unit::Query { .. }
            | Unit::Status { .. }
            | Unit::HttpRequest { .. }) => panic!("etl got unexpected {other:?}"),
        }
    }

    fn drain_completions(&mut self, now: Tick) {
        for _ in 0..1000 {
            let completions = std::mem::take(&mut self.completions);
            if completions.is_empty() {
                return;
            }
            for completion in completions {
                match completion {
                    Completion::Store { owner, job, value } => {
                        if !self.resume(owner, Wait::Store(job), value, now) {
                            match owner {
                                Owner::Apply => self.on_loop_checkpoint(LoopId::Apply, job, now),
                                Owner::Catchup(worker) => {
                                    self.on_loop_checkpoint(LoopId::Catchup(worker), job, now);
                                }
                                Owner::Boot | Owner::Worker(_) | Owner::Child(..) => {}
                            }
                        }
                    }
                    Completion::Write { owner, write } => {
                        if !self.resume(owner, Wait::Write(write), 0, now) {
                            match owner {
                                Owner::Apply => self.on_loop_write(LoopId::Apply, write, now),
                                Owner::Catchup(worker) => {
                                    self.on_loop_write(LoopId::Catchup(worker), write, now);
                                }
                                Owner::Child(worker, child) => {
                                    self.on_child_write(worker, child, write);
                                }
                                Owner::Boot | Owner::Worker(_) => {}
                            }
                        }
                    }
                }
            }
        }
        panic!("store and write completions kept producing more completions");
    }
}

impl Agent<Payload> for EtlAgent {
    fn on_message(&mut self, ctx: &mut AgentContext<'_, Payload>, message: &Message<Payload>) {
        let now = ctx.time();
        let events = self
            .net
            .receive(message, now, &self.config.faults, &mut self.rng);
        if !self.up {
            return;
        }
        for event in events {
            self.on_event(event, now);
        }
        self.drain_completions(now);
    }

    fn on_tick(&mut self, ctx: &mut AgentContext<'_, Payload>) {
        let now = ctx.time();
        let crashes = &self.config.faults.etl_crashes;
        if self.next_crash < crashes.len() && crashes[self.next_crash] <= now {
            self.next_crash += 1;
            self.crash(now);
        }
        if self.restart_at.is_some_and(|at| now >= at) {
            self.restart_at = None;
            self.boot(now);
        }
        if self.up {
            if let ApplyState::Retrying(at) = self.apply.state {
                if now >= at {
                    self.start_apply(now);
                }
            }
            self.start_workers(now);
            self.run_loop(LoopId::Apply, now);
            for worker in 0..self.workers.len() {
                let Some(entry) = self.workers[worker].as_ref() else {
                    continue;
                };
                if entry.phase == Phase::Catchup {
                    self.run_loop(LoopId::Catchup(worker), now);
                }
                let children = self.workers[worker]
                    .as_ref()
                    .map_or(0, |entry| entry.children.len());
                for child in 0..children {
                    self.run_child(worker, child, now);
                }
            }
            self.run_store_pool(now);
            self.run_dest_pool(now);
            self.drain_completions(now);
        }
        let copy_complete = {
            let mut recorder = self.recorder.borrow_mut();
            recorder.cross_region_bytes[1] = self.net.cross_region_bytes;
            recorder.copy_done_at
        };
        // A copy run ends once every table has been ready for 30 s, so the
        // report also covers streaming right after the handover.
        if copy_complete.is_some_and(|done| now >= done + ticks(30.0)) {
            ctx.send_halt_interrupt("every table is ready");
        }
        self.net.pump(ctx, now, &self.config.faults, &mut self.rng);
    }
}
