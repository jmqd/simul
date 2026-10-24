//! Protocol units carried over simulated TCP connections, and their encoding
//! into `simul` message payloads.
//!
//! A payload is a sequence of `(connection, unit)` frames: everything one
//! agent delivers to one peer in a tick travels as one engine message, because
//! the engine hands each agent at most one message per tick.

use std::sync::Arc;

/// The engine message type: encoded frames, or `None` for an empty delivery.
pub(crate) type Payload = Option<Arc<[u8]>>;

pub(crate) type ConnId = u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ConnKind {
    /// A logical replication connection (walsender), also used for slot SQL.
    Replication,
    /// An ordinary SQL connection: state store, copy children, metadata.
    Sql,
    /// An HTTPS keep-alive connection to the destination.
    Http,
}

/// A statement executed by a Postgres backend. Each costs one round trip.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Op {
    Begin,
    Commit,
    AdvisoryLock,
    /// `etl.replication_progress` upsert; the argument is the checkpoint LSN.
    UpsertCheckpoint,
    /// Store cache load; replies with the stored checkpoint LSN.
    LoadCheckpoint,
    /// Table-state upsert; arguments are the table and the state code.
    SetTableState,
    StoreSchema,
    /// A small catalog or metadata query.
    Select,
    /// Source migrations run at startup.
    Migrations,
    /// `pg_replication_slots` lookup; replies with the slot's confirmed flush.
    GetSlot,
    /// `CREATE_REPLICATION_SLOT … USE_SNAPSHOT` for a table; replies with the
    /// consistent point.
    CreateSlot,
    ExportSnapshot,
    SetSnapshot,
    LockTable,
    CommitSlotTransaction,
    DropSlot,
    /// `START_REPLICATION` from the LSN argument.
    StartReplication,
    /// Table-sync catch-up streaming from the table slot up to the second
    /// argument.
    StartCatchup,
    /// `COPY … TO STDOUT` of a ctid range: rows, then wire bytes.
    StartCopy,
}

/// Rows committed in one table during one millisecond, or a decoded slice of
/// them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WalChunk {
    pub(crate) table: u64,
    pub(crate) rows: u64,
    pub(crate) wire_bytes: u64,
    /// Transactions whose commit record ends this chunk; zero for a slice that
    /// stops mid-record.
    pub(crate) commits: u64,
    /// The commit end LSN when `commits > 0`.
    pub(crate) end_lsn: u64,
    /// Rows-weighted mean commit time.
    pub(crate) commit_tick: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HttpKind {
    Insert,
    /// Durability poll for an accepted write.
    Poll,
    /// Table drop/create during copy setup.
    Ddl,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Unit {
    Syn {
        kind: ConnKind,
        one_way_us: u64,
    },
    Ping,
    Pong,
    Close,
    Reset,
    /// Receive-window update: total bytes the application has read.
    Credit {
        consumed: u64,
    },
    Query {
        id: u64,
        op: Op,
        a: u64,
        b: u64,
    },
    Reply {
        id: u64,
        value: u64,
    },
    Wal(WalChunk),
    Keepalive {
        wal_end: u64,
        reply: bool,
    },
    Status {
        write: u64,
        flush: u64,
    },
    CopyData {
        rows: u64,
        bytes: u64,
        done: bool,
    },
    HttpRequest {
        id: u64,
        bytes: u64,
        kind: HttpKind,
        of: u64,
    },
    /// `durable_at` is zero when the write was accepted but is not durable.
    HttpResponse {
        id: u64,
        durable_at: u64,
    },
}

impl Unit {
    /// Bytes this unit occupies on the TCP stream.
    pub(crate) const fn wire_bytes(&self) -> u64 {
        match *self {
            Self::Syn { .. } | Self::Ping | Self::Pong | Self::Reply { .. } => 120,
            Self::Close | Self::Reset | Self::Credit { .. } => 40,
            Self::Query { .. } => 160,
            Self::Wal(chunk) => chunk.wire_bytes,
            // PrimaryKeepalive and StandbyStatusUpdate inside CopyData.
            Self::Keepalive { .. } => 23,
            Self::Status { .. } => 39,
            Self::CopyData { bytes, .. } => bytes,
            Self::HttpRequest { bytes, .. } => bytes + 400,
            Self::HttpResponse { .. } => 300,
        }
    }
}

const fn kind_code(kind: ConnKind) -> u64 {
    match kind {
        ConnKind::Replication => 0,
        ConnKind::Sql => 1,
        ConnKind::Http => 2,
    }
}

fn kind_from(code: u64) -> ConnKind {
    match code {
        0 => ConnKind::Replication,
        1 => ConnKind::Sql,
        2 => ConnKind::Http,
        other => panic!("unknown connection kind {other}"),
    }
}

const OPS: [Op; 19] = [
    Op::Begin,
    Op::Commit,
    Op::AdvisoryLock,
    Op::UpsertCheckpoint,
    Op::LoadCheckpoint,
    Op::SetTableState,
    Op::StoreSchema,
    Op::Select,
    Op::Migrations,
    Op::GetSlot,
    Op::CreateSlot,
    Op::ExportSnapshot,
    Op::SetSnapshot,
    Op::LockTable,
    Op::CommitSlotTransaction,
    Op::DropSlot,
    Op::StartReplication,
    Op::StartCatchup,
    Op::StartCopy,
];

fn op_code(op: Op) -> u64 {
    let index = OPS
        .iter()
        .position(|candidate| *candidate == op)
        .unwrap_or_else(|| panic!("{op:?} is listed in OPS"));
    index as u64
}

fn op_from(code: u64) -> Op {
    usize::try_from(code)
        .ok()
        .and_then(|index| OPS.get(index).copied())
        .unwrap_or_else(|| panic!("unknown op code {code}"))
}

const HTTP_KINDS: [HttpKind; 3] = [HttpKind::Insert, HttpKind::Poll, HttpKind::Ddl];

const fn http_code(kind: HttpKind) -> u64 {
    match kind {
        HttpKind::Insert => 0,
        HttpKind::Poll => 1,
        HttpKind::Ddl => 2,
    }
}

/// Accumulates frames for one peer.
#[derive(Clone, Debug, Default)]
pub(crate) struct FrameWriter {
    words: Vec<u64>,
}

impl FrameWriter {
    pub(crate) fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    pub(crate) fn push(&mut self, conn: ConnId, unit: Unit) {
        let words = &mut self.words;
        words.push(conn);
        match unit {
            Unit::Syn { kind, one_way_us } => words.extend([0, kind_code(kind), one_way_us]),
            Unit::Ping => words.push(1),
            Unit::Pong => words.push(2),
            Unit::Close => words.push(3),
            Unit::Reset => words.push(4),
            Unit::Credit { consumed } => words.extend([5, consumed]),
            Unit::Query { id, op, a, b } => words.extend([6, id, op_code(op), a, b]),
            Unit::Reply { id, value } => words.extend([7, id, value]),
            Unit::Wal(chunk) => words.extend([
                8,
                chunk.table,
                chunk.rows,
                chunk.wire_bytes,
                chunk.commits,
                chunk.end_lsn,
                chunk.commit_tick,
            ]),
            Unit::Keepalive { wal_end, reply } => words.extend([9, wal_end, u64::from(reply)]),
            Unit::Status { write, flush } => words.extend([10, write, flush]),
            Unit::CopyData { rows, bytes, done } => {
                words.extend([11, rows, bytes, u64::from(done)]);
            }
            Unit::HttpRequest {
                id,
                bytes,
                kind,
                of,
            } => words.extend([12, id, bytes, http_code(kind), of]),
            Unit::HttpResponse { id, durable_at } => words.extend([13, id, durable_at]),
        }
    }

    /// Encodes the frames and clears the writer.
    pub(crate) fn take_payload(&mut self) -> Option<Arc<[u8]>> {
        if self.words.is_empty() {
            return None;
        }
        let bytes: Vec<u8> = self
            .words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        self.words.clear();
        Some(Arc::from(bytes))
    }
}

/// Decodes a payload written by [`FrameWriter`].
pub(crate) fn decode(payload: &[u8]) -> Vec<(ConnId, Unit)> {
    let words: Vec<u64> = payload
        .chunks_exact(8)
        .map(|chunk| {
            let mut word = [0_u8; 8];
            word.copy_from_slice(chunk);
            u64::from_le_bytes(word)
        })
        .collect();
    let mut frames = Vec::new();
    let mut rest = words.as_slice();
    while let [conn, tag, tail @ ..] = rest {
        let (unit, used) = match (*tag, tail) {
            (0, [kind, one_way_us, ..]) => (
                Unit::Syn {
                    kind: kind_from(*kind),
                    one_way_us: *one_way_us,
                },
                2,
            ),
            (1, _) => (Unit::Ping, 0),
            (2, _) => (Unit::Pong, 0),
            (3, _) => (Unit::Close, 0),
            (4, _) => (Unit::Reset, 0),
            (5, [consumed, ..]) => (
                Unit::Credit {
                    consumed: *consumed,
                },
                1,
            ),
            (6, [id, op, a, b, ..]) => (
                Unit::Query {
                    id: *id,
                    op: op_from(*op),
                    a: *a,
                    b: *b,
                },
                4,
            ),
            (7, [id, value, ..]) => (
                Unit::Reply {
                    id: *id,
                    value: *value,
                },
                2,
            ),
            (8, [table, rows, wire_bytes, commits, end_lsn, commit_tick, ..]) => (
                Unit::Wal(WalChunk {
                    table: *table,
                    rows: *rows,
                    wire_bytes: *wire_bytes,
                    commits: *commits,
                    end_lsn: *end_lsn,
                    commit_tick: *commit_tick,
                }),
                6,
            ),
            (9, [wal_end, reply, ..]) => (
                Unit::Keepalive {
                    wal_end: *wal_end,
                    reply: *reply != 0,
                },
                2,
            ),
            (10, [write, flush, ..]) => (
                Unit::Status {
                    write: *write,
                    flush: *flush,
                },
                2,
            ),
            (11, [rows, bytes, done, ..]) => (
                Unit::CopyData {
                    rows: *rows,
                    bytes: *bytes,
                    done: *done != 0,
                },
                3,
            ),
            (12, [id, bytes, kind, of, ..]) => (
                Unit::HttpRequest {
                    id: *id,
                    bytes: *bytes,
                    kind: usize::try_from(*kind)
                        .ok()
                        .and_then(|index| HTTP_KINDS.get(index).copied())
                        .unwrap_or_else(|| panic!("unknown HTTP kind {kind}")),
                    of: *of,
                },
                4,
            ),
            (13, [id, durable_at, ..]) => (
                Unit::HttpResponse {
                    id: *id,
                    durable_at: *durable_at,
                },
                2,
            ),
            (tag, _) => panic!("malformed frame with tag {tag}"),
        };
        frames.push((*conn, unit));
        rest = &tail[used..];
    }
    assert!(rest.is_empty(), "trailing bytes in a frame payload");
    frames
}
