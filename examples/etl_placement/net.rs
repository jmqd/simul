//! Flow-level TCP between agents.
//!
//! Each agent owns the sending half of its connections. A flow moves bytes
//! when its congestion window, the receiver's window, the per-flow line rate,
//! and the instance's share of NIC bandwidth all allow, and delivers each unit
//! one-way delay after its last byte leaves. In-order delivery, CUBIC growth,
//! loss recovery, idle restart, and retransmission timeouts follow Linux
//! defaults (`tcp_cubic.c`, `tcp_output.c`); see `config.rs` for sources.

use crate::rng::Rng;
use crate::time::{secs, ticks, ticks_at_least_one, Tick};
use crate::wire::{decode, ConnId, ConnKind, FrameWriter, Unit};
use simul::agent::{AgentContext, AgentId};
use simul::message::Message;
use std::collections::{BTreeMap, VecDeque};

/// Round-trip and loss characteristics of the network between two regions,
/// or within one.
#[derive(Clone, Debug)]
pub(crate) struct PathModel {
    pub(crate) cross_region: bool,
    /// Each connection draws its base RTT once, `LogNormal(ln median, sigma)`,
    /// because AWS hashes every 5-tuple onto its own backbone path.
    pub(crate) rtt_median_s: f64,
    pub(crate) rtt_sigma: f64,
    pub(crate) rtt_floor_s: f64,
    /// Mean of the exponential queueing delay added to each send.
    pub(crate) jitter_mean_s: f64,
    /// Baseline per-packet loss probability.
    pub(crate) loss: f64,
    /// Path re-routes: mean seconds between changes and the RTT step range.
    pub(crate) path_change: Option<PathChange>,
}

/// Path re-routes: the mean time between changes and the range of the RTT
/// step each change makes.
#[derive(Clone, Copy, Debug)]
#[allow(
    clippy::struct_field_names,
    reason = "`_s` marks seconds, as on every duration in the model"
)]
pub(crate) struct PathChange {
    pub(crate) mean_interval_s: f64,
    pub(crate) rtt_step_min_s: f64,
    pub(crate) rtt_step_max_s: f64,
}

/// Linux TCP stack parameters.
#[derive(Clone, Debug)]
pub(crate) struct TcpModel {
    pub(crate) mss_bytes: f64,
    pub(crate) initial_cwnd_segments: f64,
    /// Largest advertised receive window: `tcp_rmem` max times the scaling
    /// ratio.
    pub(crate) receive_window_bytes: f64,
    /// `tcp_wmem` max: bytes the sender may hold unacknowledged plus unsent.
    pub(crate) send_buffer_bytes: f64,
    pub(crate) rto_min_s: f64,
    pub(crate) rto_max_s: f64,
    pub(crate) per_flow_bytes_per_s: f64,
    pub(crate) slow_start_after_idle: bool,
    pub(crate) cubic_c: f64,
    pub(crate) cubic_beta: f64,
}

/// EC2 network allowance: a baseline rate plus burst credits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NicModel {
    pub(crate) baseline_bytes_per_s: f64,
    pub(crate) burst_bytes_per_s: f64,
    /// Seconds of full burst the credit bucket holds when full.
    pub(crate) burst_seconds: f64,
}

#[derive(Clone, Debug)]
struct Nic {
    baseline: f64,
    burst: f64,
    credits: f64,
    capacity: f64,
}

impl Nic {
    fn new(model: NicModel) -> Self {
        let baseline = crate::time::per_tick(model.baseline_bytes_per_s);
        let burst = crate::time::per_tick(model.burst_bytes_per_s).max(baseline);
        let capacity =
            (model.burst_bytes_per_s - model.baseline_bytes_per_s).max(0.0) * model.burst_seconds;
        Self {
            baseline,
            burst,
            credits: capacity,
            capacity,
        }
    }

    fn budget(&self) -> f64 {
        self.baseline + self.credits.min(self.burst - self.baseline)
    }

    fn spend(&mut self, used: f64) {
        self.credits = (self.credits + self.baseline - used).clamp(0.0, self.capacity);
    }
}

/// A half-open interval of simulated time.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Window {
    pub(crate) start: Tick,
    pub(crate) end: Tick,
}

impl Window {
    pub(crate) const fn contains(self, time: Tick) -> bool {
        self.start <= time && time < self.end
    }

    const fn overlaps(self, start: Tick, end: Tick) -> bool {
        self.start < end && start < self.end
    }
}

/// Injected faults, fixed before the run so every agent sees the same plan.
#[derive(Clone, Debug, Default)]
pub(crate) struct FaultPlan {
    /// Cross-region loss episodes and their per-packet loss probability.
    pub(crate) loss_bursts: Vec<(Window, f64)>,
    /// Cross-region partitions: every packet in either direction is lost.
    pub(crate) outages: Vec<Window>,
    /// The source resets the apply worker's replication connection.
    pub(crate) replication_resets: Vec<Tick>,
    /// The ETL process dies and restarts after `EtlModel::restart_delay_s`.
    pub(crate) etl_crashes: Vec<Tick>,
    /// Destination service time multiplied by a factor.
    pub(crate) destination_brownouts: Vec<(Window, f64)>,
}

impl FaultPlan {
    /// When the earliest injected fault begins.
    pub(crate) fn first_fault(&self) -> Option<Tick> {
        self.loss_bursts
            .iter()
            .map(|(window, _)| window.start)
            .chain(self.outages.iter().map(|window| window.start))
            .chain(self.replication_resets.iter().copied())
            .chain(self.etl_crashes.iter().copied())
            .chain(
                self.destination_brownouts
                    .iter()
                    .map(|(window, _)| window.start),
            )
            .min()
    }

    fn loss(&self, cross_region: bool, now: Tick) -> f64 {
        if !cross_region {
            return 0.0;
        }
        self.loss_bursts
            .iter()
            .filter(|(window, _)| window.contains(now))
            .map(|(_, probability)| *probability)
            .fold(0.0, f64::max)
    }

    pub(crate) fn partitioned(&self, cross_region: bool, now: Tick) -> bool {
        cross_region && self.outages.iter().any(|window| window.contains(now))
    }

    /// When a segment sent at `now` is lost to a partition, returns when its
    /// retransmission finally gets through: Linux retries after the RTO and
    /// doubles the wait each time, up to `rto_max`.
    fn retransmit_time(
        &self,
        cross_region: bool,
        now: Tick,
        rtt: Tick,
        rto: f64,
        rto_max: f64,
    ) -> Option<Tick> {
        let blocked = |start: Tick| {
            self.outages
                .iter()
                .any(|window| window.overlaps(start, start + rtt.max(1)))
        };
        if !cross_region || !blocked(now) {
            return None;
        }
        let mut wait = rto;
        let mut attempt = now + ticks(wait);
        while blocked(attempt) {
            wait = (wait * 2.0).min(rto_max);
            attempt += ticks(wait);
        }
        Some(attempt)
    }
}

#[derive(Clone, Debug)]
struct Pending {
    unit: Unit,
    left: f64,
}

#[derive(Clone, Copy, Debug)]
struct CubicEpoch {
    start: Tick,
    k: f64,
    origin: f64,
    reno_segments: f64,
}

/// The sending half of one TCP connection.
#[derive(Clone, Debug)]
struct Flow {
    one_way_s: f64,
    path: PathModel,
    next_path_change: Option<Tick>,
    cwnd: f64,
    ssthresh: f64,
    w_max_segments: f64,
    epoch: Option<CubicEpoch>,
    recovery_until: Tick,
    inflight: VecDeque<(Tick, f64)>,
    inflight_bytes: f64,
    last_ack_at: Tick,
    cwnd_limited: bool,
    last_send: Option<Tick>,
    /// The `last_send` whose idle period already restarted the window.
    restarted_after: Option<Tick>,
    blocked_until: Tick,
    queue: VecDeque<Pending>,
    queued_bytes: f64,
    deliveries: VecDeque<(Tick, Unit)>,
    last_delivery: Tick,
    /// Bytes the receiving application has read, when it reads lazily.
    credit: Option<f64>,
    sent_total: f64,
}

impl Flow {
    fn new(path: PathModel, one_way_s: f64, tcp: &TcpModel, now: Tick, rng: &mut Rng) -> Self {
        let next_path_change = path
            .path_change
            .filter(|_| path.cross_region)
            .map(|change| now + ticks(rng.exponential(change.mean_interval_s)));
        Self {
            one_way_s,
            path,
            next_path_change,
            cwnd: tcp.initial_cwnd_segments * tcp.mss_bytes,
            ssthresh: f64::INFINITY,
            w_max_segments: 0.0,
            epoch: None,
            recovery_until: 0,
            inflight: VecDeque::new(),
            inflight_bytes: 0.0,
            last_ack_at: 0,
            cwnd_limited: false,
            last_send: None,
            restarted_after: None,
            blocked_until: 0,
            queue: VecDeque::new(),
            queued_bytes: 0.0,
            deliveries: VecDeque::new(),
            last_delivery: 0,
            credit: None,
            sent_total: 0.0,
        }
    }

    fn rtt_s(&self) -> f64 {
        2.0 * self.one_way_s
    }

    /// Linux sets RTO to SRTT plus `max(4·RTTVAR, rto_min)`; RTTVAR is small
    /// on a stable path, so the floor dominates.
    fn rto_s(&self, tcp: &TcpModel) -> f64 {
        (self.rtt_s() + tcp.rto_min_s).min(tcp.rto_max_s)
    }

    fn push(&mut self, unit: Unit) {
        let bytes = unit.wire_bytes() as f64;
        self.queued_bytes += bytes;
        self.queue.push_back(Pending { unit, left: bytes });
    }

    /// Bytes buffered in the socket: unsent plus unacknowledged.
    fn buffered(&self) -> f64 {
        self.queued_bytes + self.inflight_bytes
    }

    fn advance(&mut self, now: Tick, tcp: &TcpModel, rng: &mut Rng) {
        let mut acked = 0.0;
        while let Some(&(at, bytes)) = self.inflight.front() {
            if at > now {
                break;
            }
            self.inflight.pop_front();
            acked += bytes;
        }
        if acked > 0.0 {
            self.inflight_bytes = (self.inflight_bytes - acked).max(0.0);
            if self.cwnd_limited {
                self.grow(now, acked, tcp);
            }
        }
        if let (Some(at), Some(change)) = (self.next_path_change, self.path.path_change) {
            if now >= at {
                let step = rng.uniform(change.rtt_step_min_s, change.rtt_step_max_s) / 2.0;
                let signed = if rng.bernoulli(0.5) { step } else { -step };
                self.one_way_s = (self.one_way_s + signed).max(self.path.rtt_floor_s / 2.0);
                self.next_path_change = Some(now + ticks(rng.exponential(change.mean_interval_s)));
            }
        }
    }

    fn grow(&mut self, now: Tick, acked: f64, tcp: &TcpModel) {
        if self.cwnd < self.ssthresh {
            self.cwnd += acked;
            return;
        }
        let mss = tcp.mss_bytes;
        let segments = self.cwnd / mss;
        let w_max = self.w_max_segments;
        let c = tcp.cubic_c;
        let rtt = self.rtt_s();
        let epoch = self.epoch.get_or_insert_with(|| CubicEpoch {
            start: now,
            k: if w_max > segments {
                ((w_max - segments) / c).cbrt()
            } else {
                0.0
            },
            origin: w_max.max(segments),
            reno_segments: segments,
        });
        let elapsed = secs(now - epoch.start) + rtt;
        let target = c.mul_add((elapsed - epoch.k).powi(3), epoch.origin);
        let acked_segments = acked / mss;
        let cubic = if target > segments {
            ((target - segments) / segments).mul_add(acked_segments, segments)
        } else {
            (0.01 * acked_segments).mul_add(1.0 / segments, segments)
        };
        // RFC 9438 §4.3: never grow slower than Reno would.
        let beta = tcp.cubic_beta;
        epoch.reno_segments += 3.0 * (1.0 - beta) / (1.0 + beta) * acked_segments / segments;
        self.cwnd = cubic.max(epoch.reno_segments) * mss;
    }

    /// Fast retransmit: one multiplicative decrease per round trip.
    fn on_loss(&mut self, now: Tick, tcp: &TcpModel) {
        if now < self.recovery_until {
            return;
        }
        let segments = self.cwnd / tcp.mss_bytes;
        // CUBIC fast convergence releases bandwidth to newer flows.
        self.w_max_segments = if segments < self.w_max_segments {
            segments * (1.0 + tcp.cubic_beta) / 2.0
        } else {
            segments
        };
        self.cwnd = (self.cwnd * tcp.cubic_beta).max(2.0 * tcp.mss_bytes);
        self.ssthresh = self.cwnd;
        self.epoch = None;
        self.recovery_until = now + ticks(self.rtt_s());
    }

    fn on_timeout(&mut self, resume_at: Tick, tcp: &TcpModel) {
        self.w_max_segments = self.cwnd / tcp.mss_bytes;
        self.ssthresh = (self.cwnd * tcp.cubic_beta).max(2.0 * tcp.mss_bytes);
        self.cwnd = tcp.mss_bytes;
        self.epoch = None;
        self.blocked_until = resume_at;
        self.recovery_until = resume_at + ticks(self.rtt_s());
    }

    /// Bytes the flow could send this tick before the NIC share is applied.
    fn want(&mut self, now: Tick, tcp: &TcpModel) -> f64 {
        if self.queue.is_empty() || now < self.blocked_until {
            return 0.0;
        }
        let window = tcp.receive_window_bytes;
        let receiver_room = match self.credit {
            Some(consumed) => window - (self.sent_total - consumed),
            None => window - self.inflight_bytes,
        };
        let buffer_room = tcp.send_buffer_bytes - self.inflight_bytes;
        // Linux checks for idleness when it is about to send, once per idle
        // period: `last_send` changes with every transmission.
        if let (true, Some(last), true, true) = (
            tcp.slow_start_after_idle,
            self.last_send,
            self.inflight.is_empty(),
            receiver_room > 0.0 && buffer_room > 0.0,
        ) {
            if self.restarted_after != Some(last) {
                self.restarted_after = Some(last);
                self.restart_after_idle(secs(now - last), tcp);
            }
        }
        let cwnd_room = self.cwnd - self.inflight_bytes;
        let line = crate::time::per_tick(tcp.per_flow_bytes_per_s);
        self.cwnd_limited =
            cwnd_room <= receiver_room.min(buffer_room) && cwnd_room < self.queued_bytes;
        cwnd_room
            .min(receiver_room)
            .min(buffer_room)
            .min(line)
            .min(self.queued_bytes)
            .max(0.0)
    }

    /// `tcp_cwnd_restart`: halve cwnd for every RTO spent idle, down to the
    /// initial window.
    fn restart_after_idle(&mut self, idle_s: f64, tcp: &TcpModel) {
        let rto = self.rto_s(tcp);
        if idle_s <= rto {
            return;
        }
        let restart = (tcp.initial_cwnd_segments * tcp.mss_bytes).min(self.cwnd);
        self.ssthresh = self.ssthresh.max(0.75 * self.cwnd);
        let mut remaining = idle_s;
        let mut cwnd = self.cwnd;
        while remaining > rto && cwnd > restart {
            cwnd /= 2.0;
            remaining -= rto;
        }
        self.cwnd = cwnd.max(restart);
        self.epoch = None;
    }

    fn transmit(
        &mut self,
        now: Tick,
        budget: f64,
        tcp: &TcpModel,
        faults: &FaultPlan,
        rng: &mut Rng,
    ) -> f64 {
        if budget <= 0.0 {
            return 0.0;
        }
        let mut remaining = budget;
        let mut completed = Vec::new();
        while let Some(front) = self.queue.front_mut().filter(|_| remaining > 0.0) {
            let take = front.left.min(remaining);
            front.left -= take;
            remaining -= take;
            if front.left <= 1e-6 {
                if let Some(done) = self.queue.pop_front() {
                    completed.push(done.unit);
                }
            }
        }
        let sent = budget - remaining;
        self.queued_bytes = (self.queued_bytes - sent).max(0.0);
        self.sent_total += sent;

        let rtt = self.rtt_s();
        let rtt_ticks = ticks(rtt);
        let (deliver_after, ack_after) = if let Some(resume) = faults.retransmit_time(
            self.path.cross_region,
            now,
            rtt_ticks,
            self.rto_s(tcp),
            tcp.rto_max_s,
        ) {
            self.on_timeout(resume, tcp);
            let wait = secs(resume - now);
            (wait + self.one_way_s, wait + rtt)
        } else {
            let packets = (sent / tcp.mss_bytes).ceil();
            let drop_probability = self.path.loss.max(faults.loss(self.path.cross_region, now));
            let lost = drop_probability > 0.0
                && rng.bernoulli(1.0 - (1.0 - drop_probability).powf(packets));
            if lost {
                self.on_loss(now, tcp);
            }
            let recovery = if lost { rtt } else { 0.0 };
            (
                self.one_way_s + rng.exponential(self.path.jitter_mean_s) + recovery,
                rtt + recovery,
            )
        };
        let deliver_at = (now + ticks_at_least_one(deliver_after)).max(self.last_delivery);
        self.last_delivery = deliver_at;
        let ack_at = (now + ticks_at_least_one(ack_after)).max(self.last_ack_at);
        self.last_ack_at = ack_at;
        self.inflight.push_back((ack_at, sent));
        self.inflight_bytes += sent;
        self.last_send = Some(now);
        for unit in completed {
            self.deliveries.push_back((deliver_at, unit));
        }
        sent
    }

    fn idle(&self) -> bool {
        self.queue.is_empty() && self.deliveries.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConnState {
    Handshaking {
        round_trips_left: u32,
    },
    Open,
    /// A `Close` is queued behind the remaining data.
    Closing,
}

#[derive(Clone, Debug)]
struct Conn {
    peer: usize,
    state: ConnState,
    flow: Flow,
    /// Deliveries are batched to this many ticks to bound message counts.
    quantum: Tick,
}

/// A peer agent and the network path to it.
#[derive(Clone, Debug)]
pub(crate) struct Peer {
    pub(crate) name: &'static str,
    pub(crate) path: PathModel,
    id: Option<AgentId>,
}

impl Peer {
    pub(crate) const fn new(name: &'static str, path: PathModel) -> Self {
        Self {
            name,
            path,
            id: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NetEvent {
    /// A connection this agent opened finished its handshake.
    Connected(ConnId),
    /// A peer opened a connection to this agent.
    Accepted(ConnId, ConnKind),
    Data(ConnId, Unit),
    /// The peer closed or reset the connection, or it never existed.
    Closed(ConnId),
}

/// Sends on a connection the caller still tracks. Every agent drops a
/// connection from its own state when its `Closed` event arrives, so a failed
/// send here is a bookkeeping bug, not a network condition.
///
/// # Panics
///
/// Panics when `conn` is no longer open.
pub(crate) fn send_tracked(net: &mut Endpoint, conn: ConnId, unit: Unit) {
    let sent = net.send(conn, unit);
    assert!(
        sent,
        "sent {unit:?} on connection {conn:#x} after it closed"
    );
}

/// One agent's sockets.
#[derive(Clone, Debug)]
pub(crate) struct Endpoint {
    id_prefix: u64,
    next_local: u64,
    tcp: TcpModel,
    nic: Nic,
    peers: Vec<Peer>,
    conns: BTreeMap<ConnId, Conn>,
    control: Vec<(Tick, usize, ConnId, Unit)>,
    writers: Vec<FrameWriter>,
    /// Bytes sent across regions, for transfer cost.
    pub(crate) cross_region_bytes: f64,
}

impl Endpoint {
    pub(crate) fn new(agent_index: u64, tcp: TcpModel, nic: NicModel, peers: Vec<Peer>) -> Self {
        let writers = peers.iter().map(|_| FrameWriter::default()).collect();
        Self {
            id_prefix: agent_index << 40,
            next_local: 0,
            tcp,
            nic: Nic::new(nic),
            peers,
            conns: BTreeMap::new(),
            control: Vec::new(),
            writers,
            cross_region_bytes: 0.0,
        }
    }

    fn peer_id(&mut self, peer: usize, ctx: &AgentContext) -> AgentId {
        let entry = &mut self.peers[peer];
        if let Some(id) = entry.id {
            return id;
        }
        let id = ctx
            .agent_id(entry.name)
            .unwrap_or_else(|| panic!("peer agent {} exists", entry.name));
        entry.id = Some(id);
        id
    }

    pub(crate) fn peer_index(&self, name: &str) -> usize {
        self.peers
            .iter()
            .position(|peer| peer.name == name)
            .unwrap_or_else(|| panic!("{name} is a configured peer"))
    }

    /// Opens a connection; `round_trips` counts the TCP, TLS, and protocol
    /// handshakes before the application may use it.
    pub(crate) fn connect(
        &mut self,
        peer: usize,
        kind: ConnKind,
        round_trips: u32,
        now: Tick,
        rng: &mut Rng,
    ) -> ConnId {
        assert!(
            round_trips >= 1,
            "a TCP handshake takes at least one round trip"
        );
        let path = self.peers[peer].path.clone();
        let rtt = rng
            .lognormal(path.rtt_median_s, path.rtt_sigma)
            .max(path.rtt_floor_s);
        let one_way_s = rtt / 2.0;
        let id = self.id_prefix | self.next_local;
        self.next_local += 1;
        let mut flow = Flow::new(path, one_way_s, &self.tcp, now, rng);
        flow.push(Unit::Syn {
            kind,
            one_way_us: crate::time::to_u64((one_way_s * 1e6).round()),
        });
        self.conns.insert(
            id,
            Conn {
                peer,
                state: ConnState::Handshaking {
                    round_trips_left: round_trips,
                },
                flow,
                quantum: 1,
            },
        );
        id
    }

    /// Queues a unit; returns false when the connection is gone or not yet
    /// open, in which case nothing is sent.
    #[must_use]
    pub(crate) fn send(&mut self, conn: ConnId, unit: Unit) -> bool {
        match self.conns.get_mut(&conn) {
            Some(entry) if entry.state == ConnState::Open => {
                entry.flow.push(unit);
                true
            }
            Some(_) | None => false,
        }
    }

    /// Bytes waiting in the socket send buffer, for writer backpressure.
    pub(crate) fn buffered(&self, conn: ConnId) -> f64 {
        self.conns
            .get(&conn)
            .map_or(0.0, |entry| entry.flow.buffered())
    }

    pub(crate) const fn send_buffer_bytes(&self) -> f64 {
        self.tcp.send_buffer_bytes
    }

    /// Switches a connection to receiver-paced flow control: the peer reports
    /// what its application read with `Unit::Credit`.
    pub(crate) fn use_receiver_credit(&mut self, conn: ConnId) {
        if let Some(entry) = self.conns.get_mut(&conn) {
            entry.flow.credit = Some(entry.flow.sent_total);
        }
    }

    pub(crate) fn set_quantum(&mut self, conn: ConnId, quantum: Tick) {
        if let Some(entry) = self.conns.get_mut(&conn) {
            entry.quantum = quantum.max(1);
        }
    }

    /// Sends a FIN after the queued data.
    pub(crate) fn close(&mut self, conn: ConnId) {
        if let Some(entry) = self.conns.get_mut(&conn) {
            if entry.state == ConnState::Open {
                entry.flow.push(Unit::Close);
            }
            entry.state = ConnState::Closing;
        }
    }

    /// Drops a connection at once; the kernel resets it at the peer.
    pub(crate) fn abort(&mut self, conn: ConnId, now: Tick, faults: &FaultPlan) {
        if let Some(entry) = self.conns.remove(&conn) {
            self.notify_peer(entry.peer, conn, entry.flow.one_way_s, now, faults);
        }
    }

    pub(crate) fn abort_all(&mut self, now: Tick, faults: &FaultPlan) {
        let conns = std::mem::take(&mut self.conns);
        for (conn, entry) in conns {
            self.notify_peer(entry.peer, conn, entry.flow.one_way_s, now, faults);
        }
    }

    fn notify_peer(
        &mut self,
        peer: usize,
        conn: ConnId,
        one_way_s: f64,
        now: Tick,
        faults: &FaultPlan,
    ) {
        if !faults.partitioned(self.peers[peer].path.cross_region, now) {
            self.control
                .push((now + ticks_at_least_one(one_way_s), peer, conn, Unit::Reset));
        }
    }

    /// Decodes a peer's message into events, answering handshakes itself.
    pub(crate) fn receive(
        &mut self,
        message: &Message,
        now: Tick,
        faults: &FaultPlan,
        rng: &mut Rng,
    ) -> Vec<NetEvent> {
        let Some(payload) = message.custom_payload.as_deref() else {
            return Vec::new();
        };
        let peer = message
            .source
            .and_then(|source| self.peers.iter().position(|peer| peer.id == Some(source)))
            .unwrap_or_else(|| panic!("messages come from configured peers"));
        let mut events = Vec::new();
        for (conn, unit) in decode(payload) {
            match unit {
                Unit::Syn { kind, one_way_us } => {
                    let path = self.peers[peer].path.clone();
                    let mut flow = Flow::new(path, one_way_us as f64 / 1e6, &self.tcp, now, rng);
                    flow.push(Unit::Pong);
                    self.conns.insert(
                        conn,
                        Conn {
                            peer,
                            state: ConnState::Open,
                            flow,
                            quantum: 1,
                        },
                    );
                    events.push(NetEvent::Accepted(conn, kind));
                }
                Unit::Reset | Unit::Close => {
                    if self.conns.remove(&conn).is_some() {
                        events.push(NetEvent::Closed(conn));
                    }
                }
                other @ (Unit::Ping
                | Unit::Pong
                | Unit::Credit { .. }
                | Unit::Query { .. }
                | Unit::Reply { .. }
                | Unit::Wal(_)
                | Unit::Keepalive { .. }
                | Unit::Status { .. }
                | Unit::CopyData { .. }
                | Unit::HttpRequest { .. }
                | Unit::HttpResponse { .. }) => match self.conns.get_mut(&conn) {
                    None => {
                        let one_way = self.peers[peer].path.rtt_median_s / 2.0;
                        self.notify_peer(peer, conn, one_way, now, faults);
                    }
                    Some(entry) => match (other, entry.state) {
                        (Unit::Ping, _) => entry.flow.push(Unit::Pong),
                        (Unit::Pong, ConnState::Handshaking { round_trips_left }) => {
                            if round_trips_left <= 1 {
                                entry.state = ConnState::Open;
                                events.push(NetEvent::Connected(conn));
                            } else {
                                entry.state = ConnState::Handshaking {
                                    round_trips_left: round_trips_left - 1,
                                };
                                entry.flow.push(Unit::Ping);
                            }
                        }
                        (Unit::Credit { consumed }, _) => {
                            if let Some(known) = entry.flow.credit.as_mut() {
                                *known = known.max(consumed as f64);
                            }
                        }
                        (Unit::Pong, ConnState::Open | ConnState::Closing) => {}
                        (data, _) => events.push(NetEvent::Data(conn, data)),
                    },
                },
            }
        }
        events
    }

    /// Moves bytes on every connection and sends this tick's deliveries.
    pub(crate) fn pump(
        &mut self,
        ctx: &mut AgentContext,
        now: Tick,
        faults: &FaultPlan,
        rng: &mut Rng,
    ) {
        let tcp = &self.tcp;
        let mut wants = Vec::new();
        let mut total = 0.0;
        for (conn, entry) in &mut self.conns {
            entry.flow.advance(now, tcp, rng);
            let want = entry.flow.want(now, tcp);
            if want > 0.0 {
                wants.push((*conn, want));
                total += want;
            }
        }
        let budget = self.nic.budget();
        let scale = if total > budget { budget / total } else { 1.0 };
        let mut used = 0.0;
        for (conn, want) in wants {
            if let Some(entry) = self.conns.get_mut(&conn) {
                let sent = entry.flow.transmit(now, want * scale, tcp, faults, rng);
                used += sent;
                if entry.flow.path.cross_region {
                    self.cross_region_bytes += sent;
                }
            }
        }
        self.nic.spend(used);

        // A message sent now is seen next tick.
        let horizon = now + 1;
        let mut finished = Vec::new();
        for (conn, entry) in &mut self.conns {
            if horizon % entry.quantum == 0 {
                while let Some(&(at, unit)) = entry.flow.deliveries.front() {
                    if at > horizon {
                        break;
                    }
                    entry.flow.deliveries.pop_front();
                    self.writers[entry.peer].push(*conn, unit);
                }
            }
            if entry.state == ConnState::Closing && entry.flow.idle() {
                finished.push(*conn);
            }
        }
        for conn in finished {
            self.conns.remove(&conn);
        }
        let mut index = 0;
        while index < self.control.len() {
            let (at, peer, conn, unit) = self.control[index];
            if at <= horizon {
                self.writers[peer].push(conn, unit);
                self.control.swap_remove(index);
            } else {
                index += 1;
            }
        }
        // Resolve every peer, not only those sent to: a peer may open a
        // connection to this agent before this agent ever sends to it.
        for peer in 0..self.peers.len() {
            let id = self.peer_id(peer, ctx);
            if !self.writers[peer].is_empty() {
                ctx.send_to(id, self.writers[peer].take_payload());
            }
        }
    }
}
