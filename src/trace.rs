//! Structured simulation trace recording and JSONL / Perfetto Chrome Trace
//! export.
//!
//! Enable tracing at build time with
//! [`SimulationBuilder::trace`](crate::SimulationBuilder::trace) or dynamically
//! between runs (for example, around a failure window after
//! [`Simulation::run_until`](crate::Simulation::run_until)) with
//! [`Simulation::enable_trace`](crate::Simulation::enable_trace) or
//! [`Simulation::enable_trace_with_payloads`](crate::Simulation::enable_trace_with_payloads).
//! Tracing is purely observational and never alters ticks, scheduling, or RNG
//! streams.

use crate::agent::{AgentId, TimerId};
use crate::experiment::{ReplayKey, TrialContext};
use crate::{DiscreteTime, Simulation, SimulationMode};
use std::io;

/// Buffer flush threshold for streaming JSONL and Perfetto exports.
const WRITE_FLUSH_THRESHOLD: usize = 8 * 1024;

/// Pre-quoted fallback agent name when an event references an unknown index.
const UNKNOWN_JSON_NAME: &str = "\"unknown\"";

/// Two-digit base-10 lookup table for fast integer formatting.
const DEC_DIGITS_LUT: &[u8; 200] = b"\
0001020304050607080910111213141516171819\
2021222324252627282930313233343536373839\
4041424344454647484950515253545556575859\
6061626364656667686970717273747576777879\
8081828384858687888990919293949596979899";

/// Lowercase hexadecimal digits for JSON control-character escapes.
const HEX_LOWER: &[u8; 16] = b"0123456789abcdef";

/// One structured lifecycle event captured by [`SimulationTrace`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TraceEvent {
    /// Simulation tick at which the event occurred.
    pub time: DiscreteTime,
    /// Primary agent associated with the event.
    pub agent: AgentId,
    /// Event-specific details.
    pub kind: TraceEventKind,
}

/// Lifecycle event kinds recorded in a [`SimulationTrace`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceEventKind {
    /// `agent` sent an immediate or delayed message.
    MessageSent {
        /// Target agent, or `None` when the destination matched no agent.
        destination: Option<AgentId>,
        /// Tick on which the message is scheduled to arrive.
        arrival: DiscreteTime,
        /// Debug-formatted payload when payload tracing is enabled.
        payload: Option<String>,
    },
    /// `agent` cancelled a pending timer before delivery.
    TimerCancelled {
        /// Cancelled timer handle.
        timer: TimerId,
    },
    /// A message was appended to `agent`'s inbox.
    MessageDelivered {
        /// Sender identity, or `None` for initial queue messages.
        source: Option<AgentId>,
        /// Receiver queue depth immediately after delivery.
        queue_depth: usize,
    },
    /// `agent` finished consuming a message.
    MessageConsumed {
        /// Sender identity, or `None` for initial queue messages.
        source: Option<AgentId>,
        /// Tick at which the message was queued.
        queued_time: DiscreteTime,
        /// Remaining queue depth immediately after handling.
        queue_depth: usize,
    },
    /// `agent` entered sleep until `until`.
    AgentSlept {
        /// Tick on which the agent will wake.
        until: DiscreteTime,
    },
    /// `agent` woke from sleep.
    AgentWoke,
    /// `agent` requested a halt interrupt (or triggered an unknown-destination halt).
    HaltInterrupt {
        /// Reason string associated with the interrupt.
        reason: String,
    },
}

/// Ordered log of structured simulation lifecycle events, exportable to JSONL
/// and Chrome Trace / Perfetto JSON.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SimulationTrace {
    /// Agent names indexed by [`AgentId::index`].
    agent_names: Vec<String>,
    /// Recorded events in causal execution order.
    events: Vec<TraceEvent>,
}

impl SimulationTrace {
    /// Creates an empty trace bound to the given agent name table.
    pub(crate) const fn new(agent_names: Vec<String>) -> Self {
        Self {
            agent_names,
            events: Vec::new(),
        }
    }

    /// Appends one event to the trace.
    pub(crate) fn push(&mut self, time: DiscreteTime, agent: AgentId, kind: TraceEventKind) {
        self.events.push(TraceEvent { time, agent, kind });
    }

    /// Returns all recorded events in execution order.
    #[must_use]
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// Returns the agent names indexed by [`AgentId::index`].
    #[must_use]
    pub fn agent_names(&self) -> &[String] {
        &self.agent_names
    }

    /// Resolves `id` to its agent name, if present in this trace.
    #[must_use]
    pub fn agent_name(&self, id: AgentId) -> Option<&str> {
        self.agent_names.get(id.index()).map(String::as_str)
    }

    /// Returns the number of recorded events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns whether no events have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Clears recorded events while keeping the agent name table.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Iterates over events whose `time` lies in `[start, end_exclusive)`.
    pub fn events_in_window(
        &self,
        start: DiscreteTime,
        end_exclusive: DiscreteTime,
    ) -> impl Iterator<Item = &TraceEvent> {
        self.events
            .iter()
            .filter(move |event| event.time >= start && event.time < end_exclusive)
    }

    /// Iterates over events associated with `agent`.
    pub fn events_for_agent(&self, agent: AgentId) -> impl Iterator<Item = &TraceEvent> {
        self.events.iter().filter(move |event| event.agent == agent)
    }

    /// Serializes the trace as newline-delimited JSON (`JSONL`), one event
    /// object per line.
    ///
    /// # Errors
    ///
    /// Returns any underlying I/O error from `writer`.
    pub fn write_jsonl(&self, writer: &mut impl io::Write) -> io::Result<()> {
        let escaped_names = self.escaped_agent_names();
        let mut buf = String::with_capacity(WRITE_FLUSH_THRESHOLD + 256);
        for event in &self.events {
            format_jsonl_event(event, &escaped_names, &mut buf);
            buf.push('\n');
            if buf.len() >= WRITE_FLUSH_THRESHOLD {
                writer.write_all(buf.as_bytes())?;
                buf.clear();
            }
        }
        if !buf.is_empty() {
            writer.write_all(buf.as_bytes())?;
        }
        Ok(())
    }

    /// Serializes the trace as a `JSONL` string.
    #[must_use]
    pub fn to_jsonl(&self) -> String {
        let escaped_names = self.escaped_agent_names();
        let mut out = String::with_capacity(self.events.len().saturating_mul(128));
        for event in &self.events {
            format_jsonl_event(event, &escaped_names, &mut out);
            out.push('\n');
        }
        out
    }

    /// Serializes the trace in Chrome Trace Event JSON format, viewable in
    /// Perfetto UI (`https://ui.perfetto.dev`) and `chrome://tracing`.
    ///
    /// Each agent is emitted as a labeled thread track with instant lifecycle
    /// markers, complete (`X`) duration slices for sleep intervals, and
    /// `queue_depth` counter tracks.
    ///
    /// # Errors
    ///
    /// Returns any underlying I/O error from `writer`.
    pub fn write_perfetto_json(&self, writer: &mut impl io::Write) -> io::Result<()> {
        let escaped_names = self.escaped_agent_names();
        let mut buf = String::with_capacity(WRITE_FLUSH_THRESHOLD + 256);
        append_perfetto_header(&escaped_names, &mut buf);

        for event in &self.events {
            append_perfetto_event_and_metrics(event, &escaped_names, &mut buf);
            if buf.len() >= WRITE_FLUSH_THRESHOLD {
                writer.write_all(buf.as_bytes())?;
                buf.clear();
            }
        }

        buf.push_str("\n]}\n");
        writer.write_all(buf.as_bytes())
    }

    /// Serializes the trace as a Chrome Trace / Perfetto JSON string.
    #[must_use]
    pub fn to_perfetto_json(&self) -> String {
        let escaped_names = self.escaped_agent_names();
        let mut out = String::with_capacity(self.events.len().saturating_mul(160) + 128);
        append_perfetto_header(&escaped_names, &mut out);
        for event in &self.events {
            append_perfetto_event_and_metrics(event, &escaped_names, &mut out);
        }
        out.push_str("\n]}\n");
        out
    }

    /// Pre-escapes every agent name as a JSON string literal once per export.
    fn escaped_agent_names(&self) -> Vec<String> {
        self.agent_names
            .iter()
            .map(|name| {
                let mut escaped = String::with_capacity(name.len() + 2);
                push_json_string(&mut escaped, name);
                escaped
            })
            .collect()
    }
}

/// Resolves an escaped JSON agent name literal by identity.
#[inline]
fn escaped_name(escaped_names: &[String], id: AgentId) -> &str {
    escaped_names
        .get(id.index())
        .map_or(UNKNOWN_JSON_NAME, String::as_str)
}

/// Formats one event as a single JSON object into `out`.
fn format_jsonl_event(event: &TraceEvent, escaped_names: &[String], out: &mut String) {
    out.push_str("{\"time\":");
    push_u64(out, event.time);
    out.push_str(",\"agent_id\":");
    push_usize(out, event.agent.index());
    out.push_str(",\"agent\":");
    out.push_str(escaped_name(escaped_names, event.agent));
    match &event.kind {
        TraceEventKind::MessageSent {
            destination,
            arrival,
            payload,
        } => {
            out.push_str(",\"event\":\"message_sent\"");
            if let Some(dst) = destination {
                out.push_str(",\"destination_id\":");
                push_usize(out, dst.index());
                out.push_str(",\"destination\":");
                out.push_str(escaped_name(escaped_names, *dst));
            } else {
                out.push_str(",\"destination_id\":null,\"destination\":null");
            }
            out.push_str(",\"arrival\":");
            push_u64(out, *arrival);
            if let Some(text) = payload {
                out.push_str(",\"payload\":");
                push_json_string(out, text);
            }
        }
        TraceEventKind::TimerCancelled { timer } => {
            out.push_str(",\"event\":\"timer_cancelled\",\"arrival\":");
            push_u64(out, timer.arrival());
        }
        TraceEventKind::MessageDelivered {
            source,
            queue_depth,
        } => {
            out.push_str(",\"event\":\"message_delivered\"");
            append_optional_source(out, *source, escaped_names);
            out.push_str(",\"queue_depth\":");
            push_usize(out, *queue_depth);
        }
        TraceEventKind::MessageConsumed {
            source,
            queued_time,
            queue_depth,
        } => {
            out.push_str(",\"event\":\"message_consumed\"");
            append_optional_source(out, *source, escaped_names);
            out.push_str(",\"queued_time\":");
            push_u64(out, *queued_time);
            out.push_str(",\"queue_depth\":");
            push_usize(out, *queue_depth);
        }
        TraceEventKind::AgentSlept { until } => {
            out.push_str(",\"event\":\"agent_slept\",\"until\":");
            push_u64(out, *until);
        }
        TraceEventKind::AgentWoke => {
            out.push_str(",\"event\":\"agent_woke\"");
        }
        TraceEventKind::HaltInterrupt { reason } => {
            out.push_str(",\"event\":\"halt_interrupt\",\"reason\":");
            push_json_string(out, reason);
        }
    }
    out.push('}');
}

/// Appends the Perfetto trace header and process/thread metadata entries.
fn append_perfetto_header(escaped_names: &[String], out: &mut String) {
    out.push_str(
        "{\"traceEvents\":[\n{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,\"args\":{\"name\":\"simul\"}}",
    );
    for (index, name) in escaped_names.iter().enumerate() {
        out.push_str(",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":");
        push_usize(out, index);
        out.push_str(",\"args\":{\"name\":");
        out.push_str(name);
        out.push_str("}}");
    }
}

/// Appends one comma-prefixed Perfetto event and optional `queue_depth` counter.
fn append_perfetto_event_and_metrics(
    event: &TraceEvent,
    escaped_names: &[String],
    out: &mut String,
) {
    out.push_str(",\n");
    format_perfetto_event(event, escaped_names, out);
    if let Some(depth) = queue_depth_of(&event.kind) {
        out.push_str(",\n{\"name\":\"queue_depth\",\"cat\":\"metrics\",\"ph\":\"C\",\"ts\":");
        push_u64(out, event.time);
        out.push_str(",\"pid\":1,\"tid\":");
        push_usize(out, event.agent.index());
        out.push_str(",\"args\":{");
        out.push_str(escaped_name(escaped_names, event.agent));
        out.push(':');
        push_usize(out, depth);
        out.push_str("}}");
    }
}

/// Appends the shared Perfetto event prefix up through `tid`.
#[inline]
fn push_perfetto_prefix(out: &mut String, head: &str, ts: DiscreteTime, tid: usize) {
    out.push_str(head);
    push_u64(out, ts);
    out.push_str(",\"pid\":1,\"tid\":");
    push_usize(out, tid);
}

/// Appends optional Perfetto `"source"` field and closes the event's `args`.
#[inline]
fn finish_perfetto_source(out: &mut String, source: Option<AgentId>, escaped_names: &[String]) {
    if let Some(src) = source {
        out.push_str(",\"source\":");
        out.push_str(escaped_name(escaped_names, src));
    }
    out.push_str("}}");
}

/// Formats one event as a Chrome Trace JSON object into `out`.
fn format_perfetto_event(event: &TraceEvent, escaped_names: &[String], out: &mut String) {
    const SEND_HEAD: &str =
        "{\"name\":\"send\",\"cat\":\"message\",\"ph\":\"i\",\"s\":\"t\",\"ts\":";
    const CANCEL_HEAD: &str =
        "{\"name\":\"cancel_timer\",\"cat\":\"timer\",\"ph\":\"i\",\"s\":\"t\",\"ts\":";
    const DELIVER_HEAD: &str =
        "{\"name\":\"deliver\",\"cat\":\"message\",\"ph\":\"i\",\"s\":\"t\",\"ts\":";
    const CONSUME_HEAD: &str =
        "{\"name\":\"consume\",\"cat\":\"message\",\"ph\":\"i\",\"s\":\"t\",\"ts\":";
    const WAKE_HEAD: &str =
        "{\"name\":\"wake\",\"cat\":\"lifecycle\",\"ph\":\"i\",\"s\":\"t\",\"ts\":";
    const HALT_HEAD: &str =
        "{\"name\":\"halt_interrupt\",\"cat\":\"control\",\"ph\":\"i\",\"s\":\"g\",\"ts\":";

    let tid = event.agent.index();
    let ts = event.time;
    match &event.kind {
        TraceEventKind::MessageSent {
            destination,
            arrival,
            payload,
        } => {
            push_perfetto_prefix(out, SEND_HEAD, ts, tid);
            out.push_str(",\"args\":{\"arrival\":");
            push_u64(out, *arrival);
            if let Some(dst) = destination {
                out.push_str(",\"destination\":");
                out.push_str(escaped_name(escaped_names, *dst));
            }
            if let Some(text) = payload {
                out.push_str(",\"payload\":");
                push_json_string(out, text);
            }
            out.push_str("}}");
        }
        TraceEventKind::TimerCancelled { timer } => {
            push_perfetto_prefix(out, CANCEL_HEAD, ts, tid);
            out.push_str(",\"args\":{\"arrival\":");
            push_u64(out, timer.arrival());
            out.push_str("}}");
        }
        TraceEventKind::MessageDelivered {
            source,
            queue_depth,
        } => {
            push_perfetto_prefix(out, DELIVER_HEAD, ts, tid);
            out.push_str(",\"args\":{\"queue_depth\":");
            push_usize(out, *queue_depth);
            finish_perfetto_source(out, *source, escaped_names);
        }
        TraceEventKind::MessageConsumed {
            source,
            queued_time,
            queue_depth,
        } => {
            push_perfetto_prefix(out, CONSUME_HEAD, ts, tid);
            out.push_str(",\"args\":{\"queued_time\":");
            push_u64(out, *queued_time);
            out.push_str(",\"wait\":");
            push_u64(out, ts.saturating_sub(*queued_time));
            out.push_str(",\"queue_depth\":");
            push_usize(out, *queue_depth);
            finish_perfetto_source(out, *source, escaped_names);
        }
        TraceEventKind::AgentSlept { until } => {
            out.push_str("{\"name\":\"sleep\",\"cat\":\"lifecycle\",\"ph\":\"X\",\"ts\":");
            push_u64(out, ts);
            out.push_str(",\"dur\":");
            push_u64(out, until.saturating_sub(ts));
            out.push_str(",\"pid\":1,\"tid\":");
            push_usize(out, tid);
            out.push_str(",\"args\":{\"until\":");
            push_u64(out, *until);
            out.push_str("}}");
        }
        TraceEventKind::AgentWoke => {
            push_perfetto_prefix(out, WAKE_HEAD, ts, tid);
            out.push('}');
        }
        TraceEventKind::HaltInterrupt { reason } => {
            push_perfetto_prefix(out, HALT_HEAD, ts, tid);
            out.push_str(",\"args\":{\"reason\":");
            push_json_string(out, reason);
            out.push_str("}}");
        }
    }
}

/// Internal trace recorder stored inside [`Simulation`](crate::Simulation) when
/// tracing is enabled.
#[derive(Clone, Debug)]
pub(crate) struct TraceRecorder<M> {
    /// Accumulated trace events and agent name table.
    pub(crate) trace: SimulationTrace,
    /// Whether recording is currently active.
    pub(crate) enabled: bool,
    /// Optional debug formatter for sent message payloads.
    pub(crate) format_payload: Option<fn(&M) -> String>,
}

impl<M> TraceRecorder<M> {
    /// Creates an active recorder for `agent_names`.
    pub(crate) const fn new(
        agent_names: Vec<String>,
        format_payload: Option<fn(&M) -> String>,
    ) -> Self {
        Self {
            trace: SimulationTrace::new(agent_names),
            enabled: true,
            format_payload,
        }
    }

    /// Formats `payload` if payload formatting is enabled.
    pub(crate) fn format(&self, payload: &M) -> Option<String> {
        self.format_payload.map(|f| f(payload))
    }
}

/// Formats `M` using [`std::fmt::Debug`] for payload-enabled traces.
pub(crate) fn debug_format_payload<M: std::fmt::Debug>(payload: &M) -> String {
    format!("{payload:?}")
}

/// Extracts the queue depth from events that observe a queue size change.
const fn queue_depth_of(kind: &TraceEventKind) -> Option<usize> {
    match kind {
        TraceEventKind::MessageDelivered { queue_depth, .. }
        | TraceEventKind::MessageConsumed { queue_depth, .. } => Some(*queue_depth),
        TraceEventKind::MessageSent { .. }
        | TraceEventKind::TimerCancelled { .. }
        | TraceEventKind::AgentSlept { .. }
        | TraceEventKind::AgentWoke
        | TraceEventKind::HaltInterrupt { .. } => None,
    }
}

/// Appends optional `source_id` and `source` fields to a JSONL event.
#[inline]
fn append_optional_source(out: &mut String, source: Option<AgentId>, escaped_names: &[String]) {
    if let Some(src) = source {
        out.push_str(",\"source_id\":");
        push_usize(out, src.index());
        out.push_str(",\"source\":");
        out.push_str(escaped_name(escaped_names, src));
    } else {
        out.push_str(",\"source_id\":null,\"source\":null");
    }
}

/// Appends a `usize` in base 10.
#[inline]
fn push_usize(out: &mut String, value: usize) {
    push_u64(out, u64::try_from(value).unwrap_or(u64::MAX));
}

/// Appends a `u64` in base 10 without invoking `std::fmt` machinery.
#[inline]
fn push_u64(out: &mut String, mut value: u64) {
    if value < 10 {
        out.push(char::from(b'0' + u8::try_from(value).unwrap_or(0)));
        return;
    }
    let mut buf = [0_u8; 20];
    let mut pos = buf.len();
    while value >= 100 {
        let rem = usize::try_from(value.rem_euclid(100)).unwrap_or(0) * 2;
        value = value.div_euclid(100);
        pos -= 2;
        buf[pos] = DEC_DIGITS_LUT[rem];
        buf[pos + 1] = DEC_DIGITS_LUT[rem + 1];
    }
    if value >= 10 {
        let rem = usize::try_from(value).unwrap_or(0) * 2;
        pos -= 2;
        buf[pos] = DEC_DIGITS_LUT[rem];
        buf[pos + 1] = DEC_DIGITS_LUT[rem + 1];
    } else {
        pos -= 1;
        buf[pos] = b'0' + u8::try_from(value).unwrap_or(0);
    }
    if let Ok(digits) = std::str::from_utf8(&buf[pos..]) {
        out.push_str(digits);
    }
}

/// Appends `value` as a valid RFC 8259 JSON string literal.
#[inline]
fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    let bytes = value.as_bytes();
    let mut start = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        if byte >= 0x20 && byte != b'"' && byte != b'\\' {
            continue;
        }
        if start < i {
            out.push_str(&value[start..i]);
        }
        match byte {
            b'"' => out.push_str(r#"\""#),
            b'\\' => out.push_str(r"\\"),
            0x08 => out.push_str(r"\b"),
            0x0C => out.push_str(r"\f"),
            b'\n' => out.push_str(r"\n"),
            b'\r' => out.push_str(r"\r"),
            b'\t' => out.push_str(r"\t"),
            control => {
                let hi = usize::from(control >> 4);
                let lo = usize::from(control & 0x0f);
                let esc = [b'\\', b'u', b'0', b'0', HEX_LOWER[hi], HEX_LOWER[lo]];
                if let Ok(text) = std::str::from_utf8(&esc) {
                    out.push_str(text);
                }
            }
        }
        start = i + 1;
    }
    if start < bytes.len() {
        out.push_str(&value[start..]);
    }
    out.push('"');
}

/// Tick window traced during a windowed forensic replay.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplayWindow {
    /// First tick at which structured tracing is enabled (earlier ticks run at
    /// full speed without tracing).
    pub start: DiscreteTime,
    /// Optional tick at which to pause after tracing; `None` runs to the
    /// simulation's halt condition or interrupt.
    pub end: Option<DiscreteTime>,
}

impl ReplayWindow {
    /// Creates a replay window tracing ticks starting at `start`, pausing at
    /// `end` if provided or running to completion otherwise.
    #[must_use]
    pub const fn new(start: DiscreteTime, end: Option<DiscreteTime>) -> Self {
        Self { start, end }
    }

    /// Creates a replay window tracing the `lookback` ticks leading up to
    /// `target_tick` and continuing until completion.
    #[must_use]
    pub const fn before(target_tick: DiscreteTime, lookback: DiscreteTime) -> Self {
        Self {
            start: target_tick.saturating_sub(lookback),
            end: None,
        }
    }

    /// Creates a replay window tracing `lookback` ticks before `target_tick`
    /// through `after` ticks past `target_tick`.
    #[must_use]
    pub const fn around(
        target_tick: DiscreteTime,
        lookback: DiscreteTime,
        after: DiscreteTime,
    ) -> Self {
        Self {
            start: target_tick.saturating_sub(lookback),
            end: Some(target_tick.saturating_add(after)),
        }
    }
}

/// Result of a windowed forensic replay.
#[derive(Clone, Debug)]
pub struct ForensicReport<M> {
    /// Final simulation state after replaying the window.
    pub simulation: Simulation<M>,
    /// Structured trace captured strictly during the replay window.
    pub trace: SimulationTrace,
    /// Tick at which tracing actually began.
    pub trace_started_at: DiscreteTime,
    /// Final tick reached by the simulation.
    pub ended_at: DiscreteTime,
}

/// Fast-forwards `simulation` untraced to `window.start`, enables structured
/// tracing, and runs through `window.end` (or completion).
///
/// ```
/// use simul::agent::{periodic_consumer, periodic_producer};
/// use simul::trace::{ReplayWindow, forensic_replay};
/// use simul::{Halt, Simulation};
///
/// let sim = Simulation::<()>::builder(Halt::AtTime(100))
///     .agents([
///         periodic_producer("producer", 5, "consumer"),
///         periodic_consumer("consumer", 5),
///     ])
///     .build()?;
/// let report = forensic_replay(sim, ReplayWindow::before(100, 15));
/// assert_eq!(85, report.trace_started_at);
/// assert_eq!(100, report.ended_at);
/// assert!(!report.trace.is_empty());
/// assert!(report.trace.events().iter().all(|e| e.time >= 85));
/// # Ok::<(), simul::SimulationError>(())
/// ```
#[must_use]
pub fn forensic_replay<M>(
    mut simulation: Simulation<M>,
    window: ReplayWindow,
) -> ForensicReport<M> {
    simulation.disable_trace();
    if window.start > simulation.time() {
        simulation.run_until(window.start);
    }
    let trace_started_at = simulation.time();
    simulation.enable_trace();
    finish_forensic_window(&mut simulation, window.end);
    let ended_at = simulation.time();
    let trace = simulation.take_trace().unwrap_or_default();
    ForensicReport {
        simulation,
        trace,
        trace_started_at,
        ended_at,
    }
}

/// Like [`forensic_replay`], also recording a `Debug` summary of each sent
/// message payload inside the trace window.
#[must_use]
pub fn forensic_replay_with_payloads<M: std::fmt::Debug>(
    mut simulation: Simulation<M>,
    window: ReplayWindow,
) -> ForensicReport<M> {
    simulation.disable_trace();
    if window.start > simulation.time() {
        simulation.run_until(window.start);
    }
    let trace_started_at = simulation.time();
    simulation.enable_trace_with_payloads();
    finish_forensic_window(&mut simulation, window.end);
    let ended_at = simulation.time();
    let trace = simulation.take_trace().unwrap_or_default();
    ForensicReport {
        simulation,
        trace,
        trace_started_at,
        ended_at,
    }
}

/// Reconstructs a trial from its [`ReplayKey`], builds the simulation via
/// `build`, and executes [`forensic_replay`] over `window`.
///
/// # Errors
///
/// Propagates any builder error returned by `build`.
pub fn forensic_replay_trial<M, E>(
    replay: ReplayKey,
    window: ReplayWindow,
    build: impl FnOnce(&TrialContext) -> Result<Simulation<M>, E>,
) -> Result<ForensicReport<M>, E> {
    let context = TrialContext::from_replay_key(replay);
    let simulation = build(&context)?;
    Ok(forensic_replay(simulation, window))
}

/// Like [`forensic_replay_trial`], also recording `Debug`-formatted payloads
/// inside the trace window.
///
/// # Errors
///
/// Propagates any builder error returned by `build`.
pub fn forensic_replay_trial_with_payloads<M: std::fmt::Debug, E>(
    replay: ReplayKey,
    window: ReplayWindow,
    build: impl FnOnce(&TrialContext) -> Result<Simulation<M>, E>,
) -> Result<ForensicReport<M>, E> {
    let context = TrialContext::from_replay_key(replay);
    let simulation = build(&context)?;
    Ok(forensic_replay_with_payloads(simulation, window))
}

/// Runs the active trace window to `end` or completion when the simulation has
/// not already completed during fast-forward.
fn finish_forensic_window<M>(simulation: &mut Simulation<M>, end: Option<DiscreteTime>) {
    if simulation.mode() != SimulationMode::Completed {
        if let Some(until) = end {
            simulation.run_until(until);
        } else {
            simulation.run();
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::missing_docs_in_private_items)]
mod tests {
    use super::*;

    #[test]
    fn json_string_escaping_handles_control_chars_quotes_and_backslashes() {
        let mut out = String::new();
        push_json_string(&mut out, "a\"b\\c\n\t\x01");
        assert_eq!(r#""a\"b\\c\n\t\u0001""#, out);
    }

    #[test]
    fn push_u64_and_json_escaping_cover_boundaries_and_utf8() {
        for value in [
            0_u64,
            1,
            9,
            10,
            11,
            99,
            100,
            999,
            1000,
            9999,
            10_000,
            123_456_789_012_345,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let mut out = String::new();
            push_u64(&mut out, value);
            assert_eq!(value.to_string(), out);
        }

        let mut all_controls = String::new();
        for b in 0_u8..0x20 {
            all_controls.push(char::from(b));
        }
        all_controls.push_str("café\"\\日本語\x08\x0c\r\n\t");
        let mut escaped = String::new();
        push_json_string(&mut escaped, &all_controls);
        assert!(escaped.starts_with(r#""\u0000\u0001"#));
        assert!(escaped.ends_with(r#"café\"\\日本語\b\f\r\n\t""#));
    }

    #[test]
    fn jsonl_and_perfetto_exports_are_well_formed_and_filterable() {
        let mut trace = SimulationTrace::new(vec!["client".into(), "server".into()]);
        trace.push(
            0,
            AgentId::new(0),
            TraceEventKind::MessageSent {
                destination: Some(AgentId::new(1)),
                arrival: 1,
                payload: Some("Req(\"hi\")".into()),
            },
        );
        trace.push(
            1,
            AgentId::new(1),
            TraceEventKind::MessageDelivered {
                source: Some(AgentId::new(0)),
                queue_depth: 1,
            },
        );
        trace.push(
            1,
            AgentId::new(1),
            TraceEventKind::MessageConsumed {
                source: Some(AgentId::new(0)),
                queued_time: 0,
                queue_depth: 0,
            },
        );
        trace.push(1, AgentId::new(1), TraceEventKind::AgentSlept { until: 5 });
        trace.push(5, AgentId::new(1), TraceEventKind::AgentWoke);
        trace.push(
            5,
            AgentId::new(0),
            TraceEventKind::HaltInterrupt {
                reason: "done".into(),
            },
        );

        assert_eq!(6, trace.len());
        assert_eq!(3, trace.events_in_window(1, 5).count());
        assert_eq!(2, trace.events_for_agent(AgentId::new(0)).count());

        let jsonl = trace.to_jsonl();
        assert_eq!(6, jsonl.lines().count());
        assert!(jsonl.contains(r#""event":"message_sent""#));
        assert!(jsonl.contains(r#""payload":"Req(\"hi\")""#));
        assert!(jsonl.contains(r#""event":"halt_interrupt","reason":"done""#));

        let perfetto = trace.to_perfetto_json();
        assert!(perfetto.starts_with("{\"traceEvents\":[\n"));
        assert!(perfetto.ends_with("\n]}\n"));
        assert!(perfetto.contains(r#""thread_name""#));
        assert!(perfetto.contains(r#""queue_depth""#));
        assert!(perfetto.contains(r#""ph":"X","ts":1,"dur":4"#));
    }

    #[test]
    fn forensic_replay_trial_reproduces_rare_failure_window_bit_for_bit() {
        use crate::agent::{Agent, AgentContext, AgentInitializer};
        use crate::experiment::{CandidateId, FixedReplications, MasterSeed, run_serial};
        use crate::{Halt, SimulationError};

        #[derive(Clone, Debug)]
        struct FragileAgent {
            threshold: f64,
        }

        impl Agent<u32> for FragileAgent {
            fn on_tick(&mut self, ctx: &mut AgentContext<'_, u32>) {
                let sample = ctx.rng().uniform_f64();
                ctx.send_to(ctx.id(), u32::from(sample < self.threshold));
                if sample < self.threshold {
                    ctx.send_halt_interrupt("rare_spike");
                }
            }
        }

        let build = |ctx: &TrialContext| -> Result<Simulation<u32>, SimulationError> {
            Simulation::builder(Halt::AtTime(500))
                .trial(ctx)
                .agent(AgentInitializer::proactive(
                    "fragile",
                    FragileAgent { threshold: 0.005 },
                ))
                .build()
        };

        let seed = MasterSeed::from_u64(42);
        let records = run_serial(
            seed,
            FixedReplications::new(20, 1).trials(CandidateId::new(0), ()),
            |(), ctx| {
                let mut sim = build(ctx)?;
                sim.run();
                Ok::<(DiscreteTime, bool), SimulationError>((
                    sim.time(),
                    sim.halt_reason() == Some("rare_spike"),
                ))
            },
        )
        .expect("serial sweep");

        let failed = records
            .iter()
            .find(|r| matches!(r.result, Ok((_, true))))
            .expect("at least one replication should hit rare_spike");
        let Ok((fail_tick, true)) = failed.result else {
            panic!("expected failed tick");
        };

        let report = forensic_replay_trial_with_payloads(
            failed.replay,
            ReplayWindow::before(fail_tick, 5),
            build,
        )
        .expect("forensic replay");

        assert_eq!(fail_tick, report.ended_at);
        assert_eq!(Some("rare_spike"), report.simulation.halt_reason());
        assert!(
            report
                .trace
                .events()
                .iter()
                .all(|e| e.time >= report.trace_started_at)
        );
        assert!(report.trace.events().iter().any(|e| matches!(
            &e.kind,
            TraceEventKind::HaltInterrupt { reason } if reason == "rare_spike"
        )));
    }
}
