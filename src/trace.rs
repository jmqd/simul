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

use crate::DiscreteTime;
use crate::agent::{AgentId, TimerId};
use std::fmt::Write as _;
use std::io;

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
        let mut line = String::with_capacity(192);
        for event in &self.events {
            line.clear();
            self.format_jsonl_event(event, &mut line);
            line.push('\n');
            writer.write_all(line.as_bytes())?;
        }
        Ok(())
    }

    /// Serializes the trace as a `JSONL` string.
    #[must_use]
    pub fn to_jsonl(&self) -> String {
        let mut out = String::with_capacity(self.events.len().saturating_mul(128));
        for event in &self.events {
            self.format_jsonl_event(event, &mut out);
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
        writer.write_all(b"{\"traceEvents\":[\n")?;
        let mut first = true;
        let mut buf = String::with_capacity(256);

        // Process metadata.
        write_perfetto_entry(
            writer,
            &mut first,
            r#"{"name":"process_name","ph":"M","pid":1,"tid":0,"args":{"name":"simul"}}"#,
        )?;

        // Per-agent thread names so Perfetto labels each swimlane.
        for (index, name) in self.agent_names.iter().enumerate() {
            buf.clear();
            let _ = write!(
                buf,
                "{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":{index},\"args\":{{\"name\":"
            );
            push_json_string(&mut buf, name);
            buf.push_str("}}");
            write_perfetto_entry(writer, &mut first, &buf)?;
        }

        // Lifecycle events + queue-depth counter points.
        for event in &self.events {
            buf.clear();
            self.format_perfetto_event(event, &mut buf);
            write_perfetto_entry(writer, &mut first, &buf)?;

            if let Some(depth) = queue_depth_of(&event.kind) {
                buf.clear();
                let tid = event.agent.index();
                let name = self.agent_name(event.agent).unwrap_or("unknown");
                let _ = write!(
                    buf,
                    "{{\"name\":\"queue_depth\",\"cat\":\"metrics\",\"ph\":\"C\",\"ts\":{},\"pid\":1,\"tid\":{tid},\"args\":{{",
                    event.time
                );
                push_json_string(&mut buf, name);
                let _ = write!(buf, ":{depth}}}}}");
                write_perfetto_entry(writer, &mut first, &buf)?;
            }
        }

        writer.write_all(b"\n]}\n")
    }

    /// Serializes the trace as a Chrome Trace / Perfetto JSON string.
    #[must_use]
    pub fn to_perfetto_json(&self) -> String {
        let mut bytes = Vec::with_capacity(self.events.len().saturating_mul(160) + 128);
        let _ = self.write_perfetto_json(&mut bytes);
        String::from_utf8(bytes).unwrap_or_default()
    }

    /// Formats one event as a single JSON object into `out`.
    fn format_jsonl_event(&self, event: &TraceEvent, out: &mut String) {
        let agent_name = self.agent_name(event.agent).unwrap_or("unknown");
        let _ = write!(
            out,
            "{{\"time\":{},\"agent_id\":{},\"agent\":",
            event.time,
            event.agent.index()
        );
        push_json_string(out, agent_name);
        match &event.kind {
            TraceEventKind::MessageSent {
                destination,
                arrival,
                payload,
            } => {
                out.push_str(",\"event\":\"message_sent\"");
                if let Some(dst) = destination {
                    let _ = write!(out, ",\"destination_id\":{},\"destination\":", dst.index());
                    push_json_string(out, self.agent_name(*dst).unwrap_or("unknown"));
                } else {
                    out.push_str(",\"destination_id\":null,\"destination\":null");
                }
                let _ = write!(out, ",\"arrival\":{arrival}");
                if let Some(text) = payload {
                    out.push_str(",\"payload\":");
                    push_json_string(out, text);
                }
            }
            TraceEventKind::TimerCancelled { timer } => {
                let _ = write!(
                    out,
                    ",\"event\":\"timer_cancelled\",\"arrival\":{}",
                    timer.arrival()
                );
            }
            TraceEventKind::MessageDelivered {
                source,
                queue_depth,
            } => {
                out.push_str(",\"event\":\"message_delivered\"");
                append_optional_source(out, *source, self);
                let _ = write!(out, ",\"queue_depth\":{queue_depth}");
            }
            TraceEventKind::MessageConsumed {
                source,
                queued_time,
                queue_depth,
            } => {
                out.push_str(",\"event\":\"message_consumed\"");
                append_optional_source(out, *source, self);
                let _ = write!(
                    out,
                    ",\"queued_time\":{queued_time},\"queue_depth\":{queue_depth}"
                );
            }
            TraceEventKind::AgentSlept { until } => {
                let _ = write!(out, ",\"event\":\"agent_slept\",\"until\":{until}");
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

    /// Formats one event as a Chrome Trace JSON object into `out`.
    fn format_perfetto_event(&self, event: &TraceEvent, out: &mut String) {
        let tid = event.agent.index();
        let ts = event.time;
        match &event.kind {
            TraceEventKind::MessageSent {
                destination,
                arrival,
                payload,
            } => {
                let _ = write!(
                    out,
                    "{{\"name\":\"send\",\"cat\":\"message\",\"ph\":\"i\",\"s\":\"t\",\"ts\":{ts},\"pid\":1,\"tid\":{tid},\"args\":{{\"arrival\":{arrival}"
                );
                if let Some(dst) = destination {
                    out.push_str(",\"destination\":");
                    push_json_string(out, self.agent_name(*dst).unwrap_or("unknown"));
                }
                if let Some(text) = payload {
                    out.push_str(",\"payload\":");
                    push_json_string(out, text);
                }
                out.push_str("}}");
            }
            TraceEventKind::TimerCancelled { timer } => {
                let _ = write!(
                    out,
                    "{{\"name\":\"cancel_timer\",\"cat\":\"timer\",\"ph\":\"i\",\"s\":\"t\",\"ts\":{ts},\"pid\":1,\"tid\":{tid},\"args\":{{\"arrival\":{}}}}}",
                    timer.arrival()
                );
            }
            TraceEventKind::MessageDelivered {
                source,
                queue_depth,
            } => {
                let _ = write!(
                    out,
                    "{{\"name\":\"deliver\",\"cat\":\"message\",\"ph\":\"i\",\"s\":\"t\",\"ts\":{ts},\"pid\":1,\"tid\":{tid},\"args\":{{\"queue_depth\":{queue_depth}"
                );
                if let Some(src) = source {
                    out.push_str(",\"source\":");
                    push_json_string(out, self.agent_name(*src).unwrap_or("unknown"));
                }
                out.push_str("}}");
            }
            TraceEventKind::MessageConsumed {
                source,
                queued_time,
                queue_depth,
            } => {
                let wait = ts.saturating_sub(*queued_time);
                let _ = write!(
                    out,
                    "{{\"name\":\"consume\",\"cat\":\"message\",\"ph\":\"i\",\"s\":\"t\",\"ts\":{ts},\"pid\":1,\"tid\":{tid},\"args\":{{\"queued_time\":{queued_time},\"wait\":{wait},\"queue_depth\":{queue_depth}"
                );
                if let Some(src) = source {
                    out.push_str(",\"source\":");
                    push_json_string(out, self.agent_name(*src).unwrap_or("unknown"));
                }
                out.push_str("}}");
            }
            TraceEventKind::AgentSlept { until } => {
                let dur = until.saturating_sub(ts);
                let _ = write!(
                    out,
                    "{{\"name\":\"sleep\",\"cat\":\"lifecycle\",\"ph\":\"X\",\"ts\":{ts},\"dur\":{dur},\"pid\":1,\"tid\":{tid},\"args\":{{\"until\":{until}}}}}"
                );
            }
            TraceEventKind::AgentWoke => {
                let _ = write!(
                    out,
                    "{{\"name\":\"wake\",\"cat\":\"lifecycle\",\"ph\":\"i\",\"s\":\"t\",\"ts\":{ts},\"pid\":1,\"tid\":{tid}}}"
                );
            }
            TraceEventKind::HaltInterrupt { reason } => {
                let _ = write!(
                    out,
                    "{{\"name\":\"halt_interrupt\",\"cat\":\"control\",\"ph\":\"i\",\"s\":\"g\",\"ts\":{ts},\"pid\":1,\"tid\":{tid},\"args\":{{\"reason\":"
                );
                push_json_string(out, reason);
                out.push_str("}}");
            }
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
fn append_optional_source(out: &mut String, source: Option<AgentId>, trace: &SimulationTrace) {
    if let Some(src) = source {
        let _ = write!(out, ",\"source_id\":{},\"source\":", src.index());
        push_json_string(out, trace.agent_name(src).unwrap_or("unknown"));
    } else {
        out.push_str(",\"source_id\":null,\"source\":null");
    }
}

/// Writes one comma-separated entry into a Perfetto `traceEvents` array.
fn write_perfetto_entry(
    writer: &mut impl io::Write,
    first: &mut bool,
    entry: &str,
) -> io::Result<()> {
    if *first {
        *first = false;
    } else {
        writer.write_all(b",\n")?;
    }
    writer.write_all(entry.as_bytes())
}

/// Appends `value` as a valid RFC 8259 JSON string literal.
fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\u{08}' => out.push_str(r"\b"),
            '\u{0C}' => out.push_str(r"\f"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
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
}
