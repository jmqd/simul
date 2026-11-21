use crate::{AgentId, DiscreteTime};

/// A Message represents an interaction between Agents.
///
/// `M` is the simulation's message type: every agent sends and receives `M`
/// payloads, so a model matches on its own enum instead of decoding bytes.
/// Messages identify agents by [`AgentId`]. A payload-free `()` message
/// allocates nothing; share large payloads by choosing a shared `M`, such as
/// `Arc<T>`. Use [`Message::new`] rather than a struct literal.
#[derive(Default, Debug, Clone, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub struct Message<M = ()> {
    /// For ordinary sends, the sender's tick (delivered at the end of that
    /// tick); for [`AgentContext::send_after`](crate::agent::AgentContext::send_after),
    /// `time + delay - 1`; for initial-queue messages, the time passed to
    /// [`Message::new`].
    pub queued_time: DiscreteTime,
    /// `None` until [`Agent::on_message`](crate::agent::Agent::on_message)
    /// finishes the message with
    /// [`MessageProcessingStatus::Done`](crate::agent::MessageProcessingStatus::Done),
    /// then the tick it was consumed.
    pub completed_time: Option<DiscreteTime>,
    /// The sending agent, or `None` for a message supplied in an initial queue.
    pub source: Option<AgentId>,
    /// The addressed agent, or `None` when the target matched no agent.
    pub destination: Option<AgentId>,
    /// The application-defined content.
    pub payload: M,
}

// Every send moves messages through queues and histories several times, so
// keep the per-message overhead from growing unnoticed.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<Message>() == 32);

impl<M> Message<M> {
    /// Creates a message carrying `payload`, such as an initial queue entry.
    #[must_use]
    pub const fn new(time: DiscreteTime, payload: M) -> Self {
        Self {
            queued_time: time,
            completed_time: None,
            source: None,
            destination: None,
            payload,
        }
    }

    /// Creates a message sent by `source` during a callback.
    #[inline]
    pub(crate) const fn sent(
        time: DiscreteTime,
        source: AgentId,
        destination: Option<AgentId>,
        payload: M,
    ) -> Self {
        Self {
            queued_time: time,
            completed_time: None,
            source: Some(source),
            destination,
            payload,
        }
    }
}
