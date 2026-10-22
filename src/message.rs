use crate::{AgentId, DiscreteTime};
use std::sync::Arc;

/// A Message represents an interaction between Agents.
///
/// Messages identify agents by [`AgentId`] and share payload bytes, so a send
/// without a payload allocates nothing, a new payload allocates once, and
/// forwarding a received payload only increments a reference count. Clones
/// share the payload while keeping timestamps and identities independent; edit
/// a payload with [`Arc::make_mut`], which copies it if others hold it. Use the
/// constructors rather than a struct literal.
#[derive(Default, Debug, Clone)]
#[non_exhaustive]
pub struct Message {
    /// When the message was first created and put onto a queue.
    pub queued_time: DiscreteTime,
    /// When the Message was consumed and processed.
    pub completed_time: Option<DiscreteTime>,
    /// The sending agent, or `None` for a message supplied in an initial queue.
    pub source: Option<AgentId>,
    /// The addressed agent, or `None` when the target matched no agent.
    pub destination: Option<AgentId>,
    /// Optional application-defined bytes, shared by clones.
    pub custom_payload: Option<Arc<[u8]>>,
}

// Every send moves messages through queues and histories several times, so
// keep the layout from growing unnoticed.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<Message>() == 48);

impl Message {
    /// Creates a message without a payload, such as an initial queue entry.
    #[must_use]
    pub const fn new(time: DiscreteTime) -> Self {
        Self::new_with_payload(time, None)
    }

    /// Creates a message carrying `payload` without copying its bytes.
    #[must_use]
    pub const fn new_with_payload(time: DiscreteTime, payload: Option<Arc<[u8]>>) -> Self {
        Self {
            queued_time: time,
            completed_time: None,
            source: None,
            destination: None,
            custom_payload: payload,
        }
    }

    /// Creates a message sent by `source` during a callback.
    #[inline]
    pub(crate) const fn sent(
        time: DiscreteTime,
        source: AgentId,
        destination: Option<AgentId>,
        payload: Option<Arc<[u8]>>,
    ) -> Self {
        Self {
            queued_time: time,
            completed_time: None,
            source: Some(source),
            destination,
            custom_payload: payload,
        }
    }
}
