use crate::{AgentId, DiscreteTime};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum Interrupt {
    /// Immediately halt the simulation (with some reason why).
    HaltSimulation(String),
}

/// A Message represents an interaction between Agents.
///
/// Messages identify agents by [`AgentId`], so sending one without a payload
/// allocates nothing. Clones share payloads and interrupts while keeping
/// timestamps and identities independent. Mutating shared contents uses
/// copy-on-write, so changing a clone never changes its original. Use the
/// constructors rather than a struct literal.
#[derive(Default, Debug, Clone)]
pub struct Message {
    /// When the message was first created and put onto a queue.
    pub queued_time: DiscreteTime,
    /// When the Message was consumed and processed.
    pub completed_time: Option<DiscreteTime>,
    /// The sending agent, or `None` for a message supplied in an initial queue.
    pub source: Option<AgentId>,
    /// The addressed agent, or `None` when the target matched no agent.
    pub destination: Option<AgentId>,
    /// Shared contents, absent until a message has a payload or interrupt.
    body: Option<Arc<MessageBody>>,
}

/// Message contents exposed through [`Message`]'s field access.
#[derive(Default, Debug, Clone)]
pub struct MessageBody {
    /// Optional application-defined bytes.
    pub custom_payload: Option<Vec<u8>>,
    /// A control interrupt to bubble up to the Simulation engine.
    pub interrupt: Option<Interrupt>,
}

/// Contents read through messages that have no shared body.
static EMPTY_BODY: MessageBody = MessageBody {
    custom_payload: None,
    interrupt: None,
};

impl Deref for Message {
    type Target = MessageBody;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.body.as_deref().unwrap_or(&EMPTY_BODY)
    }
}

impl DerefMut for Message {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(self.body.get_or_insert_with(Arc::default))
    }
}

impl Message {
    /// Creates a message without a payload, such as an initial queue entry.
    #[must_use]
    pub const fn new(time: DiscreteTime) -> Self {
        Self {
            queued_time: time,
            completed_time: None,
            source: None,
            destination: None,
            body: None,
        }
    }

    /// Creates a message, taking ownership of its payload without copying bytes.
    #[must_use]
    pub fn new_with_payload(time: DiscreteTime, payload: Option<Vec<u8>>) -> Self {
        let mut message = Self::new(time);
        if payload.is_some() {
            message.body = Some(Arc::new(MessageBody {
                custom_payload: payload,
                interrupt: None,
            }));
        }
        message
    }

    /// Creates a message sent by `source` during a callback.
    #[inline]
    pub(crate) fn sent(
        time: DiscreteTime,
        source: AgentId,
        destination: Option<AgentId>,
        payload: Option<Vec<u8>>,
    ) -> Self {
        let mut message = Self::new_with_payload(time, payload);
        message.source = Some(source);
        message.destination = destination;
        message
    }
}
