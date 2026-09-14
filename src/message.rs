use crate::DiscreteTime;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum Interrupt {
    /// Immediately halt the simulation (with some reason why).
    HaltSimulation(String),
}

/// A Message represents an interaction between Agents.
///
/// Clones share message contents while keeping timestamps independent. Mutating
/// shared contents uses copy-on-write, so changing a clone never changes its
/// original. Use the constructors rather than a struct literal.
#[derive(Default, Debug, Clone)]
pub struct Message {
    /// When the message was first created and put onto a queue.
    pub queued_time: DiscreteTime,
    /// When the Message was consumed and processed.
    pub completed_time: Option<DiscreteTime>,
    /// Shared contents; delivery and history do not duplicate strings or payloads.
    body: Arc<MessageBody>,
}

/// Message contents exposed through [`Message`]'s field access.
#[derive(Default, Debug, Clone)]
pub struct MessageBody {
    /// The name of the Agent that created this Message.
    pub source: String,
    /// The name of the Agent that received this Message.
    pub destination: String,
    /// Optional application-defined bytes.
    pub custom_payload: Option<Vec<u8>>,
    /// A control interrupt to bubble up to the Simulation engine.
    pub interrupt: Option<Interrupt>,
}

impl Deref for Message {
    type Target = MessageBody;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.body
    }
}

impl DerefMut for Message {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.body)
    }
}

impl Message {
    /// Creates a message without a payload.
    #[must_use]
    pub fn new<S: Into<String>>(time: DiscreteTime, src: S, dst: S) -> Self {
        Self::new_with_payload(time, src, dst, None)
    }

    /// Creates a message, taking ownership of its payload without copying bytes.
    #[must_use]
    pub fn new_with_payload<S: Into<String>>(
        time: DiscreteTime,
        src: S,
        dst: S,
        payload: Option<Vec<u8>>,
    ) -> Self {
        Self {
            queued_time: time,
            completed_time: None,
            body: Arc::new(MessageBody {
                source: src.into(),
                destination: dst.into(),
                custom_payload: payload,
                interrupt: None,
            }),
        }
    }
}
