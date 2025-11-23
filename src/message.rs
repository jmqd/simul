use crate::DiscreteTime;

#[derive(Clone, Debug)]
pub enum Interrupt {
    /// Immediately halt the simulation (with some reason why).
    HaltSimulation(String),
}

/// A Message represents an interaction between Agents.
#[derive(Default, Debug, Clone)]
pub struct Message {
    /// When the message was first created and put onto a queue.
    pub queued_time: DiscreteTime,
    /// When the Message was consumed and processed.
    pub completed_time: Option<DiscreteTime>,
    /// The name of the Agent that created this Message.
    pub source: String,
    /// The name of the Agent that received this Message.
    pub destination: String,
    pub custom_payload: Option<Vec<u8>>,
    /// A control interrupt to bubble up to the Simulation engine.
    pub interrupt: Option<Interrupt>,
}

impl Message {
    pub fn new<S>(time: DiscreteTime, src: S, dst: S) -> Self
    where
        S: Into<String>,
    {
        Self {
            queued_time: time,
            completed_time: None,
            source: src.into(),
            destination: dst.into(),
            ..Default::default()
        }
    }
}
