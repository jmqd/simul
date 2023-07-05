/// A Message represents an interaction between Agents.
#[derive(Default, Debug, Clone)]
pub struct Message {
    /// When the message was first created and put onto a queue.
    pub queued_time: u64,
    /// When the Message was consumed and processed.
    pub completed_time: Option<u64>,
    /// The name of the Agent that created this Message.
    pub source: String,
    /// The name of the Agent that received this Message.
    pub destination: String,
}

impl Message {
    pub fn new<S>(time: u64, src: S, dst: S) -> Message
    where
        S: Into<String>,
    {
        Message {
            queued_time: time,
            completed_time: None,
            source: src.into(),
            destination: dst.into(),
        }
    }
}
