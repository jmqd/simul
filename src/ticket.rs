/// A ticket is a message between Agents that captures the necessary data to
/// perform the interaction.
#[derive(Default, Debug, Clone)]
pub struct Ticket {
    pub queued_time: u64,
    pub completed_time: Option<u64>,
    pub source: String,
    pub destination: String,
}

impl Ticket {
    pub fn new<S>(time: u64, src: S, dst: S) -> Ticket
    where
        S: Into<String>,
    {
        Ticket {
            queued_time: time,
            completed_time: None,
            source: src.into(),
            destination: dst.into(),
        }
    }
}
