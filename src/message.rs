/// A message is an interaction between Agents w/ the necessary data to perform
/// the interaction.
#[derive(Default, Debug, Clone)]
pub struct Message {
    pub queued_time: u64,
    pub completed_time: Option<u64>,
    pub source: String,
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
