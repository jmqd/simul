enum AgentState {
    Active,
    Sleeping,
    Dead,
}

struct PeriodicAgent {
    period: u32,
    state: AgentState,
}

struct Simulation <'s> {
    agents: Vec<StatefulAgent<'s>>,
    halt_condition: bool
}

struct StatefulAgent<'s> {
    agent: &'s PeriodicAgent,
    action_count: u64
}

impl Simulation <'s> {
    fn new(agents: Vec<PeriodicAgent>, halt_condition: bool) -> Simulation <'s> {
        Simulation {
            agents: agents.iter().map(|a| StatefulAgent {agent: a, action_count: 0}).collect(),
            halt_condition: halt_condition
        }
    }

    fn run() {
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let simulation = Simulation { agents: vec![], halt_condition: false };
        assert_eq!(Some(simulation).is_some(), true);
    }
}
