pub mod agent;
pub mod experiment;
pub mod message;

use agent::*;
use log::debug;
use message::*;
use std::collections::HashMap;

/// The current state of a Simulation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum SimulationState {
    /// The Simulation has only been constructed.
    Constructed,
    /// The Simulation is actively simulating.
    Running,
    /// The Simulation successfully reached the halt condition.
    Completed,
    /// The Simulation catastrophically crashed.
    Failed,
}

/// A Simulation struct holds all the state for any given simulation.
///
/// A Simulation is an engine that advances through its own discrete definition
/// of time and has a collection of Agents that live in its domain.
///
/// At each moment in time, it asks each Agent whether it has any action to
/// perform.
///
/// The Simulation engine uses a concept of `Messages` to communicate between
/// agents. Agents can receive messages and send messages to other Agents.
#[derive(Clone, Debug)]
pub struct Simulation {
    /// The agents within the simulation, e.g. adaptive agents.
    /// See here: https://authors.library.caltech.edu/60491/1/MGM%20113.pdf
    pub agents: Vec<Agent>,
    /// A halt check function: given the state of the Simulation determine halt or not.
    pub halt_check: fn(&Simulation) -> bool,
    /// The current discrete time of the Simulation.
    pub time: u64,
    /// Whether to record metrics on queue depths. Takes space.
    pub enable_queue_depth_metrics: bool,
    /// Space to store queue depth metrics. Maps from Agent to a Vec<Time, Depth>
    pub queue_depth_metrics: HashMap<String, Vec<usize>>,
    /// The state of the Simulation.
    pub state: SimulationState,
}

/// The parameters to create a Simulation.
#[derive(Clone, Debug)]
pub struct SimulationParameters {
    /// The agents within the simulation, e.g. adaptive agents.
    /// See here: https://authors.library.caltech.edu/60491/1/MGM%20113.pdf
    pub agents: Vec<Agent>,
    /// Given the state of the Simulation a function that determines if the Simulation is complete.
    pub halt_check: fn(&Simulation) -> bool,
    /// The discrete time at which the simulation should begin.
    /// For the vast majority of simulations, 0 is the correct default.
    pub starting_time: u64,
    /// Whether to record metrics on queue depths at every tick of the simulation.
    /// Takes time and space.
    pub enable_queue_depth_telemetry: bool,
}

impl Simulation {
    pub fn new(parameters: SimulationParameters) -> Simulation {
        Simulation {
            state: SimulationState::Constructed,
            queue_depth_metrics: parameters
                .agents
                .iter()
                .map(|a| (a.name.to_owned(), vec![]))
                .collect(),
            agents: parameters.agents,
            halt_check: parameters.halt_check,
            time: parameters.starting_time,
            enable_queue_depth_metrics: parameters.enable_queue_depth_telemetry,
        }
    }

    /// Finds an agent in the simulation and return a copy.
    pub fn find_agent(&self, name: &str) -> Option<Agent> {
        self.agents.iter().find(|a| a.name == name).cloned()
    }

    /// Returns the consumed messages for a given Agent during the Simulation.
    pub fn consumed_for_agent(&self, name: &str) -> Option<Vec<Message>> {
        let agent = self.agents.iter().find(|a| a.name == name)?;
        Some(agent.consumed.clone())
    }

    /// Returns the produced messages for a given Agent during the Simulation.
    pub fn produced_for_agent(&self, name: &str) -> Option<Vec<Message>> {
        let agent = self.agents.iter().find(|a| a.name == name)?;
        Some(agent.produced.clone())
    }

    /// Returns the queue depth timeseries for a given Agent during the Simulation.
    pub fn queue_depth_metrics(&self, agent_name: &str) -> Option<Vec<usize>> {
        self.queue_depth_metrics.get(agent_name).cloned()
    }

    /// Runs the simulation. This should only be called after adding all the beginning state.
    pub fn run(&mut self) {
        self.state = SimulationState::Running;

        while !(self.halt_check)(self) {
            debug!("Running next tick of simulation at time {}", self.time);
            let mut message_bus = vec![];
            self.wakeup_agents_scheduled_to_wakeup_now();
            for mut agent in self.agents.iter_mut() {
                if self.enable_queue_depth_metrics {
                    self.queue_depth_metrics
                        .get_mut(&agent.name)
                        .expect("Failed to find agent in metrics")
                        .push(agent.queue.len());
                }

                match agent.state {
                    AgentState::Active => match (agent.consumption_fn)(&mut agent, self.time) {
                        Some(messages) => {
                            message_bus.extend(messages);
                        }
                        None => debug!("No messages produced."),
                    },
                    AgentState::Dead | AgentState::AsleepUntil(_) => {}
                }
            }

            // Consume all the new messages in the bus and deliver to agents.
            self.disperse_bus_messages_to_agents(message_bus);

            debug!("Finished this tick; incrementing time.");
            self.time += 1;
        }

        self.state = SimulationState::Completed;
        self.emit_completed_simulation_debug_logging();
    }

    /// A helper to calculate the average waiting time to process items.
    /// Note: This function will likely go away; it is an artifact of prototyping.
    pub fn calc_avg_wait_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();
        for agent in self.agents.iter() {
            if agent.consumed.len() == 0 {
                continue;
            }

            let mut sum_of_times: u64 = 0;
            for completed in agent.consumed.iter() {
                sum_of_times += completed.completed_time.unwrap() - completed.queued_time;
            }

            data.insert(
                agent.name.clone(),
                sum_of_times as usize / agent.consumed.len(),
            );
        }

        data
    }

    /// Calculates the statistics of queue lengths.
    /// Mostly useful for checking which agents still have queues of work after halting.
    pub fn calc_queue_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(agent.name.clone(), agent.queue.len());
        }

        data
    }

    /// Calculates the length of the consumed messages for each Agent.
    pub fn calc_consumed_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(agent.name.clone(), agent.consumed.len());
        }

        data
    }

    /// Calculates the length of the produced messages for each Agent.
    pub fn calc_produced_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(agent.name.clone(), agent.produced.len());
        }

        data
    }

    fn emit_completed_simulation_debug_logging(&self) {
        let queue_len_stats = self.calc_queue_len_statistics();
        let consumed_len_stats = self.calc_consumed_len_statistics();
        let avg_wait_stats = self.calc_avg_wait_statistics();
        let produced_len_stats = self.calc_produced_len_statistics();

        debug!("Queues: {:?}", queue_len_stats);
        debug!("Consumed: {:?}", consumed_len_stats);
        debug!("Produced: {:?}", produced_len_stats);
        debug!("Average processing time: {:?}", avg_wait_stats);
    }

    /// Consume a message_bus of messages and disperse those messages to the agents.
    fn disperse_bus_messages_to_agents(&mut self, mut message_bus: Vec<Message>) {
        while let Some(message) = message_bus.pop() {
            for agent in self.agents.iter_mut() {
                if agent.name == message.clone().destination {
                    agent.push_message(message.clone());
                }

                if agent.name == message.clone().source {
                    agent.produced.push(message.clone());
                }
            }
        }
    }

    /// An internal function used to wakeup sleeping Agents due to wake.
    fn wakeup_agents_scheduled_to_wakeup_now(&mut self) {
        for agent in self.agents.iter_mut() {
            match agent.state {
                AgentState::AsleepUntil(scheduled_wakeup) => {
                    if self.time >= scheduled_wakeup {
                        agent.state = AgentState::Active;
                    }
                }
                _ => (),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Poisson;
    use std::collections::VecDeque;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn basic_periodic_test() {
        init();
        let mut simulation = Simulation::new(SimulationParameters {
            agents: vec![
                periodic_producing_agent("producer", 1, "consumer"),
                periodic_consuming_agent("consumer", 1),
            ],
            starting_time: 0,
            enable_queue_depth_telemetry: false,
            halt_check: |s: &Simulation| s.time == 5,
        });
        simulation.run();
        let produced_stats = simulation.calc_produced_len_statistics();
        assert_eq!(produced_stats.get("producer"), Some(&5));
        assert_eq!(produced_stats.get("consumer"), Some(&0));

        let consumed_stats = simulation.calc_consumed_len_statistics();
        assert_eq!(consumed_stats.get("producer"), Some(&0));
        assert_eq!(consumed_stats.get("consumer"), Some(&4));
    }

    #[test]
    fn starbucks_clerk() {
        init();
        let mut simulation = Simulation::new(SimulationParameters {
            agents: vec![
                Agent {
                    queue: VecDeque::with_capacity(8),
                    state: AgentState::Active,
                    name: "Starbucks Clerk".to_owned(),
                    consumed: vec![],
                    produced: vec![],
                    consumption_fn: |a: &mut Agent, t: u64| {
                        debug!("{} looking for a customer.", a.name);
                        if let Some(last) = a.consumed.last() {
                            if last.completed_time? + 60 > t {
                                debug!("Sorry, we're still serving the last customer.");
                                return None;
                            }
                        }

                        if let Some(message) = a.queue.pop_front() {
                            if message.queued_time + 100 > t {
                                debug!("Still making your coffee, sorry!");
                                a.queue.push_front(message);
                                return None;
                            }

                            debug!("Serviced a customer!");
                            a.consumed.push(Message {
                                completed_time: Some(t),
                                ..message
                            });
                        }
                        return None;
                    },
                    extensions: None,
                },
                poisson_distributed_producing_agent(
                    "Starbucks Customers",
                    Poisson::new(80.0).unwrap(),
                    "Starbucks Clerk",
                ),
            ],
            starting_time: 1,
            enable_queue_depth_telemetry: false,
            halt_check: |s: &Simulation| s.time > 500,
        });
        simulation.run();
        assert_eq!(Some(simulation).is_some(), true);
    }
}
