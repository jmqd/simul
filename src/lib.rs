pub mod agent;
pub mod experiment;
pub mod message;

use agent::*;
use log::{debug, info};
use message::*;
use std::collections::HashMap;

/// DiscreteTime is a Simulation's internal representation of time.
pub type DiscreteTime = u64;

/// The current mode of a Simulation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum SimulationMode {
    /// The Simulation has only been constructed.
    Constructed,
    /// The Simulation is actively simulating.
    Running,
    /// The Simulation successfully reached the halt condition.
    Completed,
    /// The Simulation catastrophically crashed.
    Failed,
}

/// State about the simulation that agents are aware of.
/// TODO: This may later just become the `Simulation` itself passed about.
#[derive(Clone, Debug)]
pub struct SimulationState {
    pub time: DiscreteTime,
    pub mode: SimulationMode,
}

/// A Simulation struct is responsible to hold all the state for a simulation
/// and coordinates the actions and interactions of the agents.
///
/// A Simulation has its own concept of time, which is implemented as discrete
/// ticks of the u64 field `time`. Every tick is modeled as an instantaneous
/// point in time at which interactions can occur. The Simulation engine uses a
/// concept of `Messages` to communicate between agents. Agents can receive
/// messages and send messages to other Agents.
#[derive(Clone, Debug)]
pub struct Simulation {
    /// The agents within the simulation, e.g. adaptive agents.
    pub agents: Vec<Box<dyn Agent>>,
    /// A halt check function: given the state of the Simulation determine halt or not.
    pub halt_check: fn(&Simulation) -> bool,
    /// The current discrete time of the Simulation.
    pub time: DiscreteTime,
    /// Whether to record metrics on queue depths. Takes space.
    pub enable_queue_depth_metrics: bool,
    /// Records a metric on the number of cycles an agent was asleep for.
    pub enable_agent_asleep_cycles_metric: bool,
    /// The mode of the Simulation.
    pub mode: SimulationMode,
    /// Maps from agent.state().id => a handle for indexing the Agent in the vec.
    agent_metadata_hash_table: HashMap<String, AgentMetadata>,

    pub current_ball: u8,
}

/// The parameters to create a Simulation.
#[derive(Clone, Debug)]
pub struct SimulationParameters {
    /// The agents within the simulation, e.g. adaptive agents.
    /// See here: https://authors.library.caltech.edu/60491/1/MGM%20113.pdf
    pub agents: Vec<Box<dyn Agent>>,
    /// Given the state of the Simulation a function that determines if the Simulation is complete.
    pub halt_check: fn(&Simulation) -> bool,
    /// The discrete time at which the simulation should begin.
    /// For the vast majority of simulations, 0 is the correct default.
    pub starting_time: DiscreteTime,
    /// Whether to record metrics on queue depths at every tick of the simulation.
    pub enable_queue_depth_metrics: bool,
    /// Records a metric on the number of cycles an agent was asleep for.
    pub enable_agent_asleep_cycles_metric: bool,
}

impl Default for SimulationParameters {
    fn default() -> Self {
        SimulationParameters {
            agents: vec![],
            halt_check: |_| true,
            starting_time: 0,
            enable_queue_depth_metrics: false,
            enable_agent_asleep_cycles_metric: false,
        }
    }
}

#[derive(Clone, Debug)]
struct AgentMetadata {
    queue_depth_metrics: Vec<usize>,
    asleep_cycle_count: DiscreteTime,
}

impl Simulation {
    pub fn new(parameters: SimulationParameters) -> Simulation {
        Simulation {
            mode: SimulationMode::Constructed,
            agent_metadata_hash_table: parameters
                .agents
                .iter()
                .map(|a| {
                    (
                        a.state().id.to_owned(),
                        AgentMetadata {
                            queue_depth_metrics: vec![],
                            asleep_cycle_count: 0,
                        },
                    )
                })
                .collect(),
            agents: parameters.agents,
            halt_check: parameters.halt_check,
            time: parameters.starting_time,
            enable_queue_depth_metrics: parameters.enable_queue_depth_metrics,
            enable_agent_asleep_cycles_metric: parameters.enable_agent_asleep_cycles_metric,
            current_ball: 1,
        }
    }

    /// Returns the consumed messages for a given Agent during the Simulation.
    pub fn consumed_for_agent(&self, name: &str) -> Option<Vec<Message>> {
        let agent = self.agents.iter().find(|a| a.state().id == name)?;
        Some(agent.state().consumed.clone())
    }

    /// Returns the produced messages for a given Agent during the Simulation.
    pub fn produced_for_agent(&self, name: &str) -> Option<Vec<Message>> {
        let agent = self.agents.iter().find(|a| a.state().id == name)?;
        Some(agent.state().produced.clone())
    }

    /// Returns the queue depth timeseries for a given Agent during the Simulation.
    pub fn queue_depth_metrics(&self, id: &str) -> Option<Vec<usize>> {
        // TODO(?): Return non option here.
        Some(
            self.agent_metadata_hash_table
                .get(id)?
                .queue_depth_metrics
                .clone(),
        )
    }

    /// Returns the asleep cycle count for a given Agent during the Simulation.
    pub fn asleep_cycle_count(&self, id: &str) -> Option<DiscreteTime> {
        // TODO(?): Return non option here.
        Some(self.agent_metadata_hash_table.get(id)?.asleep_cycle_count)
    }

    /// Runs the simulation. This should only be called after adding all the beginning state.
    pub fn run(&mut self) {
        self.mode = SimulationMode::Running;

        while !(self.halt_check)(self) {
            debug!("Running next tick of simulation at time {}", self.time);
            let mut message_bus = vec![];
            self.wakeup_agents_scheduled_to_wakeup_now();

            let tick_message = Message::new(self.time, "SIM_SRC".to_string(), "ANY".to_string());
            let simulation_state = SimulationState {
                time: self.time,
                mode: self.mode.clone(),
            };

            for agent in self.agents.iter_mut() {
                if self.enable_queue_depth_metrics {
                    self.agent_metadata_hash_table
                        .get_mut(&agent.state().id)
                        .expect("Failed to find agent in metrics")
                        .queue_depth_metrics
                        .push(agent.state().queue.len());
                }

                let queued_msg = agent.state_mut().queue.pop_front();

                match agent.state().mode {
                    AgentMode::Proactive => {
                        if let Some(messages) = agent.as_mut().process(
                            simulation_state.clone(),
                            queued_msg.as_ref().unwrap_or(&tick_message),
                        ) {
                            message_bus.extend(messages);
                        }
                    }
                    AgentMode::Reactive => {
                        if queued_msg.is_some() {
                            if let Some(new_msgs) = agent
                                .as_mut()
                                .process(simulation_state.clone(), &queued_msg.unwrap())
                            {
                                message_bus.extend(new_msgs);
                            }
                        }
                    }
                    AgentMode::AsleepUntil(_) => {
                        if self.enable_agent_asleep_cycles_metric {
                            self.agent_metadata_hash_table
                                .get_mut(&agent.state().id)
                                .expect("Failed to find agent in metrics")
                                .asleep_cycle_count += 1
                        }
                    }
                    AgentMode::Dead => {}
                }
            }

            // Consume all the new messages in the bus and deliver to agents.
            self.process_message_bus(message_bus);

            debug!("Finished this tick; incrementing time.");
            self.time += 1;
        }

        self.mode = SimulationMode::Completed;
        self.emit_completed_simulation_debug_logging();
    }

    /// A helper to calculate the average waiting time to process items.
    /// Note: This function will likely go away; it is an artifact of prototyping.
    pub fn calc_avg_wait_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();
        for agent in self
            .agents
            .iter()
            .filter(|a| !a.state().consumed.is_empty())
        {
            let mut sum_of_times: u64 = 0;
            for completed in agent.state().consumed.iter() {
                sum_of_times += completed.completed_time.unwrap() - completed.queued_time;
            }

            data.insert(
                agent.state().id.clone(),
                sum_of_times as usize / agent.state().consumed.len(),
            );
        }

        data
    }

    /// Calculates the statistics of queue lengths.
    /// Mostly useful for checking which agents still have queues of work after halting.
    pub fn calc_queue_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(agent.state().id.clone(), agent.state().queue.len());
        }

        data
    }

    /// Calculates the length of the consumed messages for each Agent.
    pub fn calc_consumed_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(agent.state().id.clone(), agent.state().consumed.len());
        }

        data
    }

    /// Calculates the length of the produced messages for each Agent.
    pub fn calc_produced_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(agent.state().id.clone(), agent.state().produced.len());
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
    /// If there are any interrupts, process those immediately.
    fn process_message_bus(&mut self, mut message_bus: Vec<Message>) {
        while let Some(message) = message_bus.pop() {
            for agent in self.agents.iter_mut() {
                if agent.state().id == message.clone().destination {
                    agent.push_message(message.clone());
                }

                if agent.state().id == message.clone().source {
                    agent.state_mut().produced.push(message.clone());
                }
            }

            if let Some(Interrupt::HaltSimulation(reason)) = message.interrupt {
                info!("Received a halt interrupt: {:?}", reason);
                self.mode = SimulationMode::Completed;
            }
        }
    }

    /// An internal function used to wakeup sleeping Agents due to wake.
    fn wakeup_agents_scheduled_to_wakeup_now(&mut self) {
        for agent in self.agents.iter_mut() {
            if let AgentMode::AsleepUntil(wakeup_at) = agent.state().mode {
                if self.time >= wakeup_at {
                    agent.state_mut().mode = agent.state().wake_mode;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Poisson;

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
            halt_check: |s: &Simulation| s.time == 5,
            ..Default::default()
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
                    name: "Starbucks Clerk".to_owned(),
                    consumption_fn: |a: &mut Agent, t: DiscreteTime| {
                        debug!("{} looking for a customer.", a.state().id);
                        if let Some(last) = a.state().consumed.last() {
                            if last.completed_time? + 60 > t {
                                debug!("Sorry, we're still serving the last customer.");
                                return None;
                            }
                        }

                        if let Some(message) = a.queue.pop_front() {
                            if message.queued_time + 100 > t {
                                debug!("Still making your coffee, sorry!");
                                a.state_mut().queue.push_front(message);
                                return None;
                            }

                            debug!("Serviced a customer!");
                            a.state_mut().consumed.push(Message {
                                completed_time: Some(t),
                                ..message
                            });
                        }
                        return None;
                    },
                    ..Default::default()
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
