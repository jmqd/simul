extern crate self as simul;
pub mod agent;
pub mod experiment;
pub mod message;

pub use agent::*;
pub use message::*;
pub use simul_macro;

use log::{debug, info};
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
    pub enable_queue_depth_metric: bool,

    /// Records a metric on the number of cycles an agent was asleep for.
    pub enable_agent_asleep_cycles_metric: bool,

    /// The mode of the Simulation.
    pub mode: SimulationMode,

    /// Maps from id to AgentMetadata.
    agent_metadata_hash_table: HashMap<String, AgentMetadata>,

    /// Maps from an Agent's id to its index, a handle for indexing the Agent.
    pub agent_id_index_map: HashMap<String, usize>,

    /// Maps from an Agent's String id to its AgentState.
    pub agent_states: Vec<AgentState>,
}

/// The parameters to create a Simulation.
#[derive(Clone, Debug)]
pub struct SimulationParameters {
    /// The agents within the simulation, e.g. adaptive agents.
    /// See here: https://authors.library.caltech.edu/60491/1/MGM%20113.pdf
    pub agent_initializers: Vec<AgentInitializer>,
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
            agent_initializers: vec![],
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
        // TODO(jmqd): Add the handle id to the agents here, use instead of mapping.
        let agent_id_index_map: HashMap<String, usize> = parameters
            .agent_initializers
            .iter()
            .enumerate()
            .map(|(i, agent_initializer)| (agent_initializer.agent.id(), i))
            .collect();

        let agent_states: Vec<AgentState> = parameters
            .agent_initializers
            .iter()
            .map(|agent_initializer| AgentState {
                mode: agent_initializer.options.initial_mode,
                wake_mode: agent_initializer.options.wake_mode,
                // TODO(jmqd): Avoid this clone of the initial_queue.
                queue: agent_initializer.options.initial_queue.clone(),
                consumed: vec![],
                produced: vec![],
            })
            .collect();

        Simulation {
            mode: SimulationMode::Constructed,
            agent_metadata_hash_table: parameters
                .agent_initializers
                .iter()
                .map(|agent_initializer| {
                    (
                        agent_initializer.agent.id(),
                        AgentMetadata {
                            queue_depth_metrics: vec![],
                            asleep_cycle_count: 0,
                        },
                    )
                })
                .collect(),
            agents: parameters
                .agent_initializers
                .into_iter()
                .map(|agent_initializer| agent_initializer.agent)
                .collect(),
            halt_check: parameters.halt_check,
            time: parameters.starting_time,
            enable_queue_depth_metric: parameters.enable_queue_depth_metrics,
            enable_agent_asleep_cycles_metric: parameters.enable_agent_asleep_cycles_metric,
            agent_id_index_map,
            agent_states,
        }
    }

    /// Returns the consumed messages for a given Agent during the Simulation.
    pub fn consumed_for_agent(&self, name: &str) -> Option<Vec<Message>> {
        let agent = self.agents.iter().find(|a| a.id() == name)?;
        Some(self.agent_state(&agent.id())?.consumed.clone())
    }

    /// Returns the produced messages for a given Agent during the Simulation.
    pub fn produced_for_agent(&self, name: &str) -> Option<Vec<Message>> {
        let agent = self.agents.iter().find(|a| a.id() == name)?;
        Some(self.agent_state(&agent.id()).unwrap().produced.clone())
    }

    pub fn agent_state(&self, id: &str) -> Option<&AgentState> {
        // SAFETY: We initialize the agent_states vec to be len(param.agents)
        unsafe {
            Some(
                self.agent_states
                    .get_unchecked(*self.agent_id_index_map.get(id)?),
            )
        }
    }

    pub fn agent_state_mut(&mut self, id: &str) -> Option<&AgentState> {
        Some(unsafe {
            self.agent_states
                .get_unchecked_mut(*self.agent_id_index_map.get(id).unwrap())
        })
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
        let mut command_buffer: Vec<AgentCommand> = vec![];

        while !(self.halt_check)(self) {
            debug!("Running next tick of simulation at time {}", self.time);
            self.wakeup_agents_scheduled_to_wakeup_now();

            for i in 0..self.agents.len() {
                let agent = &mut self.agents[i];
                let agent_id = agent.id();
                let agent_handle = self.agent_id_index_map[&agent_id];
                let queued_msg = self
                    .agent_states
                    .get_mut(agent_handle)
                    .unwrap()
                    .queue
                    .pop_front();
                let agent_state = self.agent_states.get_mut(agent_handle).unwrap();

                if self.enable_queue_depth_metric {
                    self.agent_metadata_hash_table
                        .get_mut(&agent_id)
                        .expect("Failed to find agent in metrics")
                        .queue_depth_metrics
                        .push(agent_state.queue.len());
                }

                let mut agent_commands: Vec<AgentCommandType> = vec![];

                let mut ctx = AgentContext {
                    id: &agent_id,
                    time: self.time,
                    commands: &mut agent_commands,
                    state: agent_state,
                    message_processing_status: MessageProcessingStatus::Initialized,
                };

                match agent_state.mode {
                    AgentMode::Proactive => agent.as_mut().on_tick(&mut ctx),
                    AgentMode::Reactive => {
                        if let Some(msg) = queued_msg {
                            agent.as_mut().on_message(&mut ctx, &msg);

                            match ctx.message_processing_status {
                                MessageProcessingStatus::Failed
                                | MessageProcessingStatus::InProgress => {
                                    self.agent_states
                                        .get_mut(agent_handle)
                                        .unwrap()
                                        .queue
                                        .push_front(msg);
                                }
                                // TODO(jmqd): For now, we assume Initialized also means completed.
                                // This is a leaky abstraction; we should find a better one.
                                MessageProcessingStatus::Initialized
                                | MessageProcessingStatus::Completed => {
                                    self.agent_states
                                        .get_mut(agent_handle)
                                        .unwrap()
                                        .consumed
                                        .push(Message {
                                            completed_time: Some(self.time),
                                            ..msg
                                        });
                                }
                            }
                        }
                    }
                    AgentMode::AsleepUntil(_) => {
                        if self.enable_agent_asleep_cycles_metric {
                            self.agent_metadata_hash_table
                                .get_mut(&agent.id())
                                .expect("Failed to find agent in metrics")
                                .asleep_cycle_count += 1
                        }
                    }
                    AgentMode::Dead => {}
                }

                command_buffer.extend(agent_commands.into_iter().map(|command_type| {
                    AgentCommand {
                        ty: command_type,
                        agent_handle,
                    }
                }));
            }

            // Consume all the new messages in the bus and deliver to agents.
            self.process_command_buffer(&mut command_buffer);

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
            .filter(|a| !self.agent_state(&a.id()).unwrap().consumed.is_empty())
        {
            let mut sum_of_times: u64 = 0;
            for completed in self.agent_state(&agent.id()).unwrap().consumed.iter() {
                sum_of_times += completed.completed_time.unwrap() - completed.queued_time;
            }

            data.insert(
                agent.id(),
                sum_of_times as usize / self.agent_state(&agent.id()).unwrap().consumed.len(),
            );
        }

        data
    }

    /// Calculates the statistics of queue lengths.
    /// Mostly useful for checking which agents still have queues of work after halting.
    pub fn calc_queue_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(
                agent.id().clone(),
                self.agent_state(&agent.id()).unwrap().queue.len(),
            );
        }

        data
    }

    /// Calculates the length of the consumed messages for each Agent.
    pub fn calc_consumed_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(
                agent.id().clone(),
                self.agent_state(&agent.id()).unwrap().consumed.len(),
            );
        }

        data
    }

    /// Calculates the length of the produced messages for each Agent.
    pub fn calc_produced_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(
                agent.id().clone(),
                self.agent_state(&agent.id()).unwrap().produced.len(),
            );
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
    fn process_command_buffer(&mut self, command_buffer: &mut Vec<AgentCommand>) {
        while let Some(command) = command_buffer.pop() {
            match command.ty {
                AgentCommandType::SendMessage(message) => {
                    let receiver_id_option = self.agent_id_index_map.get(&message.destination);

                    if let Some(receiver_id) = receiver_id_option {
                        let receiver_queue = &mut self.agent_states[*receiver_id].queue;
                        receiver_queue.push_back(message.clone());
                    }

                    self.agent_states[command.agent_handle]
                        .produced
                        .push(message.clone());
                }

                AgentCommandType::HaltSimulation(reason) => {
                    info!("Received a halt interrupt: {:?}", reason);
                    self.mode = SimulationMode::Completed;
                }

                AgentCommandType::Sleep(ticks) => {
                    self.agent_states[command.agent_handle].mode =
                        AgentMode::AsleepUntil(self.time + ticks);
                }
            }
        }
    }

    /// An internal function used to wakeup sleeping Agents due to wake.
    fn wakeup_agents_scheduled_to_wakeup_now(&mut self) {
        for i in 0..self.agents.len() {
            let agent_state = &mut self.agent_states[i];

            if let AgentMode::AsleepUntil(wakeup_at) = agent_state.mode {
                if self.time >= wakeup_at {
                    agent_state.mode = agent_state.wake_mode;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Poisson;
    use simul_macro::agent;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn basic_periodic_test() {
        init();
        let mut simulation = Simulation::new(SimulationParameters {
            agent_initializers: vec![
                periodic_producing_agent("producer".to_string(), 1, "consumer".to_string()),
                periodic_consuming_agent("consumer".to_string(), 1),
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

        #[agent]
        struct Clerk {}

        impl Agent for Clerk {
            fn id(&self) -> String {
                self.id.clone()
            }

            fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
                debug!("{} looking for a customer.", self.id());
                if let Some(last) = ctx.state.consumed.last() {
                    if last.completed_time.unwrap() + 60 > ctx.time {
                        debug!("Sorry, we're still serving the last customer.");
                    }
                }

                if let Some(message) = ctx.state.queue.front() {
                    if msg.queued_time + 100 > ctx.time {
                        debug!("Still making your coffee, sorry!");
                        ctx.set_processing_status(MessageProcessingStatus::InProgress);
                    }

                    debug!("Serviced a customer!");
                }
            }
        }

        let mut simulation = Simulation::new(SimulationParameters {
            starting_time: 1,
            enable_queue_depth_metrics: false,
            enable_agent_asleep_cycles_metric: false,
            halt_check: |s: &Simulation| s.time > 500,
            agent_initializers: vec![
                poisson_distributed_producing_agent(
                    "Starbucks Customers".to_string(),
                    Poisson::new(80.0).unwrap(),
                    "Starbucks Clerk".to_string(),
                ),
                AgentInitializer {
                    agent: Box::new(Clerk {
                        id: "Starbucks Clerk".to_string(),
                    }),
                    options: AgentOptions::default(),
                },
            ],
        });

        simulation.run();
        assert!(Some(simulation).is_some());
    }
}
