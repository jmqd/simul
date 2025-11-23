extern crate self as simul;
pub mod agent;
pub mod experiment;
pub mod message;

pub use agent::*;
pub use message::*;

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
    pub agents: Vec<SimulationAgent>,

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

    /// Maps from an Agent's id to its index, a handle for indexing the Agent.
    pub agent_name_handle_map: HashMap<String, usize>,
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

impl Simulation {
    pub fn new(parameters: SimulationParameters) -> Simulation {
        // TODO(jmqd): Add the handle id to the agents here, use instead of mapping.
        let agent_name_handle_map: HashMap<String, usize> = parameters
            .agent_initializers
            .iter()
            .enumerate()
            .map(|(i, agent_initializer)| (agent_initializer.options.name.clone(), i))
            .collect();

        let agents: Vec<SimulationAgent> = parameters
            .agent_initializers
            .into_iter()
            .map(|agent_initializer| SimulationAgent {
                agent: agent_initializer.agent,
                name: agent_initializer.options.name,
                metadata: AgentMetadata::default(),
                state: AgentState {
                    mode: agent_initializer.options.initial_mode,
                    wake_mode: agent_initializer.options.wake_mode,
                    queue: agent_initializer.options.initial_queue,
                    consumed: vec![],
                    produced: vec![],
                },
            })
            .collect();

        Simulation {
            mode: SimulationMode::Constructed,
            agents,
            halt_check: parameters.halt_check,
            time: parameters.starting_time,
            enable_queue_depth_metric: parameters.enable_queue_depth_metrics,
            enable_agent_asleep_cycles_metric: parameters.enable_agent_asleep_cycles_metric,
            agent_name_handle_map,
        }
    }

    /// Returns the consumed messages for a given Agent during the Simulation.
    pub fn consumed_for_agent(&self, name: &str) -> Option<Vec<Message>> {
        Some(self.find_by_name(name)?.state.consumed.clone())
    }

    /// Returns a SimulationAgent by name.
    pub fn find_by_name(&self, name: &str) -> Option<&SimulationAgent> {
        self.agent_name_handle_map
            .get(name)
            .map(|id| self.agents.get(*id))?
    }

    /// Returns a SimulationAgent by name.
    pub fn find_by_name_mut(&mut self, name: &str) -> Option<&mut SimulationAgent> {
        self.agent_name_handle_map
            .get(name)
            .map(|id| self.agents.get_mut(*id))?
    }

    /// Returns the produced messages for a given Agent during the Simulation.
    pub fn produced_for_agent(&self, name: &str) -> Option<Vec<Message>> {
        Some(self.find_by_name(name)?.state.produced.clone())
    }

    /// Returns the queue depth timeseries for a given Agent during the Simulation.
    pub fn queue_depth_metrics(&self, name: &str) -> Option<Vec<usize>> {
        // TODO(?): Return non option here.
        Some(
            self.find_by_name(name)?
                .metadata
                .queue_depth_metrics
                .clone(),
        )
    }

    /// Returns the asleep cycle count for a given Agent during the Simulation.
    pub fn asleep_cycle_count(&self, name: &str) -> Option<DiscreteTime> {
        // TODO(?): Return non option here.
        Some(self.find_by_name(name)?.metadata.asleep_cycle_count)
    }

    /// Runs the simulation. This should only be called after adding all the beginning state.
    pub fn run(&mut self) {
        self.mode = SimulationMode::Running;
        let mut command_buffer: Vec<AgentCommand> = vec![];

        while !(self.halt_check)(self) {
            debug!("Running next tick of simulation at time {}", self.time);
            self.wakeup_agents_scheduled_to_wakeup_now();

            for agent_handle in 0..self.agents.len() {
                let agent = &mut self.agents[agent_handle];
                let queued_msg = agent.state.queue.pop_front();

                if self.enable_queue_depth_metric {
                    agent
                        .metadata
                        .queue_depth_metrics
                        .push(agent.state.queue.len());
                }

                let mut agent_commands: Vec<AgentCommandType> = vec![];

                let mut ctx = AgentContext {
                    handle: agent_handle,
                    name: &agent.name,
                    time: self.time,
                    commands: &mut agent_commands,
                    state: &agent.state,
                    message_processing_status: MessageProcessingStatus::Initialized,
                };

                match agent.state.mode {
                    AgentMode::Proactive => agent.agent.on_tick(&mut ctx),
                    AgentMode::Reactive => {
                        if let Some(msg) = queued_msg {
                            // TODO(jmqd): agent.agent is not pretty; fix this composition naming.
                            agent.agent.on_message(&mut ctx, &msg);

                            match ctx.message_processing_status {
                                MessageProcessingStatus::Failed
                                | MessageProcessingStatus::InProgress => {
                                    agent.state.queue.push_front(msg);
                                }
                                // TODO(jmqd): For now, we assume Initialized also means completed.
                                // This is a leaky abstraction; we should find a better one.
                                MessageProcessingStatus::Initialized
                                | MessageProcessingStatus::Completed => {
                                    agent.state.consumed.push(Message {
                                        completed_time: Some(self.time),
                                        ..msg
                                    });
                                }
                            }
                        }
                    }
                    AgentMode::AsleepUntil(_) => {
                        if self.enable_agent_asleep_cycles_metric {
                            agent.metadata.asleep_cycle_count += 1
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
            .filter(|agent| !agent.state.consumed.is_empty())
        {
            let mut sum_of_times: u64 = 0;
            for completed in agent.state.consumed.iter() {
                sum_of_times += completed.completed_time.unwrap() - completed.queued_time;
            }

            data.insert(
                agent.name.clone(),
                sum_of_times as usize / agent.state.consumed.len(),
            );
        }

        data
    }

    /// Calculates the statistics of queue lengths.
    /// Mostly useful for checking which agents still have queues of work after halting.
    pub fn calc_queue_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(agent.name.clone(), agent.state.queue.len());
        }

        data
    }

    /// Calculates the length of the consumed messages for each Agent.
    pub fn calc_consumed_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(agent.name.clone(), agent.state.consumed.len());
        }

        data
    }

    /// Calculates the length of the produced messages for each Agent.
    pub fn calc_produced_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(agent.name.clone(), agent.state.produced.len());
        }

        data
    }

    /// SAFETY: The caller must ensure that `handle` is within the bounds of `self.agents`.
    unsafe fn agent_by_handle_mut_unchecked(&mut self, handle: usize) -> &mut SimulationAgent {
        unsafe { self.agents.get_unchecked_mut(handle) }
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
                    if let Some(receiver) = self.find_by_name_mut(&message.destination) {
                        receiver.state.queue.push_back(message.clone());
                    }

                    let commanding_agent =
                        unsafe { self.agent_by_handle_mut_unchecked(command.agent_handle) };

                    commanding_agent.state.produced.push(message.clone());
                }

                AgentCommandType::HaltSimulation(reason) => {
                    info!("Received a halt interrupt: {:?}", reason);
                    self.mode = SimulationMode::Completed;
                }

                AgentCommandType::Sleep(ticks) => {
                    let sleep_until = self.time + ticks;
                    let commanding_agent =
                        unsafe { self.agent_by_handle_mut_unchecked(command.agent_handle) };

                    commanding_agent.state.mode = AgentMode::AsleepUntil(sleep_until);
                }
            }
        }
    }

    /// An internal function used to wakeup sleeping Agents due to wake.
    fn wakeup_agents_scheduled_to_wakeup_now(&mut self) {
        for agent in self.agents.iter_mut() {
            if let AgentMode::AsleepUntil(wakeup_at) = agent.state.mode {
                if self.time >= wakeup_at {
                    agent.state.mode = agent.state.wake_mode;
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

        #[derive(Debug, Clone)]
        struct Clerk {}

        impl Agent for Clerk {
            fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
                debug!("{} looking for a customer.", ctx.name);
                if let Some(last) = ctx.state.consumed.last() {
                    if last.completed_time.unwrap() + 60 > ctx.time {
                        debug!("Sorry, we're still serving the last customer.");
                    }
                }

                if let Some(_msg) = ctx.state.queue.front() {
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
                    agent: Box::new(Clerk {}),
                    options: AgentOptions::defaults_with_name("Starbucks Clerk".to_string()),
                },
            ],
        });

        simulation.run();
        assert!(Some(simulation).is_some());
    }
}
