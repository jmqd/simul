//! `simul` is a discrete-event simulation library for running high-level
//! simulations of real-world problems and for running simulated experiments.
//!
//! `simul` is a *discrete-event simulator* using *incremental time
//! progression*, with [M/M/c queues](https://en.wikipedia.org/wiki/M/M/c_queue)
//! for interactions between agents. It also supports some forms of
//! experimentation and simulated annealing to replicate a simulation many
//! times, varying the simulation parameters.
//!
//! Use-cases:
//! - [Discrete-event simulation](https://en.wikipedia.org/wiki/Discrete-event_simulation)
//! - [Complex adaptive systems](https://authors.library.caltech.edu/60491/1/MGM%20113.pdf)
//! - [Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
//! - [Job-shop scheduling](https://en.wikipedia.org/wiki/Job-shop_scheduling)
//! - [Birth-death processes](https://en.wikipedia.org/wiki/Birth%E2%80%93death_process)
//! - [Computer experiments](https://en.wikipedia.org/wiki/Computer_experiment)
//! - Other: simulating logistics, operations research problems, running
//!   experiments to approximate a global optimum, simulating queueing systems,
//!   distributed systems, performance engineering/analysis, and so on.
//!

extern crate self as simul;
pub mod agent;
pub mod experiment;
pub mod message;

pub use agent::*;
pub use message::*;

use log::{debug, info, log_enabled, Level};
use std::collections::HashMap;

/// `DiscreteTime` is a Simulation's internal representation of time.
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
    agents: Vec<SimulationAgent>,

    /// The current discrete time of the Simulation.
    time: DiscreteTime,

    /// A halt check function: given the state of the Simulation determine halt or not.
    halt_check: fn(&Self) -> bool,

    /// Whether to record metrics on queue depths. Takes space.
    enable_queue_depth_metric: bool,

    /// Records a metric on the number of cycles an agent was asleep for.
    enable_agent_asleep_cycles_metric: bool,

    /// The mode of the Simulation.
    mode: SimulationMode,

    /// Maps from an Agent's id to its index, a handle for indexing the Agent.
    agent_name_handle_map: HashMap<String, usize>,
}

/// The parameters to create a Simulation.
#[derive(Clone, Debug)]
pub struct SimulationParameters {
    /// The agents within the simulation, e.g. adaptive agents.
    /// See here: <https://authors.library.caltech.edu/60491/1/MGM%20113.pdf>
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

/// Errors returned when simulation parameters are invalid.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimulationError {
    /// More than one agent was configured with the same name.
    DuplicateAgentName(String),
}

impl std::fmt::Display for SimulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateAgentName(name) => {
                write!(f, "duplicate agent name in simulation parameters: {name}")
            }
        }
    }
}

impl std::error::Error for SimulationError {}

impl Default for SimulationParameters {
    fn default() -> Self {
        Self {
            agent_initializers: vec![],
            halt_check: |_| true,
            starting_time: 0,
            enable_queue_depth_metrics: false,
            enable_agent_asleep_cycles_metric: false,
        }
    }
}

impl Simulation {
    /// Builds a simulation.
    ///
    /// # Panics
    ///
    /// Panics when parameters are invalid. Use [`Self::try_new`] to handle errors explicitly.
    #[must_use]
    pub fn new(parameters: SimulationParameters) -> Self {
        match Self::try_new(parameters) {
            Ok(simulation) => simulation,
            Err(err) => panic!("{err}"),
        }
    }

    /// Builds a simulation, returning an error when parameters are invalid.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::DuplicateAgentName`] when two agents share a name.
    pub fn try_new(parameters: SimulationParameters) -> Result<Self, SimulationError> {
        let mut agent_name_handle_map = HashMap::with_capacity(parameters.agent_initializers.len());

        for (i, agent_initializer) in parameters.agent_initializers.iter().enumerate() {
            let name = agent_initializer.options.name.clone();
            if agent_name_handle_map.insert(name.clone(), i).is_some() {
                return Err(SimulationError::DuplicateAgentName(name));
            }
        }

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

        Ok(Self {
            mode: SimulationMode::Constructed,
            agents,
            halt_check: parameters.halt_check,
            time: parameters.starting_time,
            enable_queue_depth_metric: parameters.enable_queue_depth_metrics,
            enable_agent_asleep_cycles_metric: parameters.enable_agent_asleep_cycles_metric,
            agent_name_handle_map,
        })
    }

    /// Returns the consumed messages for a given Agent during the Simulation.
    #[must_use]
    pub fn consumed_for_agent(&self, name: &str) -> Option<&[Message]> {
        Some(&self.find_by_name(name)?.state.consumed)
    }

    /// Returns a `SimulationAgent` by name.
    #[must_use]
    #[inline]
    pub fn find_by_name(&self, name: &str) -> Option<&SimulationAgent> {
        self.find_handle_by_name(name)
            .and_then(|id| self.agents.get(id))
    }

    /// Returns a `SimulationAgent` by name.
    #[inline]
    pub fn find_by_name_mut(&mut self, name: &str) -> Option<&mut SimulationAgent> {
        let id = self.find_handle_by_name(name)?;
        self.agents.get_mut(id)
    }

    /// Returns the produced messages for a given Agent during the Simulation.
    #[must_use]
    pub fn produced_for_agent(&self, name: &str) -> Option<&[Message]> {
        Some(&self.find_by_name(name)?.state.produced)
    }

    /// Returns the queue depth timeseries for a given Agent during the Simulation.
    #[must_use]
    pub fn queue_depth_metrics(&self, name: &str) -> Option<&[usize]> {
        Some(&self.find_by_name(name)?.metadata.queue_depth_metrics)
    }

    /// Returns the asleep cycle count for a given Agent during the Simulation.
    #[must_use]
    pub fn asleep_cycle_count(&self, name: &str) -> Option<DiscreteTime> {
        Some(self.find_by_name(name)?.metadata.asleep_cycle_count)
    }

    /// Runs the simulation. This should only be called after adding all the beginning state.
    pub fn run(&mut self) {
        self.mode = SimulationMode::Running;
        let mut command_buffer: Vec<AgentCommand> = Vec::new();
        let mut requested_sleep_until: Option<DiscreteTime>;

        while !(self.halt_check)(self) {
            debug!("Running next tick of simulation at time {}", self.time);

            for agent_handle in 0..self.agents.len() {
                let agent = &mut self.agents[agent_handle];
                if let AgentMode::AsleepUntil(wakeup_at) = agent.state.mode {
                    if self.time >= wakeup_at {
                        agent.state.mode = agent.state.wake_mode;
                    }
                }
                let queued_msg = agent.state.queue.pop_front();

                if self.enable_queue_depth_metric {
                    agent
                        .metadata
                        .queue_depth_metrics
                        .push(agent.state.queue.len());
                }

                requested_sleep_until = None;

                match agent.state.mode {
                    AgentMode::Proactive => {
                        let mut ctx = AgentContext {
                            handle: agent_handle,
                            name: &agent.name,
                            time: self.time,
                            commands: &mut command_buffer,
                            requested_sleep_until: &mut requested_sleep_until,
                            state: &agent.state,
                            message_processing_status: MessageProcessingStatus::NoError,
                        };

                        agent.agent.on_tick(&mut ctx);
                    }
                    AgentMode::Reactive => {
                        if let Some(msg) = queued_msg {
                            let mut ctx = AgentContext {
                                handle: agent_handle,
                                name: &agent.name,
                                time: self.time,
                                commands: &mut command_buffer,
                                requested_sleep_until: &mut requested_sleep_until,
                                state: &agent.state,
                                message_processing_status: MessageProcessingStatus::NoError,
                            };

                            // TODO(jmqd): agent.agent is not pretty; fix this composition naming.
                            agent.agent.on_message(&mut ctx, &msg);

                            match ctx.message_processing_status {
                                MessageProcessingStatus::InProgress => {
                                    agent.state.queue.push_front(msg);
                                }
                                MessageProcessingStatus::NoError => {
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
                            agent.metadata.asleep_cycle_count += 1;
                        }
                    }
                    AgentMode::Dead => {}
                }

                if let Some(sleep_until) = requested_sleep_until {
                    agent.state.mode = AgentMode::AsleepUntil(sleep_until);
                }
            }

            // Consume all the new messages in the bus and deliver to agents.
            self.process_command_buffer(&mut command_buffer);

            debug!("Finished this tick; incrementing time.");
            self.time += 1;
        }

        self.mode = SimulationMode::Completed;
        if log_enabled!(Level::Debug) {
            self.emit_completed_simulation_debug_logging();
        }
    }

    /// A helper to calculate the average waiting time to process items.
    /// Note: This function will likely go away; it is an artifact of prototyping.
    #[must_use]
    pub fn calc_avg_wait_statistics(&self) -> HashMap<String, f64> {
        let mut data = HashMap::new();
        for agent in self
            .agents
            .iter()
            .filter(|agent| !agent.state.consumed.is_empty())
        {
            let mut sum_of_times: f64 = 0f64;
            for completed in &agent.state.consumed {
                sum_of_times += completed.completed_time.unwrap_or(completed.queued_time) as f64
                    - completed.queued_time as f64;
            }

            data.insert(
                agent.name.clone(),
                sum_of_times / agent.state.consumed.len() as f64,
            );
        }

        data
    }

    /// Calculates the statistics of queue lengths.
    /// Mostly useful for checking which agents still have queues of work after halting.
    #[must_use]
    pub fn calc_queue_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in &self.agents {
            data.insert(agent.name.clone(), agent.state.queue.len());
        }

        data
    }

    /// Calculates the length of the consumed messages for each Agent.
    #[must_use]
    pub fn calc_consumed_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in &self.agents {
            data.insert(agent.name.clone(), agent.state.consumed.len());
        }

        data
    }

    /// Calculates the length of the produced messages for each Agent.
    #[must_use]
    pub fn calc_produced_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in &self.agents {
            data.insert(agent.name.clone(), agent.state.produced.len());
        }

        data
    }

    /// SAFETY: The caller must ensure that `handle` is within the bounds of `self.agents`.
    unsafe fn agent_by_handle_mut_unchecked(&mut self, handle: usize) -> &mut SimulationAgent {
        unsafe { self.agents.get_unchecked_mut(handle) }
    }

    /// Emits debug logging w/ analytical stats.
    fn emit_completed_simulation_debug_logging(&self) {
        let queue_len_stats = self.calc_queue_len_statistics();
        let consumed_len_stats = self.calc_consumed_len_statistics();
        let avg_wait_stats = self.calc_avg_wait_statistics();
        let produced_len_stats = self.calc_produced_len_statistics();

        debug!("Queues: {queue_len_stats:?}");
        debug!("Consumed: {consumed_len_stats:?}");
        debug!("Produced: {produced_len_stats:?}");
        debug!("Average processing time: {avg_wait_stats:?}");
    }

    /// Returns an agent handle by name, using a linear scan for small simulations.
    #[inline]
    fn find_handle_by_name(&self, name: &str) -> Option<usize> {
        if self.agents.len() <= 8 {
            self.agents.iter().position(|agent| agent.name == name)
        } else {
            self.agent_name_handle_map.get(name).copied()
        }
    }

    /// Consume a `message_bus` of messages and disperse those messages to the agents.
    /// If there are any interrupts, process those immediately.
    #[inline]
    fn process_command_buffer(&mut self, command_buffer: &mut Vec<AgentCommand>) {
        while let Some(command) = command_buffer.pop() {
            match command.ty {
                AgentCommandType::SendMessage(message) => {
                    if let Some(receiver) = self.find_by_name_mut(&message.destination) {
                        receiver.state.queue.push_back(message.clone());
                    }

                    let commanding_agent =
                        unsafe { self.agent_by_handle_mut_unchecked(command.agent_handle) };

                    commanding_agent.state.produced.push(message);
                }

                AgentCommandType::HaltSimulation(reason) => {
                    info!("Received a halt interrupt: {reason:?}");
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

    /// Searches for an agent in the Simulation matching the given predicate.
    pub fn find_agent<P>(&self, predicate: P) -> Option<&SimulationAgent>
    where
        P: FnMut(&&SimulationAgent) -> bool,
    {
        self.agents.iter().find(predicate)
    }

    /// Checks whether all agents match the given predicate.
    pub fn all_agents<P>(&self, predicate: P) -> bool
    where
        P: FnMut(&SimulationAgent) -> bool,
    {
        self.agents.iter().all(predicate)
    }

    /// Returns a slice of the Agents in the Simulation.
    #[must_use]
    pub fn agents(&self) -> &[SimulationAgent] {
        self.agents.iter().as_slice()
    }

    /// Returns the current `DiscreteTime` tick for the Simulation.
    #[must_use]
    pub const fn time(&self) -> DiscreteTime {
        self.time
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
                periodic_producer("producer".to_string(), 1, "consumer".to_string()),
                periodic_consumer("consumer".to_string(), 1),
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
    fn starbucks_clerk() -> Result<(), Box<dyn std::error::Error>> {
        #[derive(Debug, Clone)]
        struct Clerk {}

        impl Agent for Clerk {
            fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
                debug!("{} looking for a customer.", ctx.name);
                if let Some(last) = ctx.state.consumed.last() {
                    if last.completed_time.is_some_and(|t| t + 60 > ctx.time) {
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

        init();

        let mut simulation = Simulation::new(SimulationParameters {
            starting_time: 1,
            enable_queue_depth_metrics: false,
            enable_agent_asleep_cycles_metric: false,
            halt_check: |s: &Simulation| s.time > 500,
            agent_initializers: vec![
                poisson_distributed_producer(
                    "Starbucks Customers".to_string(),
                    Poisson::new(80.0_f64)?,
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
        Ok(())
    }

    #[test]
    fn finds_agents_with_small_and_large_lookup_paths() {
        let small_simulation = Simulation::new(SimulationParameters {
            agent_initializers: vec![
                periodic_consumer("first".to_string(), 1),
                periodic_consumer("second".to_string(), 1),
            ],
            ..Default::default()
        });

        assert_eq!(
            small_simulation.find_by_name("second").map(|a| &a.name),
            Some(&"second".to_string())
        );

        let large_simulation = Simulation::new(SimulationParameters {
            agent_initializers: (0..9)
                .map(|i| periodic_consumer(format!("agent-{i}"), 1))
                .collect(),
            ..Default::default()
        });

        assert_eq!(
            large_simulation.find_by_name("agent-8").map(|a| &a.name),
            Some(&"agent-8".to_string())
        );
    }

    #[test]
    fn rejects_duplicate_agent_names() {
        let result = Simulation::try_new(SimulationParameters {
            agent_initializers: vec![
                periodic_consumer("worker".to_string(), 1),
                periodic_consumer("worker".to_string(), 2),
            ],
            ..Default::default()
        });

        assert!(
            matches!(result, Err(SimulationError::DuplicateAgentName(name)) if name == "worker")
        );
    }
}
