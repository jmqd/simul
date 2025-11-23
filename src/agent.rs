use crate::{experiment::ObjectiveScore, message::*, DiscreteTime};
use dyn_clone::DynClone;
use rand::prelude::*;
use rand_distr::Poisson;
use std::collections::VecDeque;

/// Possible states an Agent can be in.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Default)]
pub enum AgentMode {
    /// The Agent is active; process() is called on every tick of the simulation.
    Proactive,

    /// The Agent is reactive; process() is called when this agent has a message.
    #[default]
    Reactive,

    /// The Agent is sleeping (or on cooldown) until a scheduled wakeup.
    AsleepUntil(DiscreteTime),

    /// The Agent is dead (inactive) and does nothing in this state.
    Dead,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct AgentMetadata {
    pub queue_depth_metrics: Vec<usize>,
    pub asleep_cycle_count: DiscreteTime,
}

#[derive(Debug, Clone)]
pub struct SimulationAgent {
    /// The agent that the user constructed; contains user state and behavior.
    pub agent: Box<dyn Agent>,

    /// The metadata associated with the agent.
    pub(crate) metadata: AgentMetadata,

    /// The name for the agent. Must be unique for the simulation.
    pub name: String,

    /// State that is mutable by the agent itself.
    pub state: AgentState,
}

#[derive(Debug, Clone)]
pub struct AgentState {
    /// The mode that the agent is in.
    pub mode: AgentMode,

    /// The mode that the agent wakes up into.
    pub wake_mode: AgentMode,

    /// The queue of incoming Messages for the Agent.
    pub queue: VecDeque<Message>,

    /// The queue of messages already consumed by the agent.
    pub consumed: Vec<Message>,

    /// The queue of messages produced by the agent.
    pub produced: Vec<Message>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            mode: AgentMode::Dead,
            wake_mode: AgentMode::Dead,
            queue: VecDeque::new(),
            consumed: vec![],
            produced: vec![],
        }
    }
}

pub struct AgentCommand {
    pub ty: AgentCommandType,
    pub agent_handle: usize,
}

/// Actions the Agent can perform.
pub enum AgentCommandType {
    /// Send a message to another agent
    SendMessage(Message),
    /// Sleep for a relative number of ticks
    Sleep(DiscreteTime),
    /// Stop the simulation
    HaltSimulation(String),
}

pub enum MessageProcessingStatus {
    Initialized,
    Completed,
    InProgress,
    Failed,
}

// The Context holds the capability for Agents to act on the world
pub struct AgentContext<'a> {
    /// The handle id of the Agent.
    pub handle: usize,

    /// The name of the Agent.
    pub name: &'a str,

    /// The current simulation time.
    pub time: DiscreteTime,

    /// Internal buffer for commands (messages, sleep requests, etc.)
    pub(crate) commands: &'a mut Vec<AgentCommandType>,

    pub state: &'a AgentState,

    pub message_processing_status: MessageProcessingStatus,
}

impl<'a> AgentContext<'a> {
    pub fn send(&mut self, target: &str, payload: Option<Vec<u8>>) {
        self.commands.push(AgentCommandType::SendMessage(Message {
            source: self.name.to_string(),
            destination: target.to_string(),
            queued_time: self.time,
            custom_payload: payload,
            ..Default::default()
        }));
    }

    /// Sends an interrupt to HALT the simulation.
    pub fn send_halt_interrupt(&mut self, reason: &str) {
        self.commands
            .push(AgentCommandType::HaltSimulation(reason.to_string()));
    }

    /// Sleeps the Agent for a relative amount of time.
    pub fn sleep_for(&mut self, ticks: DiscreteTime) {
        self.commands.push(AgentCommandType::Sleep(ticks));
    }

    pub fn set_processing_status(&mut self, status: MessageProcessingStatus) {
        self.message_processing_status = status;
    }
}

/// Configuration for how an agent should start the simulation.
#[derive(Debug, Clone, Default)]
pub struct AgentOptions {
    pub initial_mode: AgentMode,
    pub wake_mode: AgentMode,
    pub initial_queue: VecDeque<Message>,
    pub name: String,
}

impl AgentOptions {
    pub fn defaults_with_name(name: String) -> Self {
        Self {
            name,
            ..Default::default()
        }
    }
}

/// An initializer for an agent. The `agent` holds the behavior and `on_`
/// functions; the options are configuration for constructing the agent.
#[derive(Debug, Clone)]
pub struct AgentInitializer {
    pub agent: Box<dyn Agent>,
    pub options: AgentOptions,
}

/// The bread and butter of the Simulation -- the Agent.
/// In a Complex Adaptive System (CAS), an Adaptive Agent does things and
/// interacts with the Simulation, itself, and other Agents.
///
/// Some examples of what an Agent might be:
/// * Barista at a coffee shop.
/// * Stoplight.
/// * Driver in traffic.
/// * A single-celled organism.
/// * A player in a game.
pub trait Agent: std::fmt::Debug + DynClone {
    /// The main action an agent performs, processing messages that come in to it.
    fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message);

    /// Some Agents do things all the time, they are `Proactive`.
    #[allow(unused_variables)]
    fn on_tick(&mut self, ctx: &mut AgentContext) {}

    /// For annealing experiments, you may implement a cost function for the agent.
    /// For example, a periodic consuming agent has cost implented equal to its period.
    fn cost(&self) -> ObjectiveScore {
        0f64
    }
}

dyn_clone::clone_trait_object!(Agent);

/// An agent that processes on a Poisson-distributed periodicity.
pub fn poisson_distributed_consuming_agent<T>(name: T, dist: Poisson<f64>) -> AgentInitializer
where
    T: Into<String>,
{
    #[derive(Debug, Clone)]
    struct PoissonAgent {
        period: Poisson<f64>,
    }

    impl Agent for PoissonAgent {
        fn on_message(&mut self, ctx: &mut AgentContext, _msg: &Message) {
            // This agent will go to sleep for a "cooldown period",
            // as determined by a poisson distribution function.
            let cooldown_period = self.period.sample(&mut rand::thread_rng()) as u64;
            ctx.sleep_for(cooldown_period);
        }
    }

    AgentInitializer {
        agent: Box::new(PoissonAgent { period: dist }),
        options: AgentOptions::defaults_with_name(name.into()),
    }
}

/// Given a poisson distribution for the production period,
/// returns an Agent that produces to Target with that frequency.
pub fn poisson_distributed_producing_agent<T>(
    name: T,
    dist: Poisson<f64>,
    target: T,
) -> AgentInitializer
where
    T: Into<String>,
{
    #[derive(Clone, Debug)]
    struct PoissonAgent {
        period: Poisson<f64>,
        target: String,
    }

    impl Agent for PoissonAgent {
        fn on_message(&mut self, ctx: &mut AgentContext, _msg: &Message) {
            // This agent will go to sleep for a "cooldown period",
            // as determined by a poisson distribution function.
            let cooldown_period = self.period.sample(&mut rand::thread_rng()) as u64;
            ctx.sleep_for(cooldown_period);
            ctx.send(&self.target, None);
        }
    }

    AgentInitializer {
        agent: Box::new(PoissonAgent {
            period: dist,
            target: target.into(),
        }),
        options: AgentOptions::defaults_with_name(name.into()),
    }
}

/// A simple agent that produces messages on a period, directed to target.
pub fn periodic_producing_agent<T>(name: T, period: DiscreteTime, target: T) -> AgentInitializer
where
    T: Into<String>,
{
    #[derive(Clone, Debug)]
    struct PeriodicProducer {
        period: DiscreteTime,
        target: String,
    }

    impl Agent for PeriodicProducer {
        fn cost(&self) -> ObjectiveScore {
            -(self.period as ObjectiveScore)
        }

        #[allow(unused_variables)]
        fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
            // TODO(jmqd): This is pretty jank, fix this interface.
        }

        fn on_tick(&mut self, ctx: &mut AgentContext) {
            ctx.sleep_for(self.period);
            ctx.send(&self.target, None);
        }
    }

    AgentInitializer {
        agent: Box::new(PeriodicProducer {
            period,
            target: target.into(),
        }),
        options: AgentOptions {
            initial_mode: AgentMode::Proactive,
            wake_mode: AgentMode::Proactive,
            name: name.into(),
            ..Default::default()
        },
    }
}

/// A simple agent that consumes messages on a period with no side effects.
/// Period can be thought of the time to consume 1 message.
pub fn periodic_consuming_agent<T>(name: T, period: DiscreteTime) -> AgentInitializer
where
    T: Into<String>,
{
    #[derive(Clone, Debug)]
    struct PeriodicConsumer {
        period: DiscreteTime,
    }

    impl Agent for PeriodicConsumer {
        fn cost(&self) -> ObjectiveScore {
            -(self.period as ObjectiveScore)
        }

        fn on_message(&mut self, ctx: &mut AgentContext, _msg: &Message) {
            ctx.sleep_for(self.period);
        }
    }

    AgentInitializer {
        agent: Box::new(PeriodicConsumer { period }),
        options: AgentOptions::defaults_with_name(name.into()),
    }
}
