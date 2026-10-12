use crate::{message::Message, DiscreteTime};
use dyn_clone::DynClone;
use rand_distr::Distribution;
use rand_distr::Poisson;
use std::collections::{HashMap, VecDeque};

/// Possible states an Agent can be in.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Default)]
pub enum AgentMode {
    /// The Agent is active; `on_tick()` is expected to do work.
    Proactive,

    /// The Agent is reactive; `on_message()` is expected to do work.
    #[default]
    Reactive,

    /// The Agent is sleeping (or on cooldown) until a scheduled wakeup.
    AsleepUntil(DiscreteTime),

    /// The Agent is dead (inactive) and does nothing in this state.
    Dead,
}

/// Per-agent bookkeeping collected by the simulation engine.
#[derive(Clone, Debug, Default)]
pub(crate) struct AgentMetadata {
    /// Metrics recording the depth of the queue at each tick of the Sim.
    /// TODO: We should store (time, depth) to minimize pushing here.
    pub queue_depth_metrics: Vec<usize>,

    /// The number of cycles this agent was sleeping for.
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
    pub receiver_handle: Option<usize>,
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
    NoError,
    InProgress,
}

// The Context holds the capability for Agents to act on the world
///
/// Routing handles are read-only to callbacks.
///
/// ```compile_fail
/// use simul::agent::AgentContext;
/// fn change_identity(ctx: &mut AgentContext<'_>) {
///     ctx.handle = usize::MAX;
/// }
/// ```
pub struct AgentContext<'a> {
    /// The handle id of the Agent.
    pub(crate) handle: usize,

    /// The name of the Agent.
    pub name: &'a str,

    /// The current simulation time.
    pub time: DiscreteTime,

    /// Internal buffer for commands (messages, sleep requests, etc.)
    pub(crate) commands: &'a mut Vec<AgentCommand>,

    /// Sleep requested by the agent during the current tick.
    pub(crate) requested_sleep_until: &'a mut Option<DiscreteTime>,

    /// Maps agent names to stable handles for internal message routing.
    pub(crate) agent_name_handle_map: &'a HashMap<String, usize>,

    pub state: &'a AgentState,

    pub message_processing_status: MessageProcessingStatus,
}

impl AgentContext<'_> {
    /// Returns the agent's immutable routing handle.
    #[must_use]
    pub const fn handle(&self) -> usize {
        self.handle
    }

    /// Queues a message for end-of-tick delivery.
    ///
    /// Multiple messages sent by one callback are delivered in call order.
    #[inline]
    pub fn send(&mut self, target: &str, payload: Option<Vec<u8>>) {
        // Repeated payload-free sends can reuse retained contents, just as
        // delivery shares contents with history. Timestamps stay independent.
        let message = match self.state.produced.last() {
            Some(previous)
                if payload.is_none()
                    && previous.custom_payload.is_none()
                    && previous.interrupt.is_none()
                    && previous.source == self.name
                    && previous.destination == target =>
            {
                let mut message = previous.clone();
                message.queued_time = self.time;
                message.completed_time = None;
                message
            }
            _ => Message::new_with_payload(self.time, self.name, target, payload),
        };
        self.commands.push(AgentCommand {
            ty: AgentCommandType::SendMessage(message),
            agent_handle: self.handle,
            receiver_handle: self.agent_name_handle_map.get(target).copied(),
        });
    }

    /// Sends an interrupt to HALT the simulation.
    #[inline]
    pub fn send_halt_interrupt(&mut self, reason: &str) {
        self.commands.push(AgentCommand {
            ty: AgentCommandType::HaltSimulation(reason.to_string()),
            agent_handle: self.handle,
            receiver_handle: None,
        });
    }

    /// Sleeps the Agent for a relative amount of time.
    ///
    /// # Panics
    ///
    /// Panics before changing the sleep request if the wake time exceeds `u64::MAX`.
    #[inline]
    pub const fn sleep_for(&mut self, ticks: DiscreteTime) {
        let Some(wake_time) = self.time.checked_add(ticks) else {
            panic!("agent wake time exceeds u64::MAX");
        };
        *self.requested_sleep_until = Some(wake_time);
    }

    /// Records whether the current message remains in progress or finished cleanly.
    #[inline]
    pub const fn set_processing_status(&mut self, status: MessageProcessingStatus) {
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
    #[must_use]
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
}

dyn_clone::clone_trait_object!(Agent);

/// An agent that processes on a Poisson-distributed periodicity.
pub fn poisson_distributed_consumer<T>(name: T, dist: Poisson<f64>) -> AgentInitializer
where
    T: Into<String>,
{
    #[derive(Debug, Clone)]
    struct PoissonAgent {
        /// The distribution of time between consuming messages.
        period: Poisson<f64>,
    }

    impl Agent for PoissonAgent {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        fn on_message(&mut self, ctx: &mut AgentContext, _msg: &Message) {
            // This agent will go to sleep for a "cooldown period",
            // as determined by a poisson distribution function.
            let cooldown_period = self.period.sample(&mut rand::rng()) as u64;
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
pub fn poisson_distributed_producer<T>(name: T, dist: Poisson<f64>, target: T) -> AgentInitializer
where
    T: Into<String>,
{
    #[derive(Clone, Debug)]
    struct PoissonAgent {
        /// The distribution of time between producing.
        period: Poisson<f64>,

        /// The target to which we produce messages.
        target: String,
    }

    impl Agent for PoissonAgent {
        fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        fn on_tick(&mut self, ctx: &mut AgentContext) {
            // This agent will go to sleep for a "cooldown period",
            // as determined by a poisson distribution function.
            let cooldown_period = self.period.sample(&mut rand::rng()) as u64;
            ctx.sleep_for(cooldown_period);
            ctx.send(&self.target, None);
        }
    }

    AgentInitializer {
        agent: Box::new(PoissonAgent {
            period: dist,
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

/// A simple agent that produces messages on a period, directed to target.
pub fn periodic_producer<T>(name: T, period: DiscreteTime, target: T) -> AgentInitializer
where
    T: Into<String>,
{
    #[derive(Clone, Debug)]
    struct PeriodicProducer {
        /// The time between producing messages.
        period: DiscreteTime,

        /// The agent to which we direct messages.
        target: String,
    }

    impl Agent for PeriodicProducer {
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
pub fn periodic_consumer<T>(name: T, period: DiscreteTime) -> AgentInitializer
where
    T: Into<String>,
{
    #[derive(Clone, Debug)]
    struct PeriodicConsumer {
        /// The amount of time between consuming.
        period: DiscreteTime,
    }

    impl Agent for PeriodicConsumer {
        fn on_message(&mut self, ctx: &mut AgentContext, _msg: &Message) {
            ctx.sleep_for(self.period);
        }
    }

    AgentInitializer {
        agent: Box::new(PeriodicConsumer { period }),
        options: AgentOptions::defaults_with_name(name.into()),
    }
}
