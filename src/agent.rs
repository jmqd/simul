use crate::{message::Message, DiscreteTime};
use dyn_clone::DynClone;
use rand_distr::Distribution;
use rand_distr::Poisson;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;

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

/// A dense agent identity: the agent's position in
/// [`SimulationParameters::agent_initializers`](crate::SimulationParameters).
///
/// Sending by identity with [`AgentContext::send_to`] skips the name lookup
/// that [`AgentContext::send`] performs. Identities store `index + 1`, so
/// `Option<AgentId>` takes four bytes.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AgentId(NonZeroU32);

impl AgentId {
    /// Creates the identity of the agent at `index` in construction order.
    ///
    /// # Panics
    ///
    /// Panics if `index` is `u32::MAX`; no simulation has that many agents.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        match NonZeroU32::new(index.wrapping_add(1)) {
            Some(stored) => Self(stored),
            None => panic!("agent index u32::MAX has no identity"),
        }
    }

    /// Returns the agent's position in construction order.
    #[must_use]
    pub const fn index(self) -> usize {
        (self.0.get() - 1) as usize
    }

    /// Converts the handle of an agent in a constructed simulation.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "simulations reject more than u32::MAX agents, so handles stay below u32::MAX"
    )]
    pub(crate) const fn from_handle(handle: usize) -> Self {
        Self(NonZeroU32::MIN.saturating_add(handle as u32))
    }
}

impl fmt::Debug for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AgentId").field(&self.index()).finish()
    }
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

    /// Engine-managed mode, queue, and message histories. Callbacks see it
    /// read-only through [`AgentContext::state`]; callers may edit it between
    /// runs.
    pub state: AgentState,
}

impl SimulationAgent {
    /// Records the current queue depth for this tick.
    #[inline]
    pub(crate) fn record_queue_depth(&mut self) {
        self.metadata
            .queue_depth_metrics
            .push(self.state.queue.len());
    }
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
    /// Stop the simulation
    HaltSimulation(String),
}

pub enum MessageProcessingStatus {
    NoError,
    InProgress,
}

/// The capability for agents to act on the world.
///
/// Routing identities are read-only to callbacks.
///
/// ```compile_fail
/// use simul::agent::{AgentContext, AgentId};
/// fn change_identity(ctx: &mut AgentContext<'_>) {
///     ctx.id = AgentId::new(0);
/// }
/// ```
pub struct AgentContext<'a> {
    /// The identity of the Agent.
    pub(crate) id: AgentId,

    /// The name of the Agent.
    pub name: &'a str,

    /// The current simulation time.
    pub time: DiscreteTime,

    /// Internal buffer for messages and halt requests.
    pub(crate) commands: &'a mut Vec<AgentCommand>,

    /// Sleep requested by the agent during the current tick.
    pub(crate) requested_sleep_until: &'a mut Option<DiscreteTime>,

    /// Agents before and after the caller, which small simulations scan to
    /// route messages instead of hashing every target.
    pub(crate) peers: Option<(&'a [SimulationAgent], &'a [SimulationAgent])>,

    /// Maps agent names to identities when `peers` is absent.
    pub(crate) agent_name_handle_map: &'a FxHashMap<String, AgentId>,

    /// Number of agents, which bounds valid identities.
    pub(crate) agent_count: usize,

    /// Identifies the agent layout, which validates cached target identities.
    pub(crate) layout: u64,

    pub state: &'a AgentState,

    pub message_processing_status: MessageProcessingStatus,
}

impl AgentContext<'_> {
    /// Returns the agent's immutable identity.
    #[must_use]
    pub const fn id(&self) -> AgentId {
        self.id
    }

    /// Queues a message to the agent named `target` for end-of-tick delivery.
    ///
    /// Multiple messages sent by one callback are delivered in call order. A
    /// target that names no agent is recorded in the sender's history without
    /// a destination and is not delivered. Forward a received payload by
    /// passing `message.custom_payload.clone()`, which shares its bytes.
    #[inline]
    pub fn send(&mut self, target: &str, payload: Option<Arc<[u8]>>) {
        let destination = self.agent_id(target);
        self.queue_message(destination, payload);
    }

    /// Queues a message to `target` without a name lookup.
    ///
    /// Delivery order matches [`Self::send`]. An identity outside this
    /// simulation is recorded like an unknown name.
    #[inline]
    pub fn send_to(&mut self, target: AgentId, payload: Option<Arc<[u8]>>) {
        let destination = (target.index() < self.agent_count).then_some(target);
        self.queue_message(destination, payload);
    }

    /// Returns the identity of the agent named `name`, if one exists.
    ///
    /// Callbacks that message one peer repeatedly can resolve it once and use
    /// [`Self::send_to`].
    #[must_use]
    #[inline]
    pub fn agent_id(&self, name: &str) -> Option<AgentId> {
        match self.peers {
            Some((before, after)) => {
                if self.name == name {
                    Some(self.id)
                } else {
                    before
                        .iter()
                        .position(|agent| agent.name == name)
                        .or_else(|| {
                            after
                                .iter()
                                .position(|agent| agent.name == name)
                                .map(|offset| self.id.index() + 1 + offset)
                        })
                        .map(AgentId::from_handle)
                }
            }
            None => self.agent_name_handle_map.get(name).copied(),
        }
    }

    /// Queues a message from this agent for end-of-tick processing.
    #[inline]
    fn queue_message(&mut self, destination: Option<AgentId>, payload: Option<Arc<[u8]>>) {
        self.commands.push(AgentCommand {
            ty: AgentCommandType::SendMessage(Message::sent(
                self.time,
                self.id,
                destination,
                payload,
            )),
            agent_handle: self.id.index(),
        });
    }

    /// Queues a message to a built-in agent's target, resolving it at most once
    /// per simulation layout.
    #[inline]
    fn send_cached(&mut self, target: &mut CachedTarget, payload: Option<Arc<[u8]>>) {
        let destination = match target.resolved {
            Some((layout, destination)) if layout == self.layout => destination,
            _ => {
                let destination = self.agent_id(&target.name);
                target.resolved = Some((self.layout, destination));
                destination
            }
        };
        self.queue_message(destination, payload);
    }

    /// Sends an interrupt to HALT the simulation.
    #[inline]
    pub fn send_halt_interrupt(&mut self, reason: &str) {
        self.commands.push(AgentCommand {
            ty: AgentCommandType::HaltSimulation(reason.to_string()),
            agent_handle: self.id.index(),
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

/// A built-in agent's message target and its identity in the last layout.
///
/// Agents can be cloned into simulations with other layouts, so the resolved
/// identity is reused only while the layout matches.
#[derive(Clone)]
struct CachedTarget {
    /// Name of the target agent.
    name: String,
    /// Layout and identity of the last resolution; `None` before the first.
    resolved: Option<(u64, Option<AgentId>)>,
}

// The layout token comes from a process-wide counter, so printing it would make
// agent Debug output depend on how many simulations ran earlier.
impl fmt::Debug for CachedTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CachedTarget")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl CachedTarget {
    /// Creates an unresolved target.
    const fn new(name: String) -> Self {
        Self {
            name,
            resolved: None,
        }
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
        target: CachedTarget,
    }

    impl Agent for PoissonAgent {
        fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        fn on_tick(&mut self, ctx: &mut AgentContext) {
            let cooldown_period = self.period.sample(&mut rand::rng()) as u64;
            ctx.sleep_for(cooldown_period);
            ctx.send_cached(&mut self.target, None);
        }
    }

    AgentInitializer {
        agent: Box::new(PoissonAgent {
            period: dist,
            target: CachedTarget::new(target.into()),
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
        target: CachedTarget,
    }

    impl Agent for PeriodicProducer {
        #[allow(unused_variables)]
        fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {}

        fn on_tick(&mut self, ctx: &mut AgentContext) {
            ctx.sleep_for(self.period);
            ctx.send_cached(&mut self.target, None);
        }
    }

    AgentInitializer {
        agent: Box::new(PeriodicProducer {
            period,
            target: CachedTarget::new(target.into()),
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
