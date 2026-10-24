use crate::experiment::{AgentStreams, TrialRng};
use crate::{message::Message, DiscreteTime};
use dyn_clone::DynClone;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::fmt;
use std::num::NonZeroU32;

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

/// A dense agent identity: the agent's position in the order agents were
/// added to the [`SimulationBuilder`](crate::SimulationBuilder).
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

/// A queue depth that holds from `time` until the next change.
///
/// The depth is the one an agent's visit at that tick observes: after it pops
/// its message, before end-of-tick deliveries. Expanding the changes step-wise
/// over the ticks run reproduces a per-tick series.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct QueueDepthChange {
    /// The first tick with this depth.
    pub time: DiscreteTime,
    /// The number of queued messages.
    pub depth: usize,
}

/// Per-agent bookkeeping collected by the simulation engine.
#[derive(Clone, Debug, Default)]
pub(crate) struct AgentMetadata {
    /// Queue depth changes in time order, without repeated depths.
    pub queue_depth_metrics: Vec<QueueDepthChange>,

    /// The number of cycles this agent was sleeping for.
    pub asleep_cycle_count: DiscreteTime,

    /// Messages this agent finished handling.
    pub consumed_count: u64,

    /// Messages this agent sent, delivered or not.
    pub produced_count: u64,

    /// Queued-to-completion waits of consumed messages, summed in
    /// consumption order; negative for messages queued in the future.
    pub total_wait: f64,
}

/// Every message an agent consumed and produced, in order.
#[derive(Clone, Debug)]
pub(crate) struct MessageHistory<M> {
    /// Consumed messages, each with its completion time.
    pub consumed: Vec<Message<M>>,

    /// Sent messages, including those addressed to no agent.
    pub produced: Vec<Message<M>>,

    /// Copies a sent message into `produced`; only history needs `M: Clone`.
    pub copy: fn(&Message<M>) -> Message<M>,
}

#[derive(Debug, Clone)]
pub struct SimulationAgent<M = ()> {
    /// The agent that the user constructed; contains user state and behavior.
    pub agent: Box<dyn Agent<M>>,

    /// The metadata associated with the agent.
    pub(crate) metadata: AgentMetadata,

    /// The name for the agent. Must be unique for the simulation.
    pub name: String,

    /// Engine-managed mode and queue. Callbacks see it read-only through
    /// [`AgentContext::state`]; callers may edit it between runs.
    pub state: AgentState<M>,

    /// `None` unless the simulation records message history.
    pub(crate) history: Option<MessageHistory<M>>,
}

impl<M> SimulationAgent<M> {
    /// Returns the agent's behavior as an `A`, or `None` if it is another
    /// type. Read a finished simulation's agents this way rather than sharing
    /// state with them through `Rc`, locks, or atomics.
    #[must_use]
    pub fn downcast_ref<A: Agent<M> + 'static>(&self) -> Option<&A> {
        // Through the trait object: the box itself is `Any` too.
        sealed::AsAny::as_any(&*self.agent).downcast_ref()
    }
}

impl<M> SimulationAgent<M> {
    /// Returns how many messages this agent finished handling.
    #[must_use]
    pub const fn consumed_count(&self) -> u64 {
        self.metadata.consumed_count
    }

    /// Returns how many messages this agent sent, including sends to names
    /// that match no agent.
    #[must_use]
    pub const fn produced_count(&self) -> u64 {
        self.metadata.produced_count
    }

    /// Returns the messages this agent finished handling, oldest first, each
    /// with its completion time.
    ///
    /// # Panics
    ///
    /// Panics unless the simulation records message history; see
    /// [`SimulationBuilder::message_history`](crate::SimulationBuilder::message_history).
    #[must_use]
    pub fn consumed(&self) -> &[Message<M>] {
        &self.history().consumed
    }

    /// Returns the messages this agent sent, oldest first, including those
    /// addressed to no agent.
    ///
    /// # Panics
    ///
    /// Panics unless the simulation records message history; see
    /// [`SimulationBuilder::message_history`](crate::SimulationBuilder::message_history).
    #[must_use]
    pub fn produced(&self) -> &[Message<M>] {
        &self.history().produced
    }

    /// Returns the recorded history; asking for one that was never recorded
    /// is a configuration bug, and an empty answer would hide it.
    fn history(&self) -> &MessageHistory<M> {
        self.history.as_ref().unwrap_or_else(|| {
            panic!(
                "message history is off; enable it with SimulationBuilder::message_history(true)"
            )
        })
    }

    /// Counts a finished message and records it when history is on.
    pub(crate) fn record_consumed(&mut self, mut message: Message<M>, time: DiscreteTime) {
        self.metadata.consumed_count += 1;
        // Subtract integers first so short waits stay exact at large times.
        self.metadata.total_wait += if time >= message.queued_time {
            (time - message.queued_time) as f64
        } else {
            -((message.queued_time - time) as f64)
        };
        if let Some(history) = &mut self.history {
            message.completed_time = Some(time);
            history.consumed.push(message);
        }
    }

    /// Records that the current queue depth holds from `time` on.
    ///
    /// Several events can land on one tick (a delivery visible from tick `t`,
    /// then the pop at `t`), so a change at the latest time is overwritten and
    /// dropped if it no longer differs from the one before it.
    pub(crate) fn record_queue_depth(&mut self, time: DiscreteTime) {
        let depth = self.state.queue.len();
        let changes = &mut self.metadata.queue_depth_metrics;
        match changes.last_mut() {
            Some(last) if last.time == time => {
                last.depth = depth;
                if let [.., previous, _] = changes.as_slice() {
                    if previous.depth == depth {
                        changes.pop();
                    }
                }
            }
            Some(last) if last.depth == depth => {}
            _ => changes.push(QueueDepthChange { time, depth }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentState<M = ()> {
    /// The mode that the agent is in.
    pub mode: AgentMode,

    /// The mode that the agent wakes up into.
    pub wake_mode: AgentMode,

    /// The queue of incoming Messages for the Agent.
    pub queue: VecDeque<Message<M>>,
}

impl<M> Default for AgentState<M> {
    fn default() -> Self {
        Self {
            mode: AgentMode::Dead,
            wake_mode: AgentMode::Dead,
            queue: VecDeque::new(),
        }
    }
}

/// A buffered callback action, applied at the end of the tick.
pub(crate) struct AgentCommand<M> {
    /// What the agent asked for.
    pub(crate) ty: AgentCommandType<M>,
    /// Handle of the agent that issued the command.
    pub(crate) agent_handle: usize,
}

/// Actions an agent can request from a callback.
pub(crate) enum AgentCommandType<M> {
    /// Record the message as produced and deliver it to its destination.
    SendMessage(Message<M>),
    /// Stop the simulation after this tick; the payload is the reason.
    HaltSimulation(String),
}

/// Agents before and after the visited one, which small simulations scan to
/// route messages by name.
pub(crate) type Peers<'a, M> = (&'a [SimulationAgent<M>], &'a [SimulationAgent<M>]);

/// Whether the message an agent just handled is finished.
///
/// Set it from [`Agent::on_message`] with
/// [`AgentContext::set_processing_status`]; each visit starts as `Done`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MessageProcessingStatus {
    /// The message is consumed: it is counted, its wait is recorded, and it
    /// leaves the queue.
    Done,
    /// The message stays at the front of the queue and is handled again on
    /// the agent's next visit.
    InProgress,
}

/// The capability for agents to act on the world.
///
/// Identity, name, time, and state are read-only to callbacks.
///
/// ```compile_fail
/// use simul::agent::AgentContext;
/// fn rename(ctx: &mut AgentContext<'_>) {
///     ctx.name = "someone else";
/// }
/// ```
pub struct AgentContext<'a, M = ()> {
    /// The identity of the Agent.
    pub(crate) id: AgentId,

    /// The name of the Agent, which small simulations also use to recognize
    /// sends to itself.
    pub(crate) name: &'a str,

    /// The current simulation time.
    pub(crate) time: DiscreteTime,

    /// Internal buffer for messages and halt requests.
    pub(crate) commands: &'a mut Vec<AgentCommand<M>>,

    /// Sleep requested by the agent during the current tick.
    pub(crate) requested_sleep_until: &'a mut Option<DiscreteTime>,

    /// Agents before and after the caller, which small simulations scan to
    /// route messages instead of hashing every target.
    pub(crate) peers: Option<Peers<'a, M>>,

    /// Maps agent names to identities when `peers` is absent.
    pub(crate) agent_name_handle_map: &'a FxHashMap<String, AgentId>,

    /// Number of agents, which bounds valid identities.
    pub(crate) agent_count: usize,

    /// Identifies the agent layout, which validates cached target identities.
    pub(crate) layout: u64,

    /// The agent's mode and queue as of this callback.
    pub(crate) state: &'a AgentState<M>,

    /// Whether the current message finished or stays queued.
    pub(crate) message_processing_status: MessageProcessingStatus,

    /// This agent's stream root.
    pub(crate) stream_root: &'a AgentStreams,

    /// Number of streams handed out during this visit.
    pub(crate) calls: u64,
}

impl<'a, M> AgentContext<'a, M> {
    /// Returns the agent's immutable identity.
    #[must_use]
    pub const fn id(&self) -> AgentId {
        self.id
    }

    /// Returns a new random stream for this agent and tick.
    ///
    /// The `n`th call during a tick (counting `on_message` and `on_tick`
    /// together) returns window `n` of the tick's stream, which is keyed by
    /// the simulation seed and this agent's name. Equal seeds therefore give
    /// equal draws regardless of other agents, their order, scheduling,
    /// clones, or how a run is split. Each call seeks a fresh `ChaCha12`
    /// stream, so keep it for several draws rather than calling this per draw.
    #[must_use]
    pub fn rng(&mut self) -> TrialRng {
        let call = self.calls;
        self.calls += 1;
        self.stream_root.rng(self.time, call)
    }

    /// Returns the agent's name.
    #[must_use]
    pub const fn name(&self) -> &'a str {
        self.name
    }

    /// Returns the tick being executed.
    #[must_use]
    pub const fn time(&self) -> DiscreteTime {
        self.time
    }

    /// Returns the agent's state as of this callback. During `on_message`, the
    /// current message has already been popped from the queue.
    #[must_use]
    pub const fn state(&self) -> &'a AgentState<M> {
        self.state
    }

    /// Queues `payload` to the agent named `target` for end-of-tick delivery.
    ///
    /// Multiple messages sent by one callback are delivered in call order. A
    /// target that names no agent is counted as produced, kept without a
    /// destination in the sender's history when that is recorded, and not
    /// delivered.
    #[inline]
    pub fn send(&mut self, target: &str, payload: M) {
        let destination = self.agent_id(target);
        self.queue_message(destination, payload);
    }

    /// Queues `payload` to `target` without a name lookup.
    ///
    /// Delivery order matches [`Self::send`]. An identity outside this
    /// simulation is recorded like an unknown name.
    #[inline]
    pub fn send_to(&mut self, target: AgentId, payload: M) {
        let destination = (target.index() < self.agent_count).then_some(target);
        self.queue_message(destination, payload);
    }

    /// Queues `payload` to `target`, visible to it from tick
    /// `time() + delay`; `send_after(1, ..)` is [`Self::send_to`].
    ///
    /// Use it for latency and for timers, such as a self-addressed timeout,
    /// instead of polling every tick: a waiting agent stays idle, so quiet
    /// ticks can still be skipped. The message counts as sent now but carries
    /// `queued_time` `time() + delay - 1`, so it queues, records, and reports
    /// its wait as if sent that tick. Messages arriving in one tick come in
    /// send order, so delayed ones precede that tick's ordinary sends. A run
    /// that halts first leaves it in [`Simulation::in_flight`](crate::Simulation::in_flight).
    ///
    /// # Panics
    ///
    /// Panics if `delay` is zero, since a message cannot arrive in the tick it
    /// is sent, or if the arrival tick exceeds `u64::MAX`.
    #[inline]
    pub fn send_after(&mut self, delay: DiscreteTime, target: AgentId, payload: M) {
        assert!(delay > 0, "a message cannot arrive in the tick it is sent");
        let Some(arrival) = self.time.checked_add(delay) else {
            panic!("message arrival exceeds u64::MAX");
        };
        let destination = (target.index() < self.agent_count).then_some(target);
        self.commands.push(AgentCommand {
            ty: AgentCommandType::SendMessage(Message::sent(
                arrival - 1,
                self.id,
                destination,
                payload,
            )),
            agent_handle: self.id.index(),
        });
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
    fn queue_message(&mut self, destination: Option<AgentId>, payload: M) {
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
    fn send_cached(&mut self, target: &mut CachedTarget, payload: M) {
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

    /// Marks the message being handled as [`MessageProcessingStatus::Done`]
    /// (the default) or [`MessageProcessingStatus::InProgress`].
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

/// How an agent starts a simulation: its name, modes, and initial queue.
///
/// [`AgentInitializer::reactive`] and [`AgentInitializer::proactive`] fill
/// these in; set the fields directly only for unusual mode combinations.
#[derive(Debug, Clone)]
pub struct AgentOptions<M = ()> {
    /// The mode the agent starts in.
    pub initial_mode: AgentMode,
    /// The mode the agent returns to when a sleep ends.
    pub wake_mode: AgentMode,
    /// Messages queued before the first tick, oldest first. The engine sets
    /// each one's `destination` to this agent.
    pub initial_queue: VecDeque<Message<M>>,
    /// The agent's name, unique within its simulation.
    pub name: String,
}

// A derive would demand `M: Default` just to start with an empty queue.
impl<M> Default for AgentOptions<M> {
    fn default() -> Self {
        Self {
            initial_mode: AgentMode::default(),
            wake_mode: AgentMode::default(),
            initial_queue: VecDeque::new(),
            name: String::new(),
        }
    }
}

/// An agent's behavior together with how it starts the simulation.
///
/// ```
/// use simul::agent::{Agent, AgentContext, AgentInitializer};
/// use simul::message::Message;
///
/// #[derive(Clone, Debug)]
/// struct Clock;
///
/// impl Agent for Clock {
///     fn on_tick(&mut self, ctx: &mut AgentContext) {
///         ctx.sleep_for(10);
///     }
/// }
///
/// // Ticks every 10 ticks, starting at tick 5, with one message waiting.
/// let clock = AgentInitializer::proactive("clock", Clock)
///     .asleep_until(5)
///     .initial_queue([Message::new(0, ())]);
/// ```
#[derive(Debug, Clone)]
pub struct AgentInitializer<M = ()> {
    /// The agent's behavior and state.
    pub agent: Box<dyn Agent<M>>,
    /// The agent's name, modes, and initial queue.
    pub options: AgentOptions<M>,
}

impl<M> AgentInitializer<M> {
    /// An agent that acts only on messages: it starts `Reactive` and returns
    /// to `Reactive` after every sleep.
    pub fn reactive(name: impl Into<String>, agent: impl Agent<M> + 'static) -> Self {
        Self::with_mode(name.into(), Box::new(agent), AgentMode::Reactive)
    }

    /// An agent that also acts on its own: it starts `Proactive`, gets
    /// [`Agent::on_tick`] every tick it is awake, and returns to `Proactive`
    /// after every sleep.
    pub fn proactive(name: impl Into<String>, agent: impl Agent<M> + 'static) -> Self {
        Self::with_mode(name.into(), Box::new(agent), AgentMode::Proactive)
    }

    /// Builds an initializer that starts in and wakes into `mode`.
    fn with_mode(name: String, agent: Box<dyn Agent<M>>, mode: AgentMode) -> Self {
        Self {
            agent,
            options: AgentOptions {
                initial_mode: mode,
                wake_mode: mode,
                initial_queue: VecDeque::new(),
                name,
            },
        }
    }

    /// Starts the agent asleep until tick `time`; it then wakes into the mode
    /// it was constructed with.
    #[must_use]
    pub const fn asleep_until(mut self, time: DiscreteTime) -> Self {
        self.options.initial_mode = AgentMode::AsleepUntil(time);
        self
    }

    /// Replaces the messages queued for the agent before the first tick,
    /// oldest first.
    #[must_use]
    pub fn initial_queue(mut self, messages: impl IntoIterator<Item = Message<M>>) -> Self {
        self.options.initial_queue = messages.into_iter().collect();
        self
    }
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
///
/// `M` is the simulation's message type; agents that never look at payloads
/// can implement `Agent<M>` for every `M` and so serve any simulation.
///
/// Each tick, an awake agent that has a queued message receives the oldest one
/// through [`Agent::on_message`]. A `Proactive` agent then also gets
/// [`Agent::on_tick`], whether or not it had a message. Sleeping and `Dead`
/// agents get neither.
pub trait Agent<M = ()>: std::fmt::Debug + DynClone + sealed::AsAny {
    /// Handles the oldest queued message, at most once per tick. The message
    /// counts as consumed afterwards unless the agent sets
    /// [`MessageProcessingStatus::InProgress`]. The default does nothing.
    #[allow(unused_variables)]
    fn on_message(&mut self, ctx: &mut AgentContext<'_, M>, msg: &Message<M>) {}

    /// Runs once per tick while the agent is `Proactive`, after any
    /// `on_message` in the same tick. The default does nothing.
    #[allow(unused_variables)]
    fn on_tick(&mut self, ctx: &mut AgentContext<'_, M>) {}
}

dyn_clone::clone_trait_object!(<M> Agent<M>);

/// The `Any` view behind [`SimulationAgent::downcast_ref`]. Every `'static`
/// type has it, so agents never implement it themselves; it is a supertrait
/// because only supertrait methods are reachable through `dyn Agent<M>`.
/// It deliberately lacks an `Any` supertrait, which would make every call
/// through `dyn Agent<M>` require `M: 'static`.
mod sealed {
    use std::any::Any;

    /// Views an agent as `Any`; implemented for every `'static` type.
    #[allow(
        unreachable_pub,
        reason = "sealed supertrait of the public Agent trait; the module is private"
    )]
    pub trait AsAny {
        /// Returns the agent as `Any`.
        fn as_any(&self) -> &dyn Any;
    }

    impl<T: Any> AsAny for T {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
}

/// A geometric number of ticks, at least one, with a fixed mean.
///
/// Geometric gaps are memoryless, the discrete-time counterpart of exponential
/// gaps, so a producer with them emits a Bernoulli arrival process.
#[derive(Clone, Copy, Debug)]
struct GeometricTicks {
    /// `ln(1 - 1/mean)`, negative or `-inf` when the mean is one tick.
    log_continue: f64,
}

impl GeometricTicks {
    /// # Panics
    ///
    /// Panics unless `mean` is finite and at least one tick.
    fn new(mean: f64) -> Self {
        assert!(
            mean.is_finite() && mean >= 1.0,
            "mean period must be finite and at least one tick, got {mean}"
        );
        Self {
            log_continue: (-mean.recip()).ln_1p(),
        }
    }

    /// Samples by inversion: `1 + floor(ln(U) / ln(1 - p))` for `U` in `(0, 1]`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the quotient is zero or positive, and `as` saturates huge gaps"
    )]
    fn sample(self, rng: &mut TrialRng) -> DiscreteTime {
        let unit = 1.0 - rng.uniform_f64();
        let extra = (unit.ln() / self.log_continue).floor() as DiscreteTime;
        extra.saturating_add(1)
    }
}

/// A `Reactive` consumer whose service time per message is a geometric number
/// of ticks with mean `mean_service_ticks`, drawn from [`AgentContext::rng`].
///
/// # Panics
///
/// Panics unless `mean_service_ticks` is finite and at least one tick.
pub fn geometric_consumer<M>(
    name: impl Into<String>,
    mean_service_ticks: f64,
) -> AgentInitializer<M> {
    #[derive(Debug, Clone)]
    struct GeometricConsumer {
        /// Ticks spent on each message.
        service: GeometricTicks,
    }

    impl<M> Agent<M> for GeometricConsumer {
        fn on_message(&mut self, ctx: &mut AgentContext<'_, M>, _msg: &Message<M>) {
            let ticks = self.service.sample(&mut ctx.rng());
            ctx.sleep_for(ticks);
        }
    }

    AgentInitializer::reactive(
        name,
        GeometricConsumer {
            service: GeometricTicks::new(mean_service_ticks),
        },
    )
}

/// A `Proactive` producer that sends `M::default()` to `target`.
///
/// After each send it waits a geometric number of ticks with mean
/// `mean_interarrival_ticks`, drawn from [`AgentContext::rng`], so each tick
/// carries an arrival with probability `1 / mean_interarrival_ticks`.
///
/// # Panics
///
/// Panics unless `mean_interarrival_ticks` is finite and at least one tick.
pub fn geometric_producer<M: Default>(
    name: impl Into<String>,
    mean_interarrival_ticks: f64,
    target: impl Into<String>,
) -> AgentInitializer<M> {
    #[derive(Clone, Debug)]
    struct GeometricProducer {
        /// Ticks between arrivals.
        interarrival: GeometricTicks,

        /// The target to which we produce messages.
        target: CachedTarget,
    }

    impl<M: Default> Agent<M> for GeometricProducer {
        fn on_tick(&mut self, ctx: &mut AgentContext<'_, M>) {
            let ticks = self.interarrival.sample(&mut ctx.rng());
            ctx.sleep_for(ticks);
            ctx.send_cached(&mut self.target, M::default());
        }
    }

    AgentInitializer::proactive(
        name,
        GeometricProducer {
            interarrival: GeometricTicks::new(mean_interarrival_ticks),
            target: CachedTarget::new(target.into()),
        },
    )
}

/// A simple agent that sends `M::default()` to `target` on a period.
pub fn periodic_producer<M: Default>(
    name: impl Into<String>,
    period: DiscreteTime,
    target: impl Into<String>,
) -> AgentInitializer<M> {
    #[derive(Clone, Debug)]
    struct PeriodicProducer {
        /// The time between producing messages.
        period: DiscreteTime,

        /// The agent to which we direct messages.
        target: CachedTarget,
    }

    impl<M: Default> Agent<M> for PeriodicProducer {
        fn on_tick(&mut self, ctx: &mut AgentContext<'_, M>) {
            ctx.sleep_for(self.period);
            ctx.send_cached(&mut self.target, M::default());
        }
    }

    AgentInitializer::proactive(
        name,
        PeriodicProducer {
            period,
            target: CachedTarget::new(target.into()),
        },
    )
}

/// A simple agent that consumes messages on a period with no side effects.
/// Period can be thought of the time to consume 1 message.
pub fn periodic_consumer<M>(name: impl Into<String>, period: DiscreteTime) -> AgentInitializer<M> {
    #[derive(Clone, Debug)]
    struct PeriodicConsumer {
        /// The amount of time between consuming.
        period: DiscreteTime,
    }

    impl<M> Agent<M> for PeriodicConsumer {
        fn on_message(&mut self, ctx: &mut AgentContext<'_, M>, _msg: &Message<M>) {
            ctx.sleep_for(self.period);
        }
    }

    AgentInitializer::reactive(name, PeriodicConsumer { period })
}
