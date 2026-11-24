use crate::experiment::{AgentStreams, MasterSeed, SeedDeriver, TrialRng};
use crate::{DiscreteTime, message::Message};
use dyn_clone::DynClone;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::fmt;
use std::num::{NonZeroU32, NonZeroUsize};

/// Possible states an Agent can be in.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Default)]
pub enum AgentMode {
    /// The Agent is active; `on_tick()` is expected to do work.
    Proactive,

    /// The Agent is reactive; `on_message()` is expected to do work.
    #[default]
    Reactive,

    /// Sleeping until the absolute tick `until`, then returning to `then`.
    ///
    /// Messages keep arriving in the agent's queue while it sleeps. On the
    /// first tick with `time >= until`, the mode becomes `then` and the agent
    /// runs in that same tick.
    AsleepUntil {
        /// The tick at which the agent wakes.
        until: DiscreteTime,
        /// The awake mode the agent returns to.
        then: WakeMode,
    },

    /// Inactive: never called back.
    ///
    /// Messages sent to a dead agent are still appended to its queue and are
    /// never consumed. This is the default mode of [`AgentState`].
    Dead,
}

impl AgentMode {
    /// Returns this mode asleep until `until`. An awake agent wakes back into
    /// its current mode, a sleeping one into the mode it was going to wake
    /// into, and a dead agent stays dead.
    #[must_use]
    pub const fn asleep_until(self, until: DiscreteTime) -> Self {
        match self {
            Self::Proactive => Self::AsleepUntil {
                until,
                then: WakeMode::Proactive,
            },
            Self::Reactive => Self::AsleepUntil {
                until,
                then: WakeMode::Reactive,
            },
            Self::AsleepUntil { then, .. } => Self::AsleepUntil { until, then },
            Self::Dead => Self::Dead,
        }
    }
}

/// The awake modes a sleeping agent can return to.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Default)]
pub enum WakeMode {
    /// Wake into [`AgentMode::Proactive`].
    Proactive,
    /// Wake into [`AgentMode::Reactive`].
    #[default]
    Reactive,
}

impl From<WakeMode> for AgentMode {
    fn from(mode: WakeMode) -> Self {
        match mode {
            WakeMode::Proactive => Self::Proactive,
            WakeMode::Reactive => Self::Reactive,
        }
    }
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

/// Handle to a delayed message scheduled with [`AgentContext::send_after`].
///
/// Pass this handle to [`AgentContext::cancel_timer`] or
/// [`Simulation::cancel_timer`](crate::Simulation::cancel_timer) to cancel the
/// delayed message before it arrives.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TimerId {
    /// Tick on which the delayed message becomes visible.
    pub(crate) arrival: DiscreteTime,
    /// Monotonic sequence number in send order.
    pub(crate) sequence: u64,
}

impl TimerId {
    /// Returns the tick on which this delayed message is scheduled to arrive.
    #[must_use]
    pub const fn arrival(self) -> DiscreteTime {
        self.arrival
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

/// An agent inside a simulation: its behavior, name, engine-managed state, and
/// message counts.
///
/// Read agents after a run with [`Simulation::agents`](crate::Simulation::agents),
/// [`Simulation::agent`](crate::Simulation::agent), or
/// [`Simulation::find_by_name`](crate::Simulation::find_by_name), and get the
/// behavior back as its own type with [`SimulationAgent::downcast_ref`] or
/// [`SimulationAgent::downcast_mut`].
#[derive(Debug, Clone)]
pub struct SimulationAgent<M = ()> {
    /// The agent that the user constructed; contains user state and behavior.
    pub agent: Box<dyn Agent<M>>,

    /// The metadata associated with the agent.
    pub(crate) metadata: AgentMetadata,

    /// The name for the agent. Must be unique for the simulation and never
    /// changes after construction.
    pub(crate) name: String,

    /// Engine-managed mode and queue. Callbacks see it read-only through
    /// [`AgentContext::state`]; callers may edit it between runs.
    pub state: AgentState<M>,

    /// `None` unless the simulation records message history.
    pub(crate) history: Option<MessageHistory<M>>,

    /// How many queued messages this agent may handle in one tick.
    pub(crate) message_drain: MessageDrain,
}

impl<M> SimulationAgent<M> {
    /// Returns the agent's unique, immutable name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns how many queued messages this agent may handle in a single tick.
    #[must_use]
    pub const fn message_drain(&self) -> MessageDrain {
        self.message_drain
    }

    /// Changes how many queued messages this agent may handle in a single tick.
    pub const fn set_message_drain(&mut self, drain: MessageDrain) {
        self.message_drain = drain;
    }

    /// Returns the agent's behavior as an `A`, or `None` if it is another
    /// type. Read a finished simulation's agents this way rather than sharing
    /// state with them through `Rc`, locks, or atomics.
    #[must_use]
    pub fn downcast_ref<A: Agent<M> + 'static>(&self) -> Option<&A> {
        self.agent.downcast_ref()
    }

    /// Returns the agent's behavior mutably as an `A`, or `None` if it is
    /// another type. Use it through [`Simulation::find_by_name_mut`] or
    /// [`Simulation::agent_mut`] to patch an agent between runs, such as one
    /// branch's policy after a clone.
    ///
    /// [`Simulation::find_by_name_mut`]: crate::Simulation::find_by_name_mut
    /// [`Simulation::agent_mut`]: crate::Simulation::agent_mut
    #[must_use]
    pub fn downcast_mut<A: Agent<M> + 'static>(&mut self) -> Option<&mut A> {
        self.agent.downcast_mut()
    }

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

    /// Returns the mean wait from queueing to completion over messages this
    /// agent finished handling, or `None` if it consumed nothing. Needs no
    /// message history.
    #[must_use]
    pub fn avg_wait(&self) -> Option<f64> {
        (self.metadata.consumed_count > 0)
            .then(|| self.metadata.total_wait / self.metadata.consumed_count as f64)
    }

    /// Returns this agent's recorded queue depth changes in time order. Empty
    /// unless [`SimulationBuilder::queue_depth_metrics`](crate::SimulationBuilder::queue_depth_metrics)
    /// is enabled.
    #[must_use]
    pub fn queue_depth_metrics(&self) -> &[QueueDepthChange] {
        &self.metadata.queue_depth_metrics
    }

    /// Returns the number of ticks this agent spent asleep before its wake
    /// tick. Always `0` unless
    /// [`SimulationBuilder::asleep_cycles_metric`](crate::SimulationBuilder::asleep_cycles_metric)
    /// is enabled.
    #[must_use]
    pub const fn asleep_cycle_count(&self) -> DiscreteTime {
        self.metadata.asleep_cycle_count
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

/// The engine-managed part of an agent: its current mode and its queue of
/// messages not yet handled.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AgentState<M = ()> {
    /// The mode that the agent is in; a sleeping agent's mode also holds the
    /// mode it wakes into.
    pub mode: AgentMode,

    /// First-in-first-out inbox of unhandled messages.
    ///
    /// Messages sent during a tick are appended at the end of that tick (or at
    /// arrival for [`AgentContext::send_after`]). Awake agents pop at most one
    /// message per tick; setting [`MessageProcessingStatus::InProgress`] puts
    /// it back at the front.
    pub queue: VecDeque<Message<M>>,
}

impl<M> Default for AgentState<M> {
    fn default() -> Self {
        Self {
            mode: AgentMode::Dead,
            queue: VecDeque::new(),
        }
    }
}

/// An ordinary end-of-tick message send, stored compactly for cache density.
pub(crate) struct PendingSend<M> {
    /// Identity of the agent that sent the message.
    pub(crate) sender: AgentId,
    /// Resolved target identity, or `None` for an unknown destination.
    pub(crate) destination: Option<AgentId>,
    /// Message payload.
    pub(crate) payload: M,
}

/// Infrequent callback actions buffered separately from ordinary sends.
pub(crate) enum SpecialCommand<M> {
    /// Record the message as produced and deliver it at `timer.arrival` unless
    /// cancelled first.
    SendDelayed {
        /// Identity of the sending agent.
        sender: AgentId,
        /// Handle identifying this delayed message.
        timer: TimerId,
        /// The delayed message to deliver on arrival.
        message: Message<M>,
    },
    /// Cancel a pending delayed message if it has not been delivered yet.
    CancelTimer {
        /// Identity of the agent issuing the cancellation.
        sender: AgentId,
        /// Handle of the timer to cancel.
        timer: TimerId,
    },
    /// Pause the simulation after this tick and yield control to an external
    /// controller with the given tag.
    YieldControl {
        /// Identity of the agent requesting the yield.
        sender: AgentId,
        /// Tag supplied to [`AgentContext::yield_control`].
        tag: String,
    },
    /// Stop the simulation after this tick; the payload is the reason.
    HaltSimulation {
        /// Identity of the agent requesting the halt.
        sender: AgentId,
        /// Reason string supplied to [`AgentContext::send_halt_interrupt`].
        reason: String,
    },
}

/// Per-tick buffer of callback actions that keeps ordinary sends compact while
/// preserving exact issue order with infrequent commands.
pub(crate) struct CommandBuffer<M> {
    /// Ordinary end-of-tick sends in issue order.
    pub(crate) sends: Vec<PendingSend<M>>,
    /// Infrequent commands paired with `sends.len()` at issue time so draining
    /// interleaves them in exact issue order.
    pub(crate) specials: Vec<(usize, SpecialCommand<M>)>,
    /// Whether `specials` contains any [`SpecialCommand::CancelTimer`].
    pub(crate) has_cancellations: bool,
}

impl<M> CommandBuffer<M> {
    /// Creates an empty command buffer without allocating.
    pub(crate) const fn new() -> Self {
        Self {
            sends: Vec::new(),
            specials: Vec::new(),
            has_cancellations: false,
        }
    }

    /// Buffers an ordinary end-of-tick send.
    #[inline]
    pub(crate) fn push_send(&mut self, sender: AgentId, destination: Option<AgentId>, payload: M) {
        self.sends.push(PendingSend {
            sender,
            destination,
            payload,
        });
    }

    /// Buffers an infrequent command out of line so ordinary send paths stay
    /// compact.
    #[inline(never)]
    pub(crate) fn push_special(&mut self, command: SpecialCommand<M>) {
        self.specials.push((self.sends.len(), command));
    }
}

/// Agents before and after the visited one, which small simulations scan to
/// route messages by name.
pub(crate) type Peers<'a, M> = (&'a [SimulationAgent<M>], &'a [SimulationAgent<M>]);

/// How many queued messages an awake agent may handle in a single tick.
///
/// Configure per agent with [`AgentInitializer::drain_messages`],
/// [`AgentInitializer::drain_all_messages`], or
/// [`SimulationAgent::set_message_drain`].
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum MessageDrain {
    /// Handle at most one queued message per tick (the default).
    #[default]
    One,
    /// Handle up to `limit` queued messages per tick, stopping early if the
    /// agent sleeps or marks a message [`MessageProcessingStatus::InProgress`].
    UpTo(NonZeroUsize),
    /// Drain every message currently queued at the start of the tick, stopping
    /// early if the agent sleeps or marks a message
    /// [`MessageProcessingStatus::InProgress`].
    All,
}

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

/// Lazily derived per-agent stream roots for one simulation.
///
/// Simulations whose agents never draw randomness allocate nothing and hash
/// nothing at build time.
#[derive(Clone, Debug)]
pub(crate) struct AgentStreamCache {
    /// Root seed for every agent's stream.
    seed: MasterSeed,
    /// Cached stream root per agent handle; empty until the first `rng()` call.
    roots: Vec<Option<AgentStreams>>,
}

impl AgentStreamCache {
    /// Creates an empty cache for `seed`.
    pub(crate) const fn new(seed: MasterSeed) -> Self {
        Self {
            seed,
            roots: Vec::new(),
        }
    }

    /// Returns the cached stream root for `handle`, deriving and storing it on
    /// the first call.
    #[inline]
    pub(crate) fn get_or_derive(
        &mut self,
        handle: usize,
        agent_count: usize,
        name: &str,
    ) -> AgentStreams {
        if let Some(&Some(root)) = self.roots.get(handle) {
            return root;
        }
        self.derive(handle, agent_count, name)
    }

    /// Allocates the cache table on first use and derives one agent's stream root.
    #[cold]
    #[inline(never)]
    fn derive(&mut self, handle: usize, agent_count: usize, name: &str) -> AgentStreams {
        if self.roots.is_empty() {
            self.roots.resize(agent_count, None);
        }
        let root = SeedDeriver::new(self.seed).agent_streams(name);
        self.roots[handle] = Some(root);
        root
    }
}

/// Read-only per-tick metadata shared across every agent callback.
pub(crate) struct TickMeta<'a> {
    /// The current simulation time.
    pub(crate) time: DiscreteTime,
    /// Identifies the agent layout, which validates cached target identities.
    pub(crate) layout: u64,
    /// Number of agents, which bounds valid identities.
    pub(crate) agent_count: usize,
    /// Maps agent names to identities when `peers` is absent.
    pub(crate) agent_name_handle_map: &'a FxHashMap<String, AgentId>,
}

/// Mutable per-simulation state accessed infrequently by callbacks through
/// [`AgentContext`].
#[derive(Clone, Debug)]
pub(crate) struct TickState {
    /// Lazily derived stream roots for this simulation.
    pub(crate) agent_streams: AgentStreamCache,
    /// Monotonic send-order sequence for delayed messages.
    pub(crate) next_in_flight: u64,
}

impl TickState {
    /// Creates initial tick state for `seed`.
    pub(crate) const fn new(seed: MasterSeed) -> Self {
        Self {
            agent_streams: AgentStreamCache::new(seed),
            next_in_flight: 0,
        }
    }
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
    /// Read-only tick metadata shared by every visit in the tick.
    pub(crate) meta: &'a TickMeta<'a>,

    /// Internal buffer for messages and halt requests.
    pub(crate) commands: &'a mut CommandBuffer<M>,

    /// Mutable stream cache and delayed-message sequence counter.
    pub(crate) tick_state: &'a mut TickState,

    /// The agent's mode and queue as of this callback.
    pub(crate) state: &'a AgentState<M>,

    /// The name of the Agent, which small simulations also use to recognize
    /// sends to itself.
    pub(crate) name: &'a str,

    /// Agents before and after the caller, which small simulations scan to
    /// route messages instead of hashing every target.
    pub(crate) peers: Option<&'a Peers<'a, M>>,

    /// Wake time requested via [`Self::sleep_for`] when `sleep_requested` is true.
    pub(crate) sleep_until: DiscreteTime,

    /// Number of streams handed out during this visit.
    pub(crate) calls: u64,

    /// The identity of the Agent.
    pub(crate) id: AgentId,

    /// Whether the current message finished or stays queued.
    pub(crate) message_processing_status: MessageProcessingStatus,

    /// Whether the agent called [`Self::sleep_for`] during this visit.
    pub(crate) sleep_requested: bool,
}

impl<'a, M> AgentContext<'a, M> {
    /// Returns the agent's immutable identity.
    #[must_use]
    pub const fn id(&self) -> AgentId {
        self.id
    }

    /// Returns the wake time requested during this visit, if any.
    #[inline]
    pub(crate) const fn requested_sleep(&self) -> Option<DiscreteTime> {
        if self.sleep_requested {
            Some(self.sleep_until)
        } else {
            None
        }
    }

    /// Returns a new random stream for this agent and tick.
    ///
    /// The `n`th call during a tick (counting `on_message` and `on_tick`
    /// together) returns window `n` of the tick's stream, which is keyed by
    /// the simulation seed and this agent's name (derived on the first call
    /// and cached). Equal seeds therefore give equal draws regardless of other
    /// agents, their order, scheduling, clones, or how a run is split. Each
    /// call seeks a fresh `ChaCha12` stream, so keep it for several draws
    /// rather than calling this per draw.
    #[must_use]
    pub fn rng(&mut self) -> TrialRng {
        let root = self.tick_state.agent_streams.get_or_derive(
            self.id.index(),
            self.meta.agent_count,
            self.name,
        );
        let call = self.calls;
        self.calls += 1;
        root.rng(self.meta.time, call)
    }

    /// Returns the agent's name.
    #[must_use]
    pub const fn name(&self) -> &'a str {
        self.name
    }

    /// Returns the tick currently being executed.
    ///
    /// Ordinary sends in this callback receive this tick as `queued_time`, and
    /// [`Self::sleep_for`] and [`Self::send_after`] count forward from it.
    #[must_use]
    pub const fn time(&self) -> DiscreteTime {
        self.meta.time
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
        let destination = (target.index() < self.meta.agent_count).then_some(target);
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
    /// ```
    /// use simul::agent::{Agent, AgentContext, AgentInitializer};
    /// use simul::message::Message;
    /// use simul::{DiscreteTime, Halt, Simulation};
    ///
    /// #[derive(Clone, Debug, Default)]
    /// struct Timer(Option<DiscreteTime>);
    ///
    /// impl Agent<bool> for Timer {
    ///     fn on_message(&mut self, ctx: &mut AgentContext<'_, bool>, msg: &Message<bool>) {
    ///         if msg.payload {
    ///             self.0 = Some(ctx.time());
    ///         } else {
    ///             ctx.send_after(10, ctx.id(), true);
    ///         }
    ///     }
    /// }
    ///
    /// let mut sim = Simulation::builder(Halt::AtTime(20))
    ///     .agent(
    ///         AgentInitializer::reactive("timer", Timer::default())
    ///             .initial_queue([Message::new(0, false)]),
    ///     )
    ///     .build()?;
    /// sim.run();
    /// let fired_at = sim
    ///     .find_by_name("timer")
    ///     .and_then(|agent| agent.downcast_ref::<Timer>())
    ///     .and_then(|timer| timer.0);
    /// assert_eq!(Some(10), fired_at);
    /// # Ok::<(), simul::SimulationError>(())
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `delay` is zero, since a message cannot arrive in the tick it
    /// is sent, or if the arrival tick exceeds `u64::MAX`.
    ///
    /// Returns a [`TimerId`] that may be passed to [`Self::cancel_timer`] or
    /// [`Simulation::cancel_timer`](crate::Simulation::cancel_timer) to prevent
    /// delivery before `time() + delay`.
    #[inline]
    pub fn send_after(&mut self, delay: DiscreteTime, target: AgentId, payload: M) -> TimerId {
        assert!(delay > 0, "a message cannot arrive in the tick it is sent");
        let Some(arrival) = self.meta.time.checked_add(delay) else {
            panic!("message arrival exceeds u64::MAX");
        };
        let sequence = self.tick_state.next_in_flight;
        self.tick_state.next_in_flight += 1;
        let timer = TimerId { arrival, sequence };
        let destination = (target.index() < self.meta.agent_count).then_some(target);
        let message = Message::sent(arrival - 1, self.id, destination, payload);
        self.commands.push_special(SpecialCommand::SendDelayed {
            sender: self.id,
            timer,
            message,
        });
        timer
    }

    /// Cancels a pending delayed message scheduled with [`Self::send_after`].
    ///
    /// If the message has not yet been delivered, it is removed before its
    /// arrival tick—including a message due on the next tick or scheduled
    /// earlier in the current tick—so it is never delivered and never wakes
    /// quiet-tick skipping. Cancelling an already delivered, already
    /// cancelled, or unknown timer is a no-op.
    #[inline]
    pub fn cancel_timer(&mut self, timer: TimerId) {
        self.commands.has_cancellations = true;
        self.commands.push_special(SpecialCommand::CancelTimer {
            sender: self.id,
            timer,
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
            Some(&(before, after)) => {
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
            None => self.meta.agent_name_handle_map.get(name).copied(),
        }
    }

    /// Queues a message from this agent for end-of-tick processing.
    #[inline]
    fn queue_message(&mut self, destination: Option<AgentId>, payload: M) {
        self.commands.push_send(self.id, destination, payload);
    }

    /// Queues a message to a built-in agent's target, resolving it at most once
    /// per simulation layout.
    #[inline]
    fn send_cached(&mut self, target: &mut CachedTarget, payload: M) {
        let destination = match target.resolved {
            Some((layout, destination)) if layout == self.meta.layout => destination,
            _ => {
                let destination = self.agent_id(&target.name);
                target.resolved = Some((self.meta.layout, destination));
                destination
            }
        };
        self.queue_message(destination, payload);
    }

    /// Requests that the simulation stop after the current tick completes.
    ///
    /// Remaining agents still run this tick and all buffered commands (including
    /// sends issued after this call) are applied in order before the run ends in
    /// [`SimulationMode::Completed`](crate::SimulationMode::Completed). The
    /// first interrupt's `reason` in issue order is returned by
    /// [`Simulation::halt_reason`](crate::Simulation::halt_reason).
    #[inline]
    pub fn send_halt_interrupt(&mut self, reason: &str) {
        self.commands.push_special(SpecialCommand::HaltSimulation {
            sender: self.id,
            reason: reason.to_string(),
        });
    }

    /// Requests that the simulation pause after the current tick completes,
    /// yielding control to an external caller (such as an RL policy, LLM
    /// agent, or counterfactual rollout planner).
    ///
    /// Remaining agents finish this tick and all buffered commands are applied
    /// in order before the simulation enters
    /// [`SimulationMode::Paused`](crate::SimulationMode::Paused). The first
    /// yield in issue order is returned by
    /// [`Simulation::run_until_yield`](crate::Simulation::run_until_yield) and
    /// readable through [`Simulation::yield_tag`](crate::Simulation::yield_tag)
    /// and [`Simulation::yielded_by`](crate::Simulation::yielded_by). A halt
    /// interrupt in the same tick takes precedence over a yield.
    #[inline]
    pub fn yield_control(&mut self, tag: &str) {
        self.commands.push_special(SpecialCommand::YieldControl {
            sender: self.id,
            tag: tag.to_string(),
        });
    }

    /// Sleeps the Agent for a relative amount of time; it then wakes back into
    /// its current mode.
    ///
    /// # Panics
    ///
    /// Panics before changing the sleep request if the wake time exceeds `u64::MAX`.
    #[inline]
    pub const fn sleep_for(&mut self, ticks: DiscreteTime) {
        let Some(wake_time) = self.meta.time.checked_add(ticks) else {
            panic!("agent wake time exceeds u64::MAX");
        };
        self.sleep_until = wake_time;
        self.sleep_requested = true;
    }

    /// Marks the message being handled as [`MessageProcessingStatus::Done`]
    /// (the default) or [`MessageProcessingStatus::InProgress`].
    #[inline]
    pub const fn set_processing_status(&mut self, status: MessageProcessingStatus) {
        self.message_processing_status = status;
    }
}

impl<M: fmt::Debug> fmt::Debug for AgentContext<'_, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentContext")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("time", &self.meta.time)
            .field("state", &self.state)
            .field("message_processing_status", &self.message_processing_status)
            .finish_non_exhaustive()
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

/// How an agent starts a simulation: its name, mode, and initial queue.
///
/// [`AgentInitializer::reactive`] and [`AgentInitializer::proactive`] fill
/// these in.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AgentOptions<M = ()> {
    /// The mode the agent starts in; a sleeping mode also says which mode
    /// the agent wakes into.
    pub initial_mode: AgentMode,
    /// Messages queued before the first tick, oldest first. The engine sets
    /// each one's `destination` to this agent.
    pub initial_queue: VecDeque<Message<M>>,
    /// The agent's name, unique within its simulation.
    pub name: String,
    /// How many queued messages this agent may handle per tick. Defaults to
    /// [`MessageDrain::One`].
    pub message_drain: MessageDrain,
}

// A derive would demand `M: Default` just to start with an empty queue.
impl<M> Default for AgentOptions<M> {
    fn default() -> Self {
        Self {
            initial_mode: AgentMode::default(),
            initial_queue: VecDeque::new(),
            name: String::new(),
            message_drain: MessageDrain::One,
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
    /// The agent's name, mode, and initial queue.
    pub options: AgentOptions<M>,
}

impl<M> AgentInitializer<M> {
    /// An agent that acts only on messages: it starts `Reactive` and returns
    /// to `Reactive` after every sleep.
    #[must_use]
    pub fn reactive(name: impl Into<String>, agent: impl Agent<M> + 'static) -> Self {
        Self::with_mode(name.into(), Box::new(agent), AgentMode::Reactive)
    }

    /// An agent that also acts on its own: it starts `Proactive`, gets
    /// [`Agent::on_tick`] every tick it is awake, and returns to `Proactive`
    /// after every sleep.
    #[must_use]
    pub fn proactive(name: impl Into<String>, agent: impl Agent<M> + 'static) -> Self {
        Self::with_mode(name.into(), Box::new(agent), AgentMode::Proactive)
    }

    /// Builds an initializer that starts in `mode`.
    fn with_mode(name: String, agent: Box<dyn Agent<M>>, mode: AgentMode) -> Self {
        Self {
            agent,
            options: AgentOptions {
                initial_mode: mode,
                initial_queue: VecDeque::new(),
                name,
                message_drain: MessageDrain::One,
            },
        }
    }

    /// Configures how many queued messages this agent may handle per tick.
    #[must_use]
    pub const fn drain_messages(mut self, drain: MessageDrain) -> Self {
        self.options.message_drain = drain;
        self
    }

    /// Configures this agent to drain all currently queued messages each tick
    /// (unless it sleeps or marks a message [`MessageProcessingStatus::InProgress`]).
    #[must_use]
    pub const fn drain_all_messages(mut self) -> Self {
        self.options.message_drain = MessageDrain::All;
        self
    }

    /// Starts the agent asleep until tick `time`; it then wakes into the mode
    /// it was constructed with.
    #[must_use]
    pub const fn asleep_until(mut self, time: DiscreteTime) -> Self {
        self.options.initial_mode = self.options.initial_mode.asleep_until(time);
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

impl<M> dyn Agent<M> {
    /// Returns this behavior as an `A`, or `None` if it is another type.
    #[must_use]
    pub fn downcast_ref<A: Agent<M> + 'static>(&self) -> Option<&A> {
        sealed::AsAny::as_any(self).downcast_ref()
    }

    /// Returns this behavior mutably as an `A`, or `None` if it is another
    /// type. Use it through [`Simulation::find_by_name_mut`] to change an
    /// agent between runs, such as one branch's policy after a clone.
    ///
    /// [`Simulation::find_by_name_mut`]: crate::Simulation::find_by_name_mut
    #[must_use]
    pub fn downcast_mut<A: Agent<M> + 'static>(&mut self) -> Option<&mut A> {
        sealed::AsAny::as_any_mut(self).downcast_mut()
    }
}

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
        /// Returns the agent as mutable `Any`.
        fn as_any_mut(&mut self) -> &mut dyn Any;
    }

    impl<T: Any> AsAny for T {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
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
#[must_use]
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
#[must_use]
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

/// A `Proactive` agent that sends `M::default()` to `target` every `period`
/// ticks, starting on its first awake tick.
///
/// `period = 0` sends every tick, because a zero-tick sleep wakes on the next
/// tick. An unknown `target` is counted as produced without delivery.
#[must_use]
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

/// A `Reactive` agent that consumes one message and sleeps for `period` ticks
/// with no side effects. `period = 0` wakes on the next tick.
#[must_use]
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
