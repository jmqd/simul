//! `simul` is a discrete-event simulation library for running high-level
//! simulations of real-world problems and for running simulated experiments.
//!
//! `simul` is a *discrete-event simulator* using *incremental time
//! progression*, with a FIFO message queue per agent for interactions between
//! agents. It also supports deterministic typed experiments and seeded searches
//! over caller-defined candidate spaces.
//!
//! Use-cases:
//! - [Discrete-event simulation](https://en.wikipedia.org/wiki/Discrete-event_simulation)
//! - [Complex adaptive systems](https://authors.library.caltech.edu/60491/1/MGM%20113.pdf)
//! - [Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
//! - [Cross-entropy method](https://en.wikipedia.org/wiki/Cross-entropy_method)
//! - [Random search](https://en.wikipedia.org/wiki/Random_search)
//! - [Job-shop scheduling](https://en.wikipedia.org/wiki/Job-shop_scheduling)
//! - [Birth-death processes](https://en.wikipedia.org/wiki/Birth%E2%80%93death_process)
//! - [Computer experiments](https://en.wikipedia.org/wiki/Computer_experiment)
//! - Other: simulating logistics, operations research problems, running
//!   experiments to approximate a global optimum, simulating queueing systems,
//!   distributed systems, performance engineering/analysis, and so on.
//!
//! # Example
//!
//! A customer orders a drink every 3 ticks; a barista serves each order and
//! rests for as long as the drink takes. Every agent sends and receives the
//! simulation's message type, here `Drink`.
//!
//! ```
//! use simul::agent::{Agent, AgentContext, AgentInitializer};
//! use simul::message::Message;
//! use simul::{Halt, Simulation};
//!
//! #[derive(Clone, Copy, Debug)]
//! enum Drink {
//!     Coffee,
//!     Tea,
//! }
//!
//! #[derive(Clone, Debug)]
//! struct Customer;
//!
//! impl Agent<Drink> for Customer {
//!     fn on_tick(&mut self, ctx: &mut AgentContext<'_, Drink>) {
//!         // Draws come from the simulation seed, so equal seeds give equal runs.
//!         let drink = if ctx.rng().uniform_f64() < 0.5 {
//!             Drink::Coffee
//!         } else {
//!             Drink::Tea
//!         };
//!         ctx.send("barista", drink);
//!         ctx.sleep_for(3);
//!     }
//! }
//!
//! #[derive(Clone, Debug, Default)]
//! struct Barista {
//!     coffees: u32,
//!     teas: u32,
//! }
//!
//! impl Agent<Drink> for Barista {
//!     fn on_message(&mut self, ctx: &mut AgentContext<'_, Drink>, order: &Message<Drink>) {
//!         match order.payload {
//!             Drink::Coffee => {
//!                 self.coffees += 1;
//!                 ctx.sleep_for(2);
//!             }
//!             Drink::Tea => {
//!                 self.teas += 1;
//!                 ctx.sleep_for(1);
//!             }
//!         }
//!     }
//! }
//!
//! let mut simulation = Simulation::builder(Halt::AtTime(30))
//!     .agents([
//!         AgentInitializer::proactive("customer", Customer),
//!         AgentInitializer::reactive("barista", Barista::default()),
//!     ])
//!     .build()?;
//! simulation.run();
//!
//! // Results stay in the agents; read the barista back as its own type.
//! let barista = simulation
//!     .find_by_name("barista")
//!     .and_then(|agent| agent.downcast_ref::<Barista>())
//!     .expect("the barista is a Barista");
//! assert_eq!(10, barista.coffees + barista.teas);
//! # Ok::<(), simul::SimulationError>(())
//! ```
#![warn(missing_docs)]

/// Agents, the per-tick context they act through, and built-in producers and
/// consumers.
pub mod agent;
/// Deterministic engine-independent experiment, reduction, and search APIs.
pub mod experiment;
/// Messages that agents send each other.
pub mod message;
/// Structured simulation tracing and JSONL / Perfetto Chrome Trace export.
pub mod trace;

// Runs the README's Rust examples as doctests so they keep compiling against
// the current API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

use agent::{
    AgentContext, AgentId, AgentInitializer, AgentMetadata, AgentMode, AgentState,
    AgentStreamCache, CommandBuffer, MessageDrain, MessageHistory, MessageProcessingStatus, Peers,
    PendingSend, QueueDepthChange, SimulationAgent, SpecialCommand, TimerId, WakeMode,
};
use experiment::{
    CandidateId, FixedReplications, MasterSeed, RunError, TrialContext, TrialRecord, TrialStream,
};
use message::Message;
use trace::{SimulationTrace, TraceEventKind, TraceRecorder, debug_format_payload};

use rustc_hash::FxHashMap;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, hash_map::Entry};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Source of agent layout tokens; each constructed simulation takes the next.
static NEXT_LAYOUT: AtomicU64 = AtomicU64::new(0);

/// Per-agent statistics keyed by agent name; public methods return std maps.
#[allow(
    clippy::disallowed_types,
    reason = "public statistics keep returning std HashMap"
)]
type NameStatistics<V> = std::collections::HashMap<String, V>;

/// A simulation's unitless discrete tick count.
///
/// One tick is one pass over runnable agents in [`Simulation::run`]. Advancing
/// past `u64::MAX` panics.
pub type DiscreteTime = u64;

/// Largest simulation that resolves agent names by linear scan instead of hashing.
const LINEAR_NAME_SCAN_LIMIT: usize = 8;

/// Most agents one simulation holds, so every [`AgentId`] fits in `u32`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "u32::MAX fits in usize on every supported target"
)]
const MAX_AGENTS: usize = u32::MAX as usize;

/// Messages of room reserved in each queue and recorded history of a small
/// simulation.
const SMALL_SIMULATION_HISTORY: usize = 16;

/// Simulations with more agents than this may skip inactive agents each tick.
const SPARSE_SCHEDULE_MIN_AGENTS: usize = 64;

/// The current mode of a Simulation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum SimulationMode {
    /// The Simulation has only been constructed.
    Constructed,
    /// The Simulation is actively simulating.
    Running,
    /// [`Simulation::run_until`] reached its time before the run ended; `run`
    /// or `run_until` continues it.
    Paused,
    /// The simulation stopped because its [`Halt`] condition held or an agent
    /// called [`AgentContext::send_halt_interrupt`].
    Completed,
    /// A callback or the halt check panicked mid-tick. The tick was partly
    /// applied, so the simulation refuses to run again.
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
pub struct Simulation<M = ()> {
    /// The agents within the simulation, e.g. adaptive agents.
    agents: Vec<SimulationAgent<M>>,

    /// The current discrete time of the Simulation.
    time: DiscreteTime,

    /// When the run stops.
    halt: Halt<M>,

    /// Whether to record metrics on queue depths. Takes space.
    enable_queue_depth_metric: bool,

    /// Records a metric on the number of cycles an agent was asleep for.
    enable_agent_asleep_cycles_metric: bool,

    /// The mode of the Simulation.
    mode: SimulationMode,

    /// Maps agent names to identities in simulations too large for linear
    /// name scans; empty otherwise.
    agent_name_handle_map: FxHashMap<String, AgentId>,

    /// Identifies this agent layout for cached target identities. Clones share
    /// the layout and its token; every constructed simulation gets a new one.
    layout: u64,

    /// Each agent's stream root, derived on first `rng()` call; names never
    /// change, so neither do these.
    agent_streams: AgentStreamCache,

    /// The reason given by the interrupt that ended the latest run, if any.
    halt_reason: Option<String>,

    /// The agent and tag from the latest [`AgentContext::yield_control`], if paused at a yield.
    yield_point: Option<(AgentId, String)>,

    /// Messages sent with a delay, keyed by arrival tick and then send order.
    in_flight: BTreeMap<(DiscreteTime, u64), Message<M>>,

    /// Earliest arrival tick in `in_flight`, or `DiscreteTime::MAX` when empty.
    next_arrival: DiscreteTime,

    /// Send-order number of the next delayed message.
    next_in_flight: u64,

    /// How messages with unknown or out-of-range destinations are handled.
    unknown_destination_policy: UnknownDestinationPolicy,

    /// Number of messages addressed to an unknown agent name or out-of-range
    /// [`AgentId`].
    unknown_destination_count: u64,

    /// Optional structured lifecycle trace recorder.
    trace: Option<TraceRecorder<M>>,
}

/// How a simulation handles a message addressed to an unknown agent name or
/// out-of-range [`AgentId`].
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum UnknownDestinationPolicy {
    /// Count the message as produced on the sender and silently drop it
    /// (the default).
    #[default]
    Drop,
    /// Count the message as produced and stop the simulation after the current
    /// tick with a halt reason naming the sender, ending in
    /// [`SimulationMode::Completed`].
    Halt,
    /// Panic immediately while processing commands, leaving the simulation in
    /// [`SimulationMode::Failed`].
    Panic,
}

/// When a simulation stops running.
///
/// The condition is checked before every tick that runs, against the state
/// after the previous tick's messages were delivered. An agent's
/// [`AgentContext::send_halt_interrupt`] also stops the run after its tick.
///
/// With `AtTime` or `Never`, and the asleep-cycle metric off, the engine skips
/// ticks in which no agent could act and jumps to the next wake-up; results
/// match running every tick. A `When` predicate is checked on every tick, so
/// it disables skipping.
pub enum Halt<M = ()> {
    /// Stop once the simulation time reaches this tick, without running it.
    AtTime(DiscreteTime),
    /// Run until an agent sends a halt interrupt.
    Never,
    /// Stop when the predicate returns `true`.
    When(Arc<HaltPredicate<M>>),
}

/// A halt check over the whole simulation state.
type HaltPredicate<M> = dyn Fn(&Simulation<M>) -> bool + Send + Sync;

impl<M> Halt<M> {
    /// Stops when `predicate` returns `true`.
    #[must_use]
    pub fn when(predicate: impl Fn(&Simulation<M>) -> bool + Send + Sync + 'static) -> Self {
        Self::When(Arc::new(predicate))
    }
}

// A derive would demand `M: Clone`, though only the predicate is shared.
impl<M> Clone for Halt<M> {
    fn clone(&self) -> Self {
        match self {
            Self::AtTime(time) => Self::AtTime(*time),
            Self::Never => Self::Never,
            Self::When(predicate) => Self::When(Arc::clone(predicate)),
        }
    }
}

impl<M> std::fmt::Debug for Halt<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AtTime(time) => f.debug_tuple("AtTime").field(time).finish(),
            Self::Never => f.write_str("Never"),
            Self::When(_) => f.write_str("When(..)"),
        }
    }
}

/// Configures and validates a [`Simulation`]; create one with
/// [`Simulation::builder`].
#[derive(Clone, Debug)]
#[must_use]
pub struct SimulationBuilder<M = ()> {
    /// Agents in construction order, which fixes their [`AgentId`]s.
    agent_initializers: Vec<AgentInitializer<M>>,
    /// When the run stops.
    halt: Halt<M>,
    /// The tick the simulation starts at.
    starting_time: DiscreteTime,
    /// Whether queue depth changes are recorded.
    queue_depth_metrics: bool,
    /// Whether visits to sleeping agents count asleep ticks.
    asleep_cycles_metric: bool,
    /// How to copy a sent message into its sender's history, if history is on.
    message_history: Option<CloneMessage<M>>,
    /// Root of every agent's [`AgentContext::rng`] stream.
    seed: MasterSeed,
    /// How sends to unknown or out-of-range destinations are handled.
    unknown_destination_policy: UnknownDestinationPolicy,
    /// Whether structured tracing starts enabled.
    trace_enabled: bool,
    /// Optional debug formatter for sent payloads in traces.
    trace_payload_formatter: Option<fn(&M) -> String>,
}

/// Copies a message; recording a history is the only reason the engine
/// needs `M: Clone`.
type CloneMessage<M> = fn(&Message<M>) -> Message<M>;

impl<M> SimulationBuilder<M> {
    /// Appends one agent.
    pub fn agent(mut self, agent: AgentInitializer<M>) -> Self {
        self.agent_initializers.push(agent);
        self
    }

    /// Appends agents in order.
    pub fn agents(mut self, agents: impl IntoIterator<Item = AgentInitializer<M>>) -> Self {
        self.agent_initializers.extend(agents);
        self
    }

    /// Sets the first tick to run. Defaults to 0.
    pub const fn starting_time(mut self, time: DiscreteTime) -> Self {
        self.starting_time = time;
        self
    }

    /// Sets the root of every agent's [`AgentContext::rng`] stream. Equal
    /// seeds give equal runs. Defaults to `MasterSeed::from_u64(0)`.
    pub const fn seed(mut self, seed: MasterSeed) -> Self {
        self.seed = seed;
        self
    }

    /// Seeds the simulation from an experiment trial, for evaluators run by
    /// [`run_serial`](experiment::run_serial) or `run_parallel`.
    ///
    /// The seed is the trial's common-random [`TrialStream::TrialNoise`]
    /// stream tagged `simul::Simulation`. Candidates evaluated in one
    /// common-random group therefore share their agents' draws replication
    /// by replication, so paired comparisons see the same noise, while each
    /// replication draws afresh. For independent draws per candidate, give
    /// [`Self::seed`] a seed from [`TrialContext::independent_seed`].
    pub fn trial(self, context: &TrialContext) -> Self {
        let seed = context.common_seed(TrialStream::TrialNoise, "simul::Simulation");
        self.seed(MasterSeed::new(seed.bytes()))
    }

    /// Records each agent's queue depth whenever it changes. Off by default.
    pub const fn queue_depth_metrics(mut self, enabled: bool) -> Self {
        self.queue_depth_metrics = enabled;
        self
    }

    /// Counts the ticks each agent spends asleep. Off by default; it forces
    /// dense scheduling.
    pub const fn asleep_cycles_metric(mut self, enabled: bool) -> Self {
        self.asleep_cycles_metric = enabled;
        self
    }

    /// Configures how messages sent to an unknown agent name or out-of-range
    /// [`AgentId`] are handled. Defaults to [`UnknownDestinationPolicy::Drop`].
    pub const fn unknown_destination_policy(mut self, policy: UnknownDestinationPolicy) -> Self {
        self.unknown_destination_policy = policy;
        self
    }

    /// Records structured lifecycle events into a [`SimulationTrace`],
    /// exportable to JSONL or Perfetto Chrome Trace JSON. Off by default.
    ///
    /// Tracing may also be toggled dynamically between runs with
    /// [`Simulation::enable_trace`] and [`Simulation::disable_trace`].
    pub const fn trace(mut self, enabled: bool) -> Self {
        self.trace_enabled = enabled;
        self
    }

    /// Validates the configuration and builds the simulation.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::DuplicateAgentName`] when two agents share a
    /// name, or [`SimulationError::TooManyAgents`] beyond `u32::MAX` agents.
    pub fn build(self) -> Result<Simulation<M>, SimulationError> {
        Simulation::try_new(self)
    }
}

impl<M: Clone> SimulationBuilder<M> {
    /// Keeps every message each agent consumes and produces, readable through
    /// [`SimulationAgent::consumed`] and [`SimulationAgent::produced`]. Off
    /// by default: a history holds every message until the simulation is
    /// dropped, and copies each sent one. Counts and mean waits are kept
    /// either way.
    pub fn message_history(mut self, enabled: bool) -> Self {
        self.message_history = enabled.then_some(Message::clone as CloneMessage<M>);
        self
    }
}

impl<M: std::fmt::Debug> SimulationBuilder<M> {
    /// Like [`Self::trace`], and also records a `Debug` summary of each sent
    /// message payload.
    pub fn trace_with_payloads(mut self, enabled: bool) -> Self {
        self.trace_enabled = enabled;
        self.trace_payload_formatter = enabled.then_some(debug_format_payload::<M>);
        self
    }
}

/// Errors returned when simulation parameters are invalid.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimulationError {
    /// More than one agent was configured with the same name.
    DuplicateAgentName(String),
    /// More agents were configured than [`AgentId`] can identify.
    TooManyAgents {
        /// Number of configured agents.
        count: usize,
    },
}

impl std::fmt::Display for SimulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateAgentName(name) => {
                write!(f, "duplicate agent name: {name}")
            }
            Self::TooManyAgents { count } => {
                write!(f, "{count} agents exceed the limit of {MAX_AGENTS}")
            }
        }
    }
}

impl std::error::Error for SimulationError {}

/// Per-tick inputs shared by every agent visit.
struct Tick<'a, M> {
    /// The current simulation time.
    time: DiscreteTime,
    /// The simulation's agent layout token.
    layout: u64,
    /// Number of agents, which bounds valid identities.
    agent_count: usize,
    /// Whether callbacks route by scanning names instead of the name index.
    scan_names: bool,
    /// Whether every visit records the queue depth.
    queue_depth_metric: bool,
    /// Whether visits to sleeping agents count asleep cycles.
    asleep_cycles_metric: bool,
    /// Name index for simulations too large to scan names.
    agent_name_handle_map: &'a FxHashMap<String, AgentId>,
    /// Commands issued during the tick, applied at its end.
    commands: &'a mut CommandBuffer<M>,
    /// Lazily derived stream roots by handle.
    agent_streams: &'a mut AgentStreamCache,
    /// Monotonic send-order sequence for delayed messages.
    next_in_flight: &'a mut u64,
    /// Active structured trace sink, if tracing is enabled.
    trace: Option<&'a mut SimulationTrace>,
}

/// Tracks runnable handles in callback order and sleeping agents by wake time.
struct SparseSchedule {
    /// One bit per agent; scanning words skips inactive agents without touching them.
    ready: Vec<u64>,
    /// Number of set bits, used to return to dense scanning when activity rises.
    ready_count: usize,
    /// Wake deadlines ordered by time, then stable agent handle.
    sleeping: BinaryHeap<Reverse<(DiscreteTime, usize)>>,
}

impl SparseSchedule {
    /// Rebuilds transient scheduling state, including changes made between runs.
    fn new<M>(agents: &[SimulationAgent<M>]) -> Self {
        let mut schedule = Self {
            ready: vec![0; agents.len().div_ceil(64)],
            ready_count: 0,
            sleeping: BinaryHeap::new(),
        };
        let mut sleeping = Vec::new();
        for (handle, agent) in agents.iter().enumerate() {
            if let AgentMode::AsleepUntil { until, .. } = agent.state.mode {
                sleeping.push(Reverse((until, handle)));
            } else {
                schedule.set_ready(handle, Self::runnable(&agent.state));
            }
        }
        schedule.sleeping = BinaryHeap::from(sleeping);
        schedule
    }

    /// Reports whether a nonsleeping agent has a callback to execute.
    fn runnable<M>(state: &AgentState<M>) -> bool {
        state.mode == AgentMode::Proactive
            || (state.mode == AgentMode::Reactive && !state.queue.is_empty())
    }

    /// Changes membership without introducing duplicate visits.
    fn set_ready(&mut self, handle: usize, ready: bool) {
        let word = &mut self.ready[handle.div_euclid(64)];
        let bit = 1 << handle.rem_euclid(64);
        if (*word & bit != 0) != ready {
            *word ^= bit;
            if ready {
                self.ready_count += 1;
            } else {
                self.ready_count -= 1;
            }
        }
    }

    /// Finds the next runnable handle at or after the cursor.
    fn next(&self, cursor: usize) -> Option<usize> {
        let mut index = cursor.div_euclid(64);
        let mut bits = *self.ready.get(index)? & (u64::MAX << cursor.rem_euclid(64));
        loop {
            if bits != 0 {
                return Some(index * 64 + bits.trailing_zeros() as usize);
            }
            index += 1;
            bits = *self.ready.get(index)?;
        }
    }

    /// Defers wake processing until a subsequent tick, even for zero-length sleeps.
    fn sleep(&mut self, handle: usize, deadline: DiscreteTime) {
        self.set_ready(handle, false);
        self.sleeping.push(Reverse((deadline, handle)));
    }

    /// Returns the earliest tick at or after `now` at which a scheduled agent
    /// can act. Stale heap entries make this a lower bound, which is safe.
    fn next_event(&self, now: DiscreteTime) -> Option<DiscreteTime> {
        if self.ready_count > 0 {
            return Some(now);
        }
        self.sleeping
            .peek()
            .map(|&Reverse((deadline, _))| deadline.max(now))
    }

    /// Marks due agents ready; their state changes only at their ordered visit.
    fn wake<M>(&mut self, time: DiscreteTime, agents: &[SimulationAgent<M>]) {
        while let Some(&Reverse((deadline, handle))) = self.sleeping.peek() {
            if deadline > time {
                break;
            }
            self.sleeping.pop();
            if matches!(
                agents[handle].state.mode,
                AgentMode::AsleepUntil { until, .. } if until == deadline
            ) {
                self.set_ready(handle, true);
            }
        }
    }
}

/// How a run chooses which agents to visit each tick.
enum Scheduler {
    /// Visits every agent every tick: the simulation is small, or the asleep
    /// metric must count every sleeping agent every tick.
    DenseOnly,
    /// Visits every agent until activity falls below the low watermark.
    Dense,
    /// Visits only runnable agents until activity rises above the high
    /// watermark.
    Sparse(SparseSchedule),
}

impl Scheduler {
    /// Starts sparse when the simulation may use a schedule at all.
    fn new<M>(agents: &[SimulationAgent<M>], asleep_cycles_metric: bool) -> Self {
        // Small scans are cheaper than maintaining the schedule, and the asleep
        // metric needs every agent visited every tick, so either keeps dense scans.
        if agents.len() > SPARSE_SCHEDULE_MIN_AGENTS && !asleep_cycles_metric {
            Self::Sparse(SparseSchedule::new(agents))
        } else {
            Self::DenseOnly
        }
    }

    /// Returns the schedule while visits are sparse.
    const fn sparse_mut(&mut self) -> Option<&mut SparseSchedule> {
        match self {
            Self::Sparse(active) => Some(active),
            Self::Dense | Self::DenseOnly => None,
        }
    }

    /// Marks agents due at `time` ready, and switches to dense visits once a
    /// quarter of the agents are ready.
    fn before_tick<M>(&mut self, time: DiscreteTime, agents: &[SimulationAgent<M>]) {
        if let Self::Sparse(active) = self {
            active.wake(time, agents);
            if active.ready_count >= agents.len().div_ceil(4) {
                *self = Self::Dense;
            }
        }
    }

    /// Switches back to sparse visits once fewer than an eighth of the agents
    /// ran a callback. Separate low and high watermarks amortize rebuilding
    /// across busy and quiet phases instead of switching around one threshold.
    fn after_tick<M>(&mut self, callbacks: usize, agents: &[SimulationAgent<M>]) {
        if matches!(self, Self::Dense) && callbacks < agents.len().div_ceil(8) {
            *self = Self::Sparse(SparseSchedule::new(agents));
        }
    }
}

/// Whether a run may jump over ticks in which no agent can act.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QuietTicks {
    /// Every tick is observed, by a `Halt::When` predicate or the asleep
    /// metric, so none may be skipped.
    Observed,
    /// Quiet ticks may be skipped, but not past the halt or pause time.
    SkipUpTo(DiscreteTime),
    /// Quiet ticks may be skipped with no halt or pause time to stop at.
    SkipUnbounded,
}

impl QuietTicks {
    /// Decides for a run that halts by `halt` and pauses at `until`.
    fn new<M>(halt: &Halt<M>, asleep_cycles_metric: bool, until: Option<DiscreteTime>) -> Self {
        match (halt, asleep_cycles_metric) {
            (Halt::When(_), _) | (_, true) => Self::Observed,
            (Halt::AtTime(time), false) => {
                Self::SkipUpTo(until.map_or(*time, |until| until.min(*time)))
            }
            (Halt::Never, false) => until.map_or(Self::SkipUnbounded, Self::SkipUpTo),
        }
    }

    /// Returns the tick to jump to after a quiet tick, given the next tick at
    /// which anything happens, or `None` to step one tick as usual.
    fn skip_to(self, next_event: Option<DiscreteTime>) -> Option<DiscreteTime> {
        match self {
            Self::Observed => None,
            Self::SkipUpTo(bound) => Some(next_event.map_or(bound, |event| event.min(bound))),
            Self::SkipUnbounded => next_event,
        }
    }
}

/// Why a simulation run stopped.
///
/// Returned by [`Simulation::run_until_yield`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RunOutcome {
    /// The simulation's [`Halt`] condition held.
    Halted,
    /// An agent sent [`AgentContext::send_halt_interrupt`] or triggered
    /// [`UnknownDestinationPolicy::Halt`].
    Interrupted(String),
    /// Reached the pause time passed to [`Simulation::run_until`] or
    /// [`Simulation::run_until_yield`] before the run ended.
    PausedAtTime,
    /// An agent called [`AgentContext::yield_control`], pausing between ticks
    /// so an external controller can inspect state, fork counterfactual
    /// rollouts, or inject an action before resuming.
    Yielded {
        /// Identity of the first agent that requested a yield in issue order.
        agent: AgentId,
        /// Tag supplied to [`AgentContext::yield_control`].
        tag: String,
    },
}

/// Early stop requested by buffered commands during a tick.
enum TickStop {
    /// First halt interrupt (or unknown-destination halt) in issue order.
    Interrupted(String),
    /// First yield request in issue order when no interrupt occurred.
    Yielded(AgentId, String),
}

/// Records a consumed message to an active trace out of line.
#[cold]
#[inline(never)]
fn push_trace_consumed<M>(
    trace: &mut SimulationTrace,
    time: DiscreteTime,
    handle: usize,
    msg: &Message<M>,
    queue_depth: usize,
) {
    trace.push(
        time,
        AgentId::from_handle(handle),
        TraceEventKind::MessageConsumed {
            source: msg.source,
            queued_time: msg.queued_time,
            queue_depth,
        },
    );
}

/// Appends an event to an active trace out of line so untraced ticks stay
/// compact.
#[cold]
#[inline(never)]
fn push_trace_event(
    trace: &mut SimulationTrace,
    time: DiscreteTime,
    handle: usize,
    kind: TraceEventKind,
) {
    trace.push(time, AgentId::from_handle(handle), kind);
}

/// Visits one agent in handle order: wakes it when due, runs its callback, and
/// applies the sleep it requested. Returns whether a callback ran and the
/// requested wake time.
///
/// Dense and sparse ticks share this body, so it must inline into both loops.
#[allow(
    clippy::inline_always,
    reason = "per-agent hot path shared by the dense and sparse tick loops"
)]
#[inline(always)]
fn visit_agent<M>(
    agents: &mut [SimulationAgent<M>],
    handle: usize,
    tick: &mut Tick<'_, M>,
) -> (bool, Option<DiscreteTime>) {
    // Only small simulations split the slice, since every visit pays for it,
    // including visits to sleeping agents.
    let (agent, peers) = if tick.scan_names {
        let (before, rest) = agents.split_at_mut(handle);
        let Some((agent, after)) = rest.split_first_mut() else {
            return (false, None);
        };
        (agent, Some((&*before, &*after)))
    } else {
        (&mut agents[handle], None)
    };
    let wake_mode = match agent.state.mode {
        AgentMode::Proactive => WakeMode::Proactive,
        AgentMode::Reactive => WakeMode::Reactive,
        AgentMode::AsleepUntil { until, then } => {
            if tick.time < until {
                // A sleeping agent pops no message, runs no callback, and
                // requests no sleep; only the optional asleep count changes.
                if tick.asleep_cycles_metric {
                    agent.metadata.asleep_cycle_count += 1;
                }
                return (false, None);
            }
            agent.state.mode = then.into();
            if let Some(trace) = tick.trace.as_deref_mut() {
                push_trace_event(trace, tick.time, handle, TraceEventKind::AgentWoke);
            }
            then
        }
        AgentMode::Dead => return (false, None),
    };
    if agent.message_drain != MessageDrain::One {
        return visit_agent_multi_drain(agent, peers, handle, wake_mode, tick);
    }
    let proactive = wake_mode == WakeMode::Proactive;
    let message = agent.state.queue.pop_front();
    if tick.queue_depth_metric && message.is_some() {
        agent.record_queue_depth(tick.time);
    }
    if !proactive && message.is_none() {
        return (false, None);
    }

    let mut requested_sleep_until = None;
    let mut ctx = AgentContext {
        id: AgentId::from_handle(handle),
        name: &agent.name,
        time: tick.time,
        commands: &mut *tick.commands,
        requested_sleep_until: &mut requested_sleep_until,
        peers,
        agent_name_handle_map: tick.agent_name_handle_map,
        agent_count: tick.agent_count,
        layout: tick.layout,
        state: &agent.state,
        message_processing_status: MessageProcessingStatus::Done,
        agent_streams: &mut *tick.agent_streams,
        calls: 0,
        next_in_flight: &mut *tick.next_in_flight,
    };
    // Every awake agent handles at most one queued message per tick; proactive
    // agents then also tick.
    if let Some(msg) = &message {
        agent.agent.on_message(&mut ctx, msg);
    }
    let status = std::mem::replace(
        &mut ctx.message_processing_status,
        MessageProcessingStatus::Done,
    );
    if proactive {
        agent.agent.on_tick(&mut ctx);
    }
    if let Some(msg) = message {
        match status {
            MessageProcessingStatus::InProgress => {
                agent.state.queue.push_front(msg);
                if tick.queue_depth_metric {
                    // The message is back in the queue from the next tick on.
                    agent.record_queue_depth(tick.time.saturating_add(1));
                }
            }
            MessageProcessingStatus::Done => {
                if let Some(trace) = tick.trace.as_deref_mut() {
                    push_trace_consumed(trace, tick.time, handle, &msg, agent.state.queue.len());
                }
                agent.record_consumed(msg, tick.time);
            }
        }
    }
    if let Some(sleep_until) = requested_sleep_until {
        if let Some(trace) = tick.trace.as_deref_mut() {
            let kind = TraceEventKind::AgentSlept { until: sleep_until };
            push_trace_event(trace, tick.time, handle, kind);
        }
        agent.state.mode = AgentMode::AsleepUntil {
            until: sleep_until,
            then: wake_mode,
        };
    }
    (true, requested_sleep_until)
}

/// Outlined visit path for agents configured with [`MessageDrain::UpTo`] or
/// [`MessageDrain::All`], keeping the default single-message loop compact.
#[inline(never)]
fn visit_agent_multi_drain<M>(
    agent: &mut SimulationAgent<M>,
    peers: Option<Peers<'_, M>>,
    handle: usize,
    wake_mode: WakeMode,
    tick: &mut Tick<'_, M>,
) -> (bool, Option<DiscreteTime>) {
    let proactive = wake_mode == WakeMode::Proactive;
    if !proactive && agent.state.queue.is_empty() {
        return (false, None);
    }
    let limit = match agent.message_drain {
        MessageDrain::One => 1,
        MessageDrain::UpTo(max) => max.get(),
        MessageDrain::All => usize::MAX,
    };
    let mut requested_sleep_until = None;
    let mut calls = 0_u64;
    let mut handled = 0_usize;
    while handled < limit && requested_sleep_until.is_none() {
        let Some(msg) = agent.state.queue.pop_front() else {
            break;
        };
        if tick.queue_depth_metric {
            agent.record_queue_depth(tick.time);
        }
        let mut ctx = AgentContext {
            id: AgentId::from_handle(handle),
            name: &agent.name,
            time: tick.time,
            commands: &mut *tick.commands,
            requested_sleep_until: &mut requested_sleep_until,
            peers,
            agent_name_handle_map: tick.agent_name_handle_map,
            agent_count: tick.agent_count,
            layout: tick.layout,
            state: &agent.state,
            message_processing_status: MessageProcessingStatus::Done,
            agent_streams: &mut *tick.agent_streams,
            calls,
            next_in_flight: &mut *tick.next_in_flight,
        };
        agent.agent.on_message(&mut ctx, &msg);
        calls = ctx.calls;
        let status = ctx.message_processing_status;
        match status {
            MessageProcessingStatus::InProgress => {
                agent.state.queue.push_front(msg);
                if tick.queue_depth_metric {
                    agent.record_queue_depth(tick.time.saturating_add(1));
                }
                break;
            }
            MessageProcessingStatus::Done => {
                if let Some(trace) = tick.trace.as_deref_mut() {
                    trace.push(
                        tick.time,
                        AgentId::from_handle(handle),
                        TraceEventKind::MessageConsumed {
                            source: msg.source,
                            queued_time: msg.queued_time,
                            queue_depth: agent.state.queue.len(),
                        },
                    );
                }
                agent.record_consumed(msg, tick.time);
                handled += 1;
            }
        }
    }
    if proactive {
        let mut ctx = AgentContext {
            id: AgentId::from_handle(handle),
            name: &agent.name,
            time: tick.time,
            commands: &mut *tick.commands,
            requested_sleep_until: &mut requested_sleep_until,
            peers,
            agent_name_handle_map: tick.agent_name_handle_map,
            agent_count: tick.agent_count,
            layout: tick.layout,
            state: &agent.state,
            message_processing_status: MessageProcessingStatus::Done,
            agent_streams: &mut *tick.agent_streams,
            calls,
            next_in_flight: &mut *tick.next_in_flight,
        };
        agent.agent.on_tick(&mut ctx);
    }
    if let Some(sleep_until) = requested_sleep_until {
        if let Some(trace) = tick.trace.as_deref_mut() {
            trace.push(
                tick.time,
                AgentId::from_handle(handle),
                TraceEventKind::AgentSlept { until: sleep_until },
            );
        }
        agent.state.mode = AgentMode::AsleepUntil {
            until: sleep_until,
            then: wake_mode,
        };
    }
    (true, requested_sleep_until)
}

impl<M> Simulation<M> {
    /// Starts configuring a simulation that stops according to `halt`.
    ///
    /// `M` is the type every message carries. When nothing else fixes it, as
    /// with only built-in agents, name it: `Simulation::<()>::builder(halt)`.
    pub const fn builder(halt: Halt<M>) -> SimulationBuilder<M> {
        SimulationBuilder {
            agent_initializers: Vec::new(),
            halt,
            starting_time: 0,
            queue_depth_metrics: false,
            asleep_cycles_metric: false,
            message_history: None,
            seed: MasterSeed::from_u64(0),
            unknown_destination_policy: UnknownDestinationPolicy::Drop,
            trace_enabled: false,
            trace_payload_formatter: None,
        }
    }

    /// Builds a validated simulation from a builder.
    fn try_new(parameters: SimulationBuilder<M>) -> Result<Self, SimulationError> {
        let initializers = &parameters.agent_initializers;
        if initializers.len() > MAX_AGENTS {
            return Err(SimulationError::TooManyAgents {
                count: initializers.len(),
            });
        }
        let mut agent_name_handle_map = FxHashMap::default();
        if initializers.len() <= LINEAR_NAME_SCAN_LIMIT {
            for (i, agent_initializer) in initializers.iter().enumerate() {
                let name = &agent_initializer.options.name;
                if initializers[..i]
                    .iter()
                    .any(|earlier| earlier.options.name == *name)
                {
                    return Err(SimulationError::DuplicateAgentName(name.clone()));
                }
            }
        } else {
            agent_name_handle_map.reserve(initializers.len());
            for (i, agent_initializer) in initializers.iter().enumerate() {
                match agent_name_handle_map.entry(agent_initializer.options.name.clone()) {
                    Entry::Occupied(_) => {
                        return Err(SimulationError::DuplicateAgentName(
                            agent_initializer.options.name.clone(),
                        ));
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(AgentId::from_handle(i));
                    }
                }
            }
        }

        // Small simulations are often run by the thousand in experiments;
        // reserving queues and histories up front avoids repeated growth.
        let history_capacity = if initializers.len() <= LINEAR_NAME_SCAN_LIMIT {
            SMALL_SIMULATION_HISTORY
        } else {
            0
        };
        let message_history = parameters.message_history;
        let agents: Vec<SimulationAgent<M>> = parameters
            .agent_initializers
            .into_iter()
            .enumerate()
            .map(|(handle, agent_initializer)| {
                let mut queue = agent_initializer.options.initial_queue;
                for message in &mut queue {
                    message.destination = Some(AgentId::from_handle(handle));
                }
                queue.reserve(history_capacity);
                SimulationAgent {
                    agent: agent_initializer.agent,
                    name: agent_initializer.options.name,
                    metadata: AgentMetadata::default(),
                    state: AgentState {
                        mode: agent_initializer.options.initial_mode,
                        queue,
                    },
                    history: message_history.map(|copy| MessageHistory {
                        consumed: Vec::with_capacity(history_capacity),
                        produced: Vec::with_capacity(history_capacity),
                        copy,
                    }),
                    message_drain: agent_initializer.options.message_drain,
                }
            })
            .collect();

        let trace = parameters.trace_enabled.then(|| {
            let names = agents.iter().map(|a| a.name.clone()).collect();
            TraceRecorder::new(names, parameters.trace_payload_formatter)
        });

        Ok(Self {
            mode: SimulationMode::Constructed,
            agents,
            halt: parameters.halt,
            time: parameters.starting_time,
            enable_queue_depth_metric: parameters.queue_depth_metrics,
            enable_agent_asleep_cycles_metric: parameters.asleep_cycles_metric,
            agent_name_handle_map,
            layout: NEXT_LAYOUT.fetch_add(1, Ordering::Relaxed),
            agent_streams: AgentStreamCache::new(parameters.seed),
            halt_reason: None,
            yield_point: None,
            in_flight: BTreeMap::new(),
            next_arrival: DiscreteTime::MAX,
            next_in_flight: 0,
            unknown_destination_policy: parameters.unknown_destination_policy,
            unknown_destination_count: 0,
            trace,
        })
    }

    /// Returns the messages the agent named `name` consumed, or `None` if no
    /// agent has this name.
    ///
    /// # Panics
    ///
    /// Panics unless message history is on; see
    /// [`SimulationBuilder::message_history`].
    #[must_use]
    pub fn consumed_for_agent(&self, name: &str) -> Option<&[Message<M>]> {
        Some(self.find_by_name(name)?.consumed())
    }

    /// Returns the agent named `name`, or `None` if no agent has this name.
    ///
    /// Simulations of 8 agents or fewer scan linearly; larger ones use a hash
    /// lookup.
    #[must_use]
    #[inline]
    pub fn find_by_name(&self, name: &str) -> Option<&SimulationAgent<M>> {
        self.find_handle_by_name(name)
            .and_then(|handle| self.agents.get(handle))
    }

    /// Returns the identity of the agent named `name`, or `None` if no agent
    /// has this name.
    #[must_use]
    #[inline]
    pub fn agent_id(&self, name: &str) -> Option<AgentId> {
        self.find_handle_by_name(name).map(AgentId::from_handle)
    }

    /// Returns the agent with identity `id`, or `None` when `id` is out of
    /// range for this simulation.
    ///
    /// Identities are construction indices, so an `AgentId` from another
    /// simulation can refer to a different agent here.
    #[must_use]
    #[inline]
    pub fn agent(&self, id: AgentId) -> Option<&SimulationAgent<M>> {
        self.agents.get(id.index())
    }

    /// Returns a mutable reference to the agent with identity `id`, or `None`
    /// when `id` is out of range for this simulation.
    #[must_use]
    #[inline]
    pub fn agent_mut(&mut self, id: AgentId) -> Option<&mut SimulationAgent<M>> {
        self.agents.get_mut(id.index())
    }

    /// Returns a mutable reference to the agent named `name`, or `None` if no
    /// agent has this name.
    ///
    /// Callers may mutate behavior and [`SimulationAgent::state`] between runs;
    /// the routing name stays read-only.
    ///
    /// ```compile_fail
    /// use simul::Simulation;
    /// fn rename(simulation: &mut Simulation) {
    ///     simulation.find_by_name_mut("worker").unwrap().name = "renamed".into();
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub fn find_by_name_mut(&mut self, name: &str) -> Option<&mut SimulationAgent<M>> {
        let handle = self.find_handle_by_name(name)?;
        self.agents.get_mut(handle)
    }

    /// Returns the messages the agent named `name` sent, or `None` if no agent
    /// has this name.
    ///
    /// # Panics
    ///
    /// Panics unless message history is on; see
    /// [`SimulationBuilder::message_history`].
    #[must_use]
    pub fn produced_for_agent(&self, name: &str) -> Option<&[Message<M>]> {
        Some(self.find_by_name(name)?.produced())
    }

    /// Returns the agent's queue depth changes in time order, or `None` if no
    /// agent has this name. Empty unless queue depth metrics are enabled.
    #[must_use]
    pub fn queue_depth_metrics(&self, name: &str) -> Option<&[QueueDepthChange]> {
        Some(self.find_by_name(name)?.queue_depth_metrics())
    }

    /// Returns the number of ticks the agent named `name` spent asleep before
    /// its wake tick, or `None` if no agent has this name. Always `Some(0)`
    /// unless [`SimulationBuilder::asleep_cycles_metric`] is enabled.
    #[must_use]
    pub fn asleep_cycle_count(&self, name: &str) -> Option<DiscreteTime> {
        Some(self.find_by_name(name)?.asleep_cycle_count())
    }

    /// Returns messages sent with [`AgentContext::send_after`] that have not
    /// arrived yet, in arrival order (then send order). A run that halts leaves
    /// them in flight, and the next run delivers them.
    pub fn in_flight(&self) -> impl ExactSizeIterator<Item = &Message<M>> + DoubleEndedIterator {
        self.in_flight.values()
    }

    /// Cancels a pending delayed message scheduled with
    /// [`AgentContext::send_after`], returning `true` if it was still in flight
    /// and removed.
    pub fn cancel_timer(&mut self, timer: TimerId) -> bool {
        let removed = self
            .in_flight
            .remove(&(timer.arrival, timer.sequence))
            .is_some();
        if removed && timer.arrival == self.next_arrival {
            self.next_arrival = self
                .in_flight
                .first_key_value()
                .map_or(DiscreteTime::MAX, |(key, _)| key.0);
        }
        removed
    }

    /// Returns whether `timer` is currently pending delivery in
    /// [`Self::in_flight`].
    #[must_use]
    pub fn is_timer_pending(&self, timer: TimerId) -> bool {
        self.in_flight
            .contains_key(&(timer.arrival, timer.sequence))
    }

    /// Returns the policy applied to messages addressed to an unknown agent
    /// name or out-of-range [`AgentId`].
    #[must_use]
    pub const fn unknown_destination_policy(&self) -> UnknownDestinationPolicy {
        self.unknown_destination_policy
    }

    /// Returns how many messages were addressed to an unknown agent name or
    /// out-of-range [`AgentId`].
    #[must_use]
    pub const fn unknown_destination_count(&self) -> u64 {
        self.unknown_destination_count
    }

    /// Enables structured lifecycle tracing from the current tick onward.
    ///
    /// Existing recorded events (if tracing was previously enabled and
    /// disabled) are preserved.
    pub fn enable_trace(&mut self) {
        if let Some(recorder) = &mut self.trace {
            recorder.enabled = true;
        } else {
            let names = self.agents.iter().map(|a| a.name.clone()).collect();
            self.trace = Some(TraceRecorder::new(names, None));
        }
    }

    /// Stops recording new trace events while keeping events recorded so far
    /// readable through [`Self::trace`] or [`Self::take_trace`].
    pub const fn disable_trace(&mut self) {
        if let Some(recorder) = &mut self.trace {
            recorder.enabled = false;
        }
    }

    /// Returns the recorded structured trace, or `None` if tracing was never
    /// enabled (or after [`Self::take_trace`]).
    #[must_use]
    pub const fn trace(&self) -> Option<&SimulationTrace> {
        match &self.trace {
            Some(recorder) => Some(&recorder.trace),
            None => None,
        }
    }

    /// Removes and returns the recorded structured trace, disabling further
    /// tracing until re-enabled.
    #[must_use]
    pub fn take_trace(&mut self) -> Option<SimulationTrace> {
        self.trace.take().map(|recorder| recorder.trace)
    }

    /// Runs the simulation until its halt condition holds or an agent sends a
    /// halt interrupt.
    ///
    /// # Panics
    ///
    /// Panics if the simulation is [`SimulationMode::Failed`], before invoking
    /// tick callbacks if advancing time would exceed `u64::MAX`, and when an
    /// agent callback or the halt check panics. Any panic during the run
    /// leaves the simulation [`SimulationMode::Failed`].
    pub fn run(&mut self) {
        let _ = self.run_guarded(None);
    }

    /// Runs like [`Self::run`], but pauses once the simulation time reaches
    /// `time`, without running that tick, as [`Halt::AtTime`] would.
    ///
    /// A paused simulation is [`SimulationMode::Paused`]; [`Self::run`] or
    /// another `run_until` continues it exactly as one uninterrupted run
    /// would, and quiet ticks are still skipped. If the halt condition or an
    /// interrupt ends the run first, the simulation is
    /// [`SimulationMode::Completed`] instead. A `time` at or before the current
    /// time runs no tick.
    ///
    /// Pausing at a decision tick and cloning gives one branch per
    /// alternative that shares the run so far. With equal seeds, a branch
    /// matches a run that took its alternative from the start when the
    /// alternatives act identically before the decision tick.
    ///
    /// ```
    /// use simul::agent::{Agent, AgentContext, AgentInitializer};
    /// use simul::{Halt, Simulation, SimulationMode};
    ///
    /// #[derive(Clone, Debug)]
    /// struct Counter {
    ///     step: u64,
    ///     total: u64,
    /// }
    ///
    /// impl Agent for Counter {
    ///     fn on_tick(&mut self, _ctx: &mut AgentContext) {
    ///         self.total += self.step;
    ///     }
    /// }
    ///
    /// let mut trunk = Simulation::builder(Halt::AtTime(10))
    ///     .agent(AgentInitializer::proactive("counter", Counter { step: 1, total: 0 }))
    ///     .build()?;
    /// trunk.run_until(4);
    /// assert_eq!(SimulationMode::Paused, trunk.mode());
    ///
    /// let mut branch = trunk.clone();
    /// branch
    ///     .find_by_name_mut("counter")
    ///     .and_then(|agent| agent.downcast_mut::<Counter>())
    ///     .expect("counter agent")
    ///     .step = 2;
    /// trunk.run();
    /// branch.run();
    ///
    /// let total = |sim: &Simulation| {
    ///     sim.find_by_name("counter")
    ///         .and_then(|agent| agent.downcast_ref::<Counter>())
    ///         .map(|counter| counter.total)
    /// };
    /// assert_eq!(Some(10), total(&trunk));
    /// assert_eq!(Some(16), total(&branch));
    /// # Ok::<(), simul::SimulationError>(())
    /// ```
    ///
    /// # Panics
    ///
    /// As [`Self::run`].
    pub fn run_until(&mut self, time: DiscreteTime) {
        let _ = self.run_guarded(Some(time));
    }

    /// Runs ticks until the simulation halts, hits an interrupt, reaches
    /// `max_time` (if `Some`), or pauses at an agent's
    /// [`AgentContext::yield_control`] decision point.
    ///
    /// When [`RunOutcome::Yielded`] is returned, the simulation is in
    /// [`SimulationMode::Paused`] at a clean between-tick boundary: callers may
    /// inspect observations, clone the simulation for counterfactual rollouts,
    /// mutate agent state or inject messages, and call `run_until_yield` again
    /// to resume deterministically.
    ///
    /// # Panics
    ///
    /// As [`Self::run`].
    pub fn run_until_yield(&mut self, max_time: Option<DiscreteTime>) -> RunOutcome {
        self.run_guarded(max_time)
    }

    /// Runs ticks until halted, interrupted, paused at `until`, or yielded by
    /// an agent, marking the simulation failed if anything panics.
    fn run_guarded(&mut self, until: Option<DiscreteTime>) -> RunOutcome {
        assert!(
            self.mode != SimulationMode::Failed,
            "a simulation that panicked mid-tick cannot run again"
        );
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.run_ticks(until))) {
            Ok(outcome) => outcome,
            Err(payload) => {
                self.mode = SimulationMode::Failed;
                std::panic::resume_unwind(payload);
            }
        }
    }

    /// Returns whether the simulation is new, running, paused, completed, or
    /// failed.
    #[must_use]
    pub const fn mode(&self) -> SimulationMode {
        self.mode
    }

    /// Returns the reason passed to [`AgentContext::send_halt_interrupt`] by
    /// the first interrupt that ended the latest run, or `None` when the run
    /// ended by its [`Halt`] condition or paused.
    #[must_use]
    pub fn halt_reason(&self) -> Option<&str> {
        self.halt_reason.as_deref()
    }

    /// Returns the tag passed to [`AgentContext::yield_control`] when the
    /// latest run paused at a yield point, or `None` otherwise.
    #[must_use]
    pub fn yield_tag(&self) -> Option<&str> {
        self.yield_point.as_ref().map(|(_, tag)| tag.as_str())
    }

    /// Returns the identity of the agent that paused the latest run via
    /// [`AgentContext::yield_control`], or `None` otherwise.
    #[must_use]
    pub fn yielded_by(&self) -> Option<AgentId> {
        self.yield_point.as_ref().map(|(agent, _)| *agent)
    }

    /// Replaces the simulation's root seed for all subsequent
    /// [`AgentContext::rng`] calls without altering agents, queues, or time.
    ///
    /// Use this on a cloned, paused simulation to evaluate multiple stochastic
    /// future rollouts from one shared decision-point prefix.
    pub fn reseed(&mut self, seed: MasterSeed) {
        self.agent_streams = AgentStreamCache::new(seed);
    }

    /// Evaluates each candidate action in `branches` from this paused snapshot:
    /// clones `self`, applies `patch(&mut branch, action)`, runs `branch.run()`,
    /// and collects `extract(&branch)`.
    ///
    /// Because agent random streams depend only on seed, name, and tick, each
    /// branch shares identical future noise with the others (Common Random
    /// Numbers) while executing the shared prefix only once.
    pub fn rollout_branches<B, O>(
        &self,
        branches: &[B],
        mut patch: impl FnMut(&mut Self, &B),
        mut extract: impl FnMut(&Self) -> O,
    ) -> Vec<O>
    where
        M: Clone,
    {
        let mut outcomes = Vec::with_capacity(branches.len());
        for action in branches {
            let mut branch = self.clone();
            patch(&mut branch, action);
            branch.run();
            outcomes.push(extract(&branch));
        }
        outcomes
    }

    /// Evaluates the halt condition against the current state.
    fn halt_reached(&self) -> bool {
        match &self.halt {
            Halt::AtTime(time) => self.time >= *time,
            Halt::Never => false,
            Halt::When(predicate) => predicate(self),
        }
    }

    /// Executes ticks until the halt condition, an interrupt, or a yield stops
    /// the run, or until the time reaches `until`.
    fn run_ticks(&mut self, until: Option<DiscreteTime>) -> RunOutcome {
        self.mode = SimulationMode::Running;
        self.halt_reason = None;
        self.yield_point = None;
        let mut command_buffer: CommandBuffer<M> = CommandBuffer::new();
        if self.enable_queue_depth_metric {
            // Queues may have been edited between runs.
            let time = self.time;
            for agent in &mut self.agents {
                agent.record_queue_depth(time);
            }
        }
        let mut scheduler = Scheduler::new(&self.agents, self.enable_agent_asleep_cycles_metric);
        let quiet_ticks =
            QuietTicks::new(&self.halt, self.enable_agent_asleep_cycles_metric, until);

        // The halt check comes first, so a run that both halts and reaches
        // `until` at one tick completes.
        // For non-predicate halts, one compare checks halt, pause, and u64::MAX
        // overflow together; `self.time < stop_at` guarantees `self.time + 1`
        // cannot overflow.
        let pause_at = until.unwrap_or(DiscreteTime::MAX);
        let (stop_at, has_predicate) = match &self.halt {
            Halt::AtTime(time) => ((*time).min(pause_at), false),
            Halt::Never => (pause_at, false),
            Halt::When(_) => (pause_at, true),
        };
        let outcome = loop {
            if has_predicate && self.halt_reached() {
                break RunOutcome::Halted;
            }
            if self.time >= stop_at {
                if !has_predicate && self.halt_reached() {
                    break RunOutcome::Halted;
                }
                if until.is_some() {
                    break RunOutcome::PausedAtTime;
                }
                panic!("simulation time exceeds u64::MAX");
            }
            let next_time = self.time + 1;
            scheduler.before_tick(self.time, &self.agents);
            let callbacks = match scheduler.sparse_mut() {
                Some(active) => self.run_sparse_tick(&mut command_buffer, active),
                None => self.run_dense_tick(&mut command_buffer),
            };

            let stop =
                self.process_command_buffer(&mut command_buffer, scheduler.sparse_mut(), next_time);

            self.time = next_time;
            if let Some(stop) = stop {
                break match *stop {
                    TickStop::Interrupted(reason) => RunOutcome::Interrupted(reason),
                    TickStop::Yielded(agent, tag) => RunOutcome::Yielded { agent, tag },
                };
            }
            scheduler.after_tick(callbacks, &self.agents);
            // A tick without callbacks issued no commands, so state can change
            // again only when an agent becomes runnable or a delayed message
            // arrives.
            if callbacks == 0 && quiet_ticks != QuietTicks::Observed {
                self.skip_quiet_ticks(&mut scheduler, quiet_ticks);
            }
        };

        (self.mode, self.halt_reason, self.yield_point) = match &outcome {
            RunOutcome::Halted => (SimulationMode::Completed, None, None),
            RunOutcome::Interrupted(reason) => {
                (SimulationMode::Completed, Some(reason.clone()), None)
            }
            RunOutcome::PausedAtTime => (SimulationMode::Paused, None, None),
            RunOutcome::Yielded { agent, tag } => {
                (SimulationMode::Paused, None, Some((*agent, tag.clone())))
            }
        };
        outcome
    }

    /// Jumps from a tick without callbacks to the next tick at which an agent
    /// can act or a delayed message arrives, within the halt or pause time,
    /// and delivers what arrives then. Kept out of line, since busy runs
    /// never take it.
    #[inline(never)]
    fn skip_quiet_ticks(&mut self, scheduler: &mut Scheduler, quiet_ticks: QuietTicks) {
        let next_agent = match scheduler {
            Scheduler::Sparse(active) => active.next_event(self.time),
            Scheduler::Dense | Scheduler::DenseOnly => self.next_event_dense(self.time),
        };
        let next_arrival = (!self.in_flight.is_empty()).then_some(self.next_arrival);
        let next_event = next_agent.into_iter().chain(next_arrival).min();
        if let Some(target) = quiet_ticks.skip_to(next_event) {
            self.time = self.time.max(target);
            if self.next_arrival <= self.time {
                self.deliver_due(self.time, scheduler.sparse_mut());
            }
        }
    }

    /// Borrows the agents mutably beside the inputs every visit shares.
    fn split_tick<'a>(
        &'a mut self,
        commands: &'a mut CommandBuffer<M>,
    ) -> (&'a mut [SimulationAgent<M>], Tick<'a, M>) {
        let agent_count = self.agents.len();
        let trace = self
            .trace
            .as_mut()
            .filter(|recorder| recorder.enabled)
            .map(|recorder| &mut recorder.trace);
        let tick = Tick {
            time: self.time,
            layout: self.layout,
            agent_count,
            scan_names: agent_count <= LINEAR_NAME_SCAN_LIMIT,
            queue_depth_metric: self.enable_queue_depth_metric,
            asleep_cycles_metric: self.enable_agent_asleep_cycles_metric,
            agent_streams: &mut self.agent_streams,
            agent_name_handle_map: &self.agent_name_handle_map,
            commands,
            next_in_flight: &mut self.next_in_flight,
            trace,
        };
        (&mut self.agents, tick)
    }

    /// Visits every agent in handle order and returns how many callbacks ran.
    fn run_dense_tick(&mut self, commands: &mut CommandBuffer<M>) -> usize {
        let (agents, mut tick) = self.split_tick(commands);
        let mut callbacks = 0;
        for handle in 0..agents.len() {
            let (ran, _) = visit_agent(agents, handle, &mut tick);
            callbacks += usize::from(ran);
        }
        callbacks
    }

    /// Visits only scheduled agents in handle order, keeps the schedule in step,
    /// and returns how many callbacks ran. Kept out of line so the sparse path
    /// does not enlarge the dense loop in `run`.
    #[inline(never)]
    fn run_sparse_tick(
        &mut self,
        commands: &mut CommandBuffer<M>,
        schedule: &mut SparseSchedule,
    ) -> usize {
        let (agents, mut tick) = self.split_tick(commands);
        let mut callbacks = 0;
        let mut cursor = 0;
        while let Some(handle) = schedule.next(cursor) {
            cursor = handle + 1;
            let (ran, requested_sleep_until) = visit_agent(agents, handle, &mut tick);
            callbacks += usize::from(ran);
            match requested_sleep_until {
                Some(deadline) => schedule.sleep(handle, deadline),
                None => schedule.set_ready(handle, SparseSchedule::runnable(&agents[handle].state)),
            }
        }
        callbacks
    }

    /// Returns each agent's mean wait from queueing to completion over its
    /// consumed messages, keyed by agent name. Agents that consumed nothing
    /// are omitted. Needs no message history.
    #[must_use]
    pub fn calc_avg_wait_statistics(&self) -> NameStatistics<f64> {
        let mut data = NameStatistics::new();
        for agent in &self.agents {
            if let Some(wait) = agent.avg_wait() {
                data.insert(agent.name.clone(), wait);
            }
        }

        data
    }

    /// Calculates the statistics of queue lengths.
    /// Mostly useful for checking which agents still have queues of work after halting.
    #[must_use]
    pub fn calc_queue_len_statistics(&self) -> NameStatistics<usize> {
        let mut data = NameStatistics::with_capacity(self.agents.len());

        for agent in &self.agents {
            data.insert(agent.name.clone(), agent.state.queue.len());
        }

        data
    }

    /// Counts the messages each agent consumed.
    #[must_use]
    pub fn calc_consumed_len_statistics(&self) -> NameStatistics<u64> {
        let mut data = NameStatistics::with_capacity(self.agents.len());

        for agent in &self.agents {
            data.insert(agent.name.clone(), agent.consumed_count());
        }

        data
    }

    /// Counts the messages each agent sent, delivered or not.
    #[must_use]
    pub fn calc_produced_len_statistics(&self) -> NameStatistics<u64> {
        let mut data = NameStatistics::with_capacity(self.agents.len());

        for agent in &self.agents {
            data.insert(agent.name.clone(), agent.produced_count());
        }

        data
    }

    /// Returns the earliest tick at or after `now` at which some agent can act,
    /// scanning every agent.
    fn next_event_dense(&self, now: DiscreteTime) -> Option<DiscreteTime> {
        self.agents
            .iter()
            .filter_map(|agent| match agent.state.mode {
                AgentMode::Proactive => Some(now),
                AgentMode::Reactive => (!agent.state.queue.is_empty()).then_some(now),
                AgentMode::AsleepUntil { until, .. } => Some(until.max(now)),
                AgentMode::Dead => None,
            })
            .min()
    }

    /// Returns an agent handle by name, using a linear scan for small simulations.
    #[inline]
    fn find_handle_by_name(&self, name: &str) -> Option<usize> {
        if self.agents.len() <= LINEAR_NAME_SCAN_LIMIT {
            self.agents.iter().position(|agent| agent.name == name)
        } else {
            self.agent_name_handle_map.get(name).map(|id| id.index())
        }
    }

    /// Applies buffered Agent commands in the order they were issued, and
    /// returns the early stop requested by the first interrupt or yield, if any.
    #[inline]
    fn process_command_buffer(
        &mut self,
        command_buffer: &mut CommandBuffer<M>,
        mut schedule: Option<&mut SparseSchedule>,
        next_time: DiscreteTime,
    ) -> Option<Box<TickStop>> {
        if !command_buffer.specials.is_empty() {
            return self.process_command_buffer_with_specials(command_buffer, schedule, next_time);
        }
        // Delayed messages arriving next tick were sent in earlier ticks, so
        // they arrive ahead of this tick's sends.
        if self.next_arrival <= next_time {
            self.deliver_due(next_time, schedule.as_deref_mut());
        }
        if command_buffer.sends.is_empty() {
            return None;
        }
        let time = self.time;
        let trace_active = self.trace_active();
        let mut interrupt: Option<String> = None;
        for send in command_buffer.sends.drain(..) {
            let message = Message::sent(time, send.sender, send.destination, send.payload);
            let handle = send.sender.index();
            self.record_produced(handle, &message);
            if trace_active {
                self.record_trace_sent(handle, &message, next_time);
            }
            if let Some(destination) = message.destination {
                self.deliver(
                    destination,
                    message,
                    schedule.as_deref_mut(),
                    next_time,
                    trace_active,
                );
            } else if let Some(reason) = self.handle_unknown_destination(handle, time) {
                interrupt.get_or_insert(reason);
            }
        }
        interrupt.map(|reason| Box::new(TickStop::Interrupted(reason)))
    }

    /// Records and delivers one ordinary end-of-tick send from the special-command
    /// slow path.
    #[inline]
    fn apply_pending_send(
        &mut self,
        send: PendingSend<M>,
        time: DiscreteTime,
        next_time: DiscreteTime,
        schedule: Option<&mut SparseSchedule>,
        trace_active: bool,
        interrupt: &mut Option<String>,
    ) {
        let message = Message::sent(time, send.sender, send.destination, send.payload);
        let handle = send.sender.index();
        self.record_produced(handle, &message);
        if trace_active {
            self.record_trace_sent(handle, &message, next_time);
        }
        if let Some(destination) = message.destination {
            self.deliver(destination, message, schedule, next_time, trace_active);
        } else if let Some(reason) = self.handle_unknown_destination(handle, time) {
            interrupt.get_or_insert(reason);
        }
    }

    /// Slow path when a tick buffered delayed sends, timer cancellations,
    /// yields, or halt interrupts; interleaves them with ordinary sends in
    /// exact issue order.
    #[inline(never)]
    fn process_command_buffer_with_specials(
        &mut self,
        command_buffer: &mut CommandBuffer<M>,
        mut schedule: Option<&mut SparseSchedule>,
        next_time: DiscreteTime,
    ) -> Option<Box<TickStop>> {
        let has_cancellations = std::mem::replace(&mut command_buffer.has_cancellations, false);
        let cancelled_next_tick = if has_cancellations {
            self.cancel_due_timers(&command_buffer.specials, next_time)
        } else {
            Vec::new()
        };
        if self.next_arrival <= next_time {
            self.deliver_due(next_time, schedule.as_deref_mut());
        }
        let time = self.time;
        let trace_active = self.trace_active();
        let mut interrupt: Option<String> = None;
        let mut yield_point: Option<(AgentId, String)> = None;
        let mut sends = command_buffer.sends.drain(..);
        let mut sent_so_far = 0_usize;

        for (at_send, special) in command_buffer.specials.drain(..) {
            while sent_so_far < at_send {
                if let Some(send) = sends.next() {
                    self.apply_pending_send(
                        send,
                        time,
                        next_time,
                        schedule.as_deref_mut(),
                        trace_active,
                        &mut interrupt,
                    );
                }
                sent_so_far += 1;
            }
            match special {
                SpecialCommand::SendDelayed {
                    sender,
                    timer,
                    message,
                } => {
                    let handle = sender.index();
                    if timer.arrival == next_time {
                        self.record_produced(handle, &message);
                        if trace_active {
                            self.record_trace_sent(handle, &message, next_time);
                        }
                        if !cancelled_next_tick.contains(&timer) {
                            if let Some(destination) = message.destination {
                                self.deliver(
                                    destination,
                                    message,
                                    schedule.as_deref_mut(),
                                    next_time,
                                    trace_active,
                                );
                            } else if let Some(reason) =
                                self.handle_unknown_destination(handle, time)
                            {
                                interrupt.get_or_insert(reason);
                            }
                        }
                    } else if let Some(reason) =
                        self.hold_delayed(handle, timer, message, next_time)
                    {
                        interrupt.get_or_insert(reason);
                    }
                }

                SpecialCommand::CancelTimer { sender, timer } => {
                    if timer.arrival > next_time && self.cancel_timer(timer) && trace_active {
                        self.record_trace_cancelled(sender.index(), timer);
                    }
                }

                SpecialCommand::YieldControl { sender, tag } => {
                    yield_point.get_or_insert((sender, tag));
                }

                SpecialCommand::HaltSimulation { sender, reason } => {
                    if trace_active {
                        self.record_trace_halt(sender.index(), &reason);
                    }
                    // The first interrupt in issue order ends the run.
                    interrupt.get_or_insert(reason);
                }
            }
        }
        for send in sends {
            self.apply_pending_send(
                send,
                time,
                next_time,
                schedule.as_deref_mut(),
                trace_active,
                &mut interrupt,
            );
        }
        interrupt.map_or_else(
            || yield_point.map(|(agent, tag)| Box::new(TickStop::Yielded(agent, tag))),
            |reason| Some(Box::new(TickStop::Interrupted(reason))),
        )
    }

    /// Returns whether structured tracing is currently recording events.
    #[inline]
    fn trace_active(&self) -> bool {
        self.trace.as_ref().is_some_and(|recorder| recorder.enabled)
    }

    /// Appends a [`TraceEventKind::MessageSent`] event when tracing is active.
    #[cold]
    #[inline(never)]
    fn record_trace_sent(&mut self, handle: usize, message: &Message<M>, arrival: DiscreteTime) {
        if let Some(recorder) = self.trace.as_mut().filter(|r| r.enabled) {
            let payload = recorder.format(&message.payload);
            recorder.trace.push(
                self.time,
                AgentId::from_handle(handle),
                TraceEventKind::MessageSent {
                    destination: message.destination,
                    arrival,
                    payload,
                },
            );
        }
    }

    /// Appends a [`TraceEventKind::MessageDelivered`] event when tracing is active.
    #[cold]
    #[inline(never)]
    fn record_trace_delivered(
        &mut self,
        visible_at: DiscreteTime,
        destination: AgentId,
        source: Option<AgentId>,
        queue_depth: usize,
    ) {
        if let Some(recorder) = self.trace.as_mut().filter(|r| r.enabled) {
            recorder.trace.push(
                visible_at,
                destination,
                TraceEventKind::MessageDelivered {
                    source,
                    queue_depth,
                },
            );
        }
    }

    /// Appends a [`TraceEventKind::TimerCancelled`] event when tracing is active.
    #[cold]
    #[inline(never)]
    fn record_trace_cancelled(&mut self, handle: usize, timer: TimerId) {
        if let Some(recorder) = self.trace.as_mut().filter(|r| r.enabled) {
            recorder.trace.push(
                self.time,
                AgentId::from_handle(handle),
                TraceEventKind::TimerCancelled { timer },
            );
        }
    }

    /// Appends a [`TraceEventKind::HaltInterrupt`] event when tracing is active.
    #[cold]
    #[inline(never)]
    fn record_trace_halt(&mut self, handle: usize, reason: &str) {
        if let Some(recorder) = self.trace.as_mut().filter(|r| r.enabled) {
            recorder.trace.push(
                self.time,
                AgentId::from_handle(handle),
                TraceEventKind::HaltInterrupt {
                    reason: reason.to_string(),
                },
            );
        }
    }

    /// Records an unknown-destination send and applies the configured policy.
    #[cold]
    #[inline(never)]
    fn handle_unknown_destination(
        &mut self,
        sender_handle: usize,
        tick: DiscreteTime,
    ) -> Option<String> {
        self.unknown_destination_count += 1;
        match self.unknown_destination_policy {
            UnknownDestinationPolicy::Drop => None,
            UnknownDestinationPolicy::Halt => {
                let reason = format!(
                    "unknown destination sent by '{}' at tick {tick}",
                    self.agents[sender_handle].name
                );
                if self.trace_active() {
                    self.record_trace_halt(sender_handle, &reason);
                }
                Some(reason)
            }
            UnknownDestinationPolicy::Panic => panic!(
                "unknown destination sent by '{}' at tick {tick}",
                self.agents[sender_handle].name
            ),
        }
    }

    /// Removes timers due at or before `next_time` that were cancelled during
    /// this tick, and returns any `delay == 1` cancellations to suppress during
    /// command draining.
    #[inline(never)]
    fn cancel_due_timers(
        &mut self,
        specials: &[(usize, SpecialCommand<M>)],
        next_time: DiscreteTime,
    ) -> Vec<TimerId> {
        let mut cancelled_next_tick = Vec::new();
        for (_, command) in specials {
            if let SpecialCommand::CancelTimer { sender, timer } = *command {
                if timer.arrival <= next_time {
                    if self.cancel_timer(timer) && self.trace_active() {
                        self.record_trace_cancelled(sender.index(), timer);
                    }
                    if timer.arrival == next_time {
                        cancelled_next_tick.push(timer);
                    }
                }
            }
        }
        cancelled_next_tick
    }

    /// Counts a delayed message as sent by the agent at `handle` and holds it
    /// until it arrives. Kept out of line so that it does not crowd the tick
    /// loop, which most simulations run without delayed sends.
    #[inline(never)]
    fn hold_delayed(
        &mut self,
        handle: usize,
        timer: TimerId,
        message: Message<M>,
        next_time: DiscreteTime,
    ) -> Option<String> {
        self.record_produced(handle, &message);
        if self.trace_active() {
            self.record_trace_sent(handle, &message, timer.arrival);
        }
        if message.destination.is_some() {
            debug_assert!(
                message.queued_time >= next_time,
                "a delayed message arrives after the next tick"
            );
            debug_assert_eq!(timer.arrival, message.queued_time + 1);
            self.in_flight
                .insert((timer.arrival, timer.sequence), message);
            self.next_arrival = self.next_arrival.min(timer.arrival);
            None
        } else {
            self.handle_unknown_destination(handle, self.time)
        }
    }

    /// Counts a message as sent by the agent at `handle`, and records it when
    /// history is on.
    #[inline]
    fn record_produced(&mut self, handle: usize, message: &Message<M>) {
        let sender = &mut self.agents[handle];
        sender.metadata.produced_count += 1;
        if let Some(history) = &mut sender.history {
            history.produced.push((history.copy)(message));
        }
    }

    /// Appends a message to its receiver's queue, visible from `visible_at`.
    #[inline]
    fn deliver(
        &mut self,
        destination: AgentId,
        message: Message<M>,
        schedule: Option<&mut SparseSchedule>,
        visible_at: DiscreteTime,
        trace_active: bool,
    ) {
        let source = message.source;
        let receiver = &mut self.agents[destination.index()];
        receiver.state.queue.push_back(message);
        if self.enable_queue_depth_metric {
            receiver.record_queue_depth(visible_at);
        }
        if let Some(active) = schedule {
            if receiver.state.mode == AgentMode::Reactive {
                active.set_ready(destination.index(), true);
            }
        }
        if trace_active {
            let depth = receiver.state.queue.len();
            self.record_trace_delivered(visible_at, destination, source, depth);
        }
    }

    /// Delivers delayed messages that arrive at or before `now`, in arrival
    /// order, then send order. Kept out of line so the tick loop only checks
    /// the cached arrival time when nothing is due.
    #[inline(never)]
    fn deliver_due(&mut self, now: DiscreteTime, mut schedule: Option<&mut SparseSchedule>) {
        let trace_active = self.trace_active();
        while let Some(entry) = self.in_flight.first_entry() {
            if entry.key().0 > now {
                break;
            }
            let message = entry.remove();
            if let Some(destination) = message.destination {
                self.deliver(
                    destination,
                    message,
                    schedule.as_deref_mut(),
                    now,
                    trace_active,
                );
            }
        }
        self.next_arrival = self
            .in_flight
            .first_key_value()
            .map_or(DiscreteTime::MAX, |(key, _)| key.0);
    }

    /// Returns the first agent in construction order matching `predicate`, or
    /// `None` if no agent matches.
    #[must_use]
    pub fn find_agent<P>(&self, mut predicate: P) -> Option<&SimulationAgent<M>>
    where
        P: FnMut(&SimulationAgent<M>) -> bool,
    {
        self.agents.iter().find(|agent| predicate(agent))
    }

    /// Returns `true` if every agent matches `predicate` (vacuously `true` when
    /// there are no agents).
    #[must_use]
    pub fn all_agents<P>(&self, predicate: P) -> bool
    where
        P: FnMut(&SimulationAgent<M>) -> bool,
    {
        self.agents.iter().all(predicate)
    }

    /// Returns a slice of the Agents in the Simulation.
    #[must_use]
    pub fn agents(&self) -> &[SimulationAgent<M>] {
        &self.agents
    }

    /// Returns the simulation's current tick.
    ///
    /// Starts at [`SimulationBuilder::starting_time`] and advances after each
    /// tick (or jumps over quiet ticks). After [`Self::run`] or
    /// [`Self::run_until`], this is the first tick that has not been executed.
    #[must_use]
    pub const fn time(&self) -> DiscreteTime {
        self.time
    }
}

impl<M: std::fmt::Debug> Simulation<M> {
    /// Enables structured lifecycle tracing from the current tick onward,
    /// recording a `Debug` summary of each sent message payload.
    pub fn enable_trace_with_payloads(&mut self) {
        if let Some(recorder) = &mut self.trace {
            recorder.enabled = true;
            recorder.format_payload = Some(debug_format_payload::<M>);
        } else {
            let names = self.agents.iter().map(|a| a.name.clone()).collect();
            self.trace = Some(TraceRecorder::new(names, Some(debug_format_payload::<M>)));
        }
    }
}

/// Evaluates counterfactual `branches` forked from a shared paused or yielded
/// `trunk` simulation across `schedule` replications using Common Random
/// Numbers (`CRN`).
///
/// Each branch index `i` is assigned a [`CandidateId`] with integer value
/// `i as u64`. For
/// every `(branch, replication)` pair, this clones `trunk`, reseeds it with
/// that replication's common-random `simul::Simulation` seed (so all branches
/// experience identical stochastic futures on each replication), applies
/// `patch(&mut branch, action)`, runs `branch.run()`, and returns
/// `evaluate(&branch)`.
///
/// The returned [`TrialRecord`]s are directly compatible with
/// [`aggregate_scalar`](experiment::aggregate_scalar),
/// [`aggregate_quantiles`](experiment::aggregate_quantiles),
/// [`compare_paired_candidates`](experiment::compare_paired_candidates), and
/// [`ParetoArchive`](experiment::ParetoArchive).
///
/// # Errors
///
/// Returns [`RunError`] if serial execution encounters a batch limit,
/// allocation error, or evaluator panic.
pub fn evaluate_branch_replications<M, B, O, E>(
    trunk: &Simulation<M>,
    master_seed: MasterSeed,
    schedule: FixedReplications,
    branches: &[B],
    patch: impl Fn(&mut Simulation<M>, &B),
    evaluate: impl Fn(&Simulation<M>) -> Result<O, E>,
) -> Result<Vec<TrialRecord<O, E>>, RunError>
where
    M: Clone,
{
    let trials = branches.iter().enumerate().flat_map(|(index, action)| {
        let candidate = CandidateId::new(u64::try_from(index).unwrap_or(u64::MAX));
        schedule.trials(candidate, action)
    });
    experiment::run_serial(master_seed, trials, |&action, ctx| {
        let mut branch = trunk.clone();
        let future_seed = ctx.common_seed(TrialStream::TrialNoise, "simul::Simulation");
        branch.reseed(MasterSeed::new(future_seed.bytes()));
        patch(&mut branch, action);
        branch.run();
        evaluate(&branch)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent::{periodic_consumer, periodic_producer};

    #[test]
    fn basic_periodic_test() {
        let mut simulation = Simulation::<()>::builder(Halt::AtTime(5))
            .agents([
                periodic_producer("producer", 1, "consumer"),
                periodic_consumer("consumer", 1),
            ])
            .build()
            .unwrap_or_else(|error| panic!("{error}"));
        simulation.run();
        let produced_stats = simulation.calc_produced_len_statistics();
        assert_eq!(produced_stats.get("producer"), Some(&5));
        assert_eq!(produced_stats.get("consumer"), Some(&0));

        let consumed_stats = simulation.calc_consumed_len_statistics();
        assert_eq!(consumed_stats.get("producer"), Some(&0));
        assert_eq!(consumed_stats.get("consumer"), Some(&4));
    }

    #[test]
    fn sleeping_reactive_agent_preserves_messages_until_wakeup() {
        let consumer = periodic_consumer("consumer", 1).asleep_until(3);
        let mut simulation = Simulation::<()>::builder(Halt::AtTime(5))
            .agents([periodic_producer("producer", 1, "consumer"), consumer])
            .build()
            .unwrap_or_else(|error| panic!("{error}"));

        simulation.run();

        assert_eq!(
            simulation.calc_produced_len_statistics().get("producer"),
            Some(&5)
        );
        assert_eq!(
            simulation.calc_consumed_len_statistics().get("consumer"),
            Some(&2)
        );
        assert_eq!(
            simulation.calc_queue_len_statistics().get("consumer"),
            Some(&3)
        );
    }

    #[test]
    fn quiet_ticks_skip_only_unobserved_ticks_and_stop_at_the_bound() {
        let when = Halt::<()>::when(|_| false);
        assert_eq!(QuietTicks::Observed, QuietTicks::new(&when, false, Some(3)));
        assert_eq!(
            QuietTicks::Observed,
            QuietTicks::new(&Halt::<()>::Never, true, None)
        );
        assert_eq!(
            QuietTicks::SkipUpTo(3),
            QuietTicks::new(&Halt::<()>::AtTime(5), false, Some(3))
        );
        assert_eq!(
            QuietTicks::SkipUpTo(5),
            QuietTicks::new(&Halt::<()>::AtTime(5), false, Some(8))
        );
        assert_eq!(
            QuietTicks::SkipUpTo(8),
            QuietTicks::new(&Halt::<()>::Never, false, Some(8))
        );
        assert_eq!(
            QuietTicks::SkipUnbounded,
            QuietTicks::new(&Halt::<()>::Never, false, None)
        );

        assert_eq!(None, QuietTicks::Observed.skip_to(Some(2)));
        assert_eq!(Some(2), QuietTicks::SkipUpTo(5).skip_to(Some(2)));
        assert_eq!(Some(5), QuietTicks::SkipUpTo(5).skip_to(Some(9)));
        assert_eq!(Some(5), QuietTicks::SkipUpTo(5).skip_to(None));
        assert_eq!(Some(9), QuietTicks::SkipUnbounded.skip_to(Some(9)));
        assert_eq!(None, QuietTicks::SkipUnbounded.skip_to(None));
    }

    #[test]
    fn rejects_duplicate_agent_names() {
        for count in [2, 8, 9] {
            let mut agent_initializers: Vec<AgentInitializer> = (1..count)
                .map(|i| periodic_consumer(format!("worker-{i}"), 1))
                .collect();
            agent_initializers.push(periodic_consumer("worker-1", 2));
            let result = Simulation::builder(Halt::AtTime(0))
                .agents(agent_initializers)
                .build();

            assert!(
                matches!(&result, Err(SimulationError::DuplicateAgentName(name)) if name == "worker-1"),
                "{count} agents: {result:?}"
            );
        }
    }
}
