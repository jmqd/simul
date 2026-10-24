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

pub mod agent;
/// Deterministic engine-independent experiment, reduction, and search APIs.
pub mod experiment;
pub mod message;

use agent::{
    Agent, AgentCommand, AgentCommandType, AgentContext, AgentId, AgentInitializer, AgentMetadata,
    AgentMode, AgentState, MessageHistory, MessageProcessingStatus, QueueDepthChange,
    SimulationAgent,
};
use experiment::{AgentStreams, MasterSeed, SeedDeriver};
use message::Message;

use rustc_hash::FxHashMap;
use std::cmp::Reverse;
use std::collections::{hash_map::Entry, BinaryHeap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Source of agent layout tokens; each constructed simulation takes the next.
static NEXT_LAYOUT: AtomicU64 = AtomicU64::new(0);

/// Per-agent statistics keyed by agent name; public methods return std maps.
#[allow(
    clippy::disallowed_types,
    reason = "public statistics keep returning std HashMap"
)]
type NameStatistics<V> = std::collections::HashMap<String, V>;

/// `DiscreteTime` is a Simulation's internal representation of time.
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
    /// The Simulation successfully reached the halt condition.
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

    /// Each agent's stream root, by handle; names never change, so neither do
    /// these.
    agent_streams: Vec<AgentStreams>,

    /// The reason given by the interrupt that ended the latest run, if any.
    halt_reason: Option<String>,
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
    commands: &'a mut Vec<AgentCommand<M>>,
    /// Stream roots by handle.
    agent_streams: &'a [AgentStreams],
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
            if let AgentMode::AsleepUntil(deadline) = agent.state.mode {
                if agent.state.mode != agent.state.wake_mode {
                    sleeping.push(Reverse((deadline, handle)));
                }
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
            if agents[handle].state.mode == AgentMode::AsleepUntil(deadline) {
                self.set_ready(handle, true);
            }
        }
    }
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
    if let AgentMode::AsleepUntil(wakeup_at) = agent.state.mode {
        if tick.time < wakeup_at {
            // A sleeping agent pops no message, runs no callback, and requests
            // no sleep; only the optional asleep count changes.
            if tick.asleep_cycles_metric {
                agent.metadata.asleep_cycle_count += 1;
            }
            return (false, None);
        }
        agent.state.mode = agent.state.wake_mode;
    }
    let proactive = match agent.state.mode {
        AgentMode::Proactive => true,
        AgentMode::Reactive => false,
        AgentMode::AsleepUntil(_) => {
            if tick.asleep_cycles_metric {
                agent.metadata.asleep_cycle_count += 1;
            }
            return (false, None);
        }
        AgentMode::Dead => return (false, None),
    };
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
        message_processing_status: MessageProcessingStatus::NoError,
        stream_root: &tick.agent_streams[handle],
        calls: 0,
    };
    // Every awake agent handles at most one queued message per tick; proactive
    // agents then also tick.
    if let Some(msg) = &message {
        agent.agent.on_message(&mut ctx, msg);
    }
    let status = std::mem::replace(
        &mut ctx.message_processing_status,
        MessageProcessingStatus::NoError,
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
            MessageProcessingStatus::NoError => agent.record_consumed(msg, tick.time),
        }
    }
    if let Some(sleep_until) = requested_sleep_until {
        agent.state.mode = AgentMode::AsleepUntil(sleep_until);
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
        let deriver = SeedDeriver::new(parameters.seed);
        let message_history = parameters.message_history;
        let mut agent_streams = Vec::with_capacity(initializers.len());
        let agents: Vec<SimulationAgent<M>> = parameters
            .agent_initializers
            .into_iter()
            .enumerate()
            .map(|(handle, agent_initializer)| {
                agent_streams.push(deriver.agent_streams(&agent_initializer.options.name));
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
                        wake_mode: agent_initializer.options.wake_mode,
                        queue,
                    },
                    history: message_history.map(|copy| MessageHistory {
                        consumed: Vec::with_capacity(history_capacity),
                        produced: Vec::with_capacity(history_capacity),
                        copy,
                    }),
                }
            })
            .collect();

        Ok(Self {
            mode: SimulationMode::Constructed,
            agents,
            halt: parameters.halt,
            time: parameters.starting_time,
            enable_queue_depth_metric: parameters.queue_depth_metrics,
            enable_agent_asleep_cycles_metric: parameters.asleep_cycles_metric,
            agent_name_handle_map,
            layout: NEXT_LAYOUT.fetch_add(1, Ordering::Relaxed),
            agent_streams,
            halt_reason: None,
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

    /// Returns a `SimulationAgent` by name.
    #[must_use]
    #[inline]
    pub fn find_by_name(&self, name: &str) -> Option<&SimulationAgent<M>> {
        self.find_handle_by_name(name)
            .and_then(|handle| self.agents.get(handle))
    }

    /// Returns the identity of the agent named `name`.
    #[must_use]
    #[inline]
    pub fn agent_id(&self, name: &str) -> Option<AgentId> {
        self.find_handle_by_name(name).map(AgentId::from_handle)
    }

    /// Returns the agent with identity `id`.
    #[must_use]
    #[inline]
    pub fn agent(&self, id: AgentId) -> Option<&SimulationAgent<M>> {
        self.agents.get(id.index())
    }

    /// Returns mutable behavior and state, without exposing the routing identity.
    ///
    /// ```compile_fail
    /// use simul::Simulation;
    /// fn rename(simulation: &mut Simulation) {
    ///     simulation.find_by_name_mut("worker").unwrap().name = "renamed".into();
    /// }
    /// ```
    #[inline]
    pub fn find_by_name_mut(&mut self, name: &str) -> Option<AgentParts<'_, M>> {
        let handle = self.find_handle_by_name(name)?;
        let agent = self.agents.get_mut(handle)?;
        Some((&mut agent.agent, &mut agent.state))
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
        Some(&self.find_by_name(name)?.metadata.queue_depth_metrics)
    }

    /// Returns the asleep cycle count for a given Agent during the Simulation.
    #[must_use]
    pub fn asleep_cycle_count(&self, name: &str) -> Option<DiscreteTime> {
        Some(self.find_by_name(name)?.metadata.asleep_cycle_count)
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
        assert!(
            self.mode != SimulationMode::Failed,
            "a simulation that panicked mid-tick cannot run again"
        );
        if let Err(payload) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.run_ticks()))
        {
            self.mode = SimulationMode::Failed;
            std::panic::resume_unwind(payload);
        }
    }

    /// Returns whether the simulation is new, running, completed, or failed.
    #[must_use]
    pub const fn mode(&self) -> SimulationMode {
        self.mode
    }

    /// Returns the reason passed to [`AgentContext::send_halt_interrupt`] by
    /// the first interrupt that ended the latest run, or `None` when the run
    /// ended by its [`Halt`] condition.
    #[must_use]
    pub fn halt_reason(&self) -> Option<&str> {
        self.halt_reason.as_deref()
    }

    /// Evaluates the halt condition against the current state.
    fn halt_reached(&self) -> bool {
        match &self.halt {
            Halt::AtTime(time) => self.time >= *time,
            Halt::Never => false,
            Halt::When(predicate) => predicate(self),
        }
    }

    /// Executes ticks until the halt condition or an interrupt stops the run.
    fn run_ticks(&mut self) {
        self.mode = SimulationMode::Running;
        self.halt_reason = None;
        let mut command_buffer: Vec<AgentCommand<M>> = Vec::new();
        if self.enable_queue_depth_metric {
            // Queues may have been edited between runs.
            let time = self.time;
            for agent in &mut self.agents {
                agent.record_queue_depth(time);
            }
        }
        // Small scans are cheaper than maintaining the schedule, and the asleep
        // metric needs every agent visited every tick, so either keeps dense scans.
        let sparse_eligible = self.agents.len() > SPARSE_SCHEDULE_MIN_AGENTS
            && !self.enable_agent_asleep_cycles_metric;
        let mut schedule = sparse_eligible.then(|| SparseSchedule::new(&self.agents));
        // Quiet ticks can be skipped only when nothing observes them: a
        // `Halt::When` predicate sees every tick, and the asleep metric counts
        // every tick. The bound is the halt time, if any.
        let skip_bound = match (&self.halt, self.enable_agent_asleep_cycles_metric) {
            (Halt::AtTime(time), false) => Some(Some(*time)),
            (Halt::Never, false) => Some(None),
            (Halt::AtTime(_) | Halt::Never, true) | (Halt::When(_), _) => None,
        };

        while self.mode == SimulationMode::Running && !self.halt_reached() {
            let Some(next_time) = self.time.checked_add(1) else {
                panic!("simulation time exceeds u64::MAX");
            };
            if let Some(active) = &mut schedule {
                active.wake(self.time, &self.agents);
                if active.ready_count >= self.agents.len().div_ceil(4) {
                    schedule = None;
                }
            }
            let callbacks = match schedule.as_mut() {
                Some(active) => self.run_sparse_tick(&mut command_buffer, active),
                None => self.run_dense_tick(&mut command_buffer),
            };

            self.process_command_buffer(&mut command_buffer, schedule.as_mut(), next_time);

            self.time = next_time;
            // Separate low and high watermarks amortize rebuilding across busy and
            // quiet phases instead of switching around one threshold.
            if sparse_eligible
                && schedule.is_none()
                && callbacks < self.agents.len().div_ceil(8)
                && self.mode == SimulationMode::Running
            {
                schedule = Some(SparseSchedule::new(&self.agents));
            }
            // A tick without callbacks issued no commands, so state can change
            // again only when an agent becomes runnable; jump to that tick.
            if let Some(bound) = skip_bound {
                if callbacks == 0 && self.mode == SimulationMode::Running {
                    let next_event = match &schedule {
                        Some(active) => active.next_event(self.time),
                        None => self.next_event_dense(self.time),
                    };
                    let target = match (next_event, bound) {
                        (Some(event), Some(halt)) => event.min(halt),
                        (None, Some(halt)) => halt,
                        (Some(event), None) => event,
                        (None, None) => self.time,
                    };
                    self.time = self.time.max(target);
                }
            }
        }

        self.mode = SimulationMode::Completed;
    }

    /// Borrows the agents mutably beside the inputs every visit shares.
    fn split_tick<'a>(
        &'a mut self,
        commands: &'a mut Vec<AgentCommand<M>>,
    ) -> (&'a mut [SimulationAgent<M>], Tick<'a, M>) {
        let agent_count = self.agents.len();
        let tick = Tick {
            time: self.time,
            layout: self.layout,
            agent_count,
            scan_names: agent_count <= LINEAR_NAME_SCAN_LIMIT,
            queue_depth_metric: self.enable_queue_depth_metric,
            asleep_cycles_metric: self.enable_agent_asleep_cycles_metric,
            agent_streams: &self.agent_streams,
            agent_name_handle_map: &self.agent_name_handle_map,
            commands,
        };
        (&mut self.agents, tick)
    }

    /// Visits every agent in handle order and returns how many callbacks ran.
    fn run_dense_tick(&mut self, commands: &mut Vec<AgentCommand<M>>) -> usize {
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
        commands: &mut Vec<AgentCommand<M>>,
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
        for agent in self
            .agents
            .iter()
            .filter(|agent| agent.consumed_count() > 0)
        {
            data.insert(
                agent.name.clone(),
                agent.metadata.total_wait / agent.consumed_count() as f64,
            );
        }

        data
    }

    /// Calculates the statistics of queue lengths.
    /// Mostly useful for checking which agents still have queues of work after halting.
    #[must_use]
    pub fn calc_queue_len_statistics(&self) -> NameStatistics<usize> {
        let mut data = NameStatistics::new();

        for agent in &self.agents {
            data.insert(agent.name.clone(), agent.state.queue.len());
        }

        data
    }

    /// Counts the messages each agent consumed.
    #[must_use]
    pub fn calc_consumed_len_statistics(&self) -> NameStatistics<u64> {
        let mut data = NameStatistics::new();

        for agent in &self.agents {
            data.insert(agent.name.clone(), agent.consumed_count());
        }

        data
    }

    /// Counts the messages each agent sent, delivered or not.
    #[must_use]
    pub fn calc_produced_len_statistics(&self) -> NameStatistics<u64> {
        let mut data = NameStatistics::new();

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
                AgentMode::AsleepUntil(deadline) => Some(deadline.max(now)),
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

    /// Applies buffered Agent commands in the order they were issued.
    #[inline]
    fn process_command_buffer(
        &mut self,
        command_buffer: &mut Vec<AgentCommand<M>>,
        mut schedule: Option<&mut SparseSchedule>,
        next_time: DiscreteTime,
    ) {
        for command in command_buffer.drain(..) {
            match command.ty {
                AgentCommandType::SendMessage(message) => {
                    let sender = &mut self.agents[command.agent_handle];
                    sender.metadata.produced_count += 1;
                    if let Some(history) = &mut sender.history {
                        history.produced.push((history.copy)(&message));
                    }

                    if let Some(destination) = message.destination {
                        let receiver = &mut self.agents[destination.index()];
                        receiver.state.queue.push_back(message);
                        if self.enable_queue_depth_metric {
                            // Deliveries become visible at the next tick.
                            receiver.record_queue_depth(next_time);
                        }
                        if let Some(active) = schedule.as_deref_mut() {
                            if receiver.state.mode == AgentMode::Reactive {
                                active.set_ready(destination.index(), true);
                            }
                        }
                    }
                }

                AgentCommandType::HaltSimulation(reason) => {
                    // The first interrupt in issue order ends the run.
                    if self.mode == SimulationMode::Running {
                        self.halt_reason = Some(reason);
                    }
                    self.mode = SimulationMode::Completed;
                }
            }
        }
    }

    /// Searches for an agent in the Simulation matching the given predicate.
    pub fn find_agent<P>(&self, predicate: P) -> Option<&SimulationAgent<M>>
    where
        P: FnMut(&&SimulationAgent<M>) -> bool,
    {
        self.agents.iter().find(predicate)
    }

    /// Checks whether all agents match the given predicate.
    pub fn all_agents<P>(&self, predicate: P) -> bool
    where
        P: FnMut(&SimulationAgent<M>) -> bool,
    {
        self.agents.iter().all(predicate)
    }

    /// Returns a slice of the Agents in the Simulation.
    #[must_use]
    pub fn agents(&self) -> &[SimulationAgent<M>] {
        self.agents.iter().as_slice()
    }

    /// Returns the current `DiscreteTime` tick for the Simulation.
    #[must_use]
    pub const fn time(&self) -> DiscreteTime {
        self.time
    }
}

/// An agent's behavior and state, borrowed for edits between runs.
type AgentParts<'a, M> = (&'a mut Box<dyn Agent<M>>, &'a mut AgentState<M>);

#[cfg(test)]
mod tests {
    use super::*;
    use agent::{periodic_consumer, periodic_producer};

    #[test]
    fn basic_periodic_test() {
        let mut simulation = Simulation::<()>::builder(Halt::AtTime(5))
            .agents(vec![
                periodic_producer("producer".to_string(), 1, "consumer".to_string()),
                periodic_consumer("consumer".to_string(), 1),
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
        let mut consumer = periodic_consumer("consumer".to_string(), 1);
        consumer.options.initial_mode = AgentMode::AsleepUntil(3);
        consumer.options.wake_mode = AgentMode::Reactive;
        let mut simulation = Simulation::<()>::builder(Halt::AtTime(5))
            .agents(vec![
                periodic_producer("producer".to_string(), 1, "consumer".to_string()),
                consumer,
            ])
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
    fn rejects_duplicate_agent_names() {
        for count in [2, 8, 9] {
            let mut agent_initializers: Vec<AgentInitializer> = (1..count)
                .map(|i| periodic_consumer(format!("worker-{i}"), 1))
                .collect();
            agent_initializers.push(periodic_consumer("worker-1".to_string(), 2));
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
