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
    AgentMode, AgentState, MessageProcessingStatus, SimulationAgent,
};
use experiment::{MasterSeed, SeedDeriver};
use message::Message;

use log::{debug, info, log_enabled, Level};
use rustc_hash::FxHashMap;
use std::cmp::Reverse;
use std::collections::{hash_map::Entry, BinaryHeap};
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Messages of room reserved in each history and queue of a small simulation.
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

    /// Maps agent names to identities in simulations too large for linear
    /// name scans; empty otherwise.
    agent_name_handle_map: FxHashMap<String, AgentId>,

    /// Identifies this agent layout for cached target identities. Clones share
    /// the layout and its token; every constructed simulation gets a new one.
    layout: u64,

    /// Source of every agent's per-tick random stream.
    seeds: SeedDeriver,
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

    /// Root of every agent's [`AgentContext::rng`] stream. Equal seeds give
    /// equal runs; the default is `MasterSeed::from_u64(0)`.
    pub seed: MasterSeed,
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
                write!(f, "duplicate agent name in simulation parameters: {name}")
            }
            Self::TooManyAgents { count } => {
                write!(f, "{count} agents exceed the limit of {MAX_AGENTS}")
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
            seed: MasterSeed::from_u64(0),
        }
    }
}

/// Per-tick inputs shared by every agent visit.
struct Tick<'a> {
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
    commands: &'a mut Vec<AgentCommand>,
    /// Source of each visited agent's random stream for this tick.
    seeds: SeedDeriver,
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
    fn new(agents: &[SimulationAgent]) -> Self {
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
    fn runnable(state: &AgentState) -> bool {
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

    /// Marks due agents ready; their state changes only at their ordered visit.
    fn wake(&mut self, time: DiscreteTime, agents: &[SimulationAgent]) {
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
fn visit_agent(
    agents: &mut [SimulationAgent],
    handle: usize,
    tick: &mut Tick<'_>,
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
            // no sleep; only the optional metrics change.
            if tick.queue_depth_metric {
                agent.record_queue_depth();
            }
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
            if tick.queue_depth_metric {
                agent.record_queue_depth();
            }
            if tick.asleep_cycles_metric {
                agent.metadata.asleep_cycle_count += 1;
            }
            return (false, None);
        }
        AgentMode::Dead => {
            if tick.queue_depth_metric {
                agent.record_queue_depth();
            }
            return (false, None);
        }
    };
    let message = agent.state.queue.pop_front();
    if tick.queue_depth_metric {
        agent.record_queue_depth();
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
        seeds: tick.seeds,
        rng: None,
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
            MessageProcessingStatus::InProgress => agent.state.queue.push_front(msg),
            MessageProcessingStatus::NoError => {
                let mut msg = msg;
                msg.completed_time = Some(tick.time);
                agent.state.consumed.push(msg);
            }
        }
    }
    if let Some(sleep_until) = requested_sleep_until {
        agent.state.mode = AgentMode::AsleepUntil(sleep_until);
    }
    (true, requested_sleep_until)
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
    /// Returns [`SimulationError::DuplicateAgentName`] when two agents share a
    /// name, or [`SimulationError::TooManyAgents`] beyond `u32::MAX` agents.
    pub fn try_new(parameters: SimulationParameters) -> Result<Self, SimulationError> {
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
        // reserving histories up front avoids repeated growth.
        let history_capacity = if initializers.len() <= LINEAR_NAME_SCAN_LIMIT {
            SMALL_SIMULATION_HISTORY
        } else {
            0
        };
        let agents: Vec<SimulationAgent> = parameters
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
                        wake_mode: agent_initializer.options.wake_mode,
                        queue,
                        consumed: Vec::with_capacity(history_capacity),
                        produced: Vec::with_capacity(history_capacity),
                    },
                }
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
            layout: NEXT_LAYOUT.fetch_add(1, Ordering::Relaxed),
            seeds: SeedDeriver::new(parameters.seed),
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
    pub fn agent(&self, id: AgentId) -> Option<&SimulationAgent> {
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
    pub fn find_by_name_mut(
        &mut self,
        name: &str,
    ) -> Option<(&mut Box<dyn Agent>, &mut AgentState)> {
        let handle = self.find_handle_by_name(name)?;
        let agent = self.agents.get_mut(handle)?;
        Some((&mut agent.agent, &mut agent.state))
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

    /// Executes ticks until the halt condition or an interrupt stops the run.
    fn run_ticks(&mut self) {
        self.mode = SimulationMode::Running;
        let mut command_buffer: Vec<AgentCommand> = Vec::new();
        // Small scans are cheaper than maintaining the schedule, and metrics need
        // every agent visited every tick, so either keeps dense scans.
        let sparse_eligible = self.agents.len() > SPARSE_SCHEDULE_MIN_AGENTS
            && !self.enable_queue_depth_metric
            && !self.enable_agent_asleep_cycles_metric;
        let mut schedule = sparse_eligible.then(|| SparseSchedule::new(&self.agents));

        while self.mode == SimulationMode::Running && !(self.halt_check)(self) {
            let Some(next_time) = self.time.checked_add(1) else {
                panic!("simulation time exceeds u64::MAX");
            };
            debug!("Running next tick of simulation at time {}", self.time);
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

            self.process_command_buffer(&mut command_buffer, schedule.as_mut());

            debug!("Finished this tick; incrementing time.");
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
        }

        self.mode = SimulationMode::Completed;
        if log_enabled!(Level::Debug) {
            self.emit_completed_simulation_debug_logging();
        }
    }

    /// Borrows the agents mutably beside the inputs every visit shares.
    fn split_tick<'a>(
        &'a mut self,
        commands: &'a mut Vec<AgentCommand>,
    ) -> (&'a mut [SimulationAgent], Tick<'a>) {
        let agent_count = self.agents.len();
        let tick = Tick {
            time: self.time,
            layout: self.layout,
            agent_count,
            scan_names: agent_count <= LINEAR_NAME_SCAN_LIMIT,
            queue_depth_metric: self.enable_queue_depth_metric,
            asleep_cycles_metric: self.enable_agent_asleep_cycles_metric,
            seeds: self.seeds,
            agent_name_handle_map: &self.agent_name_handle_map,
            commands,
        };
        (&mut self.agents, tick)
    }

    /// Visits every agent in handle order and returns how many callbacks ran.
    fn run_dense_tick(&mut self, commands: &mut Vec<AgentCommand>) -> usize {
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
        commands: &mut Vec<AgentCommand>,
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
    /// are omitted.
    #[must_use]
    pub fn calc_avg_wait_statistics(&self) -> NameStatistics<f64> {
        let mut data = NameStatistics::new();
        for agent in self
            .agents
            .iter()
            .filter(|agent| !agent.state.consumed.is_empty())
        {
            let mut sum_of_times: f64 = 0f64;
            for completed in &agent.state.consumed {
                let completed_time = completed.completed_time.unwrap_or(completed.queued_time);
                sum_of_times += if completed_time >= completed.queued_time {
                    (completed_time - completed.queued_time) as f64
                } else {
                    -((completed.queued_time - completed_time) as f64)
                };
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
    pub fn calc_queue_len_statistics(&self) -> NameStatistics<usize> {
        let mut data = NameStatistics::new();

        for agent in &self.agents {
            data.insert(agent.name.clone(), agent.state.queue.len());
        }

        data
    }

    /// Calculates the length of the consumed messages for each Agent.
    #[must_use]
    pub fn calc_consumed_len_statistics(&self) -> NameStatistics<usize> {
        let mut data = NameStatistics::new();

        for agent in &self.agents {
            data.insert(agent.name.clone(), agent.state.consumed.len());
        }

        data
    }

    /// Calculates the length of the produced messages for each Agent.
    #[must_use]
    pub fn calc_produced_len_statistics(&self) -> NameStatistics<usize> {
        let mut data = NameStatistics::new();

        for agent in &self.agents {
            data.insert(agent.name.clone(), agent.state.produced.len());
        }

        data
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
        command_buffer: &mut Vec<AgentCommand>,
        mut schedule: Option<&mut SparseSchedule>,
    ) {
        for command in command_buffer.drain(..) {
            match command.ty {
                AgentCommandType::SendMessage(message) => {
                    if let Some(destination) = message.destination {
                        let receiver = &mut self.agents[destination.index()];
                        receiver.state.queue.push_back(message.clone());
                        if let Some(active) = schedule.as_deref_mut() {
                            if receiver.state.mode == AgentMode::Reactive {
                                active.set_ready(destination.index(), true);
                            }
                        }
                    }

                    let commanding_agent = &mut self.agents[command.agent_handle];

                    commanding_agent.state.produced.push(message);
                }

                AgentCommandType::HaltSimulation(reason) => {
                    info!("Received a halt interrupt: {reason:?}");
                    self.mode = SimulationMode::Completed;
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
    use agent::{periodic_consumer, periodic_producer, poisson_distributed_producer};
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
    fn poisson_producer_emits_without_an_input_message() -> Result<(), Box<dyn std::error::Error>> {
        let mut simulation = Simulation::new(SimulationParameters {
            agent_initializers: vec![
                poisson_distributed_producer(
                    "producer".to_string(),
                    Poisson::new(1.0)?,
                    "consumer".to_string(),
                ),
                periodic_consumer("consumer".to_string(), 1),
            ],
            halt_check: |simulation| simulation.time == 1,
            ..Default::default()
        });

        simulation.run();

        assert_eq!(
            simulation.calc_produced_len_statistics().get("producer"),
            Some(&1)
        );
        assert_eq!(
            simulation.calc_queue_len_statistics().get("consumer"),
            Some(&1)
        );
        Ok(())
    }

    #[test]
    fn sleeping_reactive_agent_preserves_messages_until_wakeup() {
        let mut consumer = periodic_consumer("consumer".to_string(), 1);
        consumer.options.initial_mode = AgentMode::AsleepUntil(3);
        consumer.options.wake_mode = AgentMode::Reactive;
        let mut simulation = Simulation::new(SimulationParameters {
            agent_initializers: vec![
                periodic_producer("producer".to_string(), 1, "consumer".to_string()),
                consumer,
            ],
            halt_check: |simulation: &Simulation| simulation.time == 5,
            ..Default::default()
        });

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
            let result = Simulation::try_new(SimulationParameters {
                agent_initializers,
                ..Default::default()
            });

            assert!(
                matches!(&result, Err(SimulationError::DuplicateAgentName(name)) if name == "worker-1"),
                "{count} agents: {result:?}"
            );
        }
    }
}
