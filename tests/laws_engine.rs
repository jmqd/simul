//! Seeded randomized laws for the agent engine.
//!
//! Every case is regenerated from its seed, so a failure names a seed that
//! reproduces it.
#![allow(clippy::missing_docs_in_private_items)]

use rand::{RngExt as _, SeedableRng as _};
use rand_chacha::ChaCha8Rng;
use simul::agent::{
    periodic_consumer, periodic_producer, Agent, AgentContext, AgentId, AgentInitializer,
    AgentMode, AgentOptions, MessageProcessingStatus, QueueDepthChange, SimulationAgent,
};
use simul::experiment::MasterSeed;
use simul::message::Message;
use simul::{DiscreteTime, Halt, Simulation};
use std::cell::{Cell, RefCell};
use std::fmt::{self, Write as _};
use std::sync::Arc;

/// Seeds per law for simulations of any size.
const CASES: u64 = 200;
/// Seeds per law for simulations large enough for sparse scheduling.
const LARGE_CASES: u64 = 100;

thread_local! {
    static HALT_AT: Cell<DiscreteTime> = const { Cell::new(0) };
    static HALT_CALLS: RefCell<Vec<DiscreteTime>> = const { RefCell::new(Vec::new()) };
    static CALLBACKS: RefCell<Vec<(DiscreteTime, usize)>> = const { RefCell::new(Vec::new()) };
}

/// The laws forward shared byte payloads, so they use them as the message type.
type Payload = Option<Arc<[u8]>>;

fn halt_at_time(simulation: &Simulation<Payload>) -> bool {
    HALT_CALLS.with(|calls| calls.borrow_mut().push(simulation.time()));
    simulation.time() >= HALT_AT.with(Cell::get)
}

fn take_halt_calls() -> Vec<DiscreteTime> {
    HALT_CALLS.with(|calls| std::mem::take(&mut *calls.borrow_mut()))
}

fn agent_id(handle: usize) -> AgentId {
    AgentId::new(u32::try_from(handle).unwrap_or_else(|_| panic!("handle {handle} fits u32")))
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Tick(DiscreteTime),
    Message {
        time: DiscreteTime,
        queued_time: DiscreteTime,
        source: Option<AgentId>,
        in_progress: bool,
    },
    Resolve(String, Option<AgentId>),
    Halt(DiscreteTime),
    Sent(DiscreteTime, Option<AgentId>, Option<Vec<u8>>),
}

/// Deterministic agent driven by its own splitmix state.
#[derive(Clone)]
struct Chaos {
    rng: u64,
    /// Real names followed by names that match no agent.
    names: Arc<[String]>,
    send_percent: u64,
    sleep_percent: u64,
    max_sleep: u64,
    in_progress_percent: u64,
    halt_per_million: u64,
    events: Vec<Event>,
    /// The tick of each send, in send order; a delayed send's `Sent` event
    /// and message carry the tick before it arrives instead.
    send_ticks: Vec<DiscreteTime>,
}

impl fmt::Debug for Chaos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Chaos")
            .field("rng", &self.rng)
            .field("events", &self.events)
            .finish_non_exhaustive()
    }
}

impl Chaos {
    const fn next(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn index_below(&mut self, len: usize) -> usize {
        let index = self.next() % len as u64;
        usize::try_from(index).unwrap_or_else(|_| panic!("{index} is below a usize length"))
    }

    fn act(&mut self, ctx: &mut AgentContext<'_, Payload>, received: Option<&Message<Payload>>) {
        // Mixing in the simulation stream makes every scheduling law also check
        // that ctx.rng() draws do not depend on how the run was executed.
        self.rng ^= ctx.rng().next_u64();
        let sends = if self.next() % 100 < self.send_percent {
            1 + self.next() % 3
        } else {
            0
        };
        for _ in 0..sends {
            let payload: Payload = match self.next() % 3 {
                0 => None,
                1 => Some(Arc::from([self.next().to_le_bytes()[0]])),
                _ => received.and_then(|message| message.payload.clone()),
            };
            let target = self.index_below(self.names.len());
            let bytes = payload.as_deref().map(<[u8]>::to_vec);
            let known = target < self.names.len() - 2;
            self.send_ticks.push(ctx.time());
            match self.next() % 3 {
                0 => {
                    let name = self.names[target].clone();
                    let id = ctx.agent_id(&name);
                    self.events.push(Event::Resolve(name.clone(), id));
                    self.events.push(Event::Sent(ctx.time(), id, bytes));
                    ctx.send(&name, payload);
                }
                1 => {
                    let id = known.then(|| agent_id(target));
                    self.events.push(Event::Sent(ctx.time(), id, bytes));
                    ctx.send_to(agent_id(target), payload);
                }
                _ => {
                    let delay = 1 + self.next() % 20;
                    let id = known.then(|| agent_id(target));
                    self.events
                        .push(Event::Sent(ctx.time() + delay - 1, id, bytes));
                    ctx.send_after(delay, agent_id(target), payload);
                }
            }
        }
        if self.next() % 100 < self.sleep_percent {
            let ticks = self.next() % (self.max_sleep + 1);
            ctx.sleep_for(ticks);
        }
        if self.next() % 1_000_000 < self.halt_per_million {
            self.events.push(Event::Halt(ctx.time()));
            ctx.send_halt_interrupt("chaos");
        }
    }
}

impl Agent<Payload> for Chaos {
    fn on_message(&mut self, ctx: &mut AgentContext<'_, Payload>, msg: &Message<Payload>) {
        let in_progress = self.next() % 100 < self.in_progress_percent;
        CALLBACKS.with(|calls| calls.borrow_mut().push((ctx.time(), ctx.id().index())));
        self.events.push(Event::Message {
            time: ctx.time(),
            queued_time: msg.queued_time,
            source: msg.source,
            in_progress,
        });
        if in_progress {
            ctx.set_processing_status(MessageProcessingStatus::InProgress);
        }
        self.act(ctx, Some(msg));
    }

    fn on_tick(&mut self, ctx: &mut AgentContext<'_, Payload>) {
        self.events.push(Event::Tick(ctx.time()));
        CALLBACKS.with(|calls| calls.borrow_mut().push((ctx.time(), ctx.id().index())));
        self.act(ctx, None);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Size {
    /// More than 64 agents, eligible for sparse scheduling.
    Large,
    /// Any size, including the linear name-scan range.
    Any,
}

struct Case {
    simulation: Simulation<Payload>,
    start: DiscreteTime,
    halt_at: DiscreteTime,
    /// Each agent's initial queue, addressed to that agent as the engine does.
    initial_queues: Vec<Vec<Message<Payload>>>,
}

/// A wake time, sometimes already in the past.
fn deadline(rng: &mut ChaCha8Rng, start: DiscreteTime, span: DiscreteTime) -> DiscreteTime {
    if rng.random_bool(0.2) {
        start.saturating_sub(rng.random_range(0..10))
    } else {
        start + rng.random_range(0..span)
    }
}

fn random_mode(rng: &mut ChaCha8Rng, start: DiscreteTime) -> AgentMode {
    match rng.random_range(0..100) {
        0..25 => AgentMode::Proactive,
        25..75 => AgentMode::Reactive,
        75..85 => AgentMode::Dead,
        _ => AgentMode::AsleepUntil(deadline(rng, start, 150)),
    }
}

fn chaos_agent(
    rng: &mut ChaCha8Rng,
    names: &Arc<[String]>,
    name: String,
    start: DiscreteTime,
    active_percent: u64,
    halt_per_million: u64,
) -> AgentInitializer<Payload> {
    let initial_mode = if rng.random_range(0..100) < active_percent {
        AgentMode::Proactive
    } else {
        match rng.random_range(0..100) {
            0..45 => AgentMode::Reactive,
            45..80 => AgentMode::AsleepUntil(deadline(rng, start, 100)),
            _ => AgentMode::Dead,
        }
    };
    AgentInitializer {
        agent: Box::new(Chaos {
            rng: rng.random(),
            names: names.clone(),
            send_percent: rng.random_range(0..=100),
            sleep_percent: rng.random_range(30..=100),
            max_sleep: rng.random_range(0..=60),
            in_progress_percent: rng.random_range(0..=40),
            halt_per_million,
            events: Vec::new(),
            send_ticks: Vec::new(),
        }),
        options: AgentOptions {
            initial_mode,
            wake_mode: random_mode(rng, start),
            name,
            ..Default::default()
        },
    }
}

fn build(seed: u64, size: Size, queue_metric: bool, asleep_metric: bool, halts: bool) -> Case {
    build_case(
        seed,
        size,
        queue_metric,
        asleep_metric,
        halts,
        HaltStyle::Predicate,
        History::Recorded,
    )
}

/// Whether a case keeps every consumed and produced message.
#[derive(Clone, Copy)]
enum History {
    Recorded,
    Off,
}

/// How a case expresses its halt time.
#[derive(Clone, Copy)]
enum HaltStyle {
    /// A predicate checked every tick, so no tick is skipped.
    Predicate,
    /// `Halt::AtTime`, which lets the engine skip quiet ticks.
    AtTime,
}

impl HaltStyle {
    fn halt(self, halt_at: DiscreteTime) -> Halt<Payload> {
        match self {
            Self::Predicate => Halt::when(halt_at_time),
            Self::AtTime => Halt::AtTime(halt_at),
        }
    }
}

fn build_case(
    seed: u64,
    size: Size,
    queue_metric: bool,
    asleep_metric: bool,
    halts: bool,
    style: HaltStyle,
    history: History,
) -> Case {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let count: usize = match (size, rng.random_range(0..4)) {
        (Size::Large, _) | (Size::Any, 0 | 1) => rng.random_range(65..=300),
        (Size::Any, 2) => rng.random_range(1..=8),
        _ => rng.random_range(9..=64),
    };
    let start: DiscreteTime = if rng.random_bool(0.5) {
        0
    } else {
        rng.random_range(0..1000)
    };
    let halt_at = start + rng.random_range(0..=200);
    let active_percent: u64 = rng.random_range(0..=30);
    let halt_per_million = if halts && rng.random_bool(0.3) {
        300
    } else {
        0
    };

    // Shuffled names keep names and identities from coinciding.
    let mut ids: Vec<usize> = (0..count).collect();
    for i in (1..count).rev() {
        ids.swap(i, rng.random_range(0..=i));
    }
    let mut names: Vec<String> = ids.iter().map(|id| format!("a{id}")).collect();
    names.push("ghost".to_string());
    names.push(String::new());
    let names: Arc<[String]> = names.into();

    let mut initial_queues = Vec::with_capacity(count);
    let agent_initializers: Vec<_> = (0..count)
        .map(|handle| {
            let name = names[handle].clone();
            let target = names[rng.random_range(0..names.len())].clone();
            let mut initializer = match rng.random_range(0..100) {
                0..20 => {
                    let mut producer = periodic_producer(name, rng.random_range(1..=40), target);
                    if rng.random_range(0..100) >= active_percent {
                        producer.options.initial_mode =
                            AgentMode::AsleepUntil(deadline(&mut rng, start, 100));
                    }
                    producer
                }
                20..45 => periodic_consumer(name, rng.random_range(0..=10)),
                _ => chaos_agent(
                    &mut rng,
                    &names,
                    name,
                    start,
                    active_percent,
                    halt_per_million,
                ),
            };
            if rng.random_bool(0.3) {
                for _ in 0..rng.random_range(1..=3) {
                    let queued_time = start.saturating_sub(rng.random_range(0..5));
                    initializer
                        .options
                        .initial_queue
                        .push_back(if rng.random_bool(0.5) {
                            Message::new(queued_time, None)
                        } else {
                            Message::new(queued_time, Some(Arc::from([rng.random::<u8>()])))
                        });
                }
            }
            let owner = agent_id(handle);
            initial_queues.push(
                initializer
                    .options
                    .initial_queue
                    .iter()
                    .cloned()
                    .map(|mut message| {
                        message.destination = Some(owner);
                        message
                    })
                    .collect(),
            );
            initializer
        })
        .collect();

    let simulation = Simulation::builder(style.halt(halt_at))
        .agents(agent_initializers)
        .starting_time(start)
        .queue_depth_metrics(queue_metric)
        .asleep_cycles_metric(asleep_metric)
        .message_history(matches!(history, History::Recorded))
        .seed(MasterSeed::from_u64(seed))
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    Case {
        simulation,
        start,
        halt_at,
        initial_queues,
    }
}

/// Everything observable except metrics and recorded messages.
fn state_snapshot(simulation: &Simulation<Payload>) -> String {
    let mut out = format!("time={}\n", simulation.time());
    let waits = simulation.calc_avg_wait_statistics();
    for agent in simulation.agents() {
        let _ = writeln!(
            out,
            "{} {:?} {:?} consumed={} produced={} wait={:?}",
            agent.name,
            agent.agent,
            agent.state,
            agent.consumed_count(),
            agent.produced_count(),
            waits.get(&agent.name)
        );
    }
    let _ = writeln!(
        out,
        "in flight {:?}",
        simulation.in_flight().collect::<Vec<_>>()
    );
    out
}

/// Everything observable except metrics.
fn snapshot(simulation: &Simulation<Payload>) -> String {
    let mut out = state_snapshot(simulation);
    for agent in simulation.agents() {
        let _ = writeln!(out, "{:?} {:?}", agent.consumed(), agent.produced());
    }
    out
}

fn metrics(simulation: &Simulation<Payload>) -> Vec<(Vec<QueueDepthChange>, DiscreteTime)> {
    simulation
        .agents()
        .iter()
        .map(|agent| {
            let name = &agent.name;
            (
                simulation
                    .queue_depth_metrics(name)
                    .unwrap_or_else(|| panic!("agent {name:?} has queue depth metrics"))
                    .to_vec(),
                simulation
                    .asleep_cycle_count(name)
                    .unwrap_or_else(|| panic!("agent {name:?} has an asleep cycle count")),
            )
        })
        .collect()
}

/// Halt-check times and the `(time, handle)` of every `Chaos` callback, in order.
type Trace = (Vec<DiscreteTime>, Vec<(DiscreteTime, usize)>);

fn run(case: &mut Case) -> Trace {
    HALT_AT.with(|halt| halt.set(case.halt_at));
    take_halt_calls();
    CALLBACKS.with(|calls| calls.borrow_mut().clear());
    case.simulation.run();
    (
        take_halt_calls(),
        CALLBACKS.with(|calls| std::mem::take(&mut *calls.borrow_mut())),
    )
}

/// Parses the `Chaos` events out of an agent's Debug output.
fn chaos_events(agent: &SimulationAgent<Payload>) -> Option<String> {
    let debug = format!("{:?}", agent.agent);
    debug.starts_with("Chaos").then_some(debug)
}

/// Parses the time that follows each `marker` in `debug`, up to `end`.
fn times_after<'a>(
    debug: &'a str,
    marker: &'a str,
    end: char,
) -> impl Iterator<Item = DiscreteTime> + 'a {
    debug.split(marker).skip(1).map(move |piece| {
        let Some(at) = piece.find(end) else {
            panic!("unterminated {marker:?} in {debug}");
        };
        piece[..at]
            .parse()
            .unwrap_or_else(|error| panic!("bad time after {marker:?} in {piece}: {error}"))
    })
}

/// Returns what follows the first `needle` in `rest`.
fn skip_past<'a>(seed: u64, rest: &'a str, needle: &str) -> &'a str {
    let at = rest
        .find(needle)
        .unwrap_or_else(|| panic!("seed {seed}: missing {needle} in {rest}"));
    &rest[at + needle.len()..]
}

/// `(queued_time, send tick, sender handle)`: messages arriving in one tick
/// come in send order, by send tick, then handle, then call order, which a
/// stable sort on this key keeps.
type ArrivalKey = (DiscreteTime, DiscreteTime, usize);

type MessageKey = (
    DiscreteTime,
    Option<AgentId>,
    Option<AgentId>,
    Option<Vec<u8>>,
);

fn message_key(message: &Message<Payload>) -> MessageKey {
    (
        message.queued_time,
        message.source,
        message.destination,
        message.payload.as_deref().map(<[u8]>::to_vec),
    )
}

fn completed_time(seed: u64, message: &Message<Payload>) -> DiscreteTime {
    message
        .completed_time
        .unwrap_or_else(|| panic!("seed {seed}: consumed message has no completion time"))
}

#[test]
fn sparse_and_dense_runs_are_observably_equivalent() {
    for seed in 0..LARGE_CASES {
        let mut sparse = build(seed, Size::Large, false, false, true);
        let sparse_calls = run(&mut sparse);
        let expected = snapshot(&sparse.simulation);
        for (queue_metric, asleep_metric) in [(true, false), (false, true), (true, true)] {
            let mut dense = build(seed, Size::Large, queue_metric, asleep_metric, true);
            let dense_calls = run(&mut dense);
            assert_eq!(
                sparse_calls, dense_calls,
                "seed {seed}: halt checks or callback order differ"
            );
            assert!(
                expected == snapshot(&dense.simulation),
                "seed {seed}: state differs with metrics ({queue_metric}, {asleep_metric})"
            );
        }
    }
}

#[test]
fn skipping_quiet_ticks_matches_running_every_tick() {
    let mut quiet_ticks = 0;
    for seed in 0..CASES {
        let queue_metric = seed % 2 == 0;
        let mut every_tick = build_case(
            seed,
            Size::Any,
            queue_metric,
            false,
            true,
            HaltStyle::Predicate,
            History::Recorded,
        );
        let (_, every_tick_callbacks) = run(&mut every_tick);
        let mut skipping = build_case(
            seed,
            Size::Any,
            queue_metric,
            false,
            true,
            HaltStyle::AtTime,
            History::Recorded,
        );
        let (_, skipping_callbacks) = run(&mut skipping);
        assert_eq!(
            every_tick_callbacks, skipping_callbacks,
            "seed {seed}: callback order"
        );
        assert_eq!(
            every_tick.simulation.time(),
            skipping.simulation.time(),
            "seed {seed}"
        );
        assert!(
            snapshot(&every_tick.simulation) == snapshot(&skipping.simulation),
            "seed {seed}: state differs"
        );
        assert_eq!(
            metrics(&every_tick.simulation),
            metrics(&skipping.simulation),
            "seed {seed}"
        );
        let mut active: Vec<_> = every_tick_callbacks.iter().map(|&(time, _)| time).collect();
        active.dedup();
        quiet_ticks += (every_tick.simulation.time() - every_tick.start)
            .saturating_sub(u64::try_from(active.len()).unwrap_or(u64::MAX));
    }
    assert!(
        quiet_ticks > 1_000,
        "cases must contain quiet ticks to skip"
    );
}

#[test]
fn recording_message_history_does_not_change_the_run() {
    for seed in 0..CASES {
        let metric = seed % 2 == 0;
        let case = |history| {
            build_case(
                seed,
                Size::Any,
                metric,
                metric,
                true,
                HaltStyle::Predicate,
                history,
            )
        };
        let mut recorded = case(History::Recorded);
        let mut off = case(History::Off);
        assert_eq!(run(&mut recorded), run(&mut off), "seed {seed}: trace");
        assert!(
            state_snapshot(&recorded.simulation) == state_snapshot(&off.simulation),
            "seed {seed}: state differs"
        );
        assert_eq!(
            metrics(&recorded.simulation),
            metrics(&off.simulation),
            "seed {seed}"
        );
    }
}

#[test]
fn running_a_clone_matches_running_the_original() {
    for seed in 0..CASES {
        let metric = seed % 2 == 0;
        let mut original = build(seed, Size::Any, metric, metric, true);
        let mut copy = Case {
            simulation: original.simulation.clone(),
            start: original.start,
            halt_at: original.halt_at,
            initial_queues: Vec::new(),
        };
        let original_calls = run(&mut original);
        let copy_calls = run(&mut copy);
        assert_eq!(original_calls, copy_calls, "seed {seed}");
        assert!(
            snapshot(&original.simulation) == snapshot(&copy.simulation),
            "seed {seed}"
        );
        assert_eq!(
            metrics(&original.simulation),
            metrics(&copy.simulation),
            "seed {seed}"
        );

        // Cloning a finished run and resuming both also agrees.
        let mut resumed_copy = Case {
            simulation: original.simulation.clone(),
            start: original.start,
            halt_at: original.halt_at + 50,
            initial_queues: Vec::new(),
        };
        original.halt_at += 50;
        let original_calls = run(&mut original);
        let copy_calls = run(&mut resumed_copy);
        assert_eq!(original_calls, copy_calls, "seed {seed} resumed");
        assert!(
            snapshot(&original.simulation) == snapshot(&resumed_copy.simulation),
            "seed {seed} resumed"
        );
        assert_eq!(
            metrics(&original.simulation),
            metrics(&resumed_copy.simulation),
            "seed {seed} resumed"
        );
    }
}

#[test]
fn segmented_runs_match_one_run_without_interrupts() {
    for seed in 0..CASES {
        let metric = seed % 3 == 0;
        let mut whole = build(seed, Size::Any, metric, metric, false);
        let mut pieces = build(seed, Size::Any, metric, metric, false);
        let full = whole.halt_at;
        run(&mut whole);
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5);
        let mut at = pieces.start;
        while at < full {
            at = rng.random_range(at + 1..=full);
            pieces.halt_at = at;
            run(&mut pieces);
        }
        assert!(
            snapshot(&whole.simulation) == snapshot(&pieces.simulation),
            "seed {seed}"
        );
        assert_eq!(
            metrics(&whole.simulation),
            metrics(&pieces.simulation),
            "seed {seed}"
        );
    }
}

/// Checks a `Chaos` agent's event log against its consumed and produced history.
fn check_chaos_history(
    seed: u64,
    simulation: &Simulation<Payload>,
    agent: &SimulationAgent<Payload>,
    debug: &str,
) {
    // Completed on_message calls, in order, are the consumed history.
    let mut rest = debug;
    for message in agent.consumed() {
        let completion = format!(
            "Message {{ time: {}, queued_time: {}, source: {:?}, in_progress: false }}",
            completed_time(seed, message),
            message.queued_time,
            message.source
        );
        rest = skip_past(seed, rest, &completion);
    }
    assert!(
        !rest.contains("in_progress: false"),
        "seed {seed}: extra completion"
    );
    let names = simulation.agents().iter().map(|a| a.name.as_str());
    for name in names.chain(["ghost", ""]) {
        let resolved = format!("Resolve({name:?}, {:?})", simulation.agent_id(name));
        let prefix = format!("Resolve({name:?}, ");
        let mentions = debug.matches(prefix.as_str()).count();
        assert_eq!(
            mentions,
            debug.matches(resolved.as_str()).count(),
            "seed {seed}"
        );
    }
    // Every send, known or not, is the produced history in call order.
    assert_eq!(
        debug.matches("Sent(").count(),
        agent.produced().len(),
        "seed {seed}: produced count"
    );
    let mut rest = debug;
    for message in agent.produced() {
        let (time, _, destination, payload) = message_key(message);
        rest = skip_past(
            seed,
            rest,
            &format!("{:?}", Event::Sent(time, destination, payload)),
        );
    }
}

#[test]
fn messages_are_conserved_and_delivered_in_issue_order() {
    let mut halted_in_flight = 0;
    for seed in 0..CASES {
        let mut case = build(seed, Size::Any, false, false, true);
        run(&mut case);
        let simulation = &case.simulation;
        let agents = simulation.agents();
        let end = simulation.time();

        let mut sent: Vec<Vec<(ArrivalKey, &Message<Payload>)>> = vec![Vec::new(); agents.len()];
        let mut in_flight = Vec::new();
        for (handle, agent) in agents.iter().enumerate() {
            let send_ticks: Vec<DiscreteTime> = agent.downcast_ref::<Chaos>().map_or_else(
                || agent.produced().iter().map(|m| m.queued_time).collect(),
                |chaos| chaos.send_ticks.clone(),
            );
            assert_eq!(send_ticks.len(), agent.produced().len(), "seed {seed}");
            for (message, &sent_at) in agent.produced().iter().zip(&send_ticks) {
                assert_eq!(message.source, Some(agent_id(handle)), "seed {seed}");
                assert_eq!(message.completed_time, None, "seed {seed}");
                assert!((case.start..end).contains(&sent_at), "seed {seed}");
                assert!(message.queued_time >= sent_at, "seed {seed}");
                if let Some(destination) = message.destination {
                    assert!(destination.index() < agents.len(), "seed {seed}");
                    if message.queued_time < end {
                        let key = (message.queued_time, sent_at, handle);
                        sent[destination.index()].push((key, message));
                    } else {
                        in_flight.push(message_key(message));
                    }
                }
            }
        }
        let mut still_in_flight: Vec<_> = simulation.in_flight().map(message_key).collect();
        in_flight.sort();
        halted_in_flight += in_flight.len();
        still_in_flight.sort();
        assert_eq!(in_flight, still_in_flight, "seed {seed}: in flight");
        for (handle, agent) in agents.iter().enumerate() {
            let mut expected: Vec<_> = case.initial_queues[handle]
                .iter()
                .map(message_key)
                .collect();
            let mut arrivals = std::mem::take(&mut sent[handle]);
            arrivals.sort_by_key(|&(key, _)| key);
            expected.extend(arrivals.iter().map(|&(_, message)| message_key(message)));
            let held: Vec<_> = agent
                .consumed()
                .iter()
                .chain(&agent.state.queue)
                .map(message_key)
                .collect();
            assert_eq!(expected, held, "seed {seed}: agent {handle} history");

            let mut wait = 0.0;
            for message in agent.consumed() {
                let completed = completed_time(seed, message);
                assert!((case.start..end).contains(&completed), "seed {seed}");
                if message.source.is_some() {
                    assert!(
                        completed > message.queued_time,
                        "seed {seed}: same-tick delivery"
                    );
                }
                wait += if completed >= message.queued_time {
                    (completed - message.queued_time) as f64
                } else {
                    -((message.queued_time - completed) as f64)
                };
            }
            // Counts and waits are kept without history, so they must match it.
            let consumed = agent.consumed().len();
            assert_eq!(
                u64::try_from(consumed).ok(),
                Some(agent.consumed_count()),
                "seed {seed}: agent {handle} consumed count"
            );
            assert_eq!(
                u64::try_from(agent.produced().len()).ok(),
                Some(agent.produced_count()),
                "seed {seed}: agent {handle} produced count"
            );
            assert_eq!(
                (consumed > 0).then(|| (wait / consumed as f64).to_bits()),
                simulation
                    .calc_avg_wait_statistics()
                    .get(&agent.name)
                    .map(|mean| mean.to_bits()),
                "seed {seed}: agent {handle} mean wait"
            );
            assert!(
                agent.state.queue.iter().all(|m| m.completed_time.is_none()),
                "seed {seed}"
            );

            if let Some(debug) = chaos_events(agent) {
                check_chaos_history(seed, simulation, agent, &debug);
            }
        }
    }
    assert!(
        halted_in_flight > 50,
        "cases must halt with delayed messages still in flight"
    );
}

#[test]
fn time_advances_per_tick_and_halts_at_first_true_check_or_interrupt() {
    for seed in 0..CASES {
        let metric = seed % 2 == 1;
        let mut case = build(seed, Size::Any, metric, metric, true);
        let (calls, callbacks) = run(&mut case);
        // Callbacks run in (time, handle) order; an agent gets at most a message
        // and a tick per tick.
        assert!(
            callbacks.windows(2).all(|pair| pair[0] <= pair[1])
                && callbacks.windows(3).all(|triple| triple[0] != triple[2]),
            "seed {seed}: callback order"
        );
        let simulation = &case.simulation;

        let mut first_interrupt: Option<DiscreteTime> = None;
        let mut callback_times: Vec<Vec<DiscreteTime>> = Vec::new();
        for agent in simulation.agents() {
            let Some(debug) = chaos_events(agent) else {
                continue;
            };
            first_interrupt = first_interrupt
                .into_iter()
                .chain(times_after(&debug, "Halt(", ')'))
                .min();
            // Each kind of callback runs at most once per agent per tick.
            callback_times.push(times_after(&debug, "Tick(", ')').collect());
            callback_times.push(times_after(&debug, "Message { time: ", ',').collect());
        }

        let (end, expected_calls): (DiscreteTime, Vec<DiscreteTime>) = match first_interrupt {
            Some(tick) if tick < case.halt_at => (tick + 1, (case.start..=tick).collect()),
            _ => (case.halt_at, (case.start..=case.halt_at).collect()),
        };
        assert_eq!(end, simulation.time(), "seed {seed}");
        assert_eq!(expected_calls, calls, "seed {seed}");
        for mut times in callback_times {
            let len = times.len();
            times.sort_unstable();
            times.dedup();
            assert_eq!(
                len,
                times.len(),
                "seed {seed}: repeated callback in one tick"
            );
            assert!(
                times.iter().all(|t| (case.start..end).contains(t)),
                "seed {seed}"
            );
        }
    }
}

#[test]
fn metrics_cover_every_tick_run() {
    for seed in 0..CASES {
        let (queue_metric, asleep_metric) = match seed % 4 {
            0 => (true, true),
            1 => (true, false),
            2 => (false, true),
            _ => (false, false),
        };
        let mut case = build(seed, Size::Any, queue_metric, asleep_metric, true);
        run(&mut case);
        let end = case.simulation.time();
        let ticks = end - case.start;
        for agent in case.simulation.agents() {
            let name = &agent.name;
            let changes = case
                .simulation
                .queue_depth_metrics(name)
                .unwrap_or_else(|| panic!("seed {seed}: agent {name:?} has queue depth metrics"));
            if queue_metric {
                // Changes start at the first tick, move forward in time within
                // the run, always change the depth, and end at the final depth.
                assert_eq!(
                    Some(case.start),
                    changes.first().map(|c| c.time),
                    "seed {seed}"
                );
                assert!(
                    changes
                        .windows(2)
                        .all(|pair| pair[0].time < pair[1].time && pair[0].depth != pair[1].depth),
                    "seed {seed}: {changes:?}"
                );
                assert!(changes.iter().all(|c| c.time <= end), "seed {seed}");
                assert_eq!(
                    Some(agent.state.queue.len()),
                    changes.last().map(|c| c.depth),
                    "seed {seed}"
                );
            } else {
                assert!(changes.is_empty(), "seed {seed}");
            }
            let asleep = case
                .simulation
                .asleep_cycle_count(name)
                .unwrap_or_else(|| panic!("seed {seed}: agent {name:?} has an asleep cycle count"));
            assert!(asleep <= ticks, "seed {seed}");
            if !asleep_metric {
                assert_eq!(0, asleep, "seed {seed}");
            }
        }
    }
}

#[test]
fn queue_depth_metrics_agree_between_sparse_and_dense_scheduling() {
    for seed in 0..LARGE_CASES {
        // The asleep metric forces dense scans; the queue metric alone does not.
        let mut sparse = build(seed, Size::Large, true, false, true);
        let mut dense = build(seed, Size::Large, true, true, true);
        run(&mut sparse);
        run(&mut dense);
        let queue_changes = |case: &Case| -> Vec<Vec<QueueDepthChange>> {
            metrics(&case.simulation)
                .into_iter()
                .map(|(changes, _)| changes)
                .collect()
        };
        assert_eq!(queue_changes(&dense), queue_changes(&sparse), "seed {seed}");
    }
}

/// Applies the same seeded edits to the agents of a halted simulation.
fn mutate_between_runs(simulation: &mut Simulation<Payload>, seed: u64, round: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_mul(31).wrapping_add(round));
    let now = simulation.time();
    let names: Vec<String> = simulation.agents().iter().map(|a| a.name.clone()).collect();
    for _ in 0..rng.random_range(0..=names.len().min(40)) {
        let name = &names[rng.random_range(0..names.len())];
        let target = names[rng.random_range(0..names.len())].clone();
        let (behavior, state) = simulation
            .find_by_name_mut(name)
            .unwrap_or_else(|| panic!("seed {seed}: agent {name:?} exists"));
        match rng.random_range(0..6) {
            0 => state.mode = random_mode(&mut rng, now),
            1 => state.wake_mode = random_mode(&mut rng, now),
            2 => state.queue.push_back(Message::new(now, None)),
            3 => state.queue.clear(),
            4 => {
                *behavior =
                    periodic_producer("ignored".to_string(), rng.random_range(1..=20), target)
                        .agent;
                state.mode = AgentMode::Proactive;
            }
            _ => {
                let mode = AgentMode::AsleepUntil(now + rng.random_range(0..5));
                state.mode = mode;
                state.wake_mode = if rng.random_bool(0.5) {
                    mode
                } else {
                    AgentMode::Reactive
                };
            }
        }
    }
}

#[test]
fn sparse_and_dense_agree_across_runs_with_edits_between() {
    for seed in 0..LARGE_CASES {
        let mut sparse = build(seed, Size::Large, false, false, true);
        let mut dense = build(seed, Size::Large, true, true, true);
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5A5A);
        let mut at = sparse.halt_at;
        for round in 0..4 {
            sparse.halt_at = at;
            dense.halt_at = at;
            let sparse_calls = run(&mut sparse);
            let dense_calls = run(&mut dense);
            assert_eq!(sparse_calls, dense_calls, "seed {seed} round {round}");
            assert!(
                snapshot(&sparse.simulation) == snapshot(&dense.simulation),
                "seed {seed} round {round}"
            );
            mutate_between_runs(&mut sparse.simulation, seed, round);
            mutate_between_runs(&mut dense.simulation, seed, round);
            at = sparse.simulation.time().max(at) + rng.random_range(1..=60);
        }
    }
}
