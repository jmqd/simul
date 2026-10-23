//! Seeded randomized laws for the agent engine.
//!
//! Every case is regenerated from its seed, so a failure names a seed that
//! reproduces it.
#![allow(clippy::missing_docs_in_private_items)]

use rand::{RngExt as _, SeedableRng as _};
use rand_chacha::ChaCha8Rng;
use simul::agent::{
    periodic_consumer, periodic_producer, Agent, AgentContext, AgentId, AgentInitializer,
    AgentMode, AgentOptions, MessageProcessingStatus, SimulationAgent,
};
use simul::experiment::MasterSeed;
use simul::message::Message;
use simul::{DiscreteTime, Simulation, SimulationParameters};
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

fn halt_at_time(simulation: &Simulation) -> bool {
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

    fn act(&mut self, ctx: &mut AgentContext, received: Option<&Message>) {
        // Mixing in the simulation stream makes every scheduling law also check
        // that ctx.rng() draws do not depend on how the run was executed.
        self.rng ^= ctx.rng().next_u64();
        let sends = if self.next() % 100 < self.send_percent {
            1 + self.next() % 3
        } else {
            0
        };
        for _ in 0..sends {
            let payload: Option<Arc<[u8]>> = match self.next() % 3 {
                0 => None,
                1 => Some(Arc::from([self.next().to_le_bytes()[0]])),
                _ => received.and_then(|message| message.custom_payload.clone()),
            };
            let target = self.index_below(self.names.len());
            let bytes = payload.as_deref().map(<[u8]>::to_vec);
            if self.next() % 2 == 0 {
                let name = self.names[target].clone();
                let id = ctx.agent_id(&name);
                self.events.push(Event::Resolve(name.clone(), id));
                self.events.push(Event::Sent(ctx.time(), id, bytes));
                ctx.send(&name, payload);
            } else {
                let known = target < self.names.len() - 2;
                let id = known.then(|| agent_id(target));
                self.events.push(Event::Sent(ctx.time(), id, bytes));
                ctx.send_to(agent_id(target), payload);
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

impl Agent for Chaos {
    fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
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

    fn on_tick(&mut self, ctx: &mut AgentContext) {
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
    simulation: Simulation,
    start: DiscreteTime,
    halt_at: DiscreteTime,
    /// Each agent's initial queue, addressed to that agent as the engine does.
    initial_queues: Vec<Vec<Message>>,
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
) -> AgentInitializer {
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
    let agent_initializers = (0..count)
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
                            Message::new(queued_time)
                        } else {
                            Message::new_with_payload(
                                queued_time,
                                Some(Arc::from([rng.random::<u8>()])),
                            )
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

    let simulation = Simulation::new(SimulationParameters {
        agent_initializers,
        halt_check: halt_at_time,
        starting_time: start,
        enable_queue_depth_metrics: queue_metric,
        enable_agent_asleep_cycles_metric: asleep_metric,
        seed: MasterSeed::from_u64(seed),
    });
    Case {
        simulation,
        start,
        halt_at,
        initial_queues,
    }
}

/// Everything observable except metrics.
fn snapshot(simulation: &Simulation) -> String {
    let mut out = format!("time={}\n", simulation.time());
    for agent in simulation.agents() {
        let _ = writeln!(out, "{} {:?} {:?}", agent.name, agent.agent, agent.state);
    }
    out
}

fn metrics(simulation: &Simulation) -> Vec<(Vec<usize>, DiscreteTime)> {
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
fn chaos_events(agent: &SimulationAgent) -> Option<String> {
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

type MessageKey = (
    DiscreteTime,
    Option<AgentId>,
    Option<AgentId>,
    Option<Vec<u8>>,
);

fn message_key(message: &Message) -> MessageKey {
    (
        message.queued_time,
        message.source,
        message.destination,
        message.custom_payload.as_deref().map(<[u8]>::to_vec),
    )
}

fn completed_time(seed: u64, message: &Message) -> DiscreteTime {
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
fn check_chaos_history(seed: u64, simulation: &Simulation, agent: &SimulationAgent, debug: &str) {
    // Completed on_message calls, in order, are the consumed history.
    let mut rest = debug;
    for message in &agent.state.consumed {
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
        agent.state.produced.len(),
        "seed {seed}: produced count"
    );
    let mut rest = debug;
    for message in &agent.state.produced {
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
    for seed in 0..CASES {
        let mut case = build(seed, Size::Any, false, false, true);
        run(&mut case);
        let simulation = &case.simulation;
        let agents = simulation.agents();
        let end = simulation.time();

        let mut sent: Vec<Vec<(DiscreteTime, usize, &Message)>> = vec![Vec::new(); agents.len()];
        for (handle, agent) in agents.iter().enumerate() {
            for message in &agent.state.produced {
                assert_eq!(message.source, Some(agent_id(handle)), "seed {seed}");
                assert_eq!(message.completed_time, None, "seed {seed}");
                assert!(
                    (case.start..end).contains(&message.queued_time),
                    "seed {seed}"
                );
                if let Some(destination) = message.destination {
                    assert!(destination.index() < agents.len(), "seed {seed}");
                    sent[destination.index()].push((message.queued_time, handle, message));
                }
            }
        }
        for (handle, agent) in agents.iter().enumerate() {
            let mut expected: Vec<_> = case.initial_queues[handle]
                .iter()
                .map(message_key)
                .collect();
            let mut arrivals = std::mem::take(&mut sent[handle]);
            arrivals.sort_by_key(|&(time, source, _)| (time, source));
            expected.extend(arrivals.iter().map(|&(_, _, message)| message_key(message)));
            let held: Vec<_> = agent
                .state
                .consumed
                .iter()
                .chain(&agent.state.queue)
                .map(message_key)
                .collect();
            assert_eq!(expected, held, "seed {seed}: agent {handle} history");

            for message in &agent.state.consumed {
                let completed = completed_time(seed, message);
                assert!((case.start..end).contains(&completed), "seed {seed}");
                if message.source.is_some() {
                    assert!(
                        completed > message.queued_time,
                        "seed {seed}: same-tick delivery"
                    );
                }
            }
            assert!(
                agent.state.queue.iter().all(|m| m.completed_time.is_none()),
                "seed {seed}"
            );

            if let Some(debug) = chaos_events(agent) {
                check_chaos_history(seed, simulation, agent, &debug);
            }
        }
    }
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
        let ticks = case.simulation.time() - case.start;
        for agent in case.simulation.agents() {
            let name = &agent.name;
            let depths = case
                .simulation
                .queue_depth_metrics(name)
                .unwrap_or_else(|| panic!("seed {seed}: agent {name:?} has queue depth metrics"));
            assert_eq!(
                if queue_metric { ticks } else { 0 },
                depths.len() as u64,
                "seed {seed}"
            );
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

/// Applies the same seeded edits to the agents of a halted simulation.
fn mutate_between_runs(simulation: &mut Simulation, seed: u64, round: u64) {
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
            2 => state.queue.push_back(Message::new(now)),
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
