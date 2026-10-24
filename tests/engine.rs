#![allow(clippy::missing_docs_in_private_items)]

use simul::agent::{
    periodic_consumer, periodic_producer, Agent, AgentContext, AgentId, AgentInitializer,
    AgentMode, AgentOptions, MessageProcessingStatus, QueueDepthChange, SimulationAgent,
};
use simul::message::Message;
use simul::{Halt, Simulation, SimulationMode};
use std::sync::Arc;

fn proactive_agent<M>(name: &str, agent: impl Agent<M> + 'static) -> AgentInitializer<M> {
    proactive_agent_boxed(name, Box::new(agent))
}

fn proactive_agent_boxed<M>(name: &str, agent: Box<dyn Agent<M>>) -> AgentInitializer<M> {
    AgentInitializer {
        agent,
        options: AgentOptions {
            initial_mode: AgentMode::Proactive,
            wake_mode: AgentMode::Proactive,
            name: name.to_string(),
            ..Default::default()
        },
    }
}

#[derive(Clone, Debug)]
struct HaltImmediately;

impl Agent for HaltImmediately {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        ctx.send_halt_interrupt("done");
    }
}

/// Sends one message each to the previous agent, itself, and the next agent.
#[derive(Clone, Debug)]
struct NeighborSender {
    previous: String,
    next: String,
}

impl Agent for NeighborSender {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        let own = ctx.name().to_string();
        for target in [&self.previous, &own, &self.next] {
            ctx.send(target, ());
        }
    }
}

#[test]
fn messages_route_to_lower_own_and_higher_handles() {
    for count in [3_u32, 8, 9] {
        let name = |handle: u32| format!("agent-{handle}");
        let neighbors = |handle: u32| [(handle + count - 1) % count, handle, (handle + 1) % count];
        let mut simulation = Simulation::builder(Halt::AtTime(1))
            .agents((0..count).map(|handle| {
                let [previous, _, next] = neighbors(handle);
                proactive_agent(
                    &name(handle),
                    NeighborSender {
                        previous: name(previous),
                        next: name(next),
                    },
                )
            }))
            .message_history(true)
            .build()
            .unwrap_or_else(|error| panic!("{error}"));
        simulation.run();

        for handle in 0..count {
            let id = AgentId::new(handle);
            assert_eq!(Some(id), simulation.agent_id(&name(handle)));
            let Some(agent) = simulation.agent(id) else {
                panic!("agent-{handle} must exist");
            };
            let destinations: Vec<Option<AgentId>> = agent
                .produced()
                .iter()
                .map(|message| message.destination)
                .collect();
            assert_eq!(
                neighbors(handle).map(|neighbor| Some(AgentId::new(neighbor))),
                destinations.as_slice(),
                "{count} agents, sender agent-{handle}"
            );

            let mut senders = neighbors(handle);
            senders.sort_unstable();
            let sources: Vec<Option<AgentId>> = agent
                .state
                .queue
                .iter()
                .map(|message| message.source)
                .collect();
            assert_eq!(
                senders.map(|sender| Some(AgentId::new(sender))),
                sources.as_slice(),
                "{count} agents, receiver agent-{handle}"
            );
        }
    }
}

/// Sends one message to every identity through one past the last agent.
#[derive(Clone, Debug)]
struct IdentitySender {
    count: u32,
}

impl Agent for IdentitySender {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        for index in 0..=self.count {
            ctx.send_to(AgentId::new(index), ());
        }
    }
}

#[test]
fn send_to_delivers_by_identity_and_records_unknown_identities() {
    for count in [3_u32, 9] {
        let mut simulation = Simulation::builder(Halt::AtTime(1))
            .agents((0..count).map(|handle| {
                proactive_agent(&format!("agent-{handle}"), IdentitySender { count })
            }))
            .message_history(true)
            .build()
            .unwrap_or_else(|error| panic!("{error}"));
        simulation.run();

        let expected: Vec<Option<AgentId>> = (0..count)
            .map(|id| Some(AgentId::new(id)))
            .chain([None])
            .collect();
        for agent in simulation.agents() {
            let destinations: Vec<Option<AgentId>> = agent
                .produced()
                .iter()
                .map(|message| message.destination)
                .collect();
            assert_eq!(expected, destinations, "{count} agents, {}", agent.name);
            assert_eq!(
                expected[..expected.len() - 1],
                agent
                    .state
                    .queue
                    .iter()
                    .map(|message| message.source)
                    .collect::<Vec<_>>(),
                "{count} agents, {}",
                agent.name
            );
        }
    }
}

#[test]
fn halt_interrupt_stops_simulation() {
    let mut simulation = Simulation::builder(Halt::Never)
        .agents(vec![proactive_agent("halter", HaltImmediately)])
        .build()
        .unwrap_or_else(|error| panic!("{error}"));

    simulation.run();

    assert_eq!(simulation.time(), 1);
}

#[derive(Clone, Debug)]
struct MissingTargetSender;

impl Agent for MissingTargetSender {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        ctx.send("missing", ());
    }
}

#[test]
fn unknown_destination_records_produced_message_without_delivery() {
    let mut simulation = Simulation::builder(Halt::AtTime(1))
        .agents(vec![proactive_agent("sender", MissingTargetSender)])
        .message_history(true)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));

    simulation.run();

    assert!(simulation.find_by_name("missing").is_none());
    assert_eq!(
        simulation
            .produced_for_agent("sender")
            .map(<[Message]>::len),
        Some(1)
    );
    assert_eq!(
        Some(None),
        simulation
            .produced_for_agent("sender")
            .and_then(|messages| messages.first())
            .map(|message| message.destination)
    );
}

#[test]
#[should_panic(expected = "message history is off")]
fn reading_unrecorded_history_panics_instead_of_looking_empty() {
    let mut simulation = Simulation::builder(Halt::AtTime(1))
        .agents(vec![proactive_agent("sender", MissingTargetSender)])
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();
    let _ = simulation.produced_for_agent("sender");
}

#[derive(Clone, Debug)]
struct BurstSender;

impl Agent<u8> for BurstSender {
    fn on_tick(&mut self, ctx: &mut AgentContext<'_, u8>) {
        ctx.send("receiver", 1);
        ctx.send("receiver", 2);
    }
}

#[test]
fn messages_sent_by_one_callback_preserve_call_order() {
    let mut simulation = Simulation::builder(Halt::AtTime(1))
        .agents(vec![
            proactive_agent("sender", BurstSender),
            periodic_consumer("receiver".to_string(), 1),
        ])
        .message_history(true)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));

    simulation.run();

    let payloads = |messages: &[Message<u8>]| {
        messages
            .iter()
            .map(|message| message.payload)
            .collect::<Vec<_>>()
    };
    let expected = Some(vec![1, 2]);
    assert_eq!(
        expected,
        simulation.produced_for_agent("sender").map(payloads)
    );
    assert_eq!(
        expected,
        simulation.find_by_name("receiver").map(|agent| {
            agent
                .state
                .queue
                .iter()
                .map(|message| message.payload)
                .collect::<Vec<_>>()
        })
    );
}

#[derive(Clone, Debug)]
struct SlowConsumer;

impl Agent for SlowConsumer {
    fn on_message(&mut self, ctx: &mut AgentContext, _msg: &Message) {
        if ctx.time() < 3 {
            ctx.set_processing_status(MessageProcessingStatus::InProgress);
        }
    }
}

#[test]
fn in_progress_message_stays_queued_until_completed() {
    let mut simulation = Simulation::builder(Halt::when(|s: &Simulation| {
        s.find_by_name("consumer")
            .is_some_and(|consumer| consumer.consumed_count() == 1)
    }))
    .agents(vec![AgentInitializer {
        agent: Box::new(SlowConsumer),
        options: AgentOptions {
            initial_queue: vec![Message::new(0, ())].into(),
            name: "consumer".to_string(),
            ..Default::default()
        },
    }])
    .message_history(true)
    .build()
    .unwrap_or_else(|error| panic!("{error}"));

    simulation.run();

    assert_eq!(
        simulation
            .consumed_for_agent("consumer")
            .and_then(|messages| messages.first())
            .and_then(|message| message.completed_time),
        Some(3)
    );
    assert_eq!(simulation.time(), 4);
}

#[derive(Clone, Debug, Default)]
struct Drawer(Arc<std::sync::Mutex<Vec<u64>>>);

impl Agent for Drawer {
    fn on_tick(&mut self, ctx: &mut AgentContext) {
        let draws = [ctx.rng().next_u64(), ctx.rng().next_u64()];
        self.0
            .lock()
            .unwrap_or_else(|error| panic!("{error}"))
            .extend(draws);
    }
}

/// Draws of an agent named `name`, with `before` idle agents ahead of it and
/// `after` behind it.
fn draws(seed: u64, name: &str, before: usize, after: usize) -> Vec<u64> {
    let drawer = Drawer::default();
    let idle = |i: usize| periodic_consumer(format!("idle-{i}"), 1);
    let mut agents: Vec<_> = (0..before).map(idle).collect();
    agents.push(proactive_agent(name, drawer.clone()));
    agents.extend((before..before + after).map(idle));
    let mut simulation = Simulation::builder(Halt::AtTime(4))
        .agents(agents)
        .seed(simul::experiment::MasterSeed::from_u64(seed))
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();
    let draws = drawer.0.lock().unwrap_or_else(|error| panic!("{error}"));
    draws.clone()
}

#[test]
fn agent_streams_depend_only_on_seed_name_and_tick() {
    let base = draws(7, "drawer", 0, 0);
    assert_eq!(8, base.len());
    assert_eq!(
        base,
        draws(7, "drawer", 0, 0),
        "equal seeds give equal draws"
    );
    assert_eq!(
        base,
        draws(7, "drawer", 0, 100),
        "later agents do not shift the stream"
    );
    assert_eq!(
        base,
        draws(7, "drawer", 100, 0),
        "earlier agents do not shift the stream"
    );
    assert_ne!(
        base,
        draws(8, "drawer", 0, 0),
        "different seeds give different draws"
    );
    assert_ne!(
        base,
        draws(7, "other", 0, 0),
        "different names give different draws"
    );
    assert_ne!(base[0], base[1], "each call in a tick returns a new stream");
    assert_ne!(base[..2], base[2..4], "each tick starts a fresh stream");
}

#[test]
fn halt_variants_stop_at_the_documented_tick() {
    let run = |starting_time, halt| {
        let mut simulation = Simulation::builder(halt)
            .agents(vec![
                periodic_producer("producer", 1, "consumer"),
                periodic_consumer("consumer", 1),
            ])
            .starting_time(starting_time)
            .build()
            .unwrap_or_else(|error| panic!("{error}"));
        simulation.run();
        simulation.time()
    };
    assert_eq!(5, run(0, Halt::AtTime(5)));
    assert_eq!(9, run(9, Halt::AtTime(5)), "a past halt time runs no ticks");
    assert_eq!(
        3,
        run(
            0,
            Halt::when(|s: &Simulation| {
                s.find_by_name("consumer")
                    .is_some_and(|consumer| consumer.consumed_count() == 2)
            })
        )
    );
}

#[test]
fn geometric_producer_arrival_gaps_have_the_configured_mean_and_spread() {
    let mean = 4.0;
    let mut simulation = Simulation::<()>::builder(Halt::AtTime(40_000))
        .agents(vec![
            simul::agent::geometric_producer("producer", mean, "consumer"),
            periodic_consumer("consumer", 1),
        ])
        .seed(simul::experiment::MasterSeed::from_u64(3))
        .message_history(true)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();
    let Some(produced) = simulation.produced_for_agent("producer") else {
        panic!("producer must exist");
    };
    let gaps: Vec<f64> = produced
        .windows(2)
        .map(|pair| (pair[1].queued_time - pair[0].queued_time) as f64)
        .collect();
    let count = gaps.len() as f64;
    let observed_mean = gaps.iter().sum::<f64>() / count;
    let variance = gaps
        .iter()
        .map(|gap| (gap - observed_mean).powi(2))
        .sum::<f64>()
        / count;
    // A geometric gap on {1, 2, ...} with mean m has variance m(m - 1).
    assert!(gaps.len() > 9_000, "{} gaps", gaps.len());
    assert!(gaps.iter().all(|gap| *gap >= 1.0));
    assert!((observed_mean - mean).abs() < 0.1, "mean {observed_mean}");
    assert!(
        mean.mul_add(-(mean - 1.0), variance).abs() < 0.8,
        "variance {variance}"
    );
}

#[test]
fn geometric_agents_with_a_one_tick_mean_act_every_tick() {
    let mut simulation = Simulation::<()>::builder(Halt::AtTime(10))
        .agents(vec![
            simul::agent::geometric_producer("producer", 1.0, "consumer"),
            simul::agent::geometric_consumer("consumer", 1.0),
        ])
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();
    assert_eq!(
        Some(10),
        simulation
            .find_by_name("producer")
            .map(SimulationAgent::produced_count)
    );
    assert_eq!(
        Some(9),
        simulation
            .find_by_name("consumer")
            .map(SimulationAgent::consumed_count)
    );
}

#[test]
#[should_panic(expected = "mean period must be finite and at least one tick")]
fn geometric_agents_reject_means_below_one_tick() {
    let _ = simul::agent::geometric_producer::<_, ()>("producer", 0.5, "consumer");
}

#[test]
fn proactive_agents_handle_one_message_then_tick() {
    #[derive(Clone, Debug, Default)]
    struct Logger(std::sync::Arc<std::sync::Mutex<Vec<(u64, &'static str)>>>);
    impl Agent for Logger {
        fn on_message(&mut self, ctx: &mut AgentContext, _: &Message) {
            self.0
                .lock()
                .unwrap_or_else(|error| panic!("{error}"))
                .push((ctx.time(), "message"));
        }
        fn on_tick(&mut self, ctx: &mut AgentContext) {
            self.0
                .lock()
                .unwrap_or_else(|error| panic!("{error}"))
                .push((ctx.time(), "tick"));
        }
    }
    let log = Logger::default();
    let mut simulation = Simulation::builder(Halt::AtTime(3))
        .agents(vec![
            periodic_producer("producer", 1, "door"),
            proactive_agent("door", log.clone()),
        ])
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();

    let events = log
        .0
        .lock()
        .unwrap_or_else(|error| panic!("{error}"))
        .clone();
    assert_eq!(
        vec![
            (0, "tick"),
            (1, "message"),
            (1, "tick"),
            (2, "message"),
            (2, "tick"),
        ],
        events
    );
    assert_eq!(
        Some(2),
        simulation
            .find_by_name("door")
            .map(SimulationAgent::consumed_count)
    );
}

#[test]
fn initial_queue_messages_are_addressed_to_their_owner() {
    let mut owner = periodic_consumer("owner", 1);
    owner.options.initial_queue.push_back(Message::new(0, ()));
    let mut simulation = Simulation::builder(Halt::AtTime(1))
        .agents(vec![periodic_consumer("other", 1), owner])
        .message_history(true)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();

    let consumed = simulation.consumed_for_agent("owner");
    assert_eq!(
        Some(simulation.agent_id("owner")),
        consumed.map(|messages| messages[0].destination)
    );
}

#[test]
fn queue_depth_metrics_record_each_tick_depth() {
    let mut simulation = Simulation::<()>::builder(Halt::AtTime(5))
        .agents(vec![
            periodic_producer("producer", 1, "consumer"),
            periodic_consumer("consumer", 3),
        ])
        .queue_depth_metrics(true)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));

    simulation.run();

    let change = |time, depth| QueueDepthChange { time, depth };
    assert_eq!(
        Some([change(0, 0)].as_slice()),
        simulation.queue_depth_metrics("producer")
    );
    // Messages arrive at the end of each tick. The consumer clears tick 0's
    // message at tick 1, sleeps until tick 4, then takes one of three, so its
    // per-tick depths are 0, 0, 1, 2, 2.
    assert_eq!(
        Some([change(0, 0), change(2, 1), change(3, 2), change(5, 3)].as_slice()),
        simulation.queue_depth_metrics("consumer")
    );
}

#[test]
fn mutable_agent_access_preserves_identity_and_routing() {
    for count in [2, 9] {
        let mut agents = vec![proactive_agent("sender", MissingTargetSender)];
        agents.extend((1..count).map(|i| periodic_consumer(format!("receiver-{i}"), 1)));
        let mut simulation = Simulation::builder(Halt::AtTime(2))
            .agents(agents)
            .message_history(true)
            .build()
            .unwrap_or_else(|error| panic!("{error}"));
        let Some((behavior, state)) = simulation.find_by_name_mut("sender") else {
            panic!("sender must exist");
        };
        *behavior = periodic_producer("replacement", 1, "receiver-1").agent;
        state.mode = AgentMode::Proactive;
        simulation.run();

        assert_eq!(
            Some(Some(AgentId::new(0))),
            simulation
                .consumed_for_agent("receiver-1")
                .map(|messages| messages[0].source)
        );
        assert!(simulation.find_by_name("replacement").is_none());
        assert_eq!(
            simulation
                .produced_for_agent("sender")
                .map(<[Message]>::len),
            Some(2)
        );
    }
}

#[derive(Clone, Debug)]
struct BoundaryCallback(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Agent for BoundaryCallback {
    fn on_message(&mut self, _: &mut AgentContext, _: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        ctx.send_halt_interrupt("done");
    }
}

#[test]
fn time_overflow_fails_before_callbacks() {
    let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut simulation = Simulation::builder(Halt::Never)
        .agents(vec![proactive_agent(
            "worker",
            BoundaryCallback(called.clone()),
        )])
        .starting_time(u64::MAX)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| simulation.run()));
    assert!(result.is_err());
    assert!(!called.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(simulation.time(), u64::MAX);
}

#[test]
fn halt_reason_reports_the_interrupt_that_ended_the_latest_run() {
    let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut simulation = Simulation::builder(Halt::Never)
        .agents(vec![proactive_agent("worker", BoundaryCallback(called))])
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(None, simulation.halt_reason());
    simulation.run();
    assert_eq!(1, simulation.time());
    assert_eq!(Some("done"), simulation.halt_reason());

    let mut timed = Simulation::<()>::builder(Halt::AtTime(2))
        .agents(vec![periodic_consumer("idle", 1)])
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    timed.run();
    assert_eq!(None, timed.halt_reason());
}

#[test]
fn sleep_overflow_does_not_wrap_or_change_mode() {
    let mut simulation = Simulation::<()>::builder(Halt::AtTime(2))
        .agents(vec![periodic_producer("sender", u64::MAX, "missing")])
        .starting_time(1)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| simulation.run()));
    assert!(result.is_err());
    assert_eq!(simulation.time(), 1);
    assert_eq!(
        simulation
            .find_by_name("sender")
            .map(|agent| agent.state.mode),
        Some(AgentMode::Proactive)
    );
}

#[test]
fn final_representable_tick_and_wake_time_are_allowed() {
    let mut simulation = Simulation::<()>::builder(Halt::AtTime(u64::MAX))
        .agents(vec![periodic_producer("sender", 1, "missing")])
        .starting_time(u64::MAX - 1)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();
    assert_eq!(simulation.time(), u64::MAX);
    assert_eq!(
        simulation
            .find_by_name("sender")
            .map(|agent| agent.state.mode),
        Some(AgentMode::AsleepUntil(u64::MAX))
    );
    assert_eq!(
        simulation
            .find_by_name("sender")
            .map(SimulationAgent::produced_count),
        Some(1)
    );
}

#[test]
fn reused_producers_resolve_targets_in_each_simulation() {
    let mut first = Simulation::builder(Halt::AtTime(2))
        .agents(vec![
            periodic_producer("producer", 1, "consumer"),
            periodic_consumer("consumer", 1),
        ])
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    first.run();
    let Some(producer) = first.find_by_name("producer") else {
        panic!("producer must exist");
    };

    // The consumer's identity from the first simulation names the bystander here.
    let mut second = Simulation::builder(Halt::AtTime(2))
        .agents(vec![
            periodic_consumer("consumer", 1),
            periodic_consumer("bystander", 1),
            proactive_agent_boxed("producer", producer.agent.clone()),
        ])
        .message_history(true)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    second.run();

    let Some(sent) = second.produced_for_agent("producer") else {
        panic!("producer must exist");
    };
    assert_eq!(
        vec![Some(AgentId::new(0)); 2],
        sent.iter()
            .map(|message| message.destination)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        Some(0),
        second.consumed_for_agent("bystander").map(<[Message]>::len)
    );
}

#[test]
fn average_wait_preserves_short_durations_at_large_timestamps() {
    for starting_time in [0, 1_u64 << 53, u64::MAX - 2] {
        let mut simulation = Simulation::builder(Halt::when(|simulation: &Simulation| {
            simulation
                .find_by_name("receiver")
                .is_some_and(|receiver| receiver.consumed_count() > 0)
        }))
        .agents(vec![
            periodic_producer("sender", 1, "receiver"),
            periodic_consumer("receiver", 1),
        ])
        .starting_time(starting_time)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
        simulation.run();

        assert_eq!(
            Some(1.0_f64.to_bits()),
            simulation
                .calc_avg_wait_statistics()
                .get("receiver")
                .map(|wait| wait.to_bits()),
            "starting time: {starting_time}"
        );
    }
}

type CallbackTrace = std::sync::Arc<std::sync::Mutex<Vec<(u64, usize)>>>;

/// The tick and sender handle of a message, so payloads tell sends apart.
type Stamp = (u64, usize);

#[derive(Clone, Debug)]
struct ScheduledAgent(CallbackTrace);

impl Agent<Stamp> for ScheduledAgent {
    fn on_tick(&mut self, ctx: &mut AgentContext<'_, Stamp>) {
        self.0
            .lock()
            .unwrap_or_else(|error| panic!("{error}"))
            .push((ctx.time(), ctx.id().index()));
        if ctx.id().index() == 0 {
            ctx.send("agent-65", (ctx.time(), 0));
            ctx.sleep_for(ctx.time() % 3);
        } else {
            ctx.sleep_for(if ctx.time() < 6 { 1 } else { 100 });
        }
    }

    fn on_message(&mut self, ctx: &mut AgentContext<'_, Stamp>, _: &Message<Stamp>) {
        self.0
            .lock()
            .unwrap_or_else(|error| panic!("{error}"))
            .push((ctx.time(), ctx.id().index()));
        if ctx.time() % 4 == 1 {
            ctx.set_processing_status(MessageProcessingStatus::InProgress);
        }
        ctx.send("agent-0", (ctx.time(), ctx.id().index()));
        ctx.sleep_for(2);
    }
}

fn scheduling_scenario(metrics: bool) -> (Simulation<Stamp>, CallbackTrace) {
    let trace = CallbackTrace::default();
    let agents: Vec<_> = (0..130)
        .map(|index| {
            let (initial_mode, wake_mode) = match index {
                0 => (AgentMode::Proactive, AgentMode::Proactive),
                65 => (AgentMode::AsleepUntil(1), AgentMode::Reactive),
                128 => (AgentMode::AsleepUntil(0), AgentMode::Dead),
                129 => (AgentMode::AsleepUntil(0), AgentMode::AsleepUntil(2)),
                _ => (AgentMode::AsleepUntil(4), AgentMode::Proactive),
            };
            AgentInitializer {
                agent: Box::new(ScheduledAgent(trace.clone())),
                options: AgentOptions {
                    name: format!("agent-{index}"),
                    initial_mode,
                    wake_mode,
                    ..Default::default()
                },
            }
        })
        .collect();
    let simulation = Simulation::builder(Halt::AtTime(12))
        .agents(agents)
        .queue_depth_metrics(metrics)
        .asleep_cycles_metric(metrics)
        .message_history(true)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    (simulation, trace)
}

#[test]
fn sparse_and_dense_scheduling_preserve_callbacks_and_message_state() {
    let (mut sparse, sparse_trace) = scheduling_scenario(false);
    let (mut dense, dense_trace) = scheduling_scenario(true);
    sparse.run();
    dense.run();
    let sparse_trace = sparse_trace
        .lock()
        .unwrap_or_else(|error| panic!("{error}"));
    let dense_trace = dense_trace.lock().unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(*dense_trace, *sparse_trace);
    drop(dense_trace);
    // A proactive agent with a message gets on_message, then on_tick.
    assert!(sparse_trace.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(
        &[(0, 0), (1, 0), (1, 65), (2, 0), (2, 0)],
        &sparse_trace[..5]
    );
    drop(sparse_trace);
    assert_eq!(12, sparse.time());
    assert_eq!(Some(12), dense.asleep_cycle_count("agent-129"));
    assert_eq!(Some(0), dense.asleep_cycle_count("agent-128"));
    for (left, right) in sparse.agents().iter().zip(dense.agents()) {
        assert_eq!(right.state.mode, left.state.mode);
        assert_eq!(right.state.wake_mode, left.state.wake_mode);
        assert_eq!(right.state.queue.len(), left.state.queue.len());
        assert_eq!(right.produced().len(), left.produced().len());
        assert_eq!(right.consumed().len(), left.consumed().len());
        let left_messages = left
            .state
            .queue
            .iter()
            .chain(left.produced())
            .chain(left.consumed());
        let right_messages = right
            .state
            .queue
            .iter()
            .chain(right.produced())
            .chain(right.consumed());
        for (left, right) in left_messages.zip(right_messages) {
            assert_eq!(right.queued_time, left.queued_time);
            assert_eq!(right.completed_time, left.completed_time);
            assert_eq!(right.source, left.source);
            assert_eq!(right.destination, left.destination);
            assert_eq!(right.payload, left.payload);
        }
    }
    for agent in dense.agents() {
        let changes = dense.queue_depth_metrics(&agent.name);
        assert_eq!(Some(0), changes.and_then(|c| c.first()).map(|c| c.time));
        assert_eq!(
            Some(agent.state.queue.len()),
            changes.and_then(|c| c.last()).map(|c| c.depth)
        );
    }
}

#[derive(Clone, Debug)]
struct ChangingPayloadSender;

impl Agent<Option<u8>> for ChangingPayloadSender {
    fn on_tick(&mut self, ctx: &mut AgentContext<'_, Option<u8>>) {
        ctx.send("receiver", (ctx.time() == 2).then_some(7));
    }
}

#[test]
fn repeated_sends_preserve_payloads_and_timestamps() {
    let mut simulation = Simulation::builder(Halt::AtTime(5))
        .agents(vec![
            proactive_agent("sender", ChangingPayloadSender),
            periodic_consumer("receiver", 1),
        ])
        .message_history(true)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();

    let Some(produced) = simulation.produced_for_agent("sender") else {
        panic!("sender must exist");
    };
    assert_eq!(produced.len(), 5);
    for (time, message) in (0_u64..).zip(produced) {
        assert_eq!(message.queued_time, time);
        assert_eq!(message.completed_time, None);
        assert_eq!(message.payload, (time == 2).then_some(7));
    }

    let Some(consumed) = simulation.consumed_for_agent("receiver") else {
        panic!("receiver must exist");
    };
    assert_eq!(consumed.len(), 4);
    for (time, message) in (0_u64..).zip(consumed) {
        assert_eq!(message.source, Some(AgentId::new(0)));
        assert_eq!(message.destination, Some(AgentId::new(1)));
        assert_eq!(message.queued_time, time);
        assert_eq!(message.completed_time, Some(time + 1));
        assert_eq!(message.payload, (time == 2).then_some(7));
    }
}

#[test]
fn quiet_ticks_check_halt_before_waking_and_rebuild_state_on_resume() {
    let trace = CallbackTrace::default();
    let mut agents: Vec<_> = (0..130)
        .map(|index| AgentInitializer {
            agent: Box::new(ScheduledAgent(trace.clone())),
            options: AgentOptions {
                name: format!("agent-{index}"),
                initial_mode: AgentMode::Dead,
                ..Default::default()
            },
        })
        .collect();
    agents[65].options.initial_mode = AgentMode::AsleepUntil(5);
    agents[65]
        .options
        .initial_queue
        .push_back(Message::new(0, (0, 65)));
    let mut simulation = Simulation::builder(Halt::when(|simulation: &Simulation<Stamp>| {
        simulation.find_by_name("agent-65").is_some_and(|agent| {
            (simulation.time() == 5 && agent.state.mode == AgentMode::AsleepUntil(5))
                || agent.consumed_count() > 0
        })
    }))
    .agents(agents)
    .message_history(true)
    .build()
    .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();
    assert_eq!(5, simulation.time());
    assert!(trace
        .lock()
        .unwrap_or_else(|error| panic!("{error}"))
        .is_empty());
    let (_, state) = simulation
        .find_by_name_mut("agent-65")
        .unwrap_or_else(|| panic!("missing agent"));
    assert_eq!(AgentMode::AsleepUntil(5), state.mode);
    state.mode = AgentMode::AsleepUntil(3);
    simulation.run();
    // Tick 5 retries the original message; tick 7 completes it.
    assert_eq!(8, simulation.time());
    assert_eq!(
        vec![(5, 65), (7, 65)],
        *trace.lock().unwrap_or_else(|error| panic!("{error}"))
    );
    assert_eq!(
        Some(7),
        simulation
            .consumed_for_agent("agent-65")
            .and_then(|messages| messages[0].completed_time)
    );
}

#[test]
fn all_sleeping_agents_still_check_halt_between_wakeups() {
    let trace = CallbackTrace::default();
    let agents: Vec<_> = (0..130)
        .map(|index| {
            let mut agent =
                proactive_agent(&format!("agent-{index}"), ScheduledAgent(trace.clone()));
            agent.options.initial_mode = AgentMode::AsleepUntil(10);
            agent
        })
        .collect();
    let mut simulation = Simulation::builder(Halt::AtTime(3))
        .agents(agents)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();
    assert_eq!(3, simulation.time());
    assert!(trace
        .lock()
        .unwrap_or_else(|error| panic!("{error}"))
        .is_empty());
}

#[test]
fn callback_panic_fails_the_simulation_without_waking_later_agents() {
    #[derive(Clone, Debug)]
    struct PanicTick;
    impl Agent for PanicTick {
        fn on_message(&mut self, _: &mut AgentContext, _: &Message) {}
        fn on_tick(&mut self, _: &mut AgentContext) {
            panic!("callback failed");
        }
    }
    let mut agents: Vec<_> = (0..130)
        .map(|index| {
            let mut agent = periodic_consumer(format!("agent-{index}"), 1);
            agent.options.initial_mode = AgentMode::AsleepUntil(u64::MAX);
            agent
        })
        .collect();
    agents[0] = proactive_agent("agent-0", PanicTick);
    agents[65].options.initial_mode = AgentMode::AsleepUntil(0);
    let mut simulation = Simulation::builder(Halt::Never)
        .agents(agents)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| simulation.run()));
    assert!(result.is_err());
    assert_eq!(SimulationMode::Failed, simulation.mode());
    assert_eq!(
        Some(AgentMode::AsleepUntil(0)),
        simulation
            .find_by_name("agent-65")
            .map(|agent| agent.state.mode)
    );
    let rerun = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| simulation.run()));
    assert!(rerun.is_err(), "a failed simulation must refuse to run");
    assert_eq!(0, simulation.time());
}

#[derive(Clone, Debug, Default)]
struct TickCounter {
    ticks: u64,
}

impl Agent for TickCounter {
    fn on_tick(&mut self, _ctx: &mut AgentContext) {
        self.ticks += 1;
    }
}

#[test]
fn finished_agents_read_back_as_their_own_type() {
    let mut simulation = Simulation::builder(Halt::AtTime(5))
        .agents(vec![
            periodic_consumer("idle", 1),
            proactive_agent("counter", TickCounter::default()),
        ])
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();

    let counter = simulation.find_by_name("counter");
    assert_eq!(
        Some(5),
        counter
            .and_then(|agent| agent.downcast_ref::<TickCounter>())
            .map(|counter| counter.ticks)
    );
    assert!(
        counter
            .and_then(|agent| agent.downcast_ref::<HaltImmediately>())
            .is_none(),
        "another agent type does not match"
    );
}

/// Sends `(tick, delay, payload)` steps to agent 1; delay 0 means `send_to`.
#[derive(Clone, Debug)]
struct ScriptedSender(Vec<(u64, u64, u8)>);

impl Agent<u8> for ScriptedSender {
    fn on_tick(&mut self, ctx: &mut AgentContext<'_, u8>) {
        for &(tick, delay, payload) in &self.0 {
            if tick != ctx.time() {
                continue;
            }
            if delay == 0 {
                ctx.send_to(AgentId::new(1), payload);
            } else {
                ctx.send_after(delay, AgentId::new(1), payload);
            }
        }
    }
}

/// Logs `(tick, payload, queued_time)` for each message it handles.
#[derive(Clone, Debug, Default)]
struct Inbox(Vec<(u64, u8, u64)>);

impl Agent<u8> for Inbox {
    fn on_message(&mut self, ctx: &mut AgentContext<'_, u8>, msg: &Message<u8>) {
        self.0.push((ctx.time(), msg.payload, msg.queued_time));
    }
}

fn delayed_run(halt_at: u64) -> Simulation<u8> {
    let script = vec![(0, 3, 1), (0, 0, 2), (2, 0, 3)];
    let mut simulation = Simulation::builder(Halt::AtTime(halt_at))
        .agents(vec![
            proactive_agent("sender", ScriptedSender(script)),
            AgentInitializer {
                agent: Box::new(Inbox::default()),
                options: AgentOptions::defaults_with_name("inbox".to_string()),
            },
        ])
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();
    simulation
}

fn inbox(simulation: &Simulation<u8>) -> Vec<(u64, u8, u64)> {
    simulation
        .find_by_name("inbox")
        .and_then(|agent| agent.downcast_ref::<Inbox>())
        .map(|inbox| inbox.0.clone())
        .unwrap_or_default()
}

#[test]
fn delayed_messages_arrive_after_their_delay_ahead_of_later_sends() {
    // Message 1 arrives at tick 3 with message 3, but was sent first; the
    // inbox takes one message per tick, so message 3 waits until tick 4.
    assert_eq!(
        vec![(1, 2, 0), (3, 1, 2), (4, 3, 2)],
        inbox(&delayed_run(10))
    );

    let halted = delayed_run(2);
    assert_eq!(vec![(1, 2, 0)], inbox(&halted));
    assert_eq!(
        vec![(1, 2)],
        halted
            .in_flight()
            .map(|message| (message.payload, message.queued_time))
            .collect::<Vec<_>>(),
        "a halted run keeps messages that have not arrived"
    );
    assert_eq!(
        Some(2),
        halted
            .find_by_name("sender")
            .map(SimulationAgent::produced_count),
        "a delayed message counts as sent when it is sent"
    );
}

#[derive(Clone, Debug)]
struct ZeroDelay;

impl Agent for ZeroDelay {
    fn on_tick(&mut self, ctx: &mut AgentContext) {
        ctx.send_after(0, ctx.id(), ());
    }
}

#[test]
#[should_panic(expected = "a message cannot arrive in the tick it is sent")]
fn delayed_messages_need_a_positive_delay() {
    let mut simulation = Simulation::builder(Halt::AtTime(1))
        .agents(vec![proactive_agent("sender", ZeroDelay)])
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();
}

/// Sends one delayed message to agent 1 at tick 0, then sleeps for good.
#[derive(Clone, Debug)]
struct SendOnceThenSleep;

impl Agent<u8> for SendOnceThenSleep {
    fn on_tick(&mut self, ctx: &mut AgentContext<'_, u8>) {
        ctx.send_after(50, AgentId::new(1), 9);
        ctx.sleep_for(1_000);
    }
}

#[test]
fn skipping_quiet_ticks_stops_at_the_next_arrival() {
    // Beyond 64 agents the sparse schedule serves the run instead of scans.
    for idle in [0, 100] {
        let mut agents = vec![
            proactive_agent("sender", SendOnceThenSleep),
            AgentInitializer {
                agent: Box::new(Inbox::default()),
                options: AgentOptions::defaults_with_name("inbox".to_string()),
            },
        ];
        agents.extend((0..idle).map(|i| periodic_consumer(format!("idle-{i}"), 1)));
        let mut simulation = Simulation::builder(Halt::AtTime(100))
            .agents(agents)
            .build()
            .unwrap_or_else(|error| panic!("{error}"));
        simulation.run();
        assert_eq!(vec![(50, 9, 49)], inbox(&simulation), "{idle} idle agents");
    }
}

/// Keeps one draw per tick from its agent stream.
#[derive(Clone, Debug, Default)]
struct Sampler(Vec<u64>);

impl Agent for Sampler {
    fn on_tick(&mut self, ctx: &mut AgentContext) {
        self.0.push(ctx.rng().next_u64());
    }
}

/// The sampler's draws for two candidates, two replications each, in trial
/// order: candidate 0 replications 0 and 1, then candidate 1.
fn trial_draws(common_random_group: u64) -> Vec<Option<Vec<u64>>> {
    use simul::experiment::{run_serial, CandidateId, FixedReplications, MasterSeed};
    let schedule = FixedReplications::new(2, common_random_group);
    let trials = schedule
        .trials(CandidateId::new(0), ())
        .chain(schedule.trials(CandidateId::new(1), ()));
    let records = run_serial(MasterSeed::from_u64(5), trials, |(), context| {
        let mut simulation = Simulation::builder(Halt::AtTime(3))
            .agents(vec![proactive_agent("sampler", Sampler::default())])
            .trial(context)
            .build()
            .unwrap_or_else(|error| panic!("{error}"));
        simulation.run();
        let sampler = simulation.find_by_name("sampler");
        Ok::<_, std::convert::Infallible>(
            sampler
                .and_then(|agent| agent.downcast_ref::<Sampler>())
                .map(|sampler| sampler.0.clone()),
        )
    })
    .unwrap_or_else(|error| panic!("{error}"));
    records
        .into_iter()
        .map(|record| match record.result {
            Ok(draws) => draws,
            Err(never) => match never {},
        })
        .collect()
}

#[test]
fn trial_seeds_pair_candidates_and_vary_by_replication() {
    let draws = trial_draws(1);
    assert_eq!(4, draws.len());
    assert!(draws[0].as_ref().is_some_and(|draws| draws.len() == 3));
    assert_eq!(
        draws[0], draws[2],
        "candidates in one group share replication 0"
    );
    assert_eq!(
        draws[1], draws[3],
        "candidates in one group share replication 1"
    );
    assert_ne!(draws[0], draws[1], "each replication draws afresh");
    assert_ne!(
        draws[0],
        trial_draws(2)[0],
        "another group draws differently"
    );
}
