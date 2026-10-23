#![allow(clippy::missing_docs_in_private_items)]

use simul::agent::{
    periodic_consumer, periodic_producer, Agent, AgentContext, AgentId, AgentInitializer,
    AgentMode, AgentOptions, MessageProcessingStatus,
};
use simul::message::Message;
use simul::{Simulation, SimulationMode, SimulationParameters};
use std::sync::Arc;

fn proactive_agent(name: &str, agent: impl Agent + 'static) -> AgentInitializer {
    proactive_agent_boxed(name, Box::new(agent))
}

fn proactive_agent_boxed(name: &str, agent: Box<dyn Agent>) -> AgentInitializer {
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
            ctx.send(target, None);
        }
    }
}

#[test]
fn messages_route_to_lower_own_and_higher_handles() {
    for count in [3_u32, 8, 9] {
        let name = |handle: u32| format!("agent-{handle}");
        let neighbors = |handle: u32| [(handle + count - 1) % count, handle, (handle + 1) % count];
        let mut simulation = Simulation::new(SimulationParameters {
            agent_initializers: (0..count)
                .map(|handle| {
                    let [previous, _, next] = neighbors(handle);
                    proactive_agent(
                        &name(handle),
                        NeighborSender {
                            previous: name(previous),
                            next: name(next),
                        },
                    )
                })
                .collect(),
            halt_check: |simulation| simulation.time() == 1,
            ..Default::default()
        });
        simulation.run();

        for handle in 0..count {
            let id = AgentId::new(handle);
            assert_eq!(Some(id), simulation.agent_id(&name(handle)));
            let Some(agent) = simulation.agent(id) else {
                panic!("agent-{handle} must exist");
            };
            let destinations: Vec<Option<AgentId>> = agent
                .state
                .produced
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
            ctx.send_to(AgentId::new(index), None);
        }
    }
}

#[test]
fn send_to_delivers_by_identity_and_records_unknown_identities() {
    for count in [3_u32, 9] {
        let mut simulation = Simulation::new(SimulationParameters {
            agent_initializers: (0..count)
                .map(|handle| proactive_agent(&format!("agent-{handle}"), IdentitySender { count }))
                .collect(),
            halt_check: |simulation| simulation.time() == 1,
            ..Default::default()
        });
        simulation.run();

        let expected: Vec<Option<AgentId>> = (0..count)
            .map(|id| Some(AgentId::new(id)))
            .chain([None])
            .collect();
        for agent in simulation.agents() {
            let destinations: Vec<Option<AgentId>> = agent
                .state
                .produced
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
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![proactive_agent("halter", HaltImmediately)],
        halt_check: |_| false,
        ..Default::default()
    });

    simulation.run();

    assert_eq!(simulation.time(), 1);
}

#[derive(Clone, Debug)]
struct MissingTargetSender;

impl Agent for MissingTargetSender {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        ctx.send("missing", None);
    }
}

#[test]
fn unknown_destination_records_produced_message_without_delivery() {
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![proactive_agent("sender", MissingTargetSender)],
        halt_check: |s: &Simulation| s.time() == 1,
        ..Default::default()
    });

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

#[derive(Clone, Debug)]
struct BurstSender;

impl Agent for BurstSender {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        ctx.send("receiver", Some(Arc::from([1])));
        ctx.send("receiver", Some(Arc::from([2])));
    }
}

#[test]
fn messages_sent_by_one_callback_preserve_call_order() {
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![
            proactive_agent("sender", BurstSender),
            periodic_consumer("receiver".to_string(), 1),
        ],
        halt_check: |simulation| simulation.time() == 1,
        ..Default::default()
    });

    simulation.run();

    let payloads = |messages: &[Message]| {
        messages
            .iter()
            .map(|message| message.custom_payload.clone())
            .collect::<Vec<_>>()
    };
    let expected = Some(vec![Some(Arc::from([1])), Some(Arc::from([2]))]);
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
                .map(|message| message.custom_payload.clone())
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
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![AgentInitializer {
            agent: Box::new(SlowConsumer),
            options: AgentOptions {
                initial_queue: vec![Message::new(0)].into(),
                name: "consumer".to_string(),
                ..Default::default()
            },
        }],
        halt_check: |s: &Simulation| {
            s.consumed_for_agent("consumer")
                .is_some_and(|messages| messages.len() == 1)
        },
        ..Default::default()
    });

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

fn draws(seed: u64, extra_agents: usize) -> Vec<u64> {
    let drawer = Drawer::default();
    let mut agents = vec![proactive_agent("drawer", drawer.clone())];
    agents.extend((0..extra_agents).map(|i| periodic_consumer(format!("idle-{i}"), 1)));
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: agents,
        halt_check: |s: &Simulation| s.time() == 4,
        seed: simul::experiment::MasterSeed::from_u64(seed),
        ..Default::default()
    });
    simulation.run();
    let draws = drawer.0.lock().unwrap_or_else(|error| panic!("{error}"));
    draws.clone()
}

#[test]
fn agent_streams_depend_only_on_seed_agent_and_tick() {
    let base = draws(7, 0);
    assert_eq!(8, base.len());
    assert_eq!(base, draws(7, 0), "equal seeds give equal draws");
    assert_eq!(base, draws(7, 100), "later agents do not shift the stream");
    assert_ne!(base, draws(8, 0), "different seeds give different draws");
    assert_ne!(base[0], base[1], "draws within a tick continue one stream");
    assert_ne!(base[..2], base[2..4], "each tick starts a fresh stream");
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
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![
            periodic_producer("producer", 1, "door"),
            proactive_agent("door", log.clone()),
        ],
        halt_check: |s: &Simulation| s.time() == 3,
        ..Default::default()
    });
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
        simulation.consumed_for_agent("door").map(<[_]>::len)
    );
}

#[test]
fn initial_queue_messages_are_addressed_to_their_owner() {
    let mut owner = periodic_consumer("owner", 1);
    owner.options.initial_queue.push_back(Message::new(0));
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![periodic_consumer("other", 1), owner],
        halt_check: |s: &Simulation| s.time() == 1,
        ..Default::default()
    });
    simulation.run();

    let consumed = simulation.consumed_for_agent("owner");
    assert_eq!(
        Some(simulation.agent_id("owner")),
        consumed.map(|messages| messages[0].destination)
    );
}

#[test]
fn queue_depth_metrics_record_each_tick_depth() {
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![
            periodic_producer("producer", 1, "consumer"),
            periodic_consumer("consumer", 3),
        ],
        enable_queue_depth_metrics: true,
        halt_check: |s: &Simulation| s.time() == 5,
        ..Default::default()
    });

    simulation.run();

    assert_eq!(
        Some([0_usize; 5].as_slice()),
        simulation.queue_depth_metrics("producer")
    );
    // Messages arrive at the end of each tick. The consumer clears tick 0's
    // message at tick 1, sleeps until tick 4, then takes one of three.
    assert_eq!(
        Some([0_usize, 0, 1, 2, 2].as_slice()),
        simulation.queue_depth_metrics("consumer")
    );
}

#[test]
fn mutable_agent_access_preserves_identity_and_routing() {
    for count in [2, 9] {
        let mut agents = vec![proactive_agent("sender", MissingTargetSender)];
        agents.extend((1..count).map(|i| periodic_consumer(format!("receiver-{i}"), 1)));
        let mut simulation = Simulation::new(SimulationParameters {
            agent_initializers: agents,
            halt_check: |simulation| simulation.time() == 2,
            ..Default::default()
        });
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
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![proactive_agent("worker", BoundaryCallback(called.clone()))],
        starting_time: u64::MAX,
        halt_check: |_| false,
        ..Default::default()
    });
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| simulation.run()));
    assert!(result.is_err());
    assert!(!called.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(simulation.time(), u64::MAX);
}

#[test]
fn sleep_overflow_does_not_wrap_or_change_mode() {
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![periodic_producer("sender", u64::MAX, "missing")],
        starting_time: 1,
        halt_check: |simulation| simulation.time() == 2,
        ..Default::default()
    });
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
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![periodic_producer("sender", 1, "missing")],
        starting_time: u64::MAX - 1,
        halt_check: |simulation| simulation.time() == u64::MAX,
        ..Default::default()
    });
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
            .produced_for_agent("sender")
            .map(<[Message]>::len),
        Some(1)
    );
}

#[test]
fn reused_producers_resolve_targets_in_each_simulation() {
    let mut first = Simulation::new(SimulationParameters {
        agent_initializers: vec![
            periodic_producer("producer", 1, "consumer"),
            periodic_consumer("consumer", 1),
        ],
        halt_check: |simulation| simulation.time() == 2,
        ..Default::default()
    });
    first.run();
    let Some(producer) = first.find_by_name("producer") else {
        panic!("producer must exist");
    };

    // The consumer's identity from the first simulation names the bystander here.
    let mut second = Simulation::new(SimulationParameters {
        agent_initializers: vec![
            periodic_consumer("consumer", 1),
            periodic_consumer("bystander", 1),
            proactive_agent_boxed("producer", producer.agent.clone()),
        ],
        halt_check: |simulation| simulation.time() == 2,
        ..Default::default()
    });
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
        let mut simulation = Simulation::new(SimulationParameters {
            agent_initializers: vec![
                periodic_producer("sender", 1, "receiver"),
                periodic_consumer("receiver", 1),
            ],
            starting_time,
            halt_check: |simulation| {
                simulation
                    .consumed_for_agent("receiver")
                    .is_some_and(|messages| !messages.is_empty())
            },
            ..Default::default()
        });
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

#[derive(Clone, Debug)]
struct ScheduledAgent(CallbackTrace);

impl Agent for ScheduledAgent {
    fn on_tick(&mut self, ctx: &mut AgentContext) {
        self.0
            .lock()
            .unwrap_or_else(|error| panic!("{error}"))
            .push((ctx.time(), ctx.id().index()));
        if ctx.id().index() == 0 {
            ctx.send("agent-65", Some(Arc::from([1])));
            ctx.sleep_for(ctx.time() % 3);
        } else {
            ctx.sleep_for(if ctx.time() < 6 { 1 } else { 100 });
        }
    }

    fn on_message(&mut self, ctx: &mut AgentContext, _: &Message) {
        self.0
            .lock()
            .unwrap_or_else(|error| panic!("{error}"))
            .push((ctx.time(), ctx.id().index()));
        if ctx.time() % 4 == 1 {
            ctx.set_processing_status(MessageProcessingStatus::InProgress);
        }
        ctx.send("agent-0", None);
        ctx.sleep_for(2);
    }
}

fn scheduling_scenario(metrics: bool) -> (Simulation, CallbackTrace) {
    let trace = CallbackTrace::default();
    let agents = (0..130)
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
    let simulation = Simulation::new(SimulationParameters {
        agent_initializers: agents,
        halt_check: |simulation| simulation.time() == 12,
        enable_queue_depth_metrics: metrics,
        enable_agent_asleep_cycles_metric: metrics,
        ..Default::default()
    });
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
        let (left, right) = (&left.state, &right.state);
        assert_eq!(right.mode, left.mode);
        assert_eq!(right.wake_mode, left.wake_mode);
        assert_eq!(right.queue.len(), left.queue.len());
        assert_eq!(right.produced.len(), left.produced.len());
        assert_eq!(right.consumed.len(), left.consumed.len());
        let left_messages = left
            .queue
            .iter()
            .chain(&left.produced)
            .chain(&left.consumed);
        let right_messages = right
            .queue
            .iter()
            .chain(&right.produced)
            .chain(&right.consumed);
        for (left, right) in left_messages.zip(right_messages) {
            assert_eq!(right.queued_time, left.queued_time);
            assert_eq!(right.completed_time, left.completed_time);
            assert_eq!(right.source, left.source);
            assert_eq!(right.destination, left.destination);
            assert_eq!(right.custom_payload, left.custom_payload);
        }
    }
    for agent in dense.agents() {
        assert_eq!(
            Some(12),
            dense.queue_depth_metrics(&agent.name).map(<[usize]>::len)
        );
    }
}

#[derive(Clone, Debug)]
struct ChangingPayloadSender;

impl Agent for ChangingPayloadSender {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        ctx.send("receiver", (ctx.time() == 2).then(|| Arc::from([7])));
    }
}

#[test]
fn repeated_sends_preserve_payloads_and_timestamps() {
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![
            proactive_agent("sender", ChangingPayloadSender),
            periodic_consumer("receiver", 1),
        ],
        halt_check: |simulation| simulation.time() == 5,
        ..Default::default()
    });
    simulation.run();

    let Some(produced) = simulation.produced_for_agent("sender") else {
        panic!("sender must exist");
    };
    assert_eq!(produced.len(), 5);
    for (time, message) in (0_u64..).zip(produced) {
        assert_eq!(message.queued_time, time);
        assert_eq!(message.completed_time, None);
        assert_eq!(
            message.custom_payload.as_deref(),
            (time == 2).then_some([7].as_slice())
        );
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
        assert_eq!(
            message.custom_payload.as_deref(),
            (time == 2).then_some([7].as_slice())
        );
    }
}

#[test]
fn new_sends_do_not_inherit_edited_history() {
    let mutations: [fn(&mut Message); 5] = [
        |message| message.source = Some(AgentId::new(9)),
        |message| message.destination = None,
        |message| message.custom_payload = Some(Arc::from([])),
        |message| message.queued_time = u64::MAX,
        |message| message.completed_time = Some(u64::MAX),
    ];
    for mutate in mutations {
        let mut simulation = Simulation::new(SimulationParameters {
            agent_initializers: vec![
                periodic_producer("sender", 1, "receiver"),
                periodic_consumer("receiver", 1),
                proactive_agent("halter", HaltImmediately),
            ],
            halt_check: |_| false,
            ..Default::default()
        });
        simulation.run();
        let Some((_, state)) = simulation.find_by_name_mut("sender") else {
            panic!("sender must exist");
        };
        mutate(&mut state.produced[0]);
        simulation.run();

        let Some(produced) = simulation.produced_for_agent("sender") else {
            panic!("sender must exist");
        };
        let message = &produced[1];
        assert_eq!(message.source, Some(AgentId::new(0)));
        assert_eq!(message.destination, Some(AgentId::new(1)));
        assert!(message.custom_payload.is_none());
        assert_eq!(message.queued_time, 1);
        assert_eq!(message.completed_time, None);
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
    agents[65].options.initial_queue.push_back(Message::new(0));
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: agents,
        halt_check: |simulation| {
            simulation.find_by_name("agent-65").is_some_and(|agent| {
                (simulation.time() == 5 && agent.state.mode == AgentMode::AsleepUntil(5))
                    || !agent.state.consumed.is_empty()
            })
        },
        ..Default::default()
    });
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
    let agents = (0..130)
        .map(|index| {
            let mut agent =
                proactive_agent(&format!("agent-{index}"), ScheduledAgent(trace.clone()));
            agent.options.initial_mode = AgentMode::AsleepUntil(10);
            agent
        })
        .collect();
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: agents,
        halt_check: |simulation| simulation.time() >= 3,
        ..Default::default()
    });
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
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: agents,
        halt_check: |_| false,
        ..Default::default()
    });
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
