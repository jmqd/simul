#![allow(clippy::missing_docs_in_private_items)]

use simul::agent::{
    periodic_consumer, periodic_producer, Agent, AgentContext, AgentInitializer, AgentMode,
    AgentOptions, MessageProcessingStatus,
};
use simul::message::Message;
use simul::{Simulation, SimulationParameters};

fn proactive_agent(name: &str, agent: impl Agent + 'static) -> AgentInitializer {
    AgentInitializer {
        agent: Box::new(agent),
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
        let own = ctx.name.to_string();
        for target in [&self.previous, &own, &self.next] {
            ctx.send(target, None);
        }
    }
}

#[test]
fn messages_route_to_lower_own_and_higher_handles() {
    for count in [3, 8, 9] {
        let name = |handle: usize| format!("agent-{handle}");
        let mut simulation = Simulation::new(SimulationParameters {
            agent_initializers: (0..count)
                .map(|handle| {
                    proactive_agent(
                        &name(handle),
                        NeighborSender {
                            previous: name((handle + count - 1) % count),
                            next: name((handle + 1) % count),
                        },
                    )
                })
                .collect(),
            halt_check: |simulation| simulation.time() == 1,
            ..Default::default()
        });
        simulation.run();

        for receiver in 0..count {
            let mut senders = [
                (receiver + count - 1) % count,
                receiver,
                (receiver + 1) % count,
            ];
            senders.sort_unstable();
            let Some(agent) = simulation.find_by_name(&name(receiver)) else {
                panic!("agent-{receiver} must exist");
            };
            let sources: Vec<&str> = agent
                .state
                .queue
                .iter()
                .map(|message| message.source.as_str())
                .collect();
            assert_eq!(
                senders.map(name),
                sources.as_slice(),
                "{count} agents, receiver agent-{receiver}"
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
        simulation
            .produced_for_agent("sender")
            .and_then(|messages| messages.first())
            .map(|message| message.destination.as_str()),
        Some("missing")
    );
}

#[derive(Clone, Debug)]
struct BurstSender;

impl Agent for BurstSender {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        ctx.send("receiver", Some(vec![1]));
        ctx.send("receiver", Some(vec![2]));
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
    assert_eq!(
        simulation.produced_for_agent("sender").map(payloads),
        Some(vec![Some(vec![1]), Some(vec![2])])
    );
    assert_eq!(
        simulation.find_by_name("receiver").map(|agent| {
            agent
                .state
                .queue
                .iter()
                .map(|message| message.custom_payload.clone())
                .collect::<Vec<_>>()
        }),
        Some(vec![Some(vec![1]), Some(vec![2])])
    );
}

#[derive(Clone, Debug)]
struct SlowConsumer;

impl Agent for SlowConsumer {
    fn on_message(&mut self, ctx: &mut AgentContext, _msg: &Message) {
        if ctx.time < 3 {
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
                initial_queue: vec![Message::new(0, "producer", "consumer")].into(),
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

#[test]
fn queue_depth_metrics_record_every_tick() {
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![
            periodic_producer("producer".to_string(), 1, "consumer".to_string()),
            periodic_consumer("consumer".to_string(), 1),
        ],
        enable_queue_depth_metrics: true,
        halt_check: |s: &Simulation| s.time() == 5,
        ..Default::default()
    });

    simulation.run();

    assert_eq!(
        simulation
            .queue_depth_metrics("producer")
            .map(<[usize]>::len),
        Some(5)
    );
    assert_eq!(
        simulation
            .queue_depth_metrics("consumer")
            .map(<[usize]>::len),
        Some(5)
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
            simulation
                .consumed_for_agent("receiver-1")
                .map(|messages| messages[0].source.as_str()),
            Some("sender")
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
fn cloned_messages_keep_independent_contents_and_timestamps() {
    let original = Message::new_with_payload(7, "sender", "receiver", Some(vec![1, 2]));
    let mut edited = original.clone();
    edited.queued_time = 8;
    edited.completed_time = Some(9);
    edited.source.push_str("-edited");
    edited.destination.push_str("-edited");
    let Some(payload) = &mut edited.custom_payload else {
        panic!("payload must exist");
    };
    payload[0] = 3;
    edited.interrupt = Some(simul::message::Interrupt::HaltSimulation("done".into()));

    assert_eq!(original.queued_time, 7);
    assert_eq!(original.completed_time, None);
    assert_eq!(original.source, "sender");
    assert_eq!(original.destination, "receiver");
    assert_eq!(original.custom_payload.as_deref(), Some([1, 2].as_slice()));
    assert!(original.interrupt.is_none());
    assert_eq!(edited.custom_payload.as_deref(), Some([3, 2].as_slice()));
    assert_eq!(edited.completed_time, Some(9));
}

#[test]
fn consumed_message_completion_does_not_change_sender_history() {
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![
            proactive_agent("sender", BurstSender),
            periodic_consumer("receiver", 1),
        ],
        halt_check: |simulation| simulation.time() == 2,
        ..Default::default()
    });
    simulation.run();
    let Some(produced) = simulation.produced_for_agent("sender") else {
        panic!("sender must exist");
    };
    let Some(consumed) = simulation.consumed_for_agent("receiver") else {
        panic!("receiver must exist");
    };
    assert_eq!(produced[0].completed_time, None);
    assert_eq!(consumed[0].completed_time, Some(1));
    assert_eq!(produced[0].custom_payload, consumed[0].custom_payload);
    assert_eq!(produced[0].queued_time, consumed[0].queued_time);
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

#[derive(Clone, Debug)]
struct ChangingPayloadSender;

impl Agent for ChangingPayloadSender {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        ctx.send("receiver", (ctx.time == 2).then(|| vec![7]));
    }
}

#[test]
fn repeated_sends_preserve_payloads_timestamps_and_independent_mutation() {
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![
            proactive_agent("sender", ChangingPayloadSender),
            periodic_consumer("receiver", 1),
        ],
        halt_check: |simulation| simulation.time() == 5,
        ..Default::default()
    });
    simulation.run();

    let Some((_, state)) = simulation.find_by_name_mut("sender") else {
        panic!("sender must exist");
    };
    assert_eq!(state.produced.len(), 5);
    for (time, message) in (0_u64..).zip(&state.produced) {
        assert_eq!(message.queued_time, time);
        assert_eq!(message.completed_time, None);
        assert_eq!(
            message.custom_payload.as_deref(),
            (time == 2).then_some([7].as_slice())
        );
    }
    state.produced[0].source.push_str("-edited");
    state.produced[0].custom_payload = Some(vec![9]);
    assert_eq!(state.produced[1].source, "sender");
    assert!(state.produced[1].custom_payload.is_none());

    let Some(consumed) = simulation.consumed_for_agent("receiver") else {
        panic!("receiver must exist");
    };
    assert_eq!(consumed.len(), 4);
    for (time, message) in (0_u64..).zip(consumed) {
        assert_eq!(message.source, "sender");
        assert_eq!(message.destination, "receiver");
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
    let mutations: [fn(&mut Message); 6] = [
        |message| message.source.push_str("-edited"),
        |message| message.destination.push_str("-edited"),
        |message| message.custom_payload = Some(vec![]),
        |message| {
            message.interrupt = Some(simul::message::Interrupt::HaltSimulation("old".into()));
        },
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
        assert_eq!(message.source, "sender");
        assert_eq!(message.destination, "receiver");
        assert!(message.custom_payload.is_none());
        assert!(message.interrupt.is_none());
        assert_eq!(message.queued_time, 1);
        assert_eq!(message.completed_time, None);
    }
}
