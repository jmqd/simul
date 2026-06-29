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
