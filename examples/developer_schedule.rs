use simul::agent::{Agent, AgentContext, AgentInitializer, AgentMode, AgentOptions};
use simul::message::Message;
use simul::{Halt, Simulation, SimulationParameters};
use std::sync::Arc;

/// Calculates productivity from each contiguous block of uninterrupted maker time.
#[derive(Debug, Clone)]
struct Developer {
    /// A developer's schedule for an 8-hour day is from 9am - 6pm. (Lunch is
    /// generally assumed at 12pm.)  This array then indicates 30-minute
    /// intervals, with true = meeting/interruption, false = maker time.
    /// Every 18 steps of the simulation will simulate a full work day for the
    /// developer.
    schedule: [bool; 18],
}

impl Agent for Developer {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        let time_of_day = usize::try_from(ctx.time() % 18).unwrap_or_default();
        let time_block = self
            .schedule
            .iter()
            .skip(time_of_day)
            .take_while(|&&slot| !slot)
            .count();
        // Being in a meeting or having only 30 minutes is 0 productivity.
        // After that, productivity is very slightly superlinear with available time.
        let productivity = match time_block {
            0..=1 => 0,
            2 => 1,
            3 => 2,
            4 => 4,
            5..=8 => time_block + 1,
            9.. => time_block + 2,
        };

        ctx.sleep_for(u64::try_from(time_block).unwrap_or(18));
        ctx.send(
            "productivity_recorder",
            Some(Arc::from(productivity.to_le_bytes())),
        );
    }
}

/// Collects and reports the productivity calculated for each maker-time block.
#[derive(Debug, Clone, Default)]
struct ProductivityRecorder {
    /// Productivity accumulated over the simulated work days.
    total_productivity: usize,
}

impl Agent for ProductivityRecorder {
    fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
        let Some(payload) = msg.custom_payload.as_deref() else {
            return;
        };
        let Ok(encoded) = <[u8; std::mem::size_of::<usize>()]>::try_from(payload) else {
            return;
        };
        let productivity = usize::from_le_bytes(encoded);
        self.total_productivity += productivity;
        println!(
            "tick {}: recorded {productivity} productivity (total {})",
            ctx.time(),
            self.total_productivity
        );
    }
}

fn main() {
    let mut simulation = Simulation::new(SimulationParameters {
        agent_initializers: vec![
            AgentInitializer {
                agent: Box::new(Developer {
                    schedule: [
                        false, false, false, false, false, false, false, false, false, false,
                        false, false, false, false, false, false, false, false,
                    ],
                }),
                options: AgentOptions {
                    initial_mode: AgentMode::Proactive,
                    wake_mode: AgentMode::Proactive,
                    name: "developer".to_owned(),
                    ..Default::default()
                },
            },
            AgentInitializer {
                agent: Box::new(ProductivityRecorder::default()),
                options: AgentOptions {
                    initial_mode: AgentMode::Reactive,
                    wake_mode: AgentMode::Reactive,
                    name: "productivity_recorder".to_owned(),
                    ..Default::default()
                },
            },
        ],
        halt: Halt::AtTime(128),
        ..Default::default()
    });

    simulation.run();
}
