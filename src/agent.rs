use crate::{message::*, DiscreteTime, SimulationState};
use dyn_clone::DynClone;
use rand::prelude::*;
use rand_distr::Poisson;
use simul_macro::agent;
use std::collections::VecDeque;

/// Possible states an Agent can be in.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
pub enum AgentMode {
    /// The Agent is active; process() is called on every tick of the simulation.
    Proactive,

    /// The Agent is reactive; process() is called when this agent has a message.
    Reactive,

    /// The Agent is sleeping (or on cooldown) until a scheduled wakeup.
    AsleepUntil(DiscreteTime),

    /// The Agent is dead (inactive) and does nothing in this state.
    Dead,
}

#[derive(Debug, Clone)]
pub struct AgentState {
    pub mode: AgentMode,
    pub wake_mode: AgentMode,
    pub id: String,
    /// The queue of incoming Messages for the Agent.
    pub queue: VecDeque<Message>,
    pub consumed: Vec<Message>,
    pub produced: Vec<Message>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            mode: AgentMode::Dead,
            wake_mode: AgentMode::Dead,
            id: "".to_string(),
            queue: VecDeque::new(),
            consumed: vec![],
            produced: vec![],
        }
    }
}

/// Internal simulation impl for an agent; this implementation is expected to
/// be the same for most Agents.
pub trait AgentCommon {
    /// The state of the agent.
    fn state(&self) -> &AgentState;

    fn state_mut(&mut self) -> &mut AgentState;

    fn push_message(&mut self, msg: Message) {
        self.state_mut().queue.push_back(msg);
    }
}

/// The bread and butter of the Simulation -- the Agent.
/// In a Complex Adaptive System (CAS), an Adaptive Agent does things and
/// interacts with the Simulation, itself, and other Agents.
///
/// Some examples of what an Agent might be:
/// * Barista at a coffee shop.
/// * Stoplight.
/// * Driver in traffic.
/// * A single-celled organism.
/// * A player in a game.
pub trait Agent: std::fmt::Debug + DynClone + AgentCommon {
    /// The main action an agent performs; it processes message that come in to it.
    /// An agent can affect other agents by returning messages here.
    fn process(&mut self, simulation_state: SimulationState, msg: &Message)
        -> Option<Vec<Message>>;

    /// For annealing experiments, you may implement a cost function for the agent.
    /// For example, a periodic consuming agent has cost implented equal to its period.
    fn cost(&self) -> i64 {
        0
    }
}

dyn_clone::clone_trait_object!(Agent);

/// An agent that processes on a Poisson-distributed periodicity.
pub fn poisson_distributed_consuming_agent<T>(id: T, dist: Poisson<f64>) -> impl Agent
where
    T: Into<String>,
{
    #[agent]
    struct PoissonAgent {
        period: Poisson<f64>,
    }

    impl Agent for PoissonAgent {
        fn process(
            &mut self,
            simulation_state: SimulationState,
            _msg: &Message,
        ) -> Option<Vec<Message>> {
            // This agent will go to sleep for a "cooldown period",
            // as determined by a poisson distribution function.
            let cooldown_period = self.period.sample(&mut rand::thread_rng()) as u64;
            self.state.mode = AgentMode::AsleepUntil(simulation_state.time + cooldown_period);
            None
        }
    }

    PoissonAgent {
        period: dist,
        state: AgentState {
            mode: AgentMode::Reactive,
            wake_mode: AgentMode::Reactive,
            id: id.into(),
            ..Default::default()
        },
    }
}

/// Given a poisson distribution for the production period,
/// returns an Agent that produces to Target with that frequency.
pub fn poisson_distributed_producing_agent<T>(
    id: T,
    dist: Poisson<f64>,
    target: T,
) -> Box<dyn Agent>
where
    T: Into<String>,
{
    #[agent]
    struct PoissonAgent {
        period: Poisson<f64>,
        target: String,
    }

    impl Agent for PoissonAgent {
        fn process(
            &mut self,
            simulation_state: SimulationState,
            _msg: &Message,
        ) -> Option<Vec<Message>> {
            // This agent will go to sleep for a "cooldown period",
            // as determined by a poisson distribution function.
            let cooldown_period = self.period.sample(&mut rand::thread_rng()) as u64;

            self.state.mode = AgentMode::AsleepUntil(simulation_state.time + cooldown_period);

            Some(vec![Message::new(
                simulation_state.time,
                self.state.id.clone(),
                self.target.clone(),
            )])
        }
    }

    Box::new(PoissonAgent {
        period: dist,
        target: target.into(),
        state: AgentState {
            id: id.into(),
            mode: AgentMode::Proactive,
            wake_mode: AgentMode::Proactive,
            ..Default::default()
        },
    })
}

/// A simple agent that produces messages on a period, directed to target.
pub fn periodic_producing_agent<T>(id: T, period: DiscreteTime, target: T) -> Box<dyn Agent>
where
    T: Into<String>,
{
    #[agent]
    struct PeriodicProducer {
        period: DiscreteTime,
        target: String,
    }

    impl Agent for PeriodicProducer {
        fn cost(&self) -> i64 {
            -(self.period as i64)
        }

        fn process(
            &mut self,
            simulation_state: SimulationState,
            _msg: &Message,
        ) -> Option<Vec<Message>> {
            self.state.mode = AgentMode::AsleepUntil(simulation_state.time + self.period);

            Some(vec![Message {
                queued_time: simulation_state.time,
                source: self.state.id.to_owned(),
                destination: self.target.to_owned(),
                ..Default::default()
            }])
        }
    }

    Box::new(PeriodicProducer {
        period,
        target: target.into(),
        state: AgentState {
            mode: AgentMode::Proactive,
            wake_mode: AgentMode::Proactive,
            id: id.into(),
            ..Default::default()
        },
    })
}

/// A simple agent that consumes messages on a period with no side effects.
/// Period can be thought of the time to consume 1 message.
pub fn periodic_consuming_agent<T>(id: T, period: DiscreteTime) -> Box<dyn Agent>
where
    T: Into<String>,
{
    #[agent]
    struct PeriodicConsumer {
        period: DiscreteTime,
    }

    impl Agent for PeriodicConsumer {
        fn cost(&self) -> i64 {
            -(self.period as i64)
        }

        fn process(
            &mut self,
            simulation_state: SimulationState,
            msg: &Message,
        ) -> Option<Vec<Message>> {
            self.state.mode = AgentMode::AsleepUntil(simulation_state.time + self.period);

            self.state.consumed.push(Message {
                completed_time: Some(simulation_state.time),
                ..msg.clone()
            });

            None
        }
    }

    Box::new(PeriodicConsumer {
        period,
        state: AgentState {
            mode: AgentMode::AsleepUntil(period),
            wake_mode: AgentMode::Reactive,
            id: id.into(),
            ..Default::default()
        },
    })
}
