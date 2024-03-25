use crate::{message::*, DiscreteTime, SimulationState};
use rand::distributions::{Alphanumeric, DistString};
use rand::prelude::*;
use rand_distr::Poisson;
use std::collections::HashMap;
use std::collections::VecDeque;

/// Possible states an Agent can be in.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AgentState {
    /// The Agent is active; if asked to do something, it can.
    Active,
    /// The Agent is sleeping (or on cooldown) until a scheduled wakeup.
    AsleepUntil(DiscreteTime),
    /// The Agent is dead (inactive) and does nothing in this state.
    Dead,
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
#[derive(Debug, Clone)]
pub trait Agent {
    fn process(&mut self, simulation_state: SimulationState, msg: Message) -> Option<Vec<Message>>;

    fn id(&self) -> String;

    /// The queue of incoming Messages for the Agent.
    pub queue: VecDeque<Message>,
    /// The current state of the Agent.
    pub state: AgentState,
    /// Messages this agent produced.
    pub produced: Vec<Message>,
    /// Mesages this agent consumed.
    pub consumed: Vec<Message>,
}

impl Agent {
    pub fn push_message(&mut self, t: Message) {
        self.queue.push_back(t);
    }

    /// The most basic message processing routine: pop -> mark done -> push to consumed.
    pub fn pop_process_msg(&mut self, t: DiscreteTime) -> Option<Vec<Message>> {
        if let Some(msg) = self.queue.pop_front() {
            self.consumed.push(Message {
                completed_time: Some(t),
                ..msg
            });
        }

        None
    }
}

/// An agent that consumes on a Poisson-distributed periodicity.
pub fn poisson_distributed_consuming_agent(id: &str, dist: Poisson<f64>) -> impl Agent {
    Agent {
        consumption_fn: |a: &mut Agent, t: DiscreteTime| {
            // This agent will go to sleep for a "cooldown period",
            // as determined by a poisson distribution function.
            let cooldown_period = a
                .extensions
                .as_ref()?
                .period_poisson_distribution?
                .sample(&mut rand::thread_rng()) as u64;
            a.state = AgentState::AsleepUntil(t + cooldown_period);

            if let Some(message) = a.queue.pop_front() {
                a.consumed.push(Message {
                    completed_time: Some(t),
                    ..message
                });
            }

            None
        },
        name: name.into(),
        extensions: Some(AgentExtensions {
            period: None,
            target: None,
            period_poisson_distribution: Some(dist),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Given a poisson distribution for the production period,
/// returns an Agent that produces to Target with that frequency.
pub fn poisson_distributed_producing_agent(name: &str, dist: Poisson<f64>, target: &str) -> Agent {
    Agent {
        consumption_fn: |a: &mut Agent, t: DiscreteTime| {
            // This agent will go to sleep for a "cooldown period",
            // as determined by a poisson distribution function.
            let cooldown_period = a
                .extensions
                .as_ref()?
                .period_poisson_distribution?
                .sample(&mut rand::thread_rng()) as u64;
            a.state = AgentState::AsleepUntil(t + cooldown_period);

            // The agent produces some new work to its target t, since it is active.
            let msg = Message::new(t, &a.name, a.extensions.as_ref()?.target.as_ref()?);
            a.produced.push(msg.clone());
            Some(vec![msg])
        },
        name: name.to_owned(),
        extensions: Some(AgentExtensions {
            period: None,
            target: Some(target.to_owned()),
            period_poisson_distribution: Some(dist),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// A simple agent that produces messages on a period, directed to target.
pub fn periodic_producing_agent(name: &str, period: DiscreteTime, target: &str) -> Agent {
    Agent {
        consumption_fn: |a: &mut Agent, t: DiscreteTime| {
            if a.produced.last().is_none()
                || a.produced.last()?.queued_time + a.extensions.as_ref()?.period? >= t
            {
                Some(vec![Message {
                    queued_time: t,
                    source: a.name.to_owned(),
                    destination: a.extensions.as_ref()?.target.as_ref()?.clone(),
                    ..Default::default()
                }])
            } else {
                None
            }
        },
        name: name.to_owned(),
        extensions: Some(AgentExtensions {
            period: Some(period),
            target: Some(target.to_owned()),
            period_poisson_distribution: None,
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// A simple agent that consumes messages on a period with no side effects.
/// Period can be thought of the time to cosume 1 message.
pub fn periodic_consuming_agent(name: &str, period: DiscreteTime) -> Agent {
    Agent {
        consumption_fn: |a: &mut Agent, t: DiscreteTime| {
            if t >= a.extensions.as_ref()?.period?
                && (a.consumed.last().is_none()
                    || a.consumed.last()?.completed_time? + a.extensions.as_ref()?.period? <= t)
            {
                if let Some(message) = a.queue.pop_front() {
                    a.consumed.push(Message {
                        completed_time: Some(t),
                        ..message
                    });
                }
            }
            None
        },
        name: name.to_owned(),
        extensions: Some(AgentExtensions {
            period: Some(period),
            target: None,
            period_poisson_distribution: None,
            ..Default::default()
        }),
        ..Default::default()
    }
}
