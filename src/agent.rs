use crate::message::*;
use rand::prelude::*;
use rand_distr::Poisson;
use std::collections::VecDeque;

/// Possible states an Agent can be in.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AgentState {
    /// The Agent is active; if asked to do something, it can.
    Active,
    /// The Agent is sleeping (or on cooldown) until a scheduled wakeup.
    AsleepUntil(u64),
    /// The Agent is dead (inactive) and does nothing in this state.
    Dead,
}

/// Sort of like extension-properties for an Agent that are sometimes used.
/// This is hacky and it should be removed/refactored away.
#[derive(Default, Debug, Clone)]
pub struct AgentExtensions {
    pub period: Option<u64>,
    pub target: Option<String>,
    pub period_poisson_distribution: Option<Poisson<f64>>,
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
#[derive(Debug, Clone)]
pub struct Agent {
    /// The queue of incoming Messages for the Agent.
    pub queue: VecDeque<Message>,
    /// The current state of the Agent.
    pub state: AgentState,
    /// Messages this agent produced.
    pub produced: Vec<Message>,
    /// Mesages this agent consumed.
    pub consumed: Vec<Message>,
    /// This function is called upon Messages when popped from the incoming queue.
    pub consumption_fn: fn(&mut Agent, u64) -> Option<Vec<Message>>,
    /// The name of the Agent. Should be unique.
    /// Note: This field is a wart in the abstraction. Ideally it is replaced with a better design.
    pub name: String,
    /// A bag of common extensions to Agent behavior.
    /// Note: This field is a wart in the abstraction. Ideally it is replaced with a better design.
    pub extensions: Option<AgentExtensions>,
}

impl Agent {
    pub fn push_message(&mut self, t: Message) {
        self.queue.push_back(t);
    }
}

/// An agent that consumes on a Poisson-distributed periodicity.
pub fn poisson_distributed_consuming_agent(name: &str, dist: Poisson<f64>) -> Agent {
    Agent {
        queue: VecDeque::with_capacity(8),
        state: AgentState::Active,
        produced: vec![],
        consumed: vec![],
        consumption_fn: |a: &mut Agent, t: u64| {
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
        }),
    }
}

/// Given a poisson distribution for the production period,
/// returns an Agent that produces to Target with that frequency.
pub fn poisson_distributed_producing_agent(name: &str, dist: Poisson<f64>, target: &str) -> Agent {
    Agent {
        queue: VecDeque::with_capacity(8),
        state: AgentState::Active,
        produced: vec![],
        consumed: vec![],
        consumption_fn: |a: &mut Agent, t: u64| {
            // This agent will go to sleep for a "cooldown period",
            // as determined by a poisson distribution function.
            let cooldown_period = a
                .extensions
                .as_ref()?
                .period_poisson_distribution?
                .sample(&mut rand::thread_rng()) as u64;
            a.state = AgentState::AsleepUntil(t + cooldown_period);

            // The agent produces some new work to its target now, since it is active.
            let t = Message::new(t, &a.name, a.extensions.as_ref()?.target.as_ref()?);
            a.produced.push(t.clone());
            Some(vec![t])
        },
        name: name.to_owned(),
        extensions: Some(AgentExtensions {
            period: None,
            target: Some(target.to_owned()),
            period_poisson_distribution: Some(dist),
        }),
    }
}

/// A simple agent that produces messages on a period, directed to target.
pub fn periodic_producing_agent(name: &str, period: u64, target: &str) -> Agent {
    Agent {
        queue: VecDeque::with_capacity(8),
        state: AgentState::Active,
        produced: vec![],
        consumed: vec![],
        consumption_fn: |a: &mut Agent, t: u64| {
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
        }),
    }
}

/// A simple agent that consumes messages on a period with no side effects.
/// Period can be thought of the time to cosume 1 message.
pub fn periodic_consuming_agent(name: &str, period: u64) -> Agent {
    Agent {
        queue: VecDeque::with_capacity(8),
        state: AgentState::Active,
        produced: vec![],
        consumed: vec![],
        consumption_fn: |a: &mut Agent, t: u64| {
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
        }),
    }
}
