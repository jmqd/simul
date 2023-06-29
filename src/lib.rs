pub mod agent;
pub mod ticket;

use agent::*;
use log::{debug, info};
use std::collections::HashMap;
use ticket::*;

/// The current state of the simultion.
pub enum SimulationState {
    /// The Simulation has only been constructed.
    Constructed,
    /// The Simulation is actively simulating.
    Running,
    /// The Simulation successfully reached the halt condition.
    Completed,
    /// The Simulation catastrophically crashed.
    Failed,
}

/// A Simulation struct holds all the state for any given simulation.
///
/// A Simulation is an engine that advances through its own discrete definition
/// of time and has a collection of Agents that live in its domain.
///
/// At each moment in time, it asks each Agent whether it has any action to
/// perform.
///
/// The Simulation engine uses a concept of `Tickets` to communicate between
/// agents. Agents can receive tickets and send tickets to other Agents.
pub struct Simulation {
    /// The agents within the simulation, e.g. adaptive agents.
    /// See here: https://authors.library.caltech.edu/60491/1/MGM%20113.pdf
    pub agents: Vec<Agent>,
    /// A halt check function: given the state of the Simulation determine halt or not.
    pub halt_check: fn(&Simulation) -> bool,
    /// The current discrete time of the Simulation.
    pub time: u64,
    /// Whether to record metrics on queue depths. Takes space.
    pub record_queue_depths: bool,
    /// Space to store queue depth metrics. Maps from Agent to a Vec<Time, Depth>
    pub queue_depth_metrics: HashMap<String, Vec<usize>>,
    /// The state of the Simulation.
    pub state: SimulationState,
}

#[allow(dead_code)]
impl Simulation {
    pub fn new(
        agents: Vec<Agent>,
        beginning_of_time: u64,
        record_queue_depths: bool,
        halt_check: fn(&Simulation) -> bool,
    ) -> Simulation {
        Simulation {
            state: SimulationState::Constructed,
            queue_depth_metrics: agents.iter().map(|a| (a.name.to_owned(), vec![])).collect(),
            agents: agents.into_iter().collect(),
            halt_check,
            time: beginning_of_time,
            record_queue_depths,
        }
    }

    /// Returns the consumed tickets for a given Agent during the Simulation.
    pub fn consumed_for_agent(&self, name: &str) -> Option<Vec<Ticket>> {
        let agent = self.agents.iter().find(|a| a.name == name)?;
        Some(agent.consumed.clone())
    }

    /// Returns the produced tickets for a given Agent during the Simulation.
    pub fn produced_for_agent(&self, name: &str) -> Option<Vec<Ticket>> {
        let agent = self.agents.iter().find(|a| a.name == name)?;
        Some(agent.produced.clone())
    }

    /// Returns the queue depth timeseries for a given Agent during the Simulation.
    pub fn queue_depth_metrics(&self, agent_name: &str) -> Option<Vec<usize>> {
        self.queue_depth_metrics.get(agent_name).cloned()
    }

    /// An internal function used to wakeup sleeping Agents due to wake.
    fn wakeup_agents_scheduled_to_wakeup_now(&mut self) {
        for agent in self.agents.iter_mut() {
            match agent.state {
                AgentState::AsleepUntil(scheduled_wakeup) => {
                    if self.time >= scheduled_wakeup {
                        agent.state = AgentState::Active;
                    }
                }
                _ => (),
            }
        }
    }

    /// Runs the simulation. This should only be called after adding all the beginning state.
    pub fn run(&mut self) {
        self.state = SimulationState::Running;

        while !(self.halt_check)(self) {
            debug!("Running next tick of simulation at time {}", self.time);
            let mut new_tickets = vec![];
            self.wakeup_agents_scheduled_to_wakeup_now();
            for mut agent in self.agents.iter_mut() {
                self.queue_depth_metrics
                    .get_mut(&agent.name)
                    .expect("Failed to find agent in metrics")
                    .push(agent.queue.len());
                match agent.state {
                    AgentState::Active => match (agent.consumption_fn)(&mut agent, self.time) {
                        Some(tickets) => {
                            new_tickets.extend(tickets);
                        }
                        None => debug!("No tickets produced."),
                    },
                    AgentState::Dead | AgentState::AsleepUntil(_) => {}
                }
            }

            while !new_tickets.is_empty() {
                let t = new_tickets.pop();
                for agent in self.agents.iter_mut() {
                    if agent.name == t.clone().unwrap().destination {
                        agent.push_ticket(t.clone().unwrap());
                    }

                    if agent.name == t.clone().unwrap().source {
                        agent.produced.push(t.clone().unwrap());
                    }
                }
            }

            debug!("Finished this tick; incrementing time.");
            self.time += 1;
        }

        self.state = SimulationState::Completed;

        let queue_len_stats = self.calc_queue_len_statistics();
        let consumed_len_stats = self.calc_consumed_len_statistics();
        let avg_wait_stats = self.calc_avg_wait_statistics();
        let produced_len_stats = self.calc_produced_len_statistics();

        info!("Queues: {:?}", queue_len_stats);
        info!("Consumed: {:?}", consumed_len_stats);
        info!("Produced: {:?}", produced_len_stats);
        info!("Average processing time: {:?}", avg_wait_stats);
    }

    /// A helper to calculate the average waiting time to process items.
    /// Note: This function will likely go away; it is an artifact of prototyping.
    pub fn calc_avg_wait_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();
        for agent in self.agents.iter() {
            if agent.consumed.len() == 0 {
                continue;
            }

            let mut sum_of_times: u64 = 0;
            for completed in agent.consumed.iter() {
                sum_of_times += completed.completed_time.unwrap() - completed.queued_time;
            }

            data.insert(
                agent.name.clone(),
                sum_of_times as usize / agent.consumed.len(),
            );
        }

        return data;
    }

    /// Calculates the statistics of queue lengths.
    /// Mostly useful for checking which agents still have queues of work after halting.
    pub fn calc_queue_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(agent.name.clone(), agent.queue.len());
        }

        return data;
    }

    /// Calculates the length of the consumed tickets for each Agent.
    pub fn calc_consumed_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(agent.name.clone(), agent.consumed.len());
        }

        return data;
    }

    /// Calculates the length of the produced tickets for each Agent.
    pub fn calc_produced_len_statistics(&self) -> HashMap<String, usize> {
        let mut data = HashMap::new();

        for agent in self.agents.iter() {
            data.insert(agent.name.clone(), agent.produced.len());
        }

        return data;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Poisson;
    use std::collections::VecDeque;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn basic_periodic_test() {
        init();
        let mut simulation = Simulation::new(
            vec![
                periodic_producing_agent("producer", 1, "consumer"),
                periodic_consuming_agent("consumer", 1),
            ],
            0,
            false,
            |s: &Simulation| s.time == 5,
        );
        simulation.run();
        let produced_stats = simulation.calc_produced_len_statistics();
        assert_eq!(produced_stats.get("producer"), Some(&5));
        assert_eq!(produced_stats.get("consumer"), Some(&0));

        let consumed_stats = simulation.calc_consumed_len_statistics();
        assert_eq!(consumed_stats.get("producer"), Some(&0));
        assert_eq!(consumed_stats.get("consumer"), Some(&4));
    }

    #[test]
    fn starbucks_clerk() {
        init();
        let mut simulation = Simulation::new(
            vec![
                Agent {
                    queue: VecDeque::with_capacity(8),
                    state: AgentState::Active,
                    name: "Starbucks Clerk".to_owned(),
                    consumed: vec![],
                    produced: vec![],
                    consumption_fn: |a: &mut Agent, t: u64| {
                        debug!("{} looking for a customer.", a.name);
                        if let Some(last) = a.consumed.last() {
                            if last.completed_time.unwrap() + 60 > t {
                                debug!("Sorry, we're still serving the last customer.");
                                return None;
                            }
                        }

                        if let Some(ticket) = a.queue.pop_front() {
                            if ticket.queued_time + 100 > t {
                                debug!("Still making your coffee, sorry!");
                                a.queue.push_front(ticket);
                                return None;
                            }

                            debug!("Serviced a customer!");
                            a.consumed.push(Ticket {
                                completed_time: Some(t),
                                ..ticket
                            });
                        }
                        return None;
                    },
                    common_traits: None,
                },
                poisson_distributed_producing_agent(
                    "Starbucks Customers",
                    Poisson::new(80.0).unwrap(),
                    "Starbucks Clerk",
                ),
            ],
            1,
            false,
            |s: &Simulation| s.time > 500,
        );
        simulation.run();
        assert_eq!(Some(simulation).is_some(), true);
    }
}
