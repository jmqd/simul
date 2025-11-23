use simul::agent::{periodic_consumer, periodic_producer};
use simul::Simulation;
use simul::SimulationParameters;

/// Example of a minimal, simple Simulation that can be executed.
fn main() {
    // Runs a simulation with a producer that produces work at every tick of
    // discrete time (period=1), and a consumer that cannot keep up (can only
    // process that work every third tick).
    let mut simulation = Simulation::new(SimulationParameters {
        // We pass in two agents:
        //   `producer`: produces a message to the consumer every tick
        //   `consumer`: consumes w/ no side effects every second tick
        // Agents are powerful, and you can pass-in custom implementations here.
        agent_initializers: vec![
            periodic_producer("producer", 1, "consumer"),
            periodic_consumer("consumer", 2),
        ],

        // We pass in a halt condition so the simulation knows when it is finished.
        // In this case, it is "when the simulation is 10 ticks old, we're done."
        halt_check: |s: &Simulation| s.time() == 10,

        ..Default::default()
    });

    // For massive simulations, you might block on this line for a long time.
    simulation.run();

    // Post-simulation, you can do analytics on the stored metrics, data, etc.
    simulation
        .agents()
        .iter()
        .for_each(|agent| println!("{agent:#?}"));
}
