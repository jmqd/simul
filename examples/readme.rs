//! An example of a simple simulation, featured in the README.md.
use simul::agent::{periodic_consumer, periodic_producer};
use simul::{Halt, Simulation, SimulationError};

/// Example of a minimal, simple Simulation that can be executed.
fn main() -> Result<(), SimulationError> {
    // Runs a simulation with a producer that produces work at every tick of
    // discrete time (period=1), and a consumer that cannot keep up (can only
    // process that work every second tick).
    //
    // The halt condition tells the simulation when it is finished: here,
    // "when the simulation is 10 ticks old, we're done."
    let mut simulation = Simulation::builder(Halt::AtTime(10))
        // We pass in two agents:
        //   `producer`: produces a message to the consumer every tick
        //   `consumer`: consumes w/ no side effects every second tick
        // Agents are powerful, and you can pass-in custom implementations here.
        .agents([
            periodic_producer("producer", 1, "consumer"),
            periodic_consumer("consumer", 2),
        ])
        .build()?;

    // For massive simulations, you might block on this line for a long time.
    simulation.run();

    // Post-simulation, you can do analytics on the stored metrics, data, etc.
    simulation
        .agents()
        .iter()
        .for_each(|agent| println!("{agent:#?}"));
    Ok(())
}
