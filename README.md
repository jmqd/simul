`simul` is a discrete-event simulation library for running high-level
simulations of real-world problems and for running simulated experiments.

`simul` is a *discrete-event simulator* using *incremental time progression*,
with a FIFO message queue per agent for interactions between agents. It also
supports deterministic typed experiments and seeded searches over
caller-defined candidate spaces.

Use-cases:
- [Discrete-event simulation](https://en.wikipedia.org/wiki/Discrete-event_simulation)
- [Complex adaptive systems](https://authors.library.caltech.edu/60491/1/MGM%20113.pdf)
- [Random search](https://en.wikipedia.org/wiki/Random_search)
- [Job-shop scheduling](https://en.wikipedia.org/wiki/Job-shop_scheduling)
- [Birth-death processes](https://en.wikipedia.org/wiki/Birth%E2%80%93death_process)
- [Computer experiments](https://en.wikipedia.org/wiki/Computer_experiment)
- Other: simulating logistics, operations research problems, running experiments
  to approximate a global optimum, simulating queueing systems, distributed
  systems, performance engineering/analysis, and so on.

# Usage

> **Warning**
>
> Experimental and unstable. Almost all APIs are expected to change.

- For some examples, see the `examples` subdirectory.
- For ideas not yet implemented, see [`examples/CANDIDATES.md`](./examples/CANDIDATES.md).
- For use cases where your agents need their own custom state, define a struct,
  implement `Agent`, and pass your agents into `Simulation` via constructing
  `AgentInitializers`.

## Typed experiments (source checkout)

`experiment` is independent of the agent engine. The consolidated API below is
available in this checkout; it has not been published as a new crates.io release.

- Schedule explicit `TrialSpec` values with `CandidateId` and `ReplicationId`.
  `run_serial_bounded` rejects duplicate identities and oversized batches before
  evaluation, and returns records ordered by `TrialId`.
  Batch independent candidates in one runner call. Pass `&candidate` to
  `FixedReplications::trials` to share large inputs without cloning their contents.
- Replay uses the V1 keyed BLAKE3/ChaCha12 protocol. Retain `ReplayKey`,
  candidate inputs, and evaluator configuration. Named common-random streams
  share noise across candidates; `ValidationHoldout` separates validation noise.
- With `parallel`, `run_parallel` uses a dedicated bounded Rayon pool and one
  mutable workspace per contiguous shard. Its factory is
  `FnMut() -> Result<W, WE>`, called on the caller thread. Explicit batch/output
  storage and the pool are prepared before factories, and every factory succeeds
  before evaluation starts. Requested workers are capped by work and available
  parallelism. Evaluators must use only their inputs, context, immutable
  configuration, and unobservable scratch for worker-count-independent results.
  Shards write directly into disjoint ranges of one final result allocation;
  initialization guards drop partial results on failure without a second buffer.
  For repeated batches, construct `ParallelExecutor::new(config)` once and call
  `executor.run(seed, trials, init_worker, evaluate)`. Worker capacity is fixed at
  construction; each batch initializes fresh workspaces on its caller thread.
  Empty batches invoke no callbacks, and failed batches do not invalidate the
  executor. Prefer serial execution when individual batches are very cheap.
- Setup failures are run-level errors; evaluator `Err` values remain ordered
  trial records. With unwinding enabled, callback panics discard partial output
  and return a run-level error without retries. This does not roll back callback
  side effects, and cannot catch panics in `panic = "abort"` builds.
- `aggregate_scalar` and `aggregate_bernoulli` apply explicit failure policies.
  Unrepresentable scalar arithmetic returns `ArithmeticOverflow`, never a
  successful non-finite summary.
- `search::SeededRandomSearch` provides tokenized ask/tell search.
  `CrossEntropyOptimizer` and `simulated_annealing_search_with_rng` remain
  engine-independent optimizer alternatives; aggregate noisy trials inside their
  objective rather than ranking a lucky single run.
- To evaluate a `Simulation` in a trial, build it with
  `Simulation::builder(..).trial(context)`. Its seed comes from the trial's
  common-random stream, so candidates in one common-random group share their
  agents' draws replication by replication; agent streams are keyed by name,
  so the pairing survives candidates that add or reorder agents.

See `examples/simple_experiment.rs` for a complete typed experiment.

## Basic usage

``` toml
[dependencies]
simul = "0.5.1"
```

``` rust
use simul::agent::{periodic_consumer, periodic_producer};
use simul::{Halt, Simulation, SimulationError};

/// Example of a minimal, simple Simulation that can be executed.
fn main() -> Result<(), SimulationError> {
    // Runs a simulation with a producer that produces work at every tick of
    // discrete time (period=1), and a consumer that cannot keep up (can only
    // process that work every second tick).
    //
    // The halt condition tells the simulation when it is finished: here,
    // "when the simulation is 10 ticks old, we're done." The messages carry
    // no payload, so their type is `()`.
    let mut simulation = Simulation::<()>::builder(Halt::AtTime(10))
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
```

## Simulation Concepts / Abstraction

- A simulation is a collection of `Agents` that interact with each other via
`Messages`.
- The simulation keeps a discrete time (u64) which is incremented on each tick
  of the Simulation.
- What an `Agent` does at each tick of the simulation is provided by you in its
  `on_tick()` and `on_message()` methods.
- `Agents` must have a unique name. Each also has a dense `AgentId`: its
  position in `agent_initializers`.
- If an `Agent` wants to interact with another `Agent`, it can send a `Message`
  via the `&mut ctx: AgentContext` passed into `on_tick` and `on_message`.
  `ctx.send(name, payload)` resolves a name; `ctx.send_to(id, payload)` skips
  the lookup, and `ctx.agent_id(name)` resolves a peer once for reuse.
- `ctx.send_after(delay, id, payload)` delivers a message `delay` ticks later
  instead of at the end of the tick. Use it for latency and, sent to `ctx.id()`,
  for timeouts: an agent waiting on a timer stays idle instead of polling every
  tick, so quiet ticks are still skipped. Messages that have not arrived when a
  run halts stay in `Simulation::in_flight` for the next run.

The simulation runs all the logic of calling `on_tick` and `on_message`,
distributing messages, tracking metrics, incrementing time, and when to halt.
A `Simulation` is finished when its `Halt` condition holds (`Halt::AtTime(t)`,
`Halt::Never`, or `Halt::when(predicate)`), or after the tick in which an
`Agent` calls `ctx.send_halt_interrupt(reason)`.

Messages identify agents by `AgentId`: `source` is the sender (absent for an
initial queue entry) and `destination` the addressed agent (absent when the
target matched no agent, in which case the message is counted but not
delivered). `Simulation::agent(id)` and `Simulation::agent_id(name)` translate
between identities and agents. Create initial messages with
`Message::new(time, payload)`, not struct literals.

Every agent counts the messages it consumed and produced
(`SimulationAgent::consumed_count`, `produced_count`) and their total wait, so
the `calc_*_statistics` methods and halt conditions on counts need nothing
else. To keep the messages themselves, build with `.message_history(true)`;
`SimulationAgent::consumed` and `produced` then return them in order. History
is off by default because it holds every message for the whole run, and the
history accessors panic when it is off rather than look empty.

Keep results in the agents themselves. After a run,
`simulation.find_by_name("door").and_then(|agent| agent.downcast_ref::<Door>())`
returns the agent as its own type, or `None` for any other type, so there is no
need to share `Rc<Cell<_>>` or locks with agents to get answers out.

Messages are typed. A simulation picks one message type `M`, usually an enum
of everything its agents say to each other, and every agent implements
`Agent<M>` and sends `M` values with `ctx.send(target, payload)`, so handlers
match on `message.payload` instead of decoding bytes. `()` is the default for
simulations whose messages carry nothing; name it with
`Simulation::<()>::builder(..)` when only built-in agents are present, since
they work with any `M` (producers send `M::default()`). A payload moves into
the receiver's queue without copying, and shared data travels as `M = Arc<T>`.
Recording message history needs `M: Clone`; running does not.

Simulations with more than 64 agents and no asleep-cycle metric use an ordered
ready bitmap and wake heap during sparse phases, switching to dense scans as
activity rises. Smaller simulations and runs with the asleep-cycle metric keep
dense scans. Callback order, end-of-tick delivery, and halt checks match dense
scans. With `Halt::AtTime` or `Halt::Never` and no asleep-cycle metric, the
engine also jumps over ticks in which no agent can act, to the next wake-up,
delayed arrival, or the halt time; `Halt::when` predicates see every tick, so
they keep every tick.

## Poisson-distributed example w/ Plotting

Here's an example of an outputted graph from a simulation run. In this
simulation, we show the average waiting time of customers in a line at a
cafe. The customers arrive at a Poisson-distributed arrival rate
(`lambda<-60.0`) and a Poisson-distributed coffee-serving rate with the
same distribution.

This simulation maps to the real world by assuming one tick of
discrete-simulation time is equal to one second.

Basically, the barista serves coffees at around 60 seconds per drink and
the customers arrive at about the same rate, both modeled by a
stochastic Poisson generator.

This simulation has a `halt_check` condition of the simulation's time
being equal to `60*60*12`, representing a full 12-hour day of the cafe
being open.

![](./readme-assets/cafe-example-queued-durations.png)

# Contributing

Issues, bugs, features are tracked in TODO.org
