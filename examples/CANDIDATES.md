# Example candidates

Ideas for future example programs. Each one is chosen because it exposes a
real-world dynamic that an averaged or spreadsheet model gets wrong: a feedback
loop, two capacity constraints that interact, or a system with more than one
stable state. Those are the dynamics a discrete-event simulation can show and a
closed-form estimate cannot.

Nothing here is implemented. Treat each entry as a brief for a possible
`examples/*.rs` program, and move an entry out of this file once it exists.

## How to read an entry

- **Dynamic.** The non-obvious behaviour the example is meant to make visible.
- **Agents.** A sketch of the topology on `Agent`, `Message`, `sleep_for`, and
  `Halt`.
- **Experiment.** What to vary and which `experiment` features apply.
- **Surprise.** The result a reader should not have guessed from averages.

## Worked brief: a billiards cafe

The most complete candidate. It uses every part of the `experiment` module and
connects to the nine-ball example through a tournament scheduler.

### Setting

Six tables, a bar and kitchen, a fixed periodic rent, hourly table pricing, and
staff on shifts. The question is not "what is the average revenue" but "what is
the probability this business is still solvent in twenty-four months, and which
policies move that probability".

### Dynamics

Six tables are servers in a queue with very long service times. The room is
either dead or saturated and rarely in between, and rent is fixed, so the whole
business is decided by what happens in the saturated hours. Four dynamics decide
that:

1. **Waiting is revenue, until it is not.** A party waiting at the bar buys
   drinks, so a short wait-list can beat an idle seventh table. A party that
   waits past its tolerance walks out, and a party that was turned away comes
   back less often. The optimal queue length is strictly positive and depends
   on the drink margin.
2. **Session length is endogenous.** Players stay longer when the room is empty
   and nobody minds. A peak-hour time cap raises table turns but can lower
   spend per party and lower return probability.
3. **The kitchen throttles the tables.** Food and drink are the high-margin
   line, but one bartender on a Saturday night caps order throughput.
   Under-staffing a busy night costs more than the wage it saves, and the loss
   shows up as lower spend per hour rather than as empty tables.
4. **Demand has memory.** Word of mouth grows the customer base and bad nights
   shrink it. This loop is what makes the model a complex adaptive system
   rather than an M/M/6 queue.

### Agents

One tick is one minute. The halt check is a two-year horizon or a ruin
condition, whichever comes first.

| Agent          | Mode      | Role                                                                                                                                                              |
| -------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Door           | Proactive | Non-homogeneous Poisson arrivals by day-part and weekday, with lambda scaled by Market. Emits one message per party with size and wait tolerance in the payload. |
| Host           | Reactive  | Holds the wait-list, implements balking and reneging with sleep timers, assigns parties to free tables, enforces reservation holds and the release policy.      |
| Table x6       | Reactive  | Sleeps for the session duration, then reports the freed table to Host and the table revenue to Ledger.                                                           |
| Bar and Kitchen| Reactive  | An M/M/c server for order messages from seated and waiting parties. `c` comes from the shift roster.                                                              |
| Calendar       | Proactive | Broadcasts day-part changes, league nights, holidays, and seasonality to Door.                                                                                   |
| Market         | Reactive  | Holds latent demand. Nudges it up on good visits and down on long waits and turn-aways.                                                                          |
| Ledger         | Reactive  | Accrues table time, orders, wages per shift, and rent on the first tick of each period. Tracks cash on hand and sends the halt interrupt on ruin.               |

Parties are message payloads rather than agents, because the engine does not
spawn agents mid-run. If party-level state is needed, pre-build a pool of party
agents and recycle them.

### Anchor number

Six tables open fourteen hours a day gives roughly 2,500 table-hours a month as
a hard ceiling, and realistic utilisation is a fraction of that. Dividing the
rent by billable table-hours gives the hourly rate needed from table time alone.
The model's job is to show how much of the rent has to come from the bar.

### Experiment

- Price schedule and the peak time cap are continuous dimensions, searched with
  `CrossEntropyOptimizer`.
- Staffing per shift, a seventh table versus more seating, and the reservation
  release policy are discrete candidates.
- Compare candidates under common random numbers (`common_rng`) so the same
  Saturday crowd hits every policy.
- Report probability of surviving twenty-four months with `BernoulliReducer`
  and its Wilson bound, and expected cash at month twenty-four with
  `ScalarReducer`.
- Use `FeasibleFirst` to require ruin probability under five percent, then
  maximise expected profit. That is the chance-constrained question a real
  operator asks.

### Friction

Parties and orders travel as variants of the model's message enum. A two-year run
at minute ticks is about a million ticks per replication, which makes the
next-event skip in `TODO.org` the main performance lever for this class of
model.

## Hospitality and consumer

### Mobile-order coffee shop

- **Dynamic.** App orders skip the line, so walk-ins see a long wait and leave
  even though the baristas are no slower.
- **Agents.** Walk-in Door, app Door, a Barista pool, a pickup shelf.
- **Experiment.** Mobile pickup lane and app-order throttling.
- **Surprise.** Throttling app orders can raise total revenue.

### Gym membership churn

- **Dynamic.** Peak-hour crowding drives churn, growth drives crowding, and
  pricing sets both. The same demand-with-memory loop as the cafe.
- **Agents.** Members as a pool, equipment stations as servers, a Market agent.
- **Experiment.** Price, off-peak discounts, and equipment count.

### EV charging station

- **Dynamic.** Long service times, overstay fees, and a grid power cap shared
  across chargers.
- **Agents.** Chargers as servers, a Grid agent that rations power.
- **Experiment.** Charger count, pricing, and overstay policy.
- **Surprise.** Adding chargers does nothing once the power cap binds.

## Software and distributed systems

### Retry storms and metastable failure

- **Dynamic.** Clients with retries, a server with capacity `c`, and a transient
  blip. The system has two stable states and never recovers from the bad one
  without load shedding or jittered backoff.
- **Agents.** A client pool, a server with a bounded queue, a fault injector.
- **Experiment.** Backoff policy, jitter, retry budget, and load shedding.
- **Surprise.** The system can stay down after the fault has cleared.

This is the most dramatic demonstration for an engineering audience and is the
strongest second candidate after the cafe.

### Merge queue throughput

- **Dynamic.** Batch size versus flaky-test rate with bisection on failure.
- **Agents.** A PR producer, the merge queue, a CI runner pool.
- **Experiment.** Batch size and flake rate.
- **Surprise.** Throughput is non-monotonic in batch size and the optimum moves
  with flakiness.

### Connection pool sizing

- **Dynamic.** Slow queries exhaust the pool, and a pool nested inside another
  pool deadlocks. Little's law in action.
- **Agents.** Request producer, a thread pool, a connection pool, a database.
- **Experiment.** Pool sizes and query latency distribution.
- **Surprise.** Smaller pools win.

### Tail latency in fan-out

- **Dynamic.** A request fans out to a hundred shards, and the P99 of the
  fan-out is roughly the P99.99 of one shard.
- **Agents.** A coordinator and a shard pool with a heavy-tailed service time.
- **Experiment.** Hedged requests and fan-out width.

### Engineering team with interrupts

- **Dynamic.** Extend `software_team.rs` with on-call interrupts, a code review
  queue, WIP limits, and onboarding drag. Cycle time explodes as utilisation
  approaches one.
- **Experiment.** WIP limit, reviewer count, and interrupt rate.

## Operations and logistics

### Beer distribution game

- **Dynamic.** Four tiers of agents with lead times and order-up-to policies.
  Demand variance amplifies upstream. The canonical bullwhip demonstration.
- **Agents.** Retailer, wholesaler, distributor, factory, and a Customer.
- **Experiment.** Order-up-to parameters and whether tiers share demand data.
- **Surprise.** Information sharing collapses the amplification.

### Emergency department boarding

- **Dynamic.** Triage priorities plus admitted patients stuck in ED beds because
  wards are full.
- **Agents.** Arrivals, triage, ED beds, wards with discharge timing.
- **Experiment.** ED bed count versus ward discharge policy.
- **Surprise.** Adding ED beds does nothing when ward discharge is the
  bottleneck.

### Call centre with abandonment

- **Dynamic.** Erlang A rather than Erlang C. Service level falls off a cliff at
  a specific staffing count.
- **Agents.** Callers with patience, an agent pool, a callback queue.
- **Experiment.** Staffing and callbacks.
- **Surprise.** Callbacks move the cliff.

### Bus bunching

- **Dynamic.** A late bus picks up more passengers and gets later. Positive
  feedback in its cleanest form.
- **Agents.** Buses, stops with passenger arrivals, a control point.
- **Experiment.** Holding at control points versus skip-stop.

### Job shop with a moving bottleneck

- **Dynamic.** Machines, routings, and setup times. Relieving one bottleneck
  exposes the next.
- **Experiment.** Capacity added at each station in turn, which is the Theory
  of Constraints as an experiment.

### Machine repairman

- **Dynamic.** A fleet with failures, repair crews, and spare-part stock.
- **Experiment.** Spares and crews trade off, and the optimum shifts with the
  failure distribution.

## Markets and two-sided systems

### Ride-hailing marketplace

- **Dynamic.** Low supply means long pickups, which wastes driver time and
  shrinks effective supply further.
- **Agents.** Drivers and riders as pools, a dispatcher, a pricing agent.
- **Experiment.** Surge pricing and dispatch radius.
- **Surprise.** Surge pricing is the stabiliser, not just a revenue tool.

### Limit order book

- **Dynamic.** Market makers, noise traders, and momentum traders. Spread,
  volatility, and flash-crash behaviour emerge from the agent mix.
- **Experiment.** Agent mix and market-maker inventory limits.

### Hiring funnel

- **Dynamic.** Interviewer capacity per stage, and offer acceptance decays with
  time in pipeline.
- **Experiment.** Interviewers per stage versus recruiters.
- **Surprise.** Faster loops beat more recruiters.

## Billiards-adjacent

### Tournament scheduling on the cafe's six tables

- **Dynamic.** A 32-player double elimination on six tables, with the nine-ball
  player model as the match engine.
- **Agents.** A Scheduler agent on top of `nine_ball_simulation.rs` players.
- **Experiment.** Format, race length, and table count.
- **Questions.** Does the final run past closing? How much walk-in revenue does
  the tournament displace? Which format most often crowns the strongest player?

### League night economics

- **Dynamic.** Four tables at a discount with guaranteed bar spend versus
  walk-ins crowded out.
- **Experiment.** Drop into the cafe model as a Calendar event and let the
  experiment decide whether the league pays.

## Suggested order

1. The billiards cafe. It exercises the whole `experiment` module, connects to
   the nine-ball example through the tournament scheduler, and the
   ruin-probability output is a question a real operator would pay to answer.
2. Retry storms. A two-state system that fails to recover is the clearest
   argument for discrete-event simulation over averages.
3. Bus bunching or the beer distribution game. Both are short, visual, and
   well known, which makes them good README material.
