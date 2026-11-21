# `simul` Python Bindings

`PyO3` / `maturin` bindings exposing `simul`'s deterministic discrete-event
engine, step/yield external-actor control loop, counterfactual snapshot
cloning/reseeding, tail-risk statistics (P99/CVaR), paired CRN comparisons,
space-filling DoE samplers (Latin Hypercube & 32D Joe & Kuo Sobol), mixed
discrete/continuous cross-entropy optimization, Pareto frontier archives, and
Perfetto / JSONL trace export.

## Build & Install

With [maturin](https://www.maturin.rs/):

```bash
cd python
maturin develop --release
```

Or with Cargo directly (e.g., for running the bundled unit tests on macOS):

```bash
cargo build --manifest-path python/Cargo.toml --release
cp python/target/release/libsimul.dylib python/simul.so
PYTHONPATH=python python3 -m unittest discover -s python/tests -v
```

## Quick Example (Step/Yield + Counterfactual Rollouts)

```python
import simul

class PlannerAgent:
    def __init__(self, gain=0.0):
        self.gain = gain
        self.reward = 0.0

    def on_tick(self, ctx):
        self.reward += self.gain + ctx.uniform_f64() * 5.0
        if ctx.time == 5:
            ctx.yield_control("decide")

sim = simul.Simulation(halt_time=20, seed=42)
sim.add_agent("actor", PlannerAgent(), proactive=True)

outcome = sim.run_until_yield()
assert outcome.kind == "yielded" and outcome.tag == "decide"

# Fork counterfactual branches from the paused snapshot with identical future noise.
branch_a = sim.clone_snapshot()
branch_b = sim.clone_snapshot()
branch_a.agent("actor").gain = 1.0
branch_b.agent("actor").gain = 3.0
branch_a.run()
branch_b.run()
assert branch_b.agent("actor").reward > branch_a.agent("actor").reward
```
