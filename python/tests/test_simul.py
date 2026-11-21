import json
import unittest

import simul


class DecisionAgent:
    def __init__(self, gain: float = 0.0):
        self.gain = gain
        self.reward = 0.0
        self.received = []
        self.pending_timer = None

    def on_message(self, ctx, msg):
        self.received.append((ctx.time, msg.payload))
        if msg.payload == "arm":
            self.pending_timer = ctx.send_after(10, ctx.id, "timeout")
        elif msg.payload == "cancel" and self.pending_timer is not None:
            ctx.cancel_timer(self.pending_timer)
            self.pending_timer = None

    def on_tick(self, ctx):
        self.reward += self.gain + ctx.uniform_f64() * 10.0
        if ctx.time == 5:
            ctx.yield_control("choose_gain")


class FailingAgent:
    def on_tick(self, ctx):
        raise ValueError("boom from python")


class TestSimulBindings(unittest.TestCase):
    def test_builtin_simulation_and_tracing(self):
        sim = simul.Simulation(halt_time=10, seed=7, trace=True)
        sim.add_periodic_producer("producer", 2, "consumer")
        sim.add_periodic_consumer("consumer", 3)
        outcome = sim.run()
        self.assertEqual(outcome.kind, "halted")
        self.assertEqual(sim.time, 10)
        self.assertEqual(sim.produced_counts()["producer"], 5)
        self.assertGreater(sim.consumed_counts()["consumer"], 0)

        jsonl = sim.trace_jsonl()
        self.assertIsNotNone(jsonl)
        lines = [json.loads(line) for line in jsonl.splitlines()]
        self.assertGreater(len(lines), 0)

        perfetto = json.loads(sim.trace_perfetto_json())
        self.assertIn("traceEvents", perfetto)
        self.assertGreater(len(perfetto["traceEvents"]), 2)

    def test_custom_python_agent_yield_branching_and_timers(self):
        sim = simul.Simulation(halt_time=20, seed=42)
        sim.add_agent("actor", DecisionAgent(gain=0.0), proactive=True, drain_all=True)
        sim.push_message("actor", "arm")

        outcome = sim.run_until_yield()
        self.assertEqual(outcome.kind, "yielded")
        self.assertEqual(outcome.tag, "choose_gain")
        self.assertEqual(sim.time, 6)
        self.assertEqual(sim.mode, "paused")

        # Fork two isolated counterfactual branches from tick 6.
        branch_low = sim.clone_snapshot()
        branch_high = sim.clone_snapshot()
        branch_low.agent("actor").gain = 1.0
        branch_high.agent("actor").gain = 4.0
        branch_high.push_message("actor", "cancel")

        branch_low.run()
        branch_high.run()

        low_actor = branch_low.agent("actor")
        high_actor = branch_high.agent("actor")
        # Identical future CRN noise -> exact 14 * (4 - 1) = 42.0 difference!
        self.assertAlmostEqual(high_actor.reward - low_actor.reward, 42.0, places=9)
        # branch_low got the timeout at tick 10; branch_high cancelled it at tick 6!
        self.assertIn((10, "timeout"), low_actor.received)
        self.assertNotIn((10, "timeout"), high_actor.received)

        # Reseeding produces distinct stochastic future while original trunk stays at tick 6.
        reseeded = sim.clone_snapshot()
        reseeded.reseed(999)
        reseeded.agent("actor").gain = 1.0
        reseeded.run()
        self.assertNotAlmostEqual(reseeded.agent("actor").reward, low_actor.reward, places=6)

        # Windowed forensic Perfetto export on a clone.
        forensic = json.loads(sim.forensic_trace_perfetto(15, 20))
        self.assertIn("traceEvents", forensic)

    def test_python_exception_propagates_cleanly(self):
        sim = simul.Simulation(halt_time=5)
        sim.add_agent("bad", FailingAgent(), proactive=True)
        with self.assertRaisesRegex(ValueError, "boom from python"):
            sim.run()

    def test_quantiles_cvar_paired_crn_and_bernoulli(self):
        q = simul.QuantileSummary.from_values([10.0, 20.0, 30.0, 40.0, 100.0])
        self.assertEqual(q.count, 5)
        self.assertAlmostEqual(q.median(), 30.0)
        self.assertAlmostEqual(q.upper_cvar(0.80), 100.0)
        self.assertAlmostEqual(q.lower_cvar(0.40), 15.0)

        s = simul.ScalarSummary.from_values([10.0, 12.0, 11.0, 13.0, 9.0])
        lo, hi = s.ci_95()
        self.assertLess(lo, s.mean)
        self.assertGreater(hi, s.mean)

        b = simul.BernoulliSummary.from_results([True, True, True, False])
        self.assertAlmostEqual(b.rate, 0.75)
        self.assertLess(b.wilson_lower, b.rate)
        self.assertGreater(b.wilson_upper, b.rate)

        rng = simul.TrialRng(7)
        base = [rng.uniform_f64() * 100.0 for _ in range(30)]
        chal = [x + 2.5 for x in base]
        paired = simul.PairedScalarSummary.from_pairs(base, chal)
        self.assertEqual(paired.challenger_wins, 30)
        self.assertAlmostEqual(paired.mean_diff, 2.5, places=12)

    def test_doe_mixed_cem_and_pareto_frontier(self):
        lhs = simul.sample_latin_hypercube(7, 16, [(0.0, 16.0), (-1.0, 1.0)])
        self.assertEqual(len(lhs), 16)
        self.assertEqual(len({int(p[0]) for p in lhs}), 16)

        sobol = simul.sample_sobol(16, [(0.0, 1.0), (0.0, 1.0)])
        scrambled = simul.sample_sobol_scrambled(42, 16, [(0.0, 1.0), (0.0, 1.0)])
        self.assertEqual(len(sobol), 16)
        self.assertNotEqual(sobol[0], scrambled[0])

        opt = simul.MixedCrossEntropyOptimizer(
            [
                simul.MixedDimension.categorical(3),
                simul.MixedDimension.integer(1, 10),
                simul.MixedDimension.continuous(0.0, 10.0),
            ],
            elite_fraction=0.2,
        )
        rng = simul.TrialRng(123)
        for _ in range(12):
            pts = opt.ask_batch(rng, 32)
            scores = [
                (10.0 if p[0] == 2 else 0.0) - (p[1] - 7) ** 2 - (p[2] - 3.5) ** 2
                for p in pts
            ]
            opt.tell(pts, scores)
        best_pt, _ = opt.best()
        self.assertEqual(best_pt[0], 2)
        self.assertEqual(best_pt[1], 7)
        self.assertAlmostEqual(best_pt[2], 3.5, delta=1.0)

        archive = simul.ParetoArchive(["min", "max"])
        self.assertTrue(archive.insert(0, "a", [1.0, 2.0]))
        self.assertTrue(archive.insert(1, "b", [2.0, 4.0]))
        self.assertFalse(archive.insert(2, "dominated", [3.0, 3.0]))
        self.assertEqual(len(archive), 2)
        self.assertAlmostEqual(archive.hypervolume_2d(5.0, 0.0), 14.0, places=9)


if __name__ == "__main__":
    unittest.main()
