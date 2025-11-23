//! A simple benchmark for baseline simulation perf.
#![allow(clippy::missing_docs_in_private_items, clippy::expect_used)]
#[macro_use]
extern crate criterion;

use criterion::criterion_group;
use criterion::Criterion;
use simul::agent::{periodic_consumer, periodic_producer};

use simul::{Simulation, SimulationParameters};

fn simple_periodic_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple periodic bench");

    group.bench_function("benchmark", |b| {
        b.iter(|| {
            let mut simulation = Simulation::new(SimulationParameters {
                agent_initializers: vec![
                    periodic_producer("producer".to_string(), 1, "consumer".to_string()),
                    periodic_consumer("consumer".to_string(), 1),
                ],
                halt_check: |s: &Simulation| s.time() == 1000,
                ..Default::default()
            });
            simulation.run();
        });
    });
}

criterion_group!(benches, simple_periodic_bench);
criterion_main!(benches);
