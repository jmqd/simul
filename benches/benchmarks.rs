//! A simple benchmark for baseline simulation perf.
#![allow(clippy::missing_docs_in_private_items, clippy::expect_used)]
#[macro_use]
extern crate criterion;

use criterion::{criterion_group, Criterion};
use simul::agent::{periodic_consumer, periodic_producer};
use simul::{Halt, Simulation};
use std::hint::black_box;

fn run_periodic_simulation(
    ticks: u64,
    enable_queue_depth_metrics: bool,
    idle_agents: usize,
) -> Simulation {
    let mut agent_initializers = vec![
        periodic_producer("producer".to_string(), 1, "consumer".to_string()),
        periodic_consumer("consumer".to_string(), 1),
    ];
    agent_initializers.extend(
        (0..idle_agents)
            .map(|index| periodic_consumer(format!("idle-{index}"), 1).asleep_until(u64::MAX)),
    );
    let mut simulation = Simulation::builder(Halt::AtTime(ticks))
        .agents(agent_initializers)
        .queue_depth_metrics(enable_queue_depth_metrics)
        .build()
        .unwrap_or_else(|error| panic!("{error}"));
    simulation.run();
    simulation
}

fn simple_periodic_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple periodic bench");

    group.bench_function("benchmark", |b| {
        b.iter(|| {
            let simulation = run_periodic_simulation(1_000, false, 0);
            black_box(simulation.time());
        });
    });

    group.bench_function("10_000_ticks", |b| {
        b.iter(|| {
            let simulation = run_periodic_simulation(10_000, false, 0);
            black_box(simulation.time());
        });
    });

    group.bench_function("1_000_ticks_with_queue_metrics", |b| {
        b.iter(|| {
            let simulation = run_periodic_simulation(1_000, true, 0);
            black_box(simulation.queue_depth_metrics("consumer"));
        });
    });

    for idle_agents in [1_000, 10_000] {
        group.bench_function(format!("10_000_ticks_{idle_agents}_idle_agents"), |b| {
            b.iter(|| {
                let simulation = run_periodic_simulation(10_000, false, idle_agents);
                black_box(simulation.time());
            });
        });
    }
}

criterion_group!(benches, simple_periodic_bench);
criterion_main!(benches);
