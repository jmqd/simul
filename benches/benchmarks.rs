#[macro_use]
extern crate criterion;

use criterion::criterion_group;
use criterion::Criterion;
use simul::agent::*;

use simul::*;

fn simple_periodic_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple periodic bench");

    group.bench_function("benchmark", |b| {
        b.iter(|| {
            let mut simulation = Simulation::new(SimulationParameters {
                agents: vec![
                    periodic_producing_agent("producer".to_string(), 1, "consumer".to_string()),
                    periodic_consuming_agent("consumer".to_string(), 1),
                ],
                halt_check: |s: &Simulation| s.time == 1000,
                ..Default::default()
            });
            simulation.run();
        })
    });
}

criterion_group!(benches, simple_periodic_bench);
criterion_main!(benches);
