#![feature(let_chains)]
use log::info;
use plotters::prelude::*;
use rand_distr::Poisson;
use simul::agent::*;
use simul::experiment::*;
use simul::*;
use std::path::PathBuf;

/// Just a janky `++` operator.
fn inc(num: &mut usize) -> usize {
    *num += 1;
    return *num;
}

/// Sandbox for running a simulated annealing experiment.
fn run_experiment() {
    let halt_condition = |s: &Simulation| {
        s.agents
            .iter()
            .find(|a| a.name == "consumer")
            .expect("No consumer agent?")
            .consumed
            .len()
            > 10
    };

    // Creates an agent generator w/ a fixed producer at interval 2 and a
    // consumer whose period randomly varies between [0, 10]
    let agent_generator = periodic_agent_generator_fixed_producer(2, 10);

    // SimulationParameters generator that holds all else static except for agents.
    let simulation_parameters_generator = move || SimulationParameters {
        agents: agent_generator(),
        halt_check: halt_condition,
        ..Default::default()
    };

    // This is the objective function which we're trying to approximately
    // optimize via simulated experiment.  This objective function tries to find
    // the simulation that completed the Simulation in the least time and
    // doesn't "overuse" consumer period.
    //
    // In this specific example, negative simulation time means that we're
    // optimizing for the simulation that completes the fastest. If this were
    // the only parameter, there would be two solutions: 0 and 1. Because the
    // producer is fixed at a period of 2, a consumer with period 0 or 1 can
    // sufficiently keep up with that producer.
    //
    // By subtracting the period of the consumer from the simulation time, we're
    // looking to get the fastest simulation time without wasting any resources.
    let objective_fn = |s: &Simulation| {
        -(s.time as i64)
            + s.agents
                .iter()
                .find(|a| a.name == "consumer")
                .as_ref()
                .unwrap()
                .extensions
                .as_ref()
                .unwrap()
                .period
                .unwrap() as i64
    };

    let replications_limit = 100;

    // Run the simulation 100 different times, randomly varying the agent
    // configuration, and return the one that maximized the objective function.
    let approx_optimal = experiment_by_annealing_objective(
        simulation_parameters_generator,
        replications_limit,
        objective_fn,
    );

    info!(
        "{:?}",
        approx_optimal
            .unwrap()
            .agents
            .iter()
            .map(|a| (&a.name, &a.extensions))
    );
}

/// Given a producer with a fixed period, returns producer-consumer two Agent
/// configurations (where only the consumer varies).
///
/// The consumer randomly varies between 0 and consumer_max_period. The two
/// agents are always named "producer" and "consumer".
fn periodic_agent_generator_fixed_producer(
    producer_period: u64,
    consumer_max_period: u64,
) -> impl Fn() -> Vec<Agent> {
    move || {
        let consumer_period = rand::random::<u32>() % (consumer_max_period + 1) as u32;
        let producer_agent = periodic_producing_agent("producer", producer_period, "consumer");
        let consumer_agent = periodic_consuming_agent("consumer", consumer_period as u64);
        vec![producer_agent, consumer_agent]
    }
}

fn test_plotting() -> Result<(), Box<dyn std::error::Error>> {
    let mut simulation = Simulation::new(SimulationParameters {
        agents: vec![
            periodic_producing_agent("producer", 1, "consumer"),
            periodic_consuming_agent("consumer", 3),
        ],
        enable_queue_depth_telemetry: true,
        halt_check: |s: &Simulation| s.time == 10,
        ..Default::default()
    });
    simulation.run();
    plot_simulation(
        &simulation,
        &["producer".into(), "consumer".into()],
        &"/tmp/0.png".to_string().into(),
    )?;
    Ok(())
}

fn test_plotting_2() -> Result<(), Box<dyn std::error::Error>> {
    let mut simulation = Simulation::new(SimulationParameters {
        agents: vec![
            poisson_distributed_consuming_agent("Barista", Poisson::new(60.0)?),
            poisson_distributed_producing_agent("Customers", Poisson::new(60.0)?, "Barista"),
        ],
        enable_queue_depth_telemetry: true,
        halt_check: |s: &Simulation| s.time == 60 * 60 * 12,
        ..Default::default()
    });
    simulation.run();
    plot_simulation(
        &simulation,
        &["Customers".into(), "Barista".into()],
        &"/tmp/cafe-example.png".to_string().into(),
    )?;
    Ok(())
}

fn test_plotting_3() -> Result<(), Box<dyn std::error::Error>> {
    let mut simulation = Simulation::new(SimulationParameters {
        agents: vec![
            poisson_distributed_consuming_agent("Barista", Poisson::new(60.0)?),
            poisson_distributed_producing_agent("Customers", Poisson::new(60.0)?, "Barista"),
        ],
        enable_queue_depth_telemetry: true,
        halt_check: |s: &Simulation| s.time == 60 * 60 * 12,
        ..Default::default()
    });
    simulation.run();
    plot_queued_durations_for_processed_messages(
        &simulation,
        &["Barista".into()],
        &"/tmp/cafe-example-queued-durations.png".to_string().into(),
    )
}

fn plot_simulation(
    simulation: &Simulation,
    agents: &[String],
    output: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut produced_series: Vec<Vec<(u64, u64)>> = vec![];
    let mut consumed_series: Vec<Vec<(u64, u64)>> = vec![];
    let mut queue_depth_series: Vec<Vec<(u64, u64)>> = vec![];

    for a in agents.iter() {
        if let Some(produced) = simulation.produced_for_agent(a) {
            produced_series.push(produced.into_iter().map(|e| (e.queued_time, 1)).collect());
        }

        if let Some(consumed) = simulation.consumed_for_agent(a) {
            consumed_series.push(
                consumed
                    .into_iter()
                    .map(|e| (e.completed_time.unwrap(), 1))
                    .collect(),
            );
        }

        if let Some(queue_depths) = simulation.queue_depth_metrics(a) {
            queue_depth_series.push(
                queue_depths
                    .into_iter()
                    .enumerate()
                    .map(|(i, e)| (i as u64, e as u64))
                    .collect(),
            );
        }
    }

    info!("Agents {:?}", &agents);
    info!("Produced {:?}", &produced_series);
    info!("Consumed {:?}", &consumed_series);
    info!("Queue depth {:?}", &queue_depth_series);

    let max_y = produced_series
        .iter()
        .chain(consumed_series.iter())
        .chain(queue_depth_series.iter())
        .flatten()
        .map(|n| n.1)
        .max()
        .unwrap() as u64;

    let root = BitMapBackend::new(output, (2560, 1920)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("producer vs consumer", ("sans-serif", 50).into_font())
        .margin(5)
        .set_all_label_area_size(64)
        .build_cartesian_2d(0u64..simulation.time + 1, 0u64..max_y + 1)?;

    chart
        .configure_mesh()
        .x_desc("Simulation Epoch (u64)")
        .y_desc("Count")
        .label_style(("sans-serif", 32, &BLACK))
        .draw()?;

    let mut series_idx = 0;
    for (i, agent) in agents.iter().enumerate() {
        let consumed = consumed_series
            .get(i)
            .expect("Failed to get consumed")
            .clone();
        let produced = produced_series
            .get(i)
            .expect("Failed to get consumed")
            .clone();
        let queue_depth = queue_depth_series
            .get(i)
            .expect("Failed to get consumed")
            .clone();

        if !consumed.is_empty() {
            let consumed_color = Palette99::pick(inc(&mut series_idx)).filled();
            chart
                .draw_series(
                    consumed
                        .iter()
                        .map(|(x, y)| Circle::new((*x, *y), 4, consumed_color.filled())),
                )?
                .label(format!("{} consumed", agent))
                .legend(move |(x, y)| {
                    Rectangle::new([(x - 16, y + 16), (x + 16, y - 16)], consumed_color)
                });
        }

        if !produced.is_empty() {
            let produced_color = Palette99::pick(inc(&mut series_idx)).filled();
            chart
                .draw_series(
                    produced
                        .iter()
                        .map(|(x, y)| Circle::new((*x, *y), 4, produced_color.filled())),
                )?
                .label(format!("{} produced", agent))
                .legend(move |(x, y)| {
                    Rectangle::new([(x - 16, y + 16), (x + 16, y - 16)], produced_color)
                });
        }

        if !queue_depth.is_empty() && !queue_depth.iter().all(|a| a.1 == 0u64) {
            let queue_depth_color = Palette99::pick(inc(&mut series_idx)).filled();
            chart
                .draw_series(
                    queue_depth
                        .iter()
                        .map(|(x, y)| Circle::new((*x, *y), 4, queue_depth_color.filled())),
                )?
                .label(format!("{} queue_depth", agent))
                .legend(move |(x, y)| {
                    Rectangle::new([(x - 16, y + 16), (x + 16, y - 16)], queue_depth_color)
                });
        }
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 32))
        .draw()?;

    root.present().expect("Presenting failed.");
    Ok(())
}

fn plot_queued_durations_for_processed_messages(
    simulation: &Simulation,
    agents: &[String],
    output: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut processing_latency: Vec<Vec<(u64, u64)>> = vec![];
    for a in agents.iter() {
        if let Some(consumed) = simulation.consumed_for_agent(a) {
            processing_latency.push(
                consumed
                    .into_iter()
                    .map(|e| {
                        (
                            e.completed_time.unwrap(),
                            e.completed_time.unwrap() - e.queued_time,
                        )
                    })
                    .collect(),
            );
        }
    }

    info!("Processing latency {:?}", &processing_latency);

    let max_y = processing_latency
        .iter()
        .flatten()
        .map(|n| n.1)
        .max()
        .unwrap() as u64;

    let root = BitMapBackend::new(output, (2560, 1920)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("queued time", ("sans-serif", 50).into_font())
        .margin(5)
        .set_all_label_area_size(64)
        .build_cartesian_2d(0u64..simulation.time + 1, 0u64..max_y + 1)?;

    let mut series_idx = 0;
    for processing_latency_series in processing_latency {
        if !processing_latency_series.is_empty() {
            let color = Palette99::pick(series_idx).filled();
            chart
                .draw_series(
                    processing_latency_series
                        .iter()
                        .map(|(x, y)| Circle::new((*x, *y), 4, color.filled())),
                )?
                .label(format!(
                    "{} processing_time",
                    agents.get(series_idx).unwrap()
                ))
                .legend(move |(x, y)| Rectangle::new([(x - 16, y + 16), (x + 16, y - 16)], color));
        }
        series_idx += 1;
    }

    chart
        .configure_mesh()
        .x_desc("Processing Epoch (u64)")
        .y_desc("Processing Latency")
        .label_style(("sans-serif", 32, &BLACK))
        .draw()?;
    Ok(())
}

/// Note, this main.rs binary file is just for library prototyping at the moment.
fn main() {
    test_plotting().expect("Plotting failed.");
    test_plotting_2().expect("Plotting 2 failed.");
    test_plotting_3().expect("Plotting 3 failed.");
    run_experiment();
}
