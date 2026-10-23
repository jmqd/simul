use simul::experiment::{
    run_serial, run_serial_bounded, select_best, BernoulliSummary, CandidateId, MasterSeed,
    MaximizeBernoulliRate, MaximizeWilsonLowerBound, ReplicationId, RunError, TrialContext,
    TrialId, TrialSpec, TrialStream,
};
use std::convert::Infallible;
use std::num::NonZeroUsize;

const MASTER_SEED: MasterSeed = MasterSeed::from_u64(918_273);
const COMMON_GROUP: u64 = 29;

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap_or_else(|| panic!("test bound must be nonzero"))
}

const fn trial(candidate: u64, replication: u32) -> TrialSpec<u64> {
    TrialSpec::new(
        CandidateId::new(candidate),
        ReplicationId::new(replication),
        COMMON_GROUP,
        candidate,
    )
}

const fn trials() -> [TrialSpec<u64>; 4] {
    [trial(8, 1), trial(2, 1), trial(8, 0), trial(2, 0)]
}

fn samples(context: &TrialContext) -> [u64; 3] {
    [
        context
            .common_rng(TrialStream::TrialNoise, "heading")
            .next_u64(),
        context
            .independent_rng(TrialStream::ExecutionNoise, "heading")
            .next_u64(),
        context
            .common_rng(TrialStream::ValidationHoldout, "heading")
            .next_u64(),
    ]
}

#[test]
fn serial_records_are_canonical_and_preserve_trial_errors() {
    let mut evaluated = Vec::new();
    let records = run_serial(MASTER_SEED, trials(), |candidate, context| {
        evaluated.push(context.key());
        if context.key().replication().value() == 0 {
            Err(*candidate)
        } else {
            Ok(*candidate * 10)
        }
    });
    let records = match records {
        Ok(records) => records,
        Err(error) => panic!("serial run failed: {error:?}"),
    };
    let expected = [(2, 0), (2, 1), (8, 0), (8, 1)].map(|(candidate, replication)| {
        TrialId::new(CandidateId::new(candidate), ReplicationId::new(replication))
    });
    assert_eq!(evaluated, expected);
    assert_eq!(
        records.iter().map(|record| record.key).collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        records
            .iter()
            .map(|record| record.result)
            .collect::<Vec<_>>(),
        [Err(2), Ok(20), Err(8), Ok(80)]
    );
}

#[test]
fn common_random_streams_separate_holdout_and_reconstruct_replay() {
    let records = match run_serial(MASTER_SEED, trials(), |_, context| {
        Ok::<_, Infallible>(samples(context))
    }) {
        Ok(records) => records,
        Err(error) => panic!("sampling run failed: {error:?}"),
    };
    let draws: Vec<_> = records
        .iter()
        .map(|record| match record.result {
            Ok(draws) => draws,
            Err(never) => match never {},
        })
        .collect();
    assert_eq!(draws[0][0], draws[2][0]);
    assert_eq!(draws[0][2], draws[2][2]);
    assert_ne!(draws[0][0], draws[1][0]);
    assert_ne!(draws[0][0], draws[0][2]);
    assert_ne!(draws[0][1], draws[2][1]);

    for record in records {
        let replay = record.replay;
        let request = TrialSpec::new(
            replay.trial().candidate(),
            replay.trial().replication(),
            replay.common_random_group(),
            (),
        );
        let replayed = match run_serial(replay.master_seed(), [request], |(), context| {
            Ok::<_, Infallible>(samples(context))
        }) {
            Ok(records) => records,
            Err(error) => panic!("replay failed: {error:?}"),
        };
        assert_eq!(replayed, [record]);
    }
}

#[test]
fn serial_rejects_duplicates_and_bounds_before_evaluation() {
    let mut evaluations = 0;
    let duplicate = run_serial(MASTER_SEED, [trial(2, 0), trial(2, 0)], |_, _| {
        evaluations += 1;
        Ok::<_, Infallible>(())
    });
    assert!(matches!(
        duplicate,
        Err(RunError::DuplicateTrial(id)) if id == trial(2, 0).key()
    ));
    let bounded = run_serial_bounded(MASTER_SEED, trials(), nonzero(3), |_, _| {
        evaluations += 1;
        Ok::<_, Infallible>(())
    });
    assert!(matches!(
        bounded,
        Err(RunError::BatchLimitExceeded { limit: 3 })
    ));
    assert_eq!(evaluations, 0);
}

#[cfg(panic = "unwind")]
#[test]
fn serial_workload_panic_discards_partial_results_without_retry() {
    let mut evaluations = 0;
    let result = run_serial(MASTER_SEED, trials(), |_, _| {
        evaluations += 1;
        assert_ne!(evaluations, 2, "workload panic");
        Ok::<_, Infallible>(())
    });
    assert!(matches!(result, Err(RunError::WorkloadPanic)));
    assert_eq!(evaluations, 2);
}

#[cfg(panic = "unwind")]
#[test]
fn serial_panic_with_panicking_payload_returns_workload_panic() {
    struct PayloadBomb;
    impl Drop for PayloadBomb {
        fn drop(&mut self) {
            panic!("payload drop panic");
        }
    }
    let result = run_serial(MASTER_SEED, trials(), |_, _| -> Result<(), Infallible> {
        std::panic::resume_unwind(Box::new(PayloadBomb))
    });
    assert!(matches!(result, Err(RunError::WorkloadPanic)));
}

#[test]
fn bernoulli_selection_ranks_nan_lowest_regardless_of_order() {
    let summary = |value| BernoulliSummary {
        attempts: 10,
        successes: 5,
        rate: value,
        wilson_lower: value,
        wilson_upper: value,
    };
    let candidates = [
        (CandidateId::new(3), summary(0.9)),
        (CandidateId::new(2), summary(f64::NAN)),
        (CandidateId::new(1), summary(0.5)),
    ];
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let shuffled = order.map(|index| candidates[index]);
        for selected in [
            select_best(&shuffled, &MaximizeBernoulliRate),
            select_best(&shuffled, &MaximizeWilsonLowerBound),
        ] {
            assert_eq!(
                Some(CandidateId::new(3)),
                selected.map(|(id, _)| id),
                "order {order:?}"
            );
        }
    }
}

#[cfg(feature = "parallel")]
mod parallel {
    use super::*;
    use simul::experiment::{run_parallel, ParallelConfig, ParallelExecutor, ParallelRunError};
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn parallel_and_reused_executor_match_serial_across_requested_worker_counts() {
        let evaluate = |candidate: &u64, context: &TrialContext| {
            if context.key().replication().value() == 0 {
                Err(*candidate)
            } else {
                Ok(samples(context))
            }
        };
        let serial = match run_serial(MASTER_SEED, trials(), evaluate) {
            Ok(records) => records,
            Err(error) => panic!("serial run failed: {error:?}"),
        };
        for workers in [1, 2, 8] {
            let parallel = run_parallel(
                MASTER_SEED,
                trials(),
                ParallelConfig::new(nonzero(workers), nonzero(4)),
                || Ok::<_, Infallible>([0_u64; 3]),
                |scratch, candidate, context| {
                    let result = evaluate(candidate, context)?;
                    *scratch = result;
                    Ok::<_, u64>(*scratch)
                },
            );
            let parallel = match parallel {
                Ok(records) => records,
                Err(error) => panic!("parallel run failed: {error:?}"),
            };
            assert_eq!(parallel, serial);
            let executor = ParallelExecutor::new(ParallelConfig::new(nonzero(workers), nonzero(4)))
                .unwrap_or_else(|error| panic!("pool construction failed: {error:?}"));
            let capacity = std::thread::available_parallelism()
                .map_or(1, NonZeroUsize::get)
                .min(workers);
            let caller = std::thread::current().id();
            let dropped = AtomicUsize::new(0);
            let mut factories = 0;
            for round in 0..3 {
                let mut batch = trials();
                batch.rotate_left(round);
                let records = executor
                    .run(
                        MASTER_SEED,
                        batch,
                        || {
                            assert_eq!(caller, std::thread::current().id());
                            factories += 1;
                            Ok::<_, Infallible>((Tracked(&dropped), [0_u64; 3]))
                        },
                        |(_, scratch), candidate, context| {
                            *scratch = evaluate(candidate, context)?;
                            Ok::<_, u64>(*scratch)
                        },
                    )
                    .unwrap_or_else(|error| panic!("reused run failed: {error:?}"));
                assert_eq!(serial, records);
                assert_eq!((round + 1) * capacity.min(4), factories);
                assert_eq!(factories, dropped.load(Ordering::SeqCst));
            }
        }
    }

    #[test]
    fn executor_rejects_invalid_batches_before_callbacks() {
        let factories = AtomicUsize::new(0);
        let evaluations = AtomicUsize::new(0);
        for (batch, limit) in [(vec![trial(2, 0), trial(2, 0)], 4), (trials().to_vec(), 3)] {
            let executor = ParallelExecutor::new(ParallelConfig::new(nonzero(2), nonzero(limit)))
                .unwrap_or_else(|error| panic!("pool construction failed: {error:?}"));
            let result = executor.run(
                MASTER_SEED,
                batch,
                || {
                    factories.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, Infallible>(())
                },
                |(), _, _| {
                    evaluations.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, Infallible>(())
                },
            );
            if limit == 4 {
                assert!(matches!(
                    result,
                    Err(ParallelRunError::Batch(RunError::DuplicateTrial(id)))
                        if id == trial(2, 0).key()
                ));
            } else {
                assert!(matches!(
                    result,
                    Err(ParallelRunError::Batch(RunError::BatchLimitExceeded {
                        limit: 3
                    }))
                ));
            }
        }
        assert_eq!(factories.load(Ordering::SeqCst), 0);
        assert_eq!(evaluations.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn executor_initialization_failure_prevents_evaluation_and_allows_reuse() {
        let workers = std::thread::available_parallelism()
            .map_or(1, NonZeroUsize::get)
            .min(2);
        let executor = ParallelExecutor::new(ParallelConfig::new(nonzero(2), nonzero(4)))
            .unwrap_or_else(|error| panic!("pool construction failed: {error:?}"));
        for fail_at in 0..workers {
            let mut factories = 0;
            let evaluations = AtomicUsize::new(0);
            let dropped = AtomicUsize::new(0);
            let result = executor.run(
                MASTER_SEED,
                trials(),
                || {
                    let worker = factories;
                    factories += 1;
                    if worker == fail_at {
                        Err(Rc::new("initialization"))
                    } else {
                        Ok(Tracked(&dropped))
                    }
                },
                |_, _, _| {
                    evaluations.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, Infallible>(())
                },
            );
            assert!(matches!(
                result,
                Err(ParallelRunError::WorkerInitialization { worker, source })
                    if worker == fail_at && *source == "initialization"
            ));
            assert_eq!(factories, fail_at + 1);
            assert_eq!(evaluations.load(Ordering::SeqCst), 0);
            assert_eq!(fail_at, dropped.load(Ordering::SeqCst));
            let recovered = executor
                .run(
                    MASTER_SEED,
                    trials(),
                    || Ok::<_, Infallible>(()),
                    |(), _, context| Ok::<_, Infallible>(samples(context)),
                )
                .unwrap_or_else(|error| panic!("recovery failed: {error:?}"));
            let serial = run_serial(MASTER_SEED, trials(), |_, context| {
                Ok::<_, Infallible>(samples(context))
            })
            .unwrap_or_else(|error| panic!("serial run failed: {error:?}"));
            assert_eq!(serial, recovered);
        }
    }

    #[test]
    fn empty_parallel_batch_skips_callbacks() {
        let result = run_parallel(
            MASTER_SEED,
            Vec::<TrialSpec<()>>::new(),
            ParallelConfig::new(nonzero(2), nonzero(4)),
            || -> Result<(), Infallible> { panic!("empty batch called worker factory") },
            |(), (), _| -> Result<(), Infallible> { panic!("empty batch evaluated a trial") },
        );
        match result {
            Ok(records) => assert_eq!(0, records.len()),
            Err(error) => panic!("empty batch failed: {error:?}"),
        }
    }

    #[test]
    fn executor_reuses_threads_with_fresh_workspaces_across_empty_batches() {
        let executor = ParallelExecutor::new(ParallelConfig::new(nonzero(1), nonzero(4)))
            .unwrap_or_else(|error| panic!("pool construction failed: {error:?}"));
        let caller = std::thread::current().id();
        let mut thread = None;
        let mut factories = 0;
        let dropped = AtomicUsize::new(0);
        for length in [1, 0, 4, 0, 2] {
            let batch: Vec<_> = trials().into_iter().take(length).collect();
            let records = executor
                .run(
                    MASTER_SEED,
                    batch,
                    || {
                        assert_eq!(caller, std::thread::current().id());
                        factories += 1;
                        Ok::<_, Infallible>((Tracked(&dropped), 0))
                    },
                    |(_, count), _, _| {
                        let ordinal = *count;
                        *count += 1;
                        Ok::<_, Infallible>((std::thread::current().id(), ordinal))
                    },
                )
                .unwrap_or_else(|error| panic!("reused run failed: {error:?}"));
            assert_eq!(length, records.len());
            for (ordinal, record) in records.iter().enumerate() {
                let result = record.result;
                let Ok((worker, count)) = result;
                assert_ne!(worker, caller);
                assert_eq!(*thread.get_or_insert(worker), worker);
                assert_eq!(ordinal, count);
            }
            assert_eq!(factories, dropped.load(Ordering::SeqCst));
        }
        assert_eq!(3, factories);
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn executor_factory_panic_prevents_evaluation_and_allows_reuse() {
        let mut factories = 0;
        let evaluations = AtomicUsize::new(0);
        let executor = ParallelExecutor::new(ParallelConfig::new(nonzero(1), nonzero(4)))
            .unwrap_or_else(|error| panic!("pool construction failed: {error:?}"));
        let result = executor.run(
            MASTER_SEED,
            trials(),
            || -> Result<(), Infallible> {
                factories += 1;
                panic!("factory panic");
            },
            |(), _, _| {
                evaluations.fetch_add(1, Ordering::SeqCst);
                Ok::<_, Infallible>(())
            },
        );
        assert!(matches!(
            result,
            Err(ParallelRunError::WorkerFactoryPanic { worker: 0 })
        ));
        assert_eq!(factories, 1);
        assert_eq!(evaluations.load(Ordering::SeqCst), 0);
        let recovered = executor
            .run(
                MASTER_SEED,
                trials(),
                || Ok::<_, Infallible>(()),
                |(), _, context| Ok::<_, Infallible>(samples(context)),
            )
            .unwrap_or_else(|error| panic!("recovery failed: {error:?}"));
        let serial = run_serial(MASTER_SEED, trials(), |_, context| {
            Ok::<_, Infallible>(samples(context))
        })
        .unwrap_or_else(|error| panic!("serial run failed: {error:?}"));
        assert_eq!(serial, recovered);
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn executor_workload_panic_discards_partial_results_and_allows_reuse() {
        let evaluations = AtomicUsize::new(0);
        let executor = ParallelExecutor::new(ParallelConfig::new(nonzero(1), nonzero(4)))
            .unwrap_or_else(|error| panic!("pool construction failed: {error:?}"));
        let dropped = AtomicUsize::new(0);
        let result = executor.run(
            MASTER_SEED,
            trials(),
            || Ok::<_, Infallible>(Tracked(&dropped)),
            |_, _, _| {
                let previous = evaluations.fetch_add(1, Ordering::SeqCst);
                assert_ne!(previous, 1, "workload panic");
                Ok::<_, Infallible>(())
            },
        );
        assert!(matches!(
            result,
            Err(ParallelRunError::WorkerWorkloadPanic { worker: 0 })
        ));
        assert_eq!(evaluations.load(Ordering::SeqCst), 2);
        assert_eq!(1, dropped.load(Ordering::SeqCst));
        let recovered = executor
            .run(
                MASTER_SEED,
                trials(),
                || Ok::<_, Infallible>(Tracked(&dropped)),
                |_, _, context| Ok::<_, Infallible>(samples(context)),
            )
            .unwrap_or_else(|error| panic!("recovery failed: {error:?}"));
        let serial = run_serial(MASTER_SEED, trials(), |_, context| {
            Ok::<_, Infallible>(samples(context))
        })
        .unwrap_or_else(|error| panic!("serial run failed: {error:?}"));
        assert_eq!(serial, recovered);
        assert_eq!(2, dropped.load(Ordering::SeqCst));
    }

    #[derive(Debug)]
    struct Tracked<'a>(&'a AtomicUsize);

    impl Drop for Tracked<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn successful_output_owns_each_outcome_and_error_exactly_once() {
        for workers in [1, 4] {
            let drops: [AtomicUsize; 8] = std::array::from_fn(|_| AtomicUsize::new(0));
            let result = run_parallel(
                MASTER_SEED,
                (0..8).rev().map(|id| trial(id, 0)),
                ParallelConfig::new(nonzero(workers), nonzero(8)),
                || Ok::<_, Infallible>(()),
                |(), id, _| {
                    let Ok(index) = usize::try_from(*id) else {
                        panic!("small test identity must fit");
                    };
                    let tracked = Tracked(&drops[index]);
                    if index == 4 {
                        Err(tracked)
                    } else {
                        Ok(tracked)
                    }
                },
            );
            let records = result.unwrap_or_else(|error| panic!("run failed: {error:?}"));
            assert_eq!(
                records
                    .iter()
                    .map(|record| record.key.candidate().value())
                    .collect::<Vec<_>>(),
                (0..8).collect::<Vec<_>>()
            );
            assert!(records[4].result.is_err());
            assert!(drops.iter().all(|count| count.load(Ordering::SeqCst) == 0));
            drop(records);
            assert!(drops.iter().all(|count| count.load(Ordering::SeqCst) == 1));
        }
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn failed_output_drops_partial_and_completed_shards_exactly_once() {
        for workers in [1, 4] {
            let drops: [AtomicUsize; 8] = std::array::from_fn(|_| AtomicUsize::new(0));
            let created: [AtomicUsize; 8] = std::array::from_fn(|_| AtomicUsize::new(0));
            let result = run_parallel(
                MASTER_SEED,
                (0..8).map(|id| trial(id, 0)),
                ParallelConfig::new(nonzero(workers), nonzero(8)),
                || Ok::<_, Infallible>(()),
                |(), id, _| {
                    let Ok(index) = usize::try_from(*id) else {
                        panic!("small test identity must fit");
                    };
                    assert_ne!(index, 1, "panic after one initialized record");
                    created[index].fetch_add(1, Ordering::SeqCst);
                    Ok::<_, Infallible>(Tracked(&drops[index]))
                },
            );
            assert!(matches!(
                result,
                Err(ParallelRunError::WorkerWorkloadPanic { worker: 0 })
            ));
            assert_eq!(created[0].load(Ordering::SeqCst), 1);
            for (created, dropped) in created.iter().zip(&drops) {
                assert_eq!(
                    created.load(Ordering::SeqCst),
                    dropped.load(Ordering::SeqCst)
                );
            }
        }
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn discarding_partial_output_drops_records_after_a_panicking_drop() {
        struct PanicOnDrop<'a>(&'a AtomicUsize);
        impl Drop for PanicOnDrop<'_> {
            fn drop(&mut self) {
                assert_ne!(
                    self.0.fetch_add(1, Ordering::SeqCst),
                    0,
                    "record drop panic"
                );
            }
        }
        let drops = AtomicUsize::new(0);
        let outer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_parallel(
                MASTER_SEED,
                (0..4).map(|id| trial(id, 0)),
                ParallelConfig::new(nonzero(1), nonzero(4)),
                || Ok::<_, Infallible>(()),
                |(), id, _| {
                    assert_ne!(*id, 3, "discard three written records");
                    Ok::<_, Infallible>(PanicOnDrop(&drops))
                },
            )
            .map(drop)
        }));
        assert!(outer.is_err());
        assert_eq!(3, drops.load(Ordering::SeqCst));
    }
}
