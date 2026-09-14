use simul::experiment::{
    run_serial, run_serial_bounded, CandidateId, MasterSeed, ReplicationId, RunError, TrialContext,
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

#[cfg(feature = "parallel")]
mod parallel {
    use super::*;
    use simul::experiment::{run_parallel, ParallelConfig, ParallelRunError};
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn parallel_matches_serial_across_requested_worker_counts() {
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
        }
    }

    #[test]
    fn parallel_rejects_invalid_batches_before_callbacks() {
        let factories = AtomicUsize::new(0);
        let evaluations = AtomicUsize::new(0);
        for (batch, limit) in [(vec![trial(2, 0), trial(2, 0)], 4), (trials().to_vec(), 3)] {
            let result = run_parallel(
                MASTER_SEED,
                batch,
                ParallelConfig::new(nonzero(2), nonzero(limit)),
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
    fn worker_initialization_failure_prevents_all_evaluation() {
        let workers = std::thread::available_parallelism()
            .map_or(1, NonZeroUsize::get)
            .min(2);
        for fail_at in 0..workers {
            let mut factories = 0;
            let evaluations = AtomicUsize::new(0);
            let result = run_parallel(
                MASTER_SEED,
                trials(),
                ParallelConfig::new(nonzero(2), nonzero(4)),
                || {
                    let worker = factories;
                    factories += 1;
                    if worker == fail_at {
                        Err(Rc::new("initialization"))
                    } else {
                        Ok(())
                    }
                },
                |(), _, _| {
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
            Ok(records) => assert!(records.is_empty()),
            Err(error) => panic!("empty batch failed: {error:?}"),
        }
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn worker_factory_panic_prevents_evaluation_without_retry() {
        let mut factories = 0;
        let evaluations = AtomicUsize::new(0);
        let result = run_parallel(
            MASTER_SEED,
            trials(),
            ParallelConfig::new(nonzero(1), nonzero(4)),
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
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn worker_workload_panic_discards_partial_results_without_retry() {
        let evaluations = AtomicUsize::new(0);
        let result = run_parallel(
            MASTER_SEED,
            trials(),
            ParallelConfig::new(nonzero(1), nonzero(4)),
            || Ok::<_, Infallible>(()),
            |(), _, _| {
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
    }
}
