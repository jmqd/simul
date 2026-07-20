use simul::experiment::{
    run_replicated, Candidate, RandomDomain, ReplayKey, ReplicationError, ReplicationPlan,
    SampleContext, SampleStream, TrialError, TrialKey, SEED_PROTOCOL,
};
use std::convert::Infallible;
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::atomic::{AtomicBool, Ordering};

const SCREENING_DOMAIN: RandomDomain = RandomDomain::new(0x5345_4152_4348_0002);
const VALIDATION_DOMAIN: RandomDomain = RandomDomain::new(0x5345_4152_4348_0003);
const HEADING_STREAM: SampleStream = SampleStream::new(0x4845_4144_494e_4701);

fn plan(domain: RandomDomain, replications: u32, workers: usize) -> ReplicationPlan {
    let Some(replications) = NonZeroU32::new(replications) else {
        panic!("test replications must be nonzero");
    };
    let Some(workers) = NonZeroUsize::new(workers) else {
        panic!("test workers must be nonzero");
    };
    ReplicationPlan {
        master_seed: 918_273,
        random_domain: domain,
        replications,
        workers,
    }
}

#[test]
fn public_sampling_and_replay_protocol_is_stable() {
    assert_eq!(SEED_PROTOCOL, "simul-v1-splitmix64-box-muller");
    let samples = SampleContext::new(918_273, SCREENING_DOMAIN, 29);
    assert_eq!(
        samples.uniform(HEADING_STREAM).to_bits(),
        0x3fe6_1efa_9df0_d72f
    );
    let normal = match samples.truncated_standard_normal(HEADING_STREAM, 3.0) {
        Ok(value) => value,
        Err(error) => panic!("normal draw failed: {error}"),
    };
    assert!((normal - -0.468_319_509_146_165_4).abs() <= 1e-15);

    let key = TrialKey {
        random_domain: SCREENING_DOMAIN,
        candidate_id: 41,
        replication_id: 29,
        common_random_group: 29,
    };
    assert_eq!(
        ReplayKey::new(918_273, key).to_string(),
        "simul-v1:918273:5345415243480002:41:29:29"
    );
}

#[test]
fn public_runner_is_ordered_common_random_and_worker_independent() {
    let candidates = [
        Candidate {
            id: 8,
            value: 3_u64,
        },
        Candidate {
            id: 2,
            value: 7_u64,
        },
    ];
    let run = |workers| {
        run_replicated(
            &candidates,
            plan(SCREENING_DOMAIN, 4, workers),
            |_| Ok::<_, Infallible>(11_u64),
            |candidate, context| {
                Ok::<_, Infallible>((
                    *candidate,
                    context.samples().uniform(HEADING_STREAM).to_bits(),
                ))
            },
            |worker, prepared| Ok::<_, Infallible>((prepared.0, prepared.1 ^ *worker)),
        )
    };

    let serial = match run(1) {
        Ok(records) => records,
        Err(error) => panic!("serial run failed: {error}"),
    };
    let parallel = match run(4) {
        Ok(records) => records,
        Err(error) => panic!("parallel run failed: {error}"),
    };
    assert_eq!(serial, parallel);

    let coordinates: Vec<_> = serial
        .iter()
        .map(|record| (record.key.candidate_id, record.key.replication_id))
        .collect();
    assert_eq!(
        coordinates,
        [
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
        ]
    );
    assert!(
        matches!(serial[0].result, Ok((3, first)) if matches!(serial[4].result, Ok((7, second)) if first == second))
    );
    assert_ne!(serial[0].result, serial[1].result);

    let validation = run_replicated(
        &candidates[..1],
        plan(VALIDATION_DOMAIN, 1, 1),
        |_| Ok::<_, Infallible>(11_u64),
        |candidate, context| {
            Ok::<_, Infallible>((
                *candidate,
                context.samples().uniform(HEADING_STREAM).to_bits(),
            ))
        },
        |worker, prepared| Ok::<_, Infallible>((prepared.0, prepared.1 ^ *worker)),
    );
    let validation = match validation {
        Ok(records) => records,
        Err(error) => panic!("validation-domain run failed: {error}"),
    };
    assert_ne!(serial[0].result, validation[0].result);
}

#[test]
fn public_runner_preserves_trial_errors_and_rejects_duplicate_ids() {
    let factory_called = AtomicBool::new(false);
    let duplicate = [Candidate { id: 5, value: 1 }, Candidate { id: 5, value: 2 }];
    let duplicate_result = run_replicated(
        &duplicate,
        plan(SCREENING_DOMAIN, 1, 1),
        |_| {
            factory_called.store(true, Ordering::SeqCst);
            Ok::<_, ()>(())
        },
        |_, _| Ok::<_, &'static str>(()),
        |(), ()| Ok::<_, &'static str>(()),
    );
    assert!(matches!(
        duplicate_result,
        Err(ReplicationError::DuplicateCandidateId { candidate_id: 5 })
    ));
    assert!(!factory_called.load(Ordering::SeqCst));

    let candidates = [Candidate { id: 9, value: () }];
    let records = run_replicated(
        &candidates,
        plan(SCREENING_DOMAIN, 2, 1),
        |_| Ok::<_, Infallible>(()),
        |(), context| {
            if context.key().replication_id == 0 {
                Err("prepare")
            } else {
                Ok(())
            }
        },
        |(), ()| Err::<(), _>("evaluate"),
    );
    let records = match records {
        Ok(records) => records,
        Err(error) => panic!("typed-error run failed: {error}"),
    };
    assert!(matches!(
        records[0].result,
        Err(TrialError::Prepare("prepare"))
    ));
    assert!(matches!(
        records[1].result,
        Err(TrialError::Evaluate("evaluate"))
    ));
}
