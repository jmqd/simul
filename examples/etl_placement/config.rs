//! Model parameters. Each value names its source; `ASSUMPTION` marks values
//! with no published source, chosen as plausible defaults to be overridden by
//! measurements.
//!
//! Sources:
//! - [etl] supabase/etl @ 2a4a0f8a (2026-10-23),
//!   <https://github.com/supabase/etl/tree/2a4a0f8a577f7f5fd9ee89daae9ae0451b0b6e7a>
//! - [nmip] AWS Network Manager Infrastructure Performance,
//!   <https://docs.aws.amazon.com/network-manager/latest/infrastructure-performance/how-nmip-works.html>
//! - [cloudping] <https://www.cloudping.co/api/latencies?percentile=p_50&timeframe=1Y>
//! - [reda] Reda et al., "Path Persistence in the Cloud", CCR 2020,
//!   <https://wreda.github.io/papers/rambler-ccr20.pdf>
//! - [az] <https://www.bitsand.cloud/posts/cross-az-latencies>
//! - [ec2bw] <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-network-bandwidth.html>,
//!   <https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html>
//! - [linux] `include/net/tcp.h`, `net/ipv4/tcp.c`, `net/ipv4/tcp_cubic.c` (v6.1–v6.15);
//!   <https://docs.kernel.org/networking/ip-sysctl.html>
//! - [rfc9438] CUBIC, <https://www.rfc-editor.org/rfc/rfc9438.txt>
//! - [pgproto] <https://www.postgresql.org/docs/current/protocol-logicalrep-message-formats.html>,
//!   <https://www.postgresql.org/docs/current/sasl-authentication.html>
//! - [pgrepl] <https://www.postgresql.org/docs/current/runtime-config-replication.html>
//! - [peerdb] <https://blog.peerdb.io/exploring-versions-of-the-postgres-logical-replication-protocol>
//! - [galaxy] single-stream pg→pg copy on RDS, <https://raw.githubusercontent.com/galaxy-io/benchmarks/main/data-movement/results/2026-08-31/taxi12-3route/native/full-load-pg-pg-taxi-12mo-remote.json>
//! - [pricing] AWS Price List API, USW2-USE1-AWS-Out-Bytes, effective 2026-06-01

use crate::net::{FaultPlan, NicModel, PathChange, PathModel, TcpModel};
use crate::time::{ticks, Tick};

/// Where the ETL binary runs. The source is in us-west-2, the destination in
/// us-east-1, and the state store inside the source database.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Placement {
    WithSource,
    WithDestination,
}

impl Placement {
    pub(crate) const ALL: [Self; 2] = [Self::WithSource, Self::WithDestination];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::WithSource => "etl@us-west-2",
            Self::WithDestination => "etl@us-east-1",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SourceModel {
    /// Logical decoding throughput of one walsender: "almost 250K inserts per
    /// second" [peerdb].
    pub(crate) decode_rows_per_s: f64,
    /// One `COPY … TO STDOUT` backend: 52.5 MB/s single stream [galaxy], a
    /// lower bound since the loading side may have been the bottleneck.
    pub(crate) copy_bytes_per_s: f64,
    /// 60 s community default [pgrepl]; AWS-managed Postgres uses 30 s.
    pub(crate) wal_sender_timeout_s: f64,
    /// ~164 B of WAL per inserted row including full-page images [galaxy].
    pub(crate) wal_bytes_per_row: f64,
    /// pgoutput per-row overhead: 30 B envelope, 'I'+Oid+'N', column count,
    /// then 5 B per column [pgproto].
    pub(crate) row_overhead_bytes: f64,
    pub(crate) column_overhead_bytes: f64,
    /// Begin (21 B) and Commit (26 B) messages, each in a 30 B envelope [pgproto].
    pub(crate) transaction_overhead_bytes: f64,
    /// ASSUMPTION: service time of a trivial statement on a warm backend.
    pub(crate) statement_s: f64,
    /// ASSUMPTION: COMMIT with a WAL fsync on network storage.
    pub(crate) commit_s: f64,
    /// ASSUMPTION: catalog lookups such as publication or schema queries.
    pub(crate) select_s: f64,
    /// ASSUMPTION: `CREATE_REPLICATION_SLOT … USE_SNAPSHOT`, which waits for
    /// running transactions and builds a snapshot.
    pub(crate) create_slot_s: f64,
    /// ASSUMPTION: startup migrations that find nothing to do.
    pub(crate) migrations_s: f64,
}

/// Bursty two-state arrival process: a high state at `multiplier` times the
/// mean rate for `fraction` of the time, a low state otherwise.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Burst {
    pub(crate) fraction: f64,
    pub(crate) multiplier: f64,
    pub(crate) mean_high_s: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct Workload {
    pub(crate) rows_per_s: f64,
    /// ASSUMPTION: mean rows per transaction (geometric).
    pub(crate) mean_rows_per_transaction: f64,
    pub(crate) tables: usize,
    /// ASSUMPTION: a 10-column row with 200 B of column data.
    pub(crate) columns: f64,
    pub(crate) data_bytes_per_row: f64,
    pub(crate) burst: Option<Burst>,
}

#[derive(Clone, Debug)]
pub(crate) struct EtlModel {
    /// `BatchConfig::DEFAULT_MAX_FILL_MS` = 10 000 [etl].
    pub(crate) max_fill_ms: f64,
    /// `BatchConfig::DEFAULT_MAX_BYTES` = 32 MiB [etl].
    pub(crate) max_batch_bytes: f64,
    /// `DEFAULT_MEMORY_BUDGET_RATIO` = 0.2 of detected memory [etl].
    pub(crate) memory_budget_ratio: f64,
    /// ASSUMPTION: a 2 GiB replicator pod.
    pub(crate) memory_bytes: f64,
    /// ASSUMPTION: apply-loop decode rate of pgoutput into rows.
    pub(crate) decode_rows_per_s: f64,
    /// ASSUMPTION: COPY text parse rate per copy connection.
    pub(crate) copy_parse_bytes_per_s: f64,
    /// ASSUMPTION: decoded in-memory row size relative to its pgoutput size;
    /// the batch byte target counts decoded size [etl].
    pub(crate) decoded_size_factor: f64,
    /// sqlx pool for the store: 2 connections, closed after 30 s idle [etl].
    pub(crate) store_connections: usize,
    pub(crate) store_idle_timeout_s: f64,
    /// Fixed `table_error_retry_delay_ms` = 10 000 and 5 attempts [etl].
    pub(crate) retry_delay_s: f64,
    pub(crate) retry_attempts: u32,
    /// `max_table_sync_workers` = 4, `max_copy_connections_per_table` = 4 [etl].
    pub(crate) table_sync_workers: usize,
    pub(crate) copy_connections_per_table: usize,
    /// ctid ranges: `max(connections × 4, rows / 250 000)`, at most 1024 [etl].
    pub(crate) rows_per_partition: f64,
    pub(crate) max_partitions: u64,
    /// Feedback debounce: unchanged flush LSNs are not resent within 100 ms [etl].
    pub(crate) status_debounce_s: f64,
    /// ASSUMPTION: time for the orchestrator to restart a crashed process.
    pub(crate) restart_delay_s: f64,
}

/// What the destination reports once it holds a write.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Durability {
    /// The response means durable (`ClickHouse` synchronous insert) [etl].
    Immediate,
    /// The response means accepted; durability is polled (Snowflake polls
    /// every 500 ms) [etl].
    Polled {
        interval_s: f64,
        commit_delay_s: f64,
    },
}

/// A request/response destination in us-east-1. Every value is a setting;
/// the defaults are ASSUMPTIONS resembling a managed analytical store over
/// HTTPS.
#[derive(Clone, Debug)]
pub(crate) struct SinkModel {
    pub(crate) latency_median_s: f64,
    pub(crate) latency_sigma: f64,
    pub(crate) ingest_bytes_per_s: f64,
    pub(crate) max_concurrent_requests: usize,
    /// Requests per write: one per table in the batch, as `ClickHouse` does [etl],
    /// capped here.
    pub(crate) max_parallel_requests: u64,
    /// Request bytes per byte of row data (format and compression).
    pub(crate) encoding_ratio: f64,
    pub(crate) keepalive_timeout_s: f64,
    pub(crate) ddl_s: f64,
    pub(crate) connection_pool: usize,
    pub(crate) durability: Durability,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TableSpec {
    pub(crate) rows: u64,
    pub(crate) data_bytes_per_row: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct Config {
    pub(crate) placement: Placement,
    pub(crate) tcp: TcpModel,
    pub(crate) cross_region: PathModel,
    pub(crate) intra_region: PathModel,
    pub(crate) etl_nic: NicModel,
    pub(crate) source_nic: NicModel,
    pub(crate) destination_nic: NicModel,
    /// TCP (1) + `SSLRequest` (1) + TLS 1.3 (1) + SCRAM-SHA-256 (3) [pgproto].
    pub(crate) postgres_handshake_round_trips: u32,
    /// TCP (1) + TLS 1.3 (1).
    pub(crate) https_handshake_round_trips: u32,
    pub(crate) source: SourceModel,
    pub(crate) workload: Workload,
    pub(crate) etl: EtlModel,
    pub(crate) sink: SinkModel,
    /// Tables to copy before streaming; empty means every table starts ready.
    pub(crate) copy_tables: Vec<TableSpec>,
    pub(crate) faults: FaultPlan,
    pub(crate) duration: Tick,
    pub(crate) warmup: Tick,
    /// $ per GB sent us-west-2 → us-east-1 (and the reverse) [pricing].
    pub(crate) cross_region_dollars_per_gb: f64,
}

const MIB: f64 = 1024.0 * 1024.0;
const GBPS: f64 = 1e9 / 8.0;

impl Config {
    pub(crate) fn baseline(placement: Placement) -> Self {
        Self {
            placement,
            tcp: TcpModel {
                mss_bytes: 1448.0,
                // TCP_INIT_CWND [linux].
                initial_cwnd_segments: 10.0,
                // tcp_rmem max 6 MiB through v6.15 (Amazon Linux 2023 ships
                // 6.1), advertised at the initial 50% scaling ratio [linux].
                receive_window_bytes: 3.0 * MIB,
                // tcp_wmem max 4 MiB [linux].
                send_buffer_bytes: 4.0 * MIB,
                // TCP_RTO_MIN 200 ms, TCP_RTO_MAX 120 s [linux].
                rto_min_s: 0.2,
                rto_max_s: 120.0,
                // 5 Gbps per flow outside a cluster placement group [ec2bw].
                per_flow_bytes_per_s: 5.0 * GBPS,
                // tcp_slow_start_after_idle = 1 [linux].
                slow_start_after_idle: true,
                // C = 0.4, β = 717/1024 [rfc9438], [linux].
                cubic_c: 0.4,
                cubic_beta: 0.7,
            },
            cross_region: PathModel {
                cross_region: true,
                // p50 62.4–62.8 ms [nmip]; per-connection LogNormal σ 0.07
                // fits CloudPing's p10/p50/p99 [cloudping]; floor 57 ms.
                rtt_median_s: 0.063,
                rtt_sigma: 0.07,
                rtt_floor_s: 0.057,
                // ASSUMPTION: no published per-packet jitter.
                jitter_mean_s: 0.0002,
                // No loss in 14.4M probes Oregon↔Virginia: p < 2e-7 [reda].
                loss: 1e-7,
                // Median persistence ~170 s; 1–5 ms shifts [reda].
                path_change: Some(PathChange {
                    mean_interval: 170.0,
                    rtt_step_min: 0.001,
                    rtt_step_max: 0.005,
                }),
            },
            intra_region: PathModel {
                cross_region: false,
                // Cross-AZ ~0.44–0.46 ms in us-west-2/us-east-1 [az].
                rtt_median_s: 0.0006,
                rtt_sigma: 0.3,
                rtt_floor_s: 0.0003,
                jitter_mean_s: 0.00002,
                loss: 1e-7,
                path_change: None,
            },
            // m7i.xlarge: 1.562 Gbps baseline, 12.5 Gbps burst [ec2bw];
            // ASSUMPTION: 10 minutes of burst credit (AWS: "typically 5–60").
            etl_nic: NicModel {
                baseline_bytes_per_s: 1.562 * GBPS,
                burst_bytes_per_s: 12.5 * GBPS,
                burst_seconds: 600.0,
            },
            // ASSUMPTION: an m7i.2xlarge-class database host (3.125 Gbps
            // baseline, 12.5 Gbps burst) [ec2bw].
            source_nic: NicModel {
                baseline_bytes_per_s: 3.125 * GBPS,
                burst_bytes_per_s: 12.5 * GBPS,
                burst_seconds: 600.0,
            },
            // ASSUMPTION: the managed destination's ingress never binds.
            destination_nic: NicModel {
                baseline_bytes_per_s: 25.0 * GBPS,
                burst_bytes_per_s: 25.0 * GBPS,
                burst_seconds: 0.0,
            },
            postgres_handshake_round_trips: 6,
            https_handshake_round_trips: 2,
            source: SourceModel {
                decode_rows_per_s: 250_000.0,
                copy_bytes_per_s: 52.5e6,
                wal_sender_timeout_s: 60.0,
                wal_bytes_per_row: 164.0,
                row_overhead_bytes: 38.0,
                column_overhead_bytes: 5.0,
                transaction_overhead_bytes: 107.0,
                statement_s: 0.0001,
                commit_s: 0.001,
                select_s: 0.0005,
                create_slot_s: 0.02,
                migrations_s: 0.005,
            },
            workload: Workload {
                rows_per_s: 1000.0,
                mean_rows_per_transaction: 5.0,
                tables: 8,
                columns: 10.0,
                data_bytes_per_row: 200.0,
                // ASSUMPTION: 10% of the time at 3× the mean rate, ~5 s bursts.
                burst: Some(Burst {
                    fraction: 0.1,
                    multiplier: 3.0,
                    mean_high_s: 5.0,
                }),
            },
            etl: EtlModel {
                max_fill_ms: 10_000.0,
                max_batch_bytes: 32.0 * MIB,
                memory_budget_ratio: 0.2,
                memory_bytes: 2048.0 * MIB,
                decode_rows_per_s: 400_000.0,
                copy_parse_bytes_per_s: 150e6,
                decoded_size_factor: 2.0,
                store_connections: 2,
                store_idle_timeout_s: 30.0,
                retry_delay_s: 10.0,
                retry_attempts: 5,
                table_sync_workers: 4,
                copy_connections_per_table: 4,
                rows_per_partition: 250_000.0,
                max_partitions: 1024,
                status_debounce_s: 0.1,
                restart_delay_s: 10.0,
            },
            sink: SinkModel {
                latency_median_s: 0.025,
                latency_sigma: 0.5,
                ingest_bytes_per_s: 100e6,
                max_concurrent_requests: 32,
                max_parallel_requests: 8,
                encoding_ratio: 1.1,
                keepalive_timeout_s: 10.0,
                ddl_s: 0.2,
                connection_pool: 16,
                durability: Durability::Immediate,
            },
            copy_tables: Vec::new(),
            faults: FaultPlan::default(),
            duration: ticks(300.0),
            warmup: ticks(60.0),
            cross_region_dollars_per_gb: 0.02,
        }
    }

    pub(crate) fn source_path(&self) -> PathModel {
        match self.placement {
            Placement::WithSource => self.intra_region.clone(),
            Placement::WithDestination => self.cross_region.clone(),
        }
    }

    pub(crate) fn destination_path(&self) -> PathModel {
        match self.placement {
            Placement::WithSource => self.cross_region.clone(),
            Placement::WithDestination => self.intra_region.clone(),
        }
    }

    /// pgoutput bytes for one row change.
    pub(crate) fn row_wire_bytes(&self, data_bytes: f64) -> f64 {
        self.source.column_overhead_bytes.mul_add(
            self.workload.columns,
            self.source.row_overhead_bytes + data_bytes,
        )
    }

    /// ETL's decoded size estimate for one row, which the batch target counts.
    pub(crate) fn decoded_bytes(&self, data_bytes: f64) -> f64 {
        self.etl.decoded_size_factor * self.row_wire_bytes(data_bytes)
    }

    pub(crate) fn request_bytes(&self, data_bytes: f64) -> f64 {
        self.sink.encoding_ratio * data_bytes
    }

    /// COPY text: column data, a delimiter per column, and a 5 B `CopyData`
    /// header.
    pub(crate) fn copy_wire_bytes(&self, data_bytes: f64) -> f64 {
        data_bytes + self.workload.columns + 5.0
    }

    /// Per-batch decoded-byte target: `min(max_bytes, ratio × memory ÷ slots)`.
    pub(crate) fn batch_target_bytes(&self, slots: usize) -> f64 {
        let share = self.etl.memory_budget_ratio * self.etl.memory_bytes / slots.max(1) as f64;
        self.etl.max_batch_bytes.min(share).max(1.0)
    }
}
