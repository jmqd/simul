//! The destination in us-east-1: an HTTPS request/response sink with bounded
//! concurrency and a lognormal service time.

use crate::config::{Config, Durability};
use crate::metrics::SharedRecorder;
use crate::net::{Endpoint, NetEvent, Peer};
use crate::rng::Rng;
use crate::time::{ticks, ticks_at_least_one, Tick};
use crate::wire::{ConnId, HttpKind, Unit};
use simul::agent::{Agent, AgentContext};
use simul::message::Message;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::rc::Rc;

pub(crate) const NAME: &str = "destination";

#[derive(Clone, Copy, Debug)]
struct Request {
    conn: ConnId,
    id: u64,
    bytes: u64,
    kind: HttpKind,
    /// For inserts, the ETL write this request is part of; for polls, the
    /// insert request being polled.
    of: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct DestinationAgent {
    config: Rc<Config>,
    recorder: SharedRecorder,
    net: Endpoint,
    network_rng: Rng,
    service_rng: Rng,
    waiting: VecDeque<Request>,
    running: Vec<(Tick, Request)>,
    /// When each accepted insert becomes durable, for polls.
    durable_at: BTreeMap<u64, Tick>,
    /// Accepted inserts not yet durable: when they land, and their write.
    landing: Vec<(Tick, u64)>,
    /// Connections the client reset. Work already running on them still
    /// lands, but its response has no reader.
    reset: BTreeSet<ConnId>,
}

impl DestinationAgent {
    pub(crate) fn new(
        config: Rc<Config>,
        recorder: SharedRecorder,
        network_rng: Rng,
        service_rng: Rng,
        etl: Peer,
    ) -> Self {
        let net = Endpoint::new(2, config.tcp.clone(), config.destination_nic, vec![etl]);
        Self {
            config,
            recorder,
            net,
            network_rng,
            service_rng,
            waiting: VecDeque::new(),
            running: Vec::new(),
            durable_at: BTreeMap::new(),
            landing: Vec::new(),
            reset: BTreeSet::new(),
        }
    }

    fn brownout(&self, now: Tick) -> f64 {
        self.config
            .faults
            .destination_brownouts
            .iter()
            .filter(|(window, _)| window.contains(now))
            .map(|(_, factor)| *factor)
            .fold(1.0, f64::max)
    }

    fn service_seconds(&mut self, request: Request, now: Tick) -> f64 {
        let sink = &self.config.sink;
        let base = match request.kind {
            HttpKind::Insert => {
                self.service_rng
                    .lognormal(sink.latency_median_s, sink.latency_sigma)
                    + request.bytes as f64 / sink.ingest_bytes_per_s
            }
            HttpKind::Poll => self.service_rng.lognormal(0.005, sink.latency_sigma),
            HttpKind::Ddl => self.service_rng.lognormal(sink.ddl_s, sink.latency_sigma),
        };
        base * self.brownout(now)
    }

    fn start_waiting(&mut self, now: Tick) {
        while self.running.len() < self.config.sink.max_concurrent_requests {
            let Some(request) = self.waiting.pop_front() else {
                break;
            };
            let done_at = now + ticks_at_least_one(self.service_seconds(request, now));
            self.running.push((done_at, request));
        }
    }

    /// Reports accepted inserts once they are durable; each part is reported
    /// exactly once, here or at completion for immediate durability.
    fn land_due(&mut self, now: Tick) {
        let mut index = 0;
        while index < self.landing.len() {
            let (at, write) = self.landing[index];
            if at <= now {
                self.landing.swap_remove(index);
                self.recorder.borrow_mut().part_durable(write, at);
            } else {
                index += 1;
            }
        }
    }

    fn finish_due(&mut self, now: Tick) {
        let mut index = 0;
        while index < self.running.len() {
            let (done_at, request) = self.running[index];
            if done_at > now {
                index += 1;
                continue;
            }
            self.running.swap_remove(index);
            let durable_at = match (request.kind, self.config.sink.durability) {
                (HttpKind::Insert, Durability::Immediate) => {
                    self.recorder.borrow_mut().part_durable(request.of, now);
                    now
                }
                (HttpKind::Insert, Durability::Polled { commit_delay_s, .. }) => {
                    let durable = now + ticks(commit_delay_s);
                    self.landing.push((durable, request.of));
                    self.durable_at.insert(request.id, durable);
                    0
                }
                (HttpKind::Ddl, _) => now,
                (HttpKind::Poll, _) => match self.durable_at.get(&request.of) {
                    Some(&at) if at <= now => {
                        self.durable_at.remove(&request.of);
                        at
                    }
                    Some(_) | None => 0,
                },
            };
            if self.reset.contains(&request.conn) {
                continue;
            }
            let sent = self.net.send(
                request.conn,
                Unit::HttpResponse {
                    id: request.id,
                    durable_at,
                },
            );
            assert!(
                sent,
                "a request's connection stays open until the client resets it"
            );
        }
    }
}

impl Agent for DestinationAgent {
    fn on_message(&mut self, ctx: &mut AgentContext, message: &Message) {
        let now = ctx.time();
        let events = self
            .net
            .receive(message, now, &self.config.faults, &mut self.network_rng);
        for event in events {
            match event {
                NetEvent::Data(
                    conn,
                    Unit::HttpRequest {
                        id,
                        bytes,
                        kind,
                        of,
                    },
                ) => self.waiting.push_back(Request {
                    conn,
                    id,
                    bytes,
                    kind,
                    of,
                }),
                NetEvent::Accepted(..) => {}
                NetEvent::Closed(conn) => {
                    self.reset.insert(conn);
                }
                NetEvent::Data(_, other) => panic!("destination got unexpected {other:?}"),
                NetEvent::Connected(_) => panic!("destination opens no connections"),
            }
        }
    }

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        let now = ctx.time();
        self.land_due(now);
        self.finish_due(now);
        self.start_waiting(now);
        self.recorder.borrow_mut().cross_region_bytes[2] = self.net.cross_region_bytes;
        self.net
            .pump(ctx, now, &self.config.faults, &mut self.network_rng);
    }
}
