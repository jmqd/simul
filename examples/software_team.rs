//! Simulates software delivery flow under meetings, Slack, CI, and decisions.
#![allow(
    clippy::integer_division,
    clippy::missing_const_for_fn,
    clippy::missing_docs_in_private_items,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "self-contained example with discrete tick arithmetic and local domain types"
)]

use rustc_hash::FxHashSet;
use simul::agent::{Agent, AgentContext, AgentId, AgentInitializer};
use simul::message::Message;
use simul::{DiscreteTime, Halt, Simulation};
use std::collections::VecDeque;

/// Every agent here sends and receives `Payload`.
type Ctx<'a> = AgentContext<'a, Payload>;

const TICKS_PER_DAY: DiscreteTime = 16;
const WORK_DAYS_PER_YEAR: DiscreteTime = 260;
const DEVELOPER_COUNT: u8 = 3;

const PRODUCT: &str = "product";
const MEETINGS: &str = "meetings";
const SLACK: &str = "slack";
const DECISIONS: &str = "decisions";
const CI: &str = "ci";
const MERGE: &str = "merge";
const STOPPER: &str = "stopper";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SoftwareTeamParameters {
    work_days: DiscreteTime,
    ticks_per_day: DiscreteTime,
    communication_load_per_mille: u16,
    meeting_load_per_mille: u16,
    ci_wait_time: DiscreteTime,
    context_switch_cost: DiscreteTime,
    decision_latency: DiscreteTime,
    release_interval: DiscreteTime,
    base_coding_ticks: u16,
    review_ticks: u16,
    rework_ticks: u16,
    decision_request_every: u16,
    ci_failure_every: u16,
}

impl Default for SoftwareTeamParameters {
    fn default() -> Self {
        Self {
            work_days: WORK_DAYS_PER_YEAR,
            ticks_per_day: TICKS_PER_DAY,
            communication_load_per_mille: 125,
            meeting_load_per_mille: 125,
            ci_wait_time: 4,
            context_switch_cost: 1,
            decision_latency: 6,
            release_interval: 8,
            base_coding_ticks: 5,
            review_ticks: 2,
            rework_ticks: 2,
            decision_request_every: 4,
            ci_failure_every: 7,
        }
    }
}

impl SoftwareTeamParameters {
    fn total_ticks(self) -> DiscreteTime {
        self.work_days.saturating_mul(self.ticks_per_day)
    }

    fn communication_ticks_per_day(self) -> DiscreteTime {
        self.load_ticks_per_day(self.communication_load_per_mille)
    }

    fn meeting_ticks_per_day(self) -> DiscreteTime {
        self.load_ticks_per_day(self.meeting_load_per_mille)
    }

    fn load_ticks_per_day(self, load_per_mille: u16) -> DiscreteTime {
        self.ticks_per_day
            .saturating_mul(DiscreteTime::from(load_per_mille))
            .div_ceil(1_000)
            .min(self.ticks_per_day)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Payload {
    Tick,
    Work {
        ticket_id: u16,
        coding_ticks: u16,
        needs_decision: bool,
    },
    Meeting {
        ticks: u16,
    },
    Slack {
        ticks: u16,
    },
    DecisionRequest {
        ticket_id: u16,
        requester: u8,
    },
    DecisionMade {
        ticket_id: u16,
    },
    ReviewRequest {
        ticket_id: u16,
        author: u8,
        review_ticks: u16,
    },
    ReviewApproved {
        ticket_id: u16,
    },
    CiRequest {
        ticket_id: u16,
        author: u8,
    },
    CiPassed {
        ticket_id: u16,
    },
    CiFailed {
        ticket_id: u16,
    },
    Merge {
        ticket_id: u16,
        author: u8,
    },
}

fn resolve_developers(ctx: &Ctx<'_>) -> Option<[AgentId; 3]> {
    Some([
        ctx.agent_id(developer_name(0))?,
        ctx.agent_id(developer_name(1))?,
        ctx.agent_id(developer_name(2))?,
    ])
}

fn has_tick_queued(ctx: &Ctx<'_>) -> bool {
    ctx.state()
        .queue
        .iter()
        .any(|message| matches!(message.payload, Payload::Tick))
}

fn ensure_self_tick(ctx: &mut Ctx<'_>) {
    if !has_tick_queued(ctx) {
        ctx.send_to(ctx.id(), Payload::Tick);
    }
}

const fn initial_tick() -> Message<Payload> {
    Message::new(0, Payload::Tick)
}

const fn discrete_multiple(value: DiscreteTime, divisor: DiscreteTime) -> bool {
    divisor != 0 && value % divisor == 0
}

const fn short_multiple(value: u16, divisor: u16) -> bool {
    divisor != 0 && value % divisor == 0
}

fn assignee_for_ticket(ticket_id: u16) -> u8 {
    let assignee = ticket_id.saturating_sub(1) % u16::from(DEVELOPER_COUNT);
    u8::try_from(assignee).unwrap_or(0)
}

fn developer_name(id: u8) -> &'static str {
    match id {
        0 => "alice",
        1 => "bob",
        _ => "carol",
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Ticket {
    id: u16,
    remaining_coding: DiscreteTime,
    needs_decision: bool,
    decision_requested: bool,
    decision_received: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReviewJob {
    ticket_id: u16,
    author: u8,
    remaining_review: DiscreteTime,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlockKind {
    Meeting,
    Slack,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkKind {
    Coding,
    Review,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DevPeers {
    devs: [AgentId; 3],
    reviewer: AgentId,
    decisions: AgentId,
    ci: AgentId,
    merge: AgentId,
}

#[derive(Clone, Debug)]
struct Developer {
    id: u8,
    params: SoftwareTeamParameters,
    peers: Option<DevPeers>,
    backlog: VecDeque<Ticket>,
    review_queue: VecDeque<ReviewJob>,
    active_coding: Option<Ticket>,
    active_review: Option<ReviewJob>,
    blocked_until: DiscreteTime,
    blocked_kind: Option<BlockKind>,
    context_switch_remaining: DiscreteTime,
    last_work_kind: Option<WorkKind>,
    review_requests: usize,
    ci_requests: usize,
    decision_requests: usize,
    coding_ticks: usize,
    review_ticks: usize,
    meeting_ticks: usize,
    slack_ticks: usize,
    context_switch_ticks: usize,
    decision_wait_ticks: usize,
}

impl Developer {
    fn new(id: u8, params: SoftwareTeamParameters) -> Self {
        Self {
            id,
            params,
            peers: None,
            backlog: VecDeque::new(),
            review_queue: VecDeque::new(),
            active_coding: None,
            active_review: None,
            blocked_until: 0,
            blocked_kind: None,
            context_switch_remaining: 0,
            last_work_kind: None,
            review_requests: 0,
            ci_requests: 0,
            decision_requests: 0,
            coding_ticks: 0,
            review_ticks: 0,
            meeting_ticks: 0,
            slack_ticks: 0,
            context_switch_ticks: 0,
            decision_wait_ticks: 0,
        }
    }

    fn resolve_peers(&mut self, ctx: &Ctx<'_>) -> Option<DevPeers> {
        if self.peers.is_none() {
            let devs = resolve_developers(ctx)?;
            let reviewer = devs[usize::from((self.id + 1) % DEVELOPER_COUNT)];
            self.peers = Some(DevPeers {
                devs,
                reviewer,
                decisions: ctx.agent_id(DECISIONS)?,
                ci: ctx.agent_id(CI)?,
                merge: ctx.agent_id(MERGE)?,
            });
        }
        self.peers
    }

    fn has_pending_work(&self) -> bool {
        self.active_coding.is_some()
            || self.active_review.is_some()
            || !self.backlog.is_empty()
            || !self.review_queue.is_empty()
    }

    fn block_for(&mut self, ctx: &Ctx<'_>, ticks: u16, block_kind: BlockKind) {
        if ticks == 0 {
            return;
        }

        if self.has_pending_work() {
            self.context_switch_remaining = self
                .context_switch_remaining
                .saturating_add(self.params.context_switch_cost);
        }

        self.blocked_until = self
            .blocked_until
            .max(ctx.time().saturating_add(DiscreteTime::from(ticks)));
        self.blocked_kind = Some(block_kind);
        self.last_work_kind = None;
    }

    fn on_tick(&mut self, ctx: &mut Ctx<'_>, peers: DevPeers) {
        if ctx.time() < self.blocked_until {
            match self.blocked_kind {
                Some(BlockKind::Meeting) => self.meeting_ticks += 1,
                Some(BlockKind::Slack) => self.slack_ticks += 1,
                None => {}
            }
            ensure_self_tick(ctx);
            return;
        }

        if self.context_switch_remaining > 0 {
            self.context_switch_remaining -= 1;
            self.context_switch_ticks += 1;
            ensure_self_tick(ctx);
            return;
        }

        if self.process_review(ctx, peers) || self.process_coding(ctx, peers) {
            ensure_self_tick(ctx);
            return;
        }

        if !self.backlog.is_empty() {
            self.decision_wait_ticks += 1;
        }

        ensure_self_tick(ctx);
    }

    fn process_review(&mut self, ctx: &mut Ctx<'_>, peers: DevPeers) -> bool {
        if self.active_review.is_none() {
            self.active_review = self.review_queue.pop_front();
        }

        let Some(mut review) = self.active_review else {
            return false;
        };

        if self.switch_to(WorkKind::Review) {
            self.active_review = Some(review);
            return true;
        }

        review.remaining_review = review.remaining_review.saturating_sub(1);
        self.review_ticks += 1;

        if review.remaining_review == 0 {
            let author_idx = usize::from(review.author.min(DEVELOPER_COUNT - 1));
            ctx.send_to(
                peers.devs[author_idx],
                Payload::ReviewApproved {
                    ticket_id: review.ticket_id,
                },
            );
            self.active_review = None;
        } else {
            self.active_review = Some(review);
        }

        true
    }

    fn process_coding(&mut self, ctx: &mut Ctx<'_>, peers: DevPeers) -> bool {
        if self.active_coding.is_none() {
            self.active_coding = self.next_ready_ticket(ctx, peers);
        }

        let Some(mut ticket) = self.active_coding else {
            return false;
        };

        if self.switch_to(WorkKind::Coding) {
            self.active_coding = Some(ticket);
            return true;
        }

        ticket.remaining_coding = ticket.remaining_coding.saturating_sub(1);
        self.coding_ticks += 1;

        if ticket.remaining_coding == 0 {
            ctx.send_to(
                peers.reviewer,
                Payload::ReviewRequest {
                    ticket_id: ticket.id,
                    author: self.id,
                    review_ticks: self.params.review_ticks,
                },
            );
            self.review_requests += 1;
            self.active_coding = None;
        } else {
            self.active_coding = Some(ticket);
        }

        true
    }

    fn next_ready_ticket(&mut self, ctx: &mut Ctx<'_>, peers: DevPeers) -> Option<Ticket> {
        let backlog_len = self.backlog.len();
        for _ in 0..backlog_len {
            let mut ticket = self.backlog.pop_front()?;

            if ticket.needs_decision && !ticket.decision_received {
                if !ticket.decision_requested {
                    ticket.decision_requested = true;
                    ctx.send_to(
                        peers.decisions,
                        Payload::DecisionRequest {
                            ticket_id: ticket.id,
                            requester: self.id,
                        },
                    );
                    self.decision_requests += 1;
                }
                self.backlog.push_back(ticket);
                continue;
            }

            return Some(ticket);
        }

        None
    }

    fn switch_to(&mut self, kind: WorkKind) -> bool {
        if self.last_work_kind.is_some_and(|last| last != kind)
            && self.params.context_switch_cost > 0
        {
            self.context_switch_remaining = self.params.context_switch_cost;
            self.last_work_kind = Some(kind);
            return true;
        }

        self.last_work_kind = Some(kind);
        false
    }

    fn receive_decision(&mut self, ticket_id: u16) {
        if let Some(ticket) = self
            .backlog
            .iter_mut()
            .find(|ticket| ticket.id == ticket_id)
        {
            ticket.decision_received = true;
        }

        if let Some(ticket) = self.active_coding.as_mut() {
            if ticket.id == ticket_id {
                ticket.decision_received = true;
            }
        }
    }
}

impl Agent<Payload> for Developer {
    fn on_message(&mut self, ctx: &mut Ctx<'_>, msg: &Message<Payload>) {
        let Some(peers) = self.resolve_peers(ctx) else {
            return;
        };
        match msg.payload {
            Payload::Tick => self.on_tick(ctx, peers),
            Payload::Work {
                ticket_id,
                coding_ticks,
                needs_decision,
            } => self.backlog.push_back(Ticket {
                id: ticket_id,
                remaining_coding: DiscreteTime::from(coding_ticks),
                needs_decision,
                decision_requested: false,
                decision_received: !needs_decision,
            }),
            Payload::Meeting { ticks } => self.block_for(ctx, ticks, BlockKind::Meeting),
            Payload::Slack { ticks } => self.block_for(ctx, ticks, BlockKind::Slack),
            Payload::DecisionMade { ticket_id } => self.receive_decision(ticket_id),
            Payload::ReviewRequest {
                ticket_id,
                author,
                review_ticks,
            } => self.review_queue.push_back(ReviewJob {
                ticket_id,
                author,
                remaining_review: DiscreteTime::from(review_ticks),
            }),
            Payload::ReviewApproved { ticket_id } => {
                ctx.send_to(
                    peers.ci,
                    Payload::CiRequest {
                        ticket_id,
                        author: self.id,
                    },
                );
                self.ci_requests += 1;
            }
            Payload::CiPassed { ticket_id } => ctx.send_to(
                peers.merge,
                Payload::Merge {
                    ticket_id,
                    author: self.id,
                },
            ),
            Payload::CiFailed { ticket_id } => {
                self.backlog.push_back(Ticket {
                    id: ticket_id,
                    remaining_coding: DiscreteTime::from(self.params.rework_ticks),
                    needs_decision: false,
                    decision_requested: false,
                    decision_received: true,
                });
            }
            Payload::DecisionRequest { .. } | Payload::CiRequest { .. } | Payload::Merge { .. } => {
            }
        }

        ensure_self_tick(ctx);
    }
}

#[derive(Clone, Debug)]
struct ProductManager {
    params: SoftwareTeamParameters,
    total_ticks: DiscreteTime,
    devs: Option<[AgentId; 3]>,
    next_ticket_id: u16,
    released: usize,
}

impl ProductManager {
    fn new(params: SoftwareTeamParameters) -> Self {
        Self {
            params,
            total_ticks: params.total_ticks(),
            devs: None,
            next_ticket_id: 1,
            released: 0,
        }
    }
}

impl Agent<Payload> for ProductManager {
    fn on_tick(&mut self, ctx: &mut Ctx<'_>) {
        if ctx.time() >= self.total_ticks || self.params.release_interval == 0 {
            return;
        }

        if !discrete_multiple(ctx.time(), self.params.release_interval) {
            return;
        }

        if self.devs.is_none() {
            self.devs = resolve_developers(ctx);
        }
        let Some(devs) = self.devs else {
            return;
        };

        let ticket_id = self.next_ticket_id;
        self.next_ticket_id = self.next_ticket_id.saturating_add(1);
        let assignee = assignee_for_ticket(ticket_id);
        let coding_ticks = self.params.base_coding_ticks.saturating_add(ticket_id % 5);
        let needs_decision = short_multiple(ticket_id, self.params.decision_request_every);

        ctx.send_to(
            devs[usize::from(assignee)],
            Payload::Work {
                ticket_id,
                coding_ticks,
                needs_decision,
            },
        );
        self.released += 1;
    }
}

#[derive(Clone, Debug)]
struct MeetingScheduler {
    ticks_per_day: DiscreteTime,
    total_ticks: DiscreteTime,
    meeting_ticks: u16,
    devs: Option<[AgentId; 3]>,
}

impl MeetingScheduler {
    fn new(params: SoftwareTeamParameters) -> Self {
        Self {
            ticks_per_day: params.ticks_per_day,
            total_ticks: params.total_ticks(),
            meeting_ticks: u16::try_from(params.meeting_ticks_per_day()).unwrap_or(u16::MAX),
            devs: None,
        }
    }
}

impl Agent<Payload> for MeetingScheduler {
    fn on_tick(&mut self, ctx: &mut Ctx<'_>) {
        if self.meeting_ticks == 0
            || ctx.time() >= self.total_ticks
            || !discrete_multiple(ctx.time(), self.ticks_per_day)
        {
            return;
        }

        if self.devs.is_none() {
            self.devs = resolve_developers(ctx);
        }
        let Some(devs) = self.devs else {
            return;
        };

        for dev in devs {
            ctx.send_to(
                dev,
                Payload::Meeting {
                    ticks: self.meeting_ticks,
                },
            );
        }
    }
}

#[derive(Clone, Debug)]
struct CommunicationScheduler {
    ticks_per_day: DiscreteTime,
    total_ticks: DiscreteTime,
    communication_ticks: u16,
    target_tick_in_day: DiscreteTime,
    devs: Option<[AgentId; 3]>,
}

impl CommunicationScheduler {
    fn new(params: SoftwareTeamParameters) -> Self {
        let target_tick_in_day = if params.ticks_per_day > 0 {
            params.meeting_ticks_per_day().min(params.ticks_per_day - 1)
        } else {
            0
        };
        Self {
            ticks_per_day: params.ticks_per_day,
            total_ticks: params.total_ticks(),
            communication_ticks: u16::try_from(params.communication_ticks_per_day())
                .unwrap_or(u16::MAX),
            target_tick_in_day,
            devs: None,
        }
    }
}

impl Agent<Payload> for CommunicationScheduler {
    fn on_tick(&mut self, ctx: &mut Ctx<'_>) {
        if self.communication_ticks == 0 || ctx.time() >= self.total_ticks {
            return;
        }

        let tick_in_day = ctx.time() % self.ticks_per_day;
        if tick_in_day != self.target_tick_in_day {
            return;
        }

        if self.devs.is_none() {
            self.devs = resolve_developers(ctx);
        }
        let Some(devs) = self.devs else {
            return;
        };

        for dev in devs {
            ctx.send_to(
                dev,
                Payload::Slack {
                    ticks: self.communication_ticks,
                },
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingDecision {
    ticket_id: u16,
    requester: u8,
    due_at: DiscreteTime,
}

#[derive(Clone, Debug)]
struct DecisionMaker {
    params: SoftwareTeamParameters,
    devs: Option<[AgentId; 3]>,
    pending: Vec<PendingDecision>,
}

impl DecisionMaker {
    const fn new(params: SoftwareTeamParameters) -> Self {
        Self {
            params,
            devs: None,
            pending: Vec::new(),
        }
    }

    fn on_tick(&mut self, ctx: &mut Ctx<'_>) {
        if self.devs.is_none() {
            self.devs = resolve_developers(ctx);
        }
        let Some(devs) = self.devs else {
            ensure_self_tick(ctx);
            return;
        };
        self.pending.retain(|decision| {
            if ctx.time() >= decision.due_at {
                let idx = usize::from(decision.requester.min(DEVELOPER_COUNT - 1));
                ctx.send_to(
                    devs[idx],
                    Payload::DecisionMade {
                        ticket_id: decision.ticket_id,
                    },
                );
                false
            } else {
                true
            }
        });
        ensure_self_tick(ctx);
    }
}

impl Agent<Payload> for DecisionMaker {
    fn on_message(&mut self, ctx: &mut Ctx<'_>, msg: &Message<Payload>) {
        match msg.payload {
            Payload::Tick => self.on_tick(ctx),
            Payload::DecisionRequest {
                ticket_id,
                requester,
            } => self.pending.push(PendingDecision {
                ticket_id,
                requester,
                due_at: ctx.time().saturating_add(self.params.decision_latency),
            }),
            Payload::Work { .. }
            | Payload::Meeting { .. }
            | Payload::Slack { .. }
            | Payload::DecisionMade { .. }
            | Payload::ReviewRequest { .. }
            | Payload::ReviewApproved { .. }
            | Payload::CiRequest { .. }
            | Payload::CiPassed { .. }
            | Payload::CiFailed { .. }
            | Payload::Merge { .. } => {}
        }
        ensure_self_tick(ctx);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CiJob {
    ticket_id: u16,
    author: u8,
    started_at: DiscreteTime,
}

#[derive(Clone, Debug)]
struct ContinuousIntegration {
    params: SoftwareTeamParameters,
    devs: Option<[AgentId; 3]>,
    queue: VecDeque<(u16, u8)>,
    active: Option<CiJob>,
    failed_once: FxHashSet<u16>,
    ci_failures: usize,
}

impl ContinuousIntegration {
    fn new(params: SoftwareTeamParameters) -> Self {
        Self {
            params,
            devs: None,
            queue: VecDeque::new(),
            active: None,
            failed_once: FxHashSet::default(),
            ci_failures: 0,
        }
    }

    fn on_tick(&mut self, ctx: &mut Ctx<'_>) {
        if self.active.is_none() {
            self.active = self.queue.pop_front().map(|(ticket_id, author)| CiJob {
                ticket_id,
                author,
                started_at: ctx.time(),
            });
        }

        let Some(job) = self.active else {
            ensure_self_tick(ctx);
            return;
        };

        if ctx.time() < job.started_at.saturating_add(self.params.ci_wait_time) {
            ensure_self_tick(ctx);
            return;
        }

        if self.devs.is_none() {
            self.devs = resolve_developers(ctx);
        }
        let Some(devs) = self.devs else {
            ensure_self_tick(ctx);
            return;
        };
        let dev = devs[usize::from(job.author.min(DEVELOPER_COUNT - 1))];

        if self.should_fail(job.ticket_id) {
            self.failed_once.insert(job.ticket_id);
            self.ci_failures += 1;
            ctx.send_to(
                dev,
                Payload::CiFailed {
                    ticket_id: job.ticket_id,
                },
            );
        } else {
            ctx.send_to(
                dev,
                Payload::CiPassed {
                    ticket_id: job.ticket_id,
                },
            );
        }

        self.active = None;
        ensure_self_tick(ctx);
    }

    fn should_fail(&self, ticket_id: u16) -> bool {
        self.params.ci_failure_every != 0
            && short_multiple(ticket_id, self.params.ci_failure_every)
            && !self.failed_once.contains(&ticket_id)
    }
}

impl Agent<Payload> for ContinuousIntegration {
    fn on_message(&mut self, ctx: &mut Ctx<'_>, msg: &Message<Payload>) {
        match msg.payload {
            Payload::Tick => self.on_tick(ctx),
            Payload::CiRequest { ticket_id, author } => {
                self.queue.push_back((ticket_id, author));
            }
            Payload::Work { .. }
            | Payload::Meeting { .. }
            | Payload::Slack { .. }
            | Payload::DecisionRequest { .. }
            | Payload::DecisionMade { .. }
            | Payload::ReviewRequest { .. }
            | Payload::ReviewApproved { .. }
            | Payload::CiPassed { .. }
            | Payload::CiFailed { .. }
            | Payload::Merge { .. } => {}
        }
        ensure_self_tick(ctx);
    }
}

#[derive(Clone, Debug, Default)]
struct MergeBot {
    merged: usize,
}

impl Agent<Payload> for MergeBot {
    fn on_message(&mut self, _ctx: &mut Ctx<'_>, msg: &Message<Payload>) {
        if matches!(msg.payload, Payload::Merge { .. }) {
            self.merged += 1;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Stopper {
    stop_at: DiscreteTime,
}

impl Agent<Payload> for Stopper {
    fn on_tick(&mut self, ctx: &mut Ctx<'_>) {
        if ctx.time() >= self.stop_at {
            ctx.send_halt_interrupt("software team year elapsed");
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SoftwareTeamReport {
    elapsed_ticks: DiscreteTime,
    released: usize,
    merged: usize,
    review_requests: usize,
    ci_requests: usize,
    ci_failures: usize,
    decision_requests: usize,
    coding_ticks: usize,
    review_ticks: usize,
    meeting_ticks: usize,
    slack_ticks: usize,
    context_switch_ticks: usize,
    decision_wait_ticks: usize,
}

impl SoftwareTeamReport {
    fn from_simulation(simulation: &Simulation<Payload>) -> Self {
        let mut report = Self {
            elapsed_ticks: simulation.time(),
            ..Default::default()
        };

        if let Some(pm) = simulation
            .find_by_name(PRODUCT)
            .and_then(|agent| agent.downcast_ref::<ProductManager>())
        {
            report.released = pm.released;
        }
        if let Some(ci) = simulation
            .find_by_name(CI)
            .and_then(|agent| agent.downcast_ref::<ContinuousIntegration>())
        {
            report.ci_failures = ci.ci_failures;
        }
        if let Some(merge) = simulation
            .find_by_name(MERGE)
            .and_then(|agent| agent.downcast_ref::<MergeBot>())
        {
            report.merged = merge.merged;
        }
        for id in 0..DEVELOPER_COUNT {
            if let Some(dev) = simulation
                .find_by_name(developer_name(id))
                .and_then(|agent| agent.downcast_ref::<Developer>())
            {
                report.review_requests += dev.review_requests;
                report.ci_requests += dev.ci_requests;
                report.decision_requests += dev.decision_requests;
                report.coding_ticks += dev.coding_ticks;
                report.review_ticks += dev.review_ticks;
                report.meeting_ticks += dev.meeting_ticks;
                report.slack_ticks += dev.slack_ticks;
                report.context_switch_ticks += dev.context_switch_ticks;
                report.decision_wait_ticks += dev.decision_wait_ticks;
            }
        }

        report
    }
}

fn build_simulation(params: SoftwareTeamParameters) -> Simulation<Payload> {
    Simulation::builder(Halt::Never)
        .agents([
            AgentInitializer::proactive(PRODUCT, ProductManager::new(params)),
            AgentInitializer::proactive(MEETINGS, MeetingScheduler::new(params)),
            AgentInitializer::proactive(SLACK, CommunicationScheduler::new(params)),
            AgentInitializer::reactive(developer_name(0), Developer::new(0, params))
                .initial_queue([initial_tick()]),
            AgentInitializer::reactive(developer_name(1), Developer::new(1, params))
                .initial_queue([initial_tick()]),
            AgentInitializer::reactive(developer_name(2), Developer::new(2, params))
                .initial_queue([initial_tick()]),
            AgentInitializer::reactive(DECISIONS, DecisionMaker::new(params))
                .initial_queue([initial_tick()]),
            AgentInitializer::reactive(CI, ContinuousIntegration::new(params))
                .initial_queue([initial_tick()]),
            AgentInitializer::reactive(MERGE, MergeBot::default()),
            AgentInitializer::proactive(
                STOPPER,
                Stopper {
                    stop_at: params.total_ticks(),
                },
            ),
        ])
        .build()
        .unwrap_or_else(|error| panic!("{error}"))
}

fn run_software_team(params: SoftwareTeamParameters) -> SoftwareTeamReport {
    let mut simulation = build_simulation(params);
    simulation.run();
    SoftwareTeamReport::from_simulation(&simulation)
}

fn print_report(params: SoftwareTeamParameters, report: SoftwareTeamReport) {
    println!("software team simulation");
    println!("work days: {}", params.work_days);
    println!("ticks per day: {}", params.ticks_per_day);
    println!(
        "communication load: {} per mille of the workday",
        params.communication_load_per_mille
    );
    println!(
        "meeting load: {} per mille of the workday",
        params.meeting_load_per_mille
    );
    println!("CI wait time: {} ticks", params.ci_wait_time);
    println!("decision latency: {} ticks", params.decision_latency);
    println!("context switch cost: {} ticks", params.context_switch_cost);
    println!();
    println!("released tickets: {}", report.released);
    println!("merged tickets: {}", report.merged);
    println!("review requests: {}", report.review_requests);
    println!("CI requests: {}", report.ci_requests);
    println!("CI failures: {}", report.ci_failures);
    println!("decision requests: {}", report.decision_requests);
    println!();
    println!("coding ticks: {}", report.coding_ticks);
    println!("review ticks: {}", report.review_ticks);
    println!("meeting ticks: {}", report.meeting_ticks);
    println!("Slack/communication ticks: {}", report.slack_ticks);
    println!("context-switch ticks: {}", report.context_switch_ticks);
    println!("decision-wait ticks: {}", report.decision_wait_ticks);
}

fn main() {
    let params = SoftwareTeamParameters::default();
    let report = run_software_team(params);
    print_report(params, report);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn short_params() -> SoftwareTeamParameters {
        SoftwareTeamParameters {
            work_days: 30,
            communication_load_per_mille: 0,
            meeting_load_per_mille: 0,
            ci_wait_time: 2,
            context_switch_cost: 0,
            decision_latency: 0,
            release_interval: 8,
            ci_failure_every: 0,
            ..Default::default()
        }
    }

    #[test]
    fn ticket_assignment_uses_the_full_identifier() {
        assert_eq!(assignee_for_ticket(255), 2);
        assert_eq!(assignee_for_ticket(256), 0);
        assert_eq!(assignee_for_ticket(257), 1);
        assert_eq!(assignee_for_ticket(258), 2);
        assert_eq!(assignee_for_ticket(520), 0);
    }

    #[test]
    fn default_simulation_runs_for_a_work_year() {
        let params = SoftwareTeamParameters::default();
        let report = run_software_team(params);

        assert_eq!(report.elapsed_ticks, params.total_ticks() + 1);
    }

    #[test]
    fn meetings_and_slack_consume_focus_and_trigger_context_switches() {
        let baseline = run_software_team(short_params());
        let loaded = run_software_team(SoftwareTeamParameters {
            communication_load_per_mille: 250,
            meeting_load_per_mille: 250,
            context_switch_cost: 2,
            ..short_params()
        });

        assert_eq!(baseline.meeting_ticks, 0);
        assert_eq!(baseline.slack_ticks, 0);
        assert!(loaded.meeting_ticks > baseline.meeting_ticks);
        assert!(loaded.slack_ticks > baseline.slack_ticks);
        assert!(loaded.context_switch_ticks > baseline.context_switch_ticks);
        assert!(loaded.coding_ticks < baseline.coding_ticks);
    }

    #[test]
    fn decision_latency_blocks_ambiguous_work() {
        let fast_decisions = run_software_team(SoftwareTeamParameters {
            decision_request_every: 1,
            decision_latency: 0,
            ..short_params()
        });
        let slow_decisions = run_software_team(SoftwareTeamParameters {
            decision_request_every: 1,
            decision_latency: 8,
            ..short_params()
        });

        assert!(slow_decisions.decision_requests > 0);
        assert!(slow_decisions.decision_wait_ticks > fast_decisions.decision_wait_ticks);
        assert!(fast_decisions.merged >= slow_decisions.merged);
    }
}
