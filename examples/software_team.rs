//! Simulates software delivery flow under meetings, Slack, CI, and decisions.
#![allow(
    clippy::integer_division,
    clippy::missing_const_for_fn,
    clippy::missing_docs_in_private_items,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "self-contained example with discrete tick arithmetic and local domain types"
)]

use simul::agent::{Agent, AgentContext, AgentInitializer, AgentMode, AgentOptions};
use simul::message::Message;
use simul::{DiscreteTime, Simulation, SimulationParameters};
use std::collections::VecDeque;

const TICKS_PER_DAY: DiscreteTime = 16;
const WORK_DAYS_PER_YEAR: DiscreteTime = 260;
const DEVELOPER_COUNT: u8 = 3;

const PRODUCT: &str = "product";
const MEETINGS: &str = "meetings";
const SLACK: &str = "slack";
const DECISIONS: &str = "decisions";
const CI: &str = "ci";
const MERGE: &str = "merge";
const METRICS: &str = "metrics";
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
enum MetricKind {
    CodingTick,
    ReviewTick,
    MeetingTick,
    SlackTick,
    ContextSwitchTick,
    DecisionWaitTick,
    WorkReleased,
    DecisionRequested,
    ReviewRequested,
    CiRequested,
    CiFailed,
}

impl MetricKind {
    const fn tag(self) -> u8 {
        match self {
            Self::CodingTick => 0,
            Self::ReviewTick => 1,
            Self::MeetingTick => 2,
            Self::SlackTick => 3,
            Self::ContextSwitchTick => 4,
            Self::DecisionWaitTick => 5,
            Self::WorkReleased => 6,
            Self::DecisionRequested => 7,
            Self::ReviewRequested => 8,
            Self::CiRequested => 9,
            Self::CiFailed => 10,
        }
    }

    const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::CodingTick),
            1 => Some(Self::ReviewTick),
            2 => Some(Self::MeetingTick),
            3 => Some(Self::SlackTick),
            4 => Some(Self::ContextSwitchTick),
            5 => Some(Self::DecisionWaitTick),
            6 => Some(Self::WorkReleased),
            7 => Some(Self::DecisionRequested),
            8 => Some(Self::ReviewRequested),
            9 => Some(Self::CiRequested),
            10 => Some(Self::CiFailed),
            _ => None,
        }
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
    Metric {
        kind: MetricKind,
    },
}

impl Payload {
    fn encode(self) -> Vec<u8> {
        match self {
            Self::Tick => vec![0],
            Self::Work {
                ticket_id,
                coding_ticks,
                needs_decision,
            } => {
                let mut bytes = Vec::with_capacity(6);
                bytes.push(1);
                bytes.extend_from_slice(&ticket_id.to_le_bytes());
                bytes.extend_from_slice(&coding_ticks.to_le_bytes());
                bytes.push(u8::from(needs_decision));
                bytes
            }
            Self::Meeting { ticks } => encode_u16(2, ticks),
            Self::Slack { ticks } => encode_u16(3, ticks),
            Self::DecisionRequest {
                ticket_id,
                requester,
            } => encode_u16_u8(4, ticket_id, requester),
            Self::DecisionMade { ticket_id } => encode_u16(5, ticket_id),
            Self::ReviewRequest {
                ticket_id,
                author,
                review_ticks,
            } => {
                let mut bytes = Vec::with_capacity(6);
                bytes.push(6);
                bytes.extend_from_slice(&ticket_id.to_le_bytes());
                bytes.push(author);
                bytes.extend_from_slice(&review_ticks.to_le_bytes());
                bytes
            }
            Self::ReviewApproved { ticket_id } => encode_u16(7, ticket_id),
            Self::CiRequest { ticket_id, author } => encode_u16_u8(8, ticket_id, author),
            Self::CiPassed { ticket_id } => encode_u16(9, ticket_id),
            Self::CiFailed { ticket_id } => encode_u16(10, ticket_id),
            Self::Merge { ticket_id, author } => encode_u16_u8(11, ticket_id, author),
            Self::Metric { kind } => vec![12, kind.tag()],
        }
    }

    fn decode(message: &Message) -> Option<Self> {
        let payload = message.custom_payload.as_deref()?;
        match payload {
            [0] => Some(Self::Tick),
            [1, id_low, id_high, coding_low, coding_high, needs_decision] => Some(Self::Work {
                ticket_id: decode_u16(*id_low, *id_high),
                coding_ticks: decode_u16(*coding_low, *coding_high),
                needs_decision: *needs_decision != 0,
            }),
            [2, low, high] => Some(Self::Meeting {
                ticks: decode_u16(*low, *high),
            }),
            [3, low, high] => Some(Self::Slack {
                ticks: decode_u16(*low, *high),
            }),
            [4, id_low, id_high, requester] => Some(Self::DecisionRequest {
                ticket_id: decode_u16(*id_low, *id_high),
                requester: *requester,
            }),
            [5, low, high] => Some(Self::DecisionMade {
                ticket_id: decode_u16(*low, *high),
            }),
            [6, id_low, id_high, author, review_low, review_high] => Some(Self::ReviewRequest {
                ticket_id: decode_u16(*id_low, *id_high),
                author: *author,
                review_ticks: decode_u16(*review_low, *review_high),
            }),
            [7, low, high] => Some(Self::ReviewApproved {
                ticket_id: decode_u16(*low, *high),
            }),
            [8, id_low, id_high, author] => Some(Self::CiRequest {
                ticket_id: decode_u16(*id_low, *id_high),
                author: *author,
            }),
            [9, low, high] => Some(Self::CiPassed {
                ticket_id: decode_u16(*low, *high),
            }),
            [10, low, high] => Some(Self::CiFailed {
                ticket_id: decode_u16(*low, *high),
            }),
            [11, id_low, id_high, author] => Some(Self::Merge {
                ticket_id: decode_u16(*id_low, *id_high),
                author: *author,
            }),
            [12, kind] => Some(Self::Metric {
                kind: MetricKind::from_tag(*kind)?,
            }),
            _ => None,
        }
    }
}

fn encode_u16(tag: u8, value: u16) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(3);
    bytes.push(tag);
    bytes.extend_from_slice(&value.to_le_bytes());
    bytes
}

fn encode_u16_u8(tag: u8, value: u16, second: u8) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4);
    bytes.push(tag);
    bytes.extend_from_slice(&value.to_le_bytes());
    bytes.push(second);
    bytes
}

const fn decode_u16(low: u8, high: u8) -> u16 {
    u16::from_le_bytes([low, high])
}

fn send_payload(ctx: &mut AgentContext, target: &str, payload: Payload) {
    ctx.send(target, Some(payload.encode()));
}

fn send_metric(ctx: &mut AgentContext, kind: MetricKind) {
    send_payload(ctx, METRICS, Payload::Metric { kind });
}

fn has_tick_queued(ctx: &AgentContext) -> bool {
    ctx.state
        .queue
        .iter()
        .any(|message| matches!(Payload::decode(message), Some(Payload::Tick)))
}

fn ensure_self_tick(ctx: &mut AgentContext) {
    if !has_tick_queued(ctx) {
        let name = ctx.name.to_string();
        send_payload(ctx, &name, Payload::Tick);
    }
}

fn initial_tick(destination: &str) -> Message {
    Message {
        queued_time: 0,
        source: "clock".to_string(),
        destination: destination.to_string(),
        custom_payload: Some(Payload::Tick.encode()),
        ..Default::default()
    }
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

fn reviewer_for(author: u8) -> &'static str {
    developer_name((author + 1) % DEVELOPER_COUNT)
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

#[derive(Clone, Debug)]
struct Developer {
    id: u8,
    params: SoftwareTeamParameters,
    backlog: VecDeque<Ticket>,
    review_queue: VecDeque<ReviewJob>,
    active_coding: Option<Ticket>,
    active_review: Option<ReviewJob>,
    blocked_until: DiscreteTime,
    blocked_kind: Option<BlockKind>,
    context_switch_remaining: DiscreteTime,
    last_work_kind: Option<WorkKind>,
}

impl Developer {
    fn new(id: u8, params: SoftwareTeamParameters) -> Self {
        Self {
            id,
            params,
            backlog: VecDeque::new(),
            review_queue: VecDeque::new(),
            active_coding: None,
            active_review: None,
            blocked_until: 0,
            blocked_kind: None,
            context_switch_remaining: 0,
            last_work_kind: None,
        }
    }

    fn has_pending_work(&self) -> bool {
        self.active_coding.is_some()
            || self.active_review.is_some()
            || !self.backlog.is_empty()
            || !self.review_queue.is_empty()
    }

    fn block_for(&mut self, ctx: &mut AgentContext, ticks: u16, block_kind: BlockKind) {
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
            .max(ctx.time.saturating_add(DiscreteTime::from(ticks)));
        self.blocked_kind = Some(block_kind);
        self.last_work_kind = None;
    }

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        if ctx.time < self.blocked_until {
            match self.blocked_kind {
                Some(BlockKind::Meeting) => send_metric(ctx, MetricKind::MeetingTick),
                Some(BlockKind::Slack) => send_metric(ctx, MetricKind::SlackTick),
                None => {}
            }
            ensure_self_tick(ctx);
            return;
        }

        if self.context_switch_remaining > 0 {
            self.context_switch_remaining -= 1;
            send_metric(ctx, MetricKind::ContextSwitchTick);
            ensure_self_tick(ctx);
            return;
        }

        if self.process_review(ctx) || self.process_coding(ctx) {
            ensure_self_tick(ctx);
            return;
        }

        if !self.backlog.is_empty() {
            send_metric(ctx, MetricKind::DecisionWaitTick);
        }

        ensure_self_tick(ctx);
    }

    fn process_review(&mut self, ctx: &mut AgentContext) -> bool {
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
        send_metric(ctx, MetricKind::ReviewTick);

        if review.remaining_review == 0 {
            send_payload(
                ctx,
                developer_name(review.author),
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

    fn process_coding(&mut self, ctx: &mut AgentContext) -> bool {
        if self.active_coding.is_none() {
            self.active_coding = self.next_ready_ticket(ctx);
        }

        let Some(mut ticket) = self.active_coding else {
            return false;
        };

        if self.switch_to(WorkKind::Coding) {
            self.active_coding = Some(ticket);
            return true;
        }

        ticket.remaining_coding = ticket.remaining_coding.saturating_sub(1);
        send_metric(ctx, MetricKind::CodingTick);

        if ticket.remaining_coding == 0 {
            send_payload(
                ctx,
                reviewer_for(self.id),
                Payload::ReviewRequest {
                    ticket_id: ticket.id,
                    author: self.id,
                    review_ticks: self.params.review_ticks,
                },
            );
            send_metric(ctx, MetricKind::ReviewRequested);
            self.active_coding = None;
        } else {
            self.active_coding = Some(ticket);
        }

        true
    }

    fn next_ready_ticket(&mut self, ctx: &mut AgentContext) -> Option<Ticket> {
        let backlog_len = self.backlog.len();
        for _ in 0..backlog_len {
            let mut ticket = self.backlog.pop_front()?;

            if ticket.needs_decision && !ticket.decision_received {
                if !ticket.decision_requested {
                    ticket.decision_requested = true;
                    send_payload(
                        ctx,
                        DECISIONS,
                        Payload::DecisionRequest {
                            ticket_id: ticket.id,
                            requester: self.id,
                        },
                    );
                    send_metric(ctx, MetricKind::DecisionRequested);
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

impl Agent for Developer {
    fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
        match Payload::decode(msg) {
            Some(Payload::Tick) => self.on_tick(ctx),
            Some(Payload::Work {
                ticket_id,
                coding_ticks,
                needs_decision,
            }) => self.backlog.push_back(Ticket {
                id: ticket_id,
                remaining_coding: DiscreteTime::from(coding_ticks),
                needs_decision,
                decision_requested: false,
                decision_received: !needs_decision,
            }),
            Some(Payload::Meeting { ticks }) => self.block_for(ctx, ticks, BlockKind::Meeting),
            Some(Payload::Slack { ticks }) => self.block_for(ctx, ticks, BlockKind::Slack),
            Some(Payload::DecisionMade { ticket_id }) => self.receive_decision(ticket_id),
            Some(Payload::ReviewRequest {
                ticket_id,
                author,
                review_ticks,
            }) => self.review_queue.push_back(ReviewJob {
                ticket_id,
                author,
                remaining_review: DiscreteTime::from(review_ticks),
            }),
            Some(Payload::ReviewApproved { ticket_id }) => {
                send_payload(
                    ctx,
                    CI,
                    Payload::CiRequest {
                        ticket_id,
                        author: self.id,
                    },
                );
                send_metric(ctx, MetricKind::CiRequested);
            }
            Some(Payload::CiPassed { ticket_id }) => send_payload(
                ctx,
                MERGE,
                Payload::Merge {
                    ticket_id,
                    author: self.id,
                },
            ),
            Some(Payload::CiFailed { ticket_id }) => {
                self.backlog.push_back(Ticket {
                    id: ticket_id,
                    remaining_coding: DiscreteTime::from(self.params.rework_ticks),
                    needs_decision: false,
                    decision_requested: false,
                    decision_received: true,
                });
            }
            Some(
                Payload::DecisionRequest { .. }
                | Payload::CiRequest { .. }
                | Payload::Merge { .. }
                | Payload::Metric { .. },
            )
            | None => {}
        }

        ensure_self_tick(ctx);
    }
}

#[derive(Clone, Debug)]
struct ProductManager {
    params: SoftwareTeamParameters,
    next_ticket_id: u16,
}

impl ProductManager {
    const fn new(params: SoftwareTeamParameters) -> Self {
        Self {
            params,
            next_ticket_id: 1,
        }
    }
}

impl Agent for ProductManager {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        if ctx.time >= self.params.total_ticks() || self.params.release_interval == 0 {
            return;
        }

        if !discrete_multiple(ctx.time, self.params.release_interval) {
            return;
        }

        let ticket_id = self.next_ticket_id;
        self.next_ticket_id = self.next_ticket_id.saturating_add(1);
        let assignee = assignee_for_ticket(ticket_id);
        let coding_ticks = self.params.base_coding_ticks.saturating_add(ticket_id % 5);
        let needs_decision = short_multiple(ticket_id, self.params.decision_request_every);

        send_payload(
            ctx,
            developer_name(assignee),
            Payload::Work {
                ticket_id,
                coding_ticks,
                needs_decision,
            },
        );
        send_metric(ctx, MetricKind::WorkReleased);
    }
}

#[derive(Clone, Debug)]
struct MeetingScheduler {
    params: SoftwareTeamParameters,
}

impl Agent for MeetingScheduler {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        let meeting_ticks = self.params.meeting_ticks_per_day();
        if meeting_ticks == 0
            || ctx.time >= self.params.total_ticks()
            || !discrete_multiple(ctx.time, self.params.ticks_per_day)
        {
            return;
        }

        let ticks = u16::try_from(meeting_ticks).unwrap_or(u16::MAX);
        for id in 0..DEVELOPER_COUNT {
            send_payload(ctx, developer_name(id), Payload::Meeting { ticks });
        }
    }
}

#[derive(Clone, Debug)]
struct CommunicationScheduler {
    params: SoftwareTeamParameters,
}

impl Agent for CommunicationScheduler {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        let communication_ticks = self.params.communication_ticks_per_day();
        if communication_ticks == 0 || ctx.time >= self.params.total_ticks() {
            return;
        }

        let tick_in_day = ctx.time % self.params.ticks_per_day;
        if tick_in_day
            != self
                .params
                .meeting_ticks_per_day()
                .min(self.params.ticks_per_day - 1)
        {
            return;
        }

        let ticks = u16::try_from(communication_ticks).unwrap_or(u16::MAX);
        for id in 0..DEVELOPER_COUNT {
            send_payload(ctx, developer_name(id), Payload::Slack { ticks });
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
    pending: Vec<PendingDecision>,
}

impl DecisionMaker {
    const fn new(params: SoftwareTeamParameters) -> Self {
        Self {
            params,
            pending: Vec::new(),
        }
    }

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        let mut remaining = Vec::with_capacity(self.pending.len());
        for decision in self.pending.drain(..) {
            if ctx.time >= decision.due_at {
                send_payload(
                    ctx,
                    developer_name(decision.requester),
                    Payload::DecisionMade {
                        ticket_id: decision.ticket_id,
                    },
                );
            } else {
                remaining.push(decision);
            }
        }
        self.pending = remaining;
        ensure_self_tick(ctx);
    }
}

impl Agent for DecisionMaker {
    fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
        match Payload::decode(msg) {
            Some(Payload::Tick) => self.on_tick(ctx),
            Some(Payload::DecisionRequest {
                ticket_id,
                requester,
            }) => self.pending.push(PendingDecision {
                ticket_id,
                requester,
                due_at: ctx.time.saturating_add(self.params.decision_latency),
            }),
            _ => {}
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
    queue: VecDeque<(u16, u8)>,
    active: Option<CiJob>,
    failed_once: Vec<u16>,
}

impl ContinuousIntegration {
    fn new(params: SoftwareTeamParameters) -> Self {
        Self {
            params,
            queue: VecDeque::new(),
            active: None,
            failed_once: Vec::new(),
        }
    }

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        if self.active.is_none() {
            self.active = self.queue.pop_front().map(|(ticket_id, author)| CiJob {
                ticket_id,
                author,
                started_at: ctx.time,
            });
        }

        let Some(job) = self.active else {
            ensure_self_tick(ctx);
            return;
        };

        if ctx.time < job.started_at.saturating_add(self.params.ci_wait_time) {
            ensure_self_tick(ctx);
            return;
        }

        if self.should_fail(job.ticket_id) {
            self.failed_once.push(job.ticket_id);
            send_payload(
                ctx,
                developer_name(job.author),
                Payload::CiFailed {
                    ticket_id: job.ticket_id,
                },
            );
            send_metric(ctx, MetricKind::CiFailed);
        } else {
            send_payload(
                ctx,
                developer_name(job.author),
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

impl Agent for ContinuousIntegration {
    fn on_message(&mut self, ctx: &mut AgentContext, msg: &Message) {
        match Payload::decode(msg) {
            Some(Payload::Tick) => self.on_tick(ctx),
            Some(Payload::CiRequest { ticket_id, author }) => {
                self.queue.push_back((ticket_id, author));
            }
            _ => {}
        }
        ensure_self_tick(ctx);
    }
}

#[derive(Clone, Debug)]
struct MergeBot;

impl Agent for MergeBot {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}
}

#[derive(Clone, Debug)]
struct MetricsSink;

impl Agent for MetricsSink {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Stopper {
    stop_at: DiscreteTime,
}

impl Agent for Stopper {
    fn on_message(&mut self, _ctx: &mut AgentContext, _msg: &Message) {}

    fn on_tick(&mut self, ctx: &mut AgentContext) {
        if ctx.time >= self.stop_at {
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
    fn from_simulation(simulation: &Simulation) -> Self {
        let mut report = Self {
            elapsed_ticks: simulation.time(),
            ..Default::default()
        };

        for agent in simulation.agents() {
            for message in &agent.state.produced {
                match Payload::decode(message) {
                    Some(Payload::Work { .. }) => report.released += 1,
                    Some(Payload::ReviewRequest { .. }) => report.review_requests += 1,
                    Some(Payload::CiRequest { .. }) => report.ci_requests += 1,
                    Some(Payload::DecisionRequest { .. }) => report.decision_requests += 1,
                    Some(Payload::CiFailed { .. }) => report.ci_failures += 1,
                    Some(Payload::Metric { kind }) => report.add_metric(kind),
                    _ => {}
                }
            }
        }

        report.merged = simulation
            .consumed_for_agent(MERGE)
            .map(|messages| {
                messages
                    .iter()
                    .filter(|message| {
                        matches!(Payload::decode(message), Some(Payload::Merge { .. }))
                    })
                    .count()
            })
            .unwrap_or_default();

        report
    }

    fn add_metric(&mut self, kind: MetricKind) {
        match kind {
            MetricKind::CodingTick => self.coding_ticks += 1,
            MetricKind::ReviewTick => self.review_ticks += 1,
            MetricKind::MeetingTick => self.meeting_ticks += 1,
            MetricKind::SlackTick => self.slack_ticks += 1,
            MetricKind::ContextSwitchTick => self.context_switch_ticks += 1,
            MetricKind::DecisionWaitTick => self.decision_wait_ticks += 1,
            MetricKind::WorkReleased
            | MetricKind::DecisionRequested
            | MetricKind::ReviewRequested
            | MetricKind::CiRequested
            | MetricKind::CiFailed => {}
        }
    }
}

fn proactive_agent(name: &str, agent: impl Agent + 'static) -> AgentInitializer {
    AgentInitializer {
        agent: Box::new(agent),
        options: AgentOptions {
            initial_mode: AgentMode::Proactive,
            wake_mode: AgentMode::Proactive,
            name: name.to_string(),
            ..Default::default()
        },
    }
}

fn reactive_agent(
    name: &str,
    agent: impl Agent + 'static,
    initial_queue: VecDeque<Message>,
) -> AgentInitializer {
    AgentInitializer {
        agent: Box::new(agent),
        options: AgentOptions {
            initial_mode: AgentMode::Reactive,
            wake_mode: AgentMode::Reactive,
            name: name.to_string(),
            initial_queue,
        },
    }
}

fn build_simulation(params: SoftwareTeamParameters) -> Simulation {
    Simulation::new(SimulationParameters {
        agent_initializers: vec![
            proactive_agent(PRODUCT, ProductManager::new(params)),
            proactive_agent(MEETINGS, MeetingScheduler { params }),
            proactive_agent(SLACK, CommunicationScheduler { params }),
            reactive_agent(
                developer_name(0),
                Developer::new(0, params),
                vec![initial_tick(developer_name(0))].into(),
            ),
            reactive_agent(
                developer_name(1),
                Developer::new(1, params),
                vec![initial_tick(developer_name(1))].into(),
            ),
            reactive_agent(
                developer_name(2),
                Developer::new(2, params),
                vec![initial_tick(developer_name(2))].into(),
            ),
            reactive_agent(
                DECISIONS,
                DecisionMaker::new(params),
                vec![initial_tick(DECISIONS)].into(),
            ),
            reactive_agent(
                CI,
                ContinuousIntegration::new(params),
                vec![initial_tick(CI)].into(),
            ),
            reactive_agent(MERGE, MergeBot, VecDeque::new()),
            reactive_agent(METRICS, MetricsSink, VecDeque::new()),
            proactive_agent(
                STOPPER,
                Stopper {
                    stop_at: params.total_ticks(),
                },
            ),
        ],
        halt_check: |_| false,
        enable_queue_depth_metrics: true,
        enable_agent_asleep_cycles_metric: false,
        ..Default::default()
    })
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
        assert!(report.released > 0);
        assert!(report.merged > 0);
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
