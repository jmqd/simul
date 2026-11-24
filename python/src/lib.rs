//! Python (`PyO3`) bindings for `simul`.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAny, PyDict, PyList, PyString, PyTuple};
use rustc_hash::FxHashMap;

use simul_core::agent::{
    Agent, AgentContext, AgentId, AgentInitializer, MessageDrain, MessageProcessingStatus,
    SimulationAgent, TimerId, geometric_consumer, geometric_producer, periodic_consumer,
    periodic_producer,
};
use simul_core::experiment::{
    AgentStreams, BernoulliReducer, BernoulliSummary, CandidateId, FailurePolicy, MasterSeed,
    MixedCrossEntropyConfig, MixedCrossEntropyOptimizer, MixedCrossEntropySample, MixedDimension,
    MixedValue, NonFinitePolicy, ObjectiveDirection, PairedScalarSummary, ParetoArchive,
    Probability, QuantileReducer, QuantileSummary, ReplayKey, ReplicationId, ScalarReducer,
    ScalarSummary, SeedDeriver, TrialContext, TrialId, TrialRecord, TrialRng, TrialSpec,
    compare_paired_scalar, run_serial,
    search::{sample_bounded, sample_latin_hypercube, sample_sobol, sample_sobol_scrambled},
};
use simul_core::message::Message;
use simul_core::trace::{ReplayWindow, SimulationTrace, forensic_replay_with_payloads};
use simul_core::{
    DiscreteTime, Halt, RunOutcome, Simulation, SimulationMode, UnknownDestinationPolicy,
};

/// Python object payload carried by simulation messages.
#[derive(Default)]
enum PyPayload {
    /// Empty payload (used by built-in periodic/geometric producers).
    #[default]
    None,
    /// Arbitrary Python object payload.
    Obj(Py<PyAny>),
}

impl Clone for PyPayload {
    fn clone(&self) -> Self {
        match self {
            Self::None => Self::None,
            Self::Obj(obj) => Python::attach(|py| Self::Obj(obj.clone_ref(py))),
        }
    }
}

impl fmt::Debug for PyPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("None"),
            Self::Obj(obj) => Python::attach(|py| match obj.bind(py).repr() {
                Ok(r) => match r.to_cow() {
                    Ok(text) => f.write_str(&text),
                    Err(_) => f.write_str("<pyobject>"),
                },
                Err(_) => f.write_str("<pyobject>"),
            }),
        }
    }
}

impl PyPayload {
    /// Clones using an already-attached Python token without re-entering `Python::attach`.
    fn clone_with_py(&self, py: Python<'_>) -> Self {
        match self {
            Self::None => Self::None,
            Self::Obj(obj) => Self::Obj(obj.clone_ref(py)),
        }
    }

    /// Converts into a Python object (`None` for [`Self::None`]).
    fn to_py(&self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::None => py.None(),
            Self::Obj(obj) => obj.clone_ref(py),
        }
    }

    /// Wraps a Python object, keeping Python `None` as [`Self::None`].
    fn from_py(obj: &Bound<'_, PyAny>) -> Self {
        if obj.is_none() {
            Self::None
        } else {
            Self::Obj(obj.clone().unbind())
        }
    }
}

/// Pinned deterministic V1 `ChaCha12` RNG stream exposed to Python.
#[pyclass(name = "TrialRng")]
#[derive(Clone)]
struct PyTrialRng {
    inner: TrialRng,
}

#[pymethods]
impl PyTrialRng {
    /// Creates a `TrialRng` from a 64-bit seed via V1 `SeedDeriver::proposal_seed(0)`.
    #[new]
    fn new(seed: u64) -> Self {
        let rng = SeedDeriver::new(MasterSeed::from_u64(seed))
            .proposal_seed(0)
            .rng();
        Self { inner: rng }
    }

    /// Draws the next uniform `u64`.
    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Draws a uniform float in `[0.0, 1.0)`.
    fn uniform_f64(&mut self) -> f64 {
        self.inner.uniform_f64()
    }

    /// Returns an independent copy positioned at the exact same stream offset.
    fn clone_rng(&self) -> Self {
        self.clone()
    }
}

/// Action buffered by a Python agent callback and replayed onto `AgentContext`.
enum PendingAction {
    SendByName(String, PyPayload),
    SendTo(u32, PyPayload),
    SendAfter {
        local_id: u64,
        delay: u64,
        target: u32,
        payload: PyPayload,
    },
    CancelTimer(u64),
    SleepFor(u64),
    SetInProgress,
    YieldControl(String),
    HaltInterrupt(String),
}

/// Read-only view of an incoming message passed to Python `on_message`.
#[pyclass(name = "Message")]
struct PyMessage {
    #[pyo3(get)]
    queued_time: u64,
    #[pyo3(get)]
    source: Option<u32>,
    #[pyo3(get)]
    destination: Option<u32>,
    payload: PyPayload,
}

#[pymethods]
impl PyMessage {
    /// Returns the message payload.
    #[getter]
    fn payload(&self, py: Python<'_>) -> Py<PyAny> {
        self.payload.to_py(py)
    }
}

/// Capability object passed into Python `on_message(ctx, msg)` and `on_tick(ctx)`.
#[pyclass(name = "AgentContext")]
struct PyAgentContext {
    #[pyo3(get)]
    id: u32,
    name: Arc<str>,
    py_name: Py<PyString>,
    #[pyo3(get)]
    time: u64,
    #[pyo3(get)]
    queue_len: usize,
    peer_ids: Arc<FxHashMap<String, u32>>,
    seed: u64,
    stream_root: Option<AgentStreams>,
    call_index: u64,
    rng: Option<TrialRng>,
    next_local_timer: u64,
    actions: Vec<PendingAction>,
}

impl PyAgentContext {
    /// Lazily initializes this callback's V1 agent stream on the first draw,
    /// matching Rust's `AgentContext::rng` without consuming a call window when
    /// a callback draws no randomness.
    fn ensure_rng(&mut self) -> &mut TrialRng {
        let seed = self.seed;
        let name = &self.name;
        let time = self.time;
        let call_index = &mut self.call_index;
        let stream_root = &mut self.stream_root;
        self.rng.get_or_insert_with(|| {
            let root = *stream_root.get_or_insert_with(|| {
                SeedDeriver::new(MasterSeed::from_u64(seed)).agent_streams(name)
            });
            let stream = root.rng(time, *call_index);
            *call_index += 1;
            stream
        })
    }
}

#[pymethods]
impl PyAgentContext {
    /// Returns the agent's name without allocating a new Python string.
    #[getter]
    fn name(&self, py: Python<'_>) -> Py<PyString> {
        self.py_name.clone_ref(py)
    }

    /// Resolves another agent's numeric ID by name, or `None` if unknown.
    fn agent_id(&self, name: &str) -> Option<u32> {
        self.peer_ids.get(name).copied()
    }

    /// Draws a deterministic uniform float in `[0.0, 1.0)` from this tick's stream.
    fn uniform_f64(&mut self) -> f64 {
        self.ensure_rng().uniform_f64()
    }

    /// Draws a deterministic `u64` from this tick's stream.
    fn next_u64(&mut self) -> u64 {
        self.ensure_rng().next_u64()
    }

    /// Queues a message to `target` (by agent name) for end-of-tick delivery.
    fn send(&mut self, target: &str, payload: &Bound<'_, PyAny>) {
        let py_payload = PyPayload::from_py(payload);
        if let Some(&target_id) = self.peer_ids.get(target) {
            self.actions
                .push(PendingAction::SendTo(target_id, py_payload));
        } else {
            self.actions
                .push(PendingAction::SendByName(target.to_owned(), py_payload));
        }
    }

    /// Queues a message to `target_id` for end-of-tick delivery.
    fn send_to(&mut self, target_id: u32, payload: &Bound<'_, PyAny>) {
        self.actions.push(PendingAction::SendTo(
            target_id,
            PyPayload::from_py(payload),
        ));
    }

    /// Schedules a delayed message arriving at `time + delay`, returning a
    /// timer handle that can be passed to `cancel_timer`.
    fn send_after(
        &mut self,
        delay: u64,
        target_id: u32,
        payload: &Bound<'_, PyAny>,
    ) -> PyResult<u64> {
        if delay == 0 {
            return Err(PyValueError::new_err("delay must be positive"));
        }
        let local_id = self.next_local_timer;
        self.next_local_timer += 1;
        self.actions.push(PendingAction::SendAfter {
            local_id,
            delay,
            target: target_id,
            payload: PyPayload::from_py(payload),
        });
        Ok(local_id)
    }

    /// Cancels a pending delayed message previously scheduled by this agent.
    fn cancel_timer(&mut self, timer_handle: u64) {
        self.actions.push(PendingAction::CancelTimer(timer_handle));
    }

    /// Puts the agent to sleep for `ticks` ticks.
    fn sleep_for(&mut self, ticks: u64) {
        self.actions.push(PendingAction::SleepFor(ticks));
    }

    /// Keeps the current message at the front of the inbox for the next visit.
    fn mark_in_progress(&mut self) {
        self.actions.push(PendingAction::SetInProgress);
    }

    /// Pauses the simulation after this tick and yields control to Python with `tag`.
    fn yield_control(&mut self, tag: String) {
        self.actions.push(PendingAction::YieldControl(tag));
    }

    /// Requests that the simulation stop after this tick with `reason`.
    fn send_halt_interrupt(&mut self, reason: String) {
        self.actions.push(PendingAction::HaltInterrupt(reason));
    }
}

/// Sentinel halt reason used when a Python callback raised an exception.
const PY_ERROR_SENTINEL: &str = "__simul_py_exception__";

/// Returns the Python reference count of `obj`.
fn py_refcnt<T>(_py: Python<'_>, obj: &Py<T>) -> isize {
    // SAFETY: Holding `Python<'_>` proves attachment to the Python interpreter,
    // and `obj` owns a valid live Python object pointer.
    unsafe { pyo3::ffi::Py_REFCNT(obj.as_ptr()) }
}

/// Wrapper implementing `simul::agent::Agent<PyPayload>` around any Python
/// object providing `on_message(ctx, msg)` and/or `on_tick(ctx)`.
struct PyCustomAgent {
    obj: Py<PyAny>,
    name: Arc<str>,
    py_name: Option<Py<PyString>>,
    peer_ids: Arc<FxHashMap<String, u32>>,
    seed_slot: Arc<AtomicU64>,
    cached_streams: Option<(u64, AgentStreams)>,
    last_rng_tick: Option<u64>,
    calls_this_tick: u64,
    timer_map: FxHashMap<u64, TimerId>,
    next_local_timer: u64,
    error_slot: Arc<Mutex<Option<PyErr>>>,
    reusable_ctx: Option<Py<PyAgentContext>>,
    reusable_msg: Option<Py<PyMessage>>,
    reusable_tick_args: Option<Py<PyTuple>>,
    reusable_msg_args: Option<Py<PyTuple>>,
    actions_buf: Vec<PendingAction>,
}

impl fmt::Debug for PyCustomAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PyCustomAgent").finish_non_exhaustive()
    }
}

/// Cached handle to Python's `copy.deepcopy` function.
static PY_DEEPCOPY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

impl Clone for PyCustomAgent {
    fn clone(&self) -> Self {
        let (cloned_obj, cloned_py_name) = Python::attach(|py| {
            let deepcopy = PY_DEEPCOPY
                .get_or_try_init(py, || {
                    py.import(pyo3::intern!(py, "copy"))?
                        .getattr(pyo3::intern!(py, "deepcopy"))
                        .map(Bound::unbind)
                })
                .and_then(|f| f.bind(py).call1((self.obj.bind(py),)));
            let obj = match deepcopy {
                Ok(val) => val.unbind(),
                Err(_) => self.obj.clone_ref(py),
            };
            let py_name = self.py_name.as_ref().map(|s| s.clone_ref(py));
            (obj, py_name)
        });
        Self {
            obj: cloned_obj,
            name: Arc::clone(&self.name),
            py_name: cloned_py_name,
            peer_ids: Arc::clone(&self.peer_ids),
            seed_slot: Arc::new(AtomicU64::new(self.seed_slot.load(Ordering::Relaxed))),
            cached_streams: self.cached_streams,
            last_rng_tick: self.last_rng_tick,
            calls_this_tick: self.calls_this_tick,
            timer_map: self.timer_map.clone(),
            next_local_timer: self.next_local_timer,
            error_slot: Arc::new(Mutex::new(None)),
            reusable_ctx: None,
            reusable_msg: None,
            reusable_tick_args: None,
            reusable_msg_args: None,
            actions_buf: Vec::new(),
        }
    }
}

impl PyCustomAgent {
    /// Replays buffered Python actions onto the real `AgentContext`, retaining
    /// the scratch vector's allocation for future callbacks.
    fn apply_actions(
        &mut self,
        ctx: &mut AgentContext<'_, PyPayload>,
        actions: &mut Vec<PendingAction>,
    ) {
        for action in actions.drain(..) {
            match action {
                PendingAction::SendByName(target, payload) => ctx.send(&target, payload),
                PendingAction::SendTo(target, payload) => {
                    ctx.send_to(AgentId::new(target), payload);
                }
                PendingAction::SendAfter {
                    local_id,
                    delay,
                    target,
                    payload,
                } => {
                    let timer = ctx.send_after(delay, AgentId::new(target), payload);
                    self.timer_map.insert(local_id, timer);
                }
                PendingAction::CancelTimer(local_id) => {
                    if let Some(timer) = self.timer_map.remove(&local_id) {
                        ctx.cancel_timer(timer);
                    }
                }
                PendingAction::SleepFor(ticks) => ctx.sleep_for(ticks),
                PendingAction::SetInProgress => {
                    ctx.set_processing_status(MessageProcessingStatus::InProgress);
                }
                PendingAction::YieldControl(tag) => ctx.yield_control(&tag),
                PendingAction::HaltInterrupt(reason) => ctx.send_halt_interrupt(&reason),
            }
        }
    }

    /// Acquires a `Py<PyAgentContext>`, reusing the cached Python shell when no
    /// Python reference escaped the previous callback.
    #[allow(clippy::too_many_arguments)]
    fn acquire_ctx(
        &mut self,
        py: Python<'_>,
        id: u32,
        time: u64,
        queue_len: usize,
        seed: u64,
        stream_root: Option<AgentStreams>,
        actions: Vec<PendingAction>,
    ) -> PyResult<Py<PyAgentContext>> {
        let expected_rc = 1
            + isize::from(self.reusable_tick_args.is_some())
            + isize::from(self.reusable_msg_args.is_some());
        if let Some(existing) = self.reusable_ctx.take() {
            if py_refcnt(py, &existing) == expected_rc {
                {
                    let mut b = existing.borrow_mut(py);
                    b.id = id;
                    b.time = time;
                    b.queue_len = queue_len;
                    b.seed = seed;
                    b.stream_root = stream_root;
                    b.call_index = self.calls_this_tick;
                    b.rng = None;
                    b.next_local_timer = self.next_local_timer;
                    b.actions = actions;
                }
                return Ok(existing);
            }
            // External reference escaped; discard cached arg tuples pointing at old ctx.
            self.reusable_tick_args = None;
            self.reusable_msg_args = None;
        }
        self.reusable_tick_args = None;
        self.reusable_msg_args = None;
        let py_name = self
            .py_name
            .get_or_insert_with(|| PyString::new(py, &self.name).unbind())
            .clone_ref(py);
        Py::new(
            py,
            PyAgentContext {
                id,
                name: Arc::clone(&self.name),
                py_name,
                time,
                queue_len,
                peer_ids: Arc::clone(&self.peer_ids),
                seed,
                stream_root,
                call_index: self.calls_this_tick,
                rng: None,
                next_local_timer: self.next_local_timer,
                actions,
            },
        )
    }

    /// Acquires a `Py<PyMessage>`, reusing the cached Python shell when no
    /// Python reference escaped the previous callback.
    fn acquire_msg(&mut self, py: Python<'_>, m: &Message<PyPayload>) -> PyResult<Py<PyMessage>> {
        let source = m
            .source
            .map(|s| u32::try_from(s.index()).unwrap_or(u32::MAX));
        let destination = m
            .destination
            .map(|d| u32::try_from(d.index()).unwrap_or(u32::MAX));
        let payload = m.payload.clone_with_py(py);
        let expected_rc = 1 + isize::from(self.reusable_msg_args.is_some());
        if let Some(existing) = self.reusable_msg.take() {
            if py_refcnt(py, &existing) == expected_rc {
                {
                    let mut b = existing.borrow_mut(py);
                    b.queued_time = m.queued_time;
                    b.source = source;
                    b.destination = destination;
                    b.payload = payload;
                }
                return Ok(existing);
            }
            self.reusable_msg_args = None;
        }
        self.reusable_msg_args = None;
        Py::new(
            py,
            PyMessage {
                queued_time: m.queued_time,
                source,
                destination,
                payload,
            },
        )
    }

    /// Calls `method` with reusable argument tuples and recycles `PyMessage`
    /// and argument tuples when Python did not retain external references.
    fn call_method_reusing_args(
        &mut self,
        py: Python<'_>,
        method: &Bound<'_, PyAny>,
        py_ctx: &Py<PyAgentContext>,
        msg: Option<&Message<PyPayload>>,
    ) -> PyResult<()> {
        if let Some(m) = msg {
            let py_msg = self.acquire_msg(py, m)?;
            let args = match self.reusable_msg_args.take() {
                Some(tuple) if py_refcnt(py, &tuple) == 1 => tuple,
                _ => {
                    PyTuple::new(py, [py_ctx.bind(py).as_any(), py_msg.bind(py).as_any()])?.unbind()
                }
            };
            let res = method.call1(&args);
            let tuple_sole = res.is_ok() && py_refcnt(py, &args) == 1;
            if tuple_sole && py_refcnt(py, &py_msg) == 2 {
                py_msg.borrow_mut(py).payload = PyPayload::None;
                self.reusable_msg = Some(py_msg);
                self.reusable_msg_args = Some(args);
            }
            res.map(|_| ())
        } else {
            let args = match self.reusable_tick_args.take() {
                Some(tuple) if py_refcnt(py, &tuple) == 1 => tuple,
                _ => PyTuple::new(py, [py_ctx.bind(py).as_any()])?.unbind(),
            };
            let res = method.call1(&args);
            if res.is_ok() && py_refcnt(py, &args) == 1 {
                self.reusable_tick_args = Some(args);
            }
            res.map(|_| ())
        }
    }

    /// Invokes a Python callback method (`on_message` or `on_tick`) if defined.
    fn invoke(
        &mut self,
        ctx: &mut AgentContext<'_, PyPayload>,
        is_message: bool,
        msg: Option<&Message<PyPayload>>,
    ) {
        let tick = ctx.time();
        if self.last_rng_tick != Some(tick) {
            self.last_rng_tick = Some(tick);
            self.calls_this_tick = 0;
        }
        let id = u32::try_from(ctx.id().index()).unwrap_or(u32::MAX);
        let queue_len = ctx.state().queue.len();
        let seed = self.seed_slot.load(Ordering::Relaxed);
        let stream_root = match self.cached_streams {
            Some((cached_seed, root)) if cached_seed == seed => Some(root),
            _ => None,
        };
        let mut actions_buf = std::mem::take(&mut self.actions_buf);
        actions_buf.clear();

        let result: Result<(u64, u64, Option<AgentStreams>, Vec<PendingAction>), PyErr> =
            Python::attach(|py| {
                let bound_obj = self.obj.bind(py);
                let method_name = if is_message {
                    pyo3::intern!(py, "on_message")
                } else {
                    pyo3::intern!(py, "on_tick")
                };
                let Some(method) = bound_obj.getattr_opt(method_name)? else {
                    return Ok((
                        self.calls_this_tick,
                        self.next_local_timer,
                        stream_root,
                        actions_buf,
                    ));
                };
                let py_ctx =
                    self.acquire_ctx(py, id, tick, queue_len, seed, stream_root, actions_buf)?;
                self.call_method_reusing_args(py, &method, &py_ctx, msg)?;
                let (next_calls, next_timer, next_root, actions) = {
                    let mut borrowed = py_ctx.borrow_mut(py);
                    (
                        borrowed.call_index,
                        borrowed.next_local_timer,
                        borrowed.stream_root,
                        std::mem::take(&mut borrowed.actions),
                    )
                };
                let expected_ctx_rc = 1
                    + isize::from(self.reusable_tick_args.is_some())
                    + isize::from(self.reusable_msg_args.is_some());
                if py_refcnt(py, &py_ctx) == expected_ctx_rc {
                    py_ctx.borrow_mut(py).rng = None;
                    self.reusable_ctx = Some(py_ctx);
                } else {
                    self.reusable_tick_args = None;
                    self.reusable_msg_args = None;
                }
                Ok((next_calls, next_timer, next_root, actions))
            });
        match result {
            Ok((next_calls, next_timer, next_root, mut actions)) => {
                self.calls_this_tick = next_calls;
                self.next_local_timer = next_timer;
                if let Some(root) = next_root {
                    self.cached_streams = Some((seed, root));
                }
                self.apply_actions(ctx, &mut actions);
                self.actions_buf = actions;
            }
            Err(err) => {
                if let Ok(mut guard) = self.error_slot.lock() {
                    if guard.is_none() {
                        *guard = Some(err);
                    }
                }
                ctx.send_halt_interrupt(PY_ERROR_SENTINEL);
            }
        }
    }
}

impl Agent<PyPayload> for PyCustomAgent {
    fn on_message(&mut self, ctx: &mut AgentContext<'_, PyPayload>, msg: &Message<PyPayload>) {
        self.invoke(ctx, true, Some(msg));
    }

    fn on_tick(&mut self, ctx: &mut AgentContext<'_, PyPayload>) {
        self.invoke(ctx, false, None);
    }
}

/// Outcome returned by `Simulation.run_until_yield`.
#[pyclass(name = "RunOutcome")]
#[derive(Clone)]
struct PyRunOutcome {
    kind: &'static str,
    #[pyo3(get)]
    agent_id: Option<u32>,
    #[pyo3(get)]
    tag: Option<String>,
    #[pyo3(get)]
    reason: Option<String>,
}

#[pymethods]
impl PyRunOutcome {
    /// Outcome kind (`"halted"`, `"interrupted"`, `"paused_at_time"`, or `"yielded"`).
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind
    }
}

impl From<RunOutcome> for PyRunOutcome {
    fn from(outcome: RunOutcome) -> Self {
        match outcome {
            RunOutcome::Halted => Self {
                kind: "halted",
                agent_id: None,
                tag: None,
                reason: None,
            },
            RunOutcome::Interrupted(reason) => Self {
                kind: "interrupted",
                agent_id: None,
                tag: None,
                reason: Some(reason),
            },
            RunOutcome::PausedAtTime => Self {
                kind: "paused_at_time",
                agent_id: None,
                tag: None,
                reason: None,
            },
            RunOutcome::Yielded { agent, tag } => Self {
                kind: "yielded",
                agent_id: Some(u32::try_from(agent.index()).unwrap_or(u32::MAX)),
                tag: Some(tag),
                reason: None,
            },
        }
    }
}

/// Configuration for a pending agent added before building `PySimulation`.
enum PendingAgentSpec {
    PeriodicProducer {
        name: String,
        period: u64,
        target: String,
        asleep_until: Option<u64>,
    },
    PeriodicConsumer {
        name: String,
        period: u64,
        asleep_until: Option<u64>,
    },
    GeometricProducer {
        name: String,
        mean_gap: f64,
        target: String,
        asleep_until: Option<u64>,
    },
    GeometricConsumer {
        name: String,
        mean_gap: f64,
        asleep_until: Option<u64>,
    },
    Custom {
        name: String,
        obj: Py<PyAny>,
        proactive: bool,
        drain_all: bool,
        asleep_until: Option<u64>,
    },
}

/// Discrete-event simulation exposed to Python, supporting custom Python
/// agents, step/yield control, counterfactual cloning/reseeding, and tracing.
#[pyclass(name = "Simulation", unsendable)]
struct PySimulation {
    halt_time: Option<DiscreteTime>,
    seed: u64,
    seed_slot: Arc<AtomicU64>,
    trace_enabled: bool,
    strict_routing: bool,
    pending: Vec<PendingAgentSpec>,
    built: Option<Simulation<PyPayload>>,
    agent_py_names: Option<Vec<Py<PyString>>>,
    error_slot: Arc<Mutex<Option<PyErr>>>,
}

/// Builds one `AgentInitializer` from a pending Python specification.
fn build_agent_initializer(
    spec: PendingAgentSpec,
    peer_ids: &Arc<FxHashMap<String, u32>>,
    seed_slot: &Arc<AtomicU64>,
    error_slot: &Arc<Mutex<Option<PyErr>>>,
) -> PyResult<AgentInitializer<PyPayload>> {
    let init = match spec {
        PendingAgentSpec::PeriodicProducer {
            name,
            period,
            target,
            asleep_until,
        } => {
            let mut i = periodic_producer(name, period, target);
            if let Some(t) = asleep_until {
                i = i.asleep_until(t);
            }
            i
        }
        PendingAgentSpec::PeriodicConsumer {
            name,
            period,
            asleep_until,
        } => {
            let mut i = periodic_consumer(name, period);
            if let Some(t) = asleep_until {
                i = i.asleep_until(t);
            }
            i
        }
        PendingAgentSpec::GeometricProducer {
            name,
            mean_gap,
            target,
            asleep_until,
        } => {
            if !mean_gap.is_finite() || mean_gap < 1.0 {
                return Err(PyValueError::new_err("mean_gap must be finite and >= 1.0"));
            }
            let mut i = geometric_producer(name, mean_gap, target);
            if let Some(t) = asleep_until {
                i = i.asleep_until(t);
            }
            i
        }
        PendingAgentSpec::GeometricConsumer {
            name,
            mean_gap,
            asleep_until,
        } => {
            if !mean_gap.is_finite() || mean_gap < 1.0 {
                return Err(PyValueError::new_err("mean_gap must be finite and >= 1.0"));
            }
            let mut i = geometric_consumer(name, mean_gap);
            if let Some(t) = asleep_until {
                i = i.asleep_until(t);
            }
            i
        }
        PendingAgentSpec::Custom {
            name,
            obj,
            proactive,
            drain_all,
            asleep_until,
        } => {
            let custom = PyCustomAgent {
                obj,
                name: Arc::from(name.as_str()),
                py_name: None,
                peer_ids: Arc::clone(peer_ids),
                seed_slot: Arc::clone(seed_slot),
                cached_streams: None,
                last_rng_tick: None,
                calls_this_tick: 0,
                timer_map: FxHashMap::default(),
                next_local_timer: 1,
                error_slot: Arc::clone(error_slot),
                reusable_ctx: None,
                reusable_msg: None,
                reusable_tick_args: None,
                reusable_msg_args: None,
                actions_buf: Vec::new(),
            };
            let mut i = if proactive {
                AgentInitializer::proactive(name, custom)
            } else {
                AgentInitializer::reactive(name, custom)
            };
            if drain_all {
                i = i.drain_messages(MessageDrain::All);
            }
            if let Some(t) = asleep_until {
                i = i.asleep_until(t);
            }
            i
        }
    };
    Ok(init)
}

/// Borrowed simulation agents paired with their cached Python `str` names.
type AgentsWithNames<'a> = (&'a [SimulationAgent<PyPayload>], &'a [Py<PyString>]);

impl PySimulation {
    /// Materializes the underlying Rust `Simulation` on first run or inspection.
    fn ensure_built(&mut self) -> PyResult<&mut Simulation<PyPayload>> {
        if self.built.is_none() {
            let halt = self.halt_time.map_or(Halt::Never, Halt::AtTime);
            let mut peer_ids = FxHashMap::default();
            for (index, spec) in self.pending.iter().enumerate() {
                let name = match spec {
                    PendingAgentSpec::PeriodicProducer { name, .. }
                    | PendingAgentSpec::PeriodicConsumer { name, .. }
                    | PendingAgentSpec::GeometricProducer { name, .. }
                    | PendingAgentSpec::GeometricConsumer { name, .. }
                    | PendingAgentSpec::Custom { name, .. } => name.clone(),
                };
                peer_ids.insert(name, u32::try_from(index).unwrap_or(u32::MAX));
            }
            let peer_ids = Arc::new(peer_ids);
            let seed_slot = Arc::new(AtomicU64::new(self.seed));
            self.seed_slot = Arc::clone(&seed_slot);
            let error_slot = Arc::new(Mutex::new(None));
            self.error_slot = Arc::clone(&error_slot);

            let mut builder = Simulation::<PyPayload>::builder(halt)
                .seed(MasterSeed::from_u64(self.seed))
                .unknown_destination_policy(if self.strict_routing {
                    UnknownDestinationPolicy::Halt
                } else {
                    UnknownDestinationPolicy::Drop
                });
            if self.trace_enabled {
                builder = builder.trace_with_payloads(true);
            }

            for spec in self.pending.drain(..) {
                builder = builder.agent(build_agent_initializer(
                    spec,
                    &peer_ids,
                    &seed_slot,
                    &error_slot,
                )?);
            }
            let sim = builder
                .build()
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            self.built = Some(sim);
        }
        self.built
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("simulation failed to initialize"))
    }

    /// Re-raises any Python exception captured during agent callbacks.
    fn take_callback_error(&self) -> PyResult<()> {
        if let Ok(mut guard) = self.error_slot.lock() {
            if let Some(err) = guard.take() {
                return Err(err);
            }
        }
        Ok(())
    }

    /// Ensures the simulation is built and returns the agent slice paired with
    /// cached Python `str` keys for direct statistics dictionary population.
    fn agents_with_py_names(&mut self, py: Python<'_>) -> PyResult<AgentsWithNames<'_>> {
        let _ = self.ensure_built()?;
        let sim = self
            .built
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("simulation failed to initialize"))?;
        let names = self.agent_py_names.get_or_insert_with(|| {
            sim.agents()
                .iter()
                .map(|agent| PyString::new(py, agent.name()).unbind())
                .collect()
        });
        Ok((sim.agents(), names.as_slice()))
    }
}

#[pymethods]
impl PySimulation {
    /// Creates a new simulation with optional `halt_time` (`None` = `Halt::Never`).
    #[new]
    #[pyo3(signature = (halt_time=None, seed=0, trace=false, strict_routing=false))]
    fn new(halt_time: Option<u64>, seed: u64, trace: bool, strict_routing: bool) -> Self {
        Self {
            halt_time,
            seed,
            seed_slot: Arc::new(AtomicU64::new(seed)),
            trace_enabled: trace,
            strict_routing,
            pending: Vec::new(),
            built: None,
            agent_py_names: None,
            error_slot: Arc::new(Mutex::new(None)),
        }
    }

    /// Adds a built-in periodic producer agent.
    #[pyo3(signature = (name, period, target, asleep_until=None))]
    fn add_periodic_producer(
        &mut self,
        name: String,
        period: u64,
        target: String,
        asleep_until: Option<u64>,
    ) -> PyResult<()> {
        if self.built.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot add agents after the simulation has started",
            ));
        }
        self.pending.push(PendingAgentSpec::PeriodicProducer {
            name,
            period,
            target,
            asleep_until,
        });
        Ok(())
    }

    /// Adds a built-in periodic consumer agent.
    #[pyo3(signature = (name, period, asleep_until=None))]
    fn add_periodic_consumer(
        &mut self,
        name: String,
        period: u64,
        asleep_until: Option<u64>,
    ) -> PyResult<()> {
        if self.built.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot add agents after the simulation has started",
            ));
        }
        self.pending.push(PendingAgentSpec::PeriodicConsumer {
            name,
            period,
            asleep_until,
        });
        Ok(())
    }

    /// Adds a built-in geometric producer agent.
    #[pyo3(signature = (name, mean_gap, target, asleep_until=None))]
    fn add_geometric_producer(
        &mut self,
        name: String,
        mean_gap: f64,
        target: String,
        asleep_until: Option<u64>,
    ) -> PyResult<()> {
        if self.built.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot add agents after the simulation has started",
            ));
        }
        self.pending.push(PendingAgentSpec::GeometricProducer {
            name,
            mean_gap,
            target,
            asleep_until,
        });
        Ok(())
    }

    /// Adds a built-in geometric consumer agent.
    #[pyo3(signature = (name, mean_gap, asleep_until=None))]
    fn add_geometric_consumer(
        &mut self,
        name: String,
        mean_gap: f64,
        asleep_until: Option<u64>,
    ) -> PyResult<()> {
        if self.built.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot add agents after the simulation has started",
            ));
        }
        self.pending.push(PendingAgentSpec::GeometricConsumer {
            name,
            mean_gap,
            asleep_until,
        });
        Ok(())
    }

    /// Adds a custom Python agent object implementing `on_message(ctx, msg)`
    /// and/or `on_tick(ctx)`.
    #[pyo3(signature = (name, agent, proactive=false, drain_all=false, asleep_until=None))]
    fn add_agent(
        &mut self,
        name: String,
        agent: Py<PyAny>,
        proactive: bool,
        drain_all: bool,
        asleep_until: Option<u64>,
    ) -> PyResult<()> {
        if self.built.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot add agents after the simulation has started",
            ));
        }
        self.pending.push(PendingAgentSpec::Custom {
            name,
            obj: agent,
            proactive,
            drain_all,
            asleep_until,
        });
        Ok(())
    }

    /// Runs the simulation until halt, interrupt, or yield.
    fn run(&mut self) -> PyResult<PyRunOutcome> {
        let outcome = self.ensure_built()?.run_until_yield(None);
        self.take_callback_error()?;
        Ok(outcome.into())
    }

    /// Runs until `time` (without executing `time`), halt, interrupt, or yield.
    fn run_until(&mut self, time: u64) -> PyResult<PyRunOutcome> {
        let outcome = self.ensure_built()?.run_until_yield(Some(time));
        self.take_callback_error()?;
        Ok(outcome.into())
    }

    /// Runs until an agent yields, `max_time` is reached, or the run halts.
    #[pyo3(signature = (max_time=None))]
    fn run_until_yield(&mut self, max_time: Option<u64>) -> PyResult<PyRunOutcome> {
        let outcome = self.ensure_built()?.run_until_yield(max_time);
        self.take_callback_error()?;
        Ok(outcome.into())
    }

    /// Deep-clones the simulation snapshot for independent counterfactual branching.
    fn clone_snapshot(&mut self) -> PyResult<Self> {
        let mut cloned = self.ensure_built()?.clone();
        // Attach fresh error and seed slots so branches evolve independently.
        let fresh_error_slot = Arc::new(Mutex::new(None));
        let fresh_seed_slot = Arc::new(AtomicU64::new(self.seed));
        for index in 0..cloned.agents().len() {
            if let Ok(idx) = u32::try_from(index) {
                if let Some(custom) = cloned
                    .agent_mut(AgentId::new(idx))
                    .and_then(|a| a.downcast_mut::<PyCustomAgent>())
                {
                    custom.error_slot = Arc::clone(&fresh_error_slot);
                    custom.seed_slot = Arc::clone(&fresh_seed_slot);
                }
            }
        }
        Ok(Self {
            halt_time: self.halt_time,
            seed: self.seed,
            seed_slot: fresh_seed_slot,
            trace_enabled: self.trace_enabled,
            strict_routing: self.strict_routing,
            pending: Vec::new(),
            built: Some(cloned),
            agent_py_names: None,
            error_slot: fresh_error_slot,
        })
    }

    /// Replaces the simulation's RNG root seed for all future ticks.
    fn reseed(&mut self, seed: u64) -> PyResult<()> {
        self.seed = seed;
        self.seed_slot.store(seed, Ordering::Relaxed);
        self.ensure_built()?.reseed(MasterSeed::from_u64(seed));
        Ok(())
    }

    /// Pushes an external message directly into `agent_name`'s inbox.
    fn push_message(&mut self, agent_name: &str, payload: &Bound<'_, PyAny>) -> PyResult<()> {
        let sim = self.ensure_built()?;
        let now = sim.time();
        let target_id = sim
            .agent_id(agent_name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown agent '{agent_name}'")))?;
        let Some(agent) = sim.agent_mut(target_id) else {
            return Err(PyValueError::new_err(format!(
                "unknown agent '{agent_name}'"
            )));
        };
        let mut msg = Message::new(now, PyPayload::from_py(payload));
        msg.destination = Some(target_id);
        agent.state.queue.push_back(msg);
        Ok(())
    }

    /// Returns the live Python object backing custom agent `name`.
    fn agent(&mut self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let sim = self.ensure_built()?;
        let custom = sim
            .find_by_name(name)
            .and_then(|a| a.downcast_ref::<PyCustomAgent>())
            .ok_or_else(|| {
                PyValueError::new_err(format!("no custom Python agent named '{name}'"))
            })?;
        Ok(custom.obj.clone_ref(py))
    }

    /// Current simulation tick.
    #[getter]
    fn time(&mut self) -> PyResult<u64> {
        Ok(self.ensure_built()?.time())
    }

    /// Current simulation mode (`"constructed"`, `"running"`, `"paused"`, `"completed"`, `"failed"`).
    #[getter]
    fn mode(&mut self) -> PyResult<&'static str> {
        Ok(match self.ensure_built()?.mode() {
            SimulationMode::Constructed => "constructed",
            SimulationMode::Running => "running",
            SimulationMode::Paused => "paused",
            SimulationMode::Completed => "completed",
            SimulationMode::Failed => "failed",
        })
    }

    /// Halt interrupt reason, if any.
    #[getter]
    fn halt_reason(&mut self) -> PyResult<Option<String>> {
        Ok(self.ensure_built()?.halt_reason().map(ToOwned::to_owned))
    }

    /// Yield tag from the latest `yield_control`, if paused at a yield.
    #[getter]
    fn yield_tag(&mut self) -> PyResult<Option<String>> {
        Ok(self.ensure_built()?.yield_tag().map(ToOwned::to_owned))
    }

    /// Number of messages addressed to unknown destinations.
    #[getter]
    fn unknown_destination_count(&mut self) -> PyResult<u64> {
        Ok(self.ensure_built()?.unknown_destination_count())
    }

    /// Consumed message counts by agent name.
    fn consumed_counts(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let (agents, py_names) = self.agents_with_py_names(py)?;
        let dict = PyDict::new(py);
        for (name, agent) in py_names.iter().zip(agents.iter()) {
            dict.set_item(name.bind(py), agent.consumed_count())?;
        }
        Ok(dict.unbind())
    }

    /// Produced message counts by agent name.
    fn produced_counts(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let (agents, py_names) = self.agents_with_py_names(py)?;
        let dict = PyDict::new(py);
        for (name, agent) in py_names.iter().zip(agents.iter()) {
            dict.set_item(name.bind(py), agent.produced_count())?;
        }
        Ok(dict.unbind())
    }

    /// Current queue depths by agent name.
    fn queue_lengths(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let (agents, py_names) = self.agents_with_py_names(py)?;
        let dict = PyDict::new(py);
        for (name, agent) in py_names.iter().zip(agents.iter()) {
            dict.set_item(name.bind(py), agent.state.queue.len())?;
        }
        Ok(dict.unbind())
    }

    /// Mean message wait durations by agent name.
    fn avg_wait_times(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let (agents, py_names) = self.agents_with_py_names(py)?;
        let dict = PyDict::new(py);
        for (name, agent) in py_names.iter().zip(agents.iter()) {
            if let Some(wait) = agent.avg_wait() {
                dict.set_item(name.bind(py), wait)?;
            }
        }
        Ok(dict.unbind())
    }

    /// Enables structured lifecycle tracing (with `repr()` payloads) from the current tick onward.
    fn enable_trace(&mut self) -> PyResult<()> {
        self.ensure_built()?.enable_trace_with_payloads();
        Ok(())
    }

    /// Disables further trace recording while retaining recorded events.
    fn disable_trace(&mut self) -> PyResult<()> {
        self.ensure_built()?.disable_trace();
        Ok(())
    }

    /// Returns recorded trace events serialized as JSONL, or `None` if tracing was never enabled.
    fn trace_jsonl(&mut self) -> PyResult<Option<String>> {
        Ok(self.ensure_built()?.trace().map(SimulationTrace::to_jsonl))
    }

    /// Returns recorded trace events serialized as Chrome Trace / Perfetto JSON.
    fn trace_perfetto_json(&mut self) -> PyResult<Option<String>> {
        Ok(self
            .ensure_built()?
            .trace()
            .map(SimulationTrace::to_perfetto_json))
    }

    /// Runs a windowed forensic trace on a clone of this simulation over `[start, end)`.
    #[pyo3(signature = (start, end=None))]
    fn forensic_trace_perfetto(&mut self, start: u64, end: Option<u64>) -> PyResult<String> {
        let cloned = self.clone_snapshot()?;
        let Some(sim) = cloned.built else {
            return Err(PyRuntimeError::new_err("simulation not initialized"));
        };
        let report = forensic_replay_with_payloads(sim, ReplayWindow::new(start, end));
        Ok(report.trace.to_perfetto_json())
    }
}

/// Finite scalar summary with two-sided 95% Student-t confidence intervals.
#[pyclass(name = "ScalarSummary")]
#[derive(Clone)]
struct PyScalarSummary {
    inner: ScalarSummary,
}

#[pymethods]
impl PyScalarSummary {
    /// Reduces an iterable of finite floats.
    #[staticmethod]
    fn from_values(values: Vec<f64>) -> PyResult<Self> {
        let inner = ScalarReducer::from_results(values)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }
    #[getter]
    fn sample_variance(&self) -> Option<f64> {
        self.inner.sample_variance()
    }
    #[getter]
    fn standard_error(&self) -> Option<f64> {
        self.inner.standard_error()
    }
    #[getter]
    fn min(&self) -> f64 {
        self.inner.min()
    }
    #[getter]
    fn max(&self) -> f64 {
        self.inner.max()
    }

    /// Two-sided 95% Student-t confidence interval `(lower, upper)`.
    fn ci_95(&self) -> Option<(f64, f64)> {
        self.inner.ci_95()
    }
}

/// Bernoulli summary with Wilson-score confidence intervals.
#[pyclass(name = "BernoulliSummary")]
#[derive(Clone)]
struct PyBernoulliSummary {
    inner: BernoulliSummary,
}

#[pymethods]
impl PyBernoulliSummary {
    /// Reduces a sequence of booleans with optional custom Wilson z-score.
    #[staticmethod]
    #[pyo3(signature = (results, wilson_z=None))]
    fn from_results(results: Vec<bool>, wilson_z: Option<f64>) -> PyResult<Self> {
        let mut reducer = match wilson_z {
            Some(z) => {
                BernoulliReducer::new(z).map_err(|err| PyValueError::new_err(err.to_string()))?
            }
            None => BernoulliReducer::default(),
        };
        for ok in results {
            reducer.observe(ok);
        }
        let inner = reducer
            .finish()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn attempts(&self) -> u64 {
        self.inner.attempts()
    }
    #[getter]
    fn successes(&self) -> u64 {
        self.inner.successes()
    }
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate()
    }
    #[getter]
    fn wilson_lower(&self) -> f64 {
        self.inner.wilson_lower()
    }
    #[getter]
    fn wilson_upper(&self) -> f64 {
        self.inner.wilson_upper()
    }
}

/// Exact empirical quantile and `CVaR` summary.
#[pyclass(name = "QuantileSummary")]
#[derive(Clone)]
struct PyQuantileSummary {
    inner: QuantileSummary,
}

#[pymethods]
impl PyQuantileSummary {
    /// Reduces a sequence of finite floats into exact order statistics.
    #[staticmethod]
    fn from_values(values: Vec<f64>) -> PyResult<Self> {
        let inner = QuantileReducer::from_results(values)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }
    #[getter]
    fn min(&self) -> f64 {
        self.inner.min()
    }
    #[getter]
    fn max(&self) -> f64 {
        self.inner.max()
    }
    fn median(&self) -> f64 {
        self.inner.median()
    }
    fn p90(&self) -> f64 {
        self.inner.p90()
    }
    fn p95(&self) -> f64 {
        self.inner.p95()
    }
    fn p99(&self) -> f64 {
        self.inner.p99()
    }
    fn p999(&self) -> f64 {
        self.inner.p999()
    }
    fn quantile(&self, p: f64) -> PyResult<f64> {
        let prob = Probability::new(p).map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(self.inner.quantile(prob))
    }
    fn upper_cvar(&self, alpha: f64) -> PyResult<f64> {
        let prob = Probability::new(alpha).map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(self.inner.upper_cvar(prob))
    }
    fn lower_cvar(&self, tail: f64) -> PyResult<f64> {
        let prob = Probability::new(tail).map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(self.inner.lower_cvar(prob))
    }
}

/// Paired-difference CRN comparison (`challenger - baseline`).
#[pyclass(name = "PairedScalarSummary")]
#[derive(Clone)]
struct PyPairedScalarSummary {
    inner: PairedScalarSummary,
}

#[pymethods]
impl PyPairedScalarSummary {
    /// Compares matched sequences `baseline` and `challenger` replication by replication.
    #[staticmethod]
    fn from_pairs(baseline: Vec<f64>, challenger: Vec<f64>) -> PyResult<Self> {
        if baseline.len() != challenger.len() {
            return Err(PyValueError::new_err(
                "baseline and challenger must have equal length",
            ));
        }
        let seed = MasterSeed::from_u64(0);
        let to_records = |cid: u64, values: Vec<f64>| -> Vec<TrialRecord<f64, ()>> {
            values
                .into_iter()
                .enumerate()
                .map(|(rep, val)| {
                    let id = TrialId::new(
                        CandidateId::new(cid),
                        ReplicationId::new(u32::try_from(rep).unwrap_or(u32::MAX)),
                    );
                    let replay =
                        TrialContext::from_replay_key(make_replay_key(seed, id)).replay_key();
                    TrialRecord {
                        key: id,
                        replay,
                        result: Ok(val),
                    }
                })
                .collect()
        };
        let b_recs = to_records(0, baseline);
        let c_recs = to_records(1, challenger);
        let inner = compare_paired_scalar(
            &b_recs,
            &c_recs,
            |&v| v,
            FailurePolicy::RejectCandidate,
            NonFinitePolicy::RejectCandidate,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn usable_pairs(&self) -> u32 {
        self.inner.usable_pairs()
    }
    #[getter]
    fn challenger_wins(&self) -> u32 {
        self.inner.challenger_wins()
    }
    #[getter]
    fn baseline_wins(&self) -> u32 {
        self.inner.baseline_wins()
    }
    #[getter]
    fn ties(&self) -> u32 {
        self.inner.ties()
    }
    #[getter]
    fn mean_diff(&self) -> f64 {
        self.inner.difference().mean()
    }
    #[getter]
    fn standard_error(&self) -> Option<f64> {
        self.inner.difference().standard_error()
    }
    fn ci_95(&self) -> Option<(f64, f64)> {
        self.inner.ci_95()
    }
    fn variance_reduction_ratio(&self) -> Option<f64> {
        self.inner.variance_reduction_ratio()
    }
}

/// Builds a synthetic `ReplayKey` via `run_serial` for Python paired sequences.
fn make_replay_key(seed: MasterSeed, key: TrialId) -> ReplayKey {
    let spec = TrialSpec::new(key.candidate(), key.replication(), 0, ());
    run_serial(seed, [spec], |(), ctx| {
        Ok::<ReplayKey, ()>(ctx.replay_key())
    })
    .ok()
    .and_then(|mut v| v.pop())
    .map_or_else(|| unreachable_fallback_replay_key(seed, key), |r| r.replay)
}

/// Deterministic fallback replay key construction.
fn unreachable_fallback_replay_key(seed: MasterSeed, key: TrialId) -> ReplayKey {
    let spec = TrialSpec::new(key.candidate(), key.replication(), 0, ());
    match run_serial(seed, [spec], |(), _| Ok::<(), ()>(())) {
        Ok(records) if !records.is_empty() => records[0].replay,
        _ => panic!("single-spec run_serial cannot fail"),
    }
}

/// Dimension specification for `MixedCrossEntropyOptimizer`.
#[pyclass(name = "MixedDimension", from_py_object)]
#[derive(Clone, Copy)]
struct PyMixedDimension {
    inner: MixedDimension,
}

#[pymethods]
impl PyMixedDimension {
    #[staticmethod]
    fn continuous(lower: f64, upper: f64) -> Self {
        Self {
            inner: MixedDimension::continuous(lower, upper),
        }
    }
    #[staticmethod]
    fn circular(lower: f64, upper: f64) -> Self {
        Self {
            inner: MixedDimension::circular(lower, upper),
        }
    }
    #[staticmethod]
    fn categorical(categories: usize) -> Self {
        Self {
            inner: MixedDimension::categorical(categories),
        }
    }
    #[staticmethod]
    fn integer(min: i64, max: i64) -> Self {
        Self {
            inner: MixedDimension::integer(min, max),
        }
    }
}

/// Mixed continuous/circular/integer/categorical cross-entropy optimizer.
#[pyclass(name = "MixedCrossEntropyOptimizer")]
struct PyMixedCrossEntropyOptimizer {
    inner: MixedCrossEntropyOptimizer,
    kinds: Vec<MixedDimension>,
}

impl PyMixedCrossEntropyOptimizer {
    /// Converts a Rust `Vec<MixedValue>` into a Python list.
    fn point_to_py(py: Python<'_>, point: &[MixedValue]) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        for val in point {
            match *val {
                MixedValue::Continuous(v) => list.append(v)?,
                MixedValue::Category(c) => list.append(c)?,
                MixedValue::Integer(i) => list.append(i)?,
            }
        }
        Ok(list.unbind())
    }

    /// Converts a Python sequence into a `Vec<MixedValue>` matching `self.kinds`.
    fn py_to_point(&self, seq: &Bound<'_, PyAny>) -> PyResult<Vec<MixedValue>> {
        let items: Vec<Bound<'_, PyAny>> = seq.extract()?;
        if items.len() != self.kinds.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} coordinates, got {}",
                self.kinds.len(),
                items.len()
            )));
        }
        let mut out = Vec::with_capacity(items.len());
        for (item, spec) in items.iter().zip(&self.kinds) {
            let val = match spec {
                MixedDimension::Continuous { .. } | MixedDimension::Circular { .. } => {
                    MixedValue::Continuous(item.extract::<f64>()?)
                }
                MixedDimension::Categorical { .. } => {
                    MixedValue::Category(item.extract::<usize>()?)
                }
                MixedDimension::Integer { .. } => MixedValue::Integer(item.extract::<i64>()?),
            };
            out.push(val);
        }
        Ok(out)
    }
}

#[pymethods]
impl PyMixedCrossEntropyOptimizer {
    #[new]
    #[pyo3(signature = (dimensions, elite_fraction=0.15, learning_rate=0.7))]
    fn new(
        dimensions: Vec<PyMixedDimension>,
        elite_fraction: f64,
        learning_rate: f64,
    ) -> PyResult<Self> {
        let kinds: Vec<MixedDimension> = dimensions.into_iter().map(|d| d.inner).collect();
        let config = MixedCrossEntropyConfig::new(kinds.clone())
            .with_elite_fraction(elite_fraction)
            .with_learning_rate(learning_rate);
        let inner = MixedCrossEntropyOptimizer::new(&config)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner, kinds })
    }

    fn ask(&self, py: Python<'_>, rng: &mut PyTrialRng) -> PyResult<Py<PyList>> {
        let pt = self.inner.ask(&mut rng.inner);
        Self::point_to_py(py, &pt)
    }

    fn ask_batch(
        &self,
        py: Python<'_>,
        rng: &mut PyTrialRng,
        count: usize,
    ) -> PyResult<Vec<Py<PyList>>> {
        self.inner
            .ask_batch(&mut rng.inner, count)
            .into_iter()
            .map(|pt| Self::point_to_py(py, &pt))
            .collect()
    }

    #[allow(clippy::needless_pass_by_value)]
    fn tell(
        &mut self,
        points: Vec<Bound<'_, PyAny>>,
        scores: Vec<f64>,
    ) -> PyResult<Option<(u64, f64)>> {
        if points.len() != scores.len() {
            return Err(PyValueError::new_err(
                "points and scores must have equal length",
            ));
        }
        let mut samples = Vec::with_capacity(points.len());
        for (pt_obj, score) in points.iter().zip(scores) {
            let pt = self.py_to_point(pt_obj)?;
            samples.push(MixedCrossEntropySample::new(pt, score));
        }
        let update = self
            .inner
            .tell(&mut samples)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(update.map(|u| (u.generation, u.best_score)))
    }

    #[getter]
    fn generation(&self) -> u64 {
        self.inner.generation()
    }

    fn best(&self, py: Python<'_>) -> PyResult<Option<(Py<PyList>, f64)>> {
        match self.inner.best() {
            Some(sample) => Ok(Some((Self::point_to_py(py, &sample.point)?, sample.score))),
            None => Ok(None),
        }
    }

    fn categorical_probabilities(&self, dimension: usize) -> Option<Vec<f64>> {
        self.inner
            .categorical_probabilities(dimension)
            .map(<[f64]>::to_vec)
    }

    fn dimension_mean(&self, dimension: usize) -> Option<f64> {
        self.inner.dimension_mean(dimension)
    }
}

/// Multi-objective Pareto frontier archive.
#[pyclass(name = "ParetoArchive")]
struct PyParetoArchive {
    inner: ParetoArchive<Py<PyAny>>,
}

#[pymethods]
impl PyParetoArchive {
    /// Creates an archive from direction strings (`"max"` or `"min"`).
    #[new]
    fn new(directions: Vec<String>) -> PyResult<Self> {
        let mut parsed = Vec::with_capacity(directions.len());
        for dir in directions {
            match dir.as_str() {
                "max" | "maximize" => parsed.push(ObjectiveDirection::Maximize),
                "min" | "minimize" => parsed.push(ObjectiveDirection::Minimize),
                other => {
                    return Err(PyValueError::new_err(format!(
                        "invalid objective direction '{other}'; expected 'max' or 'min'"
                    )));
                }
            }
        }
        let inner =
            ParetoArchive::new(parsed).map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    fn insert(
        &mut self,
        candidate_id: u64,
        payload: Py<PyAny>,
        objectives: Vec<f64>,
    ) -> PyResult<bool> {
        self.inner
            .insert(CandidateId::new(candidate_id), payload, objectives)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Returns frontier entries as list of dicts `{"candidate_id", "payload", "objectives"}`.
    fn frontier(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let mut out = Vec::with_capacity(self.inner.len());
        for entry in self.inner.frontier() {
            let dict = PyDict::new(py);
            dict.set_item("candidate_id", entry.candidate.value())?;
            dict.set_item("payload", entry.payload.clone_ref(py))?;
            dict.set_item("objectives", entry.objectives.clone())?;
            out.push(dict.unbind());
        }
        Ok(out)
    }

    fn crowding_distances(&self) -> Vec<f64> {
        self.inner.crowding_distances()
    }

    fn hypervolume_2d(&self, ref_x: f64, ref_y: f64) -> PyResult<f64> {
        self.inner
            .hypervolume_2d([ref_x, ref_y])
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

/// Uniform bounded random sampling using V1 proposal streams.
#[pyfunction(name = "sample_bounded")]
#[allow(clippy::needless_pass_by_value)]
fn py_sample_bounded(seed: u64, budget: usize, bounds: Vec<(f64, f64)>) -> PyResult<Vec<Vec<f64>>> {
    sample_bounded(MasterSeed::from_u64(seed), budget, &bounds)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Latin Hypercube space-filling sampling.
#[pyfunction(name = "sample_latin_hypercube")]
#[allow(clippy::needless_pass_by_value)]
fn py_sample_latin_hypercube(
    seed: u64,
    samples: usize,
    bounds: Vec<(f64, f64)>,
) -> PyResult<Vec<Vec<f64>>> {
    sample_latin_hypercube(MasterSeed::from_u64(seed), samples, &bounds)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Unscrambled Joe & Kuo Sobol low-discrepancy sequence (up to 32 dimensions).
#[pyfunction(name = "sample_sobol")]
#[allow(clippy::needless_pass_by_value)]
fn py_sample_sobol(samples: usize, bounds: Vec<(f64, f64)>) -> PyResult<Vec<Vec<f64>>> {
    sample_sobol(samples, &bounds).map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Seeded digitally shifted Sobol quasi-random sequence (up to 32 dimensions).
#[pyfunction(name = "sample_sobol_scrambled")]
#[allow(clippy::needless_pass_by_value)]
fn py_sample_sobol_scrambled(
    seed: u64,
    samples: usize,
    bounds: Vec<(f64, f64)>,
) -> PyResult<Vec<Vec<f64>>> {
    sample_sobol_scrambled(MasterSeed::from_u64(seed), samples, &bounds)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Native Python module `simul`.
#[pymodule(name = "simul")]
fn simul_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrialRng>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PyAgentContext>()?;
    m.add_class::<PyRunOutcome>()?;
    m.add_class::<PySimulation>()?;
    m.add_class::<PyScalarSummary>()?;
    m.add_class::<PyBernoulliSummary>()?;
    m.add_class::<PyQuantileSummary>()?;
    m.add_class::<PyPairedScalarSummary>()?;
    m.add_class::<PyMixedDimension>()?;
    m.add_class::<PyMixedCrossEntropyOptimizer>()?;
    m.add_class::<PyParetoArchive>()?;
    m.add_function(wrap_pyfunction!(py_sample_bounded, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_latin_hypercube, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_sobol, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_sobol_scrambled, m)?)?;
    Ok(())
}
