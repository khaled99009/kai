"""The Agent Runtime: orchestrates one vertical slice end to end.

``AgentRuntime`` owns the fixed pipeline of the slice:

    submit -> CREATED
    trigger: CREATED -> UNDERSTANDING  (one PlannerModel call -> scenario)
             -> resolve attendee identities (memory hint + Identity Resolver)
             -> PLANNING   (existing PlannerService.create_plan)
             -> POLICY_CHECK -> EXECUTING
             -> WAITING_APPROVAL  (a mid-flight approval checkpoint)
             -> build the validated KAI-IR program for the bound plan
             -> execute its actions as ToolIntents through the Tool Gateway
             -> VERIFYING -> COMPLETED  (or FAILED / REQUIRES_ATTENTION)

The IR program is the description of *what* is being executed: a closed,
fingerprinted, derived value built from the sealed task contract, the bound plan,
and the saved progress. It grants nothing and decides nothing — every action
still goes to the Tool Gateway, which remains the only authority on policy,
content, approval, account, and verification, and every task state still goes
through the Task Manager machine.

The Task Manager machine is the only writer of ``Task.state``: every step above
goes through :func:`kai.tasks.service.transition_task`, so the task itself
carries the flow (including a real ``WAITING_APPROVAL`` pause) instead of the
runtime reporting a state the task never entered.

Guarantees kept here:

* Every tool execution goes through the Tool Gateway (policy, content,
  approval, verification, account). The runtime never calls the calendar,
  contacts, or any adapter directly. The KAI-IR program describes the actions;
  it does not reach a tool any other way.
* All LLM output is data. Understanding and planning are gated on task state so
  a resume never re-runs the model; persistence (``RuntimeExecution``) carries
  scenario + identities + evidence + cursor for deterministic, LLM-free resume.
* The cursor only advances past a step whose intents all succeeded, and
  completed steps are never re-executed — a stopped approval checkpoint or a
  transient network failure can never cause a duplicate calendar insert.
* The cursor counts the steps of exactly one plan: it is bound to a plan
  version and that version's content hash. A plan re-proposed after execution
  started is refused, never silently resumed against the old cursor, and the
  refusal rewinds nothing and re-executes nothing.
* One trigger owns one execution at a time. The per-task claim
  (``RuntimeExecution.claim_token``) is claimed and released in short committed
  transactions and is *never* held across an external provider call, so a
  concurrent or duplicate resume writes nothing at all and can never overwrite
  the owner's result; a claim abandoned by a crashed worker expires.
* A live owner is never replaced. While it works, a heartbeat renews its lease
  on its own short transaction, so a slow model or provider call cannot let the
  lease lapse and hand the task to a second worker; and every step boundary and
  side effect re-proves ownership, so a worker that *did* lose the claim aborts
  before it acts on the task or records anything about it.
* Failure is deterministic: unknown identity, unsupported capability, no free
  slot, a denied/expired approval, or an unverified execution all move the task
  through the existing Task Manager machine.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final
from uuid import UUID, uuid4

import structlog
from sqlalchemy import or_, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kai.approvals.models import Approval, ApprovalStatus
from kai.db.engine import async_session_factory
from kai.db.models import GoogleAccount
from kai.identity.models import IdentityStatus
from kai.identity.schemas import IdentityQuery
from kai.identity.service import IdentityResolver
from kai.ir.adapter import (
    action_for,
    attach,
    evidence_from_observation,
    intent_from_action,
    observation_from_evidence,
    observation_from_result,
    record_error,
    record_result,
)
from kai.ir.build import allowed_capabilities_from_task, program_from_plan
from kai.ir.control import ControlFacts, next_step
from kai.ir.errors import IrConditionError, IrControlFlowError, IrValidationError
from kai.ir.nodes import ObservationNode
from kai.ir.program import KaiProgram
from kai.logging import get_logger
from kai.memory.models import MemoryType
from kai.memory.service import retrieve_memory
from kai.planner.model import PlannerModel
from kai.planner.models import TaskPlan
from kai.planner.schemas import PlanProposal
from kai.planner.service import (
    ModelUnavailableError,
    PlannerService,
    PlanNotAllowedError,
    PlanNotFoundError,
    PlanRejectedError,
)
from kai.runtime.evidence import qualifying_creation, qualifying_creations
from kai.runtime.fulfillment import (
    CALENDAR_MEETING_EVALUATOR_VERSION,
    GMAIL_SEND_EVALUATOR_VERSION,
    TASK_PREPARATION_EVALUATOR_VERSION,
    FulfillmentInput,
    FulfillmentOutcome,
    FulfillmentVerdict,
    calendar_meeting_criteria,
    evaluate_calendar_meeting,
    evaluate_gmail_send,
    evaluate_task_preparation,
    gmail_send_criteria,
    task_preparation_criteria,
)
from kai.runtime.ledger import append_event
from kai.runtime.models import RuntimeExecution
from kai.runtime.schemas import (
    EXTERNAL_OUTCOME,
    EXTERNAL_OUTCOME_UNKNOWN,
    GoalSubmission,
    UnderstandingOutput,
)
from kai.runtime.translator import (
    CAPABILITY_TO_TOOL,
    PROFILE_ALLOWED_CAPABILITIES,
    build_slice_contract,
    candidate_days,
    capability_to_tool,
    content_category_for,
    create_intent_args,
    memory_email_hint,
    parse_free_intervals,
    pick_slot,
    plan_fingerprint,
    profile_scope_violation,
    read_intent_args,
)
from kai.runtime.understanding import UnderstandingFailed, understand_scenario
from kai.tasks.models import Task
from kai.tasks.service import TaskNotFoundError, create_task, get_task, transition_task
from kai.tasks.state import (
    COMPLETED,
    CREATED,
    EXECUTING,
    FAILED,
    PLANNING,
    POLICY_CHECK,
    REQUIRES_ATTENTION,
    UNDERSTANDING,
    VERIFYING,
    WAITING_APPROVAL,
    WAITING_EXTERNAL,
    is_retryable_failure,
    is_terminal,
)
from kai.tools.context import ToolContext
from kai.tools.definition import VerificationContract
from kai.tools.errors import (
    APPROVAL_STALE,
    is_unresolved_external,
)
from kai.tools.gateway import ToolGateway
from kai.tools.result import ToolOutcome, ToolResult

_RESUMABLE_STATES: frozenset[str] = frozenset(
    {
        CREATED,
        UNDERSTANDING,
        PLANNING,
        POLICY_CHECK,
        EXECUTING,
        WAITING_APPROVAL,
        VERIFYING,
    }
)

#: States in which the durable identity set may legitimately still be empty.
#: The runtime resolves identities exactly once, between UNDERSTANDING and
#: PLANNING (W1/W2); every later state must already carry the set its plan and
#: evidence were built from. ``CREATED`` is pre-identity: stage 1 moves it to
#: ``UNDERSTANDING`` before anything is resolved.
_PRE_IDENTITY_STATES: frozenset[str] = frozenset({CREATED, UNDERSTANDING})

#: States where the runtime is blocked on somebody else, so triggering the task
#: again would achieve nothing. ``WAITING_APPROVAL`` is not one of them: an
#: approval that is already settled leaves the task able to move on.
_WAITING_ON_OTHERS: frozenset[str] = frozenset({WAITING_EXTERNAL})

#: The fulfillment profiles this dispatcher answers to. Named here so the
#: dispatch reads as a closed vocabulary and the value it records carries the
#: same static type the sealed contract does. There is no third name, and adding
#: one is an edit to this function rather than a lookup.
_CALENDAR_MEETING: Final = "calendar_meeting"
_GMAIL_SEND: Final = "gmail_send"
_TASK_PREPARATION: Final = "task_preparation"

#: How long a trigger's execution claim stays valid before another trigger may
#: take it over. A live trigger finishes in seconds; a claim this old belongs
#: to a worker that died mid-flight, and leaving it forever would strand the
#: task. A live worker never reaches this age: :meth:`AgentRuntime._heartbeat`
#: renews the lease while it works, and this is only the crash-detection window.
_CLAIM_LEASE: Final[timedelta] = timedelta(minutes=5)

#: How often the owner renews its claim lease, on its own session. Comfortably
#: inside :data:`_CLAIM_LEASE` so a few slow or failed renewals cannot expire a
#: live worker's claim, and rare enough to be a rounding error on the database.
_CLAIM_HEARTBEAT_SECONDS: Final[float] = 30.0

_POSITIVE_IDENTITY: frozenset[IdentityStatus] = frozenset(
    {
        IdentityStatus.CONTACT_MATCH,
        IdentityStatus.PROVIDER_VERIFIED,
        IdentityStatus.CHANNEL_VERIFIED,
    }
)


class RuntimeNotFoundError(ValueError):
    """No task resolves for the requested ``(tenant, user, id)``."""


class RuntimeResumeError(ValueError):
    """The task state does not permit a runtime trigger."""


class RuntimeFatal(ValueError):
    """A deterministic stop condition the runtime translates into a failure."""


class RuntimeScopeError(RuntimeFatal):
    """A contract's sealed profile does not licence its tool/capability scope.

    Raised at the governed live-sealing boundary (:meth:`AgentRuntime.submit`):
    the runtime refuses to seal an executable task whose profile and
    ``allowed_tools`` disagree, rather than persisting a task the governed path
    could never honour.
    """


class ClaimLostError(Exception):
    """The execution claim moved to another worker while this one was running.

    Raised at a fence: the caller has proved it is no longer the owner, so it
    must abandon the trigger without writing anything. It is deliberately not a
    :class:`RuntimeFatal` — a lost claim is never this task's failure to record.
    """


def next_action_for(state: str, *, approval_ids: Sequence[str] = ()) -> str:
    """What the caller can usefully do about a task sitting in ``state``.

    One mapping, derived from the Task Manager machine instead of a second
    hand-written list of "finished" states that could drift away from it:

    * a pending approval outranks everything - a human is the bottleneck;
    * a terminal or failed task is ``DONE``, because the runtime will not touch
      it again and only the caller (retry, escalate) can;
    * a task waiting on a third party is ``WAIT``, because triggering it again
      would achieve nothing;
    * everything else is ``EXECUTE``, which is a call the runtime can honour.
    """
    if approval_ids:
        return "APPROVE"
    if is_terminal(state) or is_retryable_failure(state):
        return "DONE"
    if state in _WAITING_ON_OTHERS:
        return "WAIT"
    return "EXECUTE"


@dataclass(frozen=True)
class TriggerOutcome:
    """What one trigger produced, for the API surface."""

    task_state: str
    outcome: str
    message: str
    approval_ids: list[str]
    steps_used: int


class AgentRuntime:
    """Coordinates the slice pipeline for one task, per trigger."""

    def __init__(
        self,
        *,
        gateway: ToolGateway,
        model: PlannerModel,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        self.gateway = gateway
        self.model = model
        # The heartbeat needs a session of its own: renewing on the owner's
        # session would queue behind whatever long call the owner is inside.
        self._session_factory = session_factory or async_session_factory
        self._log = logger or get_logger("kai.runtime.service")

    # ------------------------------------------------------------------ #
    # Submit
    # ------------------------------------------------------------------ #

    async def submit(
        self,
        session: AsyncSession,
        *,
        tenant_id: UUID,
        user_id: UUID,
        submission: GoalSubmission,
    ) -> Task:
        """Create a ``CREATED`` task with the sealed slice contract and its
        execution envelope (timezone validated at the schema boundary)."""
        from kai.runtime.translator import build_preparation_contract

        if getattr(submission, "intent", "schedule_meeting") == "prepare_task":
            contract = build_preparation_contract(submission.goal)
        else:
            contract = build_slice_contract(submission.goal)
        # The governed live-sealing gate: a contract that its profile does not
        # licence cannot become an executable task. Checked before the write, so
        # no incompatible task and no execution envelope is ever persisted.
        refused = profile_scope_violation(
            contract.fulfillment_profile, contract.allowed_tools
        )
        if refused:
            raise RuntimeScopeError(
                f"fulfillment profile {contract.fulfillment_profile!r} does not "
                f"permit tools/capabilities: {list(refused)!r}"
            )
        task = await create_task(
            session, tenant_id=tenant_id, user_id=user_id, contract=contract
        )
        execution = RuntimeExecution(
            task_id=task.id,
            tenant_id=task.tenant_id,
            user_id=task.user_id,
            timezone=submission.timezone.strip(),
            scenario=None,
            identities=[],
            evidence=[],
            cursor=0,
            steps_used=0,
        )
        session.add(execution)
        await session.commit()
        await session.refresh(execution)
        return task

    # ------------------------------------------------------------------ #
    # Trigger
    # ------------------------------------------------------------------ #

    async def trigger(
        self,
        session: AsyncSession,
        *,
        tenant_id: UUID,
        user_id: UUID,
        task_id: UUID,
        transport: Any,
    ) -> TriggerOutcome:
        """Drive the task as far as it can go in one call.

        Stages are gated on the task's current state; a terminal task is
        idempotent, a failed task refuses a trigger, and a resumed task never
        re-runs the model or re-executes completed steps.

        Exactly one trigger may own a task's execution at a time. Ownership is
        the ``RuntimeExecution`` claim: it is taken in one short committed
        transaction, it is *not* a held transaction or row lock while the
        provider is called, and it is released in a ``finally``. A caller that
        cannot take it writes nothing and returns ``RESUME_IN_PROGRESS``, which
        is what makes a concurrent/duplicate resume deterministic instead of a
        second writer racing the owner into another final state.

        Ownership is kept alive for that whole run by a background heartbeat on
        its own session, and re-proved at every step boundary and before every
        tool call. So the lease only ever lapses for a worker that really died:
        a live slow owner is never taken over, and an owner that *was* taken
        over returns ``CLAIM_LOST`` having written nothing.
        """
        try:
            task = await get_task(
                session, task_id=task_id, tenant_id=tenant_id, user_id=user_id
            )
        except TaskNotFoundError as exc:
            raise RuntimeNotFoundError(str(exc)) from exc

        if is_terminal(task.state):
            return TriggerOutcome(
                task.state,
                "ALREADY_TERMINAL",
                "task already reached a terminal state",
                [],
                0,
            )
        if task.state not in _RESUMABLE_STATES:
            raise RuntimeResumeError(
                f"task state {task.state} is not resumable by the agent runtime"
            )

        row = await self._load_execution(session, task_id=task.id, tenant_id=tenant_id)
        if row is None:
            raise RuntimeResumeError(
                "task was not submitted through the agent runtime"
            )

        claim = await self._claim_execution(
            session, task_id=task.id, tenant_id=tenant_id
        )
        if claim is None:
            return TriggerOutcome(
                task.state,
                "RESUME_IN_PROGRESS",
                "another trigger already owns this task's execution",
                [],
                row.steps_used,
            )
        # Captured before the ``finally`` rollback expires the ORM instances.
        owned_task_id = task.id
        # Renewed in the background for as long as this trigger runs, so a live
        # but slow worker keeps its lease instead of being replaced mid-flight.
        heartbeat_stop = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat(
                task_id=owned_task_id,
                tenant_id=tenant_id,
                claim=claim,
                stop=heartbeat_stop,
            )
        )
        state_at_start = task.state
        try:
            # The claim is a bulk UPDATE, so the identity-mapped row must be
            # re-read before the pipeline trusts its cursor/evidence. The task
            # is re-read too: a caller that queued behind the owner can only
            # take the claim after the owner finished, and then the persisted
            # state is the only truth.
            await session.refresh(row)
            await session.refresh(task)
            if is_terminal(task.state):
                return TriggerOutcome(
                    task.state,
                    "ALREADY_TERMINAL",
                    "task already reached a terminal state",
                    [],
                    row.steps_used,
                )
            if task.state not in _RESUMABLE_STATES:
                raise RuntimeResumeError(
                    f"task state {task.state} is not resumable by the agent runtime"
                )
            return await self._drive(
                session, task=task, row=row, transport=transport, claim=claim
            )
        except ClaimLostError:
            self._log.warning(
                "execution claim lost to another worker",
                task_id=str(owned_task_id),
                tenant_id=str(tenant_id),
            )
            return await self._lost_claim_outcome(
                task_id=owned_task_id,
                tenant_id=tenant_id,
                fallback_state=state_at_start,
            )
        finally:
            # Discard anything an aborted trigger left uncommitted, stop
            # renewing, then hand ownership back in its own short transaction.
            heartbeat_stop.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            await session.rollback()
            await self._release_execution(
                session, task_id=owned_task_id, tenant_id=tenant_id, claim=claim
            )

    async def _drive(
        self,
        session: AsyncSession,
        *,
        task: Task,
        row: RuntimeExecution,
        transport: Any,
        claim: UUID,
    ) -> TriggerOutcome:
        """Run the owned pipeline for one task (see :meth:`trigger`).

        Every stage below opens with a fence: the first write of the stage, and
        the external call it wraps, only ever happen while this worker is still
        the owner.
        """
        await self._fence_execution(
            session, task_id=task.id, tenant_id=task.tenant_id, claim=claim
        )
        # ``task_preparation`` is a local, provider-free profile: it names no
        # calendar and no guest, so it reaches neither the account resolver nor
        # the model. Everything below that is gated on ``is_prepare`` skips a
        # boundary the profile genuinely does not need; every Google-bound
        # profile still takes the same resolvers, checks, and fail-closed paths
        # it always did.
        is_prepare = task.fulfillment_profile == _TASK_PREPARATION
        if is_prepare:
            account = None
        else:
            account = await self._resolve_account(session, task)
            if account is None:
                return await self._fail(
                    session, task, "no connected google account for this tenant"
                )

        scenario = (
            UnderstandingOutput.model_validate(row.scenario)
            if row.scenario is not None
            else None
        )

        # Lifecycle integrity -- W1/W2 are the only windows where identities may
        # still be empty. An advanced state already has a plan and possibly
        # evidence; re-resolving identity there would let it silently acquire an
        # identity those durable facts were never bound to. The refusal is
        # governed: nothing is re-resolved, rewound, or re-executed.
        if not is_prepare and task.state not in _PRE_IDENTITY_STATES:
            refusal = _advanced_state_prerequisite_error(
                task_state=task.state,
                scenario=scenario,
                execution=row,
            )
            if refusal is not None:
                # Work on record means a human has to decide what the task means
                # now, so the evidence and cursor are kept exactly as they are.
                # A task that never ran is simply failed.
                if row.cursor > 0 or row.evidence or row.steps_used > 0:
                    return await self._halt(
                        session, task, refusal, steps_used=row.steps_used
                    )
                return await self._fail(
                    session, task, refusal, steps_used=row.steps_used
                )

        # A task that died between the VERIFYING transition and its outcome is
        # finished here rather than re-run. The verdict is a pure function of
        # the scenario, the identities, and the evidence, all of which are
        # already durable, so the crash window costs one decision and never a
        # re-execution, a re-plan, or a second model call.
        if task.state == VERIFYING:
            if scenario is None and not is_prepare:
                return await self._fail(
                    session, task, "cannot verify: no understanding on record"
                )
            return await self._settle_verification(
                session, task=task, row=row, scenario=scenario, claim=claim
            )

        # Stage 1a - the local preparation carries no model understanding: its
        # objective is the persisted goal itself, handed to the tool by the
        # runtime. It still walks the machine's own UNDERSTANDING state so the
        # plan below is created through the same gate every profile uses, but no
        # model is consulted and no scenario is written.
        if is_prepare and task.state == CREATED:
            await self._fence_execution(
                session, task_id=task.id, tenant_id=task.tenant_id, claim=claim
            )
            await transition_task(
                session,
                task_id=task.id,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                target=UNDERSTANDING,
            )
            await session.refresh(task)

        # Stage 1 - understanding, exactly once per task. The scenario is
        # committed while the task is still in UNDERSTANDING, so a resume can
        # find one already on record; reading that one back is what keeps a
        # crash in this window from costing a second model call that might not
        # even agree with the first one about which day the meeting is on.
        if (
            not is_prepare
            and task.state in {CREATED, UNDERSTANDING}
            and scenario is None
        ):
            await self._fence_execution(
                session, task_id=task.id, tenant_id=task.tenant_id, claim=claim
            )
            if task.state == CREATED:
                await transition_task(
                    session,
                    task_id=task.id,
                    tenant_id=task.tenant_id,
                    user_id=task.user_id,
                    target=UNDERSTANDING,
                )
                await session.refresh(task)
            notes = await self._memory_notes(session, task)
            try:
                scenario = await understand_scenario(
                    self.model,
                    goal=task.goal,
                    timezone=row.timezone,
                    memory_notes=notes,
                )
            except UnderstandingFailed as exc:
                return await self._fail(session, task, str(exc))
            row.scenario = scenario.model_dump(mode="json")
            await self._fence_execution(
                session, task_id=task.id, tenant_id=task.tenant_id, claim=claim
            )

        # Stage 2 - identity, exactly once per task. A preparation resolves no
        # guests: its sealed contract names no attendee, so there is nothing to
        # resolve and no identity gate to pass. The guard that reads and writes
        # the identity row is unchanged for every profile that does name guests.
        if not is_prepare:
            if scenario is None:
                # Unreachable after the lifecycle guard above: only CREATED and
                # UNDERSTANDING reach this point without a scenario, and stage 1
                # has just resolved it. Kept as a governed refusal rather than
                # an ``assert`` so no lifecycle corruption can surface as an
                # ``AssertionError``.
                return await self._fail(
                    session, task, "cannot plan: no understanding on record"
                )
            assert account is not None
            if not row.identities:
                await self._fence_execution(
                    session, task_id=task.id, tenant_id=task.tenant_id, claim=claim
                )
                try:
                    identities = await self._resolve_guest_identities(
                        session,
                        task,
                        scenario=scenario,
                        account=account,
                        transport=transport,
                    )
                except RuntimeFatal as exc:
                    return await self._fail(session, task, str(exc))
                row.identities = identities
                await self._fence_execution(
                    session, task_id=task.id, tenant_id=task.tenant_id, claim=claim
                )

        # Stage 3 - planning (UNDERSTANDING/PLANNING only; resume reuses the row).
        service = PlannerService(self.model)
        try:
            if task.state == UNDERSTANDING:
                await self._fence_execution(
                    session, task_id=task.id, tenant_id=task.tenant_id, claim=claim
                )
                await service.create_plan(
                    session,
                    task_id=task.id,
                    tenant_id=task.tenant_id,
                    user_id=task.user_id,
                )
                await session.refresh(task)
            plan = await service.get_plan(
                session,
                task_id=task.id,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
            )
        except PlanNotAllowedError as exc:
            return await self._fail(session, task, f"planning refused: {exc}")
        except PlanRejectedError as exc:
            # PlannerService already moved the task to FAILED itself.
            return TriggerOutcome("FAILED", "FAILED", str(exc), [], row.steps_used)
        except ModelUnavailableError as exc:
            return TriggerOutcome("FAILED", "FAILED", str(exc), [], row.steps_used)
        except PlanNotFoundError as exc:
            return await self._fail(session, task, str(exc))

        # Stage 3b - bind the cursor to the plan it is counting on. The cursor
        # and the persisted scenario/evidence describe *these* steps; a plan
        # that changed underneath them can no longer be resumed from it, and
        # guessing would either skip the new plan's steps or re-run finished
        # ones (a duplicate calendar insert).
        bound, plan_version, plan_hash = await self._bind_plan(session, row, plan, claim)
        if not bound:
            return await self._halt(
                session,
                task,
                _plan_diverged(row.plan_version, plan.version),
            )

        # Stage 4 - promote through the machine. A task parked in
        # WAITING_APPROVAL whose approval is still pending is a no-op: it stays
        # in that real state instead of bouncing through EXECUTING.
        if task.state in {PLANNING, WAITING_APPROVAL, POLICY_CHECK}:
            await self._fence_execution(
                session, task_id=task.id, tenant_id=task.tenant_id, claim=claim
            )
        if task.state == PLANNING:
            await transition_task(
                session,
                task_id=task.id,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                target=POLICY_CHECK,
            )
            await session.refresh(task)
        if task.state == WAITING_APPROVAL:
            pending = await self.pending_approval_ids(session, task)
            if pending:
                return TriggerOutcome(
                    WAITING_APPROVAL,
                    WAITING_APPROVAL,
                    "human approval is still pending",
                    pending,
                    row.steps_used,
                )
            await transition_task(
                session,
                task_id=task.id,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                target=EXECUTING,
            )
        if task.state == POLICY_CHECK:
            await transition_task(
                session,
                task_id=task.id,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                target=EXECUTING,
            )
        await session.refresh(task)

        # Stage 5 - execute (resume-safe).
        proposal = PlanProposal.model_validate(plan.plan_json)
        return await self._execute(
            session,
            task=task,
            row=row,
            scenario=scenario,
            proposal=proposal,
            plan_version=plan_version,
            plan_hash=plan_hash,
            plan_status=plan.status,
            transport=transport,
            account=account,
            claim=claim,
        )

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    async def _memory_notes(self, session: AsyncSession, task: Task) -> list[str]:
        """Memory as context for the understanding stage — never instructions."""
        scored = await retrieve_memory(
            session,
            tenant_id=task.tenant_id,
            user_id=task.user_id,
            memory_types={MemoryType.PERSON, MemoryType.PREFERENCE},
            limit=10,
        )
        return [
            f"{item.entry.memory_type}: {item.entry.content}" for item in scored
        ]

    async def _resolve_guest_identities(
        self,
        session: AsyncSession,
        task: Task,
        *,
        scenario: UnderstandingOutput,
        account: GoogleAccount,
        transport: Any,
    ) -> list[dict[str, Any]]:
        """Establish each attendee through exact identifiers plus live evidence.

        A name alone is never enough; the memory email hint is only a hint. A
        guest that does not resolve to a positive identity fails the whole task
        before any tool runs.
        """
        resolver = IdentityResolver()
        results: list[dict[str, Any]] = []
        for guest in scenario.guests:
            email = await memory_email_hint(
                session,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                name=guest,
            )
            query = IdentityQuery(email=email) if email else IdentityQuery(name=guest)
            context = ToolContext(
                session=session,
                task_id=task.id,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                account_id=account.id,
                transport=transport,
                identity=query,
            )
            resolved = await resolver.resolve(query=query, context=context)
            if resolved.status not in _POSITIVE_IDENTITY:
                raise RuntimeFatal(f"attendee identity not established: {guest}")
            # A positive verdict establishes *who*, not *where*: the row may only
            # carry the address the resolver itself returned. Falling back to the
            # memory hint here would persist an address the resolver never named,
            # so a positive result without one is a refusal, not a partial row.
            if resolved.email is None or not resolved.email.strip():
                raise RuntimeFatal(f"attendee address not established: {guest}")
            results.append(
                {
                    "name": guest,
                    "email": resolved.email,
                    "status": resolved.status.value,
                    "confidence": resolved.confidence,
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    async def _execute(
        self,
        session: AsyncSession,
        *,
        task: Task,
        row: RuntimeExecution,
        scenario: UnderstandingOutput | None,
        proposal: PlanProposal,
        plan_version: int,
        plan_hash: str,
        plan_status: str,
        transport: Any,
        account: GoogleAccount | None,
        claim: UUID,
    ) -> TriggerOutcome:
        evidence: list[dict[str, Any]] = list(row.evidence)
        identities: list[dict[str, Any]] = list(row.identities)
        cursor = row.cursor
        steps_used = row.steps_used

        collected = _read_slot_items(evidence)
        created_event_id = _created_event_id(evidence)

        try:
            # Stage 5b - the program this trigger is executing. Derived from the
            # sealed contract, the bound plan, and the saved progress, so it is
            # the same audited program on every resume; nothing about it is
            # persisted as a new authority, because the cursor, the evidence, and
            # the plan binding on ``RuntimeExecution`` remain what actually drive
            # recovery. An unsupported or unbuildable plan is refused here and
            # fails the task through the machine, exactly as an unsupported
            # capability used to — validation runs before any tool is reached.
            program = self._program_for(
                task,
                proposal=proposal,
                plan_status=plan_status,
                plan_version=plan_version,
                plan_hash=plan_hash,
                row=row,
            )

            # Which step runs next is the program's answer, not this loop's. The
            # IR declares the control flow and the runtime asks it, handing over
            # only the facts the execution row already holds. A plan with no
            # branches walks as the ordered chain it is, so the sequence is
            # unchanged — but the sequence now lives in the program, and a step
            # finished on an earlier trigger is skipped on the IR's word rather
            # than this loop's arithmetic, which is the same guarantee the
            # comparison made: never re-run a step, or a finished calendar insert
            # would be inserted twice. Each pass walks from the program's entry,
            # so a resume derives its position from durable facts instead of
            # inheriting one.
            while True:
                step = next_step(
                    program,
                    _control_facts(
                        cursor=cursor,
                        task_state=task.state,
                        steps_used=steps_used,
                        evidence=evidence,
                    ),
                )
                if step is None:
                    break
                await self._fence_execution(
                    session, task_id=task.id, tenant_id=task.tenant_id, claim=claim
                )
                intents = self._intents_for(
                    step.capability,
                    scenario=scenario,
                    identities=identities,
                    collected=collected,
                    created_event_id=created_event_id,
                    timezone=row.timezone,
                    objective=task.goal,
                )
                for index, (tool_name, args, identity) in enumerate(intents):
                    steps_used += 1
                    # The invocation is described as IR first and translated into
                    # a ToolIntent second. Nothing else changes: the same Gateway,
                    # the same fences, the same durable record, the same request
                    # id — the IR sits between the plan and the call.
                    action = action_for(
                        step=step,
                        index=index,
                        capability=step.capability,
                        arguments=args,
                        task_id=task.id,
                        plan_version=plan_version,
                    )
                    intent = intent_from_action(
                        action,
                        tool_name=tool_name,
                        task_id=task.id,
                        tenant_id=task.tenant_id,
                        user_id=task.user_id,
                    )
                    context = ToolContext(
                        session=session,
                        task_id=task.id,
                        tenant_id=task.tenant_id,
                        user_id=task.user_id,
                        account_id=account.id if account is not None else None,
                        transport=transport,
                        metadata={
                            "steps_used": steps_used,
                            "content_category": content_category_for(tool_name),
                        },
                        identity=identity,
                    )
                    # Last gate before the one call that can change the world:
                    # a worker that lost the claim never reaches the provider.
                    await self._fence_execution(
                        session,
                        task_id=task.id,
                        tenant_id=task.tenant_id,
                        claim=claim,
                    )
                    result = await self.gateway.execute(intent, context)
                    observation = observation_from_result(
                        action=action, step_order=step.order, result=result
                    )
                    attach(program, observation)
                    # The IR's own error taxonomy, applied to what the Gateway
                    # actually returned. Bookkeeping only: the failure decision
                    # below still reads the ToolResult, so a run behaves exactly
                    # as it did before this line existed. What changes is that a
                    # failure is now also recorded in the IR's own vocabulary
                    # rather than only in the task state machine.
                    record_error(program, observation)
                    record = evidence_from_observation(observation, tool_name=tool_name)
                    record["ir_program_id"] = str(program.program_id)
                    record["ir_fingerprint"] = program.fingerprint
                    if _external_outcome_unestablished(self.gateway, result):
                        # What the runtime concluded about the provider's
                        # side, kept apart from the error code that says why
                        # the gateway refused. Written before the branch below
                        # so the conclusion is durable even if this process
                        # dies next.
                        record[EXTERNAL_OUTCOME] = EXTERNAL_OUTCOME_UNKNOWN
                    await append_event(
                        session,
                        tenant_id=task.tenant_id,
                        task_id=task.id,
                        execution_id=row.id,
                        event_type="tool_observation",
                        payload=record,
                    )
                    evidence.append(record)
                    row.evidence = list(evidence)
                    row.steps_used = steps_used

                    if result.outcome is ToolOutcome.APPROVAL_REQUIRED:
                        await self._fence_execution(
                            session,
                            task_id=task.id,
                            tenant_id=task.tenant_id,
                            claim=claim,
                        )
                        # The task really parks in WAITING_APPROVAL; the runtime
                        # never reports a state the task did not enter.
                        await session.refresh(task)
                        if task.state != WAITING_APPROVAL:
                            await transition_task(
                                session,
                                task_id=task.id,
                                tenant_id=task.tenant_id,
                                user_id=task.user_id,
                                target=WAITING_APPROVAL,
                            )
                        pending = await self.pending_approval_ids(session, task)
                        # No result is recorded: a task parked for a human has
                        # not ended, and WAITING_APPROVAL is deliberately not a
                        # reportable result state. The next attempt rebuilds the
                        # program and finishes the record.
                        return TriggerOutcome(
                            WAITING_APPROVAL,
                            "WAITING_APPROVAL",
                            "human approval required to create the calendar event",
                            pending,
                            steps_used,
                        )
                    if (
                        result.outcome is not ToolOutcome.EXECUTION_SUCCEEDED
                        or not result.ok
                    ):
                        if _external_outcome_unestablished(self.gateway, result):
                            # The approval that authorised this write is spent
                            # (or the provider never answered), so the step
                            # cannot run again, and nothing durable says whether
                            # it already did. Retrying is refused by the gateway
                            # and asserting failure is a guess, so the honest
                            # answer is INDETERMINATE: park for a human to
                            # reconcile instead of reporting a failure the
                            # provider may already have contradicted.
                            #
                            # The verdict is not asserted here. The task moves to
                            # VERIFYING and the one settlement path decides, from
                            # the durable log, whether the lost answer matters:
                            # a witness may still prove the intent, in which case
                            # this completes rather than parks. That keeps
                            # "which evaluator concluded what" true on a recovery
                            # exactly as it is on the live path.
                            await self._fence_execution(
                                session,
                                task_id=task.id,
                                tenant_id=task.tenant_id,
                                claim=claim,
                            )
                            await transition_task(
                                session,
                                task_id=task.id,
                                tenant_id=task.tenant_id,
                                user_id=task.user_id,
                                target=VERIFYING,
                            )
                            record_result(
                                program,
                                result="REQUIRES_ATTENTION",
                                reason="external outcome was never recorded",
                            )
                            return await self._settle_verification(
                                session,
                                task=task,
                                row=row,
                                scenario=scenario,
                                claim=claim,
                            )
                        message = result.message or f"tool {tool_name} did not succeed"
                        target = (
                            "REQUIRES_ATTENTION"
                            if result.outcome is ToolOutcome.REQUIRES_ATTENTION
                            else "FAILED"
                        )
                        await self._fence_execution(
                            session,
                            task_id=task.id,
                            tenant_id=task.tenant_id,
                            claim=claim,
                        )
                        await transition_task(
                            session,
                            task_id=task.id,
                            tenant_id=task.tenant_id,
                            user_id=task.user_id,
                            target=target,
                            fail_reason=message,
                        )
                        # The run ends here, so the program says so. The
                        # transition above is unchanged and remains the only
                        # thing that decides the task's fate; this only writes
                        # down the outcome that was already reached.
                        record_result(program, result=target, reason=message)
                        return TriggerOutcome(target, target, message, [], steps_used)

                    if tool_name == "calendar.read_slots":
                        collected.extend(_read_slot_items([record]))
                    elif tool_name == "calendar.create_event":
                        # Through the one qualification boundary, on the record
                        # this attempt just wrote — the same boundary the
                        # fulfillment evaluator applies, and the same one the
                        # resume path below reads the log through. There is no
                        # second, looser reading of "an event was created" on
                        # the live path.
                        witness = qualifying_creation(record)
                        if witness is not None:
                            created_event_id = witness.provider_event_id

                cursor = step.order
                row.cursor = step.order
                row.evidence = list(evidence)
                row.steps_used = steps_used
                await self._fence_execution(
                    session, task_id=task.id, tenant_id=task.tenant_id, claim=claim
                )

            await self._fence_execution(
                session, task_id=task.id, tenant_id=task.tenant_id, claim=claim
            )
            await transition_task(
                session,
                task_id=task.id,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                target=VERIFYING,
            )
            settled = await self._settle_verification(
                session, task=task, row=row, scenario=scenario, claim=claim
            )
            # The flow is finished, so the run has ended one way or another.
            # COMPLETED, FAILED, and REQUIRES_ATTENTION are all reportable
            # result states, so this records whichever one verification settled
            # on — after the transitions, never instead of them.
            record_result(program, result=settled.outcome, reason=settled.message)
            return settled
        except (IrConditionError, IrControlFlowError) as exc:
            # A graph the runtime cannot walk is a fault in the program, not a
            # tool failure and not something to retry: the branch structure does
            # not match the facts, and the only safe reading of a control-flow
            # decision this program cannot make is to make none of them.
            return await self._fail(session, task, f"control flow is not traversable: {exc}")
        except RuntimeFatal as exc:
            return await self._fail(session, task, str(exc))

    async def _settle_verification(
        self,
        session: AsyncSession,
        *,
        task: Task,
        row: RuntimeExecution,
        scenario: UnderstandingOutput | None,
        claim: UUID,
    ) -> TriggerOutcome:
        """Give a task in ``VERIFYING`` the verdict its evidence already implies.

        Shared by the live execution path and by the recovery path, so a task
        that crashed a moment ago and a task that ran cleanly are decided by
        exactly the same code, on exactly the same inputs. There is one call to
        the evaluator in this function and no second settlement path anywhere, so
        "what was recorded" cannot differ between the two.

        This is where the fulfillment verdict becomes a task state, and it is
        the only place in the runtime that does. The evaluator below is handed
        the sealed contract and the durable row and returns an opinion; the
        transitions below are this function's alone.

        The audit triple is written *before* the transition, onto the row the
        evaluator was just given, so the record and the state it produced commit
        together. It is written from the freshly derived outcome and is never
        read back: recovery re-derives exactly as the live path does, which is
        what keeps the persisted verdict an artifact rather than a second source
        of truth.
        """
        await self._fence_execution(
            session, task_id=task.id, tenant_id=task.tenant_id, claim=claim
        )
        outcome = _fulfillment_outcome(
            task,
            scenario=scenario,
            identities=list(row.identities),
            evidence=list(row.evidence),
        )
        # The audit record, not the decision. Nothing below reads it back, and
        # every settlement re-derives and overwrites it — so a record that ever
        # disagreed with the evidence would be corrected, not obeyed.
        row.fulfillment_profile = outcome.profile
        row.evaluator_version = outcome.version
        row.fulfillment_verdict = outcome.verdict.value
        verdict = outcome.verdict
        if verdict is FulfillmentVerdict.FULFILLED:
            await transition_task(
                session,
                task_id=task.id,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                target=COMPLETED,
            )
            message = (
                "task prepared and recorded"
                if outcome.profile == _TASK_PREPARATION
                else "meeting scheduled and verified"
            )
            return TriggerOutcome(
                COMPLETED,
                COMPLETED,
                message,
                [],
                row.steps_used,
            )
        if verdict is FulfillmentVerdict.NOT_FULFILLED:
            return await self._fail(
                session,
                task,
                "verified execution did not satisfy the task success criteria",
                steps_used=row.steps_used,
            )
        # INDETERMINATE is not a failure: the run happened, and the evidence
        # available cannot settle whether the Intent was fulfilled. That is
        # exactly what REQUIRES_ATTENTION is for.
        return await self._halt(
            session,
            task,
            f"fulfillment could not be settled from the available evidence ({verdict.value})",
            steps_used=row.steps_used,
        )

    def _program_for(
        self,
        task: Task,
        *,
        proposal: PlanProposal,
        plan_status: str,
        plan_version: int,
        plan_hash: str,
        row: RuntimeExecution,
    ) -> KaiProgram:
        """Build and validate the KAI-IR program for this task's current plan.

        Refuses rather than repairs: an unbuildable program means the plan, the
        contract, and the catalog disagree, and the only safe response is to stop
        instead of executing a plan whose declared permissions cannot be trusted
        (spec §2.1).
        """
        registry = getattr(self.gateway, "registry", None)
        # The profile↔capability licence, checked first and fail-closed: a
        # profile present in no execution entry has no licence at all (the
        # registry maps only the live-sealable profiles), so the task is refused
        # by name here — before any capability is derived and before the program
        # is built or any step is dispatched. This is the runtime's own
        # execution-side authority (``PROFILE_ALLOWED_CAPABILITIES``, the same
        # registry :func:`kai.runtime.translator.profile_scope_violation` reads
        # at :meth:`submit`); ``LIVE_SEALABLE_PROFILES`` remains the
        # storage/creation gate in ``kai.tasks.service``. Reading a missing
        # profile as "no restriction" (the old ``if licensed is not None``) let a
        # row written by any other route reach dispatch only to be refused later
        # by the fulfillment evaluator, so an unknown profile could run a
        # representable step first.
        licensed = PROFILE_ALLOWED_CAPABILITIES.get(task.fulfillment_profile)
        if licensed is None:
            raise RuntimeFatal(
                f"fulfillment profile {task.fulfillment_profile!r} is not "
                "licensed for execution"
            )
        # An unsupported capability is refused here, before the program is built,
        # with the runtime's own wording — the same refusal the runtime used to
        # make mid-step, moved earlier so nothing can be reached first. The IR
        # validates the same constraint as a backstop, but a plan the slice
        # cannot execute is a runtime fact, not a validation detail.
        allowed = set(allowed_capabilities_from_task(task, CAPABILITY_TO_TOOL))
        # Profile↔capability compatibility: a sealed profile licences a fixed
        # capability set, and a contract whose tools derive a capability outside
        # it is refused here — before the program is built and before any step is
        # dispatched. This is the second seam behind :meth:`submit`, so a task
        # that reached the row by any other route still cannot execute a
        # capability its profile did not licence.
        refused = tuple(sorted(allowed - licensed))
        if refused:
            raise RuntimeFatal(
                f"fulfillment profile {task.fulfillment_profile!r} does not "
                f"permit tools/capabilities: {list(refused)!r}"
            )
        for step in proposal.steps:
            if step.action not in allowed:
                raise RuntimeFatal(
                    f"capability not supported by the agent runtime slice: "
                    f"{step.action}"
                )
        try:
            return program_from_plan(
                task=task,
                plan_version=plan_version,
                plan_status=plan_status,
                proposal=proposal,
                plan_hash=plan_hash,
                registry=registry,
                capability_to_tool=CAPABILITY_TO_TOOL,
                cursor=row.cursor,
                steps_used=row.steps_used,
            )
        except IrValidationError as exc:
            raise RuntimeFatal(f"plan is not valid KAI-IR: {exc}") from exc

    def _intents_for(
        self,
        action: str,
        *,
        scenario: UnderstandingOutput | None,
        identities: list[dict[str, Any]],
        collected: list[tuple[datetime, datetime]],
        created_event_id: str | None,
        timezone: str,
        objective: str | None = None,
    ) -> list[tuple[str, dict[str, Any], IdentityQuery | None]]:
        tool_name = capability_to_tool(action)
        if tool_name is None:
            raise RuntimeFatal(
                f"capability not supported by the agent runtime slice: {action}"
            )

        if tool_name == "task.prepare":
            # The one argument is the task's own persisted goal — the sealed user
            # intent — never a planner- or model-supplied string. A planner step
            # names the capability; the runtime is what fills in the objective,
            # so the local tool cannot be told to "prepare" something the user
            # never asked for.
            if not objective or not objective.strip():
                raise RuntimeFatal("cannot prepare: the task has no goal on record")
            return [(tool_name, {"objective": objective}, None)]

        if scenario is None:
            raise RuntimeFatal(
                f"capability {action!r} requires a sealed understanding on record"
            )

        if tool_name == "contacts.match":
            query = scenario.guests[0] if scenario.guests else scenario.window.title
            return [(tool_name, {"query": query}, None)]

        if tool_name == "calendar.read_slots":
            days = candidate_days(scenario.window)
            if not days:
                raise RuntimeFatal("no candidate days in the requested window")
            return [
                (
                    tool_name,
                    read_intent_args(
                        day=day, timezone=timezone, earliest_time=scenario.window.earliest_time
                    ),
                    None,
                )
                for day in days
            ]

        if tool_name == "calendar.create_event":
            slot = pick_slot(collected)
            if slot is None:
                raise RuntimeFatal("no free one-hour slot in the requested window")
            emails = [item["email"] for item in identities if item.get("email")]
            if not emails:
                raise RuntimeFatal("no verified attendee address")
            args = create_intent_args(
                slot=slot,
                title=scenario.window.title,
                timezone=timezone,
                attendee_emails=emails,
            )
            queried_identity = IdentityQuery(email=emails[0])
            return [(tool_name, args, queried_identity)]

        if tool_name == "calendar.get_event":
            if not created_event_id:
                raise RuntimeFatal("cannot verify: no created event id on record")
            return [(tool_name, {"event_id": created_event_id}, None)]

        raise RuntimeFatal(
            f"capability not supported by the agent runtime slice: {action}"
        )

    # ------------------------------------------------------------------ #
    # Support
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _load_execution(
        session: AsyncSession, *, task_id: UUID, tenant_id: UUID
    ) -> RuntimeExecution | None:
        result = await session.execute(
            select(RuntimeExecution).where(
                RuntimeExecution.task_id == task_id,
                RuntimeExecution.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def _bind_plan(
        session: AsyncSession, row: RuntimeExecution, plan: TaskPlan, claim: UUID
    ) -> tuple[bool, int, str]:
        """Bind the saved cursor to the plan version it counts, or refuse.

        Returns ``(bound, plan_version, plan_hash)``. ``bound`` is False only when
        the row is already counting a *different* plan and has work on record: the
        cursor then describes steps that no longer exist, and reusing it would
        silently skip the current plan's steps or re-execute finished ones. In
        that case the trigger must stop instead of guessing.

        Re-binding is allowed only while nothing has run yet (no cursor
        progress, no evidence, no step spent), which is the one case where the
        saved state cannot contradict a different plan.

        The hash comes back because the IR program is bound to exactly this
        plan content: the same digest that pins the cursor also pins the program,
        so a program can never be executed against a plan it was not built from.
        """
        version = plan.version
        digest = plan_fingerprint(plan.plan_json)
        if row.plan_version == version and row.plan_hash == digest:
            return True, version, digest
        if row.cursor > 0 or row.evidence or row.steps_used > 0:
            return False, version, digest
        row.plan_version = version
        row.plan_hash = digest
        await AgentRuntime._fence_execution(
            session,
            task_id=row.task_id,
            tenant_id=row.tenant_id,
            claim=claim,
        )
        return True, version, digest

    @staticmethod
    async def _claim_execution(
        session: AsyncSession, *, task_id: UUID, tenant_id: UUID
    ) -> UUID | None:
        """Take the per-task execution claim, or return ``None`` if taken.

        One conditional UPDATE decides the winner: a second caller's statement
        blocks on the row lock, then re-reads the committed row and matches
        nothing, so exactly one caller ever owns the execution. The statement is
        committed immediately — nothing is held while the provider is called —
        and a claim older than ``_CLAIM_LEASE`` is treated as abandoned by a
        crashed worker and may be taken over.
        """
        token = uuid4()
        now = datetime.now(UTC)
        result = await session.execute(
            update(RuntimeExecution)
            .where(
                RuntimeExecution.task_id == task_id,
                RuntimeExecution.tenant_id == tenant_id,
                or_(
                    RuntimeExecution.claim_token.is_(None),
                    RuntimeExecution.claimed_at < now - _CLAIM_LEASE,
                ),
            )
            .values(claim_token=token, claimed_at=now)
            .returning(RuntimeExecution.task_id)
        )
        claimed = result.scalar_one_or_none() is not None
        await session.commit()
        return token if claimed else None

    @staticmethod
    async def _renew_claim(
        session: AsyncSession, *, task_id: UUID, tenant_id: UUID, claim: UUID
    ) -> bool:
        """Renew the lease if and only if ``claim`` still owns the row.

        One conditional statement decides it, so the answer is never stale: it
        matches the row only while the stored token is this worker's, and
        matching rows get a fresh ``claimed_at``. The caller commits.
        """
        result = await session.execute(
            update(RuntimeExecution)
            .where(
                RuntimeExecution.task_id == task_id,
                RuntimeExecution.tenant_id == tenant_id,
                RuntimeExecution.claim_token == claim,
            )
            .values(claimed_at=datetime.now(UTC))
            .returning(RuntimeExecution.task_id)
        )
        return result.scalar_one_or_none() is not None

    @staticmethod
    async def _fence_execution(
        session: AsyncSession, *, task_id: UUID, tenant_id: UUID, claim: UUID
    ) -> None:
        """Prove this worker still owns the claim, or abandon the trigger.

        The renewal is committed, so a fence that passes both proves ownership
        and buys a fresh :data:`_CLAIM_LEASE` before the caller's next external
        call or write — nobody else can take the claim in that window. A fence
        that fails rolls back first, discarding everything this worker had
        pending, and then raises: by then its cursor, evidence, and state are
        known stale, and the new owner is the only writer that counts.
        """
        if await AgentRuntime._renew_claim(
            session, task_id=task_id, tenant_id=tenant_id, claim=claim
        ):
            await session.commit()
            return
        await session.rollback()
        raise ClaimLostError("the execution claim is no longer owned by this trigger")

    async def _heartbeat(
        self,
        *,
        task_id: UUID,
        tenant_id: UUID,
        claim: UUID,
        stop: asyncio.Event,
    ) -> None:
        """Keep the owner's claim lease fresh for as long as it works.

        Renews on its own short-lived session, never on the owner's, so a slow
        model or a slow provider call cannot serialize the renewal behind the
        very work it is protecting — which is the whole point: the lease has to
        stay young while the owner is inside a call, not only between them.

        Stops on ``stop``, and gives up for good the moment a renewal matches
        nothing: from then on this worker is not the executor, and the owner's
        next fence is what stops it before it writes anything.
        """
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=_CLAIM_HEARTBEAT_SECONDS)
                return
            except TimeoutError:
                pass
            try:
                async with self._session_factory() as beat:
                    if await self._renew_claim(
                        beat, task_id=task_id, tenant_id=tenant_id, claim=claim
                    ):
                        await beat.commit()
                    else:
                        await beat.rollback()
                        self._log.warning(
                            "execution claim heartbeat lost the claim",
                            task_id=str(task_id),
                            tenant_id=str(tenant_id),
                        )
                        return
            except asyncio.CancelledError:
                raise
            except SQLAlchemyError as exc:
                # A renewal that never reached the database proves nothing, and
                # the lease is still valid for the rest of its window. Say so
                # and let the next tick try again rather than dropping the
                # heartbeat on a transient connection failure.
                self._log.warning(
                    "execution claim heartbeat could not renew the lease",
                    task_id=str(task_id),
                    tenant_id=str(tenant_id),
                    error=str(exc),
                )

    async def _lost_claim_outcome(
        self, *, task_id: UUID, tenant_id: UUID, fallback_state: str
    ) -> TriggerOutcome:
        """Report where the task actually is, not what this worker last saw.

        Read through a fresh session because this worker's own session is being
        abandoned: anything uncommitted in it is stale by definition, and the
        owner now driving the task is the only one whose state counts.
        """
        async with self._session_factory() as probe:
            state = await probe.scalar(
                select(Task.state).where(
                    Task.id == task_id, Task.tenant_id == tenant_id
                )
            )
            steps_used = await probe.scalar(
                select(RuntimeExecution.steps_used).where(
                    RuntimeExecution.task_id == task_id,
                    RuntimeExecution.tenant_id == tenant_id,
                )
            )
        return TriggerOutcome(
            state or fallback_state,
            "CLAIM_LOST",
            "this trigger lost the execution claim to another worker; it ran no "
            "step and wrote no state",
            [],
            int(steps_used or 0),
        )

    @staticmethod
    async def _release_execution(
        session: AsyncSession, *, task_id: UUID, tenant_id: UUID, claim: UUID
    ) -> None:
        """Hand the claim back; only the owner can release it."""
        await session.execute(
            update(RuntimeExecution)
            .where(
                RuntimeExecution.task_id == task_id,
                RuntimeExecution.tenant_id == tenant_id,
                RuntimeExecution.claim_token == claim,
            )
            .values(claim_token=None, claimed_at=None)
        )
        await session.commit()

    @staticmethod
    async def _resolve_account(
        session: AsyncSession, task: Task
    ) -> GoogleAccount | None:
        result = await session.execute(
            select(GoogleAccount)
            .where(
                GoogleAccount.tenant_id == task.tenant_id,
                GoogleAccount.user_id == task.user_id,
            )
            .order_by(GoogleAccount.created_at, GoogleAccount.id)
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def pending_approval_ids(
        session: AsyncSession, task: Task
    ) -> list[str]:
        result = await session.execute(
            select(Approval).where(
                Approval.task_id == task.id,
                Approval.tenant_id == task.tenant_id,
                Approval.status == ApprovalStatus.PENDING.value,
            )
        )
        return [str(approval.id) for approval in result.scalars().all()]

    @staticmethod
    async def execution_row(
        session: AsyncSession, *, task_id: UUID, tenant_id: UUID
    ) -> RuntimeExecution | None:
        return await AgentRuntime._load_execution(
            session, task_id=task_id, tenant_id=tenant_id
        )

    @classmethod
    async def _fail(
        cls,
        session: AsyncSession,
        task: Task,
        reason: str,
        *,
        steps_used: int = 0,
    ) -> TriggerOutcome:
        await transition_task(
            session,
            task_id=task.id,
            tenant_id=task.tenant_id,
            user_id=task.user_id,
            target=FAILED,
            fail_reason=reason,
        )
        return TriggerOutcome("FAILED", "FAILED", reason, [], steps_used)

    @classmethod
    async def _halt(
        cls,
        session: AsyncSession,
        task: Task,
        reason: str,
        *,
        steps_used: int = 0,
    ) -> TriggerOutcome:
        """Stop a task that has run, but cannot continue on its own.

        ``REQUIRES_ATTENTION`` and not ``FAILED``: the work already happened, so
        a human has to decide what the task means now. The saved cursor and its
        evidence are left exactly as they are — nothing is rewound, nothing is
        re-executed.
        """
        await transition_task(
            session,
            task_id=task.id,
            tenant_id=task.tenant_id,
            user_id=task.user_id,
            target=REQUIRES_ATTENTION,
            fail_reason=reason,
        )
        return TriggerOutcome("REQUIRES_ATTENTION", "REQUIRES_ATTENTION", reason, [], steps_used)


def _plan_diverged(bound_version: int | None, current_version: int) -> str:
    """Deterministic, non-sensitive reason for a refused resume."""
    return (
        f"plan changed after execution started (saved cursor is bound to plan v"
        f"{bound_version}, current plan is v{current_version}); the saved cursor "
        "was not reused and no step was executed"
    )


def _advanced_state_prerequisite_error(
    *,
    task_state: str,
    scenario: UnderstandingOutput | None,
    execution: RuntimeExecution,
) -> str | None:
    """Why an advanced task cannot be driven from its durable state, if it cannot.

    ``None`` means both prerequisites an advanced state relies on are on record.
    Identity resolution is a pre-planning stage: the runtime resolves the guests
    exactly once, while the task is in ``UNDERSTANDING``. Replaying it from an
    advanced state would bind a new identity to a plan and to evidence that were
    sealed without it, so a missing prerequisite there is a refusal rather than
    work to redo.

    The check is presence-only, deliberately: this module does not classify
    persisted identity statuses, and the fulfillment criteria own what a sealed
    identity set means.
    """
    if scenario is None:
        return (
            f"cannot resume a task in {task_state}: no understanding on record; "
            "the understanding is resolved before planning and is not replayed "
            "from an advanced state"
        )
    if not execution.identities:
        return (
            f"cannot resume a task in {task_state}: no runtime identities on "
            "record; identity resolution must complete before the task leaves "
            "UNDERSTANDING and is not replayed from an advanced state"
        )
    return None


def _control_facts(
    *, cursor: int, task_state: str, steps_used: int, evidence: list[dict[str, Any]]
) -> ControlFacts:
    """The durable facts a KAI-IR condition may read, rebuilt from the row.

    Assembled from what is already persisted — the cursor, the step count, the
    evidence log — so a branch taken after a resume is decided by the same facts
    it would have been before the crash, and no new column is needed to know where
    the program is. Evidence rows that are not observations are skipped rather
    than guessed at: a condition reading a step with no observation is refused by
    the IR, not satisfied by a record that happens to be nearby.
    """
    observations: list[ObservationNode] = []
    for record in evidence:
        observation = observation_from_evidence(record)
        if observation is not None:
            observations.append(observation)
    return ControlFacts(
        cursor=cursor,
        task_state=task_state,
        steps_used=steps_used,
        observations=tuple(observations),
    )


def _read_slot_items(
    evidence: list[dict[str, Any]],
) -> list[tuple[datetime, datetime]]:
    intervals: list[tuple[datetime, datetime]] = []
    for entry in evidence:
        if entry.get("tool") != "calendar.read_slots":
            continue
        if entry.get("outcome") != "EXECUTION_SUCCEEDED" or not entry.get("ok"):
            continue
        intervals.extend(parse_free_intervals((entry.get("data") or {}).get("free")))
    return intervals


def _external_outcome_unestablished(
    gateway: ToolGateway, result: ToolResult
) -> bool:
    """Did this run of a side-effecting tool end without a knowable outcome?

    Three things have to hold, and the first two are the ones that keep the
    layers apart:

    * the tool **changes the world**, per its declared ``verification_contract``;
    * and either the provider's answer did not settle what happened
      (:func:`kai.tools.errors.is_unresolved_external` -- dispatched with no
      complete response, or a 5xx that does not say whether the write was
      applied); **or** the approval's single use was already spent, which means
      an earlier attempt reached the point of dispatch and left no receipt.

    **``APPROVAL_STALE`` is an authorization fact, not an outcome, and the two
    are not the same claim.** The gateway consumes an approval immediately
    *before* dispatch, so the refusal this invocation received provably stopped
    *this* invocation before the provider was reached -- no side effect was
    attempted here. What it does not establish is what the *earlier* attempt that
    held the same single use did. That earlier attempt left no durable record at
    all, so this process can neither confirm nor exclude that a side effect
    occurred.

    Because that inference spans two attempts rather than describing this one,
    it is written as its own conclusion under :data:`EXTERNAL_OUTCOME` rather than
    being smuggled into the error taxonomy. A reader can then tell a refusal
    from an unresolved world without interpreting anything.

    A provably-never-contacted provider is excluded: no bytes left the process,
    so nothing is unknown about the world. So is a 4xx: the provider rejected the
    request, which for a write is positive evidence that it did not apply.
    """
    if not (
        is_unresolved_external(result.error_code)
        or result.error_code == APPROVAL_STALE
    ):
        return False
    registry = getattr(gateway, "registry", None)
    if registry is None:
        return False
    definition = registry.get_if_present(result.tool_name)
    if definition is None:
        return False
    return definition.verification_contract is not VerificationContract.READ_BACKED


def _fulfillment_outcome(
    task: Task,
    *,
    scenario: UnderstandingOutput | None,
    identities: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> FulfillmentOutcome:
    """Decide fulfillment for one task, and say which semantics decided it.

    The Sealed Contract's ``fulfillment_profile`` is the only thing that selects
    an evaluator, and it is read off the task row — so a live run and a recovered
    run pick the same evaluator, and neither a planner proposal nor a model
    response can redirect the decision. The dispatch is an explicit closed
    vocabulary rather than a registry: the branches are written out here, an
    unknown profile is a fatal refusal rather than a lookup, and a further domain
    is an amendment to this function, not a runtime resolution.

    **``gmail_send`` is the second branch, and it is not reachable from the live
    slice.** :data:`kai.runtime.translator.SLICE_FULFILLMENT_PROFILE` seals
    ``calendar_meeting`` for every task the runtime drives, and driving a send
    would need a second understanding intent, a second capability table, and a
    KAI-IR program for it — none of which this experiment added or may add. What
    is implemented and tested is the *binding*: that a contract naming
    ``gmail_send`` gets the gmail criteria, the gmail evaluator and the gmail
    version identity, and that the recovery path reaches the same branch from the
    same durable row. Whether a send can be *driven* is deferred, and is recorded
    as deferred rather than approximated by loosening the slice.

    The profile is *reported*, not resolved: it is the value already sealed on
    the task, carried through untouched, so the recorded profile is the
    authoritative one by construction rather than by a second lookup agreeing
    with the first. The version comes from the same branch that runs the
    evaluator, so an identity can never be recorded for semantics that did not
    run.

    This is also the single place the typed Success Criteria are built, from the
    sealed scenario window and the sealed identities, immediately before they are
    evaluated. Both inputs are already committed and never recomputed, so the
    criteria are as fixed as the rows they come from — and because that is true,
    nothing needs to be stored a second time to keep them sealed, and a resumed
    run reconstructs the identical value.

    This function returns an opinion and nothing else. It performs no transition,
    and the profile it reads is validated on the way into the contract.
    """
    profile = task.fulfillment_profile
    if profile == _CALENDAR_MEETING:
        # The constant is the same value as ``profile`` — the branch is an
        # equality — and it carries the sealed profile's own type, so the
        # criteria and the outcome both name the profile that was checked.
        if scenario is None:
            raise RuntimeFatal(
                "calendar fulfillment requires a sealed understanding on record"
            )
        criteria = calendar_meeting_criteria(
            profile=_CALENDAR_MEETING,
            window=scenario.window,
            identities=identities,
        )
        return FulfillmentOutcome(
            profile=_CALENDAR_MEETING,
            version=CALENDAR_MEETING_EVALUATOR_VERSION,
            verdict=evaluate_calendar_meeting(
                FulfillmentInput(criteria=criteria, evidence=tuple(evidence))
            ),
        )
    if profile == _GMAIL_SEND:
        # The second sealed question, dispatched the same way and for the same
        # reasons: the contract's profile selects the evaluator, the criteria are
        # built here from sealed rows, the version comes from the same branch that
        # runs the evaluator, and the function still returns an opinion and
        # performs no transition.
        #
        # ``scenario`` is deliberately not read. A calendar contract is sealed
        # with a meeting window, so its criteria need one; a send contract has no
        # window, and its whole question is answered from the resolved identities
        # already on the row. That the two domains need *different* sealed inputs
        # is the finding — not a shortcut, and not a case of one branch quietly
        # borrowing the other's inputs.
        criteria = gmail_send_criteria(
            profile=_GMAIL_SEND,
            identities=identities,
        )
        return FulfillmentOutcome(
            profile=_GMAIL_SEND,
            version=GMAIL_SEND_EVALUATOR_VERSION,
            verdict=evaluate_gmail_send(
                FulfillmentInput(criteria=criteria, evidence=tuple(evidence))
            ),
        )
    if profile == _TASK_PREPARATION:
        # The third sealed question. Its one intent-anchored criterion is the
        # digest of the persisted task goal — a sealed statement of what the user
        # asked for — while the step count is a request-side condition. Neither
        # the scenario nor the resolved identities are read: a preparation names
        # no guest and carries no window, and the whole question is answered from
        # the durable evidence log and the sealed goal on the task row.
        criteria = task_preparation_criteria(
            profile=_TASK_PREPARATION,
            objective=task.goal,
        )
        return FulfillmentOutcome(
            profile=_TASK_PREPARATION,
            version=TASK_PREPARATION_EVALUATOR_VERSION,
            verdict=evaluate_task_preparation(
                FulfillmentInput(criteria=criteria, evidence=tuple(evidence))
            ),
        )
    raise RuntimeFatal(
        f"no fulfillment evaluator for sealed profile {profile!r}"
    )


def _created_event_id(evidence: list[dict[str, Any]]) -> str | None:
    """The created meeting this run may read back, or ``None``.

    A **sequencing** answer, not a fulfillment one: the runtime has to name *one*
    meeting when the next step's ``calendar.get_event`` needs an id, and the
    earliest qualifying creation in an append-only log is that meeting. Nothing
    here decides whether anything was fulfilled — the verdict is the evaluator's,
    over the whole witness set, and this choice cannot reach it.

    What it does decide is *what counts as a creation*, so it does not decide
    that for itself: the answer is read off
    :func:`kai.runtime.evidence.qualifying_creations`, the same boundary the
    evaluator uses. Concretely, that means the id comes from the record's
    verification ``evidence`` and never from ``data``, and a record that is not a
    verified, provider-identified attempt contributes nothing. Before this was
    shared, this helper accepted a ``data.event_id`` fallback and did not require
    ``ok``, so a record the evaluator called non-qualifying could still have been
    acted on here; both of those are now the evaluator's reading, by construction.
    """
    return next(
        (witness.provider_event_id for witness in qualifying_creations(evidence)),
        None,
    )


__all__ = [
    "AgentRuntime",
    "ClaimLostError",
    "RuntimeFatal",
    "RuntimeNotFoundError",
    "RuntimeResumeError",
    "TriggerOutcome",
    "_fulfillment_outcome",
]
