# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded admission for recovery plans (hardening H4).

A DETECTED plan is the drain-first admission of its steps: every OTHER
request is frozen while the plan waits for a budget it can fit, so an
unbounded deferral is an engine-wide stall, not a per-request one. The
pre-hardening code incremented plan.num_deferrals and never read it.

The admission condition reduces to input_budget - draft_slots >=
max(E, n_eff) once the long-prefill threshold is exempted (the repair
step is exclusive, so the threshold's starvation protection has no one
to protect, and the span cannot be chunked). These tests pin:

- the static never-admissible gate (a plan no step could ever fund --
  running arm: E + n_eff must fit the input budget beside the draft
  slots; waiting arm: max(E, n_eff) -- dissolves at its first budget
  call, not after freezing the engine);
- the long-prefill threshold exemption (the pre-fix engine froze
  forever when the threshold fell below the span);
- the deferral bound (a plan deferred REPAIR_MAX_ADMISSION_DEFERRALS
  consecutive steps dissolves to the recompute fallback);
- cancellation/completion/preemption cleaning up pending plans;
- one repair admission per step with competing plans;
- ordinary traffic resuming in the very step a plan dissolves.
"""

import pytest

from tests.v1.core.test_scheduler_repair_budget import (
    BS,
    E,
    N_EFF,
    admit_async_external_hit,
    admit_sync_external_hit,
    fail_and_truncate,
    install_plan,
    make_scheduler,
)
from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _plan_installed(scheduler, request, e=E, span=(96 * BS, 96 * BS + N_EFF)):
    return install_plan(scheduler, request, e, span=span)


def test_admission_arithmetic_of_the_abort_fixture():
    """The abort fixture's arithmetic (max_num_batched_tokens=8192, E=2623,
    n_eff=1536): the static gate passes (max(E, n_eff) = 2623 <= 8192)
    and the admission debits exactly the surcharge beside the span --
    the formula the fixture's comment documents."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    scheduler.schedule()
    plan = _plan_installed(scheduler, r1)
    r1.num_computed_tokens = plan.repair_span[0]

    out = scheduler.schedule()
    assert plan.state == "ADMITTED"
    assert out.num_scheduled_tokens[r1.request_id] == N_EFF
    assert out.repair_works is not None


def test_static_never_admissible_plan_dissolves_immediately():
    """A span that can never fit the input budget (n_eff=4992 with
    max_num_batched_tokens=4096 but max_num_scheduled_tokens=8192: the
    pre-hardening gate only checked E against the batched budget and
    n_eff against the scheduled budget, so this plan deferred forever
    while freezing the engine) dissolves at its first budget call."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=4096
    )
    scheduler.max_num_scheduled_tokens = 8192
    r1 = create_request(num_tokens=5888)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 4992)
    out1 = scheduler.schedule()
    assert r1.request_id in out1.num_scheduled_tokens

    # n_eff = 4992 > input_budget 4096 but <= max_num_scheduled_tokens.
    plan = _plan_installed(
        scheduler, r1, e=1500, span=(96 * BS, 96 * BS + 4992)
    )
    r1.num_computed_tokens = plan.repair_span[0]
    out2 = scheduler.schedule()
    # Dissolved: no repair work, the ordinary reschedule owns the
    # request from the rollback boundary (the dissolution step's chunk
    # is still reserve-capped -- benign, one smaller chunk -- and the
    # next step is unconstrained).
    assert scheduler.recovery_plans == {}
    assert out2.repair_works is None
    assert out2.num_scheduled_tokens[r1.request_id] > 0
    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens[r1.request_id] > 0


def test_long_prefill_threshold_never_freezes_the_engine():
    """The threshold deadlock (pre-fix): with another request present the
    long-prefill cap applied to the damaged request, and a threshold
    below the repair span deferred the plan forever -- while the drain-
    first freeze stalled the whole engine. The exemption admits the
    repair; the competitor rides the very next step."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    scheduler.scheduler_config.long_prefill_token_threshold = 256
    r1 = create_request(num_tokens=3328)
    r2 = create_request(num_tokens=256)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    assert r2.request_id in out1.num_scheduled_tokens
    # Process out1's output so r2 carries a sampled token (without it,
    # the fully-prefilled r2 has no work and would be skipped for
    # reasons unrelated to the freeze under test).
    output1 = create_model_runner_output(reqs=[r1, r2], use_eos=False)
    scheduler.update_from_output(out1, output1)

    plan = _plan_installed(scheduler, r1)
    r1.num_computed_tokens = plan.repair_span[0]
    # The threshold (256) is far below the span (1536): without the
    # exemption this admission would cap at 256 and defer forever.
    out2 = scheduler.schedule()
    assert plan.state == "ADMITTED"
    assert out2.repair_works is not None
    assert out2.num_scheduled_tokens[r1.request_id] == N_EFF
    # The repair step is exclusive: r2 waits exactly one step, then
    # resumes.
    assert r2.request_id not in out2.num_scheduled_tokens
    out3 = scheduler.schedule()
    assert r2.request_id in out3.num_scheduled_tokens


def test_budget_exactly_at_threshold_admits():
    """E + n_eff == input_budget: the running arm's boundary is inclusive
    -- the full-charge reserve leaves exactly n_eff, which suffices."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=E + N_EFF
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    scheduler.schedule()
    plan = _plan_installed(scheduler, r1)
    r1.num_computed_tokens = plan.repair_span[0]

    out = scheduler.schedule()
    assert plan.state == "ADMITTED"
    assert out.num_scheduled_tokens[r1.request_id] == N_EFF


def test_deferral_bound_dissolves_to_recompute_fallback():
    """A plan that repeatedly fails admission (here: the dynamic caps
    keep num_new_tokens below the span) dissolves after
    REPAIR_MAX_ADMISSION_DEFERRALS consecutive deferrals and the
    ordinary chunked recompute owns the request."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    assert r1.request_id in out1.num_scheduled_tokens

    plan = _plan_installed(scheduler, r1)
    r1.num_computed_tokens = plan.repair_span[0]
    # Stand one deferral away from the bound, then defer once more
    # through the real budget path (num_new_tokens below the span is
    # exactly what the dynamic caps -- encoder budgets, lookahead
    # reserves, mamba alignment -- produce).
    plan.num_deferrals = Scheduler.REPAIR_MAX_ADMISSION_DEFERRALS - 1
    verdict = scheduler._recovery_plan_budget(
        r1, num_new_tokens=N_EFF - 1, input_budget=8192, draft_slots=0
    )
    assert verdict == (0, 0)
    assert scheduler.recovery_plans == {}

    # The ordinary path owns the request from the rollback boundary.
    out2 = scheduler.schedule()
    assert out2.repair_works is None
    assert out2.num_scheduled_tokens[r1.request_id] == 3328 - 96 * BS


def test_deferral_below_bound_keeps_waiting():
    """The mirror of the bound: a deferral below the bound keeps the
    plan pending (the budget call returns None and the freeze holds)
    -- the bound must not dissolve plans eagerly."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    scheduler.schedule()
    plan = _plan_installed(scheduler, r1)

    verdict = scheduler._recovery_plan_budget(
        r1, num_new_tokens=N_EFF - 1, input_budget=8192, draft_slots=0
    )
    assert verdict is None
    assert scheduler.recovery_plans == {r1.request_id: plan}
    assert plan.num_deferrals == 1


def test_cancellation_during_deferral_purges_plan():
    """A client disconnect (finish_requests) while the plan waits: the
    plan is purged with the request -- no leaked plan, no leaked
    freeze."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    scheduler.schedule()
    _plan_installed(scheduler, r1)

    scheduler.finish_requests(
        [r1.request_id], finished_status=RequestStatus.FINISHED_ABORTED
    )
    assert scheduler.recovery_plans == {}
    # The engine schedules normally afterwards.
    out = scheduler.schedule()
    assert r1.request_id not in out.num_scheduled_tokens


def test_completion_during_deferral_purges_plan():
    """The stopped-completion variant of the cancellation path."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    scheduler.schedule()
    _plan_installed(scheduler, r1)

    scheduler.finish_requests(
        [r1.request_id], finished_status=RequestStatus.FINISHED_STOPPED
    )
    assert scheduler.recovery_plans == {}


def test_competing_plans_admit_one_per_step():
    """Two damaged requests: the repair step is exclusive, so the plans
    admit one per step (the second waits -- deferred, not dissolved)."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r1, 3072)
    admit_sync_external_hit(scheduler, r2, 3072)
    scheduler.schedule()

    plan1 = _plan_installed(scheduler, r1)
    plan2 = _plan_installed(scheduler, r2)
    r1.num_computed_tokens = plan1.repair_span[0]
    r2.num_computed_tokens = plan2.repair_span[0]

    out2 = scheduler.schedule()
    assert out2.repair_works is not None
    assert len(out2.repair_works) == 1
    first = next(iter(out2.repair_works))
    assert {plan1.state, plan2.state} == {"ADMITTED", "DETECTED"}

    out3 = scheduler.schedule()
    assert out3.repair_works is not None
    assert len(out3.repair_works) == 1
    assert next(iter(out3.repair_works)) != first


def test_dissolving_step_resumes_ordinary_traffic_immediately():
    """When a plan dissolves mid-step, the drain-first freeze lifts for
    the rest of THAT step: an ordinary request scheduled after the
    damaged one is not made to wait an extra step."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=4096
    )
    scheduler.max_num_scheduled_tokens = 8192
    r1 = create_request(num_tokens=5888)
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r1, 4992)
    out1 = scheduler.schedule()
    # r1's external-hit chunk plus r2's ordinary chunk exhaust out1's
    # input budget, leaving r2 with work in hand for the next step.
    assert r1.request_id in out1.num_scheduled_tokens
    assert r2.request_id in out1.num_scheduled_tokens
    assert r2.num_tokens - r2.num_computed_tokens > 0

    plan = _plan_installed(scheduler, r1, e=1500, span=(96 * BS, 96 * BS + 4992))
    r1.num_computed_tokens = plan.repair_span[0]
    plan.num_deferrals = Scheduler.REPAIR_MAX_ADMISSION_DEFERRALS - 1
    # The next step: r1's budget call dissolves the never-admissible
    # plan, and r2 (behind r1 in the running queue) schedules in the
    # SAME step -- the freeze does not outlive the plan.
    out2 = scheduler.schedule()
    assert scheduler.recovery_plans == {}
    assert out2.repair_works is None
    assert r2.request_id in out2.num_scheduled_tokens
    assert r1.request_id in out2.num_scheduled_tokens


def test_waiting_queue_plan_admits_repair():
    """The waiting-queue admission arm: an async external claim that
    fails leaves the request WAITING with a plan; the repair admits
    from the waiting queue with the same exclusivity."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    plan = install_plan(
        scheduler, r1, E, failed_block_ids=[block_ids[96]]
    )
    out2 = scheduler.schedule()
    assert plan.state == "ADMITTED"
    assert out2.repair_works is not None
    assert out2.num_scheduled_tokens[r1.request_id] == N_EFF
