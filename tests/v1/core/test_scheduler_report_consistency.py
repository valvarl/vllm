# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-rank failure-report consistency (the hardening series, H1).

The report channel's distributed contract: every TP rank reports the
same fault, the output aggregator concatenates the per-rank lists, and
the scheduler folds the duplicates into ONE recovery plan per request.
These tests drive the real Scheduler with the aggregated (already
concatenated) report lists -- the exact shape update_from_output
receives -- and pin the consensus semantics:

- identical duplicates (any multiplicity, any order) fold to one plan;
- a silent majority (one rank reporting alone) still installs the plan
  (the evidence is genuine; the repair rewrites every rank);
- divergent evidence -- different regions, or the same region with
  different metadata -- NEVER authorizes a recovery: the request falls
  back to the conservative rollback-recompute path and the engine keeps
  scheduling;
- malformed evidence fails closed to a full recompute;
- independent requests keep their independent plans.
"""

import pytest

from tests.v1.core.test_scheduler_repair_budget import (
    BS,
    _deliver_report,
    _structured_report,
    _terminal_attestation,
    admit_sync_external_hit,
    make_scheduler,
)
from tests.v1.kv_connector.unit.utils import create_request
from vllm.v1.outputs import KVConnectorOutput

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _deliver_reports(scheduler, out, reports, reqs, attestations=None):
    """update_from_output carrying an AGGREGATED report list (the shape
    the TP output aggregator produces: one entry per reporting rank)
    plus the CONFORMING positive evidence: a terminal, fully certified
    attestation per request (the fixture scheduler is TP=1, so the
    completeness gate requires exactly one per request). Tests
    exercising the incomplete-evidence shapes pass their own (or no)
    attestations."""
    from tests.v1.kv_connector.unit.utils import create_model_runner_output

    output = create_model_runner_output(reqs=reqs, use_eos=False)
    if attestations is None:
        attestations = [
            _terminal_attestation(req_id) for req_id in {r.request_id for r in reports}
        ]
    output.kv_connector_output = KVConnectorOutput(
        load_failure_reports=reports,
        load_attestations=attestations,
    )
    scheduler.update_from_output(out, output)


def _sync_claim_scenario(scheduler=None, num_tokens=3328):
    scheduler = scheduler or make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=num_tokens)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    return scheduler, r1, out1


def test_identical_reports_from_every_rank_fold_to_one_plan():
    """The canonical TP shape: every rank reports the identical fault.
    N identical reports (here 4, standing in for ws=4) install exactly
    one plan; nothing diverges, nothing is dropped."""
    scheduler, r1, out1 = _sync_claim_scenario()
    reports = [_structured_report(scheduler, r1) for _ in range(4)]
    for i, report in enumerate(reports):
        report.source_rank = i
    _deliver_reports(scheduler, out1, reports, reqs=[r1])

    assert r1.request_id in scheduler.recovery_plans
    assert len(scheduler.recovery_plans) == 1
    assert r1.num_computed_tokens == 96 * BS


def test_report_ordering_does_not_matter():
    """Per-request grouping makes the aggregated order irrelevant: the
    same set of reports in a different order yields the same plans."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r1, 3072)
    admit_sync_external_hit(scheduler, r2, 3072)
    out1 = scheduler.schedule()

    reports_a = [
        _structured_report(scheduler, r1),
        _structured_report(scheduler, r2),
    ]
    reports_b = list(reversed(reports_a))
    _deliver_reports(scheduler, out1, reports_b, reqs=[r1, r2])
    assert set(scheduler.recovery_plans) == {r1.request_id, r2.request_id}
    assert r1.num_computed_tokens == 96 * BS
    assert r2.num_computed_tokens == 96 * BS


def test_one_rank_reporting_alone_installs_plan():
    """A rank-asymmetric fault (only one rank observed the failed load)
    is still genuine evidence WHEN THAT RANK ALSO ATTESTS its terminal,
    verified completion of the whole attempt (the fixture scheduler is
    TP=1, so its own attestation completes the evidence): the single
    report installs the plan. The repair replays the span on EVERY
    rank, so a silent majority is not a divergence -- but silence is
    only tolerated when the positive evidence proves the missing ranks
    completed and verified their portions (see
    test_scheduler_transfer_evidence for the silent-rank rejection)."""
    scheduler, r1, out1 = _sync_claim_scenario()
    _deliver_reports(scheduler, out1, [_structured_report(scheduler, r1)], reqs=[r1])
    assert r1.request_id in scheduler.recovery_plans


def test_divergent_regions_reject_recovery_not_engine():
    """Two ranks name different regions of one request: the evidence
    cannot be represented by one recovery. The union is NEVER taken --
    the request truncates to the minimum certified rollback and the
    engine keeps scheduling."""
    scheduler, r1, out1 = _sync_claim_scenario()
    a = _structured_report(scheduler, r1)
    b = _structured_report(
        scheduler,
        r1,
        failed_block_idx=97,
        token_span=(97 * BS, 98 * BS),
        rollback_boundary=97 * BS,
    )
    a.source_rank = 0
    b.source_rank = 1
    _deliver_reports(scheduler, out1, [a, b], reqs=[r1])

    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 96 * BS
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens[r1.request_id] == 3328 - 96 * BS


def test_divergent_metadata_same_region_reject_recovery():
    """The region agrees but the metadata does not (a different
    restoration target): the pre-hardening code silently absorbed this
    shape -- the first report's metadata drove the commit while the
    conflicting certification was never compared. The hardening contract
    refuses it like any other divergence."""
    scheduler, r1, out1 = _sync_claim_scenario()
    a = _structured_report(scheduler, r1)
    b = _structured_report(scheduler, r1, restored_extent=2048)
    a.source_rank = 0
    b.source_rank = 1
    _deliver_reports(scheduler, out1, [a, b], reqs=[r1])

    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 96 * BS


def test_divergent_healthy_above_span_certification_reject_recovery():
    """The same region with conflicting above-span health: one rank
    certifies the above-span ranges byte-exact, the other does not. The
    commit's extent restoration depends on this field, so the conflict
    is a safety-relevant divergence."""
    scheduler, r1, out1 = _sync_claim_scenario()
    a = _structured_report(scheduler, r1, healthy_above_span=True)
    b = _structured_report(scheduler, r1, healthy_above_span=False)
    _deliver_reports(scheduler, out1, [a, b], reqs=[r1])
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 96 * BS


def test_malformed_report_fails_closed_to_full_recompute():
    """Malformed evidence: even the rollback boundary is untrustworthy,
    so the fallback recomputes the whole request (computed=0) and the
    engine survives."""
    scheduler, r1, out1 = _sync_claim_scenario()
    bad = _structured_report(scheduler, r1, rollback_boundary=96 * BS - 3)
    _deliver_reports(scheduler, out1, [bad], reqs=[r1])

    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0
    out2 = scheduler.schedule()
    assert r1.request_id in out2.num_scheduled_tokens


def test_duplicate_after_plan_consumed_same_step_absorbed():
    """Same-step re-delivery against an installed plan: identical
    evidence is absorbed (the plan object is untouched); this is the
    report-side analogue of the stale-result rejection."""
    scheduler, r1, out1 = _sync_claim_scenario()
    _deliver_report(scheduler, r1, out1, _structured_report(scheduler, r1))
    plan = scheduler.recovery_plans[r1.request_id]
    _deliver_report(scheduler, r1, out1, _structured_report(scheduler, r1))
    assert scheduler.recovery_plans[r1.request_id] is plan


def test_multiple_requests_independent_failures():
    """Two requests, one failed region each: two independent plans, no
    cross-talk -- including when one request's evidence diverges (only
    that request falls back)."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r1, 3072)
    admit_sync_external_hit(scheduler, r2, 3072)
    out1 = scheduler.schedule()

    good = _structured_report(scheduler, r1)
    divergent_pair = [
        _structured_report(scheduler, r2),
        _structured_report(
            scheduler,
            r2,
            failed_block_idx=97,
            token_span=(97 * BS, 98 * BS),
            rollback_boundary=97 * BS,
        ),
    ]
    _deliver_reports(scheduler, out1, [good] + divergent_pair, reqs=[r1, r2])

    assert set(scheduler.recovery_plans) == {r1.request_id}
    assert r1.num_computed_tokens == 96 * BS
    assert r2.num_computed_tokens == 96 * BS
    # Both requests keep making progress (r2 through the ordinary path --
    # possibly via preemption and re-admission under the fixture's KV
    # pressure, which is an ordinary scheduler decision, not a stall).
    scheduled = set()
    for _ in range(12):
        out = scheduler.schedule()
        scheduled.update(out.num_scheduled_tokens)
        if {r1.request_id, r2.request_id} <= scheduled:
            break
    assert {r1.request_id, r2.request_id} <= scheduled


def test_unrelated_request_progresses_after_rejected_recovery():
    """A rejected recovery must not stall unrelated traffic: the healthy
    request keeps its ordinary schedule while the damaged one falls
    back."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    r2 = create_request(num_tokens=512)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    assert r2.request_id in out1.num_scheduled_tokens

    a = _structured_report(scheduler, r1)
    b = _structured_report(scheduler, r1, restored_extent=2048)
    _deliver_reports(scheduler, out1, [a, b], reqs=[r1, r2])

    assert scheduler.recovery_plans == {}
    scheduled = set()
    for _ in range(12):
        out = scheduler.schedule()
        scheduled.update(out.num_scheduled_tokens)
        if {r1.request_id, r2.request_id} <= scheduled:
            break
    assert {r1.request_id, r2.request_id} <= scheduled
