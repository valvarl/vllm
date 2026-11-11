# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import (
    KVConnectorOutput,
    KVLoadFailureReport,
    ModelRunnerOutput,
    RepairResultReport,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class DummyModelRunnerOutput(ModelRunnerOutput):
    def __init__(
        self,
        finished_sending: set[str] | None = None,
        finished_recving: set[str] | None = None,
        invalid_block_ids: set[int] | None = None,
        failed_recving: set[str] | None = None,
        expected_finished_count: int = 0,
        load_failure_reports: list[KVLoadFailureReport] | None = None,
        repair_results: list[RepairResultReport] | None = None,
    ):
        self.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            invalid_block_ids=invalid_block_ids or set(),
            failed_recving=failed_recving or set(),
            expected_finished_count=expected_finished_count,
            load_failure_reports=load_failure_reports,
            repair_results=repair_results,
        )

    def __repr__(self):
        return (
            f"DummyModelRunnerOutput("
            f"finished_sending={self.kv_connector_output.finished_sending},"
            f"finished_recving={self.kv_connector_output.finished_recving})"
            f"invalid_block_ids={self.kv_connector_output.invalid_block_ids})"
        )


def test_aggregate_workers_output():
    aggregator = KVOutputAggregator(expected_finished_count=2)

    output1 = DummyModelRunnerOutput()
    output2 = DummyModelRunnerOutput()

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert not aggregated.invalid_block_ids

    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, finished_recving={"req2"}
    )
    output2 = DummyModelRunnerOutput(invalid_block_ids={1})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {1}

    output1 = DummyModelRunnerOutput(invalid_block_ids={2})
    output2 = DummyModelRunnerOutput(finished_sending={"req1"})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending == {"req1"}
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {2}

    output1 = DummyModelRunnerOutput(invalid_block_ids={3, 4})
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"},
        invalid_block_ids={4, 5},
        failed_recving={"req3"},
    )

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {"req2"}
    assert aggregated.invalid_block_ids == {3, 4, 5}
    assert not aggregated.failed_recving

    output1 = DummyModelRunnerOutput(finished_recving={"req3"})
    output2 = DummyModelRunnerOutput(finished_recving={"req3"})
    aggregated = aggregator.aggregate([output1, output2])
    assert aggregated.kv_connector_output.failed_recving == {"req3"}


def test_aggregate_workers_output_with_expected_finished_count():
    # We create the aggregator expecting to collect from 4 workers
    aggregator = KVOutputAggregator(expected_finished_count=4)
    assert aggregator._expected_finished_count == 4
    # Some request with default expected finished requests
    output1 = DummyModelRunnerOutput(finished_sending={"req1"})
    aggregated = aggregator.aggregate([output1])
    # still expecting to collect from 4 workers
    assert aggregator._send_remaining_count["req1"] == 3
    assert not aggregated.kv_connector_output.finished_sending
    assert not aggregated.kv_connector_output.finished_recving

    # Workers discover and find that in this setup they only need to
    # collect from 2
    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, expected_finished_count=2
    )
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"}, expected_finished_count=2
    )
    output3 = DummyModelRunnerOutput(finished_recving={"req2"})
    # Req2 only needs 2 acks
    aggregated = aggregator.aggregate([output1, output2, output3])
    assert aggregated.kv_connector_output.expected_finished_count == 2

    assert not aggregated.kv_connector_output.finished_sending

    # Req2 is finished
    assert "req2" not in aggregator._recv_remaining_count
    assert aggregated.kv_connector_output.finished_recving == {"req2"}

    # Req1 is still waiting for 2 more acks (expected_finished_count has no effect)
    # NOTE: This is to showcase dynamic update. Workers are responsible for
    # ensuring "req1" termination in this case
    assert aggregator._send_remaining_count["req1"] == 2


# ---- The per-rank report/verdict carry-through fields ----


def make_load_failure_report(request_id: str) -> KVLoadFailureReport:
    """A minimal legal per-rank load-failure report."""
    return KVLoadFailureReport(
        request_id=request_id,
        failed_block_ids=[7],
        token_span=(96, 112),
        rollback_boundary=96,
        observed_failed_layers=[3],
        healthy_above_span=True,
        restored_extent=3072,
    )


def make_repair_result(request_id: str, plan_id: int = 1) -> RepairResultReport:
    """A minimal legal per-rank repair verdict."""
    return RepairResultReport(
        request_id=request_id,
        plan_id=plan_id,
        epoch=7,
        structural_check=True,
        executed_layers=4,
        span=(96, 112),
    )


def test_aggregate_load_failure_reports_from_all_ranks():
    """Every TP rank reports the same fault and the scheduler's consensus
    machinery consumes the per-rank reports: the aggregated output must
    carry ALL ranks' reports, not just the output rank's."""
    aggregator = KVOutputAggregator(expected_finished_count=2)
    report_rank0 = make_load_failure_report("req-1")
    report_rank1 = make_load_failure_report("req-2")
    output1 = DummyModelRunnerOutput(load_failure_reports=[report_rank0])
    output2 = DummyModelRunnerOutput(load_failure_reports=[report_rank1])

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    kv_output = aggregated.kv_connector_output
    assert kv_output.load_failure_reports == [report_rank0, report_rank1]
    assert {r.request_id for r in kv_output.load_failure_reports} == {
        "req-1",
        "req-2",
    }


def test_aggregate_repair_verdicts_across_ranks():
    """The repair verdicts survive aggregation: the per-rank
    repair_results lists are concatenated across ranks."""
    aggregator = KVOutputAggregator(expected_finished_count=2)
    result_rank0 = make_repair_result("req-1", plan_id=1)
    result_rank1a = make_repair_result("req-2", plan_id=2)
    result_rank1b = make_repair_result("req-3", plan_id=3)
    output1 = DummyModelRunnerOutput(repair_results=[result_rank0])
    output2 = DummyModelRunnerOutput(repair_results=[result_rank1a, result_rank1b])

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    assert aggregated.kv_connector_output.repair_results == [
        result_rank0,
        result_rank1a,
        result_rank1b,
    ]


def test_aggregate_with_empty_or_absent_report_fields():
    """Ranks carrying None/empty for the carry-through fields (or
    no kv_connector_output at all) must not break the aggregation."""
    aggregator = KVOutputAggregator(expected_finished_count=2)
    output1 = DummyModelRunnerOutput()
    output2 = DummyModelRunnerOutput()

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    kv_output = aggregated.kv_connector_output
    assert not kv_output.load_failure_reports
    assert not kv_output.repair_results

    # Explicitly empty lists must not break the aggregation either.
    output1 = DummyModelRunnerOutput(load_failure_reports=[], repair_results=[])
    output2 = DummyModelRunnerOutput(load_failure_reports=[], repair_results=[])
    aggregator = KVOutputAggregator(expected_finished_count=2)
    aggregated = aggregator.aggregate([output1, output2])
    assert not aggregated.kv_connector_output.load_failure_reports
    assert not aggregated.kv_connector_output.repair_results

    # A rank whose kv_connector_output is None entirely is skipped, and
    # the surviving rank's report still carries through.
    report = make_load_failure_report("req-1")
    output1 = DummyModelRunnerOutput(load_failure_reports=[report])
    output2 = DummyModelRunnerOutput()
    output2.kv_connector_output = None
    aggregator = KVOutputAggregator(expected_finished_count=2)
    aggregated = aggregator.aggregate([output1, output2])
    assert aggregated.kv_connector_output.load_failure_reports == [report]


def test_aggregate_invalid_block_ids_union_across_ranks():
    """invalid_block_ids remains a set union across ranks (the
    pre-existing behavior the carry-through fix must not regress)."""
    aggregator = KVOutputAggregator(expected_finished_count=2)
    output1 = DummyModelRunnerOutput(invalid_block_ids={1, 2})
    output2 = DummyModelRunnerOutput(invalid_block_ids={2, 3})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated.kv_connector_output.invalid_block_ids == {1, 2, 3}


def test_single_rank_passes_reports_and_verdicts_through():
    """Single-rank compatibility: the lone rank's reports and verdicts
    pass through the aggregation unchanged."""
    aggregator = KVOutputAggregator(expected_finished_count=1)
    report = make_load_failure_report("req-1")
    verdict = make_repair_result("req-1")
    output1 = DummyModelRunnerOutput(
        load_failure_reports=[report], repair_results=[verdict]
    )

    aggregated = aggregator.aggregate([output1])

    assert aggregated is output1
    kv_output = aggregated.kv_connector_output
    assert kv_output.load_failure_reports == [report]
    assert kv_output.repair_results == [verdict]

    # The multi-verdict shape: the list passes through unchanged.
    mixed_verdicts = [
        make_repair_result("req-1", plan_id=2),
        make_repair_result("req-2", plan_id=3),
    ]
    output1 = DummyModelRunnerOutput(repair_results=mixed_verdicts)
    aggregator = KVOutputAggregator(expected_finished_count=1)
    aggregated = aggregator.aggregate([output1])
    assert aggregated.kv_connector_output.repair_results == mixed_verdicts
