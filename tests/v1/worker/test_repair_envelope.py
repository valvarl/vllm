# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The repair failure envelope (hardening H2 + H3).

The pre-hardening containment covered ONLY the per-layer forward call:
an exception anywhere else on the path between receiving RepairWork and
entering TP consensus (position fetch, metadata access, host-side fence
fetch, embedding, re-zero, the collective itself) escaped the executor
-- on one rank of a TP group that meant the other ranks blocked inside
all_gather forever (an engine hang), and the scheduler never saw a
verdict.

These tests inject a failure at every boundary and pin the envelope
contract:

- every rank-local recoverable exception converges to a FAILED verdict
  (structural_check False, aborted True, error_detail preserving the
  original exception);
- the verdict collective is reached UNCONDITIONALLY, once per work
  item, in work order -- the deadlock regression;
- the abort re-zero covers the RAISING layer's partial writes too (the
  pre-fix code re-zeroed only fully-executed layers);
- neighboring healthy state is untouched.

The multi-rank collective behavior itself (genuine processes, world
sizes 2 and 3) lives in test_repair_tp_consensus.py.
"""

from unittest.mock import patch

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from tests.v1.worker.test_repair_commit_checks import (
    BS,
    _FakeLayer,
    _FakeMeta,
    _FakeModel,
    _multi_work_setup,
    _sub_output,
)
from vllm.forward_context import set_forward_context
from vllm.v1.worker.kv_connector_model_runner_mixin import execute_repair_only

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _fake_tp_ctx:
    """In-process stand-in for a multi-rank world: _tp_world_size > 1 so
    the consensus path executes, with torch.distributed.all_gather
    patched by the caller (the collective is recorded, never a real
    one)."""

    def __enter__(self):
        import vllm.v1.worker.kv_connector_model_runner_mixin as mixin

        self._saved = mixin._tp_world_size
        mixin._tp_world_size = lambda: 2
        return self

    def __exit__(self, *exc):
        import vllm.v1.worker.kv_connector_model_runner_mixin as mixin

        mixin._tp_world_size = self._saved


def _run(model, works, tokens, positions, input_ids, meta, req_ids=None):
    sub = _sub_output(works, tokens)
    with set_forward_context(
        {"model.layers.0": meta}, create_vllm_config(model="unsloth/Llama-3.2-1B")
    ):
        return execute_repair_only(
            model, sub, input_ids, positions, req_ids or list(works)
        )


def _counting_all_gather(collect):
    def fake_all_gather(world, tensor):
        # A faithful stand-in: every remote slot receives a copy of the
        # local summary (the all-ranks-agree outcome). Divergence and
        # failure outcomes are covered by the direct consensus matrix
        # and the distributed tests.
        collect.append(tuple(int(v) for v in tensor.tolist()))
        for remote in world:
            remote.copy_(tensor)
        return None

    return fake_all_gather


def test_positions_fetch_failure_converges_to_failed_verdict():
    """A failure BEFORE the fence (the position fetch): the verdict is a
    failed one and the collective still runs for every work -- no
    escape, no hang."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    model = _FakeModel(n_layers=4)

    class _BoomPositions:
        def __getitem__(self, idx):
            raise RuntimeError("injected positions fetch failure")

    collected = []
    with patch(
        "torch.distributed.all_gather", _counting_all_gather(collected)
    ), _fake_tp_ctx():
        results = _run(
            model, works, tokens, _BoomPositions(), input_ids, meta
        )

    assert all(r.aborted for r in results)
    assert all(not r.structural_check for r in results)
    assert all("positions fetch failure" in (r.error_detail or "") for r in results)
    # One collective per work, unconditional.
    assert len(collected) == 2


def test_fence_host_fetch_failure_converges_to_failed_verdict():
    """A failure DURING the fence (the block-table host fetch): the same
    convergence -- failed verdicts, collectives reached."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    model = _FakeModel(n_layers=4)

    class _BoomTables(list):
        def __getitem__(self, idx):
            raise RuntimeError("injected block-table fetch failure")

    bad_meta = _FakeMeta(
        block_tables=_BoomTables(meta.block_tables),
        slot_mapping=meta.slot_mapping,
    )
    collected = []
    with patch(
        "torch.distributed.all_gather", _counting_all_gather(collected)
    ), _fake_tp_ctx():
        results = _run(model, works, tokens, positions, input_ids, bad_meta)

    assert all(r.aborted for r in results)
    assert all("block-table fetch failure" in (r.error_detail or "") for r in results)
    assert len(collected) == 2


def test_embedding_failure_converges_to_failed_verdict():
    """A failure at the embedding (after the fence passed, before any
    layer): failed verdicts, zero layers executed, collectives reached."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    model = _FakeModel(n_layers=4)

    def boom_embed(input_ids):
        raise RuntimeError("injected embedding failure")

    model.model.embed_input_ids = boom_embed
    collected = []
    with patch(
        "torch.distributed.all_gather", _counting_all_gather(collected)
    ), _fake_tp_ctx():
        results = _run(model, works, tokens, positions, input_ids, meta)

    assert all(r.aborted for r in results)
    assert all(r.executed_layers == 0 for r in results)
    assert all("embedding failure" in (r.error_detail or "") for r in results)
    assert len(collected) == 2


def test_collective_failure_converges_to_failed_verdict():
    """The collective ITSELF failing (a process-group error, or a backend
    timeout where the backend provides one): consensus is unachievable,
    which by fail-closure is a failed verdict -- not an escape."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    model = _FakeModel(n_layers=4)

    def boom_gather(world, tensor):
        raise RuntimeError("injected collective failure")

    with patch("torch.distributed.all_gather", boom_gather), _fake_tp_ctx():
        results = _run(model, works, tokens, positions, input_ids, meta)

    assert all(r.aborted for r in results)
    assert all(not r.structural_check for r in results)
    assert all(r.commit_check["ranks_agree"] is False for r in results)
    assert all("collective failure" in (r.error_detail or "") for r in results)


def test_rezero_failure_converges_to_failed_verdict(monkeypatch):
    """A failure DURING the abort re-zero: the scheduler-side eviction is
    the authoritative invalidation, so the envelope contains the re-zero
    failure too and still returns a verdict."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    # req-b (layer_end=2) hits the raising layer; req-a (layer_end=1)
    # stays clean.
    model = _FakeModel(n_layers=4)

    class _BoomLayer(_FakeLayer):
        def __call__(self, positions, hidden_states, residual):
            if int(positions.min()) >= BS:
                raise RuntimeError("injected layer failure")
            return hidden_states + 1.0, residual

    model.model.layers[1] = _BoomLayer("model.layers.1")

    import vllm.v1.worker.kv_connector_model_runner_mixin as mixin

    rezero_calls = []

    def boom_rezero(layers, meta, lo=0, hi=None):
        rezero_calls.append((len(layers), lo, hi))
        raise RuntimeError("injected re-zero failure")

    monkeypatch.setattr(mixin, "_rezero_repair_span", boom_rezero)
    collected = []
    with patch(
        "torch.distributed.all_gather", _counting_all_gather(collected)
    ), _fake_tp_ctx():
        results = _run(model, works, tokens, positions, input_ids, meta)

    ra, rb = results
    assert not ra.aborted and ra.structural_check
    assert rb.aborted
    assert not rb.structural_check
    # The ORIGINAL exception (the layer failure) stays the diagnostic of
    # record; the re-zero failure is contained beside it (the scheduler-
    # side eviction remains the authoritative invalidation) -- the
    # envelope keeps the first, root-cause exception.
    assert "layer failure" in (rb.error_detail or "")
    # The abort re-zero WAS attempted for the failed work, and its
    # failure did not propagate.
    assert len(rezero_calls) == 1
    assert rezero_calls[0][1:] == (128, 256)
    assert len(collected) == 2


def test_envelope_never_skips_the_collective_multi_work():
    """THE deadlock regression: a step of two works, the FIRST failing
    inside its layer loop. Both works still enter the collective,
    exactly once each, in work order -- a rank that failed locally must
    never strand its TP peers inside all_gather."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    model = _FakeModel(n_layers=4)

    class _BoomFirstWork(_FakeLayer):
        def __call__(self, positions, hidden_states, residual):
            if int(positions.min()) < BS:
                raise RuntimeError("injected first-work failure")
            return hidden_states + 1.0, residual

    model.model.layers[0] = _BoomFirstWork("model.layers.0")

    collected = []
    with patch(
        "torch.distributed.all_gather", _counting_all_gather(collected)
    ), _fake_tp_ctx():
        results = _run(
            model, works, tokens, positions, input_ids, meta, ["req-a", "req-b"]
        )

    ra, rb = results
    assert ra.aborted and not ra.structural_check
    assert not rb.aborted and rb.structural_check
    # One collective per work, in work order, regardless of the failure.
    assert len(collected) == 2
    # The failed work's summary carries its failure; the healthy work's
    # does not.
    assert collected[0][0] == 1  # the aborted flag
    assert collected[1][0] == 0


def test_missing_forward_context_converges_to_failed_verdict():
    """No usable forward context (the earliest possible failure): the
    verdicts still come back failed instead of raising out of the
    executor."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    model = _FakeModel(n_layers=4)
    sub = _sub_output(works, tokens)

    with set_forward_context(
        {}, create_vllm_config(model="unsloth/Llama-3.2-1B")
    ):
        results = execute_repair_only(
            model, sub, input_ids, positions, ["req-a", "req-b"]
        )

    assert all(r.aborted for r in results)
    assert all(not r.structural_check for r in results)
    assert all(r.error_detail is not None for r in results)


def test_error_detail_preserves_original_exception_type_and_message():
    """The envelope preserves the original exception for diagnostics: the
    report carries its type and message."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    model = _FakeModel(n_layers=4)

    class _BoomLayer(_FakeLayer):
        def __call__(self, positions, hidden_states, residual):
            if int(positions.min()) >= BS:
                raise ValueError("very specific kernel failure xyz")
            return hidden_states + 1.0, residual

    model.model.layers[1] = _BoomLayer("model.layers.1")

    ra, rb = _run(model, works, tokens, positions, input_ids, meta)
    assert not ra.aborted
    assert rb.aborted
    assert rb.error_detail is not None
    assert rb.error_detail.startswith("ValueError:")
    assert "very specific kernel failure xyz" in rb.error_detail
