# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Partial-write re-zero coverage (hardening H3, worker side).

The pre-fix abort re-zero covered only FULLY-executed layers -- the
layer that RAISED may have written some KV rows before raising, and its
partial writes survived until the scheduler-side eviction made the
blocks unmatchable. The hardening extends the re-zero depth to the
raising layer (and to the full executed range when a clean run aborts
in the verdict phase), leaving no partially written state readable.

The scheduler-side half of the invariant (eviction unmatchability,
cross-request reuse) lives in
tests/v1/core/test_scheduler_repair_invalidation.py.
"""

import pytest

from tests.v1.worker.test_repair_commit_checks import (
    BS,
    _FakeLayer,
    _FakeModel,
    _multi_work_setup,
)
from tests.v1.worker.test_repair_envelope import _run

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


# ---- H3: the raising layer's partial writes are re-zeroed ----


class _PartialWriteBoomLayer(_FakeLayer):
    """A layer that writes SOME KV rows and then raises -- the realistic
    partial-write shape (a kernel that partially completed). The write
    and the raise are scoped to req-a's row window (its positions start
    below BS)."""

    def __call__(self, positions, hidden_states, residual):
        if int(positions.min()) < BS:
            cache = self.self_attn.attn.kv_cache[0]
            cache[2, :64] = 7.5  # block 2 (req-a's span block), rows 0..63
            raise RuntimeError("injected partial-write layer failure")
        return hidden_states + 1.0, residual


def test_abort_rezero_covers_the_raising_layer_partial_writes():
    """The pre-fix gap: the re-zero covered only FULLY-executed layers,
    so the RAISING layer's partially written rows survived until the
    scheduler-side eviction made the blocks unmatchable. Defense in
    depth: the abort re-zero now covers the raising layer too, leaving
    no partially written state readable even if the eviction path were
    ever relaxed."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    model = _FakeModel(n_layers=4)
    # req-a spans [0, 128) over block 2 with layer_end=2: layer 0 runs
    # clean, layer 1 partially writes then raises. req-b keeps its
    # original layer_end=2 but its rows never trigger the scoped boom.
    works["req-a"].layer_end = 2
    model.model.layers[1] = _PartialWriteBoomLayer("model.layers.1")
    # Pre-fill every layer's caches with a sentinel (AFTER the layer
    # replacement, so the boom layer's fresh caches are covered too) so
    # the test can see exactly which rows the repair touched/re-zeroed.
    for layer in model.model.layers:
        for cache in layer.self_attn.attn.kv_cache:
            cache.fill_(3.25)

    results = _run(model, works, tokens, positions, input_ids, meta)

    ra, rb = results
    assert ra.aborted
    assert ra.executed_layers == 1
    assert not ra.structural_check
    assert "partial-write layer failure" in (ra.error_detail or "")
    # req-b ran clean (its layer 1 call took the benign branch).
    assert not rb.aborted and rb.structural_check

    layer0, layer1, layer2, layer3 = model.model.layers
    # Layer 0 (fully executed by req-a): its span rows were re-zeroed.
    assert layer0.self_attn.attn.kv_cache[0][2].abs().max().item() == 0
    # Layer 1 (the RAISING layer): its partial writes (rows 0..63 of
    # block 2) must be re-zeroed too -- the H3 fix.
    assert layer1.self_attn.attn.kv_cache[0][2].abs().max().item() == 0
    # Layers beyond req-a's layer_end never execute: untouched sentinels.
    assert layer2.self_attn.attn.kv_cache[0][2, 0, 0].item() == 3.25
    assert layer3.self_attn.attn.kv_cache[0][2, 0, 0].item() == 3.25
    # Neighbor state (req-b's block 3, any layer): untouched sentinels.
    for layer in model.model.layers:
        assert layer.self_attn.attn.kv_cache[0][3, 0, 0].item() == 3.25


def test_clean_run_writes_layers_and_leaves_rest_untouched():
    """The GREEN control for the re-zero test: a clean run writes layers
    [0, layer_end) (visible as changed rows) and leaves everything else
    at the sentinel."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    model = _FakeModel(n_layers=4)

    # Make layer 0 write req-a's span block so the changed-rows half of
    # the control is real.
    class _WritingLayer(_FakeLayer):
        def __call__(self, positions, hidden_states, residual):
            if int(positions.min()) < BS:
                self.self_attn.attn.kv_cache[0][2] = 1.5
            return hidden_states + 1.0, residual

    model.model.layers[0] = _WritingLayer("model.layers.0")
    for layer in model.model.layers:
        for cache in layer.self_attn.attn.kv_cache:
            cache.fill_(3.25)

    results = _run(model, works, tokens, positions, input_ids, meta)

    ra, rb = results
    assert not ra.aborted and ra.structural_check
    assert not rb.aborted and rb.structural_check
    # req-a's layer 0 wrote its span block.
    assert model.model.layers[0].self_attn.attn.kv_cache[0][2, 0, 0].item() == 1.5
    # Beyond req-a's layer_end (1): the sentinel survives.
    assert model.model.layers[1].self_attn.attn.kv_cache[0][2, 0, 0].item() == 3.25
    # Neighbor block untouched.
    assert model.model.layers[0].self_attn.attn.kv_cache[0][3, 0, 0].item() == 3.25
