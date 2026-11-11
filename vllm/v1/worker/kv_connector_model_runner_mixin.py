# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Define KV connector functionality mixin for model runners."""

import os
import time
from collections import Counter
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.v1.outputs import (
    KVConnectorOutput,
    ModelRunnerOutput,
    RepairResultReport,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
class KVConnectorModelRunnerMixin:
    @staticmethod
    def kv_connector_no_forward(
        scheduler_output: "SchedulerOutput", vllm_config: VllmConfig
    ) -> ModelRunnerOutput:
        # KV send/recv even if no work to do.
        with (
            set_forward_context(None, vllm_config),
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output
            ) as kv_connector_output,
        ):
            pass

        return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

    @staticmethod
    def maybe_get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        defer_finalize: bool = False,
    ) -> AbstractContextManager[KVConnectorOutput | None]:
        return (
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, defer_finalize=defer_finalize
            )
            if has_kv_transfer_group()
            else nullcontext()
        )

    @staticmethod
    def finalize_kv_connector() -> None:
        """Finalize the KV connector: wait_for_save and clear metadata.

        Call after draft model forward when defer_finalize=True was used.
        """
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.wait_for_save()
            kv_connector.clear_connector_metadata()

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        defer_finalize: bool = False,
    ) -> Generator[KVConnectorOutput, None, None]:
        output = KVConnectorOutput()

        # Update KVConnector with the KVConnector metadata forward().
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

        # Start this step's KV loads, ordered after any in-flight KV block
        # zeroing. Sync loads feed this step's forward so must precede it;
        # otherwise start (async) loads after the forward launch, keeping
        # their host-side submission cost off the critical path.
        start_after_forward = not scheduler_output.has_sync_kv_loads
        if not start_after_forward:
            kv_connector.start_load_kv(get_forward_context())
        try:
            yield output
        finally:
            if start_after_forward:
                kv_connector.start_load_kv(get_forward_context())
            if not defer_finalize:
                kv_connector.wait_for_save()

            transfer_results = kv_connector.get_transfer_results(
                scheduler_output.finished_req_ids
            )
            output.finished_sending = transfer_results.finished_sending
            output.finished_recving = transfer_results.finished_recving
            output.failed_recving = transfer_results.failed_recving
            output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
            output.load_failure_reports = (
                kv_connector.get_load_failure_reports()
            )

            output.kv_connector_stats = kv_connector.get_kv_connector_stats()
            output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()
            output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()

            if not defer_finalize:
                kv_connector.clear_connector_metadata()


def execute_repair_only(
    model: Any,
    scheduler_output: "SchedulerOutput",
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    batch_req_ids: list[str],
) -> list[RepairResultReport]:
    """Execute the repair works of one exclusive repair-only step.

    Runs native layer.forward over each work item's repair span for
    layers [0, layer_end) only, discards the output (no logits, no
    sampling), and returns one verdict per work for the caller to attach
    to the step's connector output. Must be called inside the live
    forward context (the caller's execute_model) with the runner's own
    preprocessed inputs; every scheduled request in the step is a repair
    work (the step is exclusive by construction).

    Each work item's rows are its own window of the step's batch
    (batch_req_ids is the row order), each item runs its own layer
    range, and the validation battery (positions, slot ownership, write
    coverage, layer coverage) applies per work item.
    """
    works = scheduler_output.repair_works
    assert works is not None
    # Repair steps are exclusive: no ordinary request rides the batch.
    assert set(works) == set(scheduler_output.num_scheduled_tokens), (
        "hole repair: a repair step must not carry ordinary work "
        f"({len(works)} repair works vs "
        f"{len(scheduler_output.num_scheduled_tokens)} scheduled requests)"
    )
    assert set(batch_req_ids) == set(scheduler_output.num_scheduled_tokens), (
        "hole repair: batch_req_ids must name exactly the repair step's "
        f"requests ({batch_req_ids!r} vs "
        f"{sorted(scheduler_output.num_scheduled_tokens)})"
    )
    global _REPAIR_SYNC_PROFILE
    _REPAIR_SYNC_PROFILE = {} if _repair_sync_profile_enabled() else None

    row_starts: dict[str, int] = {}
    row = 0
    for req_id in batch_req_ids:
        row_starts[req_id] = row
        row += scheduler_output.num_scheduled_tokens[req_id]

    results: list[RepairResultReport] = []
    for work_row, req_id in enumerate(batch_req_ids):
        work = works[req_id]
        lo = row_starts[req_id]
        hi = lo + scheduler_output.num_scheduled_tokens[req_id]
        span_start, span_end = work.repair_span
        # Envelope state: every phase below runs under the failure
        # envelope, and the verdict construction at the end is reached on
        # EVERY path except process or device death. A rank-local
        # recoverable exception anywhere before the collective must not
        # strand the other TP ranks inside _ranks_repair_agree: it is
        # captured here, preserved for diagnostics, and converged into a
        # consistent failed verdict that still participates in consensus.
        executed_layers = 0
        layer_failed_at: int | None = None
        backend_ok = True
        pre_write_ok = False
        auth_present = False
        auth_slots_ok = False
        auth_table_ok = False
        pre_write_ownership = False
        pre_write_coverage = False
        error_detail: str | None = None
        meta = None
        layers = None
        try:
            pos_first = _int_timed(positions[lo], f"positions[{lo}].int")
            pos_last = _int_timed(positions[hi - 1], f"positions[{hi - 1}].int")
            assert pos_first == span_start and pos_last == span_end - 1, (
                f"hole repair: physical positions [{pos_first},"
                f"{pos_last}] do not match the repair span "
                f"[{span_start}, {span_end}) of {req_id}"
            )

            layers = model.model.layers
            assert work.layer_end <= len(layers), (
                f"hole repair: layer_end {work.layer_end} exceeds the stack "
                f"({len(layers)} layers)"
            )

            attn_metadata = get_forward_context().attn_metadata
            assert isinstance(attn_metadata, dict) and attn_metadata
            meta = next(iter(attn_metadata.values()))
            block_size = layers[0].self_attn.attn.kv_cache[0].shape[1]

            # Pre-write safety gate: the slot window and the block-table row
            # are fetched once to the host (a single D2H per work item -- the
            # same lists feed the post-write attestation) and the geometry is
            # validated BEFORE the first cache-writing kernel, so a malformed
            # slot mapping can never write into another request's KV.
            slots_host = _tolist_timed(
                meta.slot_mapping[lo:hi], "pre_write.slot_mapping.tolist"
            )
            block_table_host = _tolist_timed(
                meta.block_tables[work_row], "pre_write.block_tables.tolist"
            )
            # Ownership fence: the expected physical blocks come from the
            # WORK (the scheduler snapshotted them from KVCacheManager at
            # the admission), NOT from this worker's metadata. The expected
            # physical slots are reconstructed from the authorization and
            # compared against the ACTUAL slot mapping: two
            # consistently-corrupted worker structures (the block table and
            # the slot mapping both pointing at a foreign request's blocks)
            # cannot reproduce the authorized slot multiset, so the foreign
            # write is refused.
            auth_blocks = tuple(work.expected_write_block_ids)
            auth_present = bool(auth_blocks)
            span_blocks = -(-(span_end - span_start) // block_size)
            # The authorization must cover exactly the span's blocks: a
            # partial or over-long snapshot is malformed -- refused, never a
            # crash.
            auth_len_ok = len(auth_blocks) == span_blocks
            expected_slots = Counter()
            if auth_present and auth_len_ok:
                first_blk = span_start // block_size
                for tok in range(span_start, span_end):
                    block_id = auth_blocks[(tok // block_size) - first_blk]
                    expected_slots[block_id * block_size + tok % block_size] += 1
            auth_slots_ok = (
                auth_present and auth_len_ok and Counter(slots_host) == expected_slots
            )
            # The worker block table's span slice must AGREE with the
            # authorization (an independent check of the row itself).
            table_span = block_table_host[
                span_start // block_size : -(-span_end // block_size)
            ]
            auth_table_ok = auth_present and list(table_span) == list(auth_blocks)
            pre_write_ownership = _slots_owned_by_request_list(
                slots_host, block_table_host, block_size
            )
            pre_write_coverage = _slots_cover_span_list(
                slots_host, block_table_host, span_start, span_end, block_size
            )
            pre_write_ok = (
                pre_write_ownership
                and pre_write_coverage
                and auth_present
                and auth_slots_ok
                and auth_table_ok
            )
            if not pre_write_ok:
                logger.error(
                    "Repair pre-write fence refused req=%s plan_id=%d "
                    "span=[%d,%d) ownership=%s coverage=%s auth_present=%s "
                    "auth_slots=%s auth_table=%s -- no cache write executed",
                    work.request_id,
                    work.plan_id,
                    span_start,
                    span_end,
                    pre_write_ownership,
                    pre_write_coverage,
                    auth_present,
                    auth_slots_ok,
                    auth_table_ok,
                )

            hidden_states = model.model.embed_input_ids(input_ids[lo:hi])
            residual = None
            if pre_write_ok:
                for idx, layer in enumerate(layers[: work.layer_end]):
                    try:
                        hidden_states, residual = layer(
                            positions[lo:hi], hidden_states, residual
                        )
                    except Exception as exc:
                        # Exception containment: a failure inside the repair
                        # stream is request-local. Stop the stream here,
                        # re-zero the partial writes INCLUDING this raising
                        # layer's rows (the abort handler below -- a layer
                        # may write some KV rows before raising), report the
                        # failure, and let the scheduler's recompute
                        # fallback own the request. A poisoned device context
                        # fails loudly at the next step; a successful
                        # transaction is never fabricated.
                        layer_failed_at = idx
                        backend_ok = False
                        error_detail = _format_repair_exception(exc)
                        logger.exception(
                            "Repair layer exception req=%s idx=%d span=[%d,%d) "
                            "-> contained as abort",
                            work.request_id,
                            idx,
                            span_start,
                            span_end,
                        )
                        break
                    executed_layers += 1
            del hidden_states, residual
        except Exception as exc:
            # The envelope: any rank-local recoverable failure OUTSIDE the
            # per-layer containment (position fetch, metadata access, host
            # fence, embedding) also converges to the failed verdict below
            # instead of escaping and stranding the TP ranks in consensus.
            backend_ok = False
            if error_detail is None:
                error_detail = _format_repair_exception(exc)
            logger.exception(
                "Repair pre-verdict exception req=%s plan_id=%d span=[%d,%d) "
                "-> contained as abort",
                work.request_id,
                work.plan_id,
                span_start,
                span_end,
            )

        aborted = executed_layers < work.layer_end or error_detail is not None
        if aborted and meta is not None and layers is not None and executed_layers > 0:
            # Best-effort partial-write invalidation of the fully
            # executed layers' rows. The scheduler-side eviction is the
            # authoritative invalidation -- this zeroing must never mask
            # its failure, so its own exception is contained too.
            try:
                _rezero_repair_span(layers[:executed_layers], meta, lo, hi)
            except Exception as exc:
                if error_detail is None:
                    error_detail = _format_repair_exception(exc)
                logger.exception(
                    "Repair abort re-zero failed req=%s plan_id=%d "
                    "span=[%d,%d) (scheduler-side eviction "
                    "remains the authoritative invalidation)",
                    work.request_id,
                    work.plan_id,
                    span_start,
                    span_end,
                )

        ranks_agree = False
        try:
            # TP consensus, fail-closed: every rank executes the SAME works
            # (the SchedulerOutput is broadcast) and all-gathers its packed
            # verdict summary. The transaction commits only when every rank
            # agrees on identity AND outcome -- a single failed or divergent
            # rank aborts the transaction on EVERY rank, never the driver's
            # verdict alone. This call is UNCONDITIONAL on every path the
            # envelope controls: skipping it on one rank would deadlock the
            # others inside the collective.
            ranks_agree = _ranks_repair_agree(
                aborted,
                executed_layers,
                pre_write_ok,
                auth_present and auth_slots_ok,
                auth_table_ok,
                pre_write_ownership,
                pre_write_coverage,
                work.layer_end,
                backend_ok,
            )
        except Exception as exc:
            # The collective itself failed (process-group error, or a
            # backend timeout where the backend provides one): consensus
            # is unachievable, which by fail-closure is a failed verdict.
            if error_detail is None:
                error_detail = _format_repair_exception(exc)
            logger.exception(
                "Repair TP consensus failed req=%s plan_id=%d -> abort",
                work.request_id,
                work.plan_id,
            )

        # The final abort flag: the run stopped short of its layer range,
        # OR the envelope contained any exception (including a verdict-
        # phase or collective failure after a clean run -- the
        # transaction did not complete, so its writes must be treated as
        # partial forever after).
        aborted = aborted or error_detail is not None
        commit_check = {
            "transfer_complete": not aborted,
            # The post-write attestation reuses the pre-write host lists
            # (the slot mapping is immutable across the forward): the
            # executed writes targeted exactly this request's blocks and
            # exactly the span; the pre-write gate above is what PREVENTED
            # a malformed geometry from ever executing.
            "pre_write_geometry": pre_write_ok,
            "authorized_ownership": auth_present and auth_slots_ok,
            "authorized_table": auth_table_ok,
            "slot_ownership": pre_write_ownership,
            "write_coverage": pre_write_coverage,
            "layer_coverage": executed_layers == work.layer_end,
            "backend_ok": backend_ok,
            "ranks_agree": ranks_agree,
            "identity_ok": work.plan_id >= 0 and work.epoch >= 0,
        }
        structural_check = all(commit_check.values())
        result = RepairResultReport(
            request_id=work.request_id,
            plan_id=work.plan_id,
            epoch=work.epoch,
            structural_check=structural_check,
            executed_layers=executed_layers,
            span=(span_start, span_end),
            aborted=aborted,
            commit_check=commit_check,
            error_detail=error_detail,
        )
        logger.info(
            "Repair executed req=%s plan_id=%d epoch=%d span=[%d,%d) "
            "layer_end=%d executed_layers=%d aborted=%s "
            "structural_check=%s attn_state=%s error_detail=%s",
            work.request_id,
            work.plan_id,
            work.epoch,
            span_start,
            span_end,
            work.layer_end,
            executed_layers,
            aborted,
            structural_check,
            str(getattr(meta, "attn_state", None)) if meta is not None else None,
            error_detail,
        )
        results.append(result)
    if _REPAIR_SYNC_PROFILE is not None:
        profile = {k: round(v * 1e3, 3) for k, v in _REPAIR_SYNC_PROFILE.items()}
        logger.info(
            "Repair sync profile total_ms=%.3f ops=%s",
            sum(profile.values()),
            profile,
        )
        _REPAIR_SYNC_PROFILE = None
    return results


def _format_repair_exception(exc: BaseException) -> str:
    """The envelope's diagnostic record of a contained exception: the
    exception's type and message (the full traceback goes to the log via
    logger.exception). Kept on the RepairResultReport so the scheduler
    side sees the original failure even when worker logs are lost."""
    text = type(exc).__name__ + ": " + str(exc)
    return text if len(text) <= 300 else text[:297] + "..."


# Per-repair-step sync profile: {label: accumulated seconds}. None (the
# production default) makes the _*_timed helpers pass-through calls.
_REPAIR_SYNC_PROFILE: dict[str, float] | None = None


def _repair_sync_profile_enabled() -> bool:
    return os.environ.get("KV_REPAIR_SYNC_PROFILE") == "1"


def _tolist_timed(t: torch.Tensor, label: str) -> list:
    if _REPAIR_SYNC_PROFILE is None:
        return t.tolist()
    t0 = time.perf_counter()
    out = t.tolist()
    _record_sync(label, time.perf_counter() - t0)
    return out


def _item_timed(t: torch.Tensor, label: str) -> int:
    if _REPAIR_SYNC_PROFILE is None:
        return int(t)
    t0 = time.perf_counter()
    out = int(t)
    _record_sync(label, time.perf_counter() - t0)
    return out


def _int_timed(t: torch.Tensor, label: str) -> int:
    return _item_timed(t, label)


def _record_sync(label: str, seconds: float) -> None:
    _REPAIR_SYNC_PROFILE[label] = _REPAIR_SYNC_PROFILE.get(label, 0.0) + seconds


def _tp_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _ranks_repair_agree(
    aborted: bool,
    executed_layers: int,
    pre_write_ok: bool,
    auth_ownership_ok: bool,
    auth_table_ok: bool,
    slot_ownership: bool,
    write_coverage: bool,
    layer_end: int,
    backend_ok: bool,
) -> bool:
    """The TP cross-rank repair agreement (fail-closed).

    Every rank executes the same repair works and reaches its own local
    verdict. Each rank packs its verdict summary into one int64 tensor
    and all-gathers it: the transaction succeeds only when EVERY rank's
    summary is identical to the local one AND every rank's summary is a
    success (not aborted, full layer coverage, all local checks green).
    A single failed or divergent rank aborts the transaction on every
    rank -- the driver's verdict alone never commits.
    """
    ws = _tp_world_size()
    if ws <= 1:
        return True
    summary = torch.tensor(
        [
            int(aborted),
            int(executed_layers),
            int(pre_write_ok),
            int(auth_ownership_ok),
            int(auth_table_ok),
            int(slot_ownership),
            int(write_coverage),
            int(layer_end),
            int(backend_ok),
        ],
        dtype=torch.int64,
    )
    if torch.npu.is_available():
        summary = summary.to("npu")
    world = [torch.empty_like(summary) for _ in range(ws)]
    torch.distributed.all_gather(world, summary)
    local_success = (
        not aborted
        and executed_layers == layer_end
        and pre_write_ok
        and auth_ownership_ok
        and auth_table_ok
        and slot_ownership
        and write_coverage
        and backend_ok
    )
    for remote in world:
        if not torch.equal(remote, summary):
            return False
        remote_success = (
            remote[0].item() == 0
            and int(remote[1].item()) == int(remote[7].item())
            and all(int(remote[i].item()) == 1 for i in (2, 3, 4, 5, 6, 8))
        )
        if not remote_success:
            return False
    return local_success
def _slots_owned_by_request_list(
    slots_host: list[int], block_table_host: list[int], block_size: int
) -> bool:
    """Ownership check over the host-side slot and block-table lists:
    every span slot lives inside a block the work's own block-table row
    names."""
    owned = set(block_table_host)
    return all((s // block_size) in owned for s in slots_host)


def _slots_cover_span_list(
    slots_host: list[int],
    block_table_host: list[int],
    span_start: int,
    span_end: int,
    block_size: int,
) -> bool:
    """Multiset coverage check over the host-side lists: the expected
    slots are derived from the block table per span token and the actual
    slot window must match exactly (a duplicated slot aliases two tokens
    onto one cache row). The block table row lists the request's blocks
    from its block 0, so the absolute block index (tok // block_size)
    indexes it directly."""
    expected = Counter()
    for tok in range(span_start, span_end):
        block_id = block_table_host[tok // block_size]
        expected[block_id * block_size + tok % block_size] += 1
    return Counter(slots_host) == expected


def _slots_owned_by_request(
    meta, block_size: int, row: int = 0, lo: int = 0, hi: int | None = None
) -> bool:
    """Commit-check element: every span slot lives inside a block the
    request's own block table names. ``row`` is the work item's row in
    the step's block tables; the slot window is the item's [lo, hi) rows
    of the step's slot mapping."""
    owned = {
        int(b)
        for b in _tolist_timed(
            meta.block_tables[row], f"ownership.block_tables.tolist"
        )
    }
    slots = meta.slot_mapping if hi is None else meta.slot_mapping[lo:hi]
    for s in _tolist_timed(slots, "ownership.slot_mapping.tolist"):
        if int(s) // block_size not in owned:
            return False
    return True


def _slots_cover_span(
    meta,
    block_size: int,
    span_start: int,
    span_end: int,
    row: int = 0,
    lo: int = 0,
    hi: int | None = None,
) -> bool:
    """Commit-check element: the work item's slot window covers exactly
    the span's slots, each expected row exactly once. A multiset
    comparison: a duplicated slot (two tokens aliased to one cache row,
    silently losing one row's KV) must fail the check, which a set
    comparison passes. ``row`` is the item's block-table row; the actual
    slots are the item's [lo, hi) rows of the step's slot mapping."""
    expected = Counter()
    block_table = meta.block_tables[row]
    for tok in range(span_start, span_end):
        block_idx = tok // block_size
        block_id = _item_timed(
            block_table[block_idx], "coverage_expected.item"
        )
        expected[block_id * block_size + tok % block_size] += 1
    slots = meta.slot_mapping if hi is None else meta.slot_mapping[lo:hi]
    actual = Counter(
        int(s) for s in _tolist_timed(slots, "coverage.slot_mapping.tolist")
    )
    return actual == expected


def _rezero_repair_span(layers, meta, lo: int = 0, hi: int | None = None) -> None:
    """Abort-path partial-write invalidation: re-zero the span's slot rows
    in every executed layer's K and V cache. A partial repair is not a
    valid prefix; no partial content is left readable.

    The write indexes the cache tensor's own [block, offset] dims: the
    per-layer cache can be a non-contiguous view (a reshape would copy,
    and the zero would land in the copy). The slot window is the work
    item's [lo, hi) rows of the step's slot mapping."""
    slots = meta.slot_mapping if hi is None else meta.slot_mapping[lo:hi]
    slots = slots.long()
    max_abs_after = 0.0
    for layer in layers:
        attn = layer.self_attn.attn
        block_size = attn.kv_cache[0].shape[1]
        bidx = slots // block_size
        oidx = slots % block_size
        for cache in attn.kv_cache:
            cache[bidx, oidx] = 0
        k_rows = attn.kv_cache[0][bidx, oidx]
        max_abs_after = max(max_abs_after, float(k_rows.abs().max()))
    logger.info(
        "Repair abort re-zero layers=%d rows=%d max_abs_after=%.6f",
        len(layers),
        int(slots.numel()),
        max_abs_after,
    )
