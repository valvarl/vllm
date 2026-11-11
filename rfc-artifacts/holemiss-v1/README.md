# Holemiss v1 — RFC reference implementation patches

This directory publishes the exact, verified reference-implementation patch series
for the Holemiss selective KV-cache recovery proposal, discussed in:

- **vLLM core RFC:** [vllm-project/vllm#65172](https://github.com/vllm-project/vllm/issues/65172) — Holemiss: Selective Recovery of Partially Lost LLM Cache State
- **vLLM-Ascend RFC:** [vllm-project/vllm-ascend#20692](https://github.com/vllm-project/vllm-ascend/issues/20692) — the Ascend implementation and backend contracts

## What this is — and what it is not

These are **reference implementation patches for architectural review**, attached to
the RFC discussions as implementation evidence. **This is not a merge-ready upstream
PR series**: field names, ownership boundaries, and lifecycle semantics are under
discussion in the RFCs and are expected to change with maintainer feedback.

## Base commit and application order

The reference implementation is based on vLLM commit
`db9527a4` ("[Misc] Add Transformers version upper bound in requirements (#59614)").
This publication branch is rooted at exactly that commit, so the series applies
directly to this branch's tree, **in this order**:

    git apply rfc-artifacts/holemiss-v1/0001-vllm-sampleless-output-fix.patch
    git apply rfc-artifacts/holemiss-v1/0002-vllm-core-recovery-lifecycle.patch
    git apply rfc-artifacts/holemiss-v1/0003-vllm-worker-repair-seam.patch

Composed, the series modifies 12 tracked files (+1,663/−21) and adds two new test
files (3,238 added lines across 14 files).

## The patches

| # | File | Contents |
|---|---|---|
| 0001 | `0001-vllm-sampleless-output-fix.patch` | **Experimental sampleless-output fallback** in `vllm/v1/engine/core.py` plus engine regression tests: a step whose `execute_model()` already returned a complete output no longer crashes the engine when `sample_tokens()` resolves `None`. Submitted upstream as [PR #66789](https://github.com/vllm-project/vllm/pull/66789) and **closed without merge**. Retained here for historical reproduction purposes; **not proposed for upstream landing**. (This file is the db9527a4-adapted form of the same fix; the PR form was rebased onto current main with its tests in `tests/v1/engine/test_engine_core_async_step.py`.) |
| 0002 | `0002-vllm-core-recovery-lifecycle.patch` | **The core recovery lifecycle** (+1,030/−16 on code files): the opt-in `kv_load_failure_policy="repair"` policy; `KVLoadFailureReport` structured failure evidence with the `get_load_failure_reports()` connector seam; the TP aggregator carry-through of the new connector-output fields; scheduler admission (precondition battery, write-set fence, equivalent-token budget charge, exclusive repair-only step); `RepairWork` on `SchedulerOutput` with the scheduler-authoritative write authorization; verdict consumption (identity/stale/missing/malformed handling) and commit/abort with delayed cache publication — plus the scheduler admission/commit/abort battery and the aggregator carry-through tests. |
| 0003 | `0003-vllm-worker-repair-seam.patch` | **The worker repair seam** (+526/−1): the eager repair-only dispatch producing a well-formed sampleless `ModelRunnerOutput`; the layer-limited native replay over the repair span; the pre-write ownership fence validating the actual slot mapping against the scheduler-authorized write set; per-layer exception containment with the abort re-zero attestation; the fail-closed TP-rank consensus verdict — plus the worker commit-checks test suite. |

**Scope note:** this v1 series contains **no vLLM-Ascend and no Mooncake patches**.
The Ascend-side increments (the AscendStore connector failure-report API, the optional
V1-runner hook, the scheduler-variant zero-skip re-homing) are specified in the
Ascend RFC as follow-up contributions; the validated reference stack ran a pristine
vLLM-Ascend checkout on its default V2 model runner, which inherits the core worker
seam unchanged.

## SHA-256 checksums

```
4255df43eb611843a9cee79d91a05d2055efe3747f3136fbb93ea525a78df18d  0001-vllm-sampleless-output-fix.patch
9a8f1e16d3b0b86a27aa31446fab6f3b52bf0e4741044551fd93cd8a6ccb8943  0002-vllm-core-recovery-lifecycle.patch
dff8ac3c3cd339c4282f5bc519829b0470aadebc750f8b64ccf294c64f30f69f  0003-vllm-worker-repair-seam.patch
```

## Verification

Applied in order to `db9527a4`, the series passes (CPU-compatible suites; the NPU
end-to-end evidence — Qwen3-32B, TP=2, a genuine Mooncake store GET failure,
token-exact continuations, fail-closed negatives — is summarized in the RFC
discussions):

- `tests/v1/core/test_scheduler_repair_budget.py` — 26 passed
- `tests/v1/worker/test_repair_commit_checks.py` — 13 passed
- `tests/v1/kv_connector/unit/test_output_aggregator.py` — 7 passed
