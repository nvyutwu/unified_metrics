# SGLang vs vLLM Metrics Comparison

> Based on source code analysis of both `feat/unified-metrics` branches.
> Consolidated from v1 (architecture-focused) and v2 (unified-metrics branch perspective).

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Unified Metrics (Present in Both)](#2-unified-metrics-present-in-both)
3. [Metrics Unique to SGLang](#3-metrics-unique-to-sglang)
4. [Metrics Unique to vLLM](#4-metrics-unique-to-vllm)
5. [Detailed Computation Differences](#5-detailed-computation-differences)
6. [Label Differences](#6-label-differences)
7. [Histogram Bucket Differences](#7-histogram-bucket-differences)
8. [HTTP-Level Metrics](#8-http-level-metrics)
9. [Request Type Counters](#9-request-type-counters)
10. [Config Info Metrics](#10-config-info-metrics)
11. [Architecture Differences](#11-architecture-differences)
12. [SGLang Streaming Requirement](#12-sglang-streaming-requirement)
13. [Remaining Gaps & Recommendations](#13-remaining-gaps--recommendations)
14. [Appendix: File Locations](#14-appendix-file-locations)
15. [Bridge-Side Metric Renaming](#15-bridge-side-metric-renaming)

---

## 1. Executive Summary

| Aspect | SGLang | vLLM |
|--------|--------|------|
| Metric Prefix | `sglang:` | `vllm:` |
| Primary Metrics File | `python/sglang/srt/metrics/collector.py` | `vllm/v1/metrics/loggers.py` |
| Stats Tracking | `tokenizer_manager.py`, `scheduler_metrics_mixin.py` | `vllm/v1/metrics/stats.py` |
| Enable Flag | `--enable-metrics` | Enabled by default |
| Total Metrics | ~100 | ~58 |
| Counter Naming | Uses `_total` suffix (OpenMetrics standard) | No `_total` suffix |
| Multiprocess Mode | Prometheus multiprocess via `PROMETHEUS_MULTIPROC_DIR` | Single-process (frontend) |

---

## 2. Unified Metrics (Present in Both)

These metrics exist in both frameworks on the `feat/unified-metrics` branches with compatible names and semantics.

### Request Lifecycle & Timestamps

Both engines track the same request lifecycle stages, though they use different variable names and clock types:

```
arrival / received        ← HTTP request hits the API server
    │                        vLLM: arrival_time (time.time())
    │                        SGLang: received_time / created_time (time.time())
    ▼  [tokenization, input preprocessing, IPC to engine core]
queued                    ← request enters scheduler's waiting queue
    │                        vLLM: queued_ts (time.monotonic())
    │                        SGLang: wait_queue_entry_time (time.perf_counter())
    ▼  [waiting for KV cache / compute budget]
scheduled                 ← scheduler picks request into a running batch
    │                        vLLM: scheduled_ts (time.monotonic())
    │                        SGLang: forward_entry_time (time.perf_counter())
    ▼  [GPU forward pass - prefill]
first_token               ← first output token produced
    │                        vLLM: first_token_ts (time.monotonic())
    │                        SGLang: first_token_time (time.time())
    ▼  [decode iterations]
last_token / finished     ← final output token produced
                             vLLM: last_token_ts (time.monotonic()) / iteration_timestamp (time.time())
                             SGLang: finished_time (time.time())
```

**Clock differences:** vLLM uses `time.monotonic()` for engine-core timestamps (queued → last_token) and `time.time()` for frontend timestamps (arrival, iteration). SGLang uses `time.perf_counter()` for scheduler timestamps and `time.time()` for API-level timestamps. Timestamps on different clocks cannot be subtracted from each other, so each metric uses a consistent clock pair.

**Key distinction:** `arrival_time` ≠ `queued_ts`. The gap between them includes tokenization, input preprocessing, and IPC transfer from frontend to engine core. The `e2e_request_latency` uses arrival/finished (wall-clock), while `request_queue_time` uses queued/scheduled (monotonic).

### Core Latency Histograms

| Metric Name | SGLang | vLLM | Computation Match? |
|-------------|--------|------|-------------------|
| `e2e_request_latency_seconds` | Histogram | Histogram | **YES** — both measure wall-clock from request arrival to completion. SGLang: `finished_time - created_time`. vLLM: `iteration_ts - arrival_time`. |
| `time_to_first_token_seconds` | Histogram | Histogram | **YES** — SGLang: `first_token_time - created_time`. vLLM: `iteration_ts - arrival_time` (on first token iteration). Both use wall-clock, both include queue wait. |
| `inter_token_latency_seconds` | Histogram | Histogram | **SIMILAR** (SGLang normalizes by num_new_tokens; vLLM measures individual gaps) |
| `request_queue_time_seconds` | Histogram | Histogram | **YES** — both: `scheduled - queued` (monotonic clocks). SGLang: `forward_entry_time - wait_queue_entry_time`. vLLM: `scheduled_ts - queued_ts`. |
| `request_inference_time_seconds` | Histogram | Histogram | **YES** — both: `last_token - scheduled` (monotonic). Excludes queue wait. |
| `request_prefill_time_seconds` | Histogram | Histogram | **YES** — both: `first_token - scheduled` (monotonic). |
| `request_decode_time_seconds` | Histogram | Histogram | **YES** — both: `last_token - first_token` (monotonic). |
| `request_time_per_output_token_seconds` | Histogram | Histogram | **YES** (both: decode_time / (gen_tokens - 1)) |

### Token Counters

| Metric Name | SGLang | vLLM | Match? |
|-------------|--------|------|--------|
| `prompt_tokens_total` (SGLang) / `prompt_tokens` (vLLM) | Counter | Counter | **YES** (name differs by `_total` suffix) |
| `generation_tokens_total` (SGLang) / `generation_tokens` (vLLM) | Counter | Counter | **YES** (name differs by `_total` suffix) |

### Token Histograms

| Metric Name | SGLang | vLLM | Match? |
|-------------|--------|------|--------|
| `request_prompt_tokens` | Histogram | Histogram | **YES** (different buckets) |
| `request_generation_tokens` | Histogram | Histogram | **YES** (different buckets) |

### Request Counters

| Metric Name | SGLang | vLLM | Match? |
|-------------|--------|------|--------|
| `request_success_total` (SGLang) / `request_success` (vLLM) | Counter w/ `finished_reason` | Counter w/ `finished_reason` | **YES** |
| `num_retracted_requests_total` | Counter | Counter | **YES** (unified name for preemption/retraction count) |

### Queue/Load Gauges

| Metric Name | SGLang | vLLM | Match? |
|-------------|--------|------|--------|
| `num_requests_running` | Gauge | Gauge | **YES** |
| `num_requests_waiting` | Gauge | Gauge | **YES** |
| `kv_cache_usage_perc` | Gauge | Gauge | **YES** (both 0-1 ratio) |
| `gen_throughput` | Gauge | Gauge | **YES** (tokens/s, both ~5s update interval) |

### Startup/Config

| Metric Name | SGLang | vLLM | Match? |
|-------------|--------|------|--------|
| `engine_startup_time` | Gauge | Gauge | **YES** |
| `engine_load_weights_time` | Gauge | Gauge | **YES** |
| `model_config_info` | Gauge (info) | Gauge (info) | **SIMILAR** (different label sets; see §10) |
| `parallel_config_info` | Gauge (info) | Gauge (info) | **SIMILAR** (different label sets; see §10) |
| `speculative_config_info` | Gauge (info) | Gauge (info) | **SIMILAR** (different label sets; see §10) |

### Cache Metrics

| Metric Name | SGLang | vLLM | Match? |
|-------------|--------|------|--------|
| `cache_hit_rate` (SGLang) / `prefix_cache_queries` + `prefix_cache_hits` (vLLM) | Gauge (ratio) | Two Counters | **DIFFERENT approach**: SGLang exposes pre-computed ratio; vLLM exposes raw counters for PromQL |
| `cached_tokens_total` (SGLang) / `prompt_tokens_cached` (vLLM) | Counter | Counter | **YES** — both count cached prompt tokens. vLLM aggregates local + external cached tokens |
| `mm_cache_queries` | Counter | Counter | **YES** (multi-modal cache queries) |
| `mm_cache_hits` | Counter | Counter | **YES** (multi-modal cache hits) |

### Speculative Decoding

| Metric Name | SGLang | vLLM | Match? |
|-------------|--------|------|--------|
| `spec_accept_rate` | Gauge | Gauge | **YES** (both: accepted/draft ratio, per-interval) |
| `spec_accept_length` | Gauge | Gauge | **YES** (both: 1 + accepted/drafts, per-interval) |
| `spec_decode_num_drafts` | Counter | Counter | **YES** (draft attempt count) |
| `spec_decode_num_accepted_tokens_per_pos` | Counter | Counter | **YES** (accepted tokens per draft position) |

### LoRA

| Metric Name | SGLang | vLLM | Match? |
|-------------|--------|------|--------|
| `lora_pool_utilization` | Gauge (0-1) | Gauge (0-1) | **YES** |

### Request Parameters

| Metric Name | SGLang | vLLM | Match? |
|-------------|--------|------|--------|
| `request_params_max_tokens` | Histogram | Histogram | **YES** (max_tokens parameter distribution) |

---

## 3. Metrics Unique to SGLang

### Latency & Throughput

| Metric Name | Type | Description |
|-------------|------|-------------|
| `per_stage_req_latency_seconds` | Histogram | Per-stage breakdown (label: `stage`) |
| `gpu_execution_seconds_total` | Counter | GPU execution time by forward mode |
| `func_latency_seconds` | Histogram | Function-level latency profiling |

### Grammar/Structured Output

| Metric Name | Type | Description |
|-------------|------|-------------|
| `grammar_compilation_time_seconds` | Histogram | Grammar compilation time |
| `grammar_schema_count` | Histogram | Number of schemas in grammar |
| `grammar_ebnf_size` | Histogram | EBNF grammar size (bytes) |
| `grammar_tree_traversal_time_avg` | Histogram | Avg grammar tree traversal time |
| `grammar_tree_traversal_time_max` | Histogram | Max grammar tree traversal time |
| `num_grammar_cache_hit_total` | Counter | Grammar cache hits |
| `num_grammar_aborted_total` | Counter | Grammar aborted requests |
| `num_grammar_timeout_total` | Counter | Grammar timeouts |
| `num_grammar_total` | Counter | Total grammar requests |
| `num_grammar_queue_reqs` | Gauge | Grammar queue depth |
| `num_so_requests_total` | Counter | Structured output request count |

### PD Disaggregation (Prefill-Decode Separation)

Both frameworks support PD disaggregation but use **fundamentally different architectures**, which is reflected in their metrics.

**SGLang** uses an explicit **4-queue pipeline** in the scheduler:
```
Prefill: bootstrap_queue → waiting_queue → inflight_queue → done
Decode:  prealloc_queue → transfer_queue → waiting_queue → done
```
Metrics are generic (scheduler-level), independent of the KV transfer backend.

**vLLM** uses **connector hooks** integrated into the forward pass:
```
Scheduler → start_load_kv() → wait_for_layer_load() → save_kv_layer() → wait_for_save()
```
Metrics are per-connector (each connector defines its own via `build_prom_metrics()`). The NIXL connector is the most instrumented.

#### Queue Depth Metrics (SGLang-only, no vLLM equivalent)

These metrics have no vLLM equivalent because vLLM's disaggregation does not use explicit queues.

| Metric Name | Type | Description |
|-------------|------|-------------|
| `num_prefill_prealloc_queue_reqs` | Gauge | Prefill bootstrap queue depth |
| `num_prefill_inflight_queue_reqs` | Gauge | Prefill inflight (transfer in progress) queue depth |
| `num_decode_prealloc_queue_reqs` | Gauge | Decode preallocation queue depth |
| `num_decode_transfer_queue_reqs` | Gauge | Decode transfer (receiving KV) queue depth |

#### KV Transfer Performance (overlap exists)

| Concept | SGLang Metric | Type | vLLM Metric (NIXL) | Type | Notes |
|---------|---------------|------|---------------------|------|-------|
| Transfer speed | `kv_transfer_speed_gb_s` | Gauge | `kv_transfer_speed_gb_s` | Gauge | **Exact match** (vLLM explicitly SGLang-compatible) |
| Transfer latency | `kv_transfer_latency_ms` | Gauge | `nixl_xfer_time_seconds` | Histogram | vLLM more detailed (histogram vs gauge), different unit |
| Bytes transferred | `kv_transfer_total_mb` | Gauge | `nixl_bytes_transferred` | Histogram | vLLM more detailed (histogram vs gauge), raw bytes |
| Bootstrap/post time | `kv_transfer_bootstrap_ms` | Gauge | `nixl_post_time_seconds` | Histogram | Similar concept, different granularity |
| Alloc wait time | `kv_transfer_alloc_ms` | Gauge | — | — | No vLLM equivalent |

#### Failure & Retry Counters

| Concept | SGLang Metric | vLLM Metric (NIXL) | Notes |
|---------|---------------|---------------------|-------|
| Transfer failures | `num_transfer_failed_reqs_total` | `nixl_num_failed_transfers` | Similar semantics |
| Bootstrap failures | `num_bootstrap_failed_reqs_total` | — | No vLLM equivalent (no separate bootstrap phase) |
| Prefill retries | `num_prefill_retries_total` | — | No vLLM equivalent (uses recompute/fail policy instead) |
| Failed notifications | — | `nixl_num_failed_notifications` | vLLM/NIXL-specific |
| KV expiration | — | `nixl_num_kv_expired_reqs` | vLLM/NIXL-specific |

#### vLLM NIXL-Specific Metrics (no SGLang equivalent)

| Metric Name | Type | Description |
|-------------|------|-------------|
| `nixl_post_time_seconds` | Histogram | Post-transfer processing time |
| `nixl_num_descriptors` | Histogram | Number of descriptors per transfer |
| `nixl_num_failed_notifications` | Counter | Failed NIXL notifications |
| `nixl_num_kv_expired_reqs` | Counter | Requests with expired KV (tracked on P instance) |

#### Architecture Impact on Unification

The queue depth gauges cannot be added to vLLM without redesigning its disaggregation architecture. The transfer performance and failure metrics have partial overlap — `kv_transfer_speed_gb_s` is already unified, but the remaining vLLM metrics are NIXL-specific (prefixed `nixl_*`) rather than generic. Adding generic transfer gauges (`kv_transfer_latency_ms`, `kv_transfer_total_mb`) to vLLM would be moderate effort — the data exists in `NixlKVConnectorStats` but would need to be surfaced as connector-agnostic metrics.

### Retraction Detail

| Metric Name | Type | SGLang | vLLM | Notes |
|-------------|------|--------|------|-------|
| `num_retracted_requests_total` | Counter | **YES** | **YES** | Unified name (was `num_preemptions_total` in vLLM) |
| `num_retractions` | Histogram | **YES** | **YES** | Per-request preemption count distribution, same buckets |
| `num_retracted_input_tokens_total` | Counter | **YES** | No | Input tokens wasted by retraction |
| `num_retracted_output_tokens_total` | Counter | **YES** | No | Output tokens wasted by retraction |
| `num_retracted_reqs` | Gauge | **YES** | No | Current retracted count (snapshot) |

### Cache Eviction

| Metric Name | Type | Description |
|-------------|------|-------------|
| `eviction_duration_seconds` | Histogram | GPU to CPU eviction time |
| `evicted_tokens_total` | Counter | Tokens evicted |
| `load_back_duration_seconds` | Histogram | CPU to GPU load time |
| `load_back_tokens_total` | Counter | Tokens loaded back |

### Routing Keys (multi-tenant)

| Metric Name | Type | Description |
|-------------|------|-------------|
| `routing_key_running_req_count` | GaugeHistogram | Running requests by routing key |
| `routing_key_all_req_count` | GaugeHistogram | All requests by routing key |
| `num_unique_running_routing_keys` | Gauge | Unique routing key count |

### CUDA Graph

| Metric Name | Type | Description |
|-------------|------|-------------|
| `cuda_graph_passes_total` | Counter | CUDA graph forward passes by mode |
| `is_cuda_graph` | Gauge | Whether using CUDA graph |

### MoE (Mixture of Experts)

| Metric Name | Type | Description |
|-------------|------|-------------|
| `eplb_balancedness` | Summary | Expert load balancing metric |
| `eplb_gpu_physical_count` | Histogram | Physical expert selection counts per layer/GPU |

### Prefill Delayer

| Metric Name | Type | Description |
|-------------|------|-------------|
| `prefill_delayer_wait_forward_passes` | Histogram | Forward passes delayed |
| `prefill_delayer_wait_seconds` | Histogram | Wait time (seconds) |
| `prefill_delayer_outcomes_total` | Counter | Delayer decisions by outcome |

### Data Parallel Cooperation

| Metric Name | Type | Description |
|-------------|------|-------------|
| `realtime_tokens_total` | Counter | Tokens by mode (prefill_compute/cache/decode) |
| `dp_cooperation_realtime_tokens_total` | Counter | Tokens with DP cooperation info |
| `dp_cooperation_gpu_execution_seconds_total` | Counter | GPU exec with DP info |

### Storage

| Metric Name | Type | Description |
|-------------|------|-------------|
| `prefetched_tokens_total` | Counter | Prefetched tokens |
| `backuped_tokens_total` | Counter | Backed-up tokens |
| `prefetch_pgs` | Histogram | Prefetch pages per batch |
| `backup_pgs` | Histogram | Backup pages per batch |
| `prefetch_bandwidth` | Histogram | Prefetch bandwidth (GB/s) |
| `backup_bandwidth` | Histogram | Backup bandwidth (GB/s) |

### Other SGLang-Only

| Metric Name | Type | Description |
|-------------|------|-------------|
| `num_used_tokens` | Gauge | Raw token count in KV cache |
| `max_total_num_tokens` | Gauge | Max KV cache capacity |
| `pending_prealloc_token_usage` | Gauge | Pending prealloc tokens |
| `swa_token_usage` | Gauge | Sliding window attention usage |
| `mamba_usage` | Gauge | Mamba SSM layer usage |
| `decode_sum_seq_lens` | Gauge | Sum of decode sequence lengths |
| `num_running_reqs_offline_batch` | Gauge | Offline batch requests |
| `num_paused_reqs` | Gauge | Paused requests |
| `utilization` | Gauge | Composite utilization metric |
| `max_running_requests_under_SLO` | Gauge | Max requests within SLO |
| `new_token_ratio` | Gauge | New vs cached token ratio |
| `num_requests_total` | Counter | Total finished requests (no label) |
| `num_aborted_requests_total` | Counter | Aborted requests |
| `lora_pool_slots_used` | Gauge | LoRA slots in use |
| `lora_pool_slots_total` | Gauge | Total LoRA slots |
| `startup_latency_breakdown_seconds_max` | Gauge | Startup breakdown by context |
| `process_cpu_seconds_total` | Counter | CPU time consumed |

---

## 4. Metrics Unique to vLLM

### Iteration-Level

| Metric Name | Type | Description |
|-------------|------|-------------|
| `iteration_tokens_total` | Histogram | Total tokens per engine step |

### Request Parameters (remaining vLLM-only)

| Metric Name | Type | Description |
|-------------|------|-------------|
| `request_params_n` | Histogram | Request "n" parameter distribution |
| `request_max_num_generation_tokens` | Histogram | Max generation tokens per request |
| `request_prefill_kv_computed_tokens` | Histogram | New KV tokens computed (excl. cached) |

### KV Cache Block Residency (optional)

| Metric Name | Type | Description |
|-------------|------|-------------|
| `kv_block_lifetime_seconds` | Histogram | Block lifetime: allocation to eviction |
| `kv_block_idle_before_evict_seconds` | Histogram | Idle time before eviction |
| `kv_block_reuse_gap_seconds` | Histogram | Time between consecutive block accesses |

### Prompt Token Breakdown (new in v2-0.16.0)

| Metric Name | Type | Extra Labels | Description |
|-------------|------|-------------|-------------|
| `prompt_tokens_by_source` | Counter | `source` (local_compute, local_cache_hit, external_kv_transfer) | Number of prompt tokens broken down by source |
| `prompt_tokens_cached` | Counter | — | Number of cached prompt tokens (local + external). Similar to SGLang's `cached_tokens_total` |
| `prompt_tokens_recomputed` | Counter | — | Number of cached tokens recomputed for forward pass (e.g., last token recompute when entire prompt is cached) |

### Cache Counters (more granular)

| Metric Name | Type | Description |
|-------------|------|-------------|
| `prefix_cache_queries` | Counter | Prefix cache query count (tokens) |
| `external_prefix_cache_queries` | Counter | Cross-instance cache queries |
| `external_prefix_cache_hits` | Counter | Cross-instance cache hits |

### Speculative Decoding Counters (more granular)

| Metric Name | Type | Description |
|-------------|------|-------------|
| `spec_decode_num_draft_tokens` | Counter | Draft token count |
| `spec_decode_num_accepted_tokens` | Counter | Accepted token count |

### Engine State

| Metric Name | Type | Description |
|-------------|------|-------------|
| `engine_sleep_state` | Gauge | Engine sleep/wake state tracking |
| `corrupted_requests` | Counter | Requests with NaN logits (opt-in) |

### LoRA Info

| Metric Name | Type | Description |
|-------------|------|-------------|
| `lora_requests_info` | Gauge | LoRA request info with adapter names |

### PD Disaggregation — NIXL Connector (see also §3 for cross-framework comparison)

| Metric Name | Type | Description |
|-------------|------|-------------|
| `nixl_xfer_time_seconds` | Histogram | Transfer duration per NIXL KV cache transfer |
| `nixl_post_time_seconds` | Histogram | Post-transfer processing time |
| `nixl_bytes_transferred` | Histogram | Bytes transferred per transfer |
| `nixl_num_descriptors` | Histogram | Number of descriptors per transfer |
| `nixl_num_failed_transfers` | Counter | Failed NIXL transfers |
| `nixl_num_failed_notifications` | Counter | Failed NIXL notifications |
| `nixl_num_kv_expired_reqs` | Counter | Requests with expired KV (P instance) |
| `kv_transfer_speed_gb_s` | Gauge | KV transfer speed in GB/s (SGLang-compatible) |

### Config

| Metric Name | Type | Description |
|-------------|------|-------------|
| `cache_config_info` | Gauge | Cache configuration info |

### Deprecated

| Metric Name | Type | Description |
|-------------|------|-------------|
| `time_per_output_token_seconds` | Histogram | Deprecated alias for ITL (hidden by default) |

---

## 5. Detailed Computation Differences

All core latency metrics are now **aligned** across both frameworks on the `feat/unified-metrics` branches. They use the same formula (different variable names for the same timestamps).

### Aligned Latency Metrics

| Metric | Unified Formula | SGLang Variables | vLLM Variables |
|--------|----------------|-----------------|----------------|
| `e2e_request_latency_seconds` | `end_time - arrival_time` | `finished_time - created_time` | `iteration_timestamp - arrival_time` |
| `time_to_first_token_seconds` | `first_token_time - arrival_time` | `first_token_time - created_time` | `iteration_timestamp - arrival_time` (at first token) |
| `request_queue_time_seconds` | `scheduled_time - queued_time` | `forward_entry_time - wait_queue_entry_time` | `scheduled_ts - queued_ts` |
| `request_prefill_time_seconds` | `first_token_time - scheduled_time` | `first_token_time_perf - forward_entry` | `first_token_ts - scheduled_ts` |
| `request_decode_time_seconds` | `end_time - first_token_time` | `finished_time_perf - first_token_time_perf` | `last_token_ts - first_token_ts` |
| `request_inference_time_seconds` | `end_time - scheduled_time` | `finished_time_perf - forward_entry` | `last_token_ts - scheduled_ts` |
| `request_time_per_output_token_seconds` | `decode_time / (gen_tokens - 1)` | Same | Same |

### Inter-Token Latency (remaining difference)

| Framework | Computation | Notes |
|-----------|-------------|-------|
| **SGLang** | `(new_time - last_time) / num_new_tokens` | Normalizes when multiple tokens arrive together |
| **vLLM** | `engine_core_timestamp - last_token_ts` | Measures individual inter-token gaps |

This is the one remaining computation difference. SGLang normalizes by `num_new_tokens` when batched token delivery occurs; vLLM measures each gap individually. Both now also provide the per-request mean TPOT via `request_time_per_output_token_seconds` which is computed identically.

### Cache Hit Rate (different approach, by design)

| Framework | Method | Notes |
|-----------|--------|-------|
| **SGLang** | `cache_hit_rate` gauge (pre-computed ratio) | Simple to read, but loses raw numbers |
| **vLLM** | `prefix_cache_queries` + `prefix_cache_hits` (two counters) | Requires PromQL to compute ratio, but more flexible |

### Throughput (`gen_throughput`)

| Framework | Method | Notes |
|-----------|--------|-------|
| **SGLang** | `num_generated_tokens / gap_latency` | Computed in scheduler process |
| **vLLM** | `accumulated_tokens / delta_time` (every ~5s) | Computed in frontend process |

Both compute tokens/second over a sliding window. The implementation differs due to architecture (SGLang: multi-process, vLLM: single frontend process) but the semantics are the same.

---

## 6. Label Differences

| Aspect | SGLang | vLLM |
|--------|--------|------|
| **Primary labels** | `model_name`, `engine_type`, `tp_rank`, `pp_rank`, `moe_ep_rank`, `experiment_group` | `model_name`, `engine` |
| **Parallelism info** | In metric labels (tp_rank, pp_rank, etc.) | In `parallel_config_info` metric |
| **Experiment tracking** | `experiment_group` label on all metrics | Not present |
| **Counter naming** | Uses `_total` suffix (e.g., `prompt_tokens_total`) | No `_total` suffix (e.g., `prompt_tokens`) |

**Recommendation**: The `_total` suffix is the OpenMetrics standard for counters. Both approaches are valid. Keep framework-specific labels but ensure the core `model_name` label is consistent.

---

## 7. Histogram Bucket Differences

### Time to First Token (TTFT) Buckets

| Aspect | SGLang | vLLM |
|--------|--------|------|
| **Buckets** | `[0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400]` | `[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0]` |
| **Smallest bucket** | 100ms | **1ms** |
| **Largest bucket** | 400s | **2560s** |
| **Sub-100ms granularity** | None | 1ms, 5ms, 10ms, 20ms, 40ms, 60ms, 80ms |
| **Number of buckets** | 18 | 22 |
| **Configurable** | Yes (`--bucket-time-to-first-token`) | No (hardcoded) |

**Note**: vLLM has much finer granularity for fast responses (1ms-100ms range), important for low-latency scenarios.

### Inter-Token Latency (TPOT) Buckets

| Aspect | SGLang | vLLM |
|--------|--------|------|
| **Buckets** | `[0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.060, 0.080, 0.100, 0.200, 0.400, 0.600, 0.800, 1.000, 2.000, 4.000, 6.000, 8.000]` | `[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0]` |
| **Smallest bucket** | **2ms** | 10ms |
| **Largest bucket** | 8s | **80s** |
| **Sub-10ms granularity** | 2ms, 4ms, 6ms, 8ms | None |
| **Number of buckets** | 23 | 19 |
| **Configurable** | Yes (`--bucket-inter-token-latency`) | No (hardcoded) |

**Note**: SGLang has finer granularity for fast token generation (2ms-10ms range), better for high-throughput scenarios.

### E2E Request Latency Buckets

| Aspect | SGLang | vLLM |
|--------|--------|------|
| **Buckets** | `[0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400, 600, 1200, 1800, 2400]` | `[0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0]` |
| **Smallest bucket** | **100ms** | 300ms |
| **Largest bucket** | 2400s (40min) | **7680s (2.1hr)** |
| **Number of buckets** | 22 | 21 |
| **Configurable** | Yes (`--bucket-e2e-request-latency`) | No (hardcoded) |

### Request Phase Timing (inference/prefill/decode)

| Aspect | SGLang | vLLM |
|--------|--------|------|
| **Buckets** | `[0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0, 7680.0]` | Same (identical) |
| **Match** | **YES** | **YES** |

### Request Token Histograms

| Aspect | SGLang | vLLM |
|--------|--------|------|
| **Buckets** | `[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]` | Dynamic `build_1_2_5_buckets(max_model_len)` |
| **Difference** | Static | Model-aware (adapts to max_model_len) |

### Queue Time

| Aspect | SGLang | vLLM |
|--------|--------|------|
| **Buckets** | `[0.0, 0.1, 0.2, 0.5, 1, 2, ..., 3000]` (36 buckets, very fine) | Same as E2E latency (21 buckets) |
| **Difference** | Dedicated fine-grained buckets | Reuses E2E buckets |

### Summary: Bucket Granularity

| Metric | Better Fine-Grained Coverage | Notes |
|--------|------------------------------|-------|
| **TTFT** | **vLLM** (1ms min vs 100ms) | vLLM better for low-latency measurement |
| **Inter-Token Latency** | **SGLang** (2ms min vs 10ms) | SGLang better for high-throughput |
| **E2E Latency** | **SGLang** (100ms min vs 300ms) | SGLang better for fast requests |
| **Phase Timing** | **Identical** | Both use same buckets |
| **Configurability** | **SGLang** | SGLang allows custom buckets via CLI args |

---

## 8. HTTP-Level Metrics

Both frameworks expose HTTP server-level metrics, but via **different mechanisms**:

- **SGLang**: Custom FastAPI middleware in `python/sglang/srt/utils/common.py` (`add_prometheus_track_response_middleware`)
- **vLLM**: `prometheus_fastapi_instrumentator` library in `vllm/entrypoints/serve/instrumentator/metrics.py`

### Comparison

| Metric | SGLang | vLLM |
|--------|--------|------|
| **Request counter** | `http_requests_total` `[endpoint, method]` | `http_requests_total` `[method, status, handler]` |
| **Response / status tracking** | `http_responses_total` `[endpoint, status_code, method]` (separate counter) | Included in `http_requests_total` via `status` label |
| **Duration** | `http_request_duration_seconds` Histogram `[endpoint, method]` | `http_request_duration_seconds` Histogram `[method, handler]` |
| **In-flight requests** | `http_requests_active` Gauge `[endpoint, method]` | `http_requests_in_progress` Gauge |
| **Request/response size** | Not tracked | `http_request_size_bytes`, `http_response_size_bytes` (Summary) |
| **Excluded endpoints** | None (all paths tracked) | `/metrics`, `/health`, `/load`, `/ping`, `/version`, `/server_info` |

### Status Code Breakdown (2xx/4xx/5xx)

Both frameworks provide HTTP status code breakdown:
- **SGLang**: Via `http_responses_total` with `status_code` label (e.g., `"200"`, `"400"`, `"500"`)
- **vLLM**: Via `http_requests_total` with `status` label (provided by `prometheus_fastapi_instrumentator`)

### SGLang HTTP Duration Buckets

`[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0, 120.0]`

vLLM uses the `prometheus_fastapi_instrumentator` default buckets.

### SGLang-Only: Routing Key Tracking

SGLang also tracks `routing_keys_active` (Gauge) — the number of unique routing keys with active requests, used for multi-tenant load balancing via the `x-smg-routing-key` header.

---

## 9. Request Type Counters

Both frameworks track request types for multi-modal and structured output workloads. These counters are defined in a shared module (`request_metrics.py`) and incremented across **all three API endpoints**: `/v1/chat/completions`, `/v1/completions`, and `/v1/responses`.

### Metrics

| Metric Name | Type | Description | Present In |
|-------------|------|-------------|------------|
| `request_type_image_total` | Counter | Requests containing images | Both |
| `request_type_video_total` | Counter | Requests containing `video_url` content parts | Both |
| `request_type_tool_call_total` | Counter | Requests with `tools` defined and `tool_choice != "none"` | Both |
| `request_type_structured_output_total` | Counter | Requests using any structured output method | Both |

### API Endpoint Coverage

| Counter | Chat (`/v1/chat/completions`) | Completions (`/v1/completions`) | Responses (`/v1/responses`) |
|---------|------|-------------|-----------|
| `request_type_image_total` | SGLang + vLLM (`image_url` parts) | N/A | SGLang + vLLM (`input_image` items) |
| `request_type_video_total` | SGLang + vLLM (`video_url` parts) | N/A | N/A (not supported by Responses API) |
| `request_type_tool_call_total` | SGLang + vLLM | N/A | SGLang + vLLM |
| `request_type_structured_output_total` | SGLang + vLLM | SGLang + vLLM | vLLM only (`text.format`) |

**Notes**:
- Completions API does not support images, videos, or tool calls in either framework.
- SGLang's Responses API does not support structured output (`text.format` / `response_format`), so the structured output counter is vLLM-only for responses.
- The Responses API does not support video input in either framework (no `input_video` type in the OpenAI Responses API spec).

### Classification Logic

#### Image

- **Chat**: Both iterate over `request.messages[*].content` parts and check `part.type == "image_url"`.
- **Responses**: Both check `request.input` items for top-level `input_image` type or nested `input_image`/`image_url` in message content parts.
- Each counter increments at most once per request (flags, not per-part counts).

#### Video

- **Chat only**: Both check `part.type == "video_url"` in message content parts.

#### Tool Call

Both use identical logic across chat and responses:
```python
if request.tools and request.tool_choice != "none":
```
This counts requests that **enable** tool calling (tools defined + not explicitly disabled). It does NOT count whether the model actually invoked a tool.

#### Structured Output

Both frameworks now cover all structured output methods:

| Method | SGLang | vLLM |
|--------|--------|------|
| `response_format: json_schema` | ✓ (via `response_format.type` check) | ✓ (via `response_format.type` check) |
| `response_format: json_object` | ✓ (via `response_format.type` check) | ✓ (via `response_format.type` check) |
| `response_format: structural_tag` | ✓ (via `response_format.type` check) | ✓ (via `response_format.type` check) |
| `regex` | ✓ (top-level `request.regex` field) | ✓ (via `request.structured_outputs`) |
| `ebnf` / `grammar` | ✓ (top-level `request.ebnf` field) | ✓ (via `request.structured_outputs`) |
| `choice` | N/A (not a SGLang chat field) | ✓ (via `request.structured_outputs`) |
| `json` (extra_body) | N/A (SGLang uses `response_format`) | ✓ (via `request.structured_outputs`) |
| `text.format` (Responses API) | N/A (not supported in SGLang) | ✓ (`json_schema`, `json_object`) |

**API design difference**: SGLang exposes structured output options as top-level request fields (`regex`, `ebnf`), while vLLM uses `extra_body={"structured_outputs": {...}}` which maps to a `StructuredOutputsParams` object.

### SGLang Grammar Metrics (Engine-Level, Distinct from Request Type Counters)

SGLang has additional **engine-level** grammar metrics that are separate from the chat-level `request_type_structured_output_total`:

| Metric | Type | Layer | Description |
|--------|------|-------|-------------|
| `num_so_requests_total` | Counter | Tokenizer Manager | All requests with grammar at completion (any endpoint, not just chat) |
| `num_grammar_total` | Counter | Scheduler | Grammar compilation events (internal engine operations) |
| `num_grammar_queue_reqs` | Gauge | Scheduler | Current grammar queue depth (live snapshot) |
| `num_grammar_cache_hit_total` | Counter | Scheduler | Grammar cache hits |
| `num_grammar_aborted_total` | Counter | Scheduler | Aborted grammar requests |
| `num_grammar_timeout_total` | Counter | Scheduler | Grammar timeouts |

**Key distinction**: `request_type_structured_output_total` fires at the HTTP/chat entrypoint when a request arrives. `num_so_requests_total` fires at the backend when a request with grammar completes (covers all endpoints: `/generate`, `/v1/completions`, `/v1/chat/completions`). `num_grammar_total` counts grammar engine operations, not user requests.

---

## 10. Config Info Metrics

Both frameworks expose configuration as info-style gauges (value always 1.0) with configuration details in labels.

### `model_config_info`

| Label | SGLang | vLLM |
|-------|--------|------|
| `model` | Yes | Yes |
| `served_model_name` | Yes | Yes |
| `dtype` | Yes | Yes |
| `max_model_len` | Yes (`context_length`) | Yes |
| `max_total_tokens` | Yes | No (vLLM-specific: see `cache_config_info`) |
| `max_output_length` | Yes | No (vLLM uses `max_new_tokens` in generation_config) |
| `quantization` | Yes | Yes |
| `enforce_eager` | Yes (`disable_cuda_graph`) | Yes |
| `gpu_type` | Yes | Yes |

### `parallel_config_info`

| Label | SGLang | vLLM |
|-------|--------|------|
| `tensor_parallel_size` | Yes | Yes |
| `pipeline_parallel_size` | Yes | Yes |
| `data_parallel_size` | Yes | Yes |
| `expert_parallel_size` / `enable_expert_parallel` | `expert_parallel_size` | `enable_expert_parallel` |
| `gpu_count` | Yes | Yes |

### `speculative_config_info`

| Label | SGLang | vLLM |
|-------|--------|------|
| `spec_enabled` | Yes | Yes |
| `spec_algorithm` / `spec_method` | `spec_algorithm` | `spec_method` |
| `spec_num_draft_tokens` / `spec_num_tokens` | `spec_num_draft_tokens` | `spec_num_tokens` |
| `spec_num_steps` | Yes | No |
| `spec_eagle_topk` | Yes | No |
| `spec_draft_model` | Yes | Yes |

### `detailed_config_info` (NEW — both frameworks)

Bundles scheduler, compilation, attention, and environment settings into a single info gauge.

| Label | SGLang | vLLM |
|-------|--------|------|
| `stream_interval` | Yes | Yes |
| `attention_backend` | Yes | Yes |
| `sampling_backend` | Yes | No |
| `grammar_backend` | Yes | No |
| `chunked_prefill_size` | Yes | No |
| `schedule_policy` | Yes | No |
| `compilation_mode` | No | Yes |
| `compilation_backend` | No | Yes |
| `cudagraph_mode` | No | Yes |
| `flash_attn_version` | No | Yes |
| `flashinfer_moe_backend` | No | Yes |

### `cache_config_info` (vLLM-only)

vLLM additionally exposes cache configuration info. SGLang does not have a direct equivalent.

---

## 11. Architecture Differences

### vLLM Architecture

vLLM records **all** per-request latency metrics in the **frontend process** (AsyncLLM/OutputProcessor):

```
┌─────────────────────────────────────────────────────────────────┐
│                     AsyncLLM (Frontend Process)                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │ OutputProcessor │───▶│ IterationStats  │───▶│ StatLogger  │ │
│  │ - compute stats │    │ - e2e_latency   │    │ Manager     │ │
│  │ - per request   │    │ - ttft          │    │ - Prometheus│ │
│  └─────────────────┘    │ - decode_time   │    │ - Logging   │ │
│                         └─────────────────┘    └─────────────┘ │
│                                                       │         │
│                                              /metrics endpoint  │
└─────────────────────────────────────────────────────────────────┘
         ▲
         │ IPC (ZMQ) - stats as dataclasses
         │
┌─────────────────────────────────────────────────────────────────┐
│                     EngineCore (Separate Process)                │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │    Scheduler    │───▶│ EngineCoreOutput│ (includes stats)    │
│  │ - batch stats   │    │ - scheduler_stats│                    │
│  └─────────────────┘    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design**: All histograms are observed in the same process that exposes `/metrics`.

### SGLang Architecture

SGLang splits metrics recording across **multiple processes**, using Prometheus multiprocess mode:

```
┌─────────────────────────────────────────────────────────────────┐
│                     HTTP Server (Main Process)                   │
│                                                                  │
│  /metrics endpoint with MultiProcessCollector                    │
│  (aggregates .db files from PROMETHEUS_MULTIPROC_DIR)           │
└─────────────────────────────────────────────────────────────────┘
         ▲                                    ▲
         │ reads .db files                    │ reads .db files
         │                                    │
┌────────┴────────┐                  ┌────────┴────────┐
│ Scheduler Proc  │                  │ TokenizerManager│
│                 │                  │ (Main Process)  │
│ SchedulerMetrics│                  │                 │
│ Collector       │    IPC (ZMQ)     │ TokenizerMetrics│
│ ─────────────── │ ──────────────▶  │ Collector       │
│ • queue_time    │  BatchTokenID    │ ─────────────── │
│ • gauges        │  Output          │ • e2e_latency   │
│                 │                  │ • ttft          │
│ writes to:      │                  │ • itl           │
│ PROMETHEUS_     │                  │                 │
│ MULTIPROC_DIR   │                  │ writes to:      │
│ /*.db           │                  │ PROMETHEUS_     │
└─────────────────┘                  │ MULTIPROC_DIR   │
                                     │ /*.db           │
                                     └─────────────────┘
```

### Multiprocess Mode Details

The Prometheus multiprocess mode **works correctly** for all SGLang metrics when `PROMETHEUS_MULTIPROC_DIR` is properly set:
- Each process writes metrics to separate `.db` files
- `MultiProcessCollector` aggregates these files when `/metrics` is scraped
- Histograms are aggregated by summing bucket counts across processes
- Gauges with `multiprocess_mode="mostrecent"` take the latest value

### Common Issue: Missing PROMETHEUS_MULTIPROC_DIR

**Symptoms when `PROMETHEUS_MULTIPROC_DIR` is not set:**
- Gauge metrics (like `gen_throughput`) may still appear
- Histogram metrics (like `e2e_request_latency_seconds`) are missing

**Solution**:
```bash
export PROMETHEUS_MULTIPROC_DIR=/tmp/sglang_prometheus
```

---

## 12. SGLang Streaming Requirement

Some SGLang metrics are **only recorded when using streaming requests** (`stream: true`):

| Metric | Non-Streaming | Streaming |
|--------|---------------|-----------|
| `e2e_request_latency_seconds` | Available | Available |
| `time_to_first_token_seconds` | Available | Available |
| `request_decode_time_seconds` | Not recorded | Available |
| `request_prefill_time_seconds` | Not recorded | Available |
| `inter_token_latency_seconds` | Not recorded | Available |

**Why?**
- `inter_token_latency_seconds`: Measured between incremental token deliveries (only happens with streaming)
- `request_decode_time_seconds`: Requires `first_token_time` timestamp, which is set during streaming

**Recommendation**: Use `stream: true` for full latency breakdown metrics.

---

## 13. Remaining Gaps & Recommendations

### High-Impact Gaps: SGLang Missing from vLLM

| Priority | Gap | Description | Impact |
|----------|-----|-------------|--------|
| **P2** | `request_prefill_kv_computed_tokens` | New KV tokens (excl. cached) per request | Cache effectiveness analysis |
| **P2** | `iteration_tokens_total` | Tokens per engine step | Batch utilization analysis |
| **P3** | `request_params_n` | "n" parameter distribution | Workload characterization |
| **P3** | `request_max_num_generation_tokens` | Max gen tokens per request | Workload characterization |
| **P3** | KV block residency metrics | Block lifetime, idle time, reuse gaps | Cache tuning |
| **P3** | `engine_sleep_state` | Sleep/wake state | Multi-engine management |

### High-Impact Gaps: vLLM Missing from SGLang

| Priority | Gap | Description | Impact |
|----------|-----|-------------|--------|
| **P1** | Grammar/SO metrics (11 metrics) | Grammar compilation, cache hits, timeouts | Structured output monitoring |
| **P1** | PD disaggregation queue depths (4 gauges) | Prefill/decode queue depths — no vLLM equivalent due to architectural difference (see §3) | Disaggregated serving monitoring |
| **P2** | `gpu_execution_seconds_total` | GPU execution time tracking | GPU utilization analysis |
| **P2** | Retraction token counters (input/output) | `num_retracted_input_tokens_total`, `num_retracted_output_tokens_total` | Memory pressure debugging |
| **P2** | GPU cache eviction metrics | Eviction/load-back duration and counts | Tiered cache performance |
| **P3** | Routing key metrics | Per-routing-key request distribution | Multi-tenant analysis |
| **P3** | CUDA graph metrics | Forward pass counts by graph mode | Performance mode analysis |
| **P3** | MoE metrics | Expert load balancing | MoE model optimization |
| **P3** | Prefill delayer metrics | Delay tracking | Scheduling optimization |
| **P3** | `utilization` composite gauge | Single-number utilization | SLO management |

### Recommendations

#### Counter Naming Alignment
The `_total` suffix difference is cosmetic but causes confusion in unified dashboards:
- SGLang: `prompt_tokens_total`, `generation_tokens_total`, `num_retracted_requests_total`, `request_success_total`
- vLLM: `prompt_tokens_total`, `generation_tokens_total`, `num_retracted_requests_total`, `request_success_total`

**Recommendation**: Align on the `_total` suffix (OpenMetrics standard for counters).

#### Bucket Alignment
For consistent P50/P99 calculations across frameworks, align bucket boundaries for:
1. `time_to_first_token_seconds` — vLLM has better sub-100ms coverage
2. `inter_token_latency_seconds` — SGLang has better sub-10ms coverage
3. `e2e_request_latency_seconds` — different ranges and granularity

**Recommendation**: Adopt the union of both bucket sets, or make vLLM buckets configurable.

#### Next Steps for Full Parity
1. **SGLang**: Add `request_prefill_kv_computed_tokens` and `iteration_tokens_total`
2. **SGLang**: Consider adding `prompt_tokens_by_source` breakdown (local_compute, local_cache_hit, external_kv_transfer) to match new vLLM token provenance tracking
3. **vLLM**: Add `prompt_tokens_by_source`, `prompt_tokens_cached`, `prompt_tokens_recomputed` to `_METRIC_NAME_MAP` in the OTel bridge for proper `_total` suffix alignment
4. **vLLM**: Add structured output/grammar metrics if/when structured output support is added
5. **Both**: Consider adopting `iteration_tokens_total` as a shared metric for batch utilization

---

## 14. Appendix: File Locations

### SGLang
| File | Purpose |
|------|---------|
| `python/sglang/srt/metrics/collector.py` | Metrics definitions (SchedulerMetricsCollector, TokenizerMetricsCollector, etc.) |
| `python/sglang/srt/managers/tokenizer_manager.py` | Recording logic for request-level metrics |
| `python/sglang/srt/managers/scheduler_metrics_mixin.py` | Scheduler-level metrics recording |
| `python/sglang/srt/utils/common.py` | HTTP-level metrics middleware |
| `python/sglang/srt/entrypoints/openai/request_metrics.py` | Shared request type counters and classification functions |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | Chat serving handler |
| `python/sglang/srt/entrypoints/openai/serving_completions.py` | Completions serving handler |
| `python/sglang/srt/entrypoints/openai/serving_responses.py` | Responses serving handler |
| `docs/references/production_metrics.md` | Official documentation |

### vLLM
| File | Purpose |
|------|---------|
| `vllm/v1/metrics/loggers.py` | Metrics definitions (PrometheusStatLogger) |
| `vllm/v1/metrics/stats.py` | Stats data structures |
| `vllm/v1/metrics/prometheus.py` | Prometheus setup |
| `vllm/v1/spec_decode/metrics.py` | Speculative decoding metrics |
| `vllm/entrypoints/openai/request_metrics.py` | Shared request type counters and classification functions |
| `vllm/entrypoints/openai/chat_completion/serving.py` | Chat serving handler |
| `vllm/entrypoints/openai/completion/serving.py` | Completions serving handler |
| `vllm/entrypoints/openai/responses/serving.py` | Responses serving handler |
| `vllm/entrypoints/serve/instrumentator/metrics.py` | HTTP-level metrics (`prometheus_fastapi_instrumentator`) |
| `vllm/config/*.py` | Config info metric labels |

---

## 15. Bridge-Side Metric Renaming

### Problem

To unify metrics naming between vLLM and SGLang, we previously renamed metrics directly in
each framework's source code (removing `vllm:` / `sglang:` prefixes, aligning `_total`
suffixes, etc.). This approach works but creates a maintenance burden:

- Every rebase onto upstream requires resolving conflicts in metrics definition files
- Upstream PRs that add new metrics need manual renaming
- Two separate codebases must be kept in sync

### Solution: Rename in the Prom-to-OTEL bridge

Instead of modifying source code, apply metric name transformations in the **bridge thread**
(`otel_instrumentation.py :: start_prom_to_otel_bridge`), which already scrapes Prometheus
metrics and re-exports them as OTEL instruments. The bridge already has a
`_sanitize_metric_name` function that transforms names — extend it with a mapping dictionary.

**File:** `vllm/otel_instrumentation.py`

### Approach

1. Keep native metric names untouched in vLLM/SGLang source code (`vllm:prompt_tokens`,
   `sglang:prompt_tokens_total`, etc.)
2. Define a mapping dictionary in `otel_instrumentation.py` that maps native Prometheus
   metric names to unified OTEL metric names
3. Apply the mapping in `_sanitize_metric_name()` before creating OTEL instruments

### Mapping dictionary

The dictionary handles three kinds of transformations:

1. **Prefix stripping** — `vllm_` / `sglang_` → common name
2. **Suffix alignment** — `_total` presence/absence
3. **Name differences** — metrics that have completely different names across frameworks

```python
# Native Prometheus name → unified OTEL name
# Only entries that need renaming; unlisted metrics pass through with prefix stripped.
_METRIC_NAME_MAP: dict[str, str] = {
    # --- Token counters (suffix alignment) ---
    # vLLM uses no _total suffix; SGLang uses _total
    "vllm_prompt_tokens":                   "prompt_tokens_total",
    "sglang_prompt_tokens_total":           "prompt_tokens_total",
    "vllm_generation_tokens":               "generation_tokens_total",
    "sglang_generation_tokens_total":       "generation_tokens_total",

    # --- Request counters (suffix alignment) ---
    "vllm_request_success":                 "request_success_total",
    "sglang_request_success_total":         "request_success_total",

    # --- Metrics with identical names after prefix strip (just strip prefix) ---
    # e.g. vllm_e2e_request_latency_seconds → e2e_request_latency_seconds
    #      sglang_e2e_request_latency_seconds → e2e_request_latency_seconds
    # These are handled by the default prefix-strip logic, no explicit entry needed.

    # --- Metrics with different names across frameworks ---
    # Add entries here when the name differs beyond just prefix/suffix.
    # Example (hypothetical):
    # "vllm_num_requests_running":          "num_requests_running",
    # "sglang_running_req_count":           "num_requests_running",
}


def _sanitize_metric_name(name: str) -> str:
    name = name.replace(":", "_")
    # Check explicit mapping first
    if name in _METRIC_NAME_MAP:
        return _METRIC_NAME_MAP[name]
    # Default: strip known prefixes
    for prefix in ("vllm_", "sglang_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name
```

### What this changes

| Aspect | Before (source-code renaming) | After (bridge-side renaming) |
|--------|-------------------------------|------------------------------|
| vLLM source code | Modified metric names | **Untouched** — keep native `vllm:` prefix |
| SGLang source code | Modified metric names | **Untouched** — keep native `sglang:` prefix |
| Rebase conflicts | Frequent (metrics files change often) | **None** (no source changes) |
| New upstream metrics | Need manual renaming | Auto-stripped prefix; add to map only if name differs |
| Mapping maintenance | Scattered across source files | **Single dictionary** in `otel_instrumentation.py` |
| Prometheus `/metrics` endpoint | Shows unified names | Shows **native** names (only OTEL export is unified) |
| OTEL collector | Receives unified names | Receives unified names (same outcome) |

### Considerations

- **Prometheus dashboards** that scrape `/metrics` directly will see native names (`vllm_*`).
  Only the OTEL export path gets unified names. This is acceptable if all production
  monitoring goes through the OTEL collector.
- The mapping dictionary should be kept in sync with this comparison doc (§2) when new
  shared metrics are added.
- For metrics unique to one framework (§3, §4), no mapping is needed — they pass through
  with prefix stripped.
- **Note (v2-0.16.0)**: The new `prompt_tokens_by_source`, `prompt_tokens_cached`, and
  `prompt_tokens_recomputed` counters are NOT yet in `_METRIC_NAME_MAP`. They will be
  auto-stripped of the `vllm_` prefix but will NOT get `_total` suffix alignment in the
  OTel export. Consider adding them to the map for consistency.
