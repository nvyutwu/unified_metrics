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

---

## 1. Executive Summary

| Aspect | SGLang | vLLM |
|--------|--------|------|
| Metric Prefix | `sglang:` | `vllm:` |
| Primary Metrics File | `python/sglang/srt/metrics/collector.py` | `vllm/v1/metrics/loggers.py` |
| Stats Tracking | `tokenizer_manager.py`, `scheduler_metrics_mixin.py` | `vllm/v1/metrics/stats.py` |
| Enable Flag | `--enable-metrics` | Enabled by default |
| Total Metrics | ~100 | ~49 |
| Counter Naming | Uses `_total` suffix (OpenMetrics standard) | No `_total` suffix |
| Multiprocess Mode | Prometheus multiprocess via `PROMETHEUS_MULTIPROC_DIR` | Single-process (frontend) |

---

## 2. Unified Metrics (Present in Both)

These metrics exist in both frameworks on the `feat/unified-metrics` branches with compatible names and semantics.

### Core Latency Histograms

| Metric Name | SGLang | vLLM | Computation Match? |
|-------------|--------|------|-------------------|
| `e2e_request_latency_seconds` | Histogram | Histogram | **YES** (finished - created vs iteration_ts - arrival) |
| `time_to_first_token_seconds` | Histogram | Histogram | **YES** (first_token - created vs iteration_ts - arrival) |
| `inter_token_latency_seconds` | Histogram | Histogram | **SIMILAR** (SGLang normalizes by num_new_tokens; vLLM measures individual gaps) |
| `request_queue_time_seconds` | Histogram | Histogram | **YES** (both: scheduled - queued) |
| `request_inference_time_seconds` | Histogram | Histogram | **YES** (both: last_token - scheduled) |
| `request_prefill_time_seconds` | Histogram | Histogram | **YES** (both: first_token - scheduled) |
| `request_decode_time_seconds` | Histogram | Histogram | **YES** (both: last_token - first_token) |
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
| `num_preemptions_total` (SGLang) / `num_preemptions` (vLLM) | Counter | Counter | **YES** (name differs by `_total` suffix) |

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
| `cached_tokens_total` (SGLang) / `prefix_cache_hits` (vLLM) | Counter | Counter | **SIMILAR** |
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

| Metric Name | Type | Description |
|-------------|------|-------------|
| `num_prefill_prealloc_queue_reqs` | Gauge | Prefill prealloc queue depth |
| `num_prefill_inflight_queue_reqs` | Gauge | Prefill inflight queue depth |
| `num_decode_prealloc_queue_reqs` | Gauge | Decode prealloc queue depth |
| `num_decode_transfer_queue_reqs` | Gauge | Decode transfer queue depth |
| `kv_transfer_speed_gb_s` | Gauge | KV transfer bandwidth (GB/s) |
| `kv_transfer_latency_ms` | Gauge | KV transfer latency (ms) |
| `kv_transfer_bootstrap_ms` | Gauge | KV transfer bootstrap time (ms) |
| `kv_transfer_alloc_ms` | Gauge | KV transfer allocation wait (ms) |
| `kv_transfer_total_mb` | Gauge | Total KV transferred (MB) |
| `num_bootstrap_failed_reqs_total` | Counter | Bootstrap failures |
| `num_transfer_failed_reqs_total` | Counter | Transfer failures |
| `num_prefill_retries_total` | Counter | Prefill retries |

### Retraction Detail (more granular than vLLM)

| Metric Name | Type | Description |
|-------------|------|-------------|
| `num_retracted_requests_total` | Counter | Retracted request count |
| `num_retracted_input_tokens_total` | Counter | Input tokens from retracted requests |
| `num_retracted_output_tokens_total` | Counter | Output tokens from retracted requests |
| `num_retractions` | Histogram | Retraction count distribution per request |
| `num_retracted_reqs` | Gauge | Current retracted count (snapshot) |

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

### End-to-End Request Latency (`e2e_request_latency_seconds`)

| Framework | Computation |
|-----------|-------------|
| **SGLang** | `finished_time - created_time` |
| **vLLM** | `iteration_timestamp - arrival_time` |

**Verdict**: **EQUIVALENT** — Both measure total time from request arrival to completion.

### Time to First Token (`time_to_first_token_seconds`)

| Framework | Computation |
|-----------|-------------|
| **SGLang** | `first_token_time - created_time` |
| **vLLM** | `iteration_timestamp - arrival_time` (when first token is generated) |

**Verdict**: **EQUIVALENT** — Both measure time from request creation/arrival to first token generation.

### Inter-Token Latency / Time Per Output Token

| Framework | Metric Name | Computation | Notes |
|-----------|-------------|-------------|-------|
| **SGLang** | `inter_token_latency_seconds` | `(new_time - last_time) / num_new_tokens` | Normalizes per token in batch updates |
| **vLLM** | `inter_token_latency_seconds` | `engine_core_timestamp - last_token_ts` | Time between consecutive tokens |
| **Both** | `request_time_per_output_token_seconds` | `decode_time / (gen_tokens - 1)` | Per-request mean |

**Differences**:
- SGLang normalizes by `num_new_tokens` when multiple tokens arrive together
- vLLM tracks individual inter-token gaps
- Both now provide the per-request mean TPOT via `request_time_per_output_token_seconds`

### Queue Time (`request_queue_time_seconds`)

| Framework | Computation |
|-----------|-------------|
| **SGLang** | `forward_entry_time - wait_queue_entry_time` |
| **vLLM** | `scheduled_ts - queued_ts` |

**Verdict**: **EQUIVALENT** — Both measure time spent waiting before entering the running state.

### Prefill and Decode Time (now unified)

| Metric | SGLang Computation | vLLM Computation |
|--------|-------------------|------------------|
| `request_prefill_time_seconds` | `first_token_time - scheduled_time` | `first_token_ts - scheduled_ts` |
| `request_decode_time_seconds` | `finished_time - first_token_time` | `last_token_ts - first_token_ts` |
| `request_inference_time_seconds` | `finished_time - scheduled_time` | `last_token_ts - scheduled_ts` |

**Verdict**: **EQUIVALENT** — Both compute phase breakdowns identically.

### Cache Hit Rate

| Framework | Method | Notes |
|-----------|--------|-------|
| **SGLang** | `cache_hit_rate` gauge (pre-computed ratio) | Simple to read, but loses raw numbers |
| **vLLM** | `prefix_cache_queries` + `prefix_cache_hits` (two counters) | Requires PromQL to compute ratio, but more flexible |

### Throughput (`gen_throughput`)

| Framework | Method | Notes |
|-----------|--------|-------|
| **SGLang** | `num_generated_tokens / gap_latency` | Computed in scheduler process |
| **vLLM** | `accumulated_tokens / delta_time` (every ~5s) | Computed in frontend process |

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

Both frameworks expose HTTP server-level metrics via `common.py` / middleware:

| Metric Name | Type | Description | Present In |
|-------------|------|-------------|------------|
| `http_request_duration_seconds` | Histogram | HTTP request duration by method/path/status | Both |

SGLang defines this in `python/sglang/srt/openai_api/common.py`; vLLM in its serving middleware. Both label by `method`, `path`, and `status_code`.

---

## 9. Request Type Counters

Both frameworks track request types for multi-modal and structured output workloads:

| Metric Name | Type | Description | Present In |
|-------------|------|-------------|------------|
| `request_type_image_total` | Counter | Requests containing image inputs | Both |
| `request_type_video_total` | Counter | Requests containing video inputs | Both |
| `request_type_tool_call_total` | Counter | Requests using tool/function calling | Both |
| `request_type_structured_output_total` | Counter | Requests using structured output (JSON schema, regex, etc.) | Both |

These are incremented in the chat serving handler (`serving_chat.py` in both codebases) based on request content analysis.

---

## 10. Config Info Metrics

Both frameworks expose configuration as info-style gauges (value always 1.0) with configuration details in labels.

### `model_config_info`

| Label | SGLang | vLLM |
|-------|--------|------|
| `model` | Yes | Yes |
| `served_model_name` | Yes | Yes |
| `dtype` | Yes | Yes |
| `max_total_tokens` / `max_model_len` | `max_total_tokens` | `max_model_len` |
| `quantization` | Yes | Yes |
| `enforce_eager` | No | Yes |
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
| **P1** | PD disaggregation metrics (12 metrics) | Prefill/decode queue depths, KV transfer | Disaggregated serving monitoring |
| **P2** | `gpu_execution_seconds_total` | GPU execution time tracking | GPU utilization analysis |
| **P2** | Retraction detail (input/output tokens) | More granular preemption tracking | Memory pressure debugging |
| **P2** | GPU cache eviction metrics | Eviction/load-back duration and counts | Tiered cache performance |
| **P3** | Routing key metrics | Per-routing-key request distribution | Multi-tenant analysis |
| **P3** | CUDA graph metrics | Forward pass counts by graph mode | Performance mode analysis |
| **P3** | MoE metrics | Expert load balancing | MoE model optimization |
| **P3** | Prefill delayer metrics | Delay tracking | Scheduling optimization |
| **P3** | `utilization` composite gauge | Single-number utilization | SLO management |

### Recommendations

#### Counter Naming Alignment
The `_total` suffix difference is cosmetic but causes confusion in unified dashboards:
- SGLang: `prompt_tokens_total`, `generation_tokens_total`, `num_preemptions_total`, `request_success_total`
- vLLM: `prompt_tokens`, `generation_tokens`, `num_preemptions`, `request_success`

**Recommendation**: Align on the `_total` suffix (OpenMetrics standard for counters).

#### Bucket Alignment
For consistent P50/P99 calculations across frameworks, align bucket boundaries for:
1. `time_to_first_token_seconds` — vLLM has better sub-100ms coverage
2. `inter_token_latency_seconds` — SGLang has better sub-10ms coverage
3. `e2e_request_latency_seconds` — different ranges and granularity

**Recommendation**: Adopt the union of both bucket sets, or make vLLM buckets configurable.

#### Next Steps for Full Parity
1. **SGLang**: Add `request_prefill_kv_computed_tokens` and `iteration_tokens_total`
2. **vLLM**: Add structured output/grammar metrics if/when structured output support is added
3. **Both**: Consider adopting `iteration_tokens_total` as a shared metric for batch utilization

---

## 14. Appendix: File Locations

### SGLang
| File | Purpose |
|------|---------|
| `python/sglang/srt/metrics/collector.py` | Metrics definitions (SchedulerMetricsCollector, TokenizerMetricsCollector, etc.) |
| `python/sglang/srt/managers/tokenizer_manager.py` | Recording logic for request-level metrics |
| `python/sglang/srt/managers/scheduler_metrics_mixin.py` | Scheduler-level metrics recording |
| `python/sglang/srt/openai_api/common.py` | HTTP-level metrics, request type counters |
| `python/sglang/srt/openai_api/v1/serving_chat.py` | Chat serving handler (request type detection) |
| `docs/references/production_metrics.md` | Official documentation |

### vLLM
| File | Purpose |
|------|---------|
| `vllm/v1/metrics/loggers.py` | Metrics definitions (PrometheusStatLogger) |
| `vllm/v1/metrics/stats.py` | Stats data structures |
| `vllm/v1/metrics/prometheus.py` | Prometheus setup |
| `vllm/v1/spec_decode/metrics.py` | Speculative decoding metrics |
| `vllm/entrypoints/openai/serving_chat.py` | Chat serving handler (request type detection) |
| `vllm/config/*.py` | Config info metric labels |
