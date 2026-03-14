# TensorRT-LLM Metrics Inventory

> Auto-generated from source code analysis of TensorRT-LLM `main` branch.

---

## Table of Contents
1. [Prometheus Metrics (trtllm-serve)](#1-prometheus-metrics-trtllm-serve)
   - [Request Metrics](#request-metrics)
   - [KV Cache Metrics](#kv-cache-metrics)
2. [Disaggregated Serving Metrics](#2-disaggregated-serving-metrics)
   - [Client-Side Metrics (per-role)](#client-side-metrics-per-role)
   - [Server-Side Metrics](#server-side-metrics)
3. [JSON Iteration Stats (/metrics endpoint)](#3-json-iteration-stats-metrics-endpoint)
   - [Top-Level Iteration Stats](#top-level-iteration-stats)
   - [KV Cache Stats](#kv-cache-stats)
   - [Inflight Batching Stats](#inflight-batching-stats)
   - [Static Batching Stats](#static-batching-stats)
   - [Speculative Decoding Stats](#speculative-decoding-stats)
4. [Per-Request Performance Metrics (/perf_metrics endpoint)](#4-per-request-performance-metrics-perf_metrics-endpoint)
   - [Timing Metrics](#timing-metrics)
   - [KV Cache Metrics (per-request)](#kv-cache-metrics-per-request)
   - [Speculative Decoding Metrics (per-request)](#speculative-decoding-metrics-per-request)
5. [PyExecutor Internal Timing](#5-pyexecutor-internal-timing)
6. [Triton Backend Custom Metrics](#6-triton-backend-custom-metrics)
7. [Cross-Reference: TRT-LLM vs SGLang vs vLLM](#7-cross-reference-trt-llm-vs-sglang-vs-vllm)
8. [Notes](#8-notes)

---

## 1. Prometheus Metrics (trtllm-serve)

**Source**: `tensorrt_llm/metrics/collector.py` (MetricsCollector class)
**Endpoint**: `/prometheus/metrics`
**Prefix**: `trtllm_`
**Labels**: User-supplied (e.g., `model_name`, `engine_type`); `finished_reason` added for request_success counter
**Condition**: Requires `return_perf_metrics: true` in LLM args

### Request Metrics

#### Counter

| # | Metric Name | Labels | Description |
|---|-------------|--------|-------------|
| 1 | `trtllm_request_success_total` | user labels + `finished_reason` | Count of successfully processed requests |

#### Histograms

| # | Metric Name | Buckets | Description |
|---|-------------|---------|-------------|
| 1 | `trtllm_e2e_request_latency_seconds` | [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0] | End-to-end request latency in seconds |
| 2 | `trtllm_time_to_first_token_seconds` | [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0] | Time to first token in seconds |
| 3 | `trtllm_time_per_output_token_seconds` | [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0] | Time per output token in seconds (TPOT) |
| 4 | `trtllm_request_queue_time_seconds` | [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0] | Time spent in WAITING phase |

### KV Cache Metrics

#### Gauges

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `trtllm_kv_cache_hit_rate` | KV cache hit rate (0.0-1.0) |
| 2 | `trtllm_kv_cache_utilization` | KV cache utilization (usedNumBlocks / maxNumBlocks) |

### Iteration-Level Metrics

Exposed via `log_iteration_stats()` from JSON iteration stats data.

#### Counter

| # | Metric Name | JSON Field | Description |
|---|-------------|------------|-------------|
| 1 | `trtllm_num_requests_completed_total` | `numCompletedRequests` | Total completed requests (incremented per-iteration) |

#### Gauges — Queue & Load

| # | Metric Name | JSON Field | Description |
|---|-------------|------------|-------------|
| 1 | `trtllm_num_requests_running` | `numActiveRequests` | Number of active requests |
| 2 | `trtllm_num_requests_waiting` | `numQueuedRequests` | Number of queued requests |
| 3 | `trtllm_max_num_active_requests` | `maxNumActiveRequests` | Maximum number of active requests |

#### Gauges — Iteration Latency

| # | Metric Name | JSON Field | Description |
|---|-------------|------------|-------------|
| 1 | `trtllm_iteration_latency_seconds` | `iterLatencyMS` (ms→s) | Iteration latency in seconds |

#### Gauges — Memory

| # | Metric Name | JSON Field | Description |
|---|-------------|------------|-------------|
| 1 | `trtllm_gpu_memory_usage_bytes` | `gpuMemUsage` | GPU memory usage in bytes |
| 2 | `trtllm_cpu_memory_usage_bytes` | `cpuMemUsage` | CPU memory usage in bytes |
| 3 | `trtllm_pinned_memory_usage_bytes` | `pinnedMemUsage` | Pinned memory usage in bytes |

#### Gauges — Batch Size

| # | Metric Name | JSON Field | Description |
|---|-------------|------------|-------------|
| 1 | `trtllm_max_batch_size_static` | `maxBatchSizeStatic` | Static maximum batch size |
| 2 | `trtllm_max_batch_size_runtime` | `maxBatchSizeRuntime` | Runtime maximum batch size |
| 3 | `trtllm_max_num_tokens_runtime` | `maxNumTokensRuntime` | Runtime maximum number of tokens |

#### Gauges — KV Cache Blocks

| # | Metric Name | JSON Field | Description |
|---|-------------|------------|-------------|
| 1 | `trtllm_kv_cache_max_blocks` | `kvCacheStats.maxNumBlocks` | Maximum KV cache blocks |
| 2 | `trtllm_kv_cache_free_blocks` | `kvCacheStats.freeNumBlocks` | Free KV cache blocks |
| 3 | `trtllm_kv_cache_used_blocks` | `kvCacheStats.usedNumBlocks` | Used KV cache blocks |
| 4 | `trtllm_kv_cache_tokens_per_block` | `kvCacheStats.tokensPerBlock` | Tokens per KV cache block |

#### Gauges — Inflight Batching (conditional)

Present only when using inflight batching scheduler.

| # | Metric Name | JSON Field | Description |
|---|-------------|------------|-------------|
| 1 | `trtllm_num_context_requests` | `inflightBatchingStats.numContextRequests` | Context (prefill) requests |
| 2 | `trtllm_num_generation_requests` | `inflightBatchingStats.numGenRequests` | Generation (decode) requests |
| 3 | `trtllm_num_paused_requests` | `inflightBatchingStats.numPausedRequests` | Paused requests |
| 4 | `trtllm_num_scheduled_requests` | `inflightBatchingStats.numScheduledRequests` | Scheduled requests |
| 5 | `trtllm_total_context_tokens` | `inflightBatchingStats.numCtxTokens` | Total context tokens |
| 6 | `trtllm_avg_decoded_tokens_per_iter` | `inflightBatchingStats.avgNumDecodedTokensPerIter` | Avg decoded tokens per iteration |

#### Gauges — Speculative Decoding (conditional)

Present only when speculative decoding is enabled.

| # | Metric Name | JSON Field | Description |
|---|-------------|------------|-------------|
| 1 | `trtllm_spec_decode_num_draft_tokens` | `specDecodingStats.numDraftTokens` | Draft tokens proposed |
| 2 | `trtllm_spec_decode_num_accepted_tokens` | `specDecodingStats.numAcceptedTokens` | Draft tokens accepted |
| 3 | `trtllm_spec_decode_acceptance_length` | `specDecodingStats.acceptanceLength` | Avg acceptance length |
| 4 | `trtllm_spec_decode_draft_overhead` | `specDecodingStats.draftOverhead` | Draft overhead ratio |

#### Gauges — Config Info (logged once at startup)

Info-style gauges (set to 1) with configuration values as labels. Follows the same pattern as vLLM/SGLang.

| # | Metric Name | Labels | Description |
|---|-------------|--------|-------------|
| 1 | `trtllm_model_config_info` | `model`, `served_model_name`, `dtype`, `quantization`, `max_model_len`, `gpu_type` | Model configuration |
| 2 | `trtllm_parallel_config_info` | `tensor_parallel_size`, `pipeline_parallel_size`, `gpu_count`, `expert_parallel_size` (optional) | Parallelism configuration |
| 3 | `trtllm_speculative_config_info` | `spec_enabled`, `spec_method`, `spec_num_tokens`, `spec_draft_model` (conditional) | Speculative decoding configuration |
| 4 | `trtllm_cache_config_info` | `page_size`, `enable_block_reuse`, `enable_partial_reuse`, `free_gpu_memory_fraction`, `cache_dtype` (conditional) | KV cache configuration |

**Total Prometheus metrics: 40** (2 Counters, 4 Histograms, 34 Gauges)

---

## 2. Disaggregated Serving Metrics

**Source**: `tensorrt_llm/serve/perf_metrics.py`
**Endpoint**: Registered in same `/prometheus/metrics` registry

### Client-Side Metrics (per-role)

Each metric is prefixed by role: `ctx_` (context server), `gen_` (generation server), `mme_` (multimodal encoder).
7 metrics x 3 roles = **21 metrics total**.

#### Counters (per role)

| # | Metric Name Pattern | Description |
|---|---------------------|-------------|
| 1 | `{role}_total_requests` | Total number of requests |
| 2 | `{role}_error_requests` | Total number of error requests |
| 3 | `{role}_retry_requests` | Total number of retry requests |
| 4 | `{role}_completed_requests` | Total number of completed requests |

#### Histograms (per role)

| # | Metric Name Pattern | Buckets | Description |
|---|---------------------|---------|-------------|
| 1 | `{role}_first_token_latency_seconds` | SHORT_TIME_BUCKETS | Latency from first token to completion |
| 2 | `{role}_complete_latency_seconds` | LONG_TIME_BUCKETS | Latency from request arrival to last token |
| 3 | `{role}_per_token_latency_seconds` | SHORT_TIME_BUCKETS | Per-token latency |

**Bucket definitions:**
- SHORT_TIME_BUCKETS: [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0]
- LONG_TIME_BUCKETS: [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0]

### Server-Side Metrics

#### Counters

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `total_requests` | Total number of requests |
| 2 | `stream_requests` | Total number of stream requests |
| 3 | `nonstream_requests` | Total number of non-stream requests |
| 4 | `validation_exceptions` | Total number of validation exceptions |
| 5 | `http_exceptions` | Total number of HTTP exceptions |
| 6 | `internal_errors` | Total number of internal errors |
| 7 | `total_responses` | Total number of responses |

#### Histograms

| # | Metric Name | Buckets | Description |
|---|-------------|---------|-------------|
| 1 | `queue_latency_seconds` | SHORT_TIME_BUCKETS | Latency from request arrival to being processed |

---

## 3. JSON Iteration Stats (/metrics endpoint)

**Source**: C++ `IterationStats` struct in `cpp/include/tensorrt_llm/executor/types.h`
**Endpoint**: `/metrics` (returns JSON array)
**Condition**: Requires `enable_iter_perf_stats: true` in LLM args

### Top-Level Iteration Stats

| # | Field Name | Type | Description |
|---|------------|------|-------------|
| 1 | `timestamp` | string/int | Iteration ending timestamp (microseconds) |
| 2 | `iter` | int | Iteration counter |
| 3 | `iterLatencyMS` | float | Iteration latency (milliseconds) |
| 4 | `newActiveRequestsQueueLatencyMS` | float | Total queue time for newly active requests (ms) |
| 5 | `numNewActiveRequests` | int | Number of newly fetched active requests |
| 6 | `numActiveRequests` | int | Number of currently active requests |
| 7 | `numQueuedRequests` | int | Number of queued requests |
| 8 | `numCompletedRequests` | int | Number of requests completed this iteration |
| 9 | `maxNumActiveRequests` | int | Maximum active request capacity |
| 10 | `maxBatchSizeStatic` | int | Static max batch size passed to executor |
| 11 | `maxBatchSizeTunerRecommended` | int | Dynamic tuner recommended batch size |
| 12 | `maxBatchSizeRuntime` | int | Runtime max batch size (min of static + upper bound) |
| 13 | `maxNumTokensStatic` | int | Static max num tokens |
| 14 | `maxNumTokensTunerRecommended` | int | Tuner recommended max num tokens |
| 15 | `maxNumTokensRuntime` | int | Runtime max num tokens |
| 16 | `gpuMemUsage` | int | GPU memory usage (bytes) |
| 17 | `cpuMemUsage` | int | CPU memory usage (bytes) |
| 18 | `pinnedMemUsage` | int | Pinned memory usage (bytes) |

### KV Cache Stats

**Nested under**: `kvCacheStats` (also `crossKvCacheStats` for encoder-decoder models)

| # | Field Name | Type | Description |
|---|------------|------|-------------|
| 1 | `maxNumBlocks` | int | Maximum KV cache blocks available |
| 2 | `freeNumBlocks` | int | Free KV cache blocks |
| 3 | `usedNumBlocks` | int | Used KV cache blocks |
| 4 | `tokensPerBlock` | int | Tokens per KV cache block |
| 5 | `allocTotalBlocks` | int | Total allocated blocks this iteration |
| 6 | `allocNewBlocks` | int | Newly allocated blocks this iteration |
| 7 | `reusedBlocks` | int | Reused blocks from prefix cache |
| 8 | `missedBlocks` | int | Missed cache lookups |
| 9 | `cacheHitRate` | float | Cache hit rate (0.0-1.0) |

### Inflight Batching Stats

**Nested under**: `inflightBatchingStats` (only with inflight batching scheduler)

| # | Field Name | Type | Description |
|---|------------|------|-------------|
| 1 | `numScheduledRequests` | int | Number of scheduled requests |
| 2 | `numContextRequests` | int | Number of context (prefill) requests |
| 3 | `numGenRequests` | int | Number of generation requests |
| 4 | `numPausedRequests` | int | Number of paused requests |
| 5 | `numCtxTokens` | int | Total context tokens in iteration |
| 6 | `microBatchId` | int | Micro batch index |
| 7 | `avgNumDecodedTokensPerIter` | float | Avg tokens decoded per request per iteration |

### Static Batching Stats

**Nested under**: `staticBatchingStats` (only with static batching scheduler)

| # | Field Name | Type | Description |
|---|------------|------|-------------|
| 1 | `numScheduledRequests` | int | Number of scheduled requests |
| 2 | `numContextRequests` | int | Number of context requests |
| 3 | `numCtxTokens` | int | Total context tokens |
| 4 | `numGenTokens` | int | Total generation tokens |
| 5 | `emptyGenSlots` | int | Empty generation slots |

### Speculative Decoding Stats

**Nested under**: `specDecodingStats` (only when speculative decoding is enabled)

| # | Field Name | Type | Description |
|---|------------|------|-------------|
| 1 | `numDraftTokens` | int | Total proposed draft tokens for all requests |
| 2 | `numAcceptedTokens` | int | Total accepted draft tokens |
| 3 | `numRequestsWithDraftTokens` | int | Requests with at least one draft token |
| 4 | `acceptanceLength` | float | Avg tokens produced per step per request with drafts |
| 5 | `iterLatencyMS` | float | Draft-only iteration latency (ms) |
| 6 | `draftOverhead` | float | Draft overhead ratio (draft latency / total latency) |

---

## 4. Per-Request Performance Metrics (/perf_metrics endpoint)

**Source**: C++ `RequestPerfMetrics` struct in `cpp/include/tensorrt_llm/executor/types.h`; Python `tensorrt_llm/metrics/enums.py`
**Endpoint**: `/perf_metrics` (returns JSON)
**Condition**: Requires `return_perf_metrics: true` in LLM args

### Timing Metrics

| # | Field Name | Type | Description |
|---|------------|------|-------------|
| 1 | `arrival_time` | float | Request arrival time (steady clock) |
| 2 | `first_scheduled_time` | float | Time request was first scheduled |
| 3 | `first_token_time` | float | Time first output token was generated |
| 4 | `last_token_time` | float | Time last output token was generated |
| 5 | `kv_cache_transfer_start` | float | KV cache transfer start (disaggregated serving) |
| 6 | `kv_cache_transfer_end` | float | KV cache transfer end (disaggregated serving) |
| 7 | `kv_cache_size` | int | Size of KV cache transferred (bytes, disaggregated) |

### KV Cache Metrics (per-request)

| # | Field Name | Type | Description |
|---|------------|------|-------------|
| 1 | `num_total_allocated_blocks` | int | Total allocated blocks for request |
| 2 | `num_new_allocated_blocks` | int | Newly allocated blocks |
| 3 | `num_reused_blocks` | int | Reused blocks (prefix cache) |
| 4 | `num_missed_blocks` | int | Missed blocks |
| 5 | `kv_cache_hit_rate` | float | Per-request KV cache hit rate |

### Speculative Decoding Metrics (per-request)

| # | Field Name | Type | Description |
|---|------------|------|-------------|
| 1 | `acceptance_rate` | float | Token acceptance rate |
| 2 | `total_accepted_draft_tokens` | int | Total accepted draft tokens |
| 3 | `total_draft_tokens` | int | Total draft tokens used |

### Additional Fields

| # | Field Name | Type | Description |
|---|------------|------|-------------|
| 1 | `first_iter` | int | First iteration where request was processed |
| 2 | `last_iter` | int | Last iteration where a token was generated |
| 3 | `iter` | int | Current iteration |

---

## 5. PyExecutor Internal Timing

**Source**: `tensorrt_llm/_torch/pyexecutor/perf_metrics_manager.py` (PerfMetricsManager class)
**Not exposed via endpoint** — internal per-request timing used to compute Prometheus metrics.

### Per-Step Metrics (appended per iteration to each request)

| # | Field Name | Type | Description |
|---|------------|------|-------------|
| 1 | `forward_start_time` | float | CPU timestamp at forward pass start |
| 2 | `forward_end_time` | float | CPU timestamp at forward pass end |
| 3 | `sample_start_time` | float | CPU timestamp at sampling start |
| 4 | `sample_end_time` | float | CPU timestamp at sampling end |
| 5 | `gpu_forward_time` | float | GPU forward pass time (ms, via CUDA events) |
| 6 | `gpu_sample_time` | float | GPU sampling time (ms, via CUDA events) |
| 7 | `token_time` | float | Timestamp when token was generated |
| 8 | `iter` | int | Decoding iteration number (generation phase only) |

### Accumulated Context Metrics

| # | Field Name | Type | Description |
|---|------------|------|-------------|
| 1 | `ctx_gpu_forward_time` | float | Cumulative GPU forward time across context chunks (ms) |
| 2 | `ctx_gpu_sample_time` | float | Cumulative GPU sample time across context chunks (ms) |

---

## 6. Triton Backend Custom Metrics

**Source**: `triton_backend/inflight_batcher_llm/src/custom_metrics_reporter/custom_metrics_reporter.cc`
**Endpoint**: Triton server `/metrics` endpoint
**Labels**: `model`, `version` on all metrics; category-specific sub-labels

### Request Metrics (Gauge)

**Family**: `nv_trt_llm_request_metrics`
**Label**: `request_type`

| # | Label Value | Description |
|---|-------------|-------------|
| 1 | `active` | Active Request Count |
| 2 | `max` | Max Request Count |
| 3 | `scheduled` | Scheduled Requests |
| 4 | `context` | Context Requests |
| 5 | `waiting` | Waiting Requests |

### Runtime Memory Metrics (Gauge)

**Family**: `nv_trt_llm_runtime_memory_metrics`
**Label**: `memory_type`

| # | Label Value | Description |
|---|-------------|-------------|
| 1 | `cpu` | Runtime CPU Memory Usage |
| 2 | `gpu` | Runtime GPU Memory Usage |
| 3 | `pinned` | Runtime Pinned Memory Usage |

### KV Cache Block Metrics (Gauge)

**Family**: `nv_trt_llm_kv_cache_block_metrics`
**Label**: `kv_cache_block_type`

| # | Label Value | Description |
|---|-------------|-------------|
| 1 | `max` | Max KV cache blocks |
| 2 | `free` | Free KV cache blocks |
| 3 | `used` | Used KV cache blocks |
| 4 | `tokens_per` | Tokens per KV cache block |
| 5 | `reused` | Reused KV cache blocks |
| 6 | `fraction` | Fraction used KV cache blocks |

### Disaggregated Serving Metrics (Counter)

**Family**: `nv_trt_llm_disaggregated_serving_metrics`
**Label**: `disaggregated_serving_type`

| # | Label Value | Description |
|---|-------------|-------------|
| 1 | `kv_cache_transfer_ms` | KV cache transfer time (ms) |
| 2 | `request_count` | Request count |

### Model-Type Metrics — V1 (Gauge)

**Family**: `nv_trt_llm_v1_metrics`
**Label**: `v1_specific_metric`

| # | Label Value | Description |
|---|-------------|-------------|
| 1 | `total_context_tokens` | Total Context Tokens |
| 2 | `total_generation_tokens` | Total Generation Tokens |
| 3 | `empty_generation_slots` | Empty Generation Slots |

### Model-Type Metrics — Inflight Batching (Gauge)

**Family**: `nv_trt_llm_inflight_batcher_metrics`
**Label**: `inflight_batcher_specific_metric`

| # | Label Value | Description |
|---|-------------|-------------|
| 1 | `total_context_tokens` | Total Context Tokens |
| 2 | `generation_requests` | Generation Requests |
| 3 | `micro_batch_id` | MicroBatch ID |
| 4 | `paused_requests` | Paused Requests |

### General Metrics (Gauge)

**Family**: `nv_trt_llm_general_metrics`
**Label**: `general_type`

| # | Label Value | Description |
|---|-------------|-------------|
| 1 | `timestamp` | Timestamp |
| 2 | `iteration_counter` | Iteration Counter |

### Output Token Metrics (Histogram)

**Family**: `nv_llm_output_token_len`
**Label**: `response_metric_type`
**Buckets**: [10.0, 50.0, 100.0, 500.0, 1000.0]

| # | Label Value | Description |
|---|-------------|-------------|
| 1 | `total_output_tokens` | Total Output Tokens |

### Input Token Metrics (Histogram)

**Family**: `nv_llm_input_token_len`
**Label**: `response_metric_type`
**Buckets**: [10.0, 50.0, 100.0, 500.0, 1000.0]

| # | Label Value | Description |
|---|-------------|-------------|
| 1 | `total_input_tokens` | Total Input Tokens |

---

## 7. Cross-Reference: TRT-LLM vs SGLang vs vLLM

| Metric Concept | TRT-LLM (trtllm-serve) | TRT-LLM (Triton) | SGLang | vLLM |
|---|---|---|---|---|
| **Core Latency** | | | | |
| E2E request latency | `trtllm_e2e_request_latency_seconds` | - | `e2e_request_latency_seconds` | `e2e_request_latency_seconds` |
| Time to first token | `trtllm_time_to_first_token_seconds` | - | `time_to_first_token_seconds` | `time_to_first_token_seconds` |
| Inter-token latency | `trtllm_time_per_output_token_seconds` | - | `inter_token_latency_seconds` | `inter_token_latency_seconds` |
| Request queue time | `trtllm_request_queue_time_seconds` | - | `request_queue_time_seconds` | `request_queue_time_seconds` |
| Request inference time | - | - | `request_inference_time_seconds` | `request_inference_time_seconds` |
| Request prefill time | - | - | `request_prefill_time_seconds` | `request_prefill_time_seconds` |
| Request decode time | - | - | `request_decode_time_seconds` | `request_decode_time_seconds` |
| Per-request mean TPOT | - | - | `request_time_per_output_token_seconds` | `request_time_per_output_token_seconds` |
| **Request Counters** | | | | |
| Request success | `trtllm_request_success_total` | - | `request_success_total` | `request_success` |
| Preemptions | - | - | `num_preemptions_total` | `num_preemptions` |
| Prompt tokens total | - | `nv_llm_input_token_len` (hist) | `prompt_tokens_total` | `prompt_tokens` |
| Generation tokens total | - | `nv_llm_output_token_len` (hist) | `generation_tokens_total` | `generation_tokens` |
| **Queue/Load Gauges** | | | | |
| Running requests | `trtllm_num_requests_running` | `nv_trt_llm_request_metrics{active}` | `num_requests_running` | `num_requests_running` |
| Waiting requests | `trtllm_num_requests_waiting` | `nv_trt_llm_request_metrics{waiting}` | `num_requests_waiting` | `num_requests_waiting` |
| KV cache usage | `trtllm_kv_cache_utilization` | `nv_trt_llm_kv_cache_block_metrics{fraction}` | `kv_cache_usage_perc` | `kv_cache_usage_perc` |
| Generation throughput | - | - | `gen_throughput` | `gen_throughput` |
| **Cache** | | | | |
| KV cache hit rate | `trtllm_kv_cache_hit_rate` | (via JSON) | `cache_hit_rate` | (computed from prefix_cache_hits/queries) |
| KV cache blocks (max/free/used) | `trtllm_kv_cache_{max,free,used}_blocks` | `nv_trt_llm_kv_cache_block_metrics` | - | - |
| Prefix cache queries | - | - | - | `prefix_cache_queries` |
| Prefix cache hits | - | - | `cached_tokens_total` | `prefix_cache_hits` |
| **Memory** | | | | |
| GPU memory usage | `trtllm_gpu_memory_usage_bytes` | `nv_trt_llm_runtime_memory_metrics{gpu}` | - | - |
| CPU memory usage | `trtllm_cpu_memory_usage_bytes` | `nv_trt_llm_runtime_memory_metrics{cpu}` | - | - |
| Pinned memory usage | `trtllm_pinned_memory_usage_bytes` | `nv_trt_llm_runtime_memory_metrics{pinned}` | - | - |
| **Batch Sizing** | | | | |
| Max batch size (static) | `trtllm_max_batch_size_static` | - | - | - |
| Max batch size (runtime) | `trtllm_max_batch_size_runtime` | - | - | - |
| Max num tokens (runtime) | `trtllm_max_num_tokens_runtime` | - | - | - |
| **Speculative Decoding** | | | | |
| Num draft tokens | `trtllm_spec_decode_num_draft_tokens` | - | `spec_decode_num_drafts` | `spec_decode_num_drafts` |
| Num accepted tokens | `trtllm_spec_decode_num_accepted_tokens` | - | `spec_decode_num_accepted_tokens_per_pos` | `spec_decode_num_accepted_tokens` |
| Acceptance length | `trtllm_spec_decode_acceptance_length` | - | `spec_accept_length` | `spec_accept_length` |
| Acceptance rate | per-request: `acceptance_rate` | - | `spec_accept_rate` | `spec_accept_rate` |
| Draft overhead | `trtllm_spec_decode_draft_overhead` | - | - | - |
| **Disaggregated Serving** | | | | |
| KV transfer time | `{role}_*` metrics, per-request timing | `nv_trt_llm_disaggregated_serving_metrics` | `kv_transfer_latency_ms` | - |
| KV transfer size | per-request: `kv_cache_size` | - | `kv_transfer_total_mb` | - |
| **Inflight Batching** | | | | |
| Context requests | `trtllm_num_context_requests` | `nv_trt_llm_inflight_batcher_metrics` | - | - |
| Generation requests | `trtllm_num_generation_requests` | `nv_trt_llm_inflight_batcher_metrics` | - | - |
| Paused requests | `trtllm_num_paused_requests` | `nv_trt_llm_inflight_batcher_metrics` | `num_paused_reqs` | - |
| **Startup/Config** | | | | |
| Engine startup time | - | - | `engine_startup_time` | `engine_startup_time` |
| Engine load weights time | - | - | `engine_load_weights_time` | `engine_load_weights_time` |
| Model config info | `trtllm_model_config_info` | - | `model_config_info` | `model_config_info` |
| Parallel config info | `trtllm_parallel_config_info` | - | `parallel_config_info` | `parallel_config_info` |

---

## 8. Notes

- **Two metric systems**: TRT-LLM exposes metrics through two very different systems:
  1. **trtllm-serve** (Python, `tensorrt_llm/serve/`): Prometheus metrics at `/prometheus/metrics` + JSON iteration stats at `/metrics` + JSON per-request perf at `/perf_metrics`
  2. **Triton backend** (C++, `triton_backend/`): Custom Triton metrics via the Triton server `/metrics` endpoint

- **Prometheus metrics expanded**: The trtllm-serve Prometheus endpoint now has 36 metrics (2 counters, 4 histograms, 30 gauges), up from 7 in the initial implementation. Iteration-level stats (queue load, memory, batch sizes, KV cache blocks, inflight batching, speculative decoding) are now exposed as Prometheus gauges/counters alongside the original request-level histograms.

- **No throughput gauge**: TRT-LLM does not expose a `gen_throughput` Prometheus gauge (both SGLang and vLLM do).

- **No prompt/generation token counters**: TRT-LLM does not expose cumulative prompt or generation token counters as Prometheus metrics. The Triton backend provides token length histograms (`nv_llm_input_token_len`, `nv_llm_output_token_len`) but not running counters.

- **Disaggregated serving has its own metric set**: When running in disaggregated mode, additional role-prefixed metrics (`ctx_`, `gen_`, `mme_`) are registered, providing per-component latency and error tracking.

- **Per-request perf metrics are detailed**: The `/perf_metrics` endpoint provides fine-grained per-request timing, KV cache reuse stats, and speculative decoding acceptance rates, which are richer than what's available via Prometheus.

- **Prefix**: trtllm-serve uses `trtllm_` prefix. Triton backend uses `nv_trt_llm_` prefix (or `nv_llm_` for token histograms).

- **Multiprocess mode**: trtllm-serve uses Prometheus multiprocess mode (`PROMETHEUS_MULTIPROC_DIR` must be set before `prometheus_client` is imported).

- **Still missing compared to SGLang/vLLM**: No LoRA metrics, no grammar/structured output metrics, no routing key metrics, no MoE balancedness metrics, no cache eviction metrics, no HTTP request duration histogram, no `gen_throughput` gauge, no prompt/generation token counters, no request-level latency histograms (`request_inference_time`, `request_prefill_time`, `request_decode_time`).

---

## Summary Table

| Metric Category | Metric System | Count | Endpoint |
|---|---|---|---|
| Prometheus (request + iteration-level + config) | trtllm-serve | 40 | `/prometheus/metrics` |
| Disaggregated client metrics | trtllm-serve | 21 (7 x 3 roles) | `/prometheus/metrics` |
| Disaggregated server metrics | trtllm-serve | 8 | `/prometheus/metrics` |
| Iteration stats (JSON) | trtllm-serve | 18 top-level + 9 KV + 7 IFB + 5 static + 6 spec = ~45 | `/metrics` |
| Per-request perf metrics (JSON) | trtllm-serve | ~18 fields per request | `/perf_metrics` |
| PyExecutor internal timing | Internal only | 10 fields per request | Not exposed |
| Triton custom metrics | Triton backend | ~28 metric series | Triton `/metrics` |

## Key Source Files

| File | Role |
|------|------|
| `tensorrt_llm/metrics/collector.py` | Main Prometheus MetricsCollector |
| `tensorrt_llm/metrics/enums.py` | MetricNames and RequestEventTiming enums |
| `tensorrt_llm/serve/perf_metrics.py` | Disaggregated serving metrics |
| `tensorrt_llm/serve/openai_server.py` | Server integration, route registration, stats collector loop |
| `tensorrt_llm/_torch/pyexecutor/perf_metrics_manager.py` | PyExecutor GPU/CPU timing instrumentation |
| `cpp/include/tensorrt_llm/executor/types.h` | C++ structs: IterationStats, KvCacheStats, RequestPerfMetrics, etc. |
| `triton_backend/inflight_batcher_llm/src/custom_metrics_reporter/custom_metrics_reporter.cc` | Triton backend custom metrics |
