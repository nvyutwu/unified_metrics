# vLLM Metrics Inventory

> Auto-generated from source code analysis of vLLM `feat/unified-metrics-v2-0.16.0` branch.

---

## Table of Contents
1. [Prometheus Metrics (PrometheusStatLogger)](#1-prometheus-metrics-prometheusstatlogger)
   - [Queue/Load Gauges](#queueload-gauges)
   - [KV Cache Gauges](#kv-cache-gauges)
   - [Throughput & Startup Gauges](#throughput--startup-gauges)
   - [Cache Counters](#cache-counters)
   - [Token Counters](#token-counters)
   - [Request Counters](#request-counters)
   - [Token Count Histograms](#token-count-histograms)
   - [Latency Histograms](#latency-histograms)
   - [KV Cache Residency Histograms (optional)](#kv-cache-residency-histograms-optional)
   - [LoRA Metrics (conditional)](#lora-metrics-conditional)
   - [Config Info Metrics](#config-info-metrics)
2. [Speculative Decoding Metrics](#2-speculative-decoding-metrics)
3. [Request Type Classification Metrics](#3-request-type-classification-metrics)
4. [KV Connector Metrics (per-connector)](#4-kv-connector-metrics-per-connector)
5. [HTTP-Level Metrics](#5-http-level-metrics)
6. [OpenTelemetry Bridge](#6-opentelemetry-bridge)
   - [Metric Name Mapping](#metric-name-mapping)
7. [Cross-Reference: vLLM vs SGLang vs TRT-LLM](#7-cross-reference-vllm-vs-sglang-vs-trt-llm)
8. [Notes](#8-notes)

---

## 1. Prometheus Metrics (PrometheusStatLogger)

**Source**: `vllm/v1/metrics/loggers.py` (PrometheusStatLogger class)
**Endpoint**: `/metrics`
**Prefix**: `vllm:`
**Labels**: `model_name`, `engine` on all per-engine metrics
**Multiprocess Mode**: Single-process (frontend); gauges use `mostrecent` mode

### Queue/Load Gauges

| # | Metric Name | Multiprocess Mode | Description |
|---|-------------|-------------------|-------------|
| 1 | `vllm:num_requests_running` | mostrecent | Number of requests in model execution batches |
| 2 | `vllm:num_requests_waiting` | mostrecent | Number of requests waiting to be processed |
| 3 | `vllm:engine_sleep_state` | mostrecent | Engine sleep state (labels: `sleep_state` = awake / weights_offloaded / discard_all) |

### KV Cache Gauges

| # | Metric Name | Multiprocess Mode | Description |
|---|-------------|-------------------|-------------|
| 4 | `vllm:kv_cache_usage_perc` | mostrecent | KV-cache usage. 1 means 100 percent usage |

### Throughput & Startup Gauges

| # | Metric Name | Multiprocess Mode | Description |
|---|-------------|-------------------|-------------|
| 5 | `vllm:gen_throughput` | mostrecent | Generation throughput in tokens per second (~5s update interval) |
| 6 | `vllm:engine_startup_time` | mostrecent | Time taken for the engine to start up in seconds |
| 7 | `vllm:engine_load_weights_time` | mostrecent | Time taken for the engine to load weights in seconds |

### Cache Counters

| # | Metric Name | Extra Labels | Description |
|---|-------------|-------------|-------------|
| 8 | `vllm:prefix_cache_queries` | — | Prefix cache queries, in terms of number of queried tokens |
| 9 | `vllm:prefix_cache_hits` | — | Prefix cache hits, in terms of number of cached tokens |
| 10 | `vllm:external_prefix_cache_queries` | — | External prefix cache queries from KV connector cross-instance cache sharing |
| 11 | `vllm:external_prefix_cache_hits` | — | External prefix cache hits from KV connector cross-instance cache sharing |
| 12 | `vllm:mm_cache_queries` | — | Multi-modal cache queries, in terms of number of queried items |
| 13 | `vllm:mm_cache_hits` | — | Multi-modal cache hits, in terms of number of cached items |
| 14 | `vllm:corrupted_requests` | — | Corrupted requests with NaNs in logits (conditional: `VLLM_COMPUTE_NANS_IN_LOGITS` env var) |

### Token Counters

| # | Metric Name | Extra Labels | Description |
|---|-------------|-------------|-------------|
| 15 | `vllm:num_preemptions` | — | Cumulative number of retracted (preempted) requests from the engine |
| 16 | `vllm:prompt_tokens` | — | Number of prefill tokens processed |
| 17 | `vllm:prompt_tokens_by_source` | `source` (local_compute, local_cache_hit, external_kv_transfer) | Number of prompt tokens by source |
| 18 | `vllm:prompt_tokens_cached` | — | Number of cached prompt tokens (local + external) |
| 19 | `vllm:prompt_tokens_recomputed` | — | Number of cached tokens recomputed for forward pass |
| 20 | `vllm:generation_tokens` | — | Number of generation tokens processed |

### Request Counters

| # | Metric Name | Extra Labels | Description |
|---|-------------|-------------|-------------|
| 21 | `vllm:request_success` | `finished_reason` | Count of successfully processed requests |
| 22 | `vllm:num_aborted_requests` | — | Number of requests aborted |

### Token Count Histograms

| # | Metric Name | Buckets | Description |
|---|-------------|---------|-------------|
| 23 | `vllm:request_prompt_tokens` | build_1_2_5_buckets(max_model_len) | Number of prefill tokens processed per request |
| 24 | `vllm:request_generation_tokens` | build_1_2_5_buckets(max_model_len) | Number of generation tokens processed per request |
| 25 | `vllm:iteration_tokens_total` | [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] | Number of tokens per engine_step |
| 26 | `vllm:request_max_num_generation_tokens` | build_1_2_5_buckets(max_model_len) | Maximum number of requested generation tokens |
| 27 | `vllm:request_params_n` | [1, 2, 5, 10, 20] | The n request parameter distribution |
| 28 | `vllm:request_params_max_tokens` | build_1_2_5_buckets(max_model_len) | The max_tokens request parameter distribution |
| 29 | `vllm:num_retractions` | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 75, 100] | Preemption counts per request |

### Latency Histograms

| # | Metric Name | Buckets | Description |
|---|-------------|---------|-------------|
| 30 | `vllm:time_to_first_token_seconds` | [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0] | Time to first token in seconds |
| 31 | `vllm:inter_token_latency_seconds` | [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0] | Inter-token latency in seconds |
| 32 | `vllm:request_time_per_output_token_seconds` | [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0] | Time_per_output_token_seconds per request |
| 33 | `vllm:e2e_request_latency_seconds` | [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0] | End-to-end request latency in seconds |
| 34 | `vllm:request_queue_time_seconds` | [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0] | Time spent in WAITING phase for request |
| 35 | `vllm:request_inference_time_seconds` | [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0] | Time spent in RUNNING phase for request |
| 36 | `vllm:request_prefill_time_seconds` | [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0] | Time spent in PREFILL phase for request |
| 37 | `vllm:request_decode_time_seconds` | [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0] | Time spent in DECODE phase for request |
| 38 | `vllm:request_prefill_kv_computed_tokens` | build_1_2_5_buckets(max_model_len) | New KV tokens computed during prefill (excluding cached tokens) |

### KV Cache Residency Histograms (optional)

**Condition**: Only created if `kv_cache_metrics_enabled` is True (controlled by `--kv-cache-metrics-sample`)

| # | Metric Name | Buckets | Description |
|---|-------------|---------|-------------|
| 39 | `vllm:kv_block_lifetime_seconds` | [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800] | KV cache block lifetime from allocation to eviction (sampled) |
| 40 | `vllm:kv_block_idle_before_evict_seconds` | [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800] | Idle time before KV cache block eviction (sampled) |
| 41 | `vllm:kv_block_reuse_gap_seconds` | [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800] | Time gaps between consecutive KV cache block accesses (sampled) |

### LoRA Metrics (conditional)

**Condition**: Only created if LoRA configuration is enabled

| # | Metric Name | Type | Extra Labels | Description |
|---|-------------|------|-------------|-------------|
| 42 | `vllm:lora_requests_info` | Gauge | max_lora, waiting_lora_adapters, running_lora_adapters | Running stats on LoRA requests |
| 43 | `vllm:lora_pool_utilization` | Gauge | — | LoRA adapter pool utilization as a ratio (0-1). active_adapters / max_loras |

### Config Info Metrics

All are Gauge type emulating Info metrics (value always 1.0), with dynamic labels from config objects.

| # | Metric Name | Description |
|---|-------------|-------------|
| 44 | `vllm:cache_config_info` | Information of the LLMEngine CacheConfig |
| 45 | `vllm:model_config_info` | Information of the LLMEngine ModelConfig |
| 46 | `vllm:parallel_config_info` | Information of the LLMEngine ParallelConfig |
| 47 | `vllm:speculative_config_info` | Information of the LLMEngine SpeculativeConfig (conditional) |
| 48 | `vllm:detailed_config_info` | Additional engine configuration details (scheduler, compilation, attention, env settings) |

---

## 2. Speculative Decoding Metrics

**Source**: `vllm/v1/spec_decode/metrics.py`
**Condition**: Only created if speculative decoding is enabled

### Counters

| # | Metric Name | Extra Labels | Description |
|---|-------------|-------------|-------------|
| 49 | `vllm:spec_decode_num_drafts` | — | Number of spec decoding drafts |
| 50 | `vllm:spec_decode_num_draft_tokens` | — | Number of draft tokens |
| 51 | `vllm:spec_decode_num_accepted_tokens` | — | Number of accepted tokens |
| 52 | `vllm:spec_decode_num_accepted_tokens_per_pos` | `position` (0 to num_speculative_tokens-1) | Accepted tokens per draft position |

### Gauges

| # | Metric Name | Multiprocess Mode | Description |
|---|-------------|-------------------|-------------|
| 53 | `vllm:spec_accept_length` | mostrecent | Average accepted sequence length in speculative decoding (including bonus token) |
| 54 | `vllm:spec_accept_rate` | mostrecent | Draft acceptance rate (accepted_tokens / draft_tokens, 0-1) |

---

## 3. Request Type Classification Metrics

**Source**: `vllm/entrypoints/openai/request_metrics.py`
**Scope**: Incremented across `/v1/chat/completions`, `/v1/completions`, and `/v1/responses`

| # | Metric Name | Type | Description |
|---|-------------|------|-------------|
| 55 | `vllm:request_type_image_total` | Counter | Total requests containing images |
| 56 | `vllm:request_type_video_total` | Counter | Total requests containing videos |
| 57 | `vllm:request_type_tool_call_total` | Counter | Total requests with tool calls enabled |
| 58 | `vllm:request_type_structured_output_total` | Counter | Total requests with structured output |

---

## 4. KV Connector Metrics (per-connector)

**Source**: `vllm/distributed/kv_transfer/kv_connector/v1/metrics.py`

KV Connector metrics are dynamically registered by the connector implementation. The framework provides:
- **Base Class**: `KVConnectorPromMetrics` — abstract base for per-connector metric registration
- **Factory**: `KVConnectorPrometheus` — support class that delegates to connector-specific implementations
- **Supported Types**: Gauge, Counter, Histogram

### NIXL Connector Metrics

| # | Metric Name | Type | Description |
|---|-------------|------|-------------|
| — | `nixl_xfer_time_seconds` | Histogram | Transfer duration per NIXL KV cache transfer |
| — | `nixl_post_time_seconds` | Histogram | Post-transfer processing time |
| — | `nixl_bytes_transferred` | Histogram | Bytes transferred per transfer |
| — | `nixl_num_descriptors` | Histogram | Number of descriptors per transfer |
| — | `nixl_num_failed_transfers` | Counter | Failed NIXL transfers |
| — | `nixl_num_failed_notifications` | Counter | Failed NIXL notifications |
| — | `nixl_num_kv_expired_reqs` | Counter | Requests with expired KV (P instance) |
| — | `kv_transfer_speed_gb_s` | Gauge | KV transfer speed in GB/s (SGLang-compatible) |

---

## 5. HTTP-Level Metrics

**Source**: `vllm/entrypoints/serve/instrumentator/metrics.py`
**Library**: `prometheus_fastapi_instrumentator`
**Excluded Endpoints**: `/metrics`, `/health`, `/load`, `/ping`, `/version`, `/server_info`

| # | Metric Name | Type | Labels | Description |
|---|-------------|------|--------|-------------|
| — | `http_requests_total` | Counter | method, status, handler | HTTP request count |
| — | `http_request_duration_seconds` | Histogram | method, handler | HTTP request duration |
| — | `http_requests_in_progress` | Gauge | — | In-flight HTTP requests |
| — | `http_request_size_bytes` | Summary | — | Request body size |
| — | `http_response_size_bytes` | Summary | — | Response body size |

---

## 6. OpenTelemetry Bridge

**Source**: `vllm/otel_instrumentation.py`
**Function**: `start_prom_to_otel_bridge()`

The bridge scrapes Prometheus metrics at a configurable interval and re-exports them as OpenTelemetry instruments. It handles:
- **Counter Translation**: Delta calculation from Prometheus counter values to OTel counter increments
- **Gauge Translation**: Direct mapping to OTel observable gauges
- **Histogram Translation**: Breakdown into _count, _sum, and _bucket counters
- **Labels-to-Attributes**: Direct mapping of label key-value pairs to OTel attributes

### Metric Name Mapping

The bridge uses `_sanitize_metric_name()` with an explicit `_METRIC_NAME_MAP` dictionary. The default behavior strips the `vllm_` prefix. Explicit mappings handle counter `_total` suffix alignment:

| Prometheus Name (native) | OTel Name (unified) |
|---|---|
| `vllm_prompt_tokens` | `prompt_tokens_total` |
| `vllm_generation_tokens` | `generation_tokens_total` |
| `vllm_request_success` | `request_success_total` |
| `vllm_num_preemptions` | `num_preemptions_total` |
| `vllm_prefix_cache_queries` | `prefix_cache_queries_total` |
| `vllm_prefix_cache_hits` | `prefix_cache_hits_total` |
| `vllm_external_prefix_cache_queries` | `external_prefix_cache_queries_total` |
| `vllm_external_prefix_cache_hits` | `external_prefix_cache_hits_total` |
| `vllm_mm_cache_queries` | `mm_cache_queries_total` |
| `vllm_mm_cache_hits` | `mm_cache_hits_total` |
| `vllm_corrupted_requests` | `corrupted_requests_total` |
| `vllm_num_aborted_requests` | `num_aborted_requests_total` |
| `vllm_request_type_image_total` | `request_type_image_total` |
| `vllm_request_type_video_total` | `request_type_video_total` |
| `vllm_request_type_tool_call_total` | `request_type_tool_call_total` |
| `vllm_request_type_structured_output_total` | `request_type_structured_output_total` |
| `vllm_spec_decode_num_drafts` | `spec_decode_num_drafts_total` |
| `vllm_spec_decode_num_draft_tokens` | `spec_decode_num_draft_tokens_total` |
| `vllm_spec_decode_num_accepted_tokens` | `spec_decode_num_accepted_tokens_total` |
| `vllm_spec_decode_num_accepted_tokens_per_pos` | `spec_decode_num_accepted_tokens_per_pos_total` |

Unlisted metrics pass through with `vllm_` prefix stripped (e.g., `vllm_e2e_request_latency_seconds` → `e2e_request_latency_seconds`).

---

## 7. Cross-Reference: vLLM vs SGLang vs TRT-LLM

| Metric Concept | vLLM | SGLang | TRT-LLM |
|---|---|---|---|
| **Core Latency** | | | |
| E2E request latency | `e2e_request_latency_seconds` | `e2e_request_latency_seconds` | `trtllm_e2e_request_latency_seconds` |
| Time to first token | `time_to_first_token_seconds` | `time_to_first_token_seconds` | `trtllm_time_to_first_token_seconds` |
| Inter-token latency | `inter_token_latency_seconds` | `inter_token_latency_seconds` | `trtllm_time_per_output_token_seconds` |
| Request queue time | `request_queue_time_seconds` | `request_queue_time_seconds` | `trtllm_request_queue_time_seconds` |
| Request inference time | `request_inference_time_seconds` | `request_inference_time_seconds` | — |
| Request prefill time | `request_prefill_time_seconds` | `request_prefill_time_seconds` | — |
| Request decode time | `request_decode_time_seconds` | `request_decode_time_seconds` | — |
| Per-request mean TPOT | `request_time_per_output_token_seconds` | `request_time_per_output_token_seconds` | — |
| **Token Counters** | | | |
| Prompt tokens | `prompt_tokens` | `prompt_tokens_total` | — |
| Generation tokens | `generation_tokens` | `generation_tokens_total` | — |
| Prompt tokens by source | `prompt_tokens_by_source` | — | — |
| Prompt tokens cached | `prompt_tokens_cached` | `cached_tokens_total` | — |
| Prompt tokens recomputed | `prompt_tokens_recomputed` | — | — |
| **Request Counters** | | | |
| Request success | `request_success` | `request_success_total` | `trtllm_request_success_total` |
| Preemptions | `num_preemptions` | `num_retracted_requests_total` | — |
| Aborted requests | `num_aborted_requests` | `num_aborted_requests_total` | — |
| **Queue/Load Gauges** | | | |
| Running requests | `num_requests_running` | `num_requests_running` | JSON: `numActiveRequests` |
| Waiting requests | `num_requests_waiting` | `num_requests_waiting` | JSON: `numQueuedRequests` |
| KV cache usage | `kv_cache_usage_perc` | `kv_cache_usage_perc` | `trtllm_kv_cache_utilization` |
| Generation throughput | `gen_throughput` | `gen_throughput` | — |
| **Cache** | | | |
| Prefix cache queries | `prefix_cache_queries` | — | — |
| Prefix cache hits | `prefix_cache_hits` | `cached_tokens_total` | — |
| Cache hit rate | (computed from queries/hits) | `cache_hit_rate` | `trtllm_kv_cache_hit_rate` |
| External cache queries | `external_prefix_cache_queries` | — | — |
| External cache hits | `external_prefix_cache_hits` | — | — |
| MM cache queries | `mm_cache_queries` | `mm_cache_queries` | — |
| MM cache hits | `mm_cache_hits` | `mm_cache_hits` | — |
| **Speculative Decoding** | | | |
| Num drafts | `spec_decode_num_drafts` | `spec_decode_num_drafts` | — |
| Num draft tokens | `spec_decode_num_draft_tokens` | — | JSON: `specDecodingStats.numDraftTokens` |
| Num accepted tokens | `spec_decode_num_accepted_tokens` | — | JSON: `specDecodingStats.numAcceptedTokens` |
| Per-position accepted | `spec_decode_num_accepted_tokens_per_pos` | `spec_decode_num_accepted_tokens_per_pos` | — |
| Accept length | `spec_accept_length` | `spec_accept_length` | JSON: `specDecodingStats.acceptanceLength` |
| Accept rate | `spec_accept_rate` | `spec_accept_rate` | per-request: `acceptance_rate` |
| **LoRA** | | | |
| Pool utilization | `lora_pool_utilization` | `lora_pool_utilization` | — |
| Request info | `lora_requests_info` | — | — |
| **Startup/Config** | | | |
| Engine startup time | `engine_startup_time` | `engine_startup_time` | — |
| Engine load weights time | `engine_load_weights_time` | `engine_load_weights_time` | — |
| Model config info | `model_config_info` | `model_config_info` | — |
| Parallel config info | `parallel_config_info` | `parallel_config_info` | — |
| Speculative config info | `speculative_config_info` | `speculative_config_info` | — |
| Detailed config info | `detailed_config_info` | `detailed_config_info` | — |
| Cache config info | `cache_config_info` | — | — |

---

## 8. Notes

- **Prefix**: All vLLM Prometheus metrics use the `vllm:` prefix (colon separator). In the OTel bridge, the colon is replaced with underscore (`vllm_`) before name mapping.

- **Counter naming**: vLLM counters do NOT use the `_total` suffix natively (e.g., `vllm:prompt_tokens`). The `_total` suffix is applied only in the OTel bridge output.

- **Prompt token breakdown**: The `prompt_tokens_by_source` counter provides a breakdown of prompt tokens into `local_compute`, `local_cache_hit`, and `external_kv_transfer` sources. The `prompt_tokens_cached` counter aggregates local + external cached tokens, while `prompt_tokens_recomputed` tracks tokens recomputed despite being cached (e.g., last token recompute when entire prompt is cached).

- **Preemption naming**: vLLM uses `num_preemptions` (counter) and `num_retractions` (histogram). SGLang uses `num_retracted_requests_total` (counter) and `num_retractions` (histogram). The histogram name is aligned but the counter name differs.

- **KV cache residency metrics**: These are optional sampled metrics controlled by `--kv-cache-metrics-sample`. They provide deep insight into cache block lifecycle but add overhead.

- **KV Connector metrics**: These are dynamically registered per-connector. The NIXL connector is the most instrumented. Other connectors may register different metrics via the `build_prom_metrics()` interface.

- **OTel bridge mapping**: New metrics added to vLLM that are counters should have entries added to `_METRIC_NAME_MAP` in `otel_instrumentation.py` for proper `_total` suffix alignment. Currently `prompt_tokens_by_source`, `prompt_tokens_cached`, and `prompt_tokens_recomputed` are NOT in the mapping (they get prefix-stripped only).

- **Missing compared to SGLang**: No grammar/structured output metrics, no routing key metrics, no MoE balancedness metrics, no CUDA graph metrics, no cache eviction metrics, no PD disaggregation queue depth metrics, no retracted token counters (input/output), no prefill delayer metrics, no composite utilization gauge.

---

## Summary Table

| Metric Category | Count | Source File |
|---|---|---|
| Queue/Load/Throughput Gauges | 7 | `vllm/v1/metrics/loggers.py` |
| Cache Counters | 7 | `vllm/v1/metrics/loggers.py` |
| Token Counters | 6 | `vllm/v1/metrics/loggers.py` |
| Request Counters | 2 | `vllm/v1/metrics/loggers.py` |
| Token Count Histograms | 7 | `vllm/v1/metrics/loggers.py` |
| Latency Histograms | 9 | `vllm/v1/metrics/loggers.py` |
| KV Cache Residency Histograms (optional) | 3 | `vllm/v1/metrics/loggers.py` |
| LoRA Metrics (conditional) | 2 | `vllm/v1/metrics/loggers.py` |
| Config Info Metrics | 5 | `vllm/v1/metrics/loggers.py` |
| Speculative Decoding | 6 | `vllm/v1/spec_decode/metrics.py` |
| Request Type Classification | 4 | `vllm/entrypoints/openai/request_metrics.py` |
| KV Connector (NIXL) | ~8 | `vllm/distributed/kv_transfer/kv_connector/v1/metrics.py` |
| HTTP-Level | ~5 | `vllm/entrypoints/serve/instrumentator/metrics.py` |
| **Total (core, excl. HTTP & connector)** | **58** | |

## Key Source Files

| File | Role |
|------|------|
| `vllm/v1/metrics/loggers.py` | Main PrometheusStatLogger — all core metric definitions |
| `vllm/v1/metrics/stats.py` | Stats data structures (IterationStats, FinishedRequestStats, etc.) |
| `vllm/v1/metrics/prometheus.py` | Prometheus registry setup |
| `vllm/v1/spec_decode/metrics.py` | Speculative decoding metric definitions |
| `vllm/entrypoints/openai/request_metrics.py` | Request type classification counters |
| `vllm/distributed/kv_transfer/kv_connector/v1/metrics.py` | KV connector metrics framework |
| `vllm/otel_instrumentation.py` | OpenTelemetry initialization & Prom-to-OTel bridge |
| `vllm/entrypoints/serve/instrumentator/metrics.py` | HTTP-level metrics (prometheus_fastapi_instrumentator) |
| `vllm/config/*.py` | Config info metric label definitions |
