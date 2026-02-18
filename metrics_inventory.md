# Combined Metrics Inventory: SGLang + vLLM

> Auto-generated from source code analysis of both `feat/unified-metrics` branches.

---

## Table of Contents
1. [SGLang Metrics](#1-sglang-metrics)
   - [Scheduler Metrics](#scheduler-metrics)
   - [Tokenizer Metrics](#tokenizer-metrics)
   - [Storage Metrics](#storage-metrics)
   - [Radix Cache Metrics](#radix-cache-metrics)
   - [Expert Dispatch Metrics](#expert-dispatch-metrics)
   - [Function-Level Metrics](#function-level-metrics)
   - [Startup Metrics](#startup-metrics)
   - [System Metrics](#system-metrics)
   - [Config Info Metrics (SGLang)](#config-info-metrics-sglang)
   - [HTTP & Request Type Metrics (SGLang)](#http--request-type-metrics-sglang)
2. [vLLM Metrics](#2-vllm-metrics)
   - [Scheduler State Metrics](#scheduler-state-metrics)
   - [Throughput & Startup Metrics](#throughput--startup-metrics)
   - [Cache Metrics](#cache-metrics)
   - [Request Counters](#request-counters)
   - [Token Histograms](#token-histograms)
   - [Timing Histograms (Latency)](#timing-histograms-latency)
   - [Request Latency Histograms](#request-latency-histograms)
   - [KV Cache Residency Metrics](#kv-cache-residency-metrics)
   - [LoRA Metrics](#lora-metrics)
   - [Speculative Decoding Metrics](#speculative-decoding-metrics)
   - [Config Info Metrics (vLLM)](#config-info-metrics-vllm)
   - [HTTP & Request Type Metrics (vLLM)](#http--request-type-metrics-vllm)
3. [Cross-Reference Quick Lookup Table](#3-cross-reference-quick-lookup-table)
4. [Notes](#4-notes)

---

## 1. SGLang Metrics

**Total**: ~110 metrics (40 Gauges, 29 Counters, 28 Histograms, 1 Summary, 2 Gauge Histograms + HTTP/request type)

### Scheduler Metrics

**Source**: `python/sglang/srt/metrics/collector.py` (SchedulerMetricsCollector)
**Common Labels**: `model_name`, `engine_type`, `tp_rank`, `pp_rank`, `moe_ep_rank`, `experiment_group`

#### Gauges

| # | Metric Name | Multiprocess Mode | Description |
|---|-------------|-------------------|-------------|
| 1 | `num_requests_running` | mostrecent | Active requests in execution batches |
| 2 | `num_requests_waiting` | mostrecent | Requests in waiting queue |
| 3 | `num_used_tokens` | mostrecent | Tokens allocated in KV cache |
| 4 | `kv_cache_usage_perc` | mostrecent | KV cache usage ratio (0-1) |
| 5 | `pending_prealloc_token_usage` | mostrecent | Tokens pending pre-allocation (disaggregation) |
| 6 | `swa_token_usage` | mostrecent | Sliding window attention token usage |
| 7 | `mamba_usage` | mostrecent | Mamba SSM layer token usage % |
| 8 | `decode_sum_seq_lens` | mostrecent | Sum of all sequence lengths in decode |
| 9 | `gen_throughput` | mostrecent | Generation throughput (tokens/s) |
| 10 | `num_grammar_queue_reqs` | mostrecent | Requests in grammar processing queue |
| 11 | `num_running_reqs_offline_batch` | mostrecent | Low-priority offline batch requests |
| 12 | `cache_hit_rate` | mostrecent | Prefix cache hit ratio (0-1) |
| 13 | `max_total_num_tokens` | mostrecent | Max KV cache pool capacity |
| 14 | `spec_accept_length` | mostrecent | Avg accepted tokens in spec decoding |
| 15 | `spec_accept_rate` | mostrecent | Spec decode acceptance rate (0-1) |
| 16 | `num_retracted_reqs` | default | Current retracted request count |
| 17 | `num_paused_reqs` | default | Requests paused by async weight sync |
| 18 | `num_prefill_prealloc_queue_reqs` | mostrecent | Prefill disaggregation prealloc queue |
| 19 | `num_prefill_inflight_queue_reqs` | mostrecent | Prefill disaggregation inflight queue |
| 20 | `num_decode_prealloc_queue_reqs` | mostrecent | Decode disaggregation prealloc queue |
| 21 | `num_decode_transfer_queue_reqs` | mostrecent | Decode disaggregation transfer queue |
| 22 | `kv_transfer_speed_gb_s` | mostrecent | KV cache transfer bandwidth (GB/s) |
| 23 | `kv_transfer_latency_ms` | mostrecent | KV transfer latency (ms) |
| 24 | `kv_transfer_bootstrap_ms` | mostrecent | KV transfer bootstrap time (ms) |
| 25 | `kv_transfer_alloc_ms` | mostrecent | KV transfer allocation wait (ms) |
| 26 | `kv_transfer_total_mb` | mostrecent | Total KV cache transferred (MB) |
| 27 | `utilization` | mostrecent | Composite utilization metric |
| 28 | `max_running_requests_under_SLO` | mostrecent | Max concurrent requests within SLO |
| 29 | `engine_startup_time` | mostrecent | Engine initialization time (seconds) |
| 30 | `engine_load_weights_time` | mostrecent | Weight loading duration (seconds) |
| 31 | `is_cuda_graph` | mostrecent | 1.0 if using CUDA graph, else 0.0 |
| 32 | `lora_pool_slots_used` | mostrecent | LoRA adapter slots in use (GPU) |
| 33 | `lora_pool_slots_total` | mostrecent | Total LoRA slots (max_loras_per_batch) |
| 34 | `lora_pool_utilization` | mostrecent | LoRA pool usage ratio (0-1) |
| 35 | `num_unique_running_routing_keys` | mostrecent | Count of unique routing keys in batch |
| 36 | `new_token_ratio` | mostrecent | Ratio of new tokens vs cached |

#### Counters

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `num_retracted_requests_total` | Retracted/preempted requests |
| 2 | `num_preemptions_total` | Preemptions (vLLM-compatible alias) |
| 3 | `num_retracted_input_tokens_total` | Input tokens from retracted requests |
| 4 | `num_retracted_output_tokens_total` | Output tokens from retracted requests |
| 5 | `num_bootstrap_failed_reqs_total` | Bootstrap failures (disaggregation) |
| 6 | `num_transfer_failed_reqs_total` | KV transfer failures (disaggregation) |
| 7 | `num_prefill_retries_total` | Prefill request retries |
| 8 | `num_grammar_cache_hit_total` | Grammar cache hits |
| 9 | `num_grammar_aborted_total` | Grammar requests aborted |
| 10 | `num_grammar_timeout_total` | Grammar processing timeouts |
| 11 | `num_grammar_total` | Total grammar requests |
| 12 | `realtime_tokens_total` | Tokens by mode (labels: `mode`) |
| 13 | `gpu_execution_seconds_total` | GPU execution time (labels: `category`) |
| 14 | `dp_cooperation_realtime_tokens_total` | Tokens with DP cooperation info |
| 15 | `dp_cooperation_gpu_execution_seconds_total` | GPU exec time with DP info |
| 16 | `prefill_delayer_outcomes_total` | Prefill delayer decisions |
| 17 | `cuda_graph_passes_total` | Forward passes by CUDA graph usage |
| 18 | `spec_decode_num_drafts` | Draft attempt count |
| 19 | `spec_decode_num_accepted_tokens_per_pos` | Accepted tokens per draft position |

#### Histograms

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `request_queue_time_seconds` | Queue time before execution |
| 2 | `grammar_compilation_time_seconds` | Grammar rule compilation time |
| 3 | `grammar_schema_count` | Number of schemas in grammar |
| 4 | `grammar_ebnf_size` | EBNF grammar size (bytes) |
| 5 | `grammar_tree_traversal_time_avg` | Avg grammar tree traversal time |
| 6 | `grammar_tree_traversal_time_max` | Max grammar tree traversal time |
| 7 | `per_stage_req_latency_seconds` | Per-stage latency (label: `stage`) |
| 8 | `prefill_delayer_wait_forward_passes` | Forward passes delayed |
| 9 | `prefill_delayer_wait_seconds` | Delay duration (seconds) |

#### Summary

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `eplb_balancedness` | Expert load balancing metric (MoE only) |

#### Gauge Histograms

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `routing_key_running_req_count` | Running requests by routing key |
| 2 | `routing_key_all_req_count` | All requests by routing key |

---

### Tokenizer Metrics

**Source**: `python/sglang/srt/metrics/collector.py` (TokenizerMetricsCollector)
**Common Labels**: `model_name`, `experiment_group`

#### Counters

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `prompt_tokens_total` | Prompt/prefill tokens processed |
| 2 | `generation_tokens_total` | Generation tokens processed |
| 3 | `cached_tokens_total` | Tokens served from prefix cache |
| 4 | `num_requests_total` | Total finished requests |
| 5 | `num_so_requests_total` | Structured output requests |
| 6 | `num_aborted_requests_total` | Aborted requests |
| 7 | `request_success_total` | Requests by finish reason (label: `finished_reason`) |

#### Histograms

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `time_to_first_token_seconds` | Time to first token |
| 2 | `inter_token_latency_seconds` | Inter-token latency |
| 3 | `e2e_request_latency_seconds` | End-to-end request latency |
| 4 | `num_retractions` | Retraction count per request |
| 5 | `request_inference_time_seconds` | Total inference time |
| 6 | `request_prefill_time_seconds` | Prefill phase time |
| 7 | `request_decode_time_seconds` | Decode phase time |
| 8 | `request_prompt_tokens` | Prompt tokens per request |
| 9 | `request_generation_tokens` | Generation tokens per request |
| 10 | `prompt_tokens_histogram` | Prompt token distribution (optional, configurable) |
| 11 | `generation_tokens_histogram` | Generation token distribution (optional, configurable) |
| 12 | `request_time_per_output_token_seconds` | Per-request mean TPOT |
| 13 | `request_params_max_tokens` | Request max_tokens parameter distribution |
| 14 | `mm_cache_queries` | Multi-modal cache queries |
| 15 | `mm_cache_hits` | Multi-modal cache hits |
| 16 | `http_request_duration_seconds` | HTTP request duration |

---

### Storage Metrics

**Source**: `python/sglang/srt/metrics/collector.py` (StorageMetricsCollector)

#### Counters

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `prefetched_tokens_total` | Prefetched prompt tokens |
| 2 | `backuped_tokens_total` | Backed-up tokens |

#### Histograms

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `prefetch_pgs` | Pages prefetched per batch |
| 2 | `backup_pgs` | Pages backed up per batch |
| 3 | `prefetch_bandwidth` | Prefetch bandwidth (GB/s) |
| 4 | `backup_bandwidth` | Backup bandwidth (GB/s) |

---

### Radix Cache Metrics

**Source**: `python/sglang/srt/metrics/collector.py` (RadixCacheMetricsCollector)

#### Counters

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `evicted_tokens_total` | Tokens evicted from GPU to CPU |
| 2 | `load_back_tokens_total` | Tokens loaded from CPU to GPU |

#### Histograms

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `eviction_duration_seconds` | GPU to CPU eviction time |
| 2 | `load_back_duration_seconds` | CPU to GPU load time |

---

### Expert Dispatch Metrics

**Source**: `python/sglang/srt/metrics/collector.py` (ExpertDispatchCollector)

#### Histograms

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `eplb_gpu_physical_count` | Physical expert selection counts per layer/GPU |

---

### Function-Level Metrics

**Source**: `python/sglang/srt/metrics/func_timer.py`

#### Histograms

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `func_latency_seconds` | Function-level latency profiling (decorator) |

---

### Startup Metrics

**Source**: `python/sglang/srt/metrics/startup_func_log_and_timer.py`

#### Gauges

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `startup_latency_breakdown_seconds_max` | Max duration per startup context |

---

### System Metrics

**Source**: `python/sglang/srt/metrics/cpu_monitor.py`

#### Counters

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `process_cpu_seconds_total` | CPU time consumed (user + system) |

---

### Config Info Metrics (SGLang)

**Source**: `python/sglang/srt/metrics/collector.py` (set once at startup)

#### Gauges (Info-style, always value 1.0)

| # | Metric Name | Labels | Description |
|---|-------------|--------|-------------|
| 1 | `model_config_info` | `model`, `served_model_name`, `dtype`, `max_total_tokens`, `quantization`, `gpu_type` | Model configuration |
| 2 | `parallel_config_info` | `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`, `expert_parallel_size`, `gpu_count` | Parallelism configuration |
| 3 | `speculative_config_info` | `spec_enabled`, `spec_algorithm`, `spec_num_draft_tokens`, `spec_num_steps`, `spec_eagle_topk`, `spec_draft_model` | Speculative decoding config (if enabled) |

---

### HTTP & Request Type Metrics (SGLang)

**Source**: `python/sglang/srt/openai_api/common.py`, `serving_chat.py`

| # | Metric Name | Type | Description |
|---|-------------|------|-------------|
| 1 | `http_request_duration_seconds` | Histogram | HTTP request duration (labels: `method`, `path`, `status_code`) |
| 2 | `request_type_image_total` | Counter | Requests containing image inputs |
| 3 | `request_type_video_total` | Counter | Requests containing video inputs |
| 4 | `request_type_tool_call_total` | Counter | Requests using tool/function calling |
| 5 | `request_type_structured_output_total` | Counter | Requests using structured output |

---

## 2. vLLM Metrics

**Total**: ~55 metrics (15 Gauges, 15 Counters, 19 Histograms + HTTP/request type)

### Scheduler State Metrics

**Source**: `vllm/v1/metrics/loggers.py` (PrometheusStatLogger)
**Common Labels**: `model_name`, `engine`

#### Gauges

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `num_requests_running` | Requests in model execution batches |
| 2 | `num_requests_waiting` | Requests waiting to be processed |
| 3 | `engine_sleep_state` | Engine sleep/wake state (label: `sleep_state`) |
| 4 | `kv_cache_usage_perc` | KV cache usage ratio (0-1) |

---

### Throughput & Startup Metrics

**Source**: `vllm/v1/metrics/loggers.py`

#### Gauges

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `gen_throughput` | Generation throughput (tokens/s) |
| 2 | `engine_startup_time` | Engine startup time (seconds) |
| 3 | `engine_load_weights_time` | Weight loading time (seconds) |

---

### Cache Metrics

**Source**: `vllm/v1/metrics/loggers.py`

#### Counters

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `prefix_cache_queries` | Prefix cache queries (token count) |
| 2 | `prefix_cache_hits` | Prefix cache hits (token count) |
| 3 | `external_prefix_cache_queries` | External (cross-instance) cache queries |
| 4 | `external_prefix_cache_hits` | External cache hits |
| 5 | `mm_cache_queries` | Multi-modal cache queries |
| 6 | `mm_cache_hits` | Multi-modal cache hits |

---

### Request Counters

**Source**: `vllm/v1/metrics/loggers.py`

#### Counters

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `num_preemptions` | Cumulative preemptions |
| 2 | `prompt_tokens` | Prefill tokens processed |
| 3 | `generation_tokens` | Generation tokens processed |
| 4 | `request_success` | Successfully processed requests (label: `finished_reason`) |
| 5 | `corrupted_requests` | Requests with NaN logits (opt-in) |

---

### Token Histograms

**Source**: `vllm/v1/metrics/loggers.py`

#### Histograms

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `request_prompt_tokens` | Prompt tokens per finished request |
| 2 | `request_generation_tokens` | Generation tokens per finished request |
| 3 | `iteration_tokens_total` | Total tokens per engine step |
| 4 | `request_max_num_generation_tokens` | Max generation tokens per request |
| 5 | `request_params_n` | Request "n" parameter distribution |
| 6 | `request_params_max_tokens` | Request max_tokens parameter |
| 7 | `request_prefill_kv_computed_tokens` | New KV tokens computed (excl. cached) |

---

### Timing Histograms (Latency)

**Source**: `vllm/v1/metrics/loggers.py`

#### Histograms

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `time_to_first_token_seconds` | Time to first token |
| 2 | `inter_token_latency_seconds` | Inter-token latency |
| 3 | `time_per_output_token_seconds` | **DEPRECATED** alias for ITL |
| 4 | `request_time_per_output_token_seconds` | Per-request mean TPOT |

---

### Request Latency Histograms

**Source**: `vllm/v1/metrics/loggers.py`

#### Histograms

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `e2e_request_latency_seconds` | End-to-end request latency |
| 2 | `request_queue_time_seconds` | Time in WAITING phase |
| 3 | `request_inference_time_seconds` | Time in RUNNING phase (prefill + decode) |
| 4 | `request_prefill_time_seconds` | Time in PREFILL phase |
| 5 | `request_decode_time_seconds` | Time in DECODE phase |

---

### KV Cache Residency Metrics

**Source**: `vllm/v1/metrics/loggers.py`, `vllm/v1/core/kv_cache_metrics.py`
**Conditional**: Only created if `kv_cache_metrics_enabled`

#### Histograms

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `kv_block_lifetime_seconds` | Block lifetime from allocation to eviction |
| 2 | `kv_block_idle_before_evict_seconds` | Idle time before eviction |
| 3 | `kv_block_reuse_gap_seconds` | Time between consecutive block accesses |

---

### LoRA Metrics

**Source**: `vllm/v1/metrics/loggers.py`
**Conditional**: Only created if `lora_config is not None`

#### Gauges

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `lora_requests_info` | LoRA request info with adapter names |
| 2 | `lora_pool_utilization` | LoRA pool utilization (0-1) |

---

### Speculative Decoding Metrics

**Source**: `vllm/v1/spec_decode/metrics.py` (SpecDecodingProm)
**Conditional**: Only created if `speculative_config is not None`

#### Counters

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `spec_decode_num_drafts` | Number of spec decoding drafts |
| 2 | `spec_decode_num_draft_tokens` | Number of draft tokens |
| 3 | `spec_decode_num_accepted_tokens` | Number of accepted tokens |
| 4 | `spec_decode_num_accepted_tokens_per_pos` | Accepted tokens per draft position (label: `position`) |

#### Gauges (per-interval)

| # | Metric Name | Description |
|---|-------------|-------------|
| 1 | `spec_accept_length` | Avg accepted sequence length |
| 2 | `spec_accept_rate` | Draft acceptance rate (0-1) |

---

### Config Info Metrics (vLLM)

**Source**: `vllm/v1/metrics/loggers.py` (created dynamically at startup)

#### Gauges (Info-style, always value 1.0)

| # | Metric Name | Labels | Description |
|---|-------------|--------|-------------|
| 1 | `cache_config_info` | Dynamic from `cache_config.metrics_info()` | Cache configuration |
| 2 | `model_config_info` | `model`, `served_model_name`, `dtype`, `max_model_len`, `quantization`, `enforce_eager`, `gpu_type` | Model configuration |
| 3 | `parallel_config_info` | `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`, `enable_expert_parallel`, `gpu_count` | Parallelism configuration |
| 4 | `speculative_config_info` | `spec_enabled`, `spec_method`, `spec_num_tokens`, `spec_draft_model` | Speculative decoding config (if enabled) |

---

### HTTP & Request Type Metrics (vLLM)

**Source**: `vllm/entrypoints/openai/serving_chat.py`, middleware

| # | Metric Name | Type | Description |
|---|-------------|------|-------------|
| 1 | `http_request_duration_seconds` | Histogram | HTTP request duration (labels: `method`, `path`, `status_code`) |
| 2 | `request_type_image_total` | Counter | Requests containing image inputs |
| 3 | `request_type_video_total` | Counter | Requests containing video inputs |
| 4 | `request_type_tool_call_total` | Counter | Requests using tool/function calling |
| 5 | `request_type_structured_output_total` | Counter | Requests using structured output |

---

## 3. Cross-Reference Quick Lookup Table

| Metric Name | SGLang | vLLM | Notes |
|-------------|:------:|:----:|-------|
| **Core Latency** | | | |
| `e2e_request_latency_seconds` | Y | Y | |
| `time_to_first_token_seconds` | Y | Y | |
| `inter_token_latency_seconds` | Y | Y | Computation differs slightly |
| `request_queue_time_seconds` | Y | Y | |
| `request_inference_time_seconds` | Y | Y | |
| `request_prefill_time_seconds` | Y | Y | |
| `request_decode_time_seconds` | Y | Y | |
| `request_time_per_output_token_seconds` | Y | Y | Per-request mean TPOT |
| **Token Counters** | | | |
| `prompt_tokens` / `prompt_tokens_total` | Y | Y | Suffix differs |
| `generation_tokens` / `generation_tokens_total` | Y | Y | Suffix differs |
| `cached_tokens_total` / `prefix_cache_hits` | Y | Y | Different naming |
| **Token Histograms** | | | |
| `request_prompt_tokens` | Y | Y | Different buckets |
| `request_generation_tokens` | Y | Y | Different buckets |
| `iteration_tokens_total` | - | Y | |
| `request_max_num_generation_tokens` | - | Y | |
| `request_params_n` | - | Y | |
| `request_params_max_tokens` | Y | Y | |
| `request_prefill_kv_computed_tokens` | - | Y | |
| **Request Counters** | | | |
| `request_success` / `request_success_total` | Y | Y | Suffix differs |
| `num_preemptions` / `num_preemptions_total` | Y | Y | Suffix differs |
| `num_requests_total` | Y | - | SGLang-only |
| `num_aborted_requests_total` | Y | - | SGLang-only |
| `corrupted_requests` | - | Y | |
| **Queue/Load Gauges** | | | |
| `num_requests_running` | Y | Y | |
| `num_requests_waiting` | Y | Y | |
| `kv_cache_usage_perc` | Y | Y | |
| `gen_throughput` | Y | Y | |
| **Startup/Config** | | | |
| `engine_startup_time` | Y | Y | |
| `engine_load_weights_time` | Y | Y | |
| `model_config_info` | Y | Y | Different labels |
| `parallel_config_info` | Y | Y | Different labels |
| `speculative_config_info` | Y | Y | Different labels |
| `cache_config_info` | - | Y | |
| **Cache** | | | |
| `cache_hit_rate` | Y | - | SGLang: pre-computed ratio |
| `prefix_cache_queries` | - | Y | |
| `prefix_cache_hits` | - | Y | |
| `external_prefix_cache_queries` | - | Y | |
| `external_prefix_cache_hits` | - | Y | |
| `mm_cache_queries` | Y | Y | Multi-modal cache |
| `mm_cache_hits` | Y | Y | Multi-modal cache |
| **Speculative Decoding** | | | |
| `spec_accept_rate` | Y | Y | |
| `spec_accept_length` | Y | Y | |
| `spec_decode_num_drafts` | Y | Y | |
| `spec_decode_num_draft_tokens` | - | Y | |
| `spec_decode_num_accepted_tokens` | - | Y | |
| `spec_decode_num_accepted_tokens_per_pos` | Y | Y | |
| **LoRA** | | | |
| `lora_pool_utilization` | Y | Y | |
| `lora_pool_slots_used` | Y | - | |
| `lora_pool_slots_total` | Y | - | |
| `lora_requests_info` | - | Y | |
| **KV Cache Residency** | | | |
| `kv_block_lifetime_seconds` | - | Y | Optional |
| `kv_block_idle_before_evict_seconds` | - | Y | Optional |
| `kv_block_reuse_gap_seconds` | - | Y | Optional |
| **Engine State** | | | |
| `engine_sleep_state` | - | Y | |
| `utilization` | Y | - | |
| `max_running_requests_under_SLO` | Y | - | |
| **Grammar/Structured Output** | | | |
| `grammar_compilation_time_seconds` | Y | - | |
| `grammar_schema_count` | Y | - | |
| `grammar_ebnf_size` | Y | - | |
| `grammar_tree_traversal_time_avg` | Y | - | |
| `grammar_tree_traversal_time_max` | Y | - | |
| `num_grammar_cache_hit_total` | Y | - | |
| `num_grammar_aborted_total` | Y | - | |
| `num_grammar_timeout_total` | Y | - | |
| `num_grammar_total` | Y | - | |
| `num_grammar_queue_reqs` | Y | - | |
| `num_so_requests_total` | Y | - | |
| **PD Disaggregation** | | | |
| `num_prefill_prealloc_queue_reqs` | Y | - | |
| `num_prefill_inflight_queue_reqs` | Y | - | |
| `num_decode_prealloc_queue_reqs` | Y | - | |
| `num_decode_transfer_queue_reqs` | Y | - | |
| `kv_transfer_speed_gb_s` | Y | - | |
| `kv_transfer_latency_ms` | Y | - | |
| `kv_transfer_bootstrap_ms` | Y | - | |
| `kv_transfer_alloc_ms` | Y | - | |
| `kv_transfer_total_mb` | Y | - | |
| `num_bootstrap_failed_reqs_total` | Y | - | |
| `num_transfer_failed_reqs_total` | Y | - | |
| `num_prefill_retries_total` | Y | - | |
| **Retraction** | | | |
| `num_retracted_requests_total` | Y | - | |
| `num_retracted_input_tokens_total` | Y | - | |
| `num_retracted_output_tokens_total` | Y | - | |
| `num_retractions` | Y | - | |
| `num_retracted_reqs` | Y | - | |
| **Cache Eviction** | | | |
| `eviction_duration_seconds` | Y | - | |
| `evicted_tokens_total` | Y | - | |
| `load_back_duration_seconds` | Y | - | |
| `load_back_tokens_total` | Y | - | |
| **GPU/CUDA** | | | |
| `gpu_execution_seconds_total` | Y | - | |
| `cuda_graph_passes_total` | Y | - | |
| `is_cuda_graph` | Y | - | |
| **MoE** | | | |
| `eplb_balancedness` | Y | - | |
| `eplb_gpu_physical_count` | Y | - | |
| **Routing** | | | |
| `routing_key_running_req_count` | Y | - | |
| `routing_key_all_req_count` | Y | - | |
| `num_unique_running_routing_keys` | Y | - | |
| **HTTP & Request Type** | | | |
| `http_request_duration_seconds` | Y | Y | |
| `request_type_image_total` | Y | Y | |
| `request_type_video_total` | Y | Y | |
| `request_type_tool_call_total` | Y | Y | |
| `request_type_structured_output_total` | Y | Y | |
| **Other SGLang-only** | | | |
| `per_stage_req_latency_seconds` | Y | - | |
| `func_latency_seconds` | Y | - | |
| `prefill_delayer_wait_forward_passes` | Y | - | |
| `prefill_delayer_wait_seconds` | Y | - | |
| `prefill_delayer_outcomes_total` | Y | - | |
| `realtime_tokens_total` | Y | - | |
| `dp_cooperation_realtime_tokens_total` | Y | - | |
| `dp_cooperation_gpu_execution_seconds_total` | Y | - | |
| `num_used_tokens` | Y | - | |
| `max_total_num_tokens` | Y | - | |
| `pending_prealloc_token_usage` | Y | - | |
| `swa_token_usage` | Y | - | |
| `mamba_usage` | Y | - | |
| `decode_sum_seq_lens` | Y | - | |
| `num_running_reqs_offline_batch` | Y | - | |
| `num_paused_reqs` | Y | - | |
| `new_token_ratio` | Y | - | |
| `startup_latency_breakdown_seconds_max` | Y | - | |
| `process_cpu_seconds_total` | Y | - | |
| `prefetched_tokens_total` | Y | - | |
| `backuped_tokens_total` | Y | - | |
| `prefetch_pgs` | Y | - | |
| `backup_pgs` | Y | - | |
| `prefetch_bandwidth` | Y | - | |
| `backup_bandwidth` | Y | - | |

---

## 4. Notes

- **No hardcoded prefix**: Both frameworks define metrics without `sglang:` / `vllm:` prefix. The prefix is applied at the Prometheus namespace level at runtime.
- **Multiprocess mode**: SGLang uses Prometheus multiprocess mode. Requires `PROMETHEUS_MULTIPROC_DIR` to be set for histogram/counter metrics to work.
- **Dynamic buckets (vLLM)**: Token histograms use `build_1_2_5_buckets(max_model_len)` for model-aware bucket sizing.
- **Configurable buckets (SGLang)**: TTFT, ITL, and E2E latency histograms accept custom bucket configurations via CLI args.
- **Counter naming**: SGLang uses `_total` suffix (OpenMetrics standard); vLLM does not. Both are valid conventions.
- **Streaming dependency (SGLang)**: TTFT and ITL metrics require `stream: true` in requests.
- **Conditional metrics**: KV cache residency (vLLM, opt-in), corrupted_requests (vLLM, opt-in), LoRA metrics (both, only if LoRA configured), speculative decoding (both, only if spec decode configured).
- **New unified metrics** (on `feat/unified-metrics` branches):
  - SGLang added from vLLM: `request_inference_time_seconds`, `request_prefill_time_seconds`, `request_decode_time_seconds`, `request_prompt_tokens`, `request_generation_tokens`, `request_success_total`, `num_preemptions_total`, `request_time_per_output_token_seconds`, `spec_decode_num_drafts`, `spec_decode_num_accepted_tokens_per_pos`, `mm_cache_queries`, `mm_cache_hits`, `request_params_max_tokens`, `http_request_duration_seconds`
  - vLLM added from SGLang: `gen_throughput`, `engine_startup_time`, `engine_load_weights_time`, `spec_accept_length`, `spec_accept_rate`, `lora_pool_utilization`
  - Both added: `request_type_image_total`, `request_type_video_total`, `request_type_tool_call_total`, `request_type_structured_output_total`, `gpu_type` in `model_config_info`, `gpu_count` in `parallel_config_info`
