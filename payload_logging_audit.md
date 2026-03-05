# Payload Logging Audit - vLLM & SGLang Unified Metrics Branches

**Date:** 2025-02-26 (updated)
**vLLM Branch:** `feat/unified-metrics-0.16.0` (rebased on `v0.16.0`)
**SGLang Branch:** `feat/unified-metrics-rebased-02122026`
**Base commit (OTEL logging):** `2d2396eb0fa70e51c3e22352e3e4453276657dc7`

---

## Summary of Fixes

Three categories of issues have been identified and resolved across both codebases.

### 1. `nca_id` header logging

The `nca_id` field (from HTTP request headers) is needed for request tracing.

| API | Old branch | Rebased branch (before fix) | After fix |
|-----|-----------|---------------------------|-----------|
| Chat Completions | **Yes** | **Lost** (rebase dropped headers collection) | **Yes** (`c57d6ae6b`) |
| Completions | **No** | **Yes** (new work added during rebase) | **Yes** |
| Responses | **No** | **Yes** (new work added during rebase) | **Yes** |

**Fix commit:** `c57d6ae6b` — restored headers collection in chat completions, also fixed
`rid` to use `self._base_request_id(raw_request, ...)` and changed all APIs from
`json.dumps` to passing dicts directly for proper OTEL structured logging.

### 2. Response payload logging (structured content)

The `payload_logger` logs structured `openai.response` records to OTEL with the full
response content (choices, output items, usage).

| API | Old branch | Rebased branch (before fix) | After fix |
|-----|-----------|---------------------------|-----------|
| Chat Completions | **Full** (choices with reasoning, content, tool_calls) | **Full** (no change needed) | **Full** |
| Completions | **None** | **Empty** (choices: [], usage only) | **Full** — choices with text + finish_reason (`2ed0d6a0b`) |
| Responses | **None** | **Empty** (output_count only, no streaming) | **Full** — structured output items for both streaming and non-streaming (`2ed0d6a0b`) |

**Fix commit:** `2ed0d6a0b` — populated completions choices and responses output items

### 3. OTEL log noise from `request_logger` overlap

`request_logger` messages (`"Received request"`, `"Generated response"`) were leaking to
the OTEL pipeline as unstructured text, duplicating the structured `payload_logger` records.

**Fix commit:** `cbd070bd3` — added `"Received request"` and `"Generated response"` patterns
to `OtelLogFilter`'s blocklist. These messages now only appear on console for debugging.

---

## Overall Architecture

There are two separate data paths: **metrics** and **logging**. Both flow to the OTEL
collector but through different mechanisms.

### Metrics path (Prometheus → OTEL)

vLLM natively exposes Prometheus metrics at `/metrics`. We added a bridge thread that
periodically scrapes these and re-exports them as OTEL instruments via gRPC/OTLP.

```
vLLM Prometheus metrics (/metrics endpoint)
    │
    ▼
Prom-to-OTEL bridge thread (scrapes every 30s, configurable)
    │  converts counters/gauges/histograms to OTEL instruments
    ▼
OTEL MeterProvider → OTLP MetricExporter (gRPC)
    │
    ▼
OTEL Collector (OTEL_EXPORTER_OTLP_METRICS_ENDPOINT)
```

### Logging path (Python logging → OTEL)

All logging flows through the root `vllm` logger, which has two handlers: console and OTEL.
The OTEL handler uses a queue to avoid blocking the main serving thread.

```
Main serving thread
    │
    ├─ payload_logger.info("openai.request", extra={...})
    │  payload_logger.info("openai.response", extra={...})
    │
    └─ request_logger.log_inputs(...) / request_logger.log_outputs(...)
           │
           ▼
    root "vllm" logger
           │
    ┌──────┴──────────────┐
    ▼                     ▼
Console handler        QueueHandler
(+ ConsoleLogFilter)       │
    │                 queue.put(record)  ← returns immediately, non-blocking
    ▼                     │
stdout/stderr             ▼
                    [queue.Queue]  ← Python stdlib thread-safe queue data structure
                          │
                          ▼  QueueListener daemon thread (separate from serving)
                          │  queue.get(record)  ← blocks here waiting for records
                          │
                     OtelLogFilter (drops high-volume engine stats)
                          │
                     KratosOffloadFilter (optional)
                          │  if record has base64 data URIs > 256KB:
                          │    → bulk_upload to Kratos S3 (synchronous, but on this thread)
                          │    → replace inline data with [offloaded:s3://...] reference
                          │
                     OTEL LoggingHandler → OTLP LogExporter (gRPC)
                          │
                          ▼
                    OTEL Collector (OTEL_EXPORTER_OTLP_LOGS_ENDPOINT)
```

**How the queue works:** The `QueueHandler` (Python stdlib `logging.handlers.QueueHandler`)
has an `emit()` that just does `queue.put(record)` — no I/O, returns immediately. The
`QueueListener` (also stdlib) spawns a **daemon thread** that loops calling `queue.get()`,
then passes each record through the filters and handler chain. This means the Kratos
`bulk_upload` (which does a synchronous HTTP call to S3) happens on the QueueListener thread,
not on the main serving thread. The queue is what provides the non-blocking behavior.

### What goes where

| Data | Path | Destination |
|------|------|-------------|
| Prometheus metrics (latency, throughput, etc.) | Prom bridge → OTLP MetricExporter | OTEL Collector |
| `openai.request` / `openai.response` payloads | `payload_logger` → QueueHandler → OTLP LogExporter | OTEL Collector |
| `request_logger` outputs ("Generated response...") | Blocked by `OtelLogFilter` (`cbd070bd3`) | Console only |
| Large binary blobs (base64 images in payloads) | KratosOffloadFilter → `kratos.bulksync.bulk_upload` | Kratos S3 bucket |

### Console vs OTEL filtering

| Filter | Applied to | Blocks |
|--------|------------|--------|
| `ConsoleLogFilter` | Console handlers | `openai.request`, `openai.response`, `Generated response.*chatcmpl-\|cmpl-\|resp_` |
| `OtelLogFilter` | OTEL handler | `Avg prompt throughput`, `vllm.v1.metrics`, `Received request`, `Generated response` |

**Clean separation (after `cbd070bd3`):** `request_logger` messages are now blocked from
**both** console (via `ConsoleLogFilter`) and OTEL (via `OtelLogFilter`). Only the structured
`payload_logger` records (`openai.request` / `openai.response`) flow to the OTEL collector.
This eliminates the duplicate/overlapping data that existed on the old branch.

### payload_logger vs request_logger

Both are Python loggers that are children of the root `vllm` logger — both flow to the
same OTEL handler and collector. The difference is what they log and why they exist.

| | `payload_logger` | `request_logger` |
|---|---|---|
| **Definition** | `logging.getLogger("vllm.payload")` | `RequestLogger` class (`vllm/entrypoints/logger.py`) |
| **Origin** | Our custom addition for OTEL | Pre-existing vLLM infrastructure |
| **Format** | Structured dicts in `extra` (`rid`, `endpoint`, `payload`, `headers`) | Plain text log lines (`"Received request %s: params: %s"`, `"Generated response %s"`) |
| **Request logging** | Full request payload dict + HTTP headers as one structured record | Params at INFO, prompt + token IDs at DEBUG |
| **Response logging** | Full `resp_summary` dict with choices/output, usage as one structured record | Individual content parts as separate plain text lines |
| **Control** | `VLLM_LOG_PAYLOADS` env var (default: `"1"` = enabled) | `--disable-log-requests` / `--enable-log-outputs` CLI flags |
| **Overlap** | ~~Minimal on request side; significant on response side~~ **Resolved** (`cbd070bd3`): `request_logger` messages now filtered from OTEL. Only `payload_logger` flows to OTEL. |

---

## Old Branch Baseline (`feat/production-otel-logging`)

The old branch only had `payload_logger` in `serving_chat.py`. The completions API
(`serving_completion.py`) and responses API (`serving_responses.py`) had **zero**
`payload_logger` usage — no structured request or response payload logging at all.

The completions/responses request payload logging on the rebased branch was **new work**
added during the rebase, not carried over from the old branch.

| API | Old branch `payload_logger` | Rebased branch (before fix) | Rebased branch (after fix) |
|-----|----------------------------|---------------------------|--------------------------|
| Chat Completions | Request + Response (full) | Request (missing headers/rid) + Response (full) | Request (full) + Response (full) |
| Completions | **None** | Request + Response (empty choices) | Request + Response (full choices) |
| Responses | **None** | Request + Response (metadata only, no streaming) | Request + Response (full output, both modes) |

---

## Current State Per API

### Chat Completions API (`/v1/chat/completions`)

**File:** `vllm/entrypoints/openai/chat_completion/serving.py`

#### Request logging (`payload_logger`)
- Logs `openai.request` with: `rid`, `endpoint`, `payload` (full request dict), `headers` (all HTTP headers)
- Logged once before any processing

#### Response logging (`payload_logger`)

| Mode | Logged? | Content |
|------|---------|---------|
| Streaming | Yes | `resp_summary` dict with `choices` containing separate `reasoning_content`, `content`, `tool_calls` fields per choice |
| Non-streaming | Yes | Same structure as streaming with `"stream": False` |

**Example response payload (Chat Completions):**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": "thinking text...",
        "content": "response text",
        "tool_calls": [
          {
            "id": "call_xxx",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\": \"SF\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  },
  "stream": false
}
```

- `reasoning_content`: Only present if model produced reasoning tokens
- `content`: The main text response, or `null` if tool-call-only
- `tool_calls`: Only present if model made tool calls; `finish_reason` changes to `"tool_calls"`
- Multiple choices: Fully supported — iterates over all `num_choices` (streaming) or `choices` (non-streaming)

#### Output logging (`request_logger.log_outputs`)

| Mode | Reasoning | Content | Tool Calls | Notes |
|------|-----------|---------|------------|-------|
| Streaming (delta) | Combined with labels | Combined with labels | Combined with labels | Only if `enable_log_deltas` is on |
| Streaming (complete) | Separate: `[reasoning] {text}` | Separate: plain text | Separate: `[tool_calls: name(args)]` | 3 separate log calls after stream ends |
| Streaming (fallback) | N/A | Full accumulated text | N/A | Only if no reasoning/content/tool_calls tracked |
| Non-streaming | Separate: `[reasoning] {text}` | Plain text | `[tool_calls: name(args)]` | **Content and tool_calls are elif (mutually exclusive)** |

---

### Completion API (`/v1/completions`)

**File:** `vllm/entrypoints/openai/completion/serving.py`

#### Request logging (`payload_logger`)
- Logs `openai.request` with: `rid`, `endpoint`, `payload` (full request dict), `headers` (all HTTP headers)

#### Response logging (`payload_logger`)

| Mode | Logged? | Content |
|------|---------|---------|
| Streaming | Yes | `resp_summary` with `choices` containing `text` and `finish_reason` per choice, plus `usage` |
| Non-streaming | Yes | `resp_summary` with `choices` containing `text` and `finish_reason` per choice, plus `usage` |

**Example response payload (Completions):**
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "model-name",
  "choices": [
    {
      "index": 0,
      "text": "generated completion text...",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  },
  "stream": false
}
```

- `choices` includes `text` (accumulated generated text) and `finish_reason` per choice ~~(GAP-1 resolved)~~
- Multiple choices fully supported — iterates over all `num_choices * num_prompts` (streaming) or `choices` list (non-streaming)
- Streaming tracks per-choice `finish_reason` via `previous_finish_reasons` list

#### Output logging (`request_logger.log_outputs`)

| Mode | Reasoning | Content | Tool Calls | Notes |
|------|-----------|---------|------------|-------|
| Streaming | N/A | Plain text (`previous_texts[i]`) | N/A | No separation |
| Non-streaming | N/A | Plain text (`choice.text`) | N/A | No separation |

---

### Responses API (`/v1/responses`)

**File:** `vllm/entrypoints/openai/responses/serving.py`

#### Request logging (`payload_logger`)
- Logs `openai.request` with: `rid`, `endpoint`, `payload` (full request dict), `headers` (all HTTP headers)

#### Response logging (`payload_logger`)

| Mode | Logged? | Content |
|------|---------|---------|
| Streaming | Yes | `resp_summary` with `output` items (reasoning, message, function_call), `status`, `usage` ~~(GAP-2 resolved)~~ |
| Non-streaming | Yes | `resp_summary` with `output` items (reasoning, message, function_call), `status`, `usage` ~~(GAP-5 resolved)~~ |

**Example response payload (Responses):**
```json
{
  "id": "resp-xxx",
  "object": "response",
  "created": 1234567890,
  "model": "model-name",
  "output": [
    {
      "type": "reasoning",
      "content": "thinking text..."
    },
    {
      "type": "message",
      "content": "response text"
    },
    {
      "type": "function_call",
      "name": "get_weather",
      "arguments": "{\"city\": \"SF\"}",
      "call_id": "call_xxx"
    }
  ],
  "status": "completed",
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  },
  "stream": false
}
```

- `output` contains structured items with `type` discriminator: `reasoning`, `message`, `function_call`
- `reasoning`: concatenated text from `ResponseReasoningItem.content`
- `message`: concatenated text from `ResponseOutputMessage.content`
- `function_call`: includes `name`, `arguments`, `call_id` from `ResponseFunctionToolCall`
- Both streaming and non-streaming use the same output structure

#### Output logging (`request_logger.log_outputs`)

| Mode | Reasoning | Content | Tool Calls | Notes |
|------|-----------|---------|------------|-------|
| Streaming | **No** | **No** | **No** | No streaming output logging |
| Non-streaming (Harmony) | Separate: `[reasoning] {text}` | Separate: plain text | Separate: `[tool_calls: name(args)]` | 3 separate isinstance checks |
| Non-streaming (simple) | N/A | `final_output.text` | N/A | Generic fallback, no separation |

---

## Feature Gaps

### ~~GAP-1: Completion API response payload logs empty choices~~ (RESOLVED)

**Status:** Resolved
**Files:** `completion/serving.py`

Both streaming and non-streaming `resp_summary` now include `choices` with `text` and
`finish_reason` per choice. Streaming uses `previous_texts` and `previous_finish_reasons`
lists to track accumulated content per choice. Non-streaming iterates over the `choices`
list of `CompletionResponseChoice` objects.

---

### ~~GAP-2: Responses API has no streaming response/output logging~~ (RESOLVED)

**Status:** Resolved
**Files:** `responses/serving.py`

Streaming now logs `payload_logger.info("openai.response")` with structured `output` items
built from `final_response.output` before yielding the `ResponseCompletedEvent`. Uses the
same `isinstance` pattern as non-streaming to build reasoning, message, and function_call
items.

---

### GAP-3: Chat Completions non-streaming output logging uses elif for content vs tool_calls

**Severity:** Low
**Files:** `chat_completion/serving.py` ~line 2076

In non-streaming mode, content and tool_calls are logged with `elif` (mutually exclusive).
If a response has both `content` AND `tool_calls`, only content gets logged. The streaming
path correctly logs them as separate calls.

---

### GAP-4: Completion API output logging has no field separation

**Severity:** Low
**Files:** `completion/serving.py`

Output logging does not separate reasoning, content, or tool_calls. It logs raw text only.
This is acceptable for the legacy completions API since it typically does not produce
reasoning or tool_calls, but if models do produce them, they will not be labeled.

---

### ~~GAP-5: Responses API non-streaming response payload has no output content~~ (RESOLVED)

**Status:** Resolved
**Files:** `responses/serving.py`

The `resp_summary` now includes structured `output` items (replacing `output_count`) with
reasoning, message, and function_call types built from the `output` list using the same
`isinstance` pattern as `request_logger.log_outputs`.

---

### ~~GAP-6: `request_logger` messages leak to OTEL as unstructured log noise~~ (RESOLVED)

**Status:** Resolved
**Files:** `api_server.py` (`OtelLogFilter`)

`request_logger` messages (`"Received request %s: params: %s"`, `"Generated response %s: output: %r"`)
were not blocked by `OtelLogFilter`, so they flowed to the OTEL collector as plain text log lines
alongside the structured `payload_logger` records. This created duplicate/overlapping data in the
OTEL pipeline.

**Fix commit:** `cbd070bd3` — added `"Received request"` and `"Generated response"` patterns to
`OtelLogFilter`'s blocklist. These messages now only appear on console (for debugging) and no
longer pollute the OTEL pipeline.

---

## 5. SGLang Payload Logging

**Branch:** `feat/unified-metrics-rebased-02122026`
**Logger:** `logging.getLogger("sglang.payload")`
**Env var:** `SGLANG_LOG_PAYLOADS=1` (default: `"0"` = disabled)

### Architecture

SGLang uses a shared base class (`OpenAIServingBase.handle_request()`) that handles
payload logging for both request and non-streaming response. Chat and Completions APIs
route through this base handler. The Responses API has its own `create_responses()` method
that bypasses the base handler.

Streaming response logging is handled per-API since the base class skips `StreamingResponse`
objects.

### Current State Per API

#### Chat Completions API

**Request logging:** Via `serving_base.py` `handle_request()` — logs `openai.request` with
`rid`, `endpoint`, `payload` (full request dict), `headers` (all HTTP headers).

**Response logging:**

| Mode | Logged? | Content |
|------|---------|---------|
| Streaming | Yes | Hand-built `choices` with separate `reasoning_content`, `content`, `tool_calls` per choice, `finish_reason`, `usage` |
| Non-streaming | Yes | Via `result.model_dump()` in base class — full pydantic serialization including all fields |

- Streaming logging in `serving_chat.py` reconstructs tool_calls from parser state
  (`detector.prev_tool_call_arr`) with proper call IDs
- Multiple choices supported (iterates all indices from `finish_reasons`)

#### Completions API

**Request logging:** Via `serving_base.py` `handle_request()` — same as chat.

**Response logging:**

| Mode | Logged? | Content |
|------|---------|---------|
| Streaming | Yes | `choices` with `text` and `finish_reason` per index, plus `usage` (`91bdd7c08`) |
| Non-streaming | Yes | Via `result.model_dump()` in base class |

**Fix commit:** `91bdd7c08` — added streaming response payload logging. Previously the
`StreamingResponse` bypassed the base class logging entirely, leaving no response record.
Now tracks `finish_reasons` and `response_id` during streaming and logs after completion.

#### Responses API

**Request logging:** Own logging in `create_responses()` — logs `responses.request` with
`rid`, `endpoint`, `payload`, `headers` (all HTTP headers).

**Fix commit:** `6a70cbda6` — added missing `headers` capture to Responses API request
logging, matching the pattern used by chat and completions via the base class.

**Response logging:**

| Mode | Logged? | Content |
|------|---------|---------|
| Non-streaming | Yes | Via `response.model_dump()` — full pydantic serialization |
| Background | Yes | Via `response.model_dump()` or `ORJSONResponse` body |
| Streaming | Yes | Via `final_response.model_dump()` after stream completion |

All three modes use `model_dump()` which includes the full response structure with
output items (reasoning, message, function_call types) from the pydantic model.

### SGLang vs vLLM Response Payload Comparison

| Feature | vLLM | SGLang |
|---------|------|--------|
| **Chat streaming** | Hand-built choices with separated reasoning/content/tool_calls | Hand-built choices with separated reasoning/content/tool_calls |
| **Chat non-streaming** | Hand-built choices with separated fields | Generic `model_dump()` (fields present via pydantic) |
| **Completions streaming** | Hand-built choices with text/finish_reason | Hand-built choices with text/finish_reason (`91bdd7c08`) |
| **Completions non-streaming** | Hand-built choices with text/finish_reason | Generic `model_dump()` (fields present via pydantic) |
| **Responses streaming** | Hand-built output items (reasoning/message/function_call) | `model_dump()` of final response |
| **Responses non-streaming** | Hand-built output items | `model_dump()` of final response |
| **Request headers** | All APIs | All APIs (Responses fixed in `6a70cbda6`) |
| **OTEL log noise filtering** | `OtelLogFilter` blocks request_logger (`cbd070bd3`) | No `request_logger` equivalent — not applicable |

### SGLang Remaining Gaps

**None critical.** The non-streaming paths use generic `model_dump()` instead of hand-built
summaries. The data is complete (all fields present via pydantic serialization) but includes
extra pydantic boilerplate fields. This is a style difference, not a functional gap.
