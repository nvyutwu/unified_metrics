# Payload Logging Audit - vLLM feat/unified-metrics-0.16.0

**Date:** 2025-02-25
**Branch:** `feat/unified-metrics-0.16.0` (rebased on `v0.16.0`)
**Base commit (OTEL logging):** `2d2396eb0fa70e51c3e22352e3e4453276657dc7`

---

## Recent Fix

### Missing headers in Chat Completions API request logging

**Commit:** `c57d6ae6b` on `feat/unified-metrics-0.16.0`

During the rebase from the old branch (`feat/production-otel-logging`) onto `v0.16.0`, the
`chat_completion/serving.py` file lost the HTTP request headers collection in the
`openai.request` payload log. The old branch had this in `serving_chat.py`, but when v0.16.0
refactored into `chat_completion/serving.py`, the cherry-picked code did not include headers.

**What was fixed:**
- Added `headers_obj = {k: v for k, v in raw_request.headers.items()}` collection
- Added `"headers": headers_obj` to the `extra` dict in `payload_logger.info("openai.request")`
- Changed `rid` from `getattr(request, "request_id", "")` to `self._base_request_id(raw_request, ...)` to match the old branch pattern

**Also fixed in the same commit:**
- `completion/serving.py` and `responses/serving.py` had headers and payloads serialized as
  JSON strings (`json.dumps`). Changed to pass dicts directly for proper OTEL structured logging,
  matching the chat_completion pattern and the old branch's comment:
  "Pass dict directly for proper OTEL structured logging"
- Removed duplicate `"otel"` key in `setup.py` (upstream v0.16.0 had a subset that would
  override our version with `kratos-cli`)

---

## Logging Architecture

There are two logging systems for payload logging:

| System | Logger | Purpose |
|--------|--------|---------|
| **Payload logger** | `payload_logger = logging.getLogger("vllm.payload")` | OTEL structured logging of full request/response payloads. Controlled by `VLLM_LOG_PAYLOADS` env var (default: `"1"` = enabled). Logs structured `extra` dicts with `rid`, `endpoint`, `payload`, `headers`. |
| **Request logger** | `RequestLogger` class (`vllm/entrypoints/logger.py`) | Pre-existing vLLM logging infrastructure. `log_inputs()` logs params at INFO, prompt at DEBUG. `log_outputs()` logs generated text at INFO. Controlled by `--enable-log-outputs` / `--disable-log-requests` flags. |

### OTEL pipeline

```
payload_logger  ──┐
                   ├──→ root "vllm" logger ──→ OTEL handler ──→ OTEL collector ──→ Kratos
request_logger  ──┘          │
                             └──→ Console handler (filtered)
```

Both loggers are children of the root `vllm` logger. The OTEL log handler is attached at the
root level, so **all log records flow to OTEL** unless explicitly filtered by `OtelLogFilter`.

### Console vs OTEL filtering

| Filter | Applied to | Blocks |
|--------|------------|--------|
| `ConsoleLogFilter` | Console handlers | `openai.request`, `openai.response`, `Generated response.*chatcmpl-` |
| `OtelLogFilter` | OTEL handler | `Avg prompt throughput`, `vllm.v1.metrics` |

**Key insight:** `request_logger` messages are blocked from console but **NOT blocked from
OTEL**. This means `"Received request"` and `"Generated response"` log lines flow through
to the OTEL collector as unstructured text alongside the structured `payload_logger` records.
This is true on both the old branch (`feat/production-otel-logging`) and the rebased branch.

### Overlap between the two loggers

**Request INPUT:**
- `payload_logger`: Full request payload dict + HTTP headers (structured)
- `request_logger.log_inputs()`: Params + LoRA request at INFO, prompt + token IDs at DEBUG
- Overlap: Minimal (prompt text appears in both, but at different log levels)

**Response OUTPUT:**
- `payload_logger`: Structured `resp_summary` dict with choices, usage (one record)
- `request_logger.log_outputs()`: Individual content parts as plain text log lines
- Overlap: **Significant** — generated text appears in both, but with different granularity

---

## Old Branch Baseline (`feat/production-otel-logging`)

The old branch only had `payload_logger` in `serving_chat.py`. The completions API
(`serving_completion.py`) and responses API (`serving_responses.py`) had **zero**
`payload_logger` usage — no structured request or response payload logging at all.

The completions/responses request payload logging on the rebased branch was **new work**
added during the rebase, not carried over from the old branch.

| API | Old branch `payload_logger` | Rebased branch `payload_logger` |
|-----|----------------------------|--------------------------------|
| Chat Completions | Request + Response (full) | Request + Response (full) |
| Completions | **None** | Request only + Response (empty choices) |
| Responses | **None** | Request only + Response (metadata only) |

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

### GAP-6: `request_logger` messages leak to OTEL as unstructured log noise

**Severity:** Medium
**Files:** `api_server.py` (`OtelLogFilter`)

`request_logger` messages (`"Received request %s: params: %s"`, `"Generated response %s: output: %r"`)
are not blocked by `OtelLogFilter`, so they flow to the OTEL collector as plain text log lines
alongside the structured `payload_logger` records. This creates duplicate/overlapping data in the
OTEL pipeline — the same response content appears once as a structured `openai.response` record
(from `payload_logger`) and again as an unstructured `"Generated response..."` text line
(from `request_logger`).

This is present on both the old branch (`feat/production-otel-logging`) and the rebased branch.

**Fix:** Add `request_logger` patterns (`"Received request"`, `"Generated response"`) to
`OtelLogFilter`'s blocklist so they only appear on console (for debugging) and do not pollute
the OTEL pipeline.
