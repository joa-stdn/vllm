# Universal HF Tokenizer Server

One CPU process, any HuggingFace model, load-on-demand. Serves `/tokenize`,
`/detokenize`, `/v1/tokenize`, `/v1/models` and `/healthz` — no GPU, no vLLM
engine, no per-model boot declaration.

## Why this exists

Today, token counting for many public models (Qwen, Llama, GLM, Kimi K3,
DeepSeek, …) requires either a full engine per model or hitting a rendered
route that pins a GPU endpoint per served model. This service replaces that
with a single CPU process that caches tokenizers keyed by their served name
and loads them lazily on first request.

Design goals — in order of priority:

1. **Universal.** Any HF model. First request for an unseen name loads it,
   subsequent requests hit RAM.
2. **Bounded blast radius.** `trust_remote_code=True` by default (needed for
   K3 and friends) — combine with `--allowed-model-pattern` in production.
3. **Small and vendored.** No `vllm.*` import; just `transformers` +
   `fastapi` + `uvicorn`. ~300 lines total. Extract at will.

Non-goals:

- **Not a rendering server.** No tool-parser massaging, no reasoning-parser
  overrides, no multimodal preprocessing. If the model's shipped
  `apply_chat_template` gets you the right count, so does this. If you need
  vLLM's model-specific glue (e.g. DeepSeek V4 reasoning-effort mapping),
  use `vllm launch render`.
- **Not for Mistral tekken.** HF has no tekken loader. Route Mistral models
  through Harmattan or `mistral_common` directly.

See [scope notes at the bottom](#what-works-what-drifts) for the honest
model-by-model coverage.

## Install

```bash
uv pip install fastapi uvicorn "transformers>=4.44" tiktoken sentencepiece
```

`tiktoken` is required for Kimi K3's `TikTokenTokenizer` (loaded via
`auto_map` on first hit). `sentencepiece` is required for a number of
Llama/Mistral tokenizers.

## Run

```bash
python -m tools.universal_tokenize_server.server \
    --host 0.0.0.0 --port 8100 \
    --allowed-model-pattern 'mistralai/*,meta-llama/*,Qwen/*,moonshotai/*,deepseek-ai/*' \
    --preload-model meta-llama/Llama-3.1-8B \
    --max-cached-models 32
```

Flags:

| Flag | Default | What it does |
|---|---|---|
| `--host`, `--port` | `0.0.0.0` / `8100` | Bind address. |
| `--max-cached-models` | `32` | LRU cap on live tokenizers. |
| `--no-trust-remote-code` | off | Disables `trust_remote_code=True`. K3 breaks without it. |
| `--allowed-model-pattern` | `*` | Comma-separated fnmatch patterns. Non-matching requests get 403. |
| `--preload-model MODEL` | `[]` | Repeatable. Warm the cache at startup. |
| `--hf-revision` | none | Pin every load to a specific HF revision. |
| `--request-timeout-s` | `60.0` | Per-request budget including cold-load. |

## Endpoints

### `POST /tokenize` (or `POST /v1/tokenize`)

```jsonc
// Non-chat
{"model": "meta-llama/Llama-3.1-8B", "prompt": "Hello, world!"}

// Chat (applies the model's shipped chat_template — Jinja or auto_map Python)
{
  "model": "moonshotai/Kimi-K3",
  "messages": [{"role": "user", "content": "Hi"}],
  "tools": [...],                   // optional
  "chat_template_kwargs": {         // model-specific extras (e.g. thinking)
    "thinking": true
  },
  "add_generation_prompt": true,
  "return_tokens": false,
  "return_token_strs": false
}
```

Response:

```jsonc
{
  "model": "moonshotai/Kimi-K3",
  "count": 42,
  "max_model_len": 131072,
  "tokens": null,                   // populated iff return_tokens=true
  "token_strs": null,               // populated iff return_token_strs=true
  "used_chat_template": true,
  "from_cache": true                // false on cold-load requests
}
```

### `POST /detokenize` (or `POST /v1/detokenize`)

```jsonc
{"model": "meta-llama/Llama-3.1-8B", "tokens": [128000, 9906], "skip_special_tokens": false}
```

### `GET /v1/models`

Lists every model currently in cache, in LRU order, with load timing and
`has_chat_template`. Also echoes the current `allowed_patterns` and
`trust_remote_code` setting so operators can confirm policy at a glance.

### `GET /healthz`

Cheap liveness. Doesn't touch the cache.

## Smoke test

```bash
# Different tokenizers -> different counts on the same prompt
curl -s localhost:8100/tokenize -H 'Content-Type: application/json' \
    -d '{"model":"meta-llama/Llama-3.1-8B","prompt":"Hello, world!"}'
curl -s localhost:8100/tokenize -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello, world!"}'

# Chat template renders per-model (K3 loads tokenization_kimi.py via auto_map)
curl -s localhost:8100/tokenize -H 'Content-Type: application/json' \
    -d '{"model":"moonshotai/Kimi-K3","messages":[{"role":"user","content":"Hi"}]}'

# LRU / cache introspection
curl -s localhost:8100/v1/models | jq
```

## What works, what drifts

| Model family | Correct? | Notes |
|---|---|---|
| Qwen / Llama / GLM / Gemma / Phi | ✅ byte-identical | Standard HF `chat_template` field. |
| **Kimi K3** | ✅ **byte-identical** | Repo ships `tokenization_kimi.TikTokenTokenizer` via `auto_map` and `encoding_k3.py`; `trust_remote_code=True` pulls both in. Same code vLLM's `KimiK3Renderer` runs (`self.get_tokenizer().apply_chat_template(...)`). |
| Cohere command R+ etc. | ⚠️ drift | Uses shipped Jinja here; vLLM uses the `cohere_melody` library. |
| DeepSeek V3.x | ⚠️ drift on tools / thinking | Uses shipped Jinja here; vLLM re-implements the encoder in `deepseek_v4_encoding.py` (~550 lines) with explicit tool placement / thinking-mode / reasoning-effort. |
| DeepSeek V4 / V3.2 (no template) | ❓ likely broken | V3.2's `tokenizer_config.json` dropped `chat_template`; without an `auto_map`-loaded encoder, HF has nothing to apply. |
| Inkling | ✅ | HF-native. |
| **Mistral tekken** | ❌ **use `mistral_common`** | HF has no tekken loader. |

Rule of thumb: **if the HF repo either (a) ships a `chat_template` in
`tokenizer_config.json` or (b) exposes an `auto_map` for a
`PreTrainedTokenizer` subclass with a custom `apply_chat_template`, this
server matches vLLM byte-for-byte.** Otherwise expect drift.

## What's next

If drift on Cohere / DeepSeek tools becomes a real problem, the two paths
are:

1. Extract the ~100-line encoders from vLLM (`deepseek_v4_encoding.py`,
   `cohere_melody`) and vendor them here behind a per-family switch. Cheap.
2. Fall back to a multi-model render server in vLLM (~300 lines,
   RFC/upstream cycle). More work but reuses the maintained encoders.

For now, this service covers the ~95% case cleanly. Route the exotic 5%
through vLLM's single-model render server or Harmattan.
