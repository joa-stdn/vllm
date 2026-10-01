# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Universal HF tokenizer server — one process, any HF model, load-on-demand.

Serves `/tokenize`, `/detokenize`, `/v1/tokenize` (OpenAI-compatible), and
`/v1/models` on a single CPU process. When a request names a model that
isn't in the LRU cache yet, the server calls `AutoTokenizer.from_pretrained`
with `trust_remote_code=True` (opt-out via CLI), applies the shipped chat
template (Jinja or `auto_map`-loaded Python — the latter handles Kimi K3's
`tokenization_kimi.TikTokenTokenizer`), and caches the result.

Not a vLLM engine — deliberately zero import of `vllm.*` so this can run
anywhere transformers can. If you need Mistral tekken parity, route through
Harmattan / `mistral_common` instead; if you need per-model tool-parser or
reasoning-parser massaging beyond what the shipped template does, use
`vllm launch render` (single-model) or the render server's future multi-model
variant.

Usage:
    uv pip install fastapi uvicorn "transformers>=4.44" tiktoken sentencepiece
    python -m tools.universal_tokenize_server.server \\
        --host 0.0.0.0 --port 8100 \\
        --allowed-model-pattern 'mistralai/*,meta-llama/*,Qwen/*,moonshotai/*' \\
        --preload-model meta-llama/Llama-3.1-8B \\
        --max-cached-models 32
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, cast

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger("universal_tokenize_server")


# ---------- Config ----------


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8100
    max_cached_models: int = 32
    trust_remote_code: bool = True
    allowed_patterns: tuple[str, ...] = ("*",)
    preload_models: tuple[str, ...] = ()
    hf_revision: str | None = None
    request_timeout_s: float = 60.0
    log_level: str = "INFO"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ServerConfig:
        return cls(
            host=args.host,
            port=args.port,
            max_cached_models=args.max_cached_models,
            trust_remote_code=not args.no_trust_remote_code,
            allowed_patterns=tuple(_split_csv(args.allowed_model_pattern)),
            preload_models=tuple(args.preload_model or ()),
            hf_revision=args.hf_revision,
            request_timeout_s=args.request_timeout_s,
            log_level=args.log_level,
        )


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return ["*"]
    return [p.strip() for p in value.split(",") if p.strip()]


# ---------- Cache ----------


@dataclass
class CachedTokenizer:
    tokenizer: PreTrainedTokenizerBase
    loaded_at: float = field(default_factory=time.time)
    # Time budget spent loading, in seconds. Surfaced on /v1/models so callers
    # can see which entries are cold vs. warm without waking the model.
    load_seconds: float = 0.0


class TokenizerCache:
    """LRU cache of `PreTrainedTokenizerBase` keyed by model name.

    One asyncio lock per model name so concurrent first-hits on the SAME
    model share the load, but hits on different models never serialize.
    """

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._entries: OrderedDict[str, CachedTokenizer] = OrderedDict()
        self._per_model_locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    async def preload(self, models: Sequence[str]) -> None:
        for model in models:
            try:
                await self.get(model)
            except Exception:
                logger.exception(
                    "Preload failed for %r; will retry on first request", model
                )

    async def get(self, model_name: str) -> CachedTokenizer:
        if model_name in self._entries:
            self._entries.move_to_end(model_name)
            return self._entries[model_name]

        if not self._is_allowed(model_name):
            raise ModelNotAllowedError(model_name, self._config.allowed_patterns)

        async with await self._lock_for(model_name):
            if model_name in self._entries:  # double-checked after acquire
                self._entries.move_to_end(model_name)
                return self._entries[model_name]

            start = time.time()
            tokenizer = await asyncio.to_thread(self._blocking_load, model_name)
            entry = CachedTokenizer(
                tokenizer=tokenizer,
                load_seconds=time.time() - start,
            )
            self._entries[model_name] = entry
            self._evict_if_full()
            logger.info(
                "Loaded tokenizer for %r in %.2fs (cache size %d/%d)",
                model_name,
                entry.load_seconds,
                len(self._entries),
                self._config.max_cached_models,
            )
            return entry

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "id": name,
                "loaded_at": entry.loaded_at,
                "load_seconds": round(entry.load_seconds, 3),
                "vocab_size": getattr(entry.tokenizer, "vocab_size", None),
                "model_max_length": getattr(entry.tokenizer, "model_max_length", None),
                "has_chat_template": bool(
                    getattr(entry.tokenizer, "chat_template", None)
                ),
            }
            for name, entry in self._entries.items()
        ]

    def _is_allowed(self, model_name: str) -> bool:
        return any(
            fnmatch.fnmatchcase(model_name, p) for p in self._config.allowed_patterns
        )

    async def _lock_for(self, model_name: str) -> asyncio.Lock:
        async with self._registry_lock:
            lock = self._per_model_locks.get(model_name)
            if lock is None:
                lock = asyncio.Lock()
                self._per_model_locks[model_name] = lock
            return lock

    def _evict_if_full(self) -> None:
        while len(self._entries) > self._config.max_cached_models:
            evicted, _ = self._entries.popitem(last=False)
            self._per_model_locks.pop(evicted, None)
            logger.info("Evicted cached tokenizer for %r (LRU)", evicted)

    def _blocking_load(self, model_name: str) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self._config.trust_remote_code,
            revision=self._config.hf_revision,
        )


class ModelNotAllowedError(Exception):
    def __init__(self, model: str, patterns: Sequence[str]) -> None:
        self.model = model
        self.patterns = tuple(patterns)
        super().__init__(f"Model {model!r} is not in the allowlist {list(patterns)}")


# ---------- Request / response schemas ----------


class TokenizeRequest(BaseModel):
    model: str
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    add_special_tokens: bool = True
    add_generation_prompt: bool = True
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    return_tokens: bool = False
    return_token_strs: bool = False

    @model_validator(mode="after")
    def _exactly_one_input(self) -> TokenizeRequest:
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("Exactly one of `prompt` or `messages` must be set")
        return self


class TokenizeResponse(BaseModel):
    model: str
    count: int
    max_model_len: int | None = None
    tokens: list[int] | None = None
    token_strs: list[str] | None = None
    used_chat_template: bool = False
    from_cache: bool


class DetokenizeRequest(BaseModel):
    model: str
    tokens: list[int]
    skip_special_tokens: bool = False


class DetokenizeResponse(BaseModel):
    model: str
    text: str


# ---------- App ----------


def build_app(config: ServerConfig) -> FastAPI:
    cache = TokenizerCache(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if config.preload_models:
            logger.info("Preloading %d model(s)", len(config.preload_models))
            await cache.preload(config.preload_models)
        yield

    app = FastAPI(title="Universal HF Tokenizer Server", lifespan=lifespan)
    app.state.cache = cache
    app.state.config = config

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"object": "model", **row} for row in cache.snapshot()],
            "allowed_patterns": list(config.allowed_patterns),
            "trust_remote_code": config.trust_remote_code,
            "max_cached_models": config.max_cached_models,
        }

    @app.post("/tokenize")
    @app.post("/v1/tokenize")
    async def tokenize(request: TokenizeRequest, raw: Request) -> TokenizeResponse:
        was_cached = request.model in cache._entries  # noqa: SLF001 — snapshot before load
        try:
            entry = await asyncio.wait_for(
                cache.get(request.model), timeout=config.request_timeout_s
            )
        except ModelNotAllowedError as exc:
            raise HTTPException(HTTPStatus.FORBIDDEN, detail=str(exc)) from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Load timed out after {config.request_timeout_s}s",
            ) from exc
        tokenizer = entry.tokenizer

        tokens: list[int]
        used_chat_template = False
        if request.messages is not None:
            try:
                rendered = tokenizer.apply_chat_template(
                    request.messages,
                    tools=cast(Any, request.tools),
                    add_generation_prompt=request.add_generation_prompt,
                    chat_template=request.chat_template,
                    tokenize=True,
                    **request.chat_template_kwargs,
                )
            except Exception as exc:
                raise HTTPException(
                    HTTPStatus.BAD_REQUEST,
                    detail=f"apply_chat_template failed: {exc}",
                ) from exc
            # `apply_chat_template(tokenize=True)` returns list[int]; coerce
            # defensively (some tokenizers return tensors / BatchEncoding).
            tokens = [int(t) for t in cast(Sequence[Any], rendered)]
            used_chat_template = True
        else:
            assert request.prompt is not None  # validator guarantees one is set
            tokens = [
                int(t)
                for t in tokenizer.encode(
                    request.prompt,
                    add_special_tokens=request.add_special_tokens,
                )
            ]

        token_strs: list[str] | None = None
        if request.return_token_strs:
            converted = tokenizer.convert_ids_to_tokens(tokens)
            token_strs = converted if isinstance(converted, list) else [converted]

        return TokenizeResponse(
            model=request.model,
            count=len(tokens),
            max_model_len=_safe_max_len(tokenizer),
            tokens=tokens if request.return_tokens else None,
            token_strs=token_strs,
            used_chat_template=used_chat_template,
            from_cache=was_cached,
        )

    @app.post("/detokenize")
    @app.post("/v1/detokenize")
    async def detokenize(request: DetokenizeRequest) -> DetokenizeResponse:
        try:
            entry = await cache.get(request.model)
        except ModelNotAllowedError as exc:
            raise HTTPException(HTTPStatus.FORBIDDEN, detail=str(exc)) from exc
        decoded = entry.tokenizer.decode(
            request.tokens, skip_special_tokens=request.skip_special_tokens
        )
        # `decode(list[int])` returns str; guard defensively.
        text = decoded if isinstance(decoded, str) else "".join(decoded)
        return DetokenizeResponse(model=request.model, text=text)

    @app.exception_handler(ValueError)
    async def _value_error(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST, content={"detail": str(exc)}
        )

    return app


def _safe_max_len(tokenizer: PreTrainedTokenizerBase) -> int | None:
    value = getattr(tokenizer, "model_max_length", None)
    # HF sets 1_000_000_000_000_000_019_884_624_838_656 as a sentinel meaning
    # "unknown" — surface `None` instead so callers don't display garbage.
    if value is None or value > 10**12:
        return None
    return int(value)


# ---------- CLI ----------


def _make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="universal_tokenize_server",
        description=(
            "One CPU process, any HF model, load-on-demand. See module docstring "
            "for scope and non-goals."
        ),
    )
    p.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8100")))
    p.add_argument(
        "--max-cached-models",
        type=int,
        default=32,
        help="LRU cap on live tokenizer instances (default: 32)",
    )
    p.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help=(
            "Disable trust_remote_code on AutoTokenizer.from_pretrained. Default is "
            "trust=ON because K3-family tokenizers ship Python via auto_map; "
            "combine with a tight --allowed-model-pattern in untrusted environments."
        ),
    )
    p.add_argument(
        "--allowed-model-pattern",
        type=str,
        default="*",
        help=(
            "Comma-separated fnmatch patterns of allowed model names. "
            "Example: 'mistralai/*,meta-llama/*,Qwen/*,moonshotai/*'. "
            "Requests for models outside the allowlist return 403. "
            "Default '*' allows anything — tighten in production."
        ),
    )
    p.add_argument(
        "--preload-model",
        action="append",
        metavar="MODEL",
        help="Repeatable: preload this HF model at startup so first request is warm.",
    )
    p.add_argument(
        "--hf-revision",
        default=None,
        help="Pin every tokenizer load to this HF revision (branch/tag/commit).",
    )
    p.add_argument(
        "--request-timeout-s",
        type=float,
        default=60.0,
        help="Per-request timeout including cold-load (default: 60s).",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> None:
    args = _make_arg_parser().parse_args()
    config = ServerConfig.from_args(args)
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        build_app(config),
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
