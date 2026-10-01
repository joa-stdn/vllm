# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline unit tests for the universal tokenizer server.

Exercises the cache, allowlist, and endpoint routing with a fake tokenizer
so tests run in <1s with no network. End-to-end tests that hit real HF
tokenizers live under `tests/entrypoints/render/` in the main tree; these
are the CI-friendly ones.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tools.universal_tokenize_server import server


def _fake_tokenizer(name: str) -> MagicMock:
    """Fake `PreTrainedTokenizerBase` — deterministic tokens keyed by model name."""
    tok = MagicMock()
    tok.name = name
    tok.vocab_size = 32000
    tok.model_max_length = 4096
    tok.chat_template = "template" if "chat" in name else None
    # Vary the token stream per model so tests can assert routing.
    signature = sum(ord(c) for c in name) % 100
    tok.encode.side_effect = lambda text, **_: [signature + i for i in range(len(text))]
    tok.apply_chat_template.side_effect = lambda messages, **_: [
        signature + i for i in range(sum(len(m["content"]) for m in messages))
    ]
    tok.convert_ids_to_tokens.side_effect = lambda ids, **_: [f"<{i}>" for i in ids]
    tok.decode.side_effect = lambda ids, **_: "".join(f"[{i}]" for i in ids)
    return tok


@pytest.fixture
def config() -> server.ServerConfig:
    return server.ServerConfig(
        max_cached_models=3,
        allowed_patterns=("test/*",),
        preload_models=(),
    )


@pytest.fixture
def app(config: server.ServerConfig, monkeypatch: pytest.MonkeyPatch):
    """FastAPI app with `AutoTokenizer.from_pretrained` monkeypatched."""

    def _fake_load(name, **_kwargs):
        return _fake_tokenizer(name)

    monkeypatch.setattr(server.AutoTokenizer, "from_pretrained", _fake_load)
    return server.build_app(config)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_tokenize_routes_by_model_and_caches(client: TestClient) -> None:
    # First hit -> cold (from_cache=False)
    r1 = client.post(
        "/tokenize", json={"model": "test/alpha", "prompt": "hello"}
    ).json()
    assert r1["model"] == "test/alpha"
    assert r1["count"] == 5
    assert r1["from_cache"] is False

    # Second hit -> warm
    r2 = client.post(
        "/tokenize", json={"model": "test/alpha", "prompt": "hello"}
    ).json()
    assert r2["from_cache"] is True

    # Different model -> different tokens
    r3 = client.post("/tokenize", json={"model": "test/beta", "prompt": "hello"}).json()
    assert r3["tokens"] is None  # return_tokens=False by default
    r3_full = client.post(
        "/tokenize",
        json={"model": "test/beta", "prompt": "hello", "return_tokens": True},
    ).json()
    r1_full = client.post(
        "/tokenize",
        json={"model": "test/alpha", "prompt": "hello", "return_tokens": True},
    ).json()
    assert r3_full["tokens"] != r1_full["tokens"], (
        "different models must give different tokens"
    )


def test_tokenize_chat_messages_uses_apply_chat_template(client: TestClient) -> None:
    r = client.post(
        "/tokenize",
        json={
            "model": "test/chat",
            "messages": [{"role": "user", "content": "hi"}],
            "return_tokens": True,
        },
    ).json()
    assert r["used_chat_template"] is True
    assert len(r["tokens"]) == 2  # "hi" has length 2


def test_tokenize_requires_exactly_one_of_prompt_or_messages(
    client: TestClient,
) -> None:
    r = client.post(
        "/tokenize",
        json={
            "model": "test/alpha",
            "prompt": "x",
            "messages": [{"role": "user", "content": "y"}],
        },
    )
    assert r.status_code == 422
    r = client.post("/tokenize", json={"model": "test/alpha"})
    assert r.status_code == 422


def test_detokenize_roundtrip(client: TestClient) -> None:
    r = client.post(
        "/detokenize", json={"model": "test/alpha", "tokens": [1, 2, 3]}
    ).json()
    assert r == {"model": "test/alpha", "text": "[1][2][3]"}


def test_disallowed_model_returns_403(client: TestClient) -> None:
    r = client.post("/tokenize", json={"model": "unlisted/model", "prompt": "hi"})
    assert r.status_code == 403
    assert "allowlist" in r.json()["detail"]


def test_lru_eviction(client: TestClient, config: server.ServerConfig) -> None:
    # Cache cap is 3; load 4 different models and confirm the LRU one evicts.
    for name in ("test/a", "test/b", "test/c", "test/d"):
        client.post("/tokenize", json={"model": name, "prompt": "x"})
    models = client.get("/v1/models").json()
    ids = [m["id"] for m in models["data"]]
    assert "test/a" not in ids, f"LRU eviction failed: {ids}"
    assert set(ids) == {"test/b", "test/c", "test/d"}
    assert len(ids) == config.max_cached_models


def test_v1_models_lists_config(client: TestClient) -> None:
    client.post("/tokenize", json={"model": "test/alpha", "prompt": "x"})
    body = client.get("/v1/models").json()
    assert body["allowed_patterns"] == ["test/*"]
    assert body["trust_remote_code"] is True
    assert body["max_cached_models"] == 3
    assert body["data"][0]["id"] == "test/alpha"
    assert (
        body["data"][0]["has_chat_template"] is False
    )  # 'alpha' didn't contain 'chat'


def test_concurrent_first_hits_share_the_load(
    config: server.ServerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two coroutines hitting the same cold model should both wait on one load call."""
    calls = 0

    def _slow_load(name, **_kwargs):
        nonlocal calls
        calls += 1
        return _fake_tokenizer(name)

    monkeypatch.setattr(server.AutoTokenizer, "from_pretrained", _slow_load)
    cache = server.TokenizerCache(config)

    async def _hit_twice() -> None:
        await asyncio.gather(cache.get("test/x"), cache.get("test/x"))

    asyncio.run(_hit_twice())
    assert calls == 1, (
        f"per-model lock leaked; from_pretrained was called {calls} times"
    )
