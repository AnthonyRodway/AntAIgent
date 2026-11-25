import json

import httpx

from antaigent.llm.ollama import Ollama, is_loopback


def test_models_can_be_loaded_without_generating_anything():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"embeddings": []} if request.url.path == "/api/embed" else {"done": True})

    with Ollama("http://127.0.0.1:11434") as client:
        client._http.close()
        client._http = httpx.Client(base_url=client.base_url, transport=httpx.MockTransport(handler))
        client.load("gpt-oss:20b", keep_alive="30m", num_ctx=16384)
        assert client.embed("nomic-embed-text", [], "30m") == []
    # The same num_ctx as real requests, or Ollama would load the model a second time for them.
    assert seen[0] == ("/api/chat", {"model": "gpt-oss:20b", "messages": [], "keep_alive": "30m", "options": {"num_ctx": 16384}})
    assert seen[1][0] == "/api/embed" and seen[1][1]["input"] == []


def test_local_ollama_ignores_proxy_settings(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    for url in ("http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"):
        assert is_loopback(url)
        with Ollama(url) as client:
            assert not client._http._mounts, url  # no proxy transport mounted

    assert not is_loopback("http://gpu-box.local:11434")
    with Ollama("http://gpu-box.local:11434") as client:
        assert client._http._mounts  # a remote Ollama still goes through the configured proxy
