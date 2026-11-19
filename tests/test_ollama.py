from antaigent.llm.ollama import Ollama, is_loopback


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
