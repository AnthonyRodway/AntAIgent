"""Minimal client for the parts of the Ollama HTTP API that AntAIgent uses."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


class OllamaError(RuntimeError):
    pass


def _error_text(response: httpx.Response) -> str:
    try:
        return f"Ollama: {response.json().get('error', response.text)}"
    except ValueError:
        return f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"


def _think_value(think: str) -> bool | str | None:
    value = think.strip().lower()
    if value in ("", "default", "none"):
        return None
    if value in ("true", "on", "yes"):
        return True
    if value in ("false", "off", "no"):
        return False
    return value


def has_model(available: list[str], name: str) -> bool:
    return name in available or (":" not in name and f"{name}:latest" in available)


class Ollama:
    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=5.0))

    def close(self) -> None:
        self._http.close()

    def _unreachable(self) -> OllamaError:
        return OllamaError(f"Can't reach Ollama at {self.base_url}. Is it running?")

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise self._unreachable() from exc
        except httpx.TimeoutException as exc:
            raise OllamaError(f"Ollama timed out on {path}") from exc
        if response.status_code >= 400:
            raise OllamaError(_error_text(response))
        return response.json()

    def version(self) -> str:
        return self._request("GET", "/api/version")["version"]

    def models(self) -> list[str]:
        return [m["name"] for m in self._request("GET", "/api/tags").get("models", [])]

    def running(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/ps").get("models", [])

    def embed(self, model: str, inputs: list[str], keep_alive: str) -> list[list[float]]:
        data = self._request(
            "POST", "/api/embed", json={"model": model, "input": inputs, "keep_alive": keep_alive, "truncate": True}
        )
        return data["embeddings"]

    def chat(
        self, model: str, messages: list[dict[str, str]], *, think: str, num_ctx: int, keep_alive: str
    ) -> Iterator[tuple[str, Any]]:
        """Stream a reply as ("thinking", text), ("content", text) and finally ("done", stats)."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": keep_alive,
            "options": {"num_ctx": num_ctx},
        }
        think_value = _think_value(think)
        if think_value is not None:
            payload["think"] = think_value
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    raise OllamaError(_error_text(response))
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("error"):
                        raise OllamaError(f"Ollama: {event['error']}")
                    message = event.get("message") or {}
                    if message.get("thinking"):
                        yield "thinking", message["thinking"]
                    if message.get("content"):
                        yield "content", message["content"]
                    if event.get("done"):
                        yield "done", event
        except httpx.ConnectError as exc:
            raise self._unreachable() from exc
        except httpx.TimeoutException as exc:
            raise OllamaError("Ollama timed out while generating a reply") from exc
