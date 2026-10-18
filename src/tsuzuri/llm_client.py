"""Process-wide shared OpenAI client factory.

Per-call ``OpenAI(...)`` instances were never closed and leaked their httpx
keep-alive sockets, surfacing as ``ResourceWarning: unclosed ssl.SSLSocket``
at GC (visible whenever traffic goes through a local proxy such as
127.0.0.1:7890). Clients are cached per (class, api_key, base_url, timeout,
extra kwargs) and reused; the sync client is thread-safe.

The client class is a parameter (not a module-level import binding) so tests
that patch e.g. ``宿主应用.OpenAI`` keep working: the patched class flows in as
``client_cls`` and is cached under its own key, never mixing with real
clients.
"""

from __future__ import annotations

import threading

_CLIENTS: dict = {}
_LOCK = threading.Lock()


def shared_openai_client(client_cls=None, *, api_key, base_url=None, timeout=None, **extra):
    if client_cls is None:
        from openai import OpenAI

        client_cls = OpenAI
    key = (
        client_cls,
        str(api_key or ""),
        str(base_url or ""),
        None if timeout is None else float(timeout),
        tuple(sorted((str(name), repr(value)) for name, value in extra.items())),
    )
    with _LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            kwargs = {"api_key": api_key, **extra}
            if base_url:
                kwargs["base_url"] = base_url
            if timeout is not None:
                kwargs["timeout"] = timeout
            client = client_cls(**kwargs)
            _CLIENTS[key] = client
    return client


__all__ = ["shared_openai_client"]
