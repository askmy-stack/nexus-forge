"""API authentication and rate limiting for the FastAPI serving layer."""

from __future__ import annotations

import os
from collections.abc import Callable

from fastapi import Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

PROTECTED_PREFIXES = ("/summarize", "/grade", "/train")
PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}


def _is_protected(path: str) -> bool:
    if path in PUBLIC_PATHS or path.startswith("/docs"):
        return False
    return any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def _api_key_configured() -> bool:
    return bool(os.getenv("API_KEY"))


def _require_train_key() -> bool:
    return os.getenv("REQUIRE_TRAIN_KEY", "0") == "1"


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def verify_train_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = os.getenv("TRAIN_API_KEY")
    if _require_train_key() and not expected:
        raise HTTPException(
            status_code=503,
            detail="TRAIN_API_KEY is required but not configured (REQUIRE_TRAIN_KEY=1)",
        )
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforce API_KEY on protected routes when configured."""

    async def dispatch(self, request: Request, call_next: Callable):
        if _api_key_configured() and _is_protected(request.url.path):
            provided = request.headers.get("X-API-Key")
            if provided != os.getenv("API_KEY"):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
        return await call_next(request)


class SimpleRateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory per-IP rate limiter (requests per minute)."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._hits: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable):
        if not _is_protected(request.url.path):
            return await call_next(request)

        import time

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60.0
        hits = [t for t in self._hits.get(client_ip, []) if t > window_start]
        if len(hits) >= self.requests_per_minute:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        hits.append(now)
        self._hits[client_ip] = hits
        return await call_next(request)
