"""Defensas HTTP de bajo nivel (ASGI puro, antes de que FastAPI lea nada del cuerpo).

- ``BodySizeLimitMiddleware``: rechaza cuerpos enormes con 413 sin cargarlos en memoria.
- ``LocalHostGuard``: en modo local solo acepta ``Host: localhost`` / ``127.0.0.1``. Así una
  página web abierta en tu navegador no puede usar un nombre de dominio propio que apunte a
  127.0.0.1 («DNS rebinding») para hablar con el servidor.
"""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

ALLOWED_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


class _BodyTooLargeError(Exception):
    pass


def _note_error(scope: Scope, message: str) -> None:
    """Deja el motivo donde lo lee el access log (``request.state.error``)."""
    scope.setdefault("state", {})["error"] = message


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit_mb = self.max_bytes / (1024 * 1024)
        detail = f"El cuerpo de la petición supera el máximo de {limit_mb:g} MB."
        _note_error(scope, detail)
        await JSONResponse({"detail": detail}, status_code=413)(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            try:
                too_large = int(declared) > self.max_bytes
            except ValueError:
                too_large = False  # cabecera mal formada: que la rechace el servidor HTTP
            if too_large:
                await self._reject(scope, receive, send)
                return

        # Sin Content-Length (chunked) también se vigila: se cuenta lo que llega.
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLargeError
            return message

        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLargeError:
            if not response_started:
                await self._reject(scope, receive, send)


def _hostname(header: str) -> str:
    """'localhost:9966' → 'localhost'; '[::1]:9966' → '[::1]'."""
    value = header.strip().lower()
    if value.startswith("["):
        return value.split("]", 1)[0] + "]"
    return value.split(":", 1)[0]


class LocalHostGuard:
    def __init__(self, app: ASGIApp, allowed: frozenset[str] = ALLOWED_LOCAL_HOSTS) -> None:
        self.app = app
        self.allowed = allowed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            host = dict(scope["headers"]).get(b"host", b"").decode("latin-1")
            if _hostname(host) not in self.allowed:
                detail = "Host no permitido: el servidor está en modo local (usa localhost)."
                _note_error(scope, f"Host rechazado: {host[:80]!r}")
                await JSONResponse({"detail": detail}, status_code=400)(scope, receive, send)
                return
        await self.app(scope, receive, send)
