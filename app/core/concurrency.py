"""Cola FIFO de un solo trabajador para las traducciones.

El servidor atiende las traducciones de una en una (el equipo no da para más), pero
SIN bloquear el event loop: las peticiones esperan su turno con ``await`` y el trabajo
pesado corre en un hilo, así que ``/``, ``/models`` y ``/pairs`` responden siempre.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.core.exceptions import ServerBusyError


class RequestQueue:
    """Ejecuta una función bloqueante cada vez, en orden de llegada (FIFO).

    ``asyncio.Lock`` despierta a los que esperan en orden de llegada, así que la
    cola es justa. Si ya hay ``max_pending`` peticiones (en curso + esperando),
    las nuevas se rechazan con ``ServerBusyError`` (HTTP 503).
    """

    def __init__(self, max_pending: int) -> None:
        self._max_pending = max_pending
        self._lock = asyncio.Lock()
        self._pending = 0

    @property
    def pending(self) -> int:
        return self._pending

    @property
    def max_pending(self) -> int:
        return self._max_pending

    async def run(self, func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        if self._pending >= self._max_pending:
            raise ServerBusyError(
                f"Cola llena ({self._max_pending} peticiones pendientes). Reintenta en unos segundos."
            )
        self._pending += 1
        try:
            async with self._lock:
                return await run_in_threadpool(func, *args, **kwargs)
        finally:
            self._pending -= 1
