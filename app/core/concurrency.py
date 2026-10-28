"""Cola FIFO de un solo trabajador para las traducciones.

El servidor atiende las traducciones de una en una (el equipo no da para más), pero
SIN bloquear el event loop: las peticiones esperan su turno con ``await`` y el trabajo
pesado corre en un hilo, así que ``/``, ``/models`` y ``/pairs`` responden siempre.

La cola también mide cuánto tarda por carácter, para poder estimar la espera de una
petición nueva (``GET /`` y las respuestas 503).
"""

import asyncio
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.core.exceptions import ClientDisconnectedError, ServerBusyError

_HISTORY = 50  # trabajos recientes que se usan para medir el rendimiento
_RETRY_MIN, _RETRY_MAX = 5, 120  # límites de la cabecera Retry-After (segundos)


def check_cancelled(cancel: threading.Event | None) -> None:
    """Corta el trabajo si el cliente ya no está esperando la respuesta."""
    if cancel is not None and cancel.is_set():
        raise ClientDisconnectedError("El cliente cerró la conexión; se abandona la traducción.")


class RequestQueue:
    """Ejecuta una función bloqueante cada vez, en orden de llegada (FIFO).

    ``asyncio.Lock`` despierta a los que esperan en orden de llegada, así que la
    cola es justa. Si ya hay ``max_pending`` peticiones (en curso + esperando),
    las nuevas se rechazan con ``ServerBusyError`` (HTTP 503).
    """

    def __init__(self, max_pending: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._max_pending = max_pending
        self._clock = clock
        self._lock = asyncio.Lock()
        self._pending = 0
        self._pending_chars = 0  # caracteres de todo lo pendiente (el trabajo en curso incluido)
        self._running_chars = 0
        self._running_since: float | None = None
        self._samples: deque[tuple[float, int]] = deque(maxlen=_HISTORY)  # (segundos, caracteres)

    @property
    def pending(self) -> int:
        return self._pending

    @property
    def max_pending(self) -> int:
        return self._max_pending

    @property
    def pending_chars(self) -> int:
        return self._pending_chars

    # ---------------- Estimación de espera ----------------

    def _seconds_per_char(self) -> float | None:
        chars = sum(c for _, c in self._samples)
        return sum(s for s, _ in self._samples) / chars if chars else None

    def _running_remaining_s(self, rate: float) -> float:
        if self._running_since is None:
            return 0.0
        return max(0.0, self._running_chars * rate - (self._clock() - self._running_since))

    def estimated_wait_s(self) -> float | None:
        """Segundos hasta vaciar la cola con el rendimiento reciente (None sin datos)."""
        rate = self._seconds_per_char()
        if rate is None:
            return None
        done_in_running = 0.0
        if self._running_since is not None:
            done_in_running = min(self._running_chars, (self._clock() - self._running_since) / rate)
        return max(0.0, (self._pending_chars - done_in_running) * rate)

    def retry_after_s(self) -> int:
        """Cuándo reintentar: lo que falta para que se libere un hueco (el trabajo en curso)."""
        rate = self._seconds_per_char()
        if rate is None:
            return _RETRY_MIN
        return max(_RETRY_MIN, min(_RETRY_MAX, math.ceil(self._running_remaining_s(rate))))

    def _record(self, seconds: float, chars: int) -> None:
        if chars > 0 and seconds > 0:
            self._samples.append((seconds, chars))

    # ---------------- Ejecución ----------------

    async def run(
        self, func: Callable[..., Any], /, *args: Any, chars: int = 0, **kwargs: Any
    ) -> Any:
        """Encola ``func``. ``chars`` es el tamaño del trabajo, solo para estimar esperas."""
        if self._pending >= self._max_pending:
            wait = self.estimated_wait_s()
            hint = f" Espera estimada: ~{wait:.0f} s." if wait is not None else ""
            raise ServerBusyError(
                f"Cola llena ({self._max_pending} peticiones pendientes).{hint} "
                "Reintenta en unos segundos.",
                retry_after=self.retry_after_s(),
                estimated_wait_s=wait,
            )
        self._pending += 1
        self._pending_chars += chars
        try:
            async with self._lock:
                started = self._clock()
                self._running_chars, self._running_since = chars, started
                try:
                    result = await run_in_threadpool(func, *args, **kwargs)
                finally:
                    self._running_chars, self._running_since = 0, None
                self._record(self._clock() - started, chars)  # solo trabajos terminados bien
                return result
        finally:
            self._pending -= 1
            self._pending_chars -= chars
