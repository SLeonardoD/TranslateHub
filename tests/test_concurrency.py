import asyncio
import time

import pytest

from app.core.concurrency import RequestQueue
from app.core.exceptions import ServerBusyError


def test_requests_run_one_at_a_time_in_arrival_order():
    order: list[str] = []
    running = 0
    max_running = 0

    def work(name: str) -> str:
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        time.sleep(0.05)
        order.append(name)
        running -= 1
        return name

    async def scenario():
        queue = RequestQueue(max_pending=10)
        tasks = []
        for name in "abcd":
            tasks.append(asyncio.create_task(queue.run(work, name)))
            await asyncio.sleep(0.01)  # llegan en orden
        return await asyncio.gather(*tasks), queue.pending

    results, pending = asyncio.run(scenario())
    assert results == ["a", "b", "c", "d"]
    assert order == ["a", "b", "c", "d"]
    assert max_running == 1
    assert pending == 0


def test_event_loop_is_not_blocked_while_working():
    ticks = 0

    async def scenario():
        nonlocal ticks
        queue = RequestQueue(max_pending=5)

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        tick_task = asyncio.create_task(ticker())
        await queue.run(time.sleep, 0.2)
        tick_task.cancel()

    asyncio.run(scenario())
    assert ticks >= 10  # con el loop bloqueado serían 0


def test_full_queue_rejects_new_requests():
    async def scenario():
        queue = RequestQueue(max_pending=2)
        first = asyncio.create_task(queue.run(time.sleep, 0.1))
        second = asyncio.create_task(queue.run(time.sleep, 0.1))
        await asyncio.sleep(0.02)
        with pytest.raises(ServerBusyError, match="Cola llena"):
            await queue.run(time.sleep, 0)
        await asyncio.gather(first, second)
        await queue.run(time.sleep, 0)  # ya hay hueco otra vez

    asyncio.run(scenario())


def test_pending_counter_is_released_on_error():
    def boom():
        raise ValueError("x")

    async def scenario():
        queue = RequestQueue(max_pending=1)
        with pytest.raises(ValueError, match="x"):
            await queue.run(boom)
        return queue.pending

    assert asyncio.run(scenario()) == 0
