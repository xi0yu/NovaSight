from __future__ import annotations

import asyncio
import contextlib
from dataclasses import asdict
from typing import Any


class StatusHub:
    def __init__(self, runtime: Any, *, interval_s: float = 0.05) -> None:
        self.runtime = runtime
        self.interval_s = interval_s
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def snapshot(self) -> dict[str, Any]:
        return asdict(self.runtime.state())

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        await self._publish_one(queue, self.snapshot())
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    async def broadcast(self) -> None:
        payload = self.snapshot()
        for queue in list(self._subscribers):
            await self._publish_one(queue, payload)

    async def pump_forever(self) -> None:
        while True:
            await self.broadcast()
            await asyncio.sleep(self.interval_s)

    async def _publish_one(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        payload: dict[str, Any],
    ) -> None:
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
        await queue.put(payload)
