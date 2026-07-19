from __future__ import annotations

import asyncio
import contextlib
from dataclasses import asdict
import json
from typing import Any


class StatusHub:
    def __init__(self, runtime: Any, *, interval_s: float = 0.5) -> None:
        self.runtime = runtime
        self.interval_s = interval_s
        self._subscribers: dict[asyncio.Queue[str], str] = {}
        self._pump_task: asyncio.Task[None] | None = None

    def snapshot(self, topic: str = "full") -> dict[str, Any]:
        status_snapshot = getattr(self.runtime, "status_snapshot", None)
        if topic != "full" and callable(status_snapshot):
            return status_snapshot(topic)
        return asdict(self.runtime.state())

    def snapshot_json(self, topic: str = "full") -> str:
        return json.dumps(
            self.snapshot(topic),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    async def subscribe(self, topic: str = "full") -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self._subscribers[queue] = topic
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.create_task(self.pump_forever())
        await self._publish_one(
            queue,
            await asyncio.to_thread(self.snapshot_json, topic),
        )
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        self._subscribers.pop(queue, None)
        if not self._subscribers and self._pump_task is not None:
            self._pump_task.cancel()
            self._pump_task = None

    def close(self) -> None:
        self._subscribers.clear()
        if self._pump_task is not None:
            self._pump_task.cancel()
            self._pump_task = None

    async def broadcast(self) -> None:
        # Runtime diagnostics include model and vision structures. Build and
        # serialize one compact frame off the ASGI event loop, then share that
        # immutable text across all subscribers.
        payloads: dict[str, str] = {}
        for topic in set(self._subscribers.values()):
            payloads[topic] = await asyncio.to_thread(self.snapshot_json, topic)
        for queue, topic in list(self._subscribers.items()):
            await self._publish_one(queue, payloads[topic])

    async def pump_forever(self) -> None:
        try:
            while self._subscribers:
                await asyncio.sleep(self.interval_s)
                await self.broadcast()
        except asyncio.CancelledError:
            return

    async def _publish_one(
        self,
        queue: asyncio.Queue[str],
        payload: str,
    ) -> None:
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
        await queue.put(payload)
