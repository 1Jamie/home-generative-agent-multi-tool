"""Prevent Postgres pool close while LangGraph + vector indexing run."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class GraphInvocationGate:
    """Track in-flight graph runs and RAG indexing that use the shared DB pool."""

    def __init__(self) -> None:
        """Initialize idle tracking."""
        self._count = 0
        self._mutex = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        """Hold while pool-backed store/checkpointer work is active."""
        async with self._mutex:
            self._count += 1
            self._idle.clear()
        try:
            yield
        finally:
            async with self._mutex:
                self._count -= 1
                if self._count == 0:
                    self._idle.set()

    async def wait_idle(self, *, timeout_s: float = 120.0) -> bool:
        """Block until no active hold(), or until timeout. Returns False on timeout."""
        try:
            async with asyncio.timeout(timeout_s):
                await self._idle.wait()
        except TimeoutError:
            return False
        else:
            return True
