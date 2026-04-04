"""Tests for GraphInvocationGate (unload vs in-flight graph)."""

from __future__ import annotations

import asyncio

import pytest

from custom_components.home_generative_agent.core.graph_invocation import (
    GraphInvocationGate,
)


@pytest.mark.asyncio
async def test_graph_invocation_gate_wait_idle_after_hold() -> None:
    """wait_idle completes once hold() exits."""
    gate = GraphInvocationGate()
    assert await gate.wait_idle(timeout_s=0.01)

    async def _work() -> None:
        async with gate.hold():
            await asyncio.sleep(0.05)

    task = asyncio.create_task(_work())
    wait = asyncio.create_task(gate.wait_idle(timeout_s=2.0))
    await asyncio.sleep(0.01)
    assert not wait.done()
    await task
    await wait
    assert await gate.wait_idle(timeout_s=0.01)
