"""
Assist pipeline ``chat_log_delta_listener`` parity (sticky role + TTS).

Mirrors ``homeassistant/components/assist_pipeline/pipeline.py`` (chat_log_delta_listener):
only assistant-role deltas with ``content`` feed streaming TTS; ``role`` is
updated only when the delta dict includes ``"role"``.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from custom_components.home_generative_agent.core.assist_chat_log import (
    _delta_stream,
)


def _simulate_pipeline_tts_chunks(deltas: list[dict[str, Any]]) -> list[str]:
    """
    Return assistant text chunks that would be enqueued for streaming TTS.

    Simplified from ``Pipeline._prepare_intent_recognition`` chat_log_delta_listener.
    """
    chat_log_role: str | None = None
    out: list[str] = []
    for delta in deltas:
        if role := delta.get("role"):
            chat_log_role = role
        if chat_log_role != "assistant":
            continue
        if content := delta.get("content"):
            out.append(content)
    return out


def test_pipeline_drops_content_after_tool_result_without_new_assistant_role() -> None:
    """Content-only deltas after tool_result are ignored until role switches back."""
    deltas = [
        {"role": "assistant", "content": "I'll check."},
        {
            "role": "tool_result",
            "tool_call_id": "1",
            "tool_name": "x",
            "tool_result": {},
        },
        {"content": "The answer is 42."},
    ]
    assert _simulate_pipeline_tts_chunks(deltas) == ["I'll check."]


def test_pipeline_accepts_content_after_explicit_assistant_role() -> None:
    """Official integrations emit ``{"role": "assistant"}`` before post-tool answer text."""
    deltas = [
        {"role": "assistant", "content": "I'll check."},
        {
            "role": "tool_result",
            "tool_call_id": "1",
            "tool_name": "x",
            "tool_result": {},
        },
        {"role": "assistant"},
        {"content": "The answer is 42."},
    ]
    assert _simulate_pipeline_tts_chunks(deltas) == [
        "I'll check.",
        "The answer is 42.",
    ]


@pytest.mark.asyncio
async def test_delta_stream_emits_assistant_role_before_final_content_after_tool() -> (
    None
):
    """HGA replay must match OpenAI entity ordering: tool_result then assistant then body."""
    turn = [
        AIMessage(
            content="One moment.",
            tool_calls=[
                {
                    "name": "mochi_seek",
                    "args": {"queries": ["weather"]},
                    "id": "call_w",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="weather entities yaml",
            tool_call_id="call_w",
            name="mochi_seek",
        ),
    ]

    async def _collect(gen: Any) -> list[dict[str, Any]]:
        return [x async for x in gen]

    deltas = await _collect(
        _delta_stream(
            turn,
            reasoning_plain="",
            has_native_thinking=False,
            debug_assist_trace=False,
            final_spoken_text="Tomorrow looks rainy.",
            ha_tool_intent_responses=None,
            omit_final_spoken_content=False,
        )
    )

    tr_idx = next(i for i, d in enumerate(deltas) if d.get("role") == "tool_result")
    after = deltas[tr_idx + 1 :]
    assert after[0].get("role") == "assistant"
    assert any(d.get("content") == "Tomorrow looks rainy." for d in after)
    merged_tts = _simulate_pipeline_tts_chunks(deltas)
    assert "One moment." in "".join(merged_tts)
    assert "Tomorrow looks rainy." in "".join(merged_tts)
