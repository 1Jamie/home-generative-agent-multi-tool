"""
Append LangGraph turn output to Home Assistant ChatLog (Assist 2026.4+ UI).

## How "Show details" actually works

The Assist frontend (`ha-assist-chat.ts`) builds the chat message from
``intent-progress`` pipeline events carrying ``chat_log_delta`` payloads.
``message.thinking`` and ``message.tool_calls`` are only populated from those
deltas — NOT from the static ``CONTENT_ADDED`` chat-log subscriber events.

The pipeline fires ``intent-progress`` through a ``chat_log_delta_listener``
that is attached to ``ChatLog.delta_listener``.  That listener is invoked
by ``ChatLog.async_add_delta_content_stream``, not by
``async_add_assistant_content_without_tools``.

Therefore the correct path is:

  1. Synthesise an async-generator of delta dicts from LangGraph output.
  2. Drive it through ``chat_log.async_add_delta_content_stream``.
  3. Consume the generator (we discard the yielded content objects — they are
     already stored on ``chat_log.content`` internally).

This ensures every delta reaches the pipeline listener and the frontend.

Official reference (Home Assistant core): ``openai_conversation`` maps one API
stream per iteration via ``entity._transform_stream`` and
``entity._async_handle_chat_log`` — see
``core/homeassistant/components/openai_conversation/entity.py``. That pattern
yields explicit ``{"role": "assistant"}`` before tool rows and again before the
next assistant text so the Assist pipeline's sticky ``chat_log_role`` (see
``assist_pipeline/pipeline.py`` ``chat_log_delta_listener``) stays correct.
HGA's ``_delta_stream`` mirrors that ordering.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Mapping, Sequence  # noqa: TC003
from typing import Any

from homeassistant.components.conversation.chat_log import (
    AssistantContent,
    ChatLog,
    ToolResultContent,
)
from homeassistant.helpers import intent, llm
from homeassistant.util.ulid import ulid_now
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

_LOGGER = logging.getLogger(__name__)
_DELTA_LOG_PREVIEW_CHARS = 80


def _summarize_delta_for_log(delta: dict[str, Any]) -> str:
    """Short single-line summary for DEBUG tracing (avoids huge tool_result bodies)."""
    parts: list[str] = []
    if "role" in delta:
        parts.append(f"role={delta['role']!r}")
    if "content" in delta:
        raw = delta["content"]
        if isinstance(raw, str):
            prev = raw[:_DELTA_LOG_PREVIEW_CHARS] + (
                "..." if len(raw) > _DELTA_LOG_PREVIEW_CHARS else ""
            )
            parts.append(f"content={len(raw)}ch {prev!r}")
        else:
            parts.append(f"content={raw!r}")
    if "tool_calls" in delta:
        tc = delta["tool_calls"]
        n = len(tc) if isinstance(tc, list) else "?"
        parts.append(f"tool_calls={n}")
    if "thinking_content" in delta:
        parts.append("thinking_content=yes")
    if delta.get("role") == "tool_result":
        parts.append(f"tool_name={delta.get('tool_name')!r}")
    return " ".join(parts) if parts else repr(delta)


async def _async_iter_with_optional_delta_logging(
    stream: AsyncGenerator[dict[str, Any]],
    *,
    enabled: bool,
) -> AsyncGenerator[dict[str, Any]]:
    """Pass through deltas; log each at DEBUG when ``enabled``."""
    async for d in stream:
        if enabled:
            _LOGGER.debug("ChatLog delta: %s", _summarize_delta_for_log(d))
        yield d


def ensure_chat_log_ends_with_assistant(
    chat_log: ChatLog,
    *,
    agent_id: str,
) -> None:
    """
    Align the chat log with ``async_get_result_from_chat_log`` invariants.

    ``conversation.async_get_result_from_chat_log`` requires ``chat_log.content[-1]``
    to be ``AssistantContent``. ``ChatLog.async_add_delta_content_stream`` can leave
    the last row as ``ToolResultContent`` when the closing assistant delta had no text
    and no thinking (e.g. live stream already emitted the answer, replay omits body).
    """
    if not chat_log.content:
        return
    last = chat_log.content[-1]
    if not isinstance(last, ToolResultContent):
        return
    chat_log.async_add_assistant_content_without_tools(
        AssistantContent(agent_id=agent_id, content=None)
    )


# ── Turn slicing ────────────────────────────────────────────────────────────


def turn_messages_for_chat_log(
    messages: Sequence[AnyMessage], input_message_count: int
) -> list[AnyMessage]:
    """
    Return messages added by this invoke (stable against summarisation).

    Uses the pre-invoke message count as the primary boundary. If that yields
    nothing (e.g. the checkpointer shrank the list), falls back to messages
    after the last ``HumanMessage``.
    """
    if input_message_count < len(messages):
        sliced = list(messages[input_message_count:])
        if sliced:
            return sliced
    last_human = -1
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            last_human = i
    if last_human >= 0:
        return list(messages[last_human + 1 :])
    return []


# ── LangChain → HA type helpers ─────────────────────────────────────────────


def _tool_call_to_external_input(tc: Any) -> llm.ToolInput:
    """Map a LangChain tool call dict/object to an external ``ToolInput``."""
    if isinstance(tc, dict):
        name = str(tc.get("name") or "unknown")
        raw_args = tc.get("args")
        tc_id = tc.get("id")
    else:
        name = str(getattr(tc, "name", None) or "unknown")
        raw_args = getattr(tc, "args", None)
        tc_id = getattr(tc, "id", None)
    args = raw_args if isinstance(raw_args, dict) else {}
    call_id = str(tc_id) if tc_id else ulid_now()
    return llm.ToolInput(
        tool_name=name,
        tool_args=args,
        id=call_id,
        external=True,
    )


def _ai_text(msg: AIMessage) -> str | None:
    """Return stripped text content of an AIMessage, or None if empty."""
    content = msg.content
    if isinstance(content, str):
        return content.strip() or None
    return str(content).strip() or None


def _coerce_tool_result(content: Any) -> Any:
    """Return content in a JSON-safe form for ``ToolResultContent.tool_result``."""
    if isinstance(content, (dict, list, str, int, float, bool)) or content is None:
        return content
    return str(content)


# ── Delta synthesis ─────────────────────────────────────────────────────────


async def _delta_stream(  # noqa: PLR0913
    turn: Sequence[AnyMessage],
    *,
    reasoning_plain: str,
    has_native_thinking: bool,
    debug_assist_trace: bool,
    final_spoken_text: str,
    ha_tool_intent_responses: Mapping[str, intent.IntentResponse] | None,
    omit_final_spoken_content: bool = False,
) -> AsyncGenerator[dict[str, Any]]:
    """
    Yield ``AssistantContentDeltaDict`` / ``ToolResultContentDeltaDict`` items.

    Row order (mirrors the OpenAI entity streaming pattern):
      - Per-tool-call ``AssistantContent`` rows  (external ``tool_calls``).
      - ``ToolResultContent`` rows for each tool response.
      - Final ``AssistantContent`` with ``content`` + optional ``thinking_content``.

    Yielding a ``{"role": "assistant"}`` signals the start of a new assistant
    message to ``async_add_delta_content_stream``.
    """
    show_trace = has_native_thinking or debug_assist_trace

    started_first_message = False

    for msg in turn:
        if isinstance(msg, AIMessage):
            tcalls = getattr(msg, "tool_calls", None) or []
            if not tcalls:
                continue
            inputs = [_tool_call_to_external_input(tc) for tc in tcalls]
            if started_first_message:
                yield {"role": "assistant"}
            else:
                yield {"role": "assistant"}
                started_first_message = True
            text = _ai_text(msg)
            if text:
                yield {"content": text}
            yield {"tool_calls": inputs}

        elif isinstance(msg, ToolMessage):
            tid_key = str(getattr(msg, "tool_call_id", "") or "")
            if ha_tool_intent_responses and tid_key in ha_tool_intent_responses:
                tool_result_payload: Any = llm.IntentResponseDict(
                    ha_tool_intent_responses[tid_key]
                )
            else:
                tool_result_payload = _coerce_tool_result(msg.content)
            yield {
                "role": "tool_result",
                "tool_call_id": tid_key,
                "tool_name": msg.name or "tool",
                "tool_result": tool_result_payload,
            }

    # Final assistant message: answer text + optional thinking_content.
    # Only emit the role signal when there is something to follow it.
    # An empty {"role": "assistant"} leaves the pipeline delta_listener in an
    # "assistant about to speak" state with nothing following, which makes the
    # Assist frontend show "..." indefinitely after a live-streamed answer.
    will_have_thinking = show_trace and bool(reasoning_plain.strip())
    will_have_content = not omit_final_spoken_content and bool(
        final_spoken_text.strip()
    )

    if will_have_thinking or will_have_content:
        yield {"role": "assistant"}
        if will_have_thinking:
            yield {"thinking_content": reasoning_plain.strip()}
        if will_have_content:
            yield {"content": final_spoken_text.strip()}


# ── Public entry point ───────────────────────────────────────────────────────


async def append_langgraph_turn_to_chat_log(  # noqa: PLR0913
    chat_log: ChatLog,
    agent_id: str,
    graph_messages: Sequence[AnyMessage],
    *,
    input_message_count: int,
    reasoning_plain: str,
    has_native_thinking: bool,
    debug_assist_trace: bool,
    final_spoken_text: str,
    ha_tool_intent_responses: Mapping[str, intent.IntentResponse] | None = None,
    omit_final_spoken_content: bool = False,
) -> None:
    """
    Drive LangGraph output through ChatLog so the Assist UI receives deltas.

    Uses ``chat_log.async_add_delta_content_stream`` which:
      - Calls ``ChatLog.delta_listener`` for every delta →
        pipeline fires ``intent-progress chat_log_delta`` →
        frontend ``ha-assist-chat`` populates ``message.thinking`` /
        ``message.tool_calls`` → "Show details" button appears.
      - Stores each ``AssistantContent`` / ``ToolResultContent`` on
        ``chat_log.content`` for persistence and
        ``async_get_result_from_chat_log``.

    ``thinking_content`` is only set when:
    - The model produced native thinking blocks (``has_native_thinking``), OR
    - ``debug_assist_trace`` is enabled (the "Debug: populate Assist Show
      details" option in HGA settings).
    """
    turn = turn_messages_for_chat_log(graph_messages, input_message_count)

    stream = _delta_stream(
        turn,
        reasoning_plain=reasoning_plain,
        has_native_thinking=has_native_thinking,
        debug_assist_trace=debug_assist_trace,
        final_spoken_text=final_spoken_text,
        ha_tool_intent_responses=ha_tool_intent_responses,
        omit_final_spoken_content=omit_final_spoken_content,
    )

    async for _ in chat_log.async_add_delta_content_stream(agent_id, stream):  # type: ignore[arg-type]
        pass


async def append_langgraph_turn_to_chat_log_from_delta_generator(
    chat_log: ChatLog,
    agent_id: str,
    delta_stream: AsyncGenerator[dict[str, Any]],
    *,
    debug_delta_log: bool = False,
) -> None:
    """Drive ``ChatLog`` from a pre-built async generator of delta dicts."""
    logged = _async_iter_with_optional_delta_logging(
        delta_stream, enabled=debug_delta_log
    )
    async for _ in chat_log.async_add_delta_content_stream(agent_id, logged):  # type: ignore[arg-type]
        pass


async def iter_replay_deltas_after_live_stream(  # noqa: PLR0913
    graph_messages: Sequence[AnyMessage],
    *,
    input_message_count: int,
    reasoning_plain: str,
    has_native_thinking: bool,
    debug_assist_trace: bool,
    final_spoken_text: str,
    ha_tool_intent_responses: Mapping[str, intent.IntentResponse] | None,
) -> AsyncGenerator[dict[str, Any]]:
    """Tool rows, tool results, thinking; omit final ``content`` (streamed live)."""
    turn = turn_messages_for_chat_log(graph_messages, input_message_count)
    async for d in _delta_stream(
        turn,
        reasoning_plain=reasoning_plain,
        has_native_thinking=has_native_thinking,
        debug_assist_trace=debug_assist_trace,
        final_spoken_text=final_spoken_text,
        ha_tool_intent_responses=ha_tool_intent_responses,
        omit_final_spoken_content=True,
    ):
        yield d
