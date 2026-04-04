"""LangGraph astream_events → ChatLog delta assembly for Assist."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping  # noqa: TC003
from typing import Any

from homeassistant.core import HomeAssistant  # noqa: TC002
from homeassistant.helpers import intent  # noqa: TC002
from langchain_core.messages import AIMessageChunk

from .assist_chat_log import (
    _delta_stream,
    iter_replay_deltas_after_live_stream,
    turn_messages_for_chat_log,
)
from .assist_reasoning_trace import build_assist_reasoning_trace
from .conversation_helpers import (
    _convert_schema_json_to_yaml,
    _fix_entity_ids_in_text,
    _maybe_fix_dashboard_entities,
)
from .streaming_assist import (
    PunctuationBuffer,
    ThinkingStateMachine,
    text_from_ai_message_chunk,
)


async def iter_merged_chat_log_deltas(  # noqa: PLR0912, PLR0913, PLR0915
    app: Any,
    app_input: Any,
    app_config: Any,
    *,
    hass: HomeAssistant,
    input_message_count: int,
    ha_tool_intent_responses: Mapping[str, intent.IntentResponse],
    conf_schema_first_yaml: bool,
    conf_debug_assist_trace: bool,
    response_holder: dict[str, Any],
) -> AsyncGenerator[dict[str, Any]]:
    """
    Consume ``astream_events`` (v2), emit live answer deltas, then tool/thinking replay.

    When nothing was streamed live, emits the full structured replay (same as batch).
    """
    think = ThinkingStateMachine()
    punct = PunctuationBuffer()
    started_live = False
    emitted_live_content = False
    async for ev in app.astream_events(app_input, app_config, version="v2"):
        if ev.get("event") != "on_chat_model_stream":
            continue
        meta = ev.get("metadata") or {}
        if meta.get("langgraph_node") != "agent":
            continue
        chunk = ev.get("data", {}).get("chunk")
        if not isinstance(chunk, AIMessageChunk):
            continue
        if getattr(chunk, "tool_call_chunks", None):
            continue
        text = text_from_ai_message_chunk(chunk)
        if not text:
            continue
        for seg in think.feed(text):
            for sent in punct.feed(seg):
                if not started_live:
                    yield {"role": "assistant"}
                    started_live = True
                yield {"content": sent}
                emitted_live_content = True
    tail_f = think.flush()
    if tail_f:
        for sent in punct.feed(tail_f):
            if not started_live:
                yield {"role": "assistant"}
                started_live = True
            yield {"content": sent}
            emitted_live_content = True
    rem = punct.flush()
    if rem:
        if not started_live:
            yield {"role": "assistant"}
            started_live = True
        yield {"content": rem}
        emitted_live_content = True

    snap = await app.aget_state(app_config)
    state_values = snap.values
    response_holder["values"] = state_values

    fc = state_values["messages"][-1].content
    if isinstance(fc, str):
        if conf_schema_first_yaml:
            fc = _maybe_fix_dashboard_entities(fc, hass)
        else:
            fc = _fix_entity_ids_in_text(fc, hass)
        fc = _convert_schema_json_to_yaml(fc, conf_schema_first_yaml)
    final_str = fc if isinstance(fc, str) else str(fc)
    reasoning_plain = build_assist_reasoning_trace(state_values)
    has_native_thinking = bool(state_values.get("redacted_thinking_chunks"))

    if started_live and emitted_live_content:
        async for d in iter_replay_deltas_after_live_stream(
            state_values["messages"],
            input_message_count=input_message_count,
            reasoning_plain=reasoning_plain,
            has_native_thinking=has_native_thinking,
            debug_assist_trace=conf_debug_assist_trace,
            final_spoken_text=final_str,
            ha_tool_intent_responses=ha_tool_intent_responses,
        ):
            yield d
    else:
        turn = turn_messages_for_chat_log(state_values["messages"], input_message_count)
        async for d in _delta_stream(
            turn,
            reasoning_plain=reasoning_plain,
            has_native_thinking=has_native_thinking,
            debug_assist_trace=conf_debug_assist_trace,
            final_spoken_text=final_str,
            ha_tool_intent_responses=ha_tool_intent_responses,
            omit_final_spoken_content=False,
        ):
            yield d
