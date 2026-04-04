"""Tests for MOCHI tool gate helpers (query keywords + actuation prefixes)."""

from __future__ import annotations

from custom_components.home_generative_agent.agent.graph import (
    _actuation_binding_tool_names_from_available,
    _available_tool_names,
    _filter_instruction_keys_respecting_enabled,
    _mochi_query_suggests_actuation,
    _mochi_retrieved_suggests_actuation,
    _tool_name_from_available_entry,
)
from custom_components.home_generative_agent.const import (
    LANGCHAIN_SYSTEM_PROMPT_NAMED_TOOL_NAMES,
    MOCHI_PRUNABLE_INFORMATIONAL_TOOL_NAMES,
)


def test_mochi_query_suggests_actuation_turn_on() -> None:
    assert _mochi_query_suggests_actuation("Please turn on the kitchen lights")


def test_mochi_query_suggests_actuation_toggle() -> None:
    assert _mochi_query_suggests_actuation("Toggle the bedroom light")


def test_mochi_query_suggests_actuation_negative() -> None:
    assert not _mochi_query_suggests_actuation("What is the status of my phone?")


def test_mochi_query_set_not_substring_offset() -> None:
    assert not _mochi_query_suggests_actuation("offset calibration")


def test_mochi_retrieved_hass_prefix() -> None:
    assert _mochi_retrieved_suggests_actuation({"HassTurnOn", "get_entity_history"})


def test_mochi_retrieved_langchain_actuation() -> None:
    assert _mochi_retrieved_suggests_actuation({"alarm_control"})


def test_mochi_retrieved_read_only_only() -> None:
    assert not _mochi_retrieved_suggests_actuation(
        {"get_entity_history", "get_camera_last_events"}
    )


def test_vlm_camera_tool_is_prompt_bound_not_mochi_prunable() -> None:
    assert "get_and_analyze_camera_image" in LANGCHAIN_SYSTEM_PROMPT_NAMED_TOOL_NAMES
    assert "get_and_analyze_camera_image" not in MOCHI_PRUNABLE_INFORMATIONAL_TOOL_NAMES


def test_tool_name_from_openai_style_dict() -> None:
    assert (
        _tool_name_from_available_entry(
            {
                "type": "function",
                "function": {"name": "HassTurnOn", "parameters": {}},
            }
        )
        == "HassTurnOn"
    )


def test_filter_instruction_keys_respecting_enabled() -> None:
    assert _filter_instruction_keys_respecting_enabled(
        ["a", "b"],
        {"a": {"enabled": True}, "b": {"enabled": False}},
    ) == ["a"]


def test_available_tool_names_from_openai_dicts() -> None:
    tools = [
        {"type": "function", "function": {"name": "mochi_seek", "parameters": {}}},
        {"type": "function", "function": {"name": "HassTurnOn", "parameters": {}}},
    ]
    assert _available_tool_names(tools) == {"mochi_seek", "HassTurnOn"}


def test_actuation_binding_tool_names_hass_and_langchain() -> None:
    available = [
        {"type": "function", "function": {"name": "HassTurnOn", "parameters": {}}},
        {"type": "function", "function": {"name": "alarm_control", "parameters": {}}},
        {"type": "function", "function": {"name": "GetLiveContext", "parameters": {}}},
        {"type": "function", "function": {"name": "mochi_seek", "parameters": {}}},
        {
            "type": "function",
            "function": {"name": "get_entity_history", "parameters": {}},
        },
        {"type": "function", "function": {"name": "add_automation", "parameters": {}}},
    ]
    names = _actuation_binding_tool_names_from_available(available)
    assert names == {"HassTurnOn", "alarm_control"}
    assert "add_automation" not in names
