"""Tests for mochi_seek condensed state and helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.home_generative_agent.agent.tools import (
    _mochi_seek_condense_entity,
    mochi_seek,
)
from custom_components.home_generative_agent.core.entity_index import (
    _embedding_content_for_entity,
)


def test_mochi_seek_condense_drops_ui_attrs_keeps_physical() -> None:
    """Strip icon, entity_picture, supported_features, friendly_name from attributes."""
    hass = MagicMock()
    st = MagicMock()
    st.entity_id = "light.living"
    st.domain = "light"
    st.state = "on"
    st.attributes = {
        "friendly_name": "Living",
        "icon": "mdi:light",
        "entity_picture": "http://x",
        "supported_features": 44,
        "brightness": 128,
        "color_temp": 300,
    }
    hass.states.get = MagicMock(return_value=st)

    out = _mochi_seek_condense_entity(
        hass, "light.living", area_lookup={"light.living": "Living room"}
    )
    assert out is not None
    assert out["entity_id"] == "light.living"
    assert out["state"] == "on"
    assert out["area"] == "Living room"
    attrs = out["attributes"]
    assert "icon" not in attrs
    assert "entity_picture" not in attrs
    assert "supported_features" not in attrs
    assert "friendly_name" not in attrs
    assert attrs["brightness"] == 128
    assert attrs["color_temp"] == 300


@pytest.mark.asyncio
async def test_mochi_seek_returns_warmup_when_index_not_ready() -> None:
    """When runtime_data.entity_index_ready is False, return warm-up message."""
    rd = MagicMock()
    rd.entity_index_ready = False
    cfg = {
        "configurable": {
            "hass": MagicMock(),
            "runtime_data": rd,
            "tool_mgr_data": {},
        }
    }
    store = MagicMock()
    out = await mochi_seek.coroutine(["kitchen"], config=cfg, store=store)
    assert "warming up" in out.lower()
    store.asearch.assert_not_called()


def test_embedding_content_includes_area_for_rag() -> None:
    """Index text must tie entities to area names (e.g. 'lights in the kitchen')."""
    st = MagicMock()
    st.entity_id = "light.kitchen_ceiling"
    st.domain = "light"
    st.attributes = {"friendly_name": "Ceiling Lights"}
    text = _embedding_content_for_entity(st, "Kitchen")
    assert "area:Kitchen" in text
    assert "in the Kitchen" in text
    assert "located in Kitchen" in text
    assert "light.kitchen_ceiling" in text
