"""Vector-store indexing for entity discovery (MochiSeek / mochi_seek)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from langgraph.store.base import BaseStore  # noqa: TC002

from ..agent.rag_embedding_text import truncate_for_embedding_index  # noqa: TID252
from ..snapshot.builder import _build_area_lookup  # noqa: TID252

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant, State

_LOGGER = logging.getLogger(__name__)

ENTITY_INDEX_NAMESPACE = ("system", "entities")
# Small chunks: LangGraph Postgres batches `aput` → `aembed_documents`. Keep batch
# token totals low; pair with per-document truncation (rag_embedding_text).
ENTITY_INDEX_BATCH_SIZE = 4
ENTITY_INDEX_INTER_CHUNK_SLEEP_S = 0.01
# Debounce before fingerprint + embed work; coalesces registry event bursts.
ENTITY_INDEX_DEBOUNCE_S = 5.0

ENTITY_INDEX_HASH_STORE_VERSION = 1
ENTITY_INDEX_HASH_STORE_KEY = "home_generative_agent.entity_index_hash"


class EntityIndexStore:
    """Persist last entity metadata fingerprint so restarts skip redundant embeds."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize JSON storage under ``.storage/``."""
        self._store = Store(
            hass, ENTITY_INDEX_HASH_STORE_VERSION, ENTITY_INDEX_HASH_STORE_KEY
        )

    async def async_load(self) -> str | None:
        """Return stored fingerprint hex digest, or None if missing/unreadable."""
        try:
            data = await self._store.async_load()
        except (HomeAssistantError, OSError, ValueError):
            return None
        if isinstance(data, dict):
            h = data.get("hash")
            return str(h) if h else None
        return None

    async def async_save(self, fingerprint: str) -> None:
        """Persist fingerprint after a successful full index."""
        try:
            await self._store.async_save({"hash": fingerprint})
        except (HomeAssistantError, OSError, ValueError):
            _LOGGER.warning("Could not persist entity index fingerprint")


async def gather_store_puts_in_chunks(
    tasks: list[Any],
    *,
    chunk_size: int = ENTITY_INDEX_BATCH_SIZE,
    sleep_s: float = ENTITY_INDEX_INTER_CHUNK_SLEEP_S,
) -> None:
    """
    Await store.aput coroutines in sequential chunks (embedding provider limits).

    Use for any vector indexing (tools, instructions, entities). Do not
    ``asyncio.gather(*hundreds_of_tasks)`` — the store will embed them in one batch.
    """
    if not tasks:
        return
    n = len(tasks)
    for i in range(0, n, chunk_size):
        await asyncio.gather(*tasks[i : i + chunk_size])
        await asyncio.sleep(sleep_s)


def build_entity_fingerprint(hass: HomeAssistant) -> str:
    """Stable digest when entity ids, names, areas, or domains change."""
    area_lookup = _build_area_lookup(hass)
    rows: list[str] = []
    for state in hass.states.async_all():
        fn = str(state.attributes.get("friendly_name") or "")
        area = str(area_lookup.get(state.entity_id) or "")
        rows.append(f"{state.entity_id}\t{state.domain}\t{fn}\t{area}")
    rows.sort()
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def _embedding_content_for_entity(state: State, area_name: str | None) -> str:
    """
    Build static text for embedding; excludes live state.

    Area is embedded explicitly (labeled + short natural phrase) so queries like
    "lights in the kitchen" match entities assigned to that area.
    """
    fn = str(state.attributes.get("friendly_name") or "")
    parts: list[str] = [
        state.entity_id,
        state.domain,
        fn,
    ]
    if area_name and str(area_name).strip():
        an = str(area_name).strip()
        parts.append(f"area:{an}")
        parts.append(f"in the {an}")
        parts.append(f"located in {an}")
    dc = state.attributes.get("device_class")
    if dc:
        parts.append(f"device_class:{dc}")
    unit = state.attributes.get("unit_of_measurement")
    if unit:
        parts.append(f"unit:{unit}")
    return truncate_for_embedding_index("\n".join(parts))


async def index_entities_for_mochi_seek(hass: HomeAssistant, store: BaseStore) -> None:
    """
    Upsert all entities using small sequential chunks (embedding provider safety).

    Each chunk is awaited before the next starts so we never queue thousands of aput
    coroutines or force the store to embed an oversized batch in one Ollama request.
    """
    area_lookup = _build_area_lookup(hass)
    namespace = ENTITY_INDEX_NAMESPACE
    states = hass.states.async_all()
    n = len(states)
    tasks = [
        store.aput(
            namespace,
            key=state.entity_id,
            value={
                "content": _embedding_content_for_entity(
                    state, area_lookup.get(state.entity_id)
                ),
                "entity_id": state.entity_id,
            },
        )
        for state in states
    ]
    await gather_store_puts_in_chunks(tasks)

    _LOGGER.debug(
        "MochiSeek entity index: upserted %d entities (%d per chunk)",
        n,
        ENTITY_INDEX_BATCH_SIZE,
    )
