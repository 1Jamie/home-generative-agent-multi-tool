"""Tool Configuration Manager subentry flow."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    SOURCE_USER,
    ConfigSubentry,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
)

from custom_components.home_generative_agent.const import (
    CONF_INSTRUCTION_RAG_INTENT_WEIGHT,
    CONF_INSTRUCTION_RELEVANCE_THRESHOLD,
    CONF_INSTRUCTION_RETRIEVAL_LIMIT,
    CONF_INSTRUCTIONS_CONFIG,
    CONF_TOOL_RELEVANCE_THRESHOLD,
    CONF_TOOL_RETRIEVAL_LIMIT,
    RECOMMENDED_INSTRUCTION_RAG_INTENT_WEIGHT,
    RECOMMENDED_INSTRUCTION_RELEVANCE_THRESHOLD,
    RECOMMENDED_INSTRUCTION_RETRIEVAL_LIMIT,
    RECOMMENDED_TOOL_RELEVANCE_THRESHOLD,
    RECOMMENDED_TOOL_RETRIEVAL_LIMIT,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    get_tool_manager_subentry,
)

_LOGGER = logging.getLogger(__name__)

# Constants for this flow's data model
CONF_TOOL_PROVIDERS = "tool_providers"
CONF_TOOLS_CONFIG = "tools"

CONF_SELECTED_PROVIDER = "selected_provider"
CONF_SELECTED_TOOL = "selected_tool"
CONF_SELECTED_INSTRUCTION = "selected_instruction"
CONF_INSTRUCTION_NAME_FIELD = "instruction_name"

ADD_NEW_INSTRUCTION_VALUE = "__add_new_instruction__"

_INTERNAL_TOOL_NAMES = (
    "get_and_analyze_camera_image",
    "get_camera_last_events",
    "upsert_memory",
    "get_entity_history",
    "confirm_sensitive_action",
    "alarm_control",
    "resolve_entity_ids",
    "write_yaml_file",
    "get_available_tools",
    "mochi_seek",
    "add_automation",
)


def _current_subentry(flow: ConfigSubentryFlow) -> ConfigSubentry | None:
    """Return the currently edited subentry."""
    entry = flow._get_entry()
    subentry_id = getattr(flow, "_subentry_id", None)
    if not subentry_id:
        subentry_id = flow.context.get("subentry_id")
    if subentry_id:
        return entry.subentries.get(subentry_id)
    return get_tool_manager_subentry(entry)


class ToolManagerSubentryFlow(ConfigSubentryFlow):
    """Config flow for Tool Manager."""

    def __init__(self) -> None:
        """Initialize."""
        self._payload: dict[str, Any] = {}
        self._provider_to_edit: str | None = None
        self._tool_to_edit: str | None = None
        self._instruction_to_edit: str | None = None

    def _ensure_payload(self) -> None:
        """Load or initialize ``self._payload`` from the current subentry."""
        if self._payload:
            return
        current = _current_subentry(self)
        self._payload = {
            CONF_TOOL_RETRIEVAL_LIMIT: RECOMMENDED_TOOL_RETRIEVAL_LIMIT,
            CONF_TOOL_RELEVANCE_THRESHOLD: RECOMMENDED_TOOL_RELEVANCE_THRESHOLD,
            CONF_INSTRUCTION_RETRIEVAL_LIMIT: RECOMMENDED_INSTRUCTION_RETRIEVAL_LIMIT,
            CONF_INSTRUCTION_RELEVANCE_THRESHOLD: RECOMMENDED_INSTRUCTION_RELEVANCE_THRESHOLD,
            CONF_INSTRUCTION_RAG_INTENT_WEIGHT: RECOMMENDED_INSTRUCTION_RAG_INTENT_WEIGHT,
            CONF_TOOL_PROVIDERS: {},
            CONF_TOOLS_CONFIG: {},
            CONF_INSTRUCTIONS_CONFIG: {},
        }
        if current:
            self._payload.update(current.data)

    def _provider_ids(self) -> list[str]:
        """Return tool provider IDs (Assist first, then others, then internal)."""
        apis = llm.async_get_apis(self.hass)
        providers = [llm.LLM_API_ASSIST] + [
            api.id for api in apis if api.id != llm.LLM_API_ASSIST
        ]
        providers.append("langchain_internal")
        return providers

    async def _active_tool_names(self) -> list[str]:
        """Collect tool names from registered LLM APIs plus built-in tools."""
        apis = llm.async_get_apis(self.hass)
        active_tools: set[str] = set()
        llm_context = llm.LLMContext(
            platform="home_generative_agent",
            context=None,
            language=None,
            assistant="conversation",
            device_id=None,
        )
        for api in apis:
            try:
                inst = await llm.async_get_api(self.hass, api.id, llm_context)
                for t in inst.tools:
                    active_tools.add(t.name)
            except Exception:
                pass
        active_tools.update(_INTERNAL_TOOL_NAMES)
        return sorted(active_tools)

    def _schedule_reload(self) -> None:
        entry = self._get_entry()
        self.hass.async_create_task(
            self.hass.config_entries.async_reload(entry.entry_id)
        )

    def _trigger_reindex(self) -> None:
        """Tell the RAG engine to reindex tools because tags changed."""
        entry = self._get_entry()
        entry.runtime_data.tools_version_hash = "FORCED_REINDEX"

    async def async_step_user(
        self, _user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Hub menu: global settings, manage lists, or finish."""
        self._ensure_payload()
        return self.async_show_menu(
            step_id="user",
            menu_options=(
                "global_settings",
                "manage_providers",
                "manage_tools",
                "manage_instructions",
                "finish",
            ),
            description_placeholders={
                "tool_manager_title": "RAG Tool & Prompt Configuration",
            },
        )

    async def async_step_global_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Edit global RAG sliders (tool / instruction retrieval limits and thresholds)."""
        self._ensure_payload()
        if user_input is not None:
            self._payload[CONF_TOOL_RETRIEVAL_LIMIT] = int(
                user_input[CONF_TOOL_RETRIEVAL_LIMIT]
            )
            self._payload[CONF_TOOL_RELEVANCE_THRESHOLD] = float(
                user_input[CONF_TOOL_RELEVANCE_THRESHOLD]
            )
            self._payload[CONF_INSTRUCTION_RETRIEVAL_LIMIT] = int(
                user_input[CONF_INSTRUCTION_RETRIEVAL_LIMIT]
            )
            self._payload[CONF_INSTRUCTION_RELEVANCE_THRESHOLD] = float(
                user_input[CONF_INSTRUCTION_RELEVANCE_THRESHOLD]
            )
            self._payload[CONF_INSTRUCTION_RAG_INTENT_WEIGHT] = float(
                user_input[CONF_INSTRUCTION_RAG_INTENT_WEIGHT]
            )
            return await self.async_step_user()

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_TOOL_RETRIEVAL_LIMIT,
                    default=self._payload.get(CONF_TOOL_RETRIEVAL_LIMIT),
                ): NumberSelector(NumberSelectorConfig(min=1, max=50, step=1)),
                vol.Required(
                    CONF_TOOL_RELEVANCE_THRESHOLD,
                    default=self._payload.get(CONF_TOOL_RELEVANCE_THRESHOLD),
                ): NumberSelector(NumberSelectorConfig(min=0.0, max=1.0, step=0.01)),
                vol.Required(
                    CONF_INSTRUCTION_RETRIEVAL_LIMIT,
                    default=self._payload.get(CONF_INSTRUCTION_RETRIEVAL_LIMIT),
                ): NumberSelector(NumberSelectorConfig(min=1, max=50, step=1)),
                vol.Required(
                    CONF_INSTRUCTION_RELEVANCE_THRESHOLD,
                    default=self._payload.get(CONF_INSTRUCTION_RELEVANCE_THRESHOLD),
                ): NumberSelector(NumberSelectorConfig(min=0.0, max=1.0, step=0.01)),
                vol.Required(
                    CONF_INSTRUCTION_RAG_INTENT_WEIGHT,
                    default=self._payload.get(
                        CONF_INSTRUCTION_RAG_INTENT_WEIGHT,
                        RECOMMENDED_INSTRUCTION_RAG_INTENT_WEIGHT,
                    ),
                ): NumberSelector(NumberSelectorConfig(min=0.0, max=1.0, step=0.01)),
            }
        )

        return self.async_show_form(
            step_id="global_settings",
            data_schema=schema,
            description_placeholders={
                "tool_manager_title": "RAG Tool & Prompt Configuration",
            },
        )

    async def async_step_manage_providers(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Pick a provider to configure."""
        self._ensure_payload()
        providers = self._provider_ids()
        if user_input is not None:
            self._provider_to_edit = str(user_input[CONF_SELECTED_PROVIDER])
            return await self.async_step_provider_editor()

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_SELECTED_PROVIDER,
                    default=providers[0],
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[SelectOptionDict(label=p, value=p) for p in providers],
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=False,
                    )
                ),
            }
        )
        return self.async_show_form(
            step_id="manage_providers",
            data_schema=schema,
        )

    async def async_step_manage_tools(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Pick a tool to configure."""
        self._ensure_payload()
        tools = await self._active_tool_names()
        if user_input is not None:
            self._tool_to_edit = str(user_input[CONF_SELECTED_TOOL])
            return await self.async_step_tool_editor()

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_SELECTED_TOOL,
                    default=tools[0],
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[SelectOptionDict(label=t, value=t) for t in tools],
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=False,
                    )
                ),
            }
        )
        return self.async_show_form(
            step_id="manage_tools",
            data_schema=schema,
        )

    async def async_step_manage_instructions(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Pick an instruction to edit or add a new one."""
        self._ensure_payload()
        instructions = self._payload.get(CONF_INSTRUCTIONS_CONFIG, {})
        option_rows: list[SelectOptionDict] = [
            SelectOptionDict(
                label="+ Add New Instruction",
                value=ADD_NEW_INSTRUCTION_VALUE,
            )
        ]
        for name in sorted(instructions.keys()):
            option_rows.append(SelectOptionDict(label=name, value=name))

        default_val = (
            ADD_NEW_INSTRUCTION_VALUE
            if not instructions
            else str(option_rows[1]["value"])
        )

        if user_input is not None:
            sel = str(user_input[CONF_SELECTED_INSTRUCTION])
            if sel == ADD_NEW_INSTRUCTION_VALUE:
                self._instruction_to_edit = None
            else:
                self._instruction_to_edit = sel
            return await self.async_step_instruction_editor()

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_SELECTED_INSTRUCTION,
                    default=default_val,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=option_rows,
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=False,
                    )
                ),
            }
        )
        return self.async_show_form(
            step_id="manage_instructions",
            data_schema=schema,
        )

    async def async_step_finish(
        self, _user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """
        Persist subentry data and reload.

        Must be async so the flow runner can await the step.
        """
        self._ensure_payload()
        return self.async_create_or_update()

    async def async_step_provider_editor(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if user_input is not None:
            providers = self._payload.setdefault(CONF_TOOL_PROVIDERS, {})
            prev = providers.get(self._provider_to_edit, {})
            providers[self._provider_to_edit] = {
                "enabled": user_input.get("enabled", True),
                "prompt": user_input.get("prompt", prev.get("prompt", "")),
                "tags": user_input.get("tags", prev.get("tags", "")),
            }
            return await self.async_step_user()

        current_cfg = self._payload.get(CONF_TOOL_PROVIDERS, {}).get(
            self._provider_to_edit, {}
        )
        schema = vol.Schema(
            {
                vol.Required(
                    "enabled", default=current_cfg.get("enabled", True)
                ): BooleanSelector(),
                vol.Optional(
                    "prompt",
                    default=current_cfg.get("prompt", ""),
                    description={"suggested_value": current_cfg.get("prompt", "")},
                ): TemplateSelector(),
                vol.Optional(
                    "tags",
                    default=current_cfg.get("tags", ""),
                    description={"suggested_value": current_cfg.get("tags", "")},
                ): TextSelector(),
            }
        )

        return self.async_show_form(
            step_id="provider_editor",
            data_schema=schema,
            description_placeholders={
                "provider_name": self._provider_to_edit or "",
                "provider_injection_label": "Provider Context (Tier 2 Prompt)",
                "semantic_tags_hint": "Comma-separated keywords to help the AI find this tool (e.g., 'chill, movie night, dim').",
            },
        )

    async def async_step_tool_editor(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if user_input is not None:
            tools = self._payload.setdefault(CONF_TOOLS_CONFIG, {})
            prev = tools.get(self._tool_to_edit, {})
            tools[self._tool_to_edit] = {
                "enabled": user_input.get("enabled", True),
                "prompt": user_input.get("prompt", prev.get("prompt", "")),
                "tags": user_input.get("tags", prev.get("tags", "")),
            }
            return await self.async_step_user()

        current_cfg = self._payload.get(CONF_TOOLS_CONFIG, {}).get(
            self._tool_to_edit, {}
        )
        schema = vol.Schema(
            {
                vol.Required(
                    "enabled", default=current_cfg.get("enabled", True)
                ): BooleanSelector(),
                vol.Optional(
                    "prompt",
                    default=current_cfg.get("prompt", ""),
                    description={"suggested_value": current_cfg.get("prompt", "")},
                ): TemplateSelector(),
                vol.Optional(
                    "tags",
                    default=current_cfg.get("tags", ""),
                    description={"suggested_value": current_cfg.get("tags", "")},
                ): TextSelector(),
            }
        )

        return self.async_show_form(
            step_id="tool_editor",
            data_schema=schema,
            description_placeholders={
                "tool_name": self._tool_to_edit or "",
                "tool_injection_label": "Specific Tool Instructions (Tier 3 Prompt)",
                "semantic_tags_hint": "Comma-separated keywords to help the AI find this tool (e.g., 'chill, movie night, dim').",
            },
        )

    async def async_step_instruction_editor(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Edit or create a custom instruction (Tier 1.5)."""
        self._ensure_payload()
        instructions = self._payload.setdefault(CONF_INSTRUCTIONS_CONFIG, {})
        is_new = self._instruction_to_edit is None

        if user_input is not None:
            if is_new:
                name_key = str(
                    user_input.get(CONF_INSTRUCTION_NAME_FIELD, "") or ""
                ).strip()
                if not name_key:
                    return await self._async_show_instruction_editor_form(
                        errors={
                            CONF_INSTRUCTION_NAME_FIELD: "instruction_name_required"
                        }
                    )
                instructions[name_key] = {
                    "enabled": user_input.get("enabled", True),
                    "prompt": user_input.get("prompt", ""),
                    "tags": user_input.get("tags", ""),
                }
            elif user_input.get("delete_entry"):
                instructions.pop(self._instruction_to_edit, None)
            else:
                prev = instructions.get(self._instruction_to_edit, {})
                instructions[self._instruction_to_edit] = {
                    "enabled": user_input.get("enabled", True),
                    "prompt": user_input.get("prompt", prev.get("prompt", "")),
                    "tags": user_input.get("tags", prev.get("tags", "")),
                }
            return await self.async_step_user()

        return await self._async_show_instruction_editor_form()

    async def _async_show_instruction_editor_form(
        self,
        *,
        errors: dict[str, str] | None = None,
    ) -> SubentryFlowResult:
        """Render instruction editor; new entries include a name field."""
        instructions = self._payload.get(CONF_INSTRUCTIONS_CONFIG, {})
        is_new = self._instruction_to_edit is None
        if is_new:
            current_cfg: dict[str, Any] = {}
        else:
            current_cfg = instructions.get(self._instruction_to_edit, {})

        schema_dict: dict[Any, Any] = {}
        if is_new:
            schema_dict[
                vol.Required(
                    CONF_INSTRUCTION_NAME_FIELD,
                    default="",
                )
            ] = TextSelector()

        schema_dict[
            vol.Required(
                "enabled",
                default=True if is_new else bool(current_cfg.get("enabled", False)),
            )
        ] = BooleanSelector()
        schema_dict[
            vol.Optional(
                "prompt",
                default=current_cfg.get("prompt", ""),
                description={"suggested_value": current_cfg.get("prompt", "")},
            )
        ] = TemplateSelector()
        schema_dict[
            vol.Optional(
                "tags",
                default=current_cfg.get("tags", ""),
                description={"suggested_value": current_cfg.get("tags", "")},
            )
        ] = TextSelector()
        if not is_new:
            schema_dict[vol.Optional("delete_entry", default=False)] = BooleanSelector()

        return self.async_show_form(
            step_id="instruction_editor",
            data_schema=vol.Schema(schema_dict),
            errors=errors or {},
            description_placeholders={
                "instruction_name": self._instruction_to_edit or "New instruction",
                "instruction_injection_label": "Instruction Text (Tier 1.5 Prompt)",
                "semantic_tags_hint": "Comma-separated keywords to help the AI find this instruction.",
            },
        )

    def async_create_or_update(self) -> SubentryFlowResult:
        current = _current_subentry(self)
        self._trigger_reindex()
        self._schedule_reload()

        if current:
            return self.async_update_and_abort(
                self._get_entry(), current, data=self._payload, title="RAG Tool Manager"
            )

        if self.source == SOURCE_RECONFIGURE:
            self._source = SOURCE_USER
            self.context["source"] = SOURCE_USER

        return self.async_create_entry(title="RAG Tool Manager", data=self._payload)

    async_step_reconfigure = async_step_user
