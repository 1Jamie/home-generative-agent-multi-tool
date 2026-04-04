"""Streaming helpers for Assist ChatLog: thinking tags + sentence buffering for TTS."""

from __future__ import annotations

import re
from typing import Any, Literal

# Opening tags (see core/utils.py _THINK_INNER).
_THINK_START = re.compile(
    r"<(?:think|redacted_thinking)\s*>",
    re.IGNORECASE,
)
_THINK_END = re.compile(
    r"</(?:think|redacted_thinking)\s*>",
    re.IGNORECASE,
)
_THINK_CLOSE_BUFFER_MAX = 96

# Trailing dots / Unicode ellipsis (…) that models emit as streaming artifacts.
# Stripped from PunctuationBuffer.flush() so they don't appear as the last
# visible fragment in the Assist UI.
_TRAILING_ELLIPSIS = re.compile(r"[.\u2026]+$")


def text_from_ai_message_chunk(chunk: Any) -> str:
    """Concatenate textual content from a streamed AIMessageChunk (incl. blocks)."""
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


class ThinkingStateMachine:
    """
    Strip `<think>` / `<redacted_thinking>` blocks from streamed tokens.

    Emits answer-only fragments suitable for Assist display and TTS. Partial tags
    at chunk boundaries are held until complete.
    """

    def __init__(self) -> None:
        """Initialize parser state."""
        self._mode: Literal["answer", "thinking"] = "answer"
        self._hold = ""

    def feed(self, chunk: str) -> list[str]:
        """Return answer-text segments emitted for this chunk (may be empty)."""
        if not chunk:
            return []
        buf = self._hold + chunk
        self._hold = ""
        emitted: list[str] = []
        while buf:
            if self._mode == "thinking":
                m = _THINK_END.search(buf)
                if m is None:
                    if len(buf) > _THINK_CLOSE_BUFFER_MAX:
                        buf = ""
                    else:
                        self._hold = buf
                        buf = ""
                    break
                buf = buf[m.end() :]
                self._mode = "answer"
                continue

            m_start = _THINK_START.search(buf)
            if m_start is None:
                lt = buf.rfind("<")
                if lt >= 0:
                    tail = buf[lt:]
                    if (
                        re.match(r"^<[^>]{0,48}$", tail)
                        and _THINK_START.search(tail) is None
                    ):
                        self._hold = tail
                        buf = buf[:lt]
                if buf:
                    emitted.append(buf)
                buf = ""
                break
            prefix = buf[: m_start.start()]
            if prefix:
                emitted.append(prefix)
            buf = buf[m_start.end() :]
            self._mode = "thinking"
        return emitted

    def flush(self) -> str | None:
        """Emit trailing answer text after the stream ends."""
        if self._mode == "thinking":
            self._hold = ""
            return None
        tail = self._hold
        self._hold = ""
        return tail or None


class PunctuationBuffer:
    """Buffer answer characters; flush on `.` `?` `!` for smoother TTS phrasing."""

    def __init__(self) -> None:
        """Initialize an empty buffer."""
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        """Return completed sentence fragments (each ends with . ? or !)."""
        if not chunk:
            return []
        self._buf += chunk
        out: list[str] = []
        while True:
            match = re.search(r"[.!?]", self._buf)
            if not match:
                break
            cut = match.end()
            sentence = self._buf[:cut]
            self._buf = self._buf[cut:]
            if sentence.strip():
                out.append(sentence)
        return out

    def flush(self) -> str | None:
        """Return any remainder after the last sentence boundary.

        Trailing dots and Unicode ellipsis characters (…) are stripped before
        returning — they are streaming artifacts the model emits while generating
        and should not appear as the final fragment in the Assist UI.
        """
        rest = _TRAILING_ELLIPSIS.sub("", self._buf).strip()
        self._buf = ""
        return rest or None
