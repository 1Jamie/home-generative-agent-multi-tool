"""Tests for streaming Assist buffers (thinking strip + punctuation)."""

from __future__ import annotations

from custom_components.home_generative_agent.core.streaming_assist import (
    PunctuationBuffer,
    ThinkingStateMachine,
    text_from_ai_message_chunk,
)


def test_text_from_chunk_string() -> None:
    """Plain string content."""
    chunk = type("C", (), {"content": "hi"})()
    assert text_from_ai_message_chunk(chunk) == "hi"


def test_thinking_fsm_strips_block() -> None:
    """Thinking tags are removed from streamed fragments."""
    fsm = ThinkingStateMachine()
    parts: list[str] = []
    for piece in (
        "Before ",
        "<redacted_thinking>",
        "secret",
        "</redacted_thinking>",
        " After.",
    ):
        parts.extend(fsm.feed(piece))
    tail = fsm.flush()
    if tail:
        parts.append(tail)
    merged = "".join(parts)
    assert "secret" not in merged
    assert "Before" in merged
    assert "After" in merged


def test_punctuation_buffer_splits_sentences() -> None:
    """Flush on . ? ! boundaries."""
    buf = PunctuationBuffer()
    out = list(buf.feed("Hello world. Next"))
    assert out == ["Hello world."]
    out2 = list(buf.feed(" sentence!"))
    assert " sentence!" in "".join(out2)


def test_punctuation_buffer_flush_remainder() -> None:
    """Trailing text without punctuation is flushed at end."""
    buf = PunctuationBuffer()
    buf.feed("No punct end")
    assert buf.flush() == "No punct end"


def test_punctuation_buffer_flush_drops_lone_unicode_ellipsis() -> None:
    """Trailing U+2026 left after a sentence split is discarded, not emitted."""
    buf = PunctuationBuffer()
    # "Done." is split and emitted; "…" is left in the buffer as a model artifact.
    sentences = buf.feed("Done.\u2026")
    assert sentences == ["Done."]
    assert buf.flush() is None


def test_punctuation_buffer_flush_drops_trailing_dots() -> None:
    """Three-dot ellipsis artifact left after a sentence split is discarded."""
    buf = PunctuationBuffer()
    sentences = buf.feed("Done....")
    # "Done." is split first; ".." remains
    assert "Done." in sentences
    assert buf.flush() is None


def test_punctuation_buffer_flush_keeps_real_content_before_ellipsis() -> None:
    """Real content before a trailing ellipsis is preserved."""
    buf = PunctuationBuffer()
    # No sentence-ending punctuation — whole string stays in buffer.
    buf.feed("processing\u2026")
    assert buf.flush() == "processing"


def test_punctuation_buffer_flush_keeps_real_content_with_space_before_ellipsis() -> None:
    """Whitespace between content and trailing ellipsis does not swallow the content."""
    buf = PunctuationBuffer()
    buf.feed("loading \u2026")
    assert buf.flush() == "loading"
