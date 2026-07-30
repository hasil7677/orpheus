"""Unit test: conversation_history stays bounded across many turns.

No network/API key needed — Groq's client is faked so this runs offline.

Run: .venv\\Scripts\\python.exe tests\\test_history_trim.py
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")

from config import ModelConfig
from models.llm import LLMProvider


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def create(self, **kwargs):
        return [_FakeChunk("ok.")]


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class FakeGroqClient:
    def __init__(self, api_key=None):
        self.chat = _FakeChat()


async def main():
    with patch("models.llm.Groq", FakeGroqClient):
        config = ModelConfig(llm_max_history_turns=2)
        llm = LLMProvider(config)

        history: list[dict] = []
        for i in range(6):
            tokens = []
            async for token in llm.generate_stream(f"message {i}", history):
                tokens.append(token)

        max_messages = config.llm_max_history_turns * 2
        assert len(history) == max_messages, (
            f"expected history capped at {max_messages} messages, got {len(history)}"
        )
        assert history[0]["content"] == "message 4", (
            f"expected oldest kept turn to be 'message 4', got {history[0]['content']!r} — "
            f"trimming should keep the most recent turns, not the oldest"
        )
        print(f"OK: history capped at {len(history)} messages after 6 turns "
              f"(limit={config.llm_max_history_turns} turns).")
        print("\nHistory trim test passed.")


if __name__ == "__main__":
    asyncio.run(main())
