"""Smoke test: Groq streaming chat completion works with the configured API key.

Run: .venv\\Scripts\\python.exe tests\\test_llm.py
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from config import AppConfig
from models.llm import LLMProvider


async def main():
    config = AppConfig()
    llm = LLMProvider(config.models)

    history = []
    tokens = []
    async for token in llm.generate_stream("Say hello in exactly three words.", history):
        tokens.append(token)

    response = "".join(tokens)
    print(f"Groq response: \"{response}\"")
    assert response.strip(), "Got an empty response from Groq"

    print("\nLLM smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
