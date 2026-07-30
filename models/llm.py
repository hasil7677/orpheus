import asyncio
import os
import queue
import threading
from typing import AsyncGenerator

from groq import Groq

from config import ModelConfig

_SENTINEL = object()


class LLMProvider:
    """Groq streaming client (the cloud 'brain' in the local+cloud hybrid).

    The Groq SDK's stream is a blocking iterator over network I/O. We pump it
    from a background thread into a queue so the async orchestrator's event
    loop never blocks on it (this matters for barge-in: the loop must stay
    free to notice a cancellation while a Groq response is still streaming in).
    """

    def __init__(self, config: ModelConfig):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy local/.env.example to local/.env "
                "and add a free key from https://console.groq.com/keys"
            )
        self.client = Groq(api_key=api_key)
        self.config = config

    async def generate_stream(
        self, user_text: str, conversation_history: list[dict]
    ) -> AsyncGenerator[str, None]:
        conversation_history.append({"role": "user", "content": user_text})

        loop = asyncio.get_running_loop()
        q: queue.Queue = queue.Queue()

        def _pump():
            try:
                extra_body = {}
                if "gpt-oss" in self.config.llm_model:
                    # Keep chain-of-thought minimal — a voice assistant needs
                    # fast TTFT, not deliberation, and reasoning tokens would
                    # otherwise delay the first spoken word. Passed via
                    # extra_body since the installed groq SDK predates this
                    # param's typed support.
                    extra_body["reasoning_effort"] = "low"

                stream = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": self.config.llm_system_prompt},
                        *conversation_history,
                    ],
                    stream=True,
                    temperature=0.7,
                    max_tokens=self.config.llm_max_tokens,
                    extra_body=extra_body,
                )
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        loop.call_soon_threadsafe(q.put_nowait, token)
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

        threading.Thread(target=_pump, daemon=True).start()

        full_response = ""
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            full_response += item
            yield item

        conversation_history.append({"role": "assistant", "content": full_response})

        # Trim in place (the orchestrator holds this same list across turns) —
        # keep only the most recent N user+assistant pairs so a long-running
        # conversation doesn't grow the request payload unboundedly.
        max_messages = self.config.llm_max_history_turns * 2
        if len(conversation_history) > max_messages:
            del conversation_history[: len(conversation_history) - max_messages]
