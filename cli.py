"""
cli.py — Local hybrid voice pipeline entrypoint.

Stack: Silero VAD (CPU) -> Moonshine STT (GPU) -> Groq LLM (cloud) -> Kokoro TTS (GPU)

Usage:
    .venv\\Scripts\\python.exe cli.py
"""

import asyncio
import os
import threading

import numpy as np
import pyaudio
from dotenv import load_dotenv

from config import AppConfig
from core.orchestrator import PipelineOrchestrator

load_dotenv()


def _env_device_index(var_name: str) -> int | None:
    value = os.environ.get(var_name, "").strip()
    return int(value) if value else None


def start_audio_threads(orchestrator: PipelineOrchestrator, config: AppConfig, loop: asyncio.AbstractEventLoop):
    p = pyaudio.PyAudio()

    input_device = _env_device_index("AUDIO_INPUT_DEVICE")
    output_device = _env_device_index("AUDIO_OUTPUT_DEVICE")

    if input_device is not None:
        print(f"Using input device #{input_device}: {p.get_device_info_by_index(input_device)['name']}")
    if output_device is not None:
        print(f"Using output device #{output_device}: {p.get_device_info_by_index(output_device)['name']}")

    mic = p.open(
        format=pyaudio.paInt16,
        channels=config.audio.channels,
        rate=config.audio.sample_rate,
        input=True,
        input_device_index=input_device,
        frames_per_buffer=config.audio.chunk_samples,
    )

    # Kokoro yields whole-sentence-sized audio chunks (multiple seconds).
    # Writing one in a single speaker.write() call would block until it
    # fully plays, making barge-in unable to cut it off mid-sentence. Write
    # in small slices AND cap frames_per_buffer to match — without an
    # explicit small buffer, PortAudio's default internal buffer is large
    # enough that consecutive write() calls return almost immediately
    # (they're just filling the buffer, not pacing to real time), so several
    # seconds of audio can get handed to the driver before any interrupt
    # check has a chance to matter. Once audio's in the driver's buffer,
    # nothing in Python can pull it back.
    write_block_samples = max(1, int(0.02 * config.audio.tts_sample_rate))  # ~20ms

    speaker = p.open(
        format=pyaudio.paFloat32,
        channels=config.audio.channels,
        rate=config.audio.tts_sample_rate,
        output=True,
        output_device_index=output_device,
        frames_per_buffer=write_block_samples,
    )

    stop_event = threading.Event()

    def mic_thread():
        while not stop_event.is_set():
            raw = mic.read(config.audio.chunk_samples, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.int16)
            asyncio.run_coroutine_threadsafe(orchestrator.push_chunk(chunk), loop)

    def speaker_thread():
        while not stop_event.is_set():
            future = asyncio.run_coroutine_threadsafe(orchestrator.speaker_queue.get(), loop)
            audio_chunk = future.result().astype(np.float32)

            interrupted = False
            for start in range(0, len(audio_chunk), write_block_samples):
                if orchestrator.interrupt_speaker.is_set():
                    interrupted = True
                    break
                block = audio_chunk[start : start + write_block_samples]
                speaker.write(block.tobytes())

            if interrupted:
                # Belt and suspenders: stop/restart forces PortAudio to drop
                # whatever's still sitting in its own internal buffer beyond
                # what frames_per_buffer bounds from the Python side.
                speaker.stop_stream()
                speaker.start_stream()

    t1 = threading.Thread(target=mic_thread, daemon=True)
    t2 = threading.Thread(target=speaker_thread, daemon=True)
    t1.start()
    t2.start()

    def cleanup():
        stop_event.set()
        mic.close()
        speaker.close()
        p.terminate()

    return cleanup


async def main():
    config = AppConfig()
    orchestrator = PipelineOrchestrator(config)
    loop = asyncio.get_running_loop()

    cleanup = start_audio_threads(orchestrator, config, loop)

    print("\nVoice pipeline ready. Speak!")
    print("Press Ctrl+C to stop.\n")

    try:
        await orchestrator.run_loop()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        cleanup()
        print("\nShutting down...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
