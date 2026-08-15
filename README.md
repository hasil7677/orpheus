# orpheus

A real-time, voice-to-voice AI pipeline — speak to it, it talks back, with low enough latency to feel like a conversation.

Runs speech detection, transcription, and voice synthesis entirely on your own GPU (tested on a 4GB GTX 1650), with the LLM ("brain") outsourced to [Groq's](https://console.groq.com) free API — a 4GB card can't run a usable local LLM alongside STT+TTS, so this hybrid gets you privacy + low latency on the ear/voice while keeping response quality high.

## Demo

<video src="https://raw.githubusercontent.com/hasil7677/orpheus/main/demo.mp4" controls width="100%"></video>

(If the player above doesn't render, [watch/download it directly](demo.mp4).)

## Stack

| Stage | Model | Where it runs |
|---|---|---|
| Voice activity detection | [Silero VAD](https://github.com/snakers4/silero-vad) | CPU |
| Speech-to-text | [Moonshine](https://github.com/moonshine-ai/moonshine) (ONNX) | GPU |
| LLM | Groq-hosted (`openai/gpt-oss-20b`) | Cloud |
| Text-to-speech | [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) (82M, ONNX) | GPU |

## Setup

1. **Python 3.11 required** — `kokoro-onnx` doesn't support 3.14 yet, and PyTorch
   has no CUDA wheels for 3.14 yet either. This repo assumes a `py -3.11` venv:

   ```
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. **Groq API key** (free): sign up at https://console.groq.com/keys, then:

   ```
   copy .env.example .env
   ```
   and paste the key into `.env`.

3. **Kokoro model files** (not bundled in the pip package — download once):
   - `kokoro-v1.0.onnx` and `voices-v1.0.bin` from
     https://github.com/thewh1teagle/kokoro-onnx/releases
   - Place both in `assets/`.

4. **Smoke-test each component before running the full pipeline:**

   ```
   .venv\Scripts\python.exe tests\test_vad.py
   .venv\Scripts\python.exe tests\test_tts.py
   .venv\Scripts\python.exe tests\test_stt.py
   .venv\Scripts\python.exe tests\test_llm.py
   ```

   `test_tts.py` prints the active ONNX Runtime providers — confirm
   `CUDAExecutionProvider` shows up, otherwise it silently fell back to CPU
   and TTS will be much slower.

   Two more tests need no GPU/mic/API key at all (models/Groq are faked) and
   run in a couple seconds:

   ```
   .venv\Scripts\python.exe tests\test_barge_in.py
   .venv\Scripts\python.exe tests\test_history_trim.py
   ```

5. **Run it:**

   ```
   .venv\Scripts\python.exe cli.py
   ```

   Speak into your mic; console prints STT/LLM-TTFT/TTS-TTFA timings for
   each turn so you can see real latency on this hardware. Try talking over
   the AI mid-response to test barge-in.

## Choosing a specific mic/speaker

By default this uses your OS default input/output devices. To pick a
different one:

```
.venv\Scripts\python.exe list_devices.py
```

This prints each device's index. Copy the ones you want into `.env`:

```
AUDIO_INPUT_DEVICE=3
AUDIO_OUTPUT_DEVICE=5
```

Leave a value unset (or remove it) to fall back to the OS default for that
device.

## Notes

- `silence_timeout_ms` (config.py, default 650ms) controls how long a pause
  has to be before the utterance is considered "done." Lower = snappier but
  more likely to cut people off mid-thought; higher = more natural but slower.
- Noise reduction runs once per full utterance (`audio/postprocessor.py`),
  not per audio chunk — spectral gating needs more context than a single
  32ms VAD frame to work well, and doing it per-chunk was too slow anyway.
- Conversation history sent to the LLM is capped at `llm_max_history_turns`
  (config.py, default 10 user+assistant turn pairs) so a long-running
  conversation doesn't grow the request payload unboundedly.
- If `kokoro-onnx` errors on phonemization, install `espeak-ng` and make
  sure it's on `PATH`.

## License

MIT — see [LICENSE](LICENSE).
