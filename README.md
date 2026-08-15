# orpheus

A real-time, voice-to-voice AI pipeline — speak to it, it talks back. About **2 seconds** from the moment you stop talking to the moment it starts, on a 4GB GTX 1650; see [Latency](#latency) for the per-stage breakdown.

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

## Latency

Measured on a GTX 1650 (4GB, driver 580.97) + Ryzen 5 5600H, Windows 11,
`openai/gpt-oss-20b` on Groq. 28 turns across 2 runs; the first turn of each
run is excluded as cold start. All numbers in milliseconds.

| Stage | Where | Median | p90 | Range |
|---|---|---:|---:|---:|
| VAD endpointing (`silence_timeout_ms`) | CPU | 650 | 650 | fixed |
| Speech-to-text (denoise + Moonshine) | local | 180 | 359 | 109–593 |
| LLM time-to-first-token | Groq (network) | 492 | 1032 | 375–1907 |
| Text-to-speech time-to-first-audio | GPU | 524 | 813 | 328–906 |
| **Mouth-to-ear** (end of speech → first audio out) | | **~2000** | ~2760 | 1510–3780 |

Measured end-to-end (STT + TTFA) was **1359ms median**, to which the fixed
650ms endpointing delay is added. Stage medians don't sum exactly to the
end-to-end median — each column is the median of its own distribution.

The three stages the console prints are *not* disjoint: `[TTS TTFA]` is timed
from the start of the LLM stream, so it already contains `[LLM TTFT]`. The
TTS row above is the derived incremental cost (`TTFA - TTFT`) — that is the
figure to compare against other TTS engines.

Latency scales with utterance length, mostly through STT:

| Utterance | STT | LLM TTFT | TTS TTFA | STT + TTFA |
|---|---:|---:|---:|---:|
| 1.26s | 133 | 446 | 867 | 992 |
| 1.77s | 180 | 469 | 953 | 1133 |
| 4.01s | 281 | 679 | 1438 | 1750 |

So the local half of the pipeline is comfortably sub-second — STT plus TTS is
~700ms at the median, and a short question is answered in under a second of
compute. The two things that dominate perceived delay are the fixed 650ms
endpointing pause and the round-trip to Groq, neither of which is GPU-bound.
Lowering `silence_timeout_ms` is the single most effective knob.

<details>
<summary>Measurement conditions</summary>

Utterances were played into the real pipeline at real-time cadence (32ms
int16 chunks) rather than spoken into a mic, so VAD, denoise, STT, LLM and
TTS all run exactly as in `cli.py`; only PyAudio is substituted. Timings are
the orchestrator's own instrumentation, unmodified.

Run on an otherwise-idle machine. This box has 5.86GB RAM against a ~5GB
pipeline working set, and under memory pressure (WSL/Docker/browsers
resident) the same code measured up to 10x slower with intermittent
allocation failures — if your numbers look nothing like these, check free
memory first.

ONNX Runtime execution providers actually active in this run: Kokoro on
`CUDAExecutionProvider`, Moonshine on `CPUExecutionProvider` — the
`moonshine_onnx` package constructs its sessions without a `providers=`
argument, so STT falls back to CPU regardless of the GPU build. The STT
figures above are therefore CPU figures.

</details>

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
   each turn so you can compare your hardware against the [Latency](#latency)
   table. Note that TTS-TTFA is timed from the start of the LLM stream and so
   includes LLM-TTFT — the three printed numbers do not sum. Try talking over
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
