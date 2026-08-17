# Benchmarks

Tooling and raw data behind the [Latency](../README.md#latency) table.

The harness drives the **real** `PipelineOrchestrator` — real VAD, denoise,
Moonshine, Groq, Kokoro, real state machine. The only substitution is the
PyAudio mic thread: WAV chunks are pushed into `push_chunk()` at real-time
cadence, and `speaker_queue` is drained in place of the speaker thread. Every
number reported is the orchestrator's own `[STT]` / `[LLM TTFT]` /
`[TTS TTFA]` instrumentation, unmodified.

## Reproducing

```
.venv\Scripts\python.exe benchmarks\make_utterances.py
.venv\Scripts\python.exe benchmarks\trace_session.py > run1.log 2>&1
.venv\Scripts\python.exe benchmarks\trace_session.py > run2.log 2>&1
.venv\Scripts\python.exe benchmarks\analyze.py run1.log run2.log
```

`trace_session.py [clips_dir] [reps]` accepts any directory of WAVs at any
sample rate, so you can benchmark against recordings of your own voice rather
than the synthesized clips. To record the same three sentences through your
real mic:

```
.venv\Scripts\python.exe benchmarks\record_utterances.py
.venv\Scripts\python.exe benchmarks\trace_session.py benchmarks\clips_real 5 > real.log 2>&1
.venv\Scripts\python.exe benchmarks\analyze.py real.log
```

`record_utterances.py` opens the mic with the same PyAudio parameters as
`cli.py`, so the audio reaching the VAD is what the live pipeline sees. It
writes to `benchmarks/clips_real/`, which is git-ignored — your voice stays
local. The prompts are the same sentences `make_utterances.py` synthesizes,
so transcripts are directly comparable and any difference is attributable to
real-mic audio rather than to different words.

## Reading the numbers

**`[TTS TTFA]` is timed from the start of the LLM stream, so it already
contains `[LLM TTFT]`.** The three printed stages are not a disjoint
breakdown and must not be summed. The incremental TTS cost is `TTFA - TTFT`,
which is what `analyze.py` reports as `TTS-only` and what the top-level README
puts in its TTS row.

`analyze.py` drops each run's first turn by default (`--keep-cold` to keep
it). Cold start is dramatic and not representative — in the two saved runs it
was 6109ms and 2750ms TTFA against a 1148ms steady-state median.

A fixed `silence_timeout_ms` (650ms, `config.py`) is paid before any stage
begins. It is not in any printed timing but is very much in the user's
perception, so mouth-to-ear adds it.

## Saved results

`results/2026-08-15-gtx1650-run{1,2}.log` — 15 turns each, GTX 1650 4GB
(driver 580.97), Ryzen 5 5600H, Windows 11, `openai/gpt-oss-20b` on Groq,
otherwise-idle machine. `results/2026-08-15-gtx1650-summary.txt` is the
pooled `analyze.py` output (n=28).

Pooled medians: STT 180ms, LLM TTFT 492ms, incremental TTS 524ms,
end-to-end 1359ms, mouth-to-ear ~2009ms.

## Caveats on this data

- **The clips are synthesized, not spoken.** Kokoro output is cleaner than a
  real mic in a real room, so the STT figures are a best case. Timing of the
  pipeline itself is unaffected, but expect real speech to cost more in STT.
- **Moonshine ran on CPU.** `moonshine_onnx` constructs its
  `InferenceSession`s without a `providers=` argument, so STT falls back to
  `CPUExecutionProvider` even on a CUDA build. Kokoro correctly used
  `CUDAExecutionProvider`. An explicit-CUDA probe measured the Moonshine
  encoder at 31ms median on GPU vs 47ms on CPU.
- **LLM TTFT is network-bound** and will vary with your distance to Groq,
  their load, and the model. It is the widest distribution here (375-1907ms).
- **Memory pressure dominates everything if you let it.** This pipeline
  commits ~5GB. On the 5.86GB box these were measured on, running with
  WSL/Docker/browsers resident produced STT maxima of 9328ms and intermittent
  numpy allocation failures — roughly 10x worse than the same code on an idle
  machine. `trace_session.py` prints working set and private commit after
  every turn for exactly this reason. Check those before trusting a slow
  result.
