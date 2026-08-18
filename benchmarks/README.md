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

## Endpointing evaluation

`eval_endpointing.py` is a separate harness for one question: when should the
pipeline decide you have stopped talking? It runs both `endpoint_mode` settings
(`fixed`, `punctuation` — see `config.py`) over the same clips and reports the
two things that matter:

- **false cuts** — turns finalized while the speaker still had audio left,
- **response-ready** — ms from the true end of speech to the moment the LLM
  would have been called.

```
.venv\Scripts\python.exe benchmarks\make_pause_clips.py
.venv\Scripts\python.exe benchmarks\eval_endpointing.py benchmarks\clips_pauses 3
```

It stubs out the LLM and TTS — they are downstream of the decision being
measured, they add network variance that would swamp it, and skipping them
keeps the process near VAD+STT memory instead of the full ~5GB. The two modes
alternate clip by clip rather than running in blocks, because this box's STT
latency drifts by hundreds of ms over a few minutes and a blocked run would
hand that drift to whichever mode went second.

Ground truth comes from a `labels.json` next to the clips:

```json
{"my_clip_16k": {"speech_end_ms": 3120, "pauses": [[1180, 2050]]}}
```

`speech_end_ms` — the last instant the speaker was still talking — is the only
required field. `make_pause_clips.py` writes one automatically for the clips it
generates; for real recordings it has to be written by hand.

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

`results/2026-08-18-endpointing-eval-synthetic.log` — 3 spliced-pause clips x
3 reps x both modes, n=9 each. Against a fixed 650ms timeout, `punctuation`
took 156ms off the median response-ready time with a flat tail (p90 +63ms) and
turned 2 of 9 false cuts into clean single turns.

**That is not a result to ship on.** The clips are Kokoro speech with digital
silence spliced in, which is not what a human pause sounds like; n=9 on 3
sentences is nothing; and the box was between 48MB and 711MB available during
the run, so individual STT figures in that log swing 10x. The useful part is
the *failure mode* it exposes, which is robust across every run: Moonshine
invents terminal punctuation on a truncated utterance — "Can you explain how a
voice assistant pipeline works briefly" comes back mid-pause as "Can you
explain how a voice assistant pipes?", complete with a question mark the
speaker never finished saying. The gate believes it and cuts. It fails *more*
often when STT is fast, because a quick answer means a shorter, more truncated
partial. Punctuation from an ASR is a hint, not a turn-completion signal.

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
