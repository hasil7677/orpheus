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
pipeline decide you have stopped talking? It runs all three `endpoint_mode`
settings (`fixed`, `punctuation`, `turn_detector` — see `config.py`) over the
same clips and reports the two things that matter:

- **false cuts** — turns finalized while the speaker still had audio left,
- **response-ready** — ms from the true end of speech to the moment the LLM
  would have been called.

```
.venv\Scripts\python.exe benchmarks\make_pause_clips.py
.venv\Scripts\python.exe benchmarks\eval_endpointing.py benchmarks\clips_pauses 3
```

`turn_detector` mode needs `assets/smart-turn-v3.2-cpu.onnx` on disk (~8MB,
not committed — same pattern as Kokoro's weights). Download it from
https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx
and `pip install transformers` (already in `requirements.txt`) for its
mel-spectrogram feature extraction; see `models/turn_detector.py`.

It stubs out the LLM and TTS — they are downstream of the decision being
measured, they add network variance that would swamp it, and skipping them
keeps the process near VAD+STT memory instead of the full ~5GB. The modes
alternate clip by clip rather than running in blocks, because this box's STT
latency drifts by hundreds of ms over a few minutes and a blocked run would
hand that drift to whichever mode went last.

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

`results/2026-08-19-endpointing-eval-real.log` — same harness, real recorded
speech: 8 clips (`benchmarks/clips_real/`, `record_pause_utterances.py`) x 3
reps x both modes, n=24 each. Labels are Silero VAD's own last-voiced-chunk
timestamp (`config.vad.threshold`, same model the pipeline runs), not the
amplitude-threshold heuristic `record_pause_utterances.py` auto-writes — see
"Caveats" below, that heuristic's output was unusable on this set. Against a
fixed 650ms timeout, `punctuation` took 345ms off the median response-ready
time (897ms -> 552ms) with a *better* max too (2065ms -> 1443ms), but cost
**5 more false-cut events** (15 -> 20 across 24 turns) — the opposite
direction from the synthetic run, where punctuation reduced false cuts.

**Real speech reinforces the "don't ship A" verdict, but not for the reason
expected going in.** The three deliberately-paused clips (mid-sentence pause,
"um"+restart) get badly split by *both* modes almost every time — fixed
because a natural 1-2s thinking pause is longer than 650ms by design, and
punctuation because Moonshine still confabulates a finished-sounding fragment
mid-pause. Neither approach handles a genuine hesitation. The clean finding is
`long_16k` — an ordinary, non-deliberately-paused question
("Can you explain how a voice assistant pipeline works briefly?") that fixed
mode answers correctly every time (single turn, correct transcript) and
punctuation mode splits into two *every single rep*: Moonshine reads a natural
sub-650ms breath as "Do you?" (a question mark the speaker never said) and
finalizes on it, then answers the rest of the sentence as a second turn. This
is the exact synthetic failure mode — invented terminal punctuation on a
truncated utterance — reproduced on real recorded speech, on a clip that
wasn't even built to test it.

`record_pause_utterances.py`'s auto-labeler (RMS-threshold, `> median*0.15`)
returned `speech_end_ms` within 4ms of the *full 10s recording window* for
all 5 clips it wrote — worthless, and not obviously so from the numbers
alone, since a clean recording legitimately can have voiced content deep into
a long window. The real problem: on this mic/room, quiet in-sentence syllables
and post-speech room tone sit only ~2-5x apart in RMS, too close for a static
threshold to separate. Re-labeled with Silero VAD (the same model and
threshold the pipeline itself runs on 32ms chunks) instead, which gives
per-clip values that track content sensibly and correctly locates the
deliberate gaps in `pause1`/`pause2` as a stretch of nonspeech. If extending
this clip set, prefer the VAD relabel approach over the RMS heuristic, or fix
`last_voiced_end_ms` in `record_pause_utterances.py` to do the same.

`results/2026-08-19-endpointing-eval-3way.log` — same real clips, all three
`endpoint_mode`s in one run (n=24 each): `fixed` 15 false cuts / median
806ms, `punctuation` 18 / 484ms, `turn_detector` 18 / 638ms.

**`turn_detector` (smart-turn-v3.2-cpu) does not beat `punctuation`, and
matches `fixed` and `punctuation` on which clips fail, not just the count.**
Per-clip breakdown (identical across all 3 reps): on the three deliberately
ambiguous clips (`pause1`, `pause2`, `restart1`) all three modes split the
utterance the exact same way, every rep — smart-turn doesn't distinguish a
genuine mid-thought pause from a finished thought any better than a punctuation
regex or a flat timeout does, on this data. More striking: on `long_16k` — an
*ordinary* question, not one of the designed-ambiguous clips — `punctuation`
and `turn_detector` **independently** split it the identical way (2 turns) on
every rep, while `fixed` gets it right. Two completely different mechanisms
(ASR-transcript regex vs. a purpose-trained audio classifier) converge on the
same false cut at the same natural breath — evidence that breath is genuinely
ambiguous, not an artifact of either method. `turn_detector`'s latency win is
also structurally smaller than `punctuation`'s (median -168ms vs -322ms
against fixed) because it never gets `punctuation`'s free STT-during-the-wait
trick — the audio classifier produces no transcript, so STT always pays after
the cut, same timing shape as `fixed`.

One data point in this log is a measurement artifact, not a `turn_detector`
finding: `short_16k` shows 0 turns in `turn_detector` mode on all 3 reps — STT
came back empty, so the LLM was never reached. Reproduced in isolation with
1.8GB free and it works correctly (`STT 125ms` → `"What time is it?"`), so
this is the box, not the code: three resident models (VAD+STT+turn_detector,
on top of the denoiser) pushed available memory under the 700MB floor by the
time the sequence reached this clip in each rep (528-659MB logged at that
point) — hard evidence that adding a third resident model measurably worsens
this 5.86GB box's headroom during a long combined run, beyond just adding
~8MB of weights.

**Conclusion: don't ship `turn_detector` either.** The bottleneck was never
which gate mechanism decides — a naive punctuation check and a purpose-built
Whisper-Tiny-based classifier fail on the *same* ambiguous real speech, at
rates neither beats a dumb fixed timeout on. `models/turn_detector.py` and
`endpoint_mode="turn_detector"` are built, tested (`tests/test_turn_detector.py`,
`tests/test_endpointing.py`), and wired through the eval harness — the code
is real and correct — the model itself just isn't better at this specific
problem than what was already ruled out. Default stays `fixed`, same as A.

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
- **The 2026-08-19 real-speech endpointing run also fought memory pressure.**
  Brave/WSL/Docker were closed first (349MB -> 1.2GB available), but the
  eval's own model loading ate back down to 365MB by the time it started and
  as low as 73-130MB mid-run. Treat the response-ready *medians* as directionally
  right (they replicate the synthetic run's shape) but the p90/max and any
  single clip's absolute ms as noisy. The false-cut counts and which clips
  get split into which number of turns are structural (same every rep,
  independent of memory) and are the trustworthy part of that result.
- **The 2026-08-19 3-way run (adding `turn_detector`) fought memory pressure
  harder than the 2-mode run**, unsurprisingly — a third resident model on
  top of VAD+STT+denoiser. Available memory was logged as low as 528-659MB at
  points and produced one outright STT failure (see the `short_16k` note
  above). Treat medians as directionally right, individual clip timings and
  the exact false-cut count as noisy; the *which clips split which way*
  pattern (identical across all 3 reps) is the trustworthy part.
- **Memory pressure dominates everything if you let it.** This pipeline
  commits ~5GB. On the 5.86GB box these were measured on, running with
  WSL/Docker/browsers resident produced STT maxima of 9328ms and intermittent
  numpy allocation failures — roughly 10x worse than the same code on an idle
  machine. `trace_session.py` prints working set and private commit after
  every turn for exactly this reason. Check those before trusting a slow
  result.
