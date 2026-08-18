"""Splice a long mid-sentence pause into existing clips, with ground truth.

    .venv\\Scripts\\python.exe benchmarks\\make_pause_clips.py [src_dir] [pause_ms]

Reads WAVs from src_dir (default benchmarks/clips), finds the widest natural
inter-word gap in the middle of each, inserts `pause_ms` of silence there, and
writes benchmarks/clips_pauses/*.wav plus a labels.json giving, per clip, the
true end of speech and the span of the inserted pause.

READ THIS BEFORE BELIEVING ANY NUMBER MEASURED ON THESE CLIPS
-------------------------------------------------------------
This is a *mechanism* test, not evidence. It answers "does the endpointer sit
through a 900ms gap and cut at the real end", which is a wiring question. It
does not answer "does this work on human speech", because:

  * the source clips are Kokoro-synthesized, so the pauses are digital silence
    and the speech either side is clean and unhesitant;
  * a spliced gap is acoustically nothing like a human hesitating — no breath,
    no filler, no drop in pitch, no trailing-off;
  * Moonshine transcribing synthesized speech is a best case, and the whole
    punctuation gate rides on how Moonshine punctuates.

The number that matters comes from recorded human utterances with hand-written
labels. Point eval_endpointing.py at those instead; the labels.json written
here is the format to copy.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "clips_pauses"
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "benchmarks" / "clips"
PAUSE_MS = int(sys.argv[2]) if len(sys.argv) > 2 else 900
SR = 16000
FRAME_MS = 10


def load_16k_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        audio = resample_poly(audio, SR // g, sr // g)
    return audio.astype(np.float32)


def widest_inner_gap(audio: np.ndarray) -> tuple[int, int] | None:
    """Sample range of the widest quiet stretch in the middle of the clip.

    Restricted to the middle half so the splice lands between words rather
    than in the leading/trailing air, where it would just look like a longer
    silence tail and test nothing.
    """
    n = int(SR * FRAME_MS / 1000)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    voiced = rms > max(np.median(rms[rms > 0]) * 0.15, 1e-4) if (rms > 0).any() else rms > 1
    lo, hi = int(len(voiced) * 0.25), int(len(voiced) * 0.75)

    best = best_start = best_len = None
    run_start = None
    for i in range(lo, hi):
        if not voiced[i]:
            run_start = i if run_start is None else run_start
        elif run_start is not None:
            if best_len is None or i - run_start > best_len:
                best_start, best_len = run_start, i - run_start
            run_start = None
    if run_start is not None and (best_len is None or hi - run_start > best_len):
        best_start, best_len = run_start, hi - run_start
    if best_start is None or best_len < 2:  # need at least a 20ms gap to widen
        return None
    best = (best_start + best_len // 2) * n
    return best, best_len * n


def main():
    clips = [c for c in sorted(SRC.glob("*.wav")) if not c.stem.endswith("_24k")]
    if not clips:
        print(f"No WAVs in {SRC}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    pause = np.zeros(int(SR * PAUSE_MS / 1000), dtype=np.float32)
    labels = {}

    for path in clips:
        audio = load_16k_mono(path)
        found = widest_inner_gap(audio)
        if found is None:
            print(f"{path.stem:20s} no usable inter-word gap — skipped")
            continue
        at, natural = found
        spliced = np.concatenate([audio[:at], pause, audio[at:]])
        stem = f"{path.stem.replace('_16k', '')}_pause{PAUSE_MS}_16k"
        sf.write(OUT / f"{stem}.wav", spliced, SR)

        labels[stem] = {
            "speech_end_ms": round(len(spliced) / SR * 1000),
            "pauses": [[round(at / SR * 1000),
                        round((at + len(pause)) / SR * 1000)]],
            "source": "synthesized + spliced silence (mechanism test only)",
        }
        print(f"{stem:28s} {len(spliced)/SR:5.2f}s  pause at "
              f"{at/SR:5.2f}s (natural gap {natural/SR*1000:3.0f}ms -> {PAUSE_MS}ms)")

    (OUT / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"\nWrote {len(labels)} clips + labels.json to {OUT.relative_to(REPO)}")
    print("\nEvaluate both endpoint modes on them:")
    print(r"  .venv\Scripts\python.exe benchmarks\eval_endpointing.py "
          r"benchmarks\clips_pauses")


if __name__ == "__main__":
    main()
