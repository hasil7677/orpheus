"""Turn one or more trace logs into per-stage latency statistics.

Usage (from repo root):
    .venv\\Scripts\\python.exe benchmarks\\analyze.py run1.log [run2.log ...]

Pass --keep-cold to include each run's first turn; by default it is dropped,
since the first utterance pays one-off CUDA kernel compilation and page-cache
warming and is not representative of steady state.
"""

import re
import statistics
import sys
from pathlib import Path

TURN = re.compile(r"--- turn (\d+) \((\S+?), ([\d.]+)s")
PATS = {
    "stt": re.compile(r"\[STT (\d+)ms\]"),
    "ttft": re.compile(r"\[LLM TTFT (\d+)ms\]"),
    "ttfa": re.compile(r"\[TTS TTFA (\d+)ms\]"),
}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
drop_cold = "--keep-cold" not in sys.argv
if not args:
    print(__doc__)
    raise SystemExit(1)


def parse(path: Path):
    # the ONNX Runtime C++ logger emits NUL bytes into stdout; strip them
    log = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    turns, cur = [], None
    for line in log.splitlines():
        m = TURN.search(line)
        if m:
            cur = {"clip": m.group(2), "dur": float(m.group(3))}
            turns.append(cur)
            continue
        if cur is None:
            continue
        for key, rx in PATS.items():
            m = rx.search(line)
            if m:
                cur[key] = int(m.group(1))
    complete = [t for t in turns if PATS.keys() <= t.keys()]
    return turns, complete


pooled = []
for arg in args:
    path = Path(arg)
    turns, complete = parse(path)
    incomplete = len(turns) - len(complete)
    note = f", {incomplete} incomplete/errored" if incomplete else ""
    if drop_cold and complete:
        cold = complete[0]
        complete = complete[1:]
        note += (f" (dropped cold start: STT {cold['stt']} / "
                 f"TTFT {cold['ttft']} / TTFA {cold['ttfa']})")
    print(f"{path.name}: {len(complete)} turns{note}")
    pooled.extend(complete)

if not pooled:
    print("\nNo complete turns found.")
    raise SystemExit(1)

print(f"\npooled n = {len(pooled)}\n")


def stats(vals):
    s = sorted(vals)
    return statistics.median(s), min(s), max(s), s[min(int(len(s) * 0.9), len(s) - 1)]


print(f"{'stage':<24}{'median':>9}{'min':>8}{'max':>8}{'p90':>8}")
print("-" * 57)
rows = [
    ("STT (denoise+ASR)", [t["stt"] for t in pooled]),
    ("LLM TTFT", [t["ttft"] for t in pooled]),
    ("TTS TTFA (from LLM)", [t["ttfa"] for t in pooled]),
    ("TTS-only (TTFA-TTFT)", [t["ttfa"] - t["ttft"] for t in pooled]),
    ("end-to-end (STT+TTFA)", [t["stt"] + t["ttfa"] for t in pooled]),
]
for label, vals in rows:
    med, lo, hi, p90 = stats(vals)
    print(f"{label:<24}{med:>7.0f}ms{lo:>7.0f}{hi:>7.0f}{p90:>7.0f}")

# [TTS TTFA] is timed from LLM-stream start, so it already contains [LLM TTFT];
# the incremental TTS cost is TTFA - TTFT. The three printed stages do not sum.
e2e = [t["stt"] + t["ttfa"] for t in pooled]
med_e2e, lo_e2e, hi_e2e, p90_e2e = stats(e2e)
VAD_MS = 650  # config.py silence_timeout_ms, paid before any stage begins
print(f"\nnote: TTFA is timed from LLM-stream start, so it contains TTFT.")
print(f"+ VAD endpointing (silence_timeout_ms): {VAD_MS}ms, fixed")
print(f"=> mouth-to-ear: median {med_e2e + VAD_MS:.0f}ms  "
      f"p90 {p90_e2e + VAD_MS:.0f}  range {lo_e2e + VAD_MS:.0f}-{hi_e2e + VAD_MS:.0f}")

print("\nby utterance (medians):")
print(f"{'clip':<16}{'dur':>7}{'n':>4}{'STT':>9}{'TTFT':>9}{'TTFA':>9}{'e2e':>9}")
print("-" * 63)
for clip in sorted({t["clip"] for t in pooled}, key=lambda c: min(
        t["dur"] for t in pooled if t["clip"] == c)):
    sel = [t for t in pooled if t["clip"] == clip]
    print(f"{clip[:15]:<16}{sel[0]['dur']:>6.2f}s{len(sel):>4}"
          f"{statistics.median([t['stt'] for t in sel]):>7.0f}ms"
          f"{statistics.median([t['ttft'] for t in sel]):>7.0f}ms"
          f"{statistics.median([t['ttfa'] for t in sel]):>7.0f}ms"
          f"{statistics.median([t['stt'] + t['ttfa'] for t in sel]):>7.0f}ms")
