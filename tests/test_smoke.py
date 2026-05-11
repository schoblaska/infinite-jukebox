"""Smoke test: synthesize an audio file with clear beats, analyze, simulate playback."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infinite_jukebox.analyze import analyze  # noqa: E402
from infinite_jukebox.player import JukeboxPlayer  # noqa: E402


def synth(path: Path, *, sr: int = 22050, seconds: float = 80.0, bpm: float = 120.0) -> None:
    n = int(sr * seconds)
    t = np.arange(n) / sr
    # Repeating two-bar pattern so analysis can find real similar beats.
    bar_period = 60.0 / bpm * 4  # 4 beats per bar
    pattern_period = bar_period * 2
    phase = (t % pattern_period) / pattern_period
    # Three different timbres rotate through the pattern.
    sig = np.zeros(n, dtype=np.float32)
    for i, freq in enumerate([220.0, 330.0, 440.0]):
        mask = (phase >= i / 3) & (phase < (i + 1) / 3)
        sig[mask] += 0.3 * np.sin(2 * np.pi * freq * t[mask])
    # Add a click on every beat.
    beat_period = 60.0 / bpm
    click_times = np.arange(0, seconds, beat_period)
    for ct in click_times:
        idx = int(ct * sr)
        end = min(idx + int(0.01 * sr), n)
        env = np.linspace(1.0, 0.0, end - idx, dtype=np.float32)
        sig[idx:end] += 0.5 * env * np.sin(2 * np.pi * 1000.0 * t[idx:end])
    stereo = np.stack([sig, sig], axis=1)
    sf.write(str(path), stereo, sr)


def main() -> int:
    tmp = Path("/tmp/infinite_jukebox_smoke")
    tmp.mkdir(parents=True, exist_ok=True)
    audio = tmp / "synth.wav"
    synth(audio)

    a = analyze(audio, tmp / "analysis")
    print(f"tempo={a.tempo:.1f}  n_beats={a.n_beats}  samplerate={a.samplerate}")
    print(f"branchable 8-bar={len(a.branches_major)}  4-bar={len(a.branches_minor)}")
    assert a.n_beats >= 40, "should have detected plenty of beats in an 80s clip"
    assert len(a.branches_minor) + len(a.branches_major) > 0, "synthetic pattern should yield branches"
    assert a.audio.ndim == 2 and a.audio.shape[1] == 2

    # Force every roll to jump; drive the audio callback manually.
    p = JukeboxPlayer(a, major_prob=1.0, minor_prob=1.0, rng_seed=0)
    block = 1024
    buf = np.zeros((block, 2), dtype=np.float32)
    samples_played = 0
    events: list[tuple[str, int, int]] = []

    # Wrap _advance_beat to record event kinds.
    orig_advance = p._advance_beat
    def spy() -> None:
        orig_advance()
        if p._last_event is not None:
            events.append(p._last_event)
    p._advance_beat = spy  # type: ignore[assignment]

    while samples_played < a.samplerate * 120:
        p._callback(buf, block, None, None)
        samples_played += block
    snap = p.snapshot()
    print(f"after sim: jumps={snap['jump_count']} beats_played={snap['total_beats_played']} coverage={snap['coverage']*100:.1f}%")
    kinds = [e[0] for e in events]
    n_major = sum(1 for k in kinds if k == "MAJOR")
    n_minor = sum(1 for k in kinds if k == "minor")
    n_play = sum(1 for k in kinds if k == "play")
    print(f"events: MAJOR={n_major}  minor={n_minor}  play={n_play}")
    assert snap["jump_count"] > 0, "expected jumps with both probs at 1.0"

    # Jumps must land on slot starts.
    for kind, _src, dst in events:
        if kind in ("MAJOR", "minor", "wrap"):
            slot = a.slot_kind(dst)
            assert slot in ("major", "minor"), f"{kind} jumped to non-slot beat {dst} (slot={slot})"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
