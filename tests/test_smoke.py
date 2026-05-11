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
    bar_period = 60.0 / bpm * 4
    pattern_period = bar_period * 2
    phase = (t % pattern_period) / pattern_period
    sig = np.zeros(n, dtype=np.float32)
    for i, freq in enumerate([220.0, 330.0, 440.0]):
        mask = (phase >= i / 3) & (phase < (i + 1) / 3)
        sig[mask] += 0.3 * np.sin(2 * np.pi * freq * t[mask])
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
    print(
        f"tempo={a.tempo:.1f}  n_beats={a.n_beats}  samplerate={a.samplerate}  "
        f"branchable_beats={len(a.branches)}  last_branch_point={a.last_branch_point}"
    )
    assert a.n_beats >= 40, "should have detected plenty of beats in an 80s clip"
    assert len(a.branches) > 0, "synthetic pattern should yield branches"
    assert 0 < a.last_branch_point < a.n_beats, "expected a usable last branch point"
    assert a.audio.ndim == 2 and a.audio.shape[1] == 2

    # Every kept jump must respect the bar-position constraint.
    bar_beats = a.bar_beats
    phase = a.phase
    for src, items in a.branches.items():
        src_pos = (src - phase) % bar_beats
        for tgt, _d in items:
            tgt_pos = (tgt - phase) % bar_beats
            assert src_pos == tgt_pos, (
                f"branch {src}->{tgt} crosses bar position {src_pos} != {tgt_pos}"
            )

    # Force a branch on every roll by saturating the rising probability.
    p = JukeboxPlayer(
        a,
        jump_chance_min=1.0,
        jump_chance_max=1.0,
        jump_chance_step=0.0,
        rng_seed=0,
    )
    block = 1024
    buf = np.zeros((block, 2), dtype=np.float32)
    samples_played = 0
    events: list[tuple[str, int, int]] = []

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
    print(
        f"after sim: jumps={snap['jump_count']} beats_played={snap['total_beats_played']} "
        f"coverage={snap['coverage']*100:.1f}%"
    )
    kinds = [e[0] for e in events]
    print(
        f"events: jump={kinds.count('jump')} forced={kinds.count('forced')} "
        f"wrap={kinds.count('wrap')} play={kinds.count('play')}"
    )
    assert snap["jump_count"] > 0, "expected jumps with saturated branch probability"

    # Every jump must land on a beat at the same bar position as where we came from
    # (since neighbors are filtered by index_in_parent during analysis).
    for kind, src, dst in events:
        if kind in ("jump", "forced"):
            assert (src + 1 - phase) % bar_beats == (dst - phase) % bar_beats, (
                f"{kind} {src}->{dst} broke bar alignment"
            )

    # Confirm we never play past last_branch_point (forced jump kicks in there).
    max_beat = max((e[2] for e in events), default=0)
    assert max_beat <= a.last_branch_point + 1, (
        f"playback reached beat {max_beat}, past last_branch_point {a.last_branch_point}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
