from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd

from .analyze import Analysis


class JukeboxPlayer:
    def __init__(
        self,
        analysis: Analysis,
        *,
        major_prob: float = 0.55,
        minor_prob: float = 0.15,
        rng_seed: int | None = None,
    ) -> None:
        self.a = analysis
        self.major_prob = float(major_prob)
        self.minor_prob = float(minor_prob)
        self.rng = np.random.default_rng(rng_seed)

        self.cursor_sample = 0
        self.beat_idx = 0
        self.play_count = np.zeros(self.a.n_beats, dtype=np.int64)
        self.jump_count = 0
        self.total_beats_played = 0
        self._last_jump_from: int | None = None
        self._last_event: tuple[str, int, int] | None = None  # (kind, from, to)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _pick_target(self, table: dict, from_beat: int) -> int | None:
        candidates = table.get(from_beat)
        if not candidates:
            return None
        targets = np.array([t for t, _ in candidates], dtype=np.int64)
        weights = np.array([w for _, w in candidates], dtype=np.float64)
        weights = weights / (1.0 + self.play_count[targets].astype(np.float64))
        if self._last_jump_from is not None:
            weights = np.where(targets == self._last_jump_from, weights * 0.1, weights)
        s = weights.sum()
        if s <= 0:
            return None
        return int(self.rng.choice(targets, p=weights / s))

    def _advance_beat(self) -> None:
        prev = self.beat_idx
        self.play_count[prev] += 1
        self.total_beats_played += 1

        a = self.a
        next_beat = prev + 1
        at_end = next_beat >= a.n_beats

        target: int | None = None
        event_kind = "play"

        if not at_end:
            # Decide based on the upcoming beat: is it the start of a new phrase?
            kind = a.slot_kind(next_beat)
            if kind == "major" and self.rng.random() < self.major_prob:
                t = self._pick_target(a.branches_major, next_beat)
                if t is not None:
                    target = t
                    event_kind = "MAJOR"
            elif kind == "minor" and self.rng.random() < self.minor_prob:
                t = self._pick_target(a.branches_minor, next_beat)
                if t is not None:
                    target = t
                    event_kind = "minor"
            if target is None:
                target = next_beat
        else:
            # End of song: must jump. Find the slot prev belongs to and use its branches.
            target = self._fallback_jump(prev)
            event_kind = "wrap"

        if event_kind in ("MAJOR", "minor"):
            self._last_jump_from = prev
            self.jump_count += 1
        elif event_kind == "wrap":
            self.jump_count += 1

        with self._lock:
            self._last_event = (event_kind, prev, target)

        self.beat_idx = target
        self.cursor_sample = int(a.beat_samples[target])

    def _fallback_jump(self, prev: int) -> int:
        """Pick a slot-start to jump to when playback hits the end."""
        a = self.a
        # Find the most recent slot start at or before prev.
        rel = prev - a.phase
        major_anchor = a.phase + (rel // a.major_step) * a.major_step if rel >= 0 else None
        minor_anchor = a.phase + (rel // a.minor_step) * a.minor_step if rel >= 0 else None
        for anchor, table in (
            (major_anchor, a.branches_major),
            (minor_anchor, a.branches_minor),
        ):
            if anchor is None:
                continue
            t = self._pick_target(table, anchor)
            if t is not None:
                return t
        # No branches available — pick any major slot start, else minor, else beat 0.
        if a.branches_major:
            return int(self.rng.choice(sorted(a.branches_major.keys())))
        if a.branches_minor:
            return int(self.rng.choice(sorted(a.branches_minor.keys())))
        return 0

    # ------------------------------------------------------------------
    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ARG002
        if status:
            pass
        audio = self.a.audio
        beat_samples = self.a.beat_samples
        written = 0
        while written < frames:
            beat_end = (
                int(beat_samples[self.beat_idx + 1])
                if (self.beat_idx + 1) < len(beat_samples)
                else audio.shape[0]
            )
            available_in_beat = beat_end - self.cursor_sample
            if available_in_beat <= 0:
                self._advance_beat()
                continue
            n = min(frames - written, available_in_beat)
            src = audio[self.cursor_sample : self.cursor_sample + n]
            if src.shape[0] < n:
                outdata[written : written + src.shape[0]] = src
                outdata[written + src.shape[0] : written + n] = 0.0
                self.cursor_sample = beat_end
            else:
                outdata[written : written + n] = src
                self.cursor_sample += n
            written += n

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            event = self._last_event
        return {
            "beat_idx": int(self.beat_idx),
            "n_beats": int(self.a.n_beats),
            "jump_count": int(self.jump_count),
            "total_beats_played": int(self.total_beats_played),
            "coverage": float(np.count_nonzero(self.play_count)) / max(self.a.n_beats, 1),
            "last_event": event,
        }

    def run(self, status_callback=None, status_interval: float = 0.25) -> None:
        stream = sd.OutputStream(
            samplerate=self.a.samplerate,
            channels=self.a.audio.shape[1],
            dtype="float32",
            callback=self._callback,
            blocksize=0,
            latency="low",
        )
        with stream:
            try:
                while True:
                    if status_callback is not None:
                        status_callback(self.snapshot())
                    time.sleep(status_interval)
            except KeyboardInterrupt:
                pass
