from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd

from .analyze import Analysis


# Defaults from rigdern/InfiniteJukeboxAlgorithm InfiniteBeats.js.
JUMP_CHANCE_MIN = 0.18
JUMP_CHANCE_MAX = 0.50
JUMP_CHANCE_STEP = 0.018


class JukeboxPlayer:
    def __init__(
        self,
        analysis: Analysis,
        *,
        jump_chance_min: float = JUMP_CHANCE_MIN,
        jump_chance_max: float = JUMP_CHANCE_MAX,
        jump_chance_step: float = JUMP_CHANCE_STEP,
        rng_seed: int | None = None,
    ) -> None:
        self.a = analysis
        self.jump_chance_min = float(jump_chance_min)
        self.jump_chance_max = float(jump_chance_max)
        self.jump_chance_step = float(jump_chance_step)
        self.jump_chance = self.jump_chance_min
        self.rng = np.random.default_rng(rng_seed)

        self.cursor_sample = 0
        self.beat_idx = 0
        self.play_count = np.zeros(self.a.n_beats, dtype=np.int64)
        self.jump_count = 0
        self.total_beats_played = 0
        self._last_event: tuple[str, int, int] | None = None
        self._next_neighbor_idx: dict[int, int] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _pick_target(self, seed: int) -> int | None:
        neighbors = self.a.branches.get(seed)
        if not neighbors:
            return None
        idx = self._next_neighbor_idx.get(seed, 0)
        target = neighbors[idx % len(neighbors)][0]
        self._next_neighbor_idx[seed] = idx + 1
        return int(target)

    def _advance_beat(self) -> None:
        prev = self.beat_idx
        self.play_count[prev] += 1
        self.total_beats_played += 1

        a = self.a
        seed = prev + 1
        wrap = False
        if seed >= a.n_beats:
            seed = max(0, min(a.last_branch_point, a.n_beats - 1))
            wrap = True

        target = seed
        event_kind = "play"
        seed_neighbors = a.branches.get(seed, [])

        if seed_neighbors:
            force = wrap or seed == a.last_branch_point
            if force:
                t = self._pick_target(seed)
                if t is not None:
                    target = t
                    event_kind = "forced" if not wrap else "wrap"
                    self.jump_chance = self.jump_chance_min
            else:
                self.jump_chance = min(
                    self.jump_chance_max, self.jump_chance + self.jump_chance_step
                )
                if self.rng.random() < self.jump_chance:
                    t = self._pick_target(seed)
                    if t is not None:
                        target = t
                        event_kind = "jump"
                        self.jump_chance = self.jump_chance_min
        elif wrap:
            # Wrapped but the wrap target has no neighbors — just snap there and keep going.
            event_kind = "wrap"

        if event_kind in ("jump", "forced", "wrap"):
            self.jump_count += 1

        with self._lock:
            self._last_event = (event_kind, prev, target)

        self.beat_idx = target
        self.cursor_sample = int(a.beat_samples[target])

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
            "jump_chance": float(self.jump_chance),
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
