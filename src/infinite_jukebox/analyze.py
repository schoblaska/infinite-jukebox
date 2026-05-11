from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


SAMPLE_RATE = 22050  # for analysis; playback uses original audio at its own rate
ANALYSIS_VERSION = 4  # bump to invalidate caches when algorithm changes


@dataclass
class Analysis:
    audio: np.ndarray              # (n_samples, n_channels) float32, playback sample rate
    samplerate: int                # playback sample rate
    beat_samples: np.ndarray       # (n_beats + 1,) sample indices marking beat boundaries
    tempo: float
    bar_beats: int                 # beats per bar (4 for 4/4)
    phase: int                     # bar-grid phase offset, in beats
    branches_major: dict[int, list[tuple[int, float]]]  # source beat -> [(target_beat, weight)]
    branches_minor: dict[int, list[tuple[int, float]]]

    @property
    def n_beats(self) -> int:
        return len(self.beat_samples) - 1

    @property
    def major_step(self) -> int:
        return self.bar_beats * 8

    @property
    def minor_step(self) -> int:
        return self.bar_beats * 4

    def slot_kind(self, beat_idx: int) -> str:
        """Return 'major', 'minor', or 'none' for the given beat index."""
        rel = beat_idx - self.phase
        if rel < 0:
            return "none"
        if rel % self.major_step == 0:
            return "major"
        if rel % self.minor_step == 0:
            return "minor"
        return "none"


def _file_fingerprint(path: Path) -> str:
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode())
    h.update(str(path.stat().st_size).encode())
    with path.open("rb") as f:
        h.update(f.read(65536))
        if path.stat().st_size > 131072:
            f.seek(-65536, 2)
            h.update(f.read(65536))
    return h.hexdigest()[:16]


def _per_beat_features(y: np.ndarray, sr: int, beat_frames: np.ndarray, hop: int) -> np.ndarray:
    """Per-beat feature vector: mean chroma_cqt (12) + mean MFCC (13) + RMS (1) = 26-d."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
    rms = librosa.feature.rms(y=y, hop_length=hop)

    feats = []
    n_frames = chroma.shape[1]
    for i in range(len(beat_frames) - 1):
        start = int(beat_frames[i])
        end = max(int(beat_frames[i + 1]), start + 1)
        end = min(end, n_frames)
        if end <= start:
            end = start + 1
        c = chroma[:, start:end].mean(axis=1)
        m = mfcc[:, start:end].mean(axis=1)
        r = rms[:, start:end].mean(axis=1)
        feats.append(np.concatenate([c, m, r]))
    return np.stack(feats, axis=0)  # (n_beats, 26)


def _segment_features(beat_features: np.ndarray, slots: list[int], length: int) -> np.ndarray:
    """For each slot start beat, mean of beat features over the next `length` beats."""
    n_beats = beat_features.shape[0]
    out = np.zeros((len(slots), beat_features.shape[1]), dtype=np.float64)
    for i, s in enumerate(slots):
        end = min(s + length, n_beats)
        if end <= s:
            out[i] = beat_features[min(s, n_beats - 1)]
        else:
            out[i] = beat_features[s:end].mean(axis=0)
    return out


def _branch_table(
    segment_feats: np.ndarray,
    slot_beats: list[int],
    *,
    top_k: int,
    max_distance_pct: float,
    min_slot_gap: int,
) -> dict[int, list[tuple[int, float]]]:
    """For each slot, find the K most similar other slots; weight by closeness."""
    n = segment_feats.shape[0]
    if n < 2:
        return {}
    diff = segment_feats[:, None, :] - segment_feats[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))

    idx = np.arange(n)
    too_close = np.abs(idx[:, None] - idx[None, :]) < min_slot_gap
    dist_masked = np.where(too_close, np.inf, dist)

    finite_mask = ~too_close & np.isfinite(dist)
    if finite_mask.any():
        cutoff = float(np.percentile(dist[finite_mask], max_distance_pct))
    else:
        cutoff = float("inf")

    table: dict[int, list[tuple[int, float]]] = {}
    for i in range(n):
        row = dist_masked[i]
        order = np.argsort(row)
        kept: list[tuple[int, float]] = []
        for j in order[:top_k]:
            d = float(row[j])
            if not np.isfinite(d) or d > cutoff:
                break
            kept.append((slot_beats[j], d))
        if kept:
            ds = np.array([d for _, d in kept])
            scale = ds.mean() + 1e-6
            weights = np.exp(-(ds - ds.min()) / scale)
            kept = [(t, float(w)) for (t, _), w in zip(kept, weights)]
            table[slot_beats[i]] = kept
    return table


def analyze(
    audio_path: Path,
    cache_dir: Path,
    *,
    bar_beats: int = 4,
    phase: int = 0,
    top_k: int = 8,
    max_distance_pct: float = 35.0,
) -> Analysis:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = _file_fingerprint(audio_path)
    cache_file = cache_dir / (
        f"{audio_path.stem}.{fp}.v{ANALYSIS_VERSION}"
        f".bb{bar_beats}.ph{phase}.k{top_k}.p{int(max_distance_pct)}.npz"
    )

    y_play, sr_play = librosa.load(str(audio_path), sr=None, mono=False)
    if y_play.ndim == 1:
        y_play = np.stack([y_play, y_play], axis=0)
    audio = y_play.T.astype(np.float32)

    if cache_file.exists():
        data = np.load(cache_file)

        def _unpack(arr: np.ndarray) -> dict[int, list[tuple[int, float]]]:
            d: dict[int, list[tuple[int, float]]] = {}
            for row in arr:
                src = int(row[0])
                d.setdefault(src, []).append((int(row[1]), float(row[2])))
            return d

        return Analysis(
            audio=audio,
            samplerate=sr_play,
            beat_samples=data["beat_samples"],
            tempo=float(data["tempo"]),
            bar_beats=int(data["bar_beats"]),
            phase=int(data["phase"]),
            branches_major=_unpack(data["branches_major"]),
            branches_minor=_unpack(data["branches_minor"]),
        )

    y_mono = librosa.to_mono(y_play)
    if sr_play != SAMPLE_RATE:
        y_mono = librosa.resample(y_mono, orig_sr=sr_play, target_sr=SAMPLE_RATE)
    sr = SAMPLE_RATE
    hop = 512

    tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr, hop_length=hop, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_frames = np.asarray(beat_frames, dtype=np.int64)
    n_frames_total = 1 + len(y_mono) // hop
    if beat_frames.size == 0 or beat_frames[0] > 0:
        beat_frames = np.concatenate([[0], beat_frames])
    if beat_frames[-1] < n_frames_total - 1:
        beat_frames = np.concatenate([beat_frames, [n_frames_total - 1]])

    features = _per_beat_features(y_mono, sr, beat_frames, hop)
    mu = features.mean(axis=0, keepdims=True)
    sd = features.std(axis=0, keepdims=True) + 1e-8
    fz = (features - mu) / sd

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    beat_samples = np.round(beat_times * sr_play).astype(np.int64)
    beat_samples = np.clip(beat_samples, 0, audio.shape[0])

    n_beats = features.shape[0]
    major_step = bar_beats * 8
    minor_step = bar_beats * 4

    major_slots = [b for b in range(phase, n_beats, major_step) if b + minor_step <= n_beats]
    minor_slots = [
        b for b in range(phase, n_beats, minor_step)
        if b + minor_step <= n_beats and (b - phase) % major_step != 0
    ]

    major_feats = _segment_features(fz, major_slots, major_step)
    minor_feats = _segment_features(fz, minor_slots, minor_step)

    # gap measured in beats — at least one full 8-bar block between matches
    branches_major = _branch_table(
        major_feats, major_slots, top_k=top_k, max_distance_pct=max_distance_pct,
        min_slot_gap=1,  # any other 8-bar slot is far enough
    )
    branches_minor = _branch_table(
        minor_feats, minor_slots, top_k=top_k, max_distance_pct=max_distance_pct,
        min_slot_gap=1,
    )

    def _pack(d: dict[int, list[tuple[int, float]]]) -> np.ndarray:
        rows: list[tuple[float, float, float]] = []
        for src, items in d.items():
            for tgt, w in items:
                rows.append((float(src), float(tgt), float(w)))
        if not rows:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array(rows, dtype=np.float64)

    np.savez(
        cache_file,
        beat_samples=beat_samples,
        tempo=np.array(tempo),
        bar_beats=np.array(bar_beats),
        phase=np.array(phase),
        branches_major=_pack(branches_major),
        branches_minor=_pack(branches_minor),
    )

    return Analysis(
        audio=audio,
        samplerate=sr_play,
        beat_samples=beat_samples,
        tempo=tempo,
        bar_beats=bar_beats,
        phase=phase,
        branches_major=branches_major,
        branches_minor=branches_minor,
    )
