"""Echo Nest-style jump finder.

Ports the algorithm from `rigdern/InfiniteJukeboxAlgorithm` (derived from Paul
Lamere's Infinite Jukebox). Each beat is described as an ordered sequence of
sub-beat onset segments; jump candidates are scored by walking those segment
sequences pairwise and summing weighted feature distances, plus a hard penalty
when the candidate sits at a different position within its bar.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE = 22050
ANALYSIS_VERSION = 5

MAX_BRANCHES = 4
MAX_BRANCH_THRESHOLD = 80.0

TIMBRE_WEIGHT = 1.0
PITCH_WEIGHT = 10.0
LOUD_START_WEIGHT = 1.0
LOUD_MAX_WEIGHT = 1.0
DURATION_WEIGHT = 100.0
CONFIDENCE_WEIGHT = 1.0

INDEX_IN_PARENT_PENALTY = 100.0
SELF_SEGUE_DISTANCE = 100.0
MISSING_SEGMENT_DISTANCE = 100.0

REACH_THRESHOLD_PCT = 50.0
LONG_BACKWARD_PCT = 50.0
BBB_TIGHT_THRESHOLD = 65.0
BBB_LOOSE_THRESHOLD = 55.0


@dataclass
class Analysis:
    audio: np.ndarray              # (n_samples, n_channels) float32
    samplerate: int
    beat_samples: np.ndarray       # (n_beats + 1,) sample indices marking boundaries
    tempo: float
    bar_beats: int
    phase: int
    branches: dict[int, list[tuple[int, float]]]  # source beat -> [(target, distance)]
    last_branch_point: int

    @property
    def n_beats(self) -> int:
        return len(self.beat_samples) - 1


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return (audio[channels, samples] float32, sample_rate).

    Tries libsndfile first; falls back to ffmpeg for formats it can't decode
    (notably AAC inside .m4a). yt-dlp already requires ffmpeg.
    """
    try:
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        return data.T, int(sr)
    except (sf.LibsndfileError, RuntimeError):
        pass

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "json", str(path),
        ],
        capture_output=True, check=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    sr = int(stream["sample_rate"])
    channels = int(stream["channels"])
    decode = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le", "-",
        ],
        capture_output=True, check=True,
    )
    audio = np.frombuffer(decode.stdout, dtype=np.float32).reshape(-1, channels).T
    return np.ascontiguousarray(audio), sr


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


# --------------------------------------------------------------------- segments


def _segment_boundaries(
    y: np.ndarray, sr: int, hop: int, n_frames_total: int, *,
    min_segment_frames: int,
) -> np.ndarray:
    """Onset-driven segment frame boundaries spanning the whole song."""
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="frames")
    boundaries = np.concatenate([[0], np.asarray(onsets, dtype=np.int64), [n_frames_total]])
    boundaries = np.unique(boundaries)
    # Drop segments that are too short to extract stable features from.
    keep = [boundaries[0]]
    for b in boundaries[1:]:
        if b - keep[-1] >= min_segment_frames:
            keep.append(b)
    if keep[-1] != boundaries[-1]:
        keep[-1] = boundaries[-1]
    return np.asarray(keep, dtype=np.int64)


def _segment_features(y: np.ndarray, sr: int, hop: int, boundaries: np.ndarray) -> dict:
    """Per-segment features: pitches (12), timbre (12), loud_start, loud_max, duration, confidence."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=12)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset_str = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    db = librosa.amplitude_to_db(np.maximum(rms, 1e-8))
    n_frames = chroma.shape[1]
    onset_norm = onset_str / (onset_str.max() + 1e-8)

    n_segs = len(boundaries) - 1
    pitches = np.zeros((n_segs, 12), dtype=np.float64)
    timbre = np.zeros((n_segs, 12), dtype=np.float64)
    loud_start = np.zeros(n_segs, dtype=np.float64)
    loud_max = np.zeros(n_segs, dtype=np.float64)
    duration = np.zeros(n_segs, dtype=np.float64)
    confidence = np.zeros(n_segs, dtype=np.float64)

    for i in range(n_segs):
        a = int(boundaries[i])
        b = max(int(boundaries[i + 1]), a + 1)
        b = min(b, n_frames)
        if b <= a:
            b = min(a + 1, n_frames)
            a = b - 1
        pitches[i] = chroma[:, a:b].mean(axis=1)
        timbre[i] = mfcc[:, a:b].mean(axis=1)
        loud_start[i] = db[min(a, len(db) - 1)]
        loud_max[i] = db[a:b].max()
        duration[i] = (boundaries[i + 1] - boundaries[i]) * hop / sr
        confidence[i] = onset_norm[min(a, len(onset_norm) - 1)]

    return {
        "pitches": pitches,
        "timbre": timbre,
        "loud_start": loud_start,
        "loud_max": loud_max,
        "duration": duration,
        "confidence": confidence,
        "starts": boundaries[:-1].astype(np.int64),
        "ends": boundaries[1:].astype(np.int64),
    }


def _beat_overlapping_segments(beat_frames: np.ndarray, seg_starts: np.ndarray, seg_ends: np.ndarray) -> list[list[int]]:
    """For each beat, segment indices that overlap it (seg_end > b_start AND seg_start < b_end)."""
    n_beats = len(beat_frames) - 1
    out: list[list[int]] = [[] for _ in range(n_beats)]
    j = 0
    n_segs = len(seg_starts)
    for i in range(n_beats):
        b_start = int(beat_frames[i])
        b_end = int(beat_frames[i + 1])
        # rewind j to the first segment that could overlap this beat
        while j > 0 and seg_ends[j - 1] > b_start:
            j -= 1
        k = j
        while k < n_segs:
            s_start = int(seg_starts[k])
            s_end = int(seg_ends[k])
            if s_end <= b_start:
                k += 1
                continue
            if s_start >= b_end:
                break
            out[i].append(k)
            k += 1
        # advance j past segments fully before next beat
        while j < n_segs and seg_ends[j] <= b_end:
            j += 1
    # guarantee every beat has at least one segment (fallback: nearest segment in time)
    for i, segs in enumerate(out):
        if not segs:
            mid = (int(beat_frames[i]) + int(beat_frames[i + 1])) // 2
            nearest = int(np.argmin(np.abs(((seg_starts + seg_ends) // 2) - mid)))
            out[i].append(nearest)
    return out


# --------------------------------------------------------------------- distance


def _build_distance_matrix(
    beat_segs: list[list[int]],
    feats: dict,
    index_in_parent: np.ndarray,
) -> np.ndarray:
    n_beats = len(beat_segs)
    if n_beats == 0:
        return np.zeros((0, 0))

    max_K = max((len(s) for s in beat_segs), default=0)
    seg_idx = np.full((n_beats, max_K), -1, dtype=np.int64)
    seg_count = np.zeros(n_beats, dtype=np.int64)
    for i, segs in enumerate(beat_segs):
        seg_count[i] = len(segs)
        if segs:
            seg_idx[i, : len(segs)] = segs

    pitches = feats["pitches"]
    timbre = feats["timbre"]
    loud_start = feats["loud_start"]
    loud_max = feats["loud_max"]
    duration = feats["duration"]
    confidence = feats["confidence"]

    D = np.full((n_beats, n_beats), np.inf, dtype=np.float64)
    for i in range(n_beats):
        K1 = int(seg_count[i])
        if K1 == 0:
            continue
        total = np.zeros(n_beats, dtype=np.float64)
        for j in range(K1):
            s1 = int(seg_idx[i, j])
            s2_col = seg_idx[:, j] if j < max_K else np.full(n_beats, -1, dtype=np.int64)
            dists = np.full(n_beats, MISSING_SEGMENT_DISTANCE)
            valid = (s2_col != -1) & (s2_col != s1)
            if valid.any():
                s2 = s2_col[valid]
                d_timbre = np.linalg.norm(timbre[s1] - timbre[s2], axis=1)
                d_pitch = np.linalg.norm(pitches[s1] - pitches[s2], axis=1)
                d_ls = np.abs(loud_start[s1] - loud_start[s2])
                d_lm = np.abs(loud_max[s1] - loud_max[s2])
                d_du = np.abs(duration[s1] - duration[s2])
                d_cf = np.abs(confidence[s1] - confidence[s2])
                seg_d = (
                    d_timbre * TIMBRE_WEIGHT
                    + d_pitch * PITCH_WEIGHT
                    + d_ls * LOUD_START_WEIGHT
                    + d_lm * LOUD_MAX_WEIGHT
                    + d_du * DURATION_WEIGHT
                    + d_cf * CONFIDENCE_WEIGHT
                )
                dists[valid] = seg_d
            # source itself contributes 0 distance to itself; suppress
            total += dists

        row = total / K1
        row += np.where(index_in_parent == index_in_parent[i], 0.0, INDEX_IN_PARENT_PENALTY)
        row[i] = np.inf
        D[i] = row
    return D


# --------------------------------------------------------- neighbor selection


def _precalc_neighbors(D: np.ndarray) -> list[list[tuple[int, float]]]:
    n = D.shape[0]
    out: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        row = D[i]
        order = np.argsort(row)
        for j in order[:MAX_BRANCHES]:
            d = float(row[j])
            if d >= MAX_BRANCH_THRESHOLD:
                break
            out[i].append((int(j), d))
    return out


def _collect_at(all_neighbors: list[list[tuple[int, float]]], threshold: float) -> tuple[list[list[tuple[int, float]]], int]:
    out = [[(t, d) for t, d in row if d <= threshold] for row in all_neighbors]
    count = sum(1 for row in out if row)
    return out, count


def _longest_backward_pct(neighbors: list[list[tuple[int, float]]], n_beats: int) -> float:
    longest = 0
    for i, row in enumerate(neighbors):
        for t, _ in row:
            delta = i - t
            if delta > longest:
                longest = delta
    return longest * 100 / max(n_beats, 1)


def _insert_best_backward_branch(
    all_neighbors: list[list[tuple[int, float]]],
    neighbors: list[list[tuple[int, float]]],
    selection_threshold: float,
    max_threshold: float,
    n_beats: int,
) -> None:
    candidates: list[tuple[float, int, int, float]] = []
    for i, row in enumerate(all_neighbors):
        for t, d in row:
            delta = i - t
            if delta > 0 and d < max_threshold:
                candidates.append((delta * 100 / n_beats, i, t, d))
    if not candidates:
        return
    candidates.sort(reverse=True)
    _pct, i, t, d = candidates[0]
    if d > selection_threshold and not any(tt == t for tt, _ in neighbors[i]):
        neighbors[i].append((t, d))


def _calculate_reachability(neighbors: list[list[tuple[int, float]]], n_beats: int) -> np.ndarray:
    reach = np.array([n_beats - i for i in range(n_beats)], dtype=np.float64)
    for _ in range(1000):
        changed_any = False
        for i in range(n_beats):
            old = reach[i]
            new = old
            if i + 1 < n_beats and reach[i + 1] > new:
                new = reach[i + 1]
            for t, _ in neighbors[i]:
                if reach[t] > new:
                    new = reach[t]
            if new > old:
                reach[i] = new
                changed_any = True
                # backward flood: predecessors must reach at least as far as i does
                for k in range(i):
                    if reach[k] < new:
                        reach[k] = new
        if not changed_any:
            break
    return reach


def _find_last_branch_point(reach: np.ndarray, neighbors: list[list[tuple[int, float]]], n_beats: int) -> int:
    longest = 0
    best = 0.0
    for i in range(n_beats - 1, -1, -1):
        distance_to_end = n_beats - i
        adj = (reach[i] - distance_to_end) * 100 / max(n_beats, 1)
        if adj > best and neighbors[i]:
            best = adj
            longest = i
            if adj >= REACH_THRESHOLD_PCT:
                break
    return longest


def _filter_bad_branches(neighbors: list[list[tuple[int, float]]], last_idx: int) -> None:
    for i in range(min(last_idx, len(neighbors))):
        neighbors[i] = [(t, d) for t, d in neighbors[i] if t < last_idx]


# --------------------------------------------------------------------- splices


def _snap_to_zero_crossings(audio: np.ndarray, beat_samples: np.ndarray, sr: int, *, window_ms: float = 10.0) -> np.ndarray:
    """Nudge each beat boundary to the nearest near-zero sample within ±window_ms."""
    window = int(window_ms * 1e-3 * sr)
    if window <= 0:
        return beat_samples
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    n = len(mono)
    snapped = np.empty_like(beat_samples)
    for i, s in enumerate(beat_samples):
        s = int(s)
        lo = max(0, s - window)
        hi = min(n, s + window)
        if hi <= lo:
            snapped[i] = s
            continue
        snapped[i] = lo + int(np.argmin(np.abs(mono[lo:hi])))
    return snapped


# --------------------------------------------------------------------- top-level


def analyze(
    audio_path: Path,
    cache_dir: Path,
    *,
    bar_beats: int = 4,
    phase: int = 0,
) -> Analysis:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = _file_fingerprint(audio_path)
    cache_file = cache_dir / (
        f"{audio_path.stem}.{fp}.v{ANALYSIS_VERSION}.bb{bar_beats}.ph{phase}.npz"
    )

    y_play, sr_play = _load_audio(audio_path)
    if y_play.ndim == 1:
        y_play = np.stack([y_play, y_play], axis=0)
    elif y_play.shape[0] == 1:
        y_play = np.repeat(y_play, 2, axis=0)
    audio = y_play.T.astype(np.float32, copy=False)

    if cache_file.exists():
        data = np.load(cache_file)
        branches: dict[int, list[tuple[int, float]]] = {}
        for row in data["branches"]:
            src = int(row[0])
            branches.setdefault(src, []).append((int(row[1]), float(row[2])))
        return Analysis(
            audio=audio,
            samplerate=sr_play,
            beat_samples=data["beat_samples"],
            tempo=float(data["tempo"]),
            bar_beats=int(data["bar_beats"]),
            phase=int(data["phase"]),
            branches=branches,
            last_branch_point=int(data["last_branch_point"]),
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
    n_beats = len(beat_frames) - 1

    # Sub-beat segments: cap shortest at ~50ms so features are stable.
    min_seg_frames = max(1, int(0.050 * sr / hop))
    boundaries = _segment_boundaries(y_mono, sr, hop, n_frames_total, min_segment_frames=min_seg_frames)
    feats = _segment_features(y_mono, sr, hop, boundaries)
    beat_segs = _beat_overlapping_segments(beat_frames, feats["starts"], feats["ends"])

    index_in_parent = np.array(
        [((i - phase) % bar_beats + bar_beats) % bar_beats for i in range(n_beats)],
        dtype=np.int64,
    )

    D = _build_distance_matrix(beat_segs, feats, index_in_parent)
    all_neighbors = _precalc_neighbors(D)

    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_beats)]
    target = n_beats / 6
    selection_threshold = MAX_BRANCH_THRESHOLD
    for threshold in range(10, int(MAX_BRANCH_THRESHOLD), 5):
        neighbors, count = _collect_at(all_neighbors, float(threshold))
        if count >= target:
            selection_threshold = float(threshold)
            break

    if _longest_backward_pct(neighbors, n_beats) < LONG_BACKWARD_PCT:
        _insert_best_backward_branch(all_neighbors, neighbors, selection_threshold, BBB_TIGHT_THRESHOLD, n_beats)
    else:
        _insert_best_backward_branch(all_neighbors, neighbors, selection_threshold, BBB_LOOSE_THRESHOLD, n_beats)

    reach = _calculate_reachability(neighbors, n_beats)
    last_branch_point = _find_last_branch_point(reach, neighbors, n_beats)
    _filter_bad_branches(neighbors, last_branch_point)

    # Sort neighbors by ascending distance for stable round-robin selection.
    branches: dict[int, list[tuple[int, float]]] = {}
    for i, row in enumerate(neighbors):
        if row:
            branches[i] = sorted(row, key=lambda td: td[1])

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    beat_samples = np.round(beat_times * sr_play).astype(np.int64)
    beat_samples = np.clip(beat_samples, 0, audio.shape[0])
    beat_samples = _snap_to_zero_crossings(audio, beat_samples, sr_play)

    rows = [(float(s), float(t), float(d)) for s, items in branches.items() for t, d in items]
    branch_arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 3))

    np.savez(
        cache_file,
        beat_samples=beat_samples,
        tempo=np.array(tempo),
        bar_beats=np.array(bar_beats),
        phase=np.array(phase),
        branches=branch_arr,
        last_branch_point=np.array(last_branch_point),
    )

    return Analysis(
        audio=audio,
        samplerate=sr_play,
        beat_samples=beat_samples,
        tempo=tempo,
        bar_beats=bar_beats,
        phase=phase,
        branches=branches,
        last_branch_point=int(last_branch_point),
    )
