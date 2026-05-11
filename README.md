# infinite-jukebox

Play any song forever. Drops you into a small terminal UI with a live map of
the song's beats, branch points, and recent jumps.

It's a small Python port of Paul Lamere's *Infinite Jukebox* idea (via
[rigdern's JS rewrite](https://github.com/rigdern/InfiniteJukeboxAlgorithm)):
detect beats, score every beat against every other one, and splice between
the matches forever. The result is a song that never quite ends.

## Quick start

```bash
# one-time install (uses uv — https://docs.astral.sh/uv/)
uv tool install git+https://github.com/schoblaska/infinite-jukebox

# play a YouTube URL or a local file — same command, your call
infinite-jukebox 'https://www.youtube.com/watch?v=K4DyBUG242c'
infinite-jukebox ~/Music/some-song.mp3
```

First run will download (if it's a URL), analyze, and cache results under
`~/.cache/infinite-jukebox/`. Re-runs are instant.

### System dependencies

Two things have to be on your system; everything else is `pip`-installable:

| | macOS | Linux | Windows |
|---|---|---|---|
| **ffmpeg** (decode YouTube audio) | `brew install ffmpeg` | `apt install ffmpeg` | `winget install Gyan.FFmpeg` |
| **PortAudio** (play sound) | `brew install portaudio` | `apt install libportaudio2` | bundled with `sounddevice` |

The CLI prints a friendly setup hint if either is missing.

## What the TUI looks like

```
╭─ - B E D - G O E S - U P - ──────────────────────────────────╮
│                                                              │
│         ·  ·····         ·>──█···· ···      ·····│      · ·  │
│                                                              │
│                136.0 bpm · beat 114/234 · jumps 13 · cov 68% │
╰──────────────────────────────────────────────────────────────╯
```

- **Beat map**: every branchable beat (`·`), the last-branch wall (`│`), the
  cursor (`◆`), and **fading arcs** showing where the last few jumps came from.

Hit `ctrl-c` when you've had enough.

## CLI flags

```
infinite-jukebox SOURCE [options]

  --bar-beats N       beats per bar (default 4)
  --phase N           bar grid offset
  --jump-chance-min   probability right after a jump (default 0.18)
  --jump-chance-max   cap on rising probability (default 0.50)
  --jump-chance-step  per-beat probability rise (default 0.018)
  --seed              fixed RNG seed for reproducible playback
  --no-tui            plain status-line mode (good for logging)
  --cache-dir         override cache location
```

## How it works (very short version)

1. **Beat tracking** with librosa.
2. For every beat, build a feature vector from the sub-beat onset segments
   that overlap it (timbre, pitch, loudness, duration, confidence).
3. **Pairwise distance** between beats; pick neighbors below a threshold,
   subject to a bar-position penalty so jumps land on the same beat of the
   bar.
4. **Reachability flood** picks a *last-branch point* — a beat near the end
   that the song can always loop back from. Past that point we always jump.
5. Playback is a sounddevice callback that rides the beat grid. Each new
   beat rolls a probability; if it wins, hop to a precomputed neighbor;
   else play through.

The audio output is sample-accurate at beat boundaries (snapped to the
nearest zero crossing) so splices don't click.

## Development

```bash
git clone … && cd infinite-jukebox
uv sync
uv run python tests/test_smoke.py     # synth track + analysis + sim check
uv run infinite-jukebox path/to/song  # actual playback
```

## Credits

- Paul Lamere — original Infinite Jukebox.
- [rigdern/InfiniteJukeboxAlgorithm](https://github.com/rigdern/InfiniteJukeboxAlgorithm) — the JS port whose tuning constants we reuse.
- [Textual](https://textual.textualize.io/) and [Rich](https://rich.readthedocs.io/) — the TUI.
