# infinite-jukebox

<div align="center">
  <img width="636" height="169" alt="Screenshot 2026-05-10 at 10 53 20 PM" src="https://github.com/user-attachments/assets/e3eb5883-636c-47eb-a101-564fe6e5c044" />
</div>

A port of Paul Lamere's *Infinite Jukebox* (via
[rigdern's JS rewrite](https://github.com/rigdern/InfiniteJukeboxAlgorithm)).

## Quick start

```bash
# one-time install (uses uv — https://docs.astral.sh/uv/)
uv tool install git+https://github.com/schoblaska/infinite-jukebox

# play a YouTube URL or a local file — same command, your call
infinite-jukebox https://www.youtube.com/watch?v=a4HuUmwWesA
infinite-jukebox ~/Music/some-song.mp3
```

First run will download (if it's a URL), analyze, and cache results under
`~/.cache/infinite-jukebox/`. Re-runs are instant.

### System dependencies

| | macOS | Linux | Windows |
|---|---|---|---|
| **ffmpeg** (decode YouTube audio) | `brew install ffmpeg` | `apt install ffmpeg` | `winget install Gyan.FFmpeg` |
| **PortAudio** (play sound) | `brew install portaudio` | `apt install libportaudio2` | bundled with `sounddevice` |


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
