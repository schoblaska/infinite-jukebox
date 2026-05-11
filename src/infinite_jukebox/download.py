from __future__ import annotations

import re
from pathlib import Path

import yt_dlp


def _is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s))


def fetch_audio(source: str, cache_dir: Path) -> tuple[Path, str]:
    """Return (audio_path, display_title). If source is a local file, return it as-is."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not _is_url(source):
        p = Path(source).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p, p.stem

    # Probe for video ID and title first so we can check the cache before downloading.
    probe_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(source, download=False)
    video_id = info["id"]
    title = info.get("title", video_id)

    out_template = str(cache_dir / f"{video_id}.%(ext)s")
    existing = list(cache_dir.glob(f"{video_id}.*"))
    audio_exts = {".m4a", ".mp3", ".opus", ".webm", ".wav", ".flac", ".ogg"}
    cached = [p for p in existing if p.suffix.lower() in audio_exts]
    if cached:
        return cached[0], title

    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "0",
            }
        ],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([source])

    for ext in (".m4a", ".mp3", ".opus", ".webm", ".wav"):
        p = cache_dir / f"{video_id}{ext}"
        if p.exists():
            return p, title
    raise RuntimeError(f"yt-dlp did not produce an audio file for {video_id}")
