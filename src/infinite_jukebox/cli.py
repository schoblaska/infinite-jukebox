from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click

from .analyze import analyze
from .download import fetch_audio
from .player import JukeboxPlayer


DEFAULT_CACHE = Path.home() / ".cache" / "infinite-jukebox"


def _print_status(snap: dict, *, prefix: str = "") -> None:
    cols = shutil.get_terminal_size((80, 24)).columns
    beat = snap["beat_idx"]
    n = snap["n_beats"]
    ev = snap["last_event"]
    ev_str = ""
    if ev is not None:
        kind, src, dst = ev
        if kind == "jump":
            ev_str = f"  jump {src}->{dst}"
        elif kind == "wrap":
            ev_str = f"  wrap {src}->{dst}"
    stats = (
        f"{prefix}beat {beat:>5}/{n}  jumps={snap['jump_count']:<4} "
        f"played={snap['total_beats_played']:<5} cov={snap['coverage']*100:5.1f}%{ev_str}"
    )
    # Fit a progress bar in whatever space is left, then truncate to terminal width.
    bar_w = max(0, cols - len(stats) - 4)
    if bar_w >= 8:
        pos = int(bar_w * beat / max(n - 1, 1))
        bar = "#" * pos + "." * (bar_w - pos)
        line = f"{stats}  [{bar}]"
    else:
        line = stats
    line = line[: cols - 1]
    sys.stdout.write("\r\x1b[2K" + line)
    sys.stdout.flush()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("source")
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_CACHE,
    show_default=True,
    help="Where to cache downloaded audio and analysis.",
)
@click.option(
    "--major-prob",
    type=click.FloatRange(0.0, 1.0),
    default=0.55,
    show_default=True,
    help="Probability of jumping at an 8-bar boundary.",
)
@click.option(
    "--minor-prob",
    type=click.FloatRange(0.0, 1.0),
    default=0.15,
    show_default=True,
    help="Probability of jumping at a 4-bar (non-8-bar) boundary.",
)
@click.option(
    "--bar-beats",
    type=click.IntRange(1, 16),
    default=4,
    show_default=True,
    help="Beats per bar (4 for 4/4 time).",
)
@click.option(
    "--phase",
    type=click.IntRange(0, 1024),
    default=0,
    show_default=True,
    help="Beats to offset the bar grid (use if downbeat doesn't align with beat 0).",
)
@click.option(
    "--top-k",
    type=click.IntRange(1, 64),
    default=8,
    show_default=True,
    help="Max branch candidates per slot.",
)
@click.option(
    "--max-distance-pct",
    type=click.FloatRange(0.1, 100.0),
    default=35.0,
    show_default=True,
    help="Keep branches whose distance is in the lowest N% of all pairwise distances.",
)
@click.option("--seed", type=int, default=None, help="RNG seed for reproducible playback.")
@click.option("--quiet", is_flag=True, help="Suppress status output.")
def main(
    source: str,
    cache_dir: Path,
    major_prob: float,
    minor_prob: float,
    bar_beats: int,
    phase: int,
    top_k: int,
    max_distance_pct: float,
    seed: int | None,
    quiet: bool,
) -> None:
    """Play SOURCE (a YouTube URL or local audio file) forever, splicing similar beats."""
    audio_cache = cache_dir / "audio"
    analysis_cache = cache_dir / "analysis"

    click.echo(f"→ fetching: {source}", err=True)
    audio_path, title = fetch_audio(source, audio_cache)
    click.echo(f"  audio: {audio_path}", err=True)

    click.echo(f"→ analyzing: {title}", err=True)
    a = analyze(
        audio_path,
        analysis_cache,
        bar_beats=bar_beats,
        phase=phase,
        top_k=top_k,
        max_distance_pct=max_distance_pct,
    )
    n_major_src = len(a.branches_major)
    n_minor_src = len(a.branches_minor)
    click.echo(
        f"  tempo={a.tempo:.1f} bpm  beats={a.n_beats}  bar={a.bar_beats}  "
        f"8-bar slots branchable={n_major_src}  4-bar slots branchable={n_minor_src}",
        err=True,
    )
    if n_major_src == 0 and n_minor_src == 0:
        click.echo("  warning: no branches found — playback will just loop.", err=True)

    click.echo(f"→ playing (ctrl-c to stop)", err=True)
    player = JukeboxPlayer(
        a,
        major_prob=major_prob,
        minor_prob=minor_prob,
        rng_seed=seed,
    )
    status_cb = None if quiet else _print_status
    try:
        player.run(status_callback=status_cb)
    finally:
        if not quiet:
            sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
