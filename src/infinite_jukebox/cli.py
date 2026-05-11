from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click

from .analyze import analyze
from .download import fetch_audio
from .player import JUMP_CHANCE_MAX, JUMP_CHANCE_MIN, JUMP_CHANCE_STEP, JukeboxPlayer


DEFAULT_CACHE = Path.home() / ".cache" / "infinite-jukebox"


def _print_status(snap: dict, *, prefix: str = "") -> None:
    cols = shutil.get_terminal_size((80, 24)).columns
    beat = snap["beat_idx"]
    n = snap["n_beats"]
    ev = snap["last_event"]
    ev_str = ""
    if ev is not None:
        kind, src, dst = ev
        if kind in ("jump", "forced", "wrap"):
            ev_str = f"  {kind} {src}->{dst}"
    stats = (
        f"{prefix}beat {beat:>5}/{n}  jumps={snap['jump_count']:<4} "
        f"played={snap['total_beats_played']:<5} cov={snap['coverage']*100:5.1f}% "
        f"p={snap['jump_chance']:.2f}{ev_str}"
    )
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


def _check_deps() -> None:
    """Friendly pre-flight check so missing system bits give helpful messages."""
    problems: list[str] = []
    if shutil.which("ffmpeg") is None:
        problems.append(
            "ffmpeg not found on PATH.\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg   (or your distro's equivalent)\n"
            "  Windows: winget install Gyan.FFmpeg"
        )
    try:
        import sounddevice  # noqa: F401
    except OSError as e:
        problems.append(
            f"audio backend (PortAudio) failed to load: {e}\n"
            "  macOS:   brew install portaudio\n"
            "  Linux:   sudo apt install libportaudio2\n"
            "  Windows: shipped with sounddevice — try reinstalling: uv pip install --reinstall sounddevice"
        )
    if problems:
        click.echo("Setup issues:", err=True)
        for p in problems:
            click.echo("  • " + p.replace("\n", "\n    "), err=True)
        click.echo("", err=True)
        sys.exit(2)


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
    "--bar-beats",
    type=click.IntRange(1, 16),
    default=4,
    show_default=True,
    help="Beats per bar (4 for 4/4 time). Jumps require matching beat position within the bar.",
)
@click.option(
    "--phase",
    type=click.IntRange(0, 1024),
    default=0,
    show_default=True,
    help="Bar-grid phase offset, in beats.",
)
@click.option(
    "--jump-chance-min",
    type=click.FloatRange(0.0, 1.0),
    default=JUMP_CHANCE_MIN,
    show_default=True,
    help="Branch probability immediately after a jump.",
)
@click.option(
    "--jump-chance-max",
    type=click.FloatRange(0.0, 1.0),
    default=JUMP_CHANCE_MAX,
    show_default=True,
    help="Cap on the rising branch probability between jumps.",
)
@click.option(
    "--jump-chance-step",
    type=click.FloatRange(0.0, 1.0),
    default=JUMP_CHANCE_STEP,
    show_default=True,
    help="Per-beat increment to branch probability.",
)
@click.option("--seed", type=int, default=None, help="RNG seed for reproducible playback.")
@click.option(
    "--tui/--no-tui",
    default=None,
    help="Force the Winamp-style TUI on or off. Default: TUI if stdout is a terminal.",
)
@click.option("--quiet", is_flag=True, help="Suppress status output (no-tui mode only).")
def main(
    source: str,
    cache_dir: Path,
    bar_beats: int,
    phase: int,
    jump_chance_min: float,
    jump_chance_max: float,
    jump_chance_step: float,
    seed: int | None,
    tui: bool | None,
    quiet: bool,
) -> None:
    """Play SOURCE (a YouTube URL or local audio file) forever, splicing between similar beats."""
    _check_deps()

    audio_cache = cache_dir / "audio"
    analysis_cache = cache_dir / "analysis"

    click.echo(f"→ fetching: {source}", err=True)
    audio_path, title = fetch_audio(source, audio_cache)
    click.echo(f"  audio: {audio_path}", err=True)

    click.echo(f"→ analyzing: {title}", err=True)
    a = analyze(audio_path, analysis_cache, bar_beats=bar_beats, phase=phase)
    n_branchable = len(a.branches)
    n_edges = sum(len(v) for v in a.branches.values())
    click.echo(
        f"  tempo={a.tempo:.1f} bpm  beats={a.n_beats}  bar={a.bar_beats}  "
        f"branchable_beats={n_branchable} edges={n_edges} last_branch_point={a.last_branch_point}",
        err=True,
    )
    if n_branchable == 0:
        click.echo("  warning: no branches found — playback will just loop.", err=True)

    player = JukeboxPlayer(
        a,
        jump_chance_min=jump_chance_min,
        jump_chance_max=jump_chance_max,
        jump_chance_step=jump_chance_step,
        rng_seed=seed,
    )

    use_tui = tui if tui is not None else sys.stdout.isatty()
    if use_tui:
        from .tui import run_tui

        run_tui(player, title)
        return

    click.echo("→ playing (ctrl-c to stop)", err=True)
    status_cb = None if quiet else _print_status
    try:
        player.run(status_callback=status_cb)
    finally:
        if not quiet:
            sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
