"""Minimal TUI: a tiny tile with one row of beat-map visualization."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static

if TYPE_CHECKING:
    from .player import JukeboxPlayer


class BeatMap(Widget):
    """Single-row mini-map: branch dots, cursor, and a line connecting recent jumps."""

    DEFAULT_CSS = """
    BeatMap {
        height: 3;
        padding: 1 1;
        background: transparent;
    }
    """

    def __init__(self, player: "JukeboxPlayer", **kw) -> None:
        super().__init__(**kw)
        self.player = player
        self._arcs: list[tuple[int, int, float]] = []  # (src_beat, dst_beat, life)

    def on_mount(self) -> None:
        self.set_interval(1 / 15, self._tick)

    def _tick(self) -> None:
        for kind, src, dst in self.player.drain_events():
            if kind in ("jump", "forced", "wrap"):
                self._arcs.append((src, dst, 1.0))
        self._arcs = [(s, d, l - 0.04) for (s, d, l) in self._arcs if l - 0.04 > 0]
        self.refresh()

    def render(self) -> Text:
        width = max(self.size.width, 1)
        a = self.player.a
        n_beats = max(a.n_beats, 1)
        beat = self.player.beat_idx
        last_bp = a.last_branch_point

        def col(b: int) -> int:
            return int(round(b * (width - 1) / max(n_beats - 1, 1)))

        chars: list[str] = [" "] * width
        styles: list[Style | None] = [None] * width

        # branch points
        branch_style = Style(color="grey50")
        for b in a.branches.keys():
            x = col(b)
            if 0 <= x < width:
                chars[x] = "·"
                styles[x] = branch_style

        # connecting lines for active arcs (newest last so it wins ties)
        for s, d, life in self._arcs:
            x1, x2 = col(s), col(d)
            lo, hi = min(x1, x2), max(x1, x2)
            style = Style(color="magenta", dim=life < 0.5)
            for x in range(lo, hi + 1):
                chars[x] = "─"
                styles[x] = style
            if 0 <= x2 < width:
                chars[x2] = "│"
                styles[x2] = style
            if 0 <= x1 < width:
                chars[x1] = ">" if d > s else "<"
                styles[x1] = style

        # last branch point
        lbp_c = col(last_bp)
        if 0 <= lbp_c < width:
            chars[lbp_c] = "│"
            styles[lbp_c] = Style(color="grey50")

        # cursor wins
        cursor_c = col(beat)
        if 0 <= cursor_c < width:
            chars[cursor_c] = "█"
            styles[cursor_c] = Style(color="red", bold=True)

        t = Text()
        for c, st in zip(chars, styles):
            t.append(c, style=st or "")
        return t


class StatsBar(Static):
    DEFAULT_CSS = """
    StatsBar {
        height: 1;
        padding: 0 1;
        background: transparent;
        text-align: right;
    }
    """

    def __init__(self, player: "JukeboxPlayer", **kw) -> None:
        super().__init__(**kw)
        self.player = player

    def on_mount(self) -> None:
        self.set_interval(1 / 4, self._update)
        self._update()

    def _update(self) -> None:
        a = self.player.a
        snap = self.player.snapshot()
        sub_style = Style(color="grey50")
        line = Text(
            f"{a.tempo:.1f} bpm · beat {snap['beat_idx']}/{snap['n_beats']} · "
            f"jumps {snap['jump_count']} · cov {snap['coverage']*100:.0f}%",
            style=sub_style,
        )
        self.update(line)


class JukeboxApp(App):
    CSS = """
    Screen {
        background: transparent;
        align: center middle;
    }
    #tile {
        width: 64;
        height: 6;
        border: round white;
        background: transparent;
    }
    """

    def __init__(self, player: "JukeboxPlayer", title: str) -> None:
        super().__init__()
        self.player = player
        self.title_text = title
        self._audio_thread: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="tile"):
            yield BeatMap(self.player)
            yield StatsBar(self.player)

    def on_mount(self) -> None:
        self.query_one("#tile").border_title = self.title_text
        self._audio_thread = threading.Thread(target=self.player.run, daemon=True)
        self._audio_thread.start()

    async def action_help_quit(self) -> None:
        # Textual 8.x rebound ctrl+c to a "you must press ctrl+q" hint;
        # we want the muscle-memory ctrl+c to actually kill the app.
        self.exit()


def run_tui(player: "JukeboxPlayer", title: str) -> None:
    app = JukeboxApp(player, title)
    try:
        app.run()
    finally:
        player.stop()
