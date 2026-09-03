"""
Prime Picks slate card renderer.

Draws a 1080x1350 (4:5) JPEG summarizing one window's picks. The same image
is posted to X, Facebook, and Instagram.

4:5 is the tallest aspect ratio Instagram accepts, so it occupies the most
feed real estate. It is also fine on X and Facebook.

Fonts must be vendored into the repo at assets/fonts/ — Render's Python
images ship with no fonts, so ImageFont.truetype() will raise OSError if you
rely on system paths. Commit the four Inter TTFs.

Usage:
    card = SlateCard(
        sport="NFL",
        week_label="Week 1",
        window_label="Early window",
        date_label="Sun, Sept 14",
        picks=[Pick("KC", "BUF", "KC +2.5", -110, 1.5), ...],
    )
    path = card.render("/tmp/slate.jpg")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------

W, H = 1080, 1350
MARGIN = 72

BG = "#0d1117"
RULE = "#21262d"
RULE_STRONG = "#30363d"
TEXT = "#ffffff"
TEXT_DIM = "#c9d1d9"
TEXT_MUTED = "#8a919c"
ACCENT = "#3fb950"

FONT_DIR = Path(os.getenv("PP_FONT_DIR", "assets/fonts"))

# Row geometry collapses as the slate grows so a 16-game card still fits.
# (max_picks, row_height, matchup_size, pick_size)
DENSITY_STEPS = [
    (8, 74, 38, 38),
    (12, 58, 32, 32),
    (16, 47, 27, 27),
    (20, 39, 23, 23),
]


@dataclass(frozen=True)
class Pick:
    away: str
    home: str
    selection: str          # "KC +2.5", "Under 47.5", "NYJ ML"
    price: int              # American odds
    units: float

    @property
    def matchup(self) -> str:
        return f"{self.away} @ {self.home}"

    @property
    def price_str(self) -> str:
        return f"+{self.price}" if self.price > 0 else str(self.price)

    @property
    def detail(self) -> str:
        return f"{self.price_str} · {self.units:g}u"


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / f"Inter-{name}.ttf"
    if not path.exists():
        raise OSError(
            f"Font not found: {path}. Vendor the Inter TTFs into "
            f"{FONT_DIR} or set PP_FONT_DIR."
        )
    return ImageFont.truetype(str(path), size)


def _density(n: int) -> tuple[int, int, int]:
    for max_picks, row_h, m_size, p_size in DENSITY_STEPS:
        if n <= max_picks:
            return row_h, m_size, p_size
    return DENSITY_STEPS[-1][1:]


class SlateCard:
    def __init__(
        self,
        *,
        sport: str,               # "NFL" / "CFB"
        week_label: str,          # "Week 1"
        window_label: str,        # "Early window"
        date_label: str,          # "Sun, Sept 14"
        picks: Sequence[Pick],
        record_label: Optional[str] = None,   # "Season 24-19 · +6.4u"
    ) -> None:
        if not picks:
            raise ValueError("SlateCard requires at least one pick")
        self.sport = sport
        self.week_label = week_label
        self.window_label = window_label
        self.date_label = date_label
        self.picks = list(picks)
        self.record_label = record_label

    # ----------------------------------------------------------------------

    def render(self, out_path: str) -> str:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)

        y = self._draw_header(d)
        y = self._draw_picks(d, y)
        self._draw_footer(d)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "JPEG", quality=92, optimize=True)
        return out_path

    # ----------------------------------------------------------------------

    def _draw_header(self, d: ImageDraw.ImageDraw) -> int:
        y = MARGIN + 8

        eyebrow = _font("SemiBold", 22)
        d.text((MARGIN, y), "PRIME PICKS", font=eyebrow, fill=TEXT_MUTED)
        y += 46

        title = _font("Bold", 62)
        d.text((MARGIN, y), f"{self.sport} {self.week_label}", font=title, fill=TEXT)
        y += 82

        sub = _font("Regular", 30)
        d.text(
            (MARGIN, y),
            f"{self.window_label} · {self.date_label}",
            font=sub,
            fill=TEXT_MUTED,
        )
        y += 60

        d.line([(MARGIN, y), (W - MARGIN, y)], fill=RULE_STRONG, width=2)
        return y + 26

    def _draw_picks(self, d: ImageDraw.ImageDraw, top: int) -> int:
        row_h, m_size, p_size = _density(len(self.picks))
        f_match = _font("Regular", m_size)
        f_pick = _font("SemiBold", p_size)
        f_detail = _font("Regular", int(p_size * 0.82))
        d_size = int(p_size * 0.82)

        # Fixed right edges keep the two right-hand columns aligned instead of
        # ragged. Detail column is sized off the widest detail string.
        detail_col_w = max(d.textlength(p.detail, font=f_detail) for p in self.picks)
        detail_right = W - MARGIN
        sel_right = detail_right - detail_col_w - 28

        # Center the block in the space between header rule and footer rule so
        # short slates don't leave a dead zone.
        bottom = H - MARGIN - 54
        block_h = row_h * len(self.picks)
        y = max(top, top + ((bottom - 30 - top) - block_h) // 2)

        for i, pick in enumerate(self.picks):
            baseline = y + (row_h - m_size) // 2

            d.text((MARGIN, baseline), pick.matchup, font=f_match, fill=TEXT_DIM)

            d.text(
                (detail_right - d.textlength(pick.detail, font=f_detail),
                 baseline + (m_size - d_size) // 2),
                pick.detail,
                font=f_detail,
                fill=TEXT_MUTED,
            )
            d.text(
                (sel_right - d.textlength(pick.selection, font=f_pick), baseline),
                pick.selection,
                font=f_pick,
                fill=ACCENT,
            )

            y += row_h
            if i < len(self.picks) - 1:
                d.line([(MARGIN, y), (W - MARGIN, y)], fill=RULE, width=1)

        return y

    def _draw_footer(self, d: ImageDraw.ImageDraw) -> None:
        y = H - MARGIN - 54
        d.line([(MARGIN, y)], fill=RULE_STRONG)
        d.line([(MARGIN, y), (W - MARGIN, y)], fill=RULE_STRONG, width=2)
        y += 22

        f = _font("Regular", 26)
        total_units = sum(p.units for p in self.picks)
        left = f"{len(self.picks)} picks · {total_units:g}u"
        if self.record_label:
            left = f"{left}   |   {self.record_label}"
        d.text((MARGIN, y), left, font=f, fill=TEXT_MUTED)

        right = "primepicks.ai"
        rw = d.textlength(right, font=f)
        d.text((W - MARGIN - rw, y), right, font=f, fill=TEXT_MUTED)


# --------------------------------------------------------------------------
# Captions — one image, three platforms, three caption lengths
# --------------------------------------------------------------------------


def caption_for(platform: str, card: SlateCard) -> str:
    head = f"{card.sport} {card.week_label} — {card.window_label}"
    n = len(card.picks)
    units = sum(p.units for p in card.picks)

    if platform == "x":
        # No URL: X bills $0.20 for a post containing a link vs $0.015 without.
        return f"{head}\n{n} picks · {units:g}u on the card.\n\n#PrimePicks #{card.sport}"

    if platform == "facebook":
        lines = [head, "", f"{n} picks, {units:g} units."]
        lines += [f"{p.matchup} — {p.selection} ({p.price_str})" for p in card.picks]
        lines += ["", "Full card and model breakdown: https://primepicks.ai"]
        return "\n".join(lines)

    if platform == "instagram":
        return (
            f"{head}\n{n} picks · {units:g}u\n\n"
            "Full card at the link in bio.\n\n"
            f"#PrimePicks #{card.sport} #SportsBetting #BettingPicks"
        )

    raise ValueError(f"unknown platform: {platform}")
