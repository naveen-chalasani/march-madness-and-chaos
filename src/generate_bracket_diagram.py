"""
Generate a bracket anatomy diagram for the README.

Produces: outputs/figures/bracket_anatomy.png

A standalone structural diagram -- no tournament data needed.
Run from project root:
    uv run python src/generate_bracket_diagram.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

# ── Visual theme (same palette as the project notebooks) ──────────────────
COLORS = {
    "primary":   "#1a1a2e",
    "accent":    "#e94560",
    "secondary": "#0f3460",
    "highlight": "#f5a623",
    "bg":        "#f8f9fa",
}

# ── Region identities ─────────────────────────────────────────────────────
REGION_STYLES = {
    "EAST":    {"bg": "#dce8f7", "header": "#1565c0"},
    "SOUTH":   {"bg": "#fde0e4", "header": "#c62828"},
    "WEST":    {"bg": "#d8f0e0", "header": "#2e7d32"},
    "MIDWEST": {"bg": "#ede0f5", "header": "#6a1b9a"},
}

# Standard NCAA first-round matchup order (top → bottom within each region)
MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]

# ── Cell & layout constants ────────────────────────────────────────────────
CW      = 1.25   # cell width
CH      = 0.38   # cell height
SPACING = 0.62   # vertical center-to-center between adjacent R64 cells
RW      = 2.05   # horizontal distance between round starts (left edge to left edge)

REPO_ROOT   = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"


# ── Primitive helpers ─────────────────────────────────────────────────────

def _cell(ax, x_left, y_center, w=CW, h=CH, text=None, fontsize=7.0,
          cell_color="white", border_color="#b0bec5", text_color=None,
          bold=False, zorder=2):
    """Draw a rounded-rectangle cell with optional centered text."""
    text_color = text_color or COLORS["primary"]
    ax.add_patch(FancyBboxPatch(
        (x_left, y_center - h / 2), w, h,
        boxstyle="round,pad=0.02",
        linewidth=0.75, edgecolor=border_color,
        facecolor=cell_color, zorder=zorder,
    ))
    if text:
        ax.text(
            x_left + w / 2, y_center, text,
            ha="center", va="center", fontsize=fontsize,
            color=text_color,
            fontweight="bold" if bold else "normal",
            zorder=zorder + 1,
        )


def _connector(ax, x_out, y_top, y_bot, x_in,
               color="#8faabf", lw=1.4):
    """
    Bracket connector: horizontal stubs from y_top and y_bot → vertical bar
    → horizontal to the midpoint destination cell.
    """
    x_mid = (x_out + x_in) / 2
    y_mid = (y_top + y_bot) / 2
    kw = dict(color=color, lw=lw, solid_capstyle="round", zorder=1)
    ax.plot([x_out, x_mid], [y_top, y_top], **kw)
    ax.plot([x_out, x_mid], [y_bot, y_bot], **kw)
    ax.plot([x_mid, x_mid], [y_top, y_bot], **kw)
    ax.plot([x_mid, x_in],  [y_mid, y_mid], **kw)


# ── Region drawing ─────────────────────────────────────────────────────────

def _draw_region(ax, x0, y0, name, direction="right"):
    """
    Draw one 16-team region bracket with styled cells.

    x0 : for "right" → left edge of R64 cells
         for "left"  → right edge of R64 cells
    y0 : y-center of the bottom-most R64 cell
    Returns (x_exit, y_champ) where x_exit is the horizontal exit point of
    the Elite-8 champion cell (toward the center of the figure).
    """
    sign   = 1 if direction == "right" else -1
    style  = REGION_STYLES[name]

    # ── Y-centers for every round (computed by successive midpoints) ──────
    ys = [[y0 + (7 - i) * SPACING for i in range(8)]]   # R64: 8 cells
    for _ in range(3):
        prev = ys[-1]
        ys.append([(prev[i] + prev[i + 1]) / 2 for i in range(0, len(prev), 2)])
    # ys[0]=R64(8)  ys[1]=R32(4)  ys[2]=S16(2)  ys[3]=E8(1)

    # ── Left edges of each round's cells ─────────────────────────────────
    if direction == "right":
        x_lefts = [x0 + i * RW for i in range(4)]
    else:
        # x0 is the RIGHT edge of R64; cells grow leftward each round
        x_lefts = [x0 - i * RW - CW for i in range(4)]

    # ── Region background rectangle ───────────────────────────────────────
    bg_xl = x_lefts[0 if direction == "right" else 3] - 0.22
    bg_xr = x_lefts[3 if direction == "right" else 0] + CW + 0.22
    bg_yb = ys[0][-1] - CH / 2 - 0.18
    bg_yt = ys[0][0]  + CH / 2 + 0.62   # extra room for header band + labels

    ax.add_patch(Rectangle(
        (bg_xl, bg_yb), bg_xr - bg_xl, bg_yt - bg_yb,
        facecolor=style["bg"], edgecolor="none", alpha=0.55, zorder=0,
    ))

    # ── Region header band ────────────────────────────────────────────────
    band_h = 0.32
    ax.add_patch(Rectangle(
        (bg_xl, bg_yt - band_h), bg_xr - bg_xl, band_h,
        facecolor=style["header"], edgecolor="none", alpha=0.90, zorder=1,
    ))
    ax.text(
        (bg_xl + bg_xr) / 2, bg_yt - band_h / 2, name,
        ha="center", va="center", fontsize=9.5, fontweight="bold",
        color="white", zorder=2,
    )

    # ── Round column labels (small italics below band) ────────────────────
    col_labels = ["R of 64", "R of 32", "Sweet 16", "Elite 8"]
    label_y = bg_yt - band_h - 0.06
    for i, lbl in enumerate(col_labels):
        x_center = x_lefts[i] + CW / 2
        ax.text(x_center, label_y, lbl, ha="center", va="top",
                fontsize=5.2, color=style["header"], fontstyle="italic",
                zorder=2)

    # ── R64 cells with seed matchup text ──────────────────────────────────
    for y, (s1, s2) in zip(ys[0], MATCHUPS):
        _cell(ax, x_lefts[0], y,
              text=f"#{s1}  vs  #{s2}", fontsize=7.2,
              cell_color="white", border_color="#90a4b5")

    # ── Later-round (empty result) cells ─────────────────────────────────
    for rnd in range(1, 4):
        bc = COLORS["highlight"] if rnd == 3 else "#90a4b5"
        bg = "#fef9ec"        if rnd == 3 else "#f5f7f9"
        for y in ys[rnd]:
            _cell(ax, x_lefts[rnd], y, cell_color=bg, border_color=bc)

    # ── Bracket connectors between rounds ─────────────────────────────────
    for rnd in range(3):
        if direction == "right":
            x_out = x_lefts[rnd] + CW      # right edge of source cells
            x_in  = x_lefts[rnd + 1]       # left  edge of dest cell
        else:
            x_out = x_lefts[rnd]            # left  edge of source cells
            x_in  = x_lefts[rnd + 1] + CW  # right edge of dest cell

        for pair in range(len(ys[rnd + 1])):
            _connector(ax, x_out,
                       ys[rnd][pair * 2],
                       ys[rnd][pair * 2 + 1],
                       x_in)

    # ── Return champion exit position ─────────────────────────────────────
    x_exit = x_lefts[3] + CW if direction == "right" else x_lefts[3]
    return x_exit, ys[3][0]


# ── Final Four & Championship ─────────────────────────────────────────────

def _draw_final_four(ax, xe, ye, xs, ys, xw, yw, xm, ym):
    """
    Draw the Final Four bracket in the centre of the figure.

    East  (ye) and West  (yw) are at the same y → Semifinal 1 (top)
    South (ys) and Midwest (ym) are at the same y → Semifinal 2 (bottom)
    Both semifinal winners meet at the Championship cell.
    """
    cx      = (xe + xw) / 2        # true horizontal centre of the figure
    SF_CW   = 1.90                  # semifinal cell width
    SF_CH   = 0.46                  # semifinal cell height
    CH_CW   = 2.10                  # championship cell width
    CH_CH   = 0.58                  # championship cell height

    y_sf1 = ye                      # top    semifinal y
    y_sf2 = ys                      # bottom semifinal y
    y_ch  = (y_sf1 + y_sf2) / 2    # championship y

    sf1_xl = cx - SF_CW / 2
    sf2_xl = cx - SF_CW / 2
    ch_xl  = cx - CH_CW / 2

    ff_kw = dict(color=COLORS["highlight"], lw=1.9,
                 solid_capstyle="round", zorder=1)

    # ── "Final Four" header band in the centre ────────────────────────────
    band_xl = sf1_xl - 0.1
    band_xr = sf1_xl + SF_CW + 0.1
    band_yt = y_sf1 + SF_CH / 2 + 0.30
    band_h  = 0.32
    ax.add_patch(Rectangle(
        (band_xl, band_yt - band_h), band_xr - band_xl, band_h,
        facecolor=COLORS["highlight"], edgecolor="none", alpha=0.85, zorder=1,
    ))
    ax.text((band_xl + band_xr) / 2, band_yt - band_h / 2,
            "FINAL FOUR",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="white", zorder=2)

    # ── Horizontal feed lines (region champions → SF cells) ───────────────
    ax.plot([xe, sf1_xl],          [ye, ye], **ff_kw)   # East  → SF1 left
    ax.plot([xw, sf1_xl + SF_CW], [yw, yw], **ff_kw)   # West  → SF1 right
    ax.plot([xs, sf2_xl],          [ys, ys], **ff_kw)   # South → SF2 left
    ax.plot([xm, sf2_xl + SF_CW], [ym, ym], **ff_kw)   # Midwest → SF2 right

    # ── Semifinal cells ───────────────────────────────────────────────────
    _cell(ax, sf1_xl, y_sf1, w=SF_CW, h=SF_CH,
          text="Semifinal", fontsize=8.0,
          cell_color="#fff8e6", border_color=COLORS["highlight"])
    _cell(ax, sf2_xl, y_sf2, w=SF_CW, h=SF_CH,
          text="Semifinal", fontsize=8.0,
          cell_color="#fff8e6", border_color=COLORS["highlight"])

    # ── Vertical connectors: SF cells → Championship ──────────────────────
    ax.plot([cx, cx], [y_sf1 - SF_CH / 2, y_ch + CH_CH / 2], **ff_kw)
    ax.plot([cx, cx], [y_sf2 + SF_CH / 2, y_ch - CH_CH / 2], **ff_kw)

    # ── Championship cell ─────────────────────────────────────────────────
    _cell(ax, ch_xl, y_ch, w=CH_CW, h=CH_CH,
          text="★   CHAMPION   ★", fontsize=9.5, bold=True,
          cell_color=COLORS["accent"], border_color=COLORS["accent"],
          text_color="white")

    # ── "Championship" sub-label ──────────────────────────────────────────
    ax.text(cx, y_ch - CH_CH / 2 - 0.18, "Championship",
            ha="center", va="top", fontsize=7.0,
            color=COLORS["accent"], fontstyle="italic")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    W, H        = 22.0, 14.5
    y_bot       = 1.10    # bottom region: y-center of lowest R64 cell
    gap_regions = 2.60    # vertical space between the two stacked regions
    y_top       = y_bot + 7 * SPACING + gap_regions

    x_l = 1.50          # left  edge of R64 cells (left-side regions)
    x_r = W - 1.50      # right edge of R64 cells (right-side regions)

    fig, ax = plt.subplots(figsize=(W, H), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")
    ax.set_xlim(0, W)
    ax.set_ylim(-0.8, H + 0.8)

    # ── Four regions ──────────────────────────────────────────────────────
    xe, ye = _draw_region(ax, x_l, y_top, "EAST",    "right")
    xs, ys = _draw_region(ax, x_l, y_bot, "SOUTH",   "right")
    xw, yw = _draw_region(ax, x_r, y_top, "WEST",    "left")
    xm, ym = _draw_region(ax, x_r, y_bot, "MIDWEST", "left")

    # ── Final Four & Championship ─────────────────────────────────────────
    _draw_final_four(ax, xe, ye, xs, ys, xw, yw, xm, ym)

    # ── Title & subtitle ──────────────────────────────────────────────────
    cx = W / 2
    ax.text(cx, H + 0.35,
            "ANATOMY OF THE BRACKET",
            ha="center", va="center", fontsize=19, fontweight="bold",
            color=COLORS["primary"])
    ax.text(cx, H - 0.22,
            "4 regions  ×  16 teams  =  64 teams  →  63 games  →  1 champion",
            ha="center", va="center", fontsize=10.5,
            color=COLORS["secondary"])

    # ── Round progression footer ───────────────────────────────────────────
    ax.text(cx, -0.35,
            "Round of 64   →   Round of 32   →   Sweet 16   →   "
            "Elite 8   →   Final Four   →   Championship",
            ha="center", va="center", fontsize=9,
            color=COLORS["secondary"], fontweight="bold")

    # ── Save ──────────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "bracket_anatomy.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
