# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Figures of the paper drawn from the code.

    python3 tools/make_doc_figures.py all                 # every figure into results/figures
    python3 tools/make_doc_figures.py architecture | overall | field | trajectory
    python3 tools/make_doc_figures.py all --formats tif,eps,pdf --out DIR   # submission files

- ``architecture`` (Fig. 1): the system pipeline.
- ``overall`` (Fig. 3): the ORL and MLP PPO actors; nodes, edges, slots and parameter counts
  are read from current_semantics/orl.py and models.py.
- ``field`` (Fig. 4): the scenario draw and the desired velocity field of the reward,
  computed by current_semantics/reward.py.
- ``trajectory`` (Fig. 9): one Isaac Sim-PX4 case replayed on the local plant (the Isaac flights
  kept no time series); the replay must reproduce the stored local result of that case.

The published figures were finished from these outputs; docs/figures holds them as printed.
Every figure is sized for the journal page: 168 mm full width (two 80 mm columns, 8 mm gap), no
title or caption inside a figure, at least 7 pt lettering at print size.  Font sizes in this
module are points at print size; ``save`` scales diagram text by the sheet-to-print ratio and
checks the printed sizes, overlapping text and text leaving its box.  PNG files are 600 dpi at
print width and SVG files declare the print width.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, FancyBboxPatch,  # noqa: E402
                                Rectangle)

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "python"))
OUT = PACKAGE / "results" / "figures"   # not docs/figures: those are the paper's own images
RESULTS = PACKAGE / "results" / "current_semantics_3d"
CONFIG = PACKAGE / "config" / "current_semantics" / "spatial_view_contact_v1.yaml"

# IJCAS print geometry (see the module docstring)
COLUMN_MM, TEXT_MM = 80.0, 168.0
MIN_PT = 7.0                   # smallest lettering on the printed page
HEAD, TITLE, BODY, NOTE = 9.5, 8.5, 8.0, 7.5     # print sizes: headings, box titles, text, notes
# Result plots (speed, budget, isaac): Springer's artwork guideline (IJCAS is a Springer journal)
# asks for lettering of 8-12 pt at final size with little variation inside one figure.
PLOT_MIN_PT = 8.0
P_HEAD, P_BODY, P_NOTE = 9.0, 8.5, 8.0
PLOT_RC = {"font.size": P_BODY, "axes.titlesize": P_HEAD, "axes.labelsize": P_BODY,
           "xtick.labelsize": P_NOTE, "ytick.labelsize": P_NOTE, "legend.fontsize": P_NOTE}
DPI = 600                     # line figures (IJCAS); colour figures need >= 300
PAD_IN = 0.04                  # margin around a figure on the page [in]
PLOT_W = TEXT_MM / 25.4 - 2 * PAD_IN   # result-plot width that prints at 168 mm with that margin
FORMATS = ("png", "svg")       # --formats adds tif / eps / pdf for submission

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": BODY, "svg.fonttype": "path",
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.titlesize": HEAD, "axes.labelsize": BODY, "xtick.labelsize": BODY,
                     "ytick.labelsize": BODY, "legend.fontsize": BODY, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0})

# paper-figure palette: pastel fill, darker edge of the same hue (one hue per role)
C = {"sim": ("#EEF1F3", "#2B2B2B"), "est": ("#CEDCF6", "#6E98DA"), "know": ("#DDF5E7", "#4FA862"),
     "pol": ("#E2C2B6", "#C0634B"), "off": ("#FBF8F0", "#7A7A7A"), "gt": ("#FCEBE2", "#B5470A"),
     "kg": ("#B9EEF4", "#2B2B2B")}
DASH = "#2F6EA6"          # dashed containers and zoom connectors
INK, MUTED, GRID = "#1F1F1F", "#5A5A5A", "#E4E4E4"
PROPOSED, OTHER = "#C0634B", "#9A9A9A"
# relation types: slots 1-5 of a contrast-checked categorical palette (light surface);
# every relation is also named in the legend and drawn with its own line style
RELATION_STYLE = {
    "qualifiesEstimate": ("#2a78d6", "-"),
    "informsApproach": ("#eb6834", (0, (5, 2))),
    "affectsVisibility": ("#1baf7a", (0, (1.2, 1.6))),
    "conditionsContact": ("#eda100", (0, (6, 2, 1.5, 2))),
    "informsLanding": ("#e87ba4", "-"),
    # ORL's two decision relations: slots 6-7 of the same palette
    "gatesDescent": ("#008300", (0, (8, 2))),
    "informsVertical": ("#6250d6", (0, (2, 1.2, 6, 1.2))),
}


# ----------------------------------------------------------------------------- drawing helpers
def _texts(fig):
    """Every visible text of the figure, without the tick labels of ticks outside the axis view
    (matplotlib keeps them as visible Text objects but never draws them)."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                if not lo - 1e-9 * (hi - lo) <= loc <= hi + 1e-9 * (hi - lo):
                    hidden.update((id(tick.label1), id(tick.label2)))
    return [t for t in fig.findobj(matplotlib.text.Text)
            if t.get_visible() and t.get_text().strip() and t.figure is fig and id(t) not in hidden]


def _tight_width(fig, renderer, pad):
    fig.draw_without_rendering()
    return fig.get_tightbbox(renderer).width + 2 * pad


def audit(fig, name, scale, min_pt=MIN_PT):
    """Print-size checks of a finished figure: lettering of at least ``min_pt``, no two texts
    overlapping, and no text leaving the box, circle or ellipse its anchor sits in.  ``scale``
    is sheet inches per printed inch.  Returns the list of problems (also printed)."""
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    texts = _texts(fig)
    problems = []
    for t in texts:
        size = t.get_fontsize() / scale
        if size < min_pt - 1e-6:
            problems.append(f"{size:.1f} pt < {min_pt:g}: {t.get_text()!r}")
    boxes = [(t, t.get_window_extent(renderer)) for t in texts]
    tol = 0.5 * fig.dpi / 72 * scale          # half a printed point, in display pixels
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if w > tol and h > tol:
                problems.append(f"overlap: {a.get_text()!r} / {b.get_text()!r}")
    shapes = [p for ax in fig.axes for p in ax.patches
              if isinstance(p, (FancyBboxPatch, Rectangle, Circle, Ellipse)) and p.get_visible()]
    for t, bt in boxes:
        anchor = t.get_transform().transform(t.get_position())
        inside = []
        for p in shapes:
            e = p.get_window_extent(renderer)
            if e.x0 <= anchor[0] <= e.x1 and e.y0 <= anchor[1] <= e.y1:
                inside.append((e.width * e.height, p, e))
        if not inside:
            continue
        _, p, e = min(inside, key=lambda r: r[0])
        if getattr(t, "_on_rim", False):
            continue
        if isinstance(p, (Circle, Ellipse)):
            if t.get_horizontalalignment() != "center":
                continue          # an annotation beside a node, not a label inside it
            cx, cy, rx, ry = (e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2, e.width / 2, e.height / 2
            out = any(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 > (1 + tol / min(rx, ry)) ** 2
                      for x in (bt.x0, bt.x1) for y in (bt.y0, bt.y1))
        else:
            out = bt.x0 < e.x0 - tol or bt.x1 > e.x1 + tol or bt.y0 < e.y0 - tol or bt.y1 > e.y1 + tol
        if out:
            problems.append(f"leaves its {type(p).__name__}: {t.get_text()!r}")
    area = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    edge = PAD_IN * scale * fig.dpi
    for t, bt in boxes:
        if (bt.x0 < area.x0 - edge or bt.x1 > area.x1 + edge or bt.y0 < area.y0 - edge
                or bt.y1 > area.y1 + edge):
            problems.append(f"cut off at the figure edge: {t.get_text()!r}")
    for line in problems:
        print(f"  [{name}] {line}")
    return problems


def save(fig, name, width_mm=TEXT_MM, out=None, formats=None, min_pt=MIN_PT):
    """Write the figure at print width ``width_mm`` (600 dpi PNG, SVG declaring the print size).

    A canvas diagram (``canvas``) is drawn on a sheet larger than the page: its text, declared in
    print points, is scaled here by the sheet-to-print ratio, measured on the tight bounding box.
    Other figures are drawn at print size and only checked; result plots pass PLOT_MIN_PT."""
    out = Path(out or OUT)
    formats = formats or ARGS_FORMATS or FORMATS
    out.mkdir(parents=True, exist_ok=True)
    target = width_mm / 25.4
    renderer = fig.canvas.get_renderer()
    texts = _texts(fig)
    if getattr(fig, "_sheet", False):
        for t in texts:
            t._print_pt = t.get_fontsize()
        scale = fig.get_figwidth() / target
        for _ in range(3):
            for t in texts:
                t.set_fontsize(t._print_pt * scale)
            pad = PAD_IN * scale
            scale = _tight_width(fig, renderer, pad) / target
    pad = PAD_IN * (fig.get_figwidth() / target if getattr(fig, "_sheet", False) else 1.0)
    width = _tight_width(fig, renderer, pad)
    scale = width / target
    if not getattr(fig, "_sheet", False) and abs(scale - 1) > 0.03:
        print(f"  [{name}] printed at {target * 25.4:.0f} mm the figure shrinks by {1 / scale:.2f}")
    audit(fig, name, scale, min_pt)
    height_mm = fig.get_tightbbox(renderer).height / scale * 25.4 + 2 * PAD_IN * 25.4
    for ext in formats:
        path = out / f"{name}.{ext}"
        kw = dict(bbox_inches="tight", pad_inches=pad)
        if ext in ("png", "tif"):
            kw["dpi"] = DPI / scale
            kw["pil_kwargs"] = {"dpi": (DPI, DPI)}
            if ext == "tif":
                kw["pil_kwargs"]["compression"] = "tiff_lzw"
        fig.savefig(path, **kw)
        if ext == "svg":
            _svg_print_size(path, width_mm, height_mm)
    plt.close(fig)
    print(f"wrote {out / name}.{{{','.join(formats)}}}  ({width_mm:.0f} x {height_mm:.0f} mm)")


def _svg_print_size(path, width_mm, height_mm):
    """Declare the printed size on the SVG root (the viewBox keeps the drawing)."""
    import re
    text = path.read_text()
    head, rest = text.split("<svg", 1)
    tag, body = rest.split(">", 1)
    tag = re.sub(r'\swidth="[^"]*"', f' width="{width_mm:.2f}mm"', tag, count=1)
    tag = re.sub(r'\sheight="[^"]*"', f' height="{height_mm:.2f}mm"', tag, count=1)
    path.write_text(head + "<svg" + tag + ">" + body)


ARGS_FORMATS = None


def canvas(w, h, width_mm=TEXT_MM):
    """A diagram sheet of w x h units (10 units per sheet inch).  Text sizes are print points;
    ``UPT`` is the number of sheet units per printed point, for laying text out in units."""
    global UPT
    fig = plt.figure(figsize=(w / 10, h / 10))
    fig._sheet = True
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    UPT = w / (width_mm / 25.4 * 72)
    return fig, ax


UPT = 1.0


def box(ax, x, y, w, h, title, lines=(), role="sim", tsize=TITLE, bsize=BODY, dashed=False,
        lw=1.6, spacing=1.18):
    """A rounded box: bold title over muted lines, the block centred vertically (print sizes)."""
    fc, ec = C[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec,
                                lw=lw, ls=(0, (4, 3)) if dashed else "-", zorder=3))
    cx = x + w / 2
    if not lines:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=tsize, weight="bold",
                color=INK, zorder=4)
        return
    th, lh, gap = tsize * 1.2 * UPT, bsize * spacing * UPT, 1.6 * UPT
    top = y + h / 2 + (th + gap + len(lines) * lh) / 2
    ax.text(cx, top - th / 2, title, ha="center", va="center", fontsize=tsize, weight="bold",
            color=INK, zorder=4)
    for i, line in enumerate(lines):
        ax.text(cx, top - th - gap - (i + 0.5) * lh, line, ha="center", va="center", fontsize=bsize,
                color=MUTED, zorder=4)


def panel(ax, x, y, w, h, title, role=None, fill="white", size=HEAD, inset=None):
    """A dashed blue container with a bold title (the paper-figure grouping box)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fill,
                                ec=DASH, lw=2.0, ls=(0, (6, 3)), zorder=1))
    inset = 1.3 if inset is None else inset
    ax.text(x + inset, y + h - inset - size * 0.6 * UPT, title, ha="left", va="center", fontsize=size,
            weight="bold", color=INK, zorder=2)


def arrow(ax, p, q, color=INK, lw=1.4, ls="-", rad=0.0, z=2, size=10):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=size, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=1,
                                 shrinkB=1))


def route(ax, pts, color=INK, lw=1.3, ls="-", label=None, lseg=1, lsize=NOTE, lpos=0.5):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls)
    if label:
        (x0, y0), (x1, y1) = pts[lseg], pts[lseg + 1]
        ax.text(x0 + (x1 - x0) * lpos, y0 + (y1 - y0) * lpos, label, ha="center", va="center",
                fontsize=lsize, color=color, zorder=5, bbox=dict(fc="white", ec="none", pad=0.3))


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8C8C8C")
    ax.tick_params(colors=INK, labelsize=plt.rcParams["xtick.labelsize"], length=2.5, pad=2)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- data extraction
def _rect(ax, x, y, w, h, text, role, size=BODY, weight="normal", z=3):
    fc, ec = C[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec,
                                lw=1.8, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight,
            color=INK, zorder=z + 1, linespacing=1.15)


def _line(ax, pts, color=INK, lw=2.0, ls="-", head=True, size=12, z=2):
    xs, ys = zip(*pts)
    if head:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=size, color=color,
                                     lw=lw, linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))
    else:
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z)


def _circles(ax, x0, y, n, r, fc, ec=INK, gap=0.55, frame=True, ellipsis_at=None):
    xs = []
    x = x0 + r
    for i in range(n):
        if ellipsis_at is not None and i == ellipsis_at:
            ax.text(x + r * 0.6, y, "...", ha="center", va="center", fontsize=BODY, color=INK, zorder=5)
            x += 2.4 * r
        ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=1.0, zorder=4))
        xs.append(x)
        x += 2 * r + gap
    if frame:
        ax.add_patch(FancyBboxPatch((x0 - 0.6, y - r - 0.6), x - x0 - gap + 1.2, 2 * r + 1.2,
                                    boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=INK, lw=1.2,
                                    zorder=3))
    return xs, x - gap


def _stack(ax, x, y, s, fc, n=3, hatch=None):
    for k in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + 0.4 * k, y + 0.4 * k), s, s, fc=fc, ec=INK, lw=0.8, hatch=hatch,
                               zorder=4 + (n - k)))


def _zoom(ax, a, b):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=DASH, lw=1.8, ls=(0, (5, 3)), zorder=1)



ORL_VERSION = "orl/1"     # the proposed model of the paper
ORL_SHORT = {"ObservationQuality": "Q", "VerticalState": "V", "HeadingState": "H",
             "RelativeGeometry": "G", "MotionState": "M", "AttitudeState": "A",
             "ApproachSituation": "P", "ViewGeometry": "O", "ContactCondition": "C",
             "LateralDecision": "Lat", "DescentEligibility": "E", "VerticalDecision": "Z"}


def _orl():
    from ontology_rgat.current_semantics import orl
    return orl


def _feature_count(orl):
    """Length of phi: the shared concepts once, the plane concepts for x and y."""
    dims = orl.input_dimensions()
    return sum(dims[c] for c in orl.SHARED_ROLES) + 2 * sum(dims[c] for c in orl.PLANE_ROLES)


PATH_C, PATH_FILL = "#7B2FBE", "#F1E6FB"      # the graph's output path (decision-node states)
HEAD_PROPOSED, HEAD_BASELINE = "#2F5597", "#6B6B6B"


def _badge(ax, x, y, text, color=PATH_C, r=1.45):
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=1.4, zorder=7))
    badge = ax.text(x, y, text, ha="center", va="center", fontsize=NOTE, weight="bold", color=color,
                    zorder=8)
    badge._on_rim = True
    return badge


def _network(ax, x0, x1, layers, r=1.0, fc=None):
    """A fully connected net drawn bottom-up: layers = [(y, n_shown, ellipsis_at, colour)]."""
    pts = []
    for y, n, cut, colour in layers:
        width = n * 2 * r + (n - 1) * 1.0 + (2.6 * r if cut is not None else 0.0)
        x = (x0 + x1) / 2 - width / 2 + r
        row = []
        for i in range(n):
            if cut is not None and i == cut:
                ax.text(x + 0.3 * r, y, "...", ha="center", va="center", fontsize=NOTE, zorder=5)
                x += 2.6 * r
            row.append((x, y, colour))
            x += 2 * r + 1.0
        pts.append(row)
    for lower, upper in zip(pts[:-1], pts[1:]):
        for (xa, ya, _) in lower:
            for (xb, yb, _) in upper:
                ax.plot([xa, xb], [ya + r, yb - r], color="#9A9A9A", lw=0.5, zorder=3)
    for row in pts:
        for (x, y, colour) in row:
            ax.add_patch(Circle((x, y), r, fc=colour, ec=INK, lw=0.8, zorder=4))
    return pts


def overall():
    """ORL next to plain PPO (the raw_ppo arm), with the graph's output path drawn out (orl/1).
    Print millimetres: 1 unit = 1 mm at 168 mm."""
    orl = _orl()
    graph = orl.GRAPHS[ORL_VERSION]
    semantic = [e for e in graph.edges if e[2] != orl.SELF]
    relations = [r for r in orl.RELATIONS if r != orl.SELF]
    dims = orl.input_dimensions()
    W, H = TEXT_MM, 180.0
    fig, ax = canvas(W, H)
    L0, L1, R0, R1 = 0.5, 115.5, 117.5, 167.5          # the two sides
    top, head = 171.0, 7.5                              # content top, header height
    pa, pb, pc = (121.0, 49.5), (68.5, 50.0), (21.0, 45.0)   # panel (y, height), both sides

    # ---- side frames, header bars, summary banners and the shared band
    for x0, x1, title, colour in ((L0, L1, "Proposed Model: ORL (Ontology + R-GAT)", HEAD_PROPOSED),
                                  (R0, R1, "PPO", HEAD_BASELINE)):
        ax.add_patch(FancyBboxPatch((x0, 19.5), x1 - x0, top + head + 0.5 - 19.5,
                                    boxstyle="round,pad=0,rounding_size=1.6", fc="white", ec="#B8B8B8",
                                    lw=1.0, zorder=0.2))
        ax.add_patch(FancyBboxPatch((x0, top + 0.5), x1 - x0, head, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=colour, ec=colour, lw=1.0, zorder=0.3))
        ax.text((x0 + x1) / 2, top + 0.5 + head / 2, title, ha="center", va="center", fontsize=10,
                weight="bold", color="white", zorder=1)
    ax.add_patch(FancyBboxPatch((L0, 11.5), L1 - L0, 6.6, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="#DCE6F4", ec=HEAD_PROPOSED, lw=1.0, zorder=0.3))
    ax.text((L0 + L1) / 2, 14.8, "Proposed: semantic ontology + R-GAT + decision-node readouts + "
            "mirror average", ha="center", va="center", fontsize=TITLE, weight="bold", zorder=1)
    ax.add_patch(FancyBboxPatch((R0, 11.5), R1 - R0, 6.6, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="#EDEDED", ec=HEAD_BASELINE, lw=1.0, zorder=0.3))
    ax.text((R0 + R1) / 2, 14.8, "PPO: raw observations\n+ feed-forward MLP", ha="center",
            va="center", fontsize=NOTE, weight="bold", zorder=1, linespacing=1.1)
    ax.add_patch(FancyBboxPatch((L0, 0.5), R1 - L0, 9.6, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="white", ec="#8C8C8C", lw=1.0, zorder=0.3))
    ax.text(2.0, 5.3, r"Same in both arms: critic $V(O_t)$ = MLP 21→55→1 on raw $O_t$ (training only); "
            r"$z_t\sim\mathcal{N}(\mu_t,\mathrm{diag}\,\sigma^2)$ with one learned $\log\sigma$;" + "\n"
            r"$a_t=a_{max}\odot\tanh(z_t)$, $a_{max}=(2.5,2.5,2.0)$ m/s$^2$; same PPO settings, reward, "
            "curriculum and seeds.  Parameters: ORL 4,016, PPO 4,022.", fontsize=BODY,
            va="center", zorder=1, linespacing=1.3)

    # ---- the two columns: input -> three blocks -> policy head -> action
    def column(x0, x1, blocks, centre, label_lines):
        for y, h, text, role in blocks:
            _rect(ax, x0, y, x1 - x0, h, text, role, size=TITLE)
        ax.text(centre, 23.0, label_lines, ha="center", va="center", fontsize=BODY, linespacing=1.1)
        _line(ax, [(centre, 25.6 if "\n" in label_lines else 24.6), (centre, blocks[0][0])], lw=2.2,
              size=12)
        _rect(ax, x0, 158.0, x1 - x0, 7.0, "", "sim")
        ax.text(centre, 162.6, "policy head", ha="center", va="center", fontsize=BODY, weight="bold",
                zorder=5)
        ax.text(centre, 159.9, "Gaussian, tanh", ha="center", va="center", fontsize=NOTE, color=MUTED,
                zorder=5)
        _line(ax, [(centre, 165.0), (centre, 166.8)], lw=1.4, size=8)
        ax.text(centre, 168.8, r"$a_t=(a_x,a_y,a_z)$", ha="center", va="center", fontsize=BODY)

    # proposed column: the ORL block as four mirrored copies of one network
    cx, bx0, bx1 = 16.25, 5.5, 27.0
    for k in (2, 1):
        ax.add_patch(FancyBboxPatch((1.5 + 0.7 * k, 29.0 + 0.7 * k), 26.5, 116.5,
                                    boxstyle="round,pad=0,rounding_size=0.3", fc="white", ec=DASH, lw=0.9,
                                    ls=(0, (3, 3)), zorder=0.5))
    ax.add_patch(FancyBboxPatch((1.5, 29.0), 26.5, 116.5, boxstyle="round,pad=0,rounding_size=0.3",
                                fc="white", ec=DASH, lw=2.2, ls=(0, (8, 4)), zorder=1))
    side = ax.text(3.4, 87.0, r"$\times 4$ mirrored copies, one set of weights: input $M_g\,\phi(O_t)$, "
                   r"output mapped back by $S_g$", fontsize=NOTE, color=DASH, rotation=90, ha="center",
                   va="center")
    side._on_rim = True               # runs along the stacked copies' edges by design
    left_blocks = [(36.0, 14.0, "Semantic\nFeatures +\nTyped Node\nEncoders", "est"),
                   (86.5, 14.0, "Relational\nGraph Attention\n(R-GAT)", "know"),
                   (123.0, 14.0, "Decision\nReadouts\n(lateral MLP,\ndescent gate)", "pol")]
    column(bx0, bx1, left_blocks, cx, r"$O_t$ (21 channels)")
    _line(ax, [(cx, 50.0), (cx, 86.5)], lw=2.2, size=12)
    ax.add_patch(FancyArrowPatch((27.0, 97.0), (27.0, 90.0), arrowstyle="-|>", mutation_scale=10,
                                 connectionstyle="arc3,rad=-1.2", color=INK, lw=1.6, zorder=3))
    ax.text(cx + 1.4, 68.0, f"{orl.HOPS} hops,\nshared", fontsize=NOTE, va="center", linespacing=1.1)
    _line(ax, [(cx, 100.5), (cx, 123.0)], color=PATH_C, lw=3.0, size=13)
    ax.text(cx + 1.6, 111.5, "graph\noutput", fontsize=NOTE, color=PATH_C, va="center", weight="bold",
            linespacing=1.1)
    ax.add_patch(Circle((cx, 149.0), 3.2, fc="white", ec=INK, lw=1.8, zorder=5))
    ax.text(cx, 149.0, r"$\frac{1}{4}\Sigma$", fontsize=BODY, ha="center", va="center", zorder=6)
    ax.text(cx - 4.4, 149.0, "mirror\naverage", fontsize=NOTE, ha="right", va="center", linespacing=1.1)
    _line(ax, [(cx, 137.0), (cx, 145.8)], lw=2.2, size=12)
    _line(ax, [(cx, 152.2), (cx, 158.0)], lw=2.2, size=12)
    ax.text(cx + 1.4, 155.0, r"$\mu_t$", fontsize=BODY, va="center")

    # baseline column
    vx, vx0, vx1 = 127.5, 119.0, 136.0
    right_blocks = [(36.0, 14.0, "Raw\nObservation\nInput", "est"),
                    (86.5, 14.0, "MLP Actor\n" + r"$\pi_\theta(O_t)$", "know"),
                    (123.0, 14.0, "Policy\nReadout\n(linear)", "pol")]
    column(vx0, vx1, right_blocks, vx, r"$O_t$" + "\n(21 channels)")
    _line(ax, [(vx, 50.0), (vx, 86.5)], lw=2.2, size=12)
    _line(ax, [(vx, 100.5), (vx, 123.0)], lw=2.2, size=12)
    _line(ax, [(vx, 137.0), (vx, 158.0)], lw=2.2, size=12)
    ax.text(vx + 1.4, 147.5, r"$\mu_t$", fontsize=BODY, va="center")

    # ---- detail panels and zoom connectors
    PX, PW = 31.0, 83.5
    QX, QW = 138.5, 28.0
    titles = (("Decision Readouts, Mirror Average and Policy", "Policy Readout"),
              ("Relational Graph Attention (R-GAT)", "MLP Actor"),
              ("Semantic Features and Typed Node Inputs", "Raw Input"))
    for (py, ph), (lt, rt), lb, rb in zip((pa, pb, pc), titles, reversed(left_blocks),
                                          reversed(right_blocks)):
        panel(ax, PX, py, PW, ph, lt)
        panel(ax, QX, py, QW, ph, rt)
        for (bx, block), (x, w) in (((bx1, lb), (PX, PW)), ((vx1, rb), (QX, QW))):
            _zoom(ax, (bx, block[0] + block[1]), (x, py + ph))
            _zoom(ax, (bx, block[0]), (x, py))

    # panel A: the four decision-node states -> readouts -> mirror average
    y0 = pa[0]
    e = orl.EMBEDDING
    ax.add_patch(FancyBboxPatch((32.3, y0 + 1.4), 81.0, 10.6, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=PATH_FILL, ec=PATH_C, lw=1.6, zorder=2))
    groups = [(34.0, "Lateral x"), (52.5, "Lateral y"), (73.5, "Vertical Z"), (93.0, "Eligibility E")]
    centres = []
    for x0, name in groups:
        _, x_end = _circles(ax, x0, y0 + 8.7, 4, 1.25, C["pol"][0], ellipsis_at=3, gap=0.45)
        ax.text((x0 + x_end) / 2, y0 + 4.3, name, fontsize=NOTE, ha="center", va="center", color=INK,
                zorder=4)
        centres.append((x0 + x_end) / 2)
    for c in centres:
        _line(ax, [(c, y0 + 10.6), (c, y0 + 15.4)], color=PATH_C, lw=1.8, size=9, z=5)
    _badge(ax, (centres[0] + centres[1]) / 2, y0 + 13.4, "1")
    _badge(ax, (centres[2] + centres[3]) / 2, y0 + 13.4, "2")
    _rect(ax, 32.3, y0 + 15.4, 38.0, 7.6, f"Lateral MLP {e}→{orl.LATERAL_HIDDEN}→1\nshared by x and y",
          "pol", size=BODY)
    _rect(ax, 72.3, y0 + 15.4, 41.0, 7.6,
          r"$\mathrm{hold}(Z)+\sigma(\mathrm{gate}(E))\cdot\mathrm{descend}(Z)$", "pol", size=BODY)
    for c in (51.3, 92.8):
        _line(ax, [(c, y0 + 23.0), (c, y0 + 25.4)], lw=1.2, size=8)
    _rect(ax, 32.3, y0 + 25.4, 81.0, 5.4, r"$f(M_g\phi)=(\mu_x,\ \mu_y,\ \mu_z)$ of one mirrored copy",
          "sim", size=BODY)
    _line(ax, [(72.8, y0 + 30.8), (72.8, y0 + 33.2)], lw=1.2, size=8)
    _rect(ax, 32.3, y0 + 33.2, 81.0, 6.4,
          r"$\mu_t=\frac{1}{4}\Sigma_g\, S_g\,f(M_g\phi)$: exact mirror equivariance", "pol", size=BODY)

    # panel B: relational attention, ending in the graph output
    y0 = pb[0]
    rel_colors = [RELATION_STYLE[r][0] for r in relations] + ["#9A9A9A"]
    for k, col in enumerate(rel_colors):
        ax.add_patch(Rectangle((32.5 + k * 2.9, y0 + 2.4), 2.3, 2.3, fc=col, ec=INK, lw=0.7, zorder=4))
    ax.text(56.5, y0 + 3.55, r"neighbour states $h_j$ (colour = relation, 7 + self)", fontsize=NOTE,
            va="center")
    _line(ax, [(35.0, y0 + 5.2), (35.0, y0 + 9.6)], lw=1.1, size=7)
    ax.text(36.4, y0 + 7.5, r"relation-specific projection $W_{r(ij)}$", fontsize=NOTE, va="center")
    for k, col in enumerate(rel_colors[:5]):
        _stack(ax, 32.5 + k * 4.4, y0 + 10.0, 2.6, col)
    ax.text(55.8, y0 + 11.4, "...", fontsize=BODY, va="center")
    ax.text(32.5, y0 + 16.8, r"messages $m_{ij}=W_{r(ij)}\,h_j$", fontsize=BODY, va="center")
    ax.add_patch(Rectangle((110.2, y0 + 6.8), 2.6, 2.6, fc="white", ec=INK, lw=0.9, zorder=4))
    ax.text(109.2, y0 + 8.1, r"target state $h_i$", fontsize=NOTE, ha="right", va="center")
    _line(ax, [(111.5, y0 + 9.4), (111.5, y0 + 12.6)], lw=1.1, size=7)
    ax.add_patch(FancyBboxPatch((64.0, y0 + 12.6), 50.0, 5.4, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="white", ec=INK, lw=1.1, zorder=3))
    ax.text(89.0, y0 + 15.3, r"$e_{ij}=\mathrm{LeakyReLU}(a_s^\top m_{ij}+a_d^\top h_i+b_{r(ij)})$",
            fontsize=BODY, ha="center", va="center", zorder=4)
    ax.plot([44.0, 44.0, 89.0, 89.0], [y0 + 18.6, y0 + 20.8, y0 + 20.8, y0 + 18.0], color=INK, lw=1.1,
            zorder=3)
    _line(ax, [(44.0, y0 + 20.8), (44.0, y0 + 25.6)], lw=1.1, size=7)
    ax.text(46.0, y0 + 22.9, r"softmax over the edges into $i$, weighted sum", fontsize=NOTE, va="center",
            bbox=dict(fc="white", ec="none", pad=0.2), zorder=4)
    _, row_end = _circles(ax, 32.5, y0 + 27.6, 8, 1.1, C["know"][0], ellipsis_at=6, gap=0.4)
    ax.text(row_end + 1.6, y0 + 28.0, r"$h_i'=\tanh(\sum_j \alpha_{ij}\,m_{ij}+b)\in\mathbb{R}^{%d}$" % e,
            fontsize=BODY, va="center")
    ax.text(32.5, y0 + 36.0, "self loop on every node; one layer,\napplied 3 times, shared by both planes",
            fontsize=NOTE, color=MUTED, va="center", linespacing=1.15)
    ax.add_patch(FancyBboxPatch((89.6, y0 + 31.2), 24.4, 12.6, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=PATH_FILL, ec=PATH_C, lw=1.6, zorder=4))
    ax.text(101.8, y0 + 37.5, "after 3 hops: the 4\ndecision-node states\n"
            + r"$h^{(3)}\in\mathbb{R}^{%d}$ each" % e, fontsize=NOTE, ha="center", va="center",
            color=PATH_C, weight="bold", zorder=5, linespacing=1.2)
    _line(ax, [(row_end + 35.0, y0 + 28.0), (101.8, y0 + 28.0), (101.8, y0 + 31.2)], color=PATH_C,
          lw=1.6, size=8)
    _line(ax, [(101.8, y0 + 43.8), (101.8, pa[0] + 1.4)], color=PATH_C, lw=3.0, size=13, z=6)

    # panel C: features and typed node inputs
    y0 = pc[0]
    colours = [("#FFC20E", 4), ("#6E98DA", 3), ("#4FA862", 7), ("#B59AD8", 7)]
    k = 0
    for col, n in colours:
        for _ in range(n):
            ax.add_patch(Rectangle((32.5 + (k % 7) * 2.4, y0 + 27.0 - (k // 7) * 2.4), 2.05, 2.05, fc=col,
                                   ec=INK, lw=0.6, zorder=4))
            k += 1
    ax.text(32.5, y0 + 32.2, r"observation $O_t$", fontsize=BODY, va="center")
    ax.text(32.5, y0 + 21.4, "quality,\ngeometry,\nmotion, attitude", fontsize=NOTE, va="top",
            color=MUTED, linespacing=1.1)
    _line(ax, [(49.5, y0 + 24.6), (53.0, y0 + 24.6)], lw=1.3, size=8)
    ax.add_patch(FancyBboxPatch((53.0, y0 + 12.0), 26.0, 24.5, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=C["know"][0], ec=C["know"][1], lw=1.5, zorder=3))
    ax.text(66.0, y0 + 32.6, "semantic features\n" + r"$\phi(O_t)$" + f", {_feature_count(orl)} values",
            fontsize=BODY, ha="center", va="center", zorder=4, linespacing=1.15)
    ax.text(66.0, y0 + 25.8, r"$B(x;s)=x/(|x|+s)$", fontsize=NOTE, ha="center", va="center", zorder=4)
    ax.text(66.0, y0 + 20.4, "y plane aligned to x\nby declared signs", fontsize=NOTE, ha="center",
            va="center", zorder=4, linespacing=1.1)
    ax.text(66.0, y0 + 14.8, r"invalid $\rightarrow$ 0 + flag", fontsize=NOTE, ha="center", va="center",
            zorder=4)
    _line(ax, [(79.0, y0 + 24.6), (82.4, y0 + 24.6)], lw=1.3, size=8)
    shared = list(orl.SHARED_ROLES)
    planar = list(orl.PLANE_ROLES)
    sx, px, pitch = 85.6, (85.6, 97.6), 2.0
    ax.text(px[0] + 4.0, y0 + 36.6, "x plane", fontsize=NOTE, ha="center", va="center", color=MUTED)
    ax.text(px[1] + 4.0, y0 + 36.6, "y plane", fontsize=NOTE, ha="center", va="center", color=MUTED)
    rows = [(c, "shared") for c in shared] + [(c, "plane") for c in planar] + [("decisions", "decision")]
    for i, (concept, kind) in enumerate(rows):
        yy = y0 + 33.4 - i * 2.6
        label = "D" if kind == "decision" else ORL_SHORT[concept]
        ax.text(84.0, yy + 1.0, label, fontsize=MIN_PT, ha="center", va="center")
        if kind == "shared":
            for j in range(dims[concept]):
                ax.add_patch(Rectangle((sx + j * pitch, yy), 1.75, 2.0, fc=C["est"][0], ec=INK, lw=0.5,
                                       zorder=4))
            if concept == "VerticalState":
                ax.text(sx + 8 * pitch + 0.8, yy + 1.0, "shared", fontsize=NOTE, color=MUTED, va="center")
        elif kind == "plane":
            for base in px:
                for j in range(dims[concept]):
                    ax.add_patch(Rectangle((base + j * pitch, yy), 1.75, 2.0, fc=C["know"][0], ec=INK,
                                           lw=0.5, zorder=4))
        else:
            for j, name in enumerate(("Lat x", "Lat y", "E", "Z")):
                ax.add_patch(Rectangle((sx + j * 7.2, yy - 0.3), 6.9, 2.6, fc=C["pol"][0], ec=INK,
                                       lw=0.5, zorder=4))
                ax.text(sx + 3.45 + j * 7.2, yy + 1.0, name, fontsize=MIN_PT, ha="center", va="center",
                        zorder=5)
    ax.text(32.5, y0 + 5.6, f"{len(dims)} typed encoders (one per concept);\n"
            f"{len(orl.NODES)} nodes, {len(semantic)} edges, {len(relations)} relation types;\n"
            "decision nodes start from learned states", fontsize=NOTE, color=MUTED, va="center",
            linespacing=1.15)

    # baseline panels
    y0 = pa[0]
    _circles(ax, QX + 3.5, y0 + 6.0, 5, 1.2, C["know"][0], ellipsis_at=4, gap=0.45)
    ax.text(QX + QW / 2, y0 + 2.4, "hidden layer (110)", fontsize=NOTE, ha="center", va="center",
            color=MUTED)
    _line(ax, [(QX + QW / 2, y0 + 7.8), (QX + QW / 2, y0 + 11.0)], lw=1.2, size=8)
    _rect(ax, QX + 1.5, y0 + 11.0, QW - 3.0, 5.6, "linear 110 → 3", "sim", size=BODY)
    _line(ax, [(QX + QW / 2, y0 + 16.6), (QX + QW / 2, y0 + 19.6)], lw=1.2, size=8)
    _rect(ax, QX + 1.5, y0 + 19.6, QW - 3.0, 6.0, r"$\mu_t=(\mu_x,\mu_y,\mu_z)$", "pol", size=BODY)
    ax.text(QX + QW / 2, y0 + 32.0, "no mirror average,\nno descent gate", fontsize=NOTE, ha="center",
            va="center", color=MUTED, linespacing=1.15)
    y0 = pb[0]
    _network(ax, QX + 1.0, QX + QW - 5.0,
             [(y0 + 6.0, 5, 4, C["est"][0]), (y0 + 19.0, 6, 5, C["know"][0]), (y0 + 32.0, 3, None, C["pol"][0])],
             r=1.05)
    for y, text in ((y0 + 6.0, "21"), (y0 + 19.0, "110"), (y0 + 32.0, "3")):
        ax.text(QX + QW - 1.0, y, text, fontsize=NOTE, ha="right", va="center", color=MUTED)
    ax.text(QX + QW / 2, y0 + 38.2, "tanh hidden layer;\nactor 2,753 parameters", fontsize=NOTE,
            ha="center", va="center", color=MUTED, linespacing=1.15)
    y0 = pc[0]
    k = 0
    for col, n in colours:
        for _ in range(n):
            ax.add_patch(Rectangle((QX + 5.4 + (k % 7) * 2.4, y0 + 22.0 - (k // 7) * 2.4), 2.05, 2.05,
                                   fc=col, ec=INK, lw=0.6, zorder=4))
            k += 1
    ax.text(QX + QW / 2, y0 + 28.2, r"observation $O_t$", fontsize=BODY, ha="center", va="center")
    ax.text(QX + QW / 2, y0 + 10.0, "no semantic features,\nno graph, no symmetry", fontsize=NOTE,
            ha="center", va="center", color=MUTED, linespacing=1.15)
    save(fig, "overall_architecture")



# ----------------------------------------------------------------------------- 1 architecture
def architecture():
    """System pipeline, drawn in print millimetres (1 unit = 1 mm at 168 mm)."""
    orl = _orl()
    graph = orl.GRAPHS[ORL_VERSION]
    semantic = [e for e in graph.edges if e[2] != orl.SELF]
    relations = [r for r in orl.RELATIONS if r != orl.SELF]
    W, H = TEXT_MM, 106.0
    fig, ax = canvas(W, H)
    PW, GAP, X0 = 38.4, 4.5, 0.5
    PX = [X0 + i * (PW + GAP) for i in range(4)]
    bh, bgap, head = 13.4, 2.6, 6.2
    ph = head + 4 * bh + 3 * bgap + 1.2
    py = H - 0.5 - ph
    rows = [py + ph - head - bh - k * (bh + bgap) for k in range(4)]
    content = {
        0: ("1  Isaac Sim + Pegasus", "sim", [
            ("Moving landing pad", ["UGV deck, ArUco board", "random bearing, heading"]),
            ("Quadrotor + PX4 SITL", ["lockstep physics", "attitude + thrust setpoints"]),
            ("Sensors", ["camera · IMU · GNSS", "seeded wind, handover kick"]),
            ("Degradation", ["vision dropout, optical noise", "pad acceleration"])]),
        1: ("2  Perception & state", "est", [
            ("ArUco detection + PnP", ["ROS 2 camera stream", "capture-time aligned"]),
            ("Pad track", ["relative position, velocity", "re-acquired after rejections"]),
            ("Own navigation state", ["PX4 EKF2 output", "no additional estimator"]),
            (r"Current observation $O_t$", ["21 channels, no history", "quality, geometry, motion"])]),
        2: ("3  Ontology semantics", "know", [
            ("Concepts and relations", [f"{len(orl.input_dimensions())} concepts × 2 planes",
                                        f"{len(relations)} relations, {len(semantic)} edges + self"]),
            (f"Semantic features ({_feature_count(orl)})",
             ["approach, view geometry,", "contact condition"]),
            ("Validity rules", ["no estimate or stale camera", "→ neutral 0 + explicit flag"]),
            (f"Typed node inputs ({len(orl.NODES)})", ["x, y planes share encoders",
                                                       "y aligned to x by signs"])]),
        3: ("4  Policy & flight control", "pol", [
            ("ORL relational encoder", ["one typed attention layer", f"shared over {orl.HOPS} hops"]),
            ("Decision readouts", ["lateral MLP shared by x, y", "descent gate on eligibility E"]),
            ("Mirror-equivariant actor",
             [r"$\mu_t=\frac{1}{4}\Sigma_g\, S_g f(M_g\phi)\rightarrow a_t$", "no raw-observation bypass"]),
            ("Attitude adapter", ["roll, pitch, thrust; yaw held", "20° command limit → PX4"])]),
    }
    for i, (title, role, items) in content.items():
        panel(ax, PX[i], py, PW, ph, title, role)
        for (t, lines), y in zip(items, rows):
            box(ax, PX[i] + 1.0, y, PW - 2.0, bh, t, lines, role=role,
                spacing=1.42 if "$" in lines[0] else 1.18)
    ymid = (rows[1] + rows[2] + bh) / 2
    arrow(ax, (PX[0] + PW, ymid), (PX[1], ymid), color=C["sim"][1], lw=2.4, size=13)
    arrow(ax, (PX[1] + PW, ymid), (PX[2], ymid), color=C["est"][1], lw=2.4, size=13)
    ax.text((PX[1] + PW + PX[2]) / 2, ymid + 2.4, r"$O_t$", ha="center", fontsize=BODY,
            color=C["est"][1])
    k = C["know"][1]
    # edges (concepts and relations) and nodes (typed node inputs) into the encoder
    y0 = rows[0] + bh / 2
    arrow(ax, (PX[2] + PW - 1.0, y0 + 1.5), (PX[3] + 1.0, y0 + 1.5), color=k, lw=1.7)
    gx = PX[2] + PW + GAP / 2
    ax.plot([PX[2] + PW - 1.0, gx, gx], [rows[3] + bh / 2, rows[3] + bh / 2, y0 - 1.5], color=k,
            lw=1.7, zorder=2)
    arrow(ax, (gx, y0 - 1.5), (PX[3] + 1.0, y0 - 1.5), color=k, lw=1.7)
    p = C["pol"][1]
    for a, b in zip(rows[:-1], rows[1:]):
        arrow(ax, (PX[3] + PW / 2, a), (PX[3] + PW / 2, b + bh), color=p, lw=1.4)

    # offline training band
    by, bh2 = 0.5, 25.0
    ax.add_patch(FancyBboxPatch((0.5, by), W - 1.0, bh2, boxstyle="round,pad=0,rounding_size=1.0",
                                fc="white", ec=C["off"][1], lw=1.5, zorder=1))
    ax.text(2.0, by + bh2 - 3.0, "Training: local plant calibrated to Isaac/PX4 response, then "
            "evaluated in Isaac Sim + PX4 SITL", fontsize=TITLE, weight="bold", color=C["off"][1],
            va="center")
    tw, tgap = 29.6, 4.0
    ty, th = by + 4.6, 13.4
    xs = [2.0 + i * (tw + tgap) for i in range(5)]
    box(ax, xs[0], ty, tw, th, "Ground truth", ["reward + verdict only", "never a policy input"],
        role="gt")
    box(ax, xs[1], ty, tw, th, "Curriculum", ["bearing × heading", "success-gated difficulty"],
        role="off")
    box(ax, xs[2], ty, tw, th, "Rollouts", ["12 full episodes", "per PPO update"], role="off")
    box(ax, xs[3], ty, tw, th, "Reward", [r"$\Phi(s')-\Phi(s)$ + terminal", "verdict table"],
        role="off")
    box(ax, xs[4], ty, tw, th, "PPO update", ["clip 0.2, GAE 0.95", r"critic $V(O_t)$ on raw $O_t$"],
        role="off")
    for a, b in zip(xs[1:-1], xs[2:]):
        arrow(ax, (a + tw, ty + th / 2), (b, ty + th / 2), color=C["off"][1])
    route(ax, [(xs[0] + tw / 2, ty), (xs[0] + tw / 2, ty - 2.2), (xs[3] + tw / 2, ty - 2.2),
               (xs[3] + tw / 2, ty)], color=C["gt"][1], ls=(0, (3, 2)))
    # closed loop back to the vehicle, the raw observation to the critic, the weights to the actor
    la, lb = py - 3.6, py - 8.4
    route(ax, [(PX[3] + PW * 0.2, py), (PX[3] + PW * 0.2, la), (PX[0] + PW * 0.5, la),
               (PX[0] + PW * 0.5, py)], color=p, lw=1.5, label="setpoints to PX4 (closed loop, ~10 Hz)",
          lseg=1, lpos=0.62)
    route(ax, [(PX[1] + PW * 0.82, py), (PX[1] + PW * 0.82, lb), (xs[4] + tw * 0.3, lb),
               (xs[4] + tw * 0.3, ty + th)], color=C["est"][1], lw=1.3, ls=(0, (4, 2)),
          label=r"raw $O_t$ reaches the critic only (training)", lseg=1)
    wx = xs[4] + tw * 0.78
    route(ax, [(wx, ty + th), (wx, (ty + th + py) / 2), (wx, py)], color=p, ls=(0, (3, 2)))
    ax.text(wx - 1.2, (la + lb) / 2, "learned\nweights", fontsize=NOTE, color=p, va="center",
            ha="right", linespacing=1.1)
    save(fig, "system_architecture")


# ----------------------------------------------------------------------------- 2 model
# ----------------------------------------------------------------------------- 3 graph
def field():
    from ontology_rgat.current_semantics.contracts import load_config
    from ontology_rgat.current_semantics.reward import desired_velocity
    vf = load_config(CONFIG).task.reward.field
    fig = plt.figure(figsize=(TEXT_MM / 25.4, 2.55), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.0, 1.0])
    # (a) scenario geometry, top view
    ax = fig.add_subplot(gs[0])
    ax.set_aspect("equal")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.95, 1.55)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        ax.plot([0, 1.45 * math.cos(a)], [0, 1.45 * math.sin(a)], color="#C8C8C8", lw=0.6,
                ls=(0, (3, 2)))
    ax.add_patch(Circle((0, 0), 1.45, fc="none", ec="#C8C8C8", lw=0.6))
    ax.add_patch(Circle((0, 0), 0.5, fc=C["est"][0], ec=C["est"][1], lw=0.9))
    ax.add_patch(Rectangle((-0.26, -0.26), 0.52, 0.52, fc="white", ec=C["est"][1], lw=0.6))
    for (dx, dy) in [(-0.13, -0.13), (0.13, -0.13), (-0.13, 0.13), (0.13, 0.13), (0, 0)]:
        ax.add_patch(Rectangle((dx - 0.05, dy - 0.05), 0.1, 0.1, fc=INK, ec="none"))
    B = math.radians(60)
    ax.annotate("", xy=(1.05 * math.cos(B), 1.05 * math.sin(B)), xytext=(0.5 * math.cos(B), 0.5 * math.sin(B)),
                arrowprops=dict(arrowstyle="-|>", color=C["est"][1], lw=1.2))
    white = dict(fc="white", ec="none", pad=0.15)
    ax.text(0.62, 1.0, "pad heading B\n(+ acceleration)", fontsize=NOTE, color=C["est"][1],
            va="bottom", ha="left", linespacing=1.1, bbox=white)
    A = math.radians(200)
    r = 0.95
    dx, dy = -r * math.cos(A), -r * math.sin(A)
    ax.plot(dx, dy, marker="X", ms=7, color=PROPOSED, mec="white", mew=0.6, zorder=5)
    ax.annotate("", xy=(0.2 * math.cos(A + math.pi), 0.2 * math.sin(A + math.pi)), xytext=(dx, dy),
                arrowprops=dict(arrowstyle="-|>", color=PROPOSED, lw=0.9, ls=(0, (3, 2))))
    ax.text(dx + 0.12, dy + 0.12, "drone at\nhandover", ha="left", va="bottom", fontsize=NOTE,
            color=PROPOSED, linespacing=1.1, bbox=white)
    ax.text(-1.5, -0.62, "bearing A to the pad,\noffset 0.3–0.9 m", fontsize=NOTE, color=PROPOSED,
            va="top", linespacing=1.1, bbox=white)
    ax.text(-1.5, -1.92, "A, B: 4 sectors each;\n4 of the 16 pairs held out", fontsize=NOTE,
            color=MUTED, va="bottom", linespacing=1.1)
    ax.set_title("(a) Scenario, top view", loc="left", color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # (b, c) approach field in the vertical plane through the pad (offset along x, pad frame)
    xs = np.linspace(-2.4, 2.4, 13)
    hs = np.linspace(0.1, 3.0, 9)
    first = None
    for idx, (age, title) in enumerate([(0.0, "(b) Approach field, pad in view"),
                                        (1.2, "(c) View lost for 1 s or more")]):
        ax = fig.add_subplot(gs[idx + 1], sharey=first)
        first = first or ax
        U = np.zeros((len(hs), len(xs)))
        V = np.zeros_like(U)
        for i, h in enumerate(hs):
            for j, x in enumerate(xs):
                v = desired_velocity([x, 0.0], h, vf, vision_age_s=age)
                U[i, j], V[i, j] = v[0], v[2]
        X, Hh = np.meshgrid(xs, hs)
        ax.quiver(X, Hh, U, V, color=PROPOSED if idx == 0 else C["sim"][1], angles="xy", scale_units="xy",
                  scale=3.2, width=0.007, headwidth=3.6, headlength=4.0, zorder=3)
        hh = np.linspace(0, 3.1, 50)
        rad = vf.alignment_radius_m + vf.alignment_radius_per_m * hh
        ax.fill_betweenx(hh, -rad, rad, color=C["know"][0], zorder=0)
        ax.plot(rad, hh, color=C["know"][1], lw=0.8, ls=(0, (4, 2)))
        ax.plot(-rad, hh, color=C["know"][1], lw=0.8, ls=(0, (4, 2)))
        ax.add_patch(Rectangle((-0.5, -0.12), 1.0, 0.12, fc=C["est"][1], ec="none"))
        if idx == 0:
            ax.text(0.0, 3.32, "descent cone " + r"$r = 0.25 + 0.5\,h$", ha="center", va="center",
                    fontsize=NOTE, color=C["know"][1], zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.2))
            ax.text(2.55, -0.02, "align, then sink at\n0.2 m/s to touchdown", ha="right", va="bottom",
                    fontsize=NOTE, color=INK, zorder=5, linespacing=1.1,
                    bbox=dict(fc="white", ec="none", pad=0.2))
        else:
            ax.axhline(vf.lost_view.search_height_m, color=C["sim"][1], lw=0.8, ls=(0, (2, 2)))
            ax.text(-2.55, vf.lost_view.search_height_m + 0.1, "search height 2.5 m", fontsize=NOTE,
                    color=INK, va="bottom", zorder=5, bbox=dict(fc="white", ec="none", pad=0.2))
            ax.add_patch(Rectangle((-0.75, 0.0), 1.5, 0.5, fc="none", ec=MUTED, lw=0.8, ls=(0, (2, 2))))
            ax.text(2.55, -0.02, "blind touchdown\nallowed (dashed box)", ha="right", va="bottom",
                    fontsize=NOTE, color=INK, zorder=5, linespacing=1.1,
                    bbox=dict(fc="white", ec="none", pad=0.2))
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-0.15, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xlabel("offset from the pad [m]", color=INK)
        if idx == 0:
            ax.set_ylabel("height above contact [m]", color=INK)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(title, loc="left", color=INK)
        style_axes(ax)
        ax.yaxis.grid(False)
    save(fig, "task_and_field")


# ----------------------------------------------------------------------------- 5 results
def _plot(build):
    """Run a result-plot builder under PLOT_RC (lettering of 8-12 pt at print size)."""
    import functools

    @functools.wraps(build)
    def run(*args, **kwargs):
        with plt.rc_context(PLOT_RC):
            return build(*args, **kwargs)
    return run


# One case of the closed-loop Isaac Sim-PX4 evaluation (paper Sec. 5.3), replayed on the local
# plant: isaac_run flew with trace=False, so no Isaac time series exists.  The case is the ORL
# success (Isaac and local reference both SUCCESS) whose Isaac policy flight time is closest to
# the median of the 140 ORL Isaac successes (6.77 s): training seed 59, budget 600,
# composition_id, environment seed 994503.
TRAJECTORY_CASE = {"training_seed": 59, "condition": "composition_id", "seed": 994503}
VIEW_LOST_S = 0.5     # the reward field's lost-view onset (task.reward.field.lost_view.after_s)


def _replay_case(case):
    """Fly ``case`` exactly as budget.isaac_local_reference flew it (same checkpoint, cell
    conditions, environment and action seeds, deterministic policy), recording the plant's
    world-frame drone and pad state after the reset and after every decision.  The replay must
    reproduce the stored local reference (status and decision count) or nothing is drawn."""
    from dataclasses import asdict

    import torch
    from ontology_rgat.current_semantics import budget as bud
    from ontology_rgat.current_semantics import cells
    from ontology_rgat.current_semantics import pilot
    from ontology_rgat.current_semantics.backend import make_local_backend
    from ontology_rgat.current_semantics.contracts import load_config

    torch.set_num_threads(1)            # the reference ran in single-threaded workers
    root = RESULTS / "spatial_current_semantics_v1"
    config = load_config(CONFIG)
    plan = bud.load_plan(bud.DEFAULT_PATH, config)
    arm = plan.proposed_arm
    key = f"{case['training_seed']}/{arm}"
    path = bud.checkpoint(root, key, max(plan.budgets))
    if not path.exists():
        raise SystemExit(f"no checkpoint {path}")
    model, _ = bud.load_weights(bud.arm_config(config, plan, arm), arm, path)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cell = asdict(plan.cell(case["condition"]))
    factors, dropout = cells.cell_conditions(config, cell, plan.dropout_start_s)
    reference = [json.loads(line) for line in
                 (bud.isaac_dir(root) / bud.LOCAL_REFERENCE).read_text().splitlines() if line]
    reference = next(r for r in reference if r["model"] == key
                     and r["condition"] == case["condition"] and r["seed"] == case["seed"])
    isaac = [json.loads(line) for line in
             (bud.isaac_dir(root) / "episodes.jsonl").read_text().splitlines() if line]
    isaac = next(r for r in isaac if r["model"] == key
                 and r["condition"] == case["condition"] and r["seed"] == case["seed"])

    class Recorder:                     # evaluator-side only: the policy never sees these rows
        def __init__(self, backend):
            self.backend, self.rows = backend, []

        def _snap(self, t):
            f = self.backend.flight
            self.rows.append({"t": float(t), "p": f.position.copy(), "v": f.velocity.copy(),
                              "pad": f.pad.copy(), "pad_v": f.pad_velocity.copy()})

        def reset(self, seed):
            out = self.backend.reset(seed)
            self._snap(0.0)
            return out

        def step(self, action, latent=None):
            step = self.backend.step(action, latent=latent)
            self._snap(step.elapsed_s)
            return step

    backend = make_local_backend(config, difficulty=1.0, randomization=None,
                                 scenario=cell["split"], factors=factors, dropout=dropout)
    recorder = Recorder(backend)
    try:
        episode = pilot.fly_policy(recorder, pilot._model(config, arm, state), config,
                                   seed=case["seed"], deterministic=True,
                                   generator=torch.Generator().manual_seed(
                                       case["seed"] + bud.ACTION_OFFSET),
                                   record=True, trace=True)
    finally:
        backend.close()
    if (episode["status"], episode["decisions"]) != (reference["status"], reference["decisions"]):
        raise SystemExit(f"replay {episode['status']}/{episode['decisions']} does not reproduce "
                         f"the local reference {reference['status']}/{reference['decisions']}")
    rows = recorder.rows
    for row, traced in zip(rows[1:], episode["trace"]):
        if not np.allclose(row["p"] - row["pad"], traced["truth_r"], atol=1e-9):
            raise SystemExit("recorded world state disagrees with the evaluator's truth")
    age = np.r_[0.0, [t["estimate"]["vision_age_s"] for t in episode["trace"]]]
    return rows, age, episode, isaac


@_plot
def trajectory():
    """One closed-loop ORL landing (TRAJECTORY_CASE) replayed on the calibrated local plant:
    (a) drone and pad horizontal paths, (b) relative position and velocity (own minus pad),
    (c) height above the pad and vertical relative velocity.  Simulator truth, plotted only."""
    rows, age, episode, isaac = _replay_case(TRAJECTORY_CASE)
    t = np.array([r["t"] for r in rows])
    p = np.array([r["p"] for r in rows])
    pad = np.array([r["pad"] for r in rows])
    rel = p - pad
    vel = np.array([r["v"] - r["pad_v"] for r in rows])
    blind = age >= VIEW_LOST_S
    xy_style = (("x", PROPOSED, "-"), ("y", INK, (0, (3.0, 1.6))))

    fig = plt.figure(figsize=(PLOT_W, 2.95), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.04)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 1.0])

    # (a) horizontal paths in the world frame, origin = pad centre at handover
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(pad[:, 0], pad[:, 1], color=MUTED, lw=1.2, ls=(0, (3.0, 1.6)), label="pad centre")
    ax.plot(p[:, 0], p[:, 1], color=PROPOSED, lw=1.4, label="drone")
    for k in range(0, len(t), 20):           # every 2 s: drone-pad pairs
        ax.plot([pad[k, 0], p[k, 0]], [pad[k, 1], p[k, 1]], color="#BDBDBD", lw=0.6, zorder=1)
        ax.plot(pad[k, 0], pad[k, 1], "o", ms=2.6, color=MUTED, mec="none", zorder=3)
        ax.plot(p[k, 0], p[k, 1], "o", ms=2.6, color=PROPOSED, mec="none", zorder=3)
    ax.add_patch(Circle(pad[-1, :2], 0.5, fill=False, ec=MUTED, lw=0.8, zorder=2))
    ax.plot(p[-1, 0], p[-1, 1], "x", ms=5.5, mew=1.3, color=PROPOSED, zorder=4,
            label="touchdown")
    ax.annotate("handover", xy=(p[0, 0], p[0, 1]), xytext=(p[0, 0] - 0.2, p[0, 1] - 0.95),
                fontsize=P_NOTE, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))
    ax.annotate("pad, r = 0.5 m", xy=(pad[-1, 0] + 0.36, pad[-1, 1] + 0.36),
                xytext=(pad[-1, 0] + 0.9, pad[-1, 1] + 1.0), fontsize=P_NOTE, color=MUTED,
                va="bottom", arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("east [m]", color=INK)
    ax.set_ylabel("north [m]", color=INK)
    ax.set_title("(a) Horizontal paths", loc="left", color=INK)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#BDBDBD", lw=0.6, marker="o", ms=2.6,
                              mfc=MUTED, mec="none", label="pair every 2 s"))
    ax.legend(handles=handles, loc="lower left", frameon=True, facecolor="white",
              edgecolor="none", framealpha=1.0, handlelength=1.8, borderaxespad=0.2)
    style_axes(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.5)

    def shade(a):
        edges = np.flatnonzero(np.diff(np.r_[0, blind.astype(int), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            a.axvspan(t[start], t[min(stop, len(t) - 1)], color="#EDEDED", lw=0, zorder=0)
        a.axhline(0.0, color="#8C8C8C", lw=0.6, zorder=1)
        a.set_xlim(0.0, t[-1])

    # (b) relative position and velocity, horizontal axes (own minus pad)
    top = fig.add_subplot(gs[0, 1])
    bottom = fig.add_subplot(gs[1, 1], sharex=top)
    for i, (name, color, ls) in enumerate(xy_style):
        top.plot(t, rel[:, i], color=color, ls=ls, lw=1.1, label=name)
        bottom.plot(t, vel[:, i], color=color, ls=ls, lw=1.1, label=name)
    for a in (top, bottom):
        shade(a)
        style_axes(a)
    top.set_ylabel("rel. position [m]", color=INK)
    bottom.set_ylabel("rel. velocity [m/s]", color=INK)
    bottom.set_xlabel("time since handover [s]", color=INK)
    top.tick_params(labelbottom=False)
    top.set_title("(b) Relative state, x and y", loc="left", color=INK)
    top.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", edgecolor="none",
               framealpha=1.0, handlelength=1.8, borderaxespad=0.2, columnspacing=0.8)

    # (c) height above the pad and vertical relative velocity
    contact_h = float(load_contact_height())
    top = fig.add_subplot(gs[0, 2], sharex=bottom)
    bottom2 = fig.add_subplot(gs[1, 2], sharex=bottom)
    top.plot(t, rel[:, 2], color=PROPOSED, lw=1.2)
    top.axhline(contact_h, color=INK, lw=0.7, ls=(0, (3, 2)))
    top.text(t[-1] * 0.03, contact_h + 0.12, f"contact height {contact_h:.3f} m",
             fontsize=P_NOTE, color=MUTED, va="bottom")
    bottom2.plot(t, vel[:, 2], color=PROPOSED, lw=1.2)
    bottom2.axhline(-0.30, color=INK, lw=0.7, ls=(0, (3, 2)))
    for a in (top, bottom2):
        shade(a)
        style_axes(a)
    top.set_ylim(bottom=-0.15)
    bottom2.set_ylim(min(-0.55, float(vel[:, 2].min()) - 0.05), max(0.15, float(vel[:, 2].max()) + 0.05))
    top.set_ylabel("height above pad [m]", color=INK)
    bottom2.set_ylabel("vertical velocity [m/s]", color=INK)
    bottom2.set_xlabel("time since handover [s]", color=INK)
    top.tick_params(labelbottom=False)
    top.set_title("(c) Vertical state", loc="left", color=INK)
    handles = [plt.Line2D([], [], color=INK, lw=0.7, ls=(0, (3, 2)),
                          label="limit at contact, −0.30 m/s")]
    if blind.any():
        handles.append(Rectangle((0, 0), 1, 1, fc="#EDEDED", ec="none",
                                 label=f"pad unseen ≥ {VIEW_LOST_S:g} s"))
    bottom2.legend(handles=handles, loc="upper right", frameon=False, handlelength=1.6,
                   borderaxespad=0.2, labelspacing=0.3)
    save(fig, "closed_loop_trajectory", min_pt=PLOT_MIN_PT)
    blind_s = float(np.sum(np.diff(t)[blind[1:]]))
    s = episode["scenario"]
    print(f"closed_loop_trajectory: ORL ({ORL_VERSION}) training seed "
          f"{TRAJECTORY_CASE['training_seed']}, budget 600, {TRAJECTORY_CASE['condition']} "
          f"(bearing bin {s['bearing_bin']}, heading bin {s['heading_bin']}), environment seed "
          f"{TRAJECTORY_CASE['seed']}; local replay {episode['status']} after "
          f"{episode['elapsed_s']:.2f} s ({episode['decisions']} decisions), max tilt "
          f"{episode['max_tilt_deg']:.1f} deg, touchdown v_z "
          f"{episode['terminal_truth']['relative_velocity_own_minus_pad_m_s'][2]:+.3f} m/s; "
          f"Isaac flight of the same case {isaac['status']} after {isaac['elapsed_s']:.2f} s "
          f"({isaac['decisions']} decisions); pad unseen >= {VIEW_LOST_S:g} s for "
          f"{blind_s:.2f} s")


def load_contact_height():
    """The local plant's contact height (config local_plant.contact_height_m)."""
    from ontology_rgat.current_semantics.contracts import load_config
    return load_config(CONFIG).local_plant.contact_height_m


FIGURES = {"architecture": architecture, "overall": overall, "field": field,
           "trajectory": trajectory}


def main():
    global OUT, ARGS_FORMATS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["all", *FIGURES])
    parser.add_argument("--formats", default=",".join(FORMATS),
                        help="comma list of png, svg, tif, eps, pdf (default png,svg)")
    parser.add_argument("--out", type=Path, default=OUT, help="output folder (default results/figures)")
    args = parser.parse_args()
    ARGS_FORMATS = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    unknown = set(ARGS_FORMATS) - {"png", "svg", "tif", "eps", "pdf"}
    if unknown:
        parser.error(f"unknown format(s): {', '.join(sorted(unknown))}")
    OUT = args.out
    if args.what == "all":
        for name, f in FIGURES.items():
            try:
                f()
            except (FileNotFoundError, SystemExit) as exc:   # a result not produced yet
                if name != "trajectory":
                    raise
                print(f"skipped {name}: {exc}")
    else:
        FIGURES[args.what]()


if __name__ == "__main__":
    main()
