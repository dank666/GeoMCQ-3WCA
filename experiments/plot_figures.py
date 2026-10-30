"""Figures 3-5: multiple-choice errors, deferral, and accuracy by domain (test results).

  python experiments/plot_figures.py

The figures are drawn at the text width of the manuscript (390 pt), so their type appears at its nominal size.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import ROOT, load_split  # noqa: E402

OUT = ROOT / "results" / "figures"
WIDTH = 390 / 72.27  # text width of the manuscript in inches
MODELS = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B"), ("GLM-4-9B-0414", "GLM-4-9B"), ("Qwen3.5-122B-A10B", "Qwen3.5-122B")]
DOMAINS = [("CL", "Climatology"), ("HY", "Hydrology"), ("TG", "Topography"), ("SV", r"Soil \& vegetation"),
           ("HEG", r"Human \& economic"), ("GPP", r"Processes \& principles")]

# Proposed framework in the accent of Figure 1, the baselines in two greys.
ACCENT, GREY, DARK_GREY = "#4256b8", "#aeb6c1", "#5f6a78"
INK, MUTED, RULE, GRID = "#1d2533", "#4f5966", "#8a94a1", "#e5e9ee"
METHODS = [("direct", "direct answering", GREY, "o"),
           ("k2", r"direct, told $\geq$2 options are correct", DARK_GREY, "D"),
           ("ours", "proposed framework", ACCENT, "o")]

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.5, "axes.edgecolor": RULE, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "axes.titlelocation": "left", "axes.titlepad": 5, "axes.labelcolor": MUTED,
})


def rows(path):
    return {r["uid"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def load(split, model, method):
    if method == "ours":
        return rows(ROOT / "results" / "twc" / f"{split}_{model}_full.jsonl")
    return rows(ROOT / "results" / "direct" / f"{split}_{model}{'_k2' if method == 'k2' else ''}.jsonl")


def answer(r):
    return set(r.get("answer", r.get("pred", [])))


def style(ax, grid="x"):
    for side in ("top", "right", "left") if grid == "x" else ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def marker(color, shape, size=4.5, line=False):
    return dict(color=color, marker=shape, markersize=size, markeredgecolor="white", markeredgewidth=0.8,
                linestyle="-" if line else "", linewidth=1.4)


def legend(fig, methods, y=1.0, x=0.5):
    handles = [Line2D([], [], label=label, **marker(color, shape)) for _, label, color, shape in methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(x, y),
               handletextpad=0.2, columnspacing=1.6, labelcolor=INK)


def errors():
    """Figure 3: share of multiple-choice questions whose answer misses a correct option or adds a wrong one."""
    kinds = [("missing", "Misses a correct option"), ("extra", "Adds a wrong option")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.25), sharey=True)
    step, gap = 1.0, 0.9
    ys = {}
    for i, (model, _) in enumerate(MODELS):
        for j, (method, *_) in enumerate(METHODS):
            ys[(model, method)] = i * (3 * step + gap) + j * step
    for ax, (kind, title) in zip(axes, kinds):
        for model, _ in MODELS:
            for method, _, color, shape in METHODS:
                rs = [r for r in load("test", model, method).values() if r["qtype"] == "multiple"]
                share = 100 * sum(
                    (answer(r) < set(r["gold"])) if kind == "missing" else (answer(r) > set(r["gold"])) for r in rs
                ) / len(rs)
                y = ys[(model, method)]
                ax.plot([0, share], [y, y], color=color, linewidth=1.3, solid_capstyle="round", zorder=2)
                ax.plot([share], [y], zorder=3, **marker(color, shape, size=5))
                ax.text(share + 0.9, y, f"{share:.1f}", va="center", ha="left", fontsize=7, color=MUTED)
        ax.set_title(title, color=INK)
        ax.set_xlim(0, 32)
        ax.set_xticks([0, 10, 20, 30])
        style(ax)
    centers = [ys[(m, "k2")] for m, _ in MODELS]
    axes[0].set_yticks(centers, [label for _, label in MODELS], color=INK, fontsize=8)
    axes[0].invert_yaxis()
    fig.supxlabel(r"share of the 480 multiple-choice test questions (\%)", fontsize=8, color=MUTED, y=0.02)
    legend(fig, METHODS, y=1.01, x=0.55)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.8, bottom=0.17, wspace=0.12)
    fig.savefig(OUT / "fig_answers.pdf")
    plt.close(fig)


def deferral():
    """Figure 4: accuracy on the questions decided by the three-way regions and on the deferred ones."""
    methods = [m for m in METHODS if m[0] != "k2"]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.3), sharex=True, sharey=True)
    for r, (split, name) in enumerate((("test", "GeoMCQ-1200"), ("gaokao", "GaoKao-Geo"))):
        for c, (model, label) in enumerate(MODELS):
            ax = axes[r, c]
            stage = {u: x["stage"] for u, x in load(split, model, "ours").items()}
            for method, _, color, shape in methods:
                rs = load(split, model, method)
                ys = []
                for st in ("context", "deferred"):
                    uids = [u for u in stage if stage[u] == st]
                    ys.append(100 * sum(rs[u]["correct"] for u in uids) / len(uids))
                ax.plot([0, 1], ys, zorder=3 if method == "ours" else 2, **marker(color, shape, size=5, line=True))
            share = 100 * sum(s == "deferred" for s in stage.values()) / len(stage)
            ax.text(0.97, 0.06, f"{share:.0f}\\% deferred", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color=MUTED)
            ax.set_xlim(-0.35, 1.35)
            ax.set_ylim(50, 100)
            ax.set_yticks([60, 70, 80, 90, 100])
            style(ax, grid="y")
            if r == 0:
                ax.set_title(label, color=INK)
            if c == 0:
                ax.set_ylabel(f"{name}\naccuracy (\\%)")
    axes[1, 0].set_xticks([0, 1], ["decided", "deferred"])
    legend(fig, methods, y=1.0)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.08, wspace=0.12, hspace=0.32)
    fig.savefig(OUT / "fig_deferral.pdf")
    plt.close(fig)


def domains():
    """Figure 5: accuracy by domain, direct answering against the framework."""
    methods = [m for m in METHODS if m[0] != "k2"]
    qs = {q.uid: q.domain for q in load_split("test")}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.25), sharey=True)
    for ax, (model, label) in zip(axes, MODELS):
        o, d = load("test", model, "ours"), load("test", model, "direct")
        for i, (dom, _) in enumerate(DOMAINS):
            uids = [u for u in qs if qs[u] == dom]
            a = 100 * sum(d[u]["correct"] for u in uids) / len(uids)
            b = 100 * sum(o[u]["correct"] for u in uids) / len(uids)
            ax.plot([a, b], [i, i], color="#d5dbe3", linewidth=2.2, solid_capstyle="round", zorder=1)
            ax.plot([a], [i], zorder=2, **marker(GREY, "o", size=5))
            ax.plot([b], [i], zorder=3, **marker(ACCENT, "o", size=5))
        ax.set_title(label, color=INK)
        ax.set_xlim(59, 101)
        ax.set_xticks([60, 80, 100])
        ax.set_xticks([70, 90], minor=True)
        ax.tick_params(axis="x", which="minor", length=0)
        ax.grid(axis="x", which="minor", color=GRID, linewidth=0.6)
        style(ax)
    axes[0].set_yticks(range(len(DOMAINS)), [name for _, name in DOMAINS], color=INK)
    axes[0].set_ylim(len(DOMAINS) - 0.5, -0.5)
    fig.supxlabel(r"accuracy on the 160 test questions of each domain (\%)", fontsize=8, color=MUTED, y=0.02)
    legend(fig, methods, y=1.01, x=0.6)
    fig.subplots_adjust(left=0.235, right=0.975, top=0.8, bottom=0.17, wspace=0.2)
    fig.savefig(OUT / "fig_domains.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    errors()
    deferral()
    domains()
    print(OUT)


if __name__ == "__main__":
    main()
