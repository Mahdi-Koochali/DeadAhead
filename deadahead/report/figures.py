"""Figures for the supplement: the warning curves of Figure 2, the overview of Table 1, the figures of S1, S2 and S4 and
the post hoc figures of S8.

Each method keeps one line style and one marker in every figure, and position or marker carries every other identity,
so the figures read in greyscale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from deadahead import paths  # noqa: E402
from deadahead.evaluation.rows import GROUPS  # noqa: E402

INK, MUTED, GRID, CHANCE = "black", "#595959", "#e6e6e6", "#8c8c8c"
DASHED, DOTTED = (0, (4, 1.6)), (0, (1, 1.3))
IMAGE_LINE = {"linestyle": DASHED, "marker": "^", "markerfacecolor": INK}
LINE_STYLES = {
    "deadahead": {"linestyle": "-", "marker": "o", "markerfacecolor": INK},
    "dinov2_handcrafted": IMAGE_LINE, "dinov2_handcrafted|refit": IMAGE_LINE, "dinov2_handcrafted_survival": IMAGE_LINE,
    "elapsed_time": {"linestyle": DOTTED, "marker": "s", "markerfacecolor": "white"},
}
IMAGE_NAME = "DINOv2 + handcrafted"
#: DINOv2 + handcrafted is post hoc in Fig. 2 and under DeadAhead's survival head in the window figure of S8
LABELS = {"deadahead": "DeadAhead", "dinov2_handcrafted": f"{IMAGE_NAME} (post hoc)",
          "dinov2_handcrafted|refit": IMAGE_NAME, "dinov2_handcrafted_survival": f"{IMAGE_NAME}, same head",
          "elapsed_time": "elapsed time", "chance": "chance"}
BANDS = {"deadahead": {"facecolor": INK, "alpha": 0.16, "linewidth": 0},
         "dinov2_handcrafted": {"facecolor": "none", "edgecolor": "#7f7f7f", "hatch": "/////", "linewidth": 0},
         "elapsed_time": {"facecolor": "none", "edgecolor": "#595959", "hatch": "...", "linewidth": 0}}
ROUTES = {"LYTIC": "Lytic death", "APOPTOSIS": "Apoptosis"}
#: the columns of Fig. 2's table: (method, column prefix)
SPREAD_SERIES = (("deadahead", "survival"), ("dinov2_handcrafted", "image"), ("elapsed_time", "clock"))
SPREAD_ROUTES = (("lytic", "(a) lytic death"), ("apop", "(b) apoptosis"))
#: Table 1's rows, top to bottom, without its chance row: (key in the reproduce output, printed name)
TABLE1_ROWS = (("elapsed_time", "Elapsed time"), ("deadahead", "DeadAhead (ours)"),
               ("videomae_finetuned", "VideoMAE (fine-tuned; ours)"), ("deephit", "DeepHit"),
               ("dinov2_handcrafted|refit", IMAGE_NAME),
               ("dinov2_finetuned_handcrafted|refit", "Fine-tuned DINOv2 + handcrafted"),
               ("handcrafted|refit", "Handcrafted"), ("dinov2|refit", "DINOv2"))
#: the test-time changes of S4: (key in the reproduce output, printed name)
TEST_TIME_ROWS = (("hidden_cell", "Cell hidden"), ("shuffled_frames", "Frames shuffled"),
                  ("repeated_last_frame", "Last frame repeated"))
HORIZONS = (30, 60, 120, 240)
ROUTE_LINE = {"LYTIC": {"linestyle": "-", "marker": "o", "markerfacecolor": INK},
              "APOPTOSIS": {"linestyle": DASHED, "marker": "^", "markerfacecolor": "white"}}
#: the files of deadahead plot-figures
TABLE1_FIG = "figT1_auprc_by_method"
HORIZONS_FIG = "figS1_horizons"
GROUPS_FIG = "figS2_group_differences"
TEST_TIME_FIG = "figS4_test_time"
ALARM_SHARE_FIG = "figS8c_alarm_share"
GROUP_MATCH_CSV = "s8_group_match.csv"
ROW_WEIGHTS_CSV = "method_row_weights.csv"


def _line(m: str, size: float = 3.4, width: float = 1.1) -> dict:
    return {"color": INK, "linewidth": width, "markersize": size, "markeredgecolor": INK, "markeredgewidth": 0.8,
            "label": LABELS[m], **LINE_STYLES[m]}


def _axes_style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.5, width=0.5, length=2.5)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Software": None})
    plt.close(fig)


def plot_warnings(table: pd.DataFrame, out: Path) -> None:
    """Fig. 2: share of deaths warned X min ahead per route, means over the groups with their interquartile bands."""
    with plt.rc_context({"hatch.linewidth": 0.5}):
        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.5), dpi=300, sharey=True)
        keys = []
        for ax, (route, title) in zip(axes, SPREAD_ROUTES):
            for m, prefix in SPREAD_SERIES:
                col = f"{prefix}_{route}_r02"
                ax.fill_between(table["X"], table[f"{col}_q1"], table[f"{col}_q3"], zorder=1, **BANDS[m])
            for m, prefix in SPREAD_SERIES:
                ax.plot(table["X"], table[f"{prefix}_{route}_r02"], zorder=3, **_line(m))
            ax.set_xlim(20, 250)
            ax.set_ylim(0, 100)
            ax.set_xticks([60, 120, 180, 240])
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.text(0.97, 0.95, title, transform=ax.transAxes, ha="right", va="top", fontsize=7)
            _axes_style(ax)
        for m, _ in SPREAD_SERIES:
            keys.append((Patch(**BANDS[m]), Line2D([], [], **_line(m))))
        axes[0].set_ylabel("deaths warned (%)", fontsize=7)
        fig.subplots_adjust(left=0.1, right=0.985, top=0.87, bottom=0.2, wspace=0.08)
        fig.text(0.54, 0.035, "warning before death, X (min)", ha="center", fontsize=7)
        fig.legend(keys, [LABELS[m] for m, _ in SPREAD_SERIES], handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
                   loc="upper center", ncol=3, frameon=False, fontsize=7, handlelength=2.6)
        _save(fig, out)


def plot_gap_window(table: pd.DataFrame, out: Path) -> None:
    """The post hoc window analysis: macro AUPRC against the gap, one panel per route and horizon, chance beside."""
    horizons = sorted(table["horizon_min"].unique())
    fig, axes = plt.subplots(len(ROUTES), len(horizons), figsize=(5.6, 4.3), dpi=300, sharey="row", squeeze=False)
    methods = list(dict.fromkeys(table["method"]))
    for i, (route, name) in enumerate(ROUTES.items()):
        for j, H in enumerate(horizons):
            ax = axes[i][j]
            d = table[(table["route"] == route) & (table["horizon_min"] == H)].sort_values("gap_min")
            first = d[d["method"] == methods[0]]
            ax.plot(first["gap_min"], first["chance"], color=CHANCE, linewidth=1.1, label=LABELS["chance"], zorder=2)
            for m in methods:
                e = d[d["method"] == m]
                ax.plot(e["gap_min"], e["auprc"], zorder=3, **_line(m))
            ax.set_xticks(sorted(d["gap_min"].unique()))
            ax.set_xlim(-8, 128)
            ax.set_title(f"{name}: deaths g to {H} min ahead", fontsize=7, loc="left")
            _axes_style(ax)
        axes[i][0].set_ylim(0, 1.05 * table.loc[table["route"] == route, "auprc"].max())
        axes[i][0].set_ylabel("macro AUPRC", fontsize=7)
    keys, labels = axes[0][0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.1, right=0.985, top=0.855, bottom=0.095, wspace=0.08, hspace=0.34)
    fig.text(0.54, 0.018, "gap g: forecasts whose cell died within g min are left out (min)", ha="center", fontsize=7)
    fig.text(0.012, 0.99, "Post hoc analysis", fontsize=7, color=MUTED, va="top")
    fig.legend(keys[1:] + keys[:1], labels[1:] + labels[:1], loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=4,
               frameon=False, fontsize=7, handlelength=2.6)
    _save(fig, out)


def plot_alarm_rate(table: pd.DataFrame, out: Path) -> None:
    """The post hoc alarm-rate analysis: deaths warned 60 min ahead against the achieved pooled alarm rate, per
    route; the two sides of the lytic matched comparison are ringed, with both achieved rates printed."""
    fig, axes = plt.subplots(1, len(ROUTES), figsize=(5.6, 3.0), dpi=300, sharey=True)
    methods = list(dict.fromkeys(table["method"]))
    for ax, (route, name) in zip(axes, ROUTES.items()):
        d = table[table["route"] == route]
        for m in methods:
            grid = d[(d["method"] == m) & (d["set"] == "grid")].sort_values("target_rate")
            sweep = d[(d["method"] == m) & (d["set"] == "sweep")].sort_values("target_rate")
            line = _line(m)
            if len(grid):                        # the fine grid as the line, the 16 targets as its markers
                ax.plot(grid["achieved_pooled"], grid["warned_last_60_min_pct"], color=INK, linewidth=1.1,
                        linestyle=line["linestyle"], zorder=2)
                ax.plot(sweep["achieved_pooled"], sweep["warned_last_60_min_pct"], zorder=3,
                        **{**line, "linestyle": "none"})
            else:
                ax.plot(sweep["achieved_pooled"], sweep["warned_last_60_min_pct"], zorder=3, **line)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(name, fontsize=7, loc="left")
        _axes_style(ax)
    mt = table[table["set"] == "matched"]
    ax = axes[0]
    for _, r in mt.iterrows():
        ax.plot([r["achieved_pooled"]], [r["warned_last_60_min_pct"]], linestyle="none", marker="o", markersize=8.5,
                markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.7, zorder=4)
    side = {r["method"]: r for _, r in mt.iterrows()}
    if len(side) == 2:
        a, b = (side[m] for m in methods[:2])
        ax.axvline(b["achieved_pooled"], color=CHANCE, linewidth=0.6, zorder=1)
        ax.text(b["achieved_pooled"] + 0.03, 15.5, f"matched rates (post hoc):\n{LABELS[methods[0]]}: "
                f"{a['achieved_pooled']:.3f}, {a['warned_last_60_min_pct']:.1f}%\n{LABELS[methods[1]]}: "
                f"{b['achieved_pooled']:.3f}, {b['warned_last_60_min_pct']:.1f}%", fontsize=5.8, color=INK, va="top",
                linespacing=1.25)
    axes[0].set_ylabel("deaths warned 60 min ahead (%)", fontsize=7)
    keys = [Line2D([], [], **_line(m)) for m in methods]
    fig.subplots_adjust(left=0.1, right=0.985, top=0.82, bottom=0.14, wspace=0.08)
    fig.text(0.54, 0.025, "achieved alarms per event-free reference cell-hour (pooled over the held-out groups)",
             ha="center", fontsize=7)
    fig.text(0.012, 0.985, "Post hoc analysis", fontsize=7, color=MUTED, va="top")
    fig.legend(keys, [LABELS[m] for m in methods], loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3,
               frameon=False, fontsize=7, handlelength=2.6)
    _save(fig, out)


def _group_name(group: str) -> str:
    return group.replace("_", " ")


def _macro_rows(route: str, name: str, block: dict) -> list[dict]:
    """One row per held-out group and one macro row, from a block with ``per_group`` and ``macro``."""
    rows = [{"route": route, "name": name, "group": _group_name(g), "auprc": v} for g, v in block["per_group"].items()]
    return rows + [{"route": route, "name": name, "group": "macro", "auprc": block["macro"]}]


def table1_table(res: dict) -> pd.DataFrame:
    """Figure T1's values: macro AUPRC at 30 min per method and route, with each held-out group's value and chance."""
    methods, rows = res["table1"]["cohort"]["methods"], []
    for route in ROUTES:
        chance = res["table1"]["chance"][route]["nine_folds"]
        rows.append({"route": route, "name": "Chance", "group": "macro", "auprc": chance})
        for key, name in TABLE1_ROWS:
            rows += _macro_rows(route, name, methods[key][route]["nine_folds"])
    return pd.DataFrame(rows)


def test_time_table(res: dict) -> pd.DataFrame:
    """Figure S4's values: the macro AUPRC of VideoMAE (fine-tuned), intact and under each test-time change."""
    rows = []
    for route in ROUTES:
        r = res["decomposition"]["test_time"]["readings"][route]
        chance = res["table1"]["chance"][route]["nine_folds"]
        rows.append({"route": route, "name": "Chance", "group": "macro", "auprc": chance})
        rows += _macro_rows(route, "Intact", r["intact"]["nine_folds"])
        for key, name in TEST_TIME_ROWS:
            rows += _macro_rows(route, name, r["changed"][key]["auprc"]["nine_folds"])
    return pd.DataFrame(rows)


def horizons_table(res: dict) -> pd.DataFrame:
    """Figure S1's values: DeadAhead's macro prevalence, AUPRC, concordance and IPA at each horizon."""
    alarms, survival = res["horizons"]["models"]["deadahead"], res["horizons"]["survival_estimators"]["deadahead"]
    rows = []
    for route in ROUTES:
        for h in HORIZONS:
            a, s = alarms[route][str(h)]["0.2_per_cell_hour"], survival[route][str(h)]
            rows.append({"route": route, "horizon_min": h, "prevalence": a["prevalence"]["nine_folds"]["macro"],
                         "auprc": a["auprc"]["nine_folds"]["macro"],
                         "concordance": s["c_no_same_track"]["nine_folds"]["macro"],
                         "ipa": s["ipa"]["nine_folds"]["macro"]})
    return pd.DataFrame(rows)


def group_differences_table(res: dict) -> pd.DataFrame:
    """Figure S2's values: DeadAhead minus DINOv2 + handcrafted (refit), macro AUPRC at 30 min, per held-out group."""
    rows = []
    for route in ROUTES:
        c = res["table1"]["cohort"]["contrasts"][f"deadahead - dinov2_handcrafted|refit|{route}"]["nine_folds"]
        for g in GROUPS:
            rows.append({"route": route, "group": _group_name(g), "difference": c["per_group_difference"][g]})
        rows.append({"route": route, "group": "macro", "difference": c["macro_difference"]})
    return pd.DataFrame(rows)


def _strip(ax, d: pd.DataFrame, names: list, chance: float, xmax: float) -> None:
    """One row per name: a dot per held-out group, a tick at the macro, the macro's value at the right edge."""
    groups = sorted(set(d["group"]) - {"macro"})
    offsets = {g: -0.27 + 0.54 * i / max(len(groups) - 1, 1) for i, g in enumerate(groups)}
    ax.axvline(chance, color=CHANCE, linewidth=1.1, zorder=1)
    for i, name in enumerate(names):
        e = d[d["name"] == name]
        dots = e[e["group"] != "macro"]
        ax.plot(dots["auprc"], [i + offsets[g] for g in dots["group"]], linestyle="none", marker="o", markersize=2.6,
                markerfacecolor="white", markeredgecolor=MUTED, markeredgewidth=0.6, zorder=2)
        macro = float(e.loc[e["group"] == "macro", "auprc"].iloc[0])
        ax.plot([macro, macro], [i - 0.36, i + 0.36], color=INK, linewidth=1.8, solid_capstyle="butt", zorder=3)
        ax.text(1.02, i, f"{macro:.3f}", transform=ax.get_yaxis_transform(), fontsize=6.5, va="center", color=INK)
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks(range(len(names)))
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.5, width=0.5, length=2.5)
    ax.tick_params(axis="y", length=0)


def _plot_strips(table: pd.DataFrame, names: list, out: Path, height: float, emphasis: str | None = None) -> None:
    """Both routes side by side, one row per name, chance as a grey line printed above each panel."""
    fig, axes = plt.subplots(1, len(ROUTES), figsize=(6.4, height), dpi=300, sharey=True)
    dots = table[table["group"] != "macro"]["auprc"]
    xmax = round(float(dots.max()) * 1.08 + 0.05, 1)
    for ax, (route, title) in zip(axes, ROUTES.items()):
        d = table[table["route"] == route]
        chance = float(d.loc[d["name"] == "Chance", "auprc"].iloc[0])
        _strip(ax, d, names, chance, xmax)
        ax.set_title(f"{title} (chance {chance:.3f})", fontsize=7, loc="left")
    axes[0].set_yticklabels(names, fontsize=6.5)
    if emphasis is not None:
        for label in axes[0].get_yticklabels():
            label.set_fontweight("bold" if label.get_text() == emphasis else "normal")
    keys = [Line2D([], [], linestyle="none", marker="o", markersize=2.6, markerfacecolor="white", markeredgecolor=MUTED,
                   markeredgewidth=0.6),
            Line2D([], [], color=INK, linewidth=1.8, solid_capstyle="butt"),
            Line2D([], [], color=CHANCE, linewidth=1.1)]
    labels = ["one held-out group", "mean over the groups (value at right)", "chance"]
    fig.subplots_adjust(left=0.265, right=0.95, top=1 - 0.44 / height, bottom=0.36 / height, wspace=0.2)
    fig.text(0.6, 0.015, "AUPRC at 30 min", ha="center", fontsize=7)
    fig.legend(keys, labels, loc="upper center", bbox_to_anchor=(0.6, 1.0), ncol=3, frameon=False, fontsize=6.5,
               handlelength=1.8)
    _save(fig, out)


def plot_table1(table: pd.DataFrame, out: Path) -> None:
    """Figure T1: Table 1 at a glance, macro AUPRC by method for both routes with one dot per held-out group."""
    _plot_strips(table, [name for _, name in TABLE1_ROWS], out, 3.0, emphasis="DeadAhead (ours)")


def plot_test_time(table: pd.DataFrame, out: Path) -> None:
    """Figure S4: the macro AUPRC of VideoMAE (fine-tuned), intact and under each test-time change."""
    _plot_strips(table, ["Intact", *[name for _, name in TEST_TIME_ROWS]], out, 1.9)


def plot_horizons(table: pd.DataFrame, out: Path) -> None:
    """Figure S1: DeadAhead's AUPRC, concordance and IPA against the horizon for both routes."""
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.45), dpi=300)
    panels = (("auprc", "(a) AUPRC"), ("concordance", "(b) concordance"), ("ipa", "(c) IPA"))
    for ax, (col, title) in zip(axes, panels):
        for route, name in ROUTES.items():
            d = table[table["route"] == route].sort_values("horizon_min")
            if col == "auprc":                   # chance of AUPRC: the prevalence, one thin line per route
                ax.plot(d["horizon_min"], d["prevalence"], color=CHANCE, linewidth=1.0,
                        linestyle=ROUTE_LINE[route]["linestyle"], zorder=2)
            ax.plot(d["horizon_min"], d[col], color=INK, linewidth=1.1, markersize=3.4, markeredgecolor=INK,
                    markeredgewidth=0.8, label=name, zorder=3, **ROUTE_LINE[route])
        ax.set_xscale("log", base=2)
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([str(h) for h in HORIZONS])
        ax.minorticks_off()
        ax.set_xlim(24, 300)
        ax.set_title(title, fontsize=7, loc="left")
        _axes_style(ax)
    top = float(table["auprc"].max())
    axes[0].set_ylim(0, round(1.15 * top + 0.05, 1))
    axes[0].text(0.03, 0.2 * top, "chance: the prevalence", transform=axes[0].get_yaxis_transform(), fontsize=6,
                 color=MUTED, va="bottom")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].text(0.03, 0.51, "0.5: chance", transform=axes[1].get_yaxis_transform(), fontsize=6, color=MUTED,
                 va="bottom")
    low, high = float(table["ipa"].min()), float(table["ipa"].max())
    axes[2].axhline(0, color=CHANCE, linewidth=1.0, zorder=1)
    axes[2].set_ylim(min(-0.1, 1.2 * low), max(0.1, 1.5 * high))
    axes[2].text(0.98, 0.06, "0: the group's own incidence", transform=axes[2].get_yaxis_transform(), fontsize=6,
                 color=MUTED, va="bottom", ha="right")
    keys, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.17, wspace=0.27)
    fig.text(0.53, 0.025, "horizon h (min)", ha="center", fontsize=7)
    fig.legend(keys, labels, loc="upper center", ncol=2, frameon=False, fontsize=7, handlelength=2.6)
    _save(fig, out)


def plot_group_differences(table: pd.DataFrame, out: Path) -> None:
    """Figure S2: DeadAhead minus DINOv2 + handcrafted per held-out group, a stem from zero to each group's value."""
    lines = ("PNT1A", "DU145", "LNCaP")          # Table S2a's order
    names = [_group_name(g) for g in sorted(GROUPS, key=lambda g: (lines.index(g.split("_")[0]), g))]
    fig, axes = plt.subplots(1, len(ROUTES), figsize=(5.2, 2.9), dpi=300, sharey=True, sharex=True)
    limit = max(0.2, 1.1 * float(table["difference"].abs().max()))
    for ax, (route, title) in zip(axes, ROUTES.items()):
        d = table[table["route"] == route].set_index("group")
        ax.axvline(0, color=INK, linewidth=0.7, zorder=2)
        macro = float(d.loc["macro", "difference"])
        ax.axvline(macro, color=CHANCE, linewidth=1.1, zorder=1)
        for i, name in enumerate(names):
            v = float(d.loc[name, "difference"])
            ax.plot([0, v], [i, i], color=INK, linewidth=0.8, zorder=3)
            ax.plot([v], [i], color=INK, zorder=4, markersize=3.6, markeredgecolor=INK, markeredgewidth=0.8,
                    linestyle="none", marker="o", markerfacecolor=INK if v >= 0 else "white")
        ax.set_title(f"{title} (mean {macro:+.3f})".replace("-", "\u2212"), fontsize=7, loc="left")
        ax.set_xlim(-limit, limit)
        ax.set_xticks([-0.1, 0, 0.1])
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_yticks(range(len(names)))
        ax.grid(True, axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.5, width=0.5, length=2.5)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticklabels(names, fontsize=6.5)
    stem = {"color": INK, "linewidth": 0.8, "marker": "o", "markersize": 3.6, "markeredgecolor": INK}
    keys = [Line2D([], [], markerfacecolor=INK, **stem), Line2D([], [], markerfacecolor="white", **stem),
            Line2D([], [], color=CHANCE, linewidth=1.1)]
    fig.subplots_adjust(left=0.12, right=0.985, top=0.83, bottom=0.14, wspace=0.08)
    fig.text(0.55, 0.02, "DeadAhead minus DINOv2 + handcrafted, AUPRC at 30 min", ha="center", fontsize=7)
    fig.legend(keys, ["DeadAhead ahead", "DeadAhead behind", "mean over the groups"], loc="upper center", ncol=3,
               frameon=False, fontsize=6.5, handlelength=2.2)
    _save(fig, out)


def plot_alarm_share(table: pd.DataFrame, out: Path) -> None:
    """The post hoc alarm share: share of forecasts with the alarm on per 30-min bin before death, by route, with the
    share of reference forecasts that raise an alarm."""
    fig, axes = plt.subplots(1, len(ROUTES), figsize=(5.2, 2.5), dpi=300, sharey=True)
    for ax, (route, title) in zip(axes, ROUTES.items()):
        d = table[table["route"] == route].sort_values("bin_end_min")
        reference = float(d["reference_alarm_share_pct"].iloc[0])
        ax.axhline(reference, color=CHANCE, linewidth=1.1, zorder=1)
        ax.text(238, reference + 2, f"reference forecasts, mean over groups: {reference:.1f}%", fontsize=6, color=MUTED,
                ha="left", va="bottom")
        ax.plot(d["bin_end_min"], d["alarm_share_pct"], zorder=3, **_line("deadahead"))
        ax.set_xlim(250, 20)
        ax.set_ylim(0, 100)
        ax.set_xticks([240, 180, 120, 60, 30])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(title, fontsize=7, loc="left")
        _axes_style(ax)
    axes[0].set_ylabel("forecasts with the alarm on (%)", fontsize=7)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.85, bottom=0.17, wspace=0.08)
    fig.text(0.54, 0.025, "time before death, end of each 30-min bin (min)", ha="center", fontsize=7)
    fig.text(0.012, 0.985, "Post hoc analysis", fontsize=7, color=MUTED, va="top")
    _save(fig, out)


def _parser(description: str, table_help: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("table", type=Path, help=table_help)
    p.add_argument("out", type=Path)
    return p


def _draw(plot, table: Path, out: Path, columns: tuple) -> int:
    """Draw ``plot`` from the table at ``table``; 3 with a message when it is missing or is not that figure's table."""
    if not table.is_file():
        print(f"error: no table at {table}: run deadahead reproduce first", file=sys.stderr)
        return 3
    t = pd.read_csv(table)
    missing = [c for c in columns if c not in t.columns]
    if missing:
        print(f"error: {table} is not this figure's table (no column {missing[0]})", file=sys.stderr)
        return 3
    plot(t, out)
    return 0


def main(argv=None) -> int:
    a = _parser("Draw Fig. 2 from the warnings table with its spread.",
                "fig2_warnings_spread.csv, written by deadahead reproduce").parse_args(argv)
    columns = [f"{p}_{r}_r02{s}" for _, p in SPREAD_SERIES for r, _ in SPREAD_ROUTES for s in ("", "_q1", "_q3")]
    return _draw(plot_warnings, a.table, a.out, ("X", *columns))


def main_gap_window(argv=None) -> int:
    a = _parser("Draw the post hoc window figure of S8.",
                "figS8a_gap_window.csv, written by deadahead reproduce --part gap-window").parse_args(argv)
    return _draw(plot_gap_window, a.table, a.out, ("horizon_min", "gap_min", "route", "method", "auprc", "chance"))


def main_alarm_rate(argv=None) -> int:
    a = _parser("Draw the post hoc alarm-rate figure of S8.",
                "figS8b_alarm_rate.csv, written by deadahead reproduce --part alarm-rate").parse_args(argv)
    return _draw(plot_alarm_rate, a.table, a.out, ("route", "method", "set", "target_rate", "achieved_pooled",
                                                   "achieved_mean_over_groups", "warned_last_60_min_pct"))


def _csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")


def draw_from_results(res: dict, out_dir: Path) -> list[str]:
    """Write every figure and its table that the reproduce output ``res`` can give; returns the file stems drawn."""
    from deadahead.reproduce.main import ALARM_RATE_CSV, FIG2_SPREAD_CSV, GAP_WINDOW_CSV, figure_tables
    data = Path(out_dir) / "data"
    data.mkdir(parents=True, exist_ok=True)
    tables = figure_tables(res, "all")
    for name, text in tables.items():
        (data / name).write_bytes(text.encode())
    drawn = []
    for name, stem, plot in ((FIG2_SPREAD_CSV, "fig2_warnings", plot_warnings),
                             (GAP_WINDOW_CSV, "figS8a_gap_window", plot_gap_window),
                             (ALARM_RATE_CSV, "figS8b_alarm_rate", plot_alarm_rate)):
        if name in tables:
            plot(pd.read_csv(data / name), Path(out_dir) / f"{stem}.png")
            drawn.append(stem)
    for need, stem, build, plot in ((("table1",), TABLE1_FIG, table1_table, plot_table1),
                                    (("horizons",), HORIZONS_FIG, horizons_table, plot_horizons),
                                    (("table1",), GROUPS_FIG, group_differences_table, plot_group_differences),
                                    (("table1", "decomposition"), TEST_TIME_FIG, test_time_table, plot_test_time)):
        if all(k in res for k in need):
            table = build(res)
            _csv(table, data / f"{stem}.csv")
            plot(table, Path(out_dir) / f"{stem}.png")
            drawn.append(stem)
    return drawn


def draw_from_bundle(bundle_dir: Path, out_dir: Path) -> list[str]:
    """Write the tables computed from the predictions bundle, and the figure of the alarm share."""
    from deadahead.report.bundle_tables import (alarm_share_table, group_match_table, row_weight_table,
                                                warning_input_rows)
    from deadahead.reproduce.bundle import load_bundle
    bundle = load_bundle(bundle_dir, verify=False)
    missing = [k for k in ("movies", "inner_predictions", "inner_predictions_by_method") if k not in bundle]
    if missing:
        raise ValueError(f"the bundle at {bundle_dir} holds no {missing[0]}")
    data = Path(out_dir) / "data"
    _csv(row_weight_table(bundle), data / ROW_WEIGHTS_CSV)
    inputs = warning_input_rows(bundle)
    table = alarm_share_table(inputs)
    _csv(table, data / f"{ALARM_SHARE_FIG}.csv")
    _csv(group_match_table(inputs), data / GROUP_MATCH_CSV)
    plot_alarm_share(table, Path(out_dir) / f"{ALARM_SHARE_FIG}.png")
    return [ALARM_SHARE_FIG]


def main_all(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Draw every figure of the supplement from the reproduce output, with the "
                                             "plotted values as CSV files in <out-dir>/data/.")
    ap.add_argument("results", type=Path, nargs="?", default=None,
                    help="the JSON of deadahead reproduce --part all (default: <runs>/reproduce/all.json)")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--bundle", type=Path, default=None,
                    help="the predictions bundle, for the alarm-share figure, the match group by group and the row "
                         "weights (default: <DEADAHEAD_DATA>/bundle)")
    ap.add_argument("--no-bundle", action="store_true", help="draw only what the reproduce output holds")
    a = ap.parse_args(argv)
    results = a.results if a.results is not None else paths.runs_dir() / "reproduce" / "all.json"
    if not results.is_file():
        print(f"error: no reproduce output at {results}: run deadahead reproduce --part all first", file=sys.stderr)
        return 3
    res = json.loads(results.read_text())
    drawn = draw_from_results(res, a.out_dir)
    if not a.no_bundle:
        bundle_dir = a.bundle if a.bundle is not None else paths.bundle_dir()
        if not (bundle_dir / "MANIFEST.json").is_file():
            print(f"error: no predictions bundle at {bundle_dir}", file=sys.stderr)
            return 3
        try:
            drawn += draw_from_bundle(bundle_dir, a.out_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 3
    if not drawn:
        print(f"error: {results} holds no part that a figure reads", file=sys.stderr)
        return 3
    print(f"{len(drawn)} figures in {a.out_dir}: {', '.join(drawn)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
