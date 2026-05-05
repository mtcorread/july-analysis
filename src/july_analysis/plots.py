"""Matplotlib plots — thin wrappers over metric DataFrames.

Every function takes a DataFrame (as produced by ``july_analysis.metrics``)
and returns a ``matplotlib.figure.Figure``. No file I/O, no show() calls:
the caller is responsible for saving or embedding.
"""
from __future__ import annotations

from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # headless — report generation runs on servers
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _fig(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def incidence_curve(daily: pl.DataFrame, *, title: str = "Daily incidence") -> plt.Figure:
    fig, ax = _fig()
    if daily.height == 0:
        ax.text(0.5, 0.5, "no infections", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.bar(daily["day"].to_numpy(), daily["count"].to_numpy(), color=_PALETTE[0])
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("New infections")
    ax.set_title(title)
    return fig


def cumulative_curve(cum: pl.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    if cum.height == 0 or "cumulative" not in cum.columns:
        ax.text(0.5, 0.5, "no infections", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(cum["day"].to_numpy(), cum["cumulative"].to_numpy(), marker="o", color=_PALETTE[2])
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Cumulative infections")
    ax.set_title("Cumulative infections")
    return fig


def stacked_incidence_by_encounter(
    daily_by_type: pl.DataFrame,
    *,
    title: str = "Infections by encounter type",
) -> plt.Figure:
    fig, ax = _fig()
    if daily_by_type.height == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig
    pivot = (
        daily_by_type.pivot(
            values="count", index="day", on="encounter_type_name", aggregate_function="sum"
        )
        .fill_null(0)
        .sort("day")
    )
    days = pivot["day"].to_numpy()
    bottom = np.zeros_like(days, dtype=float)
    for i, col in enumerate(c for c in pivot.columns if c != "day"):
        vals = pivot[col].to_numpy()
        ax.bar(days, vals, bottom=bottom, label=col, color=_PALETTE[i % len(_PALETTE)])
        bottom += vals
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("New infections")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    return fig


def encounter_type_contribution_bar(contribution: pl.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    if contribution.height == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig
    names = contribution["encounter_type_name"].to_list()
    shares = contribution["share"].to_numpy()
    ax.barh(names, shares, color=_PALETTE[0])
    ax.set_xlabel("Share of non-seed infections")
    ax.set_title("Encounter-type contribution to transmission")
    ax.invert_yaxis()
    return fig


def degree_distribution_bar(degree_dist: pl.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    if degree_dist.height == 0:
        ax.text(0.5, 0.5, "no relationships", ha="center", va="center", transform=ax.transAxes)
        return fig
    ax.bar(
        degree_dist["degree"].to_numpy(),
        degree_dist["count"].to_numpy(),
        color=_PALETTE[4],
    )
    ax.set_xlabel("Partners per person (simulation window)")
    ax.set_ylabel("Number of people")
    ax.set_title("Degree distribution")
    ax.set_yscale("log")
    return fig


def concurrency_curve(series: pl.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    if series.height == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig
    ax.plot(
        series["time"].to_numpy(),
        series["concurrent_people"].to_numpy(),
        marker="o",
        color=_PALETTE[1],
    )
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("People with ≥ 2 active partnerships")
    ax.set_title("Point-prevalence concurrency")
    return fig


def mixing_matrix_heatmap(mm: pl.DataFrame, attr: str = "age_group") -> plt.Figure:
    fig, ax = _fig(figsize=(6, 5))
    if mm.height == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig
    a_col, b_col = f"{attr}_a", f"{attr}_b"
    vals_a = sorted(mm[a_col].drop_nulls().unique().to_list())
    vals_b = sorted(mm[b_col].drop_nulls().unique().to_list())
    idx_a = {v: i for i, v in enumerate(vals_a)}
    idx_b = {v: i for i, v in enumerate(vals_b)}
    M = np.zeros((len(vals_a), len(vals_b)), dtype=float)
    for row in mm.iter_rows(named=True):
        va, vb, c = row[a_col], row[b_col], row["count"]
        if va is None or vb is None:
            continue
        M[idx_a[va], idx_b[vb]] = c
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(vals_b)))
    ax.set_xticklabels([str(v) for v in vals_b], rotation=90, fontsize=7)
    ax.set_yticks(range(len(vals_a)))
    ax.set_yticklabels([str(v) for v in vals_a], fontsize=7)
    ax.set_xlabel(f"{attr} (infectee)")
    ax.set_ylabel(f"{attr} (infector)")
    ax.set_title(f"Mixing matrix by {attr}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="events")
    return fig


def generation_time_hist(gen: pl.DataFrame, *, bins: int = 30) -> plt.Figure:
    fig, ax = _fig()
    if gen.height == 0:
        ax.text(0.5, 0.5, "no secondary cases", ha="center", va="center", transform=ax.transAxes)
        return fig
    ax.hist(gen["generation_time"].to_numpy(), bins=bins, color=_PALETTE[3], edgecolor="white")
    ax.set_xlabel("Generation time (days)")
    ax.set_ylabel("Count")
    ax.set_title("Generation-time distribution")
    return fig


def reff_curve(reff: pl.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    if reff.height == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig
    ax.plot(
        reff["day"].to_numpy(),
        reff["r_effective"].to_numpy(),
        marker="o",
        color=_PALETTE[5],
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("R_effective (rolling proxy)")
    ax.set_title("Rolling effective reproduction number")
    return fig


def group_size_distribution_bar(sizes: pl.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    if sizes.height == 0:
        ax.text(0.5, 0.5, "no group encounters", ha="center", va="center", transform=ax.transAxes)
        return fig
    ax.bar(sizes["size"].to_numpy(), sizes["count"].to_numpy(), color=_PALETTE[6])
    ax.set_xlabel("Encounter size")
    ax.set_ylabel("Number of encounters")
    ax.set_title("Group-encounter size distribution")
    return fig


def profile_type_distribution_bar(
    dist: pl.DataFrame, *, title: str = "Profile type distribution"
) -> plt.Figure:
    fig, ax = _fig()
    if dist.height == 0:
        ax.text(0.5, 0.5, "no profile data", ha="center", va="center", transform=ax.transAxes)
        return fig
    names = dist["profile_type_name"].to_list()
    counts = dist["count"].to_numpy()
    ax.barh(names, counts, color=_PALETTE[2])
    ax.set_xlabel("Agents")
    ax.set_title(title)
    ax.invert_yaxis()
    return fig


def infections_by_profile_bar(
    inf_by_profile: pl.DataFrame, *, title: str = "Infections by profile"
) -> plt.Figure:
    fig, ax = _fig()
    if inf_by_profile.height == 0:
        ax.text(0.5, 0.5, "no profile data", ha="center", va="center", transform=ax.transAxes)
        return fig
    names = inf_by_profile["profile_type_name"].to_list()
    shares = inf_by_profile["share"].to_numpy()
    ax.barh(names, shares, color=_PALETTE[3])
    ax.set_xlabel("Share of non-seed infections")
    ax.set_title(title)
    ax.invert_yaxis()
    return fig


def infections_by_venue_type_bar(
    df: pl.DataFrame, *, title: str = "Infections by venue type"
) -> plt.Figure:
    fig, ax = _fig()
    if df.height == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig
    names = df["venue_type"].to_list()
    shares = df["share"].to_numpy()
    ax.barh(names, shares, color=_PALETTE[4])
    ax.set_xlabel("Share of non-seed infections")
    ax.set_title(title)
    ax.invert_yaxis()
    return fig


def encounter_type_share_bar(share: pl.DataFrame) -> plt.Figure:
    fig, ax = _fig()
    if share.height == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig
    ax.barh(
        share["encounter_type_name"].to_list(),
        share["share"].to_numpy(),
        color=_PALETTE[7],
    )
    ax.set_xlabel("Share of coordinated encounter pair-records")
    ax.set_title("Encounter type mix (exposure side)")
    ax.invert_yaxis()
    return fig


# ---------------------------------------------------------------------------
# world_state.h5 sexual demographics
# ---------------------------------------------------------------------------

def orientation_overall_bar(counts, labels):
    fig, ax = _fig(figsize=(6, 3.5))
    ax.bar(labels, counts, color=_PALETTE[0])
    ax.set_ylabel("people")
    ax.set_title("Sexual orientation — population total")
    for i, c in enumerate(counts):
        if c:
            ax.text(i, c, f"{int(c):,}", ha="center", va="bottom", fontsize=8)
    return fig


def orientation_by_sex_bar(table, orientation_labels, sex_labels):
    fig, ax = _fig(figsize=(6, 3.5))
    width = 0.28
    x = np.arange(len(orientation_labels))
    for i, label in sex_labels.items():
        col = table[:, i]
        if not col.any():
            continue
        ax.bar(x + (i - 1) * width, col, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(orientation_labels)
    ax.set_ylabel("people")
    ax.set_title("Orientation × sex")
    ax.legend(fontsize=8)
    return fig


def orientation_by_age_stacked(table_pct, age_labels, orientation_labels):
    fig, ax = _fig(figsize=(7.5, 4))
    bottom = np.zeros(len(age_labels))
    for j, label in enumerate(orientation_labels):
        ax.bar(age_labels, table_pct[:, j], bottom=bottom, label=label)
        bottom += table_pct[:, j]
    ax.set_ylabel("% within age band")
    ax.set_title("Orientation share by age band")
    ax.legend(fontsize=8, loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return fig


def orientation_by_lgu_stacked(table_pct, labels, orientation_labels):
    fig, ax = _fig(figsize=(8, max(3.5, 0.35 * len(labels) + 1)))
    bottom = np.zeros(len(labels))
    for j, lab in enumerate(orientation_labels):
        ax.barh(labels, table_pct[:, j], left=bottom, label=lab)
        bottom += table_pct[:, j]
    ax.set_xlabel("% within LGU")
    ax.set_title(f"Orientation share by LGU (top {len(labels)} by population)")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")
    return fig


def relationship_status_bar(rows):
    """rows: iterable of (status_label, count). Order is preserved on the x-axis."""
    rows = list(rows)
    fig, ax = _fig(figsize=(max(6, 0.9 * len(rows) + 2), 3.5))
    if not rows:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig
    labels = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    ax.bar(range(len(labels)), counts, color=_PALETTE[2])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("people")
    ax.set_title("Relationship status — raw values from world snapshot")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for i, c in enumerate(counts):
        if c:
            ax.text(i, c, f"{int(c):,}", ha="center", va="bottom", fontsize=8)
    return fig


def couple_sex_pairing_bar(values, labels):
    fig, ax = _fig(figsize=(5, 3.2))
    ax.bar(labels, values, color=_PALETTE[3])
    ax.set_ylabel("couples")
    ax.set_title("Cohabiting couples — sex pairing")
    for i, c in enumerate(values):
        if c:
            ax.text(i, c, f"{int(c):,}", ha="center", va="bottom", fontsize=8)
    return fig


def couple_age_diff_bar(hist, edges):
    if hist.size == 0:
        fig, ax = _fig(figsize=(6, 3.2))
        ax.text(0.5, 0.5, "no couples", ha="center", va="center", transform=ax.transAxes)
        return fig
    labels = [f"{int(edges[i])}–{int(edges[i+1])}" for i in range(len(edges) - 1)]
    labels[-1] = f"{int(edges[-2])}+"
    fig, ax = _fig(figsize=(6, 3.2))
    ax.bar(labels, hist, color=_PALETTE[4])
    ax.set_xlabel("|age_a − age_b| (years)")
    ax.set_ylabel("couples")
    ax.set_title("Cohabiting couples — age difference")
    return fig
