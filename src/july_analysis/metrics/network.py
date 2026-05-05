"""Network / exposure-structure metrics computed from relationships and
coordinated encounters.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def _canonical_pairs(df: pl.DataFrame, a: str = "person_a", b: str = "person_b") -> pl.DataFrame:
    """Return a new frame with ``(p_lo, p_hi)`` = sorted (a, b) — used so that
    (A, B) and (B, A) collapse to one edge when counting unique partnerships.
    """
    return df.with_columns([
        pl.min_horizontal([pl.col(a), pl.col(b)]).alias("p_lo"),
        pl.max_horizontal([pl.col(a), pl.col(b)]).alias("p_hi"),
    ])


def degree_per_person(relationships: pl.DataFrame) -> pl.DataFrame:
    """Total number of distinct partners per person across the full run.

    Columns: ``person_id, degree``.
    """
    if relationships.height == 0:
        return pl.DataFrame(schema={"person_id": pl.Int32, "degree": pl.UInt32})
    pairs = _canonical_pairs(relationships).select(["p_lo", "p_hi"]).unique()
    # person -> degree by unioning both sides
    left = pairs.select(pl.col("p_lo").alias("person_id"), pl.col("p_hi").alias("partner_id"))
    right = pairs.select(pl.col("p_hi").alias("person_id"), pl.col("p_lo").alias("partner_id"))
    return (
        pl.concat([left, right])
        .unique()
        .group_by("person_id")
        .agg(pl.len().alias("degree"))
        .sort("person_id")
    )


def degree_distribution(relationships: pl.DataFrame) -> pl.DataFrame:
    """Histogram of distinct-partner counts. Columns: ``degree, count``."""
    deg = degree_per_person(relationships)
    if deg.height == 0:
        return pl.DataFrame(schema={"degree": pl.UInt32, "count": pl.UInt32})
    return deg.group_by("degree").agg(pl.len().alias("count")).sort("degree")


def concurrency_series(
    relationships: pl.DataFrame,
    *,
    times: np.ndarray | list[float],
    exclude_cohabiting_edges: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """At each probe time, count people currently in ≥ 2 simultaneous
    partnerships. Columns: ``time, concurrent_people, mean_partners``.

    A relationship is considered active at time *t* iff
    ``time ≤ t < dissolution_time``.

    Cohabiting partnerships in the sim are given near-permanent dissolution
    times, so they dominate raw counts without contributing to the churny
    concurrency we care about for STI-style transmission. Pass
    ``exclude_cohabiting_edges`` — the DataFrame returned by
    ``EventStore.population_network("cohabiting_couple")`` — to strip those
    relationship events *before* counting. This is a person-level filter:
    any relationship whose ``(person_a, person_b)`` pair matches a
    cohabiting pair in either direction is dropped.
    """
    out_schema = {"time": pl.Float64, "concurrent_people": pl.UInt32, "mean_partners": pl.Float64}
    if relationships.height == 0:
        return pl.DataFrame(schema=out_schema)
    rel = relationships.select(["person_a", "person_b", "time", "dissolution_time"])
    if exclude_cohabiting_edges is not None and exclude_cohabiting_edges.height > 0:
        coh = exclude_cohabiting_edges.select(
            pl.min_horizontal("person_id", "partner_id").alias("p_lo"),
            pl.max_horizontal("person_id", "partner_id").alias("p_hi"),
        ).unique()
        rel = (
            rel.with_columns(
                pl.min_horizontal("person_a", "person_b").alias("p_lo"),
                pl.max_horizontal("person_a", "person_b").alias("p_hi"),
            )
            .join(coh.with_columns(pl.lit(True).alias("_coh")), on=["p_lo", "p_hi"], how="left")
            .filter(pl.col("_coh").is_null())
            .drop(["p_lo", "p_hi", "_coh"])
        )
        if rel.height == 0:
            return pl.DataFrame(
                [{"time": float(t), "concurrent_people": 0, "mean_partners": 0.0} for t in times],
                schema=out_schema,
            )
    rows = []
    for t in times:
        active = rel.filter((pl.col("time") <= t) & (pl.col("dissolution_time") > t))
        if active.height == 0:
            rows.append({"time": float(t), "concurrent_people": 0, "mean_partners": 0.0})
            continue
        counts = (
            pl.concat([
                active.select(pl.col("person_a").alias("p")),
                active.select(pl.col("person_b").alias("p")),
            ])
            .group_by("p")
            .agg(pl.len().alias("n_partners"))
        )
        rows.append({
            "time": float(t),
            "concurrent_people": int((counts["n_partners"] >= 2).sum()),
            "mean_partners": float(counts["n_partners"].mean()),
        })
    return pl.DataFrame(rows, schema=out_schema)


def mixing_matrix(
    events: pl.DataFrame,
    demographics: pl.DataFrame,
    attr: str,
    *,
    a: str = "person_a",
    b: str = "person_b",
) -> pl.DataFrame:
    """Long-form mixing matrix: counts of events by ``(attr_a, attr_b)``.

    ``events`` must have two person-id columns (default ``person_a`` /
    ``person_b`` — override for infection events via ``a="infector_id"``,
    ``b="person_id"``). ``demographics`` must have ``person_id`` and the
    chosen ``attr`` column.

    Columns: ``<attr>_a, <attr>_b, count``.
    """
    out_schema = {f"{attr}_a": pl.Unknown, f"{attr}_b": pl.Unknown, "count": pl.UInt32}
    if events.height == 0 or demographics.height == 0 or attr not in demographics.columns:
        return pl.DataFrame(schema={f"{attr}_a": pl.Utf8, f"{attr}_b": pl.Utf8, "count": pl.UInt32})
    dem_a = demographics.select(
        pl.col("person_id").alias(a), pl.col(attr).alias(f"{attr}_a")
    )
    dem_b = demographics.select(
        pl.col("person_id").alias(b), pl.col(attr).alias(f"{attr}_b")
    )
    return (
        events.select([a, b])
        .join(dem_a, on=a, how="left")
        .join(dem_b, on=b, how="left")
        .group_by([f"{attr}_a", f"{attr}_b"])
        .agg(pl.len().alias("count"))
        .sort([f"{attr}_a", f"{attr}_b"])
    )


def encounters_per_person_per_day(encounters: pl.DataFrame) -> pl.DataFrame:
    """How many coordinated-encounter pair-records each person appears in per
    day. Returns ``person_id, day, encounters``. Treats host and guest
    symmetrically (both sides are counted).
    """
    if encounters.height == 0:
        return pl.DataFrame(
            schema={"person_id": pl.Int32, "day": pl.Int64, "encounters": pl.UInt32}
        )
    with_day = encounters.with_columns(pl.col("time").floor().cast(pl.Int64).alias("day"))
    both = pl.concat([
        with_day.select(pl.col("person_a").alias("person_id"), pl.col("day")),
        with_day.select(pl.col("person_b").alias("person_id"), pl.col("day")),
    ])
    return both.group_by(["person_id", "day"]).agg(pl.len().alias("encounters")).sort(["person_id", "day"])
