"""Group-encounter reconstruction and encounter-mix metrics."""
from __future__ import annotations

import polars as pl


def encounter_type_share(encounters: pl.DataFrame) -> pl.DataFrame:
    """Fraction of all pair-records by encounter type.

    Columns: ``encounter_type_name, count, share``.
    """
    if encounters.height == 0 or "encounter_type_name" not in encounters.columns:
        return pl.DataFrame(
            schema={
                "encounter_type_name": pl.Utf8,
                "count": pl.UInt32,
                "share": pl.Float64,
            }
        )
    total = encounters.height
    return (
        encounters.group_by("encounter_type_name")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / total).alias("share"))
        .sort("count", descending=True)
    )


def reconstruct_group_encounters(encounters: pl.DataFrame) -> pl.DataFrame:
    """Collapse per-pair rows back into one row per real encounter.

    The simulator fans one row out per host↔guest pair for every coordinated
    encounter, and stamps every row of the same real event with a shared
    ``group_id`` (``uint64``, rank-qualified in the high 16 bits so it's
    globally unique). This function groups on ``group_id`` to recover the
    participant set and treats ``person_a`` as the host (the writer
    convention; all rows of the same group share the same ``person_a``).

    Returns columns: ``group_id, host_id, time, slot, encounter_type_name,
    size, guest_ids``. ``size`` includes the host, so dyadic encounter types
    (romantic/cohabiting/ooe) come out as ``size == 2``.

    For a focused group-only view, filter to
    ``encounter_type_name == "group_sex"`` before calling.
    """
    if encounters.height == 0 or "group_id" not in encounters.columns:
        return pl.DataFrame(
            schema={
                "group_id": pl.UInt64,
                "host_id": pl.Int32,
                "time": pl.Float64,
                "slot": pl.Int32,
                "encounter_type_name": pl.Utf8,
                "size": pl.UInt32,
                "guest_ids": pl.List(pl.Int32),
            }
        )
    type_col = (
        "encounter_type_name"
        if "encounter_type_name" in encounters.columns
        else "encounter_type_id"
    )
    grouped = encounters.group_by("group_id").agg([
        pl.col("person_a").first().alias("host_id"),
        pl.col("time").first().alias("time"),
        pl.col("slot").first().alias("slot"),
        pl.col(type_col).first().alias(type_col),
        pl.col("person_b").unique().alias("guest_ids"),
        (pl.col("person_b").n_unique() + 1).alias("size"),
    ])
    order = ["group_id", "host_id", "time", "slot", type_col, "size", "guest_ids"]
    return grouped.select(order).sort(["time", "host_id"])


def channel_efficiency(
    encounters: pl.DataFrame, infections: pl.DataFrame
) -> pl.DataFrame:
    """Per-encounter-type event count vs resulting infections.

    For each ``encounter_type_name``: number of real events (pair-records
    collapsed via ``group_id``), number of non-seed infections attributed
    to that channel, and the ratio.

    The ratio answers the question "did this channel produce few
    transmissions because it fired rarely, or because the per-event
    transmission rate is low?" — the former is a scheduling/config story,
    the latter usually a contact-matrix or eligibility story.

    Columns: ``encounter_type_name, n_events, n_infections,
    infections_per_event``.
    """
    empty = pl.DataFrame(
        schema={
            "encounter_type_name": pl.Utf8,
            "n_events": pl.UInt32,
            "n_infections": pl.UInt32,
            "infections_per_event": pl.Float64,
        }
    )
    if encounters.height == 0 or "encounter_type_name" not in encounters.columns:
        return empty

    rec = reconstruct_group_encounters(encounters)
    if rec.height == 0:
        return empty
    events_by_type = rec.group_by("encounter_type_name").agg(
        pl.len().cast(pl.UInt32).alias("n_events")
    )

    if (
        infections.height == 0
        or "encounter_type_name" not in infections.columns
    ):
        inf_by_type = pl.DataFrame(
            schema={
                "encounter_type_name": pl.Utf8,
                "n_infections": pl.UInt32,
            }
        )
    else:
        non_seed = (
            infections.filter(~pl.col("is_seed"))
            if "is_seed" in infections.columns
            else infections.filter(pl.col("infector_id") != -1)
        )
        inf_by_type = non_seed.group_by("encounter_type_name").agg(
            pl.len().cast(pl.UInt32).alias("n_infections")
        )

    return (
        events_by_type.join(inf_by_type, on="encounter_type_name", how="left")
        .with_columns(pl.col("n_infections").fill_null(0))
        .with_columns(
            (pl.col("n_infections") / pl.col("n_events")).alias(
                "infections_per_event"
            )
        )
        .sort("n_events", descending=True)
    )


def group_size_distribution(reconstructed: pl.DataFrame) -> pl.DataFrame:
    """Histogram of encounter sizes. Columns: ``size, count``."""
    if reconstructed.height == 0:
        return pl.DataFrame(schema={"size": pl.UInt32, "count": pl.UInt32})
    return (
        reconstructed.group_by("size")
        .agg(pl.len().alias("count"))
        .sort("size")
    )
