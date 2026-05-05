"""Epidemiological curves and contribution breakdowns."""
from __future__ import annotations

import polars as pl

from july_analysis import schema as S


def _day_col(time_expr: pl.Expr) -> pl.Expr:
    """Convert a ``time`` column (float days since start) to an integer day."""
    return time_expr.floor().cast(pl.Int64).alias("day")


def _drop_seeds(infections: pl.DataFrame) -> pl.DataFrame:
    if "is_seed" in infections.columns:
        return infections.filter(~pl.col("is_seed"))
    return infections.filter(pl.col("infector_id") != S.INFECTOR_ID_NONE)


def incidence_by_day(
    infections: pl.DataFrame, *, exclude_seeds: bool = True
) -> pl.DataFrame:
    """Daily new-infection count. Returns columns ``day, count``."""
    if infections.height == 0:
        return pl.DataFrame({"day": [], "count": []}, schema={"day": pl.Int64, "count": pl.UInt32})
    df = _drop_seeds(infections) if exclude_seeds else infections
    return (
        df.with_columns(_day_col(pl.col("time")))
        .group_by("day")
        .agg(pl.len().alias("count"))
        .sort("day")
    )


def cumulative_incidence(
    infections: pl.DataFrame, *, exclude_seeds: bool = True
) -> pl.DataFrame:
    """Cumulative infections by day. Columns: ``day, count, cumulative``."""
    daily = incidence_by_day(infections, exclude_seeds=exclude_seeds)
    if daily.height == 0:
        return daily.with_columns(pl.lit(0, dtype=pl.UInt32).alias("cumulative"))
    return daily.with_columns(pl.col("count").cum_sum().alias("cumulative"))


def incidence_by_encounter_type(
    infections: pl.DataFrame, *, exclude_seeds: bool = True
) -> pl.DataFrame:
    """Daily infections split by ``encounter_type_name``.

    Columns: ``day, encounter_type_name, count``.
    """
    if infections.height == 0 or "encounter_type_name" not in infections.columns:
        return pl.DataFrame(
            schema={"day": pl.Int64, "encounter_type_name": pl.Utf8, "count": pl.UInt32}
        )
    df = _drop_seeds(infections) if exclude_seeds else infections
    return (
        df.with_columns(_day_col(pl.col("time")))
        .group_by(["day", "encounter_type_name"])
        .agg(pl.len().alias("count"))
        .sort(["day", "encounter_type_name"])
    )


def incidence_by_transmission_mode(
    infections: pl.DataFrame, *, exclude_seeds: bool = True
) -> pl.DataFrame:
    """Daily infections split by transmission-mode label."""
    if infections.height == 0 or "transmission_mode_name" not in infections.columns:
        return pl.DataFrame(
            schema={"day": pl.Int64, "transmission_mode_name": pl.Utf8, "count": pl.UInt32}
        )
    df = _drop_seeds(infections) if exclude_seeds else infections
    return (
        df.with_columns(_day_col(pl.col("time")))
        .group_by(["day", "transmission_mode_name"])
        .agg(pl.len().alias("count"))
        .sort(["day", "transmission_mode_name"])
    )


def encounter_type_contribution(
    infections: pl.DataFrame, *, exclude_seeds: bool = True
) -> pl.DataFrame:
    """Share of non-seed infections attributable to each encounter type.

    Columns: ``encounter_type_name, count, share``.
    """
    if infections.height == 0 or "encounter_type_name" not in infections.columns:
        return pl.DataFrame(
            schema={
                "encounter_type_name": pl.Utf8,
                "count": pl.UInt32,
                "share": pl.Float64,
            }
        )
    df = _drop_seeds(infections) if exclude_seeds else infections
    total = df.height
    if total == 0:
        return pl.DataFrame(
            schema={
                "encounter_type_name": pl.Utf8,
                "count": pl.UInt32,
                "share": pl.Float64,
            }
        )
    return (
        df.group_by("encounter_type_name")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / total).alias("share"))
        .sort("count", descending=True)
    )


def infections_by_venue_type(
    infections: pl.DataFrame,
    venues: pl.DataFrame,
    *,
    exclude_seeds: bool = True,
) -> pl.DataFrame:
    """Non-seed infections grouped by venue ``type`` (household, office, etc.).

    Joins ``infections.venue_id`` against ``venues.venue_id`` and aggregates
    on the venue's ``type`` string. Seeds (``venue_id == VENUE_ID_SEED``) are
    dropped. Rows whose venue isn't in the lookup become ``<unknown>``.

    Columns: ``venue_type, count, share``.
    """
    empty = pl.DataFrame(
        schema={"venue_type": pl.Utf8, "count": pl.UInt32, "share": pl.Float64}
    )
    if infections.height == 0:
        return empty
    df = _drop_seeds(infections) if exclude_seeds else infections
    df = df.filter(pl.col("venue_id") != S.VENUE_ID_SEED)
    if df.height == 0:
        return empty
    if venues.height == 0 or "type" not in venues.columns:
        joined = df.with_columns(pl.lit("<unknown>").alias("venue_type"))
    else:
        joined = df.join(
            venues.select(["venue_id", pl.col("type").alias("venue_type")]),
            on="venue_id",
            how="left",
        ).with_columns(pl.col("venue_type").fill_null("<unknown>"))
    total = joined.height
    return (
        joined.group_by("venue_type")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / total).alias("share"))
        .sort("count", descending=True)
    )


def transmission_mode_contribution(
    infections: pl.DataFrame, *, exclude_seeds: bool = True
) -> pl.DataFrame:
    if infections.height == 0 or "transmission_mode_name" not in infections.columns:
        return pl.DataFrame(
            schema={
                "transmission_mode_name": pl.Utf8,
                "count": pl.UInt32,
                "share": pl.Float64,
            }
        )
    df = _drop_seeds(infections) if exclude_seeds else infections
    total = df.height
    if total == 0:
        return pl.DataFrame(
            schema={
                "transmission_mode_name": pl.Utf8,
                "count": pl.UInt32,
                "share": pl.Float64,
            }
        )
    return (
        df.group_by("transmission_mode_name")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / total).alias("share"))
        .sort("count", descending=True)
    )


def time_to_symptom(
    infections: pl.DataFrame, symptom_changes: pl.DataFrame
) -> pl.DataFrame:
    """Per-person delay from infection to first symptom change.

    Columns: ``person_id, infection_time, first_symptom_time, delay``.
    """
    if infections.height == 0 or symptom_changes.height == 0:
        return pl.DataFrame(
            schema={
                "person_id": pl.Int32,
                "infection_time": pl.Float64,
                "first_symptom_time": pl.Float64,
                "delay": pl.Float64,
            }
        )
    first_sym = (
        symptom_changes.group_by("person_id")
        .agg(pl.col("time").min().alias("first_symptom_time"))
    )
    infs = infections.select(["person_id", pl.col("time").alias("infection_time")])
    return (
        infs.join(first_sym, on="person_id", how="inner")
        .with_columns((pl.col("first_symptom_time") - pl.col("infection_time")).alias("delay"))
        .filter(pl.col("delay") >= 0.0)
    )


def summary_scalars(
    infections: pl.DataFrame, deaths: pl.DataFrame
) -> dict[str, float | int]:
    """Tiny roll-up used for report headers and sweep collection."""
    non_seed = _drop_seeds(infections) if infections.height > 0 else infections
    return {
        "total_infections": int(infections.height),
        "seed_infections": int(infections.height - non_seed.height),
        "transmission_events": int(non_seed.height),
        "total_deaths": int(deaths.height),
        "peak_incidence_day": int(
            incidence_by_day(infections).sort("count", descending=True)["day"][0]
        )
        if non_seed.height > 0
        else -1,
        "peak_incidence": int(
            incidence_by_day(infections).sort("count", descending=True)["count"][0]
        )
        if non_seed.height > 0
        else 0,
    }
