"""Transmission-tree and effective-reproduction-number metrics."""
from __future__ import annotations

import polars as pl

from july_analysis import schema as S


def transmission_edges(infections: pl.DataFrame) -> pl.DataFrame:
    """Edge list of the transmission graph.

    Columns: ``infector_id, person_id, time, encounter_type_name,
    transmission_mode_name``. Seed infections (infector == -1) are excluded.
    """
    if infections.height == 0:
        return pl.DataFrame(
            schema={
                "infector_id": pl.Int32,
                "person_id": pl.Int32,
                "time": pl.Float64,
                "encounter_type_name": pl.Utf8,
                "transmission_mode_name": pl.Utf8,
            }
        )
    keep = ["infector_id", "person_id", "time"]
    if "encounter_type_name" in infections.columns:
        keep.append("encounter_type_name")
    if "transmission_mode_name" in infections.columns:
        keep.append("transmission_mode_name")
    return (
        infections.filter(pl.col("infector_id") != S.INFECTOR_ID_NONE)
        .select(keep)
    )


def secondary_cases_per_infector(infections: pl.DataFrame) -> pl.DataFrame:
    """Per-infector count of onward infections. Columns: ``infector_id, n_secondary``."""
    edges = transmission_edges(infections)
    if edges.height == 0:
        return pl.DataFrame(schema={"infector_id": pl.Int32, "n_secondary": pl.UInt32})
    return (
        edges.group_by("infector_id")
        .agg(pl.len().alias("n_secondary"))
        .sort("n_secondary", descending=True)
    )


def reff_by_encounter_type(infections: pl.DataFrame) -> pl.DataFrame:
    """Share of onward transmissions and mean secondary cases per encounter
    type. Columns: ``encounter_type_name, secondary_cases, share,
    infectors, mean_secondary_per_infector``.

    Note: this is *not* the textbook Rₑ. It is the share of non-seed
    transmissions attributable to each encounter type plus the mean onward
    count of infectors who ever transmitted via that channel. Useful for
    "which channel drives the epidemic" questions without requiring a
    generation-time model.
    """
    edges = transmission_edges(infections)
    if edges.height == 0 or "encounter_type_name" not in edges.columns:
        return pl.DataFrame(
            schema={
                "encounter_type_name": pl.Utf8,
                "secondary_cases": pl.UInt32,
                "share": pl.Float64,
                "infectors": pl.UInt32,
                "mean_secondary_per_infector": pl.Float64,
            }
        )
    total = edges.height
    return (
        edges.group_by("encounter_type_name")
        .agg([
            pl.len().alias("secondary_cases"),
            pl.col("infector_id").n_unique().alias("infectors"),
        ])
        .with_columns([
            (pl.col("secondary_cases") / total).alias("share"),
            (pl.col("secondary_cases") / pl.col("infectors")).alias(
                "mean_secondary_per_infector"
            ),
        ])
        .sort("secondary_cases", descending=True)
    )


def generation_time_distribution(infections: pl.DataFrame) -> pl.DataFrame:
    """Per-secondary generation interval.

    For every non-seed infection, compute ``time(secondary) - time(infector)``
    by self-joining on ``infector_id = person_id``. Columns: ``infector_id,
    person_id, generation_time``.
    """
    if infections.height == 0:
        return pl.DataFrame(
            schema={
                "infector_id": pl.Int32,
                "person_id": pl.Int32,
                "generation_time": pl.Float64,
            }
        )
    secondaries = infections.filter(pl.col("infector_id") != S.INFECTOR_ID_NONE).select(
        ["infector_id", "person_id", pl.col("time").alias("secondary_time")]
    )
    infectors = infections.select(
        pl.col("person_id").alias("infector_id"),
        pl.col("time").alias("infector_time"),
    )
    return (
        secondaries.join(infectors, on="infector_id", how="left")
        .with_columns((pl.col("secondary_time") - pl.col("infector_time")).alias("generation_time"))
        .filter(pl.col("generation_time") >= 0.0)
        .select(["infector_id", "person_id", "generation_time"])
    )


def reff_rolling(
    infections: pl.DataFrame, *, window_days: int = 3
) -> pl.DataFrame:
    """Rolling effective reproduction number, computed as
    ``Rₑ(t) = infections(t) / infections(t - generation_time_mean)``.

    This is a coarse proxy — good for trend direction, not for calibration.
    Columns: ``day, r_effective``.
    """
    if infections.height == 0:
        return pl.DataFrame(schema={"day": pl.Int64, "r_effective": pl.Float64})
    from july_analysis.metrics.epi import incidence_by_day

    daily = incidence_by_day(infections)
    if daily.height < window_days + 1:
        return daily.with_columns(pl.lit(None, pl.Float64).alias("r_effective")).select(
            ["day", "r_effective"]
        )
    shifted = daily.with_columns(
        pl.col("count").shift(window_days).alias("prev_count")
    )
    return shifted.with_columns(
        (pl.col("count") / pl.col("prev_count")).alias("r_effective")
    ).select(["day", "r_effective"])
