"""Profile-stratified metrics.

These metrics depend on `/lookups/profile_assignments/<facet>` being present
in the HDF5 — i.e. the simulation had at least one profile facet loaded when
it ran. Each function takes ``infections`` / ``relationships`` /
``encounters`` plus a ``profile_assignments`` DataFrame (as produced by
``EventStore.profile_assignments()``) and returns a tidy DataFrame.

All functions return an empty DataFrame when ``profile_assignments`` is empty,
so they're safe to call unconditionally — the report just renders a "no
profile data" placeholder in that case.
"""
from __future__ import annotations

import polars as pl

from july_analysis import schema as S


def _label_col() -> str:
    return "profile_type_name"


def _require_profile(profiles: pl.DataFrame) -> bool:
    return profiles.height > 0 and _label_col() in profiles.columns


def profile_type_distribution(profiles: pl.DataFrame) -> pl.DataFrame:
    """How many agents sit in each profile type, and their share of the
    population. Columns: ``profile_type_name, count, share``.
    """
    if not _require_profile(profiles):
        return pl.DataFrame(
            schema={
                _label_col(): pl.Utf8,
                "count": pl.UInt32,
                "share": pl.Float64,
            }
        )
    total = profiles.height
    return (
        profiles.group_by(_label_col())
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / total).alias("share"))
        .sort("count", descending=True)
    )


def profile_derived_field_summary(profiles: pl.DataFrame) -> pl.DataFrame:
    """Mean of every numeric derived field, grouped by profile type.

    Useful sanity check that ProfileAssigner produced the distributions the
    client expected (e.g. only ``non_exclusive_ooe_anytime`` has nonzero
    ``ooe_probability`` in-relationship).
    """
    if not _require_profile(profiles):
        return pl.DataFrame()
    numeric_cols = [
        c
        for c, dt in profiles.schema.items()
        if c not in ("person_id", "profile_id", _label_col())
        and dt.is_numeric()
    ]
    if not numeric_cols:
        return pl.DataFrame()
    return (
        profiles.group_by(_label_col())
        .agg([pl.col(c).mean().alias(f"mean_{c}") for c in numeric_cols])
        .sort(_label_col())
    )


def infections_by_profile(
    infections: pl.DataFrame,
    profiles: pl.DataFrame,
    *,
    side: str = "infectee",
) -> pl.DataFrame:
    """Non-seed infections attributed to each profile type.

    ``side="infectee"`` breaks down by the *victim's* profile — i.e. who is
    most often being infected. ``side="infector"`` breaks down by the
    *transmitter's* profile — "which profile types are driving the epidemic".
    Seeds are excluded (no infector profile to attribute).

    Columns: ``profile_type_name, count, share``.
    """
    out_schema = {
        _label_col(): pl.Utf8,
        "count": pl.UInt32,
        "share": pl.Float64,
    }
    if not _require_profile(profiles) or infections.height == 0:
        return pl.DataFrame(schema=out_schema)
    if side not in ("infectee", "infector"):
        raise ValueError(f"side must be 'infectee' or 'infector', got {side!r}")

    join_col = "person_id" if side == "infectee" else "infector_id"
    lookup = profiles.select([
        pl.col("person_id").alias(join_col),
        pl.col(_label_col()),
    ])
    non_seed = infections.filter(pl.col("infector_id") != S.INFECTOR_ID_NONE)
    if non_seed.height == 0:
        return pl.DataFrame(schema=out_schema)
    joined = non_seed.join(lookup, on=join_col, how="left")
    total = joined.height
    return (
        joined.group_by(_label_col())
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / total).alias("share"))
        .sort("count", descending=True)
    )


def reff_by_profile(
    infections: pl.DataFrame, profiles: pl.DataFrame
) -> pl.DataFrame:
    """For each profile type, how many onward infections did its agents
    produce in total and on average (conditional on being an infector).

    Columns: ``profile_type_name, infectors, secondary_cases,
    mean_secondary_per_infector``.
    """
    out_schema = {
        _label_col(): pl.Utf8,
        "infectors": pl.UInt32,
        "secondary_cases": pl.UInt32,
        "mean_secondary_per_infector": pl.Float64,
    }
    if not _require_profile(profiles) or infections.height == 0:
        return pl.DataFrame(schema=out_schema)
    non_seed = infections.filter(pl.col("infector_id") != S.INFECTOR_ID_NONE)
    if non_seed.height == 0:
        return pl.DataFrame(schema=out_schema)
    lookup = profiles.select([
        pl.col("person_id").alias("infector_id"),
        pl.col(_label_col()),
    ])
    joined = non_seed.join(lookup, on="infector_id", how="left")
    return (
        joined.group_by(_label_col())
        .agg([
            pl.len().alias("secondary_cases"),
            pl.col("infector_id").n_unique().alias("infectors"),
        ])
        .with_columns(
            (pl.col("secondary_cases") / pl.col("infectors")).alias(
                "mean_secondary_per_infector"
            )
        )
        .sort("secondary_cases", descending=True)
    )


def relationships_by_profile_pair(
    relationships: pl.DataFrame, profiles: pl.DataFrame
) -> pl.DataFrame:
    """Mixing matrix of relationships by profile-type pair.

    Columns: ``profile_type_name_a, profile_type_name_b, count``.
    """
    if not _require_profile(profiles) or relationships.height == 0:
        return pl.DataFrame(
            schema={
                "profile_type_name_a": pl.Utf8,
                "profile_type_name_b": pl.Utf8,
                "count": pl.UInt32,
            }
        )
    lookup_a = profiles.select([
        pl.col("person_id").alias("person_a"),
        pl.col(_label_col()).alias("profile_type_name_a"),
    ])
    lookup_b = profiles.select([
        pl.col("person_id").alias("person_b"),
        pl.col(_label_col()).alias("profile_type_name_b"),
    ])
    return (
        relationships.select(["person_a", "person_b"])
        .join(lookup_a, on="person_a", how="left")
        .join(lookup_b, on="person_b", how="left")
        .group_by(["profile_type_name_a", "profile_type_name_b"])
        .agg(pl.len().alias("count"))
        .sort(["profile_type_name_a", "profile_type_name_b"])
    )
