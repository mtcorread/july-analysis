"""Sexual-orientation, relationship-status and couple statistics derived
from a ``world_state.h5`` snapshot (see :mod:`july_analysis.world_state`).

All functions return plain dicts / numpy arrays — kept dependency-free
(no polars) because the inputs are themselves numpy arrays from the
streaming loader.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from july_analysis.world_state import (
    AGE_BIN_LABELS,
    ORIENTATION_LABELS,
    SEX_LABELS,
    WorldStats,
    age_bins,
    ids_to_rows,
)


PAIR_LABELS = ["M-M", "M-F", "F-F", "other"]
AGE_DIFF_EDGES = np.array([0, 1, 3, 5, 10, 15, 20, 30, 200], dtype=np.float32)


def crosstab(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> np.ndarray:
    """Dense 2-D count table. ``a`` in [0, na), ``b`` in [0, nb)."""
    flat = a.astype(np.int64) * nb + b.astype(np.int64)
    counts = np.bincount(flat, minlength=na * nb)
    return counts.reshape(na, nb)


def percent_rows(counts: np.ndarray) -> np.ndarray:
    """Row-normalised percentages. Rows that sum to 0 stay at 0."""
    total = counts.sum(axis=1, keepdims=True)
    return np.divide(counts * 100.0, total,
                     out=np.zeros_like(counts, dtype=float),
                     where=total != 0)


@dataclass
class CoupleStats:
    n_couples: int
    sex_pairing: np.ndarray            # (4,) M-M, M-F, F-F, other
    sex_pair_labels: list[str]
    orientation_pair_counts: dict[tuple[str, str], int]
    age_diff_hist: np.ndarray
    age_diff_edges: np.ndarray
    same_lgu_share: float
    same_sgu_share: float


def couple_stats(stats: WorldStats) -> CoupleStats:
    """Per-couple sex pairing, orientation pairing, age-diff and geo-overlap."""
    if stats.couple_pairs.shape[0] == 0:
        return CoupleStats(
            n_couples=0,
            sex_pairing=np.zeros(4, dtype=np.int64),
            sex_pair_labels=PAIR_LABELS,
            orientation_pair_counts={},
            age_diff_hist=np.zeros(AGE_DIFF_EDGES.size - 1, dtype=np.int64),
            age_diff_edges=AGE_DIFF_EDGES,
            same_lgu_share=0.0,
            same_sgu_share=0.0,
        )

    a_id = stats.couple_pairs[:, 0]
    b_id = stats.couple_pairs[:, 1]
    a_pos = ids_to_rows(stats, a_id)
    b_pos = ids_to_rows(stats, b_id)

    a_sex = stats.sexes[a_pos]
    b_sex = stats.sexes[b_pos]
    a_ori = stats.orientations[a_pos]
    b_ori = stats.orientations[b_pos]
    a_age = stats.ages[a_pos]
    b_age = stats.ages[b_pos]
    a_geo = stats.geo_sgu[a_pos]
    b_geo = stats.geo_sgu[b_pos]

    is_mm = (a_sex == 0) & (b_sex == 0)
    is_ff = (a_sex == 1) & (b_sex == 1)
    is_mf = ((a_sex == 0) & (b_sex == 1)) | ((a_sex == 1) & (b_sex == 0))
    is_other = ~(is_mm | is_ff | is_mf)
    sex_pairing = np.array(
        [is_mm.sum(), is_mf.sum(), is_ff.sum(), is_other.sum()],
        dtype=np.int64,
    )

    pair_counts: dict[tuple[str, str], int] = {}
    for i in range(len(ORIENTATION_LABELS)):
        for j in range(i, len(ORIENTATION_LABELS)):
            mask = ((a_ori == i) & (b_ori == j)) | ((a_ori == j) & (b_ori == i))
            n = int(mask.sum())
            if n:
                pair_counts[(ORIENTATION_LABELS[i], ORIENTATION_LABELS[j])] = n

    age_diff = np.abs(a_age - b_age)
    hist, _ = np.histogram(age_diff, bins=AGE_DIFF_EDGES)

    geo = stats.geography
    same_sgu = float((a_geo == b_geo).mean()) if a_geo.size else 0.0
    a_lgu = geo.parent_at_level(a_geo, target_level=1)
    b_lgu = geo.parent_at_level(b_geo, target_level=1)
    same_lgu = float((a_lgu == b_lgu).mean()) if a_lgu.size else 0.0

    return CoupleStats(
        n_couples=int(stats.couple_pairs.shape[0]),
        sex_pairing=sex_pairing,
        sex_pair_labels=PAIR_LABELS,
        orientation_pair_counts=pair_counts,
        age_diff_hist=hist,
        age_diff_edges=AGE_DIFF_EDGES,
        same_lgu_share=same_lgu,
        same_sgu_share=same_sgu,
    )


@dataclass
class OrientationBreakdown:
    overall: np.ndarray                 # (4,)
    by_sex: np.ndarray                  # (4, 3)
    by_age: np.ndarray                  # (n_age_bins, 4)
    by_lgu: np.ndarray                  # (n_lgu, 4)
    lgu_ids: np.ndarray                 # (n_lgu,)
    lgu_population: np.ndarray          # (n_lgu,)


def orientation_breakdown(stats: WorldStats) -> OrientationBreakdown:
    """Cross-tabulate orientation against sex / age band / LGU."""
    sex_codes = np.clip(stats.sexes, 0, 2)
    overall = np.bincount(stats.orientations, minlength=len(ORIENTATION_LABELS))
    by_sex = crosstab(stats.orientations, sex_codes, len(ORIENTATION_LABELS), 3)
    age_b = age_bins(stats.ages)
    by_age = crosstab(age_b, stats.orientations, len(AGE_BIN_LABELS),
                      len(ORIENTATION_LABELS))

    lgu = stats.geography.parent_at_level(stats.geo_sgu, target_level=1)
    unique_lgu, inv = np.unique(lgu, return_inverse=True)
    lgu_pop = np.bincount(inv)
    by_lgu = crosstab(inv, stats.orientations, unique_lgu.size,
                      len(ORIENTATION_LABELS))
    return OrientationBreakdown(
        overall=overall,
        by_sex=by_sex,
        by_age=by_age,
        by_lgu=by_lgu,
        lgu_ids=unique_lgu,
        lgu_population=lgu_pop,
    )


def relationship_status_rows(stats: WorldStats) -> list[tuple[str, int]]:
    """Return ``(status, count)`` pairs sorted by descending count.

    Status strings are whatever the world writer stored verbatim — the
    schema is upstream and isn't pinned here, so we report what's actually
    present rather than inventing a vocabulary.
    """
    return sorted(stats.rel_status_counts.items(), key=lambda kv: -kv[1])
