"""Multi-run aggregation for parameter sweeps / replicate ensembles."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import polars as pl

from july_analysis import schema as S
from july_analysis.io import EventStore
from july_analysis.metrics import epi


def find_runs(root: str | Path) -> list[Path]:
    """Walk ``root`` and return every directory that contains a
    ``simulation_events.h5`` file. Order is sorted by path for reproducibility.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    matches = sorted(root.rglob(S.MERGED_EVENTS_FILENAME))
    return [m.parent for m in matches]


def collect_runs(
    root: str | Path,
    *,
    run_id_fn=lambda p: p.name,
    extra_scalars: Iterable[str] = (),
) -> pl.DataFrame:
    """For every run under ``root``, compute a fixed set of scalar summaries
    and return them as one tidy DataFrame.

    Columns: ``run_id, path, total_infections, seed_infections,
    transmission_events, total_deaths, peak_incidence_day, peak_incidence,
    n_relationships, n_coordinated_encounters, end_time``.

    Caveats:
    - Uses ``EventStore.meta()`` and ``epi.summary_scalars`` — both are cheap
      on a per-run basis (count-only), so this scales to hundreds of runs.
    - Fails soft: runs that raise are recorded with ``error`` instead of
      metrics, so one broken run does not break the whole sweep.
    """
    run_dirs = find_runs(root)
    rows = []
    for d in run_dirs:
        entry: dict = {"run_id": run_id_fn(d), "path": str(d)}
        try:
            store = EventStore(d)
            m = store.meta()
            inf = store.infections()
            deaths = store.deaths()
            scalars = epi.summary_scalars(inf, deaths)
            entry.update(scalars)
            entry["n_relationships"] = m.n_relationships
            entry["n_coordinated_encounters"] = m.n_coordinated_encounters
            entry["end_time"] = m.simulation_end_time or 0.0
            entry["error"] = None
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(entry)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def compare_scenarios(
    summary: pl.DataFrame, *, by: str, metrics: Iterable[str] = ("transmission_events", "peak_incidence")
) -> pl.DataFrame:
    """Aggregate a ``collect_runs`` table across replicates. ``by`` should be
    a column the caller already attached to ``summary`` identifying the
    scenario / parameter set (e.g. via ``with_columns``). Returns per-scenario
    median + IQR for each metric.
    """
    aggs: list[pl.Expr] = []
    for m in metrics:
        aggs.extend([
            pl.col(m).median().alias(f"{m}_median"),
            pl.col(m).quantile(0.25).alias(f"{m}_q25"),
            pl.col(m).quantile(0.75).alias(f"{m}_q75"),
            pl.len().alias("n_runs"),
        ])
    return summary.group_by(by).agg(aggs).sort(by)
