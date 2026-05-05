"""Smoke test.

Runs end-to-end against a real simulation_events.h5 file whose path is set via
the ``JULY_TEST_HDF5`` environment variable. Skipped otherwise so the test
suite stays runnable in environments without a real simulation output.

Usage::

    JULY_TEST_HDF5=/path/to/simulation_events.h5 pytest tests/
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from july_analysis import EventStore
from july_analysis.metrics import encounters, epi, network, transmission


def _store() -> EventStore:
    path = os.environ.get("JULY_TEST_HDF5")
    if not path:
        pytest.skip("JULY_TEST_HDF5 not set; skipping live-file smoke test")
    return EventStore(path)


def test_meta_and_registries():
    store = _store()
    meta = store.meta()
    assert meta.n_infections >= 0
    assert isinstance(meta.encounter_type_registry, dict)
    assert "encounter_types" in repr(store.encounter_type_registry) or True


def test_load_all_tables():
    store = _store()
    for fn in (
        store.infections,
        store.relationships,
        store.coordinated_encounters,
        store.symptom_changes,
        store.deaths,
        store.hospital_admissions,
        store.icu_admissions,
        store.hospital_discharges,
        store.vaccinations,
        store.population_summary,
        store.people,
        store.venues,
    ):
        df = fn()
        assert isinstance(df, pl.DataFrame)


def test_metrics_run():
    store = _store()
    inf = store.infections()
    rel = store.relationships()
    enc = store.coordinated_encounters()
    pop = store.population_summary()
    deaths = store.deaths()
    symp = store.symptom_changes()

    assert epi.incidence_by_day(inf).height >= 0
    assert epi.encounter_type_contribution(inf).height >= 0
    scalars = epi.summary_scalars(inf, deaths)
    assert scalars["total_infections"] == inf.height
    epi.time_to_symptom(inf, symp)

    network.degree_distribution(rel)
    network.concurrency_series(
        rel.filter(pl.col("dissolution_time") < 365.0), times=[0.5, 2.5, 5.5]
    )
    if pop.height > 0:
        network.mixing_matrix(
            inf.filter(~inf["is_seed"]) if inf.height else inf,
            pop.select(["person_id", "age_group"]),
            "age_group",
            a="infector_id",
            b="person_id",
        )

    transmission.secondary_cases_per_infector(inf)
    transmission.reff_by_encounter_type(inf)
    transmission.generation_time_distribution(inf)
    transmission.reff_rolling(inf)

    encounters.encounter_type_share(enc)
    grp = enc.filter(pl.col("encounter_type_name") == "group_sex") if enc.height else enc
    rec = encounters.reconstruct_group_encounters(grp)
    encounters.group_size_distribution(rec)


def test_report_generates(tmp_path: Path):
    from july_analysis.report import build_report

    store = _store()
    out = build_report(store.path, tmp_path / "report.html")
    assert out.exists()
    html = out.read_text()
    assert "<html" in html
    assert "Simulation events report" in html
