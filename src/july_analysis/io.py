"""Loaders for simulation_events.h5.

Design:
- Zero coupling to the C++ writer. Everything known about the schema lives in
  `schema.py`; this module only knows how to read HDF5 structured arrays and
  turn them into typed polars DataFrames.
- `EventStore` is the main entry point. Wrap one HDF5 file, call one method
  per event type. Results are cached in-memory so repeated metric calls are
  cheap. Caching is opt-out via ``cache=False`` on the constructor.
- Missing datasets are returned as empty DataFrames with the correct schema
  rather than raising, because the writer skips zero-row tables.
- For runs too large to fit in RAM, call ``EventStore.to_parquet(out_dir)``
  once, then use ``polars.scan_parquet`` for lazy queries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import polars as pl

from july_analysis import schema as S


# ----------------------------------------------------------------------------
# low-level helpers
# ----------------------------------------------------------------------------

def _decode_bytes_col(col: np.ndarray) -> np.ndarray:
    """Decode fixed-width S* byte-string columns to Python strings (vectorized)."""
    return np.char.rstrip(np.char.decode(col, "utf-8"), "\x00").astype(object)


def _structured_to_polars(arr: np.ndarray) -> pl.DataFrame:
    """Convert a 1-D numpy structured array to a polars DataFrame.

    - Bytes columns are decoded to utf-8.
    - 2-D subfields (e.g. ``extra_codes`` shape (N, k)) are unpacked into
      ``<name>_0`` .. ``<name>_{k-1}`` scalar columns.
    """
    if arr.dtype.names is None:
        raise TypeError(f"expected structured array, got dtype={arr.dtype}")
    cols: dict[str, np.ndarray] = {}
    for name in arr.dtype.names:
        sub = arr[name]
        if sub.dtype.kind == "S":
            cols[name] = _decode_bytes_col(sub)
        elif sub.ndim > 1:
            for i in range(sub.shape[1]):
                cols[f"{name}_{i}"] = sub[:, i]
        else:
            cols[name] = sub
    return pl.DataFrame(cols)


def _read_dataset(h5: h5py.File, path: str) -> pl.DataFrame | None:
    """Read one HDF5 dataset into a polars DataFrame, or None if missing."""
    if path not in h5:
        return None
    arr = h5[path][:]
    if arr.dtype.names is None:
        # 1-D non-structured (property arrays). Callers handle these directly.
        return None
    return _structured_to_polars(arr)


# ----------------------------------------------------------------------------
# path resolution
# ----------------------------------------------------------------------------

def _resolve_path(path: str | Path) -> Path:
    """Accept a direct file path or a run directory; return the merged file."""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        merged = p / S.MERGED_EVENTS_FILENAME
        if merged.exists():
            return merged
        raise FileNotFoundError(
            f"no {S.MERGED_EVENTS_FILENAME} inside directory {p}. "
            f"If you have unmerged rank files ({S.RANK_EVENTS_GLOB}), pass "
            f"one of them directly or merge them first."
        )
    raise FileNotFoundError(f"path does not exist: {p}")


# ----------------------------------------------------------------------------
# EventStore
# ----------------------------------------------------------------------------

@dataclass
class EventStoreMeta:
    """Metadata summary attached to each store (cheap to compute)."""
    path: Path
    n_infections: int
    n_relationships: int
    n_coordinated_encounters: int
    n_deaths: int
    n_symptom_changes: int
    encounter_type_registry: dict[int, str]
    activities_registry: dict[int, str]
    simulation_end_time: float | None  # latest event time observed


class EventStore:
    """Read-side wrapper around one simulation_events.h5 file.

    Usage::

        store = EventStore("runs_local/run_2001")
        infections = store.infections()           # polars.DataFrame
        enc = store.coordinated_encounters()
        tree = store.infections(decode=True)      # with encounter_type_name

    All loader methods return polars DataFrames. Missing datasets come back as
    empty frames with the expected columns — metric functions can assume their
    input exists.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        cache: bool = True,
        transmission_mode_labels: dict[int, str] | None = None,
    ) -> None:
        self.path = _resolve_path(path)
        self._cache_enabled = cache
        self._cache: dict[str, pl.DataFrame] = {}
        self.transmission_mode_labels = dict(
            transmission_mode_labels or S.DEFAULT_TRANSMISSION_MODE_LABELS
        )

    # -- context manager so callers can close explicitly if they want --------
    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.clear_cache()

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- registries ----------------------------------------------------------

    def encounter_type_registry(self) -> dict[int, str]:
        """index -> encounter-type name (e.g. ``{0: "ooe_encounter", ...}``)."""
        return self._read_registry(S.REGISTRY_ENCOUNTER_TYPES)

    def activities_registry(self) -> dict[int, str]:
        return self._read_registry(S.REGISTRY_ACTIVITIES)

    def symptom_registry(self) -> dict[int, str]:
        """index -> symptom name (e.g. ``{0: "susceptible", 2: "infectious", ...}``)."""
        return self._read_registry(S.REGISTRY_SYMPTOMS)

    def _read_registry(self, path: str) -> dict[int, str]:
        with h5py.File(self.path, "r") as h5:
            if path not in h5:
                return {}
            raw = h5[path][:]
        out: dict[int, str] = {}
        for i, v in enumerate(raw):
            if isinstance(v, bytes):
                out[i] = v.decode("utf-8", errors="replace")
            else:
                out[i] = str(v)
        return out

    # -- event tables --------------------------------------------------------

    def infections(self, *, decode: bool = True) -> pl.DataFrame:
        """Load `/events/infections`.

        Columns: ``person_id, infector_id, venue_id, time, encounter_type_id,
        transmission_mode_index, infector_symptom_id``. If ``decode=True``
        (default), add ``encounter_type_name`` and ``transmission_mode_name``.

        Note on ``encounter_type_id == 255``: this is a sentinel used for both
        (a) seed imports (``infector_id == -1``) and (b) regular venue-contact
        transmissions at household/classroom/office/etc. that don't flow through
        a coordinated-encounter path. The decoder disambiguates:
        ``infector_id == -1`` → ``"seed"``, otherwise → ``"venue_contact"``.
        """
        df = self._load(S.DATASET_INFECTIONS, _EMPTY_INFECTIONS)
        if decode and df.height > 0:
            reg = self.encounter_type_registry()
            enc_old = list(reg.keys())
            enc_new = list(reg.values())
            tm_old = list(self.transmission_mode_labels.keys())
            tm_new = list(self.transmission_mode_labels.values())
            df = df.with_columns([
                pl.when(pl.col("encounter_type_id") == S.ENCOUNTER_TYPE_SEED)
                .then(
                    pl.when(pl.col("infector_id") == S.INFECTOR_ID_NONE)
                    .then(pl.lit("seed"))
                    .otherwise(pl.lit("venue_contact"))
                )
                .otherwise(
                    pl.col("encounter_type_id").replace_strict(
                        enc_old, enc_new, default=None, return_dtype=pl.Utf8
                    )
                )
                .alias("encounter_type_name"),
                pl.col("transmission_mode_index")
                .replace_strict(tm_old, tm_new, default=None, return_dtype=pl.Utf8)
                .alias("transmission_mode_name"),
                (pl.col("infector_id") == S.INFECTOR_ID_NONE).alias("is_seed"),
            ])
        return df

    def relationships(self) -> pl.DataFrame:
        """Load `/events/relationships`.

        Columns: ``person_a, person_b, time, dissolution_time, tie_tag``.
        Empty tag strings are kept as ``""`` — in current runs they appear to
        denote LTR / untagged ties while ``"ooe"`` is the explicit
        one-off-encounter tag.
        """
        df = self._load(S.DATASET_RELATIONSHIPS, _EMPTY_RELATIONSHIPS)
        if df.height > 0:
            df = df.with_columns(
                pl.when(pl.col("tie_tag") == "")
                .then(pl.lit("ltr_or_untagged"))
                .otherwise(pl.col("tie_tag"))
                .alias("tie_tag_decoded")
            )
        return df

    def coordinated_encounters(self, *, decode: bool = True) -> pl.DataFrame:
        """Load `/events/coordinated_encounters`.

        Columns: ``person_a, person_b, time, encounter_type_id, slot``. Group
        encounters appear as one row per host↔guest pair.
        """
        df = self._load(S.DATASET_COORDINATED_ENCOUNTERS, _EMPTY_COORDINATED)
        if decode and df.height > 0:
            reg = self.encounter_type_registry()
            df = df.with_columns(
                pl.col("encounter_type_id")
                .replace_strict(list(reg.keys()), list(reg.values()), default=None, return_dtype=pl.Utf8)
                .alias("encounter_type_name")
            )
        return df

    def symptom_changes(self, *, decode: bool = True) -> pl.DataFrame:
        """Load `/events/symptom_changes`.

        Columns: ``person_id, venue_id, time, old_symptom_id, new_symptom_id``.
        If ``decode=True`` (default) and the symptom registry is present, adds
        ``old_symptom_name`` and ``new_symptom_name`` string columns.
        """
        df = self._load(S.DATASET_SYMPTOM_CHANGES, _EMPTY_SYMPTOM)
        if decode and df.height > 0:
            reg = self.symptom_registry()
            if reg:
                df = df.with_columns(
                    pl.col("old_symptom_id")
                    .replace_strict(list(reg.keys()), list(reg.values()), default=None, return_dtype=pl.Utf8)
                    .alias("old_symptom_name"),
                    pl.col("new_symptom_id")
                    .replace_strict(list(reg.keys()), list(reg.values()), default=None, return_dtype=pl.Utf8)
                    .alias("new_symptom_name"),
                )
        return df

    def deaths(self) -> pl.DataFrame:
        return self._load(S.DATASET_DEATHS, _EMPTY_DEATHS)

    def hospital_admissions(self) -> pl.DataFrame:
        return self._load(S.DATASET_HOSPITAL_ADMISSIONS, _EMPTY_HOSP_ADM)

    def icu_admissions(self) -> pl.DataFrame:
        return self._load(S.DATASET_ICU_ADMISSIONS, _EMPTY_ICU_ADM)

    def hospital_discharges(self) -> pl.DataFrame:
        return self._load(S.DATASET_HOSPITAL_DISCHARGES, _EMPTY_HOSP_DIS)

    def vaccinations(self) -> pl.DataFrame:
        return self._load(S.DATASET_VACCINATIONS, _EMPTY_VACC)

    # -- lookups -------------------------------------------------------------

    def population_summary(self) -> pl.DataFrame:
        """Load `/lookups/population_summary` — compact per-person demographics."""
        return self._load(S.DATASET_POPULATION_SUMMARY, _EMPTY_POPSUM)

    def available_population_networks(self) -> list[str]:
        """Return the list of person-network names present in this file
        (e.g. ``cohabiting_couple``, ``friendships``). Empty when the world
        was loaded without network-valued person properties.
        """
        with h5py.File(self.path, "r") as h5:
            if S.GROUP_POPULATION_NETWORKS not in h5:
                return []
            return sorted(h5[S.GROUP_POPULATION_NETWORKS].keys())

    def population_network(self, name: str) -> pl.DataFrame:
        """One row per (person_id, partner_id) pair for the given network.

        Covers the full population regardless of ``save_full_person_details`` —
        ``people_properties()`` is often filtered to the infected subset, but
        this dataset always has every person who has at least one partner in
        the network.

        Returns an empty frame if the named network isn't present.
        """
        path = f"{S.GROUP_POPULATION_NETWORKS}/{name}"
        with h5py.File(self.path, "r") as h5:
            if path not in h5:
                return pl.DataFrame(
                    {"person_id": np.array([], dtype=np.int32),
                     "partner_id": np.array([], dtype=np.int32)}
                )
            grp = h5[path]
            persons = grp["person_id"][:]
            partners = grp["partner_id"][:]
        return pl.DataFrame({"person_id": persons, "partner_id": partners})

    def people(self) -> pl.DataFrame:
        """Load `/lookups/people` — per-person detail (typically infected-only)."""
        return self._load(S.DATASET_PEOPLE, _EMPTY_PEOPLE)

    def venues(self) -> pl.DataFrame:
        return self._load(S.DATASET_VENUES, _EMPTY_VENUES)

    # -- profile facet assignments (optional) --------------------------------

    def available_profile_facets(self) -> list[str]:
        """Return the list of profile facets present in this file. Empty list
        if no facets were configured when the simulation ran.
        """
        with h5py.File(self.path, "r") as h5:
            if S.GROUP_PROFILE_ASSIGNMENTS not in h5:
                return []
            return sorted(h5[S.GROUP_PROFILE_ASSIGNMENTS].keys())

    def profile_types_registry(self, facet: str) -> dict[int, str]:
        """profile_id -> profile-type name, for a given facet.

        Returns an empty dict if the facet has no registry entry (e.g. no
        profiles were loaded). Profile ids are not guaranteed contiguous —
        the writer pads unused positions with empty strings.
        """
        path = f"{S.REGISTRY_PROFILE_TYPES_GROUP}/{facet}"
        with h5py.File(self.path, "r") as h5:
            if path not in h5:
                return {}
            arr = h5[path][:]
        out: dict[int, str] = {}
        for i, v in enumerate(arr):
            name = v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
            if name:  # skip empty padding slots
                out[i] = name
        return out

    def profile_assignments(
        self, facet: str, *, decode: bool = True
    ) -> pl.DataFrame:
        """One row per agent with their profile facet state.

        Columns: ``person_id`` + one column per field in the facet's
        ``field_defs`` (e.g. ``profile_id``, ``max_partners``,
        ``ooe_probability``, ``group_sex_willingness``,
        ``cooldown_duration_days``, ``actively_seeking``, etc.). If
        ``decode=True`` (default) and a
        ``/metadata/registries/profile_types/<facet>`` registry exists, a
        ``profile_type_name`` column is added.

        Returns an empty frame (just ``person_id: Int32``) if the facet has
        no entry in the file. Check ``available_profile_facets()`` first if
        you need to discover what's present.
        """
        cache_key = f"profile_assignments/{facet}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        group_path = f"{S.GROUP_PROFILE_ASSIGNMENTS}/{facet}"
        with h5py.File(self.path, "r") as h5:
            if group_path not in h5:
                empty = pl.DataFrame({"person_id": np.array([], dtype=np.int32)})
                if self._cache_enabled:
                    self._cache[cache_key] = empty
                return empty
            grp = h5[group_path]
            cols: dict[str, np.ndarray] = {}
            for name in grp.keys():
                arr = grp[name][:]
                if arr.dtype.kind == "S" or arr.dtype == object:
                    cols[name] = _decode_bytes_col(arr)
                else:
                    cols[name] = arr
        # person_id first for readability
        ordered = {"person_id": cols.pop("person_id")} if "person_id" in cols else {}
        ordered.update(cols)
        df = pl.DataFrame(ordered)
        if decode and "profile_id" in df.columns:
            reg = self.profile_types_registry(facet)
            if reg:
                df = df.with_columns(
                    pl.col("profile_id")
                    .map_elements(
                        lambda i: reg.get(int(i), "" if int(i) == 0 else f"unknown_{int(i)}"),
                        return_dtype=pl.Utf8,
                    )
                    .alias("profile_type_name")
                )
        if self._cache_enabled:
            self._cache[cache_key] = df
        return df

    def people_properties(self) -> pl.DataFrame:
        """Return a DataFrame with one row per /lookups/people row, plus all
        property columns (ethnicity, sexual_orientation, ...). Indexed to
        /lookups/people by row position; the returned frame includes a
        ``person_id`` column joined from /lookups/people for convenience.
        """
        if "people_properties" in self._cache:
            return self._cache["people_properties"]
        with h5py.File(self.path, "r") as h5:
            if S.GROUP_PEOPLE_PROPERTIES not in h5 or S.DATASET_PEOPLE not in h5:
                out = pl.DataFrame({"person_id": np.array([], dtype=np.int32)})
                if self._cache_enabled:
                    self._cache["people_properties"] = out
                return out
            people_ids = h5[S.DATASET_PEOPLE]["person_id"][:]
            cols: dict[str, np.ndarray] = {"person_id": people_ids}
            grp = h5[S.GROUP_PEOPLE_PROPERTIES]
            for name in grp.keys():
                arr = grp[name][:]
                cols[name] = _decode_bytes_col(arr)
        out = pl.DataFrame(cols)
        if self._cache_enabled:
            self._cache["people_properties"] = out
        return out

    # -- joined views --------------------------------------------------------

    def infections_with_demographics(self) -> pl.DataFrame:
        """Infections joined against ``population_summary`` for both the
        infectee (``person_id``) and the infector (``infector_id``). Columns
        from population_summary are prefixed ``infectee_`` and ``infector_``.

        Rows where the infector is not in the population_summary (e.g.
        infector_id == -1 for seed events) keep null demographic fields.
        """
        inf = self.infections(decode=True)
        pop = self.population_summary()
        if pop.height == 0 or inf.height == 0:
            return inf
        pop_r = pop.rename({c: f"infectee_{c}" for c in pop.columns if c != "person_id"})
        out = inf.join(pop_r, on="person_id", how="left")
        pop_l = pop.rename(
            {c: f"infector_{c}" for c in pop.columns if c != "person_id"}
        ).rename({"person_id": "infector_id"})
        out = out.join(pop_l, on="infector_id", how="left")
        return out

    # -- I/O utilities -------------------------------------------------------

    def to_parquet(self, out_dir: str | Path, *, decode: bool = True) -> Path:
        """Dump every table as a Parquet file into ``out_dir``. Intended for
        large runs where subsequent analysis should use ``pl.scan_parquet``
        rather than re-reading HDF5. Idempotent — overwrites existing files.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        loaders = {
            "infections": lambda: self.infections(decode=decode),
            "relationships": self.relationships,
            "coordinated_encounters": lambda: self.coordinated_encounters(decode=decode),
            "symptom_changes": self.symptom_changes,
            "deaths": self.deaths,
            "hospital_admissions": self.hospital_admissions,
            "icu_admissions": self.icu_admissions,
            "hospital_discharges": self.hospital_discharges,
            "vaccinations": self.vaccinations,
            "population_summary": self.population_summary,
            "people": self.people,
            "people_properties": self.people_properties,
            "venues": self.venues,
        }
        for name, fn in loaders.items():
            df = fn()
            df.write_parquet(out / f"{name}.parquet")
        return out

    def meta(self) -> EventStoreMeta:
        """Cheap summary of the store — used by the CLI/report header."""
        reg = self.encounter_type_registry()
        acts = self.activities_registry()
        end_time = None
        with h5py.File(self.path, "r") as h5:
            n_inf = h5[S.DATASET_INFECTIONS].shape[0] if S.DATASET_INFECTIONS in h5 else 0
            n_rel = h5[S.DATASET_RELATIONSHIPS].shape[0] if S.DATASET_RELATIONSHIPS in h5 else 0
            n_enc = h5[S.DATASET_COORDINATED_ENCOUNTERS].shape[0] if S.DATASET_COORDINATED_ENCOUNTERS in h5 else 0
            n_death = h5[S.DATASET_DEATHS].shape[0] if S.DATASET_DEATHS in h5 else 0
            n_symp = h5[S.DATASET_SYMPTOM_CHANGES].shape[0] if S.DATASET_SYMPTOM_CHANGES in h5 else 0
            for ds in (S.DATASET_INFECTIONS, S.DATASET_COORDINATED_ENCOUNTERS, S.DATASET_SYMPTOM_CHANGES):
                if ds in h5 and h5[ds].shape[0] > 0:
                    t = float(h5[ds]["time"][-1])
                    end_time = max(end_time, t) if end_time is not None else t
        return EventStoreMeta(
            path=self.path,
            n_infections=n_inf,
            n_relationships=n_rel,
            n_coordinated_encounters=n_enc,
            n_deaths=n_death,
            n_symptom_changes=n_symp,
            encounter_type_registry=reg,
            activities_registry=acts,
            simulation_end_time=end_time,
        )

    # -- internals -----------------------------------------------------------

    def _load(self, dataset_path: str, empty_schema: dict) -> pl.DataFrame:
        if dataset_path in self._cache:
            return self._cache[dataset_path]
        with h5py.File(self.path, "r") as h5:
            df = _read_dataset(h5, dataset_path)
        if df is None:
            df = pl.DataFrame(schema=empty_schema)
        if self._cache_enabled:
            self._cache[dataset_path] = df
        return df


# ----------------------------------------------------------------------------
# empty-schema constants (used when a dataset is absent)
# ----------------------------------------------------------------------------

_EMPTY_INFECTIONS = {
    "person_id": pl.Int32,
    "infector_id": pl.Int32,
    "venue_id": pl.Int32,
    "time": pl.Float64,
    "encounter_type_id": pl.UInt8,
    "transmission_mode_index": pl.UInt8,
    "infector_symptom_id": pl.UInt16,
}
_EMPTY_RELATIONSHIPS = {
    "person_a": pl.Int32,
    "person_b": pl.Int32,
    "time": pl.Float64,
    "dissolution_time": pl.Float64,
    "tie_tag": pl.Utf8,
}
_EMPTY_COORDINATED = {
    "person_a": pl.Int32,
    "person_b": pl.Int32,
    "time": pl.Float64,
    "encounter_type_id": pl.UInt8,
    "slot": pl.Int32,
}
_EMPTY_SYMPTOM = {
    "person_id": pl.Int32,
    "venue_id": pl.Int32,
    "time": pl.Float64,
    "old_symptom_id": pl.UInt16,
    "new_symptom_id": pl.UInt16,
}
_EMPTY_DEATHS = {"person_id": pl.Int32, "venue_id": pl.Int32, "time": pl.Float64}
_EMPTY_HOSP_ADM = {
    "person_id": pl.Int32,
    "hospital_id": pl.Int32,
    "time": pl.Float64,
    "reason": pl.Utf8,
}
_EMPTY_ICU_ADM = {"person_id": pl.Int32, "hospital_id": pl.Int32, "time": pl.Float64}
_EMPTY_HOSP_DIS = {
    "person_id": pl.Int32,
    "hospital_id": pl.Int32,
    "time": pl.Float64,
    "outcome": pl.Utf8,
}
_EMPTY_VACC = {
    "person_id": pl.Int32,
    "vaccine_type": pl.Utf8,
    "dose_index": pl.Int32,
    "time": pl.Float64,
}
_EMPTY_POPSUM = {
    "person_id": pl.Int32,
    "age_group": pl.UInt8,
    "sex_code": pl.UInt8,
    "schedule_type_code": pl.UInt8,
    "reserved": pl.UInt8,
    "geo_unit_id": pl.Int32,
    "extra_codes_0": pl.UInt8,
    "extra_codes_1": pl.UInt8,
    "extra_codes_2": pl.UInt8,
    "extra_codes_3": pl.UInt8,
}
_EMPTY_PEOPLE = {
    "person_id": pl.Int32,
    "age": pl.Float64,
    "sex": pl.Utf8,
    "geo_unit_id": pl.Int32,
    "is_dead": pl.Int32,
    "death_time": pl.Float64,
    "schedule_type": pl.Utf8,
    "num_activities": pl.Int32,
    "num_residence_venues": pl.Int32,
    "num_primary_activities": pl.Int32,
    "num_leisure_venues": pl.Int32,
    "num_medical_facilities": pl.Int32,
}
_EMPTY_VENUES = {
    "venue_id": pl.Int32,
    "name": pl.Utf8,
    "type": pl.Utf8,
    "geo_unit_id": pl.Int32,
    "n_subsets": pl.Int32,
}


def load_run(path: str | Path, **kwargs) -> EventStore:
    """Convenience constructor: ``load_run("path/to/run_dir")``."""
    return EventStore(path, **kwargs)
