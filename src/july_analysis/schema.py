"""HDF5 schema constants for simulation_events.h5.

These are derived from the on-disk structure and kept here so the package has
zero coupling to the C++ codebase that produced the file. If the writer
changes, update this module and nothing else.
"""
from __future__ import annotations

# --- dataset paths -----------------------------------------------------------

EVENTS_GROUP = "/events"
LOOKUPS_GROUP = "/lookups"
REGISTRIES_GROUP = "/metadata/registries"

DATASET_INFECTIONS = f"{EVENTS_GROUP}/infections"
DATASET_RELATIONSHIPS = f"{EVENTS_GROUP}/relationships"
DATASET_COORDINATED_ENCOUNTERS = f"{EVENTS_GROUP}/coordinated_encounters"
DATASET_SYMPTOM_CHANGES = f"{EVENTS_GROUP}/symptom_changes"
DATASET_DEATHS = f"{EVENTS_GROUP}/deaths"
DATASET_HOSPITAL_ADMISSIONS = f"{EVENTS_GROUP}/hospital_admissions"
DATASET_ICU_ADMISSIONS = f"{EVENTS_GROUP}/icu_admissions"
DATASET_HOSPITAL_DISCHARGES = f"{EVENTS_GROUP}/hospital_discharges"
DATASET_VACCINATIONS = f"{EVENTS_GROUP}/vaccinations"

DATASET_PEOPLE = f"{LOOKUPS_GROUP}/people"
DATASET_POPULATION_SUMMARY = f"{LOOKUPS_GROUP}/population_summary"
DATASET_VENUES = f"{LOOKUPS_GROUP}/venues"
GROUP_PEOPLE_PROPERTIES = f"{LOOKUPS_GROUP}/people_properties"

REGISTRY_ENCOUNTER_TYPES = f"{REGISTRIES_GROUP}/encounter_types"
REGISTRY_ACTIVITIES = f"{REGISTRIES_GROUP}/activities"
REGISTRY_SYMPTOMS = f"{REGISTRIES_GROUP}/symptoms"
# Registry group for profile types — one var-length-string dataset per facet,
# indexed by profile_id. Absent when no facets are configured.
REGISTRY_PROFILE_TYPES_GROUP = f"{REGISTRIES_GROUP}/profile_types"

# Per-facet profile assignments — one subgroup per facet, containing
# ``person_id`` plus one 1-D array per derived field_def. Absent when no
# facets are configured.
GROUP_PROFILE_ASSIGNMENTS = f"{LOOKUPS_GROUP}/profile_assignments"

# Full-population person-network tables. One subgroup per network name (e.g.
# ``cohabiting_couple``, ``friendships``), each with parallel ``person_id``
# and ``partner_id`` int32 datasets. Absent when no network properties were
# loaded from the world file.
GROUP_POPULATION_NETWORKS = f"{LOOKUPS_GROUP}/population_networks"

# Datasets that may legitimately be absent from a file when zero rows were
# written (the writer skips empty tables).
OPTIONAL_DATASETS = frozenset({
    DATASET_HOSPITAL_DISCHARGES,
    DATASET_VACCINATIONS,
    DATASET_HOSPITAL_ADMISSIONS,
    DATASET_ICU_ADMISSIONS,
    DATASET_DEATHS,
    DATASET_SYMPTOM_CHANGES,
    DATASET_INFECTIONS,
    DATASET_RELATIONSHIPS,
    DATASET_COORDINATED_ENCOUNTERS,
})

# --- sentinels ---------------------------------------------------------------

# encounter_type_id == 255 is written on infection-seed events (no encounter).
ENCOUNTER_TYPE_SEED = 255

# venue_id == -999 is the infection-seed pseudo-venue.
VENUE_ID_SEED = -999

# infector_id == -1 on seeded (imported) infections.
INFECTOR_ID_NONE = -1

# --- default transmission-mode label map -------------------------------------
# The sim does not embed a registry for transmission modes, so callers can
# override this. In current runs only {0, 1} are observed.
DEFAULT_TRANSMISSION_MODE_LABELS: dict[int, str] = {
    0: "mode_0",
    1: "mode_1",
}

# Filename patterns the loader recognises inside a run directory.
MERGED_EVENTS_FILENAME = "simulation_events.h5"
RANK_EVENTS_GLOB = "simulation_events_rank*.h5"
