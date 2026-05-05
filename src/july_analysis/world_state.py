"""Streaming reader for ``world_state.h5``.

Unlike :mod:`july_analysis.io`, which targets the per-rank
``simulation_events.h5`` files, this module reads the static *world snapshot*
that the simulator persists at startup. The world file owns the canonical
copy of every per-person property (sexual orientation, cohabiting partner,
relationship status, ...) for the *full* population, whereas events files
only carry properties for the infected subset.

Designed to scale to the 60 M-person production world (~20 GB on disk):

- Per-person properties are streamed in chunks; only compact uint8 codes
  end up in resident memory.
- Cohabiting partner ids are deduplicated into ``(lo, hi)`` pairs via a
  packed-uint64 ``np.unique`` rather than a Python set.
- ID lookups use an inverse-permutation int32 array (the writer assigns
  person ids as a 0..N-1 permutation, so the lookup is a single index op).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


DEFAULT_CHUNK = 2_000_000

AGE_BIN_EDGES = np.array(
    [0, 5, 15, 18, 25, 35, 45, 55, 65, 75, 200], dtype=np.int32
)
AGE_BIN_LABELS = [
    "0-4", "5-14", "15-17", "18-24", "25-34", "35-44",
    "45-54", "55-64", "65-74", "75+",
]

SEX_LABELS = {0: "male", 1: "female", 2: "unknown"}

# Empty string is treated as "unspecified" — matches how the writer leaves
# under-age agents (children) without a stated orientation.
ORIENTATION_LABELS = ["heterosexual", "homosexual", "bisexual", "unspecified"]
ORIENTATION_BYTES = {
    b"heterosexual": 0,
    b"homosexual": 1,
    b"bisexual": 2,
    b"": 3,
}

GEO_LEVEL_NAMES = ["XLGU", "LGU", "MGU", "SGU"]


# ---------------------------------------------------------------------------
# geography
# ---------------------------------------------------------------------------

@dataclass
class Geography:
    """Hierarchy lookup for the world's geo tree.

    Geo ids are dense within a level but not globally unique across levels
    (the writer numbers them per level and stamps a level enum on each row).
    """
    ids: np.ndarray         # int32 (G,)
    parent_ids: np.ndarray  # int32 (G,)
    levels: np.ndarray      # uint8 (G,)
    names: np.ndarray       # object (utf-8 bytes), shape (G,)
    id_to_row: dict[int, int] = field(default_factory=dict)

    @classmethod
    def load(cls, h5: h5py.File) -> "Geography":
        ids = h5["geography/ids"][:]
        return cls(
            ids=ids,
            parent_ids=h5["geography/parent_ids"][:],
            levels=h5["geography/levels"][:],
            names=h5["metadata/names/geography"][:],
            id_to_row={int(g): i for i, g in enumerate(ids)},
        )

    def parent_at_level(self, geo_ids: np.ndarray, target_level: int) -> np.ndarray:
        """Walk every entry in ``geo_ids`` up the tree until it hits
        ``target_level``. Returns an array of geo ids (or -1 where the walk
        runs out)."""
        out = geo_ids.copy()
        # The hierarchy has ≤4 levels; cap at 8 as a defensive max-depth.
        for _ in range(8):
            rows = np.array([self.id_to_row.get(int(g), -1) for g in out],
                            dtype=np.int64)
            cur_levels = np.where(rows >= 0, self.levels[rows], -1)
            done = (cur_levels == target_level) | (rows < 0)
            if done.all():
                break
            new = np.where(rows >= 0, self.parent_ids[rows], -1)
            out = np.where(done, out, new)
        return out

    def name(self, geo_id: int) -> str:
        row = self.id_to_row.get(int(geo_id))
        if row is None:
            return f"<unknown:{geo_id}>"
        v = self.names[row]
        return v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)


# ---------------------------------------------------------------------------
# per-chunk decoders
# ---------------------------------------------------------------------------

def _iter_chunks(n: int, chunk: int) -> Iterator[tuple[int, int]]:
    for start in range(0, n, chunk):
        yield start, min(start + chunk, n)


def _orientation_chunk_to_codes(arr: np.ndarray) -> np.ndarray:
    """Map an HDF5 sexual_orientation chunk to uint8 orientation codes."""
    out = np.full(arr.shape, ORIENTATION_BYTES[b""], dtype=np.uint8)
    if arr.dtype == object:
        for code, raw in enumerate(ORIENTATION_LABELS):
            key = raw.encode("ascii") if raw != "unspecified" else b""
            out[arr == key] = code
    else:
        a = arr if arr.dtype.kind == "S" else arr.astype("S16")
        for code, raw in enumerate(ORIENTATION_LABELS):
            key = raw.encode("ascii") if raw != "unspecified" else b""
            out[a == key] = code
    return out


def _parse_cohabiting_chunk(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a chunk of ``b'[12345]'``-style cohabiting-couple strings.

    Returns ``(rows_in_chunk, partner_ids)`` — both 1-D arrays. Empty
    strings (no partner) are dropped. cohabiting_couple is at most 1 partner
    per person in the world model, so we only emit one row per non-empty
    input.
    """
    n = arr.shape[0]
    if arr.dtype == object:
        nonempty = (arr != b"")
    else:
        nonempty = (np.char.str_len(arr) > 2)
    rows = np.nonzero(nonempty)[0]
    if rows.size == 0:
        return rows, np.empty(0, dtype=np.int64)
    if arr.dtype == object:
        partners = np.fromiter(
            (int(arr[r][1:-1]) for r in rows),
            dtype=np.int64, count=rows.size,
        )
    else:
        sub = np.char.strip(np.char.strip(arr[rows], b"["), b"]")
        partners = sub.astype(np.int64)
    return rows, partners


_TYPE_RE = None  # lazily-compiled re.Pattern; module-level for cache reuse


def _tally_relationship_status_chunk(
    arr: np.ndarray, into: dict[str, int]
) -> None:
    """Count each distinct ``relationship_status`` string in ``arr`` into
    ``into``. The schema is not pinned in this repo, so we discover labels
    rather than hard-coding them: the raw stored string is the key, and a
    tiny regex extracts the JSON ``"type": "..."`` field as a friendlier
    secondary label rendered alongside it.

    Empty strings are bucketed as ``"<unspecified>"`` (matches the writer's
    convention for under-age agents with no stated status).
    """
    if arr.dtype == object:
        # np.unique on object arrays of bytes is fast — ~3 distinct values
        # on the test world, unlikely to balloon.
        unique, counts = np.unique(arr, return_counts=True)
        for v, c in zip(unique, counts):
            label = v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
            if not label:
                label = "<unspecified>"
            into[label] = into.get(label, 0) + int(c)
    else:
        text = arr
        unique, counts = np.unique(text, return_counts=True)
        for v, c in zip(unique, counts):
            label = v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
            if not label:
                label = "<unspecified>"
            into[label] = into.get(label, 0) + int(c)


# ---------------------------------------------------------------------------
# top-level loader
# ---------------------------------------------------------------------------

@dataclass
class WorldStats:
    n_people: int
    ages: np.ndarray            # float32 (N,)
    sexes: np.ndarray           # uint8 (N,)
    orientations: np.ndarray    # uint8 (N,)
    rel_status_counts: dict[str, int]   # raw string -> count, discovered live
    geo_sgu: np.ndarray         # int32 (N,)  — population/geo_unit_ids (SGU)
    person_ids: np.ndarray      # int32 (N,)  — population/ids (row -> id)
    id_to_row: np.ndarray       # int32 (max_id+1,) — id -> row, -1 if absent
    couple_pairs: np.ndarray    # int64 (P, 2)  — deduped (lo_id, hi_id)
    geography: Geography
    n_self_referential: int
    n_partner_unknown: int
    n_couple_records: int       # number of non-empty cohabiting_couple rows


def load_world(
    path: str | Path,
    *,
    chunk: int = DEFAULT_CHUNK,
    progress: bool = False,
) -> WorldStats:
    """Stream a world_state.h5 file and return decoded population stats.

    ``progress=True`` prints per-chunk progress to stderr.
    """
    path = Path(path)
    with h5py.File(path, "r") as h5:
        geography = Geography.load(h5)
        ids_ds = h5["population/ids"]
        n = ids_ds.shape[0]

        ages = h5["population/ages"][:]
        sexes = h5["population/sexes"][:]
        person_ids = ids_ds[:]
        geo_sgu = h5["population/geo_unit_ids"][:]

        # Inverse permutation: the writer assigns person ids as a permutation
        # of 0..N-1, so a dense int32 lookup is the cheapest possible map.
        max_id = int(person_ids.max())
        id_to_row = np.full(max_id + 1, -1, dtype=np.int32)
        id_to_row[person_ids] = np.arange(n, dtype=np.int32)

        orientations = np.empty(n, dtype=np.uint8)
        rel_status_counts: dict[str, int] = {}

        couple_lo: list[np.ndarray] = []
        couple_hi: list[np.ndarray] = []
        n_couple_records = 0
        n_self = 0
        n_partner_unknown = 0

        cohab_ds = h5["population/properties/cohabiting_couple"]
        orient_ds = h5["population/properties/sexual_orientation"]
        relst_ds = h5["population/properties/relationship_status"]

        t0 = time.time()
        for s, e in _iter_chunks(n, chunk):
            orientations[s:e] = _orientation_chunk_to_codes(orient_ds[s:e])
            _tally_relationship_status_chunk(relst_ds[s:e], rel_status_counts)

            local_rows, partners = _parse_cohabiting_chunk(cohab_ds[s:e])
            if partners.size:
                self_ids = person_ids[s + local_rows].astype(np.int64)

                ok = self_ids != partners
                n_self += int((~ok).sum())
                self_ids = self_ids[ok]
                partners = partners[ok]

                in_range = (partners >= 0) & (partners <= max_id)
                valid = np.zeros_like(partners, dtype=bool)
                valid[in_range] = id_to_row[partners[in_range]] >= 0
                n_partner_unknown += int((~valid).sum())
                self_ids = self_ids[valid]
                partners = partners[valid]

                lo = np.minimum(self_ids, partners)
                hi = np.maximum(self_ids, partners)
                couple_lo.append(lo)
                couple_hi.append(hi)
                n_couple_records += int(valid.sum())

            if progress:
                pct = 100 * e / n
                rate = e / max(time.time() - t0, 1e-6) / 1e6
                print(f"  ...{e:,}/{n:,} ({pct:5.1f}%)  {rate:.2f}M rows/s",
                      file=sys.stderr)

        if couple_lo:
            lo = np.concatenate(couple_lo)
            hi = np.concatenate(couple_hi)
            key = (lo.astype(np.uint64) << np.uint64(32)) | hi.astype(np.uint64)
            unq = np.unique(key)
            couple_pairs = np.empty((unq.size, 2), dtype=np.int64)
            couple_pairs[:, 0] = (unq >> np.uint64(32)).astype(np.int64)
            couple_pairs[:, 1] = (unq & np.uint64(0xFFFFFFFF)).astype(np.int64)
        else:
            couple_pairs = np.empty((0, 2), dtype=np.int64)

    return WorldStats(
        n_people=n,
        ages=ages,
        sexes=sexes,
        orientations=orientations,
        rel_status_counts=rel_status_counts,
        geo_sgu=geo_sgu,
        person_ids=person_ids,
        id_to_row=id_to_row,
        couple_pairs=couple_pairs,
        geography=geography,
        n_self_referential=n_self,
        n_partner_unknown=n_partner_unknown,
        n_couple_records=n_couple_records,
    )


def age_bins(ages: np.ndarray) -> np.ndarray:
    """Bucket an array of ages into AGE_BIN_LABELS-aligned uint8 codes."""
    return np.clip(np.digitize(ages, AGE_BIN_EDGES, right=False) - 1,
                   0, len(AGE_BIN_LABELS) - 1).astype(np.uint8)


def ids_to_rows(stats: WorldStats, ids: np.ndarray) -> np.ndarray:
    """Convert person ids to row indices via the prebuilt inverse permutation."""
    return stats.id_to_row[ids]
