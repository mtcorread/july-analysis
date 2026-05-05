"""Analytical groupings of encounter types.

Groupings are declared in YAML on the analyser side (not in the simulator
config) because "sexual" / "leisure" / etc. are reporting categories, not
runtime concepts. They are validated against the events file's
encounter-type registry on load — any name in the YAML that the events file
doesn't know about raises a ``UnknownEncounterTypesError`` listing the
offenders and the actual registry contents.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Mapping

import yaml


@dataclass(frozen=True)
class Grouping:
    name: str
    display_name: str
    encounter_types: tuple[str, ...]


class UnknownEncounterTypesError(ValueError):
    """Raised when a groupings YAML names encounter types absent from the
    events file's registry. Listed alongside the actual registry so the user
    can immediately see what was renamed or removed."""


def load_groupings(path: str | Path | None = None) -> dict[str, Grouping]:
    """Load groupings from a YAML file. ``None`` loads the package default
    (``configs/groupings.yaml`` shipped with july-analysis).
    """
    if path is None:
        text = resources.files("july_analysis.configs").joinpath(
            "groupings.yaml"
        ).read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")

    raw = yaml.safe_load(text) or {}
    block = raw.get("groupings") or {}
    out: dict[str, Grouping] = {}
    for name, body in block.items():
        body = body or {}
        out[name] = Grouping(
            name=name,
            display_name=body.get("display_name", name),
            encounter_types=tuple(body.get("encounter_types") or ()),
        )
    return out


def validate_against_registry(
    groupings: Mapping[str, Grouping],
    registry: Mapping[int, str],
) -> None:
    """Raise ``UnknownEncounterTypesError`` if any grouping references an
    encounter type not in the events file's registry. The error lists every
    offender across all groupings, plus the registry contents."""
    known = set(registry.values())
    bad: list[tuple[str, str]] = []
    for g in groupings.values():
        for t in g.encounter_types:
            if t not in known:
                bad.append((g.name, t))
    if not bad:
        return
    by_group: dict[str, list[str]] = {}
    for gname, t in bad:
        by_group.setdefault(gname, []).append(t)
    parts = [f"  groupings.{g}: {ts}" for g, ts in by_group.items()]
    raise UnknownEncounterTypesError(
        "groupings YAML references encounter types absent from the events "
        "file's registry:\n"
        + "\n".join(parts)
        + f"\nregistry contains: {sorted(known)}"
    )
