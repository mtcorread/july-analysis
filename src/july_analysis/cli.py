"""Command-line entry point.

Subcommands:
    july-analyze inspect    --events FILE                  — schema + row counts
    july-analyze report     --events FILE [--world FILE] -o out.html
    july-analyze to-parquet --events FILE -o DIR           — dump all tables as parquet
    july-analyze collect-runs <root> -o out.csv            — sweep summary table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from july_analysis import EventStore, __version__
from july_analysis.io import load_run
from july_analysis.runs import collect_runs


def _parse_tm_map(s: str | None) -> dict[int, str] | None:
    if not s:
        return None
    out: dict[int, str] = {}
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        k, _, v = piece.partition("=")
        out[int(k)] = v
    return out


def _cmd_inspect(args: argparse.Namespace) -> int:
    store = load_run(args.events, transmission_mode_labels=_parse_tm_map(args.transmission_modes))
    meta = store.meta()
    print(f"path                 {meta.path}")
    print(f"simulation end time  {meta.simulation_end_time}")
    print(f"infections           {meta.n_infections}")
    print(f"relationships        {meta.n_relationships}")
    print(f"coord. encounters    {meta.n_coordinated_encounters}")
    print(f"deaths               {meta.n_deaths}")
    print(f"symptom changes      {meta.n_symptom_changes}")
    print("\nencounter_types:")
    for k, v in meta.encounter_type_registry.items():
        print(f"  {k}: {v}")
    print("\nactivities:")
    for k, v in meta.activities_registry.items():
        print(f"  {k}: {v}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from july_analysis.report import build_report

    out = build_report(args.events, args.output, world_path=args.world)
    print(f"wrote {out}")
    return 0


def _cmd_to_parquet(args: argparse.Namespace) -> int:
    store = load_run(args.events, transmission_mode_labels=_parse_tm_map(args.transmission_modes))
    out = store.to_parquet(args.output)
    print(f"wrote parquet files into {out}")
    return 0


def _cmd_collect_runs(args: argparse.Namespace) -> int:
    df = collect_runs(args.root)
    if df.height == 0:
        print(f"no runs found under {args.root}", file=sys.stderr)
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        df.write_parquet(output)
    else:
        df.write_csv(output)
    print(f"wrote {df.height} rows to {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="july-analyze")
    parser.add_argument("--version", action="version", version=f"july-analysis {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    events_kw = dict(
        required=True,
        help="run directory or simulation_events.h5 path",
    )

    p_inspect = sub.add_parser("inspect", help="print schema and row counts")
    p_inspect.add_argument("--events", **events_kw)
    p_inspect.add_argument("--transmission-modes", help="e.g. 0=respiratory,1=physical_contact")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_report = sub.add_parser("report", help="generate self-contained HTML report")
    p_report.add_argument("--events", **events_kw)
    p_report.add_argument(
        "--world", default=None,
        help="optional world_state.h5 — adds a sexual-demographics section "
             "(orientation, cohabiting couples, relationship status) covering "
             "the full population",
    )
    p_report.add_argument("-o", "--output", required=True, help="output HTML path")
    p_report.set_defaults(func=_cmd_report)

    p_parquet = sub.add_parser("to-parquet", help="dump all tables as parquet")
    p_parquet.add_argument("--events", **events_kw)
    p_parquet.add_argument("-o", "--output", required=True, help="output directory")
    p_parquet.add_argument("--transmission-modes", help="e.g. 0=respiratory,1=physical_contact")
    p_parquet.set_defaults(func=_cmd_to_parquet)

    p_collect = sub.add_parser("collect-runs", help="aggregate scalars across runs")
    p_collect.add_argument("root", help="directory containing run subdirectories")
    p_collect.add_argument("-o", "--output", required=True, help="output .csv or .parquet")
    p_collect.set_defaults(func=_cmd_collect_runs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
