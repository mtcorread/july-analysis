"""Single-run HTML report. Runs every metric, renders every plot, and embeds
them as base64 PNGs inside a self-contained HTML file (no external assets).
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from jinja2 import Environment, PackageLoader, select_autoescape

import h5py

from july_analysis import plots, schema as S
from july_analysis.io import EventStore
from july_analysis.metrics import encounters, epi, network, profile, transmission
from july_analysis.metrics import sex_stats as sex_stats_m
from july_analysis import world_state as ws


def _aggregate_encounters_chunked(
    store: EventStore,
    inf: pl.DataFrame,
    chunk_size: int = 2_000_000,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Read coordinated_encounters in chunks and return pre-computed aggregates.

    Returns: (enc_share, channel_eff, size_dist) — same schemas as
    encounters.encounter_type_share / channel_efficiency / group_size_distribution.
    Never loads the full dataset into memory.
    """
    _empty_share = pl.DataFrame(schema={"encounter_type_name": pl.Utf8, "count": pl.UInt32, "share": pl.Float64})
    _empty_eff = pl.DataFrame(schema={"encounter_type_name": pl.Utf8, "n_events": pl.UInt32, "n_infections": pl.UInt32, "infections_per_event": pl.Float64})
    _empty_size = pl.DataFrame(schema={"size": pl.UInt32, "count": pl.UInt32})

    reg = store.encounter_type_registry()
    id_to_name: dict[int, str] = reg
    group_enc_ids = {k for k, v in reg.items() if v == "group_sex"}

    with h5py.File(store.path, "r") as h5:
        if S.DATASET_COORDINATED_ENCOUNTERS not in h5:
            return _empty_share, _empty_eff, _empty_size

        ds = h5[S.DATASET_COORDINATED_ENCOUNTERS]
        n = ds.shape[0]
        has_group_id = "group_id" in ds.dtype.names

        type_pair_counts: dict[int, int] = {}
        grp_ids_chunks: list[np.ndarray] = []
        grp_cnt_chunks: list[np.ndarray] = []

        for start in range(0, n, chunk_size):
            chunk = ds[start: start + chunk_size]
            enc_ids = chunk["encounter_type_id"]

            unique_ids, counts = np.unique(enc_ids, return_counts=True)
            for uid, cnt in zip(unique_ids.tolist(), counts.tolist()):
                type_pair_counts[uid] = type_pair_counts.get(uid, 0) + cnt

            if has_group_id and group_enc_ids:
                grp_ids = chunk["group_id"]
                g_mask = np.isin(enc_ids, list(group_enc_ids))
                if g_mask.any():
                    unique_g, cnt_g = np.unique(grp_ids[g_mask], return_counts=True)
                    grp_ids_chunks.append(unique_g)
                    grp_cnt_chunks.append(cnt_g)

    # --- enc_share ---
    total_pairs = sum(type_pair_counts.values()) or 1
    enc_share = pl.DataFrame(
        [{"encounter_type_name": id_to_name.get(t, f"unknown_{t}"), "count": c, "share": c / total_pairs}
         for t, c in sorted(type_pair_counts.items(), key=lambda x: -x[1])],
        schema={"encounter_type_name": pl.Utf8, "count": pl.UInt32, "share": pl.Float64},
    ) if type_pair_counts else _empty_share

    # --- merge group_id counts across chunks ---
    n_group_events = 0
    group_sizes_arr: np.ndarray = np.array([], dtype=np.int64)
    if grp_ids_chunks:
        all_gids = np.concatenate(grp_ids_chunks)
        all_cnts = np.concatenate(grp_cnt_chunks)
        unique_gids, inv_idx = np.unique(all_gids, return_inverse=True)
        total_cnts = np.bincount(inv_idx, weights=all_cnts.astype(np.float64)).astype(np.int64)
        n_group_events = len(unique_gids)
        group_sizes_arr = total_cnts + 1  # +1 for host

    # --- channel_eff ---
    n_events_by_name: dict[str, int] = {}
    for tid, cnt in type_pair_counts.items():
        name = id_to_name.get(tid, f"unknown_{tid}")
        n_events_by_name[name] = n_group_events if tid in group_enc_ids else cnt

    if inf.height and "encounter_type_name" in inf.columns and "is_seed" in inf.columns:
        inf_by_type = (
            inf.filter(~pl.col("is_seed"))
            .group_by("encounter_type_name")
            .agg(pl.len().cast(pl.UInt32).alias("n_infections"))
        )
    else:
        inf_by_type = pl.DataFrame(schema={"encounter_type_name": pl.Utf8, "n_infections": pl.UInt32})

    if n_events_by_name:
        channel_eff = (
            pl.DataFrame({"encounter_type_name": list(n_events_by_name.keys()),
                          "n_events": list(n_events_by_name.values())})
            .with_columns(pl.col("n_events").cast(pl.UInt32))
            .join(inf_by_type, on="encounter_type_name", how="left")
            .with_columns(pl.col("n_infections").fill_null(0))
            .with_columns((pl.col("n_infections").cast(pl.Float64) / pl.col("n_events")).alias("infections_per_event"))
            .sort("n_events", descending=True)
        )
    else:
        channel_eff = _empty_eff

    # --- size_dist ---
    if len(group_sizes_arr) > 0:
        unique_sizes, size_cnts = np.unique(group_sizes_arr.astype(np.uint32), return_counts=True)
        size_dist = pl.DataFrame({"size": unique_sizes, "count": size_cnts.astype(np.uint32)})
    else:
        size_dist = _empty_size

    return enc_share, channel_eff, size_dist


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _df_to_rows(df: pl.DataFrame, *, max_rows: int = 20) -> tuple[list[str], list[list[str]]]:
    if df.height == 0:
        return ([], [])
    headers = df.columns
    rows = []
    for i, row in enumerate(df.iter_rows()):
        if i >= max_rows:
            break
        rows.append([
            f"{v:.4g}" if isinstance(v, float) else (str(v) if v is not None else "")
            for v in row
        ])
    return (headers, rows)


def _build_world_section(world_path: Path) -> dict:
    """Stream the world snapshot and return template variables for the
    sexual-demographics section. Empty dict on missing/unreadable file.
    """
    world = ws.load_world(world_path, progress=False)
    breakdown = sex_stats_m.orientation_breakdown(world)
    couples = sex_stats_m.couple_stats(world)
    rs_rows_data = sex_stats_m.relationship_status_rows(world)

    n = world.n_people
    n_male = int((world.sexes == 0).sum())
    n_female = int((world.sexes == 1).sum())

    # Tables (header, rows). Reuse the shared _df_to_rows formatter where
    # possible by going through tiny ad-hoc lists.
    def fmt_count(v: int, total: int) -> str:
        pct = (v / total * 100) if total else 0.0
        return f"{v:,} ({pct:.1f}%)"

    overall = breakdown.overall
    overall_rows = [
        [lab, int(c), f"{(c / n * 100):.2f}%"]
        for lab, c in zip(ws.ORIENTATION_LABELS, overall)
    ]

    by_sex_rows = []
    for i, lab in enumerate(ws.ORIENTATION_LABELS):
        row = [lab] + [int(breakdown.by_sex[i, j]) for j in range(3)] + [
            int(breakdown.by_sex[i].sum())
        ]
        by_sex_rows.append(row)

    by_age_rows = []
    for i, lab in enumerate(ws.AGE_BIN_LABELS):
        total = int(breakdown.by_age[i].sum())
        row = [lab, total]
        for j in range(len(ws.ORIENTATION_LABELS)):
            row.append(fmt_count(int(breakdown.by_age[i, j]), total))
        by_age_rows.append(row)

    order = np.argsort(breakdown.lgu_population)[::-1][:30]
    lgu_rows = []
    for k in order:
        lgu_id = int(breakdown.lgu_ids[k])
        name = world.geography.name(lgu_id)
        row = [name, int(breakdown.lgu_population[k])]
        total = int(breakdown.by_lgu[k].sum())
        for j in range(len(ws.ORIENTATION_LABELS)):
            row.append(fmt_count(int(breakdown.by_lgu[k, j]), total))
        lgu_rows.append(row)

    couple_pair_rows = [
        [f"{a} ↔ {b}", int(cnt)]
        for (a, b), cnt in sorted(
            couples.orientation_pair_counts.items(), key=lambda kv: -kv[1]
        )
    ]
    rs_rows = [[lab, int(c)] for lab, c in rs_rows_data]

    n_couple_rec = world.n_couple_records
    expected = n_couple_rec / 2
    reciprocity = (couples.n_couples / expected * 100) if expected else 0.0

    # Plots
    by_age_pct = sex_stats_m.percent_rows(breakdown.by_age.astype(float))
    by_lgu_top = breakdown.by_lgu[order]
    by_lgu_pct = sex_stats_m.percent_rows(by_lgu_top.astype(float))
    lgu_labels = [world.geography.name(int(breakdown.lgu_ids[k]))
                  for k in order[:12]]
    by_lgu_plot_pct = sex_stats_m.percent_rows(
        breakdown.by_lgu[order[:12]].astype(float)
    )

    figures = {
        "world_orientation_overall": _fig_to_base64(
            plots.orientation_overall_bar(overall, ws.ORIENTATION_LABELS)),
        "world_orientation_by_sex": _fig_to_base64(
            plots.orientation_by_sex_bar(
                breakdown.by_sex, ws.ORIENTATION_LABELS, ws.SEX_LABELS)),
        "world_orientation_by_age": _fig_to_base64(
            plots.orientation_by_age_stacked(
                by_age_pct, ws.AGE_BIN_LABELS, ws.ORIENTATION_LABELS)),
        "world_orientation_by_lgu": _fig_to_base64(
            plots.orientation_by_lgu_stacked(
                by_lgu_plot_pct, lgu_labels, ws.ORIENTATION_LABELS)),
        "world_relationship_status": _fig_to_base64(
            plots.relationship_status_bar(rs_rows_data)),
        "world_couple_sex_pairing": _fig_to_base64(
            plots.couple_sex_pairing_bar(couples.sex_pairing,
                                         couples.sex_pair_labels)),
        "world_couple_age_diff": _fig_to_base64(
            plots.couple_age_diff_bar(couples.age_diff_hist,
                                      couples.age_diff_edges)),
    }

    tables = {
        "world_orientation_overall": (["orientation", "people", "share"],
                                      overall_rows),
        "world_orientation_by_sex": (
            ["orientation", "male", "female", "unknown", "total"], by_sex_rows),
        "world_orientation_by_age": (
            ["age_band", "population"] + list(ws.ORIENTATION_LABELS), by_age_rows),
        "world_orientation_by_lgu": (
            ["LGU", "population"] + list(ws.ORIENTATION_LABELS), lgu_rows),
        "world_relationship_status": (["status", "people"], rs_rows),
        "world_couple_orientation_pairs": (["orientation_pair", "couples"],
                                           couple_pair_rows),
    }

    return {
        "world_path": str(world_path),
        "world_n_people": n,
        "world_n_male": n_male,
        "world_n_female": n_female,
        "world_n_sgu": int((world.geography.levels == 3).sum()),
        "world_n_mgu": int((world.geography.levels == 2).sum()),
        "world_n_lgu": int((world.geography.levels == 1).sum()),
        "world_couples": couples.n_couples,
        "world_couple_records": n_couple_rec,
        "world_reciprocity_pct": reciprocity,
        "world_same_sgu_pct": couples.same_sgu_share * 100,
        "world_same_lgu_pct": couples.same_lgu_share * 100,
        "world_self_referential": world.n_self_referential,
        "world_partner_unknown": world.n_partner_unknown,
        "world_orientation_labels": list(ws.ORIENTATION_LABELS),
        "world_figures": figures,
        "world_tables": tables,
    }


def build_report(
    run_path: str | Path,
    output: str | Path,
    *,
    world_path: str | Path | None = None,
    groupings_path: str | Path | None = None,
) -> Path:
    """Generate a self-contained HTML report for one run. Returns the output path.

    If ``world_path`` is provided, also streams that world_state.h5 file and
    appends a sexual-demographics section (orientation, relationship status,
    cohabiting couples) covering the *full* population.

    ``groupings_path`` overrides the bundled groupings YAML (see
    :mod:`july_analysis.groupings`). Defaults to the package config; the
    "sexual" grouping drives the "Sexual transmission focus" section.
    """
    from july_analysis.groupings import load_groupings, validate_against_registry

    store = EventStore(run_path)
    meta = store.meta()

    groupings = load_groupings(groupings_path)
    validate_against_registry(groupings, store.encounter_type_registry())

    inf = store.infections()
    rel = store.relationships()
    pop = store.population_summary()
    deaths = store.deaths()
    symp = store.symptom_changes()
    venues = store.venues()

    # --- metrics ------------------------------------------------------------
    daily = epi.incidence_by_day(inf)
    cum = epi.cumulative_incidence(inf)
    daily_by_type = epi.incidence_by_encounter_type(inf)
    sexual_types = list(groupings["sexual"].encounter_types) if "sexual" in groupings else []
    if inf.height and "encounter_type_name" in inf.columns and sexual_types:
        inf_sexual = inf.filter(pl.col("encounter_type_name").is_in(sexual_types))
    else:
        inf_sexual = inf
    daily_by_type_sexual = epi.incidence_by_encounter_type(inf_sexual)
    contribution_sexual = epi.encounter_type_contribution(inf_sexual)
    venue_type_contrib = epi.infections_by_venue_type(inf, venues)
    contribution = epi.encounter_type_contribution(inf)
    mode_contribution = epi.transmission_mode_contribution(inf)
    scalars = epi.summary_scalars(inf, deaths)
    tts = epi.time_to_symptom(inf, symp)

    deg_dist = network.degree_distribution(rel)
    end_t = meta.simulation_end_time or 7.0
    probe_times = np.linspace(0.0, float(end_t), num=min(8, max(2, int(end_t) + 1)))
    # Strip cohabiting relationships at the person level so they don't dominate
    # the churny-concurrency signal. Falls back to the whole population if the
    # file was produced before population_networks was added.
    cohabiting_edges = (
        store.population_network("cohabiting_couple")
        if "cohabiting_couple" in store.available_population_networks()
        else None
    )
    conc = network.concurrency_series(
        rel, times=probe_times, exclude_cohabiting_edges=cohabiting_edges
    )

    mix = network.mixing_matrix(
        inf.filter(~inf["is_seed"]) if inf.height and "is_seed" in inf.columns else inf,
        pop.select(["person_id", "age_group"]) if pop.height else pop,
        "age_group",
        a="infector_id",
        b="person_id",
    )

    reff = transmission.reff_by_encounter_type(inf)
    reff_rolling = transmission.reff_rolling(inf, window_days=3)
    gen_time = transmission.generation_time_distribution(inf)
    top_superspreaders = transmission.secondary_cases_per_infector(inf).head(10)

    # --- profile facets (optional) -----------------------------------------
    profile_facet: str | None = None
    profile_assignments = pl.DataFrame()
    profile_dist = pl.DataFrame()
    inf_by_infectee_profile = pl.DataFrame()
    inf_by_infector_profile = pl.DataFrame()
    reff_profile = pl.DataFrame()
    facets = store.available_profile_facets()
    if facets:
        profile_facet = facets[0]
        profile_assignments = store.profile_assignments(profile_facet)
        profile_dist = profile.profile_type_distribution(profile_assignments)
        inf_by_infectee_profile = profile.infections_by_profile(
            inf, profile_assignments, side="infectee"
        )
        inf_by_infector_profile = profile.infections_by_profile(
            inf, profile_assignments, side="infector"
        )
        reff_profile = profile.reff_by_profile(inf, profile_assignments)

    enc_share, channel_eff, size_dist = _aggregate_encounters_chunked(store, inf)

    # --- figures ------------------------------------------------------------
    figures = {
        "incidence": _fig_to_base64(plots.incidence_curve(daily)),
        "cumulative": _fig_to_base64(plots.cumulative_curve(cum)),
        "stacked_incidence": _fig_to_base64(plots.stacked_incidence_by_encounter(daily_by_type)),
        "stacked_incidence_sexual": _fig_to_base64(
            plots.stacked_incidence_by_encounter(
                daily_by_type_sexual,
                title="Infections by sexual encounter type",
            )
        ),
        "venue_type_contribution": _fig_to_base64(
            plots.infections_by_venue_type_bar(venue_type_contrib)
        ),
        "sexual_contribution": _fig_to_base64(
            plots.encounter_type_contribution_bar(contribution_sexual)
        ),
        "contribution": _fig_to_base64(plots.encounter_type_contribution_bar(contribution)),
        "degree": _fig_to_base64(plots.degree_distribution_bar(deg_dist)),
        "concurrency": _fig_to_base64(plots.concurrency_curve(conc)),
        "mixing": _fig_to_base64(plots.mixing_matrix_heatmap(mix, attr="age_group")),
        "gen_time": _fig_to_base64(plots.generation_time_hist(gen_time)),
        "reff_rolling": _fig_to_base64(plots.reff_curve(reff_rolling)),
        "group_sizes": _fig_to_base64(plots.group_size_distribution_bar(size_dist)),
        "enc_share": _fig_to_base64(plots.encounter_type_share_bar(enc_share)),
        "profile_dist": _fig_to_base64(plots.profile_type_distribution_bar(profile_dist)),
        "infections_by_infector_profile": _fig_to_base64(
            plots.infections_by_profile_bar(
                inf_by_infector_profile, title="Infections by infector profile"
            )
        ),
        "infections_by_infectee_profile": _fig_to_base64(
            plots.infections_by_profile_bar(
                inf_by_infectee_profile, title="Infections by infectee profile"
            )
        ),
    }

    # --- tables -------------------------------------------------------------
    tables = {
        "encounter_contribution": _df_to_rows(contribution),
        "venue_type_contribution": _df_to_rows(venue_type_contrib),
        "sexual_contribution": _df_to_rows(contribution_sexual),
        "channel_efficiency": _df_to_rows(channel_eff),
        "mode_contribution": _df_to_rows(mode_contribution),
        "reff_by_encounter": _df_to_rows(reff),
        "top_superspreaders": _df_to_rows(top_superspreaders),
        "profile_distribution": _df_to_rows(profile_dist),
        "reff_by_profile": _df_to_rows(reff_profile),
        "infections_by_infector_profile": _df_to_rows(inf_by_infector_profile),
        "time_to_symptom_summary": _df_to_rows(
            tts.select([
                pl.col("delay").mean().alias("mean_days"),
                pl.col("delay").median().alias("median_days"),
                pl.col("delay").min().alias("min_days"),
                pl.col("delay").max().alias("max_days"),
                pl.len().alias("n_people"),
            ])
            if tts.height
            else tts
        ),
    }

    world_ctx: dict = {}
    if world_path is not None:
        world_ctx = _build_world_section(Path(world_path))

    env = Environment(
        loader=PackageLoader("july_analysis", "templates"),
        autoescape=select_autoescape(["html"]),
    )
    tmpl = env.get_template("report.html")
    html = tmpl.render(
        meta=meta,
        scalars=scalars,
        figures=figures,
        tables=tables,
        encounter_registry=meta.encounter_type_registry,
        profile_facet=profile_facet,
        world=world_ctx,
    )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
