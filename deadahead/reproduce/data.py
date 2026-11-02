"""Reproduce the data counts of the paper from the bundle's tables: the cells of the 63 movies and of the 62 scored
movies by outcome, the same at the cut, and the forecasts and positives at 30 min.

The events table gives a typed death dated after its movie ends the type EXCLUDED and no route. The paper counts these
13 deaths in its route totals (646 apoptotic, 1,119 lytic), so the totals are not sums of the table: their routes
(EXCLUDED_ROUTES) come from the label spreadsheets, which ``deadahead data-tables`` keeps in ``events.death_type_raw``.
"""

from __future__ import annotations

from deadahead.evaluation.alarms import CUT_FRAME, FRAME_MIN
from deadahead.evaluation.rows import LABEL_CENSORED, LABEL_UNTYPED, TABLE1_HORIZON, table1_rows

EXCLUDED = "EXCLUDED"
#: the routes of the deaths dated after their movie ends, which the bundle does not hold
EXCLUDED_ROUTES = {"APOPTOSIS": 11, "LYTIC": 2}


def cell_counts(events) -> dict:
    """One scope's cells: by event type, the route totals with the excluded deaths, and at the cut (typed deaths after
    frame 438 censored there; untyped deaths kept apart, with those before the cut)."""
    n = {t: int((events["event_type"] == t).sum()) for t in ("APOPTOSIS", "LYTIC", "CENSORED", LABEL_UNTYPED, EXCLUDED)}
    inside = events["t_event"] <= CUT_FRAME
    at_cut = {t: int(((events["event_type"] == t) & inside).sum()) for t in ("APOPTOSIS", "LYTIC", LABEL_UNTYPED)}
    late = n["APOPTOSIS"] + n["LYTIC"] - at_cut["APOPTOSIS"] - at_cut["LYTIC"]
    return {"cells": len(events), "by_event_type": n,
            "at_the_cut": {"apoptosis": at_cut["APOPTOSIS"], "lytic": at_cut["LYTIC"],
                           "censored": n["CENSORED"] + late, "typed_deaths_after_the_cut": late,
                           "untyped": n[LABEL_UNTYPED], "untyped_before_the_cut": at_cut[LABEL_UNTYPED],
                           "excluded": n[EXCLUDED]}}


def route_totals(n: dict) -> dict:
    """The paper's route totals: the typed deaths of the events table with the excluded deaths added to their routes;
    not computed when the bundle's excluded deaths are not those of the released dataset."""
    if n[EXCLUDED] != sum(EXCLUDED_ROUTES.values()):
        return {"not_computed": f"{n[EXCLUDED]} excluded deaths, not the dataset's {sum(EXCLUDED_ROUTES.values())}"}
    return {"apoptosis": n["APOPTOSIS"] + EXCLUDED_ROUTES["APOPTOSIS"],
            "lytic": n["LYTIC"] + EXCLUDED_ROUTES["LYTIC"],
            "excluded_routes": {k.lower(): v for k, v in EXCLUDED_ROUTES.items()},
            "note": "the excluded deaths are counted in these totals; the bundle holds no route for them, the label "
                    "spreadsheets do"}


def part_data(bundle: dict) -> dict:
    """The counts over all movies and the scored movies, and the 30-min forecasts of the evaluation cohort."""
    if "movies" not in bundle:
        raise ValueError("the bundle holds no movie table: the data counts cannot be computed")
    events = bundle["events"].assign(movie_id=lambda d: d["track_id"].astype(str).str.rsplit(":", n=1).str[0])
    movies, rows = bundle["movies"], bundle["rows"]
    scored = set(rows["movie_id"])
    everyone = cell_counts(events)
    n = everyone["by_event_type"]
    tr = table1_rows(rows, bundle["untyped"])
    c, co = tr["cohort"], tr["counts"]
    r30 = rows[rows["horizon_min"] == TABLE1_HORIZON]
    frames = movies["num_frames"]
    return {"movies": {"all": len(movies), "scored": len(scored),
                       "left_out": sorted(set(movies["movie_id"]) - scored),
                       "frames_min": int(frames.min()), "frames_max": int(frames.max()),
                       "hours_min": float(frames.min() * FRAME_MIN / 60),
                       "hours_max": float(frames.max() * FRAME_MIN / 60)},
            "cells_all_movies": everyone,
            "cells_scored_movies": cell_counts(events[events["movie_id"].isin(scored)]),
            "route_totals": route_totals(n),
            "forecasts": {"horizon_min": TABLE1_HORIZON, "rows_scored_movies": len(r30),
                          "follow_up_ends_inside_the_horizon": int((r30["label"] == LABEL_CENSORED).sum()),
                          "cohort_removed": co["cohort_removed"], "U_excluded_on_the_cohort": co["U_excluded"],
                          "cohort_rows": len(c), "lytic_positives": int((c["label"] == "LYTIC").sum()),
                          "apoptosis_positives": int((c["label"] == "APOPTOSIS").sum()),
                          "lytic_rate": float((c["label"] == "LYTIC").mean()),
                          "apoptosis_rate": float((c["label"] == "APOPTOSIS").mean())}}


def _scope(name: str, c: dict) -> list[str]:
    n, a = c["by_event_type"], c["at_the_cut"]
    return [f"{name}: {c['cells']:,} cells",
            f"  apoptosis {n['APOPTOSIS']:,}, lytic {n['LYTIC']:,}, censored {n['CENSORED']:,}, untyped "
            f"{n[LABEL_UNTYPED]:,}, excluded {n[EXCLUDED]:,}",
            f"  at the cut (frame {CUT_FRAME}): apoptosis {a['apoptosis']:,}, lytic {a['lytic']:,}, censored "
            f"{a['censored']:,} ({a['typed_deaths_after_the_cut']:,} typed deaths after the cut among them), untyped "
            f"{a['untyped']:,} ({a['untyped_before_the_cut']:,} before the cut)"]


def format_data(res: dict) -> str:
    """The counts as text."""
    m, r, f = res["movies"], res["route_totals"], res["forecasts"]
    co = f["cohort_removed"]
    routes = (f"  route totals: {r['not_computed']}" if "not_computed" in r else
              f"  route totals: apoptosis {r['apoptosis']:,}, lytic {r['lytic']:,}, with the excluded deaths, dated "
              f"after their movie ends: {r['excluded_routes']['apoptosis']} apoptotic and "
              f"{r['excluded_routes']['lytic']} lytic (routes from the label spreadsheets, not in the bundle)")
    return "\n".join([
        f"Data: {m['all']} movies, {m['frames_min']} to {m['frames_max']} frames ({m['hours_min']:.2f} to "
        f"{m['hours_max']:.2f} h); {m['scored']} scored, left out: {', '.join(m['left_out'])}", "",
        *_scope(f"All {m['all']} movies", res["cells_all_movies"]),
        routes, "",
        *_scope(f"The {m['scored']} scored movies", res["cells_scored_movies"]), "",
        f"Forecasts at {f['horizon_min']} min, {m['scored']} scored movies: {f['rows_scored_movies']:,} rows",
        f"  left out: {co['landmarks']:,} rows of the {co['tracks']} untyped-death cells, then "
        f"{f['U_excluded_on_the_cohort']:,} rows whose follow-up ends inside the horizon with no death",
        f"  evaluation cohort: {f['cohort_rows']:,} rows, lytic {f['lytic_positives']:,} "
        f"({100 * f['lytic_rate']:.2f}%), apoptosis {f['apoptosis_positives']:,} ({100 * f['apoptosis_rate']:.2f}%)"])
