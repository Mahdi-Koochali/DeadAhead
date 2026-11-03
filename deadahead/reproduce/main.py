"""deadahead reproduce: recompute the paper's Table 1, the warnings and the supplementary results from a
predictions bundle.

    deadahead reproduce [--bundle DIR]
                        [--part data|table1|warnings|image-warnings|gap|horizons|sensitivity|decomposition|
                                censoring|gap-window|alarm-rate|within-time|all]
                        [--out FILE] [--no-verify]

Parts: data (the counts of cells, deaths and forecasts), table1 (Table 1, its chance row and the elapsed-time fit with
and without the row weights), warnings (DeadAhead and elapsed time, Fig. 2), image-warnings (post hoc: the image baseline with its own thresholds), gap (post hoc: deaths 60 to 120 min ahead, forecasts whose cell died within 60 min
left out), horizons (the fixed-horizon results at 30, 60, 120 and 240 min: alarms, AUPRC and the survival estimators),
sensitivity (Table 1 at the common cut, the warnings at the common landmark and, with the horizon inputs, the
fixed-horizon results at the common cut), decomposition (the survival head on fine-tuned against adapted video
features, and the video model under three test-time changes), censoring (the injected-censoring experiment),
gap-window (post hoc: the gap analysis at gaps of 0 to 120 min and horizons of 120 and 240 min), alarm-rate (post hoc:
deaths warned against the alarm rate, and the comparison at a matched achieved rate), within-time (post hoc: the 30-min
ranking scored within each hour of imaging, on its own scale). With --part all, a part whose input files are not in the
bundle is skipped.

Writes every number as JSON (default: <runs>/reproduce/<part>.json), then prints a readable table; the time of each part
goes to stderr, so the JSON repeats byte for byte. The tables behind the figures go next to the JSON file:
fig2_warnings.csv and fig2_warnings_spread.csv (--part image-warnings and --part all), figS8a_gap_window.csv
(--part gap-window) and figS8b_alarm_rate.csv (--part alarm-rate).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from deadahead import paths

PARTS = ("data", "table1", "warnings", "image-warnings", "gap", "horizons", "sensitivity", "decomposition",
         "censoring", "gap-window", "alarm-rate", "within-time")
FIG2_CSV = "fig2_warnings.csv"                 # the warning curves at both rates
FIG2_SPREAD_CSV = "fig2_warnings_spread.csv"   # Fig. 2 with its spread, the input of deadahead plot-warnings
GAP_WINDOW_CSV = "figS8a_gap_window.csv"       # the input of deadahead plot-gap-window
ALARM_RATE_CSV = "figS8b_alarm_rate.csv"       # the input of deadahead plot-alarm-rate


def jsonable(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def available_parts(bundle: dict) -> tuple:
    """The parts whose input files the bundle holds (--part all runs these)."""
    from deadahead.reproduce.horizons import has_inputs
    warn = "inner_predictions" in bundle and "movies" in bundle
    have = {"data": "movies" in bundle, "table1": bool(bundle.get("predictions")), "warnings": warn,
            "sensitivity": warn,
            "image-warnings": warn and "dinov2_handcrafted|refit" in bundle.get("inner_predictions_by_method", {}),
            "gap": "landmarks" in bundle and "cumulative_incidence" in bundle and "movies" in bundle,
            "horizons": has_inputs(bundle),
            "decomposition": bool(bundle.get("predictions")) and "adapted_predictions" in bundle
            and "test_time_predictions" in bundle,
            "censoring": "censoring_predictions" in bundle and "censoring_units" in bundle,
            "gap-window": has_inputs(bundle),
            "alarm-rate": warn and "dinov2_handcrafted|refit" in bundle.get("inner_predictions_by_method", {}),
            "within-time": {"deadahead", "dinov2_handcrafted|refit"} <= set(bundle.get("predictions") or ())}
    return tuple(p for p in PARTS if have[p])


def run(bundle: dict, parts) -> dict:
    out = {}
    for p in parts:
        t0 = time.time()
        if p == "data":
            from deadahead.reproduce.data import part_data
            out[p] = part_data(bundle)
        elif p == "table1":
            from deadahead.reproduce.table1 import part_table1
            out[p] = part_table1(bundle)
        elif p == "warnings":
            from deadahead.reproduce.warning_times import part_warnings
            out[p] = part_warnings(bundle)
        elif p == "image-warnings":
            from deadahead.reproduce.image_warnings import part_image_warnings
            from deadahead.reproduce.warning_times import part_warnings
            if "warnings" not in out:                # DeadAhead's and elapsed time's values, kept for the Fig. 2 tables
                out["warnings"] = part_warnings(bundle)
            out[p] = part_image_warnings(bundle, out["warnings"])
        elif p == "gap":
            from deadahead.reproduce.gap import part_gap
            out[p] = part_gap(bundle)
        elif p == "horizons":
            from deadahead.reproduce.horizons import part_horizons
            out[p] = part_horizons(bundle)
        elif p == "sensitivity":
            from deadahead.reproduce.horizons import has_inputs, part_common_cut
            from deadahead.reproduce.sensitivity import COMMON_CUT, part_sensitivity
            out[p] = part_sensitivity(bundle)
            if has_inputs(bundle):
                out[p][COMMON_CUT] = part_common_cut(bundle)
        elif p == "decomposition":
            from deadahead.reproduce.decomposition import part_decomposition
            out[p] = part_decomposition(bundle)
        elif p == "censoring":
            from deadahead.reproduce.censoring import part_censoring
            out[p] = part_censoring(bundle)
        elif p == "gap-window":
            from deadahead.reproduce.gap_window import part_gap_window
            out[p] = part_gap_window(bundle, out.get("horizons"), out.get("gap"))
        elif p == "alarm-rate":
            from deadahead.reproduce.alarm_rate import part_alarm_rate
            out[p] = part_alarm_rate(bundle, out.get("warnings"), out.get("image-warnings"))
        elif p == "within-time":
            from deadahead.reproduce.within_time import part_within_time
            out[p] = part_within_time(bundle, out.get("sensitivity"))
        print(f"{p}: {time.time() - t0:.1f} s", file=sys.stderr)      # on stderr, so the JSON repeats byte for byte
    return out


def report(res: dict) -> str:
    text = []
    if "data" in res:
        from deadahead.reproduce.data import format_data
        text.append(format_data(res["data"]))
    if "table1" in res:
        from deadahead.reproduce.table1 import format_table1
        text.append(format_table1(res["table1"]))
    if "warnings" in res:
        from deadahead.reproduce.warning_times import format_warnings
        text.append(format_warnings(res["warnings"]))
    if "image-warnings" in res:
        from deadahead.reproduce.image_warnings import format_image_warnings
        text.append(format_image_warnings(res["image-warnings"]))
    if "gap" in res:
        from deadahead.reproduce.gap import format_gap
        text.append(format_gap(res["gap"]))
    if "horizons" in res:
        from deadahead.reproduce.horizons import format_horizons
        text.append(format_horizons(res["horizons"]))
    if "sensitivity" in res:
        from deadahead.reproduce.sensitivity import format_sensitivity
        text.append(format_sensitivity(res["sensitivity"]))
    if "decomposition" in res:
        from deadahead.reproduce.decomposition import format_decomposition
        text.append(format_decomposition(res["decomposition"]))
    if "censoring" in res:
        from deadahead.reproduce.censoring import format_censoring
        text.append(format_censoring(res["censoring"]))
    if "gap-window" in res:
        from deadahead.reproduce.gap_window import format_gap_window
        text.append(format_gap_window(res["gap-window"]))
    if "alarm-rate" in res:
        from deadahead.reproduce.alarm_rate import format_alarm_rate
        text.append(format_alarm_rate(res["alarm-rate"]))
    if "within-time" in res:
        from deadahead.reproduce.within_time import format_within_time
        text.append(format_within_time(res["within-time"]))
    return "\n\n".join(text)


def figure_tables(res: dict, part: str) -> dict:
    """{file name: CSV text} of the tables behind the figures that ``res`` holds."""
    out = {}
    if "warnings" in res and (part == "all" or "image-warnings" in res):
        from deadahead.reproduce.image_warnings import fig2_csv, spread_csv
        out[FIG2_CSV] = fig2_csv(res["warnings"], res.get("image-warnings"))
        if "image-warnings" in res:
            out[FIG2_SPREAD_CSV] = spread_csv(res["warnings"], res["image-warnings"])
    if "gap-window" in res:
        from deadahead.reproduce.gap_window import figure_csv
        out[GAP_WINDOW_CSV] = figure_csv(res["gap-window"])
    if "alarm-rate" in res:
        from deadahead.reproduce.alarm_rate import figure_csv
        out[ALARM_RATE_CSV] = figure_csv(res["alarm-rate"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="deadahead reproduce", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=None, help="the bundle folder (default: <DEADAHEAD_DATA>/bundle)")
    ap.add_argument("--part", choices=(*PARTS, "all"), default="table1")
    ap.add_argument("--out", type=Path, default=None, help="the JSON file to write")
    ap.add_argument("--no-verify", action="store_true", help="skip the size and sha256 check of the bundle files")
    a = ap.parse_args(argv)
    from deadahead.reproduce.bundle import load_bundle
    bundle_dir = a.bundle if a.bundle is not None else paths.bundle_dir()
    try:
        bundle = load_bundle(bundle_dir, verify=not a.no_verify)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        if not (bundle_dir / "MANIFEST.json").is_file():
            print(f"download deadahead_bundle_v1.zip and unzip it into {bundle_dir.parent} "
                  "(docs/REPRODUCE.md, Check it in 5 minutes)", file=sys.stderr)
        return 3
    parts = available_parts(bundle) if a.part == "all" else (a.part,)
    skipped = [p for p in PARTS if p not in parts] if a.part == "all" else []
    if not parts:
        print("error: the bundle holds the input files of no part", file=sys.stderr)
        return 3
    try:
        res = run(bundle, parts)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    out = a.out if a.out is not None else paths.runs_dir() / "reproduce" / f"{a.part}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(res, indent=1, default=jsonable) + "\n").encode())
    written = [out]
    for name, text in figure_tables(res, a.part).items():
        (out.parent / name).write_bytes(text.encode())
        written.append(out.parent / name)
    print(report(res))
    if skipped:
        print()
        for p in skipped:
            print(f"skipped {p}: its inputs are not in the bundle")
    print()
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
