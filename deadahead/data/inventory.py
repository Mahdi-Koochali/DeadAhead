"""Movie table of the extracted CellDeathDetect archives: one row per movie with its phase stack, mask stack and
label spreadsheet, found by file name.

    deadahead data-inventory [--raw-dir DIR] [--out tables/movies.csv]

Paths in the table are relative to the raw data folder (``deadahead.paths.raw_dir()``), so the table stays valid when
the folder moves.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from deadahead import paths

EXPECTED_MOVIES = 63
EXPECTED_LINES = {"PNT1A", "DU145", "LNCaP"}
EXPECTED_TREATMENTS = {"bp", "do", "st"}
EXPECTED_FOVS = set(range(1, 8))

# One pattern per file role. Case-insensitive and tolerant of .tif/.tiff, so that a file with an unexpected name is
# reported instead of being skipped.
PATTERNS = {
    "image": re.compile(r"^QPI_(?P<line>[A-Za-z0-9]+)_(?P<treat>[a-z]{2})_(?P<fov>\d+)\.tiff?$", re.IGNORECASE),
    "mask": re.compile(r"^mask_(?P<line>[A-Za-z0-9]+)_(?P<treat>[a-z]{2})_(?P<fov>\d+)\.tiff?$", re.IGNORECASE),
    "labels": re.compile(r"^labels_(?P<line>[A-Za-z0-9]+)_(?P<treat>[a-z]{2})_(?P<fov>\d+)\.xlsx$", re.IGNORECASE),
}


def classify(path: Path):
    """The role (image, mask or labels) of a file and its cell line, treatment and field of view, or (None, None)."""
    for role, pat in PATTERNS.items():
        m = pat.match(path.name)
        if m:
            return role, m.groupdict()
    return None, None


def find_movies(root: Path) -> tuple[pd.DataFrame, dict]:
    """(movie table, problems) of the .tif, .tiff and .xlsx files under ``root``; other files, such as the downloaded
    .zip archives, are skipped.

    One row per movie with all three files: paths relative to ``root`` and sizes in bytes, sorted by cell line,
    treatment and field. ``problems`` lists unmatched files, duplicated roles and movies with a missing file.
    """
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".xlsx"}]

    movies: dict[str, dict] = defaultdict(dict)
    unmatched: list[str] = []
    duplicates: list[str] = []

    for p in sorted(files):
        role, g = classify(p)
        if role is None:
            unmatched.append(str(p.relative_to(root)))
            continue
        line, treat, fov = g["line"], g["treat"].lower(), int(g["fov"])
        movie_id = f"{line}_{treat}_{fov}"
        rec = movies[movie_id]
        if role in rec:
            duplicates.append(f"{movie_id}:{role}:{p.relative_to(root)}")
            continue
        rec[role] = p
        rec.update(cell_line=line, treatment=treat, fov=fov)

    rows = []
    incomplete = []
    for movie_id in sorted(movies):
        rec = movies[movie_id]
        missing = [r for r in ("image", "mask", "labels") if r not in rec]
        if missing:
            incomplete.append(f"{movie_id}: missing {'+'.join(missing)}")
            continue
        rows.append(
            {
                "movie_id": movie_id,
                "cell_line": rec["cell_line"],
                "treatment": rec["treatment"],
                "fov": rec["fov"],
                "image_path": str(rec["image"].relative_to(root)),
                "mask_path": str(rec["mask"].relative_to(root)),
                "labels_path": str(rec["labels"].relative_to(root)),
                "image_bytes": rec["image"].stat().st_size,
                "mask_bytes": rec["mask"].stat().st_size,
                "labels_bytes": rec["labels"].stat().st_size,
            }
        )

    columns = ["movie_id", "cell_line", "treatment", "fov", "image_path", "mask_path", "labels_path",
               "image_bytes", "mask_bytes", "labels_bytes"]
    df = pd.DataFrame(rows, columns=columns).sort_values(["cell_line", "treatment", "fov"]).reset_index(drop=True)
    return df, {"unmatched": unmatched, "duplicates": duplicates, "incomplete": incomplete}


def dataset_problems(df: pd.DataFrame) -> list[str]:
    """How the table differs from the full dataset: 63 movies, 3 cell lines x 3 treatments x 7 fields."""
    out = []
    if len(df) != EXPECTED_MOVIES:
        out.append(f"{len(df)} complete movies, {EXPECTED_MOVIES} expected")
    if set(df["cell_line"]) != EXPECTED_LINES:
        out.append(f"cell lines {sorted(set(df['cell_line']))}, expected {sorted(EXPECTED_LINES)}")
    if set(df["treatment"]) != EXPECTED_TREATMENTS:
        out.append(f"treatments {sorted(set(df['treatment']))}, expected {sorted(EXPECTED_TREATMENTS)}")
    if {int(x) for x in df["fov"]} != EXPECTED_FOVS:
        out.append(f"fields {sorted({int(x) for x in df['fov']})}, expected 1 to 7")
    per_group = df.groupby(["cell_line", "treatment"]).size()
    if len(per_group) != 9 or not (per_group == 7).all():
        out.append(f"movies per group: {per_group.to_dict()}, expected 7 in each of 9 groups")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write the movie table of the extracted CellDeathDetect archives.")
    ap.add_argument("--raw-dir", type=Path, default=None, help="extracted archives (default: raw/celldeathdetect)")
    ap.add_argument("--out", type=Path, default=None, help="movie table (.csv); default: tables/movies.csv")
    ap.add_argument("--any-size", action="store_true",
                    help="accept a subset or another dataset with the same file names (no 63-movie check)")
    a = ap.parse_args(argv)
    root = a.raw_dir or paths.raw_dir()
    out = a.out or paths.tables_dir() / "movies.csv"
    if not root.is_dir():
        print(f"raw data folder not found: {root}", file=sys.stderr)
        return 3

    df, problems = find_movies(root)
    if df.empty:
        print(f"no movie files under {root}", file=sys.stderr)
        return 3
    found = [f"{k}: {v}" for k, items in problems.items() for v in items]
    if not a.any_size:
        found += dataset_problems(df)
    for line in found[:40]:
        print(f"  {line}", file=sys.stderr)
    if found:
        print(f"{len(found)} problem(s) in {root}; the table was not written", file=sys.stderr)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} movies, {df.groupby(['cell_line', 'treatment']).ngroups} groups -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
