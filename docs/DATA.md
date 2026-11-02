# Data

This is the data card of the release: what each dataset is, how to get it and what a correct build prints. The models have
their own [model card](MODEL_CARD.md).

## CellDeathDetect

Vicar et al. (2019), *Quantitative phase microscopy timelapse dataset of PNT1A, DU-145 and LNCaP cells with annotated
caspase 3,7-dependent and independent cell death*, Zenodo record 2601562, doi:10.5281/zenodo.2601562, CC BY 4.0;
described in Vicar et al. (2020), *The quantitative-phase dynamics of apoptosis and lytic cell death*, Scientific
Reports 10, 1566.
The three archives total about 64.6 GB (LNCaP 17,284,897,824 bytes; PNT1A 22,177,069,075; DU145 25,165,027,366). They
unpack to 66.1 GB, so allow about 131 GB while both exist.

Inside Python, `DEADAHEAD_DATA` defaults to `data/` in the checkout; the shell commands below set it the same way when
it is unset, so the archives go to `data/raw/celldeathdetect/`. Run them from the checkout:

```bash
export DEADAHEAD_DATA="${DEADAHEAD_DATA:-$PWD/data}"
REPO=$(pwd)                                   # the DeadAhead checkout
mkdir -p "$DEADAHEAD_DATA/raw/celldeathdetect" && cd "$DEADAHEAD_DATA/raw/celldeathdetect"
for f in QPI_annotated_timelapse_LNCaP.zip QPI_annotated_timelapse_PNT1A.zip QPI_annotated_timelapse_DU145.zip; do
  wget -c -O "$f" "https://zenodo.org/api/records/2601562/files/$f/content"
done
md5sum -c "$REPO/data/celldeathdetect_md5.txt"
for f in *.zip; do unzip -n "$f"; done
```

## Building the tables

From the extracted archives, four commands build the tables that every later stage reads, under
`$DEADAHEAD_DATA/tables/`:

| Command | Output |
|---|---|
| `deadahead data-inventory` | `movies.csv`: every movie's phase stack, outlines and annotation files |
| `deadahead data-tables` | `movies.parquet`, `tracks.parquet`, `events.parquet` from the annotation spreadsheets |
| `deadahead data-cells` | `cells.parquet`: per-cell, per-frame shape and phase measures (about 1.5 h on one CPU; split with `--movies`, and the last run joins the shards) |
| `deadahead data-windows` | `windows.parquet` (the forecast rows) and `folds.json` (equal to `splits/folds.json`) |

A correct build prints these counts:

| Command | Printed |
|---|---|
| `deadahead data-inventory` | 63 movies, 9 groups |
| `deadahead data-tables` | LYTIC 1117, APOPTOSIS 635, CENSORED 378, EXCLUDED 13, UNTYPED_DEATH 12; 63 movies, 2155 tracks, 2155 events |
| `deadahead data-cells` | 1,240,935 rows, 2155 tracks, 63 movies |
| `deadahead data-windows` | primary_439 213200 rows, full_length 234084 rows, each 2134 tracks; 9 folds |

`tracks.parquet` describes the curated tracks, and no later stage reads it.

Our models locate each cell with the dataset's curated outlines: the video and image crops are centred on them, and
the handcrafted features are measured inside them. A visual check of 9 sampled movies, blind to model output, covered
one movie per group at random, in the 5 frames before death in 79 sampled tracks: 59 tracks were usable (14 of 30 under
staurosporine). One movie failed it: DU145 st, field 2 (`DU145_st_2`), with 1 of 10 tracks usable, against 60 to 100% in
the other eight. It is left out before any model is scored: its death labels train no model on the primary panel (the default; the
optional `all_movies` panel of `train-head` and `baseline-dinov2-survival` keeps them) and it is in no score, so
results cover 62 movies. It stays in the tables (`morphology_excluded` in `movies.parquet` and `cells.parquet`), in
`windows.parquet` and in `folds.json`, and its frames stay in the unlabelled adaptation of the eight folds that train
on its group.

Thirteen typed deaths (11 apoptotic, 2 lytic) are dated 3 to 14 frames after their movie ends and are also left out,
with event type `EXCLUDED` in the tables and the bundle. `data-tables` therefore prints 635 apoptotic and 1,117 lytic
events, where the paper's 646 and 1,119 count these 13. `events.parquet` from `data-tables` keeps their route in
`death_type_raw`; the bundle's `events.parquet` holds only `track_id`, `event_type` and `t_event`.

## DetecDiv (the yeast probe, S7)

Aspert (2021), *Annotated images from yeast cell lifespans - Training & Test sets - DetecDiv (id01)*, Zenodo record 6078462,
doi:10.5281/zenodo.6078462, CC BY 4.0; described in Aspert et al. (2022), *DetecDiv, a generalist deep-learning
platform for automated cell division tracking and survival analysis*, eLife 11, e79519. `deadahead yeast-data` reads
one file, `lifespanAnnotation.zip` (3,372,360,116 bytes, md5 `66cfd66c1ac4e31112efb102df6f4f65`), in memory, without
extracting it, and refuses an archive whose counts differ (250 traps, 249 lifespans, 173 deaths, 24,346 landmarks) before
writing anything. Run the commands from the checkout, as above:

```bash
export DEADAHEAD_DATA="${DEADAHEAD_DATA:-$PWD/data}"
mkdir -p "$DEADAHEAD_DATA/raw/detecdiv" && cd "$DEADAHEAD_DATA/raw/detecdiv"
wget -c -O lifespanAnnotation.zip "https://zenodo.org/api/records/6078462/files/lifespanAnnotation.zip/content"
echo "66cfd66c1ac4e31112efb102df6f4f65  lifespanAnnotation.zip" | md5sum -c
deadahead yeast-data --zip "$DEADAHEAD_DATA/raw/detecdiv/lifespanAnnotation.zip"
```

The scores of S7 need only the yeast predictions bundle (`deadahead yeast-score`), not this archive. `yeast-score` checks
the bundle against its `MANIFEST.json` and returns 3 on a mismatch; `--no-verify` skips the check for your own runs.

## The predictions bundle

To recompute the paper's results without the raw data, download the predictions bundle (CC BY 4.0, listed in
[`WEIGHTS.md`](WEIGHTS.md)), unzip it into `$DEADAHEAD_DATA` (it creates `bundle/`), and run `deadahead reproduce --part all` (see [`REPRODUCE.md`](REPRODUCE.md)).
