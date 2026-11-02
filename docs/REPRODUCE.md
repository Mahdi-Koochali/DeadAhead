# Check and reproduce

[Back to the README](../README.md)

## Check it in 5 minutes

Recompute Table 1 on a CPU, with no GPU and no raw movies (Python 3.10 or newer; on macOS, `shasum -a 256 -c -` replaces
`sha256sum -c -`):

```bash
git clone https://github.com/Mahdi-Koochali/DeadAhead.git && cd DeadAhead
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
export DEADAHEAD_DATA="$PWD/data"
curl -L -o data/deadahead_bundle_v1.zip "https://zenodo.org/records/23883611/files/deadahead_bundle_v1.zip?download=1"
echo "e99a7a7a6afa290fd6e0c7c0bd9421a988793d7b29c84bf558418e3c69f65060  data/deadahead_bundle_v1.zip" | sha256sum -c -
unzip -q data/deadahead_bundle_v1.zip -d data
deadahead reproduce --part table1
```

The last command prints Table 1 in about a minute. Compare these rows (`dinov2_handcrafted|refit` is the paper's DINOv2 +
handcrafted; `(8)`: the eight groups other than PNT1A bp):

```text
method                                       lytic  apoptosis  lytic (8)  apopt. (8)
chance (prevalence)                          0.024      0.012      0.023       0.008
deadahead                                    0.322      0.213      0.324       0.195
dinov2_handcrafted|refit                     0.254      0.180      0.268       0.155
```

and, further down, DeadAhead's lead over that baseline:

```text
  dinov2_handcrafted|refit                LYTIC        +0.068   8/9  DeadAhead ahead (meets the decision rule)
```

The [claim table](#which-command-checks-which-claim) maps each claim of the abstract, the post hoc matched and within-hour results, Table 1, Table 2 and Figure 2 to one command and the lines of output to read.

**Tests only, no download:**

```bash
pip install -e ".[dev]"
pytest tests/test_cli.py tests/test_metrics.py tests/test_paths.py tests/test_packaging.py tests/test_row_weights.py tests/test_horizons_metrics.py tests/test_table1.py
```

This gives `60 passed, 2 skipped` in about 35 s without PyTorch, and includes a reproduce run on a synthetic bundle. The whole suite, `pytest -m "not data"`,
takes about 9 minutes.

**Supplement cited by the paper:** [S1 fixed horizons](SUPPLEMENT.md#s1-fixed-horizon-forecasts), [S2 per group](SUPPLEMENT.md#s2-per-group-differences),
[S3 warnings](SUPPLEMENT.md#s3-warnings-in-detail), [S4 fine-tuning and test-time changes](SUPPLEMENT.md#s4-fine-tuning-and-test-time-changes),
[S5 VideoMAE alone, same head, elapsed time](SUPPLEMENT.md#s5-videomae-alone-the-same-head-comparison-and-elapsed-time),
[S6 gap, sensitivity, censoring, labels](SUPPLEMENT.md#s6-gap-analysis-sensitivity-analyses-censoring-and-label-audit),
[S7 yeast probe](SUPPLEMENT.md#s7-exploratory-probe-yeast-death), [S8 post hoc analyses](SUPPLEMENT.md#s8-post-hoc-analyses).

## Reproducing the paper's numbers

The results listed below can be recomputed on a CPU, without the raw movies or a GPU, from the released predictions
bundle (`deadahead_bundle_v1.zip`, 211 MB; CC BY 4.0; doi:10.5281/zenodo.23883611). It holds the forecast rows, the outcomes and every
method's held-out predictions (elapsed time is refit from the forecast rows, which takes seconds); its `MANIFEST.json` lists
every file with its sha256 and records the source data, the licence and the changes made:

```bash
export DEADAHEAD_DATA="${DEADAHEAD_DATA:-$PWD/data}"
curl -L -o "$DEADAHEAD_DATA/deadahead_bundle_v1.zip" "https://zenodo.org/records/23883611/files/deadahead_bundle_v1.zip?download=1"
echo "e99a7a7a6afa290fd6e0c7c0bd9421a988793d7b29c84bf558418e3c69f65060  $DEADAHEAD_DATA/deadahead_bundle_v1.zip" | sha256sum -c -
unzip -q "$DEADAHEAD_DATA/deadahead_bundle_v1.zip" -d "$DEADAHEAD_DATA"      # creates bundle/
deadahead reproduce --part all
```

`--part all` takes 15 to 34 min with three threads, depending on the load of the machine, and up to three times as long at full load. It needs about 2.2 GB of RAM.

### Which command checks which claim

Each row is one claim of the paper, one command and the line of its output to read; spaces in the lines are collapsed. Times are
wall times of real runs with three threads on the shared server named below, at moderate load; at full load they can take up to three times as long.

| Claim | Command | Line to read | Time |
|---|---|---|---|
| 30-min lytic macro AUPRC of 0.322 | `deadahead reproduce --part table1` | `deadahead 0.322 0.213`, column `lytic` | 50 to 64 s |
| 0.254 for the best image baseline | `deadahead reproduce --part table1` | `dinov2_handcrafted\|refit 0.254 0.180`, the paper's DINOv2 + handcrafted; the `\|original` row beside it is the fit of [Table S2c](SUPPLEMENT.md#s2-per-group-differences) | 50 to 64 s |
| Ahead in 8 of 9 groups | `deadahead reproduce --part table1` | `dinov2_handcrafted\|refit LYTIC +0.068 8/9 DeadAhead ahead (meets the decision rule)` | 50 to 64 s |
| Lytic deaths at 13 times chance | `deadahead reproduce --part table1` | `chance (prevalence) 0.024 0.012` and `deadahead 0.322 0.213`, column `lytic` | 50 to 64 s |
| Apoptosis far above chance, 18 times | `deadahead reproduce --part table1` | `chance (prevalence) 0.024 0.012` and `deadahead 0.322 0.213`, column `apoptosis` | 50 to 64 s |
| Alarm on at every forecast of the final hour before 66.2% of lytic deaths | `deadahead reproduce --part warnings` | under `target 0.2, lytic`: `deadahead achieved 0.24 pooled, 0.29 mean`, then `60: 66.2` | 37 to 48 s |
| Far more than an elapsed-time control | `deadahead reproduce --part warnings` | under `target 0.2, lytic`: `elapsed_time achieved 0.21 pooled, 0.18 mean`, then `60: 10.5`; `DeadAhead minus elapsed time`, then `60:+55.7 9/9` | 37 to 48 s |
| Both routes far above chance out to 240 min | `deadahead reproduce --part horizons` | `lytic 240 deadahead 18.9 0.20 0.489 0.429 0.482 -0.082` and `apoptosis 240 deadahead 9.7 0.25 0.311 0.421 0.284 -0.267`: column `AUPRC` against column `Prev.`, the positive rate in %; every horizon is in [S1](SUPPLEMENT.md#s1-fixed-horizon-forecasts) | 8 to 18 min |
| Post hoc, at a matched alarm rate: 72.8% against 63.5%, ahead in 7 of 9 groups | `deadahead reproduce --part alarm-rate` | `DeadAhead at target 0.26: achieved 0.309 pooled, 0.365 mean over groups, warned 72.8%`; `image baseline at target 0.2: achieved 0.307 pooled, 0.348 mean over groups, warned 63.5%`; `DeadAhead minus the image baseline: +9.3 points, ahead in 7 of 9 groups, DeadAhead ahead (meets the decision rule)` | 4 to 5 min |
| Post hoc, within each hour of imaging: within-hour AUPRC 0.426 lytic and 0.325 apoptosis, a random order 0.070 and 0.070, DINOv2 + handcrafted 0.353 and 0.299; leads +0.073 in 8 of 9 groups, which meets our decision rule, and +0.027 in 5 of 9, which does not | `deadahead reproduce --part within-time` | `Post hoc within-hour AUPRC, lytic: DeadAhead 0.426, DINOv2 + handcrafted 0.353, a random order 0.070; lead +0.073, ahead in 8 of 9 groups, DeadAhead ahead (meets the decision rule)`; `Post hoc within-hour AUPRC, apoptosis: DeadAhead 0.325, DINOv2 + handcrafted 0.299, a random order 0.070; lead +0.027, ahead in 5 of 9 groups, below the decision rule` | 75 to 81 s |
| Tested on held-out line-by-treatment groups | `python -c "import json; [print(f['test_group'], len(f['train_movies']), len(f['test_movies'])) for f in json.load(open('splits/folds.json'))['folds']]"` | nine lines, from `DU145_bp 56 7` to `PNT1A_st 56 7`: each group held out, its movies never in training | under 1 s |
| [Table 1](SUPPLEMENT.md#table-1), every row | `deadahead reproduce --part table1` | the block under `Table 1: AUPRC at 30 min`. Names: `deadahead`, `videomae_finetuned`, `deephit`, `dinov2_handcrafted\|refit` (DINOv2 + handcrafted), `dinov2_finetuned_handcrafted\|refit` (Fine-tuned DINOv2 + handcrafted), `handcrafted\|refit`, `dinov2\|refit`, `elapsed_time` | 50 to 64 s |
| [Table 2](SUPPLEMENT.md#table-2), h = 30 min | `deadahead reproduce --part horizons` | `lytic 30 deadahead 2.4 0.26 0.141 0.777 0.321 0.031` | 8 to 18 min |
| [Table 2](SUPPLEMENT.md#table-2), h = 60 min | `deadahead reproduce --part horizons` | `lytic 60 deadahead 4.8 0.25 0.224 0.648 0.380 0.051` | 8 to 18 min |
| [Table 2](SUPPLEMENT.md#table-2), h = 120 min | `deadahead reproduce --part horizons` | `lytic 120 deadahead 9.6 0.23 0.339 0.533 0.402 -0.002` | 8 to 18 min |
| [Table 2](SUPPLEMENT.md#table-2), h = 240 min | `deadahead reproduce --part horizons` | `lytic 240 deadahead 18.9 0.20 0.489 0.429 0.482 -0.082` | 8 to 18 min |
| [Figure 2](SUPPLEMENT.md#figure-2), the curves | `deadahead reproduce --part image-warnings` | the lines above for DeadAhead and elapsed time; `target 0.2, lytic achieved 0.31 pooled`, then `60: 63.5`, for DINOv2 + handcrafted (post hoc); the bands are in `fig2_warnings_spread.csv` | 69 to 77 s |
| [Figure 2](SUPPLEMENT.md#figure-2), the image | that command, then `deadahead plot-warnings $DEADAHEAD_DATA/runs/reproduce/fig2_warnings_spread.csv docs/figures/fig2_warnings.png` | `git status docs/figures` shows no change with the versions of `environment/cpu-py312.txt` | 69 to 77 s, then a few seconds |
| [Figure 2](SUPPLEMENT.md#figure-2), second row (post hoc) | `deadahead reproduce --part alarm-rate`, then `deadahead plot-alarm-rate $DEADAHEAD_DATA/runs/reproduce/figS8b_alarm_rate.csv docs/figures/figS8b_alarm_rate.png` | `git status docs/figures` shows no change with those versions | 4 to 5 min, then a few seconds |

`--part horizons` prints the columns `Prev.`, `rate`, `precision`, `sensitivity`, `AUPRC`, `IPA`, `IPA (train)`, `<0`, `C`, `slope`
and `P/O`; the lines above stop at `IPA`, and [Table 2](SUPPLEMENT.md#table-2)'s `C` is further along the same line.

### What each part recomputes

| Part | What it recomputes | Time inside `--part all` |
|---|---|:---:|
| `data` | the counts of the [Data and Forecast tables](METHOD.md#method-in-brief): cells, deaths and forecasts over all 63 movies and the 62 scored movies, the cut, the positives, and the movie lengths (439 to 863 frames, 21.95 to 43.15 h) | under 1 s |
| `table1` | Table 1 and its leads, S2, the Table 1 numbers of S5, the fewest and most forecasts per group; also `table1.chance` (`nine_folds`, `eight_folds_without_PNT1A_bp`) and `table1.elapsed_time_fit` (per route, weighted and unweighted: `slope_per_group`, `groups_slope_below_zero`, `groups_slope_above_zero`, `auprc`) | 28 to 53 s |
| `warnings` | Figure 2 (DeadAhead, elapsed time) and S3; prints the mean rate, the row without PNT1A_bp, the eligible deaths and DeadAhead minus elapsed time | 25 to 49 s |
| `image-warnings` | Figure 2 (DINOv2 + handcrafted), that baseline's alarm coverage, S3 (post hoc) | 36 to 100 s |
| `horizons` | Table 2, S1 (Slope is `ipcw_slope`, P/O is `e_over_o`), the same-head comparison and the alarm score of S5 (the comparison is also quoted in S6's gap analysis), the probability accuracy of S6; prints the prevalence, the rate, the groups below zero, Slope, P/O and a block of same-head contrasts | 8 to 17 min |
| `gap` | S6, gap analysis (post hoc); also prints the score by the whole 120-min risk and by the increment given survival to 60 min | 5 to 52 s |
| `sensitivity` | S6, complete follow-up, the follow-up row of S5, and the fixed horizons inside the cut quoted in S8; prints the macro AUPRC per method, the contrasts at the common landmark and the fixed-horizon tables at the cut | 5 to 8 min |
| `decomposition` | S4 | 10 to 11 s |
| `censoring` | S6, keeping censored cells | about 1 s |
| `gap-window` | S8, forecasting beyond the last minutes (post hoc) | 2 to 3 s |
| `alarm-rate` | S8, alarm coverage against the alarm rate and the matched comparison (post hoc); prints the reverse match and the apoptosis match | about 4 min |
| `within-time` | S8, scoring within each hour of imaging (post hoc); prints every method at 1-hour bands, the lead over DINOv2 + handcrafted at each band width and the second design with its ceiling | about 23 s |

The 646 and 1,119 of the `data` part count 11 and 2 excluded deaths dated after their movie ends. Their routes come from the
label spreadsheets (`events.death_type_raw` from `deadahead data-tables`), not from the bundle: in the code they are the
constant `EXCLUDED_ROUTES` in `deadahead/reproduce/data.py`.

These times come from two runs of `--part all` with three threads on a shared 2.2 GHz Xeon E5-2698 v4 server, at different
loads. Most of the time goes to `horizons`, `sensitivity` and `alarm-rate`. A part run alone also loads and checks the bundle
first, and `image-warnings` also computes `warnings`.

The tables behind the figures go next to the JSON output (by default in `$DEADAHEAD_DATA/runs/reproduce/`):
`--part image-warnings` and `--part all` write `fig2_warnings.csv` and `fig2_warnings_spread.csv`, `--part gap-window`
writes `figS8a_gap_window.csv` and `--part alarm-rate` writes `figS8b_alarm_rate.csv`. One command then draws every figure
of the supplement except Figure 1, which is the paper's own figure:

```bash
deadahead plot-figures --out-dir docs/figures
```

It writes each figure to `docs/figures/` and its values to `docs/figures/data/`. Every figure comes from the JSON, except
`figS8c_alarm_share`, which comes from the bundle, as do `s8_group_match.csv` and `method_row_weights.csv` (`--bundle`, or
`--no-bundle` to skip the three). The command took 51 to 70 s on that server, and about 8 min under heavy load, most of it
reading the bundle. `plot-warnings`, `plot-gap-window` and `plot-alarm-rate` draw single
figures from their tables; they print an error and exit with code 3 on a missing or wrong table, and create the output
folder.


`reproduce` checks the bundle against its `MANIFEST.json`, writes every value as JSON and the figure tables, then prints
each table; the time of each part goes to stderr. The JSON of the post hoc parts carries `"post_hoc": true`, and `--help`
marks them. With numpy 2.x the results equal, bit for bit, the values behind the paper's numbers (S1's training-reference
IPA, recorded from a second implementation of the same estimator, agrees to within 1e-15 absolute); with numpy 1.26 they
agree to within about 1e-12 relative, and every printed value is the same. `configs/evaluation.yaml` holds the alarm and
horizon settings (`horizons_min`, `positivity_min_g`).

## Installation

| Part | Python | Packages |
|---|---|---|
| Base install: tables, features, survival head, evaluation, `deadahead reproduce`, `deadahead yeast-score` | 3.10 or newer | LightGBM 4.7.0 and the CPU packages of [`pyproject.toml`](../pyproject.toml) |
| `[video]` extra: VideoMAE, the DINOv2 baselines, DeepHit | 3.10 to 3.13 (no torch 2.5.1 wheel for 3.14); tested fresh on 3.10 and 3.12 | torch 2.5.1, transformers 5.15.1, safetensors 0.8.0, huggingface_hub 1.28.0 |
| `[dev]` extra: tests and lint | 3.10 or newer | pytest and ruff 0.17.3 |

```bash
git clone https://github.com/Mahdi-Koochali/DeadAhead.git
cd DeadAhead
pip install -e .            # CPU: tables, features, survival head, evaluation, reproduction
pip install -e ".[video]"   # adds PyTorch and Transformers 5 (VideoMAE, the DINOv2 baselines, DeepHit)
pip install -e ".[dev]"     # adds pytest and ruff
```

Install in editable mode (`-e`): the settings in `configs/` are read from the checkout. Do not upgrade the `[video]` pins:
a newer Transformers needs torch 2.6 or later and turns PyTorch off. The exact versions of the paper's runs are listed in
`environment/`.

`ruff check` uses the settings in `pyproject.toml`. `pytest` runs the tests on synthetic data; when the yeast bundle is
installed and `DEADAHEAD_DATA` is set it also runs the S7 data test (about 12 minutes at one thread under load), and
`pytest -m "not data"` skips it.
